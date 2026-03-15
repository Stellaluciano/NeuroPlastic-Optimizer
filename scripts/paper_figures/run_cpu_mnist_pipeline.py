from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Any

import yaml
from torchvision import datasets, transforms

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.paper_figures.generate_cpu_paper_figures import generate_paper_figures

DEFAULT_BENCHMARK_CONFIGS = [
    "configs/mnist/neuroplastic.yaml",
    "configs/mnist/ablation_grad_only.yaml",
    "configs/mnist/adamw.yaml",
    "configs/mnist/adam.yaml",
    "configs/mnist/sgd.yaml",
]


def _validate_environment() -> None:
    missing: list[str] = []
    for module_name in ("torch", "torchvision", "matplotlib", "yaml"):
        try:
            __import__(module_name)
        except ImportError:
            missing.append(module_name)
    if missing:
        raise SystemExit(
            "Missing required packages: "
            + ", ".join(missing)
            + ". Install the repo first with `python -m pip install -e .[dev]`."
        )


def _ensure_mnist(data_root: Path) -> dict[str, Any]:
    transform = transforms.ToTensor()
    train = datasets.MNIST(str(data_root), train=True, download=True, transform=transform)
    test = datasets.MNIST(str(data_root), train=False, download=True, transform=transform)
    sample, _ = train[0]
    classes = getattr(train, "classes", list(range(10)))
    stats = {
        "train_samples": len(train),
        "test_samples": len(test),
        "input_shape": tuple(sample.shape),
        "num_classes": len(classes),
    }
    print("[pipeline] MNIST ready")
    print(f"[pipeline] training samples: {stats['train_samples']}")
    print(f"[pipeline] test samples: {stats['test_samples']}")
    print(f"[pipeline] input shape: {stats['input_shape']}")
    print(f"[pipeline] classes: {stats['num_classes']}")
    return stats


def _run_command(command: list[str], cwd: Path) -> None:
    print("[pipeline] running:", " ".join(command))
    subprocess.run(command, cwd=cwd, check=True)


def _write_smoke_config(
    repo_root: Path,
    results_dir: Path,
    checkpoints_dir: Path,
    data_root: Path,
    generated_configs_dir: Path,
) -> Path:
    base_config = yaml.safe_load((repo_root / "configs" / "mnist" / "neuroplastic.yaml").read_text(encoding="utf-8"))
    experiment = dict(base_config.get("experiment", {}))
    experiment.update(
        {
            "run_name": "neuroplastic_cpu_smoke",
            "epochs": 1,
            "num_workers": 0,
            "device": "cpu",
            "data_root": str(data_root),
            "download": True,
            "output_dir": str(results_dir),
            "checkpoint_dir": str(checkpoints_dir),
        }
    )
    base_config["experiment"] = experiment

    generated_configs_dir.mkdir(parents=True, exist_ok=True)
    smoke_path = generated_configs_dir / "neuroplastic_cpu_smoke.yaml"
    smoke_path.write_text(yaml.safe_dump(base_config, sort_keys=False), encoding="utf-8")
    return smoke_path


def _write_benchmark_override(
    source_config: Path,
    results_dir: Path,
    checkpoints_dir: Path,
    data_root: Path,
    temp_dir: Path,
    seed: int,
    epochs: int | None,
) -> Path:
    payload = yaml.safe_load(source_config.read_text(encoding="utf-8"))
    experiment = dict(payload.get("experiment", {}))
    run_name = str(experiment.get("run_name") or source_config.stem)
    experiment.update(
        {
            "run_name": f"{run_name}_seed{seed}",
            "seed": seed,
            "device": "cpu",
            "data_root": str(data_root),
            "download": True,
            "num_workers": 0,
            "persistent_workers": False,
            "pin_memory": False,
            "output_dir": str(results_dir),
            "checkpoint_dir": str(checkpoints_dir),
        }
    )
    if epochs is not None:
        experiment["epochs"] = epochs
    payload["experiment"] = experiment
    output_path = temp_dir / f"{source_config.stem}_seed{seed}.yaml"
    output_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return output_path


def _run_smoke_test(
    repo_root: Path,
    results_dir: Path,
    checkpoints_dir: Path,
    data_root: Path,
    generated_configs_dir: Path,
) -> Path:
    smoke_config = _write_smoke_config(
        repo_root,
        results_dir,
        checkpoints_dir,
        data_root,
        generated_configs_dir,
    )
    _run_command(
        [sys.executable, "-m", "neuroplastic_optimizer.training.runner", "--config", str(smoke_config)],
        repo_root,
    )
    return smoke_config


def _run_benchmark(
    repo_root: Path,
    results_dir: Path,
    checkpoints_dir: Path,
    data_root: Path,
    generated_configs_dir: Path,
    seeds: list[int],
    epochs: int | None,
) -> None:
    default_results_dir = repo_root / "results"
    default_checkpoints_dir = repo_root / "checkpoints"
    if (
        len(seeds) == 1
        and results_dir.resolve() == default_results_dir.resolve()
        and checkpoints_dir.resolve() == default_checkpoints_dir.resolve()
    ):
        try:
            _run_command([sys.executable, "scripts/benchmark_all.py"], repo_root)
            return
        except subprocess.CalledProcessError as exc:
            print(
                "[pipeline] benchmark_all.py failed in this environment; "
                "falling back to CPU-safe config overrides with num_workers=0."
            )
            print(f"[pipeline] benchmark_all.py exit code: {exc.returncode}")

    temp_dir = generated_configs_dir / f"mnist_benchmark_{uuid.uuid4().hex}"
    temp_dir.mkdir(parents=True, exist_ok=True)
    try:
        override_paths = []
        for seed in seeds:
            for config in DEFAULT_BENCHMARK_CONFIGS:
                override_paths.append(
                    _write_benchmark_override(
                        repo_root / config,
                        results_dir,
                        checkpoints_dir,
                        data_root,
                        temp_dir,
                        seed,
                        epochs,
                    )
                )
        for config_path in override_paths:
            _run_command(
                [sys.executable, "-m", "neuroplastic_optimizer.training.runner", "--config", str(config_path)],
                repo_root,
            )
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a CPU-only MNIST smoke test / benchmark and generate paper figures")
    parser.add_argument("--smoke-only", action="store_true", help="Run only the 1-epoch NeuroPlastic smoke test")
    parser.add_argument("--skip-benchmark", action="store_true", help="Skip the full MNIST optimizer sweep")
    parser.add_argument("--results-dir", default="results", help="Directory to write and read run artifacts")
    parser.add_argument(
        "--output-dir",
        default="paper_artifacts/cpu_mnist",
        help="Directory for generated paper figures and summary artifacts",
    )
    parser.add_argument("--data-root", default="data", help="Local dataset directory")
    parser.add_argument("--checkpoints-dir", default="checkpoints", help="Checkpoint directory")
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[42],
        help="One or more random seeds for the full benchmark sweep",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override benchmark epochs for all generated configs",
    )
    parser.add_argument(
        "--skip-smoke",
        action="store_true",
        help="Do not run the 1-epoch smoke config before the benchmark",
    )
    args = parser.parse_args()

    repo_root = REPO_ROOT
    results_dir = (repo_root / args.results_dir).resolve()
    output_dir = (repo_root / args.output_dir).resolve()
    data_root = (repo_root / args.data_root).resolve()
    checkpoints_dir = (repo_root / args.checkpoints_dir).resolve()
    generated_configs_dir = output_dir / "_generated_configs"

    _validate_environment()
    results_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    generated_configs_dir.mkdir(parents=True, exist_ok=True)

    _ensure_mnist(data_root)
    smoke_config: Path | None = None
    if not args.skip_smoke:
        smoke_config = _run_smoke_test(
            repo_root,
            results_dir,
            checkpoints_dir,
            data_root,
            generated_configs_dir,
        )

    benchmark_ran = False
    if not args.smoke_only and not args.skip_benchmark:
        _run_benchmark(
            repo_root,
            results_dir,
            checkpoints_dir,
            data_root,
            generated_configs_dir,
            args.seeds,
            args.epochs,
        )
        benchmark_ran = True

    figure_summary = generate_paper_figures(results_dir, output_dir)

    print("[pipeline] smoke config:", smoke_config if smoke_config is not None else "skipped")
    print(f"[pipeline] results dir: {results_dir}")
    print(f"[pipeline] checkpoints dir: {checkpoints_dir}")
    print(f"[pipeline] paper artifacts dir: {output_dir}")
    print(f"[pipeline] benchmark seeds: {args.seeds}")
    print(f"[pipeline] benchmark epochs override: {args.epochs}")
    print(f"[pipeline] benchmark ran: {benchmark_ran}")
    print("[pipeline] generated outputs:")
    for path in figure_summary["generated_files"]:
        print(f"[pipeline]   {path}")


if __name__ == "__main__":
    main()

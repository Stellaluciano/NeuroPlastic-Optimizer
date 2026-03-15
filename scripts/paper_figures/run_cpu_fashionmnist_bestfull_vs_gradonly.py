# ruff: noqa: E501

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import yaml
from torchvision import datasets, transforms

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from neuroplastic_optimizer.training.runner import (  # noqa: E402
    _artifact_stem,
    run_experiment,
)
from scripts.paper_figures.run_cpu_mnist_pipeline import (  # noqa: E402
    _validate_environment,
)
from scripts.paper_figures.study_helpers import (  # noqa: E402
    RunArtifactStatus,
    classify_config_group,
    default_seed_values,
    inspect_run_artifacts,
    sanitize_token,
)


def _load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _ensure_fashionmnist(data_root: Path) -> None:
    transform = transforms.ToTensor()
    train = datasets.FashionMNIST(str(data_root), train=True, download=True, transform=transform)
    test = datasets.FashionMNIST(str(data_root), train=False, download=True, transform=transform)
    sample, _ = train[0]
    print("[fashion_pipeline] Fashion-MNIST ready")
    print(f"[fashion_pipeline] training samples: {len(train)}")
    print(f"[fashion_pipeline] test samples: {len(test)}")
    print(f"[fashion_pipeline] input shape: {tuple(sample.shape)}")
    print(f"[fashion_pipeline] classes: {len(getattr(train, 'classes', []))}")


def _resolve_output_paths(repo_root: Path, output_root_arg: str) -> tuple[Path, Path, Path]:
    requested = Path(output_root_arg)
    if requested.name == "results_fashionmnist_bestfull_vs_gradonly_clean":
        results_dir = (repo_root / requested).resolve()
        checkpoints_dir = (
            repo_root / "checkpoints_fashionmnist_bestfull_vs_gradonly_clean"
        ).resolve()
        paper_dir = (
            repo_root / "paper_artifacts" / "fashionmnist_bestfull_vs_gradonly_clean"
        ).resolve()
        return results_dir, checkpoints_dir, paper_dir
    output_root = (repo_root / requested).resolve()
    return (
        output_root / "results_fashionmnist_bestfull_vs_gradonly_clean",
        output_root / "checkpoints_fashionmnist_bestfull_vs_gradonly_clean",
        output_root / "paper_artifacts" / "fashionmnist_bestfull_vs_gradonly_clean",
    )


def _make_payload(
    base_config: dict[str, Any],
    *,
    run_name: str,
    seed: int,
    epochs: int,
    lr: float,
    dataset_name: str,
    results_dir: Path,
    checkpoints_dir: Path,
    data_root: Path,
    mode: str,
    warmup_epochs: int,
    plasticity_scale: float,
) -> dict[str, Any]:
    payload = json.loads(json.dumps(base_config))
    experiment = dict(payload.get("experiment", {}))
    experiment.update(
        {
            "run_name": run_name,
            "dataset": dataset_name,
            "seed": seed,
            "epochs": epochs,
            "lr": lr,
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
    payload["experiment"] = experiment
    plasticity = dict(payload.get("plasticity", {}))
    plasticity.update(
        {
            "mode": mode,
            "warmup_epochs": warmup_epochs,
            "plasticity_scale": plasticity_scale,
        }
    )
    payload["plasticity"] = plasticity
    return payload


def _artifact_paths(config_path: Path, payload: dict[str, Any]) -> dict[str, Path]:
    experiment = payload["experiment"]
    results_dir = Path(experiment["output_dir"])
    checkpoints_dir = Path(experiment["checkpoint_dir"])
    stem = _artifact_stem(str(config_path), SimpleNamespace(**experiment))
    return {
        "summary": results_dir / f"{stem}_summary.json",
        "metrics": results_dir / f"{stem}_metrics.json",
        "events": results_dir / f"{stem}_events.jsonl",
        "checkpoint": checkpoints_dir / f"{stem}_model.pt",
    }


def _group_label(kind: str, lr: float, warmup_epochs: int, plasticity_scale: float) -> str:
    if kind == "baseline":
        return f"baseline lr={lr:g}"
    return (
        f"best_full lr={lr:g}, warmup={warmup_epochs}, "
        f"plasticity_scale={plasticity_scale:g}"
    )


def _print_status_matrix(groups: list[dict[str, Any]]) -> None:
    print("[fashion_pipeline] config status matrix:")
    for state in ("completed", "partial", "missing", "invalid"):
        matching = [group for group in groups if group["group_status"] == state]
        print(f"[fashion_pipeline] {state}: {len(matching)}")
        for group in matching:
            seed_summary = ", ".join(
                f"{seed_status.seed}:{seed_status.state}"
                for seed_status in group["seed_statuses"]
            )
            print(
                f"[fashion_pipeline]   {_group_label(group['kind'], group['lr'], group['warmup_epochs'], group['plasticity_scale'])} "
                f"[{seed_summary}]"
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Fashion-MNIST best-full-vs-grad-only validation on CPU"
    )
    parser.add_argument("--epochs", type=int, default=10, help="Epoch count for each run.")
    parser.add_argument("--seeds", type=int, default=3, help="Number of seeds to run.")
    parser.add_argument("--skip-smoke", action="store_true", help="Reserved for CLI symmetry.")
    parser.add_argument(
        "--best-config",
        required=True,
        help="Path to the selected best full NeuroPlastic config from the MNIST tuning study.",
    )
    parser.add_argument(
        "--output-root",
        default="results_fashionmnist_bestfull_vs_gradonly_clean",
        help="Results directory or root under which Fashion-MNIST outputs are created.",
    )
    args = parser.parse_args()

    repo_root = REPO_ROOT
    results_dir, checkpoints_dir, paper_dir = _resolve_output_paths(repo_root, args.output_root)
    generated_configs_dir = paper_dir / "_generated_configs"
    data_root = (repo_root / "data").resolve()
    manifest_path = paper_dir / "manifest.json"
    best_config = _load_json((repo_root / args.best_config).resolve())

    _validate_environment()
    results_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    paper_dir.mkdir(parents=True, exist_ok=True)
    generated_configs_dir.mkdir(parents=True, exist_ok=True)
    _ensure_fashionmnist(data_root)

    seed_values = default_seed_values(args.seeds)
    baseline_base = _load_yaml(repo_root / "configs" / "mnist" / "ablation_grad_only.yaml")
    full_base = _load_yaml(repo_root / "configs" / "mnist" / "neuroplastic.yaml")

    planned_groups = []
    baseline_seed_plans = []
    for seed in seed_values:
        run_name = f"fashionmnist_grad_only_lr0p1_seed{seed}"
        payload = _make_payload(
            baseline_base,
            run_name=run_name,
            seed=seed,
            epochs=args.epochs,
            lr=0.1,
            dataset_name="fashionmnist",
            results_dir=results_dir,
            checkpoints_dir=checkpoints_dir,
            data_root=data_root,
            mode="ablation_grad_only",
            warmup_epochs=0,
            plasticity_scale=1.0,
        )
        config_path = generated_configs_dir / f"{run_name}.yaml"
        _write_yaml(config_path, payload)
        baseline_seed_plans.append({"seed": seed, "config_path": str(config_path)})
    planned_groups.append(
        {
            "kind": "baseline",
            "lr": 0.1,
            "warmup_epochs": 0,
            "plasticity_scale": 1.0,
            "seed_plans": baseline_seed_plans,
        }
    )

    full_seed_plans = []
    for seed in seed_values:
        run_name = (
            "fashionmnist_best_full_"
            f"lr{sanitize_token(best_config['lr'])}_"
            f"w{best_config['warmup_epochs']}_"
            f"ps{sanitize_token(best_config['plasticity_scale'])}_seed{seed}"
        )
        payload = _make_payload(
            full_base,
            run_name=run_name,
            seed=seed,
            epochs=args.epochs,
            lr=float(best_config["lr"]),
            dataset_name="fashionmnist",
            results_dir=results_dir,
            checkpoints_dir=checkpoints_dir,
            data_root=data_root,
            mode="rule_based",
            warmup_epochs=int(best_config["warmup_epochs"]),
            plasticity_scale=float(best_config["plasticity_scale"]),
        )
        config_path = generated_configs_dir / f"{run_name}.yaml"
        _write_yaml(config_path, payload)
        full_seed_plans.append({"seed": seed, "config_path": str(config_path)})
    planned_groups.append(
        {
            "kind": "full",
            "lr": float(best_config["lr"]),
            "warmup_epochs": int(best_config["warmup_epochs"]),
            "plasticity_scale": float(best_config["plasticity_scale"]),
            "seed_plans": full_seed_plans,
        }
    )

    for group in planned_groups:
        seed_statuses: list[RunArtifactStatus] = []
        for seed_plan in group["seed_plans"]:
            config_path = Path(seed_plan["config_path"])
            payload = _load_yaml(config_path)
            artifact_paths = _artifact_paths(config_path, payload)
            seed_statuses.append(
                inspect_run_artifacts(
                    seed=seed_plan["seed"],
                    summary_path=artifact_paths["summary"],
                    metrics_path=artifact_paths["metrics"],
                    checkpoint_path=artifact_paths["checkpoint"],
                    expected_epochs=payload["experiment"]["epochs"],
                )
            )
        group["seed_statuses"] = seed_statuses
        group["group_status"] = classify_config_group(seed_statuses)

    _print_status_matrix(planned_groups)

    executed_configs: list[dict[str, Any]] = []
    for group in planned_groups:
        for seed_plan, seed_status in zip(group["seed_plans"], group["seed_statuses"], strict=True):
            config_path = Path(seed_plan["config_path"])
            payload = _load_yaml(config_path)
            artifact_paths = _artifact_paths(config_path, payload)
            status = "executed"
            if seed_status.state == "completed":
                status = "skipped_completed"
            else:
                checkpoint_path = artifact_paths["checkpoint"]
                if checkpoint_path.exists():
                    payload["experiment"]["resume_from"] = str(checkpoint_path)
                    _write_yaml(config_path, payload)
                run_experiment(str(config_path))
            executed_configs.append(
                {
                    "config_path": str(config_path),
                    "pre_run_status": seed_status.state,
                    "status": status,
                    "artifact_paths": {key: str(value) for key, value in artifact_paths.items()},
                }
            )
            print(f"[fashion_pipeline] {status}: {config_path.name}")

    manifest = {
        "study_name": "fashionmnist_bestfull_vs_gradonly_clean",
        "dataset": "fashionmnist",
        "question": (
            "Does the best full NeuroPlastic config from the final MNIST tuning study "
            "remain competitive against the grad-only ablation on Fashion-MNIST?"
        ),
        "baseline_config": {
            "optimizer_name": "ablation_grad_only",
            "lr": 0.1,
            "warmup_epochs": 0,
            "plasticity_scale": 1.0,
        },
        "best_full_config": best_config,
        "seeds": seed_values,
        "epoch_count": args.epochs,
        "results_dir": str(results_dir),
        "checkpoints_dir": str(checkpoints_dir),
        "paper_artifacts_dir": str(paper_dir),
        "actual_executed_configs": executed_configs,
        "config_status_matrix": [
            {
                "kind": group["kind"],
                "lr": group["lr"],
                "warmup_epochs": group["warmup_epochs"],
                "plasticity_scale": group["plasticity_scale"],
                "group_status": group["group_status"],
                "seed_statuses": [
                    {
                        "seed": seed_status.seed,
                        "state": seed_status.state,
                        "recorded_epochs": seed_status.recorded_epochs,
                    }
                    for seed_status in group["seed_statuses"]
                ],
            }
            for group in planned_groups
        ],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"[fashion_pipeline] results dir: {results_dir}")
    print(f"[fashion_pipeline] checkpoints dir: {checkpoints_dir}")
    print(f"[fashion_pipeline] paper artifacts dir: {paper_dir}")
    print(f"[fashion_pipeline] manifest: {manifest_path}")


if __name__ == "__main__":
    main()

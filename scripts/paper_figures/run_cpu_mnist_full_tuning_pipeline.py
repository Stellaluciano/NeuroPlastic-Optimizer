from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from neuroplastic_optimizer.training.runner import (  # noqa: E402
    _artifact_stem,
    run_experiment,
)
from scripts.paper_figures.run_cpu_mnist_pipeline import (  # noqa: E402
    _ensure_mnist,
    _run_smoke_test,
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


def _make_baseline_payload(
    base_config: dict[str, Any],
    *,
    seed: int,
    epochs: int,
    lr: float,
    results_dir: Path,
    checkpoints_dir: Path,
    data_root: Path,
) -> dict[str, Any]:
    payload = json.loads(json.dumps(base_config))
    experiment = dict(payload.get("experiment", {}))
    experiment.update(
        {
            "run_name": f"mnist_full_tuning_grad_only_lr{sanitize_token(lr)}_seed{seed}",
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
    plasticity["mode"] = "ablation_grad_only"
    payload["plasticity"] = plasticity
    return payload


def _make_full_payload(
    base_config: dict[str, Any],
    *,
    seed: int,
    epochs: int,
    lr: float,
    warmup_epochs: int,
    plasticity_scale: float,
    results_dir: Path,
    checkpoints_dir: Path,
    data_root: Path,
) -> dict[str, Any]:
    payload = json.loads(json.dumps(base_config))
    experiment = dict(payload.get("experiment", {}))
    experiment.update(
        {
            "run_name": (
                "mnist_full_tuning_full_"
                f"lr{sanitize_token(lr)}_w{warmup_epochs}_ps{sanitize_token(plasticity_scale)}_seed{seed}"
            ),
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
            "mode": "rule_based",
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
    stem = _artifact_stem(
        str(config_path),
        SimpleNamespace(**experiment),
    )
    return {
        "summary": results_dir / f"{stem}_summary.json",
        "metrics": results_dir / f"{stem}_metrics.json",
        "events": results_dir / f"{stem}_events.jsonl",
        "checkpoint": checkpoints_dir / f"{stem}_model.pt",
    }


def _looks_complete(metrics_path: Path, expected_epochs: int) -> bool:
    if not metrics_path.exists():
        return False
    try:
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    test_entries = payload.get("test")
    return isinstance(test_entries, list) and len(test_entries) >= expected_epochs


def _print_matrix(configs: list[dict[str, Any]]) -> None:
    print("[full_tuning] experiment matrix:")
    for item in configs:
        print(
            "[full_tuning]   "
            f"{item['kind']}: lr={item['lr']}, warmup={item['warmup_epochs']}, "
            f"plasticity_scale={item['plasticity_scale']}, seed={item['seed']}"
        )


def _write_manifest(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _resolve_output_paths(repo_root: Path, output_root_arg: str) -> tuple[Path, Path, Path]:
    requested = Path(output_root_arg)
    if requested.name == "results_mnist_full_tuning_clean":
        results_dir = (repo_root / requested).resolve()
        checkpoints_dir = (repo_root / "checkpoints_mnist_full_tuning_clean").resolve()
        paper_dir = (repo_root / "paper_artifacts" / "mnist_full_tuning_clean").resolve()
        return results_dir, checkpoints_dir, paper_dir
    output_root = (repo_root / requested).resolve()
    return (
        output_root / "results_mnist_full_tuning_clean",
        output_root / "checkpoints_mnist_full_tuning_clean",
        output_root / "paper_artifacts" / "mnist_full_tuning_clean",
    )


def _group_label(kind: str, lr: float, warmup_epochs: int, plasticity_scale: float) -> str:
    if kind == "baseline":
        return f"baseline lr={lr:g}"
    return (
        f"full lr={lr:g}, warmup={warmup_epochs}, "
        f"plasticity_scale={plasticity_scale:g}"
    )


def _print_status_matrix(groups: list[dict[str, Any]]) -> None:
    print("[full_tuning] config status matrix:")
    ordered_states = ("completed", "partial", "missing", "invalid")
    for state in ordered_states:
        matching = [group for group in groups if group["group_status"] == state]
        print(f"[full_tuning] {state}: {len(matching)}")
        for group in matching:
            seed_summary = ", ".join(
                f"{seed_status.seed}:{seed_status.state}"
                for seed_status in group["seed_statuses"]
            )
            group_label = _group_label(
                group["kind"],
                group["lr"],
                group["warmup_epochs"],
                group["plasticity_scale"],
            )
            print(
                "[full_tuning]   "
                f"{group_label} "
                f"[{seed_summary}]"
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run a focused CPU MNIST tuning sweep "
            "for full NeuroPlastic vs grad-only ablation"
        )
    )
    parser.add_argument(
        "--seeds",
        type=int,
        default=3,
        help="Number of seeds to run. Defaults to 3.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Epoch count for each tuning run.",
    )
    parser.add_argument(
        "--skip-smoke",
        action="store_true",
        help="Skip the 1-epoch smoke validation run.",
    )
    parser.add_argument(
        "--output-root",
        default=".",
        help=(
            "Root directory under which dedicated "
            "results/checkpoints/artifacts directories are created."
        ),
    )
    parser.add_argument(
        "--lr-values",
        nargs="+",
        type=float,
        default=[0.1, 0.05, 0.03, 0.01],
        help="Learning rates to evaluate for full NeuroPlastic.",
    )
    parser.add_argument(
        "--warmup-values",
        nargs="+",
        type=int,
        default=[0, 1, 2],
        help="Warmup epochs to evaluate for full NeuroPlastic.",
    )
    parser.add_argument(
        "--plasticity-scale-values",
        nargs="+",
        type=float,
        default=[0.5, 1.0, 2.0],
        help="Plasticity scaling coefficients to evaluate for full NeuroPlastic.",
    )
    parser.add_argument(
        "--include-baseline",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include the `ablation_grad_only` baseline at the start of the sweep.",
    )
    parser.add_argument(
        "--baseline-lr",
        type=float,
        default=0.1,
        help="Learning rate for the `ablation_grad_only` baseline.",
    )
    args = parser.parse_args()

    repo_root = REPO_ROOT
    results_dir, checkpoints_dir, paper_dir = _resolve_output_paths(
        repo_root,
        args.output_root,
    )
    generated_configs_dir = paper_dir / "_generated_configs"
    data_root = (repo_root / "data").resolve()
    manifest_path = paper_dir / "manifest.json"

    _validate_environment()
    results_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    paper_dir.mkdir(parents=True, exist_ok=True)
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

    seed_values = default_seed_values(args.seeds)
    full_base_config = _load_yaml(repo_root / "configs" / "mnist" / "neuroplastic.yaml")
    baseline_base_config = _load_yaml(repo_root / "configs" / "mnist" / "ablation_grad_only.yaml")

    planned_groups: list[dict[str, Any]] = []
    if args.include_baseline:
        seed_plans: list[dict[str, Any]] = []
        for seed in seed_values:
            payload = _make_baseline_payload(
                baseline_base_config,
                seed=seed,
                epochs=args.epochs,
                lr=args.baseline_lr,
                results_dir=results_dir,
                checkpoints_dir=checkpoints_dir,
                data_root=data_root,
            )
            config_path = generated_configs_dir / f"{payload['experiment']['run_name']}.yaml"
            _write_yaml(config_path, payload)
            seed_plans.append(
                {
                    "seed": seed,
                    "config_path": str(config_path),
                }
            )
        planned_groups.append(
            {
                "kind": "baseline",
                "lr": args.baseline_lr,
                "warmup_epochs": 0,
                "plasticity_scale": 1.0,
                "seed_plans": seed_plans,
            }
        )

    for lr in args.lr_values:
        for warmup in args.warmup_values:
            for plasticity_scale in args.plasticity_scale_values:
                seed_plans = []
                for seed in seed_values:
                    payload = _make_full_payload(
                        full_base_config,
                        seed=seed,
                        epochs=args.epochs,
                        lr=lr,
                        warmup_epochs=warmup,
                        plasticity_scale=plasticity_scale,
                        results_dir=results_dir,
                        checkpoints_dir=checkpoints_dir,
                        data_root=data_root,
                    )
                    config_path = (
                        generated_configs_dir / f"{payload['experiment']['run_name']}.yaml"
                    )
                    _write_yaml(config_path, payload)
                    seed_plans.append(
                        {
                            "seed": seed,
                            "config_path": str(config_path),
                        }
                    )
                planned_groups.append(
                    {
                        "kind": "full",
                        "lr": lr,
                        "warmup_epochs": warmup,
                        "plasticity_scale": plasticity_scale,
                        "seed_plans": seed_plans,
                    }
                )

    flat_matrix = []
    generated_paths: list[Path] = []
    for group in planned_groups:
        for seed_plan in group["seed_plans"]:
            generated_paths.append(Path(seed_plan["config_path"]))
            flat_matrix.append(
                {
                    "kind": group["kind"],
                    "seed": seed_plan["seed"],
                    "lr": group["lr"],
                    "warmup_epochs": group["warmup_epochs"],
                    "plasticity_scale": group["plasticity_scale"],
                }
            )
    _print_matrix(flat_matrix)

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
            print(f"[full_tuning] {status}: {config_path.name}")

    manifest = {
        "study_name": "mnist_full_tuning_clean",
        "question": (
            "Can full NeuroPlastic outperform the grad-only ablation "
            "after targeted tuning?"
        ),
        "sweep_definition": {
            "full_optimizer": "neuroplastic",
            "lr_values": args.lr_values,
            "warmup_values": args.warmup_values,
            "plasticity_scale_values": args.plasticity_scale_values,
        },
        "baseline_config": {
            "included": args.include_baseline,
            "optimizer_name": "ablation_grad_only",
            "lr": args.baseline_lr,
            "warmup_epochs": 0,
            "plasticity_scale": 1.0,
        },
        "seeds": seed_values,
        "epoch_count": args.epochs,
        "results_dir": str(results_dir),
        "checkpoints_dir": str(checkpoints_dir),
        "paper_artifacts_dir": str(paper_dir),
        "generated_configs": [str(path) for path in generated_paths],
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
        "smoke_config": str(smoke_config) if smoke_config is not None else None,
    }
    _write_manifest(manifest_path, manifest)

    print(f"[full_tuning] results dir: {results_dir}")
    print(f"[full_tuning] checkpoints dir: {checkpoints_dir}")
    print(f"[full_tuning] paper artifacts dir: {paper_dir}")
    print(f"[full_tuning] manifest: {manifest_path}")


if __name__ == "__main__":
    main()

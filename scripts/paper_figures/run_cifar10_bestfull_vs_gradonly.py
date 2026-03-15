# ruff: noqa: E501

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import yaml
from torchvision import datasets, transforms

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from neuroplastic_optimizer.training.runner import _artifact_stem, run_experiment  # noqa: E402
from scripts.paper_figures.run_cpu_mnist_pipeline import _validate_environment  # noqa: E402
from scripts.paper_figures.study_helpers import (  # noqa: E402
    LockedBestConfig,
    RunArtifactStatus,
    classify_config_group,
    default_seed_values,
    format_locked_best_config,
    inspect_run_artifacts,
    load_locked_best_config,
    sanitize_token,
    write_json,
)


def _load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _ensure_cifar10(data_root: Path) -> None:
    transform = transforms.ToTensor()
    try:
        train = datasets.CIFAR10(str(data_root), train=True, download=False, transform=transform)
        test = datasets.CIFAR10(str(data_root), train=False, download=False, transform=transform)
    except RuntimeError:
        train = datasets.CIFAR10(str(data_root), train=True, download=True, transform=transform)
        test = datasets.CIFAR10(str(data_root), train=False, download=True, transform=transform)
    sample, _ = train[0]
    print("[cifar10_pipeline] CIFAR-10 ready")
    print(f"[cifar10_pipeline] training samples: {len(train)}")
    print(f"[cifar10_pipeline] test samples: {len(test)}")
    print(f"[cifar10_pipeline] input shape: {tuple(sample.shape)}")
    print(f"[cifar10_pipeline] classes: {len(getattr(train, 'classes', []))}")


def _resolve_output_paths(repo_root: Path, output_root_arg: str) -> tuple[Path, Path, Path]:
    requested = Path(output_root_arg)
    if requested.name == "results_cifar10_bestfull_vs_gradonly_clean":
        return (
            (repo_root / requested).resolve(),
            (repo_root / "checkpoints_cifar10_bestfull_vs_gradonly_clean").resolve(),
            (repo_root / "paper_artifacts" / "cifar10_bestfull_vs_gradonly_clean").resolve(),
        )
    output_root = (repo_root / requested).resolve()
    return (
        output_root / "results_cifar10_bestfull_vs_gradonly_clean",
        output_root / "checkpoints_cifar10_bestfull_vs_gradonly_clean",
        output_root / "paper_artifacts" / "cifar10_bestfull_vs_gradonly_clean",
    )


def _make_payload(
    base_config: dict[str, Any],
    *,
    run_name: str,
    seed: int,
    epochs: int,
    batch_size: int,
    dataset_root: Path,
    results_dir: Path,
    checkpoints_dir: Path,
    best_config: LockedBestConfig | None,
    baseline: bool,
) -> dict[str, Any]:
    payload = copy.deepcopy(base_config)
    experiment = dict(payload.get("experiment", {}))
    experiment.update(
        {
            "run_name": run_name,
            "dataset": "cifar10",
            "seed": seed,
            "epochs": epochs,
            "batch_size": batch_size,
            "device": "cpu",
            "data_root": str(dataset_root),
            "download": True,
            "num_workers": 0,
            "persistent_workers": False,
            "pin_memory": False,
            "output_dir": str(results_dir),
            "checkpoint_dir": str(checkpoints_dir),
            "tags": {
                "study_name": "cifar10_bestfull_vs_gradonly_clean",
                "comparison_target": "ablation_grad_only_vs_locked_best_full",
                "optimizer_variant": "ablation_grad_only" if baseline else "locked_best_full",
            },
        }
    )
    if baseline:
        experiment["lr"] = 0.1
    else:
        assert best_config is not None
        experiment["lr"] = best_config.lr
    payload["experiment"] = experiment

    plasticity = dict(payload.get("plasticity", {}))
    if baseline:
        plasticity.update({"mode": "ablation_grad_only", "warmup_epochs": 0, "plasticity_scale": 1.0})
    else:
        assert best_config is not None
        plasticity.update(
            {
                "mode": "rule_based",
                "warmup_epochs": best_config.warmup_epochs,
                "plasticity_scale": best_config.plasticity_scale,
            }
        )
    payload["plasticity"] = plasticity
    return payload


def _artifact_paths(config_path: Path, payload: dict[str, Any]) -> dict[str, Path]:
    experiment = payload["experiment"]
    stem = _artifact_stem(str(config_path), SimpleNamespace(**experiment))
    return {
        "summary": Path(experiment["output_dir"]) / f"{stem}_summary.json",
        "metrics": Path(experiment["output_dir"]) / f"{stem}_metrics.json",
        "events": Path(experiment["output_dir"]) / f"{stem}_events.jsonl",
        "checkpoint": Path(experiment["checkpoint_dir"]) / f"{stem}_model.pt",
    }


def _print_status_matrix(groups: list[dict[str, Any]]) -> None:
    print("[cifar10_pipeline] config status matrix:")
    for state in ("completed", "partial", "missing", "invalid"):
        matching = [group for group in groups if group["group_status"] == state]
        print(f"[cifar10_pipeline] {state}: {len(matching)}")
        for group in matching:
            seed_summary = ", ".join(
                f"{seed_status.seed}:{seed_status.state}" for seed_status in group["seed_statuses"]
            )
            print(f"[cifar10_pipeline]   {group['label']} [{seed_summary}]")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run CIFAR-10 locked-best-full vs grad-only validation")
    parser.add_argument("--epochs", type=int, default=5, help="Epoch count for each CIFAR-10 run. Defaults to 5 for CPU practicality.")
    parser.add_argument("--seeds", type=int, default=3, help="Number of seeds to execute.")
    parser.add_argument("--skip-smoke", action="store_true", help="Reserved for symmetry with the other paper scripts.")
    parser.add_argument("--best-config", required=True, help="Locked best full NeuroPlastic config from the MNIST tuning study.")
    parser.add_argument("--output-root", default="results_cifar10_bestfull_vs_gradonly_clean", help="Results directory or parent root for the CIFAR-10 study bundle.")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for both optimizer settings.")
    parser.add_argument("--dataset-root", default="data", help="Dataset root used for CIFAR-10 downloads/cache.")
    args = parser.parse_args()

    repo_root = REPO_ROOT
    results_dir, checkpoints_dir, paper_dir = _resolve_output_paths(repo_root, args.output_root)
    generated_configs_dir = paper_dir / "_generated_configs"
    dataset_root = (repo_root / args.dataset_root).resolve()
    best_config = load_locked_best_config((repo_root / args.best_config).resolve())

    print("[cifar10_pipeline] loaded locked best full config:")
    print(format_locked_best_config(best_config))

    _validate_environment()
    results_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    paper_dir.mkdir(parents=True, exist_ok=True)
    generated_configs_dir.mkdir(parents=True, exist_ok=True)

    baseline_base = _load_yaml(repo_root / "configs" / "mnist" / "ablation_grad_only.yaml")
    full_base = _load_yaml(repo_root / "configs" / "mnist" / "neuroplastic.yaml")
    seed_values = default_seed_values(args.seeds)

    planned_groups: list[dict[str, Any]] = []
    baseline_plans = []
    full_plans = []
    for seed in seed_values:
        baseline_run_name = f"cifar10_grad_only_lr0p1_seed{seed}"
        baseline_payload = _make_payload(
            baseline_base,
            run_name=baseline_run_name,
            seed=seed,
            epochs=args.epochs,
            batch_size=args.batch_size,
            dataset_root=dataset_root,
            results_dir=results_dir,
            checkpoints_dir=checkpoints_dir,
            best_config=None,
            baseline=True,
        )
        baseline_path = generated_configs_dir / f"{baseline_run_name}.yaml"
        _write_yaml(baseline_path, baseline_payload)
        baseline_plans.append({"seed": seed, "config_path": str(baseline_path)})

        full_run_name = (
            "cifar10_best_full_"
            f"lr{sanitize_token(best_config.lr)}_"
            f"w{best_config.warmup_epochs}_"
            f"ps{sanitize_token(best_config.plasticity_scale)}_seed{seed}"
        )
        full_payload = _make_payload(
            full_base,
            run_name=full_run_name,
            seed=seed,
            epochs=args.epochs,
            batch_size=args.batch_size,
            dataset_root=dataset_root,
            results_dir=results_dir,
            checkpoints_dir=checkpoints_dir,
            best_config=best_config,
            baseline=False,
        )
        full_path = generated_configs_dir / f"{full_run_name}.yaml"
        _write_yaml(full_path, full_payload)
        full_plans.append({"seed": seed, "config_path": str(full_path)})

    planned_groups.append({"label": "ablation_grad_only", "kind": "baseline", "seed_plans": baseline_plans})
    planned_groups.append({"label": "locked_best_full", "kind": "full", "seed_plans": full_plans})

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

    needs_execution = any(
        seed_status.state != "completed"
        for group in planned_groups
        for seed_status in group["seed_statuses"]
    )
    if needs_execution:
        _ensure_cifar10(dataset_root)
    else:
        print("[cifar10_pipeline] all requested seeds already completed; skipping dataset setup")

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
                if artifact_paths["checkpoint"].exists():
                    payload["experiment"]["resume_from"] = str(artifact_paths["checkpoint"])
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
            print(f"[cifar10_pipeline] {status}: {config_path.name}")

    manifest_path = paper_dir / "manifest.json"
    write_json(
        manifest_path,
        {
            "study_name": "cifar10_bestfull_vs_gradonly_clean",
            "dataset": "cifar10",
            "epoch_count": args.epochs,
            "batch_size": args.batch_size,
            "dataset_root": str(dataset_root),
            "seeds": seed_values,
            "results_dir": str(results_dir),
            "checkpoints_dir": str(checkpoints_dir),
            "paper_artifacts_dir": str(paper_dir),
            "best_full_config": {
                "path": str((repo_root / args.best_config).resolve()),
                "payload": {
                    "study_name": best_config.study_name,
                    "dataset": best_config.dataset,
                    "optimizer_name": best_config.optimizer_name,
                    "lr": best_config.lr,
                    "warmup_epochs": best_config.warmup_epochs,
                    "plasticity_scale": best_config.plasticity_scale,
                    "selected_by": best_config.selected_by,
                },
            },
            "config_status_matrix": [
                {
                    "label": group["label"],
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
            "actual_executed_configs": executed_configs,
        },
    )

    print(f"[cifar10_pipeline] results dir: {results_dir}")
    print(f"[cifar10_pipeline] checkpoints dir: {checkpoints_dir}")
    print(f"[cifar10_pipeline] paper artifacts dir: {paper_dir}")
    print(f"[cifar10_pipeline] manifest: {manifest_path}")


if __name__ == "__main__":
    main()

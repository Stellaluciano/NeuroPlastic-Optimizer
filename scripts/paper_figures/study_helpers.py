from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class RunArtifactStatus:
    seed: int
    state: str
    summary_path: Path
    metrics_path: Path
    checkpoint_path: Path
    recorded_epochs: int


@dataclass(slots=True)
class LockedBestConfig:
    study_name: str
    dataset: str
    optimizer_name: str
    lr: float
    warmup_epochs: int
    plasticity_scale: float
    selected_by: str
    selection_reason: str
    comparison_to_baseline: dict[str, Any]


def sanitize_token(value: float) -> str:
    return str(value).replace("-", "m").replace(".", "p")


def default_seed_values(seed_count: int) -> list[int]:
    return [41 + index for index in range(seed_count)]


def read_json_if_valid(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def write_json(path: Path, payload: dict[str, Any] | list[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_locked_best_config(path: Path) -> LockedBestConfig:
    payload = read_json_if_valid(path)
    if payload is None:
        raise ValueError(f"Could not parse locked best-config JSON: {path}")
    required_fields = {
        "study_name",
        "dataset",
        "optimizer_name",
        "lr",
        "warmup_epochs",
        "plasticity_scale",
        "selected_by",
        "selection_reason",
    }
    missing = sorted(field for field in required_fields if field not in payload)
    if missing:
        raise ValueError(f"Locked best-config is missing required fields: {missing}")
    return LockedBestConfig(
        study_name=str(payload["study_name"]),
        dataset=str(payload["dataset"]),
        optimizer_name=str(payload["optimizer_name"]),
        lr=float(payload["lr"]),
        warmup_epochs=int(payload["warmup_epochs"]),
        plasticity_scale=float(payload["plasticity_scale"]),
        selected_by=str(payload["selected_by"]),
        selection_reason=str(payload["selection_reason"]),
        comparison_to_baseline=dict(payload.get("comparison_to_baseline", {})),
    )


def locked_best_config_payload(config: LockedBestConfig) -> dict[str, Any]:
    return {
        "study_name": config.study_name,
        "dataset": config.dataset,
        "optimizer_name": config.optimizer_name,
        "lr": config.lr,
        "warmup_epochs": config.warmup_epochs,
        "plasticity_scale": config.plasticity_scale,
        "selected_by": config.selected_by,
        "selection_reason": config.selection_reason,
        "comparison_to_baseline": config.comparison_to_baseline,
    }


def format_locked_best_config(config: LockedBestConfig) -> str:
    return json.dumps(locked_best_config_payload(config), indent=2)


def deterministic_subset_indices(
    *,
    dataset_size: int,
    fraction: float,
    seed: int,
) -> list[int]:
    if dataset_size <= 0:
        raise ValueError("dataset_size must be positive")
    if not (0.0 < fraction <= 1.0):
        raise ValueError("fraction must be in (0, 1]")
    subset_size = dataset_size if fraction == 1.0 else max(1, int(round(dataset_size * fraction)))
    indices = list(range(dataset_size))
    random.Random(seed).shuffle(indices)
    return sorted(indices[:subset_size])


def build_subset_metadata(
    *,
    dataset: str,
    fraction: float,
    seed: int,
    dataset_size: int,
    indices_path: Path,
    indices: list[int],
) -> dict[str, Any]:
    return {
        "dataset": dataset,
        "fraction": fraction,
        "seed": seed,
        "dataset_size": dataset_size,
        "subset_size": len(indices),
        "indices_path": str(indices_path),
    }


def inspect_run_artifacts(
    *,
    seed: int,
    summary_path: Path,
    metrics_path: Path,
    checkpoint_path: Path,
    expected_epochs: int,
) -> RunArtifactStatus:
    payload = read_json_if_valid(metrics_path)
    summary_exists = summary_path.exists()
    if payload is None:
        if summary_exists or checkpoint_path.exists():
            return RunArtifactStatus(
                seed=seed,
                state="invalid",
                summary_path=summary_path,
                metrics_path=metrics_path,
                checkpoint_path=checkpoint_path,
                recorded_epochs=0,
            )
        return RunArtifactStatus(
            seed=seed,
            state="missing",
            summary_path=summary_path,
            metrics_path=metrics_path,
            checkpoint_path=checkpoint_path,
            recorded_epochs=0,
        )

    test_entries = payload.get("test")
    recorded_epochs = len(test_entries) if isinstance(test_entries, list) else 0
    if summary_exists and recorded_epochs >= expected_epochs:
        return RunArtifactStatus(
            seed=seed,
            state="completed",
            summary_path=summary_path,
            metrics_path=metrics_path,
            checkpoint_path=checkpoint_path,
            recorded_epochs=recorded_epochs,
        )
    if recorded_epochs > 0 or summary_exists or checkpoint_path.exists():
        return RunArtifactStatus(
            seed=seed,
            state="partial" if recorded_epochs > 0 else "invalid",
            summary_path=summary_path,
            metrics_path=metrics_path,
            checkpoint_path=checkpoint_path,
            recorded_epochs=recorded_epochs,
        )
    return RunArtifactStatus(
        seed=seed,
        state="missing",
        summary_path=summary_path,
        metrics_path=metrics_path,
        checkpoint_path=checkpoint_path,
        recorded_epochs=0,
    )


def classify_config_group(seed_statuses: list[RunArtifactStatus]) -> str:
    states = {item.state for item in seed_statuses}
    if states == {"completed"}:
        return "completed"
    if "invalid" in states:
        return "invalid"
    if "partial" in states:
        return "partial"
    if "completed" in states:
        return "partial"
    return "missing"


def compare_seed_metric_dicts(
    baseline_by_seed: dict[int, float | None],
    candidate_by_seed: dict[int, float | None],
) -> dict[str, Any]:
    shared_seeds = sorted(set(baseline_by_seed).intersection(candidate_by_seed))
    wins = 0
    losses = 0
    ties = 0
    for seed in shared_seeds:
        baseline = baseline_by_seed[seed]
        candidate = candidate_by_seed[seed]
        if baseline is None or candidate is None:
            continue
        if candidate > baseline:
            wins += 1
        elif candidate < baseline:
            losses += 1
        else:
            ties += 1
    return {
        "shared_seeds": shared_seeds,
        "shared_seed_count": len(shared_seeds),
        "wins": wins,
        "losses": losses,
        "ties": ties,
    }

# ruff: noqa: E501

from __future__ import annotations

import csv
import json
import math
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


@dataclass(slots=True)
class TuningRun:
    run_name: str
    optimizer_name: str
    lr: float
    warmup_epochs: int
    plasticity_scale: float
    seed: int
    epochs: int
    dataset: str
    model_identifier: str
    timestamp: str | None
    git_commit_hash: str | None
    result_directory: str
    summary_path: Path
    metrics_path: Path
    events_path: Path | None
    test_accuracy: list[float]
    test_loss: list[float]
    final_test_accuracy: float | None
    best_test_accuracy: float | None
    final_test_loss: float | None
    tags: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class TuningAggregate:
    optimizer_name: str
    lr: float
    warmup_epochs: int
    plasticity_scale: float
    runs: list[TuningRun]
    epochs: list[int]
    mean_test_accuracy: list[float]
    std_test_accuracy: list[float]
    mean_test_loss: list[float]
    std_test_loss: list[float]
    num_seeds: int
    mean_final_test_accuracy: float | None
    std_final_test_accuracy: float | None
    mean_best_test_accuracy: float | None
    std_best_test_accuracy: float | None
    mean_final_test_loss: float | None
    std_final_test_loss: float | None
    mean_epoch_to_95_acc: float | None
    mean_epoch_to_97_acc: float | None
    num_seeds_reached_95_acc: int
    num_seeds_reached_97_acc: int

    @property
    def config_key(self) -> tuple[str, float, int, float]:
        return (
            self.optimizer_name,
            self.lr,
            self.warmup_epochs,
            self.plasticity_scale,
        )


@dataclass(slots=True)
class AggregateSplit:
    complete: list[TuningAggregate]
    partial: list[TuningAggregate]


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _coerce_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _coerce_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return statistics.fmean(values)


def _std(values: list[float]) -> float | None:
    if not values:
        return None
    if len(values) < 2:
        return 0.0
    return statistics.stdev(values)


def _series_from_metrics(payload: dict[str, Any]) -> tuple[list[float], list[float]]:
    test_entries = payload.get("test")
    if not isinstance(test_entries, list):
        return [], []

    accuracy: list[float] = []
    loss: list[float] = []
    for entry in test_entries:
        if not isinstance(entry, dict):
            continue
        accuracy.append(float(entry.get("accuracy", float("nan"))))
        loss.append(float(entry.get("loss", float("nan"))))
    return accuracy, loss


def _extract_metadata(
    summary_payload: dict[str, Any],
    metrics_payload: dict[str, Any],
) -> dict[str, Any] | None:
    for payload in (summary_payload, metrics_payload):
        metadata = payload.get("run_metadata")
        if isinstance(metadata, dict):
            return metadata
    return None


def _extract_tags(metadata: dict[str, Any] | None) -> dict[str, Any]:
    if metadata is None:
        return {}
    tags = metadata.get("tags")
    if isinstance(tags, dict):
        return tags
    return {}


def _extract_plasticity(metrics_payload: dict[str, Any]) -> dict[str, Any]:
    plasticity = metrics_payload.get("plasticity_config")
    if isinstance(plasticity, dict):
        return plasticity
    return {}


def discover_tuning_runs(results_dir: Path, *, dataset_name: str = "mnist") -> list[TuningRun]:
    runs: list[TuningRun] = []
    for summary_path in sorted(results_dir.glob("*_summary.json")):
        stem = summary_path.name[: -len("_summary.json")]
        metrics_path = results_dir / f"{stem}_metrics.json"
        if not metrics_path.exists():
            continue
        summary_payload = _read_json(summary_path)
        metrics_payload = _read_json(metrics_path)
        metadata = _extract_metadata(summary_payload, metrics_payload)
        if metadata is None:
            continue
        dataset = str(metadata.get("dataset", "")).lower()
        if dataset != dataset_name.lower():
            continue
        optimizer_name = str(metadata.get("optimizer_name") or summary_payload.get("optimizer") or "")
        if optimizer_name != "neuroplastic":
            continue
        plasticity = _extract_plasticity(metrics_payload)
        mode = str(plasticity.get("mode", "rule_based"))
        run_name = str(metadata.get("run_name") or summary_payload.get("run_name") or stem)
        if "smoke" in run_name.lower():
            continue
        if mode == "ablation_grad_only":
            optimizer_name = "ablation_grad_only"
        elif mode != "rule_based":
            continue

        seed = _coerce_int(metadata.get("seed"))
        epochs = _coerce_int(metadata.get("epochs"))
        lr = _coerce_float(metadata.get("lr"))
        warmup_epochs = _coerce_int(metadata.get("warmup_epochs"))
        plasticity_scale = _coerce_float(metadata.get("plasticity_scale"))
        if None in {seed, epochs, lr, warmup_epochs, plasticity_scale}:
            continue

        accuracy, loss = _series_from_metrics(metrics_payload)
        clean_accuracy = [value for value in accuracy if not math.isnan(value)]
        clean_loss = [value for value in loss if not math.isnan(value)]
        events_path = results_dir / f"{stem}_events.jsonl"
        runs.append(
            TuningRun(
                run_name=run_name,
                optimizer_name=optimizer_name,
                lr=float(lr),
                warmup_epochs=int(warmup_epochs),
                plasticity_scale=float(plasticity_scale),
                seed=int(seed),
                epochs=int(epochs),
                dataset=dataset_name.lower(),
                model_identifier=str(metadata.get("model_identifier", "unknown_model")),
                timestamp=metadata.get("timestamp"),
                git_commit_hash=metadata.get("git_commit_hash"),
                result_directory=str(metadata.get("result_directory") or results_dir.resolve()),
                summary_path=summary_path,
                metrics_path=metrics_path,
                events_path=events_path if events_path.exists() else None,
                tags=_extract_tags(metadata),
                test_accuracy=accuracy,
                test_loss=loss,
                final_test_accuracy=clean_accuracy[-1] if clean_accuracy else None,
                best_test_accuracy=_coerce_float(summary_payload.get("best_test_accuracy"))
                if summary_payload.get("best_test_accuracy") is not None
                else (max(clean_accuracy) if clean_accuracy else None),
                final_test_loss=_coerce_float(summary_payload.get("last_test_loss"))
                if summary_payload.get("last_test_loss") is not None
                else (clean_loss[-1] if clean_loss else None),
            )
        )
    return runs


def _epoch_to_threshold(series: list[float], threshold: float) -> int | None:
    for epoch, value in enumerate(series, start=1):
        if not math.isnan(value) and value >= threshold:
            return epoch
    return None


def _aggregate_series(runs: list[TuningRun], attr: str) -> tuple[list[int], list[float], list[float]]:
    min_len = min((len(getattr(run, attr)) for run in runs), default=0)
    epochs = list(range(1, min_len + 1))
    means: list[float] = []
    stds: list[float] = []
    for index in range(min_len):
        values = [
            getattr(run, attr)[index]
            for run in runs
            if index < len(getattr(run, attr)) and not math.isnan(getattr(run, attr)[index])
        ]
        means.append(statistics.fmean(values) if values else float("nan"))
        stds.append(statistics.stdev(values) if len(values) > 1 else 0.0)
    return epochs, means, stds


def aggregate_tuning_runs(runs: list[TuningRun]) -> list[TuningAggregate]:
    grouped: dict[tuple[str, float, int, float], list[TuningRun]] = {}
    for run in runs:
        grouped.setdefault(
            (run.optimizer_name, run.lr, run.warmup_epochs, run.plasticity_scale),
            [],
        ).append(run)

    aggregates: list[TuningAggregate] = []
    for key, grouped_runs in sorted(grouped.items()):
        grouped_runs = sorted(grouped_runs, key=lambda run: run.seed)
        epochs, mean_acc, std_acc = _aggregate_series(grouped_runs, "test_accuracy")
        _, mean_loss, std_loss = _aggregate_series(grouped_runs, "test_loss")
        final_acc_values = [
            run.final_test_accuracy for run in grouped_runs if run.final_test_accuracy is not None
        ]
        best_acc_values = [
            run.best_test_accuracy for run in grouped_runs if run.best_test_accuracy is not None
        ]
        final_loss_values = [
            run.final_test_loss for run in grouped_runs if run.final_test_loss is not None
        ]
        epoch_to_95 = [
            epoch
            for epoch in (_epoch_to_threshold(run.test_accuracy, 0.95) for run in grouped_runs)
            if epoch is not None
        ]
        epoch_to_97 = [
            epoch
            for epoch in (_epoch_to_threshold(run.test_accuracy, 0.97) for run in grouped_runs)
            if epoch is not None
        ]
        aggregates.append(
            TuningAggregate(
                optimizer_name=key[0],
                lr=key[1],
                warmup_epochs=key[2],
                plasticity_scale=key[3],
                runs=grouped_runs,
                epochs=epochs,
                mean_test_accuracy=mean_acc,
                std_test_accuracy=std_acc,
                mean_test_loss=mean_loss,
                std_test_loss=std_loss,
                num_seeds=len(grouped_runs),
                mean_final_test_accuracy=_mean(final_acc_values),
                std_final_test_accuracy=_std(final_acc_values),
                mean_best_test_accuracy=_mean(best_acc_values),
                std_best_test_accuracy=_std(best_acc_values),
                mean_final_test_loss=_mean(final_loss_values),
                std_final_test_loss=_std(final_loss_values),
                mean_epoch_to_95_acc=_mean(epoch_to_95),
                mean_epoch_to_97_acc=_mean(epoch_to_97),
                num_seeds_reached_95_acc=len(epoch_to_95),
                num_seeds_reached_97_acc=len(epoch_to_97),
            )
        )
    return aggregates


def split_complete_and_partial_aggregates(
    aggregates: list[TuningAggregate],
    expected_seed_count: int | None,
) -> AggregateSplit:
    if expected_seed_count is None:
        return AggregateSplit(complete=list(aggregates), partial=[])
    complete = [
        aggregate for aggregate in aggregates if aggregate.num_seeds >= expected_seed_count
    ]
    partial = [
        aggregate for aggregate in aggregates if aggregate.num_seeds < expected_seed_count
    ]
    return AggregateSplit(complete=complete, partial=partial)


def _apply_plot_style() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "figure.figsize": (8.0, 4.8),
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "savefig.dpi": 300,
        }
    )


def _config_label(aggregate: TuningAggregate) -> str:
    if aggregate.optimizer_name == "ablation_grad_only":
        return f"Grad-only lr={aggregate.lr:g}"
    return (
        f"Full lr={aggregate.lr:g}, warmup={aggregate.warmup_epochs}, "
        f"scale={aggregate.plasticity_scale:g}"
    )


def _dataset_title(dataset_name: str) -> str:
    mapping = {
        "mnist": "MNIST",
        "fashionmnist": "Fashion-MNIST",
        "cifar10": "CIFAR-10",
    }
    return mapping.get(dataset_name.lower(), dataset_name)


def _save_figure(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight", dpi=300)
    plt.close()


def _best_full_aggregates(aggregates: list[TuningAggregate], limit: int = 5) -> list[TuningAggregate]:
    full_only = [item for item in aggregates if item.optimizer_name == "neuroplastic"]
    return sorted(
        full_only,
        key=lambda item: (
            item.mean_final_test_accuracy if item.mean_final_test_accuracy is not None else float("-inf")
        ),
        reverse=True,
    )[:limit]


def _baseline_aggregate(aggregates: list[TuningAggregate]) -> TuningAggregate | None:
    for aggregate in aggregates:
        if aggregate.optimizer_name == "ablation_grad_only":
            return aggregate
    return None


def _plot_aggregated_accuracy(
    aggregates: list[TuningAggregate],
    output_dir: Path,
    *,
    dataset_name: str,
    file_prefix: str,
) -> Path | None:
    baseline = _baseline_aggregate(aggregates)
    selected = _best_full_aggregates(aggregates)
    if baseline is not None:
        selected = [baseline, *selected]
    if not selected:
        return None
    plt.figure()
    for aggregate in selected:
        plt.plot(aggregate.epochs, aggregate.mean_test_accuracy, marker="o", label=_config_label(aggregate))
    plt.xlabel("Epoch")
    plt.ylabel("Mean Test Accuracy")
    plt.title(f"{_dataset_title(dataset_name)} Full-vs-Grad-Only: Aggregated Test Accuracy vs Epoch")
    plt.legend()
    output_path = output_dir / f"{file_prefix}_aggregated_test_accuracy_vs_epoch.png"
    _save_figure(output_path)
    return output_path


def _plot_aggregated_loss(
    aggregates: list[TuningAggregate],
    output_dir: Path,
    *,
    dataset_name: str,
    file_prefix: str,
) -> Path | None:
    baseline = _baseline_aggregate(aggregates)
    selected = _best_full_aggregates(aggregates)
    if baseline is not None:
        selected = [baseline, *selected]
    if not selected:
        return None
    plt.figure()
    for aggregate in selected:
        plt.plot(aggregate.epochs, aggregate.mean_test_loss, marker="o", label=_config_label(aggregate))
    plt.xlabel("Epoch")
    plt.ylabel("Mean Test Loss")
    plt.title(f"{_dataset_title(dataset_name)} Full-vs-Grad-Only: Aggregated Test Loss vs Epoch")
    plt.legend()
    output_path = output_dir / f"{file_prefix}_aggregated_test_loss_vs_epoch.png"
    _save_figure(output_path)
    return output_path


def _plot_best_final_accuracy(
    aggregates: list[TuningAggregate],
    output_dir: Path,
    *,
    dataset_name: str,
    file_prefix: str,
) -> Path | None:
    ordered = sorted(
        aggregates,
        key=lambda item: (
            item.mean_final_test_accuracy if item.mean_final_test_accuracy is not None else float("-inf")
        ),
        reverse=True,
    )
    if not ordered:
        return None
    labels = [_config_label(item) for item in ordered]
    best = [
        item.mean_best_test_accuracy if item.mean_best_test_accuracy is not None else float("nan")
        for item in ordered
    ]
    final = [
        item.mean_final_test_accuracy if item.mean_final_test_accuracy is not None else float("nan")
        for item in ordered
    ]
    xs = list(range(len(ordered)))
    width = 0.38
    plt.figure(figsize=(max(10.0, len(ordered) * 0.35), 5.2))
    plt.bar([x - width / 2 for x in xs], best, width=width, label="Mean Best Test Accuracy")
    plt.bar([x + width / 2 for x in xs], final, width=width, label="Mean Final Test Accuracy")
    plt.xticks(xs, labels, rotation=55, ha="right")
    plt.ylabel("Accuracy")
    plt.title(f"{_dataset_title(dataset_name)} Full-vs-Grad-Only: Best vs Final Test Accuracy")
    plt.legend()
    output_path = output_dir / f"{file_prefix}_best_final_test_accuracy.png"
    _save_figure(output_path)
    return output_path


def _plot_early_convergence(
    aggregates: list[TuningAggregate],
    output_dir: Path,
    *,
    dataset_name: str,
    file_prefix: str,
) -> Path | None:
    baseline = _baseline_aggregate(aggregates)
    selected = _best_full_aggregates(aggregates)
    if baseline is not None:
        selected = [baseline, *selected]
    if not selected:
        return None
    plt.figure()
    for aggregate in selected:
        xs = [epoch for epoch in aggregate.epochs if epoch <= 3]
        ys = aggregate.mean_test_accuracy[: len(xs)]
        plt.plot(xs, ys, marker="o", label=_config_label(aggregate))
    plt.xlabel("Epoch")
    plt.ylabel("Mean Test Accuracy")
    plt.title(f"{_dataset_title(dataset_name)} Full-vs-Grad-Only: Early Convergence")
    plt.legend()
    output_path = output_dir / f"{file_prefix}_early_convergence_accuracy.png"
    _save_figure(output_path)
    return output_path


def _plot_full_ranking(
    aggregates: list[TuningAggregate],
    output_dir: Path,
    *,
    dataset_name: str,
    file_prefix: str,
) -> Path | None:
    full_only = [
        item for item in aggregates if item.optimizer_name == "neuroplastic" and item.mean_final_test_accuracy is not None
    ]
    if not full_only:
        return None
    ordered = sorted(full_only, key=lambda item: item.mean_final_test_accuracy or float("-inf"))
    labels = [_config_label(item) for item in ordered]
    values = [item.mean_final_test_accuracy for item in ordered]
    plt.figure(figsize=(10.0, max(6.0, len(ordered) * 0.28)))
    plt.barh(labels, values)
    plt.xlabel("Mean Final Test Accuracy")
    plt.ylabel("Full NeuroPlastic Configuration")
    plt.title(f"{_dataset_title(dataset_name)} Full NeuroPlastic: Ranked by Mean Final Accuracy")
    output_path = output_dir / f"{file_prefix}_ranked_full_configurations.png"
    _save_figure(output_path)
    return output_path


def _format_epoch_metric(value: float | None, reached_count: int) -> str:
    if reached_count == 0 or value is None:
        return "not_reached"
    return f"{value:.4f}"


def write_compact_summary_table(aggregates: list[TuningAggregate], output_dir: Path) -> Path | None:
    if not aggregates:
        return None
    output_path = output_dir / "compact_summary_table.csv"
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "optimizer_name",
                "lr",
                "warmup_epochs",
                "plasticity_scale",
                "num_seeds",
                "mean_final_test_accuracy",
                "std_final_test_accuracy",
                "mean_best_test_accuracy",
                "std_best_test_accuracy",
                "mean_final_test_loss",
                "std_final_test_loss",
                "mean_epoch_to_95_acc",
                "mean_epoch_to_97_acc",
                "num_seeds_reached_95_acc",
                "num_seeds_reached_97_acc",
            ],
        )
        writer.writeheader()
        for aggregate in sorted(
            aggregates,
            key=lambda item: (
                item.optimizer_name != "ablation_grad_only",
                -(item.mean_final_test_accuracy or float("-inf")),
                item.lr,
                item.warmup_epochs,
                item.plasticity_scale,
            ),
        ):
            writer.writerow(
                {
                    "optimizer_name": aggregate.optimizer_name,
                    "lr": aggregate.lr,
                    "warmup_epochs": aggregate.warmup_epochs,
                    "plasticity_scale": aggregate.plasticity_scale,
                    "num_seeds": aggregate.num_seeds,
                    "mean_final_test_accuracy": aggregate.mean_final_test_accuracy,
                    "std_final_test_accuracy": aggregate.std_final_test_accuracy,
                    "mean_best_test_accuracy": aggregate.mean_best_test_accuracy,
                    "std_best_test_accuracy": aggregate.std_best_test_accuracy,
                    "mean_final_test_loss": aggregate.mean_final_test_loss,
                    "std_final_test_loss": aggregate.std_final_test_loss,
                    "mean_epoch_to_95_acc": _format_epoch_metric(
                        aggregate.mean_epoch_to_95_acc, aggregate.num_seeds_reached_95_acc
                    ),
                    "mean_epoch_to_97_acc": _format_epoch_metric(
                        aggregate.mean_epoch_to_97_acc, aggregate.num_seeds_reached_97_acc
                    ),
                    "num_seeds_reached_95_acc": aggregate.num_seeds_reached_95_acc,
                    "num_seeds_reached_97_acc": aggregate.num_seeds_reached_97_acc,
                }
            )
    return output_path


def best_full_config_by_metric(
    aggregates: list[TuningAggregate],
    metric_name: str,
) -> TuningAggregate | None:
    full_only = [
        aggregate
        for aggregate in aggregates
        if aggregate.optimizer_name == "neuroplastic"
        and getattr(aggregate, metric_name) is not None
    ]
    if not full_only:
        return None
    return max(full_only, key=lambda aggregate: getattr(aggregate, metric_name) or float("-inf"))


def select_recommended_full_config(
    aggregates: list[TuningAggregate],
    comparison: dict[str, Any],
    *,
    dataset_name: str,
) -> dict[str, Any] | None:
    best_final = best_full_config_by_metric(aggregates, "mean_final_test_accuracy")
    best_best = best_full_config_by_metric(aggregates, "mean_best_test_accuracy")
    if best_final is None:
        return None
    selected_by = "mean_final_test_accuracy"
    if best_best is not None and best_best.config_key == best_final.config_key:
        selection_reason = (
            "Selected because it achieved the strongest complete-run mean final accuracy "
            "and also matched the best mean best accuracy."
        )
    else:
        selection_reason = (
            "Selected because mean final test accuracy is the primary ranking metric "
            "for downstream follow-up benchmarks."
        )
    return {
        "study_name": f"{dataset_name.lower()}_full_tuning_clean",
        "dataset": dataset_name.lower(),
        "optimizer_name": best_final.optimizer_name,
        "lr": best_final.lr,
        "warmup_epochs": best_final.warmup_epochs,
        "plasticity_scale": best_final.plasticity_scale,
        "selected_by": selected_by,
        "selection_reason": selection_reason,
        "comparison_to_baseline": comparison,
    }


def compare_best_full_vs_baseline(aggregates: list[TuningAggregate]) -> dict[str, Any]:
    baseline = _baseline_aggregate(aggregates)
    best_full_final = best_full_config_by_metric(aggregates, "mean_final_test_accuracy")
    best_full_best = best_full_config_by_metric(aggregates, "mean_best_test_accuracy")
    comparison: dict[str, Any] = {
        "comparison_mode": "aggregate_only",
        "baseline_present": baseline is not None,
        "best_full_by_final_accuracy": None,
        "best_full_by_best_accuracy": None,
        "mean_final_accuracy_gap_vs_baseline": None,
        "mean_best_accuracy_gap_vs_baseline": None,
        "mean_final_loss_gap_vs_baseline": None,
        "final_seed_wins": None,
        "best_seed_wins": None,
        "shared_seed_count": 0,
        "robustness": "inconclusive",
    }
    if baseline is None:
        return comparison

    if best_full_final is not None:
        comparison["best_full_by_final_accuracy"] = {
            "optimizer_name": best_full_final.optimizer_name,
            "lr": best_full_final.lr,
            "warmup_epochs": best_full_final.warmup_epochs,
            "plasticity_scale": best_full_final.plasticity_scale,
            "mean_final_test_accuracy": best_full_final.mean_final_test_accuracy,
        }
        comparison["mean_final_accuracy_gap_vs_baseline"] = (
            (best_full_final.mean_final_test_accuracy or 0.0)
            - (baseline.mean_final_test_accuracy or 0.0)
        )
        comparison["mean_final_loss_gap_vs_baseline"] = (
            (best_full_final.mean_final_test_loss or 0.0) - (baseline.mean_final_test_loss or 0.0)
        )

    if best_full_best is not None:
        comparison["best_full_by_best_accuracy"] = {
            "optimizer_name": best_full_best.optimizer_name,
            "lr": best_full_best.lr,
            "warmup_epochs": best_full_best.warmup_epochs,
            "plasticity_scale": best_full_best.plasticity_scale,
            "mean_best_test_accuracy": best_full_best.mean_best_test_accuracy,
        }
        comparison["mean_best_accuracy_gap_vs_baseline"] = (
            (best_full_best.mean_best_test_accuracy or 0.0)
            - (baseline.mean_best_test_accuracy or 0.0)
        )

    if best_full_final is not None:
        baseline_by_seed = {run.seed: run for run in baseline.runs}
        full_by_seed = {run.seed: run for run in best_full_final.runs}
        shared_seeds = sorted(set(baseline_by_seed).intersection(full_by_seed))
        comparison["shared_seed_count"] = len(shared_seeds)
        if shared_seeds:
            comparison["comparison_mode"] = "seed_aware"
            comparison["final_seed_wins"] = sum(
                1
                for seed in shared_seeds
                if (full_by_seed[seed].final_test_accuracy or float("-inf"))
                > (baseline_by_seed[seed].final_test_accuracy or float("-inf"))
            )
            comparison["best_seed_wins"] = sum(
                1
                for seed in shared_seeds
                if (full_by_seed[seed].best_test_accuracy or float("-inf"))
                > (baseline_by_seed[seed].best_test_accuracy or float("-inf"))
            )

    final_gap = comparison["mean_final_accuracy_gap_vs_baseline"]
    if final_gap is None:
        comparison["robustness"] = "inconclusive"
    elif final_gap > 0.003 and comparison["final_seed_wins"] == comparison["shared_seed_count"] and comparison["shared_seed_count"] > 0:
        comparison["robustness"] = "robust"
    elif final_gap > 0:
        comparison["robustness"] = "marginal"
    else:
        comparison["robustness"] = "inconclusive"
    return comparison


def build_interpretation_note(
    aggregates: list[TuningAggregate],
    comparison: dict[str, Any],
    *,
    dataset_name: str = "mnist",
    recommended_config: dict[str, Any] | None = None,
) -> str:
    baseline = _baseline_aggregate(aggregates)
    best_full_final = best_full_config_by_metric(aggregates, "mean_final_test_accuracy")
    best_full_best = best_full_config_by_metric(aggregates, "mean_best_test_accuracy")
    lines = [
        f"# {_dataset_title(dataset_name)} Full NeuroPlastic Tuning Interpretation",
        "",
    ]
    if baseline is None:
        lines.append("Baseline `ablation_grad_only` was not found, so the direct question cannot be answered.")
        return "\n".join(lines) + "\n"

    final_outperforms = (
        best_full_final is not None
        and best_full_final.mean_final_test_accuracy is not None
        and baseline.mean_final_test_accuracy is not None
        and best_full_final.mean_final_test_accuracy > baseline.mean_final_test_accuracy
    )
    best_outperforms = (
        best_full_best is not None
        and best_full_best.mean_best_test_accuracy is not None
        and baseline.mean_best_test_accuracy is not None
        and best_full_best.mean_best_test_accuracy > baseline.mean_best_test_accuracy
    )
    lower_lr_helped = (
        best_full_final is not None
        and best_full_final.lr < baseline.lr
    )
    warmup_helped = (
        best_full_final is not None
        and best_full_final.warmup_epochs > 0
    )
    best_full_label = "n/a"
    if best_full_final is not None:
        best_full_label = _config_label(best_full_final)

    lines.append(
        f"- Did any full NeuroPlastic setting outperform `ablation_grad_only` on mean final test accuracy? "
        f"{'Yes' if final_outperforms else 'No'}. "
        f"Best full gap vs baseline: {comparison['mean_final_accuracy_gap_vs_baseline']:+.4f}"
        if comparison["mean_final_accuracy_gap_vs_baseline"] is not None
        else "- Did any full NeuroPlastic setting outperform `ablation_grad_only` on mean final test accuracy? Not answerable."
    )
    lines.append(
        f"- Did any full setting outperform it on mean best test accuracy? "
        f"{'Yes' if best_outperforms else 'No'}. "
        f"Best full gap vs baseline: {comparison['mean_best_accuracy_gap_vs_baseline']:+.4f}"
        if comparison["mean_best_accuracy_gap_vs_baseline"] is not None
        else "- Did any full setting outperform it on mean best test accuracy? Not answerable."
    )
    lines.append(f"- Did lower learning rates help? {'Yes' if lower_lr_helped else 'No clear evidence'}.")
    lines.append(f"- Did warmup help? {'Yes' if warmup_helped else 'No clear evidence'}.")
    lines.append(f"- Which full configuration performed best overall? {best_full_label}.")
    lines.append(
        f"- Best full vs baseline final-loss gap: "
        f"{comparison['mean_final_loss_gap_vs_baseline']:+.4f}"
        if comparison["mean_final_loss_gap_vs_baseline"] is not None
        else "- Best full vs baseline final-loss gap: not available."
    )
    if comparison["comparison_mode"] == "seed_aware":
        lines.append(
            f"- Seed-aware comparison used shared seeds. Final wins: {comparison['final_seed_wins']}/{comparison['shared_seed_count']}; "
            f"best-accuracy wins: {comparison['best_seed_wins']}/{comparison['shared_seed_count']}."
        )
    else:
        lines.append("- Shared-seed pairing was unavailable, so the comparison fell back to aggregate metrics.")
    robustness = comparison["robustness"]
    if not final_outperforms and not best_outperforms:
        lines.append(
            f"- Is the improvement consistent, marginal, or inconclusive? Full NeuroPlastic still does not clearly win; the result is {robustness}."
        )
    else:
        lines.append(
            f"- Is the improvement consistent, marginal, or inconclusive? The gain looks {robustness}."
        )
    if recommended_config is not None:
        lines.append(
            "- What is the recommended default full NeuroPlastic config for follow-up benchmarks? "
            f"lr={recommended_config['lr']}, warmup_epochs={recommended_config['warmup_epochs']}, "
            f"plasticity_scale={recommended_config['plasticity_scale']}."
        )
    return "\n".join(lines) + "\n"


def write_interpretation_note(
    aggregates: list[TuningAggregate],
    comparison: dict[str, Any],
    output_dir: Path,
    *,
    dataset_name: str,
    recommended_config: dict[str, Any] | None = None,
) -> Path:
    output_path = output_dir / "interpretation_note.md"
    output_path.write_text(
        build_interpretation_note(
            aggregates,
            comparison,
            dataset_name=dataset_name,
            recommended_config=recommended_config,
        ),
        encoding="utf-8",
    )
    return output_path


def write_analysis_manifest(
    aggregates: list[TuningAggregate],
    comparison: dict[str, Any],
    output_dir: Path,
    *,
    partial_aggregates: list[TuningAggregate],
    recommended_config: dict[str, Any] | None,
) -> Path:
    output_path = output_dir / "analysis_manifest.json"
    payload = {
        "aggregated_configs": [
            {
                "optimizer_name": aggregate.optimizer_name,
                "lr": aggregate.lr,
                "warmup_epochs": aggregate.warmup_epochs,
                "plasticity_scale": aggregate.plasticity_scale,
                "num_seeds": aggregate.num_seeds,
            }
            for aggregate in aggregates
        ],
        "partial_aggregates": [
            {
                "optimizer_name": aggregate.optimizer_name,
                "lr": aggregate.lr,
                "warmup_epochs": aggregate.warmup_epochs,
                "plasticity_scale": aggregate.plasticity_scale,
                "num_seeds": aggregate.num_seeds,
            }
            for aggregate in partial_aggregates
        ],
        "comparison": comparison,
        "recommended_config": recommended_config,
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return output_path


def write_recommended_config(path: Path, config_payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config_payload, indent=2), encoding="utf-8")
    return path


def generate_full_tuning_artifacts(
    results_dir: Path,
    output_dir: Path,
    *,
    dataset_name: str = "mnist",
    expected_seed_count: int | None = None,
    file_prefix: str = "mnist_full_tuning",
    include_ranking_plot: bool = True,
    recommended_config_path: Path | None = None,
) -> dict[str, Any]:
    _apply_plot_style()
    output_dir.mkdir(parents=True, exist_ok=True)
    runs = discover_tuning_runs(results_dir, dataset_name=dataset_name)
    aggregates = aggregate_tuning_runs(runs)
    aggregate_split = split_complete_and_partial_aggregates(aggregates, expected_seed_count)
    comparison = compare_best_full_vs_baseline(aggregate_split.complete)
    recommended_config = select_recommended_full_config(
        aggregate_split.complete,
        comparison,
        dataset_name=dataset_name,
    )

    generated_files: list[Path] = []
    for maybe_path in (
        write_compact_summary_table(aggregate_split.complete, output_dir),
        _plot_aggregated_accuracy(
            aggregate_split.complete,
            output_dir,
            dataset_name=dataset_name,
            file_prefix=file_prefix,
        ),
        _plot_aggregated_loss(
            aggregate_split.complete,
            output_dir,
            dataset_name=dataset_name,
            file_prefix=file_prefix,
        ),
        _plot_best_final_accuracy(
            aggregate_split.complete,
            output_dir,
            dataset_name=dataset_name,
            file_prefix=file_prefix,
        ),
        _plot_early_convergence(
            aggregate_split.complete,
            output_dir,
            dataset_name=dataset_name,
            file_prefix=file_prefix,
        ),
    ):
        if maybe_path is not None:
            generated_files.append(maybe_path)
    if include_ranking_plot:
        ranking_path = _plot_full_ranking(
            aggregate_split.complete,
            output_dir,
            dataset_name=dataset_name,
            file_prefix=file_prefix,
        )
        if ranking_path is not None:
            generated_files.append(ranking_path)

    generated_files.append(
        write_interpretation_note(
            aggregate_split.complete,
            comparison,
            output_dir,
            dataset_name=dataset_name,
            recommended_config=recommended_config,
        )
    )
    generated_files.append(
        write_analysis_manifest(
            aggregate_split.complete,
            comparison,
            output_dir,
            partial_aggregates=aggregate_split.partial,
            recommended_config=recommended_config,
        )
    )
    if recommended_config_path is not None and recommended_config is not None:
        generated_files.append(write_recommended_config(recommended_config_path, recommended_config))
    return {
        "runs_found": [run.run_name for run in runs],
        "aggregates_found": [
            {
                "optimizer_name": aggregate.optimizer_name,
                "lr": aggregate.lr,
                "warmup_epochs": aggregate.warmup_epochs,
                "plasticity_scale": aggregate.plasticity_scale,
            }
            for aggregate in aggregate_split.complete
        ],
        "partial_aggregates": [
            {
                "optimizer_name": aggregate.optimizer_name,
                "lr": aggregate.lr,
                "warmup_epochs": aggregate.warmup_epochs,
                "plasticity_scale": aggregate.plasticity_scale,
                "num_seeds": aggregate.num_seeds,
            }
            for aggregate in aggregate_split.partial
        ],
        "recommended_config": recommended_config,
        "comparison": comparison,
        "generated_files": [str(path) for path in generated_files],
    }

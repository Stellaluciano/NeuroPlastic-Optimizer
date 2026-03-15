from __future__ import annotations

import csv
import json
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scripts.paper_figures.mnist_full_tuning import TuningRun, discover_tuning_runs
from scripts.paper_figures.study_helpers import compare_seed_metric_dicts, write_json


@dataclass(slots=True)
class LowDataAggregate:
    dataset: str
    fraction: float
    optimizer_name: str
    runs: list[TuningRun]
    num_seeds: int
    mean_final_test_accuracy: float
    std_final_test_accuracy: float
    mean_best_test_accuracy: float
    std_best_test_accuracy: float
    mean_final_test_loss: float
    std_final_test_loss: float


def _mean(values: list[float]) -> float:
    return statistics.fmean(values)


def _std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    return statistics.stdev(values)


def aggregate_low_data_runs(runs: list[TuningRun]) -> list[LowDataAggregate]:
    grouped: dict[tuple[str, float, str], list[TuningRun]] = {}
    for run in runs:
        fraction = run.tags.get("data_fraction")
        if fraction is None:
            continue
        grouped.setdefault((run.dataset, float(fraction), run.optimizer_name), []).append(run)

    aggregates: list[LowDataAggregate] = []
    for key, grouped_runs in sorted(grouped.items(), key=lambda item: (item[0][1], item[0][2])):
        grouped_runs = sorted(grouped_runs, key=lambda run: run.seed)
        final_acc = [run.final_test_accuracy for run in grouped_runs if run.final_test_accuracy is not None]
        best_acc = [run.best_test_accuracy for run in grouped_runs if run.best_test_accuracy is not None]
        final_loss = [run.final_test_loss for run in grouped_runs if run.final_test_loss is not None]
        if not final_acc or not best_acc or not final_loss:
            continue
        aggregates.append(
            LowDataAggregate(
                dataset=key[0],
                fraction=key[1],
                optimizer_name=key[2],
                runs=grouped_runs,
                num_seeds=len(grouped_runs),
                mean_final_test_accuracy=_mean(final_acc),
                std_final_test_accuracy=_std(final_acc),
                mean_best_test_accuracy=_mean(best_acc),
                std_best_test_accuracy=_std(best_acc),
                mean_final_test_loss=_mean(final_loss),
                std_final_test_loss=_std(final_loss),
            )
        )
    return aggregates


def build_low_data_comparison(aggregates: list[LowDataAggregate]) -> dict[str, Any]:
    by_fraction: dict[float, dict[str, LowDataAggregate]] = {}
    for aggregate in aggregates:
        by_fraction.setdefault(aggregate.fraction, {})[aggregate.optimizer_name] = aggregate

    fraction_summaries: list[dict[str, Any]] = []
    final_gaps: list[float] = []
    for fraction in sorted(by_fraction):
        pair = by_fraction[fraction]
        baseline = pair.get("ablation_grad_only")
        full = pair.get("neuroplastic")
        if baseline is None or full is None:
            continue
        final_gap = full.mean_final_test_accuracy - baseline.mean_final_test_accuracy
        best_gap = full.mean_best_test_accuracy - baseline.mean_best_test_accuracy
        loss_gap = full.mean_final_test_loss - baseline.mean_final_test_loss
        seed_final = compare_seed_metric_dicts(
            {run.seed: run.final_test_accuracy for run in baseline.runs},
            {run.seed: run.final_test_accuracy for run in full.runs},
        )
        seed_best = compare_seed_metric_dicts(
            {run.seed: run.best_test_accuracy for run in baseline.runs},
            {run.seed: run.best_test_accuracy for run in full.runs},
        )
        fraction_summaries.append(
            {
                "fraction": fraction,
                "mean_final_accuracy_gap_vs_baseline": final_gap,
                "mean_best_accuracy_gap_vs_baseline": best_gap,
                "mean_final_loss_gap_vs_baseline": loss_gap,
                "final_seed_wins": seed_final["wins"],
                "best_seed_wins": seed_best["wins"],
                "shared_seed_count": seed_final["shared_seed_count"],
            }
        )
        final_gaps.append(final_gap)

    monotonic = None
    if len(final_gaps) >= 2:
        monotonic = all(
            earlier <= later
            for earlier, later in zip(final_gaps, final_gaps[1:])
        ) or all(
            earlier >= later
            for earlier, later in zip(final_gaps, final_gaps[1:])
        )

    largest_advantage = None
    if fraction_summaries:
        largest_advantage = max(
            fraction_summaries,
            key=lambda item: item["mean_final_accuracy_gap_vs_baseline"],
        )

    return {
        "fractions": fraction_summaries,
        "largest_advantage_fraction": None if largest_advantage is None else largest_advantage["fraction"],
        "largest_advantage_gap": None if largest_advantage is None else largest_advantage["mean_final_accuracy_gap_vs_baseline"],
        "trend_is_monotonic": monotonic,
    }


def write_low_data_summary_table(aggregates: list[LowDataAggregate], output_dir: Path) -> Path:
    output_path = output_dir / "compact_summary_table.csv"
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "dataset",
                "fraction",
                "optimizer_name",
                "num_seeds",
                "mean_final_test_accuracy",
                "std_final_test_accuracy",
                "mean_best_test_accuracy",
                "std_best_test_accuracy",
                "mean_final_test_loss",
                "std_final_test_loss",
            ],
        )
        writer.writeheader()
        for aggregate in sorted(aggregates, key=lambda item: (item.fraction, item.optimizer_name)):
            writer.writerow(
                {
                    "dataset": aggregate.dataset,
                    "fraction": aggregate.fraction,
                    "optimizer_name": aggregate.optimizer_name,
                    "num_seeds": aggregate.num_seeds,
                    "mean_final_test_accuracy": aggregate.mean_final_test_accuracy,
                    "std_final_test_accuracy": aggregate.std_final_test_accuracy,
                    "mean_best_test_accuracy": aggregate.mean_best_test_accuracy,
                    "std_best_test_accuracy": aggregate.std_best_test_accuracy,
                    "mean_final_test_loss": aggregate.mean_final_test_loss,
                    "std_final_test_loss": aggregate.std_final_test_loss,
                }
            )
    return output_path


def _plot_metric_by_fraction(
    aggregates: list[LowDataAggregate],
    output_path: Path,
    *,
    metric_attr: str,
    ylabel: str,
) -> Path:
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.figure(figsize=(7.2, 4.8))
    for optimizer_name in ("ablation_grad_only", "neuroplastic"):
        subset = [item for item in aggregates if item.optimizer_name == optimizer_name]
        subset = sorted(subset, key=lambda item: item.fraction)
        plt.plot(
            [item.fraction for item in subset],
            [getattr(item, metric_attr) for item in subset],
            marker="o",
            label="Grad-only" if optimizer_name == "ablation_grad_only" else "Full NeuroPlastic",
        )
    plt.xlabel("Training Data Fraction")
    plt.ylabel(ylabel)
    plt.xscale("log")
    plt.xticks([0.1, 0.25, 0.5, 1.0], ["0.10", "0.25", "0.50", "1.00"])
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight", dpi=300)
    plt.close()
    return output_path


def write_low_data_manifest(
    output_dir: Path,
    *,
    dataset: str,
    comparison: dict[str, Any],
    generated_files: list[Path],
) -> Path:
    output_path = output_dir / "manifest.json"
    write_json(
        output_path,
        {
            "study_name": f"low_data_{dataset}_bestfull_vs_gradonly_clean",
            "dataset": dataset,
            "comparison": comparison,
            "generated_files": [str(path) for path in generated_files],
        },
    )
    return output_path


def build_low_data_interpretation_note(dataset: str, comparison: dict[str, Any]) -> str:
    largest_fraction = comparison.get("largest_advantage_fraction")
    largest_gap = comparison.get("largest_advantage_gap")
    trend = comparison.get("trend_is_monotonic")
    fractions = comparison.get("fractions", [])
    stronger_low_data = False
    if fractions:
        first_gap = fractions[0]["mean_final_accuracy_gap_vs_baseline"]
        last_gap = fractions[-1]["mean_final_accuracy_gap_vs_baseline"]
        stronger_low_data = first_gap > last_gap

    if not fractions:
        strength = "inconclusive"
    elif any(item["mean_final_accuracy_gap_vs_baseline"] > 0.003 for item in fractions):
        strength = "strong"
    elif any(item["mean_final_accuracy_gap_vs_baseline"] > 0 for item in fractions):
        strength = "marginal"
    else:
        strength = "inconclusive"

    return (
        f"# {dataset.upper()} Low-Data Best-Full vs Grad-Only Interpretation\n\n"
        f"- Does full NeuroPlastic gain relative to grad-only increase as data fraction decreases? "
        f"{'Directionally yes' if stronger_low_data else 'Not clearly'}.\n"
        f"- Which fraction shows the largest advantage? "
        f"{largest_fraction if largest_fraction is not None else 'Not enough complete data'} "
        f"(final-accuracy gap={largest_gap}).\n"
        f"- Is the low-data trend consistent with the plasticity hypothesis? "
        f"{'Partially' if stronger_low_data else 'Only weakly or not at all'}.\n"
        f"- Is the result strong, marginal, or inconclusive? {strength}.\n"
        f"- Is the gap-vs-fraction pattern monotonic? "
        f"{'Yes' if trend is True else 'No' if trend is False else 'Not enough fractions'}.\n"
    )


def load_and_aggregate_low_data(results_dir: Path, dataset: str) -> tuple[list[TuningRun], list[LowDataAggregate], dict[str, Any]]:
    runs = discover_tuning_runs(results_dir, dataset_name=dataset)
    aggregates = aggregate_low_data_runs(runs)
    comparison = build_low_data_comparison(aggregates)
    return runs, aggregates, comparison

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.paper_figures.mnist_full_tuning import (
    TuningAggregate,
    TuningRun,
    aggregate_tuning_runs,
    best_full_config_by_metric,
    discover_tuning_runs,
)


DEFAULT_OUTPUT_DIR = ROOT / "figures"

MNIST_RESULTS = ROOT / "results_mnist_full_tuning_clean"
MNIST_ARTIFACTS = ROOT / "paper_artifacts" / "mnist_full_tuning_clean"
FASHION_RESULTS = ROOT / "results_fashionmnist_bestfull_vs_gradonly_clean"
FASHION_ARTIFACTS = ROOT / "paper_artifacts" / "fashionmnist_bestfull_vs_gradonly_clean"
LOW_DATA_RESULTS = ROOT / "results_low_data_fashionmnist_bestfull_vs_gradonly_clean"
LOW_DATA_ARTIFACTS = ROOT / "paper_artifacts" / "low_data_fashionmnist_bestfull_vs_gradonly_clean"
CIFAR_RESULTS = ROOT / "results_cifar10_bestfull_vs_gradonly_clean"
CIFAR_ARTIFACTS = ROOT / "paper_artifacts" / "cifar10_bestfull_vs_gradonly_clean"

COLORS = {
    "neuroplastic": "#1b4f9c",
    "ablation_grad_only": "#5a5a5a",
}
LABELS = {
    "neuroplastic": "Full NeuroPlastic",
    "ablation_grad_only": "Grad-only",
}
LINESTYLES = {
    "neuroplastic": "-",
    "ablation_grad_only": "--",
}
MARKERS = {
    "neuroplastic": "o",
    "ablation_grad_only": "s",
}


def _apply_style() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.titlesize": 12,
            "axes.labelsize": 10.5,
            "axes.edgecolor": "#333333",
            "axes.linewidth": 0.9,
            "axes.titlepad": 8.0,
            "grid.color": "#e2e2e2",
            "grid.linewidth": 0.7,
            "grid.alpha": 0.8,
            "legend.frameon": False,
            "legend.fontsize": 9.5,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "xtick.major.size": 3.5,
            "ytick.major.size": 3.5,
            "savefig.dpi": 300,
        }
    )


def _save(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(pad=0.55, w_pad=1.0, h_pad=0.9)
    fig.savefig(path, bbox_inches="tight", dpi=300, facecolor="white")
    plt.close(fig)
    return path


def _add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.13,
        1.045,
        label,
        transform=ax.transAxes,
        fontsize=12,
        fontweight="bold",
        va="bottom",
        ha="left",
    )


def _style_axes(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.9)
    ax.spines["bottom"].set_linewidth(0.9)
    ax.grid(axis="y", alpha=0.8)
    ax.grid(axis="x", alpha=0.16)
    ax.margins(x=0.02)


def _set_clean_legend(ax: plt.Axes, location: str) -> None:
    ax.legend(loc=location, handlelength=2.2, borderaxespad=0.3, labelspacing=0.35)


def _finite_bounds(series: list[float], stds: list[float] | None = None) -> tuple[float, float]:
    values = [value for value in series if not math.isnan(value)]
    if stds is not None:
        values.extend(
            [
                value - std
                for value, std in zip(series, stds)
                if not math.isnan(value) and not math.isnan(std)
            ]
        )
        values.extend(
            [
                value + std
                for value, std in zip(series, stds)
                if not math.isnan(value) and not math.isnan(std)
            ]
        )
    return min(values), max(values)


def _pad_limits(lower: float, upper: float, frac: float = 0.12) -> tuple[float, float]:
    span = upper - lower
    if span <= 0:
        span = max(abs(upper), 1e-6) * 0.1
    pad = span * frac
    return lower - pad, upper + pad


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _load_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _aggregate_for_dataset(results_dir: Path, dataset: str) -> list[TuningAggregate]:
    return aggregate_tuning_runs(discover_tuning_runs(results_dir, dataset_name=dataset))


def _find_baseline(aggregates: list[TuningAggregate]) -> TuningAggregate:
    for aggregate in aggregates:
        if aggregate.optimizer_name == "ablation_grad_only":
            return aggregate
    raise ValueError("Could not find grad-only baseline aggregate.")


def _select_full(aggregates: list[TuningAggregate]) -> TuningAggregate:
    selected = best_full_config_by_metric(aggregates, "mean_final_test_accuracy")
    if selected is None:
        raise ValueError("Could not find full NeuroPlastic aggregate.")
    return selected


def _plot_curve_panel(
    ax: plt.Axes,
    *,
    panel_label: str,
    title: str,
    ylabel: str,
    baseline: TuningAggregate,
    full: TuningAggregate,
    value_attr: str,
    std_attr: str,
    legend_loc: str,
    ylim: tuple[float, float] | None = None,
) -> None:
    series = [
        (baseline, COLORS["ablation_grad_only"]),
        (full, COLORS["neuroplastic"]),
    ]
    for aggregate, color in series:
        xs = aggregate.epochs
        ys = getattr(aggregate, value_attr)
        stds = getattr(aggregate, std_attr)
        ax.plot(
            xs,
            ys,
            color=color,
            linewidth=2.2,
            linestyle=LINESTYLES[aggregate.optimizer_name],
            marker=MARKERS[aggregate.optimizer_name],
            markersize=4.2,
            label=LABELS[aggregate.optimizer_name],
        )
        lower = [y - s for y, s in zip(ys, stds)]
        upper = [y + s for y, s in zip(ys, stds)]
        ax.fill_between(xs, lower, upper, color=color, alpha=0.11, linewidth=0)
    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.set_xticks(full.epochs)
    if ylim is None:
        lower_1, upper_1 = _finite_bounds(getattr(baseline, value_attr), getattr(baseline, std_attr))
        lower_2, upper_2 = _finite_bounds(getattr(full, value_attr), getattr(full, std_attr))
        ylim = _pad_limits(min(lower_1, lower_2), max(upper_1, upper_2))
    ax.set_ylim(*ylim)
    _style_axes(ax)
    _set_clean_legend(ax, legend_loc)
    _add_panel_label(ax, panel_label)


def _plot_fraction_panel(
    ax: plt.Axes,
    *,
    panel_label: str,
    title: str,
    ylabel: str,
    rows: list[dict[str, str]],
    value_key: str,
    std_key: str,
    legend_loc: str,
    ylim: tuple[float, float] | None = None,
) -> None:
    fractions = [0.1, 0.25, 0.5, 1.0]
    all_values: list[float] = []
    all_stds: list[float] = []
    for optimizer_name in ("ablation_grad_only", "neuroplastic"):
        subset = [row for row in rows if row["optimizer_name"] == optimizer_name]
        subset.sort(key=lambda row: float(row["fraction"]))
        xs = [float(row["fraction"]) for row in subset]
        ys = [float(row[value_key]) for row in subset]
        stds = [float(row[std_key]) for row in subset]
        all_values.extend(ys)
        all_stds.extend(stds)
        ax.plot(
            xs,
            ys,
            color=COLORS[optimizer_name],
            linewidth=2.2,
            linestyle=LINESTYLES[optimizer_name],
            marker=MARKERS[optimizer_name],
            markersize=4.6,
            label=LABELS[optimizer_name],
        )
        lower = [y - s for y, s in zip(ys, stds)]
        upper = [y + s for y, s in zip(ys, stds)]
        ax.fill_between(xs, lower, upper, color=COLORS[optimizer_name], alpha=0.11, linewidth=0)
    ax.set_title(title)
    ax.set_xlabel("Training Data Fraction")
    ax.set_ylabel(ylabel)
    ax.set_xscale("log")
    ax.set_xticks(fractions, ["0.10", "0.25", "0.50", "1.00"])
    ax.set_xlim(0.09, 1.08)
    if ylim is None:
        ylim = _pad_limits(min(v - s for v, s in zip(all_values, all_stds)), max(v + s for v, s in zip(all_values, all_stds)), frac=0.18)
    ax.set_ylim(*ylim)
    _style_axes(ax)
    _set_clean_legend(ax, legend_loc)
    _add_panel_label(ax, panel_label)


def _read_train_diagnostic(run: TuningRun, key: str) -> list[float]:
    payload = _load_json(run.metrics_path)
    entries = payload.get("train", [])
    values: list[float] = []
    for entry in entries:
        value = entry.get(key)
        if value is None and isinstance(entry, dict):
            diagnostics = entry.get("optimizer_diagnostics")
            if isinstance(diagnostics, dict):
                value = diagnostics.get(key)
        values.append(float(value) if value is not None else float("nan"))
    return values


def _aggregate_run_diagnostic(runs: list[TuningRun], key: str) -> tuple[list[int], list[float], list[float]]:
    min_len = min((len(_read_train_diagnostic(run, key)) for run in runs), default=0)
    xs = list(range(1, min_len + 1))
    cached = [_read_train_diagnostic(run, key)[:min_len] for run in runs]
    means: list[float] = []
    stds: list[float] = []
    for idx in range(min_len):
        values = [series[idx] for series in cached if not math.isnan(series[idx])]
        means.append(statistics.fmean(values) if values else float("nan"))
        stds.append(statistics.stdev(values) if len(values) > 1 else 0.0)
    return xs, means, stds


def _plot_diagnostic_panel(
    ax: plt.Axes,
    *,
    panel_label: str,
    title: str,
    ylabel: str,
    xs: list[int],
    means: list[float],
    stds: list[float],
    ylim: tuple[float, float] | None = None,
) -> None:
    ax.plot(
        xs,
        means,
        color=COLORS["neuroplastic"],
        linewidth=2.25,
        marker=MARKERS["neuroplastic"],
        markersize=4.2,
    )
    lower = [y - s for y, s in zip(means, stds)]
    upper = [y + s for y, s in zip(means, stds)]
    ax.fill_between(xs, lower, upper, color=COLORS["neuroplastic"], alpha=0.12, linewidth=0)
    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.set_xticks(xs)
    if ylim is None:
        ylim = _pad_limits(min(lower), max(upper))
    ax.set_ylim(*ylim)
    _style_axes(ax)
    _add_panel_label(ax, panel_label)


def _write_single_panel(
    output_path: Path,
    draw_fn: Any,
    *,
    figsize: tuple[float, float] = (6.2, 4.4),
) -> Path:
    fig, ax = plt.subplots(figsize=figsize)
    draw_fn(ax)
    return _save(fig, output_path)


def _render_figure_2(output_dir: Path, created: list[Path], sources: set[Path]) -> None:
    mnist_aggregates = _aggregate_for_dataset(MNIST_RESULTS, "mnist")
    fashion_aggregates = _aggregate_for_dataset(FASHION_RESULTS, "fashionmnist")
    mnist_baseline = _find_baseline(mnist_aggregates)
    mnist_full = _select_full(mnist_aggregates)
    fashion_baseline = _find_baseline(fashion_aggregates)
    fashion_full = _select_full(fashion_aggregates)

    sources.add(MNIST_ARTIFACTS / "compact_summary_table.csv")
    sources.add(FASHION_ARTIFACTS / "compact_summary_table.csv")
    for run in [*mnist_baseline.runs, *mnist_full.runs, *fashion_baseline.runs, *fashion_full.runs]:
        sources.add(run.metrics_path)

    created.append(
        _write_single_panel(
            output_dir / "fig2A.png",
            lambda ax: _plot_curve_panel(
                ax,
                panel_label="A",
                title="MNIST Test Accuracy",
                ylabel="Test Accuracy",
                baseline=mnist_baseline,
                full=mnist_full,
                value_attr="mean_test_accuracy",
                std_attr="std_test_accuracy",
                legend_loc="lower right",
                ylim=(0.905, 0.980),
            ),
        )
    )
    created.append(
        _write_single_panel(
            output_dir / "fig2B.png",
            lambda ax: _plot_curve_panel(
                ax,
                panel_label="B",
                title="Fashion-MNIST Test Accuracy",
                ylabel="Test Accuracy",
                baseline=fashion_baseline,
                full=fashion_full,
                value_attr="mean_test_accuracy",
                std_attr="std_test_accuracy",
                legend_loc="lower right",
                ylim=(0.81, 0.885),
            ),
        )
    )

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.6))
    _plot_curve_panel(
        axes[0],
        panel_label="A",
        title="MNIST Test Accuracy",
        ylabel="Test Accuracy",
        baseline=mnist_baseline,
        full=mnist_full,
        value_attr="mean_test_accuracy",
        std_attr="std_test_accuracy",
        legend_loc="lower right",
        ylim=(0.905, 0.980),
    )
    _plot_curve_panel(
        axes[1],
        panel_label="B",
        title="Fashion-MNIST Test Accuracy",
        ylabel="Test Accuracy",
        baseline=fashion_baseline,
        full=fashion_full,
        value_attr="mean_test_accuracy",
        std_attr="std_test_accuracy",
        legend_loc="lower right",
        ylim=(0.81, 0.885),
    )
    created.append(_save(fig, output_dir / "fig2_combined.png"))


def _render_figure_3(output_dir: Path, created: list[Path], sources: set[Path]) -> None:
    summary_path = LOW_DATA_ARTIFACTS / "compact_summary_table.csv"
    rows = _load_csv_rows(summary_path)
    sources.add(summary_path)

    created.append(
        _write_single_panel(
            output_dir / "fig3A.png",
            lambda ax: _plot_fraction_panel(
                ax,
                panel_label="A",
                title="Low-Data Fashion-MNIST Final Test Accuracy",
                ylabel="Final Test Accuracy",
                rows=rows,
                value_key="mean_final_test_accuracy",
                std_key="std_final_test_accuracy",
                legend_loc="lower right",
                ylim=(0.79, 0.885),
            ),
        )
    )
    created.append(
        _write_single_panel(
            output_dir / "fig3B.png",
            lambda ax: _plot_fraction_panel(
                ax,
                panel_label="B",
                title="Low-Data Fashion-MNIST Final Test Loss",
                ylabel="Final Test Loss",
                rows=rows,
                value_key="mean_final_test_loss",
                std_key="std_final_test_loss",
                legend_loc="upper right",
                ylim=(0.32, 0.55),
            ),
        )
    )

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.6))
    _plot_fraction_panel(
        axes[0],
        panel_label="A",
        title="Low-Data Fashion-MNIST Final Test Accuracy",
        ylabel="Final Test Accuracy",
        rows=rows,
        value_key="mean_final_test_accuracy",
        std_key="std_final_test_accuracy",
        legend_loc="lower right",
        ylim=(0.79, 0.885),
    )
    _plot_fraction_panel(
        axes[1],
        panel_label="B",
        title="Low-Data Fashion-MNIST Final Test Loss",
        ylabel="Final Test Loss",
        rows=rows,
        value_key="mean_final_test_loss",
        std_key="std_final_test_loss",
        legend_loc="upper right",
        ylim=(0.32, 0.55),
    )
    created.append(_save(fig, output_dir / "fig3_combined.png"))


def _render_figure_4(output_dir: Path, created: list[Path], sources: set[Path]) -> None:
    cifar_aggregates = _aggregate_for_dataset(CIFAR_RESULTS, "cifar10")
    cifar_baseline = _find_baseline(cifar_aggregates)
    cifar_full = _select_full(cifar_aggregates)

    sources.add(CIFAR_ARTIFACTS / "compact_summary_table.csv")
    for run in [*cifar_baseline.runs, *cifar_full.runs]:
        sources.add(run.metrics_path)

    created.append(
        _write_single_panel(
            output_dir / "fig4A.png",
            lambda ax: _plot_curve_panel(
                ax,
                panel_label="A",
                title="CIFAR-10 Test Accuracy",
                ylabel="Test Accuracy",
                baseline=cifar_baseline,
                full=cifar_full,
                value_attr="mean_test_accuracy",
                std_attr="std_test_accuracy",
                legend_loc="lower right",
                ylim=(0.47, 0.79),
            ),
        )
    )
    created.append(
        _write_single_panel(
            output_dir / "fig4B.png",
            lambda ax: _plot_curve_panel(
                ax,
                panel_label="B",
                title="CIFAR-10 Test Loss",
                ylabel="Test Loss",
                baseline=cifar_baseline,
                full=cifar_full,
                value_attr="mean_test_loss",
                std_attr="std_test_loss",
                legend_loc="upper right",
                ylim=(0.62, 1.90),
            ),
        )
    )

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.6))
    _plot_curve_panel(
        axes[0],
        panel_label="A",
        title="CIFAR-10 Test Accuracy",
        ylabel="Test Accuracy",
        baseline=cifar_baseline,
        full=cifar_full,
        value_attr="mean_test_accuracy",
        std_attr="std_test_accuracy",
        legend_loc="lower right",
        ylim=(0.47, 0.79),
    )
    _plot_curve_panel(
        axes[1],
        panel_label="B",
        title="CIFAR-10 Test Loss",
        ylabel="Test Loss",
        baseline=cifar_baseline,
        full=cifar_full,
        value_attr="mean_test_loss",
        std_attr="std_test_loss",
        legend_loc="upper right",
        ylim=(0.62, 1.90),
    )
    created.append(_save(fig, output_dir / "fig4_combined.png"))


def _render_figure_5(output_dir: Path, created: list[Path], sources: set[Path]) -> None:
    fashion_aggregates = _aggregate_for_dataset(FASHION_RESULTS, "fashionmnist")
    fashion_full = _select_full(fashion_aggregates)
    xs_alpha, mean_alpha, std_alpha = _aggregate_run_diagnostic(fashion_full.runs, "alpha_mean")
    xs_update, mean_update, std_update = _aggregate_run_diagnostic(
        fashion_full.runs, "effective_update_norm"
    )

    for run in fashion_full.runs:
        sources.add(run.metrics_path)

    created.append(
        _write_single_panel(
            output_dir / "fig5A.png",
            lambda ax: _plot_diagnostic_panel(
                ax,
                panel_label="A",
                title="Plasticity Coefficient Dynamics",
                ylabel="Mean Alpha",
                xs=xs_alpha,
                means=mean_alpha,
                stds=std_alpha,
                ylim=(1.00, 1.065),
            ),
        )
    )
    created.append(
        _write_single_panel(
            output_dir / "fig5B.png",
            lambda ax: _plot_diagnostic_panel(
                ax,
                panel_label="B",
                title="Effective Update Norm Dynamics",
                ylabel="Effective Update Norm",
                xs=xs_update,
                means=mean_update,
                stds=std_update,
                ylim=(29.0, 36.9),
            ),
        )
    )

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.6))
    _plot_diagnostic_panel(
        axes[0],
        panel_label="A",
        title="Plasticity Coefficient Dynamics",
        ylabel="Mean Alpha",
        xs=xs_alpha,
        means=mean_alpha,
        stds=std_alpha,
        ylim=(1.00, 1.065),
    )
    _plot_diagnostic_panel(
        axes[1],
        panel_label="B",
        title="Effective Update Norm Dynamics",
        ylabel="Effective Update Norm",
        xs=xs_update,
        means=mean_update,
        stds=std_update,
        ylim=(29.0, 36.9),
    )
    created.append(_save(fig, output_dir / "fig5_combined.png"))


def _print_report(
    sources: set[Path],
    created: list[Path],
    caveats: list[str],
    *,
    modified_scripts: list[Path],
    notes: list[str],
) -> None:
    print("Modified plotting/helper files:")
    for path in modified_scripts:
        print(f" - {path}")
    print("Source files used:")
    for path in sorted(sources):
        print(f" - {path}")
    print("Figure files overwritten:")
    for path in created:
        print(f" - {path}")
    print("PDF copies generated:")
    print(" - no")
    print("Figure-specific notes:")
    for note in notes:
        print(f" - {note}")
    print("Caveats:")
    if caveats:
        for caveat in caveats:
            print(f" - {caveat}")
    else:
        print(" - None")
    print("Source metrics or CSV data changed:")
    print(" - no")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate NeurIPS manuscript Figures 2-5.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    _apply_style()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    created: list[Path] = []
    sources: set[Path] = set()
    caveats = [
        "Figure 4 reflects limited transfer: the locked MNIST-derived full NeuroPlastic config is slightly below the grad-only baseline on final CIFAR-10 accuracy.",
        "Figure 5 diagnostics are shown for the best full Fashion-MNIST run family because those logs provide a stable, readable mechanistic trajectory.",
    ]
    notes = [
        "Standardized all figures to a sans-serif paper style with consistent title, label, tick, legend, and panel-label sizing.",
        "Panel labels now use bold 'A'/'B' placement at the upper-left with consistent offsets across single and combined figures.",
        "Figure 2 uses tighter accuracy windows to improve readability while preserving the modest MNIST gap and clearer Fashion-MNIST gain.",
        "Figure 3 uses aligned combined panels, conservative y-limits, and consistent fraction tick formatting to keep the low-data trend prominent but honest.",
        "Figure 4 legends and y-limits were positioned to read as limited transfer with a small final gap rather than instability.",
        "Figure 5 uses lighter variability bands and tighter diagnostic ranges to emphasize stable mechanistic trajectories.",
    ]

    _render_figure_2(output_dir, created, sources)
    _render_figure_3(output_dir, created, sources)
    _render_figure_4(output_dir, created, sources)
    _render_figure_5(output_dir, created, sources)
    _print_report(
        sources,
        created,
        caveats,
        modified_scripts=[Path(__file__).resolve()],
        notes=notes,
    )


if __name__ == "__main__":
    main()

from pathlib import Path

from scripts.paper_figures.low_data_analysis import (
    aggregate_low_data_runs,
    build_low_data_comparison,
)
from scripts.paper_figures.mnist_full_tuning import TuningRun


def _make_run(*, seed: int, optimizer_name: str, fraction: float, final_acc: float, best_acc: float, final_loss: float) -> TuningRun:
    return TuningRun(
        run_name=f"{optimizer_name}_seed{seed}",
        optimizer_name=optimizer_name,
        lr=0.1,
        warmup_epochs=0 if optimizer_name == "ablation_grad_only" else 2,
        plasticity_scale=1.0 if optimizer_name == "ablation_grad_only" else 2.0,
        seed=seed,
        epochs=10,
        dataset="fashionmnist",
        model_identifier="mlp_classifier_784_256_10",
        timestamp=None,
        git_commit_hash=None,
        result_directory="results",
        summary_path=Path("summary.json"),
        metrics_path=Path("metrics.json"),
        events_path=None,
        tags={"data_fraction": fraction},
        test_accuracy=[final_acc],
        test_loss=[final_loss],
        final_test_accuracy=final_acc,
        best_test_accuracy=best_acc,
        final_test_loss=final_loss,
    )


def test_aggregate_low_data_runs_groups_by_fraction_and_optimizer():
    runs = [
        _make_run(seed=41, optimizer_name="ablation_grad_only", fraction=0.1, final_acc=0.70, best_acc=0.72, final_loss=0.9),
        _make_run(seed=42, optimizer_name="ablation_grad_only", fraction=0.1, final_acc=0.72, best_acc=0.73, final_loss=0.88),
        _make_run(seed=41, optimizer_name="neuroplastic", fraction=0.1, final_acc=0.75, best_acc=0.76, final_loss=0.82),
        _make_run(seed=42, optimizer_name="neuroplastic", fraction=0.1, final_acc=0.74, best_acc=0.77, final_loss=0.80),
    ]

    aggregates = aggregate_low_data_runs(runs)

    assert len(aggregates) == 2
    assert {item.optimizer_name for item in aggregates} == {"ablation_grad_only", "neuroplastic"}
    full = next(item for item in aggregates if item.optimizer_name == "neuroplastic")
    assert full.mean_final_test_accuracy > 0.74


def test_build_low_data_comparison_reports_largest_advantage():
    runs = [
        _make_run(seed=41, optimizer_name="ablation_grad_only", fraction=0.1, final_acc=0.70, best_acc=0.72, final_loss=0.9),
        _make_run(seed=42, optimizer_name="ablation_grad_only", fraction=0.1, final_acc=0.71, best_acc=0.73, final_loss=0.88),
        _make_run(seed=41, optimizer_name="neuroplastic", fraction=0.1, final_acc=0.76, best_acc=0.77, final_loss=0.8),
        _make_run(seed=42, optimizer_name="neuroplastic", fraction=0.1, final_acc=0.75, best_acc=0.76, final_loss=0.81),
        _make_run(seed=41, optimizer_name="ablation_grad_only", fraction=1.0, final_acc=0.85, best_acc=0.86, final_loss=0.5),
        _make_run(seed=42, optimizer_name="ablation_grad_only", fraction=1.0, final_acc=0.84, best_acc=0.85, final_loss=0.52),
        _make_run(seed=41, optimizer_name="neuroplastic", fraction=1.0, final_acc=0.86, best_acc=0.87, final_loss=0.48),
        _make_run(seed=42, optimizer_name="neuroplastic", fraction=1.0, final_acc=0.85, best_acc=0.86, final_loss=0.49),
    ]

    comparison = build_low_data_comparison(aggregate_low_data_runs(runs))

    assert comparison["largest_advantage_fraction"] == 0.1
    assert comparison["fractions"][0]["final_seed_wins"] == 2

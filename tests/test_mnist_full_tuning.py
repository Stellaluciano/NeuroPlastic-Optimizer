import csv
import shutil
from pathlib import Path

from scripts.paper_figures.mnist_full_tuning import (
    TuningAggregate,
    TuningRun,
    aggregate_tuning_runs,
    build_interpretation_note,
    compare_best_full_vs_baseline,
    write_compact_summary_table,
)


def _make_run(
    *,
    optimizer_name: str,
    lr: float,
    warmup_epochs: int,
    plasticity_scale: float,
    seed: int,
    test_accuracy: list[float],
    test_loss: list[float],
) -> TuningRun:
    return TuningRun(
        run_name=f"{optimizer_name}_seed{seed}",
        optimizer_name=optimizer_name,
        lr=lr,
        warmup_epochs=warmup_epochs,
        plasticity_scale=plasticity_scale,
        seed=seed,
        epochs=len(test_accuracy),
        dataset="mnist",
        model_identifier="mlp_classifier_784_256_10",
        timestamp=None,
        git_commit_hash=None,
        result_directory="results_mnist_full_tuning_clean",
        summary_path=Path(f"{optimizer_name}_seed{seed}_summary.json"),
        metrics_path=Path(f"{optimizer_name}_seed{seed}_metrics.json"),
        events_path=None,
        test_accuracy=test_accuracy,
        test_loss=test_loss,
        final_test_accuracy=test_accuracy[-1],
        best_test_accuracy=max(test_accuracy),
        final_test_loss=test_loss[-1],
    )


def test_aggregate_tuning_runs_groups_multiple_configs():
    runs = [
        _make_run(
            optimizer_name="ablation_grad_only",
            lr=0.1,
            warmup_epochs=0,
            plasticity_scale=1.0,
            seed=41,
            test_accuracy=[0.90, 0.95],
            test_loss=[0.2, 0.1],
        ),
        _make_run(
            optimizer_name="ablation_grad_only",
            lr=0.1,
            warmup_epochs=0,
            plasticity_scale=1.0,
            seed=42,
            test_accuracy=[0.91, 0.94],
            test_loss=[0.19, 0.11],
        ),
        _make_run(
            optimizer_name="neuroplastic",
            lr=0.03,
            warmup_epochs=1,
            plasticity_scale=2.0,
            seed=41,
            test_accuracy=[0.92, 0.96],
            test_loss=[0.18, 0.09],
        ),
        _make_run(
            optimizer_name="neuroplastic",
            lr=0.03,
            warmup_epochs=1,
            plasticity_scale=2.0,
            seed=42,
            test_accuracy=[0.93, 0.97],
            test_loss=[0.17, 0.08],
        ),
    ]

    aggregates = aggregate_tuning_runs(runs)

    assert len(aggregates) == 2
    baseline = next(item for item in aggregates if item.optimizer_name == "ablation_grad_only")
    full = next(item for item in aggregates if item.optimizer_name == "neuroplastic")
    assert baseline.num_seeds == 2
    assert full.num_seeds == 2
    assert full.mean_final_test_accuracy == 0.965
    assert full.mean_epoch_to_95_acc == 2.0
    assert full.num_seeds_reached_97_acc == 1


def test_write_compact_summary_table_encodes_unreached_thresholds():
    aggregate = TuningAggregate(
        optimizer_name="neuroplastic",
        lr=0.03,
        warmup_epochs=1,
        plasticity_scale=2.0,
        runs=[],
        epochs=[1, 2],
        mean_test_accuracy=[0.90, 0.94],
        std_test_accuracy=[0.0, 0.0],
        mean_test_loss=[0.2, 0.1],
        std_test_loss=[0.0, 0.0],
        num_seeds=2,
        mean_final_test_accuracy=0.94,
        std_final_test_accuracy=0.01,
        mean_best_test_accuracy=0.94,
        std_best_test_accuracy=0.01,
        mean_final_test_loss=0.1,
        std_final_test_loss=0.01,
        mean_epoch_to_95_acc=None,
        mean_epoch_to_97_acc=None,
        num_seeds_reached_95_acc=0,
        num_seeds_reached_97_acc=0,
    )

    output_dir = Path(".mnist_full_tuning_test_artifacts")
    output_dir.mkdir(exist_ok=True)
    try:
        output_path = write_compact_summary_table([aggregate], output_dir)
        rows = list(csv.DictReader(output_path.open(encoding="utf-8")))
        assert rows[0]["mean_epoch_to_95_acc"] == "not_reached"
        assert rows[0]["mean_epoch_to_97_acc"] == "not_reached"
    finally:
        shutil.rmtree(output_dir, ignore_errors=True)


def test_compare_best_full_vs_baseline_uses_seed_aware_counts():
    runs = [
        _make_run(
            optimizer_name="ablation_grad_only",
            lr=0.1,
            warmup_epochs=0,
            plasticity_scale=1.0,
            seed=41,
            test_accuracy=[0.90, 0.95],
            test_loss=[0.2, 0.1],
        ),
        _make_run(
            optimizer_name="ablation_grad_only",
            lr=0.1,
            warmup_epochs=0,
            plasticity_scale=1.0,
            seed=42,
            test_accuracy=[0.89, 0.94],
            test_loss=[0.21, 0.11],
        ),
        _make_run(
            optimizer_name="neuroplastic",
            lr=0.03,
            warmup_epochs=1,
            plasticity_scale=2.0,
            seed=41,
            test_accuracy=[0.92, 0.96],
            test_loss=[0.18, 0.09],
        ),
        _make_run(
            optimizer_name="neuroplastic",
            lr=0.03,
            warmup_epochs=1,
            plasticity_scale=2.0,
            seed=42,
            test_accuracy=[0.90, 0.95],
            test_loss=[0.19, 0.1],
        ),
    ]

    comparison = compare_best_full_vs_baseline(aggregate_tuning_runs(runs))

    assert comparison["comparison_mode"] == "seed_aware"
    assert comparison["final_seed_wins"] == 2
    assert comparison["best_seed_wins"] == 2
    assert comparison["robustness"] in {"robust", "marginal"}


def test_build_interpretation_note_says_plainly_when_full_does_not_win():
    baseline = TuningAggregate(
        optimizer_name="ablation_grad_only",
        lr=0.1,
        warmup_epochs=0,
        plasticity_scale=1.0,
        runs=[],
        epochs=[1, 2],
        mean_test_accuracy=[0.93, 0.97],
        std_test_accuracy=[0.0, 0.0],
        mean_test_loss=[0.15, 0.08],
        std_test_loss=[0.0, 0.0],
        num_seeds=2,
        mean_final_test_accuracy=0.97,
        std_final_test_accuracy=0.0,
        mean_best_test_accuracy=0.97,
        std_best_test_accuracy=0.0,
        mean_final_test_loss=0.08,
        std_final_test_loss=0.0,
        mean_epoch_to_95_acc=2.0,
        mean_epoch_to_97_acc=2.0,
        num_seeds_reached_95_acc=2,
        num_seeds_reached_97_acc=2,
    )
    full = TuningAggregate(
        optimizer_name="neuroplastic",
        lr=0.03,
        warmup_epochs=1,
        plasticity_scale=2.0,
        runs=[],
        epochs=[1, 2],
        mean_test_accuracy=[0.92, 0.965],
        std_test_accuracy=[0.0, 0.0],
        mean_test_loss=[0.16, 0.09],
        std_test_loss=[0.0, 0.0],
        num_seeds=2,
        mean_final_test_accuracy=0.965,
        std_final_test_accuracy=0.0,
        mean_best_test_accuracy=0.965,
        std_best_test_accuracy=0.0,
        mean_final_test_loss=0.09,
        std_final_test_loss=0.0,
        mean_epoch_to_95_acc=2.0,
        mean_epoch_to_97_acc=None,
        num_seeds_reached_95_acc=2,
        num_seeds_reached_97_acc=0,
    )

    comparison = {
        "mean_final_accuracy_gap_vs_baseline": -0.005,
        "mean_best_accuracy_gap_vs_baseline": -0.005,
        "mean_final_loss_gap_vs_baseline": 0.01,
        "comparison_mode": "aggregate_only",
        "robustness": "inconclusive",
    }

    note = build_interpretation_note([baseline, full], comparison)

    assert "Full NeuroPlastic still does not clearly win" in note

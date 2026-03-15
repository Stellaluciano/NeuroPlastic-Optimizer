import json
from pathlib import Path

from scripts.paper_figures.mnist_full_tuning import (
    TuningAggregate,
    select_recommended_full_config,
    split_complete_and_partial_aggregates,
)
from scripts.paper_figures.study_helpers import (
    RunArtifactStatus,
    classify_config_group,
    compare_seed_metric_dicts,
    deterministic_subset_indices,
    load_locked_best_config,
)


def test_classify_config_group_recognizes_partial_and_invalid_states():
    completed = [
        RunArtifactStatus(41, "completed", Path("a"), Path("b"), Path("c"), 10),
        RunArtifactStatus(42, "completed", Path("a"), Path("b"), Path("c"), 10),
    ]
    partial = [
        RunArtifactStatus(41, "completed", Path("a"), Path("b"), Path("c"), 10),
        RunArtifactStatus(42, "missing", Path("a"), Path("b"), Path("c"), 0),
    ]
    invalid = [
        RunArtifactStatus(41, "completed", Path("a"), Path("b"), Path("c"), 10),
        RunArtifactStatus(42, "invalid", Path("a"), Path("b"), Path("c"), 0),
    ]

    assert classify_config_group(completed) == "completed"
    assert classify_config_group(partial) == "partial"
    assert classify_config_group(invalid) == "invalid"


def test_split_complete_and_partial_aggregates_uses_expected_seed_count():
    complete = TuningAggregate(
        optimizer_name="neuroplastic",
        lr=0.1,
        warmup_epochs=0,
        plasticity_scale=2.0,
        runs=[],
        epochs=[1],
        mean_test_accuracy=[0.97],
        std_test_accuracy=[0.0],
        mean_test_loss=[0.1],
        std_test_loss=[0.0],
        num_seeds=3,
        mean_final_test_accuracy=0.97,
        std_final_test_accuracy=0.0,
        mean_best_test_accuracy=0.98,
        std_best_test_accuracy=0.0,
        mean_final_test_loss=0.1,
        std_final_test_loss=0.0,
        mean_epoch_to_95_acc=2.0,
        mean_epoch_to_97_acc=4.0,
        num_seeds_reached_95_acc=3,
        num_seeds_reached_97_acc=3,
    )
    partial = TuningAggregate(
        optimizer_name="neuroplastic",
        lr=0.1,
        warmup_epochs=1,
        plasticity_scale=1.0,
        runs=[],
        epochs=[1],
        mean_test_accuracy=[0.96],
        std_test_accuracy=[0.0],
        mean_test_loss=[0.11],
        std_test_loss=[0.0],
        num_seeds=2,
        mean_final_test_accuracy=0.96,
        std_final_test_accuracy=0.0,
        mean_best_test_accuracy=0.97,
        std_best_test_accuracy=0.0,
        mean_final_test_loss=0.11,
        std_final_test_loss=0.0,
        mean_epoch_to_95_acc=2.0,
        mean_epoch_to_97_acc=5.0,
        num_seeds_reached_95_acc=2,
        num_seeds_reached_97_acc=2,
    )

    split = split_complete_and_partial_aggregates([complete, partial], 3)

    assert split.complete == [complete]
    assert split.partial == [partial]


def test_select_recommended_full_config_prefers_best_final_accuracy():
    best = TuningAggregate(
        optimizer_name="neuroplastic",
        lr=0.1,
        warmup_epochs=0,
        plasticity_scale=2.0,
        runs=[],
        epochs=[1],
        mean_test_accuracy=[0.975],
        std_test_accuracy=[0.0],
        mean_test_loss=[0.09],
        std_test_loss=[0.0],
        num_seeds=3,
        mean_final_test_accuracy=0.975,
        std_final_test_accuracy=0.0,
        mean_best_test_accuracy=0.977,
        std_best_test_accuracy=0.0,
        mean_final_test_loss=0.09,
        std_final_test_loss=0.0,
        mean_epoch_to_95_acc=2.0,
        mean_epoch_to_97_acc=5.0,
        num_seeds_reached_95_acc=3,
        num_seeds_reached_97_acc=3,
    )
    comparison = {"mean_final_accuracy_gap_vs_baseline": 0.001}

    selected = select_recommended_full_config([best], comparison, dataset_name="mnist")

    assert selected["optimizer_name"] == "neuroplastic"
    assert selected["lr"] == 0.1
    assert selected["warmup_epochs"] == 0
    assert selected["plasticity_scale"] == 2.0


def test_load_locked_best_config_reads_saved_payload(tmp_path):
    config_path = tmp_path / "best_config.json"
    config_path.write_text(
        json.dumps(
            {
                "study_name": "mnist_full_tuning_clean",
                "dataset": "mnist",
                "optimizer_name": "neuroplastic",
                "lr": 0.1,
                "warmup_epochs": 2,
                "plasticity_scale": 2.0,
                "selected_by": "mean_final_test_accuracy",
                "selection_reason": "Locked for downstream benchmarks",
            }
        ),
        encoding="utf-8",
    )

    config = load_locked_best_config(config_path)

    assert config.study_name == "mnist_full_tuning_clean"
    assert config.lr == 0.1
    assert config.warmup_epochs == 2


def test_deterministic_subset_indices_are_seed_controlled():
    first = deterministic_subset_indices(dataset_size=100, fraction=0.25, seed=41)
    second = deterministic_subset_indices(dataset_size=100, fraction=0.25, seed=41)
    third = deterministic_subset_indices(dataset_size=100, fraction=0.25, seed=42)

    assert first == second
    assert len(first) == 25
    assert first != third


def test_compare_seed_metric_dicts_counts_wins_losses_and_ties():
    comparison = compare_seed_metric_dicts(
        {41: 0.8, 42: 0.7, 43: 0.9},
        {41: 0.81, 42: 0.7, 43: 0.85},
    )

    assert comparison["shared_seed_count"] == 3
    assert comparison["wins"] == 1
    assert comparison["losses"] == 1
    assert comparison["ties"] == 1

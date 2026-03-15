import csv
import json


def test_generate_cpu_paper_figures_handles_summary_and_metrics(tmp_path):
    from scripts.paper_figures.generate_cpu_paper_figures import generate_paper_figures

    results_dir = tmp_path / "results"
    output_dir = tmp_path / "paper_artifacts"
    results_dir.mkdir()

    (results_dir / "neuroplastic_mnist_neuroplastic_summary.json").write_text(
        json.dumps(
            {
                "run_name": "neuroplastic",
                "best_test_accuracy": 0.92,
                "last_test_loss": 0.14,
                "optimizer": "neuroplastic",
                "dataset": "mnist",
            }
        ),
        encoding="utf-8",
    )
    (results_dir / "neuroplastic_mnist_neuroplastic_metrics.json").write_text(
        json.dumps(
            {
                "test": [
                    {"loss": 0.40, "accuracy": 0.88},
                    {"loss": 0.20, "accuracy": 0.91},
                    {"loss": 0.14, "accuracy": 0.92},
                ],
                "config": {"dataset": "mnist", "run_name": "neuroplastic", "optimizer": "neuroplastic"},
                "device": "cpu",
            }
        ),
        encoding="utf-8",
    )
    (results_dir / "adam_mnist_adam_summary.json").write_text(
        json.dumps(
            {
                "run_name": "adam",
                "best_test_accuracy": 0.89,
                "last_test_loss": 0.18,
                "optimizer": "adam",
                "dataset": "mnist",
            }
        ),
        encoding="utf-8",
    )
    (results_dir / "adam_mnist_adam_events.jsonl").write_text(
        "\n".join(
            [
                json.dumps({"epoch": 1, "test_loss": 0.45, "test_acc": 0.84, "device": "cpu"}),
                json.dumps({"epoch": 2, "test_loss": 0.22, "test_acc": 0.88, "device": "cpu"}),
                json.dumps({"epoch": 3, "test_loss": 0.18, "test_acc": 0.89, "device": "cpu"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    summary = generate_paper_figures(results_dir, output_dir)

    assert "neuroplastic" in summary["runs_found"]
    assert "adam" in summary["runs_found"]
    assert (output_dir / "mnist_test_accuracy_vs_epoch.png").exists()
    assert (output_dir / "mnist_best_final_test_accuracy.png").exists()
    assert (output_dir / "mnist_test_loss_vs_epoch.png").exists()
    assert (output_dir / "mnist_early_convergence_accuracy.png").exists()
    assert (output_dir / "run_notes.md").exists()

    table_rows = list(csv.DictReader((output_dir / "benchmark_table.csv").open(encoding="utf-8")))
    assert len(table_rows) == 2
    assert {row["run_name"] for row in table_rows} == {"neuroplastic", "adam"}


def test_generate_cpu_paper_figures_writes_seed_aggregates(tmp_path):
    from scripts.paper_figures.generate_cpu_paper_figures import generate_paper_figures

    results_dir = tmp_path / "results"
    output_dir = tmp_path / "paper_artifacts"
    results_dir.mkdir()

    for seed, final_acc in [(41, 0.90), (42, 0.94)]:
        (results_dir / f"adam_seed{seed}_mnist_adam_summary.json").write_text(
            json.dumps(
                {
                    "run_name": f"adam_seed{seed}",
                    "best_test_accuracy": final_acc,
                    "last_test_loss": 0.10,
                    "optimizer": "adam",
                    "dataset": "mnist",
                }
            ),
            encoding="utf-8",
        )
        (results_dir / f"adam_seed{seed}_mnist_adam_metrics.json").write_text(
            json.dumps(
                {
                    "test": [
                        {"loss": 0.30, "accuracy": final_acc - 0.08},
                        {"loss": 0.18, "accuracy": final_acc - 0.03},
                        {"loss": 0.10, "accuracy": final_acc},
                    ],
                    "config": {"dataset": "mnist", "run_name": f"adam_seed{seed}", "optimizer": "adam"},
                    "device": "cpu",
                }
            ),
            encoding="utf-8",
        )

    summary = generate_paper_figures(results_dir, output_dir)

    assert "adam" in summary["seed_aggregates_found"]
    assert (output_dir / "mnist_seed_aggregated_test_accuracy_vs_epoch.png").exists()
    assert (output_dir / "mnist_seed_aggregated_best_final_test_accuracy.png").exists()
    assert (output_dir / "mnist_seed_aggregated_early_convergence_accuracy.png").exists()
    assert (output_dir / "seed_aggregate_table.csv").exists()

    aggregate_rows = list(csv.DictReader((output_dir / "seed_aggregate_table.csv").open(encoding="utf-8")))
    assert len(aggregate_rows) == 1
    assert aggregate_rows[0]["run_group"] == "adam"
    assert aggregate_rows[0]["run_count"] == "2"

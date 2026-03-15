from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.paper_figures.low_data_analysis import (  # noqa: E402
    build_low_data_interpretation_note,
    load_and_aggregate_low_data,
    write_low_data_manifest,
    write_low_data_summary_table,
    _plot_metric_by_fraction,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate low-data best-full-vs-grad-only artifacts")
    parser.add_argument("--results-dir", default="results_low_data_fashionmnist_bestfull_vs_gradonly_clean", help="Directory containing low-data run artifacts.")
    parser.add_argument("--output-dir", default="paper_artifacts/low_data_fashionmnist_bestfull_vs_gradonly_clean", help="Directory where low-data figures/tables are written.")
    parser.add_argument("--dataset", default="fashionmnist", choices=["fashionmnist", "mnist", "cifar10"], help="Dataset to analyze.")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _, aggregates, comparison = load_and_aggregate_low_data(Path(args.results_dir), args.dataset)

    generated_files = [
        write_low_data_summary_table(aggregates, output_dir),
        _plot_metric_by_fraction(
            aggregates,
            output_dir / "fraction_accuracy_plot.png",
            metric_attr="mean_final_test_accuracy",
            ylabel="Mean Final Test Accuracy",
        ),
        _plot_metric_by_fraction(
            aggregates,
            output_dir / "fraction_loss_plot.png",
            metric_attr="mean_final_test_loss",
            ylabel="Mean Final Test Loss",
        ),
    ]
    note_path = output_dir / "interpretation_note.md"
    note_path.write_text(build_low_data_interpretation_note(args.dataset, comparison), encoding="utf-8")
    generated_files.append(note_path)
    generated_files.append(
        write_low_data_manifest(
            output_dir,
            dataset=args.dataset,
            comparison=comparison,
            generated_files=generated_files,
        )
    )

    for path in generated_files:
        print(f"[low_data_figures] wrote: {path}")


if __name__ == "__main__":
    main()

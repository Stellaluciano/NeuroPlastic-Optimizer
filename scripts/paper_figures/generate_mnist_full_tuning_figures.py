from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.paper_figures.mnist_full_tuning import (  # noqa: E402
    generate_full_tuning_artifacts,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate paper-ready figures and summary tables "
            "for the CPU MNIST full-tuning study"
        )
    )
    parser.add_argument(
        "--results-dir",
        default="results_mnist_full_tuning_clean",
        help="Directory containing tuning result artifacts.",
    )
    parser.add_argument(
        "--output-dir",
        default="paper_artifacts/mnist_full_tuning_clean",
        help="Directory where tuning figures and tables will be written.",
    )
    parser.add_argument(
        "--expected-seeds",
        type=int,
        default=3,
        help="Seed count required for a config to be treated as final in the artifact bundle.",
    )
    parser.add_argument(
        "--recommended-config-out",
        default="configs/paper/best_full_neuroplastic_mnist.json",
        help="Path where the selected best full NeuroPlastic config will be written.",
    )
    args = parser.parse_args()

    summary = generate_full_tuning_artifacts(
        Path(args.results_dir),
        Path(args.output_dir),
        dataset_name="mnist",
        expected_seed_count=args.expected_seeds,
        file_prefix="mnist_full_tuning",
        include_ranking_plot=True,
        recommended_config_path=Path(args.recommended_config_out),
    )
    print(
        "[mnist_full_tuning_figures] runs found:",
        ", ".join(summary["runs_found"]) if summary["runs_found"] else "none",
    )
    for path in summary["generated_files"]:
        print(f"[mnist_full_tuning_figures] wrote: {path}")


if __name__ == "__main__":
    main()

# ruff: noqa: E501

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.paper_figures.mnist_full_tuning import (  # noqa: E402
    generate_full_tuning_artifacts,
)


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _build_fashion_interpretation_note(
    output_dir: Path,
    *,
    best_config: dict,
) -> str:
    analysis_manifest = _load_json(output_dir / "analysis_manifest.json")
    comparison = analysis_manifest["comparison"]
    base_note = (output_dir / "interpretation_note.md").read_text(encoding="utf-8").strip()
    mnist_gap = best_config.get("comparison_to_baseline", {}).get(
        "mean_final_accuracy_gap_vs_baseline"
    )
    fashion_gap = comparison.get("mean_final_accuracy_gap_vs_baseline")
    if fashion_gap is None:
        direction_line = "The Fashion-MNIST run does not yet provide a complete final-accuracy comparison."
    elif mnist_gap is None:
        direction_line = "Direction relative to MNIST could not be assessed from the saved best-config metadata."
    elif fashion_gap > 0 and mnist_gap > 0:
        direction_line = "The effect direction is consistent with MNIST: full NeuroPlastic remains slightly ahead."
    elif fashion_gap <= 0 < mnist_gap:
        direction_line = "The effect direction is weaker than MNIST: the MNIST advantage did not carry over cleanly."
    else:
        direction_line = "The effect direction is mixed relative to MNIST."

    if fashion_gap is None or mnist_gap is None:
        strength_line = "Relative evidence strength versus MNIST is inconclusive."
    elif abs(fashion_gap) > abs(mnist_gap):
        strength_line = "The Fashion-MNIST effect is stronger in magnitude than the MNIST result."
    elif abs(fashion_gap) < abs(mnist_gap):
        strength_line = "The Fashion-MNIST effect is weaker than the MNIST result."
    else:
        strength_line = "The Fashion-MNIST effect magnitude is about the same as MNIST."

    lines = base_note.splitlines()
    lines.extend(
        [
            f"- Does the best full NeuroPlastic config remain competitive on Fashion-MNIST? "
            f"{'Yes' if comparison.get('mean_final_accuracy_gap_vs_baseline', 0) >= 0 else 'Not clearly'}.",
            f"- Does it outperform ablation_grad_only on final accuracy, best accuracy, or loss? "
            f"final_gap={comparison.get('mean_final_accuracy_gap_vs_baseline')}, "
            f"best_gap={comparison.get('mean_best_accuracy_gap_vs_baseline')}, "
            f"final_loss_gap={comparison.get('mean_final_loss_gap_vs_baseline')}.",
            f"- Is the effect directionally consistent with MNIST? {direction_line}",
            f"- Is the evidence stronger, weaker, or inconclusive relative to MNIST? {strength_line}",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate Fashion-MNIST best-full-vs-grad-only paper artifacts"
    )
    parser.add_argument(
        "--results-dir",
        default="results_fashionmnist_bestfull_vs_gradonly_clean",
        help="Directory containing Fashion-MNIST result artifacts.",
    )
    parser.add_argument(
        "--output-dir",
        default="paper_artifacts/fashionmnist_bestfull_vs_gradonly_clean",
        help="Directory where Fashion-MNIST figures and tables will be written.",
    )
    parser.add_argument(
        "--best-config",
        default="configs/paper/best_full_neuroplastic_mnist.json",
        help="Best full NeuroPlastic config selected from the MNIST tuning study.",
    )
    parser.add_argument(
        "--expected-seeds",
        type=int,
        default=3,
        help="Seed count required for a config to be treated as final.",
    )
    args = parser.parse_args()

    best_config = _load_json(Path(args.best_config))
    summary = generate_full_tuning_artifacts(
        Path(args.results_dir),
        Path(args.output_dir),
        dataset_name="fashionmnist",
        expected_seed_count=args.expected_seeds,
        file_prefix="fashionmnist_bestfull_vs_gradonly",
        include_ranking_plot=False,
        recommended_config_path=None,
    )
    note_path = Path(args.output_dir) / "interpretation_note.md"
    note_path.write_text(
        _build_fashion_interpretation_note(Path(args.output_dir), best_config=best_config),
        encoding="utf-8",
    )
    if str(note_path) not in summary["generated_files"]:
        summary["generated_files"].append(str(note_path))

    print(
        "[fashionmnist_figures] runs found:",
        ", ".join(summary["runs_found"]) if summary["runs_found"] else "none",
    )
    for path in summary["generated_files"]:
        print(f"[fashionmnist_figures] wrote: {path}")


if __name__ == "__main__":
    main()

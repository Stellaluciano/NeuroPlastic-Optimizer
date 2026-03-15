# ruff: noqa: E501

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.paper_figures.mnist_full_tuning import generate_full_tuning_artifacts  # noqa: E402
from scripts.paper_figures.study_helpers import (  # noqa: E402
    format_locked_best_config,
    load_locked_best_config,
    read_json_if_valid,
    write_json,
)


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _build_note(output_dir: Path, *, best_config_path: Path) -> str:
    analysis = _load_json(output_dir / "analysis_manifest.json")
    comparison = analysis["comparison"]
    best_config = load_locked_best_config(best_config_path)
    fashion_manifest = read_json_if_valid(
        REPO_ROOT / "paper_artifacts" / "fashionmnist_bestfull_vs_gradonly_clean" / "analysis_manifest.json"
    )
    fashion_gap = None
    if fashion_manifest is not None:
        fashion_gap = (
            fashion_manifest.get("comparison", {}).get("mean_final_accuracy_gap_vs_baseline")
        )
    cifar_gap = comparison.get("mean_final_accuracy_gap_vs_baseline")
    if cifar_gap is None:
        relative_strength = "CIFAR-10 is incomplete, so the comparison to Fashion-MNIST is inconclusive."
    elif fashion_gap is None:
        relative_strength = "Fashion-MNIST reference metadata was incomplete, so cross-dataset strength is inconclusive."
    elif abs(cifar_gap) > abs(fashion_gap):
        relative_strength = "CIFAR-10 is stronger in magnitude than the Fashion-MNIST result."
    elif abs(cifar_gap) < abs(fashion_gap):
        relative_strength = "CIFAR-10 is weaker than the Fashion-MNIST result."
    else:
        relative_strength = "CIFAR-10 is about as strong as the Fashion-MNIST result."
    return (
        "# CIFAR-10 Locked Best-Full vs Grad-Only Interpretation\n\n"
        f"Locked config source:\n\n```json\n{format_locked_best_config(best_config)}\n```\n\n"
        f"- Does full NeuroPlastic remain competitive on CIFAR-10? "
        f"{'Yes' if (comparison.get('mean_final_accuracy_gap_vs_baseline') or 0.0) >= 0 else 'Not clearly'}.\n"
        f"- Does it outperform `ablation_grad_only` on mean final accuracy? "
        f"{comparison.get('mean_final_accuracy_gap_vs_baseline')}.\n"
        f"- Does it outperform on mean best accuracy? "
        f"{comparison.get('mean_best_accuracy_gap_vs_baseline')}.\n"
        f"- Does it outperform on final loss? "
        f"{comparison.get('mean_final_loss_gap_vs_baseline')}.\n"
        f"- Are wins consistent across shared seeds? "
        f"final={comparison.get('final_seed_wins')}/{comparison.get('shared_seed_count')}, "
        f"best={comparison.get('best_seed_wins')}/{comparison.get('shared_seed_count')}.\n"
        f"- Is the CIFAR-10 evidence stronger, weaker, or inconclusive relative to Fashion-MNIST? "
        f"{relative_strength}\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate CIFAR-10 best-full-vs-grad-only artifact bundle")
    parser.add_argument("--results-dir", default="results_cifar10_bestfull_vs_gradonly_clean", help="Directory containing CIFAR-10 result artifacts.")
    parser.add_argument("--output-dir", default="paper_artifacts/cifar10_bestfull_vs_gradonly_clean", help="Directory for CIFAR-10 figures/tables.")
    parser.add_argument("--best-config", default="configs/paper/best_full_neuroplastic_mnist.json", help="Locked best full NeuroPlastic config from the MNIST tuning study.")
    parser.add_argument("--expected-seeds", type=int, default=3, help="Seed count expected for a complete CIFAR-10 comparison.")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    summary = generate_full_tuning_artifacts(
        Path(args.results_dir),
        output_dir,
        dataset_name="cifar10",
        expected_seed_count=args.expected_seeds,
        file_prefix="cifar10",
        include_ranking_plot=False,
        recommended_config_path=None,
    )
    note_path = output_dir / "interpretation_note.md"
    note_path.write_text(
        _build_note(output_dir, best_config_path=(REPO_ROOT / args.best_config).resolve()),
        encoding="utf-8",
    )
    manifest_path = output_dir / "manifest.json"
    write_json(
        manifest_path,
        {
            "study_name": "cifar10_bestfull_vs_gradonly_clean",
            "results_dir": str(Path(args.results_dir).resolve()),
            "comparison": summary["comparison"],
            "generated_files": sorted(set(summary["generated_files"] + [str(note_path), str(manifest_path)])),
        },
    )

    print("[cifar10_figures] runs found:", ", ".join(summary["runs_found"]) if summary["runs_found"] else "none")
    for path in sorted(set(summary["generated_files"] + [str(note_path), str(manifest_path)])):
        print(f"[cifar10_figures] wrote: {path}")


if __name__ == "__main__":
    main()

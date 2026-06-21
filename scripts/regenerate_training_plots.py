"""Regenerate training curve PNGs from saved history JSON (no retraining)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.train import load_training_history, plot_training_curves
from src.utils import METRICS_DIR

PLOTS = {
    "custom_cnn_history.json": "custom_cnn_training_curves.png",
    "resnet18_head_history.json": "resnet18_head_training_curves.png",
    "resnet18_finetune_history.json": "resnet18_finetune_training_curves.png",
    "resnet18_history.json": "resnet18_training_curves.png",
}


def main() -> None:
    found = False
    for history_name, figure_name in PLOTS.items():
        path = METRICS_DIR / history_name
        if not path.exists():
            print(f"Skip (missing): {path}")
            continue
        history = load_training_history(path)
        if not history.train_accuracy:
            print(f"Skip (no train accuracy in file): {path}")
            print("  Re-run the matching training cell in the notebook to record train accuracy.")
            continue
        plot_training_curves(history, filename=figure_name)
        print(f"Saved: outputs/figures/{figure_name}")
        found = True

    if not found:
        print("\nNo history files with train accuracy yet.")
        print("Re-run training cells in notebooks/di504_pet_project.ipynb, then run this script again.")


if __name__ == "__main__":
    main()

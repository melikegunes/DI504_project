"""
Isolated ResNet-18 run with 100 total epochs (25 head + 75 fine-tune).

Writes ONLY under outputs/experiments/epoch100_test/ — main project files are untouched.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data import get_dataloaders, load_datasets
from src.evaluate import evaluate_model
from src.models import build_resnet18
from src.train import get_experiment_dirs, train_resnet18_phased
from src.utils import get_device, save_json, set_seed

EXPERIMENT = "epoch100_test"
TOTAL_EPOCHS = 100
HEAD_EPOCHS = 25
FINETUNE_EPOCHS = 75


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print plan only; do not train.",
    )
    args = parser.parse_args()

    figures_dir, metrics_dir, checkpoints_dir = get_experiment_dirs(EXPERIMENT)

    print(f"Experiment folder: outputs/experiments/{EXPERIMENT}/")
    print(f"Epochs: {HEAD_EPOCHS} (head) + {FINETUNE_EPOCHS} (fine-tune) = {TOTAL_EPOCHS}")
    print("Main checkpoints/metrics/figures will NOT be modified.")

    if args.dry_run:
        return

    set_seed(42)
    device = get_device()
    print(f"Device: {device}")

    train_ds, val_ds, test_ds, _ = load_datasets(download=False)
    train_loader, val_loader, test_loader = get_dataloaders(
        train_ds, val_ds, test_ds, batch_size=32
    )

    t0 = time.time()
    model, history = train_resnet18_phased(
        build_resnet18(num_classes=37, pretrained=True),
        train_loader,
        val_loader,
        device,
        head_epochs=HEAD_EPOCHS,
        finetune_epochs=FINETUNE_EPOCHS,
        head_lr=1e-3,
        finetune_lr=1e-4,
        weight_decay=1e-4,
        checkpoint_name="resnet18_standard.pt",
        experiment_name=EXPERIMENT,
    )
    train_minutes = (time.time() - t0) / 60

    metrics = evaluate_model(
        model,
        test_loader,
        device,
        model_name="ResNet-18 (100 epochs)",
        save_prefix="resnet18_standard",
        metrics_dir=metrics_dir,
        figures_dir=figures_dir,
    )
    metrics["head_epochs"] = HEAD_EPOCHS
    metrics["finetune_epochs"] = FINETUNE_EPOCHS
    metrics["total_epochs"] = TOTAL_EPOCHS
    metrics["train_minutes"] = round(train_minutes, 1)
    save_json(metrics, metrics_dir / "run_summary.json")

    print(f"\nDone in {train_minutes:.1f} minutes")
    print(f"Test accuracy: {metrics['accuracy']:.4f}")
    print(f"Macro F1: {metrics['macro_f1']:.4f}")
    print(f"Results: outputs/experiments/{EXPERIMENT}/")


if __name__ == "__main__":
    main()

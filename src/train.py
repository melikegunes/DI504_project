"""Training loops and checkpointing."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.utils import CHECKPOINTS_DIR, FIGURES_DIR, METRICS_DIR, OUTPUTS_DIR, ensure_output_dirs, save_json


def get_experiment_dirs(experiment_name: str) -> tuple[Path, Path, Path]:
    """Isolated outputs under outputs/experiments/<name>/ (does not touch main files)."""
    base = OUTPUTS_DIR / "experiments" / experiment_name
    figures_dir = base / "figures"
    metrics_dir = base / "metrics"
    checkpoints_dir = base / "checkpoints"
    for path in (figures_dir, metrics_dir, checkpoints_dir):
        path.mkdir(parents=True, exist_ok=True)
    return figures_dir, metrics_dir, checkpoints_dir


@dataclass
class TrainHistory:
    train_loss: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    train_accuracy: list[float] = field(default_factory=list)
    val_accuracy: list[float] = field(default_factory=list)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(loader, desc="train", leave=False):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return running_loss / total, correct / total


def validate_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="val", leave=False):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return running_loss / total, correct / total


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    *,
    epochs: int = 15,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    checkpoint_name: str = "model.pt",
    model_label: str = "model",
    setup_fn: Callable[[nn.Module], None] | None = None,
    experiment_name: str | None = None,
    early_stopping_patience: int | None = None,
) -> tuple[nn.Module, TrainHistory]:
    """
    Full training loop with best-checkpoint saving, early stopping, and loss-curve plot.
    """
    ensure_output_dirs()
    if setup_fn is not None:
        setup_fn(model)

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

    if experiment_name:
        figures_dir, metrics_dir, checkpoints_dir = get_experiment_dirs(experiment_name)
    else:
        figures_dir, metrics_dir, checkpoints_dir = FIGURES_DIR, METRICS_DIR, CHECKPOINTS_DIR

    history = TrainHistory()
    best_val_acc = -1.0
    epochs_no_improve = 0
    checkpoint_path = checkpoints_dir / checkpoint_name

    for epoch in range(1, epochs + 1):
        print(f"\n[{model_label}] Epoch {epoch}/{epochs}")
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate_epoch(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        history.train_loss.append(train_loss)
        history.val_loss.append(val_loss)
        history.train_accuracy.append(train_acc)
        history.val_accuracy.append(val_acc)

        print(
            f"  train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_no_improve = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "val_accuracy": val_acc,
                    "epoch": epoch,
                },
                checkpoint_path,
            )
        else:
            epochs_no_improve += 1

        if early_stopping_patience is not None and epochs_no_improve >= early_stopping_patience:
            print(
                f"\n[Early Stopping] Triggered at epoch {epoch}. "
                f"Validation accuracy did not improve for {early_stopping_patience} consecutive epochs."
            )
            break

    if checkpoint_path.exists():
        state = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state["model_state_dict"])
        print(f"Loaded best checkpoint (val_acc={state['val_accuracy']:.4f} from epoch {state.get('epoch', epoch)})")

    history_path = metrics_dir / f"{model_label}_history.json"
    save_training_history(history, history_path)
    plot_training_curves(
        history, filename=f"{model_label}_training_curves.png", figures_dir=figures_dir
    )
    return model, history


def save_training_history(history: TrainHistory, path: Path) -> None:
    ensure_output_dirs()
    save_json(
        {
            "train_loss": history.train_loss,
            "val_loss": history.val_loss,
            "train_accuracy": history.train_accuracy,
            "val_accuracy": history.val_accuracy,
        },
        path,
    )


def load_training_history(path: Path) -> TrainHistory:
    import json

    data = json.loads(path.read_text())
    return TrainHistory(
        train_loss=data.get("train_loss", []),
        val_loss=data.get("val_loss", []),
        train_accuracy=data.get("train_accuracy", []),
        val_accuracy=data.get("val_accuracy", []),
    )


def plot_training_curves(
    history: TrainHistory, filename: str, figures_dir: Path | None = None
) -> Path:
    out_dir = figures_dir or FIGURES_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    epochs = range(1, len(history.train_loss) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].plot(epochs, history.train_loss, label="train")
    axes[0].plot(epochs, history.val_loss, label="val")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    if history.train_accuracy:
        axes[1].plot(epochs, history.train_accuracy, label="train accuracy")
    axes[1].plot(epochs, history.val_accuracy, label="val accuracy")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    fig.tight_layout()
    out_path = out_dir / filename
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    return out_path


def train_resnet18_phased(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    *,
    head_epochs: int = 5,
    finetune_epochs: int = 10,
    head_lr: float = 1e-3,
    finetune_lr: float = 1e-4,
    weight_decay: float = 1e-4,
    checkpoint_name: str = "resnet18.pt",
    experiment_name: str | None = None,
) -> tuple[nn.Module, TrainHistory]:
    """Two-phase ResNet-18 fine-tuning: frozen backbone, then full model."""
    from src.models import freeze_backbone, unfreeze_all

    if experiment_name:
        figures_dir, metrics_dir, _ = get_experiment_dirs(experiment_name)
    else:
        figures_dir, metrics_dir = FIGURES_DIR, METRICS_DIR

    combined = TrainHistory()

    print("Phase 1: train classification head only")
    freeze_backbone(model)
    _, hist_head = train_model(
        model,
        train_loader,
        val_loader,
        device,
        epochs=head_epochs,
        lr=head_lr,
        weight_decay=weight_decay,
        checkpoint_name=f"tmp_{checkpoint_name}",
        model_label="resnet18_head",
        experiment_name=experiment_name,
    )
    combined.train_loss.extend(hist_head.train_loss)
    combined.val_loss.extend(hist_head.val_loss)
    combined.train_accuracy.extend(hist_head.train_accuracy)
    combined.val_accuracy.extend(hist_head.val_accuracy)

    print("Phase 2: fine-tune full ResNet-18")
    unfreeze_all(model)
    model, hist_ft = train_model(
        model,
        train_loader,
        val_loader,
        device,
        epochs=finetune_epochs,
        lr=finetune_lr,
        weight_decay=weight_decay,
        checkpoint_name=checkpoint_name,
        model_label="resnet18_finetune",
        experiment_name=experiment_name,
    )
    combined.train_loss.extend(hist_ft.train_loss)
    combined.val_loss.extend(hist_ft.val_loss)
    combined.train_accuracy.extend(hist_ft.train_accuracy)
    combined.val_accuracy.extend(hist_ft.val_accuracy)

    save_training_history(combined, metrics_dir / "resnet18_history.json")
    plot_training_curves(
        combined, filename="resnet18_training_curves.png", figures_dir=figures_dir
    )
    return model, combined

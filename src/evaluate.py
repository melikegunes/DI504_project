"""Evaluation metrics, confusion matrix, comparison table, Optuna HPO."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.utils import FIGURES_DIR, METRICS_DIR, ensure_output_dirs, get_device, save_figure, save_json


def collect_predictions(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_preds: list[int] = []
    all_labels: list[int] = []

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="eval", leave=False):
            images = images.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.numpy().tolist())

    return np.array(all_labels), np.array(all_preds)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
    }


def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    model_name: str,
    save_prefix: str,
    metrics_dir: Path | None = None,
    figures_dir: Path | None = None,
) -> dict:
    ensure_output_dirs()
    metrics_out = metrics_dir or METRICS_DIR
    figures_out = figures_dir or FIGURES_DIR
    metrics_out.mkdir(parents=True, exist_ok=True)
    figures_out.mkdir(parents=True, exist_ok=True)

    model = model.to(device)
    y_true, y_pred = collect_predictions(model, loader, device)
    metrics = compute_metrics(y_true, y_pred)
    metrics["model"] = model_name

    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    save_json(metrics, metrics_out / f"{save_prefix}_test_metrics.json")
    save_json(report, metrics_out / f"{save_prefix}_classification_report.json")

    plot_confusion_matrix(
        y_true,
        y_pred,
        filename=f"{save_prefix}_confusion_matrix.png",
        title=f"Confusion matrix — {model_name}",
        figures_dir=figures_out,
    )
    print(f"{model_name}: accuracy={metrics['accuracy']:.4f}, macro_f1={metrics['macro_f1']:.4f}")
    return metrics


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    filename: str,
    title: str,
    max_labels: int = 37,
    figures_dir: Path | None = None,
) -> Path:
    out_dir = figures_dir or FIGURES_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    cm = confusion_matrix(y_true, y_pred, labels=list(range(max_labels)))
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    out_path = out_dir / filename
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    return out_path


def build_comparison_table(results: list[dict]) -> pd.DataFrame:
    rows = []
    for r in results:
        rows.append(
            {
                "Model": r.get("model", ""),
                "Initialization": r.get("initialization", ""),
                "Augmentation": r.get("augmentation", ""),
                "Epochs": r.get("epochs", ""),
                "Best Validation Accuracy": r.get("best_val_accuracy", np.nan),
                "Test Accuracy": r.get("accuracy", np.nan),
                "Macro F1": r.get("macro_f1", np.nan),
            }
        )
    df = pd.DataFrame(rows)
    ensure_output_dirs()
    df.to_csv(METRICS_DIR / "final_comparison.csv", index=False)
    return df


def run_optuna_study(
    *,
    num_classes: int = 37,
    n_trials: int = 10,
    head_epochs: int = 3,
) -> dict:
    """
    Limited hyperparameter search for ResNet-18 (lr, batch size, weight decay).
    Uses validation accuracy as the objective.
    """
    import optuna

    from src.data import get_dataloaders, load_datasets
    from src.models import build_resnet18, freeze_backbone

    device = get_device()

    def objective(trial: optuna.Trial) -> float:
        lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
        batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])

        train_ds, val_ds, test_ds, _ = load_datasets()
        tr_loader, va_loader, _ = get_dataloaders(train_ds, val_ds, test_ds, batch_size=batch_size)

        model = build_resnet18(num_classes=num_classes, pretrained=True)
        freeze_backbone(model)

        model = model.to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=weight_decay
        )

        best_val_acc = 0.0
        for _ in range(head_epochs):
            model.train()
            for images, labels in tr_loader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                loss = criterion(model(images), labels)
                loss.backward()
                optimizer.step()

            model.eval()
            correct = total = 0
            with torch.no_grad():
                for images, labels in va_loader:
                    images, labels = images.to(device), labels.to(device)
                    preds = model(images).argmax(dim=1)
                    correct += (preds == labels).sum().item()
                    total += labels.size(0)
            best_val_acc = max(best_val_acc, correct / total)

        return best_val_acc

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    save_json(
        {"best_params": study.best_params, "best_value": study.best_value, "n_trials": n_trials},
        METRICS_DIR / "optuna_best_params.json",
    )
    print("Optuna best params:", study.best_params)
    return study.best_params

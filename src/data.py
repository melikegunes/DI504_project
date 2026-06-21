"""Oxford-IIIT Pet dataset loading, transforms, and exploration."""

from __future__ import annotations

from collections import Counter
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, Subset, random_split
from torchvision import datasets, transforms
from torchvision.datasets import OxfordIIITPet

from src.utils import DATA_DIR, FIGURES_DIR, IMAGENET_MEAN, IMAGENET_STD, ensure_output_dirs, save_figure

INPUT_SIZE = 224


def get_transforms(split: Literal["train", "val", "test"]) -> transforms.Compose:
    """Basic preprocessing for train/val/test (resize/crop, flip on train, normalize)."""
    normalize = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)

    if split == "train":
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(INPUT_SIZE, scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]
        )

    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(INPUT_SIZE),
            transforms.ToTensor(),
            normalize,
        ]
    )


def _unwrap_target(target) -> int:
    if isinstance(target, (list, tuple)):
        return int(target[0])
    return int(target)


class PetDataset(Dataset):
    """Thin wrapper so targets are always integer class indices."""

    def __init__(self, base: Dataset):
        self.base = base

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int):
        image, target = self.base[index]
        return image, _unwrap_target(target)


def load_datasets(
    val_ratio: float = 0.2,
    seed: int = 42,
    download: bool = True,
) -> tuple[PetDataset, PetDataset, PetDataset, list[str]]:
    """
    Load Oxford-IIIT Pet with official trainval/test split.

    trainval is subdivided into train and validation subsets.
    """
    train_tf = get_transforms("train")
    eval_tf = get_transforms("val")

    trainval_base = OxfordIIITPet(
        root=str(DATA_DIR),
        split="trainval",
        target_types="category",
        transform=train_tf,
        download=download,
    )
    test_base = OxfordIIITPet(
        root=str(DATA_DIR),
        split="test",
        target_types="category",
        transform=eval_tf,
        download=download,
    )

    class_names = trainval_base.classes
    n_trainval = len(trainval_base)
    n_val = int(n_trainval * val_ratio)
    n_train = n_trainval - n_val

    generator = torch.Generator().manual_seed(seed)
    train_subset, val_subset = random_split(range(n_trainval), [n_train, n_val], generator=generator)

    train_base = OxfordIIITPet(
        root=str(DATA_DIR),
        split="trainval",
        target_types="category",
        transform=train_tf,
        download=False,
    )
    val_base = OxfordIIITPet(
        root=str(DATA_DIR),
        split="trainval",
        target_types="category",
        transform=eval_tf,
        download=False,
    )

    train_ds = PetDataset(Subset(train_base, train_subset.indices))
    val_ds = PetDataset(Subset(val_base, val_subset.indices))
    test_ds = PetDataset(test_base)

    return train_ds, val_ds, test_ds, class_names


def get_dataloaders(
    train_ds: Dataset,
    val_ds: Dataset,
    test_ds: Dataset,
    batch_size: int = 32,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory
    )
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader, test_loader


def dataset_summary(train_ds: Dataset, val_ds: Dataset, test_ds: Dataset) -> pd.DataFrame:
    rows = [
        ("train", len(train_ds)),
        ("validation", len(val_ds)),
        ("test", len(test_ds)),
        ("total", len(train_ds) + len(val_ds) + len(test_ds)),
    ]
    return pd.DataFrame(rows, columns=["split", "num_samples"])


def class_distribution(dataset: Dataset) -> pd.Series:
    labels = [_unwrap_target(dataset[i][1]) for i in range(len(dataset))]
    counts = Counter(labels)
    return pd.Series(counts).sort_index()


def plot_class_distribution(train_ds: Dataset, class_names: list[str], filename: str = "class_distribution.png") -> None:
    ensure_output_dirs()
    counts = class_distribution(train_ds)
    names = [class_names[i] for i in counts.index]

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(range(len(counts)), counts.values)
    ax.set_xticks(range(len(counts)))
    ax.set_xticklabels(names, rotation=90, fontsize=7)
    ax.set_title("Training set class distribution")
    ax.set_ylabel("Count")
    fig.tight_layout()
    save_figure(fig, filename)


def plot_sample_images(
    dataset: Dataset,
    class_names: list[str],
    n_samples: int = 12,
    filename: str = "sample_images.png",
    title: str = "Sample pet images",
) -> None:
    ensure_output_dirs()
    indices = np.linspace(0, len(dataset) - 1, n_samples, dtype=int)

    cols = 4
    rows = int(np.ceil(n_samples / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(12, 3 * rows))
    axes = np.array(axes).reshape(-1)

    for ax, idx in zip(axes, indices):
        image, label = dataset[idx]
        if torch.is_tensor(image):
            img = image.permute(1, 2, 0).numpy()
            img = np.clip(img * np.array(IMAGENET_STD) + np.array(IMAGENET_MEAN), 0, 1)
        else:
            img = np.array(image) / 255.0
        ax.imshow(img)
        ax.set_title(class_names[label], fontsize=8)
        ax.axis("off")

    for ax in axes[n_samples:]:
        ax.axis("off")

    fig.suptitle(title)
    fig.tight_layout()
    save_figure(fig, filename)


def plot_raw_vs_transformed(
    class_names: list[str],
    index: int = 0,
    filename: str = "raw_vs_transformed.png",
) -> None:
    """Show one image before and after standard train transforms."""
    ensure_output_dirs()
    raw_ds = OxfordIIITPet(
        root=str(DATA_DIR),
        split="trainval",
        target_types="category",
        transform=None,
        download=False,
    )
    raw_image, target = raw_ds[index]
    label = _unwrap_target(target)

    standard_tf = get_transforms("train")
    transformed = standard_tf(raw_image)

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(raw_image)
    axes[0].set_title(f"Raw — {class_names[label]}")
    axes[0].axis("off")

    img = transformed.permute(1, 2, 0).numpy()
    img = np.clip(img * np.array(IMAGENET_STD) + np.array(IMAGENET_MEAN), 0, 1)
    axes[1].imshow(img)
    axes[1].set_title("After standard train transforms")
    axes[1].axis("off")

    fig.tight_layout()
    save_figure(fig, filename)

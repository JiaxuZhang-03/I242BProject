"""Plot helpers used by the entry scripts."""

from __future__ import annotations

import os
import random
from pathlib import Path

from .paths import RESULT_DIR, ensure_dir

os.environ.setdefault("MPLCONFIGDIR", str(ensure_dir(RESULT_DIR / ".matplotlib")))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from PIL import Image
from sklearn.metrics import auc, roc_curve
from torchvision import datasets


def plot_dataset_counts(summary: pd.DataFrame, output_path: str | Path) -> None:
    output_path = Path(output_path)
    ensure_dir(output_path.parent)
    plt.figure(figsize=(8, 5))
    sns.barplot(data=summary, x="split", y="count", hue="class")
    plt.title("Dataset image counts")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_sample_grid(
    dataset_root: str | Path,
    output_path: str | Path,
    split: str = "train",
    samples_per_class: int = 6,
    seed: int = 42,
) -> None:
    dataset = datasets.ImageFolder(Path(dataset_root) / split)
    rng = random.Random(seed)
    class_to_paths: dict[int, list[str]] = {idx: [] for idx in range(len(dataset.classes))}
    for path, class_idx in dataset.samples:
        class_to_paths[class_idx].append(path)

    rows = len(dataset.classes)
    cols = samples_per_class
    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(cols * 2.2, rows * 2.4),
        squeeze=False,
    )

    for class_idx, class_name in enumerate(dataset.classes):
        paths = class_to_paths[class_idx]
        rng.shuffle(paths)
        selected = paths[:samples_per_class]
        for col_idx in range(cols):
            axis = axes[class_idx][col_idx]
            axis.axis("off")
            if col_idx < len(selected):
                image = Image.open(selected[col_idx]).convert("RGB")
                axis.imshow(image)
            if col_idx == 0:
                axis.set_title(class_name)

    output_path = Path(output_path)
    ensure_dir(output_path.parent)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_history(history: pd.DataFrame, output_path: str | Path) -> None:
    output_path = Path(output_path)
    ensure_dir(output_path.parent)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    if "train_loss" in history:
        axes[0].plot(history["epoch"], history["train_loss"], label="train")
    if "val_loss" in history:
        axes[0].plot(history["epoch"], history["val_loss"], label="val")
    if "supcon_loss" in history:
        axes[0].plot(history["epoch"], history["supcon_loss"], label="supcon")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    if axes[0].get_legend_handles_labels()[0]:
        axes[0].legend()

    if "train_acc" in history:
        axes[1].plot(history["epoch"], history["train_acc"], label="train")
    if "val_acc" in history:
        axes[1].plot(history["epoch"], history["val_acc"], label="val")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    if axes[1].get_legend_handles_labels()[0]:
        axes[1].legend()
    else:
        axes[1].axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_confusion_matrix(
    matrix: pd.DataFrame,
    output_path: str | Path,
) -> None:
    output_path = Path(output_path)
    ensure_dir(output_path.parent)
    plt.figure(figsize=(6, 5))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.ylabel("True")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_binary_roc(
    y_true,
    positive_scores,
    output_path: str | Path,
) -> None:
    output_path = Path(output_path)
    ensure_dir(output_path.parent)
    fpr, tpr, _ = roc_curve(y_true, positive_scores)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("ROC curve")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()

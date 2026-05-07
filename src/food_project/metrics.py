"""Metrics and prediction helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
from tqdm import tqdm


def predict(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    y_true: list[np.ndarray] = []
    y_prob: list[np.ndarray] = []

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Predict", leave=False):
            images = images.to(device)
            logits = model(images)
            probs = torch.softmax(logits, dim=1)
            y_true.append(labels.cpu().numpy())
            y_prob.append(probs.cpu().numpy())

    y_true_array = np.concatenate(y_true)
    y_prob_array = np.concatenate(y_prob)
    y_pred_array = y_prob_array.argmax(axis=1)
    return y_true_array, y_pred_array, y_prob_array


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    class_names: list[str],
) -> dict[str, object]:
    labels = list(range(len(class_names)))
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        zero_division=0,
    )
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )
    weighted_precision, weighted_recall, weighted_f1, _ = (
        precision_recall_fscore_support(
            y_true,
            y_pred,
            average="weighted",
            zero_division=0,
        )
    )

    result: dict[str, object] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "weighted_precision": float(weighted_precision),
        "weighted_recall": float(weighted_recall),
        "weighted_f1": float(weighted_f1),
        "per_class": {},
    }

    for idx, name in enumerate(class_names):
        result["per_class"][name] = {
            "precision": float(precision[idx]),
            "recall": float(recall[idx]),
            "f1": float(f1[idx]),
            "support": int(support[idx]),
        }

    if len(class_names) == 2:
        try:
            result["roc_auc"] = float(roc_auc_score(y_true, y_prob[:, 1]))
        except ValueError:
            result["roc_auc"] = None

    return result


def classification_report_text(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
) -> str:
    return classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        zero_division=0,
    )


def confusion_matrix_frame(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
) -> pd.DataFrame:
    matrix = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    return pd.DataFrame(matrix, index=class_names, columns=class_names)


def predictions_frame(
    sample_paths: list[str | Path],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    class_names: list[str],
) -> pd.DataFrame:
    rows = []
    for path, true_idx, pred_idx, probs in zip(sample_paths, y_true, y_pred, y_prob):
        row = {
            "path": str(path),
            "true_label": class_names[int(true_idx)],
            "pred_label": class_names[int(pred_idx)],
            "correct": bool(true_idx == pred_idx),
        }
        for idx, name in enumerate(class_names):
            row[f"prob_{name}"] = float(probs[idx])
        rows.append(row)
    return pd.DataFrame(rows)


"""Evaluate a trained classifier checkpoint on val or test data."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torchvision import datasets

from food_project.checkpointing import load_classifier_from_checkpoint
from food_project.data import eval_transform
from food_project.metrics import (
    classification_report_text,
    compute_metrics,
    confusion_matrix_frame,
    predictions_frame,
    predict,
)
from food_project.paths import DATA_DIR, RESULT_DIR, ensure_dir, find_dataset_root, timestamp
from food_project.plots import plot_binary_roc, plot_confusion_matrix
from food_project.utils import get_device, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=RESULT_DIR / "evaluation")
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--model-name", type=str, default=None)
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", type=str, default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = get_device(args.device)
    model, checkpoint, class_names = load_classifier_from_checkpoint(
        args.checkpoint,
        device=device,
        model_name=args.model_name,
    )

    config = checkpoint.get("config", {})
    image_size = args.image_size or int(config.get("image_size", 224))
    dataset_root = find_dataset_root(args.data_dir)
    dataset = datasets.ImageFolder(
        dataset_root / args.split,
        transform=eval_transform(image_size),
    )
    if dataset.classes != class_names:
        raise ValueError(
            f"Dataset classes {dataset.classes} do not match checkpoint classes {class_names}."
        )

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    run_dir = ensure_dir(args.output_dir / f"{args.split}_{timestamp()}")
    y_true, y_pred, y_prob = predict(model, loader, device)
    metrics = compute_metrics(y_true, y_pred, y_prob, class_names)
    matrix = confusion_matrix_frame(y_true, y_pred, class_names)
    sample_paths = [path for path, _class_idx in dataset.samples]
    predictions = predictions_frame(sample_paths, y_true, y_pred, y_prob, class_names)

    save_json(
        {
            "checkpoint": str(args.checkpoint),
            "dataset_root": str(dataset_root),
            "split": args.split,
            "image_size": image_size,
            "metrics": metrics,
        },
        run_dir / "metrics.json",
    )
    matrix.to_csv(run_dir / "confusion_matrix.csv")
    predictions.to_csv(run_dir / "predictions.csv", index=False)
    (run_dir / "classification_report.txt").write_text(
        classification_report_text(y_true, y_pred, class_names),
        encoding="utf-8",
    )
    plot_confusion_matrix(matrix, run_dir / "confusion_matrix.png")
    if len(class_names) == 2:
        plot_binary_roc(y_true, y_prob[:, 1], run_dir / "roc_curve.png")

    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Wrote evaluation outputs to: {run_dir}")


if __name__ == "__main__":
    main()


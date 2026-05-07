"""Generate Grad-CAM visualizations for a trained classifier checkpoint."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import pandas as pd
from torchvision import datasets

from food_project.checkpointing import load_classifier_from_checkpoint
from food_project.data import eval_transform
from food_project.gradcam import save_gradcam_overlay
from food_project.paths import DATA_DIR, RESULT_DIR, ensure_dir, find_dataset_root, timestamp
from food_project.utils import get_device, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--image", type=Path, default=None)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=RESULT_DIR / "gradcam")
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--num-images", type=int, default=12)
    parser.add_argument("--model-name", type=str, default=None)
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--target-class", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    return parser.parse_args()


def resolve_target_class(value: str | None, class_names: list[str]) -> int | None:
    if value is None:
        return None
    if value.isdigit():
        class_idx = int(value)
        if class_idx < 0 or class_idx >= len(class_names):
            raise ValueError(f"target-class index out of range: {value}")
        return class_idx
    if value not in class_names:
        raise ValueError(f"target-class must be one of {class_names} or an index.")
    return class_names.index(value)


def select_images(args: argparse.Namespace) -> list[tuple[Path, int | None]]:
    if args.image is not None:
        return [(args.image, None)]

    dataset_root = find_dataset_root(args.data_dir)
    dataset = datasets.ImageFolder(dataset_root / args.split)
    samples = [(Path(path), class_idx) for path, class_idx in dataset.samples]
    rng = random.Random(args.seed)
    rng.shuffle(samples)
    return samples[: args.num_images]


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
    transform = eval_transform(image_size)
    target_class = resolve_target_class(args.target_class, class_names)

    run_dir = ensure_dir(args.output_dir / f"gradcam_{timestamp()}")
    rows = []
    for idx, (image_path, true_idx) in enumerate(select_images(args), start=1):
        safe_stem = image_path.stem[:60].replace(" ", "_")
        output_path = run_dir / f"{idx:03d}_{safe_stem}_gradcam.png"
        result = save_gradcam_overlay(
            model,
            image_path,
            transform,
            output_path,
            device=device,
            image_size=image_size,
            target_class=target_class,
        )
        probabilities = result.pop("probabilities")
        pred_idx = int(max(range(len(probabilities)), key=probabilities.__getitem__))
        row = {
            **result,
            "true_label": class_names[true_idx] if true_idx is not None else None,
            "pred_label": class_names[pred_idx],
            "target_label": class_names[int(result["target_class"])],
        }
        for class_idx, class_name in enumerate(class_names):
            row[f"prob_{class_name}"] = float(probabilities[class_idx])
        rows.append(row)

    manifest = pd.DataFrame(rows)
    manifest.to_csv(run_dir / "gradcam_manifest.csv", index=False)
    save_json(
        {
            "checkpoint": str(args.checkpoint),
            "image_size": image_size,
            "class_names": class_names,
            "target_class": target_class,
            "num_images": len(rows),
        },
        run_dir / "gradcam_config.json",
    )
    print(f"Wrote Grad-CAM outputs to: {run_dir}")


if __name__ == "__main__":
    main()


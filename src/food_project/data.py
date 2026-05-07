"""Dataset discovery, transforms, and dataloaders."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from torchvision import datasets, transforms

from .paths import find_dataset_root


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class TwoCropTransform:
    """Create two augmented views of one image for contrastive learning."""

    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, image):
        return self.base_transform(image), self.base_transform(image)


def train_transform(image_size: int = 224):
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size, scale=(0.65, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply(
                [transforms.ColorJitter(0.3, 0.3, 0.3, 0.08)], p=0.6
            ),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def supcon_transform(image_size: int = 224):
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size, scale=(0.5, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply(
                [transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8
            ),
            transforms.RandomGrayscale(p=0.15),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def eval_transform(image_size: int = 224):
    resize_size = int(image_size * 1.15)
    return transforms.Compose(
        [
            transforms.Resize(resize_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def get_class_names(dataset_root: str | Path) -> list[str]:
    train_dir = Path(dataset_root) / "train"
    dataset = datasets.ImageFolder(train_dir)
    return dataset.classes


def build_datasets(
    dataset_root: str | Path,
    image_size: int = 224,
    augment: bool = True,
) -> dict[str, datasets.ImageFolder]:
    dataset_root = Path(dataset_root)
    split_transforms = {
        "train": train_transform(image_size) if augment else eval_transform(image_size),
        "val": eval_transform(image_size),
        "test": eval_transform(image_size),
    }
    return {
        split: datasets.ImageFolder(dataset_root / split, transform=transform)
        for split, transform in split_transforms.items()
        if (dataset_root / split).is_dir()
    }


def build_dataloaders(
    dataset_root: str | Path,
    image_size: int = 224,
    batch_size: int = 32,
    num_workers: int = 2,
    augment: bool = True,
    pin_memory: bool = False,
) -> tuple[dict[str, torch.utils.data.DataLoader], list[str]]:
    dataset_root = Path(dataset_root)
    datasets_by_split = build_datasets(dataset_root, image_size=image_size, augment=augment)
    class_names = datasets_by_split["train"].classes

    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True

    loaders = {
        split: torch.utils.data.DataLoader(
            dataset,
            shuffle=(split == "train"),
            **loader_kwargs,
        )
        for split, dataset in datasets_by_split.items()
    }
    return loaders, class_names


def build_supcon_loader(
    dataset_root: str | Path,
    image_size: int = 224,
    batch_size: int = 64,
    num_workers: int = 2,
    pin_memory: bool = False,
):
    dataset_root = Path(dataset_root)
    dataset = datasets.ImageFolder(
        dataset_root / "train",
        transform=TwoCropTransform(supcon_transform(image_size)),
    )
    loader_kwargs = {
        "batch_size": batch_size,
        "shuffle": True,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "drop_last": True,
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
    return torch.utils.data.DataLoader(dataset, **loader_kwargs), dataset.classes


def dataset_summary(data_dir: str | Path) -> tuple[pd.DataFrame, Path]:
    dataset_root = find_dataset_root(data_dir)
    rows: list[dict[str, object]] = []

    for split in ("train", "val", "test"):
        split_dir = dataset_root / split
        if not split_dir.exists():
            continue
        for class_dir in sorted(path for path in split_dir.iterdir() if path.is_dir()):
            count = sum(
                1
                for path in class_dir.rglob("*")
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
            )
            rows.append({"split": split, "class": class_dir.name, "count": count})

    return pd.DataFrame(rows), dataset_root


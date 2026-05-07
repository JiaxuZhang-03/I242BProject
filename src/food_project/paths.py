"""Path helpers for the project layout."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
RESULT_DIR = REPO_ROOT / "result"


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def find_dataset_root(data_dir: str | Path = DATA_DIR) -> Path:
    """Find the directory that contains train/val/test ImageFolder splits."""
    data_dir = Path(data_dir)
    split_names = ("train", "val", "test")

    candidates: list[Path] = []
    if all((data_dir / split).is_dir() for split in split_names):
        candidates.append(data_dir)

    if data_dir.exists():
        for train_dir in data_dir.rglob("train"):
            parent = train_dir.parent
            if parent in candidates:
                continue
            if all((parent / split).is_dir() for split in split_names):
                candidates.append(parent)

    if not candidates:
        raise FileNotFoundError(
            f"Could not find train/val/test folders under {data_dir}."
        )

    return sorted(candidates, key=lambda path: (len(path.parts), str(path)))[0]


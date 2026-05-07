"""Generate dataset summary tables and exploratory plots."""

from __future__ import annotations

import argparse
from pathlib import Path

from food_project.data import dataset_summary
from food_project.paths import DATA_DIR, RESULT_DIR, ensure_dir
from food_project.plots import plot_dataset_counts, plot_sample_grid
from food_project.utils import save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=RESULT_DIR / "eda")
    parser.add_argument("--samples-per-class", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = ensure_dir(args.output_dir)

    summary, dataset_root = dataset_summary(args.data_dir)
    summary_path = output_dir / "dataset_counts.csv"
    summary.to_csv(summary_path, index=False)

    split_totals = summary.groupby("split")["count"].sum().to_dict()
    class_totals = summary.groupby("class")["count"].sum().to_dict()
    save_json(
        {
            "dataset_root": str(dataset_root),
            "total_images": int(summary["count"].sum()),
            "split_totals": {key: int(value) for key, value in split_totals.items()},
            "class_totals": {key: int(value) for key, value in class_totals.items()},
        },
        output_dir / "dataset_summary.json",
    )

    plot_dataset_counts(summary, output_dir / "dataset_counts.png")
    plot_sample_grid(
        dataset_root,
        output_dir / "sample_grid_train.png",
        split="train",
        samples_per_class=args.samples_per_class,
        seed=args.seed,
    )

    print(f"Dataset root: {dataset_root}")
    print(f"Wrote EDA outputs to: {output_dir}")


if __name__ == "__main__":
    main()


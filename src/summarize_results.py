"""Summarize training histories and evaluation metrics into report-ready tables."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from food_project.paths import RESULT_DIR, ensure_dir
from food_project.utils import load_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", type=Path, default=RESULT_DIR)
    parser.add_argument("--output-csv", type=Path, default=RESULT_DIR / "experiment_summary.csv")
    parser.add_argument("--output-md", type=Path, default=RESULT_DIR / "experiment_summary.md")
    return parser.parse_args()


def load_best_val_accuracy(result_dir: Path) -> dict[str, float]:
    best_by_experiment: dict[str, float] = {}
    for history_path in sorted((result_dir / "classifier").glob("*/history.csv")):
        history = pd.read_csv(history_path)
        if "val_acc" not in history:
            continue
        best_by_experiment[history_path.parent.name] = float(history["val_acc"].max())
    return best_by_experiment


def build_summary(result_dir: Path) -> pd.DataFrame:
    best_val = load_best_val_accuracy(result_dir)
    rows: list[dict[str, object]] = []

    for metrics_path in sorted((result_dir / "evaluation").glob("*/*/metrics.json")):
        payload = load_json(metrics_path)
        metrics = payload["metrics"]
        experiment = metrics_path.parents[1].name
        rows.append(
            {
                "experiment": experiment,
                "split": payload["split"],
                "best_val_acc": best_val.get(experiment),
                "test_accuracy": metrics.get("accuracy"),
                "macro_f1": metrics.get("macro_f1"),
                "weighted_f1": metrics.get("weighted_f1"),
                "roc_auc": metrics.get("roc_auc"),
                "metrics_path": str(metrics_path),
            }
        )

    summary = pd.DataFrame(rows)
    if not summary.empty:
        summary = summary.sort_values("test_accuracy", ascending=False)
    return summary


def dataframe_to_markdown(frame: pd.DataFrame) -> str:
    """Render a small dataframe as a markdown table without optional deps."""
    if frame.empty:
        return "_No evaluation metrics found._\n"

    display = frame.copy()
    for column in display.columns:
        if pd.api.types.is_float_dtype(display[column]):
            display[column] = display[column].map(
                lambda value: "" if pd.isna(value) else f"{value:.4f}"
            )
        else:
            display[column] = display[column].map(
                lambda value: "" if pd.isna(value) else str(value)
            )

    headers = [str(column) for column in display.columns]
    rows = display.values.tolist()
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    summary = build_summary(args.result_dir)
    ensure_dir(args.output_csv.parent)
    summary.to_csv(args.output_csv, index=False)
    args.output_md.write_text(dataframe_to_markdown(summary), encoding="utf-8")
    print(summary.to_string(index=False))
    print(f"Wrote summary CSV to: {args.output_csv}")
    print(f"Wrote summary markdown to: {args.output_md}")


if __name__ == "__main__":
    main()

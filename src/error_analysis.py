"""Analyze model errors from an evaluation predictions CSV."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / "result" / ".matplotlib"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from PIL import Image, ImageOps

from food_project.paths import RESULT_DIR, ensure_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions-csv", type=Path, default=None)
    parser.add_argument("--experiment", type=str, default="resnet18_unfrozen_from_frozen_lr3e-5_128_e8")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--plot-dir", type=Path, default=RESULT_DIR / "summary_plots")
    parser.add_argument("--summary-csv", type=Path, default=RESULT_DIR / "error_analysis_summary.csv")
    parser.add_argument("--summary-md", type=Path, default=RESULT_DIR / "error_analysis_summary.md")
    parser.add_argument("--max-images", type=int, default=16)
    return parser.parse_args()


def find_latest_predictions(experiment: str) -> Path:
    candidates = sorted((RESULT_DIR / "evaluation" / experiment).glob("*/predictions.csv"))
    if not candidates:
        raise FileNotFoundError(
            f"No predictions.csv found for {experiment}. Run evaluate_model.py first."
        )
    return max(candidates, key=lambda path: path.stat().st_mtime)


def infer_experiment(predictions_csv: Path, fallback: str) -> str:
    parts = predictions_csv.parts
    if "evaluation" in parts:
        idx = parts.index("evaluation")
        if len(parts) > idx + 1:
            return parts[idx + 1]
    return fallback


def normalize_predictions(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    result = frame.copy()
    if result["correct"].dtype != bool:
        result["correct"] = result["correct"].astype(str).str.lower().eq("true")

    prob_columns = [column for column in result.columns if column.startswith("prob_")]
    class_names = [column.replace("prob_", "", 1) for column in prob_columns]
    if not class_names:
        raise ValueError("Predictions CSV must contain probability columns named prob_<class>.")

    for column in prob_columns:
        result[column] = pd.to_numeric(result[column], errors="coerce")

    def probability_for_label(row: pd.Series, label_column: str) -> float:
        column = f"prob_{row[label_column]}"
        return float(row[column]) if column in row else float("nan")

    result["true_probability"] = result.apply(
        lambda row: probability_for_label(row, "true_label"),
        axis=1,
    )
    result["pred_probability"] = result.apply(
        lambda row: probability_for_label(row, "pred_label"),
        axis=1,
    )
    result["confidence"] = result[prob_columns].max(axis=1)
    if len(prob_columns) == 2:
        result["prob_margin"] = (result[prob_columns[0]] - result[prob_columns[1]]).abs()
    else:
        result["prob_margin"] = result["confidence"] - result["true_probability"]
    result["error_type"] = result["true_label"] + " -> " + result["pred_label"]
    return result, class_names


def write_markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No rows._\n"
    display = frame.copy()
    for column in display.columns:
        if pd.api.types.is_float_dtype(display[column]):
            display[column] = display[column].map(lambda value: f"{value:.4f}")
        else:
            display[column] = display[column].astype(str)
    lines = [
        "| " + " | ".join(display.columns) + " |",
        "| " + " | ".join("---" for _ in display.columns) + " |",
    ]
    for row in display.itertuples(index=False):
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines) + "\n"


def plot_confidence_histogram(frame: pd.DataFrame, path: Path) -> None:
    plt.figure(figsize=(8, 4.5))
    sns.histplot(
        data=frame,
        x="confidence",
        hue="correct",
        bins=20,
        multiple="layer",
        alpha=0.55,
        palette={True: "#4E79A7", False: "#E15759"},
    )
    plt.title("Prediction confidence: correct vs. incorrect")
    plt.xlabel("Predicted-class probability")
    plt.ylabel("Number of samples")
    plt.tight_layout()
    plt.savefig(path, dpi=220)
    plt.close()


def plot_error_types(summary: pd.DataFrame, path: Path) -> None:
    if summary.empty:
        return
    plt.figure(figsize=(7, 4.5))
    axis = sns.barplot(
        data=summary,
        x="error_type",
        y="count",
        hue="error_type",
        legend=False,
        palette=["#E15759", "#F28E2B"],
    )
    axis.bar_label(axis.containers[0], fmt="%.0f", padding=3)
    plt.title("Misclassification types")
    plt.xlabel("")
    plt.ylabel("Number of errors")
    plt.tight_layout()
    plt.savefig(path, dpi=220)
    plt.close()


def plot_misclassified_grid(misclassified: pd.DataFrame, path: Path, max_images: int) -> None:
    samples = misclassified.sort_values("confidence", ascending=False).head(max_images)
    if samples.empty:
        return

    columns = 4
    rows = (len(samples) + columns - 1) // columns
    fig, axes = plt.subplots(rows, columns, figsize=(columns * 3.2, rows * 3.6))
    axes_list = list(axes.flat) if hasattr(axes, "flat") else [axes]

    for axis, (_, row) in zip(axes_list, samples.iterrows()):
        image_path = Path(row["path"])
        with Image.open(image_path) as image:
            image = ImageOps.exif_transpose(image).convert("RGB")
            axis.imshow(image)
        axis.set_title(
            f"T: {row['true_label']}\nP: {row['pred_label']} ({row['confidence']:.2f})",
            fontsize=9,
        )
        axis.axis("off")

    for axis in axes_list[len(samples) :]:
        axis.axis("off")

    plt.suptitle("Most confident misclassified test images", y=0.995)
    plt.tight_layout()
    plt.savefig(path, dpi=220)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    predictions_csv = args.predictions_csv or find_latest_predictions(args.experiment)
    experiment = infer_experiment(predictions_csv, args.experiment)
    output_dir = ensure_dir(args.output_dir or RESULT_DIR / "error_analysis" / experiment)
    plot_dir = ensure_dir(args.plot_dir)

    predictions, class_names = normalize_predictions(pd.read_csv(predictions_csv))
    misclassified = predictions.loc[~predictions["correct"]].copy()
    correct = predictions.loc[predictions["correct"]].copy()

    error_summary = (
        misclassified.groupby(["true_label", "pred_label", "error_type"])
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )
    class_error_rates = (
        predictions.assign(error=~predictions["correct"])
        .groupby("true_label")
        .agg(samples=("path", "count"), errors=("error", "sum"))
        .reset_index()
    )
    class_error_rates["error_rate"] = class_error_rates["errors"] / class_error_rates["samples"]

    most_confident_errors = misclassified.sort_values("confidence", ascending=False)
    low_confidence_correct = correct.sort_values("confidence", ascending=True)

    misclassified.to_csv(output_dir / "misclassified.csv", index=False)
    most_confident_errors.head(args.max_images).to_csv(
        output_dir / "most_confident_errors.csv",
        index=False,
    )
    low_confidence_correct.head(args.max_images).to_csv(
        output_dir / "low_confidence_correct.csv",
        index=False,
    )
    error_summary.to_csv(output_dir / "error_type_summary.csv", index=False)
    class_error_rates.to_csv(output_dir / "class_error_rates.csv", index=False)

    summary = {
        "experiment": experiment,
        "predictions_csv": str(predictions_csv),
        "total_samples": int(len(predictions)),
        "num_errors": int(len(misclassified)),
        "accuracy": float(predictions["correct"].mean()),
        "mean_error_confidence": float(misclassified["confidence"].mean()),
        "median_error_confidence": float(misclassified["confidence"].median()),
        "high_confidence_errors_0.80": int((misclassified["confidence"] >= 0.80).sum()),
        "classes": class_names,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    summary_row = pd.DataFrame([summary])
    summary_row.to_csv(args.summary_csv, index=False)

    plot_confidence_histogram(predictions, plot_dir / "error_confidence_histogram.png")
    plot_error_types(error_summary, plot_dir / "error_type_counts.png")
    plot_misclassified_grid(
        most_confident_errors,
        plot_dir / "misclassified_examples_grid.png",
        args.max_images,
    )

    markdown = [
        f"# Error Analysis: {experiment}",
        "",
        f"- Predictions file: `{predictions_csv}`",
        f"- Total test samples: `{summary['total_samples']}`",
        f"- Misclassified samples: `{summary['num_errors']}`",
        f"- Accuracy: `{summary['accuracy']:.4f}`",
        f"- Mean confidence among errors: `{summary['mean_error_confidence']:.4f}`",
        f"- High-confidence errors (confidence >= 0.80): `{summary['high_confidence_errors_0.80']}`",
        "",
        "## Error Types",
        "",
        write_markdown_table(error_summary),
        "## Class Error Rates",
        "",
        write_markdown_table(class_error_rates),
        "## Most Confident Errors",
        "",
        write_markdown_table(
            most_confident_errors[
                ["path", "true_label", "pred_label", "confidence", "true_probability"]
            ].head(args.max_images)
        ),
    ]
    args.summary_md.write_text("\n".join(markdown), encoding="utf-8")

    print(f"Wrote error analysis outputs to: {output_dir}")
    print(f"Wrote summary CSV to: {args.summary_csv}")
    print(f"Wrote summary markdown to: {args.summary_md}")
    print(f"Wrote plots to: {plot_dir}")


if __name__ == "__main__":
    main()

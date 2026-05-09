"""Build a focused ResNet18 fine-tuning ablation summary."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / "result" / ".matplotlib"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from food_project.paths import RESULT_DIR, ensure_dir
from food_project.utils import load_json


DEFAULT_EXPERIMENTS = [
    "resnet18_pretrained_frozen_128_e5",
    "resnet18_layer4_from_frozen_lr3e-5_128_e8",
    "resnet18_unfrozen_from_frozen_lr1e-5_128_e8",
    "resnet18_unfrozen_from_frozen_lr3e-5_128_e8",
    "resnet18_unfrozen_from_frozen_lr1e-4_128_e8",
]

METADATA = {
    "resnet18_pretrained_frozen_128_e5": {
        "display_name": "Frozen head",
        "initialization": "ImageNet",
        "trainable_scope": "classifier head",
        "ablation_axis": "freeze policy",
        "order": 1,
    },
    "resnet18_layer4_from_frozen_lr3e-5_128_e8": {
        "display_name": "Layer4 fine-tune",
        "initialization": "Frozen checkpoint",
        "trainable_scope": "classifier + layer4",
        "ablation_axis": "freeze policy",
        "order": 2,
    },
    "resnet18_unfrozen_from_frozen_lr1e-5_128_e8": {
        "display_name": "Full fine-tune LR 1e-5",
        "initialization": "Frozen checkpoint",
        "trainable_scope": "full backbone",
        "ablation_axis": "learning rate",
        "order": 3,
    },
    "resnet18_unfrozen_from_frozen_lr3e-5_128_e8": {
        "display_name": "Full fine-tune LR 3e-5",
        "initialization": "Frozen checkpoint",
        "trainable_scope": "full backbone",
        "ablation_axis": "learning rate",
        "order": 4,
    },
    "resnet18_unfrozen_from_frozen_lr1e-4_128_e8": {
        "display_name": "Full fine-tune LR 1e-4",
        "initialization": "Frozen checkpoint",
        "trainable_scope": "full backbone",
        "ablation_axis": "learning rate",
        "order": 5,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", type=Path, default=RESULT_DIR)
    parser.add_argument("--summary-csv", type=Path, default=RESULT_DIR / "experiment_summary.csv")
    parser.add_argument("--output-csv", type=Path, default=RESULT_DIR / "ablation_summary.csv")
    parser.add_argument("--output-md", type=Path, default=RESULT_DIR / "ablation_summary.md")
    parser.add_argument("--plot-dir", type=Path, default=RESULT_DIR / "summary_plots")
    parser.add_argument("--experiments", nargs="*", default=DEFAULT_EXPERIMENTS)
    return parser.parse_args()


def load_history_metadata(result_dir: Path, experiment: str) -> dict[str, object]:
    run_dir = result_dir / "classifier" / experiment
    config_path = run_dir / "config.json"
    history_path = run_dir / "history.csv"

    metadata: dict[str, object] = {}
    if config_path.exists():
        config = load_json(config_path)
        metadata.update(
            {
                "lr": config.get("lr"),
                "epochs_configured": config.get("epochs"),
                "trainable_parameters": config.get("trainable_parameters"),
                "total_parameters": config.get("total_parameters"),
                "trainable_backbone": config.get("trainable_backbone"),
                "init_checkpoint": config.get("init_checkpoint"),
            }
        )

    if history_path.exists():
        history = pd.read_csv(history_path)
        best_idx = history["val_acc"].idxmax()
        metadata.update(
            {
                "epochs_run": int(history["epoch"].max()),
                "best_epoch": int(history.loc[best_idx, "epoch"]),
                "final_train_acc": float(history.iloc[-1]["train_acc"]),
                "final_val_acc": float(history.iloc[-1]["val_acc"]),
            }
        )
    return metadata


def dataframe_to_markdown(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No ablation rows found._\n"

    display = frame.copy()
    for column in display.columns:
        if column == "lr":
            display[column] = display[column].map(
                lambda value: "" if pd.isna(value) else f"{float(value):g}"
            )
        elif pd.api.types.is_float_dtype(display[column]):
            display[column] = display[column].map(
                lambda value: "" if pd.isna(value) else f"{value:.4f}"
            )
        else:
            display[column] = display[column].map(
                lambda value: "" if pd.isna(value) else str(value)
            )

    lines = [
        "| " + " | ".join(display.columns) + " |",
        "| " + " | ".join("---" for _ in display.columns) + " |",
    ]
    for row in display.itertuples(index=False):
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines) + "\n"


def build_ablation_frame(args: argparse.Namespace) -> pd.DataFrame:
    if not args.summary_csv.exists():
        raise FileNotFoundError(
            f"Missing {args.summary_csv}. Run `python src/summarize_results.py` first."
        )

    summary = pd.read_csv(args.summary_csv)
    rows = []
    for experiment in args.experiments:
        matches = summary.loc[summary["experiment"] == experiment]
        if matches.empty:
            continue
        row = matches.sort_values("test_accuracy", ascending=False).iloc[0].to_dict()
        row.update(METADATA.get(experiment, {"display_name": experiment, "order": 999}))
        row.update(load_history_metadata(args.result_dir, experiment))
        rows.append(row)

    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame = frame.sort_values("order")
    return frame


def plot_accuracy_bars(frame: pd.DataFrame, plot_dir: Path) -> Path:
    plot_frame = frame.melt(
        id_vars=["display_name"],
        value_vars=["best_val_acc", "test_accuracy"],
        var_name="metric",
        value_name="accuracy",
    )
    plot_frame["metric"] = plot_frame["metric"].map(
        {"best_val_acc": "Best validation", "test_accuracy": "Test"}
    )

    plt.figure(figsize=(11, 5))
    axis = sns.barplot(
        data=plot_frame,
        x="display_name",
        y="accuracy",
        hue="metric",
        palette=["#4E79A7", "#F28E2B"],
    )
    for container in axis.containers:
        axis.bar_label(container, fmt="%.3f", fontsize=8, padding=2)
    axis.set_title("ResNet18 fine-tuning ablation")
    axis.set_xlabel("")
    axis.set_ylabel("Accuracy")
    axis.set_ylim(0.80, 1.0)
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    path = plot_dir / "resnet18_ablation_accuracy.png"
    plt.savefig(path, dpi=220)
    plt.close()
    return path


def plot_learning_rate_curve(frame: pd.DataFrame, plot_dir: Path) -> Path | None:
    lr_frame = frame.loc[
        frame["trainable_scope"].eq("full backbone") & frame["lr"].notna()
    ].copy()
    if lr_frame.empty:
        return None

    lr_frame["lr"] = pd.to_numeric(lr_frame["lr"])
    lr_frame = lr_frame.sort_values("lr")
    plt.figure(figsize=(7, 4.5))
    plt.plot(lr_frame["lr"], lr_frame["test_accuracy"], marker="o", linewidth=2)
    for _, row in lr_frame.iterrows():
        plt.text(row["lr"], row["test_accuracy"] + 0.002, f"{row['test_accuracy']:.3f}", ha="center")
    plt.xscale("log")
    plt.title("Full fine-tuning learning-rate sensitivity")
    plt.xlabel("Learning rate")
    plt.ylabel("Test accuracy")
    plt.ylim(0.88, 1.0)
    plt.grid(alpha=0.25)
    plt.tight_layout()
    path = plot_dir / "resnet18_ablation_lr_sensitivity.png"
    plt.savefig(path, dpi=220)
    plt.close()
    return path


def main() -> None:
    args = parse_args()
    plot_dir = ensure_dir(args.plot_dir)
    frame = build_ablation_frame(args)
    if frame.empty:
        raise ValueError("No ablation experiments were found in the experiment summary.")

    output_columns = [
        "display_name",
        "experiment",
        "initialization",
        "trainable_scope",
        "lr",
        "epochs_run",
        "best_epoch",
        "best_val_acc",
        "test_accuracy",
        "macro_f1",
        "roc_auc",
        "trainable_parameters",
    ]
    output = frame[[column for column in output_columns if column in frame.columns]]
    ensure_dir(args.output_csv.parent)
    output.to_csv(args.output_csv, index=False)
    args.output_md.write_text(dataframe_to_markdown(output), encoding="utf-8")

    paths = [plot_accuracy_bars(frame, plot_dir)]
    lr_path = plot_learning_rate_curve(frame, plot_dir)
    if lr_path is not None:
        paths.append(lr_path)

    print(output.to_string(index=False))
    print(f"Wrote ablation CSV to: {args.output_csv}")
    print(f"Wrote ablation markdown to: {args.output_md}")
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()

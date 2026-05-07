"""Create report-ready summary plots from completed experiment outputs."""

from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(__file__).resolve().parent / ".matplotlib"),
)

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


RESULT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = RESULT_DIR / "summary_plots"

DISPLAY_NAMES = {
    "simple_cnn_128_e5": "Simple CNN",
    "simple_cnn_supcon_finetune_128_e5": "Simple CNN + SupCon",
    "resnet18_pretrained_frozen_128_e5": "ResNet18 pretrained",
}


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def pretty_name(experiment: str) -> str:
    return DISPLAY_NAMES.get(experiment, experiment.replace("_", " "))


def load_summary() -> pd.DataFrame:
    summary_path = RESULT_DIR / "experiment_summary.csv"
    if not summary_path.exists():
        raise FileNotFoundError(
            f"Missing {summary_path}. Run `python src/summarize_results.py` first."
        )
    summary = pd.read_csv(summary_path)
    summary["display_name"] = summary["experiment"].map(pretty_name)
    return summary.sort_values("test_accuracy", ascending=False)


def resolve_metrics_dir(row: pd.Series) -> Path | None:
    metrics_path = Path(str(row["metrics_path"]))
    if metrics_path.exists():
        return metrics_path.parent

    candidates = sorted((RESULT_DIR / "evaluation" / row["experiment"]).glob("*/metrics.json"))
    if candidates:
        return candidates[-1].parent
    return None


def annotate_bars(axis) -> None:
    for container in axis.containers:
        axis.bar_label(container, fmt="%.3f", fontsize=8, padding=2)


def plot_metric_comparison(summary: pd.DataFrame, output_dir: Path) -> Path:
    metric_columns = ["test_accuracy", "macro_f1", "roc_auc"]
    plot_frame = summary.melt(
        id_vars=["display_name"],
        value_vars=metric_columns,
        var_name="metric",
        value_name="score",
    )
    metric_labels = {
        "test_accuracy": "Test accuracy",
        "macro_f1": "Macro F1",
        "roc_auc": "ROC AUC",
    }
    plot_frame["metric"] = plot_frame["metric"].map(metric_labels)

    plt.figure(figsize=(10, 5))
    axis = sns.barplot(
        data=plot_frame,
        x="display_name",
        y="score",
        hue="metric",
        palette=["#3B7EA1", "#59A14F", "#E15759"],
    )
    axis.set_title("Model performance summary")
    axis.set_xlabel("")
    axis.set_ylabel("Score")
    axis.set_ylim(0.72, 1.0)
    annotate_bars(axis)
    plt.xticks(rotation=12, ha="right")
    plt.tight_layout()

    path = output_dir / "model_metric_comparison.png"
    plt.savefig(path, dpi=220)
    plt.close()
    return path


def plot_val_test_accuracy(summary: pd.DataFrame, output_dir: Path) -> Path:
    plot_frame = summary.melt(
        id_vars=["display_name"],
        value_vars=["best_val_acc", "test_accuracy"],
        var_name="split",
        value_name="accuracy",
    )
    plot_frame["split"] = plot_frame["split"].map(
        {"best_val_acc": "Best validation", "test_accuracy": "Test"}
    )

    plt.figure(figsize=(9, 5))
    axis = sns.barplot(
        data=plot_frame,
        x="display_name",
        y="accuracy",
        hue="split",
        palette=["#4E79A7", "#F28E2B"],
    )
    axis.set_title("Validation-to-test comparison")
    axis.set_xlabel("")
    axis.set_ylabel("Accuracy")
    axis.set_ylim(0.72, 0.92)
    annotate_bars(axis)
    plt.xticks(rotation=12, ha="right")
    plt.tight_layout()

    path = output_dir / "validation_vs_test_accuracy.png"
    plt.savefig(path, dpi=220)
    plt.close()
    return path


def plot_training_curves(summary: pd.DataFrame, output_dir: Path, metric: str) -> Path:
    metric_title = "Accuracy" if metric == "acc" else "Loss"
    plt.figure(figsize=(10, 5))

    for _, row in summary.iterrows():
        history_path = RESULT_DIR / "classifier" / row["experiment"] / "history.csv"
        if not history_path.exists():
            continue
        history = pd.read_csv(history_path)
        train_column = f"train_{metric}"
        val_column = f"val_{metric}"
        if train_column in history:
            plt.plot(
                history["epoch"],
                history[train_column],
                linestyle="--",
                alpha=0.55,
                label=f"{row['display_name']} train",
            )
        if val_column in history:
            plt.plot(
                history["epoch"],
                history[val_column],
                marker="o",
                linewidth=2,
                label=f"{row['display_name']} val",
            )

    plt.title(f"Training and validation {metric_title.lower()} curves")
    plt.xlabel("Epoch")
    plt.ylabel(metric_title)
    plt.grid(alpha=0.25)
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()

    path = output_dir / f"training_{metric_title.lower()}_curves.png"
    plt.savefig(path, dpi=220)
    plt.close()
    return path


def plot_supcon_losses(output_dir: Path) -> Path | None:
    histories = sorted((RESULT_DIR / "supcon").glob("*/history.csv"))
    if not histories:
        return None

    plt.figure(figsize=(8, 4.5))
    for history_path in histories:
        history = pd.read_csv(history_path)
        if "supcon_loss" not in history:
            continue
        label = pretty_name(history_path.parent.name)
        plt.plot(
            history["epoch"],
            history["supcon_loss"],
            marker="o",
            linewidth=2,
            label=label,
        )

    plt.title("Supervised contrastive pretraining loss")
    plt.xlabel("Epoch")
    plt.ylabel("SupCon loss")
    plt.grid(alpha=0.25)
    plt.legend(fontsize=8)
    plt.tight_layout()

    path = output_dir / "supcon_loss_curve.png"
    plt.savefig(path, dpi=220)
    plt.close()
    return path


def plot_confusion_matrices(summary: pd.DataFrame, output_dir: Path) -> Path:
    matrices: list[tuple[str, pd.DataFrame]] = []
    for _, row in summary.iterrows():
        metrics_dir = resolve_metrics_dir(row)
        if metrics_dir is None:
            continue
        matrix_path = metrics_dir / "confusion_matrix.csv"
        if matrix_path.exists():
            matrices.append((str(row["display_name"]), pd.read_csv(matrix_path, index_col=0)))

    if not matrices:
        raise FileNotFoundError("No confusion_matrix.csv files found.")

    fig, axes = plt.subplots(1, len(matrices), figsize=(5 * len(matrices), 4))
    if len(matrices) == 1:
        axes = [axes]

    for axis, (label, matrix) in zip(axes, matrices):
        sns.heatmap(
            matrix,
            annot=True,
            fmt="d",
            cmap="YlGnBu",
            cbar=False,
            ax=axis,
        )
        axis.set_title(label)
        axis.set_xlabel("Predicted")
        axis.set_ylabel("True")

    plt.tight_layout()
    path = output_dir / "test_confusion_matrices.png"
    plt.savefig(path, dpi=220)
    plt.close(fig)
    return path


def save_manifest(paths: list[Path], output_dir: Path) -> None:
    manifest = {
        "output_dir": str(output_dir),
        "plots": [str(path) for path in paths if path is not None],
    }
    (output_dir / "summary_plots_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    sns.set_theme(style="whitegrid")
    output_dir = ensure_dir(OUTPUT_DIR)
    summary = load_summary()

    paths = [
        plot_metric_comparison(summary, output_dir),
        plot_val_test_accuracy(summary, output_dir),
        plot_training_curves(summary, output_dir, metric="acc"),
        plot_training_curves(summary, output_dir, metric="loss"),
    ]
    supcon_path = plot_supcon_losses(output_dir)
    if supcon_path is not None:
        paths.append(supcon_path)
    paths.append(plot_confusion_matrices(summary, output_dir))
    save_manifest(paths, output_dir)

    print(f"Wrote {len(paths)} summary plots to: {output_dir}")
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()


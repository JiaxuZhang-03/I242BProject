"""Train a classifier baseline, transfer model, or SupCon-initialized model."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn

from food_project.checkpointing import save_classifier_checkpoint
from food_project.data import build_dataloaders
from food_project.models import MODEL_NAMES, build_classifier, load_supcon_encoder
from food_project.paths import DATA_DIR, RESULT_DIR, ensure_dir, find_dataset_root, timestamp
from food_project.plots import plot_history
from food_project.train_utils import classifier_epoch, save_history
from food_project.utils import count_parameters, get_device, save_json, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=RESULT_DIR / "classifier")
    parser.add_argument("--experiment-name", type=str, default=None)
    parser.add_argument("--model-name", choices=MODEL_NAMES, default="simple_cnn")
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--freeze-backbone", action="store_true")
    parser.add_argument(
        "--init-checkpoint",
        type=Path,
        default=None,
        help="Optional classifier checkpoint used to initialize model weights before training.",
    )
    parser.add_argument("--supcon-checkpoint", type=Path, default=None)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.init_checkpoint is not None and args.supcon_checkpoint is not None:
        raise ValueError("Use either --init-checkpoint or --supcon-checkpoint, not both.")

    set_seed(args.seed)
    device = get_device(args.device)
    dataset_root = find_dataset_root(args.data_dir)

    run_name = args.experiment_name or f"{args.model_name}_{timestamp()}"
    run_dir = ensure_dir(args.output_dir / run_name)

    loaders, class_names = build_dataloaders(
        dataset_root,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        augment=True,
        pin_memory=(device.type == "cuda"),
    )

    model = build_classifier(
        args.model_name,
        num_classes=len(class_names),
        pretrained=args.pretrained,
        freeze_backbone=args.freeze_backbone,
        dropout=args.dropout,
    )
    if args.init_checkpoint is not None:
        checkpoint = torch.load(args.init_checkpoint, map_location="cpu", weights_only=False)
        load_result = model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        print(f"Initialized classifier from {args.init_checkpoint}: {load_result}")

    if args.supcon_checkpoint is not None:
        load_result = load_supcon_encoder(model, args.supcon_checkpoint, strict=False)
        print(f"Loaded SupCon encoder from {args.supcon_checkpoint}: {load_result}")

    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(args.epochs, 1),
    )

    config = {
        **vars(args),
        "data_dir": str(args.data_dir),
        "dataset_root": str(dataset_root),
        "output_dir": str(run_dir),
        "model_name": args.model_name,
        "class_names": class_names,
        "device": str(device),
        "trainable_parameters": count_parameters(model, trainable_only=True),
        "total_parameters": count_parameters(model, trainable_only=False),
    }
    save_json(config, run_dir / "config.json")

    history: list[dict[str, float]] = []
    best_val_acc = -1.0
    stale_epochs = 0

    for epoch in range(1, args.epochs + 1):
        train_stats = classifier_epoch(
            model,
            loaders["train"],
            criterion,
            device,
            optimizer=optimizer,
            desc=f"Epoch {epoch}/{args.epochs} train",
        )
        val_stats = classifier_epoch(
            model,
            loaders["val"],
            criterion,
            device,
            optimizer=None,
            desc=f"Epoch {epoch}/{args.epochs} val",
        )
        scheduler.step()

        row = {
            "epoch": epoch,
            "train_loss": train_stats["loss"],
            "train_acc": train_stats["acc"],
            "val_loss": val_stats["loss"],
            "val_acc": val_stats["acc"],
            "lr": optimizer.param_groups[0]["lr"],
        }
        history.append(row)
        print(
            f"Epoch {epoch:03d}: "
            f"train_loss={row['train_loss']:.4f} train_acc={row['train_acc']:.4f} "
            f"val_loss={row['val_loss']:.4f} val_acc={row['val_acc']:.4f}"
        )

        save_classifier_checkpoint(
            run_dir / "last_model.pt",
            model,
            optimizer,
            epoch,
            config,
            class_names,
            metrics=row,
        )

        if row["val_acc"] > best_val_acc:
            best_val_acc = row["val_acc"]
            stale_epochs = 0
            save_classifier_checkpoint(
                run_dir / "best_model.pt",
                model,
                optimizer,
                epoch,
                config,
                class_names,
                metrics=row,
            )
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                print(f"Early stopping after {args.patience} stale epochs.")
                break

    history_frame = save_history(history, run_dir / "history.csv")
    plot_history(history_frame, run_dir / "training_curves.png")
    print(f"Best val_acc: {best_val_acc:.4f}")
    print(f"Wrote training outputs to: {run_dir}")


if __name__ == "__main__":
    main()

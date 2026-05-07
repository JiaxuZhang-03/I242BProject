"""Run supervised contrastive pretraining and save encoder checkpoints."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from food_project.checkpointing import save_supcon_checkpoint
from food_project.data import build_supcon_loader
from food_project.losses import SupConLoss
from food_project.models import MODEL_NAMES, build_supcon_model
from food_project.paths import DATA_DIR, RESULT_DIR, ensure_dir, find_dataset_root, timestamp
from food_project.plots import plot_history
from food_project.train_utils import save_history, supcon_epoch
from food_project.utils import count_parameters, get_device, save_json, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=RESULT_DIR / "supcon")
    parser.add_argument("--experiment-name", type=str, default=None)
    parser.add_argument("--model-name", choices=MODEL_NAMES, default="simple_cnn")
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--projection-dim", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = get_device(args.device)
    dataset_root = find_dataset_root(args.data_dir)

    run_name = args.experiment_name or f"{args.model_name}_supcon_{timestamp()}"
    run_dir = ensure_dir(args.output_dir / run_name)

    loader, class_names = build_supcon_loader(
        dataset_root,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    model = build_supcon_model(
        args.model_name,
        pretrained=args.pretrained,
        projection_dim=args.projection_dim,
    ).to(device)
    criterion = SupConLoss(temperature=args.temperature)
    optimizer = torch.optim.AdamW(
        model.parameters(),
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
    best_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        stats = supcon_epoch(
            model,
            loader,
            criterion,
            device,
            optimizer,
            desc=f"Epoch {epoch}/{args.epochs} supcon",
        )
        scheduler.step()
        row = {
            "epoch": epoch,
            "supcon_loss": stats["loss"],
            "lr": optimizer.param_groups[0]["lr"],
        }
        history.append(row)
        print(f"Epoch {epoch:03d}: supcon_loss={row['supcon_loss']:.4f}")

        save_supcon_checkpoint(
            run_dir / "last_supcon.pt",
            model,
            optimizer,
            epoch,
            config,
            class_names,
            metrics=row,
        )
        if row["supcon_loss"] < best_loss:
            best_loss = row["supcon_loss"]
            save_supcon_checkpoint(
                run_dir / "best_supcon.pt",
                model,
                optimizer,
                epoch,
                config,
                class_names,
                metrics=row,
            )

    history_frame = save_history(history, run_dir / "history.csv")
    plot_history(history_frame, run_dir / "supcon_curves.png")
    print(f"Best SupCon loss: {best_loss:.4f}")
    print(f"Wrote SupCon outputs to: {run_dir}")


if __name__ == "__main__":
    main()


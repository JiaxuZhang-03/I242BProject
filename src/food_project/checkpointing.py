"""Checkpoint save/load helpers."""

from __future__ import annotations

from pathlib import Path

import torch

from .models import build_classifier
from .paths import ensure_dir


def save_classifier_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    epoch: int,
    config: dict,
    class_names: list[str],
    metrics: dict | None = None,
) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    payload = {
        "epoch": epoch,
        "config": config,
        "class_names": class_names,
        "model_state_dict": model.state_dict(),
        "metrics": metrics or {},
    }
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    torch.save(payload, path)


def save_supcon_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    epoch: int,
    config: dict,
    class_names: list[str],
    metrics: dict | None = None,
) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    payload = {
        "epoch": epoch,
        "config": config,
        "class_names": class_names,
        "encoder_state_dict": model.encoder.state_dict(),
        "projection_state_dict": model.projection_head.state_dict(),
        "metrics": metrics or {},
    }
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    torch.save(payload, path)


def load_classifier_from_checkpoint(
    checkpoint_path: str | Path,
    device: torch.device,
    model_name: str | None = None,
):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint.get("config", {})
    class_names = checkpoint["class_names"]
    resolved_model_name = model_name or config.get("model_name") or config.get("model")
    if resolved_model_name is None:
        raise ValueError("Checkpoint config is missing model_name.")

    model = build_classifier(
        resolved_model_name,
        num_classes=len(class_names),
        pretrained=False,
        freeze_backbone=False,
        dropout=float(config.get("dropout", 0.2)),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, checkpoint, class_names


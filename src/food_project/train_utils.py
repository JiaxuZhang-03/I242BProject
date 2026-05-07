"""Training loops for classifier and supervised contrastive pretraining."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm

from .paths import ensure_dir


def classifier_epoch(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    desc: str = "",
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(loader, desc=desc, leave=False):
        images = images.to(device)
        labels = labels.to(device)

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            logits = model(images)
            loss = criterion(logits, labels)
            if training:
                loss.backward()
                optimizer.step()

        batch_size = labels.size(0)
        running_loss += loss.item() * batch_size
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += batch_size

    return {
        "loss": running_loss / max(total, 1),
        "acc": correct / max(total, 1),
    }


def supcon_epoch(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer,
    desc: str = "",
) -> dict[str, float]:
    model.train()
    running_loss = 0.0
    total = 0

    for views, labels in tqdm(loader, desc=desc, leave=False):
        if not isinstance(views, (list, tuple)) or len(views) != 2:
            raise ValueError("SupCon loader must return two augmented views.")

        view_1, view_2 = views
        batch_size = labels.size(0)
        images = torch.cat([view_1, view_2], dim=0).to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)
        projections = model(images)
        features = torch.stack(torch.split(projections, batch_size, dim=0), dim=1)
        loss = criterion(features, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * batch_size
        total += batch_size

    return {"loss": running_loss / max(total, 1)}


def save_history(history: list[dict[str, float]], path: str | Path) -> pd.DataFrame:
    path = Path(path)
    ensure_dir(path.parent)
    frame = pd.DataFrame(history)
    frame.to_csv(path, index=False)
    return frame


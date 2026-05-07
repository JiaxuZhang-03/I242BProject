"""Loss functions for representation learning."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class SupConLoss(nn.Module):
    """Supervised contrastive loss for a batch with multiple views per sample."""

    def __init__(self, temperature: float = 0.07, base_temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature

    def forward(
        self,
        features: torch.Tensor,
        labels: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if features.ndim < 3:
            raise ValueError("features must have shape [batch, views, dim].")

        device = features.device
        features = F.normalize(features, dim=-1)
        batch_size = features.shape[0]

        if labels is not None and mask is not None:
            raise ValueError("Cannot define both labels and mask.")
        if labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32, device=device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError("labels length does not match features batch size.")
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        anchor_feature = contrast_feature
        anchor_count = contrast_count

        logits = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature,
        )
        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits = logits - logits_max.detach()

        mask = mask.repeat(anchor_count, contrast_count)
        logits_mask = torch.ones_like(mask)
        logits_mask.scatter_(
            1,
            torch.arange(batch_size * anchor_count, device=device).view(-1, 1),
            0,
        )
        mask = mask * logits_mask

        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-12)

        positive_count = mask.sum(dim=1).clamp_min(1.0)
        mean_log_prob_pos = (mask * log_prob).sum(dim=1) / positive_count
        loss = -(self.temperature / self.base_temperature) * mean_log_prob_pos
        return loss.view(anchor_count, batch_size).mean()


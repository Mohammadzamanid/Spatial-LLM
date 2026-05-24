"""
src/training/loss.py
Geo-aware auxiliary losses for spatial output heads.
These can be combined with the standard LM cross-entropy loss
when the model is trained to also predict coordinates.
"""

import torch
import torch.nn as nn
import math


class HaversineLoss(nn.Module):
    """
    Differentiable Haversine distance loss for coordinate regression.
    Use when training a coordinate prediction head alongside the LM head.
    """

    def __init__(self, reduction: str = "mean"):
        super().__init__()
        assert reduction in ("mean", "sum", "none")
        self.reduction = reduction

    def forward(
        self,
        pred_coords: torch.Tensor,  # (B, 2) [lat, lon] in degrees
        true_coords: torch.Tensor,  # (B, 2)
    ) -> torch.Tensor:
        pred_rad = pred_coords * (math.pi / 180.0)
        true_rad = true_coords * (math.pi / 180.0)

        dlat = true_rad[:, 0] - pred_rad[:, 0]
        dlon = true_rad[:, 1] - pred_rad[:, 1]

        a = (
            torch.sin(dlat / 2) ** 2
            + torch.cos(pred_rad[:, 0])
            * torch.cos(true_rad[:, 0])
            * torch.sin(dlon / 2) ** 2
        )
        # Clamp for numerical stability
        c = 2 * torch.asin(torch.clamp(torch.sqrt(a), 0.0, 1.0))
        dist_km = 6371.0 * c  # Earth radius in km

        if self.reduction == "mean":
            return dist_km.mean()
        elif self.reduction == "sum":
            return dist_km.sum()
        return dist_km


class SpatialLMLoss(nn.Module):
    """
    Combined loss: LM cross-entropy + weighted Haversine for coord regression.
    Set coord_weight=0.0 to use pure LM loss.
    """

    def __init__(self, coord_weight: float = 0.1):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(ignore_index=-100)
        self.haversine = HaversineLoss()
        self.coord_weight = coord_weight

    def forward(
        self,
        logits: torch.Tensor,       # (B, T, vocab_size)
        labels: torch.Tensor,       # (B, T)
        pred_coords: torch.Tensor | None = None,  # (B, 2)
        true_coords: torch.Tensor | None = None,  # (B, 2)
    ) -> dict[str, torch.Tensor]:
        lm_loss = self.ce(logits.view(-1, logits.size(-1)), labels.view(-1))

        total = lm_loss
        losses = {"lm_loss": lm_loss}

        if pred_coords is not None and true_coords is not None and self.coord_weight > 0:
            geo_loss = self.haversine(pred_coords, true_coords)
            total = lm_loss + self.coord_weight * geo_loss
            losses["geo_loss"] = geo_loss

        losses["total_loss"] = total
        return losses

"""Acoustic model losses."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean over positions where mask is False (mask is True at padding)."""
    keep = (~mask).float()
    while keep.dim() < x.dim():
        keep = keep.unsqueeze(-1)
    return (x * keep).sum() / keep.expand_as(x).sum().clamp(min=1.0)


def acoustic_loss(batch: dict, mel, mel_post, v) -> dict[str, torch.Tensor]:
    """Mel L1 before and after the postnet, MSE on log-duration, pitch and energy."""
    target = batch["mel"]
    losses = {
        "mel": masked_mean((mel - target).abs(), batch["mel_mask"]),
        "mel_post": masked_mean((mel_post - target).abs(), batch["mel_mask"]),
        "duration": masked_mean(F.mse_loss(v.log_duration, torch.log(batch["duration"].float() + 1), reduction="none"), batch["id_mask"]),
        "pitch": masked_mean(F.mse_loss(v.pitch, batch["pitch"], reduction="none"), batch["id_mask"]),
        "energy": masked_mean(F.mse_loss(v.energy, batch["energy"], reduction="none"), batch["id_mask"]),
    }
    losses["total"] = sum(losses.values())
    return losses

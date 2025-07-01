"""Acoustic model losses."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .models.aligner import bin_loss, forward_sum_loss
from .models.variance import average_by_duration


def masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean over positions where mask is False (mask is True at padding)."""
    keep = (~mask).float()
    while keep.dim() < x.dim():
        keep = keep.unsqueeze(-1)
    return (x * keep).sum() / keep.expand_as(x).sum().clamp(min=1.0)


def acoustic_loss(batch: dict, mel, mel_post, v, bin_weight: float = 1.0) -> dict[str, torch.Tensor]:
    """Mel L1 before and after the postnet, MSE on log-duration, pitch and energy, and the two aligner losses.

    Duration targets are the learned durations; pitch and energy targets are averaged per phoneme with them.
    """
    target = batch["mel"]
    logp, hard = v.alignment
    durations = v.durations.detach()
    mse = lambda a, b: F.mse_loss(a, b, reduction="none")  # noqa: E731
    losses = {
        "mel": masked_mean((mel - target).abs(), batch["mel_mask"]),
        "mel_post": masked_mean((mel_post - target).abs(), batch["mel_mask"]),
        "duration": masked_mean(mse(v.log_duration, torch.log(durations.float() + 1)), batch["id_mask"]),
        "pitch": masked_mean(mse(v.pitch, average_by_duration(batch["pitch"], durations)), batch["id_mask"]),
        "energy": masked_mean(mse(v.energy, average_by_duration(batch["energy"], durations)), batch["id_mask"]),
        "align_forward_sum": forward_sum_loss(logp, (~batch["id_mask"]).sum(1), (~batch["mel_mask"]).sum(1)),
        "align_bin": bin_weight * bin_loss(logp, hard),
    }
    losses["total"] = sum(losses.values())
    return losses

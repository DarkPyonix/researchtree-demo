"""Variance adaptor: learned alignment, duration, pitch and energy prediction, and the length regulator."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn

from .aligner import Aligner


class VariancePredictor(nn.Module):
    """Two 1D convolutions (ReLU, layer norm, dropout), then a linear layer: one value per step."""

    def __init__(self, hidden: int, layers: int = 2, kernel: int = 3, dropout: float = 0.5):
        super().__init__()
        self.convs = nn.ModuleList(nn.Conv1d(hidden, hidden, kernel, padding=kernel // 2) for _ in range(layers))
        self.norms = nn.ModuleList(nn.LayerNorm(hidden) for _ in range(layers))
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # x: (batch, time, hidden); mask: True at padded steps.
        for conv, norm in zip(self.convs, self.norms):
            x = conv(x.transpose(1, 2)).transpose(1, 2)
            x = self.dropout(norm(torch.relu(x)))
        return self.out(x).squeeze(-1).masked_fill(mask, 0.0)


class LengthRegulator(nn.Module):
    """Repeat each phoneme's hidden vector as many times as its duration in frames."""

    def forward(self, x: torch.Tensor, durations: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        frames = [torch.repeat_interleave(xi, di, dim=0) for xi, di in zip(x, durations)]
        lengths = torch.tensor([f.shape[0] for f in frames], device=x.device)
        out = nn.utils.rnn.pad_sequence(frames, batch_first=True)
        mel_mask = torch.arange(out.shape[1], device=x.device)[None, :] >= lengths[:, None]
        return out, mel_mask


def average_by_duration(values: torch.Tensor, durations: torch.Tensor) -> torch.Tensor:
    """(batch, frames) values -> (batch, phonemes) means over each phoneme's frames."""
    ends = durations.cumsum(1)
    starts = ends - durations
    cums = torch.nn.functional.pad(values.cumsum(1), (1, 0))
    total = cums.gather(1, ends) - cums.gather(1, starts)
    return total / durations.clamp(min=1)


@dataclass
class VarianceOutput:
    hidden: torch.Tensor
    mel_mask: torch.Tensor
    log_duration: torch.Tensor
    pitch: torch.Tensor
    energy: torch.Tensor
    durations: torch.Tensor | None = None
    alignment: tuple | None = None  # (log soft alignment, hard alignment) during training


class VarianceAdaptor(nn.Module):
    """Predict durations, add quantized pitch and energy embeddings per phoneme, then expand to frames."""

    def __init__(self, cfg: dict, hidden: int):
        super().__init__()
        self.duration = VariancePredictor(hidden, **cfg["predictor"])
        self.aligner = Aligner(n_mels=80, hidden=hidden)
        self.pitch = VariancePredictor(hidden, **cfg["predictor"])
        self.energy = VariancePredictor(hidden, **cfg["predictor"])
        # Pitch and energy are normalized to zero mean and unit variance over the corpus.
        self.register_buffer("pitch_bins", torch.linspace(-3.0, 3.0, cfg["pitch"]["bins"] - 1))
        self.register_buffer("energy_bins", torch.linspace(-3.0, 3.0, cfg["energy"]["bins"] - 1))
        self.pitch_embed = nn.Embedding(cfg["pitch"]["bins"], hidden)
        self.energy_embed = nn.Embedding(cfg["energy"]["bins"], hidden)
        self.regulate = LengthRegulator()
        self.log_f0_std = cfg["pitch"]["log_f0_std"]
        self.pitch_min, self.pitch_max = cfg["pitch"]["clamp"]

    def shift_pitch(self, p: torch.Tensor, scale: float) -> torch.Tensor:
        """Multiply F0 by `scale`. p is normalized log-F0, so this adds log(scale) / std.

        The result is kept inside the speaker's range: pitch pushed below it makes the vocoder buzz.
        """
        return torch.clamp(p + math.log(scale) / self.log_f0_std, self.pitch_min, self.pitch_max)

    def forward(self, x, mask, durations=None, pitch=None, energy=None, pitch_scale: float = 1.0, rate: float = 1.0,
                mel=None, mel_mask=None, prior=None) -> VarianceOutput:
        log_duration = self.duration(x, mask)
        alignment = None
        if mel is not None:
            # Training: durations come from the hard alignment between phonemes and mel frames, and
            # the per-frame pitch and energy targets are averaged over each phoneme with them.
            logp, hard = self.aligner(x.detach(), mel, mask, mel_mask, prior)
            durations = hard.sum(1).long()
            alignment = (logp, hard)
            pitch = average_by_duration(pitch, durations)
            energy = average_by_duration(energy, durations)
        if durations is None:
            # rate > 1 speaks faster: every predicted duration is divided by it before rounding.
            durations = torch.clamp(torch.round((torch.exp(log_duration) - 1) / rate), min=1).long()
            durations = durations.masked_fill(mask, 0)
        # Pitch and energy are predicted per phoneme, before the length regulator, so every
        # frame of a phoneme gets the same pitch and energy embedding.
        pitch_pred = self.pitch(x, mask)
        energy_pred = self.energy(x, mask)
        p = pitch if pitch is not None else pitch_pred
        if pitch is None and pitch_scale != 1.0:
            p = self.shift_pitch(p, pitch_scale)
        e = energy if energy is not None else energy_pred
        x = x + self.pitch_embed(torch.bucketize(p, self.pitch_bins))
        x = x + self.energy_embed(torch.bucketize(e, self.energy_bins))
        x, mel_mask = self.regulate(x, durations)
        return VarianceOutput(x, mel_mask, log_duration, pitch_pred, energy_pred, durations, alignment)

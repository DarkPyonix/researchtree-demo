"""Variance adaptor: duration, pitch and energy prediction, and the length regulator."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from ..cwt import cwt_reconstruct


class VariancePredictor(nn.Module):
    """Two 1D convolutions (ReLU, layer norm, dropout), then a linear layer: one value per step."""

    def __init__(self, hidden: int, layers: int = 2, kernel: int = 3, dropout: float = 0.5, out_dim: int = 1):
        super().__init__()
        self.convs = nn.ModuleList(nn.Conv1d(hidden, hidden, kernel, padding=kernel // 2) for _ in range(layers))
        self.norms = nn.ModuleList(nn.LayerNorm(hidden) for _ in range(layers))
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(hidden, out_dim)
        self.out_dim = out_dim

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # x: (batch, time, hidden); mask: True at padded steps.
        for conv, norm in zip(self.convs, self.norms):
            x = conv(x.transpose(1, 2)).transpose(1, 2)
            x = self.dropout(norm(torch.relu(x)))
        y = self.out(x).masked_fill(mask.unsqueeze(-1), 0.0)
        return y.squeeze(-1) if self.out_dim == 1 else y


class LengthRegulator(nn.Module):
    """Repeat each phoneme's hidden vector as many times as its duration in frames."""

    def forward(self, x: torch.Tensor, durations: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        frames = [torch.repeat_interleave(xi, di, dim=0) for xi, di in zip(x, durations)]
        lengths = torch.tensor([f.shape[0] for f in frames], device=x.device)
        out = nn.utils.rnn.pad_sequence(frames, batch_first=True)
        mel_mask = torch.arange(out.shape[1], device=x.device)[None, :] >= lengths[:, None]
        return out, mel_mask


@dataclass
class VarianceOutput:
    hidden: torch.Tensor
    mel_mask: torch.Tensor
    log_duration: torch.Tensor
    pitch: torch.Tensor
    energy: torch.Tensor


class VarianceAdaptor(nn.Module):
    """Predict durations, expand to frames, then add quantized pitch and energy embeddings."""

    def __init__(self, cfg: dict, hidden: int):
        super().__init__()
        self.duration = VariancePredictor(hidden, **cfg["predictor"])
        # Pitch: 10 wavelet coefficients per frame, plus the utterance's pitch mean and spread
        # predicted from the average encoder state.
        self.pitch = VariancePredictor(hidden, out_dim=cfg["pitch"]["scales"], **cfg["predictor"])
        self.pitch_stats = nn.Linear(hidden, 2)
        self.energy = VariancePredictor(hidden, **cfg["predictor"])
        # Pitch and energy are normalized to zero mean and unit variance over the corpus.
        self.register_buffer("pitch_bins", torch.linspace(-3.0, 3.0, cfg["pitch"]["bins"] - 1))
        self.register_buffer("energy_bins", torch.linspace(-3.0, 3.0, cfg["energy"]["bins"] - 1))
        self.pitch_embed = nn.Embedding(cfg["pitch"]["bins"], hidden)
        self.energy_embed = nn.Embedding(cfg["energy"]["bins"], hidden)
        self.regulate = LengthRegulator()

    def forward(self, x, mask, durations=None, pitch=None, energy=None) -> VarianceOutput:
        log_duration = self.duration(x, mask)
        if durations is None:
            durations = torch.clamp(torch.round(torch.exp(log_duration) - 1), min=1).long()
            durations = durations.masked_fill(mask, 0)
        x, mel_mask = self.regulate(x, durations)

        # Pitch and energy are predicted per mel frame, after the length regulator.
        stats = self.pitch_stats(x.mean(1))  # (batch, 2): mean and spread of the contour
        pitch_pred = cwt_reconstruct(self.pitch(x, mel_mask)) * stats[:, 1:] + stats[:, :1]
        energy_pred = self.energy(x, mel_mask)
        p = pitch if pitch is not None else pitch_pred
        e = energy if energy is not None else energy_pred
        x = x + self.pitch_embed(torch.bucketize(p, self.pitch_bins))
        x = x + self.energy_embed(torch.bucketize(e, self.energy_bins))
        return VarianceOutput(x, mel_mask, log_duration, pitch_pred, energy_pred)

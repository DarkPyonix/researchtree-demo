"""Conformer block (Gulati et al., 2020): feed-forward, self-attention, convolution, feed-forward."""

from __future__ import annotations

import torch
from torch import nn

from .fastspeech2 import sinusoid_positions


class FeedForward(nn.Module):
    def __init__(self, hidden: int, mult: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(hidden), nn.Linear(hidden, hidden * mult), nn.SiLU(), nn.Dropout(dropout),
            nn.Linear(hidden * mult, hidden), nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ConvModule(nn.Module):
    """Pointwise convolution with GLU, depthwise convolution, batch norm, SiLU, pointwise convolution."""

    def __init__(self, hidden: int, kernel: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(hidden)
        self.pointwise1 = nn.Conv1d(hidden, hidden * 2, 1)
        self.depthwise = nn.Conv1d(hidden, hidden, kernel, padding=kernel // 2, groups=hidden)
        self.bn = nn.BatchNorm1d(hidden)
        self.pointwise2 = nn.Conv1d(hidden, hidden, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        y = self.norm(x).transpose(1, 2)
        y = nn.functional.glu(self.pointwise1(y), dim=1)
        y = self.pointwise2(torch.nn.functional.silu(self.bn(self.depthwise(y))))
        return self.dropout(y.transpose(1, 2))


class ConformerBlock(nn.Module):
    def __init__(self, hidden: int, heads: int, conv_kernel: int, ff_mult: int, dropout: float):
        super().__init__()
        self.ff1 = FeedForward(hidden, ff_mult, dropout)
        self.attn_norm = nn.LayerNorm(hidden)
        self.attn = nn.MultiheadAttention(hidden, heads, dropout=dropout, batch_first=True)
        self.conv = ConvModule(hidden, conv_kernel, dropout)
        self.ff2 = FeedForward(hidden, ff_mult, dropout)
        self.out_norm = nn.LayerNorm(hidden)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = x + 0.5 * self.ff1(x)
        y = self.attn_norm(x)
        x = x + self.attn(y, y, y, key_padding_mask=mask)[0]
        x = x + self.conv(x, mask)
        x = x + 0.5 * self.ff2(x)
        return self.out_norm(x).masked_fill(mask.unsqueeze(-1), 0.0)


class ConformerStack(nn.Module):
    """Same interface as FFTStack: sinusoidal positions plus a stack of blocks."""

    def __init__(self, layers: int, hidden: int, heads: int, conv_kernel: int, dropout: float, ff_mult: int = 4, **_):
        super().__init__()
        self.hidden = hidden
        self.blocks = nn.ModuleList(ConformerBlock(hidden, heads, conv_kernel, ff_mult, dropout) for _ in range(layers))

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = x + sinusoid_positions(x.shape[1], self.hidden, x.device)
        for block in self.blocks:
            x = block(x, mask)
        return x

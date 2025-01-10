"""HiFi-GAN vocoder (Kong et al., 2020): generator, multi-period and multi-scale discriminators."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils.parametrizations import weight_norm

LRELU_SLOPE = 0.1


class ResBlock(nn.Module):
    """Residual block with dilated 1D convolutions (HiFi-GAN "ResBlock1")."""

    def __init__(self, channels: int, kernel: int, dilations: list[int]):
        super().__init__()
        self.convs1 = nn.ModuleList(
            weight_norm(nn.Conv1d(channels, channels, kernel, dilation=d, padding=d * (kernel - 1) // 2)) for d in dilations
        )
        self.convs2 = nn.ModuleList(
            weight_norm(nn.Conv1d(channels, channels, kernel, padding=(kernel - 1) // 2)) for _ in dilations
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for c1, c2 in zip(self.convs1, self.convs2):
            y = c2(F.leaky_relu(c1(F.leaky_relu(x, LRELU_SLOPE)), LRELU_SLOPE))
            x = x + y
        return x


class Generator(nn.Module):
    """Mel spectrogram (batch, n_mels, frames) to waveform (batch, 1, frames * hop)."""

    def __init__(self, n_mels: int, upsample_rates, upsample_kernel_sizes, upsample_initial_channel,
                 resblock_kernel_sizes, resblock_dilation_sizes, **_):
        super().__init__()
        ch = upsample_initial_channel
        self.pre = weight_norm(nn.Conv1d(n_mels, ch, 7, padding=3))
        self.ups = nn.ModuleList()
        self.resblocks = nn.ModuleList()
        for i, (rate, kernel) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            out_ch = ch // (2 ** (i + 1))
            self.ups.append(weight_norm(nn.ConvTranspose1d(ch // (2**i), out_ch, kernel, rate, padding=(kernel - rate) // 2)))
            for k, d in zip(resblock_kernel_sizes, resblock_dilation_sizes):
                self.resblocks.append(ResBlock(out_ch, k, d))
        self.n_kernels = len(resblock_kernel_sizes)
        self.post = weight_norm(nn.Conv1d(out_ch, 1, 7, padding=3))

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        x = self.pre(mel)
        for i, up in enumerate(self.ups):
            x = up(F.leaky_relu(x, LRELU_SLOPE))
            blocks = self.resblocks[i * self.n_kernels : (i + 1) * self.n_kernels]
            x = sum(block(x) for block in blocks) / self.n_kernels
        return torch.tanh(self.post(F.leaky_relu(x)))


class PeriodDiscriminator(nn.Module):
    """Looks at the waveform folded into a 2D grid with a fixed period."""

    def __init__(self, period: int):
        super().__init__()
        self.period = period
        chans = [1, 32, 128, 512, 1024, 1024]
        self.convs = nn.ModuleList(
            weight_norm(nn.Conv2d(chans[i], chans[i + 1], (5, 1), (3 if i < 4 else 1, 1), padding=(2, 0)))
            for i in range(5)
        )
        self.out = weight_norm(nn.Conv2d(1024, 1, (3, 1), padding=(1, 0)))

    def forward(self, wav: torch.Tensor):
        b, c, t = wav.shape
        if t % self.period:
            wav = F.pad(wav, (0, self.period - t % self.period), "reflect")
        x = wav.view(b, c, -1, self.period)
        features = []
        for conv in self.convs:
            x = F.leaky_relu(conv(x), LRELU_SLOPE)
            features.append(x)
        x = self.out(x)
        features.append(x)
        return x.flatten(1), features


class ScaleDiscriminator(nn.Module):
    """1D convolutions over the raw (or average-pooled) waveform."""

    def __init__(self):
        super().__init__()
        self.convs = nn.ModuleList([
            weight_norm(nn.Conv1d(1, 128, 15, 1, padding=7)),
            weight_norm(nn.Conv1d(128, 256, 41, 4, groups=16, padding=20)),
            weight_norm(nn.Conv1d(256, 1024, 41, 4, groups=64, padding=20)),
            weight_norm(nn.Conv1d(1024, 1024, 5, 1, padding=2)),
        ])
        self.out = weight_norm(nn.Conv1d(1024, 1, 3, padding=1))

    def forward(self, wav: torch.Tensor):
        x, features = wav, []
        for conv in self.convs:
            x = F.leaky_relu(conv(x), LRELU_SLOPE)
            features.append(x)
        x = self.out(x)
        features.append(x)
        return x.flatten(1), features


def build_discriminators(names: list[str]) -> nn.ModuleList:
    out = nn.ModuleList()
    for name in names:
        if name == "mpd":
            out.extend(PeriodDiscriminator(p) for p in (2, 3, 5, 7, 11))
        elif name == "msd":
            out.extend(ScaleDiscriminator() for _ in range(3))
        else:
            raise ValueError(f"unknown discriminator: {name}")
    return out

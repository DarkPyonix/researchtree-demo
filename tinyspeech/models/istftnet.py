"""iSTFTNet generator (Kaneko et al., 2022): HiFi-GAN upsampling stages, then an inverse STFT.

Configuration C8C8I: two transposed-convolution stages upsample the mel frames by 8 and 8 (64 times).
The last layer then predicts, for every step, the magnitude and phase of a 16-point STFT with hop 4,
and torch.istft turns them into the waveform (4 more times, 256 in total). The inverse STFT replaces
HiFi-GAN's last two upsampling stages, which run at the highest time resolution and cost the most.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils.parametrizations import weight_norm

from .hifigan import LRELU_SLOPE, ResBlock


class ISTFTNetGenerator(nn.Module):
    """Mel spectrogram (batch, n_mels, frames) to waveform (batch, 1, frames * 256)."""

    def __init__(self, n_mels: int, upsample_rates, upsample_kernel_sizes, upsample_initial_channel,
                 resblock_kernel_sizes, resblock_dilation_sizes, istft_n_fft: int = 16, istft_hop: int = 4, **_):
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
        self.n_fft, self.hop = istft_n_fft, istft_hop
        # n_fft / 2 + 1 magnitudes and as many phases per step.
        self.post = weight_norm(nn.Conv1d(out_ch, istft_n_fft + 2, 7, padding=3))
        self.register_buffer("window", torch.hann_window(istft_n_fft))

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        x = self.pre(mel)
        for i, up in enumerate(self.ups):
            x = up(F.leaky_relu(x, LRELU_SLOPE))
            blocks = self.resblocks[i * self.n_kernels : (i + 1) * self.n_kernels]
            x = sum(block(x) for block in blocks) / self.n_kernels
        x = self.post(F.leaky_relu(x))
        bins = self.n_fft // 2 + 1
        magnitude = torch.exp(x[:, :bins])
        phase = torch.pi * torch.sin(x[:, bins:])
        wav = torch.istft(torch.polar(magnitude, phase), self.n_fft, self.hop, self.n_fft, self.window)
        return wav.unsqueeze(1)

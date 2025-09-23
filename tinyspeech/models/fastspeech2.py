"""FastSpeech2-style acoustic model: phoneme ids to an 80-bin log-mel spectrogram."""

from __future__ import annotations

import math

import torch
from torch import nn

from .context import PhraseContext
from .variance import VarianceAdaptor


class FFTBlock(nn.Module):
    """Feed-forward Transformer block: self-attention, then two 1D convolutions."""

    def __init__(self, hidden: int, heads: int, conv_kernel: int, dropout: float):
        super().__init__()
        self.attn = nn.MultiheadAttention(hidden, heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(hidden)
        self.conv = nn.Sequential(
            nn.Conv1d(hidden, hidden * 4, conv_kernel, padding=conv_kernel // 2),
            nn.ReLU(),
            nn.Conv1d(hidden * 4, hidden, 1),
        )
        self.norm2 = nn.LayerNorm(hidden)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # mask: True at padded positions, shape (batch, time).
        y, _ = self.attn(x, x, x, key_padding_mask=mask)
        x = self.norm1(x + self.dropout(y))
        y = self.conv(x.transpose(1, 2)).transpose(1, 2)
        x = self.norm2(x + self.dropout(y))
        return x.masked_fill(mask.unsqueeze(-1), 0.0)


def sinusoid_positions(length: int, hidden: int, device: torch.device) -> torch.Tensor:
    pos = torch.arange(length, device=device, dtype=torch.float32)[:, None]
    div = torch.exp(torch.arange(0, hidden, 2, device=device) * (-math.log(10000.0) / hidden))
    table = torch.zeros(length, hidden, device=device)
    table[:, 0::2] = torch.sin(pos * div)
    table[:, 1::2] = torch.cos(pos * div)
    return table


class FFTStack(nn.Module):
    """Sinusoidal positions plus a stack of FFT blocks."""

    def __init__(self, layers: int, hidden: int, heads: int, conv_kernel: int, dropout: float, **_):
        super().__init__()
        self.hidden = hidden
        self.blocks = nn.ModuleList(FFTBlock(hidden, heads, conv_kernel, dropout) for _ in range(layers))

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = x + sinusoid_positions(x.shape[1], self.hidden, x.device)
        for block in self.blocks:
            x = block(x, mask)
        return x


def build_decoder(cfg: dict) -> nn.Module:
    if cfg["type"] == "fft":
        return FFTStack(**cfg)
    if cfg["type"] == "conformer":
        from .conformer import ConformerStack  # conformer.py imports this module, so import it late

        return ConformerStack(**cfg)
    raise ValueError(f"unknown decoder type: {cfg['type']}")


class Postnet(nn.Module):
    """1D convolutions that predict a residual correction for the decoder's mel output."""

    def __init__(self, n_mels: int, layers: int, channels: int, kernel: int):
        super().__init__()
        dims = [n_mels] + [channels] * (layers - 1) + [n_mels]
        self.convs = nn.ModuleList(
            nn.Sequential(nn.Conv1d(dims[i], dims[i + 1], kernel, padding=kernel // 2), nn.BatchNorm1d(dims[i + 1]))
            for i in range(layers)
        )

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        x = mel.transpose(1, 2)
        for i, conv in enumerate(self.convs):
            x = conv(x)
            if i < len(self.convs) - 1:
                x = torch.tanh(x)
        return x.transpose(1, 2)


class FastSpeech2(nn.Module):
    def __init__(self, cfg: dict, n_symbols: int):
        super().__init__()
        hidden = cfg["encoder"]["hidden"]
        n_mels = cfg["audio"]["n_mels"]
        self.embed = nn.Embedding(n_symbols, hidden, padding_idx=0)
        self.encoder = FFTStack(**cfg["encoder"])
        self.context = PhraseContext(hidden, cfg["context"]["model"]) if cfg.get("context") else None
        self.variance = VarianceAdaptor(cfg["variance"], hidden)
        self.decoder = build_decoder(cfg["decoder"])
        self.to_mel = nn.Linear(cfg["decoder"]["hidden"], n_mels)
        self.postnet = Postnet(n_mels, **cfg["postnet"])

    def forward(self, ids, id_mask, durations=None, pitch=None, energy=None, pitch_scale: float = 1.0,
                mel=None, mel_mask=None, prior=None, words=None, word_ids=None):
        """With `mel` (training), durations come from the learned alignment; without it, from the predictor.

        `words` (a list of words per item) and `word_ids` (each phoneme's word index) feed the phrase context.
        """
        x = self.encoder(self.embed(ids), id_mask)
        if self.context is not None:
            x = x + self.context(words, word_ids).masked_fill(id_mask.unsqueeze(-1), 0.0)
        v = self.variance(x, id_mask, durations, pitch, energy, pitch_scale=pitch_scale,
                          mel=mel, mel_mask=mel_mask, prior=prior)
        mel = self.to_mel(self.decoder(v.hidden, v.mel_mask))
        mel_post = mel + self.postnet(mel)
        return mel, mel_post, v

    @torch.no_grad()
    def synthesize(self, ids: torch.Tensor, pitch_scale: float = 1.0, words=None, word_ids=None) -> torch.Tensor:
        """One sentence (batch of 1) to a mel spectrogram, using predicted durations, pitch and energy.

        pitch_scale multiplies the predicted F0: 1.2 is 20% higher, 0.8 is 20% lower.
        """
        mask = torch.zeros_like(ids, dtype=torch.bool)
        _, mel, _ = self(ids, mask, pitch_scale=pitch_scale, words=words, word_ids=word_ids)
        return mel

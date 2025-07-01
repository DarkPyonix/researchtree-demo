"""Alignment learned together with the model (Badlani et al., 2022, "One TTS Alignment To Rule Them All").

Soft alignment: a softmax over negative distances between convolution-encoded phonemes and mel frames,
optionally multiplied by a beta-binomial prior that favors the diagonal. Hard alignment: the most likely
monotonic path through the soft alignment (monotonic alignment search). The durations used as targets
for the duration predictor are the number of frames the hard alignment gives each phoneme.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import betabinom
from torch import nn


class Aligner(nn.Module):
    def __init__(self, n_mels: int, hidden: int, attn_dim: int = 80, temperature: float = 0.0005):
        super().__init__()
        self.key = nn.Sequential(
            nn.Conv1d(hidden, hidden * 2, 3, padding=1), nn.ReLU(), nn.Conv1d(hidden * 2, attn_dim, 1)
        )
        self.query = nn.Sequential(
            nn.Conv1d(n_mels, n_mels * 2, 3, padding=1), nn.ReLU(),
            nn.Conv1d(n_mels * 2, n_mels, 1), nn.ReLU(),
            nn.Conv1d(n_mels, attn_dim, 1),
        )
        self.temperature = temperature

    def forward(self, text, mel, id_mask, mel_mask, prior=None):
        """text: (batch, phonemes, hidden), mel: (batch, frames, n_mels).

        Returns the log soft alignment and the hard alignment, both (batch, frames, phonemes).
        """
        k = self.key(text.transpose(1, 2))   # (batch, attn_dim, phonemes)
        q = self.query(mel.transpose(1, 2))  # (batch, attn_dim, frames)
        dist = ((q[:, :, :, None] - k[:, :, None, :]) ** 2).sum(1)
        logp = F.log_softmax(-self.temperature * dist, dim=-1)
        if prior is not None:
            logp = logp + torch.log(prior + 1e-8)
        logp = F.log_softmax(logp.masked_fill(id_mask[:, None, :], -1e4), dim=-1)
        return logp, monotonic_alignment_search(logp, id_mask, mel_mask)


def beta_binomial_prior(n_phonemes: int, n_frames: int, scale: float = 1.0) -> torch.Tensor:
    """(frames, phonemes) prior that puts phoneme i near frame i * frames / phonemes."""
    k = np.arange(n_phonemes)
    rows = [betabinom(n_phonemes - 1, scale * t, scale * (n_frames + 1 - t)).pmf(k) for t in range(1, n_frames + 1)]
    return torch.tensor(np.array(rows), dtype=torch.float32)


@torch.no_grad()
def monotonic_alignment_search(logp, id_mask, mel_mask) -> torch.Tensor:
    """Most likely monotonic path through the soft alignment (Viterbi), as a 0/1 matrix per item."""
    out = torch.zeros_like(logp)
    for b in range(logp.shape[0]):
        frames, phonemes = int((~mel_mask[b]).sum()), int((~id_mask[b]).sum())
        lp = logp[b, :frames, :phonemes].cpu().numpy()
        score = np.full((frames, phonemes), -np.inf)
        score[0, 0] = lp[0, 0]
        for t in range(1, frames):
            advance = np.concatenate([[-np.inf], score[t - 1, :-1]])
            score[t] = np.maximum(score[t - 1], advance) + lp[t]
        n = phonemes - 1
        for t in range(frames - 1, -1, -1):
            out[b, t, n] = 1.0
            if t > 0 and n > 0 and score[t - 1, n - 1] >= score[t - 1, n]:
                n -= 1
    return out


def forward_sum_loss(logp, id_lens, mel_lens) -> torch.Tensor:
    """CTC loss over the soft alignment: every phoneme must be visited, in order."""
    logp = F.pad(logp, (1, 0), value=np.log(1e-4))  # column 0 is the CTC blank
    targets = torch.arange(1, logp.shape[-1], device=logp.device)[None].expand(logp.shape[0], -1)
    return F.ctc_loss(logp.log_softmax(-1).transpose(0, 1), targets, mel_lens, id_lens, zero_infinity=True)


def bin_loss(logp, hard) -> torch.Tensor:
    """Pulls the soft alignment toward the hard one (mean negative log-probability on the hard path)."""
    return -(logp * hard).sum() / hard.sum().clamp(min=1.0)

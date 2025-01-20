"""Continuous wavelet transform (CWT) of a log-F0 contour, as used for pitch in FastSpeech2.

The contour is split into 10 scales with a Mexican hat wavelet. Scale i covers about
2^(i+1) frames, so low scales hold fast pitch movements and high scales hold phrase-level ones.
"""

from __future__ import annotations

import numpy as np
import torch
from scipy import signal

SCALES = 10


def cwt_decompose(lf0: np.ndarray, scales: int = SCALES) -> np.ndarray:
    """(frames,) log-F0, normalized per utterance -> (frames, scales) wavelet coefficients."""
    widths = 2.0 ** (np.arange(scales) + 1)
    return signal.cwt(lf0, signal.ricker, widths).T.astype(np.float32)


def cwt_reconstruct(coefs: torch.Tensor) -> torch.Tensor:
    """Weighted sum over scales (Suni et al., 2013), the inverse used at inference."""
    scales = coefs.shape[-1]
    weights = (torch.arange(scales, device=coefs.device) + 2.5) ** -2.5
    return (coefs * weights).sum(-1)

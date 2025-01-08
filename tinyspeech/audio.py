"""Mel spectrogram and energy with the settings in configs/acoustic.yaml (audio section)."""

from __future__ import annotations

import functools

import librosa
import torch

SAMPLE_RATE = 22050
N_FFT = 1024
HOP = 256
WIN = 1024
N_MELS = 80
FMIN, FMAX = 0, 8000


@functools.cache
def _mel_basis(device: str) -> torch.Tensor:
    basis = librosa.filters.mel(sr=SAMPLE_RATE, n_fft=N_FFT, n_mels=N_MELS, fmin=FMIN, fmax=FMAX)
    return torch.from_numpy(basis).float().to(device)


def stft_magnitude(wav: torch.Tensor) -> torch.Tensor:
    window = torch.hann_window(WIN, device=wav.device)
    spec = torch.stft(wav, N_FFT, HOP, WIN, window, center=True, return_complex=True)
    return spec.abs()  # (batch, n_fft // 2 + 1, frames)


def mel_spectrogram(wav: torch.Tensor) -> torch.Tensor:
    """Log-mel spectrogram, shape (batch, n_mels, frames)."""
    mel = _mel_basis(str(wav.device)) @ stft_magnitude(wav)
    return torch.log(torch.clamp(mel, min=1e-5))


def energy(wav: torch.Tensor) -> torch.Tensor:
    """L2 norm of each STFT frame, shape (batch, frames)."""
    return torch.linalg.norm(stft_magnitude(wav), dim=1)

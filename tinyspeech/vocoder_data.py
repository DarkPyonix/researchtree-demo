"""Random waveform segments and their mels for vocoder training."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset

from .audio import HOP, mel_spectrogram


class VocoderSegments(Dataset):
    """Pairs of (mel, waveform) segments.

    With `predicted_mels`, the mel comes from the acoustic model run with ground-truth durations
    (saved as data/predicted_mels/<clip>.npy), so the vocoder learns to fix the model's errors.
    """

    def __init__(self, segment_size: int, predicted_mels: str | None = None, root: str = "data/LJSpeech-1.1"):
        self.wavs = sorted(Path(root, "wavs").glob("*.wav"))
        self.segment_size = segment_size
        self.predicted = Path("data/predicted_mels") if predicted_mels else None

    def __len__(self) -> int:
        return len(self.wavs)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        wav, _ = sf.read(self.wavs[index], dtype="float32")
        frames = self.segment_size // HOP
        if self.predicted is not None:
            mel = torch.from_numpy(np.load(self.predicted / f"{self.wavs[index].stem}.npy")).T
        else:
            mel = mel_spectrogram(torch.from_numpy(wav)[None])[0]
        start = random.randint(0, max(mel.shape[1] - frames, 0))
        mel = mel[:, start : start + frames]
        seg = torch.from_numpy(wav[start * HOP : start * HOP + self.segment_size])
        seg = torch.nn.functional.pad(seg, (0, self.segment_size - len(seg)))
        return mel, seg[None]

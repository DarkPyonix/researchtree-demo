"""Precompute mel, pitch and energy for every clip.

Durations are not extracted: the acoustic model learns them from its own alignment during training.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pyworld
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tinyspeech.audio import HOP, SAMPLE_RATE, energy, mel_spectrogram  # noqa: E402

OUT = Path("data/features")


def log_f0(wav: np.ndarray, n_frames: int) -> np.ndarray:
    """log-F0 per frame (WORLD DIO + StoneMask), linearly interpolated through unvoiced frames."""
    f0, t = pyworld.dio(wav.astype(np.float64), SAMPLE_RATE, frame_period=HOP / SAMPLE_RATE * 1000)
    f0 = pyworld.stonemask(wav.astype(np.float64), f0, t, SAMPLE_RATE)[:n_frames]
    voiced = f0 > 0
    lf0 = np.zeros_like(f0)
    lf0[voiced] = np.log(f0[voiced])
    lf0 = np.interp(np.arange(len(f0)), np.flatnonzero(voiced), lf0[voiced])
    return lf0.astype(np.float32)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    stats: dict[str, list[np.ndarray]] = {"pitch": [], "energy": []}
    items = {}
    for wav_path in sorted(Path("data/LJSpeech-1.1/wavs").glob("*.wav")):
        wav, _ = sf.read(wav_path, dtype="float32")
        mel = mel_spectrogram(torch.from_numpy(wav)[None])[0].T.numpy()
        n = mel.shape[0]
        item = {"mel": mel, "pitch": log_f0(wav, n), "energy": energy(torch.from_numpy(wav)[None])[0, :n].numpy()}
        stats["pitch"].append(item["pitch"])
        stats["energy"].append(item["energy"])
        items[wav_path.stem] = item

    # Normalize pitch and energy with the corpus mean and standard deviation. They stay per frame;
    # the model averages them per phoneme with its learned durations.
    norm = {k: (np.concatenate(v).mean(), np.concatenate(v).std()) for k, v in stats.items()}
    np.save(OUT / "stats.npy", norm)
    for clip, item in items.items():
        for k, (mean, std) in norm.items():
            item[k] = (item[k] - mean) / std
        np.savez(OUT / f"{clip}.npz", **item)


if __name__ == "__main__":
    main()

"""Precompute mel, duration, pitch and energy for every clip.

Durations come from Montreal Forced Aligner TextGrids (data/alignments/<clip>.TextGrid), made with:

    mfa align data/LJSpeech-1.1/wavs data/cmudict-0.7b english_us_arpa data/alignments

Clips that MFA could not align are skipped and left out of training.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pyworld
import soundfile as sf
import textgrid
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tinyspeech.audio import HOP, SAMPLE_RATE, energy, mel_spectrogram  # noqa: E402

OUT = Path("data/features")


def durations_from_textgrid(path: Path, n_frames: int) -> np.ndarray:
    """Frame count per phoneme (silences included) from the MFA "phones" tier."""
    tier = textgrid.TextGrid.fromFile(str(path)).getFirst("phones")
    ends = np.array([round(iv.maxTime * SAMPLE_RATE / HOP) for iv in tier])
    ends[-1] = n_frames
    return np.diff(np.concatenate([[0], ends])).astype(np.int64)


def log_f0(wav: np.ndarray, n_frames: int) -> np.ndarray:
    """log-F0 per frame (WORLD DIO + StoneMask), linearly interpolated through unvoiced frames."""
    f0, t = pyworld.dio(wav.astype(np.float64), SAMPLE_RATE, frame_period=HOP / SAMPLE_RATE * 1000)
    f0 = pyworld.stonemask(wav.astype(np.float64), f0, t, SAMPLE_RATE)[:n_frames]
    voiced = f0 > 0
    lf0 = np.zeros_like(f0)
    lf0[voiced] = np.log(f0[voiced])
    lf0 = np.interp(np.arange(len(f0)), np.flatnonzero(voiced), lf0[voiced])
    return lf0.astype(np.float32)


def phoneme_average(values: np.ndarray, durations: np.ndarray) -> np.ndarray:
    """Mean of a per-frame contour over each phoneme's frames (0 for zero-length phonemes)."""
    ends = np.cumsum(durations)
    starts = ends - durations
    return np.array([values[s:e].mean() if e > s else 0.0 for s, e in zip(starts, ends)], dtype=np.float32)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    stats: dict[str, list[np.ndarray]] = {"pitch": [], "energy": []}
    items = {}
    for wav_path in sorted(Path("data/LJSpeech-1.1/wavs").glob("*.wav")):
        grid = Path("data/alignments") / f"{wav_path.stem}.TextGrid"
        if not grid.exists():
            continue  # MFA failed on this clip
        wav, _ = sf.read(wav_path, dtype="float32")
        mel = mel_spectrogram(torch.from_numpy(wav)[None])[0].T.numpy()
        n = mel.shape[0]
        item = {
            "mel": mel,
            "duration": durations_from_textgrid(grid, n),
            "pitch": log_f0(wav, n),
            "energy": energy(torch.from_numpy(wav)[None])[0, :n].numpy(),
        }
        stats["pitch"].append(item["pitch"])
        stats["energy"].append(item["energy"])
        items[wav_path.stem] = item

    # Normalize pitch and energy with the corpus mean and standard deviation.
    norm = {k: (np.concatenate(v).mean(), np.concatenate(v).std()) for k, v in stats.items()}
    np.save(OUT / "stats.npy", norm)
    for clip, item in items.items():
        for k, (mean, std) in norm.items():
            item[k] = phoneme_average((item[k] - mean) / std, item["duration"])
        np.savez(OUT / f"{clip}.npz", **item)


if __name__ == "__main__":
    main()

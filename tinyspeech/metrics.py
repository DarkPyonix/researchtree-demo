"""Evaluation metrics on synthesized audio. Heavy models are loaded once and cached."""

from __future__ import annotations

import functools
import re

import librosa
import numpy as np
import pysptk
import torch
from fastdtw import fastdtw

SAMPLE_RATE = 22050


@functools.cache
def _utmos():
    return torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True)


@functools.cache
def _whisper():
    import whisper

    return whisper.load_model("large-v3")


def utmos_score(wav: np.ndarray) -> float:
    """UTMOS22 strong learner: predicted mean opinion score on a 1-5 scale."""
    return float(_utmos()(torch.from_numpy(wav)[None], SAMPLE_RATE).item())


def whisper_transcribe(wav: np.ndarray) -> str:
    audio = librosa.resample(wav, orig_sr=SAMPLE_RATE, target_sr=16000)
    return _whisper().transcribe(audio.astype(np.float32), language="en")["text"]


def _clean(text: str) -> str:
    return re.sub(r"[^a-z' ]", "", text.lower().replace("-", " ")).strip()


def character_error_rate(hypothesis: str, reference: str) -> float:
    """Levenshtein distance over characters divided by the reference length, after lowercasing."""
    hyp, ref = _clean(hypothesis), _clean(reference)
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        cur = [i]
        for j, h in enumerate(hyp, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (r != h)))
        prev = cur
    return prev[-1] / max(len(ref), 1)


def _mcep(wav: np.ndarray) -> np.ndarray:
    frames = librosa.util.frame(wav, frame_length=1024, hop_length=256).T * np.hanning(1024)
    return np.stack([pysptk.mcep(f, order=24, alpha=0.455, etype=1, eps=1e-8) for f in frames])[:, 1:14]


def mel_cepstral_distortion(wav: np.ndarray, reference_path: str) -> float:
    """MCD in dB over 13 mel-cepstral coefficients (c0 excluded), frames aligned with DTW."""
    ref, _ = librosa.load(reference_path, sr=SAMPLE_RATE)
    a, b = _mcep(wav.astype(np.float64)), _mcep(ref.astype(np.float64))
    _, path = fastdtw(a, b, dist=lambda x, y: np.sqrt(2 * np.sum((x - y) ** 2)))
    const = 10.0 / np.log(10.0)
    return float(np.mean([const * np.sqrt(2 * np.sum((a[i] - b[j]) ** 2)) for i, j in path]))

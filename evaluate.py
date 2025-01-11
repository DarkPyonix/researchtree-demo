"""The five metrics every experiment reports (see SPEC.md, Evaluation).

mel_l1  L1 distance to the recorded mel on the validation set, ground-truth durations (lower is better)
mcd     mel cepstral distortion to the recording in dB, DTW-aligned, test set (lower is better)
utmos   UTMOS22 predicted MOS on the test set, 1 to 5 (higher is better)
cer     character error rate of Whisper large-v3 on the synthesized test set, percent (lower is better)
rtf     synthesis time / audio duration, acoustic model + vocoder, 4 CPU threads (lower is better)
"""

from __future__ import annotations

import time

import numpy as np
import torch

from tinyspeech.data import LJSpeech, collate, read_metadata
from tinyspeech.metrics import character_error_rate, mel_cepstral_distortion, utmos_score, whisper_transcribe
from tinyspeech.synthesis import Synthesizer
from tinyspeech.text.cmudict import CMUDict


@torch.no_grad()
def validation_mel_l1(model, val_ids: list[str]) -> float:
    ds = LJSpeech("data/LJSpeech-1.1", "data/features", val_ids, CMUDict("data/cmudict-0.7b"))
    total, frames = 0.0, 0
    for item in ds:
        batch = {k: v.cuda() for k, v in collate([item]).items()}
        _, mel_post, _ = model(batch["ids"], batch["id_mask"], batch["duration"], batch["pitch"], batch["energy"])
        total += (mel_post - batch["mel"]).abs().sum().item() / batch["mel"].shape[-1]
        frames += batch["mel"].shape[1]
    return total / frames


def evaluate(model, val_ids: list[str], test_ids: list[str] | None) -> dict[str, float]:
    model.eval()
    out = {"mel_l1": round(validation_mel_l1(model, val_ids), 3)}
    if test_ids:
        rows = {r[0]: r for r in read_metadata("data/LJSpeech-1.1")}
        synth = Synthesizer(model.cpu(), vocoder="checkpoints/vocoder.pt", threads=4)
        mcd, mos, cer, rtf = [], [], [], []
        for clip in test_ids:
            _, raw_text, normalized = rows[clip]
            start = time.perf_counter()
            wav = synth(raw_text)  # raw text, as a user would type it
            rtf.append((time.perf_counter() - start) / (len(wav) / synth.sample_rate))
            mcd.append(mel_cepstral_distortion(wav, f"data/LJSpeech-1.1/wavs/{clip}.wav"))
            mos.append(utmos_score(wav))
            cer.append(character_error_rate(whisper_transcribe(wav), normalized))
        out.update(mcd=round(float(np.mean(mcd)), 2), utmos=round(float(np.mean(mos)), 2),
                   cer=round(100 * float(np.mean(cer)), 1), rtf=round(float(np.median(rtf)), 2))
        model.cuda()
    model.train()
    return out

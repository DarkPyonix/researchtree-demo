"""LJSpeech dataset with precomputed features (see scripts/extract_features.py)."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .models.aligner import beta_binomial_prior
from .text import phonemize
from .text.cmudict import CMUDict


def read_metadata(root: str | Path) -> list[tuple[str, str, str]]:
    """(clip id, raw text, normalized text) for every row of metadata.csv."""
    with open(Path(root) / "metadata.csv", encoding="utf-8") as f:
        return [(row[0], row[1], row[2]) for row in csv.reader(f, delimiter="|", quoting=csv.QUOTE_NONE)]


class LJSpeech(Dataset):
    """One item per clip: phoneme ids, mel, per-frame pitch and energy, and the alignment prior.

    There are no duration targets on disk: the model learns durations from its own alignment.
    """

    def __init__(self, root: str, features: str, ids: list[str], lexicon: CMUDict):
        self.features = Path(features)
        rows = {r[0]: r for r in read_metadata(root)}
        self.items = [rows[i] for i in ids]
        self.lexicon = lexicon

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict:
        clip, _, text = self.items[index]
        f = np.load(self.features / f"{clip}.npz")
        ids = phonemize(text, self.lexicon).ids
        mel = torch.from_numpy(f["mel"])  # (frames, n_mels)
        return {
            "ids": torch.tensor(ids),
            "mel": mel,
            "pitch": torch.from_numpy(f["pitch"]),    # normalized log-F0 per frame
            "energy": torch.from_numpy(f["energy"]),  # normalized energy per frame
            "prior": beta_binomial_prior(len(ids), len(mel)),
        }


def pad_2d(xs: list[torch.Tensor]) -> torch.Tensor:
    out = torch.zeros(len(xs), max(x.shape[0] for x in xs), max(x.shape[1] for x in xs))
    for i, x in enumerate(xs):
        out[i, : x.shape[0], : x.shape[1]] = x
    return out


def collate(batch: list[dict]) -> dict:
    pad = torch.nn.utils.rnn.pad_sequence
    out = {k: pad([b[k] for b in batch], batch_first=True) for k in batch[0] if k != "prior"}
    out["prior"] = pad_2d([b["prior"] for b in batch])
    id_len = torch.tensor([len(b["ids"]) for b in batch])
    mel_len = torch.tensor([len(b["mel"]) for b in batch])
    out["id_mask"] = torch.arange(out["ids"].shape[1])[None] >= id_len[:, None]
    out["mel_mask"] = torch.arange(out["mel"].shape[1])[None] >= mel_len[:, None]
    return out

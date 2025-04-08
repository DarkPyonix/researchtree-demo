"""LJSpeech dataset with precomputed features (see scripts/extract_features.py)."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .text import phonemize
from .text.cmudict import CMUDict


def read_metadata(root: str | Path) -> list[tuple[str, str, str]]:
    """(clip id, raw text, normalized text) for every row of metadata.csv."""
    with open(Path(root) / "metadata.csv", encoding="utf-8") as f:
        return [(row[0], row[1], row[2]) for row in csv.reader(f, delimiter="|", quoting=csv.QUOTE_NONE)]


class LJSpeech(Dataset):
    """One item per clip: phoneme ids, mel, and the duration, pitch and energy targets.

    Durations come from Montreal Forced Aligner TextGrids. Clips that MFA failed to align have no
    feature file and are skipped.
    """

    def __init__(self, root: str, features: str, ids: list[str], lexicon: CMUDict):
        self.features = Path(features)
        rows = {r[0]: r for r in read_metadata(root)}
        self.items = [rows[i] for i in ids if (self.features / f"{i}.npz").exists()]
        self.lexicon = lexicon

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict:
        clip, _, text = self.items[index]
        f = np.load(self.features / f"{clip}.npz")
        return {
            "ids": torch.tensor(phonemize(text, self.lexicon).ids),
            "mel": torch.from_numpy(f["mel"]),            # (frames, n_mels)
            "duration": torch.from_numpy(f["duration"]),  # frames per phoneme
            "pitch": torch.from_numpy(f["pitch"]),        # normalized log-F0 per phoneme
            "energy": torch.from_numpy(f["energy"]),      # normalized energy per phoneme
        }


def collate(batch: list[dict]) -> dict:
    pad = torch.nn.utils.rnn.pad_sequence
    out = {k: pad([b[k] for b in batch], batch_first=True) for k in batch[0]}
    id_len = torch.tensor([len(b["ids"]) for b in batch])
    mel_len = torch.tensor([len(b["mel"]) for b in batch])
    out["id_mask"] = torch.arange(out["ids"].shape[1])[None] >= id_len[:, None]
    out["mel_mask"] = torch.arange(out["mel"].shape[1])[None] >= mel_len[:, None]
    return out

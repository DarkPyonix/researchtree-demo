"""Text to waveform with a trained acoustic model and vocoder."""

from __future__ import annotations

import numpy as np
import torch
import yaml

from .models import build_generator
from .text import phonemize
from .text.cmudict import CMUDict


class Synthesizer:
    def __init__(self, model, vocoder: str, threads: int = 4, vocoder_config: str = "configs/vocoder.yaml"):
        torch.set_num_threads(threads)
        self.model = model.eval()
        cfg = yaml.safe_load(open(vocoder_config))
        self.vocoder = build_generator(cfg["generator"])
        self.vocoder.load_state_dict(torch.load(vocoder, map_location="cpu"))
        self.vocoder.eval()
        self.lexicon = CMUDict("data/cmudict-0.7b")
        self.sample_rate = 22050

    @torch.no_grad()
    def __call__(self, text: str, pitch_scale: float = 1.0) -> np.ndarray:
        ids = torch.tensor([phonemize(text, self.lexicon).ids])
        mel = self.model.synthesize(ids, pitch_scale=pitch_scale)
        wav = self.vocoder(mel.transpose(1, 2))
        return wav.squeeze().numpy()

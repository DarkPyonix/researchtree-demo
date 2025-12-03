"""Text to waveform with a trained acoustic model and vocoder."""

from __future__ import annotations

import numpy as np
import torch
import yaml

from .models import build_generator
from .text import phonemize
from .text.cmudict import CMUDict


class Synthesizer:
    def __init__(self, model, vocoder: str, threads: int = 4, vocoder_config: str = "configs/vocoder.yaml", int8: bool = False):
        torch.set_num_threads(threads)
        self.model = model.eval()
        if int8:
            # Linear layers get int8 weights; activations are quantized on the fly at each call.
            self.model = torch.ao.quantization.quantize_dynamic(self.model, {torch.nn.Linear}, dtype=torch.qint8)
        cfg = yaml.safe_load(open(vocoder_config))
        self.vocoder = build_generator(cfg["generator"])
        self.vocoder.load_state_dict(torch.load(vocoder, map_location="cpu"))
        self.vocoder.eval()
        self.lexicon = CMUDict("data/cmudict-0.7b")
        self.sample_rate = 22050

    @torch.no_grad()
    def __call__(self, text: str, pitch_scale: float = 1.0) -> np.ndarray:
        ph = phonemize(text, self.lexicon)
        ids = torch.tensor([ph.ids])
        mel = self.model.synthesize(ids, pitch_scale=pitch_scale, words=[ph.words], word_ids=torch.tensor([ph.word_ids]))
        wav = self.vocoder(mel.transpose(1, 2))
        return wav.squeeze().numpy()

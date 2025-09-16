from .fastspeech2 import FastSpeech2
from .hifigan import Generator, build_discriminators
from .istftnet import ISTFTNetGenerator


def build_generator(cfg: dict):
    """The vocoder generator named by `type` in the generator section of configs/vocoder.yaml."""
    if cfg["type"] == "hifigan":
        return Generator(n_mels=80, **cfg)
    if cfg["type"] == "istftnet":
        return ISTFTNetGenerator(n_mels=80, **cfg)
    raise ValueError(f"unknown generator type: {cfg['type']}")


__all__ = ["FastSpeech2", "Generator", "ISTFTNetGenerator", "build_discriminators", "build_generator"]

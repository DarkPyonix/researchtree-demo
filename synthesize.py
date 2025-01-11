"""Command-line synthesis.

    python synthesize.py "The quick brown fox." -o fox.wav
"""

from __future__ import annotations

import argparse

import soundfile as sf
import torch
import yaml

from tinyspeech.models import FastSpeech2
from tinyspeech.synthesis import Synthesizer
from tinyspeech.text.symbols import SYMBOLS


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("text")
    ap.add_argument("-o", "--output", default="out.wav")
    ap.add_argument("--acoustic", default="checkpoints/acoustic.pt")
    ap.add_argument("--vocoder", default="checkpoints/vocoder.pt")
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()

    cfg = yaml.safe_load(open("configs/acoustic.yaml"))
    model = FastSpeech2(cfg, len(SYMBOLS))
    model.load_state_dict(torch.load(args.acoustic, map_location="cpu"))
    synth = Synthesizer(model, args.vocoder, threads=args.threads)
    sf.write(args.output, synth(args.text), synth.sample_rate)


if __name__ == "__main__":
    main()

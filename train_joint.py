"""Fine-tune the acoustic model and the vocoder together on waveform losses.

Starts from trained checkpoints of both. The acoustic model runs with its learned durations, the vocoder
turns the predicted mel into audio, and the vocoder's GAN, feature matching and mel losses flow back into
the acoustic model as well.

    python train_joint.py --acoustic checkpoints/acoustic.pt --vocoder checkpoints/vocoder.pt
"""

from __future__ import annotations

import argparse

import soundfile as sf
import torch
import torch.nn.functional as F
import wandb
import yaml

import researchtree as rt
from evaluate import evaluate
from tinyspeech.audio import HOP, mel_spectrogram
from tinyspeech.data import LJSpeech, collate
from tinyspeech.models import FastSpeech2, Generator, build_discriminators
from tinyspeech.text.cmudict import CMUDict
from tinyspeech.text.symbols import SYMBOLS
from train import split
from train_vocoder import discriminator_loss, generator_losses

SEGMENT_FRAMES = 32  # frames of predicted mel per training segment (8,192 samples)


class WithAudio(LJSpeech):
    """LJSpeech items plus the recorded waveform, which the vocoder losses compare against."""

    def __getitem__(self, index: int) -> dict:
        item = super().__getitem__(index)
        wav, _ = sf.read(f"data/LJSpeech-1.1/wavs/{self.items[index][0]}.wav", dtype="float32")
        item["wav"] = torch.from_numpy(wav)
        return item


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--acoustic", required=True)
    ap.add_argument("--vocoder", required=True)
    ap.add_argument("--steps", type=int, default=30000)
    args = ap.parse_args()
    cfg = yaml.safe_load(open("configs/acoustic.yaml"))
    vcfg = yaml.safe_load(open("configs/vocoder.yaml"))
    tcfg = yaml.safe_load(open("configs/train.yaml"))
    run = wandb.init(project="tts", name="joint-finetune")
    rt.set(wandb=run.url)

    model = FastSpeech2(cfg, len(SYMBOLS)).cuda()
    model.load_state_dict(torch.load(args.acoustic))
    gen = Generator(n_mels=80, **vcfg["generator"]).cuda()
    gen.load_state_dict(torch.load(args.vocoder))
    discs = build_discriminators(vcfg["discriminators"]).cuda()
    opt_g = torch.optim.AdamW([*model.parameters(), *gen.parameters()], 5e-5, betas=(0.8, 0.99))
    opt_d = torch.optim.AdamW(discs.parameters(), 5e-5, betas=(0.8, 0.99))

    train_ids, val_ids, test_ids = split(tcfg["data"]["root"], tcfg["data"]["val_size"], tcfg["data"]["test_size"], tcfg["seed"])
    data = WithAudio(tcfg["data"]["root"], tcfg["data"]["features"], train_ids, CMUDict("data/cmudict-0.7b"))
    loader = torch.utils.data.DataLoader(data, batch_size=8, shuffle=True, collate_fn=collate, drop_last=True)
    step = 0
    while step < args.steps:
        for batch in loader:
            batch = {k: v.cuda() for k, v in batch.items()}
            _, mel, _ = model(batch["ids"], batch["id_mask"], pitch=batch["pitch"], energy=batch["energy"],
                              mel=batch["mel"], mel_mask=batch["mel_mask"])
            start = torch.randint(0, max(mel.shape[1] - SEGMENT_FRAMES, 1), (1,)).item()
            seg = mel[:, start : start + SEGMENT_FRAMES].transpose(1, 2)
            wav = batch["wav"][:, None, start * HOP : (start + SEGMENT_FRAMES) * HOP]
            fake = gen(seg)
            loss_d = discriminator_loss([d(wav)[0] for d in discs], [d(fake.detach())[0] for d in discs])
            opt_d.zero_grad()
            loss_d.backward()
            opt_d.step()
            real_out, fake_out = [d(wav) for d in discs], [d(fake) for d in discs]
            adv, fm = generator_losses([s for s, _ in fake_out], [f for _, f in real_out], [f for _, f in fake_out])
            loss_g = adv + 2 * fm + 45 * F.l1_loss(mel_spectrogram(fake.squeeze(1)), seg)
            opt_g.zero_grad()
            loss_g.backward()
            opt_g.step()
            step += 1
            if step >= args.steps:
                break
    torch.save(model.state_dict(), "checkpoints/acoustic-joint.pt")
    torch.save(gen.state_dict(), "checkpoints/vocoder-joint.pt")
    rt.log(**evaluate(model, val_ids, test_ids=test_ids))


if __name__ == "__main__":
    main()

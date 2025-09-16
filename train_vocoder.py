"""Train the vocoder on ground-truth mels, then fine-tune it on mels predicted by the acoustic model.

    python train_vocoder.py --stage gt
    python train_vocoder.py --stage finetune --acoustic checkpoints/acoustic.pt
"""

from __future__ import annotations

import argparse
import itertools

import torch
import torch.nn.functional as F
import wandb
import yaml

import researchtree as rt
from tinyspeech.audio import mel_spectrogram
from tinyspeech.models import build_discriminators, build_generator
from tinyspeech.vocoder_data import VocoderSegments


def discriminator_loss(real_scores, fake_scores) -> torch.Tensor:
    return sum(torch.mean((1 - r) ** 2) + torch.mean(f**2) for r, f in zip(real_scores, fake_scores))


def generator_losses(fake_scores, real_feats, fake_feats) -> tuple[torch.Tensor, torch.Tensor]:
    adv = sum(torch.mean((1 - f) ** 2) for f in fake_scores)
    fm = sum(F.l1_loss(f, r.detach()) for rs, fs in zip(real_feats, fake_feats) for r, f in zip(rs, fs))
    return adv, fm


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/vocoder.yaml")
    ap.add_argument("--stage", choices=["gt", "finetune"], required=True)
    ap.add_argument("--acoustic", help="acoustic model checkpoint that predicts the mels for fine-tuning")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    tcfg, lcfg = cfg["training"], cfg["loss"]
    run = wandb.init(project="tts", name=f"vocoder-{args.stage}", config=cfg)
    rt.set(wandb=run.url)

    gen = build_generator(cfg["generator"]).cuda()
    discs = build_discriminators(cfg["discriminators"]).cuda()
    opt_g = torch.optim.AdamW(gen.parameters(), tcfg["lr"], betas=tuple(tcfg["betas"]))
    opt_d = torch.optim.AdamW(discs.parameters(), tcfg["lr"], betas=tuple(tcfg["betas"]))
    data = VocoderSegments(tcfg["segment_size"], predicted_mels=args.acoustic if args.stage == "finetune" else None)
    loader = torch.utils.data.DataLoader(data, batch_size=tcfg["batch_size"], shuffle=True, num_workers=4, drop_last=True)
    steps = tcfg["max_steps"] if args.stage == "gt" else tcfg["finetune_on_predicted_mels"]

    for step, (mel, wav) in zip(range(steps), itertools.cycle(loader)):
        mel, wav = mel.cuda(), wav.cuda()
        fake = gen(mel)
        real_out = [d(wav) for d in discs]
        fake_out = [d(fake.detach()) for d in discs]
        loss_d = discriminator_loss([s for s, _ in real_out], [s for s, _ in fake_out])
        opt_d.zero_grad()
        loss_d.backward()
        opt_d.step()

        fake_out = [d(fake) for d in discs]
        adv, fm = generator_losses([s for s, _ in fake_out], [f for _, f in real_out], [f for _, f in fake_out])
        mel_l1 = F.l1_loss(mel_spectrogram(fake.squeeze(1)), mel)
        loss_g = adv + lcfg["feature_matching_weight"] * fm + lcfg["mel_weight"] * mel_l1
        opt_g.zero_grad()
        loss_g.backward()
        opt_g.step()
        if step % 100 == 0:
            wandb.log({"loss_d": loss_d.item(), "loss_g": loss_g.item(), "mel_l1": mel_l1.item()}, step=step)
    torch.save(gen.state_dict(), "checkpoints/vocoder.pt")


if __name__ == "__main__":
    main()

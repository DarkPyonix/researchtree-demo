"""Train the acoustic model, then evaluate and write the final metrics into the experiment's PR.

    python train.py --run-name pitch-phoneme-level
"""

from __future__ import annotations

import argparse
import random

import numpy as np
import torch
import wandb
import yaml
from torch.utils.data import DataLoader

import researchtree as rt
from evaluate import evaluate
from tinyspeech.data import LJSpeech, collate, read_metadata
from tinyspeech.losses import acoustic_loss
from tinyspeech.models import FastSpeech2
from tinyspeech.text.cmudict import CMUDict
from tinyspeech.text.symbols import SYMBOLS


def noam(step: int, warmup: int, hidden: int) -> float:
    """Noam learning-rate factor: linear warmup, then decay with 1/sqrt(step)."""
    step = max(step, 1)
    return hidden**-0.5 * min(step**-0.5, step * warmup**-1.5)


def split(root: str, val_size: int, test_size: int, seed: int) -> tuple[list[str], list[str], list[str]]:
    ids = [r[0] for r in read_metadata(root)]
    random.Random(seed).shuffle(ids)
    return ids[val_size + test_size :], ids[:val_size], ids[val_size : val_size + test_size]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--acoustic", default="configs/acoustic.yaml")
    ap.add_argument("--train", default="configs/train.yaml")
    ap.add_argument("--run-name", required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.acoustic))
    tcfg = yaml.safe_load(open(args.train))

    torch.manual_seed(tcfg["seed"])
    np.random.seed(tcfg["seed"])
    run = wandb.init(project="tts", name=args.run_name, config={"acoustic": cfg, "train": tcfg})
    rt.set(wandb=run.url)

    lexicon = CMUDict("data/cmudict-0.7b")
    train_ids, val_ids, test_ids = split(tcfg["data"]["root"], tcfg["data"]["val_size"], tcfg["data"]["test_size"], tcfg["seed"])
    train_set = LJSpeech(tcfg["data"]["root"], tcfg["data"]["features"], train_ids, lexicon)
    loader = DataLoader(train_set, batch_size=tcfg["batch_size"], shuffle=True, collate_fn=collate, num_workers=4, drop_last=True)

    model = FastSpeech2(cfg, len(SYMBOLS)).cuda()
    opt = torch.optim.Adam(model.parameters(), lr=tcfg["optimizer"]["lr"], betas=tuple(tcfg["optimizer"]["betas"]))
    warmup, hidden = tcfg["scheduler"]["warmup_steps"], cfg["encoder"]["hidden"]
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: noam(s, warmup, hidden) / noam(warmup, warmup, hidden))

    step = 0
    while step < tcfg["max_steps"]:
        for batch in loader:
            batch = {k: v.cuda() for k, v in batch.items()}
            mel, mel_post, v = model(batch["ids"], batch["id_mask"], batch["duration"], batch["pitch"], batch["energy"])
            losses = acoustic_loss(batch, mel, mel_post, v)
            opt.zero_grad()
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg["grad_clip"])
            opt.step()
            sched.step()
            step += 1
            if step % 100 == 0:
                wandb.log({**{f"train/{k}": x.item() for k, x in losses.items()}, "train/lr": sched.get_last_lr()[0]}, step=step)
            if step % tcfg["eval_every"] == 0:
                wandb.log({f"val/{k}": x for k, x in evaluate(model, val_ids, test_ids=None).items()}, step=step)
                torch.save(model.state_dict(), f"checkpoints/{args.run_name}.pt")
            if step >= tcfg["max_steps"]:
                break

    final = evaluate(model, val_ids, test_ids=test_ids)
    wandb.log({f"test/{k}": x for k, x in final.items()}, step=step)
    rt.log(**final)  # final metrics into the PR's YAML block


if __name__ == "__main__":
    main()

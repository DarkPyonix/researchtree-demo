"""Train the student text encoder to reproduce BERT-base word vectors.

Text: the LJSpeech transcripts plus the LibriTTS train-clean-360 transcripts (about 116k sentences), so the
student sees more sentence structures than one book-reading corpus has.

    python distill_context.py --steps 150000
"""

from __future__ import annotations

import argparse
import random

import torch
import torch.nn.functional as F
import wandb
from transformers import AutoModel, AutoTokenizer

import researchtree as rt
from tinyspeech.models.context_student import StudentTextEncoder


def sentences() -> list[str]:
    lines = open("data/distill_text.txt", encoding="utf-8").read().splitlines()
    return [line for line in lines if 3 <= len(line.split()) <= 60]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=150000)
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args()
    run = wandb.init(project="tts", name="distill-context")
    rt.set(wandb=run.url)

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    teacher = AutoModel.from_pretrained("bert-base-uncased").eval().requires_grad_(False).cuda()
    student = StudentTextEncoder(teacher.embeddings.word_embeddings).cuda()
    opt = torch.optim.AdamW([p for p in student.parameters() if p.requires_grad], 3e-4, weight_decay=0.01)
    texts = sentences()

    for step in range(args.steps):
        batch = tokenizer(random.sample(texts, args.batch_size), padding=True, return_tensors="pt").to("cuda")
        with torch.no_grad():
            target = teacher(**batch).last_hidden_state
        pred = student(batch["input_ids"], batch["attention_mask"])
        keep = batch["attention_mask"].bool()
        loss = F.mse_loss(pred[keep], target[keep]) + (1 - F.cosine_similarity(pred[keep], target[keep])).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 100 == 0:
            wandb.log({"distill/loss": loss.item()}, step=step)
        if step % 10000 == 0:
            torch.save(student.state_dict(), "checkpoints/context-student.pt")


if __name__ == "__main__":
    main()

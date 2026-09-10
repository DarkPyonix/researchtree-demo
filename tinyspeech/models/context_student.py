"""A small text encoder trained to reproduce BERT-base word vectors (open decision D4).

It uses BERT's own sub-word vocabulary and embeddings table (frozen), followed by 4 Transformer layers of
width 384 that are trained to map them to the word vectors BERT-base produces for the same sentence.
"""

from __future__ import annotations

import torch
from torch import nn


class StudentTextEncoder(nn.Module):
    def __init__(self, teacher_embeddings: nn.Embedding, layers: int = 4, width: int = 384, out_dim: int = 768):
        super().__init__()
        self.embed = teacher_embeddings.requires_grad_(False)
        self.inp = nn.Linear(teacher_embeddings.embedding_dim, width)
        layer = nn.TransformerEncoderLayer(width, nhead=6, dim_feedforward=width * 4, batch_first=True)
        self.layers = nn.TransformerEncoder(layer, layers)
        self.out = nn.Linear(width, out_dim)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """(batch, tokens) sub-word ids -> (batch, tokens, 768), the same shape as BERT-base's last layer."""
        x = self.inp(self.embed(input_ids))
        x = self.layers(x, src_key_padding_mask=attention_mask == 0)
        return self.out(x)

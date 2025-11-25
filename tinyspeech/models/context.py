"""Phrase context: one vector per word from a frozen pretrained text model, added to the word's phonemes.

The text model reads the whole sentence, so a word's vector reflects its role in the phrase (for example,
the new information that usually carries the pitch accent), which the phoneme encoder cannot see.
"""

from __future__ import annotations

import torch
from torch import nn


class PhraseContext(nn.Module):
    def __init__(self, hidden: int, model_name: str = "bert-base-uncased"):
        super().__init__()
        from transformers import AutoModel, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.text_model = AutoModel.from_pretrained(model_name).eval().requires_grad_(False)
        self.proj = nn.Linear(self.text_model.config.hidden_size, hidden)

    @torch.no_grad()
    def word_vectors(self, words: list[str]) -> torch.Tensor:
        """(words, text model size): the mean of each word's sub-word vectors from the last layer."""
        enc = self.tokenizer(words, is_split_into_words=True, return_tensors="pt").to(self.proj.weight.device)
        states = self.text_model(**enc).last_hidden_state[0]
        sums = torch.zeros(len(words), states.shape[-1], device=states.device)
        counts = torch.zeros(len(words), 1, device=states.device)
        for i, w in enumerate(enc.word_ids()):
            if w is not None:
                sums[w] += states[i]
                counts[w] += 1
        return sums / counts.clamp(min=1)

    def forward(self, words: list[list[str]], word_ids: torch.Tensor) -> torch.Tensor:
        """(batch, phonemes, hidden). Word gaps and punctuation (word id -1) get zeros."""
        out = []
        for item_words, ids in zip(words, word_ids):
            vecs = self.proj(self.word_vectors(item_words))
            vecs = torch.cat([vecs, torch.zeros_like(vecs[:1])])  # extra last row, picked by word id -1
            out.append(vecs[ids])
        return torch.stack(out)

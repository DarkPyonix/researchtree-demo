"""Text frontend: raw English text to phoneme ids, with the word each phoneme came from."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .cmudict import CMUDict
from .symbols import PUNCTUATION, SPACE, SYMBOL_TO_ID

_WHITESPACE = re.compile(r"\s+")
_TOKEN = re.compile(r"[a-z']+|\d|[,.?!;:]|\*")
_DIGITS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]


def normalize(text: str) -> str:
    """Lowercase, straighten quotes and collapse whitespace.

    Numbers and abbreviations are not expanded here: each digit is later read on its own.
    """
    text = text.lower().replace("’", "'").replace("“", '"').replace("”", '"')
    return _WHITESPACE.sub(" ", text).strip()


@dataclass
class Phonemized:
    ids: list[int]
    # Index of the source word for each id; -1 for word gaps and punctuation.
    word_ids: list[int]
    words: list[str]
    # 1 for phonemes of words written between asterisks, else 0.
    emphasis: list[int] = field(default_factory=list)


def phonemize(text: str, lexicon: CMUDict) -> Phonemized:
    """Normalize, split into words and punctuation, and look every word up in CMUdict.

    Words between asterisks (*like this*) are marked as emphasized.
    """
    ids: list[int] = []
    word_ids: list[int] = []
    words: list[str] = []
    emphasis: list[int] = []
    emphasized = False
    for token in _TOKEN.findall(normalize(text)):
        if token == "*":
            emphasized = not emphasized
            continue
        if token in PUNCTUATION:
            ids.append(SYMBOL_TO_ID[token])
            word_ids.append(-1)
            emphasis.append(0)
            continue
        if token.isdigit():
            token = _DIGITS[int(token)]
        if words:
            ids.append(SYMBOL_TO_ID[SPACE])
            word_ids.append(-1)
            emphasis.append(0)
        phones = lexicon.lookup(token) or spell_out(token, lexicon)
        ids.extend(SYMBOL_TO_ID[p] for p in phones)
        word_ids.extend([len(words)] * len(phones))
        emphasis.extend([int(emphasized)] * len(phones))
        words.append(token)
    return Phonemized(ids, word_ids, words, emphasis)


def spell_out(word: str, lexicon: CMUDict) -> list[str]:
    """Pronounce a word that is missing from CMUdict letter by letter ("ljs" -> "el jay es")."""
    phones: list[str] = []
    for letter in word.replace("'", ""):
        phones.extend(lexicon.lookup(letter) or [])
    return phones

"""CMU Pronouncing Dictionary lookup."""

from __future__ import annotations

from pathlib import Path


class CMUDict:
    """Word -> ARPAbet phonemes, using the first pronunciation listed for each word."""

    def __init__(self, path: str | Path):
        self.entries: dict[str, list[str]] = {}
        for line in Path(path).read_text(encoding="latin-1").splitlines():
            if not line or line.startswith(";;;"):
                continue
            word, phones = line.split("  ", 1)
            word = word.lower()
            if word.endswith(")"):  # alternative pronunciation, e.g. "read(1)"
                continue
            self.entries[word] = phones.split()

    def lookup(self, word: str) -> list[str] | None:
        return self.entries.get(word.lower())

    def __contains__(self, word: str) -> bool:
        return word.lower() in self.entries

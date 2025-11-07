"""Expand common English abbreviations before tokenizing.

The list is the one from Tacotron's English text cleaners. "St." is always read "saint", so "Elm St."
comes out wrong; the list has no context to tell the two apart.
"""

from __future__ import annotations

import re

_ABBREVIATIONS = [
    (re.compile(rf"\b{short}\.", re.IGNORECASE), full)
    for short, full in [
        ("mrs", "missus"), ("mr", "mister"), ("dr", "doctor"), ("st", "saint"), ("co", "company"),
        ("jr", "junior"), ("maj", "major"), ("gen", "general"), ("drs", "doctors"), ("rev", "reverend"),
        ("lt", "lieutenant"), ("hon", "honorable"), ("sgt", "sergeant"), ("capt", "captain"),
        ("esq", "esquire"), ("ltd", "limited"), ("col", "colonel"), ("ft", "fort"),
    ]
]


def expand_abbreviations(text: str) -> str:
    for pattern, full in _ABBREVIATIONS:
        text = pattern.sub(full, text)
    return text

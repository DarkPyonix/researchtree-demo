"""Input symbols for the acoustic model: padding, word gap, punctuation and ARPAbet phonemes."""

_ARPABET = [
    "AA", "AE", "AH", "AO", "AW", "AY", "B", "CH", "D", "DH", "EH", "ER", "EY", "F", "G", "HH",
    "IH", "IY", "JH", "K", "L", "M", "N", "NG", "OW", "OY", "P", "R", "S", "SH", "T", "TH",
    "UH", "UW", "V", "W", "Y", "Z", "ZH",
]
_VOWELS = {"AA", "AE", "AH", "AO", "AW", "AY", "EH", "ER", "EY", "IH", "IY", "OW", "OY", "UH", "UW"}

PAD = "<pad>"
SPACE = "<sp>"
PUNCTUATION = [",", ".", "?", "!", ";", ":"]

# Vowels carry a CMUdict stress marker (0 none, 1 primary, 2 secondary).
PHONEMES = [p + s for p in _ARPABET for s in (("0", "1", "2") if p in _VOWELS else ("",))]

SYMBOLS = [PAD, SPACE, *PUNCTUATION, *PHONEMES]
SYMBOL_TO_ID = {s: i for i, s in enumerate(SYMBOLS)}

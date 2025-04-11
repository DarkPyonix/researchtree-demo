"""Write the pronunciation dictionary for Montreal Forced Aligner: CMUdict plus G2P guesses.

MFA cannot align a clip that contains a word missing from its dictionary, and skips it. In
LJSpeech that is 2.7% of the clips, mostly names and old spellings. This script adds a
pronunciation from the g2p_en model for every missing word.

    python scripts/prepare_mfa_lexicon.py > data/lexicon.dict
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

from g2p_en import G2p

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tinyspeech.data import read_metadata  # noqa: E402
from tinyspeech.text.cmudict import CMUDict  # noqa: E402


def main() -> None:
    lexicon = CMUDict("data/cmudict-0.7b")
    words = {w for _, _, text in read_metadata("data/LJSpeech-1.1") for w in re.findall(r"[a-z']+", text.lower())}
    missing = sorted(w for w in words if w not in lexicon)
    g2p = G2p()
    for word, phones in sorted(lexicon.entries.items()):
        print(word, " ".join(phones))
    for word in missing:
        print(word, " ".join(p for p in g2p(word) if p.strip()))
    print(f"added {len(missing)} words with G2P", file=sys.stderr)


if __name__ == "__main__":
    main()

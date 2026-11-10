"""M8 misspelling and the Yak virama-joiner probes. Original text is never rewritten in place."""

from __future__ import annotations

import random

import regex

VIRAMA = "्"
ZWJ = "\u200d"
ZWNJ = "\u200c"
_VIRAMA_BEFORE_CONSONANT = regex.compile(r"्(?=[क-हक़-य़ॸ-ॿ])")
_SWAP = {"ि": "ी", "ी": "ि", "ु": "ू", "ू": "ु"}
_SIB = {"श": "स", "स": "ष", "ष": "श"}


def misspell_text(text: str, rng: random.Random) -> str:
    """Same character walk as ancoda_tok.m8_final.misspell."""
    out: list[str] = []
    for ch in text:
        if (ch in _SWAP or ch in _SIB) and rng.random() < 0.1:
            ch = _SWAP[ch] if ch in _SWAP else _SIB[ch]
        out.append(ch)
    return "".join(out)


def insert_joiner(text: str, joiner: str) -> str:
    """Put joiner after every virama that precedes a consonant."""
    return _VIRAMA_BEFORE_CONSONANT.sub(VIRAMA + joiner, text)


def misspell_corpus(texts: list[str], seed: int = 20261104) -> list[str]:
    rng = random.Random(seed)
    return [misspell_text(text, rng) for text in texts]

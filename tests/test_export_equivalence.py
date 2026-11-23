"""docs/SPEC.md 3.5 export check.

Full merges are every split ``(A, B)`` of a vocab token ``T = A + B``, sorted by
``(rank(T), rank(A), rank(B))``. Hugging Face BPE uses ``ignore_merges=True``.
Tiktoken is built with the same ranks and a pattern that keeps each word as one
pre-token. Training merges are only the pair that created each token.
Seed 20261104.
"""

from __future__ import annotations

import random

import tiktoken
from tokenizers import Tokenizer
from tokenizers.models import BPE

SEED = 20261104
TOY_VOCAB = {"a": 0, "b": 1, "c": 2, "d": 3, "bc": 4, "ab": 5, "abc": 6}
TOY_TRAINING = [("b", "c"), ("a", "b"), ("ab", "c")]


def _all_splits(vocab: dict[str, int]) -> list[tuple[str, str]]:
    pairs: set[tuple[str, str]] = set()
    for token in vocab:
        if len(token) < 2:
            continue
        for cut in range(1, len(token)):
            left, right = token[:cut], token[cut:]
            if left in vocab and right in vocab:
                pairs.add((left, right))
    return _sort_merges(pairs, vocab)


def _sort_merges(pairs: set[tuple[str, str]] | list[tuple[str, str]], vocab: dict[str, int]):
    return sorted(
        pairs, key=lambda pair: (vocab[pair[0] + pair[1]], vocab[pair[0]], vocab[pair[1]])
    )


def _hf(vocab: dict[str, int], merges: list[tuple[str, str]]) -> Tokenizer:
    model = BPE(
        vocab=vocab,
        merges=merges,
        ignore_merges=True,
        byte_fallback=False,
        dropout=None,
        unk_token=None,
        fuse_unk=False,
    )
    return Tokenizer(model)


def _tiktoken(vocab: dict[str, int]) -> tiktoken.Encoding:
    ranks = {token.encode("utf-8"): rank for token, rank in vocab.items()}
    return tiktoken.Encoding(
        name="ancoda-pretokenizer",
        pat_str=".+",
        mergeable_ranks=ranks,
        special_tokens={},
    )


def _same(hf: Tokenizer, tt: tiktoken.Encoding, word: str) -> bool:
    return hf.encode(word).ids == tt.encode_ordinary(word)


def _random_word(rng: random.Random, alphabet: list[str]) -> str:
    return "".join(rng.choice(alphabet) for _ in range(rng.randint(1, 10)))


def _random_vocab(rng: random.Random) -> tuple[dict[str, int], list[tuple[str, str]], list[str]]:
    alphabet = [chr(ord("a") + index) for index in range(rng.randint(3, 6))]
    vocab = {char: index for index, char in enumerate(alphabet)}
    pieces = list(alphabet)
    training: list[tuple[str, str]] = []
    target = rng.randint(5, 20)
    tries = 0
    while len(training) < target and tries < 300:
        tries += 1
        left = rng.choice(pieces)
        right = rng.choice(pieces)
        if len(left) + len(right) > 6:
            continue
        merged = left + right
        if merged in vocab:
            continue
        vocab[merged] = len(vocab)
        pieces.append(merged)
        training.append((left, right))
    return vocab, training, alphabet


def test_toy_and_random_vocabularies() -> None:
    rng = random.Random(SEED)
    full_bad: list[str] = []
    training_bad = 0

    toy_words = ["abcd"] + [_random_word(rng, list("abcd")) for _ in range(200)]
    toy_full = _all_splits(TOY_VOCAB)
    toy_training = _sort_merges(TOY_TRAINING, TOY_VOCAB)
    assert ("a", "bc") in toy_full
    assert ("a", "bc") not in toy_training
    toy_tt = _tiktoken(TOY_VOCAB)
    toy_hf_full = _hf(TOY_VOCAB, toy_full)
    toy_hf_train = _hf(TOY_VOCAB, toy_training)
    for word in toy_words:
        if not _same(toy_hf_full, toy_tt, word):
            full_bad.append(f"toy full {word!r}")
        if not _same(toy_hf_train, toy_tt, word):
            training_bad += 1
    assert not _same(toy_hf_train, toy_tt, "abcd")
    assert _same(toy_hf_full, toy_tt, "abcd")

    for _ in range(1000):
        vocab, training, alphabet = _random_vocab(rng)
        full = _all_splits(vocab)
        trained = _sort_merges(training, vocab)
        words = [_random_word(rng, alphabet) for _ in range(200)]
        tt = _tiktoken(vocab)
        hf_full = _hf(vocab, full)
        hf_train = _hf(vocab, trained)
        for word in words:
            if not _same(hf_full, tt, word):
                full_bad.append(f"random full {word!r} vocab={sorted(vocab)}")
                if len(full_bad) >= 5:
                    break
            if not _same(hf_train, tt, word):
                training_bad += 1
        if len(full_bad) >= 5:
            break

    assert full_bad == [], full_bad
    assert training_bad >= 1

"""Document permutation and byte accounting. Independent of Python's random module."""

from __future__ import annotations

import numpy as np


def fisher_yates(n: int, seed: int) -> np.ndarray:
    """Permutation of 0..n-1. LCG state starts at `seed` (mod 2^64)."""
    idx = np.arange(n, dtype=np.int64)
    state = seed & 0xFFFFFFFFFFFFFFFF
    mul = 6364136223846793005
    for i in range(n - 1, 0, -1):
        state = (state * mul + 1) & 0xFFFFFFFFFFFFFFFF
        j = state % (i + 1)
        idx[i], idx[j] = idx[j], idx[i]
    return idx


def stream_prefix(order: np.ndarray, tok_counts: np.ndarray) -> np.ndarray:
    """cum[i] = stream tokens (content + one EOS) before permuted document i. Length n+1."""
    n = int(order.shape[0])
    cum = np.zeros(n + 1, dtype=np.int64)
    if n:
        cum[1:] = np.cumsum(tok_counts[order] + np.int64(1), dtype=np.int64)
    return cum


def byte_prefix(order: np.ndarray, byte_counts: np.ndarray) -> np.ndarray:
    """bpre[i] = original UTF-8 bytes before permuted document i. Length n+1."""
    n = int(order.shape[0])
    bpre = np.zeros(n + 1, dtype=np.int64)
    if n:
        bpre[1:] = np.cumsum(byte_counts[order], dtype=np.int64)
    return bpre


def bytes_at(
    pos: int,
    order: np.ndarray,
    tok_counts: np.ndarray,
    byte_counts: np.ndarray,
    cum: np.ndarray,
    bpre: np.ndarray,
) -> int:
    """Original-text bytes covered by the first `pos` stream tokens.

    Content tokens share a document's bytes by integer division. The EOS id
    adds no bytes. A finished document contributes its full byte count.
    """
    if pos <= 0:
        return 0
    total = int(cum[-1])
    if pos >= total:
        return int(bpre[-1])
    j = int(np.searchsorted(cum, pos, side="right") - 1)
    base = int(bpre[j])
    into = pos - int(cum[j])
    if into <= 0:
        return base
    doc = int(order[j])
    tc = int(tok_counts[doc])
    bc = int(byte_counts[doc])
    content = min(tc, into)
    if tc == 0 or content >= tc:
        return base + bc
    return base + (bc * content) // tc


def first_step_reaching(byte_fn, bstar: int, tokens_per_step: int, max_steps: int) -> int | None:
    """Smallest completed step whose covered bytes reach bstar. Steps are 1-indexed counts."""
    lo, hi = 1, max_steps
    answer: int | None = None
    while lo <= hi:
        mid = (lo + hi) // 2
        if byte_fn(mid * tokens_per_step) >= bstar:
            answer = mid
            hi = mid - 1
        else:
            lo = mid + 1
    return answer


def content_offsets(tok_counts: np.ndarray) -> np.ndarray:
    """offsets[i] is the start of document i in the canonical content-token bin."""
    off = np.zeros(int(tok_counts.shape[0]) + 1, dtype=np.int64)
    if tok_counts.size:
        off[1:] = np.cumsum(tok_counts, dtype=np.int64)
    return off

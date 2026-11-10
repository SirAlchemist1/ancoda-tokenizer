"""Rank-prefix selection. The numpy arrays are ranks and index byte counts, aligned."""

from __future__ import annotations

import numpy as np


def prefix_spec(ranks: np.ndarray, nbytes: np.ndarray, target: int) -> dict:
    """Describe the (rank, doc_id) prefix whose byte sum first reaches `target`.

    `mode` is "none", "all", or "prefix". For "prefix", take every document with
    rank < cutoff, then documents with rank == cutoff in doc_id order until
    `need_at_cutoff` bytes are reached (the crossing document is included).
    """
    n = int(ranks.shape[0])
    available = int(nbytes.sum()) if n else 0
    if target <= 0 or n == 0:
        return {
            "mode": "none",
            "cutoff": None,
            "bytes_below": 0,
            "need_at_cutoff": 0,
            "available": available,
            "shortfall": max(target, 0),
        }
    if available < target:
        return {
            "mode": "all",
            "cutoff": None,
            "bytes_below": available,
            "need_at_cutoff": 0,
            "available": available,
            "shortfall": target - available,
        }
    order = np.argsort(ranks, kind="stable")
    sorted_ranks = ranks[order]
    sorted_bytes = nbytes[order]
    csum = np.cumsum(sorted_bytes, dtype=np.int64)
    idx = int(np.searchsorted(csum, np.int64(target), side="left"))
    cutoff = int(sorted_ranks[idx])
    first = idx
    while first > 0 and int(sorted_ranks[first - 1]) == cutoff:
        first -= 1
    bytes_below = int(csum[first - 1]) if first > 0 else 0
    return {
        "mode": "prefix",
        "cutoff": cutoff,
        "bytes_below": bytes_below,
        "need_at_cutoff": int(target - bytes_below),
        "available": available,
        "shortfall": 0,
    }

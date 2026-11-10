"""Memory-mapped, permuted, EOS-joined token stream."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ancoda_tok.m9.perm import (
    byte_prefix,
    bytes_at,
    content_offsets,
    fisher_yates,
    stream_prefix,
)


class TokenStream:
    def __init__(self, token_dir: Path, seed: int) -> None:
        manifest = json.loads((token_dir / "manifest.json").read_text(encoding="utf-8"))
        self.manifest = manifest
        self.eos = int(manifest["eos_id"])
        dtype = np.dtype("<u2" if manifest["dtype"] == "uint16" else "<u4")
        self.tokens = np.memmap(token_dir / "tokens.bin", dtype=dtype, mode="r")
        self.tok_counts = np.memmap(token_dir / "tok_counts.i32", dtype="<i4", mode="r")
        self.byte_counts = np.memmap(token_dir / "byte_counts.i64", dtype="<i8", mode="r")
        n = int(self.tok_counts.shape[0])
        if int(manifest["n_docs"]) != n:
            raise SystemExit(f"STOP: n_docs {manifest['n_docs']} != tok_counts {n}")
        self.order = fisher_yates(n, seed)
        self.cum = stream_prefix(self.order, np.asarray(self.tok_counts))
        self.bpre = byte_prefix(self.order, np.asarray(self.byte_counts))
        self.off = content_offsets(np.asarray(self.tok_counts))
        self.n_docs = n
        self.stream_tokens = int(self.cum[-1])

    def bytes_covered(self, pos: int) -> int:
        return bytes_at(
            pos,
            self.order,
            np.asarray(self.tok_counts),
            np.asarray(self.byte_counts),
            self.cum,
            self.bpre,
        )

    def read(self, pos: int, length: int) -> np.ndarray:
        if pos < 0 or length < 0 or pos + length > self.stream_tokens:
            raise SystemExit(
                f"STOP: read [{pos}, {pos + length}) outside stream of {self.stream_tokens}"
            )
        out = np.empty(length, dtype=np.int64)
        filled = 0
        cursor = pos
        while filled < length:
            j = int(np.searchsorted(self.cum, cursor, side="right") - 1)
            doc = int(self.order[j])
            into = cursor - int(self.cum[j])
            tc = int(self.tok_counts[doc])
            if into < tc:
                take = min(length - filled, tc - into)
                start = int(self.off[doc]) + into
                out[filled : filled + take] = self.tokens[start : start + take]
                filled += take
                cursor += take
            else:
                out[filled] = self.eos
                filled += 1
                cursor += 1
        return out

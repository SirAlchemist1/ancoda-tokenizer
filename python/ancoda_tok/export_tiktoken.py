"""Tiktoken export (PLAN.md 3.5). Ranks are the token bytes, not the byte-level unicode."""

from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path

import tiktoken

from ancoda_tok.export_hf import read_pattern


def export_tiktoken(
    tokens: list[bytes],
    specials: list[tuple[str, int]],
    out_dir: Path,
    pattern: str | None = None,
) -> Path:
    pattern = read_pattern() if pattern is None else pattern
    directory = out_dir / "tiktoken"
    directory.mkdir(parents=True, exist_ok=True)
    lines = [
        f"{base64.b64encode(raw).decode('ascii')} {rank}" for rank, raw in enumerate(tokens)
    ]
    rank_path = directory / "ancoda.tiktoken"
    rank_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    meta = {
        "pattern": pattern,
        "pattern_sha256": hashlib.sha256(pattern.encode("utf-8")).hexdigest(),
        "special_tokens": {name: ident for name, ident in specials},
    }
    (directory / "tiktoken.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    # The pattern must survive JSON escaping unchanged.
    loaded = json.loads((directory / "tiktoken.json").read_text(encoding="utf-8"))
    if loaded["pattern"] != pattern:
        raise RuntimeError("tiktoken.json pattern did not round-trip")
    return rank_path


def load_tiktoken(out_dir: Path, name: str = "ancoda") -> tiktoken.Encoding:
    directory = out_dir / "tiktoken"
    meta = json.loads((directory / "tiktoken.json").read_text(encoding="utf-8"))
    ranks: dict[bytes, int] = {}
    for line in (directory / "ancoda.tiktoken").read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        encoded, rank = line.split(" ")
        ranks[base64.b64decode(encoded)] = int(rank)
    specials = {name: int(ident) for name, ident in meta["special_tokens"].items()}
    return tiktoken.Encoding(
        name=name,
        pat_str=meta["pattern"],
        mergeable_ranks=ranks,
        special_tokens=specials,
    )

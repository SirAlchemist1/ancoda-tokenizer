"""Extra frontier metrics for T1–T4 diagnostics (PLAN.md 3.9 + held-out fertilities)."""

from __future__ import annotations

import json
import random
from pathlib import Path

import pyarrow.parquet as pq

from ancoda_tok.eval.morph import SAMPLE_N, sample_words, score_tokenizer
from ancoda_tok.export_hf import load_hf
from ancoda_tok.storage import REPO_ROOT

SEED = 20261104
RARE_LINES = 1_000_000
RARE_THRESHOLD = 100
N60_SNAP = REPO_ROOT / "data" / "snapshots" / "n60_cf8066dbbdd0da93"
FW2_TEST = REPO_ROOT / "data" / "raw" / "fw2_npi_test"

_CACHE: dict[str, object] = {}


def _cached_morph_words() -> list[tuple[str, int]]:
    key = "morph_words"
    if key not in _CACHE:
        _CACHE[key] = sample_words(SAMPLE_N)
    return _CACHE[key]  # type: ignore[return-value]


def _sample_snapshot_lines(snap: Path, n: int = RARE_LINES) -> list[str]:
    key = f"lines:{snap.resolve()}:{n}"
    if key in _CACHE:
        return _CACHE[key]  # type: ignore[return-value]
    lines: list[str] = []
    for shard in sorted(snap.glob("shard-*.parquet")):
        pf = pq.ParquetFile(shard)
        for batch in pf.iter_batches(columns=["text"], batch_size=256):
            for text in batch.column(0).to_pylist():
                if not text:
                    continue
                for line in text.splitlines():
                    if line.strip():
                        lines.append(line)
                        if len(lines) >= n * 2:
                            break
                if len(lines) >= n * 2:
                    break
        if len(lines) >= n * 2:
            break
    rng = random.Random(SEED)
    if len(lines) > n:
        # Deterministic subsample of n lines from the prefix pool.
        rng.shuffle(lines)
        lines = lines[:n]
    _CACHE[key] = lines
    meta = {
        "snapshot": str(snap.relative_to(REPO_ROOT)),
        "seed": SEED,
        "requested_lines": n,
        "actual_lines": len(lines),
        "selection": "shuffle_prefix_pool_seed_fixed",
    }
    _CACHE[f"meta:{key}"] = meta
    return lines


def fertility_on_texts(tok_path: Path, texts: list[str]) -> float:
    tok = load_hf(tok_path if tok_path.name == "tokenizer.json" else tok_path / "tokenizer.json")
    tokens = words = 0
    for text in texts:
        if not text:
            continue
        w = text.split()
        if not w:
            continue
        words += len(w)
        tokens += len(tok.encode(text, add_special_tokens=False).ids)
    if words == 0:
        return 0.0
    return tokens / words


def fw2_npi_test_texts(limit_docs: int = 50_000) -> list[str]:
    key = f"fw2:{limit_docs}"
    if key in _CACHE:
        return _CACHE[key]  # type: ignore[return-value]
    texts: list[str] = []
    for shard in sorted(FW2_TEST.glob("shard-*.parquet")):
        pf = pq.ParquetFile(shard)
        for batch in pf.iter_batches(columns=["text"], batch_size=256):
            for text in batch.column(0).to_pylist():
                if text:
                    texts.append(text)
                    if len(texts) >= limit_docs:
                        break
            if len(texts) >= limit_docs:
                break
        if len(texts) >= limit_docs:
            break
    _CACHE[key] = texts
    return texts


def rare_token_count(
    tok_dir: Path,
    *,
    snap: Path,
    threshold: int = RARE_THRESHOLD,
    n_lines: int = RARE_LINES,
) -> dict:
    """Vocab entries with occurrence < threshold in n_lines.

    Ancoda: mergeable ranks tokens (ids < reserved_start).
    HF-only (Yak): all vocab ids from get_vocab().
    """
    tok_json = tok_dir / "tokenizer.json" if tok_dir.is_dir() else tok_dir
    tok = load_hf(tok_json)
    ranks_path = tok_json.parent / "ranks.json"
    if ranks_path.is_file():
        layout = json.loads(ranks_path.read_text(encoding="utf-8"))["layout"]
        n_mergeable = int(layout["reserved_start"])
        id_set = range(n_mergeable)
    else:
        vocab = tok.get_vocab()
        n_mergeable = len(vocab)
        id_set = sorted(vocab.values())
    lines = _sample_snapshot_lines(snap, n_lines)
    meta = _CACHE[f"meta:lines:{snap.resolve()}:{n_lines}"]
    counts = {i: 0 for i in id_set}
    batch = 512
    for start in range(0, len(lines), batch):
        chunk = lines[start : start + batch]
        encoded = tok.encode_batch(chunk, add_special_tokens=False)
        for enc in encoded:
            for tid in enc.ids:
                if tid in counts:
                    counts[tid] += 1
        if start % 20000 == 0:
            print(f"  rare-count lines {start}/{len(lines)}", flush=True)
    rare = sum(1 for c in counts.values() if c < threshold)
    return {
        "rare_tokens_lt_100": rare,
        "mergeable_tokens": n_mergeable,
        "rare_threshold": threshold,
        "rare_lines": meta,
    }


def compute_extra_metrics(
    tok_dir: Path,
    *,
    snap: Path | None = None,
    aya_fertility: float | None = None,
) -> dict:
    """Aya + fw2 fertilities, rare tokens, morphology F1 / whole-word."""
    snap = snap or N60_SNAP
    tok_json = tok_dir / "tokenizer.json" if tok_dir.is_dir() else tok_dir
    tok_dir_resolved = tok_json.parent if tok_json.name == "tokenizer.json" else tok_dir

    fw2_texts = fw2_npi_test_texts()
    fw2_fert = fertility_on_texts(tok_json, fw2_texts)

    aya = aya_fertility
    if aya is None:
        # Compute from harness everyday file if present.
        aya_path = (
            REPO_ROOT
            / "third_party"
            / "yak-tokenizer"
            / "data"
            / "eval"
            / "npi_everyday"
            / "aya_npi.jsonl"
        )
        texts = []
        if aya_path.is_file():
            for line in aya_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                obj = json.loads(line)
                texts.append(obj.get("text") or obj.get("prompt", "") + "\n" + obj.get("answer", ""))
        aya = fertility_on_texts(tok_json, texts) if texts else None

    # score_tokenizer expects a directory containing tokenizer.json
    morph_dir = tok_dir_resolved
    if not (morph_dir / "tokenizer.json").is_file() and tok_json.is_file():
        morph_dir = tok_json.parent
    morph = score_tokenizer(morph_dir, _cached_morph_words())
    rare = rare_token_count(morph_dir, snap=snap)

    return {
        "aya_fertility": aya,
        "fw2_npi_test_fertility": fw2_fert,
        "fw2_npi_test_docs": len(fw2_texts),
        "morph_f1_unweighted": morph["unweighted"]["f1"],
        "morph_f1_frequency_weighted": morph["frequency_weighted"]["f1"],
        "whole_word_unweighted": morph["whole_word_preservation_unweighted"],
        "whole_word_frequency_weighted": morph["whole_word_preservation_frequency_weighted"],
        "morph_words": morph["words"],
        **rare,
    }


def attach_extra_metrics(
    row: dict,
    tok_dir: Path,
    *,
    snap: Path | None = None,
    harness_aya: float | None = None,
) -> None:
    extra = compute_extra_metrics(tok_dir, snap=snap, aya_fertility=harness_aya)
    row.update(extra)

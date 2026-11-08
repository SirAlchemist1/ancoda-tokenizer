"""Morphology alignment and whole-word preservation (PLAN.md 3.9)."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import pyarrow.parquet as pq

from ancoda_tok.clean.common import base_manifest, finish_manifest, save_json
from ancoda_tok.export_hf import load_hf
from ancoda_tok.storage import REPO_ROOT

SEED = 20261104
SAMPLE_N = 20_000
AFFIX_CANDIDATES = [
    REPO_ROOT / "data" / "affix_inventory.tsv",
    REPO_ROOT / "data" / "raw" / "papaya_affixes" / "data" / "affix_inventory.tsv",
]


def _affix_path() -> Path:
    for path in AFFIX_CANDIDATES:
        if path.is_file():
            return path
    raise SystemExit("STOP: Papaya affix inventory not found under data/")


def load_suffixes(path: Path | None = None) -> list[str]:
    """Longest-first Devanagari suffix strings from the Papaya inventory."""
    path = path or _affix_path()
    suffixes: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines()[1:]:
        if not line.strip():
            continue
        fields = line.split("\t")
        if len(fields) < 4:
            continue
        affix = fields[3].strip()
        affix = affix.removeprefix("-")
        if not affix or affix == "∅":
            continue
        if any("\u0900" <= ch <= "\u097f" for ch in affix):
            suffixes.add(affix)
    return sorted(suffixes, key=lambda s: (-len(s), s))


def reference_boundaries(word: str, suffixes: list[str]) -> list[int]:
    """Character offsets of suffix cuts via repeated longest-match from the end."""
    cuts: list[int] = []
    end = len(word)
    while end > 0:
        matched = None
        for suffix in suffixes:
            if len(suffix) >= end:
                continue
            if word.endswith(suffix, 0, end):
                matched = suffix
                break
        if matched is None:
            break
        cut = end - len(matched)
        if cut <= 0:
            break
        cuts.append(cut)
        end = cut
    cuts.reverse()
    return cuts


def sample_words(limit: int = SAMPLE_N) -> list[tuple[str, int]]:
    """Distinct Nepali words from fw2_npi_test with frequencies, seed-fixed."""
    root = REPO_ROOT / "data" / "raw" / "fw2_npi_test"
    freq: dict[str, int] = {}
    for shard in sorted(root.glob("shard-*.parquet")):
        parquet = pq.ParquetFile(shard)
        for batch in parquet.iter_batches(columns=["text"], batch_size=512):
            for value in batch.column(0).to_pylist():
                if not value:
                    continue
                for word in value.split():
                    if any("\u0900" <= ch <= "\u097f" for ch in word):
                        freq[word] = freq.get(word, 0) + 1
    words = sorted(freq)
    rng = random.Random(SEED)
    rng.shuffle(words)
    return [(w, freq[w]) for w in words[:limit]]


def token_char_bounds(tokenizer, word: str) -> list[int]:
    """Internal character offsets of token boundaries for ``" " + word``."""
    text = " " + word
    encoded = tokenizer.encode(text, add_special_tokens=False)
    bounds: list[int] = []
    for i in range(1, len(encoded.ids) + 1):
        bounds.append(len(tokenizer.decode(encoded.ids[:i], skip_special_tokens=False)))
    full = tokenizer.decode(encoded.ids, skip_special_tokens=False)
    if full != text:
        return []
    internal = []
    for end in bounds[:-1]:
        if end <= 1:
            continue
        internal.append(end - 1)
    return internal


def _prf(tp: float, fp: float, hits: float, misses: float) -> dict[str, float]:
    precision = 0.0 if (tp + fp) == 0 else tp / (tp + fp)
    recall = 0.0 if (hits + misses) == 0 else hits / (hits + misses)
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return {"precision": precision, "recall": recall, "f1": f1}


def score_tokenizer(tokenizer_dir: Path, words: list[tuple[str, int]] | None = None) -> dict:
    suffixes = load_suffixes()
    sample = words if words is not None else sample_words()
    tokenizer = load_hf(tokenizer_dir / "tokenizer.json")

    tp = fp = hits = misses = 0
    tp_w = fp_w = hits_w = misses_w = 0.0
    whole = 0
    whole_w = 0.0
    total_w = 0.0
    for word, count in sample:
        weight = float(count)
        total_w += weight
        ref = set(reference_boundaries(word, suffixes))
        ids = tokenizer.encode(" " + word, add_special_tokens=False).ids
        if len(ids) == 1:
            whole += 1
            whole_w += weight
        pred = set(token_char_bounds(tokenizer, word))
        for boundary in pred:
            if boundary in ref:
                tp += 1
                tp_w += weight
            else:
                fp += 1
                fp_w += weight
        for boundary in ref:
            if boundary in pred:
                hits += 1
                hits_w += weight
            else:
                misses += 1
                misses_w += weight

    n = len(sample)
    return {
        "words": n,
        "affix_path": str(_affix_path().relative_to(REPO_ROOT)),
        "suffixes": len(suffixes),
        "unweighted": _prf(float(tp), float(fp), float(hits), float(misses)),
        "frequency_weighted": _prf(tp_w, fp_w, hits_w, misses_w),
        "whole_word_preservation_unweighted": 0.0 if n == 0 else whole / n,
        "whole_word_preservation_frequency_weighted": (
            0.0 if total_w == 0 else whole_w / total_w
        ),
        "counts": {
            "tp": tp,
            "fp": fp,
            "ref_hits": hits,
            "ref_misses": misses,
            "whole_word": whole,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    result = score_tokenizer(args.tokenizer_dir)
    out = args.out or (
        REPO_ROOT / "artifacts" / "eval" / "m7" / f"morph_{args.tokenizer_dir.name}.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        **result,
        "tokenizer_dir": str(args.tokenizer_dir.relative_to(REPO_ROOT)),
    }
    save_json(out, finish_manifest(base_manifest(kind="morphology", **payload)))
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()

"""Morphology alignment and whole-word preservation (PLAN.md 3.9)."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import ancoda_seg
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
        "whole_word_preservation_frequency_weighted": (0.0 if total_w == 0 else whole_w / total_w),
        "counts": {
            "tp": tp,
            "fp": fp,
            "ref_hits": hits,
            "ref_misses": misses,
            "whole_word": whole,
        },
    }


# Nepali case markers for the boundary metric. Longest match wins.
# This list is fixed; the spot-check sample is not allowed to change it.
CASE_MARKERS: tuple[str, ...] = (
    "ले",
    "लाई",
    "मा",
    "को",
    "का",
    "की",
    "बाट",
    "देखि",
    "सँग",
    "सम्म",
    "भन्दा",
)
STEM_MIN_COUNT = 20


def case_markers_longest_first() -> tuple[str, ...]:
    return tuple(sorted(CASE_MARKERS, key=lambda marker: (-len(marker), marker)))


def split_case_marker(word: str, markers: tuple[str, ...] | None = None) -> tuple[str, str] | None:
    """Return (stem, marker) when ``word`` ends in a listed marker and the stem is nonempty."""
    ordered = markers if markers is not None else case_markers_longest_first()
    for marker in ordered:
        if len(word) > len(marker) and word.endswith(marker):
            return word[: -len(marker)], marker
    return None


VERB_FORM_MARKERS: tuple[str, ...] = ("को", "का", "की", "मा")
VOWEL_SIGN_E = "\u0947"
INDEPENDENT_E = "\u090f"
EXCLUSION_REASONS: tuple[str, ...] = (
    "non_devanagari",
    "single_unit",
    "e_sign_verb_form",
    "independent_e_verb_form",
)


def is_devanagari_block(ch: str) -> bool:
    return "\u0900" <= ch <= "\u097f"


def case_marker_exclusion(
    word: str, stem: str, marker: str, *, independent_e: bool = True
) -> str | None:
    """Reason a stem|marker candidate is not a noun plus case marker, or None to keep it.

    Reasons are checked in ``EXCLUSION_REASONS`` order and the first match is returned.
    ``single_unit`` uses the Ancoda protected unit (``ancoda_seg.aksharas``).
    The verb-form reasons drop participles (को/का/की) and conditionals (मा) whose stem
    ends in the vowel sign े, or in the independent vowel ए after another vowel
    (आएको, खुलाएका). Other markers after such a stem are kept.
    """
    if not all(is_devanagari_block(ch) for ch in word):
        return "non_devanagari"
    if len(ancoda_seg.aksharas(stem)) == 1:
        return "single_unit"
    if marker in VERB_FORM_MARKERS:
        if stem.endswith(VOWEL_SIGN_E):
            return "e_sign_verb_form"
        if independent_e and stem.endswith(INDEPENDENT_E):
            return "independent_e_verb_form"
    return None


def markers_missing_from_inventory(suffixes: list[str] | set[str]) -> list[str]:
    """Markers with no exact affix-string match in the Papaya inventory."""
    have = set(suffixes)
    return [marker for marker in CASE_MARKERS if marker not in have]


def pretoken_surface(pretoken: str) -> str:
    """Standalone-word surface of a pre-token.

    The n60j5 table stores Ancoda pre-tokens, which may carry leading whitespace
    from the pre-tokenizer. Only that leading whitespace is removed. The stored
    text is not normalized.
    """
    return pretoken.lstrip()


def classify_marker_spans(spans: list[tuple[int, int]], cut: int, nchars: int) -> str:
    """Classify one word from token spans in word-character coordinates.

    ``spans`` are half-open intervals inside the word. The leading space used at
    encode time is already removed. ``cut`` is the stem|marker boundary.
    """
    if cut <= 0 or cut >= nchars:
        raise ValueError("case-marker cut must fall inside the word")
    if not spans:
        return "other"
    starts = {start for start, _end in spans}
    ends = {end for _start, end in spans}
    if cut in starts and cut in ends:
        return "respected"
    if len(spans) == 1 and spans[0] == (0, nchars):
        return "fused"
    if len(spans) > 1 and any(start < cut < end for start, end in spans):
        return "crossed"
    return "other"


def word_spans_from_pieces(word: str, pieces: list[bytes]) -> list[tuple[int, int]] | None:
    """Character spans of ``pieces`` inside ``word``.

    ``pieces`` are the token byte strings of ``" " + word``, in order, and must
    concatenate to that text. A token that only covers the leading space is dropped.
    Returns None when the pieces are not a byte partition of the text.
    """
    text = " " + word
    raw = text.encode("utf-8")
    if b"".join(pieces) != raw:
        return None
    byte_to_char = [0] * (len(raw) + 1)
    pos = 0
    for index, ch in enumerate(text):
        width = len(ch.encode("utf-8"))
        for _ in range(width):
            byte_to_char[pos] = index
            pos += 1
    byte_to_char[pos] = len(text)
    spans: list[tuple[int, int]] = []
    pos = 0
    for piece in pieces:
        start_b = pos
        pos += len(piece)
        start_c = byte_to_char[start_b]
        end_c = byte_to_char[pos]
        word_start = max(start_c, 1) - 1
        word_end = min(end_c, len(text)) - 1
        if word_end > word_start:
            spans.append((word_start, word_end))
    return spans


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

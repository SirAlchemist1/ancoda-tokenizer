"""Nepali case-marker boundary metric. Does not train and does not touch the server.

Extends ancoda_tok.eval.morph without changing its existing morphology outputs.
The 200-word spot-check is written for a native speaker and is not an input.
"""

from __future__ import annotations

import io
import json
import random
from collections import Counter, defaultdict
from typing import Any

import numpy as np
import tokenizers
import zstandard as zstd
from analysis_extra_per_sentence import himalaya_specs

from ancoda_tok.counts import iter_entries
from ancoda_tok.eval.morph import (
    CASE_MARKERS,
    STEM_MIN_COUNT,
    _affix_path,
    classify_marker_spans,
    load_suffixes,
    markers_missing_from_inventory,
    pretoken_surface,
    split_case_marker,
    word_spans_from_pieces,
)
from ancoda_tok.m8_final import BOOT_N, REPO_ROOT, SEED, Encoder, verdict
from ancoda_tok.m8_flores_per_sentence import read_sentences
from ancoda_tok.m8_followup_report import load_specs

OUT_JSON = REPO_ROOT / "artifacts" / "analysis" / "case_markers.json"
OUT_MD = REPO_ROOT / "docs" / "ANALYSIS_CASE_MARKERS.md"
SPOT = REPO_ROOT / "docs" / "case_marker_spotcheck.tsv"
SEALED = REPO_ROOT / "data" / "m9" / "eval" / "sealed.jsonl.zst"
COUNTS = REPO_ROOT / "artifacts" / "counts" / "n60j5" / "pretoken_counts.bin"
COUNTS_MANIFEST = REPO_ROOT / "artifacts" / "counts" / "n60j5" / "manifest.json"
PLAIN = REPO_ROOT / "artifacts" / "tokenizers" / "m9" / "plain-64k" / "tokenizer.json"

CLASS_OURS = {
    "32k": ["ancoda-32k"],
    "64k": ["ancoda-64k", "ancoda-ne-64k"],
    "128k": ["ancoda-128k"],
    "250k": ["ancoda-250k"],
}
CLASS_COMPETITORS = {
    "32k": ["yak-32k-v1"],
    "64k": [
        "yak-64k-v1",
        "yak-64k-v2.0",
        "sidskarki-64k",
        "arkios",
        "sarvam-1",
        "himalaya-64k",
        "plain-64k",
    ],
    "128k": [
        "yak-128k-v1",
        "yak-128k-v2.0",
        "llama3",
        "sarvam-m",
        "qwen2.5",
        "qwen3",
        "himalaya-100k",
        "himalaya-128k",
    ],
    "250k": ["indicbert-v2", "bloom", "gemma3", "sarvam-30b", "o200k"],
}
LABELS = ("respected", "fused", "crossed", "other")


def log(msg: str) -> None:
    print(msg, flush=True)


def panel_names() -> list[str]:
    names: list[str] = []
    for size in ("32k", "64k", "128k", "250k"):
        names.extend(CLASS_OURS[size])
        names.extend(CLASS_COMPETITORS[size])
    return names


def panel_specs() -> dict[str, str]:
    specs = load_specs()
    specs.update(himalaya_specs())
    specs["o200k"] = "o200k"
    if not PLAIN.is_file():
        raise SystemExit(f"STOP: missing {PLAIN}")
    specs["plain-64k"] = str(PLAIN)
    names = panel_names()
    missing = [name for name in names if name not in specs]
    if missing:
        raise SystemExit(f"STOP: missing tokenizer specs {missing}")
    return {name: specs[name] for name in names}


def sealed_nepali() -> list[str]:
    texts: list[str] = []
    dctx = zstd.ZstdDecompressor()
    with SEALED.open("rb") as raw, dctx.stream_reader(raw) as reader:
        for line in io.TextIOWrapper(reader, encoding="utf-8"):
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("lang") == "Nepali":
                texts.append(row["text"])
    if len(texts) != 28094:
        raise SystemExit(f"STOP: sealed Nepali docs {len(texts)}, expected 28094")
    return texts


def scan(texts: list[str]) -> list[list[tuple[str, str, str]]]:
    docs: list[list[tuple[str, str, str]]] = []
    for text in texts:
        hits: list[tuple[str, str, str]] = []
        for word in text.split():
            split = split_case_marker(word)
            if split is None:
                continue
            stem, marker = split
            hits.append((word, stem, marker))
        docs.append(hits)
    return docs


def tally(docs: list[list[tuple[str, str, str]]]) -> tuple[dict[str, int], dict[str, int]]:
    types: dict[str, set[str]] = {marker: set() for marker in CASE_MARKERS}
    tokens: dict[str, int] = {marker: 0 for marker in CASE_MARKERS}
    for hits in docs:
        for word, _stem, marker in hits:
            types[marker].add(word)
            tokens[marker] += 1
    return {marker: len(types[marker]) for marker in CASE_MARKERS}, tokens


def stem_counts(stems: set[str]) -> dict[str, int]:
    totals = dict.fromkeys(stems, 0)
    seen = 0
    for raw, count in iter_entries(COUNTS):
        seen += 1
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            continue
        surface = pretoken_surface(text)
        if surface in totals:
            totals[surface] += count
    log(f"scanned {seen} n60j5 pretokens")
    return totals


def filter_docs(
    docs: list[list[tuple[str, str, str]]], counts: dict[str, int]
) -> list[list[tuple[str, str, str]]]:
    kept: list[list[tuple[str, str, str]]] = []
    for hits in docs:
        kept.append([hit for hit in hits if counts[hit[1]] >= STEM_MIN_COUNT])
    return kept


def _decode(enc: Encoder, ids: list[int]) -> str:
    if enc.kind == "tiktoken":
        return str(enc.tt.decode(ids))
    if enc.kind == "sentencepiece":
        return str(enc.sp.decode(ids))
    return str(enc.tok.decode(ids, skip_special_tokens=False))


def _clip(offsets: list[tuple[int, int]], text_len: int) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    for start, end in offsets:
        if end <= start:
            continue
        word_start = max(start, 1) - 1
        word_end = min(end, text_len) - 1
        if word_end > word_start:
            spans.append((word_start, word_end))
    return spans


def _clean_word_spans(
    offsets: list[tuple[int, int]], text_len: int
) -> list[tuple[int, int]] | None:
    """Clip offsets that cover the word once and do not overlap.

    The leading space may be inside a token or outside every token. Overlapping
    character offsets (byte-level models such as Qwen) are rejected so the
    caller can fall back to decode lengths.
    """
    if text_len <= 1:
        return None
    cover = [0] * text_len
    for start, end in offsets:
        if start < 0 or end > text_len or end <= start:
            return None
        for index in range(start, end):
            if cover[index]:
                return None
            cover[index] = 1
    if any(cover[index] != 1 for index in range(1, text_len)):
        return None
    if cover[0] not in (0, 1):
        return None
    return _clip(offsets, text_len)


def _spans_from_decode(enc: Encoder, word: str) -> list[tuple[int, int]] | None:
    text = " " + word
    ids = enc.ids(text)
    if not ids:
        return None
    ends: list[int] = []
    for index in range(1, len(ids) + 1):
        ends.append(len(_decode(enc, ids[:index])))
    if _decode(enc, ids) != text:
        return None
    spans: list[tuple[int, int]] = []
    prev = 0
    for end in ends:
        word_start = max(prev, 1) - 1
        word_end = min(end, len(text)) - 1
        if word_end > word_start:
            spans.append((word_start, word_end))
        prev = end
    return spans


def word_spans(enc: Encoder, word: str) -> list[tuple[int, int]] | None:
    """Token spans inside ``word``. The leading space of ``" " + word`` is ignored."""
    text = " " + word
    if enc.kind == "hf":
        encoded = enc.tok.encode(text, add_special_tokens=False)
        offsets = [(int(start), int(end)) for start, end in encoded.offsets]
        spans = _clean_word_spans(offsets, len(text))
        if spans is not None:
            return spans
        return _spans_from_decode(enc, word)
    if enc.kind == "sentencepiece":
        mapping = enc.sp.encode_as_offset_mapping(text)
        offsets = [(int(start), int(end)) for start, end in mapping["offsets"]]
        spans = _clean_word_spans(offsets, len(text))
        if spans is not None:
            return spans
        return _spans_from_decode(enc, word)
    pieces = enc.pieces(text)
    if pieces is not None and all(_utf8(piece) for piece in pieces):
        return word_spans_from_pieces(word, pieces)
    return _spans_from_decode(enc, word)


def _utf8(piece: bytes) -> bool:
    try:
        piece.decode("utf-8")
    except UnicodeDecodeError:
        return False
    return True


def classify_word(enc: Encoder, word: str, stem: str) -> str:
    spans = word_spans(enc, word)
    if spans is None:
        return "other"
    return classify_marker_spans(spans, len(stem), len(word))


def share_block(labels: list[str], weights: list[int]) -> dict[str, Any]:
    type_n = {label: 0 for label in LABELS}
    token_n = {label: 0 for label in LABELS}
    for label, weight in zip(labels, weights, strict=True):
        type_n[label] += 1
        token_n[label] += weight
    type_total = sum(type_n.values())
    token_total = sum(token_n.values())

    def pack(counts: dict[str, int], total: int) -> dict[str, Any]:
        return {
            "counts": counts,
            "total": total,
            "share": {label: (counts[label] / total if total else None) for label in LABELS},
        }

    return {"type": pack(type_n, type_total), "token": pack(token_n, token_total)}


def _rate_ci(left: np.ndarray, right: np.ndarray, weight: np.ndarray) -> dict[str, Any]:
    """Paired bootstrap of crossed share, left minus right. Same percentile as fertility."""
    total = float(weight.sum())
    if total <= 0:
        raise SystemExit("STOP: bootstrap population has no kept words")
    point = float(left.sum() / total - right.sum() / total)
    size = len(weight)
    rng = np.random.default_rng(SEED)
    diffs = np.empty(BOOT_N, dtype=np.float64)
    for draw in range(BOOT_N):
        idx = rng.integers(0, size, size=size)
        denom = float(weight[idx].sum())
        diffs[draw] = float(left[idx].sum() / denom - right[idx].sum() / denom)
    diffs.sort()
    out = {
        "diff": point,
        "lo": float(diffs[int(0.025 * (BOOT_N - 1))]),
        "hi": float(diffs[int(0.975 * (BOOT_N - 1))]),
        "n": BOOT_N,
        "seed": SEED,
    }
    out["verdict"] = verdict(out)
    return out


def population(
    docs: list[list[tuple[str, str, str]]],
    labels_by_word: dict[str, str],
) -> tuple[np.ndarray, np.ndarray, list[str], list[int]]:
    """Document crossed-counts, plus one row per kept type with its token weight."""
    usable = [hits for hits in docs if hits]
    hit = np.zeros(len(usable), dtype=np.float64)
    weight = np.zeros(len(usable), dtype=np.float64)
    type_weight: dict[str, int] = defaultdict(int)
    for index, hits in enumerate(usable):
        weight[index] = len(hits)
        crossed = 0
        for word, _stem, _marker in hits:
            type_weight[word] += 1
            if labels_by_word[word] == "crossed":
                crossed += 1
        hit[index] = crossed
    words = sorted(type_weight)
    type_labels = [labels_by_word[word] for word in words]
    type_weights = [type_weight[word] for word in words]
    return hit, weight, type_labels, type_weights


def fmt_share(value: float | None) -> str:
    if value is None:
        return "—"
    return f"{value:.4f}"


def fmt_ci(ci: dict[str, Any]) -> str:
    return f"{ci['verdict']} {ci['diff']:+.4f} [{ci['lo']:+.4f}, {ci['hi']:+.4f}]"


def write_spotcheck(
    words: list[str],
    meta: dict[str, tuple[str, str, int]],
    sealed_n: Counter[str],
    flores_n: Counter[str],
) -> None:
    rng = random.Random(SEED)
    pool = sorted(words)
    rng.shuffle(pool)
    # A type that occurs only in FLORES+ stays in the metric and stays out of
    # this file, so the committed sample does not re-host FLORES+ text.
    chosen = [word for word in pool if sealed_n[word] > 0][:200]
    if len(chosen) < 200:
        raise SystemExit(f"STOP: spot-check pool has {len(chosen)} sealed types")
    chosen.sort(key=lambda word: (meta[word][1], word))
    lines = ["word\tstem\tmarker\tstem_count\tsealed_tokens\tflores_tokens"]
    for word in chosen:
        stem, marker, stem_count = meta[word]
        lines.append(f"{word}\t{stem}\t{marker}\t{stem_count}\t{sealed_n[word]}\t{flores_n[word]}")
    SPOT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    log(f"wrote {SPOT} rows={len(chosen)}")


def marker_table(rows: dict[str, dict[str, dict[str, int]]], corpus: str) -> list[str]:
    lines = [
        f"| Marker | {corpus} candidate types | candidate tokens | kept types | kept tokens |",
        "|---|---:|---:|---:|---:|",
    ]
    cand_t = cand_n = kept_t = kept_n = 0
    for marker in CASE_MARKERS:
        row = rows[corpus][marker]
        cand_t += row["candidate_types"]
        cand_n += row["candidate_tokens"]
        kept_t += row["kept_types"]
        kept_n += row["kept_tokens"]
        lines.append(
            f"| {marker} | {row['candidate_types']} | {row['candidate_tokens']} | "
            f"{row['kept_types']} | {row['kept_tokens']} |"
        )
    lines.append(f"| total | {cand_t} | {cand_n} | {kept_t} | {kept_n} |")
    lines.append("")
    return lines


def share_table(scores: dict[str, Any], corpus: str) -> list[str]:
    lines = [
        (
            "| Tokenizer | respected types | fused types | crossed types | other types | "
            "respected tokens | fused tokens | crossed tokens | other tokens |"
        ),
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in panel_names():
        block = scores[corpus][name]
        cells = []
        for kind in ("type", "token"):
            for label in LABELS:
                cells.append(fmt_share(block[kind]["share"][label]))
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    lines.append("")
    return lines


def ci_table(cis: dict[str, Any], corpus: str, kind: str) -> list[str]:
    lines = [
        "| Ours | Competitor | Crossed share, ours − competitor |",
        "|---|---|---|",
    ]
    for size in ("32k", "64k", "128k", "250k"):
        for ours in CLASS_OURS[size]:
            for comp in CLASS_COMPETITORS[size]:
                hit = cis[f"{corpus}|{kind}|{ours}|{comp}"]
                lines.append(f"| {ours} | {comp} | {fmt_ci(hit)} |")
    lines.append("")
    return lines


def write_md(payload: dict[str, Any]) -> None:
    missing = payload["papaya_missing"]
    missing_text = ", ".join(missing) if missing else "none"
    lines = [
        "# Nepali case-marker boundaries",
        "",
        (
            "Analysis only. No training run, and the server was not used. "
            "Reproduction: `scripts/analysis_case_markers.py` "
            f"(tokenizers {payload['tokenizers_version']}). "
            "The morphology scorer in `python/ancoda_tok/eval/morph.py` is unchanged; "
            "this metric is a separate function in that module."
        ),
        "",
        "Words are whitespace-separated tokens from sealed Nepali (28,094 documents "
        "in the no-joiner sealed set) and from FLORES+ Nepali devtest (1,012 lines "
        "that contain a word). A word is a candidate when it ends in one of: "
        + ", ".join(CASE_MARKERS)
        + ". Longest match wins, and the stem (the word minus the marker) must be "
        "nonempty. No Unicode normalization is applied.",
        "",
        (
            "A candidate is kept when its stem occurs as a standalone word at least "
            f"{STEM_MIN_COUNT} times in the n60j5 pre-token counts "
            "(`artifacts/counts/n60j5/pretoken_counts.bin`). A pre-token counts toward "
            "that stem when dropping its leading whitespace leaves exactly the stem. "
            f"Count-table sha256 `{payload['counts_sha256']}`."
        ),
        "",
        "## Papaya inventory",
        "",
        (
            f"Affix file: `{payload['affix_path']}`. "
            "A marker is present when the affix column, with one leading hyphen removed, "
            "equals the marker. Missing markers: "
            f"**{missing_text}**. "
            "The missing marker stays in the metric. The inventory was not edited, "
            "and the spot-check below was not used to add or drop a marker."
        ),
        "",
        "## Words kept",
        "",
        "Candidate counts are before the frequency filter. Kept counts are after it.",
        "",
        *marker_table(payload["markers"], "sealed"),
        "",
        *marker_table(payload["markers"], "flores"),
        "",
        (
            f"Kept word types in total, counting a type once if it occurs in either corpus: "
            f"**{payload['kept_types']}**. "
            f"Sealed kept tokens: **{payload['sealed_kept_tokens']}** "
            f"in {payload['sealed_docs_with_kept']} of {payload['sealed_docs']} documents. "
            f"FLORES+ kept tokens: **{payload['flores_kept_tokens']}** "
            f"in {payload['flores_lines_with_kept']} of {payload['flores_lines']} lines."
        ),
        "",
        "## Spot-check",
        "",
        (
            f"`docs/case_marker_spotcheck.tsv` is {payload['spotcheck_n']} kept types that "
            f"occur in the sealed set, taken in order from a `random.Random({SEED})` shuffle "
            "of the kept types and then sorted by marker and word for reading. "
            "Types that occur only in FLORES+ are left out of this file. "
            "It is for a native-speaker check of the stem|marker split. "
            "It was not used to change the marker list, the count threshold, or the classifier."
        ),
        "",
        "## Boundary classes",
        "",
        (
            "Each kept occurrence is classified from the tokenization of a leading space "
            "plus the word. Offsets are character offsets inside the word; the leading "
            "space is ignored."
        ),
        "",
        "- **respected**: a token boundary falls exactly at the stem|marker point.",
        "- **fused**: the whole word is one token.",
        (
            "- **crossed**: some token covers the last character of the stem and the first "
            "character of the marker, and the word is more than one token."
        ),
        "- **other**: none of those three. A round-trip failure is recorded here.",
        "",
        (
            "Type shares give each kept word type equal weight. Token shares weight by "
            "occurrences in that corpus. The panel is the 16 per-sentence comparison "
            "tokenizers, the five Ancoda builds in those tables, himalaya-64k, "
            "himalaya-100k, himalaya-128k, o200k, and plain-64k. "
            "himalaya-100k is paired with ancoda-128k, as in the fertility tables."
        ),
        "",
        "### Sealed Nepali",
        "",
        *share_table(payload["scores"], "sealed"),
        "### FLORES+ Nepali",
        "",
        *share_table(payload["scores"], "flores"),
        "## Crossed-share intervals",
        "",
        (
            f"Paired bootstrap, {BOOT_N} resamples, seed {SEED}. "
            "The difference is Ancoda minus the competitor, so a negative value means "
            "Ancoda crosses the stem|marker point less often. "
            "WIN means the 95% interval is entirely below 0. "
            "The token interval resamples documents (sealed) or sentences (FLORES+) that "
            "contain at least one kept word, and recomputes the occurrence-weighted share. "
            "The type interval resamples kept word types in that corpus, each with weight 1. "
            "Each comparison uses its own generator started at the same seed."
        ),
        "",
        "### Sealed, by token",
        "",
        *ci_table(payload["crossed_ci"], "sealed", "token"),
        "### Sealed, by type",
        "",
        *ci_table(payload["crossed_ci"], "sealed", "type"),
        "### FLORES+, by token",
        "",
        *ci_table(payload["crossed_ci"], "flores", "token"),
        "### FLORES+, by type",
        "",
        *ci_table(payload["crossed_ci"], "flores", "type"),
        "JSON: `artifacts/analysis/case_markers.json`.",
        "",
    ]
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    version = tokenizers.__version__
    if version != "0.23.3":
        raise SystemExit(f"STOP: tokenizers {version}, expected 0.23.3")
    suffixes = load_suffixes()
    missing = markers_missing_from_inventory(suffixes)
    log(f"papaya missing {missing}")
    flores = [line for line in read_sentences("npi_Deva") if line.split()]
    if len(flores) != 1012:
        raise SystemExit(f"STOP: FLORES Nepali lines {len(flores)}")
    log("loading sealed Nepali")
    sealed = sealed_nepali()
    flo_raw = scan(flores)
    sea_raw = scan(sealed)
    stems: set[str] = set()
    for docs in (flo_raw, sea_raw):
        for hits in docs:
            for _word, stem, _marker in hits:
                stems.add(stem)
    log(f"candidate stems {len(stems)}")
    counts = stem_counts(stems)
    flo_docs = filter_docs(flo_raw, counts)
    sea_docs = filter_docs(sea_raw, counts)
    flo_cand_types, flo_cand_tokens = tally(flo_raw)
    sea_cand_types, sea_cand_tokens = tally(sea_raw)
    flo_kept_types, flo_kept_tokens = tally(flo_docs)
    sea_kept_types, sea_kept_tokens = tally(sea_docs)

    meta: dict[str, tuple[str, str, int]] = {}
    sealed_n: Counter[str] = Counter()
    flores_n: Counter[str] = Counter()
    for hits in sea_docs:
        for word, stem, marker in hits:
            meta[word] = (stem, marker, counts[stem])
            sealed_n[word] += 1
    for hits in flo_docs:
        for word, stem, marker in hits:
            meta[word] = (stem, marker, counts[stem])
            flores_n[word] += 1
    if len(meta) < 200:
        raise SystemExit(f"STOP: only {len(meta)} kept types")
    write_spotcheck(list(meta), meta, sealed_n, flores_n)

    markers_out: dict[str, dict[str, dict[str, int]]] = {"sealed": {}, "flores": {}}
    for marker in CASE_MARKERS:
        markers_out["sealed"][marker] = {
            "candidate_types": sea_cand_types[marker],
            "candidate_tokens": sea_cand_tokens[marker],
            "kept_types": sea_kept_types[marker],
            "kept_tokens": sea_kept_tokens[marker],
        }
        markers_out["flores"][marker] = {
            "candidate_types": flo_cand_types[marker],
            "candidate_tokens": flo_cand_tokens[marker],
            "kept_types": flo_kept_types[marker],
            "kept_tokens": flo_kept_tokens[marker],
        }
        log(
            f"{marker} sealed kept {sea_kept_types[marker]} types / "
            f"{sea_kept_tokens[marker]} tokens; "
            f"flores {flo_kept_types[marker]} / {flo_kept_tokens[marker]}"
        )

    words = sorted(meta)
    specs = panel_specs()
    by_name: dict[str, dict[str, str]] = {}
    for name in panel_names():
        log(f"classify {name}")
        enc = Encoder(specs[name])
        probe_word = "नेपालको"
        probe = word_spans(enc, probe_word)
        bad = probe is None or any(
            start < 0 or end > len(probe_word) or start >= end for start, end in probe
        )
        if bad:
            raise SystemExit(f"STOP: bad spans for {name}: {probe}")
        labels = {}
        for index, word in enumerate(words):
            labels[word] = classify_word(enc, word, meta[word][0])
            if index and index % 20000 == 0:
                log(f"  {index}/{len(words)}")
        by_name[name] = labels
        n_other = sum(1 for label in labels.values() if label == "other")
        log(f"  other types {n_other} / {len(words)}")

    scores: dict[str, Any] = {"sealed": {}, "flores": {}}
    pops: dict[str, dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]] = {
        "sealed": {},
        "flores": {},
    }
    corpus_docs = {"sealed": sea_docs, "flores": flo_docs}
    for corpus, docs in corpus_docs.items():
        for name in panel_names():
            hit, weight, type_labels, type_weights = population(docs, by_name[name])
            scores[corpus][name] = share_block(type_labels, type_weights)
            type_hit = np.asarray(
                [1.0 if label == "crossed" else 0.0 for label in type_labels], dtype=np.float64
            )
            type_w = np.ones(len(type_labels), dtype=np.float64)
            pops[corpus][name] = (hit, weight, type_hit, type_w)
            token_share = scores[corpus][name]["token"]["share"]["crossed"]
            type_share = scores[corpus][name]["type"]["share"]["crossed"]
            if weight.sum() == 0 or abs(token_share - float(hit.sum() / weight.sum())) > 1e-12:
                raise SystemExit(f"STOP: token share mismatch {corpus} {name}")
            if abs(type_share - float(type_hit.mean())) > 1e-12:
                raise SystemExit(f"STOP: type share mismatch {corpus} {name}")
            log(f"  {corpus} {name} crossed token share {token_share:.4f}")

    cis: dict[str, Any] = {}
    for corpus in ("sealed", "flores"):
        for kind, hit_i, weight_i in (("token", 0, 1), ("type", 2, 3)):
            for size in ("32k", "64k", "128k", "250k"):
                for ours in CLASS_OURS[size]:
                    for comp in CLASS_COMPETITORS[size]:
                        left_hit, left_w = pops[corpus][ours][hit_i], pops[corpus][ours][weight_i]
                        right_hit = pops[corpus][comp][hit_i]
                        # Occurrence weights do not depend on the tokenizer.
                        if not np.array_equal(left_w, pops[corpus][comp][weight_i]):
                            raise SystemExit(f"STOP: weight mismatch {ours} vs {comp}")
                        key = f"{corpus}|{kind}|{ours}|{comp}"
                        cis[key] = _rate_ci(left_hit, right_hit, left_w)
                        log(f"CI {corpus} {kind} {ours} - {comp} {fmt_ci(cis[key])}")

    manifest = json.loads(COUNTS_MANIFEST.read_text(encoding="utf-8"))
    payload: dict[str, Any] = {
        "kind": "case_marker_boundaries",
        "tokenizers_version": version,
        "seed": SEED,
        "bootstrap_n": BOOT_N,
        "stem_min_count": STEM_MIN_COUNT,
        "marker_list": list(CASE_MARKERS),
        "papaya_missing": missing,
        "affix_path": str(_affix_path().relative_to(REPO_ROOT)),
        "counts_sha256": manifest["counts_sha256"],
        "counts_unique": manifest["unique"],
        "spotcheck": "docs/case_marker_spotcheck.tsv",
        "spotcheck_n": 200,
        "spotcheck_note": (
            "Drawn after the frequency filter from kept types that occur in the sealed set. "
            "FLORES+-only types are omitted so this file does not re-host FLORES+ text. "
            "Not used to tune the marker list, the threshold, or the classifier."
        ),
        "kept_types": len(meta),
        "sealed_docs": len(sealed),
        "sealed_docs_with_kept": sum(1 for hits in sea_docs if hits),
        "sealed_kept_tokens": sum(sea_kept_tokens.values()),
        "flores_lines": len(flores),
        "flores_lines_with_kept": sum(1 for hits in flo_docs if hits),
        "flores_kept_tokens": sum(flo_kept_tokens.values()),
        "markers_by_corpus": markers_out,
        "scores": scores,
        "crossed_ci": cis,
        "panel": panel_names(),
    }
    # The markdown helper expects payload["markers"].
    payload["markers"] = markers_out
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_md(payload)
    log(f"wrote {OUT_JSON}")
    log(f"wrote {OUT_MD}")


if __name__ == "__main__":
    main()

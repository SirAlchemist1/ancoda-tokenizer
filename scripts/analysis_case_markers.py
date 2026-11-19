"""Nepali case-marker boundary metric. Does not train and does not touch the server.

Extends ancoda_tok.eval.morph without changing its existing morphology outputs.
The 200-word spot-check is written for a native speaker and is not an input.
"""

from __future__ import annotations

import io
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import tokenizers
import zstandard as zstd
from analysis_extra_per_sentence import himalaya_specs

from ancoda_tok.clean.common import (
    base_manifest,
    finish_manifest,
    save_json,
    sha256_bytes,
    sha256_file,
)
from ancoda_tok.counts import iter_entries
from ancoda_tok.eval.morph import (
    CASE_MARKERS,
    EXCLUSION_REASONS,
    STEM_MIN_COUNT,
    _affix_path,
    case_marker_exclusion,
    classify_marker_spans,
    load_suffixes,
    markers_missing_from_inventory,
    pretoken_surface,
    split_case_marker,
    word_spans_from_pieces,
)
from ancoda_tok.m8_final import BOOT_N, REPO_ROOT, SEED, Encoder, verdict
from ancoda_tok.m8_flores_per_sentence import flores_path, read_sentences
from ancoda_tok.m8_followup_report import load_specs

OUT_JSON = REPO_ROOT / "artifacts" / "analysis" / "case_markers.json"
OUT_MANIFEST = REPO_ROOT / "artifacts" / "analysis" / "case_markers.manifest.json"
LABEL_CACHE = REPO_ROOT / "artifacts" / "analysis" / "case_markers_labels"
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
COMPARE_64K = (
    "yak-64k-v1",
    "yak-64k-v2.0",
    "sidskarki-64k",
    "plain-64k",
    "sarvam-1",
    "himalaya-64k",
)
CLASS_64K = CLASS_OURS["64k"] + CLASS_COMPETITORS["64k"]
VARIANTS = ("unfiltered", "filtered", "literal")


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


def kept_meta(
    sea_docs: list[list[tuple[str, str, str]]],
    flo_docs: list[list[tuple[str, str, str]]],
    counts: dict[str, int],
) -> tuple[dict[str, tuple[str, str, int]], Counter[str], Counter[str]]:
    meta: dict[str, tuple[str, str, int]] = {}
    sealed_n: Counter[str] = Counter()
    flores_n: Counter[str] = Counter()
    for docs, tally_n in ((sea_docs, sealed_n), (flo_docs, flores_n)):
        for hits in docs:
            for word, stem, marker in hits:
                meta[word] = (stem, marker, counts[stem])
                tally_n[word] += 1
    return meta, sealed_n, flores_n


def exclusion_map(
    meta: dict[str, tuple[str, str, int]], *, independent_e: bool
) -> dict[str, str | None]:
    return {
        word: case_marker_exclusion(word, stem, marker, independent_e=independent_e)
        for word, (stem, marker, _count) in sorted(meta.items())
    }


def drop_excluded(
    docs: list[list[tuple[str, str, str]]], reasons: dict[str, str | None]
) -> list[list[tuple[str, str, str]]]:
    return [[hit for hit in hits if reasons[hit[0]] is None] for hits in docs]


def removal_report(
    docs: list[list[tuple[str, str, str]]], reasons: dict[str, str | None]
) -> dict[str, Any]:
    """What the exclusion filter removes from the frequency-kept words of one corpus."""
    kept_types: set[str] = set()
    kept_tokens = 0
    rem_types: dict[str, set[str]] = {reason: set() for reason in EXCLUSION_REASONS}
    rem_tokens = dict.fromkeys(EXCLUSION_REASONS, 0)
    removed_n: Counter[str] = Counter()
    for hits in docs:
        for word, _stem, _marker in hits:
            kept_types.add(word)
            kept_tokens += 1
            reason = reasons[word]
            if reason is not None:
                rem_types[reason].add(word)
                rem_tokens[reason] += 1
                removed_n[word] += 1
    ranked = sorted(removed_n.items(), key=lambda item: (-item[1], item[0]))
    top = ranked[:10]
    top_by_reason = {
        reason: [
            {"word": word, "count": count} for word, count in ranked if reasons[word] == reason
        ][:8]
        for reason in EXCLUSION_REASONS
    }
    removed_types = sum(len(words) for words in rem_types.values())
    removed_tokens = sum(rem_tokens.values())
    return {
        "kept_types_before": len(kept_types),
        "kept_tokens_before": kept_tokens,
        "removed_types": removed_types,
        "removed_tokens": removed_tokens,
        "removed_type_share": removed_types / len(kept_types) if kept_types else None,
        "removed_token_share": removed_tokens / kept_tokens if kept_tokens else None,
        "by_reason": {
            reason: {"types": len(rem_types[reason]), "tokens": rem_tokens[reason]}
            for reason in EXCLUSION_REASONS
        },
        "top_removed": [
            {"word": word, "count": count, "reason": reasons[word]} for word, count in top
        ],
        "top_by_reason": top_by_reason,
    }


def score_corpus(
    docs: list[list[tuple[str, str, str]]],
    by_name: dict[str, dict[str, str]],
) -> tuple[dict[str, Any], dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]]:
    scores: dict[str, Any] = {}
    pops: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
    for name in panel_names():
        hit, weight, type_labels, type_weights = population(docs, by_name[name])
        scores[name] = share_block(type_labels, type_weights)
        type_hit = np.asarray(
            [1.0 if label == "crossed" else 0.0 for label in type_labels], dtype=np.float64
        )
        type_w = np.ones(len(type_labels), dtype=np.float64)
        pops[name] = (hit, weight, type_hit, type_w)
        token_share = scores[name]["token"]["share"]["crossed"]
        type_share = scores[name]["type"]["share"]["crossed"]
        if weight.sum() == 0 or abs(token_share - float(hit.sum() / weight.sum())) > 1e-12:
            raise SystemExit(f"STOP: token share mismatch {name}")
        if abs(type_share - float(type_hit.mean())) > 1e-12:
            raise SystemExit(f"STOP: type share mismatch {name}")
    return scores, pops


def pair_ci(
    pops: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    kind: str,
    ours: str,
    comp: str,
) -> dict[str, Any]:
    hit_i, weight_i = (0, 1) if kind == "token" else (2, 3)
    left_hit, left_w = pops[ours][hit_i], pops[ours][weight_i]
    # Occurrence weights do not depend on the tokenizer.
    if not np.array_equal(left_w, pops[comp][weight_i]):
        raise SystemExit(f"STOP: weight mismatch {ours} vs {comp}")
    return _rate_ci(left_hit, pops[comp][hit_i], left_w)


def ordering(scores: dict[str, Any], kind: str) -> list[str]:
    return sorted(CLASS_64K, key=lambda name: (scores[name][kind]["share"]["crossed"], name))


def cached_labels(
    name: str, spec: str, words: list[str], stems: dict[str, str], words_sha: str
) -> dict[str, str]:
    """Labels for one tokenizer, reused from disk when the word list and spec match."""
    path = LABEL_CACHE / f"{name}.json"
    if path.is_file():
        cached = json.loads(path.read_text(encoding="utf-8"))
        if cached.get("words_sha256") == words_sha and cached.get("spec") == spec:
            log(f"classify {name}: cached")
            return dict(zip(words, cached["labels"], strict=True))
    log(f"classify {name}")
    enc = Encoder(spec)
    probe_word = "नेपालको"
    probe = word_spans(enc, probe_word)
    bad = probe is None or any(
        start < 0 or end > len(probe_word) or start >= end for start, end in probe
    )
    if bad:
        raise SystemExit(f"STOP: bad spans for {name}: {probe}")
    labels: list[str] = []
    for index, word in enumerate(words):
        labels.append(classify_word(enc, word, stems[word]))
        if index and index % 20000 == 0:
            log(f"  {index}/{len(words)}")
    n_other = sum(1 for label in labels if label == "other")
    log(f"  other types {n_other} / {len(words)}")
    save_json(path, {"words_sha256": words_sha, "spec": spec, "labels": labels})
    return dict(zip(words, labels, strict=True))


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
    cols = ("candidate_types", "candidate_tokens", "freq_types", "freq_tokens")
    cols += ("kept_types", "kept_tokens")
    lines = [
        (
            f"| Marker | {corpus} candidate types | candidate tokens | frequency-kept types | "
            "frequency-kept tokens | kept types | kept tokens |"
        ),
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    totals = dict.fromkeys(cols, 0)
    for marker in CASE_MARKERS:
        row = rows[corpus][marker]
        for col in cols:
            totals[col] += row[col]
        lines.append(f"| {marker} | " + " | ".join(str(row[col]) for col in cols) + " |")
    lines.append("| total | " + " | ".join(str(totals[col]) for col in cols) + " |")
    lines.append("")
    return lines


def share_table(scores: dict[str, Any]) -> list[str]:
    lines = [
        (
            "| Tokenizer | respected types | fused types | crossed types | other types | "
            "respected tokens | fused tokens | crossed tokens | other tokens |"
        ),
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in panel_names():
        block = scores[name]
        cells = []
        for kind in ("type", "token"):
            for label in LABELS:
                cells.append(fmt_share(block[kind]["share"][label]))
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    lines.append("")
    return lines


def ci_table(after: dict[str, Any], before: dict[str, Any], corpus: str, kind: str) -> list[str]:
    lines = [
        "| Ours | Competitor | Crossed share, ours − competitor | Before the filter |",
        "|---|---|---|---|",
    ]
    for size in ("32k", "64k", "128k", "250k"):
        for ours in CLASS_OURS[size]:
            for comp in CLASS_COMPETITORS[size]:
                key = f"{corpus}|{kind}|{ours}|{comp}"
                lines.append(f"| {ours} | {comp} | {fmt_ci(after[key])} | {fmt_ci(before[key])} |")
    lines.append("")
    return lines


REASON_TEXT = {
    "non_devanagari": "word has a character outside U+0900–U+097F",
    "single_unit": "stem is one letter unit",
    "e_sign_verb_form": "stem ends in े (U+0947), marker को/का/की/मा",
    "independent_e_verb_form": "stem ends in ए (U+090F), marker को/का/की/मा",
}


def removal_tables(payload: dict[str, Any]) -> list[str]:
    lines: list[str] = [
        (
            "| Corpus | Variant | kept types before | removed types | share | "
            "kept tokens before | removed tokens | share |"
        ),
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for corpus in ("sealed", "flores"):
        for variant in ("filtered", "literal"):
            rep = payload["removal"][variant][corpus]
            lines.append(
                f"| {corpus} | {variant} | {rep['kept_types_before']} | {rep['removed_types']} | "
                f"{rep['removed_type_share']:.4f} | {rep['kept_tokens_before']} | "
                f"{rep['removed_tokens']} | {rep['removed_token_share']:.4f} |"
            )
    lines += ["", "Removed by reason (main filter; first matching reason wins):", ""]
    lines += [
        "| Reason | sealed types | sealed tokens | FLORES+ types | FLORES+ tokens |",
        "|---|---:|---:|---:|---:|",
    ]
    for reason in EXCLUSION_REASONS:
        sea = payload["removal"]["filtered"]["sealed"]["by_reason"][reason]
        flo = payload["removal"]["filtered"]["flores"]["by_reason"][reason]
        lines.append(
            f"| {REASON_TEXT[reason]} | {sea['types']} | {sea['tokens']} | "
            f"{flo['types']} | {flo['tokens']} |"
        )
    lines.append("")
    for corpus, title in (("sealed", "Sealed"), ("flores", "FLORES+")):
        lines += [
            f"Ten most frequent removed words, {title}:",
            "",
            "| Word | occurrences | reason |",
            "|---|---:|---|",
        ]
        for row in payload["removal"]["filtered"][corpus]["top_removed"]:
            word = row["word"]
            if corpus == "flores" and payload["sealed_n_unfiltered"].get(word, 0) == 0:
                word = "(FLORES+-only word, withheld)"
            lines.append(f"| {word} | {row['count']} | {REASON_TEXT[row['reason']]} |")
        lines.append("")
    lines += [
        "Most frequent removed words per reason, sealed:",
        "",
        "| Reason | words (occurrences) |",
        "|---|---|",
    ]
    for reason in EXCLUSION_REASONS:
        top = payload["removal"]["filtered"]["sealed"]["top_by_reason"][reason]
        cells = ", ".join(f"{row['word']} ({row['count']})" for row in top)
        lines.append(f"| {REASON_TEXT[reason]} | {cells} |")
    lines += [
        "",
        (
            "The rules are applied as specified and are not tuned on these lists. Known "
            "costs: the one-unit rule also drops the case-marked pronouns मैले and मलाई "
            "(म + ले, म + लाई); the े + मा rule also drops the postposition बारेमा; and the "
            "non-Devanagari rule mostly drops case-marked words that carry an opening quote "
            "or bracket (‘हामीले), not verb forms."
        ),
        "",
    ]
    return lines


def compare_table(payload: dict[str, Any], corpus: str) -> list[str]:
    scores = payload["scores_by_variant"]
    cmp = payload["compare_64k"]
    lines = [
        (
            "| Competitor | crossed tokens, ancoda-64k / competitor, before | after | "
            "ours − competitor, before | after | U+0947-only filter |"
        ),
        "|---|---|---|---|---|---|",
    ]
    for comp in COMPARE_64K:
        cells = []
        for variant in ("unfiltered", "filtered"):
            ours = scores[variant][corpus]["ancoda-64k"]["token"]["share"]["crossed"]
            theirs = scores[variant][corpus][comp]["token"]["share"]["crossed"]
            cells.append(f"{ours:.4f} / {theirs:.4f}")
        for variant in VARIANTS:
            cells.append(fmt_ci(cmp[f"{variant}|{corpus}|token|ancoda-64k|{comp}"]))
        lines.append(f"| {comp} | " + " | ".join(cells) + " |")
    lines.append("")
    return lines


def ordering_lines(payload: dict[str, Any]) -> list[str]:
    lines = [
        "| Corpus | Share | before the filter (lowest crossed first) | after | changed |",
        "|---|---|---|---|---|",
    ]
    for corpus in ("sealed", "flores"):
        for kind in ("token", "type"):
            before = payload["ordering"]["unfiltered"][corpus][kind]
            after = payload["ordering"]["filtered"][corpus][kind]
            changed = "no" if before == after else "**yes**"
            lines.append(
                f"| {corpus} | {kind} | {', '.join(before)} | {', '.join(after)} | {changed} |"
            )
    lines.append("")
    changes = payload["verdict_changes"]
    if changes:
        lines.append("Verdict changes across all interval rows (before → after):")
        lines.append("")
        for row in changes:
            lines.append(f"- {row}")
    else:
        lines.append("No WIN/TIE/LOSE verdict changes in any interval row.")
    lines.append("")
    return lines


def write_md(payload: dict[str, Any]) -> None:
    missing = payload["papaya_missing"]
    missing_text = ", ".join(missing) if missing else "none"
    after = payload["scores_by_variant"]["filtered"]
    lines = [
        "# Nepali case-marker boundaries",
        "",
        (
            "Analysis only. No training run, and the server was not used. "
            "Reproduction: `scripts/analysis_case_markers.py` "
            f"(tokenizers {payload['tokenizers_version']}). "
            "The morphology scorer in `python/ancoda_tok/eval/morph.py` is unchanged; "
            "this metric and its exclusion filter are separate functions in that module."
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
            "A candidate is frequency-kept when its stem occurs as a standalone word at least "
            f"{STEM_MIN_COUNT} times in the n60j5 pre-token counts "
            "(`artifacts/counts/n60j5/pretoken_counts.bin`). A pre-token counts toward "
            "that stem when dropping its leading whitespace leaves exactly the stem. "
            f"Count-table sha256 `{payload['counts_sha256']}`."
        ),
        "",
        "## Verb-form and noise filter",
        "",
        (
            "A native-speaker check of the first spot-check found verb forms counted as "
            "case markers: past participles (stem ending in े or ए before को/का/की, "
            "e.g. खुलाएका, नआएको, भागेकी) and conditionals (stem ending in े before मा, "
            "e.g. घटेमा, बसेमा), plus a one-unit stem (बिकी) and a word with punctuation "
            "((बीमा). A frequency-kept candidate is now dropped when, checking in this order:"
        ),
        "",
        (
            "1. the word contains any code point outside the Devanagari block U+0900–U+097F "
            "(this includes ZWJ, ZWNJ, digits outside the block, and punctuation other than "
            "danda and double danda);"
        ),
        (
            "2. the stem is a single Ancoda letter unit (`ancoda_seg.aksharas`, "
            "DECISIONS.md: a consonant with its signs, or an independent vowel);"
        ),
        "3. the marker is को, का, की or मा and the stem ends in the vowel sign े (U+0947);",
        "4. the marker is को, का, की or मा and the stem ends in the independent vowel ए (U+090F).",
        "",
        (
            "Rule 4 is not in the written instruction, which named only U+0947, but two of "
            "its examples (खुलाएका, नआएको) end in ए (U+090F), and so do the most common "
            "participles (भएको, गएको). The **filtered** tables below use rules 1–4. "
            "The **literal** variant uses rules 1–3 only and is reported next to them. "
            "ले, लाई, बाट, सम्म, देखि, सँग and भन्दा after a participle stem are kept "
            "(e.g. लागिएकाले)."
        ),
        "",
        "### What the filter removes",
        "",
        *removal_tables(payload),
        "## Papaya inventory",
        "",
        (
            f"Affix file: `{payload['affix_path']}`. "
            "A marker is present when the affix column, with one leading hyphen removed, "
            "equals the marker. Missing markers: "
            f"**{missing_text}**. "
            "The missing marker stays in the metric. The inventory was not edited."
        ),
        "",
        "## Words kept",
        "",
        (
            "Candidate counts are before the frequency filter, frequency-kept counts after it, "
            "and kept counts after the verb-form and noise filter."
        ),
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
            f"`docs/case_marker_spotcheck.tsv` is {payload['spotcheck_n']} kept types "
            "(after the verb-form and noise filter) that occur in the sealed set, taken in "
            f"order from a `random.Random({SEED})` shuffle of the sorted kept types and then "
            "sorted by marker and word for reading. Types that occur only in FLORES+ are "
            "left out of this file. It replaces the first spot-check and is for a second "
            "native-speaker check."
        ),
        "",
        "## Ancoda-64k against the 64k panel, before and after the filter",
        "",
        (
            "Crossed token share (occurrence-weighted). Intervals as in the section below. "
            "The last column is the literal U+0947-only filter."
        ),
        "",
        "### Sealed",
        "",
        *compare_table(payload, "sealed"),
        "### FLORES+",
        "",
        *compare_table(payload, "flores"),
        "### Ordering of the 64k class by crossed share",
        "",
        *ordering_lines(payload),
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
            "himalaya-100k is paired with ancoda-128k, as in the fertility tables. "
            "Tables in this section are after the filter."
        ),
        "",
        "### Sealed Nepali",
        "",
        *share_table(after["sealed"]),
        "### FLORES+ Nepali",
        "",
        *share_table(after["flores"]),
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
            "Each comparison uses its own generator started at the same seed. "
            "The last column is the same interval on the frequency-kept words before the "
            "verb-form and noise filter."
        ),
        "",
        "### Sealed, by token",
        "",
        *ci_table(payload["crossed_ci"], payload["crossed_ci_unfiltered"], "sealed", "token"),
        "### Sealed, by type",
        "",
        *ci_table(payload["crossed_ci"], payload["crossed_ci_unfiltered"], "sealed", "type"),
        "### FLORES+, by token",
        "",
        *ci_table(payload["crossed_ci"], payload["crossed_ci_unfiltered"], "flores", "token"),
        "### FLORES+, by type",
        "",
        *ci_table(payload["crossed_ci"], payload["crossed_ci_unfiltered"], "flores", "type"),
        (
            "JSON: `artifacts/analysis/case_markers.json` "
            "(manifest `artifacts/analysis/case_markers.manifest.json`)."
        ),
        "",
    ]
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")


def tally_markers(
    raw: list[list[tuple[str, str, str]]],
    freq: list[list[tuple[str, str, str]]],
    kept: list[list[tuple[str, str, str]]],
) -> dict[str, dict[str, int]]:
    cand_types, cand_tokens = tally(raw)
    freq_types, freq_tokens = tally(freq)
    kept_types, kept_tokens = tally(kept)
    return {
        marker: {
            "candidate_types": cand_types[marker],
            "candidate_tokens": cand_tokens[marker],
            "freq_types": freq_types[marker],
            "freq_tokens": freq_tokens[marker],
            "kept_types": kept_types[marker],
            "kept_tokens": kept_tokens[marker],
        }
        for marker in CASE_MARKERS
    }


def full_cis(pops: dict[str, Any], label: str) -> dict[str, Any]:
    cis: dict[str, Any] = {}
    for corpus in ("sealed", "flores"):
        for kind in ("token", "type"):
            for size in ("32k", "64k", "128k", "250k"):
                for ours in CLASS_OURS[size]:
                    for comp in CLASS_COMPETITORS[size]:
                        key = f"{corpus}|{kind}|{ours}|{comp}"
                        cis[key] = pair_ci(pops[corpus], kind, ours, comp)
                        log(f"CI {label} {key} {fmt_ci(cis[key])}")
    return cis


def main() -> None:
    version = tokenizers.__version__
    if version != "0.23.3":
        raise SystemExit(f"STOP: tokenizers {version}, expected 0.23.3")
    manifest = base_manifest(kind="case_marker_boundaries")
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
    flo_freq = filter_docs(flo_raw, counts)
    sea_freq = filter_docs(sea_raw, counts)
    meta_all, sealed_all_n, _flores_all_n = kept_meta(sea_freq, flo_freq, counts)

    reasons = {
        "filtered": exclusion_map(meta_all, independent_e=True),
        "literal": exclusion_map(meta_all, independent_e=False),
    }
    variant_docs = {"unfiltered": {"sealed": sea_freq, "flores": flo_freq}}
    removal: dict[str, dict[str, Any]] = {}
    for variant, by_word in reasons.items():
        variant_docs[variant] = {
            "sealed": drop_excluded(sea_freq, by_word),
            "flores": drop_excluded(flo_freq, by_word),
        }
        removal[variant] = {
            "sealed": removal_report(sea_freq, by_word),
            "flores": removal_report(flo_freq, by_word),
        }
        for corpus in ("sealed", "flores"):
            rep = removal[variant][corpus]
            log(
                f"{variant} {corpus}: removed {rep['removed_types']}/{rep['kept_types_before']} "
                f"types, {rep['removed_tokens']}/{rep['kept_tokens_before']} tokens"
            )
    sea_docs = variant_docs["filtered"]["sealed"]
    flo_docs = variant_docs["filtered"]["flores"]

    meta, sealed_n, flores_n = kept_meta(sea_docs, flo_docs, counts)
    if len(meta) < 200:
        raise SystemExit(f"STOP: only {len(meta)} kept types")
    write_spotcheck(list(meta), meta, sealed_n, flores_n)

    markers_out = {
        "sealed": tally_markers(sea_raw, sea_freq, sea_docs),
        "flores": tally_markers(flo_raw, flo_freq, flo_docs),
    }

    words = sorted(meta_all)
    words_sha = sha256_bytes("\n".join(words).encode("utf-8"))
    word_stems = {word: meta_all[word][0] for word in words}
    specs = panel_specs()
    by_name = {
        name: cached_labels(name, specs[name], words, word_stems, words_sha)
        for name in panel_names()
    }

    scores_v: dict[str, dict[str, Any]] = {}
    pops_v: dict[str, dict[str, Any]] = {}
    for variant in VARIANTS:
        scores_v[variant] = {}
        pops_v[variant] = {}
        for corpus in ("sealed", "flores"):
            scores, pops = score_corpus(variant_docs[variant][corpus], by_name)
            scores_v[variant][corpus] = scores
            pops_v[variant][corpus] = pops
            for name in CLASS_64K:
                share = scores[name]["token"]["share"]["crossed"]
                log(f"  {variant} {corpus} {name} crossed token share {share:.4f}")

    cis = full_cis(pops_v["filtered"], "filtered")
    cis_before = full_cis(pops_v["unfiltered"], "unfiltered")
    compare: dict[str, Any] = {}
    for variant in VARIANTS:
        for corpus in ("sealed", "flores"):
            for kind in ("token", "type"):
                for comp in COMPARE_64K:
                    key = f"{variant}|{corpus}|{kind}|ancoda-64k|{comp}"
                    if variant == "filtered":
                        compare[key] = cis[f"{corpus}|{kind}|ancoda-64k|{comp}"]
                    elif variant == "unfiltered":
                        compare[key] = cis_before[f"{corpus}|{kind}|ancoda-64k|{comp}"]
                    else:
                        compare[key] = pair_ci(pops_v[variant][corpus], kind, "ancoda-64k", comp)
                    log(f"CI {key} {fmt_ci(compare[key])}")

    orders = {
        variant: {
            corpus: {kind: ordering(scores_v[variant][corpus], kind) for kind in ("token", "type")}
            for corpus in ("sealed", "flores")
        }
        for variant in VARIANTS
    }
    verdict_changes = [
        f"{key.replace('|', ' ')}: {cis_before[key]['verdict']} → {cis[key]['verdict']}"
        for key in cis
        if cis[key]["verdict"] != cis_before[key]["verdict"]
    ]

    counts_manifest = json.loads(COUNTS_MANIFEST.read_text(encoding="utf-8"))
    payload: dict[str, Any] = {
        "kind": "case_marker_boundaries",
        "tokenizers_version": version,
        "seed": SEED,
        "bootstrap_n": BOOT_N,
        "stem_min_count": STEM_MIN_COUNT,
        "marker_list": list(CASE_MARKERS),
        "exclusion_reasons": list(EXCLUSION_REASONS),
        "papaya_missing": missing,
        "affix_path": str(_affix_path().relative_to(REPO_ROOT)),
        "counts_sha256": counts_manifest["counts_sha256"],
        "counts_unique": counts_manifest["unique"],
        "spotcheck": "docs/case_marker_spotcheck.tsv",
        "spotcheck_n": 200,
        "spotcheck_note": (
            "Drawn after the frequency filter and the verb-form and noise filter, from kept "
            "types that occur in the sealed set. FLORES+-only types are omitted so this file "
            "does not re-host FLORES+ text."
        ),
        "kept_types": len(meta),
        "kept_types_unfiltered": len(meta_all),
        "sealed_docs": len(sealed),
        "sealed_docs_with_kept": sum(1 for hits in sea_docs if hits),
        "sealed_kept_tokens": sum(len(hits) for hits in sea_docs),
        "flores_lines": len(flores),
        "flores_lines_with_kept": sum(1 for hits in flo_docs if hits),
        "flores_kept_tokens": sum(len(hits) for hits in flo_docs),
        "markers": markers_out,
        "removal": removal,
        "sealed_n_unfiltered": dict(sorted(sealed_all_n.items())),
        "scores_by_variant": scores_v,
        "crossed_ci": cis,
        "crossed_ci_unfiltered": cis_before,
        "compare_64k": compare,
        "ordering": orders,
        "verdict_changes": verdict_changes,
        "panel": panel_names(),
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    write_md(payload)
    tok_files = {
        name: sha256_file(Path(spec)) for name, spec in specs.items() if Path(spec).is_file()
    }
    manifest.update(
        input_sha256={
            "sealed": sha256_file(SEALED),
            "flores_npi": sha256_file(flores_path("npi_Deva")),
            "pretoken_counts": sha256_file(COUNTS),
            "affix_inventory": sha256_file(_affix_path()),
            "tokenizers": tok_files,
        },
        tokenizer_specs_without_file=sorted(n for n in specs if n not in tok_files),
        outputs={
            "json": sha256_file(OUT_JSON),
            "markdown": sha256_file(OUT_MD),
            "spotcheck": sha256_file(SPOT),
        },
    )
    save_json(OUT_MANIFEST, finish_manifest(manifest))
    log(f"wrote {OUT_JSON}")
    log(f"wrote {OUT_MD}")
    log(f"wrote {OUT_MANIFEST}")


if __name__ == "__main__":
    main()

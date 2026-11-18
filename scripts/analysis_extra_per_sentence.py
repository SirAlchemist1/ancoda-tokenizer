"""Measure-only per-sentence rows for o200k and HimalayaAI.

Same fertility definition as artifacts/eval/m8/followup/flores_per_sentence.json:
each sentence or document is encoded alone, fertility = tokens / whitespace words,
paired bootstrap 10,000 resamples, seed 20261104.

HimalayaAI files stay in the Hugging Face cache. This script does not copy them
into the repo and does not redistribute them. The Hub cards state no licence.
o200k is tiktoken o200k_base (MIT). Does not touch the server.
"""

from __future__ import annotations

import io
import json
import unicodedata
from typing import Any

import ancoda_seg
import tokenizers
import zstandard as zstd

from ancoda_tok.m8_final import (
    BOOT_N,
    REPO_ROOT,
    SEED,
    Encoder,
    _joiner_text,
    verdict,
)
from ancoda_tok.m8_flores_per_sentence import (
    DEVA7,
    LANGS,
    bootstrap_deva_mean,
    corpus_fert,
    fmt_cell,
    fmt_fert,
    per_sentence_pairs,
    read_sentences,
)
from ancoda_tok.m8_followup_report import bootstrap_fert_diff, load_specs

OUT_JSON = REPO_ROOT / "artifacts" / "analysis" / "extra_per_sentence.json"
MD_OUT = REPO_ROOT / "docs" / "RESULTS_PER_SENTENCE.md"
SEALED = REPO_ROOT / "data" / "m9" / "eval" / "sealed.jsonl.zst"
SECTION = "## Additional measure-only tokenizers"
HF = REPO_ROOT / "data" / "hf_cache" / "hub"

# Revisions are the ones recorded in docs/LICENSES.md. Paths stay in the cache.
def _himalaya(repo: str, revision: str, filename: str) -> tuple[str, str]:
    rel = f"{repo}/snapshots/{revision}/{filename}"
    return rel, revision


HIMALAYA: dict[str, tuple[str, str]] = {
    "himalaya-64k": _himalaya(
        "models--himalaya-ai--nepali-tokenizer-bpe-64k-5b",
        "1c39f7a61b78211e3c5bce64e29b5ffd10b438d5",
        "himalaya-ai_nepali-tokenizer-corpus-64000.model",
    ),
    "himalaya-100k": _himalaya(
        "models--himalaya-ai--nepali-tokenizer-bpe-100k-5b",
        "83beb6d9e7a9af6f8449399aa5e16b438f7cf396",
        "himalaya-ai_nepali-tokenizer-corpus-100000.model",
    ),
    "himalaya-128k": _himalaya(
        "models--himalaya-ai--nepali-tokenizer-bpe-128k-5b",
        "c525ebe0b8f72705e549e6e62d36fb1b6a495df3",
        "himalaya-ai_nepali-tokenizer-corpus-128000.model",
    ),
}

# (class heading, ours, competitor). 100k is scored against ancoda-128k.
PAIRS = (
    ("64k", "ancoda-64k", "himalaya-64k"),
    ("128k", "ancoda-128k", "himalaya-100k"),
    ("128k", "ancoda-128k", "himalaya-128k"),
    ("250k", "ancoda-250k", "o200k"),
)
OURS_NEEDED = ("ancoda-64k", "ancoda-128k", "ancoda-250k")
NEW_MODELS = ("himalaya-64k", "himalaya-100k", "himalaya-128k", "o200k")
# Reported per-sentence FLORES+ Nepali fertilities. Confirmed before new rows.
BASELINE_NPI = {"ancoda-64k": "1.4508", "yak-64k-v1": "1.4566"}


def log(msg: str) -> None:
    print(msg, flush=True)


def confirm_baselines(specs: dict[str, str]) -> dict[str, str]:
    """Re-encode the two published rows before any new row is written."""
    lines = read_sentences("npi_Deva")
    shown: dict[str, str] = {}
    version = tokenizers.__version__
    if version != "0.23.3":
        raise SystemExit(f"STOP: tokenizers {version}, expected 0.23.3")
    for name, want in BASELINE_NPI.items():
        if name not in specs:
            raise SystemExit(f"STOP: missing baseline tokenizer {name}")
        got = corpus_fert(per_sentence_pairs(Encoder(specs[name]), lines))
        text = f"{got:.4f}"
        shown[name] = text
        log(f"baseline {name} FLORES+ Nepali {text} raw={got:.10f} tokenizers={version}")
        if text != want:
            raise SystemExit(f"STOP: {name} FLORES+ Nepali {text} != {want}")
    log("baselines match; new rows may be added")
    return shown


def himalaya_specs() -> dict[str, str]:
    specs: dict[str, str] = {}
    for name, (rel, revision) in HIMALAYA.items():
        path = HF / rel
        if not path.is_file():
            raise SystemExit(f"STOP: missing cached Himalaya file {path}")
        if revision not in path.parts:
            raise SystemExit(f"STOP: {name} is not at the LICENSES.md revision")
        specs[name] = str(path)
    return specs


def sealed_nepali() -> list[str]:
    texts: list[str] = []
    dctx = zstd.ZstdDecompressor()
    with SEALED.open("rb") as raw, dctx.stream_reader(raw) as reader:
        for line in io.TextIOWrapper(reader, encoding="utf-8"):
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("lang") == "Nepali" and row["text"].split():
                texts.append(row["text"])
    if len(texts) != 28094:
        raise SystemExit(f"STOP: sealed Nepali docs {len(texts)}, expected 28094")
    return texts


def decode(enc: Encoder, ids: list[int]) -> str:
    if enc.kind == "tiktoken":
        return str(enc.tt.decode(ids))
    if enc.kind == "sentencepiece":
        return str(enc.sp.decode(ids))
    return str(enc.tok.decode(ids, skip_special_tokens=False))


def round_trip(enc: Encoder, texts: list[str]) -> dict[str, Any]:
    ok = 0
    for text in texts:
        if decode(enc, enc.ids(text)) == text:
            ok += 1
    n = len(texts)
    return {"ok": ok, "n": n, "rate": (ok / n) if n else None}


def is_mark_initial(piece: bytes) -> bool:
    if not piece:
        return False
    try:
        ch = piece.decode("utf-8")[0]
    except UnicodeDecodeError:
        return False
    return unicodedata.category(ch) in ("Mn", "Mc", "Me")


def mark_initial_share(enc: Encoder, lines: list[str]) -> dict[str, Any]:
    orphan = 0
    total = 0
    uncovered = 0
    for line in lines:
        if not line.split():
            continue
        pieces = enc.pieces(line)
        ids = enc.ids(line)
        total += len(ids)
        if pieces is None:
            uncovered += len(ids)
            continue
        orphan += sum(1 for piece in pieces if is_mark_initial(piece))
    return {
        "orphan_tokens": orphan,
        "total_tokens": total,
        "uncovered_tokens": uncovered,
        "share": (orphan / total) if total else None,
        "method": "per-sentence",
    }


def joiner_ratio(enc: Encoder, original: list[str], rewritten: list[str]) -> dict[str, Any]:
    po = [(len(enc.ids(line)), len(line.split())) for line in original]
    pr = [(len(enc.ids(line)), len(line.split())) for line in rewritten]
    den = sum(t for t, _ in po)
    num = sum(t for t, _ in pr)
    return {
        "ratio": (num / den) if den else None,
        "tokens_original": den,
        "tokens_rewritten": num,
        "method": "per-sentence",
    }


def upsert_row(lines: list[str], heading: str, key: str, row: str) -> None:
    """Insert or replace a table row inside the section that starts at heading."""
    start = next(i for i, line in enumerate(lines) if line == heading)
    end = len(lines)
    level = len(heading) - len(heading.lstrip("#"))
    for i in range(start + 1, len(lines)):
        if lines[i].startswith("#"):
            other = len(lines[i]) - len(lines[i].lstrip("#"))
            if other <= level:
                end = i
                break
    for i in range(start, end):
        if lines[i].startswith(key):
            lines[i] = row
            return
    last = None
    for i in range(start, end):
        if lines[i].startswith("|") and not lines[i].startswith("|---") and " | " in lines[i]:
            # skip header rows whose second cell is not numeric-looking; keep last body or header
            last = i
    if last is None:
        raise SystemExit(f"STOP: no table under {heading}")
    # If the last '|' line is the separator, insert after the header pair's last body.
    lines.insert(last + 1, row)


def ensure_100k_section(lines: list[str], row: str) -> None:
    heading = "## Absolute fertilities — 100k"
    if heading in lines:
        upsert_row(lines, heading, "| himalaya-100k |", row)
        return
    anchor = "## Absolute fertilities — 128k"
    at = lines.index(anchor)
    block = [
        heading,
        "",
        "himalaya-100k is the 100k SentencePiece model. Its paired comparison below is against ancoda-128k.",
        "",
        "| Model | npi | hin | eng | mai | bho | awa | mag | san | mar | hne | Deva mean |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        row,
        "",
    ]
    lines[at:at] = block


def render_extra(payload: dict[str, Any]) -> list[str]:
    fert = payload["fertility"]
    intro = (
        "Measured with the same per-sentence harness as the tables above "
        f"({BOOT_N} resamples, seed {SEED}, tokenizers {payload['tokenizers_version']}). "
        "o200k is tiktoken `o200k_base` (MIT). "
        "HimalayaAI `himalaya-ai/nepali-tokenizer-bpe-{64k,100k,128k}-5b` "
        "are SentencePiece files at the revisions in `docs/LICENSES.md`. "
        "Those Hub repos state no licence: measure only, files stay in the "
        "Hugging Face cache, and nothing from them is copied or redistributed. "
        "himalaya-64k is compared with ancoda-64k. himalaya-100k and himalaya-128k "
        "are compared with ancoda-128k. o200k is compared with ancoda-250k."
    )
    sealed_note = (
        "Each of the 28,094 Nepali documents in the no-joiner sealed set is encoded "
        "on its own. Fertility = total tokens / total whitespace-separated words. "
        "The interval is a paired document bootstrap of ours − competitor."
    )
    mark_note = (
        "FLORES+ Nepali devtest, each sentence encoded alone. "
        "Mark-initial share is tokens whose first decoded character is a combining mark "
        "(Mn, Mc, or Me), divided by all tokens. "
        "ZWJ and ZWNJ cost is tokens on the Yak joiner rewrite divided by tokens on the "
        "original sentences. The rewrite inserts U+200D or U+200C after every virama "
        "that precedes a Devanagari consonant."
    )
    trip_note = (
        "Rate of `decode(encode(x)) == x`, with no special tokens added. "
        "FLORES+ Nepali is the 1,012 devtest lines that contain a word. "
        "Sealed Nepali is the same 28,094 documents."
    )
    lines = [
        SECTION,
        "",
        intro,
        "",
        "### Sealed Nepali fertility",
        "",
        sealed_note,
        "",
        "| Model | Sealed Nepali |",
        "|---|---:|",
    ]
    for name in ("ancoda-64k", "himalaya-64k", "ancoda-128k", "himalaya-100k", "himalaya-128k", "ancoda-250k", "o200k"):
        lines.append(f"| {name} | {fmt_fert(fert[name]['sealed_npi'])} |")
    lines += [
        "",
        "| Ours | Competitor | Sealed Nepali |",
        "|---|---|---|",
    ]
    for cls, left, right in PAIRS:
        cell = payload["sealed_cells"][f"{left}|{right}"]
        lines.append(f"| {left} | {right} | {fmt_cell(cell)} |")
    lines += [
        "",
        "### Mark-initial share and ZWJ / ZWNJ cost",
        "",
        mark_note,
        "",
        "| Model | Mark-initial share | ZWJ cost | ZWNJ cost |",
        "|---|---:|---:|---:|",
    ]
    for name in NEW_MODELS:
        mark = payload["mark_initial"][name]["share"]
        zwj = payload["zwj_zwnj"][name]["zwj"]["ratio"]
        zwnj = payload["zwj_zwnj"][name]["zwnj"]["ratio"]
        lines.append(f"| {name} | {mark:.6f} | {zwj:.6f} | {zwnj:.6f} |")
    lines += [
        "",
        "### Round-trip",
        "",
        trip_note,
        "",
        "| Model | FLORES+ Nepali | Sealed Nepali |",
        "|---|---:|---:|",
    ]
    for name in (*OURS_NEEDED, *NEW_MODELS):
        flo = payload["round_trip"][name]["flores_npi"]
        sea = payload["round_trip"][name]["sealed_npi"]
        lines.append(
            f"| {name} | {flo['ok']}/{flo['n']} ({flo['rate']:.6f}) | "
            f"{sea['ok']}/{sea['n']} ({sea['rate']:.6f}) |"
        )
    lines.append("")
    return lines


def update_markdown(payload: dict[str, Any]) -> None:
    text = MD_OUT.read_text(encoding="utf-8")
    if SECTION in text:
        text = text[: text.index(SECTION)].rstrip() + "\n"
    lines = text.splitlines()
    fert = payload["fertility"]
    langs = list(LANGS)
    for name, cls in (
        ("himalaya-64k", "64k"),
        ("himalaya-128k", "128k"),
        ("o200k", "250k"),
    ):
        vals = " | ".join(fmt_fert(fert[name][code]) for code in langs)
        row = f"| {name} | {vals} | {fmt_fert(fert[name]['deva_mean'])} |"
        upsert_row(lines, f"## Absolute fertilities — {cls}", f"| {name} |", row)
    vals = " | ".join(fmt_fert(fert["himalaya-100k"][code]) for code in langs)
    ensure_100k_section(
        lines,
        f"| himalaya-100k | {vals} | {fmt_fert(fert['himalaya-100k']['deva_mean'])} |",
    )
    metrics = langs + ["deva_mean"]
    for cls, left, right in PAIRS:
        cells = []
        for metric in metrics:
            hit = payload["flores_cells"][f"{left}|{right}|{metric}"]
            cells.append(fmt_cell(hit))
        row = "| " + " | ".join([left, right, *cells]) + " |"
        upsert_row(lines, f"### {cls}", f"| {left} | {right} |", row)
    body = "\n".join(lines).rstrip() + "\n\n" + "\n".join(render_extra(payload))
    MD_OUT.write_text(body if body.endswith("\n") else body + "\n", encoding="utf-8")


def main() -> None:
    specs = load_specs()
    for name in OURS_NEEDED:
        if name not in specs:
            raise SystemExit(f"STOP: missing {name}")
    baselines = confirm_baselines(specs)
    specs.update(himalaya_specs())
    if specs.get("o200k") != "o200k":
        specs["o200k"] = "o200k"

    sentences = {code: read_sentences(stem) for code, stem in LANGS.items()}
    npi_lines = [line for line in sentences["npi"] if line.split()]
    if len(npi_lines) != 1012:
        raise SystemExit(f"STOP: FLORES Nepali lines {len(npi_lines)}")
    letters = frozenset(int(x) for x in ancoda_seg.devanagari_letters())
    zwj_lines = [_joiner_text(line, "\u200d", letters) for line in npi_lines]
    zwnj_lines = [_joiner_text(line, "\u200c", letters) for line in npi_lines]
    log("loading sealed Nepali")
    sealed = sealed_nepali()
    log(f"sealed Nepali docs={len(sealed)}")

    names = list(OURS_NEEDED) + list(NEW_MODELS)
    pairs: dict[str, dict[str, list[tuple[int, int]]]] = {}
    fertility: dict[str, dict[str, float]] = {}
    mark: dict[str, Any] = {}
    joiners: dict[str, Any] = {}
    trips: dict[str, Any] = {}
    for name in names:
        log(f"encode {name}")
        enc = Encoder(specs[name])
        pairs[name] = {}
        fertility[name] = {}
        for code, lines in sentences.items():
            got = per_sentence_pairs(enc, lines)
            pairs[name][code] = got
            fertility[name][code] = corpus_fert(got)
        fertility[name]["deva_mean"] = sum(fertility[name][c] for c in DEVA7) / len(DEVA7)
        sealed_pairs = per_sentence_pairs(enc, sealed)
        fertility[name]["sealed_npi"] = corpus_fert(sealed_pairs)
        pairs[name]["sealed_npi"] = sealed_pairs
        if name in NEW_MODELS:
            mark[name] = mark_initial_share(enc, npi_lines)
            joiners[name] = {
                "zwj": joiner_ratio(enc, npi_lines, zwj_lines),
                "zwnj": joiner_ratio(enc, npi_lines, zwnj_lines),
            }
        trips[name] = {
            "flores_npi": round_trip(enc, npi_lines),
            "sealed_npi": round_trip(enc, sealed),
        }
        log(
            f"  npi={fertility[name]['npi']:.4f} "
            f"deva={fertility[name]['deva_mean']:.4f} "
            f"sealed={fertility[name]['sealed_npi']:.4f}"
        )

    # Harness check against the published per-sentence table.
    published = json.loads(
        (REPO_ROOT / "artifacts/eval/m8/followup/flores_per_sentence.json").read_text(
            encoding="utf-8"
        )
    )
    for name in OURS_NEEDED:
        for code in (*LANGS, "deva_mean"):
            got = fertility[name][code]
            want = published["fertility"][name][code]
            if abs(got - want) >= 0.00015:
                raise SystemExit(
                    f"STOP: {name} {code} {got:.6f} != published {want:.6f}"
                )
    log("published FLORES fertilities match")

    flat: dict[str, list[tuple[int, int]]] = {}
    for name in names:
        for code in DEVA7:
            flat[f"{name}:{code}"] = pairs[name][code]

    flores_cells: dict[str, Any] = {}
    sealed_cells: dict[str, Any] = {}
    for _cls, left, right in PAIRS:
        log(f"CI {left} vs {right}")
        for code in LANGS:
            a, b = [], []
            for x, y in zip(pairs[left][code], pairs[right][code], strict=True):
                if x[1] > 0 and y[1] > 0:
                    a.append(x)
                    b.append(y)
            ci = bootstrap_fert_diff(a, b)
            ci["verdict"] = verdict(ci)
            flores_cells[f"{left}|{right}|{code}"] = ci
        ci_m = bootstrap_deva_mean(flat, left, right)
        flores_cells[f"{left}|{right}|deva_mean"] = ci_m
        ci_s = bootstrap_fert_diff(pairs[left]["sealed_npi"], pairs[right]["sealed_npi"])
        ci_s["verdict"] = verdict(ci_s)
        sealed_cells[f"{left}|{right}"] = ci_s

    payload: dict[str, Any] = {
        "kind": "extra_per_sentence",
        "tokenizers_version": tokenizers.__version__,
        "baseline_flores_npi": baselines,
        "seed": SEED,
        "bootstrap_n": BOOT_N,
        "himalaya_revisions": {name: rev for name, (_rel, rev) in HIMALAYA.items()},
        "o200k": "tiktoken o200k_base",
        "note": (
            "Measure only. HimalayaAI files were not copied out of the Hugging Face "
            "cache and are not redistributed. Hub repos state no licence."
        ),
        "fertility": fertility,
        "flores_cells": flores_cells,
        "sealed_cells": sealed_cells,
        "mark_initial": mark,
        "zwj_zwnj": joiners,
        "round_trip": trips,
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    log(f"wrote {OUT_JSON}")
    update_markdown(payload)
    log(f"wrote {MD_OUT}")


if __name__ == "__main__":
    main()

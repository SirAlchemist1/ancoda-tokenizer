"""M8 report-only: whitespace fertility, sealed-by-source, joiners, code/math KB.

No training, no config or tokenizer changes. Seed 20261104.
"""

from __future__ import annotations

import json
from collections import defaultdict
from typing import Any

import numpy as np
import pyarrow.parquet as pq

from ancoda_tok.frontier_common import held_out_code_math
from ancoda_tok.frontier_metrics import fw2_npi_test_texts
from ancoda_tok.m8_final import (
    BOOT_N,
    M8,
    SEED,
    SNAP_J5,
    TOK,
    Encoder,
    aya_texts,
    panel_paths,
    read_lines,
    verdict,
)
from ancoda_tok.m8_followup_report import (
    BOOT_N as _BN,
)
from ancoda_tok.m8_followup_report import (
    bootstrap_fert_diff,
    corpus_fert,
    recover_sealed,
)
from ancoda_tok.storage import REPO_ROOT

assert _BN == BOOT_N

OUT = M8 / "followup"
WS_NAMES = [
    "ancoda-64k",
    "ancoda-ne-64k",
    "yak-64k-v1",
    "yak-64k-v2.0",
    "sidskarki-64k",
]
SP_MARK = "\u2581"  # SentencePiece word-start
JOINERS = "\u200c\u200d"


def log(msg: str) -> None:
    print(msg, flush=True)


def load_specs() -> dict[str, str]:
    paths = panel_paths()
    chosen = json.loads((M8 / "chosen.json").read_text(encoding="utf-8"))
    specs = {
        "ancoda-32k": str(TOK / "ancoda-32k/tokenizer.json"),
        "ancoda-64k": str(TOK / "ancoda-64k/tokenizer.json"),
        "ancoda-128k": str(TOK / "ancoda-128k/tokenizer.json"),
        "ancoda-ne-64k": str(TOK / "ancoda-ne-64k/tokenizer.json"),
        "ancoda-250k": str(REPO_ROOT / chosen["dir"] / "tokenizer.json"),
    }
    for name, spec in paths.items():
        if spec != "skipped: license":
            specs[name] = spec
    return specs


def _piece_is_whitespace_only(piece: bytes) -> bool:
    try:
        text = piece.decode("utf-8")
    except UnicodeDecodeError:
        return False
    # SP ▁ counts as whitespace; also spaces/tabs/newlines.
    cleaned = text.replace(SP_MARK, " ")
    return len(cleaned) > 0 and cleaned.isspace()


def encode_ws_stats(enc: Encoder, text: str) -> dict[str, int]:
    words = len(text.split())
    ws_chars = sum(1 for ch in text if ch.isspace())
    newlines = text.count("\n") + text.count("\r")
    ids = enc.ids(text)
    n_all = len(ids)
    pieces = enc.pieces(text)
    if pieces is None:
        # Fallback: decode each id when possible.
        n_non_ws = n_all
        if enc.kind == "sentencepiece":
            n_non_ws = 0
            for tid in ids:
                piece = enc.sp.id_to_piece(int(tid))
                cleaned = piece.replace(SP_MARK, " ")
                if cleaned and not cleaned.isspace():
                    n_non_ws += 1
        elif enc.kind == "hf":
            n_non_ws = 0
            for tid in ids:
                tok = enc.tok.id_to_token(tid)
                # byte-fallback pieces may not be plain text; count as non-ws if any non-ws
                cleaned = tok.replace(SP_MARK, " ")
                if cleaned and not cleaned.isspace():
                    n_non_ws += 1
        else:
            n_non_ws = n_all
    else:
        n_non_ws = sum(1 for p in pieces if not _piece_is_whitespace_only(p))
    return {
        "tokens": n_all,
        "tokens_non_ws": n_non_ws,
        "words": words,
        "ws_chars": ws_chars,
        "newlines": newlines,
    }


def corpus_from_stats(rows: list[dict[str, int]], key: str) -> float | None:
    t = sum(r[key] for r in rows)
    w = sum(r["words"] for r in rows)
    return None if w == 0 else t / w


def bootstrap_diff_stats(
    left: list[dict[str, int]],
    right: list[dict[str, int]],
    key: str,
    n: int = BOOT_N,
    seed: int = SEED,
) -> dict[str, Any]:
    usable = [
        (a, b)
        for a, b in zip(left, right, strict=True)
        if a["words"] > 0 and b["words"] > 0
    ]
    if not usable:
        return {"diff": None, "lo": None, "hi": None, "n": 0, "verdict": "TIE"}
    lt = np.asarray([a[key] for a, _ in usable], dtype=np.int64)
    lw = np.asarray([a["words"] for a, _ in usable], dtype=np.int64)
    rt = np.asarray([b[key] for _, b in usable], dtype=np.int64)
    rw = np.asarray([b["words"] for _, b in usable], dtype=np.int64)
    point = float(lt.sum() / lw.sum() - rt.sum() / rw.sum())
    m = len(usable)
    rng = np.random.default_rng(seed)
    diffs = np.empty(n, dtype=np.float64)
    for i in range(n):
        idx = rng.integers(0, m, size=m)
        diffs[i] = lt[idx].sum() / lw[idx].sum() - rt[idx].sum() / rw[idx].sum()
    diffs.sort()
    lo = float(diffs[int(0.025 * (n - 1))])
    hi = float(diffs[int(0.975 * (n - 1))])
    out = {"diff": point, "lo": lo, "hi": hi, "n": n, "seed": seed}
    out["verdict"] = verdict(out)
    return out


def ws_corpus_payload(
    encoders: dict[str, Encoder],
    texts: list[str],
    names: list[str],
    *,
    line_mode: bool,
) -> dict[str, Any]:
    per: dict[str, list[dict[str, int]]] = {}
    for name in names:
        log(f"  encode {name} docs={len(texts)}")
        per[name] = [encode_ws_stats(encoders[name], t) for t in texts]
    fert = {}
    for name in names:
        rows = per[name]
        fert[name] = {
            "fertility_a": corpus_from_stats(rows, "tokens"),
            "fertility_b": corpus_from_stats(rows, "tokens_non_ws"),
            "tokens": sum(r["tokens"] for r in rows),
            "tokens_non_ws": sum(r["tokens_non_ws"] for r in rows),
            "words": sum(r["words"] for r in rows),
            "ws_chars_per_word": (
                sum(r["ws_chars"] for r in rows) / sum(r["words"] for r in rows)
                if sum(r["words"] for r in rows)
                else None
            ),
            "newlines_per_word": (
                sum(r["newlines"] for r in rows) / sum(r["words"] for r in rows)
                if sum(r["words"] for r in rows)
                else None
            ),
        }
    # corpus-level ws/newline rates are text-side, same for all tokenizers
    text_ws = {
        "ws_chars_per_word": fert[names[0]]["ws_chars_per_word"],
        "newlines_per_word": fert[names[0]]["newlines_per_word"],
        "n_units": len(texts),
        "unit": "line" if line_mode else "doc",
    }
    cis = []
    for left in ("ancoda-64k", "ancoda-ne-64k"):
        for right in ("sidskarki-64k", "yak-64k-v1"):
            for key, label in (("tokens", "a"), ("tokens_non_ws", "b")):
                ci = bootstrap_diff_stats(per[left], per[right], key)
                cis.append({"ours": left, "base": right, "metric": label, **ci})
    return {"fertility": fert, "text_ws": text_ws, "cis": cis}


def sealed_nepali_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [r for r in rows if r.get("correct_lang") == "Nepali" or r.get("lang") == "Nepali"]


def assign_correct(rows: list[dict[str, Any]]) -> None:
    for row in rows:
        if row["bucket"] in ("Code", "Math"):
            row["correct_lang"] = row["bucket"]
        elif row.get("stored_glotlid") in (
            "npi_Deva",
            "hin_Deva",
            "mar_Deva",
            "san_Deva",
            "eng_Latn",
        ):
            row["correct_lang"] = {
                "npi_Deva": "Nepali",
                "hin_Deva": "Hindi",
                "mar_Deva": "Marathi",
                "san_Deva": "Sanskrit",
                "eng_Latn": "English",
            }[row["stored_glotlid"]]
        elif row.get("stored_glotlid"):
            row["correct_lang"] = row["stored_glotlid"]
        else:
            row["correct_lang"] = row.get("lang") or row["bucket"]


def source_bucket(source: str) -> str:
    if source in ("indiccorp_npi", "sangraha_nep", "fw2_npi"):
        return source
    return f"other:{source}" if source else "other:unknown"


def joiner_overlap(sealed_ids: set[str], sealed_texts: dict[str, str]) -> dict[str, Any]:
    from ancoda_tok.clean.common import load_mix
    from ancoda_tok.clean.minhash_eval import _minhash_config, _patch_xxhash, _signature_helper

    out: dict[str, Any] = {}
    _patch_xxhash()
    mix = load_mix()
    helper = _signature_helper(_minhash_config(mix))
    n_buckets = int(mix["minhash"]["num_buckets"])

    # Sealed bands after stripping joiners.
    sealed_bands: list[dict[tuple[int, ...], set[str]]] = [defaultdict(set) for _ in range(n_buckets)]
    for doc_id, text in sealed_texts.items():
        stripped = text.replace("\u200c", "").replace("\u200d", "")
        shingles = helper.get_shingles(stripped)
        if shingles.size == 0:
            continue
        for bi, band in enumerate(helper.get_signature(shingles)):
            sealed_bands[bi][tuple(int(x) for x in band)].add(doc_id)

    for label, directory in (
        ("n60j5", REPO_ROOT / "data" / "clean" / "_synthetic_joiners_n60j5"),
        ("n60j10", REPO_ROOT / "data" / "clean" / "_synthetic_joiners_n60j10"),
    ):
        source_ids: list[str] = []
        texts: list[str] = []
        for shard in sorted(directory.glob("shard-*.parquet")):
            table = pq.read_table(shard, columns=["doc_id", "text"])
            for doc_id, text in zip(
                table.column(0).to_pylist(), table.column(1).to_pylist(), strict=True
            ):
                assert str(doc_id).startswith("synthetic\x1f")
                source_ids.append(str(doc_id)[len("synthetic\x1f") :])
                texts.append(text or "")
        source_set = set(source_ids)
        id_overlap = sorted(source_set & sealed_ids)
        # Near-dup: stripped synthetic vs sealed (stripped).
        collided: set[str] = set()
        collided_sources: set[str] = set()
        exact_stripped = 0
        sealed_stripped_text = {
            did: (txt or "").replace("\u200c", "").replace("\u200d", "")
            for did, txt in sealed_texts.items()
        }
        sealed_by_text: dict[str, list[str]] = defaultdict(list)
        for did, txt in sealed_stripped_text.items():
            if txt:
                sealed_by_text[txt].append(did)
        for src_id, syn in zip(source_ids, texts, strict=True):
            stripped = (syn or "").replace("\u200c", "").replace("\u200d", "")
            if stripped and stripped in sealed_by_text:
                exact_stripped += 1
                collided.update(sealed_by_text[stripped])
                collided_sources.add(src_id)
            shingles = helper.get_shingles(stripped)
            if shingles.size == 0:
                continue
            for bi, band in enumerate(helper.get_signature(shingles)):
                hits = sealed_bands[bi].get(tuple(int(x) for x in band))
                if hits:
                    collided.update(hits)
                    collided_sources.add(src_id)
        near_extra = sorted(collided - set(id_overlap))
        out[label] = {
            "synthetic_docs": len(source_ids),
            "unique_source_ids": len(source_set),
            "sealed_id_overlap": len(id_overlap),
            "sealed_id_overlap_ids_sample": id_overlap[:20],
            "exact_stripped_matches": exact_stripped,
            "minhash_or_exact_sealed_docs_hit": len(collided),
            "minhash_or_exact_beyond_id_overlap": len(near_extra),
            "near_extra_sample": near_extra[:20],
            "joiner_sources_that_hit_sealed": len(collided_sources),
        }
        log(
            f"joiner {label}: synth={len(source_ids)} id_overlap={len(id_overlap)} "
            f"near_hits={len(collided)} beyond_id={len(near_extra)}"
        )
    return out


def _tokens_per_kb_enc(enc: Encoder, texts: list[str]) -> float:
    total_tok = total_bytes = 0
    for text in texts:
        if not text:
            continue
        total_tok += len(enc.ids(text))
        total_bytes += len(text.encode("utf-8"))
    if total_bytes == 0:
        return 0.0
    return 1024.0 * total_tok / total_bytes


def code_math_kb(specs: dict[str, str]) -> dict[str, Any]:
    names = [
        "ancoda-32k",
        "ancoda-64k",
        "ancoda-128k",
        "ancoda-250k",
        "yak-32k-v1",
        "yak-64k-v1",
        "yak-64k-v2.0",
        "yak-128k-v1",
        "yak-128k-v2.0",
        "sidskarki-64k",
        "arkios",
        "indicbert-v2",
    ]
    code_texts, math_texts = held_out_code_math(SNAP_J5)
    code_b = sum(len(t.encode("utf-8")) for t in code_texts)
    math_b = sum(len(t.encode("utf-8")) for t in math_texts)
    rows = {}
    for name in names:
        enc = Encoder(specs[name])
        c = _tokens_per_kb_enc(enc, code_texts)
        m = _tokens_per_kb_enc(enc, math_texts)
        rows[name] = {"code_per_kb": c, "math_per_kb": m}
        log(f"kb {name}: code={c:.4f} math={m:.4f}")
    return {
        "snapshot": str(SNAP_J5.relative_to(REPO_ROOT)),
        "limit_bytes": 5_000_000,
        "code_docs": len(code_texts),
        "math_docs": len(math_texts),
        "code_bytes": code_b,
        "math_bytes": math_b,
        "rows": rows,
        "note": (
            "Whole-file tokens/KB on n60j5 Code/Math held-out "
            "(frontier_common.held_out_code_math). Uses Encoder (HF + SentencePiece)."
        ),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    specs = load_specs()
    encoders = {n: Encoder(specs[n]) for n in WS_NAMES}

    log("recovering sealed docs")
    rows = recover_sealed()
    assign_correct(rows)
    nep = [r for r in rows if r["correct_lang"] == "Nepali"]
    log(f"sealed Nepali n={len(nep)}")

    # --- 1 whitespace ---
    log("whitespace: sealed Nepali")
    ws = {
        "seed": SEED,
        "definition": (
            "(a) tokens/words; (b) count only tokens whose decoded piece has ≥1 non-whitespace "
            "character (SentencePiece ▁ treated as whitespace along with spaces/newlines/tabs); "
            "(c) whitespace characters and newlines per word on the raw text."
        ),
        "sets": {},
    }
    ws["sets"]["sealed_nepali"] = ws_corpus_payload(
        encoders, [r["text"] for r in nep], WS_NAMES, line_mode=False
    )
    log("whitespace: Aya npi")
    aya = aya_texts()
    ws["sets"]["aya_npi"] = ws_corpus_payload(encoders, aya, WS_NAMES, line_mode=False)
    log("whitespace: fw2_npi_test")
    fw2 = fw2_npi_test_texts()
    ws["sets"]["fw2_npi_test"] = ws_corpus_payload(encoders, fw2, WS_NAMES, line_mode=False)
    log("whitespace: FLORES npi_Deva")
    flores = read_lines("npi_Deva")
    ws["sets"]["flores_npi_Deva"] = ws_corpus_payload(
        encoders, flores, WS_NAMES, line_mode=True
    )
    (OUT / "whitespace_fertility.json").write_text(
        json.dumps(ws, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    log("wrote whitespace_fertility.json")

    # --- 2 by source ---
    log("sealed Nepali by source")
    by_src: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in nep:
        by_src[source_bucket(str(r.get("source") or ""))].append(r)
    # encode all 5 tokenizers once on nep order
    nep_texts = [r["text"] for r in nep]
    per_tok_pairs: dict[str, list[tuple[int, int]]] = {}
    for name in WS_NAMES:
        log(f"  source-encode {name}")
        enc = encoders[name]
        per_tok_pairs[name] = [
            (len(enc.ids(t)), len(t.split())) for t in nep_texts
        ]
    # index nep rows
    src_report = {"docs_total": len(nep), "sources": {}}
    for src, docs in sorted(by_src.items(), key=lambda kv: -len(kv[1])):
        idxs = [i for i, r in enumerate(nep) if source_bucket(str(r.get("source") or "")) == src]
        bytes_ = sum(int(nep[i]["text_bytes"]) for i in idxs)
        ferts = {}
        for name in WS_NAMES:
            subset = [per_tok_pairs[name][i] for i in idxs]
            ferts[name] = {
                "fertility": corpus_fert(subset),
                "tokens": sum(a for a, _ in subset),
                "words": sum(b for _, b in subset),
            }
        cis = []
        for left in ("ancoda-64k", "ancoda-ne-64k"):
            for right in ("sidskarki-64k", "yak-64k-v1"):
                ci = bootstrap_fert_diff(
                    [per_tok_pairs[left][i] for i in idxs],
                    [per_tok_pairs[right][i] for i in idxs],
                )
                cis.append({"ours": left, "base": right, **ci})
        src_report["sources"][src] = {
            "docs": len(idxs),
            "bytes": bytes_,
            "fertility": ferts,
            "cis": cis,
        }
        log(f"  {src}: docs={len(idxs)} bytes={bytes_}")
    (OUT / "sealed_nepali_by_source.json").write_text(
        json.dumps(src_report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # --- 3 training overlap (stated sources only; filled in summary) ---
    train_overlap = {
        "sidskarki-64k": {
            "stated_sources": [
                "CulturaX Nepali",
                "Sangraha verified Nepali (AI4Bharat)",
                "CC-100 Nepali",
                "Nepali books/documents (public)",
            ],
            "refs": [
                "https://huggingface.co/sidskarki/nepali-bpe-tokenizer",
                "https://siddhantskarki.com/case-studies/nepali-tokenizer",
            ],
            "sealed_sources_named": {
                "sangraha_nep": "yes (Sangraha)",
                "indiccorp_npi": "not stated",
                "fw2_npi": "not stated (CulturaX/CC-100 are related web crawls; FineWeb-2 not named)",
            },
            "note": "Card does not name FineWeb-2, IndicCorp, or our exact source ids.",
        },
        "yak-v1-v2": {
            "stated_sources": [
                "FineWeb-2",
                "Sangraha",
                "HPLT 2.0",
                "IndicCorp v2",
                "Wikipedia / Wikisource / Wiktionary",
                "Sentiment-Analysis-Nepali",
                "Nepali-Flow-Roman",
                "Nepali-Roman-Transliteration",
                "nepglish-banking-dataset",
                "local-states-nepal",
                "FineWeb-Edu",
                "DCLM baseline",
                "FineMath",
                "OpenWebMath",
                "github-code-clean",
                "The Stack (permissive)",
                "synthetic Ampixa text/romanization",
            ],
            "refs": [
                "https://huggingface.co/datasets/freaktopus/yak-corpus",
                "https://ampixa.com/blog/yak-an-open-lossless-nepali-first-tokenizer/",
            ],
            "sealed_sources_named": {
                "fw2_npi": "yes (FineWeb-2)",
                "sangraha_nep": "yes (Sangraha)",
                "indiccorp_npi": "yes (IndicCorp v2)",
            },
            "note": (
                "Tokenizer files are public; Ampixa states training text is not released as "
                "the tokenizer artifact. The yak-corpus dataset card lists the sources above."
            ),
        },
    }
    (OUT / "training_source_overlap.json").write_text(
        json.dumps(train_overlap, indent=2), encoding="utf-8"
    )

    # --- 4 joiner overlap ---
    log("joiner-source overlap")
    sealed_ids = {r["doc_id"] for r in rows}
    sealed_text_map = {r["doc_id"]: r["text"] for r in rows}
    join = joiner_overlap(sealed_ids, sealed_text_map)
    (OUT / "joiner_sealed_overlap.json").write_text(
        json.dumps(join, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # --- 5 code/math KB ---
    log("code/math held-out KB")
    kb = code_math_kb(specs)
    (OUT / "code_math_kb.json").write_text(json.dumps(kb, indent=2), encoding="utf-8")

    # --- 6 corrected sealed table already in sealed_relabel.json ---
    summary = {
        "seed": SEED,
        "whitespace": "whitespace_fertility.json",
        "by_source": "sealed_nepali_by_source.json",
        "training_overlap": "training_source_overlap.json",
        "joiner": "joiner_sealed_overlap.json",
        "code_math_kb": "code_math_kb.json",
        "corrected_sealed": "sealed_relabel.json",
        "note": "Sealed results are reported only; they do not drive tokenizer changes.",
    }
    (OUT / "ws_report_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log("done")


if __name__ == "__main__":
    main()

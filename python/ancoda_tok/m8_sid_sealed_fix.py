"""Report-only: sidskarki FLORES path audit + sealed joiner-source cleanup.

Frozen M8 tokenizers. Seed 20261104. No training / config / tokenizer changes.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import sentencepiece as spm

from ancoda_tok.frontier_metrics import fw2_npi_test_texts
from ancoda_tok.m8_final import (
    BOOT_N,
    CLASSES,
    M8,
    OURS,
    SEED,
    TOK,
    Encoder,
    aya_texts,
    panel_paths,
    read_lines,
    verdict,
)
from ancoda_tok.m8_followup_report import (
    bootstrap_fert_diff,
    corpus_fert,
    recover_sealed,
)
from ancoda_tok.storage import REPO_ROOT

OUT = M8 / "followup"
SID_MODEL = Path(panel_paths()["sidskarki-64k"])
DEVA11 = (
    "Hindi",
    "Marathi",
    "Nepali",
    "Sanskrit",
    "anp_Deva",
    "awa_Deva",
    "bho_Deva",
    "gom_Deva",
    "hne_Deva",
    "mai_Deva",
    "new_Deva",
)


def log(msg: str) -> None:
    print(msg, flush=True)


def sha256_hex(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def list_sha256(hashes: list[str]) -> str:
    return hashlib.sha256("\n".join(hashes).encode("utf-8")).hexdigest()


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


def sp_stats(sp: spm.SentencePieceProcessor, text: str) -> dict[str, Any]:
    ids = list(sp.encode(text, out_type=int))
    unk_id = sp.unk_id()
    bos_id = sp.bos_id()
    eos_id = sp.eos_id()
    pieces = [sp.id_to_piece(i) for i in ids]
    nl = sum(1 for p in pieces if p == "<0x0A>" or "\n" in p)
    bos = sum(1 for i in ids if bos_id >= 0 and i == bos_id)
    eos = sum(1 for i in ids if eos_id >= 0 and i == eos_id)
    unk = sum(1 for i in ids if i == unk_id)
    decoded = sp.decode(ids)
    return {
        "tokens": len(ids),
        "bos": bos,
        "eos": eos,
        "unk": unk,
        "newline_tokens": nl,
        "roundtrip_exact": decoded == text,
        "ids": ids,
        "pieces": pieces,
        "decoded": decoded,
    }


def sidskarki_path_audit() -> dict[str, Any]:
    assert SID_MODEL.suffix == ".model"
    assert not (SID_MODEL.parent / "tokenizer.json").is_file()
    sp = spm.SentencePieceProcessor(model_file=str(SID_MODEL))
    lines = read_lines("npi_Deva")
    joined = "\n".join(lines)
    words = len(joined.split())

    whole = sp_stats(sp, joined)
    per_rows = [sp_stats(sp, line) for line in lines]
    per_tokens = sum(r["tokens"] for r in per_rows)
    per_bos = sum(r["bos"] for r in per_rows)
    per_eos = sum(r["eos"] for r in per_rows)
    per_unk = sum(r["unk"] for r in per_rows)
    per_nl = sum(r["newline_tokens"] for r in per_rows)
    per_rt = sum(1 for r in per_rows if r["roundtrip_exact"]) / len(per_rows)

    # Also mirror M8 Encoder (SP .model) for confirmation
    enc = Encoder(str(SID_MODEL))
    enc_whole = len(enc.ids(joined))
    enc_per = sum(len(enc.ids(line)) for line in lines)

    # 3 differing line pairs: alone vs as second line after newline
    diffs = []
    for i in range(len(lines) - 1):
        a, b = lines[i], lines[i + 1]
        alone = sp.encode(b, out_type=str)
        after = sp.encode(a + "\n" + b, out_type=str)
        # tokens belonging to b in joined form ≈ after[len(encode(a)):] but boundary interacts;
        # show alone vs nl-prefix encode of b
        prefixed = sp.encode("\n" + b, out_type=str)
        if alone != prefixed:
            diffs.append(
                {
                    "line_index": i + 1,
                    "text": b,
                    "per_line_tokens": alone,
                    "whole_file_style_nl_prefix_tokens": prefixed,
                    "per_line_n": len(alone),
                    "nl_prefix_n": len(prefixed),
                }
            )
        if len(diffs) >= 3:
            break

    choice = {
        "path": "sentencepiece_.model_per_unit",
        "file": str(SID_MODEL.relative_to(REPO_ROOT)),
        "library": "sentencepiece.SentencePieceProcessor",
        "unit": "one encode() call per eval unit (FLORES line; Aya/fw2/sealed document)",
        "add_bos_eos": False,
        "why": (
            "Hub ships only nepali_bpe_64k.model (no tokenizer.json). The model defines "
            "no BOS/EOS. Whole-file FLORES join emits one <0x0A> byte-fallback token per "
            "newline and retokenizes the next line (fertility 1.4448); per-unit encode "
            "matches Yak fertility_segments / npi-per-line (1.3699), has the same 99.51% "
            "exact line round-trip as the Yak harness segment rate, and matches how sealed/"
            "Aya/fw2 already encode each document alone. Prefer per-unit: no special tokens "
            "added, closest to lossless, no cross-unit newline tokens."
        ),
    }

    return {
        "model_file": str(SID_MODEL.relative_to(REPO_ROOT)),
        "tokenizer_json_present": False,
        "vocab_size": sp.get_piece_size(),
        "bos_id": sp.bos_id(),
        "eos_id": sp.eos_id(),
        "unk_id": sp.unk_id(),
        "paths": {
            "whole_file_sentencepiece": {
                "file": str(SID_MODEL.relative_to(REPO_ROOT)),
                "library": "sentencepiece.SentencePieceProcessor.encode",
                "mode": "whole file: '\\n'.join(lines) then one encode()",
                "tokens": whole["tokens"],
                "words": words,
                "fertility": whole["tokens"] / words,
                "bos": whole["bos"],
                "eos": whole["eos"],
                "unk": whole["unk"],
                "newline_tokens": whole["newline_tokens"],
                "roundtrip_exact": whole["roundtrip_exact"],
                "m8_Encoder_ids_tokens": enc_whole,
                "matches_yak_fertility": abs(whole["tokens"] / words - 1.4447890487139725) < 1e-12,
            },
            "per_line_sentencepiece": {
                "file": str(SID_MODEL.relative_to(REPO_ROOT)),
                "library": "sentencepiece.SentencePieceProcessor.encode",
                "mode": "per line: encode(line) for each line, sum tokens",
                "tokens": per_tokens,
                "words": words,
                "fertility": per_tokens / words,
                "bos": per_bos,
                "eos": per_eos,
                "unk": per_unk,
                "newline_tokens": per_nl,
                "roundtrip_exact_rate": per_rt,
                "m8_Encoder_ids_tokens": enc_per,
                "matches_yak_fertility_segments": abs(per_tokens / words - 1.3699267418854606)
                < 1e-12,
            },
        },
        "prior_run_usage": {
            "m8_scoreboard_flores_npi": "whole_file_sentencepiece (1.4448)",
            "m8_misspell_absolute": "whole_file_sentencepiece on FLORES npi (orig 1.4448)",
            "whitespace_report_flores": "per_line_sentencepiece (1.3699)",
            "sealed_aya_fw2": (
                "per-unit document encode via Encoder → SentencePiece .model "
                "(same library/file; each doc alone, so internal newlines still emit "
                "<0x0A> inside a doc, but units are not joined)"
            ),
        },
        "example_line_diffs": diffs,
        "choice": choice,
    }


def fert_pairs(enc: Encoder, texts: list[str]) -> list[tuple[int, int]]:
    out = []
    for t in texts:
        w = len(t.split())
        if w == 0:
            out.append((0, 0))
        else:
            out.append((len(enc.ids(t)), w))
    return out


def bootstrap_mean_deva(
    per_tok: dict[str, list[tuple[int, int]]],
    rows: list[dict[str, Any]],
    left: str,
    right: str,
) -> dict[str, Any]:
    """Doc-level bootstrap of unweighted mean fertility over the 11 Deva groups.

    Each resample redraws documents within each language, recomputes 11 fertilities,
    then takes their unweighted mean; CI on (mean_left - mean_right).
    """
    lang_idxs: dict[str, list[int]] = {lang: [] for lang in DEVA11}
    for i, r in enumerate(rows):
        lang = r["correct_lang"]
        if lang in lang_idxs:
            lang_idxs[lang].append(i)
    # drop empty langs from mean
    active = [lang for lang in DEVA11 if lang_idxs[lang]]
    if not active:
        return {"diff": None, "lo": None, "hi": None, "n": 0, "verdict": "TIE", "langs": []}

    def mean_fert(name: str, idx_map: dict[str, list[int]]) -> float:
        vals = []
        for lang in active:
            subset = [per_tok[name][i] for i in idx_map[lang]]
            f = corpus_fert(subset)
            if f is not None:
                vals.append(f)
        return sum(vals) / len(vals)

    point = mean_fert(left, lang_idxs) - mean_fert(right, lang_idxs)
    rng = np.random.default_rng(SEED)
    diffs = np.empty(BOOT_N, dtype=np.float64)
    for b in range(BOOT_N):
        boot_map: dict[str, list[int]] = {}
        for lang in active:
            idxs = lang_idxs[lang]
            m = len(idxs)
            pick = rng.integers(0, m, size=m)
            boot_map[lang] = [idxs[j] for j in pick]
        diffs[b] = mean_fert(left, boot_map) - mean_fert(right, boot_map)
    diffs.sort()
    lo = float(diffs[int(0.025 * (BOOT_N - 1))])
    hi = float(diffs[int(0.975 * (BOOT_N - 1))])
    out = {
        "diff": float(point),
        "lo": lo,
        "hi": hi,
        "n": BOOT_N,
        "seed": SEED,
        "langs": active,
        "n_langs": len(active),
    }
    out["verdict"] = verdict(out)
    return out


def joiner_source_hashes() -> set[str]:
    hashes: set[str] = set()
    for name in ("_synthetic_joiners_n60j5", "_synthetic_joiners_n60j10"):
        directory = REPO_ROOT / "data" / "clean" / name
        for shard in sorted(directory.glob("shard-*.parquet")):
            table = pq.read_table(shard, columns=["doc_id"])
            for doc_id in table.column(0).to_pylist():
                src = str(doc_id)[len("synthetic\x1f") :]
                hashes.add(sha256_hex(src))
    return hashes


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    log("1. sidskarki path audit")
    audit = sidskarki_path_audit()
    (OUT / "sidskarki_flores_path_audit.json").write_text(
        json.dumps(
            {k: v for k, v in audit.items() if k != "example_line_diffs"}
            | {"example_line_diffs": audit["example_line_diffs"]},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    log(
        f"whole={audit['paths']['whole_file_sentencepiece']['fertility']:.4f} "
        f"per_line={audit['paths']['per_line_sentencepiece']['fertility']:.4f}"
    )

    # Recompute chosen path fertilities + CIs
    log("recompute sidskarki (per-unit SP) vs Ancoda on FLORES/Aya/fw2/sealed-Nepali")
    specs = {
        "ancoda-64k": str(TOK / "ancoda-64k/tokenizer.json"),
        "ancoda-ne-64k": str(TOK / "ancoda-ne-64k/tokenizer.json"),
        "sidskarki-64k": str(SID_MODEL),
    }
    encoders = {n: Encoder(s) for n, s in specs.items()}

    sets: dict[str, list[str]] = {
        "flores_npi_Deva": read_lines("npi_Deva"),
        "aya_npi": aya_texts(),
        "fw2_npi_test": fw2_npi_test_texts(),
    }
    log("recover sealed for Nepali + cleanup")
    rows = recover_sealed()
    assign_correct(rows)
    nep_all = [r for r in rows if r["correct_lang"] == "Nepali"]
    sets["sealed_nepali_full"] = [r["text"] for r in nep_all]

    recompute: dict[str, Any] = {"choice": audit["choice"], "sets": {}}
    for set_name, texts in sets.items():
        log(f"  encode {set_name} n={len(texts)}")
        pairs = {n: fert_pairs(encoders[n], texts) for n in specs}
        fert = {n: corpus_fert(pairs[n]) for n in specs}
        cis = []
        for left in ("ancoda-64k", "ancoda-ne-64k"):
            ci = bootstrap_fert_diff(pairs[left], pairs["sidskarki-64k"])
            cis.append({"ours": left, "base": "sidskarki-64k", **ci})
        recompute["sets"][set_name] = {
            "n_units": len(texts),
            "fertility": fert,
            "cis_vs_sidskarki": cis,
        }
        log(f"    fert { {k: round(v,4) if v else None for k,v in fert.items()} }")

    (OUT / "sidskarki_chosen_path_recompute.json").write_text(
        json.dumps(recompute, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # --- 2 sealed cleanup ---
    log("2. sealed cleanup")
    original = json.loads((REPO_ROOT / "docs" / "sealed_test_set.json").read_text(encoding="utf-8"))
    join_hashes = joiner_source_hashes()
    orig_list = list(original["doc_id_sha256"])
    drop = [h for h in orig_list if h in join_hashes]
    keep = [h for h in orig_list if h not in join_hashes]
    keep_sorted = sorted(keep)
    reduced = {
        **{k: v for k, v in original.items() if k not in ("doc_id_sha256", "n_docs", "list_sha256")},
        "n_docs": len(keep_sorted),
        "doc_id_sha256": keep_sorted,
        "list_sha256": list_sha256(keep_sorted),
        "parent_list_sha256": original["list_sha256"],
        "dropped_joiner_source_docs": len(drop),
        "dropped_sha256": sorted(drop),
        "note": (
            "Reduced sealed list: dropped docs whose doc_id was a synthetic joiner "
            "source in n60j5 or n60j10. Original docs/sealed_test_set.json unchanged."
        ),
        "seed": SEED,
    }
    reduced_path = REPO_ROOT / "docs" / "sealed_test_set_no_joiners.json"
    reduced_path.write_text(json.dumps(reduced, indent=2) + "\n", encoding="utf-8")
    # mirror under data/clean for tooling that reads there
    (REPO_ROOT / "data" / "clean" / "sealed_test_set_no_joiners.json").write_text(
        json.dumps(reduced, indent=2) + "\n", encoding="utf-8"
    )
    log(f"dropped {len(drop)} new_n={len(keep_sorted)} sha={reduced['list_sha256']}")

    drop_set = set(drop)
    rows_kept = [r for r in rows if sha256_hex(r["doc_id"]) not in drop_set]
    for r in rows_kept:
        # already has correct_lang
        pass
    log(f"kept sealed rows {len(rows_kept)}")

    # Encode all needed tokenizers once on kept rows
    need_names = [
        "ancoda-32k",
        "ancoda-64k",
        "ancoda-ne-64k",
        "ancoda-128k",
        "ancoda-250k",
        "sidskarki-64k",
        "yak-64k-v1",
        "yak-64k-v2.0",
        "yak-32k-v1",
        "yak-128k-v1",
        "yak-128k-v2.0",
        "arkios",
        "sarvam-1",
        "llama3",
        "sarvam-m",
        "qwen2.5",
        "qwen3",
        "indicbert-v2",
        "bloom",
        "gemma3",
        "sarvam-30b",
    ]
    paths = panel_paths()
    chosen = json.loads((M8 / "chosen.json").read_text(encoding="utf-8"))
    all_specs = {
        "ancoda-32k": str(TOK / "ancoda-32k/tokenizer.json"),
        "ancoda-64k": str(TOK / "ancoda-64k/tokenizer.json"),
        "ancoda-128k": str(TOK / "ancoda-128k/tokenizer.json"),
        "ancoda-ne-64k": str(TOK / "ancoda-ne-64k/tokenizer.json"),
        "ancoda-250k": str(REPO_ROOT / chosen["dir"] / "tokenizer.json"),
    }
    for n, s in paths.items():
        if s != "skipped: license" and n in need_names:
            all_specs[n] = s

    texts = [r["text"] for r in rows_kept]
    per_tok: dict[str, list[tuple[int, int]]] = {}
    for name in need_names:
        log(f"  sealed-encode {name}")
        enc = Encoder(all_specs[name])
        per_tok[name] = fert_pairs(enc, texts)

    def idxs_for(lang: str) -> list[int]:
        return [i for i, r in enumerate(rows_kept) if r["correct_lang"] == lang]

    nep_i = idxs_for("Nepali")
    headline = {"docs": len(nep_i), "dropped": len(drop), "fertility": {}, "cis": []}
    for name in (
        "ancoda-64k",
        "ancoda-ne-64k",
        "sidskarki-64k",
        "yak-64k-v1",
        "yak-64k-v2.0",
    ):
        subset = [per_tok[name][i] for i in nep_i]
        headline["fertility"][name] = corpus_fert(subset)
    for left in ("ancoda-64k", "ancoda-ne-64k"):
        for right in ("sidskarki-64k", "yak-64k-v1", "yak-64k-v2.0"):
            ci = bootstrap_fert_diff(
                [per_tok[left][i] for i in nep_i],
                [per_tok[right][i] for i in nep_i],
            )
            headline["cis"].append({"ours": left, "base": right, **ci})

    extra_cis = []
    for left, right, lang in (
        ("ancoda-250k", "indicbert-v2", "Hindi"),
        ("ancoda-250k", "indicbert-v2", "Marathi"),
        ("ancoda-250k", "indicbert-v2", "awa_Deva"),
        ("ancoda-128k", "yak-128k-v1", "Nepali"),
        ("ancoda-64k", "yak-64k-v2.0", "bho_Deva"),
        ("ancoda-64k", "yak-64k-v2.0", "mai_Deva"),
        ("ancoda-64k", "yak-64k-v2.0", "new_Deva"),
    ):
        ix = idxs_for(lang)
        ci = bootstrap_fert_diff(
            [per_tok[left][i] for i in ix],
            [per_tok[right][i] for i in ix],
        )
        extra_cis.append(
            {
                "lang": lang,
                "docs": len(ix),
                "ours": left,
                "base": right,
                "fertility_ours": corpus_fert([per_tok[left][i] for i in ix]),
                "fertility_base": corpus_fert([per_tok[right][i] for i in ix]),
                **ci,
            }
        )
        log(f"  extra {left} vs {right} {lang}: {ci['verdict']} {ci['diff']:+.4f}")

    log("deva11 mean CIs")
    deva_mean_cis = []
    for cls, members in CLASSES.items():
        ours = [n for n in members if n in OURS and n in per_tok]
        panel = [n for n in members if n not in OURS and n in per_tok]
        for left in ours:
            for right in panel:
                ci = bootstrap_mean_deva(per_tok, rows_kept, left, right)
                deva_mean_cis.append({"class": cls, "ours": left, "base": right, **ci})
                log(
                    f"  mean11 {left} vs {right}: {ci['verdict']} "
                    f"{ci['diff']:+.4f} [{ci['lo']:+.4f},{ci['hi']:+.4f}]"
                )

    # per-lang fertilities on reduced set for GATE_A table
    by_lang: dict[str, Any] = {}
    langs = sorted({r["correct_lang"] for r in rows_kept}, key=lambda L: (-sum(1 for r in rows_kept if r["correct_lang"]==L), L))
    for lang in langs:
        ix = idxs_for(lang)
        ferts = {}
        for name in need_names:
            ferts[name] = corpus_fert([per_tok[name][i] for i in ix])
        by_lang[lang] = {"docs": len(ix), "fertility": ferts}

    payload = {
        "seed": SEED,
        "reduced_list": str(reduced_path.relative_to(REPO_ROOT)),
        "artifact": "sealed_nojjoiners_remeasure.json",
        "n_docs_original": original["n_docs"],
        "n_docs_reduced": reduced["n_docs"],
        "dropped": len(drop),
        "list_sha256_original": original["list_sha256"],
        "list_sha256_reduced": reduced["list_sha256"],
        "sealed_nepali_headline": headline,
        "extra_cis": extra_cis,
        "deva11_mean_cis": deva_mean_cis,
        "by_lang": by_lang,
        "note": "Sealed results reported only; do not drive tokenizer changes.",
    }
    (OUT / "sealed_nojjoiners_remeasure.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    log("done")


if __name__ == "__main__":
    main()

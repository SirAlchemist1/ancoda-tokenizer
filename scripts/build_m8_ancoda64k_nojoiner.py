"""Train matched M8 Ancoda-64k with joiner copy removed. Do not queue or train models.

Uses the same budget build as m8/ancoda-64k (B_d=48000, covered_64k_3pct,
code_math_scale=1.0, seed 20261104) on the n60j5_nojoiner snapshot.
Writes reports under artifacts/analysis/ and the tokenizer under
artifacts/tokenizers/m8/ancoda-64k-nojoiner/.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
from pathlib import Path

import zstandard as zstd

from ancoda_tok.counts import iter_entries, write_counts
from ancoda_tok.export_hf import bytes_to_unicode, load_hf
from ancoda_tok.frontier_common import is_deva_bytes
from ancoda_tok.clean.common import free_bytes, save_json, sha256_file
from ancoda_tok.m8_final import (
    REPO_ROOT,
    Encoder,
    _joiner_text,
    _ratio_ci,
    count_snapshot_buckets,
    train_budget,
)
from ancoda_tok.m8_flores_per_sentence import (
    LANGS,
    corpus_fert,
    flores_path,
    per_sentence_pairs,
)
from ancoda_tok.m8_followup_report import bootstrap_fert_diff
import ancoda_seg

OUT_DIR = REPO_ROOT / "artifacts" / "tokenizers" / "m8" / "ancoda-64k-nojoiner"
COUNTS = REPO_ROOT / "artifacts" / "eval" / "m8" / "n60j5_nojoiner_counts"
REPORT = REPO_ROOT / "artifacts" / "analysis" / "ancoda64k_nojoiner_report.json"
MD = REPO_ROOT / "docs" / "ANALYSIS_NOJOINER_ARM.md"
M8_REF = REPO_ROOT / "artifacts" / "tokenizers" / "m8" / "ancoda-64k"
STREAM = REPO_ROOT / "data" / "m9" / "text"


def find_snapshot() -> Path:
    snaps = sorted((REPO_ROOT / "data" / "snapshots").glob("n60j5_nojoiner_*"))
    if not snaps:
        raise SystemExit("STOP: n60j5_nojoiner snapshot missing; build it first")
    return snaps[0]


def joiner_vocab_count(tok_path: Path) -> dict:
    tok = load_hf(tok_path)
    inverse = {ch: b for b, ch in bytes_to_unicode().items()}
    zwj = zwnj = either = 0
    for tid in range(tok.get_vocab_size(with_added_tokens=True)):
        piece = tok.id_to_token(tid)
        try:
            text = bytes(inverse[c] for c in piece).decode("utf-8")
        except Exception:
            continue
        has_z = "\u200d" in text
        has_n = "\u200c" in text
        if has_z:
            zwj += 1
        if has_n:
            zwnj += 1
        if has_z or has_n:
            either += 1
    return {
        "entries_with_zwj": zwj,
        "entries_with_zwnj": zwnj,
        "entries_with_either": either,
    }


def roundtrip_stream(tok_path: Path, n: int = 1000) -> dict:
    tok = load_hf(tok_path)
    ok = 0
    total = 0
    shards = sorted(STREAM.glob("shard-*.jsonl.zst"))
    for shard in shards:
        dctx = zstd.ZstdDecompressor()
        with shard.open("rb") as raw, dctx.stream_reader(raw) as reader:
            for line in io.TextIOWrapper(reader, encoding="utf-8"):
                if not line.strip():
                    continue
                text = json.loads(line)["text"]
                ids = tok.encode(text, add_special_tokens=False).ids
                if tok.decode(ids, skip_special_tokens=False) == text:
                    ok += 1
                total += 1
                if total >= n:
                    return {"n": total, "ok": ok, "rate": ok / total}
    return {"n": total, "ok": ok, "rate": (ok / total) if total else None}


def tokens_per_byte_stream(tok_path: Path, max_docs: int = 10_000) -> dict:
    enc = Encoder(str(tok_path))
    tokens = 0
    nbytes = 0
    docs = 0
    for shard in sorted(STREAM.glob("shard-*.jsonl.zst")):
        dctx = zstd.ZstdDecompressor()
        with shard.open("rb") as raw, dctx.stream_reader(raw) as reader:
            for line in io.TextIOWrapper(reader, encoding="utf-8"):
                if not line.strip():
                    continue
                text = json.loads(line)["text"]
                tokens += len(enc.ids(text))
                nbytes += len(text.encode("utf-8"))
                docs += 1
                if docs >= max_docs:
                    return {
                        "docs": docs,
                        "tokens": tokens,
                        "bytes": nbytes,
                        "tokens_per_byte": tokens / nbytes if nbytes else None,
                        "note": f"first {max_docs} stream docs by shard order",
                    }
    return {
        "docs": docs,
        "tokens": tokens,
        "bytes": nbytes,
        "tokens_per_byte": tokens / nbytes if nbytes else None,
    }


def fert_compare(ref: Path, new: Path) -> dict:
    letters = frozenset(int(x) for x in ancoda_seg.devanagari_letters())
    out = {"langs": {}, "zwj": {}, "zwnj": {}, "bootstrap": {}}
    for code in ("npi", "hin", "eng"):
        lines = flores_path(LANGS[code]).read_text(encoding="utf-8").splitlines()
        a = per_sentence_pairs(Encoder(str(ref)), lines)
        b = per_sentence_pairs(Encoder(str(new)), lines)
        fa, fb = corpus_fert(a), corpus_fert(b)
        # bootstrap_fert_diff(left, right): left − right; WIN when left lower.
        ci = bootstrap_fert_diff(b, a)
        out["langs"][code] = {
            "m8_ancoda64k": fa,
            "nojoiner": fb,
            "diff_nojoiner_minus_m8": fb - fa,
            "bootstrap": ci,
            "verdict_nojoiner_vs_m8": ci.get("verdict")
            or ("WIN" if ci["hi"] < 0 else "LOSE" if ci["lo"] > 0 else "TIE"),
        }
        out["bootstrap"][code] = ci
    npi_lines = flores_path(LANGS["npi"]).read_text(encoding="utf-8").splitlines()
    for label, path in (("m8", ref), ("nojoiner", new)):
        enc = Encoder(str(path))
        out["zwj"][label] = _ratio_ci(
            enc,
            npi_lines,
            [_joiner_text(line, "\u200d", letters) for line in npi_lines],
        )
        out["zwnj"][label] = _ratio_ci(
            enc,
            npi_lines,
            [_joiner_text(line, "\u200c", letters) for line in npi_lines],
        )
    return out


def config_diff(snap: Path, dewa: Path, latin: Path) -> dict:
    m8_man = json.loads((M8_REF / "m8_manifest.json").read_text(encoding="utf-8"))
    return {
        "unchanged": {
            "vocab_size": 65536,
            "B_d": 48000,
            "covered": "artifacts/inventory/covered_64k_3pct.json",
            "code_math_scale": 1.0,
            "seed": 20261104,
            "slack": 4096,
            "block0": True,
            "block1": True,
            "bans": True,
            "gb9c": False,
            "plain": False,
        },
        "changed": {
            "snapshot": {
                "arm_A": m8_man["snapshot"],
                "arm_F": str(snap.relative_to(REPO_ROOT)),
                "joiner_share": {"arm_A": 0.01, "arm_F": 0.0},
                "note": "1% joiner copy replaced by natural Nepali (DECISIONS 3.8)",
            },
            "deva_counts": str(dewa.relative_to(REPO_ROOT)),
            "latin_counts": str(latin.relative_to(REPO_ROOT)),
        },
        "not_used": "configs/build_64k.yaml (points at n50; stale for this arm)",
        "ladder_final_nojoiner_not_matched": {
            "path": "artifacts/tokenizers/ladder-final_nojoiner",
            "reason": "n50_nojoiner snapshot; greets tokenize differently from m8/ancoda-64k",
        },
    }


def write_md(report: dict) -> None:
    f = report["fertility"]
    lines = [
        "# Matched no-joiner Ancoda-64k (arm F prep)",
        "",
        "Preparation only. Tokenizer trained; **not** queued for M9. "
        "Marked in `docs/M9_PREREG.md` as not yet registered.",
        "",
        "## Config diff vs m8/ancoda-64k",
        "",
        "```json",
        json.dumps(report["config_diff"], indent=2, sort_keys=True),
        "```",
        "",
        f"Joiner-containing vocab entries: **{report['joiner_vocab']['entries_with_either']}** "
        f"(zwj={report['joiner_vocab']['entries_with_zwj']}, "
        f"zwnj={report['joiner_vocab']['entries_with_zwnj']}).",
        "",
        "## FLORES+ per-sentence fertility",
        "",
        "| Lang | m8/ancoda-64k | nojoiner | Δ (nojoiner−m8) | Verdict |",
        "|---|---:|---:|---:|---|",
    ]
    for code in ("npi", "hin", "eng"):
        row = f["langs"][code]
        lines.append(
            f"| {code} | {row['m8_ancoda64k']:.6f} | {row['nojoiner']:.6f} | "
            f"{row['diff_nojoiner_minus_m8']:+.6f} | {row['verdict_nojoiner_vs_m8']} |"
        )
    lines += [
        "",
        "## ZWJ / ZWNJ cost (FLORES npi rewrite ÷ original)",
        "",
        f"- ZWJ m8={f['zwj']['m8']['ratio']:.6f}, nojoiner={f['zwj']['nojoiner']['ratio']:.6f}",
        f"- ZWNJ m8={f['zwnj']['m8']['ratio']:.6f}, nojoiner={f['zwnj']['nojoiner']['ratio']:.6f}",
        "",
        f"Round-trip 1,000 stream docs: {report['roundtrip']['ok']}/{report['roundtrip']['n']} "
        f"(rate {report['roundtrip']['rate']:.6f}).",
        "",
        f"Tokens/byte on M9 stream sample ({report['tokens_per_byte']['docs']} docs): "
        f"nojoiner={report['tokens_per_byte']['tokens_per_byte']:.6f}; "
        f"m8 reference on full stream was 0.150967.",
        "",
        f"Tokenizer: `{report['tokenizer_dir']}` sha256=`{report['tokenizer_sha256']}`.",
        "",
    ]
    MD.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    snap = find_snapshot()
    free = free_bytes()
    print(f"snapshot={snap} free_bytes={free}", flush=True)
    if free < 50_000_000_000:
        raise SystemExit(f"STOP: free_bytes={free} under 50 GB")

    count_snapshot_buckets(snap, COUNTS)
    dewa = COUNTS / "deva_pretoken_counts.bin"
    latin = COUNTS / "latin_cm100_pretoken_counts.bin"
    covered = REPO_ROOT / "artifacts" / "inventory" / "covered_64k_3pct.json"
    if not covered.is_file():
        raise SystemExit(f"STOP: missing {covered}")

    row = train_budget(
        name="ancoda-64k-nojoiner",
        vocab=65_536,
        b_d=48_000,
        scale=1.0,
        dest=OUT_DIR,
        dewa_counts=dewa,
        latin_path=latin,
        covered=covered,
    )
    print(f"train {row}", flush=True)
    if row.get("stopped"):
        raise SystemExit(f"STOP: training stopped: {row}")

    # train_budget hardcodes snapshot=n60j5 in the manifest; correct it.
    man_path = OUT_DIR / "m8_manifest.json"
    if man_path.is_file():
        man = json.loads(man_path.read_text(encoding="utf-8"))
        man["snapshot"] = str(snap.relative_to(REPO_ROOT))
        man["joiner_share"] = 0.0
        man["matched_to"] = "artifacts/tokenizers/m8/ancoda-64k"
        man_path.write_text(json.dumps(man, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    tok = OUT_DIR / "tokenizer.json"
    fert = fert_compare(M8_REF / "tokenizer.json", tok)
    report = {
        "kind": "m8_ancoda64k_nojoiner_prep",
        "snapshot": str(snap.relative_to(REPO_ROOT)),
        "tokenizer_dir": str(OUT_DIR.relative_to(REPO_ROOT)),
        "tokenizer_sha256": sha256_file(tok),
        "train": row,
        "config_diff": config_diff(snap, dewa, latin),
        "joiner_vocab": joiner_vocab_count(tok),
        "fertility": fert,
        "roundtrip": roundtrip_stream(tok, 1000),
        "tokens_per_byte": tokens_per_byte_stream(tok, 10_000),
        "m8_ref_sha256": sha256_file(M8_REF / "tokenizer.json"),
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    save_json(REPORT, report)
    write_md(report)
    print(json.dumps({k: report[k] for k in ("tokenizer_sha256", "joiner_vocab", "roundtrip")}, indent=2))


if __name__ == "__main__":
    main()

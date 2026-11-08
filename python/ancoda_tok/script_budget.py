"""Script-budget frontier diagnostic (M7 follow-up H).

Split n60 pretokens into Devanagari vs Latin/other, train final-config sides
with fixed Devanagari budgets, interleave Block 2 merges by count, evaluate.
Does not change the shipping final config.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

from ancoda_tok.clean.common import base_manifest, finish_manifest, save_json, sha256_file
from ancoda_tok.counts import iter_entries, write_counts
from ancoda_tok.export_cli import export_dir
from ancoda_tok.export_hf import load_hf
from ancoda_tok.m7 import M7_DIR, TOK_DIR, _caffeinate, _env, harness_add
from ancoda_tok.storage import REPO_ROOT

SEED = 20261104
COUNTS_N60 = REPO_ROOT / "artifacts" / "counts" / "n60" / "pretoken_counts.bin"
COVERED = REPO_ROOT / "artifacts" / "inventory" / "covered_64k.json"
UNICODE = REPO_ROOT / "third_party" / "unicode" / "16.0.0" / "UnicodeData.txt"
OUT_ROOT = REPO_ROOT / "artifacts" / "eval" / "m7" / "script_budget"

# Devanagari-related blocks used for the side split of pretokens / tokens.
DEVA_RANGES = (
    range(0x0900, 0x0980),
    range(0xA8E0, 0xA900),
    range(0x1CD0, 0x1D00),
    range(0x11B00, 0x11B60),
)


def is_deva_bytes(raw: bytes) -> bool:
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        return False
    for ch in text:
        cp = ord(ch)
        for r in DEVA_RANGES:
            if cp in r:
                return True
    return False


def split_counts(dest_dir: Path) -> dict:
    dest_dir.mkdir(parents=True, exist_ok=True)
    dewa_path = dest_dir / "deva_pretoken_counts.bin"
    latin_path = dest_dir / "latin_pretoken_counts.bin"
    if dewa_path.is_file() and latin_path.is_file() and (dest_dir / "split_manifest.json").is_file():
        print("split counts already present", flush=True)
        return json.loads((dest_dir / "split_manifest.json").read_text(encoding="utf-8"))
    dewa: list[tuple[bytes, int]] = []
    latin: list[tuple[bytes, int]] = []
    for raw, count in iter_entries(COUNTS_N60):
        if is_deva_bytes(raw):
            dewa.append((raw, count))
        else:
            latin.append((raw, count))
    write_counts(dewa_path, dewa)
    write_counts(latin_path, latin)
    meta = base_manifest(
        kind="script_budget_split",
        source_counts=str(COUNTS_N60.relative_to(REPO_ROOT)),
        source_sha256=sha256_file(COUNTS_N60),
        dewa_unique=len(dewa),
        dewa_occurrences=sum(c for _r, c in dewa),
        latin_unique=len(latin),
        latin_occurrences=sum(c for _r, c in latin),
        dewa_sha256=sha256_file(dewa_path),
        latin_sha256=sha256_file(latin_path),
    )
    save_json(dest_dir / "split_manifest.json", finish_manifest(meta))
    print(
        f"split dewa unique={len(dewa)} occ={meta['dewa_occurrences']} "
        f"latin unique={len(latin)} occ={meta['latin_occurrences']}",
        flush=True,
    )
    return meta


def current_n60_deva_budget() -> dict:
    ranks = json.loads((TOK_DIR / "ancoda-64k-n60" / "ranks.json").read_text(encoding="utf-8"))
    layout = ranks["layout"]
    tokens = [bytes.fromhex(x) for x in ranks["tokens_hex"]]
    block0 = layout["block0_end"] - layout["byte_end"]
    block1 = layout["block1_end"] - layout["block0_end"]
    dewa_b2 = 0
    latin_b2 = 0
    for tid in range(layout["block1_end"], layout["reserved_start"]):
        if is_deva_bytes(tokens[tid]):
            dewa_b2 += 1
        else:
            latin_b2 += 1
    # B_d includes Block 0 + Block 1 + Devanagari Block 2 (bytes shared, not in B_d).
    b_d = block0 + block1 + dewa_b2
    return {
        "byte_end": layout["byte_end"],
        "block0": block0,
        "block1": block1,
        "deva_block2": dewa_b2,
        "latin_block2": latin_b2,
        "B_d": b_d,
        "reserved": 1024,
        "vocab_size": layout["vocab_size"],
    }


def _write_side_config(
    *,
    name: str,
    counts_rel: str,
    output_rel: str,
    vocab_size: int,
    block0: bool,
    block1: bool,
) -> Path:
    text = "\n".join(
        [
            f"vocab_size: {vocab_size}",
            "plain: false",
            f"block0: {'true' if block0 else 'false'}",
            f"block1: {'true' if block1 else 'false'}",
            "bans: true",
            "gb9c: false",
            f"seed: {SEED}",
            f"counts: {counts_rel}",
            f"block1_path: artifacts/inventory/covered_64k.json",
            "unicode_data: third_party/unicode/16.0.0/UnicodeData.txt",
            f"output_dir: {output_rel}",
            "require_full: true",
            "",
        ]
    )
    path = OUT_ROOT / "configs" / f"{name}.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def train_sides(b_d: int) -> tuple[Path, Path]:
    """Train Devanagari and Latin sides for budget B_d (excl. shared bytes)."""
    block0 = 133
    # Block 1 size from covered file
    covered = json.loads(COVERED.read_text(encoding="utf-8"))
    block1 = int(covered["block1_merge_count"])
    fixed_deva = block0 + block1  # 3409 at 64k covered
    if b_d < fixed_deva:
        raise SystemExit(f"STOP: B_d={b_d} < block0+block1={fixed_deva}")
    dewa_b2 = b_d - fixed_deva
    # mergeable = 64512 = 256 + B_d + latin_b2
    latin_b2 = 64512 - 256 - b_d
    if latin_b2 < 0:
        raise SystemExit(f"STOP: latin_b2 negative for B_d={b_d}")
    # Deva vocab_size: bytes + B_d + reserved
    dewa_vocab = 256 + b_d + 1024
    # Latin vocab_size: bytes + latin_b2 + reserved
    latin_vocab = 256 + latin_b2 + 1024
    tag = f"bd{b_d}"
    dewa_out = OUT_ROOT / tag / "side-deva"
    latin_out = OUT_ROOT / tag / "side-latin"
    dewa_cfg = _write_side_config(
        name=f"{tag}_deva",
        counts_rel="artifacts/eval/m7/script_budget/deva_pretoken_counts.bin",
        output_rel=str(dewa_out.relative_to(REPO_ROOT)),
        vocab_size=dewa_vocab,
        block0=True,
        block1=True,
    )
    latin_cfg = _write_side_config(
        name=f"{tag}_latin",
        counts_rel="artifacts/eval/m7/script_budget/latin_pretoken_counts.bin",
        output_rel=str(latin_out.relative_to(REPO_ROOT)),
        vocab_size=latin_vocab,
        block0=False,
        block1=False,
    )
    for cfg, out in ((dewa_cfg, dewa_out), (latin_cfg, latin_out)):
        if (out / "ranks.json").is_file():
            print(f"resume {out}", flush=True)
            continue
        _caffeinate(
            [
                "cargo",
                "run",
                "--release",
                "--offline",
                "-p",
                "ancoda-bpe",
                "--",
                "train",
                "--config",
                str(cfg),
            ]
        )
    return dewa_out, latin_out


def combine(deva_dir: Path, latin_dir: Path, dest: Path) -> Path:
    """Interleave Block 2 merges by count; bytes + Block0/1 from Deva side."""
    if (dest / "ranks.json").is_file() and (dest / "tokenizer.json").is_file():
        print(f"combine resume {dest}", flush=True)
        return dest
    dewa = json.loads((deva_dir / "ranks.json").read_text(encoding="utf-8"))
    latin = json.loads((latin_dir / "ranks.json").read_text(encoding="utf-8"))
    dewa_tokens = [bytes.fromhex(x) for x in dewa["tokens_hex"]]
    latin_tokens = [bytes.fromhex(x) for x in latin["tokens_hex"]]
    d_layout = dewa["layout"]
    l_layout = latin["layout"]
    # Shared bytes from either (identical).
    tokens: list[bytes] = dewa_tokens[:256]
    # Block 0 + Block 1 from Deva.
    tokens.extend(dewa_tokens[256 : d_layout["block1_end"]])
    index = {t: i for i, t in enumerate(tokens)}

    # Block 2 queues: merges in learning order with counts.
    # Only emit when both parents already exist (interleaving two dependency orders).
    dewa_q = list(dewa["block2_merges"])
    latin_q = list(latin["block2_merges"])
    i = j = 0
    combined_merges = []
    skipped_dup = 0
    while (i < len(dewa_q) or j < len(latin_q)) and len(tokens) < 64512:
        cand = []
        if i < len(dewa_q):
            row = dewa_q[i]
            left_b = dewa_tokens[row["left"]]
            right_b = dewa_tokens[row["right"]]
            if left_b in index and right_b in index:
                cand.append(("deva", i, row, left_b, right_b, row["count"]))
        if j < len(latin_q):
            row = latin_q[j]
            left_b = latin_tokens[row["left"]]
            right_b = latin_tokens[row["right"]]
            if left_b in index and right_b in index:
                cand.append(("latin", j, row, left_b, right_b, row["count"]))
        if not cand:
            # Advance a stuck head (should resolve once the other side catches up).
            if i < len(dewa_q):
                i += 1
            elif j < len(latin_q):
                j += 1
            else:
                break
            continue
        # Higher count first; on tie prefer Devanagari.
        cand.sort(key=lambda item: (-item[5], 0 if item[0] == "deva" else 1))
        side, idx, row, left_b, right_b, count = cand[0]
        if side == "deva":
            i = idx + 1
        else:
            j = idx + 1
        merged = left_b + right_b
        if merged in index:
            skipped_dup += 1
            continue
        new_id = len(tokens)
        index[merged] = new_id
        tokens.append(merged)
        combined_merges.append(
            {
                "id": new_id,
                "left": index[left_b],
                "right": index[right_b],
                "count": count,
                "side": side,
            }
        )
    if len(tokens) < 64512:
        print(
            f"warning: combined only {len(tokens)} mergeable tokens, want 64512 "
            f"(skipped_dup={skipped_dup} dewa_i={i}/{len(dewa_q)} "
            f"latin_j={j}/{len(latin_q)})",
            flush=True,
        )

    # Truncate if over; if under (cross-side duplicates), shrink reserved_start.
    tokens = tokens[:64512]
    reserved_start = len(tokens)
    vocab_size = reserved_start + 1024
    specials = [
        {"name": s["name"], "id": reserved_start + k}
        for k, s in enumerate(dewa["special_tokens"])
    ]
    ranks = {
        "version": 1,
        "seed": SEED,
        "layout": {
            "byte_end": 256,
            "block0_end": d_layout["block0_end"],
            "block1_end": d_layout["block1_end"],
            "reserved_start": reserved_start,
            "vocab_size": vocab_size,
        },
        "tokens_hex": [t.hex() for t in tokens],
        "special_tokens": specials,
        "block2_merges": combined_merges,
        "stats": {
            "combined_from": {
                "deva": str(deva_dir.relative_to(REPO_ROOT)),
                "latin": str(latin_dir.relative_to(REPO_ROOT)),
            },
            "deva_merges_used": i,
            "latin_merges_used": j,
            "mergeable_tokens": len(tokens),
            "skipped_dup": skipped_dup,
            "target_mergeable": 64512,
        },
    }
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "ranks.json").write_text(json.dumps(ranks, indent=2) + "\n", encoding="utf-8")
    export_dir(dest / "ranks.json")
    return dest


def guarantee_quick(tok_dir: Path, tag: str) -> dict:
    """Run full guarantee tests; write report under script_budget."""
    report = OUT_ROOT / f"guarantee_{tag}.json"
    orphans = OUT_ROOT / f"orphans_{tag}.txt"
    if report.is_file():
        data = json.loads(report.read_text(encoding="utf-8"))
        if not data.get("failures"):
            return data
    _caffeinate(
        [
            "uv",
            "run",
            "python",
            "-u",
            "-m",
            "ancoda_tok.guarantee",
            "--tokenizer-dir",
            str(tok_dir),
            "--covered",
            str(COVERED),
            "--snapshot-prefix",
            "n60",
            "--report",
            str(report),
            "--orphans",
            str(orphans),
        ]
    )
    return json.loads(report.read_text(encoding="utf-8"))


def tokens_per_kb(tok_dir: Path, texts: list[str]) -> float:
    tok = load_hf(tok_dir / "tokenizer.json")
    total_tok = 0
    total_bytes = 0
    for text in texts:
        if not text:
            continue
        total_tok += len(tok.encode(text, add_special_tokens=False).ids)
        total_bytes += len(text.encode("utf-8"))
    if total_bytes == 0:
        return 0.0
    return 1024.0 * total_tok / total_bytes


def held_out_code_math(limit_bytes: int = 5_000_000) -> tuple[list[str], list[str]]:
    import pyarrow.parquet as pq

    snap = REPO_ROOT / "data" / "snapshots" / "n60_cf8066dbbdd0da93"
    code: list[str] = []
    math: list[str] = []
    code_b = math_b = 0
    for shard in sorted(snap.glob("shard-*.parquet")):
        if code_b >= limit_bytes and math_b >= limit_bytes:
            break
        pf = pq.ParquetFile(shard)
        for batch in pf.iter_batches(columns=["text", "bucket"], batch_size=128):
            for text, bucket in zip(
                batch.column(0).to_pylist(), batch.column(1).to_pylist(), strict=True
            ):
                if not text:
                    continue
                if bucket == "Code" and code_b < limit_bytes:
                    code.append(text)
                    code_b += len(text.encode("utf-8"))
                elif bucket == "Math" and math_b < limit_bytes:
                    math.append(text)
                    math_b += len(text.encode("utf-8"))
    return code, math


def flores_deva_paths() -> list[Path]:
    root = (
        REPO_ROOT
        / "third_party"
        / "yak-tokenizer"
        / "data"
        / "eval"
        / "flores_plus"
        / "devtest"
    )
    return sorted(root.glob("*_Deva.txt"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--budgets",
        default="current,32000,36000,40000,44000",
        help="Comma list; 'current' uses measured n60 Devanagari budget",
    )
    parser.add_argument("--skip-guarantee", action="store_true")
    args = parser.parse_args()
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    split_counts(OUT_ROOT)
    current = current_n60_deva_budget()
    save_json(
        OUT_ROOT / "current_n60_split.json",
        finish_manifest(base_manifest(kind="script_budget_current", **current)),
    )
    print("current n60 Devanagari budget:", json.dumps(current), flush=True)

    budgets: list[int] = []
    for item in args.budgets.split(","):
        item = item.strip()
        if item == "current":
            budgets.append(int(current["B_d"]))
        else:
            budgets.append(int(item))
    # unique preserve order
    seen = set()
    ordered = []
    for b in budgets:
        if b not in seen:
            seen.add(b)
            ordered.append(b)

    code_texts, math_texts = held_out_code_math()
    rows = []
    adds = {}
    for b_d in ordered:
        tag = f"bd{b_d}"
        print(f"=== budget B_d={b_d} ===", flush=True)
        dewa_dir, latin_dir = train_sides(b_d)
        dest = OUT_ROOT / tag / "combined"
        combine(dewa_dir, latin_dir, dest)
        if not args.skip_guarantee:
            g = guarantee_quick(dest, tag)
            violations = g.get("hf_covered_violations")
            failures = g.get("failures")
        else:
            violations = None
            failures = None
        adds[f"sb_{tag}"] = dest
        rows.append(
            {
                "B_d": b_d,
                "dir": str(dest.relative_to(REPO_ROOT)),
                "hf_covered_violations": violations,
                "guarantee_failures": failures,
                "code_tokens_per_kb": tokens_per_kb(dest, code_texts),
                "math_tokens_per_kb": tokens_per_kb(dest, math_texts),
            }
        )
    harness_add(adds, "script_budget")
    harness = json.loads((M7_DIR / "harness_script_budget.json").read_text(encoding="utf-8"))
    # Enrich with FLORES Deva fertilities from harness JSON files
    for row in rows:
        tag = f"bd{row['B_d']}"
        name = f"sb_{tag}"
        h = harness["tokenizers"].get(name, {})
        row["npi"] = h.get("npi")
        row["eng"] = h.get("eng")
        row["hin"] = h.get("hin")
        row["orphan"] = h.get("npi_orphan_mark_rate")
        # per-language from raw yak artifact
        raw = M7_DIR / f"script_budget_{name}.json"
        if raw.is_file():
            rec = json.loads(raw.read_text(encoding="utf-8"))
            results = rec.get("results") or {}
            dewa = {}
            for key, val in results.items():
                if key.startswith("flores_devtest/") and key.endswith("_Deva"):
                    lang = key.split("/")[-1]
                    dewa[lang] = val.get("fertility")
            row["flores_deva"] = dewa
            if dewa:
                row["flores_deva_mean"] = sum(dewa.values()) / len(dewa)
    save_json(
        OUT_ROOT / "frontier.json",
        finish_manifest(
            base_manifest(
                kind="script_budget_frontier",
                current_n60=current,
                rows=rows,
            )
        ),
    )
    print(json.dumps({"current": current, "rows": rows}, indent=2), flush=True)


if __name__ == "__main__":
    main()

"""Shared helpers for script-budget-style Devanagari/Latin frontiers (diagnostics)."""

from __future__ import annotations

import json
from pathlib import Path

from ancoda_tok.clean.common import base_manifest, finish_manifest, save_json, sha256_file
from ancoda_tok.counts import iter_entries, write_counts
from ancoda_tok.export_cli import export_dir
from ancoda_tok.export_hf import load_hf
from ancoda_tok.mix_sweep import MIX_SWEEP_DIR, _caffeinate, harness_add
from ancoda_tok.storage import REPO_ROOT

SEED = 20261104
COUNTS_N60 = REPO_ROOT / "artifacts" / "counts" / "n60" / "pretoken_counts.bin"
UNICODE = REPO_ROOT / "third_party" / "unicode" / "16.0.0" / "UnicodeData.txt"

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


def split_counts(dest_dir: Path, counts_path: Path = COUNTS_N60) -> dict:
    dest_dir.mkdir(parents=True, exist_ok=True)
    dewa_path = dest_dir / "deva_pretoken_counts.bin"
    latin_path = dest_dir / "latin_pretoken_counts.bin"
    manifest = dest_dir / "split_manifest.json"
    if dewa_path.is_file() and latin_path.is_file() and manifest.is_file():
        print("split counts already present", flush=True)
        return json.loads(manifest.read_text(encoding="utf-8"))
    # Reuse H split if identical source.
    h_dir = REPO_ROOT / "artifacts" / "eval" / "mix_sweep" / "script_budget"
    h_man = h_dir / "split_manifest.json"
    if h_man.is_file() and counts_path.resolve() == COUNTS_N60.resolve():
        h = json.loads(h_man.read_text(encoding="utf-8"))
        if h.get("source_sha256") == sha256_file(COUNTS_N60):
            import shutil

            shutil.copy2(h_dir / "deva_pretoken_counts.bin", dewa_path)
            shutil.copy2(h_dir / "latin_pretoken_counts.bin", latin_path)
            save_json(manifest, h)
            print("reused H split counts", flush=True)
            return h
    dewa: list[tuple[bytes, int]] = []
    latin: list[tuple[bytes, int]] = []
    for raw, count in iter_entries(counts_path):
        if is_deva_bytes(raw):
            dewa.append((raw, count))
        else:
            latin.append((raw, count))
    write_counts(dewa_path, dewa)
    write_counts(latin_path, latin)
    meta = base_manifest(
        kind="script_budget_split",
        source_counts=str(counts_path.relative_to(REPO_ROOT)),
        source_sha256=sha256_file(counts_path),
        dewa_unique=len(dewa),
        dewa_occurrences=sum(c for _r, c in dewa),
        latin_unique=len(latin),
        latin_occurrences=sum(c for _r, c in latin),
        dewa_sha256=sha256_file(dewa_path),
        latin_sha256=sha256_file(latin_path),
    )
    save_json(manifest, finish_manifest(meta))
    print(
        f"split dewa unique={len(dewa)} occ={meta['dewa_occurrences']} "
        f"latin unique={len(latin)} occ={meta['latin_occurrences']}",
        flush=True,
    )
    return meta


def measure_bd(ranks_path: Path) -> dict:
    ranks = json.loads(ranks_path.read_text(encoding="utf-8"))
    layout = ranks["layout"]
    tokens = [bytes.fromhex(x) for x in ranks["tokens_hex"]]
    block0 = layout["block0_end"] - layout["byte_end"]
    block1 = layout["block1_end"] - layout["block0_end"]
    dewa_b2 = latin_b2 = 0
    for tid in range(layout["block1_end"], layout["reserved_start"]):
        if is_deva_bytes(tokens[tid]):
            dewa_b2 += 1
        else:
            latin_b2 += 1
    return {
        "byte_end": layout["byte_end"],
        "block0": block0,
        "block1": block1,
        "deva_block2": dewa_b2,
        "latin_block2": latin_b2,
        "B_d": block0 + block1 + dewa_b2,
        "reserved": layout["vocab_size"] - layout["reserved_start"],
        "vocab_size": layout["vocab_size"],
        "mergeable": layout["reserved_start"],
    }


def write_side_config(
    *,
    path: Path,
    counts_rel: str,
    output_rel: str,
    vocab_size: int,
    block0: bool,
    block1: bool,
    block1_path: str,
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
            f"block1_path: {block1_path}",
            "unicode_data: third_party/unicode/16.0.0/UnicodeData.txt",
            f"output_dir: {output_rel}",
            "require_full: true",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def train_sides(
    *,
    out_root: Path,
    tag: str,
    b_d: int,
    mergeable: int,
    block0: int,
    block1: int,
    block1_path: str,
    dewa_counts_rel: str,
    latin_counts_rel: str,
) -> tuple[Path, Path]:
    fixed = block0 + block1
    if b_d < fixed:
        raise SystemExit(f"STOP: B_d={b_d} < block0+block1={fixed}")
    latin_b2 = mergeable - 256 - b_d
    if latin_b2 < 0:
        raise SystemExit(f"STOP: latin_b2 negative for B_d={b_d} mergeable={mergeable}")
    dewa_vocab = 256 + b_d + 1024
    latin_vocab = 256 + latin_b2 + 1024
    dewa_out = out_root / tag / "side-deva"
    latin_out = out_root / tag / "side-latin"
    dewa_cfg = write_side_config(
        path=out_root / "configs" / f"{tag}_deva.yaml",
        counts_rel=dewa_counts_rel,
        output_rel=str(dewa_out.relative_to(REPO_ROOT)),
        vocab_size=dewa_vocab,
        block0=True,
        block1=True,
        block1_path=block1_path,
    )
    latin_cfg = write_side_config(
        path=out_root / "configs" / f"{tag}_latin.yaml",
        counts_rel=latin_counts_rel,
        output_rel=str(latin_out.relative_to(REPO_ROOT)),
        vocab_size=latin_vocab,
        block0=False,
        block1=False,
        block1_path=block1_path,
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
        # Optional RSS check via ranks stats
        stats = json.loads((out / "ranks.json").read_text(encoding="utf-8")).get("stats", {})
        peak = int(stats.get("peak_rss_bytes") or 0)
        print(f"trained {out} peak_rss_bytes={peak}", flush=True)
    return dewa_out, latin_out


def combine(
    dewa_dir: Path,
    latin_dir: Path,
    dest: Path,
    *,
    mergeable: int,
) -> Path:
    if (dest / "ranks.json").is_file() and (dest / "tokenizer.json").is_file():
        print(f"combine resume {dest}", flush=True)
        return dest
    dewa = json.loads((dewa_dir / "ranks.json").read_text(encoding="utf-8"))
    latin = json.loads((latin_dir / "ranks.json").read_text(encoding="utf-8"))
    dewa_tokens = [bytes.fromhex(x) for x in dewa["tokens_hex"]]
    latin_tokens = [bytes.fromhex(x) for x in latin["tokens_hex"]]
    d_layout = dewa["layout"]
    tokens: list[bytes] = dewa_tokens[:256]
    tokens.extend(dewa_tokens[256 : d_layout["block1_end"]])
    index = {t: i for i, t in enumerate(tokens)}
    dewa_q = list(dewa["block2_merges"])
    latin_q = list(latin["block2_merges"])
    i = j = 0
    combined_merges = []
    skipped_dup = 0
    while (i < len(dewa_q) or j < len(latin_q)) and len(tokens) < mergeable:
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
            if i < len(dewa_q):
                i += 1
            elif j < len(latin_q):
                j += 1
            else:
                break
            continue
        cand.sort(key=lambda item: (-item[5], 0 if item[0] == "deva" else 1))
        side, idx, _row, left_b, right_b, count = cand[0]
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
    if len(tokens) < mergeable:
        print(
            f"warning: combined only {len(tokens)} mergeable tokens, want {mergeable} "
            f"(skipped_dup={skipped_dup})",
            flush=True,
        )
    tokens = tokens[:mergeable]
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
                "deva": str(dewa_dir.relative_to(REPO_ROOT)),
                "latin": str(latin_dir.relative_to(REPO_ROOT)),
            },
            "deva_merges_used": i,
            "latin_merges_used": j,
            "mergeable_tokens": len(tokens),
            "skipped_dup": skipped_dup,
            "target_mergeable": mergeable,
        },
    }
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "ranks.json").write_text(json.dumps(ranks, indent=2) + "\n", encoding="utf-8")
    export_dir(dest / "ranks.json")
    return dest


def guarantee(
    tok_dir: Path,
    *,
    report: Path,
    orphans: Path,
    covered: Path,
    snapshot_prefix: str = "n60",
    snapshot: Path | None = None,
    snapshot_line_limit: int = 1_000_000,
) -> dict:
    """Run guarantee tests. Diagnostics use 1M snapshot lines (final-build does 10M)."""
    if report.is_file():
        data = json.loads(report.read_text(encoding="utf-8"))
        if not data.get("failures") and data.get("snapshot_lines") == snapshot_line_limit:
            return data
        # Stale/partial report (e.g. interrupted 10M run): re-run.
        report.unlink(missing_ok=True)
    cmd = [
        "uv",
        "run",
        "python",
        "-u",
        "-m",
        "ancoda_tok.guarantee",
        "--tokenizer-dir",
        str(tok_dir),
        "--covered",
        str(covered),
        "--report",
        str(report),
        "--orphans",
        str(orphans),
        "--snapshot-line-limit",
        str(snapshot_line_limit),
    ]
    if snapshot is not None:
        cmd.extend(["--snapshot", str(snapshot)])
    else:
        cmd.extend(["--snapshot-prefix", snapshot_prefix])
    _caffeinate(cmd)
    return json.loads(report.read_text(encoding="utf-8"))


def tokens_per_kb(tok_path: Path, texts: list[str]) -> float:
    tok = load_hf(tok_path if tok_path.name == "tokenizer.json" else tok_path / "tokenizer.json")
    total_tok = total_bytes = 0
    for text in texts:
        if not text:
            continue
        total_tok += len(tok.encode(text, add_special_tokens=False).ids)
        total_bytes += len(text.encode("utf-8"))
    if total_bytes == 0:
        return 0.0
    return 1024.0 * total_tok / total_bytes


def held_out_code_math(
    snap: Path | None = None, limit_bytes: int = 5_000_000
) -> tuple[list[str], list[str]]:
    import pyarrow.parquet as pq

    snap = snap or (REPO_ROOT / "data" / "snapshots" / "n60_cf8066dbbdd0da93")
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


def enrich_from_harness(row: dict, harness_tag: str, name: str) -> None:
    hpath = MIX_SWEEP_DIR / f"harness_{harness_tag}.json"
    if not hpath.is_file():
        return
    harness = json.loads(hpath.read_text(encoding="utf-8"))
    h = harness.get("tokenizers", {}).get(name, {})
    row["npi"] = h.get("npi")
    row["eng"] = h.get("eng")
    row["hin"] = h.get("hin")
    row["orphan"] = h.get("npi_orphan_mark_rate")
    row["zwj"] = h.get("zwj")
    row["zwnj"] = h.get("zwnj")
    raw = MIX_SWEEP_DIR / f"{harness_tag}_{name}.json"
    if raw.is_file():
        rec = json.loads(raw.read_text(encoding="utf-8"))
        results = rec.get("results") or {}
        aya = results.get("npi_everyday_aya") or {}
        if aya.get("fertility") is not None:
            row["aya_fertility"] = aya.get("fertility")
        dewa = {}
        for key, val in results.items():
            if key.startswith("flores_devtest/") and key.endswith("_Deva"):
                dewa[key.split("/")[-1]] = val.get("fertility")
        row["flores_deva"] = dewa
        if dewa:
            row["flores_deva_mean"] = sum(dewa.values()) / len(dewa)


def run_harness(adds: dict[str, Path], tag: str) -> None:
    harness_add(adds, tag)

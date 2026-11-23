"""T3: 10 GB n60-share snapshot WITH 1% joiner; frontier B_d 44/46/48k."""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

from ancoda_tok.clean.augment_joiners import build_joiner_copy
from ancoda_tok.clean.audit_snapshots import audit_snapshot
from ancoda_tok.clean.common import (
    Stop,
    base_manifest,
    finish_manifest,
    free_bytes,
    iter_parquet_rows,
    load_json,
    load_mix,
    save_json,
    utf8_len,
)
from ancoda_tok.clean.snapshot import (
    SEED,
    build_index,
    fetch_texts,
    hashlib_int,
    load_index_rows,
    pool_sample_index,
    scale_shares,
    write_snapshot,
)
from ancoda_tok.frontier_common import (
    combine,
    enrich_from_harness,
    guarantee,
    held_out_code_math,
    is_deva_bytes,
    run_harness,
    tokens_per_kb,
    train_sides,
)
from ancoda_tok.frontier_metrics import attach_extra_metrics
from ancoda_tok.counts import iter_entries, write_counts
from ancoda_tok.mix_sweep import _caffeinate
from ancoda_tok.storage import REPO_ROOT

OUT = REPO_ROOT / "artifacts" / "eval" / "mix_sweep" / "t3_10gb_joiner"
BUDGET = 10_000_000_000
NEPALI_SHARE = 0.60
JOINER_SHARE = 0.01
NAME = "n60j10"
MERGEABLE = 64512
BLOCK0 = 133
BDS = (44000, 46000, 48000)
RSS_CAP = 12 * 1024**3
COVERED = REPO_ROOT / "artifacts" / "inventory" / "covered_64k.json"


def build_10gb_joiner_snapshot() -> Path:
    existing = list((REPO_ROOT / "data" / "snapshots").glob(f"{NAME}_*"))
    if len(existing) == 1 and (existing[0] / "manifest.json").is_file():
        print(f"resume snapshot {existing[0]}", flush=True)
        return existing[0]
    free = free_bytes()
    print(f"df check free_bytes={free}", flush=True)
    if free - BUDGET < 50_000_000_000:
        raise Stop(
            f"STOP: fewer than 50 GB would remain after 10GB write "
            f"(free={free}, budget={BUDGET})"
        )
    mix = load_mix()
    temperature = float(mix["temperature"])
    index_path = build_index(mix)
    index_rows = load_index_rows(index_path)
    by_bucket: dict[str, list] = defaultdict(list)
    for row in index_rows:
        by_bucket[row["bucket"]].append(row)

    shares = scale_shares(mix, NEPALI_SHARE)
    natural_nepali = shares["Nepali"] - JOINER_SHARE
    targets = {b: int(BUDGET * s) for b, s in shares.items()}
    targets["Nepali"] = int(BUDGET * natural_nepali)

    selected_index = []
    used_ids: set[str] = set()
    for bucket, target in targets.items():
        picked = pool_sample_index(
            by_bucket.get(bucket, []),
            target,
            seed=SEED + int(NEPALI_SHARE * 1000) + hashlib_int(bucket) + 10_000,
            temperature=temperature,
        )
        for p in picked:
            used_ids.add(p["doc_id"])
        selected_index.extend(picked)
    chosen = fetch_texts(selected_index)

    unused = [d for d in by_bucket.get("Nepali", []) if d["doc_id"] not in used_ids]
    joiner_target = int(BUDGET * JOINER_SHARE)
    unused_sorted = sorted(unused, key=lambda d: d["doc_id"])
    rng = random.Random(SEED)
    rng.shuffle(unused_sorted)
    pool_rows = []
    pool_bytes = 0
    for item in unused_sorted:
        pool_rows.append(item)
        pool_bytes += item["text_bytes"]
        if pool_bytes >= joiner_target * 3:
            break
    unused_docs = fetch_texts(pool_rows)
    joiner_dir = REPO_ROOT / "data" / "clean" / f"_synthetic_joiners_{NAME}"
    joiner_manifest = build_joiner_copy(
        unused_docs,
        target_bytes=joiner_target,
        seed=SEED,
        out_dir=joiner_dir,
    )
    for row in iter_parquet_rows(joiner_dir):
        chosen.append(
            {
                "source": row["source"],
                "doc_id": row["doc_id"],
                "license": row["license"],
                "bucket": "Nepali",
                "text_bytes": utf8_len(row["text"]),
                "text": row["text"],
                "glotlid_label": "synthetic",
                "glotlid_prob": 1.0,
                "synthetic": True,
            }
        )
    mix_meta = {
        "nepali_share": NEPALI_SHARE,
        "shares": shares,
        "joiner_share": JOINER_SHARE,
        "joiner_docs": joiner_manifest["docs"],
        "joiner_bytes": joiner_manifest["text_bytes"],
        "budget": BUDGET,
        "diagnostic": "T3_10gb_n60_joiner",
        "synthetic": True,
    }
    path = write_snapshot(NAME, chosen, mix_meta=mix_meta)
    counts = audit_snapshot(path)
    print(f"audit {NAME} {counts}", flush=True)
    if any(v != 0 for v in counts.values()):
        raise Stop(f"STOP: snapshot {NAME} failed decontam audit: {counts}")
    return path


def count_snapshot(snap: Path) -> Path:
    out = OUT / "counts"
    if (out / "pretoken_counts.bin").is_file():
        print("resume counts", flush=True)
        return out / "pretoken_counts.bin"
    out.mkdir(parents=True, exist_ok=True)
    _caffeinate(
        [
            "uv",
            "run",
            "python",
            "-u",
            "-m",
            "ancoda_tok.counts",
            "--snapshot",
            str(snap),
            "--out",
            str(out),
        ]
    )
    return out / "pretoken_counts.bin"


def split_from_counts(counts_path: Path) -> None:
    dewa_path = OUT / "deva_pretoken_counts.bin"
    latin_path = OUT / "latin_pretoken_counts.bin"
    if dewa_path.is_file() and latin_path.is_file():
        print("split already present", flush=True)
        return
    dewa = []
    latin = []
    for raw, count in iter_entries(counts_path):
        if is_deva_bytes(raw):
            dewa.append((raw, count))
        else:
            latin.append((raw, count))
    write_counts(dewa_path, dewa)
    write_counts(latin_path, latin)
    print(f"split dewa={len(dewa)} latin={len(latin)}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-guarantee", action="store_true")
    parser.add_argument("--skip-snapshot", action="store_true")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    if args.skip_snapshot:
        snaps = list((REPO_ROOT / "data" / "snapshots").glob(f"{NAME}_*"))
        if len(snaps) != 1:
            raise SystemExit(f"STOP: need exactly one {NAME}_* snapshot")
        snap = snaps[0]
    else:
        snap = build_10gb_joiner_snapshot()

    counts_path = count_snapshot(snap)
    split_from_counts(counts_path)

    covered = json.loads(COVERED.read_text(encoding="utf-8"))
    block1 = int(covered["block1_merge_count"])
    code_texts, math_texts = held_out_code_math(snap)
    rows = []
    adds: dict[str, Path] = {}
    for b_d in BDS:
        tag = f"bd{b_d}"
        print(f"=== T3 budget B_d={b_d} ===", flush=True)
        dewa_dir, latin_dir = train_sides(
            out_root=OUT,
            tag=tag,
            b_d=b_d,
            mergeable=MERGEABLE,
            block0=BLOCK0,
            block1=block1,
            block1_path="artifacts/inventory/covered_64k.json",
            dewa_counts_rel="artifacts/eval/mix_sweep/t3_10gb_joiner/deva_pretoken_counts.bin",
            latin_counts_rel="artifacts/eval/mix_sweep/t3_10gb_joiner/latin_pretoken_counts.bin",
        )
        # RSS check on both sides
        for side in (dewa_dir, latin_dir):
            stats = json.loads((side / "ranks.json").read_text(encoding="utf-8")).get(
                "stats", {}
            )
            peak = int(stats.get("peak_rss_bytes") or 0)
            if peak > RSS_CAP:
                print(
                    f"STOP RSS: {side} peak_rss_bytes={peak} > {RSS_CAP}; skipping combine",
                    flush=True,
                )
                rows.append(
                    {
                        "arm": "T3",
                        "label": f"10gb_joiner_Bd{b_d}",
                        "B_d": b_d,
                        "stopped": "rss_cap",
                        "peak_rss_bytes": peak,
                    }
                )
                break
        else:
            dest = OUT / tag / "combined"
            combine(dewa_dir, latin_dir, dest, mergeable=MERGEABLE)
            if not args.skip_guarantee:
                # Diagnostics: all eval texts + 1M snapshot lines (final-build does 10M).
                # Prefer the 10 GB joiner snapshot when present.
                g = guarantee(
                    dest,
                    report=OUT / f"guarantee_{tag}.json",
                    orphans=OUT / f"orphans_{tag}.txt",
                    covered=COVERED,
                    snapshot=snap,
                    snapshot_line_limit=1_000_000,
                )
                viol = g.get("hf_covered_violations")
                fails = g.get("failures")
            else:
                viol = fails = None
            name = f"t3_{tag}"
            adds[name] = dest
            rows.append(
                {
                    "arm": "T3",
                    "label": f"10gb_joiner_Bd{b_d}",
                    "B_d": b_d,
                    "dir": str(dest.relative_to(REPO_ROOT)),
                    "hf_covered_violations": viol,
                    "guarantee_failures": fails,
                    "code_tokens_per_kb": tokens_per_kb(dest, code_texts),
                    "math_tokens_per_kb": tokens_per_kb(dest, math_texts),
                }
            )
    if adds:
        run_harness(adds, "t3")
        for row in rows:
            if row.get("B_d") is not None and "stopped" not in row:
                enrich_from_harness(row, "t3", f"t3_bd{row['B_d']}")
                attach_extra_metrics(
                    row,
                    REPO_ROOT / row["dir"],
                    snap=snap,
                    harness_aya=row.get("aya_fertility"),
                )
    save_json(
        OUT / "frontier.json",
        finish_manifest(
            base_manifest(
                kind="t3_10gb_joiner",
                snapshot=str(snap.relative_to(REPO_ROOT)),
                budget=BUDGET,
                joiner_share=JOINER_SHARE,
                rows=rows,
            )
        ),
    )
    print(json.dumps({"snapshot": str(snap), "rows": rows}, indent=2), flush=True)


if __name__ == "__main__":
    main()

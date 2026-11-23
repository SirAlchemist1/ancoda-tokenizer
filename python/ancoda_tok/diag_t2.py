"""T2: Latin-side code/math rebalance at B_d=46000/48000 (64k; diagnostic)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ancoda_tok.clean.common import base_manifest, finish_manifest, save_json, sha256_file
from ancoda_tok.counts import iter_entries, write_counts
from ancoda_tok.frontier_common import (
    combine,
    enrich_from_harness,
    guarantee,
    held_out_code_math,
    run_harness,
    split_counts,
    tokens_per_kb,
    train_sides,
    write_side_config,
)
from ancoda_tok.frontier_metrics import attach_extra_metrics
from ancoda_tok.mix_sweep import _caffeinate
from ancoda_tok.storage import REPO_ROOT
N60_SNAP = REPO_ROOT / "data" / "snapshots" / "n60_cf8066dbbdd0da93"

OUT = REPO_ROOT / "artifacts" / "eval" / "mix_sweep" / "t2_latin_rebalance"
SNAP = REPO_ROOT / "data" / "snapshots" / "n60_cf8066dbbdd0da93"
COVERED = REPO_ROOT / "artifacts" / "inventory" / "covered_64k.json"
MERGEABLE = 64512
BLOCK0 = 133
BDS = (46000, 48000)
SCALES = (1.0, 0.5, 0.25)


def _count_bucket(bucket: str) -> Path:
    dest = OUT / f"counts_{bucket.lower()}"
    if (dest / "pretoken_counts.bin").is_file() and (dest / "manifest.json").is_file():
        print(f"resume counts {bucket}", flush=True)
        return dest / "pretoken_counts.bin"
    dest.mkdir(parents=True, exist_ok=True)
    _caffeinate(
        [
            "uv",
            "run",
            "python",
            "-u",
            "-m",
            "ancoda_tok.counts",
            "--snapshot",
            str(SNAP),
            "--out",
            str(dest),
            "--bucket",
            bucket,
        ]
    )
    return dest / "pretoken_counts.bin"


def build_scaled_latin(scale: float) -> Path:
    """English + scale*Code + scale*Math (downsample only)."""
    tag = f"latin_cm{int(scale * 100)}"
    dest = OUT / f"{tag}_pretoken_counts.bin"
    if dest.is_file():
        print(f"resume {dest}", flush=True)
        return dest
    eng = dict(iter_entries(OUT / "counts_english" / "pretoken_counts.bin"))
    code = dict(iter_entries(OUT / "counts_code" / "pretoken_counts.bin"))
    math = dict(iter_entries(OUT / "counts_math" / "pretoken_counts.bin"))
    merged: dict[bytes, int] = {}
    for raw, c in eng.items():
        merged[raw] = merged.get(raw, 0) + c
    for raw, c in code.items():
        add = int(c * scale)
        if add > 0:
            merged[raw] = merged.get(raw, 0) + add
    for raw, c in math.items():
        add = int(c * scale)
        if add > 0:
            merged[raw] = merged.get(raw, 0) + add
    entries = sorted(merged.items(), key=lambda kv: kv[0])
    write_counts(dest, entries)
    save_json(
        OUT / f"{tag}_manifest.json",
        finish_manifest(
            base_manifest(
                kind="t2_scaled_latin",
                scale=scale,
                unique=len(entries),
                occurrences=sum(c for _r, c in entries),
                sha256=sha256_file(dest),
            )
        ),
    )
    print(f"wrote {dest} unique={len(entries)} scale={scale}", flush=True)
    return dest


def train_one_side(cfg: Path, out: Path) -> None:
    if (out / "ranks.json").is_file():
        print(f"resume {out}", flush=True)
        return
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-guarantee", action="store_true")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    # Also keep full dewa split for Devanagari side.
    split_counts(OUT)
    print("counting English / Code / Math pretokens from n60…", flush=True)
    _count_bucket("English")
    _count_bucket("Code")
    _count_bucket("Math")
    for scale in SCALES:
        build_scaled_latin(scale)

    covered = json.loads(COVERED.read_text(encoding="utf-8"))
    block1 = int(covered["block1_merge_count"])
    code_texts, math_texts = held_out_code_math()
    rows = []
    adds: dict[str, Path] = {}

    for b_d in BDS:
        # Shared Devanagari side for this B_d (5% Block1).
        dewa_tag = f"bd{b_d}_deva"
        # Reuse T1 side-deva if present.
        t1_deva = REPO_ROOT / "artifacts/eval/mix_sweep/t1_block1_cap" / f"bd{b_d}_5pct" / "side-deva"
        dewa_out = OUT / dewa_tag / "side-deva"
        if t1_deva.is_dir() and (t1_deva / "ranks.json").is_file():
            if not (dewa_out / "ranks.json").is_file():
                dewa_out.mkdir(parents=True, exist_ok=True)
                import shutil

                for name in ("ranks.json", "tokenizer.json", "manifest.json", "tiktoken"):
                    src = t1_deva / name
                    if src.exists():
                        if src.is_dir():
                            shutil.copytree(src, dewa_out / name, dirs_exist_ok=True)
                        else:
                            shutil.copy2(src, dewa_out / name)
                # export artifacts if missing
                if not (dewa_out / "tokenizer.json").is_file():
                    from ancoda_tok.export_cli import export_dir

                    export_dir(dewa_out / "ranks.json")
            print(f"reused T1 dewa side for B_d={b_d}", flush=True)
        else:
            train_sides(
                out_root=OUT,
                tag=dewa_tag,
                b_d=b_d,
                mergeable=MERGEABLE,
                block0=BLOCK0,
                block1=block1,
                block1_path="artifacts/inventory/covered_64k.json",
                dewa_counts_rel="artifacts/eval/mix_sweep/t2_latin_rebalance/deva_pretoken_counts.bin",
                latin_counts_rel="artifacts/eval/mix_sweep/t2_latin_rebalance/latin_pretoken_counts.bin",
            )
            # train_sides trains both; we only need dewa — ok.

        for scale in SCALES:
            tag = f"bd{b_d}_cm{int(scale * 100)}"
            print(f"=== T2 B_d={b_d} code+math scale={scale} ===", flush=True)
            latin_counts = build_scaled_latin(scale)
            latin_b2 = MERGEABLE - 256 - b_d
            latin_vocab = 256 + latin_b2 + 1024
            latin_out = OUT / tag / "side-latin"
            latin_cfg = write_side_config(
                path=OUT / "configs" / f"{tag}_latin.yaml",
                counts_rel=str(latin_counts.relative_to(REPO_ROOT)),
                output_rel=str(latin_out.relative_to(REPO_ROOT)),
                vocab_size=latin_vocab,
                block0=False,
                block1=False,
                block1_path="artifacts/inventory/covered_64k.json",
            )
            train_one_side(latin_cfg, latin_out)
            dest = OUT / tag / "combined"
            combine(dewa_out if (dewa_out / "ranks.json").is_file() else OUT / dewa_tag / "side-deva", latin_out, dest, mergeable=MERGEABLE)
            if not args.skip_guarantee:
                g = guarantee(
                    dest,
                    report=OUT / f"guarantee_{tag}.json",
                    orphans=OUT / f"orphans_{tag}.txt",
                    covered=COVERED,
                )
                viol = g.get("hf_covered_violations")
                fails = g.get("failures")
            else:
                viol = fails = None
            name = f"t2_{tag}"
            adds[name] = dest
            rows.append(
                {
                    "arm": "T2",
                    "label": f"Bd{b_d}_cm{int(scale * 100)}",
                    "B_d": b_d,
                    "code_math_scale": scale,
                    "dir": str(dest.relative_to(REPO_ROOT)),
                    "hf_covered_violations": viol,
                    "guarantee_failures": fails,
                    "code_tokens_per_kb": tokens_per_kb(dest, code_texts),
                    "math_tokens_per_kb": tokens_per_kb(dest, math_texts),
                }
            )
    run_harness(adds, "t2")
    for row in rows:
        enrich_from_harness(
            row,
            "t2",
            f"t2_bd{row['B_d']}_cm{int(row['code_math_scale'] * 100)}",
        )
        attach_extra_metrics(
            row,
            REPO_ROOT / row["dir"],
            snap=N60_SNAP,
            harness_aya=row.get("aya_fertility"),
        )
    save_json(
        OUT / "frontier.json",
        finish_manifest(base_manifest(kind="t2_latin_rebalance", rows=rows)),
    )
    print(json.dumps({"rows": rows}, indent=2), flush=True)


if __name__ == "__main__":
    main()

"""T1: Block-1 cap ablations at B_d=46000 and 48000 (64k; diagnostic)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ancoda_tok.clean.common import base_manifest, finish_manifest, save_json
from ancoda_tok.frontier_common import (
    combine,
    enrich_from_harness,
    guarantee,
    held_out_code_math,
    run_harness,
    split_counts,
    tokens_per_kb,
    train_sides,
)
from ancoda_tok.frontier_metrics import attach_extra_metrics
from ancoda_tok.storage import REPO_ROOT
N60_SNAP = REPO_ROOT / "data" / "snapshots" / "n60_cf8066dbbdd0da93"

OUT = REPO_ROOT / "artifacts" / "eval" / "m7" / "t1_block1_cap"
MERGEABLE = 64512
BLOCK0 = 133
BDS = (46000, 48000)
CAPS = ("5pct", "3pct", "2pct")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-guarantee", action="store_true")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    split_counts(OUT)
    code_texts, math_texts = held_out_code_math()
    rows = []
    adds: dict[str, Path] = {}
    for b_d in BDS:
        for pct in CAPS:
            covered_path = OUT / f"covered_64k_{pct}.json"
            covered = json.loads(covered_path.read_text(encoding="utf-8"))
            block1 = int(covered["block1_merge_count"])
            coverage = covered["coverage"]
            tag = f"bd{b_d}_{pct}"
            print(
                f"=== T1 B_d={b_d} cap={pct} block1={block1} coverage={coverage:.10f} ===",
                flush=True,
            )
            dewa_dir, latin_dir = train_sides(
                out_root=OUT,
                tag=tag,
                b_d=b_d,
                mergeable=MERGEABLE,
                block0=BLOCK0,
                block1=block1,
                block1_path=str(covered_path.relative_to(REPO_ROOT)),
                dewa_counts_rel="artifacts/eval/m7/t1_block1_cap/deva_pretoken_counts.bin",
                latin_counts_rel="artifacts/eval/m7/t1_block1_cap/latin_pretoken_counts.bin",
            )
            dest = OUT / tag / "combined"
            combine(dewa_dir, latin_dir, dest, mergeable=MERGEABLE)
            if not args.skip_guarantee:
                g = guarantee(
                    dest,
                    report=OUT / f"guarantee_{tag}.json",
                    orphans=OUT / f"orphans_{tag}.txt",
                    covered=covered_path,
                )
                viol = g.get("hf_covered_violations")
                fails = g.get("failures")
                tail = g.get("tail_hf_rate_lo_hi")
                cov_fail = g.get("coverage_failures")
                orphan_occ = g.get("orphan_occurrences_npi_aya")
            else:
                viol = fails = tail = cov_fail = orphan_occ = None
            name = f"t1_{tag}"
            adds[name] = dest
            rows.append(
                {
                    "arm": "T1",
                    "label": f"Bd{b_d}_block1_{pct}",
                    "B_d": b_d,
                    "block1_cap": pct,
                    "block1": block1,
                    "block1_coverage": coverage,
                    "dir": str(dest.relative_to(REPO_ROOT)),
                    "hf_covered_violations": viol,
                    "coverage_failures": cov_fail,
                    "tail_hf_rate_lo_hi": tail,
                    "guarantee_failures": fails,
                    "orphan_occurrences_npi_aya": orphan_occ,
                    "code_tokens_per_kb": tokens_per_kb(dest, code_texts),
                    "math_tokens_per_kb": tokens_per_kb(dest, math_texts),
                }
            )
    run_harness(adds, "t1")
    for row in rows:
        enrich_from_harness(row, "t1", f"t1_bd{row['B_d']}_{row['block1_cap']}")
        attach_extra_metrics(
            row,
            REPO_ROOT / row["dir"],
            snap=N60_SNAP,
            harness_aya=row.get("aya_fertility"),
        )
    save_json(
        OUT / "frontier.json",
        finish_manifest(base_manifest(kind="t1_block1_cap", rows=rows)),
    )
    print(json.dumps({"rows": rows}, indent=2), flush=True)


if __name__ == "__main__":
    main()

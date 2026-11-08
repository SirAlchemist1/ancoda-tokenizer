"""T4: 128k script-budget frontier on n60 counts (diagnostic; no shipping change)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ancoda_tok.clean.common import base_manifest, finish_manifest, save_json, sha256_file
from ancoda_tok.frontier_common import (
    combine,
    enrich_from_harness,
    guarantee,
    held_out_code_math,
    measure_bd,
    run_harness,
    split_counts,
    tokens_per_kb,
    train_sides,
)
from ancoda_tok.frontier_metrics import attach_extra_metrics
from ancoda_tok.storage import REPO_ROOT
N60_SNAP = REPO_ROOT / "data" / "snapshots" / "n60_cf8066dbbdd0da93"

OUT = REPO_ROOT / "artifacts" / "eval" / "m7" / "script_budget_128k"
COVERED = REPO_ROOT / "artifacts" / "inventory" / "covered_128k.json"
ANC128 = REPO_ROOT / "artifacts" / "tokenizers" / "ancoda-128k-n60"
YAK = (
    REPO_ROOT
    / "data/hf_cache/hub/models--ampixa--yak-tokenizer/snapshots"
    / "12b6d3010ec542b484ec3b673354852fe8bb85fb"
)
MERGEABLE = 130048
BLOCK0 = 133


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-guarantee", action="store_true")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    split_counts(OUT)
    current = measure_bd(ANC128 / "ranks.json")
    save_json(
        OUT / "current_128k_split.json",
        finish_manifest(base_manifest(kind="t4_current_128k", **current)),
    )
    print("T4 current 128k Devanagari split:", json.dumps(current), flush=True)
    b0 = int(current["B_d"])
    budgets = [b0, b0 + 6000, b0 + 12000, b0 + 18000, b0 + 24000]
    covered = json.loads(COVERED.read_text(encoding="utf-8"))
    block1 = int(covered["block1_merge_count"])
    code_texts, math_texts = held_out_code_math()
    rows = []
    adds: dict[str, Path] = {}
    for b_d in budgets:
        tag = f"bd{b_d}"
        print(f"=== T4 budget B_d={b_d} ===", flush=True)
        dewa_dir, latin_dir = train_sides(
            out_root=OUT,
            tag=tag,
            b_d=b_d,
            mergeable=MERGEABLE,
            block0=BLOCK0,
            block1=block1,
            block1_path="artifacts/inventory/covered_128k.json",
            dewa_counts_rel="artifacts/eval/m7/script_budget_128k/deva_pretoken_counts.bin",
            latin_counts_rel="artifacts/eval/m7/script_budget_128k/latin_pretoken_counts.bin",
        )
        dest = OUT / tag / "combined"
        combine(dewa_dir, latin_dir, dest, mergeable=MERGEABLE)
        if not args.skip_guarantee:
            g = guarantee(
                dest,
                report=OUT / f"guarantee_{tag}.json",
                orphans=OUT / f"orphans_{tag}.txt",
                covered=COVERED,
            )
            viol = g.get("hf_covered_violations")
            fails = g.get("failures")
            orphan_occ = g.get("orphan_occurrences_npi_aya")
        else:
            viol = fails = orphan_occ = None
        name = f"t4_{tag}"
        adds[name] = dest
        rows.append(
            {
                "arm": "T4",
                "label": f"128k_interleave_Bd{b_d}",
                "B_d": b_d,
                "dir": str(dest.relative_to(REPO_ROOT)),
                "hf_covered_violations": viol,
                "guarantee_failures": fails,
                "orphan_occurrences_npi_aya": orphan_occ,
                "code_tokens_per_kb": tokens_per_kb(dest, code_texts),
                "math_tokens_per_kb": tokens_per_kb(dest, math_texts),
            }
        )

    # Shipping anc128 + Yak for table / F verify
    adds["anc128_n60"] = ANC128
    adds["yak_v1_128k"] = YAK / "v1.0/128k/tokenizer.json"
    adds["yak_v2_128k"] = YAK / "v2.0/128k/tokenizer.json"
    run_harness(adds, "t4_128k")
    for row in rows:
        enrich_from_harness(row, "t4_128k", f"t4_bd{row['B_d']}")
        attach_extra_metrics(
            row,
            REPO_ROOT / row["dir"],
            snap=N60_SNAP,
            harness_aya=row.get("aya_fertility"),
        )

    # Extra rows: shipping + Yak
    extra = []
    for name, path in (
        ("anc128_n60", ANC128 / "tokenizer.json"),
        ("yak_v1_128k", YAK / "v1.0/128k/tokenizer.json"),
        ("yak_v2_128k", YAK / "v2.0/128k/tokenizer.json"),
    ):
        row = {
            "arm": "T4",
            "label": name,
            "B_d": current["B_d"] if name == "anc128_n60" else None,
            "dir": str(path.relative_to(REPO_ROOT)),
            "path_sha256": sha256_file(path),
            "code_tokens_per_kb": tokens_per_kb(path, code_texts),
            "math_tokens_per_kb": tokens_per_kb(path, math_texts),
        }
        enrich_from_harness(row, "t4_128k", name)
        attach_extra_metrics(
            row,
            path.parent if path.name == "tokenizer.json" else path,
            snap=N60_SNAP,
            harness_aya=row.get("aya_fertility"),
        )
        extra.append(row)

    # F verify: Hindi equality with fingerprint
    f_verify = {}
    for name in ("anc128_n60", "yak_v1_128k"):
        raw = REPO_ROOT / "artifacts" / "eval" / "m7" / f"t4_128k_{name}.json"
        rec = json.loads(raw.read_text(encoding="utf-8"))
        hin = rec["results"]["flores_devtest/hin_Deva"]["fertility"]
        path = (
            ANC128 / "tokenizer.json"
            if name == "anc128_n60"
            else YAK / "v1.0/128k/tokenizer.json"
        )
        f_verify[name] = {
            "path": str(path),
            "sha256": sha256_file(path),
            "fingerprint": rec.get("fingerprint"),
            "hin_fertility": hin,
            "harness_artifact": str(raw.relative_to(REPO_ROOT)),
        }
    equal = abs(f_verify["anc128_n60"]["hin_fertility"] - f_verify["yak_v1_128k"]["hin_fertility"]) < 1e-12
    print("F verify Hindi equal:", equal, json.dumps(f_verify, indent=2), flush=True)

    out = finish_manifest(
        base_manifest(
            kind="t4_128k_frontier",
            current_128k=current,
            rows=rows + extra,
            f_verify_hindi=f_verify,
            f_verify_equal=equal,
        )
    )
    save_json(OUT / "frontier.json", out)
    print(json.dumps({"current": current, "rows": rows + extra}, indent=2), flush=True)


if __name__ == "__main__":
    main()

"""Build n60j5_nojoiner: same selection as n60j5, joiner 1% → natural Nepali.

Matched to data/snapshots/n60j5_20af1fffbec0fe86 (diag_combined_candidates.build_5gb_n60_joiner)
with joiner_share=0 and Nepali target = full 60% of the 5 GB budget.
Does not touch the server queue.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from ancoda_tok.clean.audit_snapshots import audit_snapshot
from ancoda_tok.clean.common import Stop, free_bytes, load_mix, save_json
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
from ancoda_tok.m9.common import REPO_ROOT

NAME = "n60j5_nojoiner"
BUDGET = 5_000_000_000
J5 = REPO_ROOT / "data" / "snapshots" / "n60j5_20af1fffbec0fe86"
OUT_NOTE = REPO_ROOT / "artifacts" / "analysis" / "n60j5_nojoiner_build.json"


def main() -> None:
    existing = list((REPO_ROOT / "data" / "snapshots").glob(f"{NAME}_*"))
    if len(existing) == 1 and (existing[0] / "manifest.json").is_file():
        print(f"resume snapshot {existing[0]}", flush=True)
        print(existing[0])
        return

    free = free_bytes()
    if free - BUDGET < 50_000_000_000:
        # Soft check: project rule is stop if fewer than 50 GB would remain after >20 GB write.
        # This snapshot is ~1.3 GB compressed / 5 GB text; still gate on free headroom.
        print(f"WARN free_bytes={free}; proceeding if free > 60 GB", flush=True)
    if free < 55_000_000_000:
        raise Stop(f"STOP: free_bytes={free} under 55 GB")

    mix = load_mix()
    temperature = float(mix["temperature"])
    nepali_share = 0.60
    index_path = build_index(mix)
    index_rows = load_index_rows(index_path)
    by_bucket: dict[str, list] = defaultdict(list)
    for row in index_rows:
        by_bucket[row["bucket"]].append(row)
    shares = scale_shares(mix, nepali_share)
    # Matched no-joiner: full Nepali share is natural (DECISIONS 3.8 / M7 follow-up).
    targets = {b: int(BUDGET * s) for b, s in shares.items()}
    selected_index = []
    for bucket, target in targets.items():
        picked = pool_sample_index(
            by_bucket.get(bucket, []),
            target,
            seed=SEED + int(nepali_share * 1000) + hashlib_int(bucket) + 20_000,
            temperature=temperature,
        )
        selected_index.extend(picked)
        print(f"picked {bucket} docs={len(picked)} target_bytes={target}", flush=True)
    chosen = fetch_texts(selected_index)
    mix_meta = {
        "nepali_share": nepali_share,
        "shares": shares,
        "joiner_share": 0.0,
        "budget": BUDGET,
        "diagnostic": "matched_n60j5_nojoiner",
        "matched_to": str(J5.relative_to(REPO_ROOT)),
        "note": (
            "Same seed offsets as build_5gb_n60_joiner, but Nepali target is the "
            "full 60% as natural text (no synthetic joiner copy)."
        ),
        "synthetic": False,
    }
    snap = write_snapshot(NAME, chosen, mix_meta=mix_meta)
    counts = audit_snapshot(snap)
    print(f"audit {NAME} {counts}", flush=True)
    if any(v != 0 for v in counts.values()):
        raise Stop(f"STOP: snapshot {NAME} failed decontam audit: {counts}")

    # Record overlap with n60j5 natural docs.
    import pyarrow.parquet as pq

    def ids(path: Path) -> set[str]:
        out: set[str] = set()
        for p in sorted(path.glob("shard-*.parquet")):
            out.update(pq.read_table(p, columns=["doc_id"]).column("doc_id").to_pylist())
        return out

    j5_ids = ids(J5)
    synth = {d for d in j5_ids if str(d).startswith("synthetic")}
    natural_j5 = j5_ids - synth
    new_ids = ids(snap)
    note = {
        "snapshot": str(snap.relative_to(REPO_ROOT)),
        "docs": len(new_ids),
        "j5_docs": len(j5_ids),
        "j5_synth": len(synth),
        "j5_natural": len(natural_j5),
        "intersection_with_j5_natural": len(new_ids & natural_j5),
        "only_in_nojoiner": len(new_ids - natural_j5),
        "only_in_j5_natural": len(natural_j5 - new_ids),
        "n60_is_matched": False,
        "n60_check": (
            "Existing n60_cf8066dbbdd0da93 shares only ~685k docs with n60j5 natural; "
            "not a joiner-swap. Built this snapshot instead."
        ),
        "audit": counts,
        "mix_meta": mix_meta,
    }
    OUT_NOTE.parent.mkdir(parents=True, exist_ok=True)
    save_json(OUT_NOTE, note)
    man = json.loads((snap / "manifest.json").read_text(encoding="utf-8"))
    print(json.dumps({"path": str(snap), "docs": man["docs"], "text_bytes": man["text_bytes"]}, indent=2))


if __name__ == "__main__":
    main()

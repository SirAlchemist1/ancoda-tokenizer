"""MinHash near-duplicate check of training docs against large web eval sets.

Same MinHash settings as M4 step 4 (mix.yaml minhash + seed 20261104).
Drops a training document if any of its 14 band signatures collides with a
band from fw2_npi_test, fw2_hin_test or c4_val.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from ancoda_tok.clean.common import (
    CLEAN_SCHEMA,
    REPO_ROOT,
    SEED,
    ShardWriter,
    Stop,
    artifacts_clean,
    base_manifest,
    clean_dir,
    ensure_room,
    finish_manifest,
    free_bytes,
    iter_parquet_rows,
    list_shards,
    load_json,
    load_mix,
    raw_dir,
    save_json,
    train_source_ids,
    utf8_len,
)

_EXACT_ONLY_EVAL_SOURCES = ("fw2_npi_test", "fw2_hin_test", "c4_val")


def _patch_xxhash() -> None:
    from ancoda_tok.clean.dedup import _patch_xxhash_for_datatrove

    _patch_xxhash_for_datatrove()


def _minhash_config(mix: dict):
    from datatrove.pipeline.dedup.minhash import MinhashConfig

    mh = mix["minhash"]
    return MinhashConfig(
        n_grams=int(mh["n_grams"]),
        num_buckets=int(mh["num_buckets"]),
        hashes_per_bucket=int(mh["hashes_per_bucket"]),
        seed=SEED,
    )


def _signature_helper(config):
    from datatrove.pipeline.dedup import MinhashDedupSignature
    from datatrove.utils.typeshelper import Languages

    return MinhashDedupSignature(
        output_folder=str(REPO_ROOT / "data" / "clean" / "_minhash_eval" / "unused_sigs"),
        config=config,
        language=Languages.nepali_individual_language__deva,
    )


def eval_bands_dir() -> Path:
    return REPO_ROOT / "data" / "clean" / "_minhash_eval" / "eval_bands"


def build_or_load_eval_bands(mix: dict | None = None) -> list[set[tuple[int, ...]]]:
    """Build (or load) per-bucket band sets for the three large eval sources."""
    mix = mix or load_mix()
    out = eval_bands_dir()
    meta_path = out / "manifest.json"
    n_buckets = int(mix["minhash"]["num_buckets"])
    band_paths = [out / f"bucket_{i:03d}.jsonl" for i in range(n_buckets)]
    if meta_path.exists() and all(p.exists() for p in band_paths):
        print(f"loading eval MinHash bands from {out}", flush=True)
        bands: list[set[tuple[int, ...]]] = []
        for path in band_paths:
            s: set[tuple[int, ...]] = set()
            with path.open(encoding="utf-8") as fh:
                for line in fh:
                    s.add(tuple(json.loads(line)))
            bands.append(s)
        return bands

    _patch_xxhash()
    config = _minhash_config(mix)
    helper = _signature_helper(config)
    bands = [set() for _ in range(n_buckets)]
    n_docs = 0
    n_empty = 0
    ensure_room(2_000_000_000, label="minhash_eval_bands")
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    for source_id in _EXACT_ONLY_EVAL_SOURCES:
        directory = raw_dir(source_id)
        if not directory.exists():
            raise Stop(f"STOP: missing eval source for MinHash near-dup: {source_id}")
        print(f"minhash-eval indexing {source_id}", flush=True)
        for row in iter_parquet_rows(directory):
            shingles = helper.get_shingles(row["text"])
            n_docs += 1
            if shingles.size == 0:
                n_empty += 1
                continue
            for bi, band in enumerate(helper.get_signature(shingles)):
                bands[bi].add(tuple(int(x) for x in band))
            if n_docs % 10_000 == 0:
                print(
                    f"  indexed docs={n_docs} band0_size={len(bands[0])} "
                    f"free_bytes={free_bytes()}",
                    flush=True,
                )

    for bi, path in enumerate(band_paths):
        with path.open("w", encoding="utf-8") as fh:
            for band in sorted(bands[bi]):
                fh.write(json.dumps(list(band)) + "\n")
    man = {
        **base_manifest(stage="minhash_eval_bands"),
        "eval_sources": list(_EXACT_ONLY_EVAL_SOURCES),
        "n_docs": n_docs,
        "n_empty_shingles": n_empty,
        "band_sizes": [len(b) for b in bands],
        "config": {
            "n_grams": int(mix["minhash"]["n_grams"]),
            "num_buckets": n_buckets,
            "hashes_per_bucket": int(mix["minhash"]["hashes_per_bucket"]),
            "seed": SEED,
        },
    }
    save_json(meta_path, finish_manifest(man))
    print(
        f"eval MinHash bands ready docs={n_docs} sizes={[len(b) for b in bands]}",
        flush=True,
    )
    return bands


def doc_minhash_collides(text: str, helper, bands: list[set[tuple[int, ...]]]) -> bool:
    shingles = helper.get_shingles(text)
    if shingles.size == 0:
        return False
    for bi, band in enumerate(helper.get_signature(shingles)):
        if tuple(int(x) for x in band) in bands[bi]:
            return True
    return False


# Process-pool worker state (initialized once per child).
_WORKER_HELPER = None
_WORKER_BANDS: list[set[tuple[int, ...]]] | None = None


def _init_filter_worker(bands: list[set[tuple[int, ...]]]) -> None:
    global _WORKER_HELPER, _WORKER_BANDS
    _patch_xxhash()
    mix = load_mix()
    _WORKER_HELPER = _signature_helper(_minhash_config(mix))
    _WORKER_BANDS = bands


def _filter_one_shard(args: tuple[str, str]) -> dict[str, Any]:
    """Filter one parquet shard; write kept rows to out_path. Returns stats."""
    shard_s, out_s = args
    shard = Path(shard_s)
    out_path = Path(out_s)
    assert _WORKER_HELPER is not None and _WORKER_BANDS is not None
    table = pq.read_table(shard)
    cols = {name: table.column(name) for name in table.column_names}
    kept_rows: list[dict[str, Any]] = []
    scanned = 0
    dropped = 0
    dropped_bytes = 0
    for i in range(table.num_rows):
        row = {name: cols[name][i].as_py() for name in cols}
        scanned += 1
        text = row["text"]
        if doc_minhash_collides(text, _WORKER_HELPER, _WORKER_BANDS):
            dropped += 1
            dropped_bytes += utf8_len(text)
            continue
        kept_rows.append(row)
    if kept_rows:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(
            __import__("pyarrow").Table.from_pylist(kept_rows, schema=CLEAN_SCHEMA),
            out_path,
            compression="zstd",
        )
    elif out_path.exists():
        out_path.unlink()
    return {
        "shard": shard_s,
        "scanned": scanned,
        "dropped": dropped,
        "dropped_bytes": dropped_bytes,
        "kept": len(kept_rows),
        "out": str(out_path) if kept_rows else "",
    }


def _filter_directory(
    src: Path,
    *,
    label: str,
    bands: list[set[tuple[int, ...]]],
    report: dict[str, Any],
) -> None:
    """Drop docs whose MinHash collides with a large eval set; rewrite in place."""
    from concurrent.futures import ProcessPoolExecutor, as_completed

    from ancoda_tok.clean.common import workers

    shards = list_shards(src)
    if not shards:
        return
    safe = label.replace("/", "_").replace(" ", "_")
    tmp = REPO_ROOT / "data" / "clean" / f".tmp_minhash_eval_{safe}"
    tmp.mkdir(parents=True, exist_ok=True)
    progress_path = tmp / "progress.json"
    progress: dict[str, Any] = {
        "completed_shards": [],
        "scanned": 0,
        "dropped": 0,
        "dropped_bytes": 0,
    }
    if progress_path.exists():
        progress = load_json(progress_path)
    completed = set(progress.get("completed_shards", []))

    scanned = int(progress.get("scanned", 0))
    dropped = int(progress.get("dropped", 0))
    dropped_bytes = int(progress.get("dropped_bytes", 0))
    pending = [s for s in shards if str(s.relative_to(REPO_ROOT)) not in completed]
    n_workers = min(workers(), max(1, len(pending)))
    print(
        f"minhash-eval {label}: {len(shards)} shards "
        f"(done={len(completed)} pending={len(pending)} workers={n_workers})",
        flush=True,
    )

    jobs = [
        (str(s), str(tmp / s.name))
        for s in pending
    ]
    if jobs:
        with ProcessPoolExecutor(
            max_workers=n_workers,
            initializer=_init_filter_worker,
            initargs=(bands,),
        ) as pool:
            futs = {pool.submit(_filter_one_shard, job): job for job in jobs}
            done = 0
            for fut in as_completed(futs):
                stats = fut.result()
                done += 1
                scanned += stats["scanned"]
                dropped += stats["dropped"]
                dropped_bytes += stats["dropped_bytes"]
                rel = str(Path(stats["shard"]).relative_to(REPO_ROOT))
                completed.add(rel)
                progress = {
                    "completed_shards": sorted(completed),
                    "scanned": scanned,
                    "dropped": dropped,
                    "dropped_bytes": dropped_bytes,
                }
                save_json(progress_path, progress)
                if done % 20 == 0 or done == len(jobs):
                    print(
                        f"  {label} shards={len(completed)}/{len(shards)} "
                        f"scanned={scanned} dropped={dropped} free_bytes={free_bytes()}",
                        flush=True,
                    )

    writer = ShardWriter(tmp / "_final", schema=CLEAN_SCHEMA)
    for path in sorted(p for p in tmp.glob("shard-*.parquet")):
        table = pq.read_table(path)
        cols = {name: table.column(name) for name in table.column_names}
        for i in range(table.num_rows):
            writer.add({name: cols[name][i].as_py() for name in cols})
    writer.close()
    final = tmp / "_final"
    for old in list_shards(src):
        old.unlink()
    for shard in list_shards(final):
        shard.rename(src / shard.name)

    man_path = src / "manifest.json"
    if man_path.exists():
        man = load_json(man_path)
        man["docs"] = writer.docs
        man["text_bytes"] = writer.text_bytes
        man["shards"] = writer.shards
        man["minhash_eval_dropped_docs"] = dropped
        man["minhash_eval_dropped_bytes"] = dropped_bytes
        save_json(man_path, finish_manifest(man))
    shutil.rmtree(tmp)

    report.setdefault("sources", {})[label] = {
        "scanned_docs": scanned,
        "dropped_docs": dropped,
        "dropped_bytes": dropped_bytes,
        "kept_docs": writer.docs,
        "kept_bytes": writer.text_bytes,
    }
    print(
        f"minhash-eval {label} kept={writer.docs} dropped={dropped} "
        f"dropped_bytes={dropped_bytes} scanned={scanned}",
        flush=True,
    )


def run_minhash_eval_near_dup() -> dict[str, Any]:
    """Drop training docs that MinHash-collide with large web eval sets."""
    mix = load_mix()
    bands = build_or_load_eval_bands(mix)
    report: dict[str, Any] = {
        **base_manifest(stage="minhash_eval_near_dup"),
        "eval_sources": list(_EXACT_ONLY_EVAL_SOURCES),
        "band_sizes": [len(b) for b in bands],
        "note": (
            "Prior M4 MinHash signature workdir was not on disk; "
            "eval bands rebuilt with the same MinHash config as step 4."
        ),
        "sources": {},
    }
    for source_id in train_source_ids(mix):
        directory = clean_dir(source_id)
        if directory.exists():
            _filter_directory(directory, label=source_id, bands=bands, report=report)
    reroute = REPO_ROOT / "data" / "clean" / "_reroute"
    if reroute.exists():
        for bucket_dir in sorted(p for p in reroute.iterdir() if p.is_dir()):
            for src_dir in sorted(p for p in bucket_dir.iterdir() if p.is_dir()):
                label = f"_reroute/{bucket_dir.name}/{src_dir.name}"
                _filter_directory(src_dir, label=label, bands=bands, report=report)

    total_dropped = sum(v["dropped_docs"] for v in report["sources"].values())
    total_bytes = sum(v["dropped_bytes"] for v in report["sources"].values())
    report["total_dropped_docs"] = total_dropped
    report["total_dropped_bytes"] = total_bytes
    report = finish_manifest(report)
    save_json(artifacts_clean() / "minhash_eval_near_dup.json", report)

    main_path = artifacts_clean() / "report.json"
    if main_path.exists():
        main = load_json(main_path)
        main["minhash_eval_near_dup"] = {
            "total_dropped_docs": total_dropped,
            "total_dropped_bytes": total_bytes,
            "sources": report["sources"],
        }
        save_json(main_path, main)

    print(
        f"minhash-eval near-dup done dropped_docs={total_dropped} "
        f"dropped_bytes={total_bytes} free_bytes={free_bytes()}",
        flush=True,
    )
    return report


def _audit_one_shard_minhash(shard_s: str) -> int:
    assert _WORKER_HELPER is not None and _WORKER_BANDS is not None
    table = pq.read_table(shard_s, columns=["text"])
    texts = table.column("text")
    n = 0
    for i in range(table.num_rows):
        if doc_minhash_collides(texts[i].as_py(), _WORKER_HELPER, _WORKER_BANDS):
            n += 1
    return n


def audit_minhash_snapshot(snapshot_dir: Path, bands=None, helper=None) -> int:
    """Count snapshot docs that MinHash-collide with large eval sets (must be 0)."""
    from concurrent.futures import ProcessPoolExecutor, as_completed

    from ancoda_tok.clean.common import workers

    mix = load_mix()
    if bands is None:
        bands = build_or_load_eval_bands(mix)
    shards = list_shards(snapshot_dir)
    if not shards:
        return 0
    # Prefer process-pool (same as filter). Fall back to sequential helper if given.
    if helper is not None:
        n = 0
        for shard in shards:
            table = pq.read_table(shard, columns=["text"])
            texts = table.column("text")
            for i in range(table.num_rows):
                if doc_minhash_collides(texts[i].as_py(), helper, bands):
                    n += 1
        return n
    n_workers = min(workers(), max(1, len(shards)))
    total = 0
    done = 0
    with ProcessPoolExecutor(
        max_workers=n_workers,
        initializer=_init_filter_worker,
        initargs=(bands,),
    ) as pool:
        futs = {pool.submit(_audit_one_shard_minhash, str(s)): s for s in shards}
        for fut in as_completed(futs):
            total += fut.result()
            done += 1
            if done % 20 == 0 or done == len(shards):
                print(
                    f"  minhash-audit {snapshot_dir.name} "
                    f"shards={done}/{len(shards)} hits={total}",
                    flush=True,
                )
    return total


if __name__ == "__main__":
    run_minhash_eval_near_dup()
    print("done", flush=True)

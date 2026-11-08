"""M4 step 4: exact sha256 then MinHash dedup across Nepali sources."""

from __future__ import annotations

import array
import struct
import shutil
from pathlib import Path
from typing import Any

from datatrove.executor.local import LocalPipelineExecutor
from datatrove.pipeline.dedup import (
    MinhashDedupBuckets,
    MinhashDedupCluster,
    MinhashDedupFilter,
    MinhashDedupSignature,
)
from datatrove.pipeline.dedup.minhash import MinhashConfig
from datatrove.pipeline.readers import ParquetReader
from datatrove.pipeline.writers import ParquetWriter
from datatrove.utils.typeshelper import Languages

from ancoda_tok.clean.common import (
    CLEAN_SCHEMA,
    REPO_ROOT,
    SEED,
    ShardWriter,
    artifacts_clean,
    base_manifest,
    clean_dir,
    finish_manifest,
    free_bytes,
    iter_parquet_rows,
    list_shards,
    load_json,
    load_mix,
    save_json,
    sha256_text,
    utf8_len,
    workers,
)

NEPALI_SOURCES_PREF = None  # filled from mix


def _preference_rank(mix: dict) -> dict[str, int]:
    return {sid: i for i, sid in enumerate(mix["nepali_dedup_preference"])}


def _patch_xxhash_for_datatrove() -> None:
    """xxhash>=4 requires bytes; datatrove still passes str. Encode UTF-8."""
    import datatrove.utils.hashes.xxhash as dt_xx
    import xxhash as _xx

    def xxhash64(data: str | bytes) -> int:
        if isinstance(data, str):
            data = data.encode("utf-8")
        return _xx.xxh64_intdigest(data)

    def xxhash32(data: str | bytes) -> int:
        if isinstance(data, str):
            data = data.encode("utf-8")
        return _xx.xxh32_intdigest(data)

    dt_xx.xxhash64 = xxhash64
    dt_xx.xxhash32 = xxhash32


def _rewrite_removes_for_preference(
    remove_dir: Path,
    staged: Path,
    pref: dict[str, int],
) -> dict[str, Any]:
    """Replace MinHash .remove lists so each cluster keeps the preferred source.

    datatrove's union-find keeper is size-based, not source-preference. PLAN.md
    requires preference order sangraha → iriis → indiccorp → fw2 → hplt.
    """
    import pyarrow.parquet as pq

    shards = list_shards(staged)
    file_ranks: list[array.array] = []
    for shard in shards:
        ranks = array.array("H")
        table = pq.read_table(shard, columns=["source"])
        for src in table.column("source").to_pylist():
            ranks.append(int(pref.get(src, 999)))
        file_ranks.append(ranks)

    clusters: dict[int, list[tuple[int, int]]] = {}
    for path in sorted(remove_dir.glob("*.clusters")):
        file_id = int(path.name.split(".", 1)[0])
        with path.open("rb") as fh:
            while True:
                blob = fh.read(8)
                if len(blob) < 8:
                    break
                doc_idx, cluster_id = struct.unpack("<2I", blob)
                clusters.setdefault(cluster_id, []).append((file_id, doc_idx))

    removes: dict[int, list[int]] = {}
    kept = 0
    dropped = 0
    for members in clusters.values():
        keeper = min(
            members,
            key=lambda fd: (file_ranks[fd[0]][fd[1]], fd[0], fd[1]),
        )
        kept += 1
        for file_id, doc_idx in members:
            if (file_id, doc_idx) != keeper:
                removes.setdefault(file_id, []).append(doc_idx)
                dropped += 1

    for path in remove_dir.glob("*.remove"):
        path.unlink()
    for file_id, docs in removes.items():
        docs_sorted = sorted(set(docs))
        with (remove_dir / f"{file_id:06d}.remove").open("wb") as fh:
            for doc_idx in docs_sorted:
                fh.write(struct.pack("<I", doc_idx))

    return {
        "clusters": len(clusters),
        "preference_keepers": kept,
        "preference_removed": dropped,
    }


def exact_dedup_nepali(mix: dict) -> dict[str, Any]:
    """Keep one doc per sha256 of UTF-8 text, preference order from mix."""
    pref = _preference_rank(mix)
    sources = [s for s in mix["nepali_dedup_preference"] if clean_dir(s).exists()]
    # Also include any Nepali bucket source not in the preference list.
    for sid in mix["buckets"]["Nepali"]["sources"]:
        if sid not in sources and clean_dir(sid).exists():
            sources.append(sid)
            pref.setdefault(sid, len(pref))

    # digest -> best (pref_rank, source_id, doc_id, text_bytes). No raw text in RAM.
    stats = {"docs_in": 0, "bytes_in": 0, "docs_out": 0, "bytes_out": 0, "dropped": 0}
    keep_ids: dict[str, set[str]] = {s: set() for s in sources}
    best: dict[str, tuple[int, str, str, int]] = {}

    for source_id in sources:
        rank = pref.get(source_id, 999)
        for row in iter_parquet_rows(clean_dir(source_id)):
            text = row["text"]
            nbytes = utf8_len(text)
            digest = sha256_text(text)
            stats["docs_in"] += 1
            stats["bytes_in"] += nbytes
            cand = (rank, source_id, row["doc_id"], nbytes)
            prev = best.get(digest)
            if prev is None:
                best[digest] = cand
            else:
                stats["dropped"] += 1
                # Lower preference rank wins; then source_id, then doc_id.
                if cand[:3] < prev[:3]:
                    best[digest] = cand

    for rank, source_id, doc_id, nbytes in best.values():
        keep_ids[source_id].add(doc_id)
        stats["docs_out"] += 1
        stats["bytes_out"] += nbytes
    # Digests with a single copy never incremented dropped; duplicates did.

    # Rewrite each source keeping winners only.
    for source_id in sources:
        src = clean_dir(source_id)
        tmp = REPO_ROOT / "data" / "clean" / f".tmp_exact_{source_id}"
        if tmp.exists():
            shutil.rmtree(tmp)
        writer = ShardWriter(tmp, schema=CLEAN_SCHEMA)
        kept = keep_ids[source_id]
        for row in iter_parquet_rows(src):
            if row["doc_id"] in kept:
                writer.add(row)
        writer.close()
        # Replace shards.
        for old in list_shards(src):
            old.unlink()
        for shard in list_shards(tmp):
            shard.rename(src / shard.name)
        shutil.rmtree(tmp)
        # Update manifest stats.
        man_path = src / "manifest.json"
        if man_path.exists():
            man = load_json(man_path)
            man["docs"] = writer.docs
            man["text_bytes"] = writer.text_bytes
            man["shards"] = writer.shards
            man["exact_dedup"] = True
            save_json(man_path, finish_manifest(man))
        print(
            f"exact_dedup {source_id} kept={writer.docs} text_bytes={writer.text_bytes}",
            flush=True,
        )

    out = {
        **base_manifest(stage="exact_dedup"),
        "stats": stats,
        "sources": sources,
    }
    save_json(artifacts_clean() / "exact_dedup.json", finish_manifest(out))
    return out


def _adapter(self, data: dict, path: str, id_in_file: int | str):
    # ParquetReader adapter: map our schema to Document fields.
    return {
        "text": data["text"],
        "id": data["doc_id"],
        "media": [],
        "metadata": {
            "source": data.get("source", ""),
            "license": data.get("license", ""),
            "bucket": data.get("bucket", "Nepali"),
            "glotlid_label": data.get("glotlid_label", ""),
            "glotlid_prob": data.get("glotlid_prob", 0.0),
            "file_path": path,
        },
    }


def minhash_dedup_nepali(mix: dict, *, restage: bool = False) -> dict[str, Any]:
    """FineWeb MinHash settings via datatrove local executor."""
    _patch_xxhash_for_datatrove()
    mh = mix["minhash"]
    config = MinhashConfig(
        n_grams=int(mh["n_grams"]),
        num_buckets=int(mh["num_buckets"]),
        hashes_per_bucket=int(mh["hashes_per_bucket"]),
        seed=SEED,
    )
    pref = _preference_rank(mix)
    # Preference order for staging (PLAN.md cluster keepers).
    sources = [s for s in mix["nepali_dedup_preference"] if clean_dir(s).exists()]
    for sid in mix["buckets"]["Nepali"]["sources"]:
        if sid not in sources and clean_dir(sid).exists():
            sources.append(sid)
            pref.setdefault(sid, len(pref))

    work = REPO_ROOT / "data" / "clean" / "_minhash"
    input_dir = work / "input"
    sig_dir = work / "signatures"
    buckets_dir = work / "buckets"
    remove_dir = work / "remove_ids"
    filtered_dir = work / "filtered"
    for d in (input_dir, sig_dir, buckets_dir, remove_dir, filtered_dir):
        d.mkdir(parents=True, exist_ok=True)

    staged = input_dir / "docs"
    exact_path = artifacts_clean() / "exact_dedup.json"
    expected_docs = None
    if exact_path.exists():
        expected_docs = int(load_json(exact_path)["stats"]["docs_out"])

    need_stage = restage or not staged.exists() or not list_shards(staged)
    if not need_stage and expected_docs is not None:
        # Count quickly from manifests / shard footers via row groups is heavy;
        # restage if shard count looks empty.
        pass
    if need_stage:
        if staged.exists():
            shutil.rmtree(staged)
        staged.mkdir(parents=True)
        writer = ShardWriter(staged, schema=CLEAN_SCHEMA)
        for source_id in sources:
            for row in iter_parquet_rows(clean_dir(source_id)):
                writer.add(row)
        writer.close()
        print(f"minhash staged docs={writer.docs} text_bytes={writer.text_bytes}", flush=True)
        # Fresh staging invalidates downstream minhash intermediates.
        for d in (sig_dir, buckets_dir, remove_dir, filtered_dir):
            if d.exists():
                shutil.rmtree(d)
            d.mkdir(parents=True, exist_ok=True)
        for log_name in ("logs_sig", "logs_buckets", "logs_cluster", "logs_filter"):
            log_dir = work / log_name
            if log_dir.exists():
                shutil.rmtree(log_dir)
    else:
        print(f"minhash reusing staged input at {staged}", flush=True)
        # Clear partial signature/log state from the failed xxhash run.
        for d in (sig_dir, buckets_dir, remove_dir, filtered_dir):
            if d.exists():
                shutil.rmtree(d)
            d.mkdir(parents=True, exist_ok=True)
        for log_name in ("logs_sig", "logs_buckets", "logs_cluster", "logs_filter"):
            log_dir = work / log_name
            if log_dir.exists():
                shutil.rmtree(log_dir)

    n_workers = workers()
    n_shards = max(1, len(list_shards(staged)))

    def reader_factory():
        return ParquetReader(
            str(staged),
            text_key="text",
            id_key="doc_id",
            adapter=_adapter,
            glob_pattern="shard-*.parquet",
        )

    stage1 = LocalPipelineExecutor(
        pipeline=[
            reader_factory(),
            MinhashDedupSignature(
                output_folder=str(sig_dir),
                config=config,
                language=Languages.nepali_individual_language__deva,
            ),
        ],
        tasks=n_shards,
        workers=n_workers,
        logging_dir=str(work / "logs_sig"),
        skip_completed=True,
    )

    stage2 = LocalPipelineExecutor(
        pipeline=[
            MinhashDedupBuckets(
                input_folder=str(sig_dir),
                output_folder=str(buckets_dir),
                config=config,
            ),
        ],
        tasks=config.num_buckets,
        workers=min(n_workers, config.num_buckets),
        logging_dir=str(work / "logs_buckets"),
        depends=stage1,
        skip_completed=True,
    )

    stage3 = LocalPipelineExecutor(
        pipeline=[
            MinhashDedupCluster(
                input_folder=str(buckets_dir),
                output_folder=str(remove_dir),
                config=config,
                save_cluster_id=True,
            ),
        ],
        tasks=1,
        workers=1,
        logging_dir=str(work / "logs_cluster"),
        depends=stage2,
        skip_completed=True,
    )

    stage4 = LocalPipelineExecutor(
        pipeline=[
            reader_factory(),
            MinhashDedupFilter(input_folder=str(remove_dir)),
            ParquetWriter(
                output_folder=str(filtered_dir),
                compression="zstd",
                adapter=lambda self, doc: {
                    "doc_id": doc.id,
                    "text": doc.text,
                    "source": doc.metadata.get("source", ""),
                    "license": doc.metadata.get("license", ""),
                    "bucket": doc.metadata.get("bucket", "Nepali"),
                    "glotlid_label": doc.metadata.get("glotlid_label", ""),
                    "glotlid_prob": float(doc.metadata.get("glotlid_prob") or 0.0),
                },
                expand_metadata=False,
            ),
        ],
        tasks=n_shards,
        workers=n_workers,
        logging_dir=str(work / "logs_filter"),
        depends=stage3,
        skip_completed=True,
    )

    print("running minhash stage1 signatures", flush=True)
    stage1.run()
    print("running minhash stage2 buckets", flush=True)
    stage2.run()
    print("running minhash stage3 cluster", flush=True)
    stage3.run()
    print("rewriting remove lists for source preference", flush=True)
    pref_stats = _rewrite_removes_for_preference(remove_dir, staged, pref)
    print(f"preference rewrite {pref_stats}", flush=True)
    print("running minhash stage4 filter", flush=True)
    stage4.run()

    redistrib = redistribute_minhash_filtered(
        filtered_dir=filtered_dir,
        sources=sources,
        pref_stats=pref_stats,
        config=config,
        mix=mix,
        work=work,
    )
    return redistrib


def redistribute_minhash_filtered(
    *,
    filtered_dir: Path,
    sources: list[str],
    pref_stats: dict[str, Any],
    config: MinhashConfig,
    mix: dict,
    work: Path | None = None,
) -> dict[str, Any]:
    """Stream filtered MinHash survivors back into per-source clean dirs (low RAM)."""
    import pyarrow.parquet as pq

    # Write into temp dirs first, then swap — so a crash doesn't wipe clean/.
    tmp_roots = {s: REPO_ROOT / "data" / "clean" / f".tmp_mh_{s}" for s in sources}
    writers: dict[str, ShardWriter] = {}
    for source_id, tmp in tmp_roots.items():
        if tmp.exists():
            shutil.rmtree(tmp)
        writers[source_id] = ShardWriter(tmp, schema=CLEAN_SCHEMA)

    kept = 0
    bytes_kept = 0
    filtered_files = sorted(
        p for p in filtered_dir.rglob("*") if p.is_file() and ".parquet" in p.name
    )
    print(f"redistributing {len(filtered_files)} filtered files", flush=True)
    for path in filtered_files:
        table = pq.read_table(path)
        cols = {name: table.column(name) for name in table.column_names}
        n = table.num_rows
        for i in range(n):
            row = {name: cols[name][i].as_py() for name in cols}
            source_id = row.get("source") or ""
            if source_id not in writers:
                source_id = str(row.get("doc_id", "")).split("\x1f", 1)[0]
            if source_id not in writers:
                writers[source_id] = ShardWriter(
                    REPO_ROOT / "data" / "clean" / f".tmp_mh_{source_id}",
                    schema=CLEAN_SCHEMA,
                )
                tmp_roots[source_id] = writers[source_id].out
            text = row["text"]
            writers[source_id].add(
                {
                    "doc_id": row["doc_id"],
                    "text": text,
                    "source": row.get("source") or source_id,
                    "license": row.get("license") or "",
                    "bucket": row.get("bucket") or "Nepali",
                    "glotlid_label": row.get("glotlid_label") or "",
                    "glotlid_prob": float(row.get("glotlid_prob") or 0.0),
                }
            )
            kept += 1
            bytes_kept += utf8_len(text)
        if kept and kept % 500_000 < n:
            print(f"redistributed ~{kept} docs free_bytes={free_bytes()}", flush=True)

    for source_id, writer in writers.items():
        writer.close()
        dest = clean_dir(source_id)
        dest.mkdir(parents=True, exist_ok=True)
        for old in list_shards(dest):
            old.unlink()
        tmp = tmp_roots[source_id]
        for shard in list_shards(tmp):
            shard.rename(dest / shard.name)
        shutil.rmtree(tmp)
        man_path = dest / "manifest.json"
        if man_path.exists():
            man = load_json(man_path)
            man["docs"] = writer.docs
            man["text_bytes"] = writer.text_bytes
            man["shards"] = writer.shards
            man["minhash_dedup"] = True
            save_json(man_path, finish_manifest(man))
        print(
            f"minhash {source_id} kept={writer.docs} text_bytes={writer.text_bytes}",
            flush=True,
        )

    result = finish_manifest(
        {
            **base_manifest(stage="minhash_dedup"),
            "docs_kept": kept,
            "bytes_kept": bytes_kept,
            "config": str(config),
            "sources": sources,
            "preference": list(mix["nepali_dedup_preference"]),
            "preference_rewrite": pref_stats,
        }
    )
    save_json(artifacts_clean() / "minhash_dedup.json", result)
    if work is not None and work.exists():
        shutil.rmtree(work)
        print(f"removed minhash workdir free_bytes={free_bytes()}", flush=True)
    print(f"minhash done kept={kept} free_bytes={free_bytes()}", flush=True)
    return result


def run_dedup() -> dict[str, Any]:
    mix = load_mix()
    exact_path = artifacts_clean() / "exact_dedup.json"
    if exact_path.exists():
        exact = load_json(exact_path)
        print(
            f"skipping exact dedup (already done docs_out={exact['stats']['docs_out']})",
            flush=True,
        )
    else:
        exact = exact_dedup_nepali(mix)
    mh = minhash_dedup_nepali(mix)
    return {"exact": exact, "minhash": mh}

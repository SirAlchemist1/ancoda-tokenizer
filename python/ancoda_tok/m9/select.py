"""Scan the clean index and choose the shared document prefix."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from ancoda_tok.m9.common import (
    GROUPS,
    REPO_ROOT,
    canonical_lang,
    doc_id_sha256,
    mix_group,
    rank_hex,
    rank_u64,
    text_sha256,
)

INDEX = REPO_ROOT / "data" / "clean" / "_snapshot_index.parquet"
SEALED = REPO_ROOT / "docs" / "sealed_test_set.json"
SELECT_SCHEMA = pa.schema(
    [
        ("doc_id", pa.string()),
        ("lang", pa.string()),
        ("group", pa.string()),
        ("text_bytes", pa.int64()),
        ("clean_shard", pa.string()),
        ("row", pa.int64()),
        ("rank_hex", pa.string()),
    ]
)


def load_sealed_hashes() -> set[str]:
    payload = json.loads(SEALED.read_text(encoding="utf-8"))
    hashes = set(payload["doc_id_sha256"])
    if len(hashes) != int(payload["n_docs"]):
        raise SystemExit(f"STOP: sealed hash set {len(hashes)} != n_docs {payload['n_docs']}")
    return hashes


def eval_source_ids() -> list[str]:
    raw = yaml.safe_load((REPO_ROOT / "configs" / "sources.yaml").read_text(encoding="utf-8"))
    return [row["id"] for row in raw["sources"] if row.get("role") in ("eval", "decontam")]


def load_eval_id_hashes(source_ids: list[str]) -> set[str]:
    hashes: set[str] = set()
    for source in source_ids:
        root = REPO_ROOT / "data" / "raw" / source
        if not root.is_dir():
            raise SystemExit(f"STOP: missing eval raw dir {root}")
        for path in sorted(root.glob("shard-*.parquet")):
            ids = pq.read_table(path, columns=["doc_id"]).column(0).to_pylist()
            hashes.update(doc_id_sha256(doc_id) for doc_id in ids)
    return hashes


def load_text_blocklist(source_ids: list[str]) -> set[str]:
    """Exact-text sha256 of FLORES lines and eval/decontam documents."""
    blocked: set[str] = set()
    flores_dirs = [
        REPO_ROOT / "third_party" / "yak-tokenizer" / "data" / "eval" / "flores_plus" / "devtest",
        REPO_ROOT / "data" / "eval" / "flores_plus_extra" / "devtest",
    ]
    n_files = 0
    for directory in flores_dirs:
        if not directory.is_dir():
            continue
        for path in sorted(directory.glob("*.txt")):
            n_files += 1
            for line in path.read_text(encoding="utf-8").splitlines():
                if line:
                    blocked.add(text_sha256(line))
    if n_files == 0:
        raise SystemExit("STOP: no FLORES devtest files found")
    for source in source_ids:
        root = REPO_ROOT / "data" / "raw" / source
        for path in sorted(root.glob("shard-*.parquet")):
            texts = pq.read_table(path, columns=["text"]).column(0).to_pylist()
            for text in texts:
                if text:
                    blocked.add(text_sha256(text))
    return blocked


def scan_index(
    sealed: set[str], eval_ids: set[str]
) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], dict]:
    chunks: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {g: [] for g in GROUPS}
    stats = {
        "rows": 0,
        "sealed_hits": 0,
        "eval_id_hits": 0,
        "synthetic_skipped": 0,
        "unassigned": 0,
    }
    pf = pq.ParquetFile(INDEX)
    columns = ["doc_id", "bucket", "text_bytes"]
    for batch in pf.iter_batches(columns=columns, batch_size=16384):
        doc_ids = batch.column(0).to_pylist()
        buckets = batch.column(1).to_pylist()
        nbytes = batch.column(2).to_pylist()
        by_group: dict[str, list[tuple[int, int]]] = {g: [] for g in GROUPS}
        for doc_id, bucket, nbyte in zip(doc_ids, buckets, nbytes, strict=True):
            stats["rows"] += 1
            if doc_id.startswith("synthetic\x1f"):
                stats["synthetic_skipped"] += 1
                continue
            digest = doc_id_sha256(doc_id)
            if digest in sealed:
                stats["sealed_hits"] += 1
                continue
            if digest in eval_ids:
                stats["eval_id_hits"] += 1
                continue
            group = mix_group(bucket)
            if group is None:
                stats["unassigned"] += 1
                continue
            by_group[group].append((rank_u64(doc_id), int(nbyte)))
        for group, rows in by_group.items():
            if not rows:
                continue
            chunks[group].append(
                (
                    np.asarray([r for r, _b in rows], dtype=np.uint64),
                    np.asarray([b for _r, b in rows], dtype=np.int64),
                )
            )
        if stats["rows"] % 2_000_000 == 0:
            print(f"  index rows={stats['rows']}", flush=True)
    arrays: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for group, parts in chunks.items():
        if not parts:
            arrays[group] = (np.zeros(0, dtype=np.uint64), np.zeros(0, dtype=np.int64))
        else:
            arrays[group] = (
                np.concatenate([p[0] for p in parts]),
                np.concatenate([p[1] for p in parts]),
            )
    return arrays, stats


def _take_ties(ties: list[dict], need: int) -> list[dict]:
    if need <= 0:
        return []
    ties.sort(key=lambda row: row["doc_id"])
    got = 0
    kept: list[dict] = []
    for row in ties:
        kept.append(row)
        got += int(row["text_bytes"])
        if got >= need:
            break
    return kept


def collect_selected(
    specs: dict[str, dict],
    dest: Path,
    sealed: set[str],
    eval_ids: set[str],
) -> dict[str, dict[str, int]]:
    """Second index scan. Write the selected rows sorted later by the caller."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    writer = pq.ParquetWriter(dest, SELECT_SCHEMA)
    batch: list[dict] = []
    counts = {g: 0 for g in GROUPS}
    bytes_out = {g: 0 for g in GROUPS}
    ties: dict[str, list[dict]] = {g: [] for g in GROUPS}

    def flush() -> None:
        if not batch:
            return
        table = pa.Table.from_pylist(batch, schema=SELECT_SCHEMA)
        writer.write_table(table)
        batch.clear()

    pf = pq.ParquetFile(INDEX)
    columns = ["doc_id", "bucket", "lang", "text_bytes", "shard", "row"]
    for arrow in pf.iter_batches(columns=columns, batch_size=16384):
        data = {name: arrow.column(name).to_pylist() for name in columns}
        n = len(data["doc_id"])
        for i in range(n):
            doc_id = data["doc_id"][i]
            if doc_id.startswith("synthetic\x1f"):
                continue
            digest = doc_id_sha256(doc_id)
            if digest in sealed or digest in eval_ids:
                continue
            bucket = data["bucket"][i]
            group = mix_group(bucket)
            if group is None:
                continue
            spec = specs[group]
            if spec["mode"] == "none":
                continue
            rank = rank_u64(doc_id)
            row = {
                "doc_id": doc_id,
                "lang": canonical_lang(bucket, data["lang"][i]),
                "group": group,
                "text_bytes": int(data["text_bytes"][i]),
                "clean_shard": data["shard"][i],
                "row": int(data["row"][i]),
                "rank_hex": rank_hex(rank),
            }
            if spec["mode"] == "all" or rank < spec["cutoff"]:
                batch.append(row)
                counts[group] += 1
                bytes_out[group] += row["text_bytes"]
            elif rank == spec["cutoff"]:
                ties[group].append(row)
            if len(batch) >= 20000:
                flush()
    for group, spec in specs.items():
        if spec["mode"] != "prefix":
            continue
        chosen = _take_ties(ties[group], int(spec["need_at_cutoff"]))
        for row in chosen:
            batch.append(row)
            counts[group] += 1
            bytes_out[group] += int(row["text_bytes"])
    flush()
    writer.close()
    return {"docs": counts, "bytes": bytes_out}

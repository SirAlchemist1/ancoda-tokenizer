"""Stream a snapshot into a sorted pre-token count table.

Raw text is not kept. Each shard is counted on its own and the bins are merged.
A finished shard bin plus its sidecar is skipped, so the job resumes.
"""

from __future__ import annotations

import argparse
import heapq
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import ancoda_seg
import pyarrow.parquet as pq

from ancoda_tok.clean.common import (
    FREE_FLOOR,
    base_manifest,
    finish_manifest,
    free_bytes,
    list_shards,
    save_json,
    sha256_file,
)
from ancoda_tok.inventory import find_n50_snapshot
from ancoda_tok.storage import GB, REPO_ROOT

MAGIC = b"ANCDACT1"
BATCH = 256


def write_counts(path: Path, entries: list[tuple[bytes, int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    previous: bytes | None = None
    with tmp.open("wb") as handle:
        handle.write(MAGIC)
        handle.write(len(entries).to_bytes(8, "little"))
        for raw, count in entries:
            if count <= 0 or count >= 2**64:
                raise ValueError(f"count out of range for {raw!r}")
            if previous is not None and raw <= previous:
                raise ValueError("refusing to write unsorted counts")
            previous = raw
            handle.write(len(raw).to_bytes(4, "little"))
            handle.write(raw)
            handle.write(int(count).to_bytes(8, "little"))
    tmp.replace(path)


class _Reader:
    def __init__(self, path: Path) -> None:
        self.handle = path.open("rb")
        magic = self.handle.read(8)
        if magic != MAGIC:
            raise ValueError(f"bad count magic in {path}")
        self.remaining = int.from_bytes(self.handle.read(8), "little")

    def next_entry(self) -> tuple[bytes, int] | None:
        if self.remaining <= 0:
            return None
        self.remaining -= 1
        length = int.from_bytes(self.handle.read(4), "little")
        raw = self.handle.read(length)
        if len(raw) != length:
            raise ValueError("truncated count entry")
        count = int.from_bytes(self.handle.read(8), "little")
        return raw, count

    def close(self) -> None:
        self.handle.close()


def iter_entries(path: Path):
    reader = _Reader(path)
    try:
        while True:
            item = reader.next_entry()
            if item is None:
                break
            yield item
    finally:
        reader.close()


def merge_counts(paths: list[Path], dest: Path) -> dict[str, int]:
    readers = [_Reader(path) for path in paths]
    heap: list[tuple[bytes, int, int]] = []
    try:
        for index, reader in enumerate(readers):
            item = reader.next_entry()
            if item is not None:
                heapq.heappush(heap, (item[0], index, item[1]))
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(dest.suffix + ".tmp")
        unique = 0
        occurrences = 0
        key_bytes = 0
        with tmp.open("wb") as handle:
            handle.write(MAGIC)
            handle.write((0).to_bytes(8, "little"))
            while heap:
                raw, index, count = heapq.heappop(heap)
                total = count
                nxt = readers[index].next_entry()
                if nxt is not None:
                    heapq.heappush(heap, (nxt[0], index, nxt[1]))
                while heap and heap[0][0] == raw:
                    _raw, other, more = heapq.heappop(heap)
                    total += more
                    nxt = readers[other].next_entry()
                    if nxt is not None:
                        heapq.heappush(heap, (nxt[0], other, nxt[1]))
                if total >= 2**64:
                    raise ValueError("merged count does not fit in u64")
                handle.write(len(raw).to_bytes(4, "little"))
                handle.write(raw)
                handle.write(total.to_bytes(8, "little"))
                unique += 1
                occurrences += total
                key_bytes += len(raw)
        with tmp.open("r+b") as handle:
            handle.seek(8)
            handle.write(unique.to_bytes(8, "little"))
        tmp.replace(dest)
    finally:
        for reader in readers:
            reader.close()
    return {"unique": unique, "occurrences": occurrences, "key_bytes": key_bytes}


def _sidecar(path: Path) -> Path:
    return path.with_suffix(".json")


def _count_shard(shard: str, dest: str, bucket: str | None = None) -> dict:
    shard_path = Path(shard)
    dest_path = Path(dest)
    side = _sidecar(dest_path)
    if dest_path.exists() and side.exists():
        meta = json.loads(side.read_text(encoding="utf-8"))
        if meta.get("bucket_filter") == bucket:
            meta["resumed"] = True
            return meta
        dest_path.unlink()
        side.unlink()
    if dest_path.exists():
        dest_path.unlink()
    counter: dict[bytes, int] = {}
    docs = 0
    text_bytes = 0
    empty = 0
    occurrences = 0
    parquet = pq.ParquetFile(shard_path)
    columns = ["text"] if bucket is None else ["text", "bucket"]
    for batch in parquet.iter_batches(columns=columns, batch_size=BATCH):
        texts = [("" if value is None else value) for value in batch.column(0).to_pylist()]
        if bucket is None:
            keep = texts
        else:
            buckets = batch.column(1).to_pylist()
            keep = [
                text
                for text, label in zip(texts, buckets, strict=True)
                if label == bucket
            ]
        if not keep:
            continue
        docs += len(keep)
        text_bytes += sum(len(text.encode("utf-8")) for text in keep)
        for parts in ancoda_seg.pretokenize_batch(keep):
            for part in parts:
                raw = part.encode("utf-8")
                if not raw:
                    empty += 1
                    continue
                counter[raw] = counter.get(raw, 0) + 1
                occurrences += 1
    entries = sorted(counter.items())
    write_counts(dest_path, entries)
    meta = {
        "shard": str(shard_path.relative_to(REPO_ROOT))
        if shard_path.is_relative_to(REPO_ROOT)
        else str(shard_path),
        "bin": str(dest_path.relative_to(REPO_ROOT))
        if dest_path.is_relative_to(REPO_ROOT)
        else str(dest_path),
        "docs": docs,
        "text_bytes": text_bytes,
        "empty_pretokens": empty,
        "unique": len(entries),
        "occurrences": occurrences,
        "key_bytes": sum(len(raw) for raw, _count in entries),
        "bin_sha256": sha256_file(dest_path),
        "shard_sha256": sha256_file(shard_path),
        "bucket_filter": bucket,
        "resumed": False,
    }
    side.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return meta


def _count_prefix(shard: Path, dest: Path, already: int, limit: int) -> dict:
    """Include whole documents from ``shard`` until text bytes reach ``limit``."""
    counter: dict[bytes, int] = {}
    docs = 0
    seen = already
    empty = 0
    occurrences = 0
    parquet = pq.ParquetFile(shard)
    stop = False
    for batch in parquet.iter_batches(columns=["text"], batch_size=BATCH):
        texts = [("" if value is None else value) for value in batch.column(0).to_pylist()]
        # Walk documents in order. A batch may pass the limit, so do not pretok the tail.
        take: list[str] = []
        for text in texts:
            if seen >= limit:
                stop = True
                break
            take.append(text)
            seen += len(text.encode("utf-8"))
            docs += 1
        if take:
            for parts in ancoda_seg.pretokenize_batch(take):
                for part in parts:
                    raw = part.encode("utf-8")
                    if not raw:
                        empty += 1
                        continue
                    counter[raw] = counter.get(raw, 0) + 1
                    occurrences += 1
        if stop or seen >= limit:
            break
    entries = sorted(counter.items())
    write_counts(dest, entries)
    return {
        "docs": docs,
        "text_bytes": seen - already,
        "empty_pretokens": empty,
        "unique": len(entries),
        "occurrences": occurrences,
        "end_text_bytes": seen,
    }


def _workers(requested: int | None) -> int:
    if requested is not None:
        return requested
    return int(os.environ.get("ANCODA_WORKERS", "5"))


def build(
    snapshot: Path, out_dir: Path, workers: int, bucket: str | None = None
) -> None:
    free = free_bytes()
    print(f"free_bytes={free}", flush=True)
    if free < FREE_FLOOR:
        raise SystemExit(f"STOP: {free} bytes free, floor is {FREE_FLOOR}")
    shards = list_shards(snapshot)
    if not shards:
        raise SystemExit(f"STOP: no shards in {snapshot}")
    shard_dir = out_dir / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    jobs = [(str(shard), str(shard_dir / f"{shard.stem}.bin")) for shard in shards]
    metas: list[dict] = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(_count_shard, shard, dest, bucket): shard for shard, dest in jobs
        }
        for done, future in enumerate(as_completed(futures), start=1):
            meta = future.result()
            metas.append(meta)
            print(
                f"shard {done}/{len(jobs)} {meta['shard']} docs={meta['docs']} "
                f"text_bytes={meta['text_bytes']} unique={meta['unique']} "
                f"resumed={meta['resumed']}",
                flush=True,
            )
    metas.sort(key=lambda item: item["shard"])
    merged = out_dir / "pretoken_counts.bin"
    print(f"merging {len(metas)} bins -> {merged}", flush=True)
    stats = merge_counts([REPO_ROOT / item["bin"] for item in metas], merged)
    manifest = base_manifest(
        snapshot=str(snapshot.relative_to(REPO_ROOT)),
        kind="pretoken_counts",
        bucket_filter=bucket,
        docs=sum(item["docs"] for item in metas),
        text_bytes=sum(item["text_bytes"] for item in metas),
        empty_pretokens=sum(item["empty_pretokens"] for item in metas),
        unique=stats["unique"],
        occurrences=stats["occurrences"],
        key_bytes=stats["key_bytes"],
        counts_sha256=sha256_file(merged),
        input_sha256={item["shard"]: item["shard_sha256"] for item in metas},
        pattern_sha256=sha256_file(REPO_ROOT / "configs" / "pretokenizer_pattern.txt"),
    )
    save_json(out_dir / "manifest.json", finish_manifest(manifest))
    print(
        f"full unique={stats['unique']} occurrences={stats['occurrences']} "
        f"key_bytes={stats['key_bytes']} text_bytes={manifest['text_bytes']}",
        flush=True,
    )
    # The 1 GB slice is only for the trainer plain-vs-HF check on n50. Other
    # snapshots must not overwrite artifacts/counts/n50_1gb.
    if out_dir.name == "n50" and bucket is None:
        _build_one_gb(metas, out_dir.parent / "n50_1gb", snapshot)


def _build_one_gb(metas: list[dict], out_dir: Path, snapshot: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    selected: list[Path] = []
    used = 0
    docs = 0
    empty = 0
    boundary: dict | None = None
    for meta in metas:
        if used >= GB:
            break
        shard_bytes = int(meta["text_bytes"])
        if used + shard_bytes < GB:
            selected.append(REPO_ROOT / meta["bin"])
            used += shard_bytes
            docs += int(meta["docs"])
            empty += int(meta["empty_pretokens"])
            continue
        partial = out_dir / "boundary.bin"
        boundary = _count_prefix(REPO_ROOT / meta["shard"], partial, used, GB)
        selected.append(partial)
        used = int(boundary["end_text_bytes"])
        docs += int(boundary["docs"])
        empty += int(boundary["empty_pretokens"])
        break
    if used < GB:
        raise SystemExit(f"STOP: snapshot has only {used} text bytes, wanted {GB}")
    merged = out_dir / "pretoken_counts.bin"
    print(f"merging 1GB slice ({used} text bytes, {docs} docs) -> {merged}", flush=True)
    stats = merge_counts(selected, merged)
    manifest = base_manifest(
        snapshot=str(snapshot.relative_to(REPO_ROOT)),
        kind="pretoken_counts_1gb",
        limit_text_bytes=GB,
        docs=docs,
        text_bytes=used,
        empty_pretokens=empty,
        unique=stats["unique"],
        occurrences=stats["occurrences"],
        key_bytes=stats["key_bytes"],
        counts_sha256=sha256_file(merged),
        boundary=boundary,
        pattern_sha256=sha256_file(REPO_ROOT / "configs" / "pretokenizer_pattern.txt"),
    )
    save_json(out_dir / "manifest.json", finish_manifest(manifest))
    print(
        f"1gb unique={stats['unique']} occurrences={stats['occurrences']} "
        f"key_bytes={stats['key_bytes']} text_bytes={used}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "artifacts" / "counts" / "n50")
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument(
        "--bucket",
        default=None,
        help="If set, count only documents whose snapshot bucket equals this label",
    )
    args = parser.parse_args()
    snapshot = args.snapshot or find_n50_snapshot()
    build(snapshot, args.out, _workers(args.workers), bucket=args.bucket)


if __name__ == "__main__":
    main()

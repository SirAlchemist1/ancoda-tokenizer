"""Build the shared zstd JSONL stream on the Mac. Never normalizes stored text."""

from __future__ import annotations

import hashlib
import io
import json
from collections import defaultdict
from pathlib import Path

import pyarrow.parquet as pq
import zstandard as zstd

from ancoda_tok.clean.common import (
    base_manifest,
    ensure_room,
    finish_manifest,
    sha256_file,
)
from ancoda_tok.m9.arms import ARMS, load_arm
from ancoda_tok.m9.common import (
    GROUPS,
    REPO_ROOT,
    SAMPLE_BYTES,
    choose_total_bytes,
    doc_id_sha256,
    group_targets,
    text_sha256,
    update_list_hash,
)
from ancoda_tok.m9.cutoff import prefix_spec
from ancoda_tok.m9.select import (
    INDEX,
    collect_selected,
    eval_source_ids,
    load_eval_id_hashes,
    load_sealed_hashes,
    load_text_blocklist,
    scan_index,
)

M9 = REPO_ROOT / "data" / "m9"
TEXT_DIR = M9 / "text"
WORK = M9 / "work"
MERGE_BATCH = 48


def _part_name(shard: str) -> str:
    return shard.replace("/", "__") + ".jsonl.zst"


def _write_jsonl_zst(path: Path, lines: list[bytes]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cctx = zstd.ZstdCompressor(level=3)
    with path.open("wb") as raw, cctx.stream_writer(raw) as writer:
        for line in lines:
            writer.write(line)


def _iter_zst_lines(path: Path):
    dctx = zstd.ZstdDecompressor()
    with path.open("rb") as raw, dctx.stream_reader(raw) as reader:
        text = io.TextIOWrapper(reader, encoding="utf-8")
        for line in text:
            if line:
                yield line


def fetch_texts(rows: list[dict]) -> list[str]:
    by_shard: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for i, row in enumerate(rows):
        by_shard[row["clean_shard"]].append((i, int(row["row"])))
    texts = [""] * len(rows)
    for shard, items in sorted(by_shard.items()):
        column = pq.read_table(REPO_ROOT / shard, columns=["text"]).column(0)
        for i, row_i in items:
            texts[i] = column[row_i].as_py() or ""
    return texts


def _rows_from_parquet(path: Path) -> list[dict]:
    table = pq.read_table(path)
    rows = table.to_pylist()
    rows.sort(key=lambda row: (row["rank_hex"], row["doc_id"]))
    return rows


def measure_sample(rows: list[dict]) -> dict:
    texts = fetch_texts(rows)
    mismatched = 0
    raw_bytes = 0
    per_group_bytes: dict[str, int] = defaultdict(int)
    for row, text in zip(rows, texts, strict=True):
        nbyte = len(text.encode("utf-8"))
        raw_bytes += nbyte
        per_group_bytes[row["group"]] += nbyte
        if nbyte != int(row["text_bytes"]):
            mismatched += 1
    if mismatched:
        raise SystemExit(
            f"STOP: sample index text_bytes mismatched the stored text on {mismatched} docs"
        )
    rates = {}
    for name in ARMS:
        arm = load_arm(name)
        tokens = 0
        for text in texts:
            tokens += len(arm.encode(text))
        rates[name] = {
            "content_tokens": tokens,
            "bytes": raw_bytes,
            "tokens_per_byte": tokens / raw_bytes if raw_bytes else 0.0,
            "docs": len(texts),
        }
        print(
            f"  sample {name} tokens={tokens} bytes={raw_bytes} "
            f"tpb={rates[name]['tokens_per_byte']:.6f}",
            flush=True,
        )
    return {"bytes": raw_bytes, "per_group_bytes": dict(per_group_bytes), "arms": rates}


def extract_parts(selection: Path, blocklist: set[str]) -> dict:
    table = pq.read_table(selection)
    by_shard: dict[str, list[dict]] = defaultdict(list)
    for row in table.to_pylist():
        by_shard[row["clean_shard"]].append(row)
    parts = WORK / "parts"
    parts.mkdir(parents=True, exist_ok=True)
    dropped = 0
    mismatched = 0
    kept = 0
    shards = sorted(by_shard)
    for n, shard in enumerate(shards, start=1):
        part = parts / _part_name(shard)
        marker = part.with_suffix(part.suffix + ".done")
        if marker.is_file() and part.is_file():
            meta = json.loads(marker.read_text(encoding="utf-8"))
            dropped += int(meta["dropped"])
            mismatched += int(meta["mismatched"])
            kept += int(meta["kept"])
            continue
        rows = by_shard[shard]
        column = pq.read_table(REPO_ROOT / shard, columns=["text"]).column(0)
        lines: list[bytes] = []
        part_dropped = 0
        part_mismatch = 0
        part_kept = 0
        for row in rows:
            text = column[int(row["row"])].as_py() or ""
            nbyte = len(text.encode("utf-8"))
            if nbyte != int(row["text_bytes"]):
                part_mismatch += 1
            if text_sha256(text) in blocklist:
                part_dropped += 1
                continue
            payload = {
                "rank_hex": row["rank_hex"],
                "doc_id": row["doc_id"],
                "lang": row["lang"],
                "group": row["group"],
                "text": text,
            }
            lines.append(
                (json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n").encode(
                    "utf-8"
                )
            )
            part_kept += 1
        _write_jsonl_zst(part, lines)
        meta = {"dropped": part_dropped, "mismatched": part_mismatch, "kept": part_kept}
        marker.write_text(json.dumps(meta), encoding="utf-8")
        dropped += part_dropped
        mismatched += part_mismatch
        kept += part_kept
        if n % 25 == 0 or n == len(shards):
            print(f"  extracted {n}/{len(shards)} shards kept={kept}", flush=True)
    if mismatched:
        raise SystemExit(
            f"STOP: index text_bytes mismatched stored text on {mismatched} selected docs"
        )
    return {"dropped_exact_text": dropped, "kept": kept, "parts": str(parts)}


def _line_key(line: str) -> tuple[str, str]:
    obj = json.loads(line)
    return obj["rank_hex"], obj["doc_id"]


def _merge_group_impl(paths: list[Path], dest: Path) -> None:
    import heapq

    iterators = [_iter_zst_lines(path) for path in paths]
    heap: list[tuple[str, str, int]] = []
    current: dict[int, str] = {}
    for i, it in enumerate(iterators):
        line = next(it, None)
        if line is None:
            continue
        rank, doc_id = _line_key(line)
        current[i] = line
        heapq.heappush(heap, (rank, doc_id, i))
    dest.parent.mkdir(parents=True, exist_ok=True)
    cctx = zstd.ZstdCompressor(level=3)
    with dest.open("wb") as raw, cctx.stream_writer(raw) as writer:
        while heap:
            _rank, _doc, i = heapq.heappop(heap)
            writer.write(current[i].encode("utf-8"))
            line = next(iterators[i], None)
            if line is None:
                continue
            rank, doc_id = _line_key(line)
            current[i] = line
            heapq.heappush(heap, (rank, doc_id, i))


def merge_to_shards(
    parts_dir: Path,
    sealed: set[str],
    eval_ids: set[str],
) -> dict:
    paths = sorted(parts_dir.glob("*.jsonl.zst"))
    if not paths:
        raise SystemExit("STOP: no extracted parts")
    level = 0
    current = paths
    while len(current) > 1:
        nxt_dir = WORK / f"merge-{level}"
        nxt_dir.mkdir(parents=True, exist_ok=True)
        merged: list[Path] = []
        for i in range(0, len(current), MERGE_BATCH):
            group = current[i : i + MERGE_BATCH]
            dest = nxt_dir / f"part-{i // MERGE_BATCH:04d}.jsonl.zst"
            marker = dest.with_suffix(dest.suffix + ".done")
            if not marker.is_file():
                print(f"  merge level {level} group {i // MERGE_BATCH}", flush=True)
                if dest.exists():
                    dest.unlink()
                _merge_group_impl(group, dest)
                marker.write_text("ok\n", encoding="utf-8")
            merged.append(dest)
        current = merged
        level += 1
    final_sorted = current[0]
    TEXT_DIR.mkdir(parents=True, exist_ok=True)
    for old in TEXT_DIR.glob("shard-*.jsonl.zst"):
        old.unlink()
    hasher = hashlib.sha256()
    overlap = 0
    docs = 0
    per_group_bytes: dict[str, int] = defaultdict(int)
    per_group_docs: dict[str, int] = defaultdict(int)
    shard_idx = 0
    raw_bytes_in_shard = 0
    shard_limit = 32 * 1024 * 1024
    cctx = zstd.ZstdCompressor(level=3)
    handle = None
    writer = None

    def open_shard() -> None:
        nonlocal handle, writer, shard_idx, raw_bytes_in_shard
        path = TEXT_DIR / f"shard-{shard_idx:05d}.jsonl.zst"
        handle = path.open("wb")
        writer = cctx.stream_writer(handle)
        raw_bytes_in_shard = 0

    def close_shard() -> None:
        nonlocal writer, handle, shard_idx
        if writer is not None:
            writer.close()
        if handle is not None:
            handle.close()
        shard_idx += 1

    open_shard()
    assert writer is not None
    for line in _iter_zst_lines(final_sorted):
        obj = json.loads(line)
        digest = doc_id_sha256(obj["doc_id"])
        if digest in sealed or digest in eval_ids:
            overlap += 1
            continue
        out = {"doc_id": obj["doc_id"], "lang": obj["lang"], "text": obj["text"]}
        encoded = (json.dumps(out, ensure_ascii=False, separators=(",", ":")) + "\n").encode(
            "utf-8"
        )
        if raw_bytes_in_shard >= shard_limit and docs:
            close_shard()
            open_shard()
            assert writer is not None
        writer.write(encoded)
        raw_bytes_in_shard += len(encoded)
        update_list_hash(hasher, obj["doc_id"])
        nbyte = len(obj["text"].encode("utf-8"))
        per_group_bytes[obj["group"]] += nbyte
        per_group_docs[obj["group"]] += 1
        docs += 1
    close_shard()
    if overlap:
        raise SystemExit(f"STOP: final stream overlap with the exclusion set is {overlap}")
    return {
        "docs": docs,
        "overlap": overlap,
        "list_sha256": hasher.hexdigest(),
        "per_group_bytes": dict(per_group_bytes),
        "per_group_docs": dict(per_group_docs),
        "shards": shard_idx,
    }


def prepare() -> dict:
    if (TEXT_DIR / "manifest.json").is_file():
        existing = json.loads((TEXT_DIR / "manifest.json").read_text(encoding="utf-8"))
        if existing.get("complete"):
            print(f"stream already complete list_sha256={existing['document_list_sha256']}")
            return existing
    ensure_room(20_000_000_000, label="m9 stream")
    WORK.mkdir(parents=True, exist_ok=True)
    print("loading exclusion sets", flush=True)
    sealed = load_sealed_hashes()
    sources = eval_source_ids()
    eval_ids = load_eval_id_hashes(sources)
    blocklist = load_text_blocklist(sources)
    print(
        f"sealed={len(sealed)} eval_doc_ids={len(eval_ids)} text_blocklist={len(blocklist)}",
        flush=True,
    )
    print("scanning clean index", flush=True)
    arrays, stats = scan_index(sealed, eval_ids)
    print(f"scan {stats}", flush=True)
    sealed_n = len(sealed)
    if stats["sealed_hits"] != sealed_n:
        raise SystemExit(f"STOP: sealed hits {stats['sealed_hits']} != sealed list {sealed_n}")
    available = {g: int(arrays[g][1].sum()) for g in GROUPS}
    print(f"available_bytes {available}", flush=True)
    sample_targets = group_targets(SAMPLE_BYTES)
    sample_specs = {}
    for group in GROUPS:
        spec = prefix_spec(arrays[group][0], arrays[group][1], sample_targets[group])
        if spec["shortfall"]:
            raise SystemExit(f"STOP: sample shortfall in {group}: {spec}")
        sample_specs[group] = spec
    sample_path = WORK / "sample.parquet"
    if not sample_path.is_file():
        print("collecting 50 MB sample", flush=True)
        collect_selected(sample_specs, sample_path, sealed, eval_ids)
    print("tokenizing sample", flush=True)
    measured = measure_sample(_rows_from_parquet(sample_path))
    min_name = min(measured["arms"], key=lambda name: measured["arms"][name]["tokens_per_byte"])
    min_tpb = measured["arms"][min_name]["tokens_per_byte"]
    total = choose_total_bytes(min_tpb, available)
    targets = group_targets(total)
    print(
        f"stream_bytes={total} targets={targets} min_arm={min_name} tpb={min_tpb:.6f}", flush=True
    )
    full_specs = {g: prefix_spec(arrays[g][0], arrays[g][1], targets[g]) for g in GROUPS}
    for group, spec in full_specs.items():
        if spec["shortfall"]:
            raise SystemExit(f"STOP: full-stream shortfall in {group}: {spec}")
    del arrays
    selection = WORK / "selection.parquet"
    if not selection.is_file():
        print("collecting full selection", flush=True)
        summary = collect_selected(full_specs, selection, sealed, eval_ids)
        (WORK / "selection_summary.json").write_text(json.dumps(summary), encoding="utf-8")
        print(f"selection {summary}", flush=True)
    summary = json.loads((WORK / "selection_summary.json").read_text(encoding="utf-8"))
    for group, target in targets.items():
        got = int(summary["bytes"][group])
        if got < target:
            raise SystemExit(f"STOP: {group} selected {got} bytes, target {target}")
    print("extracting text", flush=True)
    extracted = extract_parts(selection, blocklist)
    print(f"extract {extracted}", flush=True)
    print("merging shards", flush=True)
    merged = merge_to_shards(WORK / "parts", sealed, eval_ids)
    manifest = finish_manifest(
        base_manifest(
            kind="m9_text_stream",
            complete=True,
            input_sha256={
                "clean_index": sha256_file(INDEX),
                "sealed_test_set": sha256_file(REPO_ROOT / "docs" / "sealed_test_set.json"),
            },
            document_list_sha256=merged["list_sha256"],
            n_docs=merged["docs"],
            shards=merged["shards"],
            overlap=merged["overlap"],
            per_group_bytes=merged["per_group_bytes"],
            per_group_docs=merged["per_group_docs"],
            targets=targets,
            total_bytes_target=total,
            available_bytes=available,
            sample=measured,
            min_arm=min_name,
            min_tokens_per_byte=min_tpb,
            scan=stats,
            exact_text_dropped=extracted["dropped_exact_text"],
            eval_sources=sources,
            note=(
                "Shared stream for every arm. Text is unmodified clean-pool text. "
                "Mix shares and the rank prefix are locked in docs/M9_PREREG.md."
            ),
        )
    )
    (TEXT_DIR / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        f"wrote stream docs={merged['docs']} list_sha256={merged['list_sha256']}",
        flush=True,
    )
    return manifest

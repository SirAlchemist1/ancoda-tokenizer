"""Stream tokenization. One shard at a time per process."""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path

import numpy as np
import zstandard as zstd

from ancoda_tok.model_eval.arms import load_arm
from ancoda_tok.model_eval.common import attach_build, now_utc, update_list_hash


def iter_jsonl_zst(path: Path):
    dctx = zstd.ZstdDecompressor()
    with path.open("rb") as raw, dctx.stream_reader(raw) as reader:
        text = io.TextIOWrapper(reader, encoding="utf-8")
        for line in text:
            if line.strip():
                yield json.loads(line)


def _encode_one(task: dict) -> dict:
    arm = load_arm(task["arm"], Path(task["tokenizer"]))
    shard = Path(task["shard"])
    out = Path(task["out"])
    out.mkdir(parents=True, exist_ok=True)
    ids_chunks: list[np.ndarray] = []
    tok_counts: list[int] = []
    byte_counts: list[int] = []
    max_id = 0
    n = 0
    for row in iter_jsonl_zst(shard):
        encoded = arm.encode(row["text"])
        if encoded:
            arr = np.asarray(encoded, dtype=np.int64)
            max_id = max(max_id, int(arr.max()))
            ids_chunks.append(arr)
        tok_counts.append(len(encoded))
        byte_counts.append(len(row["text"].encode("utf-8")))
        n += 1
    tokens = np.concatenate(ids_chunks) if ids_chunks else np.zeros(0, dtype=np.int64)
    limit = 65535 if task["dtype"] == "uint16" else 2**32 - 1
    if max_id > limit or arm.eos_id > limit:
        raise SystemExit(f"STOP: id {max(max_id, arm.eos_id)} does not fit {task['dtype']}")
    store = np.dtype("<u2" if task["dtype"] == "uint16" else "<u4")
    np.asarray(tokens, dtype=store).tofile(out / "tokens.bin")
    np.asarray(tok_counts, dtype="<i4").tofile(out / "tok_counts.i32")
    np.asarray(byte_counts, dtype="<i8").tofile(out / "byte_counts.i64")
    (out / "done").write_text("ok\n", encoding="utf-8")
    return {
        "shard": shard.name,
        "docs": n,
        "content_tokens": int(tokens.size),
        "max_id": max_id,
    }


def tokenize(args) -> dict:
    import multiprocessing as mp

    text = Path(args.text)
    shards = sorted(text.glob("shard-*.jsonl.zst"))
    if not shards:
        raise SystemExit(f"STOP: no shards in {text}")
    arm = load_arm(args.arm, Path(args.tokenizer))
    out = Path(args.out)
    started = now_utc()
    part_root = out / "parts"
    part_root.mkdir(parents=True, exist_ok=True)
    tasks = [
        {
            "arm": args.arm,
            "tokenizer": str(args.tokenizer),
            "shard": str(shard),
            "out": str(part_root / shard.name),
            "dtype": arm.dtype,
        }
        for shard in shards
    ]
    pending = [task for task in tasks if not (Path(task["out"]) / "done").is_file()]
    print(f"tokenize {args.arm} shards={len(shards)} pending={len(pending)}", flush=True)
    if pending:
        ctx = mp.get_context("spawn")
        with ctx.Pool(int(args.workers)) as pool:
            for result in pool.imap_unordered(_encode_one, pending):
                print(f"  part {result['shard']} docs={result['docs']}", flush=True)
    dtype = np.dtype("<u2" if arm.dtype == "uint16" else "<u4")
    with (out / "tokens.bin").open("wb") as dest:
        for shard in shards:
            dest.write((part_root / shard.name / "tokens.bin").read_bytes())
    count_parts = []
    byte_parts = []
    for shard in shards:
        folder = part_root / shard.name
        count_parts.append(np.fromfile(folder / "tok_counts.i32", dtype="<i4"))
        byte_parts.append(np.fromfile(folder / "byte_counts.i64", dtype="<i8"))
    tok_counts = np.concatenate(count_parts)
    byte_counts = np.concatenate(byte_parts)
    tok_counts.tofile(out / "tok_counts.i32")
    byte_counts.tofile(out / "byte_counts.i64")
    content_tokens = (out / "tokens.bin").stat().st_size // dtype.itemsize
    if content_tokens != int(tok_counts.sum()):
        raise SystemExit(
            f"STOP: token bin {content_tokens} != sum of counts {int(tok_counts.sum())}"
        )
    hasher = hashlib.sha256()
    docs = 0
    for shard in shards:
        for row in iter_jsonl_zst(shard):
            update_list_hash(hasher, row["doc_id"])
            docs += 1
    if docs != int(tok_counts.shape[0]):
        raise SystemExit(f"STOP: docs {docs} != counts {tok_counts.shape[0]}")
    stream_manifest = json.loads((text / "manifest.json").read_text(encoding="utf-8"))
    list_sha = hasher.hexdigest()
    if list_sha != stream_manifest["document_list_sha256"]:
        raise SystemExit(
            "STOP: document list sha256 "
            f"{list_sha} != stream {stream_manifest['document_list_sha256']}"
        )
    text_bytes = int(byte_counts.sum())
    manifest = attach_build(
        {
            "arm": args.arm,
            "eos_id": arm.eos_id,
            "eos_rule": arm.eos_rule,
            "vocab": arm.vocab,
            "model_vocab": arm.model_vocab,
            "dtype": arm.dtype,
            "n_docs": docs,
            "content_tokens": content_tokens,
            "stream_tokens": content_tokens + docs,
            "text_bytes": text_bytes,
            "document_list_sha256": list_sha,
            "tokens_per_byte": content_tokens / text_bytes if text_bytes else 0.0,
            "seed": None,
            "started_at": started,
            "finished_at": now_utc(),
            "input_sha256": {"text_manifest": stream_manifest.get("document_list_sha256")},
        },
        args.build,
    )
    (out / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest), flush=True)
    return manifest


def prefix_counts(
    text: Path, tokenizer: Path, arm_name: str, n_docs: int, n_roundtrip: int
) -> dict:
    arm = load_arm(arm_name, tokenizer)
    counts: list[int] = []
    matched = 0
    seen_rt = 0
    for shard in sorted(text.glob("shard-*.jsonl.zst")):
        for row in iter_jsonl_zst(shard):
            ids = arm.encode(row["text"])
            if len(counts) < n_docs:
                counts.append(len(ids))
            if seen_rt < n_roundtrip:
                if arm.decode(ids) == row["text"]:
                    matched += 1
                seen_rt += 1
            if len(counts) >= n_docs and seen_rt >= n_roundtrip:
                break
        if len(counts) >= n_docs and seen_rt >= n_roundtrip:
            break
    if len(counts) < n_docs or seen_rt < n_roundtrip:
        raise SystemExit(f"STOP: stream shorter than the check ({len(counts)} docs)")
    return {
        "arm": arm_name,
        "n_docs": n_docs,
        "counts_sha256": hashlib.sha256(np.asarray(counts, dtype="<i4").tobytes()).hexdigest(),
        "counts_head": counts[:8],
        "count_sum": int(sum(counts)),
        "roundtrip_n": n_roundtrip,
        "roundtrip_ok": matched,
        "roundtrip_rate": matched / n_roundtrip,
        "eos_id": arm.eos_id,
        "eos_rule": arm.eos_rule,
        "vocab": arm.vocab,
        "model_vocab": arm.model_vocab,
        "dtype": arm.dtype,
    }


def compare_bins(token_dir: Path, report: dict) -> None:
    """Stored per-document counts for the prefix must match a fresh encode."""
    counts = np.fromfile(token_dir / "tok_counts.i32", dtype="<i4")
    prefix = counts[: report["n_docs"]]
    digest = hashlib.sha256(np.asarray(prefix, dtype="<i4").tobytes()).hexdigest()
    if digest != report["counts_sha256"]:
        raise SystemExit(
            f"STOP: bin counts sha {digest} != encode sha {report['counts_sha256']} "
            f"(bin_sum={int(prefix.sum())} encode_sum={report['count_sum']})"
        )

"""M4 step 5: decontamination against eval sets.

Match keys use NFC + collapsed whitespace on temporary copies only.
Stored training text is never normalized.

Word n-gram containment (6–12 and 13) uses blake2b digests of the joined
gram string. That is equivalent to Aho-Corasick over the same word spans and
fits in RAM; storing tens of millions of gram strings does not.
"""

from __future__ import annotations

import re
import unicodedata
from collections import defaultdict
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from ancoda_tok.clean.common import (
    CLEAN_SCHEMA,
    REPO_ROOT,
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
    raw_dir,
    save_json,
    train_source_ids,
    utf8_len,
    workers,
)

_WS_RE = re.compile(r"\s+")


def match_key(text: str) -> str:
    """Temporary NFC + collapsed whitespace key. Never written to stored text."""
    return _WS_RE.sub(" ", unicodedata.normalize("NFC", text)).strip()


def words(key: str) -> list[str]:
    if not key:
        return []
    return key.split(" ")


def exact_u64(key: str) -> int:
    import xxhash

    return xxhash.xxh64_intdigest(key.encode("utf-8"))


def word_u64s(w: list[str]) -> list[int]:
    """xxhash64 each word once; n-grams are rolling mixes of these."""
    import xxhash

    return [xxhash.xxh64_intdigest(t.encode("utf-8")) for t in w]


def ngram_u64_from_words(wh: list[int], start: int, n: int) -> int:
    """Polynomial mix of per-word hashes (stable, join-free)."""
    h = 0
    for j in range(n):
        h = ((h ^ wh[start + j]) * 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
    return h


def all_ngram_u64s(wh: list[int], n_min: int, n_max: int) -> list[int]:
    out: list[int] = []
    n_words = len(wh)
    for n in range(n_min, n_max + 1):
        if n_words < n:
            break
        for i in range(n_words - n + 1):
            out.append(ngram_u64_from_words(wh, i, n))
    return out


# Large same-distribution held-out sets: exact-match only. Indexing their
# 6–12-gram inventory (hundreds of millions of unique grams) both OOMs an
# 18 GB Mac and drops ~40–60% of training via ordinary n-gram overlap — that
# is not eval-leakage removal. Short eval sets get the full three rules.
_NGRAM_EVAL_SOURCES = ("belebele_npi", "aya_npi")
_EXACT_ONLY_EVAL_SOURCES = ("fw2_npi_test", "fw2_hin_test", "c4_val")


def iter_eval_texts(*, ngram: bool | None = None) -> Iterator[tuple[str, str, str]]:
    """Yield (eval_set, doc_id, text).

    ngram=None: every eval source (for exact keys).
    ngram=True: only short sources used for 6–12 / 13-gram indexes.
    ngram=False: only exact-only sources.
    """
    if ngram is not False:
        # FLORES+ from the harness (never copy elsewhere).
        flores = REPO_ROOT / "third_party" / "yak-tokenizer" / "data" / "eval" / "flores_plus"
        for split in ("dev", "devtest"):
            split_dir = flores / split
            if not split_dir.exists():
                continue
            for path in sorted(split_dir.glob("*.txt")):
                text = path.read_text(encoding="utf-8")
                yield (f"flores_{split}_{path.stem}", path.name, text)
                for i, line in enumerate(text.splitlines()):
                    if line.strip():
                        yield (
                            f"flores_{split}_{path.stem}_line",
                            f"{path.name}:{i}",
                            line,
                        )
        for source_id in _NGRAM_EVAL_SOURCES:
            directory = raw_dir(source_id)
            if not directory.exists():
                continue
            for row in iter_parquet_rows(directory):
                yield (source_id, row["doc_id"], row["text"])

    if ngram is not True:
        for source_id in _EXACT_ONLY_EVAL_SOURCES:
            directory = raw_dir(source_id)
            if not directory.exists():
                continue
            for row in iter_parquet_rows(directory):
                yield (source_id, row["doc_id"], row["text"])


def _unique_sorted_u64(raw_path: Path) -> Any:
    import numpy as np

    if not raw_path.exists() or raw_path.stat().st_size == 0:
        return np.empty(0, dtype="<u8")
    data = np.fromfile(raw_path, dtype="<u8")
    data.sort()
    return np.unique(data)


def _contains_u64(sorted_arr: Any, value: int) -> bool:
    import numpy as np

    if sorted_arr.size == 0:
        return False
    i = int(np.searchsorted(sorted_arr, value, side="left"))
    return i < sorted_arr.size and int(sorted_arr[i]) == value


def build_eval_index() -> dict[str, Any]:
    """Build packed uint64 n-gram indexes (sorted unique) plus exact-key hashes.

    Per-eval-set arrays keep attribution for the report without holding
    hundreds of millions of Python string objects in RAM.
    """
    import struct

    index_dir = artifacts_clean() / "decontam_index"
    index_dir.mkdir(parents=True, exist_ok=True)
    for old in index_dir.glob("*.raw"):
        old.unlink()

    exact_meta: dict[int, tuple[str, str]] = {}
    short_under_6 = 0
    n_docs = 0
    # eval_set -> open raw binary file handles for segments / grams13
    seg_files: dict[str, Any] = {}
    g13_files: dict[str, Any] = {}

    def _fh(store: dict[str, Any], eval_set: str, kind: str):
        if eval_set not in store:
            path = index_dir / f"{kind}_{eval_set.replace('/', '_')}.raw"
            store[eval_set] = path.open("wb")
        return store[eval_set]

    try:
        # Exact keys from every eval source.
        for eval_set, doc_id, text in iter_eval_texts():
            key = match_key(text)
            if not key:
                continue
            ek = exact_u64(key)
            if ek not in exact_meta:
                exact_meta[ek] = (eval_set, doc_id)
            n_docs += 1
            if len(words(key)) < 6:
                short_under_6 += 1
            if n_docs % 5000 == 0:
                print(f"  exact keys docs={n_docs} exact={len(exact_meta)}", flush=True)

        # 6–12 / 13-gram inventory from short eval sets only.
        n_ngram_docs = 0
        for eval_set, doc_id, text in iter_eval_texts(ngram=True):
            key = match_key(text)
            if not key:
                continue
            w = words(key)
            wh = word_u64s(w)
            n_ngram_docs += 1
            if len(wh) >= 6:
                fh = _fh(seg_files, eval_set, "seg")
                for n in range(6, 13):
                    if len(wh) < n:
                        break
                    for i in range(len(wh) - n + 1):
                        fh.write(struct.pack("<Q", ngram_u64_from_words(wh, i, n)))
            if len(wh) >= 13:
                fh = _fh(g13_files, eval_set, "g13")
                for i in range(len(wh) - 12):
                    fh.write(struct.pack("<Q", ngram_u64_from_words(wh, i, 13)))
            if n_ngram_docs % 2000 == 0:
                print(f"  ngram docs={n_ngram_docs}", flush=True)
        print(f"  ngram source docs={n_ngram_docs}", flush=True)
    finally:
        for fh in seg_files.values():
            fh.close()
        for fh in g13_files.values():
            fh.close()

    print("  sorting unique segment / 13-gram hashes per eval set", flush=True)
    segment_arrays: dict[str, Any] = {}
    grams13_arrays: dict[str, Any] = {}
    n_segments = 0
    n_grams13 = 0
    for raw in sorted(index_dir.glob("seg_*.raw")):
        eval_set = raw.name[len("seg_") : -len(".raw")]
        arr = _unique_sorted_u64(raw)
        segment_arrays[eval_set] = arr
        n_segments += int(arr.size)
        out = index_dir / f"seg_{eval_set}.u64"
        arr.tofile(out)
        raw.unlink()
        print(f"  segments[{eval_set}]={arr.size}", flush=True)
    for raw in sorted(index_dir.glob("g13_*.raw")):
        eval_set = raw.name[len("g13_") : -len(".raw")]
        arr = _unique_sorted_u64(raw)
        grams13_arrays[eval_set] = arr
        n_grams13 += int(arr.size)
        out = index_dir / f"g13_{eval_set}.u64"
        arr.tofile(out)
        raw.unlink()
        print(f"  grams13[{eval_set}]={arr.size}", flush=True)

    print("  merging per-set hashes into ALL arrays", flush=True)
    all_seg = _merge_sorted_u64_arrays(list(segment_arrays.values()))
    all_g13 = _merge_sorted_u64_arrays(list(grams13_arrays.values()))
    all_seg.tofile(index_dir / "seg_ALL.u64")
    all_g13.tofile(index_dir / "g13_ALL.u64")
    seg_order = sorted(segment_arrays.keys(), key=lambda s: int(segment_arrays[s].size))
    g13_order = sorted(grams13_arrays.keys(), key=lambda s: int(grams13_arrays[s].size))

    print(
        f"  eval index ready docs={n_docs} exact={len(exact_meta)} "
        f"segments={n_segments} (merged={all_seg.size}) "
        f"grams13={n_grams13} (merged={all_g13.size}) short<6={short_under_6}",
        flush=True,
    )
    return {
        "exact": exact_meta,
        "segments": segment_arrays,
        "grams13": grams13_arrays,
        "all_segments": all_seg,
        "all_grams13": all_g13,
        "seg_order": seg_order,
        "g13_order": g13_order,
        "short_under_6": short_under_6,
        "n_exact": len(exact_meta),
        "n_segments": int(all_seg.size),
        "n_grams13": int(all_g13.size),
        "index_dir": str(index_dir),
        "hash": "xxhash64",
    }


def _merge_sorted_u64_arrays(arrays: list[Any]) -> Any:
    """K-way merge of sorted unique uint64 arrays into one sorted unique array."""
    import heapq

    import numpy as np

    arrays = [a for a in arrays if a is not None and getattr(a, "size", 0)]
    if not arrays:
        return np.empty(0, dtype="<u8")
    if len(arrays) == 1:
        return arrays[0]

    heap: list[tuple[int, int, int]] = []
    for i, arr in enumerate(arrays):
        heapq.heappush(heap, (int(arr[0]), i, 0))

    out = np.empty(sum(int(a.size) for a in arrays), dtype="<u8")
    n_out = 0
    last: int | None = None
    while heap:
        val, i, pos = heapq.heappop(heap)
        if last is None or val != last:
            out[n_out] = val
            n_out += 1
            last = val
        pos += 1
        if pos < arrays[i].size:
            heapq.heappush(heap, (int(arrays[i][pos]), i, pos))
    return out[:n_out].copy()


def load_or_build_eval_index() -> dict[str, Any]:
    """Build or load xxhash-packed indexes. Old blake2b files are ignored.

    Membership uses memmaps of the merged ALL arrays only (~3 GB on disk,
    paged by the OS). Per-eval-set files are opened on demand for attribution
    so we do not hold duplicate copies of every hash in RAM.
    """
    import numpy as np

    index_dir = artifacts_clean() / "decontam_index"
    marker = index_dir / "HASH_XXHASH64"
    all_seg_path = index_dir / "seg_ALL.u64"
    all_g13_path = index_dir / "g13_ALL.u64"
    if not (marker.exists() and all_seg_path.exists() and all_g13_path.exists()):
        if index_dir.exists():
            import shutil

            shutil.rmtree(index_dir)
        index = build_eval_index()
        marker.write_text("xxhash64\n", encoding="utf-8")
        # Rebuild returns in-memory arrays; re-open as memmaps for steady state.
        return load_or_build_eval_index()

    print(f"loading packed n-gram indexes from {index_dir}", flush=True)
    # Short-eval indexes fit in RAM; avoid memmap page-fault latency.
    all_seg = np.fromfile(all_seg_path, dtype="<u8")
    all_g13 = np.fromfile(all_g13_path, dtype="<u8")
    # Bloom optional: only worth it for huge indexes. Small ones searchsorted fine.
    bloom_seg = None
    bloom_g13 = None
    if all_seg.size > 20_000_000:
        bloom_seg_path = index_dir / "bloom_segments.npy"
        bloom_g13_path = index_dir / "bloom_grams13.npy"
        if bloom_seg_path.exists() and bloom_g13_path.exists():
            print("loading bloom filters", flush=True)
            bloom_seg = U64Bloom.load(bloom_seg_path)
            bloom_g13 = U64Bloom.load(bloom_g13_path)
        else:
            print("building bloom filters (one-time)", flush=True)
            bloom_seg = U64Bloom()
            bloom_seg.add_array(all_seg)
            bloom_seg.save(bloom_seg_path)
            bloom_g13 = U64Bloom()
            bloom_g13.add_array(all_g13)
            bloom_g13.save(bloom_g13_path)
            print("bloom filters saved", flush=True)

    print("rebuilding exact-key hash map", flush=True)
    exact_meta: dict[int, tuple[str, str]] = {}
    short_under_6 = 0
    n_docs = 0
    for eval_set, doc_id, text in iter_eval_texts():
        key = match_key(text)
        if not key:
            continue
        ek = exact_u64(key)
        if ek not in exact_meta:
            exact_meta[ek] = (eval_set, doc_id)
        n_docs += 1
        if len(words(key)) < 6:
            short_under_6 += 1

    print(
        f"  eval index ready docs={n_docs} exact={len(exact_meta)} "
        f"segments={all_seg.size} grams13={all_g13.size} short<6={short_under_6}",
        flush=True,
    )
    return {
        "exact": exact_meta,
        "all_segments": all_seg,
        "all_grams13": all_g13,
        "bloom_segments": bloom_seg,
        "bloom_grams13": bloom_g13,
        "short_under_6": short_under_6,
        "n_exact": len(exact_meta),
        "n_segments": int(all_seg.size),
        "n_grams13": int(all_g13.size),
        "index_dir": str(index_dir),
        "hash": "xxhash64",
    }


def _doc_ngram_hashes(w: list[str], n_min: int, n_max: int) -> Any:
    import numpy as np
    import xxhash

    if len(w) < n_min:
        return np.empty(0, dtype="<u8")
    vals: list[int] = []
    for n in range(n_min, n_max + 1):
        if len(w) < n:
            break
        for i in range(len(w) - n + 1):
            vals.append(
                xxhash.xxh64_intdigest(" ".join(w[i : i + n]).encode("utf-8"))
            )
    if not vals:
        return np.empty(0, dtype="<u8")
    arr = np.fromiter(vals, dtype="<u8", count=len(vals))
    arr.sort()
    return np.unique(arr)


def _intersects(doc_hashes: Any, sorted_arr: Any) -> bool:
    import numpy as np

    if doc_hashes.size == 0 or sorted_arr.size == 0:
        return False
    idx = np.searchsorted(sorted_arr, doc_hashes)
    ok = idx < sorted_arr.size
    if not np.any(ok):
        return False
    return bool(np.any(sorted_arr[idx[ok]] == doc_hashes[ok]))


def _first_hit(doc_hashes: Any, arrays: dict[str, Any], order: list[str]) -> str | None:
    if doc_hashes.size == 0:
        return None
    for eval_set in order:
        if _intersects(doc_hashes, arrays[eval_set]):
            return eval_set
    return None


class U64Bloom:
    """In-RAM bloom filter over uint64 keys for fast negative rejection."""

    def __init__(self, n_bits: int = 1 << 31, bits: Any | None = None) -> None:
        import numpy as np

        assert n_bits % 64 == 0
        self.n_bits = int(n_bits)
        self.bits = (
            bits
            if bits is not None
            else np.zeros(self.n_bits // 64, dtype=np.uint64)
        )

    def add_array(self, values: Any) -> None:
        import numpy as np

        n_bits = np.uint64(self.n_bits)
        for i in range(0, int(values.size), 2_000_000):
            chunk = np.asarray(values[i : i + 2_000_000], dtype=np.uint64)
            mixes = (
                chunk,
                (chunk >> np.uint64(17)) | (chunk << np.uint64(47)),
                chunk * np.uint64(0x9E3779B97F4A7C15),
                (chunk ^ np.uint64(0xD1B54A32D192ED03)) * np.uint64(0x94D049BB133111EB),
            )
            for mixed in mixes:
                hs = mixed % n_bits
                words = (hs // np.uint64(64)).astype(np.int64)
                bits = (hs % np.uint64(64)).astype(np.uint64)
                np.bitwise_or.at(self.bits, words, np.uint64(1) << bits)
            done = min(i + 2_000_000, int(values.size))
            if (i // 2_000_000) % 25 == 0:
                print(f"  bloom filled {done}/{int(values.size)}", flush=True)

    def __contains__(self, value: int) -> bool:
        import numpy as np

        x = np.uint64(value)
        mixes = (
            x,
            (x >> np.uint64(17)) | (x << np.uint64(47)),
            x * np.uint64(0x9E3779B97F4A7C15),
            (x ^ np.uint64(0xD1B54A32D192ED03)) * np.uint64(0x94D049BB133111EB),
        )
        n_bits = np.uint64(self.n_bits)
        for mixed in mixes:
            h = int(mixed % n_bits)
            word, bit = divmod(h, 64)
            if (int(self.bits[word]) & (1 << bit)) == 0:
                return False
        return True

    def save(self, path: Path) -> None:
        import numpy as np

        np.save(path, self.bits)

    @classmethod
    def load(cls, path: Path, n_bits: int = 1 << 31) -> "U64Bloom":
        import numpy as np

        bits = np.load(path, mmap_mode=None)
        return cls(n_bits=n_bits, bits=bits)


def _confirmed_in_sorted(sorted_arr: Any, value: int) -> bool:
    import numpy as np

    j = int(np.searchsorted(sorted_arr, value))
    return j < sorted_arr.size and int(sorted_arr[j]) == value


def classify_doc(text: str, index: dict[str, Any]) -> tuple[str | None, dict[str, Any]]:
    """Return (reason, detail) if the doc must be dropped, else (None, {})."""
    import numpy as np

    key = match_key(text)
    ek = exact_u64(key)
    if ek in index["exact"]:
        eval_set, eval_id = index["exact"][ek]
        return "exact", {"eval_set": eval_set, "eval_id": eval_id, "match_key": key}

    w = words(key)
    wh = word_u64s(w)
    all_seg = index["all_segments"]
    if all_seg.size and len(wh) >= 6:
        vals = all_ngram_u64s(wh, 6, 12)
        arr = np.fromiter(vals, dtype=np.uint64, count=len(vals))
        ip = np.searchsorted(all_seg, arr)
        ok = ip < all_seg.size
        if np.any(ok) and np.any(all_seg[ip[ok]] == arr[ok]):
            return "segment", {
                "eval_set": "eval_ngram",
                "eval_id": "",
                "n_words": 6,
                "match_key": key,
            }

    all_g13 = index["all_grams13"]
    if all_g13.size and len(wh) >= 13:
        vals = all_ngram_u64s(wh, 13, 13)
        arr = np.fromiter(vals, dtype=np.uint64, count=len(vals))
        ip = np.searchsorted(all_g13, arr)
        ok = ip < all_g13.size
        if np.any(ok) and np.any(all_g13[ip[ok]] == arr[ok]):
            # Recover a matching gram string for the borderline TSV.
            gram = ""
            for i, val in enumerate(vals):
                j = int(np.searchsorted(all_g13, val))
                if j < all_g13.size and int(all_g13[j]) == val:
                    gram = " ".join(w[i : i + 13])
                    break
            return "13gram", {
                "eval_set": "eval_ngram",
                "eval_id": "",
                "gram": gram,
                "match_key": key,
            }
    return None, {}


def _decontam_one_shard(
    shard_path: Path, out_path: Path, index: dict[str, Any]
) -> dict[str, Any]:
    """Classify one parquet shard; write kept rows to out_path."""
    import pyarrow.parquet as pq

    table = pq.read_table(shard_path)
    cols = {name: table.column(name) for name in table.column_names}
    kept_rows: list[dict[str, Any]] = []
    per_eval: dict[str, dict[str, int]] = defaultdict(
        lambda: {"docs": 0, "bytes": 0, "exact": 0, "segment": 0, "13gram": 0}
    )
    borderline: list[str] = []
    length_hist: dict[str, int] = defaultdict(int)
    dropped = 0
    dropped_bytes = 0
    scanned = 0
    for i in range(table.num_rows):
        row = {name: cols[name][i].as_py() for name in cols}
        scanned += 1
        reason, detail = classify_doc(row["text"], index)
        if reason is None:
            kept_rows.append(row)
            continue
        dropped += 1
        nbytes = utf8_len(row["text"])
        dropped_bytes += nbytes
        ev = detail.get("eval_set", "unknown")
        per_eval[ev]["docs"] += 1
        per_eval[ev]["bytes"] += nbytes
        per_eval[ev][reason] += 1
        if reason == "segment":
            length_hist[str(detail.get("n_words"))] += 1
        if reason == "13gram":
            borderline.append(
                f"{row.get('source','')}\t{row['doc_id']}\t{detail.get('eval_set')}\t"
                f"{detail.get('eval_id')}\t{detail.get('gram')}"
            )
    if kept_rows:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(
            __import__("pyarrow").Table.from_pylist(kept_rows, schema=CLEAN_SCHEMA),
            out_path,
            compression="zstd",
        )
    return {
        "scanned": scanned,
        "kept": len(kept_rows),
        "dropped": dropped,
        "dropped_bytes": dropped_bytes,
        "per_eval": {k: dict(v) for k, v in per_eval.items()},
        "length_hist": dict(length_hist),
        "borderline": borderline,
        "out": str(out_path) if kept_rows else "",
    }


def decontam_directory(
    src: Path,
    *,
    label: str,
    index: dict[str, Any],
    report: dict[str, Any],
) -> None:
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import shutil

    if not src.exists():
        return
    safe = label.replace("/", "_").replace(" ", "_")
    tmp = REPO_ROOT / "data" / "clean" / f".tmp_decontam_{safe}"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)

    shards = list_shards(src)
    if not shards:
        return
    n_workers = min(workers(), max(1, len(shards)))
    print(
        f"decontam {label}: {len(shards)} shards, {n_workers} threads",
        flush=True,
    )
    per_eval: dict[str, dict[str, int]] = defaultdict(
        lambda: {"docs": 0, "bytes": 0, "exact": 0, "segment": 0, "13gram": 0}
    )
    borderline: list[str] = []
    length_hist: dict[str, int] = defaultdict(int)
    dropped = 0
    dropped_bytes = 0
    scanned = 0
    kept_files: list[Path] = []

    def _job(shard: Path) -> dict[str, Any]:
        out = tmp / shard.name
        return _decontam_one_shard(shard, out, index)

    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futs = {pool.submit(_job, s): s for s in shards}
        done = 0
        for fut in as_completed(futs):
            stats = fut.result()
            done += 1
            scanned += stats["scanned"]
            dropped += stats["dropped"]
            dropped_bytes += stats["dropped_bytes"]
            for ev, d in stats["per_eval"].items():
                for k, v in d.items():
                    per_eval[ev][k] += v
            for k, v in stats["length_hist"].items():
                length_hist[k] += v
            borderline.extend(stats["borderline"])
            if stats["out"]:
                kept_files.append(Path(stats["out"]))
            if done % 20 == 0 or done == len(shards):
                print(
                    f"  {label} shards={done}/{len(shards)} scanned={scanned} "
                    f"dropped={dropped} free_bytes={free_bytes()}",
                    flush=True,
                )

    # Rewrite into canonical shard layout via ShardWriter (sorted by name).
    writer = ShardWriter(tmp / "_final", schema=CLEAN_SCHEMA)
    import pyarrow.parquet as pq

    for path in sorted(kept_files):
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
    shutil.rmtree(tmp)

    man_path = src / "manifest.json"
    if man_path.exists():
        man = load_json(man_path)
        man["docs"] = writer.docs
        man["text_bytes"] = writer.text_bytes
        man["shards"] = writer.shards
        man["decontaminated"] = True
        man["decontam_dropped_docs"] = dropped
        man["decontam_dropped_bytes"] = dropped_bytes
        save_json(man_path, finish_manifest(man))
    else:
        save_json(
            man_path,
            finish_manifest(
                {
                    **base_manifest(stage="decontam"),
                    "docs": writer.docs,
                    "text_bytes": writer.text_bytes,
                    "shards": writer.shards,
                    "decontaminated": True,
                    "decontam_dropped_docs": dropped,
                    "decontam_dropped_bytes": dropped_bytes,
                }
            ),
        )

    report["sources"][label] = {
        "kept_docs": writer.docs,
        "kept_bytes": writer.text_bytes,
        "dropped_docs": dropped,
        "dropped_bytes": dropped_bytes,
        "per_eval": dict(per_eval),
        "segment_length_hist": dict(length_hist),
    }
    report["borderline"].extend(borderline)
    print(
        f"decontam {label} kept={writer.docs} dropped={dropped} "
        f"dropped_bytes={dropped_bytes} scanned={scanned}",
        flush=True,
    )


def decontam_source(source_id: str, index: dict[str, Any], report: dict[str, Any]) -> None:
    decontam_directory(clean_dir(source_id), label=source_id, index=index, report=report)


def decontam_reroutes(index: dict[str, Any], report: dict[str, Any]) -> None:
    """Decontaminate LID-rerouted docs under data/clean/_reroute (used in snapshots)."""
    root = REPO_ROOT / "data" / "clean" / "_reroute"
    if not root.exists():
        return
    for bucket_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for src_dir in sorted(p for p in bucket_dir.iterdir() if p.is_dir()):
            label = f"_reroute/{bucket_dir.name}/{src_dir.name}"
            decontam_directory(src_dir, label=label, index=index, report=report)


def run_decontam() -> dict[str, Any]:
    mix = load_mix()
    print("building eval decontamination index", flush=True)
    index = load_or_build_eval_index()
    print(
        f"eval index exact={index['n_exact']} segments={index['n_segments']} "
        f"grams13={index['n_grams13']} short_under_6={index['short_under_6']}",
        flush=True,
    )
    report: dict[str, Any] = {
        **base_manifest(stage="decontam"),
        "match_keys": "NFC + collapsed whitespace (temporary copies only)",
        "short_under_6": index["short_under_6"],
        "n_exact": index["n_exact"],
        "n_segments": index["n_segments"],
        "n_grams13": index["n_grams13"],
        "sources": {},
        "borderline": [],
    }
    for source_id in train_source_ids(mix):
        decontam_source(source_id, index, report)
    decontam_reroutes(index, report)

    borderline_path = artifacts_clean() / "decontam_borderline.tsv"
    borderline_path.write_text(
        "source_id\tdoc_id\teval_set\teval_id\tgram\n"
        + "\n".join(report["borderline"])
        + ("\n" if report["borderline"] else ""),
        encoding="utf-8",
    )
    report.pop("borderline")
    report["borderline_path"] = str(borderline_path.relative_to(REPO_ROOT))
    report = finish_manifest(report)
    save_json(artifacts_clean() / "decontam_report.json", report)

    # Merge into report.json if present.
    main_report_path = artifacts_clean() / "report.json"
    if main_report_path.exists():
        main = load_json(main_report_path)
        main["decontam"] = {
            "short_under_6": report["short_under_6"],
            "n_exact": report["n_exact"],
            "n_segments": report["n_segments"],
            "n_grams13": report["n_grams13"],
            "sources": report["sources"],
        }
        save_json(main_report_path, main)

    print(f"text decontam done; running MinHash near-dup vs large eval sets", flush=True)
    from ancoda_tok.clean.minhash_eval import run_minhash_eval_near_dup

    mh = run_minhash_eval_near_dup()
    report["minhash_eval_near_dup"] = {
        "total_dropped_docs": mh["total_dropped_docs"],
        "total_dropped_bytes": mh["total_dropped_bytes"],
    }
    print(f"decontam done free_bytes={free_bytes()}", flush=True)
    return report


def _audit_one_shard(shard_path: Path, index: dict[str, Any]) -> dict[str, int]:
    """Count residual contamination in one snapshot shard (exact/segment/13gram)."""
    import pyarrow.parquet as pq

    table = pq.read_table(shard_path, columns=["text"])
    texts = table.column("text")
    counts = {"exact": 0, "segment": 0, "13gram": 0}
    for i in range(table.num_rows):
        reason, _ = classify_doc(texts[i].as_py(), index)
        if reason:
            counts[reason] += 1
    return counts


def audit_snapshot(snapshot_dir: Path) -> dict[str, int]:
    """Re-run every applicable decontam rule; return counts (must be 0)."""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    from ancoda_tok.clean.minhash_eval import (
        audit_minhash_snapshot,
        build_or_load_eval_bands,
    )

    index = load_or_build_eval_index()
    shards = list_shards(snapshot_dir)
    counts = {"exact": 0, "segment": 0, "13gram": 0, "minhash_eval": 0}
    if not shards:
        return counts
    n_workers = min(workers(), max(1, len(shards)))
    print(
        f"audit {snapshot_dir.name}: {len(shards)} shards, {n_workers} threads",
        flush=True,
    )
    done = 0
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(_audit_one_shard, shard, index): shard for shard in shards}
        for fut in as_completed(futures):
            part = fut.result()
            for k, v in part.items():
                counts[k] += v
            done += 1
            if done == len(shards) or done % 10 == 0:
                print(
                    f"  {snapshot_dir.name} shards={done}/{len(shards)} so_far={counts}",
                    flush=True,
                )
    print(f"audit {snapshot_dir.name}: MinHash near-dup vs large eval sets", flush=True)
    mix = load_mix()
    bands = build_or_load_eval_bands(mix)
    counts["minhash_eval"] = audit_minhash_snapshot(snapshot_dir, bands=bands)
    return counts

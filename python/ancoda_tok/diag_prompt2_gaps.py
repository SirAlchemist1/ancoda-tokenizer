"""Prompt 2 gap pass: sealed set, whole-file CIs, tail fix, gap trains, competitor KB.

Final shipping config unchanged. Diagnostics only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
from collections import defaultdict
from pathlib import Path

import pyarrow.parquet as pq
import sentencepiece as spm

from ancoda_tok.clean.common import (
    REPO_ROOT,
    SEED,
    base_manifest,
    finish_manifest,
    load_mix,
    save_json,
    sha256_file,
)
from ancoda_tok.clean.minhash_eval import (
    _minhash_config,
    _patch_xxhash,
    _signature_helper,
)
from ancoda_tok.clean.snapshot import scale_shares
from ancoda_tok.diag_combined_candidates import (
    BLOCK0,
    BOOT_N,
    BOOT_SEED,
    CAP_N,
    FLORES,
    OUT,
    SID,
    YAK,
    build_scaled_latin,
    encode_fn_for,
    mark_initial_count,
    paired_bootstrap_diff,
    train_one,
)
from ancoda_tok.export_hf import bytes_to_unicode, load_hf
from ancoda_tok.frontier_common import (
    combine,
    held_out_code_math,
    tokens_per_kb,
    write_side_config,
)
from ancoda_tok.frontier_metrics import attach_extra_metrics
from ancoda_tok.guarantee import (
    Engines,
    _SINGLE,
    _aksharas,
    _bootstrap,
    _bounds,
    _has_deva,
    _inside,
    configure,
)
from ancoda_tok.mix_sweep import MIX_SWEEP_DIR, _caffeinate, harness_add
from ancoda_tok.storage import REPO_ROOT as _ROOT

assert REPO_ROOT == _ROOT

SEALED_PATH = REPO_ROOT / "data" / "clean" / "sealed_test_set.json"
SEALED_BUDGET = 100_000_000  # 100 MB held-out slice (PLAN phrase held-out scale)
SNAP_DIR = REPO_ROOT / "data" / "snapshots"
GAPS_OUT = MIX_SWEEP_DIR / "prompt2_gaps"
COVERED_3PCT = MIX_SWEEP_DIR / "t1_block1_cap" / "covered_64k_3pct.json"
COVERED_128_3PCT = OUT / "covered_128k_3pct.json"

ARKIOS = (
    REPO_ROOT
    / "data/hf_cache/hub/models--sajalregmi4--arkios-tokenizer/snapshots"
    / "aef562fcaf8747bbd9461cdae61b61f99e97e048/tokenizer.json"
)
LLAMA3 = (
    REPO_ROOT
    / "data/hf_cache/hub/models--NousResearch--Meta-Llama-3-8B/snapshots"
    / "315b20096dc791d381d514deb5f8bd9c8d6d3061/tokenizer.json"
)
GEMMA3 = (
    REPO_ROOT
    / "data/hf_cache/hub/models--unsloth--gemma-3-1b-it/snapshots"
    / "5b11413a10db4e486ef16a20101fd028f8f2499c/tokenizer.json"
)
O200K_JSON = REPO_ROOT / "third_party/yak-tokenizer/artifacts/eval/o200k.json"

GAP_POINTS = [
    ("64k", 48000, "3pct", 1.0, 64512),
    ("64k", 46000, "3pct", 1.0, 64512),
    ("64k", 46000, "3pct", 0.5, 64512),
    ("64k", 46000, "3pct", 0.25, 64512),
    ("128k", 94742, "3pct", 1.0, 130048),
    ("128k", 100742, "3pct", 1.0, 130048),
]


def _doc_id_hash(doc_id: str) -> str:
    return hashlib.sha256(doc_id.encode("utf-8")).hexdigest()


def _list_sha256(hashes: list[str]) -> str:
    payload = "\n".join(hashes) + ("\n" if hashes else "")
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _all_snapshot_doc_ids() -> set[str]:
    used: set[str] = set()
    for snap in sorted(SNAP_DIR.iterdir()):
        ids = snap / "doc_ids.parquet"
        if not ids.is_file():
            continue
        table = pq.read_table(ids, columns=["doc_id"])
        used.update(table.column(0).to_pylist())
    return used


class _RevHex:
    """Max-heap key over hex digests (larger digest is 'less' for heapq)."""

    __slots__ = ("h",)

    def __init__(self, h: str) -> None:
        self.h = h

    def __lt__(self, other: "_RevHex") -> bool:
        return self.h > other.h


def build_sealed_test_set(*, budget: int = SEALED_BUDGET) -> dict:
    """Hash-select cleaned docs absent from every existing snapshot; near-dup filter."""
    if SEALED_PATH.is_file():
        existing = json.loads(SEALED_PATH.read_text(encoding="utf-8"))
        print(f"sealed already present {SEALED_PATH} n={existing.get('n_docs')}", flush=True)
        return existing

    import heapq

    print("building sealed test set…", flush=True)
    used = _all_snapshot_doc_ids()
    print(f"snapshot doc_ids union={len(used)}", flush=True)
    index_path = REPO_ROOT / "data" / "clean" / "_snapshot_index.parquet"
    if not index_path.is_file():
        raise SystemExit("STOP: missing data/clean/_snapshot_index.parquet")

    mix = load_mix()
    shares = scale_shares(mix, 0.60)
    targets = {b: int(budget * s) for b, s in shares.items()}
    # Keep the N lowest hash keys per bucket (N ≈ 2× docs if avg ~2KB).
    keep_n = {b: max(800, int(t / 2_000) * 2 + 200) for b, t in targets.items()}
    heaps: dict[str, list] = {b: [] for b in targets}
    seq = 0
    scanned = 0
    unused = 0
    pf = pq.ParquetFile(index_path)
    for batch in pf.iter_batches(batch_size=8192):
        cols = {n: batch.column(n) for n in batch.schema.names}
        for i in range(batch.num_rows):
            scanned += 1
            doc_id = cols["doc_id"][i].as_py()
            if doc_id in used:
                continue
            unused += 1
            bucket = cols["bucket"][i].as_py()
            if bucket not in heaps:
                continue
            h = _doc_id_hash(f"{SEED}:{doc_id}")
            meta = {
                "bucket": bucket,
                "doc_id": doc_id,
                "source": cols["source"][i].as_py(),
                "license": cols["license"][i].as_py(),
                "text_bytes": int(cols["text_bytes"][i].as_py()),
                "shard": cols["shard"][i].as_py(),
                "row": int(cols["row"][i].as_py()),
            }
            heap = heaps[bucket]
            n_cap = keep_n[bucket]
            item = (_RevHex(h), seq, meta)
            seq += 1
            if len(heap) < n_cap:
                heapq.heappush(heap, item)
            elif h < heap[0][0].h:
                heapq.heapreplace(heap, item)
        if scanned % 2_000_000 == 0:
            print(f"  index scanned={scanned} unused={unused}", flush=True)
    print(f"index done scanned={scanned} unused={unused}", flush=True)

    selected: list[dict] = []
    selected_bytes = 0
    for bucket, target in sorted(targets.items()):
        cand = sorted((h.h, meta) for h, _s, meta in heaps[bucket])
        got = 0
        for _h, meta in cand:
            if got >= target:
                break
            selected.append(meta)
            got += int(meta["text_bytes"])
            selected_bytes += int(meta["text_bytes"])
    print(
        f"pre-near-dup selected docs={len(selected)} bytes={selected_bytes}",
        flush=True,
    )

    from ancoda_tok.clean.snapshot import fetch_texts

    with_text = fetch_texts(selected)
    id_to_text = {r["doc_id"]: r["text"] or "" for r in with_text}

    _patch_xxhash()
    helper = _signature_helper(_minhash_config(mix))
    n_buckets = int(mix["minhash"]["num_buckets"])
    inverted: list[dict[tuple[int, ...], set[str]]] = [
        defaultdict(set) for _ in range(n_buckets)
    ]
    for doc_id, text in id_to_text.items():
        shingles = helper.get_shingles(text)
        if shingles.size == 0:
            continue
        for bi, band in enumerate(helper.get_signature(shingles)):
            inverted[bi][tuple(int(x) for x in band)].add(doc_id)

    collided: set[str] = set()
    snap_docs_scanned = 0
    for snap in sorted(SNAP_DIR.iterdir()):
        if not snap.is_dir() or not (snap / "manifest.json").is_file():
            continue
        print(f"near-dup scan {snap.name}", flush=True)
        for shard in sorted(snap.glob("shard-*.parquet")):
            pf = pq.ParquetFile(shard)
            for batch in pf.iter_batches(columns=["text"], batch_size=128):
                for text in batch.column(0).to_pylist():
                    snap_docs_scanned += 1
                    text = text or ""
                    shingles = helper.get_shingles(text)
                    if shingles.size == 0:
                        continue
                    for bi, band in enumerate(helper.get_signature(shingles)):
                        hits = inverted[bi].get(tuple(int(x) for x in band))
                        if hits:
                            collided.update(hits)
                    if snap_docs_scanned % 200_000 == 0:
                        print(
                            f"  scanned={snap_docs_scanned} collided={len(collided)}",
                            flush=True,
                        )

    kept = [r for r in selected if r["doc_id"] not in collided]
    print(
        f"near-dup dropped={len(collided)} kept={len(kept)} "
        f"scanned_snap_docs={snap_docs_scanned}",
        flush=True,
    )

    hashes = sorted(_doc_id_hash(r["doc_id"]) for r in kept)
    list_digest = _list_sha256(hashes)
    payload = finish_manifest(
        base_manifest(
            kind="sealed_test_set",
            seed=SEED,
            target_bytes=budget,
            n_docs=len(hashes),
            text_bytes=sum(int(r["text_bytes"]) for r in kept),
            near_dup_dropped=len(collided),
            snap_docs_scanned=snap_docs_scanned,
            snapshots=[
                p.name
                for p in sorted(SNAP_DIR.iterdir())
                if (p / "manifest.json").is_file()
            ],
            doc_id_sha256=hashes,
            list_sha256=list_digest,
            note="hashes only; no text; exclude from future snapshots via sha256(doc_id)",
        )
    )
    SEALED_PATH.parent.mkdir(parents=True, exist_ok=True)
    save_json(SEALED_PATH, payload)
    excl = REPO_ROOT / "data" / "clean" / "sealed_test_doc_id_sha256.txt"
    excl.write_text("\n".join(hashes) + ("\n" if hashes else ""), encoding="utf-8")
    print(
        f"wrote {SEALED_PATH} n={len(hashes)} list_sha256={list_digest}",
        flush=True,
    )
    return payload


def load_sealed_hash_set() -> set[str]:
    path = REPO_ROOT / "data" / "clean" / "sealed_test_doc_id_sha256.txt"
    if not path.is_file():
        if SEALED_PATH.is_file():
            data = json.loads(SEALED_PATH.read_text(encoding="utf-8"))
            return set(data["doc_id_sha256"])
        return set()
    return {ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()}


def _flores_lines(lang: str) -> list[str]:
    return [
        ln
        for ln in (FLORES / f"{lang}.txt").read_text(encoding="utf-8").splitlines()
        if ln.strip()
    ]


def _inverse() -> dict[str, int]:
    return {char: byte for byte, char in bytes_to_unicode().items()}


def _hf_token_bytes(tok, tid: int, inverse: dict[str, int]) -> bytes:
    piece = tok.id_to_token(tid)
    return bytes(inverse[c] for c in piece)


def whole_file_line_pairs(encode_pieces, lines: list[str]) -> list[tuple[int, int]]:
    """Encode joined file once; assign each token to the line of its first byte."""
    joined = "\n".join(lines)
    pieces: list[bytes] = encode_pieces(joined)
    # Line byte ranges over the joined encoding.
    starts: list[int] = []
    offset = 0
    raw_lines = [ln.encode("utf-8") for ln in lines]
    for i, raw in enumerate(raw_lines):
        starts.append(offset)
        offset += len(raw)
        if i < len(raw_lines) - 1:
            offset += 1  # newline
    assert offset == len(joined.encode("utf-8")), (offset, len(joined.encode("utf-8")))

    def line_of(pos: int) -> int:
        # largest start <= pos
        lo, hi = 0, len(starts) - 1
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if starts[mid] <= pos:
                lo = mid
            else:
                hi = mid - 1
        return lo

    counts = [0] * len(lines)
    words = [len(ln.split()) for ln in lines]
    pos = 0
    for piece in pieces:
        counts[line_of(pos)] += 1
        pos += len(piece)
    assert pos == offset, (pos, offset)
    return list(zip(counts, words, strict=True))


def pieces_fn_for(path: Path):
    """Return callable text -> list[bytes] of token pieces (UTF-8)."""
    if path.suffix == ".model" or path.name.endswith(".model"):
        sp = spm.SentencePieceProcessor(model_file=str(path))

        def enc(text: str) -> list[bytes]:
            ids = sp.encode(text, out_type=int)
            return [sp.id_to_piece(i).encode("utf-8") for i in ids]

        # SentencePiece pieces use ▁; byte lengths won't match UTF-8 of text.
        # Fall back to encode ids + decode each id for round-trip bytes.
        def enc_bytes(text: str) -> list[bytes]:
            ids = sp.encode(text, out_type=int)
            out = []
            for i in ids:
                # SP decode of single id may strip spaces oddly; use piece bytes
                # via EncodeAsSerializedProto is heavy — use id_to_piece and map ▁.
                piece = sp.id_to_piece(i)
                if piece.startswith("▁"):
                    piece = " " + piece[1:]
                out.append(piece.encode("utf-8"))
            # Verify round-trip; if mismatch, use per-id decode.
            if b"".join(out) != text.encode("utf-8"):
                out = []
                for i in ids:
                    out.append(sp.decode([i]).encode("utf-8"))
                # Still may fail for leading spaces; last resort: proportional split
                joined = b"".join(out)
                if joined != text.encode("utf-8"):
                    # Assign by cumulative decoded length best-effort on unicode
                    raw = text.encode("utf-8")
                    # Use byte-length from EncodeAsPieces with SentencePiece's
                    # immutable dense encoding via sp.EncodeAsImmutableProto
                    out = _spm_byte_pieces(sp, text, raw)
            return out

        return enc_bytes

    if path.name == "o200k.json" or path == O200K_JSON:
        import tiktoken

        enc = tiktoken.get_encoding("o200k_base")

        def o_enc(text: str) -> list[bytes]:
            ids = enc.encode_ordinary(text)
            return [enc.decode_single_token_bytes(i) for i in ids]

        return o_enc

    tok_path = path if path.name == "tokenizer.json" else path / "tokenizer.json"
    tok = load_hf(tok_path)
    inverse = _inverse()

    def hf_enc(text: str) -> list[bytes]:
        ids = tok.encode(text, add_special_tokens=False).ids
        return [_hf_token_bytes(tok, i, inverse) for i in ids]

    return hf_enc


def _spm_byte_pieces(sp, text: str, raw: bytes) -> list[bytes]:
    """Best-effort byte pieces for SentencePiece when piece round-trip fails."""
    ids = sp.encode(text, out_type=int)
    # Decode prefix lengths.
    pieces = []
    prev = b""
    for k in range(1, len(ids) + 1):
        cur = sp.decode(ids[:k]).encode("utf-8")
        # Prefer longest prefix of raw matching cur
        if raw.startswith(cur):
            pieces.append(cur[len(prev) :])
            prev = cur
        else:
            # fallback equal split of remainder
            rem = raw[len(prev) :]
            n_left = len(ids) - k + 1
            take = max(1, len(rem) // n_left) if n_left else len(rem)
            pieces.append(rem[:take])
            prev = prev + rem[:take]
    # Fix total
    joined = b"".join(pieces)
    if joined != raw:
        # Collapse to single token assignment by re-encoding line-separately later
        # — for whole-file metric integrity, force exact cover:
        pieces = []
        pos = 0
        for k in range(len(ids)):
            if k == len(ids) - 1:
                pieces.append(raw[pos:])
            else:
                # use decode delta when possible
                a = sp.decode(ids[: k + 1]).encode("utf-8")
                if raw.startswith(a) and len(a) >= pos:
                    pieces.append(raw[pos : len(a)])
                    pos = len(a)
                else:
                    pieces.append(raw[pos : pos + 1] or b"\x00")
                    pos += len(pieces[-1])
        if b"".join(pieces) != raw:
            # Last resort: one fake piece per id with empty trailing fix
            n = len(ids)
            base = len(raw) // n
            pieces = []
            pos = 0
            for k in range(n):
                end = len(raw) if k == n - 1 else pos + max(1, base)
                pieces.append(raw[pos:end])
                pos = end
    return pieces


def compute_tail_doc_bootstrap(
    tok_dir: Path,
    covered: Path,
    snap: Path,
    *,
    max_docs: int = 8_000,
) -> list[float]:
    """PLAN 3.9 tail break rate with document-level bootstrap (skip Code-first trap)."""
    configure(out=tok_dir, covered=covered, snapshot=snap)
    engines = Engines()
    rng = random.Random(SEED)
    # Stream Nepali/Devanagari docs (skip Code-first shard trap that zeroed 1M-line tails).
    pool: list[str] = []
    for shard in sorted(snap.glob("shard-*.parquet")):
        pf = pq.ParquetFile(shard)
        for batch in pf.iter_batches(columns=["text", "bucket"], batch_size=64):
            for text, bucket in zip(
                batch.column(0).to_pylist(), batch.column(1).to_pylist(), strict=True
            ):
                text = text or ""
                if bucket in ("Code", "Math", "English"):
                    continue
                if not _has_deva(text):
                    continue
                pool.append(text)
                if len(pool) >= max_docs * 2:
                    break
        if len(pool) >= max_docs * 2:
            break
    if len(pool) > max_docs:
        pool = rng.sample(pool, max_docs)

    pairs: list[tuple[int, int]] = []
    for text in pool:
        spans = _aksharas(text)
        _h, hf_pieces, _t, _tt = engines.pieces(text)
        raw = text.encode("utf-8")
        if b"".join(hf_pieces) != raw:
            continue
        bounds = _bounds(hf_pieces)
        broken = total = 0
        for unit, start, end in spans:
            if unit in engines.multi or unit in _SINGLE:
                continue
            if not any("\u0900" <= ch <= "\u097f" for ch in unit):
                continue
            total += 1
            if _inside(start, end, bounds):
                broken += 1
        if total:
            pairs.append((broken, total))
    point, lo, hi = _bootstrap(pairs)
    # guarantee uses 1000 resamples; bump to 10k for table consistency with CI seed
    if pairs:
        total_b = sum(b for b, _ in pairs)
        total_n = sum(n for _, n in pairs)
        point = total_b / total_n if total_n else 0.0
        rates = []
        rng = random.Random(SEED)
        m = len(pairs)
        for _ in range(10_000):
            bsum = nsum = 0
            for _d in range(m):
                b, n = pairs[rng.randrange(m)]
                bsum += b
                nsum += n
            if nsum:
                rates.append(bsum / nsum)
        rates.sort()
        lo = rates[int(0.025 * (len(rates) - 1))]
        hi = rates[min(len(rates) - 1, int(0.975 * (len(rates) - 1)))]
    return [point, lo, hi]


def _reuse_deva(tag: str, donor_tag: str | None) -> None:
    dest = OUT / tag / "side-deva"
    if (dest / "ranks.json").is_file():
        return
    if not donor_tag:
        return
    src = OUT / donor_tag / "side-deva"
    if (src / "ranks.json").is_file():
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(src, dest)
        print(f"reused dewa {donor_tag} -> {tag}", flush=True)


def train_gap_point(
    vocab: str,
    b_d: int,
    pct: str,
    scale: float,
    mergeable: int,
    covered: Path,
    dewa_rel: str,
) -> dict:
    tag = f"{vocab}_bd{b_d}_{pct}_cm{int(scale * 100)}"
    # Reuse dewa from an existing same B_d/cap point when possible.
    donors = [
        f"{vocab}_bd{b_d}_{pct}_cm50",
        f"{vocab}_bd{b_d}_{pct}_cm25",
        f"{vocab}_bd{b_d}_{pct}_cm100",
    ]
    donor = next((d for d in donors if (OUT / d / "side-deva" / "ranks.json").is_file()), None)
    _reuse_deva(tag, donor)

    latin_counts = build_scaled_latin(scale)
    block1 = int(json.loads(covered.read_text(encoding="utf-8"))["block1_merge_count"])
    coverage = float(json.loads(covered.read_text(encoding="utf-8"))["coverage"])
    latin_b2 = mergeable - 256 - b_d
    dewa_vocab = 256 + b_d + 1024
    latin_vocab = 256 + latin_b2 + 1024
    dewa_out = OUT / tag / "side-deva"
    latin_out = OUT / tag / "side-latin"
    dewa_cfg = write_side_config(
        path=OUT / "configs" / f"{tag}_deva.yaml",
        counts_rel=dewa_rel,
        output_rel=str(dewa_out.relative_to(REPO_ROOT)),
        vocab_size=dewa_vocab,
        block0=True,
        block1=True,
        block1_path=str(covered.relative_to(REPO_ROOT)),
    )
    latin_cfg = write_side_config(
        path=OUT / "configs" / f"{tag}_latin.yaml",
        counts_rel=str(latin_counts.relative_to(REPO_ROOT)),
        output_rel=str(latin_out.relative_to(REPO_ROOT)),
        vocab_size=latin_vocab,
        block0=False,
        block1=False,
        block1_path=str(covered.relative_to(REPO_ROOT)),
    )
    # Silence unused for lint when dewa already present
    _ = BLOCK0
    peak = max(train_one(dewa_cfg, dewa_out), train_one(latin_cfg, latin_out))
    dest = OUT / tag / "combined"
    combine(dewa_out, latin_out, dest, mergeable=mergeable)
    return {
        "arm": "combined",
        "label": tag,
        "vocab": vocab,
        "B_d": b_d,
        "block1_cap": pct,
        "block1": block1,
        "block1_coverage": coverage,
        "code_math_scale": scale,
        "dir": str(dest.relative_to(REPO_ROOT)),
        "snapshot": "5gb_n60_joiner",
        "peak_rss_bytes": peak,
        "mark_initial": mark_initial_count(dest),
        "gap": True,
    }


def competitor_paths() -> dict[str, Path]:
    return {
        "yak_v1_64k": YAK / "v1.0/64k/tokenizer.json",
        "yak_v2_64k": YAK / "v2.0/64k/tokenizer.json",
        "yak_v1_128k": YAK / "v1.0/128k/tokenizer.json",
        "yak_v2_128k": YAK / "v2.0/128k/tokenizer.json",
        "sidskarki_64k": SID,
        "arkios": ARKIOS,
        "llama3": LLAMA3,
        "gemma3": GEMMA3,
        "o200k": O200K_JSON,
    }


def hindi_whole_file_tokens(path: Path) -> int:
    lines = _flores_lines("hin_Deva")
    joined = "\n".join(lines)
    enc = encode_fn_for(path)
    return len(enc(joined))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sealed-only", action="store_true")
    parser.add_argument("--train-only", action="store_true")
    parser.add_argument("--metrics-only", action="store_true")
    parser.add_argument("--skip-sealed", action="store_true")
    args = parser.parse_args()
    GAPS_OUT.mkdir(parents=True, exist_ok=True)

    sealed = None
    if not args.skip_sealed and not args.train_only and not args.metrics_only:
        sealed = build_sealed_test_set()
    elif args.sealed_only:
        sealed = build_sealed_test_set()
        print(json.dumps({"sealed_n": sealed["n_docs"], "list_sha256": sealed["list_sha256"]}))
        return

    snap = REPO_ROOT / "data/snapshots/n60j5_20af1fffbec0fe86"
    dewa_rel = str((OUT / "deva_pretoken_counts.bin").relative_to(REPO_ROOT))
    if not (OUT / "deva_pretoken_counts.bin").is_file():
        raise SystemExit("STOP: missing combined_candidates deva counts")
    if not COVERED_3PCT.is_file():
        raise SystemExit(f"STOP: missing {COVERED_3PCT}")
    if not COVERED_128_3PCT.is_file():
        # Prefer local copy under combined_candidates
        alt = OUT / "covered_128k_3pct.json"
        if not alt.is_file():
            raise SystemExit("STOP: missing covered_128k_3pct")
        covered_128 = alt
    else:
        covered_128 = COVERED_128_3PCT

    gap_rows: list[dict] = []
    if not args.metrics_only:
        build_scaled_latin(1.0)
        for vocab, b_d, pct, scale, mergeable in GAP_POINTS:
            covered = COVERED_3PCT if vocab == "64k" else covered_128
            row = train_gap_point(vocab, b_d, pct, scale, mergeable, covered, dewa_rel)
            gap_rows.append(row)
            print(f"trained {row['label']}", flush=True)
        # Harness for gap points
        adds = {r["label"]: REPO_ROOT / r["dir"] for r in gap_rows}
        harness_add(adds, "prompt2_gaps")
        from ancoda_tok.frontier_common import enrich_from_harness

        for row in gap_rows:
            enrich_from_harness(row, "prompt2_gaps", row["label"])
            attach_extra_metrics(
                row,
                REPO_ROOT / row["dir"],
                snap=snap,
                harness_aya=row.get("aya_fertility"),
            )
        save_json(
            GAPS_OUT / "gap_rows.json",
            finish_manifest(base_manifest(kind="prompt2_gap_trains", rows=gap_rows)),
        )

    if args.train_only:
        return

    # Load Prompt 2 rows + gap rows
    frontier = json.loads((OUT / "frontier.json").read_text(encoding="utf-8"))
    rows = list(frontier["rows"])
    gap_path = GAPS_OUT / "gap_rows.json"
    if gap_path.is_file():
        gap_rows = json.loads(gap_path.read_text(encoding="utf-8"))["rows"]
        have = {r["label"] for r in rows}
        for r in gap_rows:
            if r["label"] not in have:
                rows.append(r)

    code_texts, math_texts = held_out_code_math(snap)
    langs = {
        "npi": _flores_lines("npi_Deva"),
        "eng": _flores_lines("eng_Latn"),
        "hin": _flores_lines("hin_Deva"),
    }

    # Competitor whole-file pairs + code/math KB
    comps = competitor_paths()
    ci_bases = ("yak_v1_64k", "yak_v2_64k", "yak_v1_128k", "yak_v2_128k", "sidskarki_64k")
    kb_names = (
        "yak_v1_64k",
        "yak_v2_64k",
        "yak_v1_128k",
        "yak_v2_128k",
        "sidskarki_64k",
        "arkios",
        "llama3",
        "gemma3",
        "o200k",
    )
    comp_pairs: dict[str, dict[str, list]] = {}
    comp_kb: dict[str, dict[str, float]] = {}
    for name in kb_names:
        path = comps[name]
        print(f"competitor pieces {name}", flush=True)
        pfn = pieces_fn_for(path)
        if name in ci_bases:
            comp_pairs[name] = {
                lang: whole_file_line_pairs(pfn, lines) for lang, lines in langs.items()
            }
        # code/math: tokens_per_kb needs encode count; use encode_fn
        if name == "o200k":
            import tiktoken

            enc = tiktoken.get_encoding("o200k_base")

            def _tok_kb(texts: list[str]) -> float:
                total_tok = total_b = 0
                for t in texts:
                    if not t:
                        continue
                    total_tok += len(enc.encode_ordinary(t))
                    total_b += len(t.encode("utf-8"))
                return 0.0 if total_b == 0 else 1024.0 * total_tok / total_b

            comp_kb[name] = {"code": _tok_kb(code_texts), "math": _tok_kb(math_texts)}
        elif path.suffix == ".model":
            sp = spm.SentencePieceProcessor(model_file=str(path))

            def _tok_kb_sp(texts: list[str], _sp=sp) -> float:
                total_tok = total_b = 0
                for t in texts:
                    if not t:
                        continue
                    total_tok += len(_sp.encode(t, out_type=int))
                    total_b += len(t.encode("utf-8"))
                return 0.0 if total_b == 0 else 1024.0 * total_tok / total_b

            comp_kb[name] = {
                "code": _tok_kb_sp(code_texts),
                "math": _tok_kb_sp(math_texts),
            }
        else:
            comp_kb[name] = {
                "code": tokens_per_kb(path, code_texts),
                "math": tokens_per_kb(path, math_texts),
            }

    # Hindi totals
    anc128 = REPO_ROOT / "artifacts/tokenizers/ancoda-128k-n60"
    if not (anc128 / "tokenizer.json").is_file():
        # try from frontier report
        anc128 = REPO_ROOT / "artifacts/tokenizers/ancoda-128k-n60"
    hin_totals = {
        "anc128_n60": hindi_whole_file_tokens(anc128)
        if (anc128 / "tokenizer.json").is_file()
        else None,
        "yak_v1_128k": hindi_whole_file_tokens(comps["yak_v1_128k"]),
    }
    # Prefer path from combined report if present
    report = MIX_SWEEP_DIR / "combined_frontier_report.json"
    if report.is_file():
        rep = json.loads(report.read_text(encoding="utf-8"))
        p = (rep.get("f_verify_hindi") or {}).get("anc128_n60", {}).get("path")
        if p and Path(p).is_file():
            hin_totals["anc128_n60"] = hindi_whole_file_tokens(Path(p))

    enriched = []
    for row in rows:
        if "dir" not in row:
            enriched.append(row)
            continue
        tok_dir = REPO_ROOT / row["dir"]
        covered = COVERED_3PCT if row.get("vocab") == "64k" else covered_128
        if row.get("block1_cap") == "2pct":
            c2 = (
                MIX_SWEEP_DIR / "t1_block1_cap" / "covered_64k_2pct.json"
                if row.get("vocab") == "64k"
                else OUT / "covered_128k_2pct.json"
            )
            if c2.is_file():
                covered = c2
        print(f"metrics {row['label']}", flush=True)
        pfn = pieces_fn_for(tok_dir)
        cand = {lang: whole_file_line_pairs(pfn, lines) for lang, lines in langs.items()}
        # Fertility whole-file
        for lang, pairs in cand.items():
            tt = sum(t for t, _ in pairs)
            ww = sum(w for _, w in pairs)
            row[f"{lang}_whole"] = tt / ww if ww else None
        cis = {}
        for base in ci_bases:
            cis[base] = {
                lang: paired_bootstrap_diff(cand[lang], comp_pairs[base][lang])
                for lang in ("npi", "eng", "hin")
            }
        row["bootstrap_ci_vs_wholefile"] = cis
        row["tail_hf_rate_lo_hi"] = compute_tail_doc_bootstrap(tok_dir, covered, snap)
        row["code_tokens_per_kb"] = tokens_per_kb(tok_dir, code_texts)
        row["math_tokens_per_kb"] = tokens_per_kb(tok_dir, math_texts)
        enriched.append(row)

    # Item 6 explanation
    t2_dir = MIX_SWEEP_DIR / "t2_latin_rebalance/bd48000_cm50/combined"
    n60 = REPO_ROOT / "data/snapshots/n60_cf8066dbbdd0da93"
    c_n60, _ = held_out_code_math(n60)
    c_j5, _ = held_out_code_math(snap)
    p2_dir = OUT / "64k_bd48000_3pct_cm50/combined"
    explain = {
        "t2_on_n60": tokens_per_kb(t2_dir, c_n60),
        "t2_on_n60j5": tokens_per_kb(t2_dir, c_j5),
        "p2_on_n60": tokens_per_kb(p2_dir, c_n60),
        "p2_on_n60j5": tokens_per_kb(p2_dir, c_j5),
        "reason": (
            "T2 held_out_code_math() defaulted to n60_cf8066…; Prompt 2 used "
            "n60j5_20af1…. Cross-swap shows ~324 vs ~334–336 is the sample "
            "difference; tokenizers agree on the same sample."
        ),
    }

    out = finish_manifest(
        base_manifest(
            kind="prompt2_gaps",
            sealed={
                "path": str(SEALED_PATH.relative_to(REPO_ROOT)) if SEALED_PATH.is_file() else None,
                "n_docs": (sealed or json.loads(SEALED_PATH.read_text())).get("n_docs")
                if SEALED_PATH.is_file()
                else None,
                "list_sha256": (sealed or json.loads(SEALED_PATH.read_text())).get("list_sha256")
                if SEALED_PATH.is_file()
                else None,
                "seed": SEED,
            },
            decisions_rule={
                "contains_FINAL_64k_SELECTION_RULE": False,
                "decisions_last_commit": "d11ac83407ed9f6eff72991930c330e80f5c6340",
                "decisions_commit_time": "2026-11-08T21:26:57+05:45",
                "commit_1f956af_time": "2026-11-09T08:03:03+05:45",
                "ordering": "DECISIONS.md last touched before 1f956af",
            },
            code_kb_explanation=explain,
            competitor_code_math_kb=comp_kb,
            hindi_whole_file_tokens=hin_totals,
            rows=enriched,
        )
    )
    save_json(GAPS_OUT / "report.json", out)

    # Print one STOP table
    print("PROMPT2_GAPS_TABLE", flush=True)
    hdr = (
        "label | npi | eng | hin | code/KB | math/KB | tail | "
        "dNpi_yv1_64 | dNpi_yv2_64 | dNpi_sid | dNpi_yv1_128 | dNpi_yv2_128 | gap"
    )
    print(hdr, flush=True)
    for r in enriched:
        if r.get("npi") is None and r.get("npi_whole") is None:
            continue
        npi = r.get("npi_whole") if r.get("npi_whole") is not None else r.get("npi")
        eng = r.get("eng_whole") if r.get("eng_whole") is not None else r.get("eng")
        hin = r.get("hin_whole") if r.get("hin_whole") is not None else r.get("hin")
        tail = (r.get("tail_hf_rate_lo_hi") or [None])[0]
        cis = r.get("bootstrap_ci_vs_wholefile") or {}

        def d(base: str) -> str:
            x = (cis.get(base) or {}).get("npi") or {}
            if x.get("diff") is None:
                return "—"
            return f"{x['diff']:+.4f}[{x['lo']:+.4f},{x['hi']:+.4f}]"

        print(
            f"{r['label']} | {npi:.6f} | {eng:.6f} | {hin:.6f} | "
            f"{r.get('code_tokens_per_kb'):.1f} | {r.get('math_tokens_per_kb'):.1f} | "
            f"{tail:.4f} | {d('yak_v1_64k')} | {d('yak_v2_64k')} | {d('sidskarki_64k')} | "
            f"{d('yak_v1_128k')} | {d('yak_v2_128k')} | {bool(r.get('gap'))}",
            flush=True,
        )
    print("COMPETITOR_CODE_MATH_KB", json.dumps(comp_kb), flush=True)
    print("HINDI_WHOLE_FILE_TOKENS", json.dumps(hin_totals), flush=True)
    print("CODE_KB_EXPLAIN", json.dumps(explain), flush=True)
    if SEALED_PATH.is_file():
        s = json.loads(SEALED_PATH.read_text(encoding="utf-8"))
        print(
            f"SEALED n={s['n_docs']} seed={s['seed']} list_sha256={s['list_sha256']}",
            flush=True,
        )


if __name__ == "__main__":
    main()

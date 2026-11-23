"""Analysis only: sealed-sample lengths, fertility spread, joiner-unit rates.

Does not train and does not touch the server queue. Writes aggregate JSON
under artifacts/analysis/. Stored text is never normalized and never written out.
"""

from __future__ import annotations

import io
import json
import math
import multiprocessing as mp
import os
import time
from collections import Counter
from pathlib import Path

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import pyarrow.parquet as pq
import sentencepiece as spm
import zstandard as zstd

from ancoda_tok.export_hf import bytes_to_unicode, load_hf
from ancoda_tok.model_eval.arms import ARMS, mac_path
from ancoda_tok.model_eval.common import REPO_ROOT

ZWJ = "\u200d"
ZWNJ = "\u200c"
OUT = REPO_ROOT / "artifacts" / "analysis" / "sealed_fertility_joiners.json"
PROGRESS = REPO_ROOT / "artifacts" / "analysis" / "joiner_shards.json"
SNAP = REPO_ROOT / "data" / "snapshots" / "n60j5_20af1fffbec0fe86"
STREAM = REPO_ROOT / "data" / "model_eval" / "text"
EVAL = REPO_ROOT / "data" / "model_eval" / "eval"
FLORES = (
    REPO_ROOT
    / "third_party"
    / "yak-tokenizer"
    / "data"
    / "eval"
    / "flores_plus"
    / "devtest"
    / "npi_Deva.txt"
)
ARMS_WANTED = (
    "ancoda-64k",
    "yak-64k-v1",
    "yak-64k-v2.0",
    "sidskarki-64k",
    "plain-64k",
)
# Per-sentence FLORES Nepali fertility, 3 d.p., from the final-build self-check.
FLORES_CHECK = {
    "ancoda-64k": 1.451,
    "yak-64k-v1": 1.457,
    "yak-64k-v2.0": 1.470,
    "sidskarki-64k": 1.370,
}
BATCH = 1024


def _log(msg: str) -> None:
    print(msg, flush=True)


def read_jsonl_zst(path: Path):
    dctx = zstd.ZstdDecompressor()
    with path.open("rb") as raw, dctx.stream_reader(raw) as reader:
        text = io.TextIOWrapper(reader, encoding="utf-8")
        for line in text:
            if line.strip():
                yield json.loads(line)


def word_spans(text: str) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    i = 0
    n = len(text)
    while i < n:
        while i < n and text[i].isspace():
            i += 1
        if i >= n:
            break
        j = i + 1
        while j < n and not text[j].isspace():
            j += 1
        spans.append((i, j))
        i = j
    return spans


def assign(spans: list[tuple[int, int]], offsets: list[tuple[int, int]]) -> tuple[list[int], int]:
    """Assign each token to one whitespace word.

    A token goes to the word it overlaps most (ties: the earlier word).
    A token that overlaps no word goes to the following word, or to the
    last word when it is trailing. The counts sum to the token count.
    """
    counts = [0] * len(spans)
    cross = 0
    if not spans:
        return counts, cross
    n = len(spans)
    wi = 0
    for a, b in offsets:
        if b < a:
            a, b = b, a
        while wi < n and spans[wi][1] <= a:
            wi += 1
        if wi >= n:
            counts[-1] += 1
            continue
        if b <= spans[wi][0]:
            counts[wi] += 1
            continue
        best = wi
        best_ov = 0
        overlapped = 0
        k = wi
        while k < n and spans[k][0] < b:
            ws, we = spans[k]
            ov = min(b, we) - max(a, ws)
            if ov > 0:
                overlapped += 1
                if ov > best_ov:
                    best = k
                    best_ov = ov
            k += 1
        if best_ov <= 0:
            counts[wi] += 1
        else:
            if overlapped > 1:
                cross += 1
            counts[best] += 1
    return counts, cross


def _self_test_assign() -> None:
    text = "hello world"
    spans = word_spans(text)
    if spans != [(0, 5), (6, 11)] or len(text.split()) != 2:
        raise SystemExit("STOP: word span self-test failed")
    counts, cross = assign(spans, [(0, 5), (5, 6), (6, 11)])
    if counts != [1, 2] or cross != 0:
        raise SystemExit(f"STOP: space-token assignment {counts} {cross}")
    counts, cross = assign(spans, [(0, 5), (5, 11)])
    if counts != [1, 1] or cross != 0:
        raise SystemExit(f"STOP: attached-space assignment {counts} {cross}")
    counts, cross = assign(spans, [(0, 11)])
    if sum(counts) != 1 or cross != 1:
        raise SystemExit(f"STOP: cross-word assignment {counts} {cross}")


def quantile_hist(hist: Counter[int], q: float) -> float:
    """Hyndman–Fan type 7 on an integer histogram. Position q*(n−1)."""
    n = sum(hist.values())
    if n == 0:
        raise SystemExit("STOP: empty histogram")
    pos = q * (n - 1)
    lo = math.floor(pos)
    hi = math.ceil(pos)
    frac = pos - lo

    def at(index: int) -> int:
        seen = 0
        for value, count in sorted(hist.items()):
            if seen + count > index:
                return value
            seen += count
        return max(hist)

    return at(lo) * (1.0 - frac) + at(hi) * frac


def quantile_list(values: list[int], q: float) -> float:
    arr = np.sort(np.asarray(values, dtype=np.int64))
    n = int(arr.size)
    if n == 0:
        raise SystemExit("STOP: empty sample")
    pos = q * (n - 1)
    lo = math.floor(pos)
    hi = math.ceil(pos)
    frac = pos - lo
    return float(arr[lo] * (1.0 - frac) + arr[hi] * frac)


def summarize(hist: Counter[int], cross_tokens: int) -> dict:
    words = sum(hist.values())
    tokens = sum(k * v for k, v in hist.items())
    ge4 = sum(v for k, v in hist.items() if k >= 4)
    return {
        "words": words,
        "tokens": tokens,
        "mean": tokens / words,
        "median": quantile_hist(hist, 0.50),
        "p90": quantile_hist(hist, 0.90),
        "p95": quantile_hist(hist, 0.95),
        "pct_words_ge4": 100.0 * ge4 / words,
        "words_ge4": ge4,
        "cross_word_tokens": cross_tokens,
        "cross_word_token_share": cross_tokens / tokens if tokens else 0.0,
        "percentile": "hyndman-fan type 7, position q*(n-1), linear interpolation",
        "word": "whitespace-separated (str.split / Unicode whitespace)",
        "assignment": (
            "each token is counted once, on the whitespace word with the greatest "
            "character overlap; ties go to the earlier word; a token that overlaps "
            "no word is counted on the following word, or the last word if trailing"
        ),
    }


class Encoder:
    def __init__(self, name: str) -> None:
        self.name = name
        spec = ARMS[name]
        self.path = mac_path(spec)
        if spec.kind == "hf":
            tok = load_hf(self.path)
            tok.no_truncation()
            tok.no_padding()
            self.kind = "hf"
            self.tok = tok
            self.sp = None
        elif spec.kind == "sp":
            self.kind = "sp"
            self.tok = None
            self.sp = spm.SentencePieceProcessor(model_file=str(self.path))
        else:
            raise SystemExit(f"STOP: unknown arm kind {spec.kind}")

    def encode(self, text: str) -> tuple[list[int], list[tuple[int, int]]]:
        if self.kind == "hf":
            assert self.tok is not None
            enc = self.tok.encode(text, add_special_tokens=False)
            return list(enc.ids), [(int(a), int(b)) for a, b in enc.offsets]
        assert self.sp is not None
        mapping = self.sp.encode_as_offset_mapping(text)
        ids = [int(i) for i in mapping["ids"]]
        offsets = [(int(a), int(b)) for a, b in mapping["offsets"]]
        if len(ids) != len(offsets):
            raise SystemExit(f"STOP: {self.name} offset/id length mismatch")
        return ids, offsets


def _piece_text(tok, tid: int, inverse: dict[str, int]) -> str:
    piece = tok.id_to_token(tid)
    if piece is None:
        return ""
    try:
        raw = bytes(inverse[ch] for ch in piece)
    except KeyError:
        return piece
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return ""


def joiner_vocab(tok) -> dict:
    inverse = {ch: byte for byte, ch in bytes_to_unicode().items()}
    vocab = int(tok.get_vocab_size(with_added_tokens=True))
    zwj_ids: list[int] = []
    zwnj_ids: list[int] = []
    both_ids: list[int] = []
    exact_zwj = 0
    exact_zwnj = 0
    for tid in range(vocab):
        text = _piece_text(tok, tid, inverse)
        has_zwj = ZWJ in text
        has_zwnj = ZWNJ in text
        if has_zwj:
            zwj_ids.append(tid)
        if has_zwnj:
            zwnj_ids.append(tid)
        if has_zwj and has_zwnj:
            both_ids.append(tid)
        if text == ZWJ:
            exact_zwj += 1
        if text == ZWNJ:
            exact_zwnj += 1
    either = sorted(set(zwj_ids) | set(zwnj_ids))
    return {
        "vocab_size": vocab,
        "entries_with_zwj": len(zwj_ids),
        "entries_with_zwnj": len(zwnj_ids),
        "entries_with_both": len(both_ids),
        "entries_with_either": len(either),
        "entries_exact_zwj": exact_zwj,
        "entries_exact_zwnj": exact_zwnj,
        "zwj_ids": zwj_ids,
        "zwnj_ids": zwnj_ids,
        "either_ids": either,
    }


def _flags(vocab: int, ids: list[int]) -> np.ndarray:
    flag = np.zeros(vocab, dtype=np.uint8)
    if ids:
        flag[np.asarray(ids, dtype=np.int32)] = 1
    return flag


def _empty_counts() -> dict:
    return {
        "docs": 0,
        "tokens": 0,
        "either_tokens": 0,
        "zwj_tokens": 0,
        "zwnj_tokens": 0,
        "text_bytes": 0,
        "zwj_chars": 0,
        "zwnj_chars": 0,
        "synthetic_docs": 0,
        "synthetic_tokens": 0,
        "synthetic_either_tokens": 0,
        "natural_docs": 0,
        "natural_tokens": 0,
        "natural_either_tokens": 0,
    }


def _add_counts(total: dict, part: dict) -> None:
    for key, value in part.items():
        total[key] = int(total.get(key, 0)) + int(value)


def _rate(count: int, tokens: int) -> float | None:
    if tokens <= 0:
        return None
    return 1_000_000.0 * count / tokens


def _rates(block: dict) -> dict:
    tokens = int(block["tokens"])
    out = {
        "content_tokens": tokens,
        "either_tokens": int(block["either_tokens"]),
        "zwj_tokens": int(block["zwj_tokens"]),
        "zwnj_tokens": int(block["zwnj_tokens"]),
        "either_per_million_tokens": _rate(int(block["either_tokens"]), tokens),
        "zwj_per_million_tokens": _rate(int(block["zwj_tokens"]), tokens),
        "zwnj_per_million_tokens": _rate(int(block["zwnj_tokens"]), tokens),
        "docs": int(block["docs"]),
        "text_bytes": int(block["text_bytes"]),
        "zwj_chars": int(block["zwj_chars"]),
        "zwnj_chars": int(block["zwnj_chars"]),
    }
    if "synthetic_tokens" in block:
        out["synthetic_docs"] = int(block["synthetic_docs"])
        out["synthetic_tokens"] = int(block["synthetic_tokens"])
        out["synthetic_either_per_million_tokens"] = _rate(
            int(block["synthetic_either_tokens"]), int(block["synthetic_tokens"])
        )
        out["natural_docs"] = int(block["natural_docs"])
        out["natural_tokens"] = int(block["natural_tokens"])
        out["natural_either_per_million_tokens"] = _rate(
            int(block["natural_either_tokens"]), int(block["natural_tokens"])
        )
    return out


def _consume_batch(
    tok, flag_either, flag_zwj, flag_zwnj, texts: list[str], synthetic: list[bool], acc: dict
) -> None:
    if not texts:
        return
    encoded = tok.encode_batch(texts, add_special_tokens=False)
    for text, syn, enc in zip(texts, synthetic, encoded, strict=True):
        ids = np.asarray(enc.ids, dtype=np.int32)
        n = int(ids.size)
        either = int(flag_either[ids].sum()) if n else 0
        zwj_n = int(flag_zwj[ids].sum()) if n else 0
        zwnj_n = int(flag_zwnj[ids].sum()) if n else 0
        acc["docs"] += 1
        acc["tokens"] += n
        acc["either_tokens"] += either
        acc["zwj_tokens"] += zwj_n
        acc["zwnj_tokens"] += zwnj_n
        acc["text_bytes"] += len(text.encode("utf-8"))
        acc["zwj_chars"] += text.count(ZWJ)
        acc["zwnj_chars"] += text.count(ZWNJ)
        if syn:
            acc["synthetic_docs"] += 1
            acc["synthetic_tokens"] += n
            acc["synthetic_either_tokens"] += either
        else:
            acc["natural_docs"] += 1
            acc["natural_tokens"] += n
            acc["natural_either_tokens"] += either


_WORKER: dict = {}


def _init_worker(
    tok_path: str, zwj_ids: list[int], zwnj_ids: list[int], either_ids: list[int]
) -> None:
    tok = load_hf(Path(tok_path))
    tok.no_truncation()
    tok.no_padding()
    vocab = int(tok.get_vocab_size(with_added_tokens=True))
    _WORKER["tok"] = tok
    _WORKER["either"] = _flags(vocab, either_ids)
    _WORKER["zwj"] = _flags(vocab, zwj_ids)
    _WORKER["zwnj"] = _flags(vocab, zwnj_ids)


def _scan_parquet(path_str: str) -> dict:
    acc = _empty_counts()
    table = pq.read_table(Path(path_str), columns=["text", "doc_id", "glotlid_label"])
    texts_col = table.column("text").to_pylist()
    ids_col = table.column("doc_id").to_pylist()
    labels = table.column("glotlid_label").to_pylist()
    batch_t: list[str] = []
    batch_s: list[bool] = []
    for text, doc_id, label in zip(texts_col, ids_col, labels, strict=True):
        raw = text or ""
        flag = str(doc_id).startswith("synthetic\x1f") or label == "synthetic"
        batch_t.append(raw)
        batch_s.append(flag)
        if len(batch_t) >= BATCH:
            _consume_batch(
                _WORKER["tok"],
                _WORKER["either"],
                _WORKER["zwj"],
                _WORKER["zwnj"],
                batch_t,
                batch_s,
                acc,
            )
            batch_t.clear()
            batch_s.clear()
    _consume_batch(
        _WORKER["tok"], _WORKER["either"], _WORKER["zwj"], _WORKER["zwnj"], batch_t, batch_s, acc
    )
    return acc


def _scan_stream(path_str: str) -> dict:
    acc = _empty_counts()
    batch_t: list[str] = []
    batch_s: list[bool] = []
    for row in read_jsonl_zst(Path(path_str)):
        raw = row["text"] or ""
        flag = str(row["doc_id"]).startswith("synthetic\x1f")
        batch_t.append(raw)
        batch_s.append(flag)
        if len(batch_t) >= BATCH:
            _consume_batch(
                _WORKER["tok"],
                _WORKER["either"],
                _WORKER["zwj"],
                _WORKER["zwnj"],
                batch_t,
                batch_s,
                acc,
            )
            batch_t.clear()
            batch_s.clear()
    _consume_batch(
        _WORKER["tok"], _WORKER["either"], _WORKER["zwj"], _WORKER["zwnj"], batch_t, batch_s, acc
    )
    return acc


def _scan_one(job: tuple[str, str]) -> tuple[str, dict]:
    kind, path_str = job
    if kind == "parquet":
        part = _scan_parquet(path_str)
    else:
        part = _scan_stream(path_str)
    return Path(path_str).name, part


def _write_progress(done: dict) -> None:
    tmp = PROGRESS.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(done), encoding="utf-8")
    tmp.replace(PROGRESS)


def _run_shards(kind: str, paths: list[Path], tok_path: str, vocab_info: dict, label: str) -> dict:
    done: dict = {}
    if PROGRESS.is_file():
        done = json.loads(PROGRESS.read_text(encoding="utf-8"))
    acc = _empty_counts()
    prefix = f"{label}:"
    pending = [p for p in paths if f"{prefix}{p.name}" not in done]
    for key, part in done.items():
        if key.startswith(prefix):
            _add_counts(acc, part)
    _log(f"{label} shards {len(paths)} pending {len(pending)}")
    if not pending:
        return acc
    ctx = mp.get_context("spawn")
    n_workers = min(4, max(1, os.cpu_count() or 1), len(pending))
    t0 = time.perf_counter()
    finished = 0
    with ctx.Pool(
        processes=n_workers,
        initializer=_init_worker,
        initargs=(
            tok_path,
            vocab_info["zwj_ids"],
            vocab_info["zwnj_ids"],
            vocab_info["either_ids"],
        ),
    ) as pool:
        jobs = [(kind, str(p)) for p in pending]
        for name, part in pool.imap_unordered(_scan_one, jobs, chunksize=1):
            done[f"{prefix}{name}"] = part
            _add_counts(acc, part)
            finished += 1
            _write_progress(done)
            elapsed = time.perf_counter() - t0
            _log(
                f"{label} {finished}/{len(pending)} {name} tokens={part['tokens']} "
                f"either={part['either_tokens']} elapsed_s={elapsed:.1f}"
            )
    return acc


def fertility_on(encoders: list[Encoder], texts: list[str]) -> dict:
    hists = {enc.name: Counter() for enc in encoders}
    cross = {enc.name: 0 for enc in encoders}
    offset_oob = {enc.name: 0 for enc in encoders}
    for text in texts:
        spans = word_spans(text)
        if len(spans) != len(text.split()):
            raise SystemExit("STOP: word span count != str.split")
        if not spans:
            continue
        for enc in encoders:
            ids, offsets = enc.encode(text)
            if len(ids) != len(offsets):
                raise SystemExit(f"STOP: {enc.name} id/offset mismatch")
            if any(a < 0 or b > len(text) for a, b in offsets):
                offset_oob[enc.name] += 1
            counts, n_cross = assign(spans, offsets)
            if sum(counts) != len(ids):
                raise SystemExit(f"STOP: {enc.name} assignment dropped tokens")
            hist = hists[enc.name]
            for count in counts:
                hist[count] += 1
            cross[enc.name] += n_cross
    out = {}
    for enc in encoders:
        summary = summarize(hists[enc.name], cross[enc.name])
        summary["docs_with_offset_oob"] = offset_oob[enc.name]
        out[enc.name] = summary
    return out


def length_report(rows: list[dict], sample_ids: list[str]) -> dict:
    nepali = [row for row in rows if row["lang"] == "Nepali"]
    nepali_sorted = sorted(nepali, key=lambda row: row["doc_id"])
    prefix = [row["doc_id"] for row in nepali_sorted[:2000]]
    sample_sorted = sorted(sample_ids)
    prefix_sorted = sorted(prefix)
    lengths_all = [int(row["original_bytes"]) for row in nepali]
    by_id = {row["doc_id"]: int(row["original_bytes"]) for row in nepali}
    if any(doc_id not in by_id for doc_id in sample_ids):
        raise SystemExit("STOP: sealed_npi_2000 doc_id missing from sealed Nepali")
    lengths_sample = [by_id[doc_id] for doc_id in sample_ids]
    return {
        "all_sealed_nepali": {
            "n": len(lengths_all),
            "median_bytes": quantile_list(lengths_all, 0.50),
            "p90_bytes": quantile_list(lengths_all, 0.90),
            "min_bytes": int(min(lengths_all)),
            "max_bytes": int(max(lengths_all)),
        },
        "sealed_npi_2000": {
            "n": len(lengths_sample),
            "median_bytes": quantile_list(lengths_sample, 0.50),
            "p90_bytes": quantile_list(lengths_sample, 0.90),
            "min_bytes": int(min(lengths_sample)),
            "max_bytes": int(max(lengths_sample)),
        },
        "sample_is_first_2000_nepali_by_doc_id": prefix_sorted == sample_sorted
        and len(sample_ids) == 2000,
        "percentile": "hyndman-fan type 7, position q*(n-1), linear interpolation",
    }


def eval_joiner_rates(tok, vocab_info: dict) -> dict:
    vocab = vocab_info["vocab_size"]
    flag_either = _flags(vocab, vocab_info["either_ids"])
    flag_zwj = _flags(vocab, vocab_info["zwj_ids"])
    flag_zwnj = _flags(vocab, vocab_info["zwnj_ids"])
    files = {
        "flores_npi_zwj": EVAL / "perturb" / "flores_npi_zwj.jsonl.zst",
        "flores_npi_zwnj": EVAL / "perturb" / "flores_npi_zwnj.jsonl.zst",
        "sealed_npi_zwj": EVAL / "perturb" / "sealed_npi_zwj.jsonl.zst",
        "sealed_npi_zwnj": EVAL / "perturb" / "sealed_npi_zwnj.jsonl.zst",
    }
    out = {}
    for name, path in files.items():
        acc = _empty_counts()
        texts = [row["text"] or "" for row in read_jsonl_zst(path)]
        syn = [False] * len(texts)
        for start in range(0, len(texts), BATCH):
            _consume_batch(
                tok,
                flag_either,
                flag_zwj,
                flag_zwnj,
                texts[start : start + BATCH],
                syn[start : start + BATCH],
                acc,
            )
        out[name] = _rates(acc)
    return out


def main() -> None:
    _self_test_assign()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    _log("loading sealed rows")
    sealed_rows = list(read_jsonl_zst(EVAL / "sealed.jsonl.zst"))
    sample_rows = list(read_jsonl_zst(EVAL / "sealed_npi_2000.jsonl.zst"))
    lengths = length_report(sealed_rows, [row["doc_id"] for row in sample_rows])
    _log(
        "lengths "
        f"all_n={lengths['all_sealed_nepali']['n']} "
        f"median={lengths['all_sealed_nepali']['median_bytes']} "
        f"sample_median={lengths['sealed_npi_2000']['median_bytes']} "
        f"prefix_ok={lengths['sample_is_first_2000_nepali_by_doc_id']}"
    )

    _log("loading encoders")
    encoders = [Encoder(name) for name in ARMS_WANTED]
    ancoda = next(enc for enc in encoders if enc.name == "ancoda-64k")
    assert ancoda.tok is not None
    vocab_info = joiner_vocab(ancoda.tok)
    _log(
        "joiner vocab "
        f"either={vocab_info['entries_with_either']} "
        f"zwj={vocab_info['entries_with_zwj']} "
        f"zwnj={vocab_info['entries_with_zwnj']} "
        f"both={vocab_info['entries_with_both']}"
    )

    flores = [line for line in FLORES.read_text(encoding="utf-8").splitlines()]
    _log(f"fertility flores lines={len(flores)}")
    flores_stats = fertility_on(encoders, flores)
    for name, want in FLORES_CHECK.items():
        got = flores_stats[name]["mean"]
        if abs(got - want) >= 0.001:
            raise SystemExit(f"STOP: {name} FLORES fertility {got} != {want} within 0.001")
        _log(f"  flores {name} mean={got:.6f} check={want}")

    nepali_text = [row["text"] for row in sealed_rows if row["lang"] == "Nepali"]
    _log(f"fertility sealed nepali docs={len(nepali_text)}")
    sealed_stats = fertility_on(encoders, nepali_text)
    for name in ARMS_WANTED:
        row = flores_stats[name]
        _log(
            f"  flores {name} median={row['median']} p90={row['p90']} "
            f"p95={row['p95']} ge4={row['pct_words_ge4']:.4f}"
        )
    for name in ARMS_WANTED:
        row = sealed_stats[name]
        _log(
            f"  sealed {name} mean={row['mean']:.6f} median={row['median']} "
            f"p90={row['p90']} p95={row['p95']} ge4={row['pct_words_ge4']:.4f}"
        )

    _log("eval-set joiner rates")
    eval_rates = eval_joiner_rates(ancoda.tok, vocab_info)
    vocab_public = {k: v for k, v in vocab_info.items() if not k.endswith("_ids")}
    partial = {
        "kind": "analysis_sealed_fertility_joiners",
        "lengths": lengths,
        "fertility": {
            "flores_npi_devtest": flores_stats,
            "sealed_nepali": sealed_stats,
            "flores_lines": len(flores),
            "sealed_nepali_docs": len(nepali_text),
        },
        "joiner_vocab": vocab_public,
        "joiner_rates": {"eval_sets": eval_rates},
    }
    OUT.write_text(json.dumps(partial, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _log(f"wrote partial {OUT}")

    tok_path = str(ancoda.path)
    snap_paths = sorted(SNAP.glob("shard-*.parquet"))
    stream_paths = sorted(STREAM.glob("shard-*.jsonl.zst"))
    if len(snap_paths) != 75:
        raise SystemExit(f"STOP: expected 75 snapshot shards, found {len(snap_paths)}")
    _log("tokenizer training text")
    snap_counts = _run_shards("parquet", snap_paths, tok_path, vocab_info, "n60j5")
    _log("model training stream")
    stream_counts = _run_shards("stream", stream_paths, tok_path, vocab_info, "m9text")

    payload = {
        "kind": "analysis_sealed_fertility_joiners",
        "lengths": lengths,
        "fertility": {
            "flores_npi_devtest": flores_stats,
            "sealed_nepali": sealed_stats,
            "flores_lines": len(flores),
            "sealed_nepali_docs": len(nepali_text),
        },
        "joiner_vocab": vocab_public,
        "joiner_rates": {
            "tokenizer_training_text_n60j5": _rates(snap_counts),
            "model_training_stream": _rates(stream_counts),
            "eval_sets": eval_rates,
            "denominator": (
                "Ancoda-64k content tokens, add_special_tokens=False, "
                "encode_special_tokens=True. End-of-text ids are not in the denominator."
            ),
        },
    }
    OUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _log(f"wrote {OUT}")


if __name__ == "__main__":
    main()

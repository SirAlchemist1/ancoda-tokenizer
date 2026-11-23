"""Recompute seed-1 byte coverage for arms A and C. Analysis only.

Writes per-shard count arrays, then the prefix totals at completed steps
5400 and 7630. Does not train and does not read the server.
"""

from __future__ import annotations

import json
import multiprocessing as mp
from pathlib import Path

import numpy as np

from ancoda_tok.model_eval.arms import load_arm
from ancoda_tok.model_eval.perm import byte_prefix, bytes_at, fisher_yates, stream_prefix
from ancoda_tok.model_eval.tokenize import iter_jsonl_zst

REPO = Path(__file__).resolve().parents[1]
TEXT = REPO / "data" / "model_eval" / "text"
OUT = REPO / "artifacts" / "analysis" / "byte_counter"
TOKENS_PER_STEP = 131072
STEPS = (5400, 7630)
# Logged row whose "step" field is 5400 is completed_steps 5401.
ALSO = (5397, 5401)


def _count_shard(task: dict) -> str:
    shard = Path(task["shard"])
    dest = Path(task["out"])
    if dest.is_file():
        return shard.name
    anc = load_arm("ancoda-64k")
    sid = load_arm("sidskarki-64k")
    texts: list[str] = []
    byts: list[int] = []
    for row in iter_jsonl_zst(shard):
        text = row["text"]
        texts.append(text)
        byts.append(len(text.encode("utf-8")))
    anc_n = [len(ids) for ids in _encode_all(anc, texts)]
    sid_n = [len(ids) for ids in _encode_all(sid, texts)]
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".tmp.npz")
    np.savez(
        tmp,
        anc=np.asarray(anc_n, dtype=np.int32),
        sid=np.asarray(sid_n, dtype=np.int32),
        byt=np.asarray(byts, dtype=np.int64),
    )
    tmp.replace(dest)
    return shard.name


def _encode_all(arm, texts: list[str]) -> list[list[int]]:
    if arm._hf is not None:
        out: list[list[int]] = []
        for start in range(0, len(texts), 1024):
            chunk = texts[start : start + 1024]
            encoded = arm._hf.encode_batch(chunk, add_special_tokens=False)
            out.extend(list(item.ids) for item in encoded)
        return out
    assert arm._sp is not None
    return [list(arm._sp.encode(text, out_type=int, add_bos=False, add_eos=False)) for text in texts]


def _load() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    shards = sorted(TEXT.glob("shard-*.jsonl.zst"))
    parts_a: list[np.ndarray] = []
    parts_c: list[np.ndarray] = []
    parts_b: list[np.ndarray] = []
    for shard in shards:
        data = np.load(OUT / "parts" / f"{shard.name}.npz")
        parts_a.append(data["anc"])
        parts_c.append(data["sid"])
        parts_b.append(data["byt"])
    return np.concatenate(parts_a), np.concatenate(parts_c), np.concatenate(parts_b)


def _walk(order: np.ndarray, tok: np.ndarray, byt: np.ndarray, pos: int) -> dict[str, int]:
    """Independent of bytes_at. Same charging rule, plus content and EOS counts."""
    if pos < 0:
        raise SystemExit(f"STOP: negative position {pos}")
    remaining = pos
    covered = 0
    content = 0
    eos = 0
    full_docs = 0
    partial = 0
    for doc in order:
        tc = int(tok[doc])
        bc = int(byt[doc])
        span = tc + 1
        if remaining >= span:
            covered += bc
            content += tc
            eos += 1
            full_docs += 1
            remaining -= span
            if remaining == 0:
                break
            continue
        take_content = min(tc, remaining)
        if tc == 0:
            # EOS not yet consumed; empty docs add bytes only when EOS is consumed.
            covered += 0
        elif take_content >= tc:
            covered += bc
        else:
            covered += (bc * take_content) // tc
            partial = 1
        content += take_content
        if remaining > tc:
            eos += 1
        remaining = 0
        break
    if remaining != 0:
        raise SystemExit(f"STOP: stream ended with {remaining} tokens still unread")
    return {
        "stream_tokens": pos,
        "content_tokens": content,
        "eos_tokens": eos,
        "bytes_walk": covered,
        "full_docs": full_docs,
        "partial_doc": partial,
    }


def _report(name: str, tok: np.ndarray, byt: np.ndarray, seed: int) -> dict:
    order = fisher_yates(int(tok.shape[0]), seed)
    cum = stream_prefix(order, tok)
    bpre = byte_prefix(order, byt)
    rows = []
    for step in (*STEPS, *ALSO):
        pos = step * TOKENS_PER_STEP
        counter = bytes_at(pos, order, tok, byt, cum, bpre)
        walked = _walk(order, tok, byt, pos)
        if walked["bytes_walk"] != counter:
            raise SystemExit(f"STOP: walk {walked['bytes_walk']} != counter {counter} at {name} step {step}")
        rows.append({"completed_steps": step, "counter": counter, **walked})
    return {
        "arm": name,
        "seed": seed,
        "docs": int(tok.shape[0]),
        "content_tokens": int(tok.sum()),
        "text_bytes": int(byt.sum()),
        "stream_tokens": int(tok.sum()) + int(tok.shape[0]),
        "rows": rows,
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    part_dir = OUT / "parts"
    part_dir.mkdir(parents=True, exist_ok=True)
    shards = sorted(TEXT.glob("shard-*.jsonl.zst"))
    tasks = [
        {"shard": str(shard), "out": str(part_dir / f"{shard.name}.npz")}
        for shard in shards
    ]
    pending = [task for task in tasks if not Path(task["out"]).is_file()]
    print(f"shards={len(shards)} pending={len(pending)}", flush=True)
    if pending:
        ctx = mp.get_context("spawn")
        with ctx.Pool(4) as pool:
            for done, name in enumerate(pool.imap_unordered(_count_shard, pending), start=1):
                print(f"  {done}/{len(pending)} {name}", flush=True)
    anc, sid, byt = _load()
    if int(anc.shape[0]) != 2_477_651 or int(byt.sum()) != 7_189_336_358:
        raise SystemExit(
            f"STOP: docs {anc.shape[0]} bytes {int(byt.sum())} "
            f"anc_tokens {int(anc.sum())} sid_tokens {int(sid.sum())}"
        )
    if int(anc.sum()) != 1_085_350_393 or int(sid.sum()) != 1_551_158_477:
        raise SystemExit(f"STOP: token totals anc {int(anc.sum())} sid {int(sid.sum())}")
    report = {
        "A": _report("ancoda-64k", anc, byt, 1),
        "C": _report("sidskarki-64k", sid, byt, 1),
    }
    path = OUT / "seed1.json"
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    print(f"wrote {path}", flush=True)


if __name__ == "__main__":
    main()

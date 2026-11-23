#!/usr/bin/env python3
"""Reference BPB on the fixed 300-doc sample using the project scoring path."""

from __future__ import annotations

import argparse
import io
import json
import math
from pathlib import Path

import torch
import zstandard as zstd

from ancoda_tok.model_eval.arms import load_arm
from ancoda_tok.model_eval.bpb import score_ids
from ancoda_tok.model_eval.common import bpb
from ancoda_tok.model_eval.model import Llama


def load_sample(sealed: Path, n: int) -> list[dict]:
    dctx = zstd.ZstdDecompressor()
    nepali = []
    with sealed.open("rb") as raw, dctx.stream_reader(raw) as reader:
        for line in io.TextIOWrapper(reader, encoding="utf-8"):
            if not line.strip():
                continue
            row = json.loads(line)
            if row["lang"] != "Nepali":
                continue
            nepali.append(row)
            if len(nepali) >= n:
                break
    if len(nepali) != n:
        raise SystemExit(f"need {n} docs, got {len(nepali)}")
    return nepali


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--tokenizer", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--sealed", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n-docs", type=int, default=300)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    device = torch.device(args.device)
    arm = load_arm(args.arm, args.tokenizer)
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = Llama(int(payload["vocab"]))
    model.load_state_dict(payload["model"])
    model.to(device)
    model.eval()
    rows = load_sample(args.sealed, args.n_docs)
    nll = 0.0
    nbytes = 0
    with torch.inference_mode():
        for i, row in enumerate(rows):
            nll += score_ids(model, arm.encode(row["text"]), arm.eos_id, device)
            nbytes += int(row["original_bytes"])
            if (i + 1) % 25 == 0:
                print(f"ref {args.arm} {i + 1}/{args.n_docs}", flush=True)
    value = bpb(nll, nbytes)
    out = {
        "arm": args.arm,
        "bpb": value,
        "bytes": nbytes,
        "n_docs": args.n_docs,
        "nll_nats": nll,
        "ln2": math.log(2),
        "source": "project ancoda_tok.model_eval.bpb.score_ids",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"arm": args.arm, "bpb": value}))


if __name__ == "__main__":
    main()

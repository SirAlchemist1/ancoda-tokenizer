#!/usr/bin/env python3
"""Standalone BPB recompute for M9 verification.

Uses only torch + tokenizers/sentencepiece. Does not import ancoda_tok.m9
eval code. Must match the project eval pipeline within 0.001 BPB on the
fixed 300-doc sealed Nepali sample (first 300 Nepali rows by doc_id order).

Scoring policy (locked in docs/M9_PREREG.md):
- denominator = UTF-8 bytes of the original text (row['original_bytes'])
- encode with add_special_tokens=False / no BOS/EOS from the tokenizer
- score windows of at most 1023 text tokens; each window is [EOS] + span
- EOS is context only and is never a scored target
"""

from __future__ import annotations

import argparse
import io
import json
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import zstandard as zstd
from torch import nn

DIM = 512
LAYERS = 8
HEADS = 8
HEAD_DIM = 64
MLP = 1408
ROPE_THETA = 10000.0
RMS_EPS = 1e-5
CONTEXT = 1024
EOS_NAMES = ("<|eos|>", "<|endoftext|>", "</s>")


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    d = x.shape[-1] // 2
    return torch.cat((-x[..., d:], x[..., :d]), dim=-1)


class RMSNorm(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig = x.dtype
        x32 = x.float()
        var = x32.pow(2).mean(dim=-1, keepdim=True)
        x32 = x32 * torch.rsqrt(var + RMS_EPS)
        return (self.weight * x32).to(orig)


class Block(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.norm1 = RMSNorm(DIM)
        self.q = nn.Linear(DIM, DIM, bias=False)
        self.k = nn.Linear(DIM, DIM, bias=False)
        self.v = nn.Linear(DIM, DIM, bias=False)
        self.o = nn.Linear(DIM, DIM, bias=False)
        self.norm2 = RMSNorm(DIM)
        self.w1 = nn.Linear(DIM, MLP, bias=False)
        self.w3 = nn.Linear(DIM, MLP, bias=False)
        self.w2 = nn.Linear(MLP, DIM, bias=False)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        b, t, _c = h.shape
        q = self.q(h).view(b, t, HEADS, HEAD_DIM).transpose(1, 2)
        k = self.k(h).view(b, t, HEADS, HEAD_DIM).transpose(1, 2)
        v = self.v(h).view(b, t, HEADS, HEAD_DIM).transpose(1, 2)
        q = q * cos + _rotate_half(q) * sin
        k = k * cos + _rotate_half(k) * sin
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(b, t, DIM)
        x = x + self.o(y)
        h = self.norm2(x)
        x = x + self.w2(F.silu(self.w1(h)) * self.w3(h))
        return x


class Llama(nn.Module):
    def __init__(self, vocab: int) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab, DIM)
        self.blocks = nn.ModuleList(Block() for _ in range(LAYERS))
        self.norm = RMSNorm(DIM)
        inv_freq = 1.0 / (
            ROPE_THETA ** (torch.arange(0, HEAD_DIM, 2, dtype=torch.float32) / HEAD_DIM)
        )
        pos = torch.arange(CONTEXT, dtype=torch.float32)
        freqs = torch.outer(pos, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos", emb.cos()[None, None, :, :], persistent=False)
        self.register_buffer("sin", emb.sin()[None, None, :, :], persistent=False)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        t = tokens.shape[1]
        h = F.embedding(tokens, self.embed.weight)
        cos = self.cos[:, :, :t, :]
        sin = self.sin[:, :, :t, :]
        for block in self.blocks:
            h = block(h, cos, sin)
        h = self.norm(h)
        return F.linear(h.float(), self.embed.weight)


def load_tokenizer(kind: str, path: Path):
    if kind == "hf":
        from tokenizers import Tokenizer

        tok = Tokenizer.from_file(str(path))
        tok.no_truncation()
        tok.no_padding()
        added = tok.get_added_tokens_decoder()
        specials = {info.content: int(tid) for tid, info in added.items() if info.special}
        vocab = int(tok.get_vocab_size(with_added_tokens=True))

        def encode(text: str) -> list[int]:
            return list(tok.encode(text, add_special_tokens=False).ids)

    elif kind == "sp":
        import sentencepiece as spm

        sp = spm.SentencePieceProcessor(model_file=str(path))
        vocab = int(sp.vocab_size())
        specials = {}
        if int(sp.eos_id()) >= 0:
            specials["</s>"] = int(sp.eos_id())

        def encode(text: str) -> list[int]:
            return list(sp.encode(text, out_type=int, add_bos=False, add_eos=False))

    else:
        raise ValueError(kind)

    eos = None
    for name in EOS_NAMES:
        if name in specials:
            eos = specials[name]
            model_vocab = vocab
            break
    if eos is None:
        eos = vocab
        model_vocab = vocab + 1
    return encode, eos, model_vocab


def text_windows(n_tokens: int) -> list[tuple[int, int]]:
    span = CONTEXT - 1
    out = []
    i = 0
    while i < n_tokens:
        j = min(n_tokens, i + span)
        out.append((i, j))
        i = j
    return out


def score_ids(model: Llama, ids: list[int], eos: int, device: torch.device) -> float:
    total = 0.0
    for a, b in text_windows(len(ids)):
        span = ids[a:b]
        if not span:
            continue
        batch = torch.zeros(1, len(span) + 1, dtype=torch.long, device=device)
        batch[0, 0] = eos
        batch[0, 1:] = torch.tensor(span, dtype=torch.long, device=device)
        with torch.inference_mode():
            logits = model(batch)
            total += float(
                F.cross_entropy(logits[0, : len(span)].float(), batch[0, 1:], reduction="sum").item()
            )
    return total


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
            if int(row["original_bytes"]) != len(row["text"].encode("utf-8")):
                raise SystemExit(f"original_bytes mismatch for {row['doc_id']}")
            nepali.append(row)
            if len(nepali) >= n:
                break
    if len(nepali) != n:
        raise SystemExit(f"need {n} Nepali docs, got {len(nepali)}")
    return nepali


def bpb(nll: float, nbytes: int) -> float:
    return nll / (math.log(2) * nbytes)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sealed", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--tokenizer", type=Path, required=True)
    ap.add_argument("--kind", choices=("hf", "sp"), required=True)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--n-docs", type=int, default=300)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    device = torch.device(args.device)
    encode, eos, model_vocab = load_tokenizer(args.kind, args.tokenizer)
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if int(payload["vocab"]) != model_vocab:
        raise SystemExit(f"vocab mismatch payload={payload['vocab']} tokenizer={model_vocab}")
    model = Llama(model_vocab)
    model.load_state_dict(payload["model"])
    model.to(device)
    model.eval()

    rows = load_sample(args.sealed, args.n_docs)
    nll = 0.0
    nbytes = 0
    for i, row in enumerate(rows):
        nll += score_ids(model, encode(row["text"]), eos, device)
        nbytes += int(row["original_bytes"])
        if (i + 1) % 25 == 0:
            print(f"{args.arm} scored {i + 1}/{args.n_docs}", flush=True)

    value = bpb(nll, nbytes)
    out = {
        "arm": args.arm,
        "checkpoint": str(args.checkpoint),
        "n_docs": args.n_docs,
        "bytes": nbytes,
        "nll_nats": nll,
        "bpb": value,
        "eos_id": eos,
        "doc_ids": [r["doc_id"] for r in rows],
        "first_doc_id": rows[0]["doc_id"],
        "last_doc_id": rows[-1]["doc_id"],
        "note": "standalone recompute; EOS context-only; original_bytes denominator",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"arm": args.arm, "bpb": value, "bytes": nbytes, "docs": args.n_docs}))
    return 0


if __name__ == "__main__":
    sys.exit(main())

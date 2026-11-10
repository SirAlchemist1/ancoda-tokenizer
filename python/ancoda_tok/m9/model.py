"""Llama-style decoder. Hidden 512, 8 layers, 8 heads, SwiGLU 1408, tied embeddings."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from ancoda_tok.m9.common import CONTEXT

DIM = 512
LAYERS = 8
HEADS = 8
HEAD_DIM = DIM // HEADS
MLP = 1408
ROPE_THETA = 10000.0
RMS_EPS = 1e-5


def parameter_count(vocab: int) -> int:
    embed = vocab * DIM
    attn = 4 * DIM * DIM
    mlp = 3 * DIM * MLP
    norms = 2 * DIM
    return embed + LAYERS * (attn + mlp + norms) + DIM


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
        self.vocab = vocab
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
        self.apply(self._init)

    @staticmethod
    def _init(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        t = tokens.shape[1]
        h = self.embed(tokens)
        cos = self.cos[:, :, :t, :]
        sin = self.sin[:, :, :t, :]
        for block in self.blocks:
            h = block(h, cos, sin)
        h = self.norm(h)
        return F.linear(h, self.embed.weight)


def build_model(vocab: int, seed: int) -> Llama:
    torch.manual_seed(seed)
    return Llama(vocab)


def trainable_groups(model: nn.Module) -> list[dict]:
    decay: list[torch.nn.Parameter] = []
    nodecay: list[torch.nn.Parameter] = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.endswith(("norm.weight", "embed.weight")):
            nodecay.append(param)
        else:
            decay.append(param)
    return [
        {"params": decay, "weight_decay": 0.1},
        {"params": nodecay, "weight_decay": 0.0},
    ]


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def assert_formula(model: nn.Module) -> None:
    vocab = model.embed.num_embeddings
    got = count_parameters(model)
    expect = parameter_count(vocab)
    if got != expect:
        raise SystemExit(f"STOP: parameter count {got} != formula {expect}")
    # RoPE buffers are not parameters. math.sqrt keeps head dim obvious for readers.
    if HEAD_DIM != DIM // HEADS or not math.isclose(1 / math.sqrt(HEAD_DIM), 0.125):
        raise SystemExit("STOP: head dimension is not 64")

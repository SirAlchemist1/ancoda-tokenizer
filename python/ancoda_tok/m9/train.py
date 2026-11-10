"""XPU training loop. Resume restores model, optimizer, and the data position."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from ancoda_tok.m9.common import FULL_STEPS, SEQUENCES_PER_STEP, TOKENS_PER_STEP
from ancoda_tok.m9.model import assert_formula, build_model, trainable_groups
from ancoda_tok.m9.perm import first_step_reaching
from ancoda_tok.m9.reader import TokenStream
from ancoda_tok.m9.schedule import learning_rate


def rss_bytes() -> int:
    try:
        with open("/proc/self/status", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        return -1
    return -1


def gpu_bytes() -> int:
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return int(torch.xpu.memory_allocated())
    return -1


def _atomic_torch(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def save_checkpoint(path: Path, payload: dict) -> None:
    _atomic_torch(path, payload)
    _atomic_torch(path.parent / "latest.pt", payload)


def retain(run_dir: Path, keep_names: set[str]) -> None:
    for path in run_dir.glob("checkpoint-*.pt"):
        if path.name not in keep_names:
            path.unlink()


def load_checkpoint(path: Path, model: torch.nn.Module, optim: torch.optim.Optimizer) -> int:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(payload["model"])
    optim.load_state_dict(payload["optim"])
    return int(payload["completed_steps"])


def raw_module(model: torch.nn.Module) -> torch.nn.Module:
    return getattr(model, "_orig_mod", model)


def _payload(args, model, optim, completed: int, stream: TokenStream) -> dict:
    return {
        "completed_steps": completed,
        "model": raw_module(model).state_dict(),
        "optim": optim.state_dict(),
        "arm": args.arm,
        "seed": args.seed,
        "vocab": args.vocab,
        "tokens_seen": completed * TOKENS_PER_STEP,
        "bytes_seen": stream.bytes_covered(completed * TOKENS_PER_STEP),
    }


def train(args) -> None:
    device = torch.device("xpu" if torch.xpu.is_available() else "cpu")
    if device.type != "xpu":
        raise SystemExit("STOP: torch.xpu is not available")
    stream = TokenStream(Path(args.tokens), args.seed)
    model = build_model(args.vocab, args.seed).to(device)
    assert_formula(model)
    optim = torch.optim.AdamW(trainable_groups(model), lr=1e-3, betas=(0.9, 0.95))
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    completed = 0
    latest = run_dir / "latest.pt"
    if args.resume and latest.is_file():
        completed = load_checkpoint(latest, model, optim)
        print(f"resumed at completed_steps={completed}", flush=True)
    if args.compile:
        model = torch.compile(model)
    bstar = None
    bstar_step = None
    if args.bstar is not None:
        bstar = int(args.bstar)
        bstar_step = first_step_reaching(stream.bytes_covered, bstar, TOKENS_PER_STEP, args.steps)
        print(f"B*={bstar} first_step={bstar_step}", flush=True)
    log_path = run_dir / "log.jsonl"
    micro = int(args.micro_batch)
    if SEQUENCES_PER_STEP % micro != 0:
        raise SystemExit(f"STOP: micro_batch {micro} does not divide {SEQUENCES_PER_STEP}")
    accum = SEQUENCES_PER_STEP // micro
    latest_name: str | None = None
    bstar_name: str | None = None
    final_name: str | None = None
    target = int(args.steps)
    model.train()
    while completed < target:
        step = completed
        lr = learning_rate(step, total_steps=FULL_STEPS)
        for group in optim.param_groups:
            group["lr"] = lr
        pos = step * TOKENS_PER_STEP
        window = stream.read(pos, TOKENS_PER_STEP).reshape(SEQUENCES_PER_STEP, 1024)
        optim.zero_grad(set_to_none=True)
        t0 = time.perf_counter()
        loss_sum = 0.0
        for i in range(accum):
            sl = window[i * micro : (i + 1) * micro]
            batch = torch.from_numpy(np.ascontiguousarray(sl)).to(device)
            with torch.autocast(device_type="xpu", dtype=torch.bfloat16):
                logits = model(batch)
            loss = F.cross_entropy(
                logits[:, :-1, :].float().reshape(-1, logits.shape[-1]),
                batch[:, 1:].reshape(-1),
            )
            (loss / accum).backward()
            loss_sum += float(loss.detach().item())
            del logits, loss, batch
        torch.nn.utils.clip_grad_norm_(raw_module(model).parameters(), 1.0)
        optim.step()
        completed += 1
        step_index = completed - 1
        if step_index % 50 == 0 or completed == target:
            record = {
                "step": step_index,
                "completed_steps": completed,
                "loss": loss_sum / accum,
                "tokens_seen": completed * TOKENS_PER_STEP,
                "bytes_seen": stream.bytes_covered(completed * TOKENS_PER_STEP),
                "step_seconds": time.perf_counter() - t0,
                "gpu_bytes": gpu_bytes(),
                "rss_bytes": rss_bytes(),
                "lr": lr,
            }
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record) + "\n")
            print(json.dumps(record), flush=True)
        names: set[str] = set()
        if completed % int(args.checkpoint_every) == 0 or completed == target:
            name = f"checkpoint-{completed:06d}.pt"
            payload = _payload(args, model, optim, completed, stream)
            save_checkpoint(run_dir / name, payload)
            latest_name = name
            names.add(name)
            print(f"checkpoint {name}", flush=True)
        if bstar_step is not None and completed == bstar_step:
            name = f"checkpoint-bstar-{completed:06d}.pt"
            payload = _payload(args, model, optim, completed, stream)
            payload["bstar"] = bstar
            _atomic_torch(run_dir / name, payload)
            bstar_name = name
            names.add(name)
        if completed == target:
            final_name = latest_name
        if names:
            keep = {n for n in (latest_name, bstar_name, final_name) if n}
            retain(run_dir, keep)

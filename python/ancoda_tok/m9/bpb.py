"""Score held-out text. Each document starts from EOS; windows do not share context."""

from __future__ import annotations

import io
import json
import math
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
import zstandard as zstd

from ancoda_tok.m9.arms import load_arm
from ancoda_tok.m9.common import attach_build, bpb, now_utc, text_windows
from ancoda_tok.m9.model import Llama


def iter_jsonl_zst(path: Path):
    dctx = zstd.ZstdDecompressor()
    with path.open("rb") as raw, dctx.stream_reader(raw) as reader:
        text = io.TextIOWrapper(reader, encoding="utf-8")
        for line in text:
            if line.strip():
                yield json.loads(line)


def _score_spans(model: Llama, spans: list[list[int]], eos: int, device: torch.device) -> float:
    width = max(len(span) for span in spans) + 1
    batch = torch.zeros(len(spans), width, dtype=torch.long, device=device)
    batch[:, 0] = eos
    for i, span in enumerate(spans):
        if span:
            batch[i, 1 : 1 + len(span)] = torch.tensor(span, dtype=torch.long, device=device)
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
        logits = model(batch)
    total = 0.0
    for i, span in enumerate(spans):
        if not span:
            continue
        n = len(span)
        total += float(
            F.cross_entropy(
                logits[i, :n].float(),
                batch[i, 1 : 1 + n],
                reduction="sum",
            ).item()
        )
    return total


def score_ids(model: Llama, ids: list[int], eos: int, device: torch.device) -> float:
    spans = [ids[a:b] for a, b in text_windows(len(ids))]
    total = 0.0
    step = 4
    for i in range(0, len(spans), step):
        total += _score_spans(model, spans[i : i + step], eos, device)
    return total


def evaluate(args) -> dict:
    started = now_utc()
    device = torch.device("xpu" if torch.xpu.is_available() else "cpu")
    if device.type != "xpu":
        raise SystemExit("STOP: torch.xpu is not available")
    arm = load_arm(args.arm, Path(args.tokenizer))
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = Llama(int(payload["vocab"]))
    model.load_state_dict(payload["model"])
    model.to(device)
    model.eval()
    groups: dict[str, dict[str, float]] = defaultdict(lambda: {"nll": 0.0, "bytes": 0, "docs": 0})
    n = 0
    with torch.inference_mode():
        for path in sorted(Path(args.eval_dir).rglob("*.jsonl.zst")):
            for row in iter_jsonl_zst(path):
                text = row["text"]
                original = int(row["original_bytes"])
                if original <= 0 or text == "":
                    continue
                nll = score_ids(model, arm.encode(text), arm.eos_id, device)
                key = f"{path.stem}:{row['lang']}"
                groups[key]["nll"] += nll
                groups[key]["bytes"] += original
                groups[key]["docs"] += 1
                n += 1
                if n % 200 == 0:
                    print(f"scored {n}", flush=True)
    report = {}
    for key, agg in sorted(groups.items()):
        report[key] = {
            "docs": agg["docs"],
            "bytes": agg["bytes"],
            "nll_nats": agg["nll"],
            "bpb": bpb(agg["nll"], int(agg["bytes"])),
        }
    out = attach_build(
        {
            "kind": "m9_bpb",
            "arm": args.arm,
            "checkpoint": str(args.checkpoint),
            "completed_steps": payload.get("completed_steps"),
            "seed": payload.get("seed"),
            "ln2": math.log(2),
            "groups": report,
            "started_at": started,
            "finished_at": now_utc(),
        },
        args.build,
    )
    dest = Path(args.out)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {dest}", flush=True)
    return out

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
from ancoda_tok.m9.train import TORCH_THREADS, configure_cpu, cpu_temp_c

EVAL_BATCH = 2


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
    for i in range(0, len(spans), EVAL_BATCH):
        total += _score_spans(model, spans[i : i + EVAL_BATCH], eos, device)
    return total


def _empty_agg() -> dict:
    return {"nll": 0.0, "bytes": 0, "docs": 0}


def _finalize_groups(groups: dict[str, dict]) -> dict:
    report = {}
    for key, agg in sorted(groups.items()):
        report[key] = {
            "docs": agg["docs"],
            "bytes": agg["bytes"],
            "nll_nats": agg["nll"],
            "bpb": bpb(agg["nll"], int(agg["bytes"])) if agg["bytes"] else None,
        }
    return report


def evaluate(args) -> dict:
    configure_cpu()
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

    sid_arm = None
    if getattr(args, "sid_tokenizer", None) is not None:
        sid_arm = load_arm("sidskarki-64k", Path(args.sid_tokenizer))

    groups: dict[str, dict] = defaultdict(_empty_agg)
    rt_groups: dict[str, dict] = defaultdict(_empty_agg)
    by_set: dict[str, dict[str, int]] = defaultdict(lambda: {"docs": 0, "ok": 0})
    by_lang: dict[str, dict[str, int]] = defaultdict(lambda: {"docs": 0, "ok": 0})
    n = 0
    peak_cpu_c: float | None = None
    with torch.inference_mode():
        for path in sorted(Path(args.eval_dir).rglob("*.jsonl.zst")):
            set_name = path.relative_to(args.eval_dir).as_posix()
            for row in iter_jsonl_zst(path):
                text = row["text"]
                original = int(row["original_bytes"])
                if original <= 0 or text == "":
                    continue
                lang = str(row["lang"])
                nll = score_ids(model, arm.encode(text), arm.eos_id, device)
                key = f"{path.stem}:{lang}"
                groups[key]["nll"] += nll
                groups[key]["bytes"] += original
                groups[key]["docs"] += 1
                if sid_arm is not None:
                    ids = sid_arm.encode(text)
                    ok = sid_arm.decode(ids) == text
                    by_set[set_name]["docs"] += 1
                    by_lang[lang]["docs"] += 1
                    if ok:
                        by_set[set_name]["ok"] += 1
                        by_lang[lang]["ok"] += 1
                        rt_groups[key]["nll"] += nll
                        rt_groups[key]["bytes"] += original
                        rt_groups[key]["docs"] += 1
                n += 1
                if n % 200 == 0:
                    temp_c = cpu_temp_c()
                    if temp_c is not None:
                        peak_cpu_c = temp_c if peak_cpu_c is None else max(peak_cpu_c, temp_c)
                    print(f"scored {n} cpu_temp_c={temp_c}", flush=True)

    roundtrip = None
    if sid_arm is not None:

        def rate_map(src: dict[str, dict[str, int]]) -> dict:
            out = {}
            for name, agg in sorted(src.items()):
                docs = agg["docs"]
                out[name] = {
                    "docs": docs,
                    "ok": agg["ok"],
                    "rate": (agg["ok"] / docs) if docs else None,
                }
            return out

        roundtrip = {
            "per_eval_set": rate_map(by_set),
            "per_language": rate_map(by_lang),
            "bpb_on_lossless_docs": _finalize_groups(rt_groups),
            "note": (
                "bpb_on_lossless_docs scores this arm's NLL only on documents where "
                "sidskarki decode(encode(x)) == x. Pair with the matching arm-C report "
                "for the secondary A vs C comparison."
            ),
        }

    out = attach_build(
        {
            "kind": "m9_bpb",
            "arm": args.arm,
            "checkpoint": str(args.checkpoint),
            "completed_steps": payload.get("completed_steps"),
            "seed": payload.get("seed"),
            "ln2": math.log(2),
            "groups": _finalize_groups(groups),
            "sidskarki_roundtrip": roundtrip,
            "torch_threads": TORCH_THREADS,
            "eval_batch": EVAL_BATCH,
            "peak_cpu_temp_c": peak_cpu_c,
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

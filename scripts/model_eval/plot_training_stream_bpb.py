#!/usr/bin/env python3
"""Training-stream BPB (approximate) from logged nats/token loss. No new scoring."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[2]
RUNS = REPO / "artifacts/model_eval/server_copy/runs"
EVAL = REPO / "artifacts/model_eval/server_copy/evals"
OUT_PLOT = REPO / "artifacts/model_eval/plots/training_stream_bpb.png"
OUT_CSV = REPO / "artifacts/model_eval/plots/training_stream_bpb.csv"

# Corpus content tokens per UTF-8 byte. Used only to turn nats/token into an
# approximate training-stream BPB. The x-axis is the logged byte counter.
TPB = {
    "A": ("ancoda-64k", 0.150967),
    "B": ("yak-64k-v1", 0.151130),
    "C": ("sidskarki-64k", 0.215758),
}
LN2 = math.log(2.0)


def bstar_step(arm: str, seed: int) -> int | None:
    path = EVAL / f"{arm}-s{seed}-bstar.json"
    if not path.exists():
        return None
    return int(json.loads(path.read_text(encoding="utf-8"))["completed_steps"])


def main() -> None:
    OUT_PLOT.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = {"A": "#2f6fed", "B": "#888888", "C": "#c45c26"}
    styles = {1: "-", 2: "--", 3: ":"}

    for letter, (arm, tpb) in TPB.items():
        for seed in (1, 2, 3):
            log = RUNS / f"{arm}-s{seed}" / "log.jsonl"
            if not log.exists():
                raise SystemExit(f"missing {log}")
            xs: list[float] = []
            ys: list[float] = []
            bstar = bstar_step(arm, seed)
            bstar_xy: tuple[float, float] | None = None
            for line in log.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                r = json.loads(line)
                step = int(r["completed_steps"])
                loss = float(r["loss"])
                bytes_read = int(r["bytes_seen"])
                train_bpb = loss * tpb / LN2
                xs.append(bytes_read)
                ys.append(train_bpb)
                rows.append(
                    {
                        "arm": letter,
                        "tokenizer": arm,
                        "seed": seed,
                        "step": step,
                        "tokens_per_byte": tpb,
                        "loss_nats_per_token": loss,
                        "bytes_read": bytes_read,
                        "training_stream_bpb": train_bpb,
                        "is_bstar": int(bstar is not None and step == bstar),
                    }
                )
                if bstar is not None and step == bstar:
                    bstar_xy = (bytes_read, train_bpb)
            # Mark B*: if that step was not logged, use the nearest logged step.
            if bstar_xy is None and bstar is not None and xs:
                steps = [
                    int(json.loads(l)["completed_steps"])
                    for l in log.read_text(encoding="utf-8").splitlines()
                    if l.strip()
                ]
                nearest = min(steps, key=lambda s: abs(s - bstar))
                for row in rows:
                    if row["arm"] == letter and row["seed"] == seed and row["step"] == nearest:
                        row["is_bstar"] = 1
                        bstar_xy = (float(row["bytes_read"]), float(row["training_stream_bpb"]))
                        break
            ax.plot(
                xs,
                ys,
                color=colors[letter],
                linestyle=styles[seed],
                linewidth=1.2,
                label=f"{letter} s{seed}",
            )
            if bstar_xy is not None:
                ax.scatter(
                    [bstar_xy[0]],
                    [bstar_xy[1]],
                    color=colors[letter],
                    marker="o",
                    s=36,
                    zorder=5,
                    edgecolors="black",
                    linewidths=0.6,
                )

    ax.set_xlabel("bytes covered")
    ax.set_ylabel("training-stream BPB (approximate)")
    ax.set_title("training-stream BPB (approximate)")
    ax.legend(fontsize=8, ncol=3)
    fig.tight_layout()
    fig.savefig(OUT_PLOT, dpi=140)
    plt.close(fig)

    with OUT_CSV.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "arm",
                "tokenizer",
                "seed",
                "step",
                "tokens_per_byte",
                "loss_nats_per_token",
                "bytes_read",
                "training_stream_bpb",
                "is_bstar",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"wrote {OUT_PLOT}")
    print(f"wrote {OUT_CSV}")
    print(f"rows={len(rows)}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Run Unigram training under a hard RSS cap (default 12 GiB).

If RSS exceeds the cap, the child is killed and exit code 2 is returned so the
caller can skip the rung.
"""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DEFAULT_CAP = 12 * 1024**3


def rss_bytes(pid: int) -> int:
    # macOS: ps rss is in KiB.
    out = subprocess.check_output(["ps", "-o", "rss=", "-p", str(pid)], text=True).strip()
    if not out:
        return 0
    return int(out.split()[0]) * 1024


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot-prefix", default="n60")
    parser.add_argument(
        "--out",
        default=str(REPO / "artifacts" / "tokenizers" / "ladder-unigram"),
    )
    parser.add_argument("--cap-bytes", type=int, default=DEFAULT_CAP)
    args = parser.parse_args()

    env = dict(os.environ)
    cargo_bin = str(Path.home() / ".cargo" / "bin")
    path = env.get("PATH", "")
    if cargo_bin not in path.split(":"):
        env["PATH"] = f"{cargo_bin}:{path}"

    cmd = [
        "uv",
        "run",
        "python",
        "-u",
        "-m",
        "ancoda_tok.unigram",
        "--snapshot-prefix",
        args.snapshot_prefix,
        "--out",
        args.out,
    ]
    print(f"+ {' '.join(cmd)}", flush=True)
    print(f"memory cap = {args.cap_bytes} bytes ({args.cap_bytes / 1024**3:.2f} GiB)", flush=True)
    proc = subprocess.Popen(cmd, cwd=REPO, env=env)
    peak = 0
    try:
        while True:
            code = proc.poll()
            try:
                # Include children (SentencePiece may spawn threads in-process).
                total = rss_bytes(proc.pid)
                # Sum children RSS.
                try:
                    kids = subprocess.check_output(
                        ["pgrep", "-P", str(proc.pid)], text=True
                    ).split()
                except subprocess.CalledProcessError:
                    kids = []
                for kid in kids:
                    try:
                        total += rss_bytes(int(kid))
                    except (subprocess.CalledProcessError, ValueError):
                        pass
                peak = max(peak, total)
                print(f"rss_bytes={total} peak={peak}", flush=True)
                if total > args.cap_bytes:
                    print(
                        f"STOP: Unigram RSS {total} exceeded cap {args.cap_bytes}; killing",
                        flush=True,
                    )
                    proc.send_signal(signal.SIGTERM)
                    try:
                        proc.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                        proc.wait(timeout=10)
                    note = REPO / "artifacts" / "eval" / "mix_sweep" / "unigram_skipped.json"
                    note.parent.mkdir(parents=True, exist_ok=True)
                    note.write_text(
                        (
                            "{\n"
                            '  "skipped": true,\n'
                            f'  "reason": "RSS {total} exceeded cap {args.cap_bytes}",\n'
                            f'  "peak_rss_bytes": {peak},\n'
                            f'  "cap_bytes": {args.cap_bytes}\n'
                            "}\n"
                        ),
                        encoding="utf-8",
                    )
                    return 2
            except subprocess.CalledProcessError:
                pass
            if code is not None:
                print(f"unigram exit={code} peak_rss_bytes={peak}", flush=True)
                return code
            time.sleep(5)
    except KeyboardInterrupt:
        proc.send_signal(signal.SIGTERM)
        proc.wait(timeout=30)
        return 130


if __name__ == "__main__":
    sys.exit(main())

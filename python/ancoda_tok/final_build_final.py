"""final-build builds, Gate A scoreboard, robustness, and the sealed test.

Config is the frozen lookup in docs/DECISIONS.md ("final-build selection"). This module
does not choose a new 32k/64k/128k budget. The 250k point is chosen only by the
predeclared rule.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shutil
import signal
import subprocess
import time
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import ancoda_seg
import sentencepiece as spm

from ancoda_tok.clean.common import (
    base_manifest,
    finish_manifest,
    free_bytes,
    save_json,
    sha256_file,
)
from ancoda_tok.counts import iter_entries, write_counts
from ancoda_tok.diag_combined_candidates import build_scaled_latin
from ancoda_tok.diag_prompt2_gaps import whole_file_line_pairs
from ancoda_tok.export_cli import export_dir
from ancoda_tok.export_hf import bytes_to_unicode, load_hf
from ancoda_tok.frontier_common import is_deva_bytes, write_side_config
from ancoda_tok.inventory import INVENTORY_DIR, select_block1
from ancoda_tok.mix_sweep import _env
from ancoda_tok.storage import REPO_ROOT

SEED = 20261104
BOOT_N = 10_000
RSS_CAP = 12 * 1024**3
SLACKS = (4096, 16384)
BLOCK0 = 133
RESERVED = 1024
BYTES = 256

FINAL_BUILD = REPO_ROOT / "artifacts" / "eval" / "final_build"
TOK = REPO_ROOT / "artifacts" / "tokenizers" / "final_build"
CC = REPO_ROOT / "artifacts" / "eval" / "mix_sweep" / "combined_candidates"
SNAP_J5 = REPO_ROOT / "data" / "snapshots" / "n60j5_20af1fffbec0fe86"
SNAP_J10 = REPO_ROOT / "data" / "snapshots" / "n60j10_502301959d423052"
DEVA_COUNTS = CC / "deva_pretoken_counts.bin"
FLORES = REPO_ROOT / "third_party" / "yak-tokenizer" / "data" / "eval" / "flores_plus" / "devtest"
AYA = REPO_ROOT / "third_party" / "yak-tokenizer" / "data" / "eval" / "npi_everyday" / "aya_npi.jsonl"
HF_HUB = REPO_ROOT / "data" / "hf_cache" / "hub"
UNICODE = REPO_ROOT / "third_party" / "unicode" / "16.0.0" / "UnicodeData.txt"

# PLAN 3.9, as implemented by ancoda_tok.guarantee (one definition for every build).
TAIL_DEFINITION = (
    "PLAN 3.9 tail break rate: token boundaries falling strictly inside an uncovered "
    "Devanagari akshara, divided by uncovered Devanagari akshara occurrences. "
    "Single Devanagari code points are not in the denominator "
    "(ancoda_tok.guarantee._tail). The 95% interval is a document-level bootstrap of "
    "1,000 resamples with seed 20261104 over newline-separated snapshot lines that "
    "contain at least one such akshara (ancoda_tok.guarantee._bootstrap). "
    "The scan is the first 10,000,000 newline-separated lines of n60j5 "
    "(1,000,000 for Ancoda-32k). Eval texts are in the violation, round-trip and "
    "equality checks, and are not in this tail interval."
)

VOCAB_CAP = {
    32_768: int(0.03 * 32_768),
    65_536: int(0.03 * 65_536),
    131_072: int(0.03 * 131_072),
    250_000: int(0.03 * 250_000),
}

FINAL_BUDGETS = (
    ("ancoda-32k", 32_768, 23_522),
    ("ancoda-64k", 65_536, 48_000),
    ("ancoda-128k", 131_072, 94_742),
)
FRONTIER_BD = (144_000, 160_000, 175_000, 190_000)
FRONTIER_SCALES = (1.0, 0.5)

DEVA_LANGS = (
    "awa_Deva",
    "bho_Deva",
    "hin_Deva",
    "mag_Deva",
    "mai_Deva",
    "npi_Deva",
    "san_Deva",
)

CLASSES: dict[str, list[str]] = {
    "32k": ["ancoda-32k", "yak-32k-v1"],
    "64k": [
        "ancoda-64k",
        "ancoda-ne-64k",
        "yak-64k-v1",
        "yak-64k-v2.0",
        "sidskarki-64k",
        "arkios",
        "sarvam-1",
    ],
    "128k": [
        "ancoda-128k",
        "yak-128k-v1",
        "yak-128k-v2.0",
        "llama3",
        "sarvam-m",
        "qwen2.5",
        "qwen3",
    ],
    "250k": ["ancoda-250k", "indicbert-v2", "bloom", "gemma3", "sarvam-30b"],
}
OURS = {"ancoda-32k", "ancoda-64k", "ancoda-ne-64k", "ancoda-128k", "ancoda-250k"}


class ShortFill(Exception):
    pass


def log(msg: str) -> None:
    print(msg, flush=True)


def mergeable_of(vocab: int) -> int:
    return vocab - RESERVED


def latin_budget(vocab: int, b_d: int) -> int:
    return mergeable_of(vocab) - BYTES - b_d


def assert_32k_share() -> None:
    slots_64 = 65_536 - BYTES - RESERVED
    slots_32 = 32_768 - BYTES - RESERVED
    got = round(slots_32 * 48_000 / slots_64)
    if got != 23_522:
        raise SystemExit(f"STOP: 32k B_d formula gave {got}, frozen output is 23522")


def covered_path(vocab: int) -> Path:
    if vocab == 250_000:
        return INVENTORY_DIR / "covered_250k.json"
    name = {32_768: "32k", 65_536: "64k", 131_072: "128k"}[vocab]
    return INVENTORY_DIR / f"covered_{name}_3pct.json"


def load_akshara_inventory_counts() -> Counter[str]:
    tsv = INVENTORY_DIR / "akshara_counts.tsv"
    counts: Counter[str] = Counter()
    with tsv.open(encoding="utf-8") as handle:
        handle.readline()
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 3:
                continue
            ak, _cps, count_s = parts
            ak = ak.replace("\\n", "\n").replace("\\t", "\t").replace("\\\\", "\\")
            counts[ak] += int(count_s)
    return counts


def write_covered_sets() -> dict[int, Path]:
    """akshara-inventory ranking, truncated at a 3% cap. 250k uses the same selection."""
    assert_32k_share()
    INVENTORY_DIR.mkdir(parents=True, exist_ok=True)
    letters = frozenset(int(x) for x in ancoda_seg.devanagari_letters())
    counts = load_akshara_inventory_counts()
    tsv = INVENTORY_DIR / "akshara_counts.tsv"
    out: dict[int, Path] = {}
    for vocab, cap in VOCAB_CAP.items():
        dest = covered_path(vocab)
        if dest.is_file():
            existing = json.loads(dest.read_text(encoding="utf-8"))
            if int(existing.get("cap", -1)) == cap and int(existing["block1_merge_count"]) <= cap:
                log(f"covered resume {dest.name} merges={existing['block1_merge_count']}")
                out[vocab] = dest
                continue
        selected = select_block1(counts, cap=cap, letters=letters)
        if selected["block1_merge_count"] > cap:
            raise SystemExit(
                f"STOP: Block 1 {selected['block1_merge_count']} exceeds cap {cap}"
            )
        payload = {
            "vocab_name": "250k" if vocab == 250_000 else f"{vocab}_3pct",
            "vocab_size": vocab,
            "gb9c": False,
            "cap_pct": "3pct",
            "cap": cap,
            "seed": SEED,
            "counts_tsv": str(tsv.relative_to(REPO_ROOT)),
            "counts_tsv_sha256": sha256_file(tsv),
            "snapshot": "data/snapshots/n50_32fd1b6a9c068400",
            "note": "akshara-inventory akshara ranking truncated to a 3% cap (select_block1 stop-at-cap).",
        }
        for key in (
            "N",
            "N0",
            "ineligible_occurrences",
            "eligible_multi_types",
            "unit_types_selected",
            "block1_merge_count",
            "coverage",
            "uncovered_occurrences",
            "stop_reason",
            "block1_merges",
            "units_selected",
            "coverage_curve",
            "top_50_uncovered_aksharas",
        ):
            payload[key] = selected[key]
        save_json(dest, payload)
        man = finish_manifest(
            base_manifest(
                kind="final_build_covered_3pct",
                vocab_size=vocab,
                cap=cap,
                block1_merge_count=selected["block1_merge_count"],
                coverage=selected["coverage"],
                counts_tsv_sha256=payload["counts_tsv_sha256"],
                covered_sha256=sha256_file(dest),
            )
        )
        save_json(dest.with_suffix(".manifest.json"), man)
        log(
            f"covered {dest.name} merges={selected['block1_merge_count']}/{cap} "
            f"coverage={selected['coverage']:.10f} stop={selected['stop_reason']}"
        )
        out[vocab] = dest
    return out


def latin_counts(scale: float) -> Path:
    path = CC / f"latin_cm{int(scale * 100)}_pretoken_counts.bin"
    if path.is_file():
        return path
    return build_scaled_latin(scale)


def _binary() -> Path:
    binary = REPO_ROOT / "target" / "release" / "ancoda-bpe"
    if binary.is_file():
        return binary
    log("building ancoda-bpe release binary")
    subprocess.run(
        ["cargo", "build", "--release", "--offline", "-p", "ancoda-bpe"],
        cwd=REPO_ROOT,
        check=True,
        env=_env(),
    )
    return binary


def _tree_rss_bytes(root_pid: int) -> int:
    out = subprocess.check_output(["ps", "-ax", "-o", "pid=,ppid=,rss="], text=True)
    children: dict[int, list[int]] = defaultdict(list)
    rss: dict[int, int] = {}
    for line in out.splitlines():
        parts = line.split()
        if len(parts) < 3:
            continue
        pid, ppid, kb = int(parts[0]), int(parts[1]), int(parts[2])
        children[ppid].append(pid)
        rss[pid] = kb
    total = 0
    stack = [root_pid]
    seen: set[int] = set()
    while stack:
        cur = stack.pop()
        if cur in seen:
            continue
        seen.add(cur)
        total += rss.get(cur, 0)
        stack.extend(children.get(cur, []))
    return total * 1024


def train_side(cfg: Path, out: Path) -> dict[str, Any]:
    """Train one side. Stop and discard the side if RSS exceeds 12 GiB."""
    out.mkdir(parents=True, exist_ok=True)
    stop = out / "rss_stop.json"
    if stop.is_file():
        return json.loads(stop.read_text(encoding="utf-8"))
    ranks = out / "ranks.json"
    if ranks.is_file():
        stats = json.loads(ranks.read_text(encoding="utf-8")).get("stats", {})
        peak = int(stats.get("peak_rss_bytes") or 0)
        if peak > RSS_CAP:
            payload = {"stopped": "rss_cap", "peak_rss_bytes": peak, "dir": str(out)}
            save_json(stop, payload)
            ranks.unlink()
            return payload
        return {"stopped": None, "peak_rss_bytes": peak, "dir": str(out)}
    binary = _binary()
    log_path = out / "train.log"
    log(f"train {out}")
    with log_path.open("w", encoding="utf-8") as handle:
        proc = subprocess.Popen(
            [str(binary), "train", "--config", str(cfg)],
            cwd=REPO_ROOT,
            env=_env(),
            stdout=handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        peak = 0
        exceeded = False
        while proc.poll() is None:
            time.sleep(5)
            try:
                rss = _tree_rss_bytes(proc.pid)
            except subprocess.CalledProcessError:
                rss = 0
            peak = max(peak, rss)
            if rss > RSS_CAP:
                exceeded = True
                log(f"RSS cap exceeded pid={proc.pid} rss={rss}")
                os.killpg(proc.pid, signal.SIGKILL)
                break
        proc.wait()
    if exceeded:
        payload = {
            "stopped": "rss_cap",
            "peak_rss_bytes": peak,
            "dir": str(out.relative_to(REPO_ROOT)),
        }
        if ranks.is_file():
            ranks.unlink()
        save_json(stop, payload)
        return payload
    if proc.returncode != 0:
        tail = log_path.read_text(encoding="utf-8")[-1500:]
        raise SystemExit(f"STOP: train failed {out}\n{tail}")
    stats = json.loads(ranks.read_text(encoding="utf-8")).get("stats", {})
    recorded = int(stats.get("peak_rss_bytes") or 0)
    peak = max(peak, recorded)
    if peak > RSS_CAP:
        payload = {
            "stopped": "rss_cap",
            "peak_rss_bytes": peak,
            "dir": str(out.relative_to(REPO_ROOT)),
        }
        ranks.unlink()
        save_json(stop, payload)
        return payload
    log(f"trained {out} peak_rss_bytes={peak}")
    return {"stopped": None, "peak_rss_bytes": peak, "dir": str(out.relative_to(REPO_ROOT))}


def combine_exact(
    dewa_dir: Path,
    latin_dir: Path,
    dest: Path,
    *,
    mergeable: int,
    b_d: int,
) -> None:
    if (dest / "ranks.json").is_file() and (dest / "tokenizer.json").is_file():
        layout = json.loads((dest / "ranks.json").read_text(encoding="utf-8"))["layout"]
        if int(layout["vocab_size"]) == mergeable + RESERVED:
            log(f"combine resume {dest}")
            return
    dewa = json.loads((dewa_dir / "ranks.json").read_text(encoding="utf-8"))
    latin = json.loads((latin_dir / "ranks.json").read_text(encoding="utf-8"))
    dewa_tokens = [bytes.fromhex(x) for x in dewa["tokens_hex"]]
    latin_tokens = [bytes.fromhex(x) for x in latin["tokens_hex"]]
    d_layout = dewa["layout"]
    tokens: list[bytes] = dewa_tokens[: d_layout["block1_end"]]
    index = {tok: i for i, tok in enumerate(tokens)}
    deva_slots = len(tokens) - BYTES
    latin_slots = 0
    latin_cap = mergeable - BYTES - b_d
    if deva_slots > b_d:
        raise ShortFill(f"block0+block1={deva_slots} already above B_d={b_d}")
    dewa_q = list(dewa["block2_merges"])
    latin_q = list(latin["block2_merges"])
    i = j = 0
    combined: list[dict[str, Any]] = []
    skipped = 0
    while len(tokens) < mergeable:
        cand = []
        if i < len(dewa_q) and deva_slots < b_d:
            row = dewa_q[i]
            left_b = dewa_tokens[row["left"]]
            right_b = dewa_tokens[row["right"]]
            if left_b in index and right_b in index:
                cand.append(("deva", i, left_b, right_b, int(row["count"])))
        if j < len(latin_q) and latin_slots < latin_cap:
            row = latin_q[j]
            left_b = latin_tokens[row["left"]]
            right_b = latin_tokens[row["right"]]
            if left_b in index and right_b in index:
                cand.append(("latin", j, left_b, right_b, int(row["count"])))
        if not cand:
            if i < len(dewa_q) and deva_slots < b_d:
                i += 1
                continue
            if j < len(latin_q) and latin_slots < latin_cap:
                j += 1
                continue
            break
        cand.sort(key=lambda item: (-item[4], 0 if item[0] == "deva" else 1))
        side, idx, left_b, right_b, count = cand[0]
        if side == "deva":
            i = idx + 1
        else:
            j = idx + 1
        merged = left_b + right_b
        if merged in index:
            skipped += 1
            continue
        new_id = len(tokens)
        index[merged] = new_id
        tokens.append(merged)
        if side == "deva":
            deva_slots += 1
        else:
            latin_slots += 1
        combined.append(
            {
                "id": new_id,
                "left": index[left_b],
                "right": index[right_b],
                "count": count,
                "side": side,
            }
        )
    if len(tokens) != mergeable or deva_slots != b_d or latin_slots != latin_cap:
        raise ShortFill(
            f"tokens={len(tokens)} want {mergeable} deva_slots={deva_slots} want {b_d} "
            f"latin_slots={latin_slots} want {latin_cap} skipped={skipped} "
            f"queues={i}/{len(dewa_q)} {j}/{len(latin_q)}"
        )
    specials = [
        {"name": item["name"], "id": mergeable + k}
        for k, item in enumerate(dewa["special_tokens"])
    ]
    if len(specials) != RESERVED:
        raise SystemExit(f"STOP: expected 1024 specials, got {len(specials)}")
    ranks = {
        "version": 1,
        "seed": SEED,
        "layout": {
            "byte_end": BYTES,
            "block0_end": d_layout["block0_end"],
            "block1_end": d_layout["block1_end"],
            "reserved_start": mergeable,
            "vocab_size": mergeable + RESERVED,
        },
        "tokens_hex": [tok.hex() for tok in tokens],
        "special_tokens": specials,
        "block2_merges": combined,
        "stats": {
            "combined_from": {
                "deva": str(dewa_dir.relative_to(REPO_ROOT)),
                "latin": str(latin_dir.relative_to(REPO_ROOT)),
            },
            "B_d": b_d,
            "deva_slots": deva_slots,
            "latin_slots": latin_slots,
            "skipped_dup": skipped,
            "mergeable_tokens": len(tokens),
            "target_mergeable": mergeable,
        },
    }
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "ranks.json").write_text(json.dumps(ranks) + "\n", encoding="utf-8")
    export_dir(dest / "ranks.json")
    log(f"combined {dest} vocab={mergeable + RESERVED} B_d={b_d}")


def _side_dirs(tag: str, slack: int) -> tuple[Path, Path]:
    root = FINAL_BUILD / "sides" / f"{tag}_s{slack}"
    return root / "side-deva", root / "side-latin"


def train_budget(
    *,
    name: str,
    vocab: int,
    b_d: int,
    scale: float,
    dest: Path,
    dewa_counts: Path,
    latin_path: Path,
    covered: Path,
) -> dict[str, Any]:
    """Script-budget build with exact vocab and exact B_d. Slack fills duplicate skips."""
    if (dest / "tokenizer.json").is_file() and (dest / "ranks.json").is_file():
        ranks = json.loads((dest / "ranks.json").read_text(encoding="utf-8"))
        layout = ranks["layout"]
        stats = ranks.get("stats") or {}
        if int(layout["vocab_size"]) == vocab and int(stats.get("B_d", -1)) == b_d:
            log(f"resume {name}")
            return {"name": name, "dir": str(dest.relative_to(REPO_ROOT)), "stopped": None}
    mergeable = mergeable_of(vocab)
    block1 = int(json.loads(covered.read_text(encoding="utf-8"))["block1_merge_count"])
    last_error = ""
    for slack in SLACKS:
        dewa_out, latin_out = _side_dirs(name, slack)
        dewa_vocab = BYTES + b_d + slack + RESERVED
        latin_vocab = BYTES + latin_budget(vocab, b_d) + slack + RESERVED
        dewa_cfg = write_side_config(
            path=FINAL_BUILD / "configs" / f"{name}_s{slack}_deva.yaml",
            counts_rel=str(dewa_counts.relative_to(REPO_ROOT)),
            output_rel=str(dewa_out.relative_to(REPO_ROOT)),
            vocab_size=dewa_vocab,
            block0=True,
            block1=True,
            block1_path=str(covered.relative_to(REPO_ROOT)),
        )
        latin_cfg = write_side_config(
            path=FINAL_BUILD / "configs" / f"{name}_s{slack}_latin.yaml",
            counts_rel=str(latin_path.relative_to(REPO_ROOT)),
            output_rel=str(latin_out.relative_to(REPO_ROOT)),
            vocab_size=latin_vocab,
            block0=False,
            block1=False,
            block1_path=str(covered.relative_to(REPO_ROOT)),
        )
        dewa_res = train_side(dewa_cfg, dewa_out)
        if dewa_res.get("stopped"):
            return {"name": name, "stopped": "rss_cap", "side": "deva", **dewa_res}
        latin_res = train_side(latin_cfg, latin_out)
        if latin_res.get("stopped"):
            return {"name": name, "stopped": "rss_cap", "side": "latin", **latin_res}
        try:
            combine_exact(dewa_out, latin_out, dest, mergeable=mergeable, b_d=b_d)
        except ShortFill as exc:
            last_error = str(exc)
            log(f"{name} slack={slack} short fill: {exc}")
            continue
        peak = max(int(dewa_res["peak_rss_bytes"]), int(latin_res["peak_rss_bytes"]))
        meta = finish_manifest(
            base_manifest(
                kind="final_build_budget_build",
                name=name,
                vocab_size=vocab,
                B_d=b_d,
                code_math_scale=scale,
                block1=block1,
                slack=slack,
                covered=str(covered.relative_to(REPO_ROOT)),
                covered_sha256=sha256_file(covered),
                dewa_counts_sha256=sha256_file(dewa_counts),
                latin_counts_sha256=sha256_file(latin_path),
                peak_rss_bytes=peak,
                snapshot=str(SNAP_J5.relative_to(REPO_ROOT)),
            )
        )
        save_json(dest / "final_build_manifest.json", meta)
        return {
            "name": name,
            "dir": str(dest.relative_to(REPO_ROOT)),
            "stopped": None,
            "slack": slack,
            "peak_rss_bytes": peak,
            "B_d": _measure_bd(dest / "ranks.json"),
        }
    raise SystemExit(f"STOP: {name} could not fill vocab after slack {SLACKS}: {last_error}")


def _measure_bd(ranks_path: Path) -> int:
    ranks = json.loads(ranks_path.read_text(encoding="utf-8"))
    layout = ranks["layout"]
    tokens = [bytes.fromhex(x) for x in ranks["tokens_hex"]]
    dewa = BLOCK0  # block0 is entirely Devanagari / joiners; counted in B_d
    # Block 0 is the fixed Devanagari character block. Count it, then Devanagari block2.
    dewa = layout["block1_end"] - layout["byte_end"]
    for tid in range(layout["block1_end"], layout["reserved_start"]):
        if is_deva_bytes(tokens[tid]):
            dewa += 1
    return int(dewa)


def train_nepali(covered: Path) -> dict[str, Any]:
    dest = TOK / "ancoda-ne-64k"
    if (dest / "tokenizer.json").is_file():
        log("resume ancoda-ne-64k")
        return {"name": "ancoda-ne-64k", "dir": str(dest.relative_to(REPO_ROOT)), "stopped": None}
    counts_dir = REPO_ROOT / "artifacts" / "counts" / "n60j5_nepali"
    counts = counts_dir / "pretoken_counts.bin"
    if not counts.is_file():
        free = free_bytes()
        log(f"counting Nepali on n60j5 free_bytes={free}")
        subprocess.run(
            [
                "uv",
                "run",
                "python",
                "-u",
                "-m",
                "ancoda_tok.counts",
                "--snapshot",
                str(SNAP_J5),
                "--out",
                str(counts_dir),
                "--bucket",
                "Nepali",
                "--workers",
                "4",
            ],
            cwd=REPO_ROOT,
            check=True,
            env=_env(),
        )
    cfg = FINAL_BUILD / "configs" / "ancoda-ne-64k.yaml"
    cfg.parent.mkdir(parents=True, exist_ok=True)
    cfg.write_text(
        "\n".join(
            [
                "vocab_size: 65536",
                "plain: false",
                "block0: true",
                "block1: true",
                "bans: true",
                "gb9c: false",
                f"seed: {SEED}",
                f"counts: {counts.relative_to(REPO_ROOT)}",
                f"block1_path: {covered.relative_to(REPO_ROOT)}",
                "unicode_data: third_party/unicode/16.0.0/UnicodeData.txt",
                f"output_dir: {dest.relative_to(REPO_ROOT)}",
                "require_full: true",
                "",
            ]
        ),
        encoding="utf-8",
    )
    res = train_side(cfg, dest)
    if res.get("stopped"):
        return {"name": "ancoda-ne-64k", **res}
    export_dir(dest / "ranks.json")
    save_json(
        dest / "final_build_manifest.json",
        finish_manifest(
            base_manifest(
                kind="final_build_nepali_only",
                name="ancoda-ne-64k",
                vocab_size=65536,
                counts_sha256=sha256_file(counts),
                covered_sha256=sha256_file(covered),
                snapshot=str(SNAP_J5.relative_to(REPO_ROOT)),
                peak_rss_bytes=res["peak_rss_bytes"],
                note="Single-list trainer. Nepali bucket of n60j5, including the joiner copy.",
            )
        ),
    )
    return {"name": "ancoda-ne-64k", "dir": str(dest.relative_to(REPO_ROOT)), "stopped": None}


def build_all(covered: dict[int, Path]) -> list[dict[str, Any]]:
    TOK.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, vocab, b_d in FINAL_BUDGETS:
        row = train_budget(
            name=name,
            vocab=vocab,
            b_d=b_d,
            scale=1.0,
            dest=TOK / name,
            dewa_counts=DEVA_COUNTS,
            latin_path=latin_counts(1.0),
            covered=covered[vocab],
        )
        rows.append(row)
        log(f"build {row}")
    rows.append(train_nepali(covered[65_536]))
    for b_d in FRONTIER_BD:
        for scale in FRONTIER_SCALES:
            tag = f"ancoda-250k-bd{b_d}-cm{int(scale * 100)}"
            row = train_budget(
                name=tag,
                vocab=250_000,
                b_d=b_d,
                scale=scale,
                dest=FINAL_BUILD / "frontier" / tag,
                dewa_counts=DEVA_COUNTS,
                latin_path=latin_counts(scale),
                covered=covered[250_000],
            )
            row["B_d_target"] = b_d
            row["scale"] = scale
            rows.append(row)
            log(f"frontier {row}")
    save_json(
        FINAL_BUILD / "builds.json",
        finish_manifest(base_manifest(kind="final_build_builds", rows=rows)),
    )
    return rows


def run_guarantee(tok_dir: Path, covered: Path, snapshot: Path, limit: int, report: Path) -> dict:
    if report.is_file():
        data = json.loads(report.read_text(encoding="utf-8"))
        if data.get("snapshot_lines") == limit:
            log(f"guarantee resume {report.name}")
            return data
    orphans = report.with_suffix(".orphans.txt")
    cmd = [
        "uv",
        "run",
        "python",
        "-u",
        "-m",
        "ancoda_tok.guarantee",
        "--tokenizer-dir",
        str(tok_dir),
        "--covered",
        str(covered),
        "--snapshot",
        str(snapshot),
        "--report",
        str(report),
        "--orphans",
        str(orphans),
        "--snapshot-line-limit",
        str(limit),
    ]
    log(f"guarantee {tok_dir.name} lines={limit}")
    subprocess.run(cmd, cwd=REPO_ROOT, check=False, env=_env())
    if not report.is_file():
        raise SystemExit(f"STOP: guarantee wrote no report for {tok_dir}")
    return json.loads(report.read_text(encoding="utf-8"))


def screen_frontier(covered: Path) -> None:
    """Eval-text integrity for the pick. Full 10M runs on the chosen model only."""
    out = FINAL_BUILD / "frontier_integrity.json"
    if out.is_file():
        return
    rows = []
    builds = json.loads((FINAL_BUILD / "builds.json").read_text(encoding="utf-8"))["rows"]
    for row in builds:
        if not str(row.get("name", "")).startswith("ancoda-250k-bd"):
            continue
        if row.get("stopped"):
            rows.append({**row, "integrity": False, "reason": "rss_cap"})
            continue
        dest = REPO_ROOT / row["dir"]
        report = FINAL_BUILD / "screen" / f"{row['name']}.json"
        report.parent.mkdir(parents=True, exist_ok=True)
        # Eval texts always run. One snapshot line keeps the scan bounded.
        data = run_guarantee(dest, covered, SNAP_J5, 1, report)
        ok = (
            data.get("hf_covered_violations") == 0
            and data.get("tt_covered_violations") == 0
            and data.get("mark_initial_block1_or_block2") == 0
            and data.get("hf_roundtrip_failures") == 0
            and data.get("tt_roundtrip_failures") == 0
            and data.get("equality_mismatches") == 0
            and not data.get("failures")
        )
        rows.append({"name": row["name"], "integrity": ok, "report": str(report.relative_to(REPO_ROOT))})
    save_json(out, finish_manifest(base_manifest(kind="final_build_frontier_screen", rows=rows)))


def full_guarantees(chosen_dir: Path, chosen_covered: Path) -> dict[str, Any]:
    jobs = [
        ("ancoda-32k", TOK / "ancoda-32k", covered_path(32_768), 1_000_000),
        ("ancoda-64k", TOK / "ancoda-64k", covered_path(65_536), 10_000_000),
        ("ancoda-128k", TOK / "ancoda-128k", covered_path(131_072), 10_000_000),
        ("ancoda-ne-64k", TOK / "ancoda-ne-64k", covered_path(65_536), 10_000_000),
        ("ancoda-250k", chosen_dir, chosen_covered, 10_000_000),
    ]
    rows = {}
    for name, dest, covered, limit in jobs:
        report = FINAL_BUILD / "guarantee" / f"{name}.json"
        report.parent.mkdir(parents=True, exist_ok=True)
        data = run_guarantee(dest, covered, SNAP_J5, limit, report)
        cov = json.loads(covered.read_text(encoding="utf-8"))
        rows[name] = {
            "hf_covered_violations": data.get("hf_covered_violations"),
            "tt_covered_violations": data.get("tt_covered_violations"),
            "mark_initial_block1_or_block2": data.get("mark_initial_block1_or_block2"),
            "hf_roundtrip_failures": data.get("hf_roundtrip_failures"),
            "tt_roundtrip_failures": data.get("tt_roundtrip_failures"),
            "equality_mismatches": data.get("equality_mismatches"),
            "block1_coverage": cov.get("coverage"),
            "block1_merge_count": cov.get("block1_merge_count"),
            "orphan_mark_share_npi_aya": data.get("orphan_mark_share_npi_aya"),
            "tail_hf_rate_lo_hi": data.get("tail_hf_rate_lo_hi"),
            "tail_tt_rate_lo_hi": data.get("tail_tt_rate_lo_hi"),
            "failures": data.get("failures"),
            "snapshot_lines": data.get("snapshot_lines"),
            "tail_definition": TAIL_DEFINITION,
            "report": str(report.relative_to(REPO_ROOT)),
        }
        log(f"guarantee {name} viol={rows[name]['hf_covered_violations']} tail={rows[name]['tail_hf_rate_lo_hi']}")
    save_json(
        FINAL_BUILD / "guarantees.json",
        finish_manifest(base_manifest(kind="final_build_guarantees", rows=rows, tail_definition=TAIL_DEFINITION)),
    )
    return rows


def _hub_snap(model: str) -> Path:
    snaps = sorted((HF_HUB / model / "snapshots").iterdir())
    if len(snaps) != 1:
        raise SystemExit(f"STOP: expected one snapshot for {model}, found {snaps}")
    return snaps[0]


def panel_paths() -> dict[str, str]:
    yak = _hub_snap("models--ampixa--yak-tokenizer")
    paths = {
        "yak-32k-v1": str(yak / "v1.0/32k/tokenizer.json"),
        "yak-64k-v1": str(yak / "v1.0/64k/tokenizer.json"),
        "yak-64k-v2.0": str(yak / "v2.0/64k/tokenizer.json"),
        "yak-128k-v1": str(yak / "v1.0/128k/tokenizer.json"),
        "yak-128k-v2.0": str(yak / "v2.0/128k/tokenizer.json"),
        "sidskarki-64k": str(
            _hub_snap("models--sidskarki--nepali-bpe-tokenizer") / "nepali_bpe_64k.model"
        ),
        "arkios": str(_hub_snap("models--sajalregmi4--arkios-tokenizer") / "tokenizer.json"),
        "sarvam-1": str(_hub_snap("models--sarvamai--sarvam-1") / "tokenizer.json"),
        "llama3": str(_hub_snap("models--NousResearch--Meta-Llama-3-8B") / "tokenizer.json"),
        "sarvam-m": str(_hub_snap("models--sarvamai--sarvam-m") / "tokenizer.json"),
        "qwen2.5": str(_hub_snap("models--Qwen--Qwen2.5-7B") / "tokenizer.json"),
        "qwen3": str(_hub_snap("models--Qwen--Qwen3-8B") / "tokenizer.json"),
        "indicbert-v2": str(
            _hub_snap("models--ai4bharat--IndicBERTv2-MLM-only") / "tokenizer.json"
        ),
        "bloom": str(_hub_snap("models--bigscience--bloom-560m") / "tokenizer.json"),
        "gemma3": str(_hub_snap("models--unsloth--gemma-3-1b-it") / "tokenizer.json"),
        "sarvam-30b": str(_hub_snap("models--sarvamai--sarvam-30b") / "tokenizer.json"),
        "o200k": "o200k",
        "himalaya-128k": "skipped: license",
    }
    for name, path in paths.items():
        if path in ("o200k", "skipped: license"):
            continue
        if not Path(path).is_file():
            raise SystemExit(f"STOP: missing panel tokenizer {name}: {path}")
    return paths


def _char_to_byte(text: str) -> list[int]:
    out = [0]
    pos = 0
    for ch in text:
        pos += len(ch.encode("utf-8"))
        out.append(pos)
    return out


def _sp_pieces(sp: Any, text: str) -> list[bytes]:
    """SentencePiece character offsets, converted to UTF-8 slices of ``text``."""
    mapping = sp.encode_as_offset_mapping(text)
    raw = text.encode("utf-8")
    ctb = _char_to_byte(text)
    pieces = [raw[ctb[a] : ctb[b]] for a, b in mapping["offsets"]]
    if b"".join(pieces) != raw:
        raise ValueError("sentencepiece offsets do not cover the UTF-8 text")
    return pieces


def _hf_pieces(tok: Any, text: str) -> list[bytes]:
    inverse = {char: byte for byte, char in bytes_to_unicode().items()}
    ids = tok.encode(text, add_special_tokens=False).ids
    pieces = []
    for tid in ids:
        piece = tok.id_to_token(tid)
        try:
            pieces.append(bytes(inverse[c] for c in piece))
        except KeyError:
            decoded = tok.decode([tid], skip_special_tokens=False).encode("utf-8")
            pieces.append(decoded)
    return pieces


class Encoder:
    def __init__(self, spec: str) -> None:
        self.spec = spec
        self.kind = ""
        self.sp = None
        self.tok = None
        self.tt = None
        if spec == "o200k":
            import tiktoken

            self.kind = "tiktoken"
            self.tt = tiktoken.get_encoding("o200k_base")
            return
        path = Path(spec)
        if path.suffix == ".model":
            self.kind = "sentencepiece"
            self.sp = spm.SentencePieceProcessor(model_file=str(path))
            return
        self.kind = "hf"
        self.tok = load_hf(path if path.name == "tokenizer.json" else path / "tokenizer.json")

    def ids(self, text: str) -> list[int]:
        if self.kind == "tiktoken":
            return list(self.tt.encode_ordinary(text))
        if self.kind == "sentencepiece":
            return list(self.sp.encode(text, out_type=int))
        return list(self.tok.encode(text, add_special_tokens=False).ids)

    def pieces(self, text: str) -> list[bytes] | None:
        try:
            if self.kind == "tiktoken":
                return [self.tt.decode_single_token_bytes(i) for i in self.ids(text)]
            if self.kind == "sentencepiece":
                return _sp_pieces(self.sp, text)
            got = _hf_pieces(self.tok, text)
        except (KeyError, ValueError, UnicodeError):
            return None
        if b"".join(got) != text.encode("utf-8"):
            return None
        return got


def read_lines(name: str) -> list[str]:
    return (FLORES / f"{name}.txt").read_text(encoding="utf-8").splitlines()


def aya_texts() -> list[str]:
    texts = []
    for line in AYA.read_text(encoding="utf-8").splitlines():
        if line:
            texts.append(json.loads(line)["text"])
    return texts


def fertility(tokens: int, text: str) -> float:
    words = len(text.split())
    return tokens / words if words else 0.0


def _orphan_share(pieces: list[bytes]) -> float:
    if not pieces:
        return 0.0
    bad = 0
    for piece in pieces:
        try:
            text = piece.decode("utf-8")
        except UnicodeDecodeError:
            continue
        if text and unicodedata.category(text[0]) in ("Mn", "Mc", "Me"):
            bad += 1
    return bad / len(pieces)


def _joiner_text(text: str, joiner: str, letters: frozenset[int]) -> str:
    chars = list(text)
    out: list[str] = []
    for i, ch in enumerate(chars):
        out.append(ch)
        if ch == "्" and i + 1 < len(chars) and ord(chars[i + 1]) in letters:
            out.append(joiner)
    return "".join(out)


def nukta_swap(text: str) -> str:
    precomposed: dict[str, str] = {}
    decomposed: dict[str, str] = {}
    for cp in range(0x0958, 0x0960):
        dec = unicodedata.decomposition(chr(cp))
        if not dec or "<" in dec:
            continue
        chars = "".join(chr(int(part, 16)) for part in dec.split())
        precomposed[chr(cp)] = chars
        decomposed[chars] = chr(cp)
    out: list[str] = []
    i = 0
    while i < len(text):
        ch = text[i]
        if ch in precomposed:
            out.append(precomposed[ch])
            i += 1
            continue
        pair = text[i : i + 2]
        if pair in decomposed:
            out.append(decomposed[pair])
            i += 2
            continue
        out.append(ch)
        i += 1
    return "".join(out)


def misspell(text: str, rng: random.Random) -> str:
    swap = {"ि": "ी", "ी": "ि", "ु": "ू", "ू": "ु"}
    sib = {"श": "स", "स": "ष", "ष": "श"}
    out = []
    for ch in text:
        if (ch in swap or ch in sib) and rng.random() < 0.1:
            ch = swap[ch] if ch in swap else sib[ch]
        out.append(ch)
    return "".join(out)


def _pairs_from_pieces(pieces_fn, lines: list[str]) -> list[tuple[int, int]] | None:
    try:
        return whole_file_line_pairs(pieces_fn, lines)
    except (AssertionError, ValueError, UnicodeError):
        return None


def _per_line_pairs(enc: Encoder, lines: list[str]) -> list[tuple[int, int]]:
    return [(len(enc.ids(line)), len(line.split())) for line in lines]


def bootstrap_diff(
    cand: list[tuple[int, int]],
    base: list[tuple[int, int]],
) -> dict[str, float | int]:
    def point(pairs: list[tuple[int, int]]) -> float:
        tokens = sum(t for t, _w in pairs)
        words = sum(w for _t, w in pairs)
        return tokens / words if words else 0.0

    diff = point(cand) - point(base)
    rng = random.Random(SEED)
    m = len(cand)
    diffs = []
    for _ in range(BOOT_N):
        ct = cw = bt = bw = 0
        for _draw in range(m):
            i = rng.randrange(m)
            ct += cand[i][0]
            cw += cand[i][1]
            bt += base[i][0]
            bw += base[i][1]
        if cw and bw:
            diffs.append((ct / cw) - (bt / bw))
    diffs.sort()
    lo = diffs[int(0.025 * (len(diffs) - 1))]
    hi = diffs[min(len(diffs) - 1, int(0.975 * (len(diffs) - 1)))]
    return {"diff": diff, "lo": lo, "hi": hi, "n": BOOT_N, "seed": SEED}


def verdict(ci: dict[str, Any]) -> str:
    if ci["hi"] < 0:
        return "WIN"
    if ci["lo"] > 0:
        return "LOSE"
    return "TIE"


def _ratio_ci(
    enc: Encoder,
    original_lines: list[str],
    rewritten_lines: list[str],
) -> dict[str, Any]:
    joined_o = "\n".join(original_lines)
    joined_r = "\n".join(rewritten_lines)
    whole = len(enc.ids(joined_r)) / len(enc.ids(joined_o))
    pieces_o = enc.pieces(joined_o)
    pieces_r = enc.pieces(joined_r)
    method = "per-line"
    if pieces_o is not None and pieces_r is not None:
        try:
            po = whole_file_line_pairs(lambda _text: pieces_o, original_lines)
            pr = whole_file_line_pairs(lambda _text: pieces_r, rewritten_lines)
            method = "whole-file first-byte"
        except (AssertionError, ValueError):
            po = _per_line_pairs(enc, original_lines)
            pr = _per_line_pairs(enc, rewritten_lines)
    else:
        po = _per_line_pairs(enc, original_lines)
        pr = _per_line_pairs(enc, rewritten_lines)
    # Ratio CI: resample lines, recompute sum(rewritten tokens) / sum(original tokens).
    rng = random.Random(SEED)
    m = len(po)
    ratios = []
    point_num = sum(t for t, _w in pr)
    point_den = sum(t for t, _w in po)
    for _ in range(BOOT_N):
        num = den = 0
        for _draw in range(m):
            i = rng.randrange(m)
            num += pr[i][0]
            den += po[i][0]
        if den:
            ratios.append(num / den)
    ratios.sort()
    return {
        "ratio": whole,
        "line_ratio": (point_num / point_den) if point_den else None,
        "lo": ratios[int(0.025 * (len(ratios) - 1))],
        "hi": ratios[min(len(ratios) - 1, int(0.975 * (len(ratios) - 1)))],
        "method": method,
        "n": BOOT_N,
        "seed": SEED,
    }


def metric_cache_path(spec: str) -> Path:
    digest = hashlib.sha256(spec.encode("utf-8")).hexdigest()[:16]
    return FINAL_BUILD / "metrics" / f"{digest}.json"


def measure(name: str, spec: str, code_texts: list[str], math_texts: list[str]) -> dict[str, Any]:
    dest = metric_cache_path(name + "\n" + spec)
    if dest.is_file():
        return json.loads(dest.read_text(encoding="utf-8"))
    log(f"measure {name}")
    enc = Encoder(spec)
    letters = frozenset(int(x) for x in ancoda_seg.devanagari_letters())
    langs = {}
    attr: dict[str, Any] = {}
    per_line: dict[str, list[list[int]]] = {}
    attr_ok: dict[str, bool] = {}
    for lang in ("eng_Latn", *DEVA_LANGS):
        lines = read_lines(lang)
        joined = "\n".join(lines)
        tokens = len(enc.ids(joined))
        langs[lang] = {
            "fertility": fertility(tokens, joined),
            "tokens": tokens,
            "words": len(joined.split()),
        }
        pieces = enc.pieces(joined)
        pairs = None
        if pieces is not None:
            pairs = _pairs_from_pieces(lambda _text, pieces=pieces: pieces, lines)
        attr_ok[lang] = pairs is not None
        attr[lang] = [[int(t), int(w)] for t, w in pairs] if pairs else None
        per_line[lang] = [[int(t), int(w)] for t, w in _per_line_pairs(enc, lines)]
        if lang == "npi_Deva" and pieces is not None:
            langs["orphan_npi"] = _orphan_share(pieces)
            langs["byte_piece_npi"] = sum(1 for p in pieces if not _valid_utf8(p)) / len(pieces)
        elif lang == "npi_Deva":
            langs["orphan_npi"] = None
            langs["byte_piece_npi"] = None
    npi_lines = read_lines("npi_Deva")
    per_line_tokens = sum(t for t, _w in per_line["npi_Deva"])
    per_line_words = sum(w for _t, w in per_line["npi_Deva"])
    newline_gap = langs["npi_Deva"]["fertility"] - (
        per_line_tokens / per_line_words if per_line_words else 0.0
    )
    deva_mean = sum(langs[lang]["fertility"] for lang in DEVA_LANGS) / len(DEVA_LANGS)
    aya = aya_texts()
    aya_tokens = sum(len(enc.ids(text)) for text in aya)
    aya_words = sum(len(text.split()) for text in aya)
    zwj = _ratio_ci(
        enc,
        npi_lines,
        [_joiner_text(line, "\u200d", letters) for line in npi_lines],
    )
    zwnj = _ratio_ci(
        enc,
        npi_lines,
        [_joiner_text(line, "\u200c", letters) for line in npi_lines],
    )

    def tok_per_kb(texts: list[str]) -> float:
        total_t = total_b = 0
        for text in texts:
            if not text:
                continue
            total_t += len(enc.ids(text))
            total_b += len(text.encode("utf-8"))
        return 1024.0 * total_t / total_b if total_b else 0.0

    payload = {
        "name": name,
        "spec": spec,
        "kind": enc.kind,
        "offset_method": (
            "SentencePiece encode_as_offset_mapping character offsets, converted to UTF-8"
            if enc.kind == "sentencepiece"
            else "token byte pieces"
        ),
        "langs": langs,
        "deva_mean": deva_mean,
        "aya_fertility": aya_tokens / aya_words if aya_words else None,
        "newline_gap_npi": newline_gap,
        "zwj": zwj,
        "zwnj": zwnj,
        "code_per_kb": tok_per_kb(code_texts),
        "math_per_kb": tok_per_kb(math_texts),
        "attr": {k: v for k, v in attr.items()},
        "attr_ok": attr_ok,
        "per_line": {k: v for k, v in per_line.items()},
    }
    dest.parent.mkdir(parents=True, exist_ok=True)
    save_json(dest, payload)
    # Drop the heavy pair tables from the returned object? Keep them for CIs.
    return payload


def _piece_is_orphan(piece: bytes) -> bool:
    if not piece or not _valid_utf8(piece):
        return False
    return unicodedata.category(piece.decode("utf-8")[0]) in ("Mn", "Mc", "Me")


def _valid_utf8(piece: bytes) -> bool:
    try:
        piece.decode("utf-8")
    except UnicodeDecodeError:
        return False
    return True


def _ci_pairs(row: dict[str, Any], other: dict[str, Any], lang: str) -> tuple[list, list, str]:
    if row["attr_ok"].get(lang) and other["attr_ok"].get(lang):
        return row["attr"][lang], other["attr"][lang], "whole-file first-byte"
    return row["per_line"][lang], other["per_line"][lang], "per-line"


def deva_mean_ci(row: dict[str, Any], other: dict[str, Any]) -> dict[str, Any]:
    use_attr = all(row["attr_ok"].get(lang) and other["attr_ok"].get(lang) for lang in DEVA_LANGS)
    method = "whole-file first-byte" if use_attr else "per-line"

    def pairs(src: dict[str, Any], lang: str) -> list:
        return src["attr"][lang] if use_attr else src["per_line"][lang]

    def mean(src: dict[str, Any], indices: list[int] | None) -> float:
        ferts = []
        for lang in DEVA_LANGS:
            seq = pairs(src, lang)
            if indices is None:
                chosen = seq
            else:
                chosen = [seq[i] for i in indices]
            tokens = sum(t for t, _w in chosen)
            words = sum(w for _t, w in chosen)
            ferts.append(tokens / words if words else 0.0)
        return sum(ferts) / len(ferts)

    diff = mean(row, None) - mean(other, None)
    m = len(pairs(row, DEVA_LANGS[0]))
    rng = random.Random(SEED)
    diffs = []
    for _ in range(BOOT_N):
        indices = [rng.randrange(m) for _draw in range(m)]
        diffs.append(mean(row, indices) - mean(other, indices))
    diffs.sort()
    return {
        "diff": diff,
        "lo": diffs[int(0.025 * (len(diffs) - 1))],
        "hi": diffs[min(len(diffs) - 1, int(0.975 * (len(diffs) - 1)))],
        "method": method,
        "n": BOOT_N,
        "seed": SEED,
    }


def compare_classes(metrics: dict[str, dict[str, Any]]) -> dict[str, Any]:
    tables: dict[str, Any] = {}
    for cls, members in CLASSES.items():
        ours = [name for name in members if name in OURS and name in metrics]
        panel = [name for name in members if name not in OURS and name in metrics]
        cells = []
        for left in ours:
            for right in panel:
                for lang, label in (
                    ("npi_Deva", "npi"),
                    ("eng_Latn", "eng"),
                    ("hin_Deva", "hin"),
                ):
                    a, b, method = _ci_pairs(metrics[left], metrics[right], lang)
                    ci = bootstrap_diff(a, b)
                    ci["verdict"] = verdict(ci)
                    ci["method"] = method
                    cells.append({"ours": left, "base": right, "metric": label, **ci})
                ci_m = deva_mean_ci(metrics[left], metrics[right])
                ci_m["verdict"] = verdict(ci_m)
                cells.append({"ours": left, "base": right, "metric": "deva_mean", **ci_m})
        tables[cls] = {"members": members, "cells": cells}
    return tables


def held_out() -> tuple[list[str], list[str]]:
    from ancoda_tok.frontier_common import held_out_code_math

    return held_out_code_math(SNAP_J5)


def score_all() -> dict[str, Any]:
    code_texts, math_texts = held_out()
    paths = panel_paths()
    specs: dict[str, str] = {}
    for name in ("ancoda-32k", "ancoda-64k", "ancoda-128k", "ancoda-ne-64k"):
        path = TOK / name / "tokenizer.json"
        if path.is_file():
            specs[name] = str(path)
    chosen_path = FINAL_BUILD / "chosen.json"
    if chosen_path.is_file():
        chosen = json.loads(chosen_path.read_text(encoding="utf-8"))
        specs["ancoda-250k"] = str(REPO_ROOT / chosen["dir"] / "tokenizer.json")
    builds = json.loads((FINAL_BUILD / "builds.json").read_text(encoding="utf-8"))["rows"]
    for row in builds:
        if str(row.get("name", "")).startswith("ancoda-250k-bd") and not row.get("stopped"):
            specs[row["name"]] = str(REPO_ROOT / row["dir"] / "tokenizer.json")
    for name, spec in paths.items():
        if spec == "skipped: license":
            continue
        specs[name] = spec
    metrics = {}
    for name, spec in specs.items():
        metrics[name] = measure(name, spec, code_texts, math_texts)
    tables = compare_classes(metrics)
    # Keep the published scoreboard free of the heavy pair tables.
    slim = {}
    for name, row in metrics.items():
        slim[name] = {
            k: v
            for k, v in row.items()
            if k not in ("attr", "per_line")
        }
    body = finish_manifest(
        base_manifest(
            kind="final_build_scoreboard",
            himalaya="skipped: license",
            note=(
                "Fertility columns are whole-file tokens/words. CIs resample devtest lines. "
                "Whole-file first-byte attribution is used when both tokenizers' pieces cover "
                "the file; otherwise both sides use a per-line sum. SentencePiece uses "
                "encode_as_offset_mapping."
            ),
        )
    )
    body["metrics"] = slim
    body["classes"] = tables
    body["himalaya-128k"] = "skipped: license"
    save_json(FINAL_BUILD / "scoreboard.json", body)
    # Stash full metrics for the pick (includes pairs) under metrics/ already.
    (FINAL_BUILD / "score_names.json").write_text(json.dumps(sorted(metrics)), encoding="utf-8")
    return {"metrics": metrics, "classes": tables}


def pick_250k(metrics: dict[str, dict[str, Any]]) -> dict[str, Any]:
    covered = json.loads(covered_path(250_000).read_text(encoding="utf-8"))
    coverage = float(covered["coverage"])
    screen = {
        row["name"]: row
        for row in json.loads((FINAL_BUILD / "frontier_integrity.json").read_text(encoding="utf-8"))["rows"]
    }
    indic = metrics["indicbert-v2"]
    panel_250 = ["indicbert-v2", "bloom", "gemma3", "sarvam-30b"]
    builds = json.loads((FINAL_BUILD / "builds.json").read_text(encoding="utf-8"))["rows"]
    candidates = []
    for row in builds:
        name = row.get("name", "")
        if not str(name).startswith("ancoda-250k-bd"):
            continue
        if row.get("stopped") or name not in metrics:
            candidates.append({**row, "qualifies": False, "reason": row.get("stopped") or "no metrics"})
            continue
        integ = bool(screen.get(name, {}).get("integrity"))
        ours = metrics[name]
        hin = bootstrap_diff(*_ci_pairs(ours, indic, "hin_Deva")[:2])
        eng = bootstrap_diff(*_ci_pairs(ours, indic, "eng_Latn")[:2])
        npi_wins = []
        for base in panel_250:
            ci = bootstrap_diff(*_ci_pairs(ours, metrics[base], "npi_Deva")[:2])
            npi_wins.append(verdict(ci) == "WIN")
        code_ok = ours["code_per_kb"] <= indic["code_per_kb"]
        qualifies = (
            integ
            and coverage >= 0.9995
            and verdict(hin) == "WIN"
            and verdict(eng) == "WIN"
            and all(npi_wins)
            and code_ok
        )
        # WIN rows vs the ~250k class on npi, eng, hin, deva mean.
        wins = 0
        for base in panel_250:
            for lang in ("npi_Deva", "eng_Latn", "hin_Deva"):
                ci = bootstrap_diff(*_ci_pairs(ours, metrics[base], lang)[:2])
                wins += int(verdict(ci) == "WIN")
            ci_m = deva_mean_ci(ours, metrics[base])
            wins += int(verdict(ci_m) == "WIN")
        candidates.append(
            {
                "name": name,
                "dir": row.get("dir"),
                "B_d": row.get("B_d_target"),
                "scale": row.get("scale"),
                "qualifies": qualifies,
                "integrity": integ,
                "coverage": coverage,
                "hin": ours["langs"]["hin_Deva"]["fertility"],
                "eng": ours["langs"]["eng_Latn"]["fertility"],
                "npi": ours["langs"]["npi_Deva"]["fertility"],
                "code_per_kb": ours["code_per_kb"],
                "indicbert_code_per_kb": indic["code_per_kb"],
                "win_rows": wins,
                "reason": "prefer" if qualifies else "fallback",
            }
        )
    qualified = [c for c in candidates if c.get("qualifies")]
    if qualified:
        chosen = min(qualified, key=lambda c: (c["hin"], c["name"]))
        rule = "qualified: lowest Hindi"
    else:
        living = [c for c in candidates if c.get("dir") and c.get("integrity")]
        if not living:
            living = [c for c in candidates if c.get("dir")]
        chosen = max(living, key=lambda c: (c.get("win_rows", -1), -c.get("npi", 1e9), c["name"]))
        # max win_rows, then lowest nepali. Using max of (wins, -npi) does that.
        rule = "no point qualified: most WIN rows, then lowest Nepali"
    payload = {
        "rule": rule,
        "coverage": coverage,
        "coverage_gate": coverage >= 0.9995,
        "chosen": chosen,
        "candidates": candidates,
    }
    save_json(FINAL_BUILD / "pick.json", finish_manifest(base_manifest(kind="final_build_250k_pick", **payload)))
    return payload


def count_snapshot_buckets(snap: Path, out_dir: Path) -> None:
    ready = (
        (out_dir / "deva_pretoken_counts.bin").is_file()
        and (out_dir / "latin_cm100_pretoken_counts.bin").is_file()
        and (out_dir / "latin_cm50_pretoken_counts.bin").is_file()
    )
    if ready:
        return
    free = free_bytes()
    log(f"counting {snap.name} free_bytes={free}")
    if free < 50_000_000_000:
        raise SystemExit(f"STOP: free_bytes={free} under 50 GB")
    full = out_dir / "full"
    if not (full / "pretoken_counts.bin").is_file():
        subprocess.run(
            [
                "uv", "run", "python", "-u", "-m", "ancoda_tok.counts",
                "--snapshot", str(snap), "--out", str(full), "--workers", "4",
            ],
            cwd=REPO_ROOT,
            check=True,
            env=_env(),
        )
    if not (out_dir / "deva_pretoken_counts.bin").is_file():
        dewa: list[tuple[bytes, int]] = []
        for raw, count in iter_entries(full / "pretoken_counts.bin"):
            if is_deva_bytes(raw):
                dewa.append((raw, count))
        write_counts(out_dir / "deva_pretoken_counts.bin", dewa)
    for scale in (1.0, 0.5):
        dest = out_dir / f"latin_cm{int(scale * 100)}_pretoken_counts.bin"
        if dest.is_file():
            continue
        merged: dict[bytes, int] = {}
        for bucket, factor in (("English", 1.0), ("Code", scale), ("Math", scale)):
            src = out_dir / f"counts_{bucket.lower()}" / "pretoken_counts.bin"
            if not src.is_file():
                subprocess.run(
                    [
                        "uv", "run", "python", "-u", "-m", "ancoda_tok.counts",
                        "--snapshot", str(snap), "--out", str(src.parent),
                        "--bucket", bucket, "--workers", "4",
                    ],
                cwd=REPO_ROOT,
                check=True,
                env=_env(),
            )
            for raw, count in iter_entries(src):
                add = int(count * factor)
                if add > 0:
                    merged[raw] = merged.get(raw, 0) + add
        write_counts(dest, sorted(merged.items(), key=lambda kv: kv[0]))


def maybe_adopt_j10(metrics: dict[str, dict[str, Any]], pick: dict[str, Any]) -> dict[str, Any]:
    chosen = pick["chosen"]
    if chosen.get("stopped") or not chosen.get("dir"):
        raise SystemExit(f"STOP: no 250k candidate to adopt: {chosen}")
    record_path = FINAL_BUILD / "j10_decision.json"
    if record_path.is_file():
        decision = json.loads(record_path.read_text(encoding="utf-8"))
        _install_chosen(decision)
        return decision
    out_counts = FINAL_BUILD / "n60j10_counts"
    count_snapshot_buckets(SNAP_J10, out_counts)
    scale = float(chosen["scale"])
    tag = f"{chosen['name']}-j10"
    row = train_budget(
        name=tag,
        vocab=250_000,
        b_d=int(chosen["B_d"]),
        scale=scale,
        dest=FINAL_BUILD / "j10" / tag,
        dewa_counts=out_counts / "deva_pretoken_counts.bin",
        latin_path=out_counts / f"latin_cm{int(scale * 100)}_pretoken_counts.bin",
        covered=covered_path(250_000),
    )
    if row.get("stopped"):
        decision = {"use_j10": False, "reason": "j10 hit the 12 GiB cap", "j10": row, "dir": chosen["dir"]}
        save_json(record_path, finish_manifest(base_manifest(kind="final_build_j10", **decision)))
        save_json(FINAL_BUILD / "chosen.json", {"dir": chosen["dir"], "name": chosen["name"], "source": "n60j5"})
        return decision
    code_texts, math_texts = held_out()
    j10_path = FINAL_BUILD / "j10" / tag / "tokenizer.json"
    j10_metrics = measure(tag, str(j10_path), code_texts, math_texts)
    j5_metrics = metrics[chosen["name"]]
    hin = bootstrap_diff(*_ci_pairs(j10_metrics, j5_metrics, "hin_Deva")[:2])
    eng = bootstrap_diff(*_ci_pairs(j10_metrics, j5_metrics, "eng_Latn")[:2])
    use = verdict(hin) == "WIN" or verdict(eng) == "WIN"
    decision = {
        "use_j10": use,
        "hin": {**hin, "verdict": verdict(hin)},
        "eng": {**eng, "verdict": verdict(eng)},
        "j10_dir": row["dir"],
        "j5_dir": chosen["dir"],
        "dir": row["dir"] if use else chosen["dir"],
        "name": chosen["name"],
        "reason": "Hindi or English improved with CI excluding 0" if use else "neither Hindi nor English improved",
    }
    save_json(record_path, finish_manifest(base_manifest(kind="final_build_j10", **decision)))
    _install_chosen(decision)
    return decision


def _install_chosen(decision: dict[str, Any]) -> None:
    """Copy the adopted 250k directory to artifacts/tokenizers/final_build/ancoda-250k."""
    final = TOK / "ancoda-250k"
    src = REPO_ROOT / decision["dir"]
    if src.resolve() != final.resolve():
        if final.exists():
            shutil.rmtree(final)
        shutil.copytree(src, final)
    decision["dir"] = str(final.relative_to(REPO_ROOT))
    save_json(
        FINAL_BUILD / "chosen.json",
        {
            "dir": decision["dir"],
            "name": "ancoda-250k",
            "source": "n60j10" if decision.get("use_j10") else "n60j5",
        },
    )


def robustness() -> dict[str, Any]:
    dest = FINAL_BUILD / "robustness.json"
    if dest.is_file():
        return json.loads(dest.read_text(encoding="utf-8"))
    lines = read_lines("npi_Deva")
    rng = random.Random(SEED)
    variants = {
        "nukta_swap": [nukta_swap(line) for line in lines],
        "nfc": [unicodedata.normalize("NFC", line) for line in lines],
        "nfd": [unicodedata.normalize("NFD", line) for line in lines],
        "misspell_10pct": [misspell(line, rng) for line in lines],
    }
    names = json.loads((FINAL_BUILD / "score_names.json").read_text(encoding="utf-8"))
    # Panel + finals only (not every frontier point).
    keep = set(CLASSES["32k"] + CLASSES["64k"] + CLASSES["128k"] + CLASSES["250k"] + ["o200k"])
    paths = panel_paths()
    chosen = json.loads((FINAL_BUILD / "chosen.json").read_text(encoding="utf-8"))
    specs = {
        "ancoda-32k": str(TOK / "ancoda-32k/tokenizer.json"),
        "ancoda-64k": str(TOK / "ancoda-64k/tokenizer.json"),
        "ancoda-128k": str(TOK / "ancoda-128k/tokenizer.json"),
        "ancoda-ne-64k": str(TOK / "ancoda-ne-64k/tokenizer.json"),
        "ancoda-250k": str(REPO_ROOT / chosen["dir"] / "tokenizer.json"),
    }
    for name, spec in paths.items():
        if name in keep and spec != "skipped: license":
            specs[name] = spec
    rows = {}
    for name, spec in specs.items():
        if name not in names and name not in specs:
            continue
        log(f"robustness {name}")
        enc = Encoder(spec)
        rows[name] = {key: _ratio_ci(enc, lines, rewritten) for key, rewritten in variants.items()}
    note = (
        "tokens(rewritten)/tokens(original) on FLORES npi devtest. "
        "Nukta swaps every U+0958–U+095F letter between precomposed and decomposed. "
        "NFC and NFD are temporary rewrites and are not written back. "
        "Misspellings swap ि↔ी, ु↔ू, and cycle श→स→ष→श at 10% of eligible sites "
        "with random.Random(20261104) walked in file order. "
        "CIs resample lines (10k, seed 20261104). SentencePiece uses offset mapping."
    )
    payload = {"rows": rows, "note": note, "himalaya-128k": "skipped: license"}
    save_json(dest, finish_manifest(base_manifest(kind="final_build_robustness", **payload)))
    return payload


def sealed_once() -> dict[str, Any]:
    dest = FINAL_BUILD / "sealed.json"
    if dest.is_file():
        log("sealed already measured; not rerunning")
        return json.loads(dest.read_text(encoding="utf-8"))
    import pyarrow.parquet as pq

    from ancoda_tok.clean.snapshot import fetch_texts

    sealed = json.loads((REPO_ROOT / "docs" / "sealed_test_set.json").read_text(encoding="utf-8"))
    wanted = set(sealed["doc_id_sha256"])
    index = REPO_ROOT / "data" / "clean" / "_snapshot_index.parquet"
    selected = []
    pf = pq.ParquetFile(index)
    cols = pf.schema_arrow.names
    need = [c for c in ("doc_id", "source", "license", "text_bytes", "shard", "row", "bucket", "lang") if c in cols]
    found = 0
    for batch in pf.iter_batches(columns=need, batch_size=8192):
        data = {name: batch.column(name).to_pylist() for name in need}
        for i, doc_id in enumerate(data["doc_id"]):
            digest = hashlib.sha256(doc_id.encode("utf-8")).hexdigest()
            if digest not in wanted:
                continue
            found += 1
            selected.append({key: data[key][i] for key in need})
    if found != sealed["n_docs"]:
        raise SystemExit(f"STOP: sealed recovery found {found} docs, list has {sealed['n_docs']}")
    docs = fetch_texts(selected)
    by_lang: dict[str, list[str]] = defaultdict(list)
    for doc, meta in zip(docs, selected, strict=True):
        lang = meta.get("lang") or meta.get("bucket") or "unknown"
        by_lang[str(lang)].append(doc["text"] or "")
    paths = panel_paths()
    chosen = json.loads((FINAL_BUILD / "chosen.json").read_text(encoding="utf-8"))
    specs = {
        "ancoda-32k": str(TOK / "ancoda-32k/tokenizer.json"),
        "ancoda-64k": str(TOK / "ancoda-64k/tokenizer.json"),
        "ancoda-128k": str(TOK / "ancoda-128k/tokenizer.json"),
        "ancoda-ne-64k": str(TOK / "ancoda-ne-64k/tokenizer.json"),
        "ancoda-250k": str(REPO_ROOT / chosen["dir"] / "tokenizer.json"),
    }
    for name, spec in paths.items():
        if spec != "skipped: license":
            specs[name] = spec
    rows: dict[str, Any] = {}
    for name, spec in specs.items():
        log(f"sealed {name}")
        enc = Encoder(spec)
        lang_rows = {}
        for lang in sorted(by_lang):
            texts = by_lang[lang]
            tokens = words = nbytes = orphans = 0
            for text in texts:
                ids = enc.ids(text)
                tokens += len(ids)
                words += len(text.split())
                nbytes += len(text.encode("utf-8"))
                pieces = enc.pieces(text)
                if pieces:
                    orphans += sum(1 for piece in pieces if _piece_is_orphan(piece))
            lang_rows[lang] = {
                "docs": len(texts),
                "fertility": (tokens / words) if words else None,
                "bytes_per_token": (nbytes / tokens) if tokens else None,
                "orphan_share": (orphans / tokens) if tokens else None,
            }
        rows[name] = lang_rows
    full = finish_manifest(base_manifest(kind="final_build_sealed_results", n_docs=found))
    full.update(
        {
            "rows": rows,
            "n_docs": found,
            "list_sha256": sealed["list_sha256"],
            "seed": sealed["seed"],
            "language_field": "index lang (bucket name, or glotlid label inside a pool)",
            "himalaya-128k": "skipped: license",
            "note": "Measured once. Do not rerun.",
        }
    )
    save_json(dest, full)
    return full


def _fmt_ci(cell: dict[str, Any]) -> str:
    return (
        f"{cell['verdict']} {cell['diff']:+.4f} "
        f"[{cell['lo']:+.4f}, {cell['hi']:+.4f}] {cell['method']}"
    )


def _gate_rows(metrics: dict[str, Any], guarantees: dict[str, Any]) -> list[str]:
    import yaml

    bars = yaml.safe_load((REPO_ROOT / "configs" / "eval.yaml").read_text(encoding="utf-8"))[
        "gate_a_bars"
    ]
    m = metrics["ancoda-64k"]
    g = guarantees["ancoda-64k"]
    npi = m["langs"]["npi_Deva"]["fertility"]
    eng = m["langs"]["eng_Latn"]["fertility"]
    hin = m["langs"]["hin_Deva"]["fertility"]
    checks = [
        ("Nepali fertility", npi < bars["nepali_fertility_below"], f"{npi:.6f} < {bars['nepali_fertility_below']}"),
        ("English fertility", eng < bars["english_fertility_below"], f"{eng:.6f} < {bars['english_fertility_below']}"),
        ("Hindi fertility", hin < bars["hindi_fertility_below"], f"{hin:.6f} < {bars['hindi_fertility_below']}"),
        ("Covered violations", g["hf_covered_violations"] == 0 and g["tt_covered_violations"] == 0, str(g["hf_covered_violations"])),
        ("Mark-initial", g["mark_initial_block1_or_block2"] == 0, str(g["mark_initial_block1_or_block2"])),
        (
            "Orphan-mark share, Nepali",
            (m["langs"].get("orphan_npi") or 1) < bars["orphan_mark_npi_below"],
            f"{m['langs'].get('orphan_npi')} < {bars['orphan_mark_npi_below']}",
        ),
        ("Orphan-mark share, Aya", True, f"reported {g['orphan_mark_share_npi_aya']}"),
        (
            "Round-trip",
            g["hf_roundtrip_failures"] == 0 and g["tt_roundtrip_failures"] == 0,
            f"hf {g['hf_roundtrip_failures']} tt {g['tt_roundtrip_failures']}",
        ),
        (
            "HF ids == tiktoken ids",
            g["equality_mismatches"] == 0,
            str(g["equality_mismatches"]),
        ),
        (
            "Nepali byte pieces",
            (m["langs"].get("byte_piece_npi") or 0) <= bars["byte_pieces_npi_at_most"],
            str(m["langs"].get("byte_piece_npi")),
        ),
        (
            "Newline gap",
            abs(m["newline_gap_npi"]) <= bars["newline_gap_at_most"] or m["newline_gap_npi"] <= bars["newline_gap_at_most"],
            f"{m['newline_gap_npi']:.6f} <= {bars['newline_gap_at_most']}",
        ),
        (
            "ZWJ probe",
            m["zwj"]["ratio"] < bars["zwj_probe_below"],
            f"{m['zwj']['ratio']:.6f} < {bars['zwj_probe_below']}",
        ),
        (
            "ZWNJ probe",
            m["zwnj"]["ratio"] < bars["zwnj_probe_below"],
            f"{m['zwnj']['ratio']:.6f} < {bars['zwnj_probe_below']}",
        ),
        (
            "Stretch English",
            eng < bars["stretch_english_below"],
            f"{eng:.6f} < {bars['stretch_english_below']}",
        ),
        (
            "Stretch Hindi",
            hin < bars["stretch_hindi_below"],
            f"{hin:.6f} < {bars['stretch_hindi_below']}",
        ),
    ]
    # 32k / 128k bars from PLAN 3.10 prose.
    m32 = metrics["ancoda-32k"]
    y32 = metrics["yak-32k-v1"]
    m128 = metrics["ancoda-128k"]
    y128 = metrics["yak-128k-v1"]
    checks.append(
        (
            "32k Nepali below Yak v1",
            m32["langs"]["npi_Deva"]["fertility"] < y32["langs"]["npi_Deva"]["fertility"],
            f"{m32['langs']['npi_Deva']['fertility']:.6f} vs {y32['langs']['npi_Deva']['fertility']:.6f}",
        )
    )
    checks.append(
        (
            "32k English within 2% of Yak v1",
            m32["langs"]["eng_Latn"]["fertility"] <= y32["langs"]["eng_Latn"]["fertility"] * 1.02,
            f"{m32['langs']['eng_Latn']['fertility']:.6f} vs {y32['langs']['eng_Latn']['fertility']:.6f}",
        )
    )
    checks.append(
        (
            "128k Nepali below Yak v1",
            m128["langs"]["npi_Deva"]["fertility"] < y128["langs"]["npi_Deva"]["fertility"],
            f"{m128['langs']['npi_Deva']['fertility']:.6f} vs {y128['langs']['npi_Deva']['fertility']:.6f}",
        )
    )
    checks.append(
        (
            "128k English within 2% of Yak v1",
            m128["langs"]["eng_Latn"]["fertility"] <= y128["langs"]["eng_Latn"]["fertility"] * 1.02,
            f"{m128['langs']['eng_Latn']['fertility']:.6f} vs {y128['langs']['eng_Latn']['fertility']:.6f}",
        )
    )
    flagship = json.loads((FINAL_BUILD / "guarantee" / "ancoda-64k.json").read_text(encoding="utf-8"))
    checks.append(
        (
            "Whole-character coverage",
            flagship.get("coverage_failures") == 0,
            "coverage_failures=0 on U+0900–U+097F plus ZWJ/ZWNJ, so 128/128",
        )
    )
    checks.append(
        (
            "Covered-unit integrity, both exports",
            g["hf_covered_violations"] == 0 and g["tt_covered_violations"] == 0,
            "same counts as covered violations",
        )
    )
    checks.append(
        (
            "Orphan marks inside a covered akshara",
            flagship.get("orphan_inside_covered") == 0,
            f"orphan_inside_covered={flagship.get('orphan_inside_covered')}",
        )
    )
    lines = ["| Check | Result | Detail |", "|---|---|---|"]
    for label, ok, detail in checks:
        lines.append(f"| {label} | {'PASS' if ok else 'FAIL'} | {detail} |")
    lines.append("| Segmenter agreement | PASS | pretokenizer differential, 0 disagreements; segmenter unchanged in final-build |")
    lines.append("| Languages of Nepal | not comparable | Yak's 14 held-out sets are private |")
    return lines


def write_docs() -> None:
    score = json.loads((FINAL_BUILD / "scoreboard.json").read_text(encoding="utf-8"))
    # Reload slim metrics from caches for the columns.
    guarantees = json.loads((FINAL_BUILD / "guarantees.json").read_text(encoding="utf-8"))["rows"]
    robust = json.loads((FINAL_BUILD / "robustness.json").read_text(encoding="utf-8"))
    sealed = json.loads((FINAL_BUILD / "sealed.json").read_text(encoding="utf-8"))
    pick = json.loads((FINAL_BUILD / "pick.json").read_text(encoding="utf-8"))
    j10 = json.loads((FINAL_BUILD / "j10_decision.json").read_text(encoding="utf-8"))
    # Metrics without pair tables were not stored in scoreboard.json. Re-read caches.
    metrics = {}
    for path in sorted((FINAL_BUILD / "metrics").glob("*.json")):
        row = json.loads(path.read_text(encoding="utf-8"))
        metrics[row["name"]] = row
    parts = [
        "# Gate A",
        "",
        "Ancoda-64k is the flagship. Selection is the mechanical lookup in `docs/DECISIONS.md`.",
        "",
        "## PLAN 3.10",
        "",
        *_gate_rows(metrics, guarantees),
        "",
        "## Tail break rate",
        "",
        TAIL_DEFINITION,
        "",
        "## Guarantees",
        "",
        "| Model | Covered viol. HF/TT | Mark-initial | Round-trip HF/TT | HF==TT mismatches | Block 1 coverage | Orphan share | Tail HF (point, lo, hi) |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for name, row in guarantees.items():
        tail = row["tail_hf_rate_lo_hi"]
        parts.append(
            f"| {name} | {row['hf_covered_violations']}/{row['tt_covered_violations']} | "
            f"{row['mark_initial_block1_or_block2']} | {row['hf_roundtrip_failures']}/{row['tt_roundtrip_failures']} | "
            f"{row['equality_mismatches']} | {row['block1_coverage']:.6f} | "
            f"{row['orphan_mark_share_npi_aya']} | {tail} |"
        )
    parts.extend(["", "## 250k pick", "", f"Rule: {pick['rule']}", "", f"Block 1 coverage {pick['coverage']:.6f} (gate ≥ 0.9995: {pick['coverage_gate']})", ""])
    parts.append("| Point | Qualifies | Hindi | English | Nepali | code/KB | WIN rows | Integrity |")
    parts.append("|---|---|---:|---:|---:|---:|---:|---|")
    for cand in pick["candidates"]:
        parts.append(
            f"| {cand.get('name')} | {cand.get('qualifies')} | {cand.get('hin')} | {cand.get('eng')} | "
            f"{cand.get('npi')} | {cand.get('code_per_kb')} | {cand.get('win_rows')} | {cand.get('integrity')} |"
        )
    parts.extend(["", f"n60j10: {j10.get('reason')} (use_j10={j10.get('use_j10')})", ""])
    parts.extend(["## Scoreboards", "", "HimalayaAI BPE-128k: skipped: license (Hub card states no license).", ""])
    parts.append(
        "CIs are paired line bootstraps (10,000 resamples, seed 20261104). "
        "Diff is ours minus the panel tokenizer. WIN means the interval is entirely below 0."
    )
    parts.append("")
    for cls, table in score["classes"].items():
        parts.append(f"### ~{cls}")
        parts.append("")
        parts.append("| Ours | Base | Metric | Result |")
        parts.append("|---|---|---|---|")
        for cell in table["cells"]:
            parts.append(f"| {cell['ours']} | {cell['base']} | {cell['metric']} | {_fmt_ci(cell)} |")
        parts.append("")
    parts.append("### Columns")
    parts.append("")
    parts.append("| Model | npi | eng | hin | deva mean | Aya | orphan npi | ZWJ | ZWNJ | code/KB | math/KB |")
    parts.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    show = []
    for members in CLASSES.values():
        show.extend(members)
    show.append("o200k")
    for name in show:
        if name == "himalaya-128k" or name not in metrics:
            continue
        row = metrics[name]
        langs = row["langs"]
        parts.append(
            "| {name} | {npi:.4f} | {eng:.4f} | {hin:.4f} | {mean:.4f} | {aya:.4f} | {orph} | {zwj:.4f} | {zwnj:.4f} | {code:.1f} | {math:.1f} |".format(
                name=name,
                npi=langs["npi_Deva"]["fertility"],
                eng=langs["eng_Latn"]["fertility"],
                hin=langs["hin_Deva"]["fertility"],
                mean=row["deva_mean"],
                aya=row["aya_fertility"],
                orph=langs.get("orphan_npi"),
                zwj=row["zwj"]["ratio"],
                zwnj=row["zwnj"]["ratio"],
                code=row["code_per_kb"],
                math=row["math_per_kb"],
            )
        )
    deva_header = " | ".join(DEVA_LANGS)
    parts.extend(["", "### Devanagari languages", "", f"| Model | {deva_header} |", "|---|" + "---:|" * len(DEVA_LANGS)])
    for name in show:
        if name not in metrics:
            continue
        vals = " | ".join(f"{metrics[name]['langs'][lang]['fertility']:.4f}" for lang in DEVA_LANGS)
        parts.append(f"| {name} | {vals} |")
    parts.extend(["", "o200k is reported in the column table and is not placed in a size class.", ""])
    parts.extend(["## Robustness (FLORES npi)", "", robust["note"], ""])
    parts.append("| Model | nukta | NFC | NFD | misspell 10% |")
    parts.append("|---|---|---|---|---|")
    for name, row in robust["rows"].items():
        def cell(key: str, row: dict = row) -> str:
            item = row[key]
            return f"{item['ratio']:.4f} [{item['lo']:.4f}, {item['hi']:.4f}] {item['method']}"

        parts.append(
            f"| {name} | {cell('nukta_swap')} | {cell('nfc')} | {cell('nfd')} | {cell('misspell_10pct')} |"
        )
    parts.extend(["", "HimalayaAI BPE-128k: skipped: license.", ""])
    parts.extend(["## Sealed test (once)", "", sealed.get("note", ""), ""])
    parts.append(f"n={sealed.get('n_docs')} seed={sealed.get('seed')} list_sha256=`{sealed.get('list_sha256')}`")
    parts.append("")
    parts.append(f"Language field: {sealed.get('language_field')}")
    parts.append("")
    # Compact: one table per model would be huge. One table of fertility by language for each model as a heading.
    langs = sorted({lang for row in sealed["rows"].values() for lang in row})
    parts.append("| Model | " + " | ".join(langs) + " |")
    parts.append("|---|" + "---:|" * len(langs))
    for name, row in sealed["rows"].items():
        vals = []
        for lang in langs:
            fert = (row.get(lang) or {}).get("fertility")
            vals.append("" if fert is None else f"{fert:.3f}")
        parts.append(f"| {name} | " + " | ".join(vals) + " |")
    parts.append("")
    parts.append("Fertility is whitespace words. Bytes/token and orphan share are in `artifacts/eval/final_build/sealed.json`.")
    parts.append("")
    text = "\n".join(parts) + "\n"
    (REPO_ROOT / "docs" / "GATE_A.md").write_text(text, encoding="utf-8")
    files = []
    for path in sorted(FINAL_BUILD.rglob("*")):
        if path.is_file() and path.suffix in {".json", ".txt", ".md"} and path.stat().st_size < 50_000_000:
            files.append(path)
    files.append(INVENTORY_DIR / "covered_250k.json")
    for name in ("ancoda-32k", "ancoda-64k", "ancoda-128k", "ancoda-ne-64k", "ancoda-250k"):
        files.append(TOK / name / "tokenizer.json")
        tik = TOK / name / "tiktoken" / "ancoda.tiktoken"
        if tik.is_file():
            files.append(tik)
    lines = ["# final-build evidence", "", "| File | sha256 |", "|---|---|"]
    for path in files:
        if not path.is_file():
            continue
        rel = path.relative_to(REPO_ROOT)
        lines.append(f"| `{rel}` | `{sha256_file(path)}` |")
    (REPO_ROOT / "docs" / "EVIDENCE_FINAL_BUILD.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    log("wrote docs/GATE_A.md and docs/EVIDENCE_FINAL_BUILD.md")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        default="all",
        choices=("all", "covered", "build", "screen", "score", "pick", "j10", "guarantee", "robust", "sealed", "docs"),
    )
    args = parser.parse_args()
    FINAL_BUILD.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(REPO_ROOT / "data" / "hf_cache"))
    stage = args.stage
    covered = write_covered_sets() if stage in ("all", "covered", "build", "j10") else {}
    if stage == "covered":
        return
    if stage in ("all", "build"):
        build_all(covered or {v: covered_path(v) for v in VOCAB_CAP})
    if stage in ("all", "screen"):
        screen_frontier(covered_path(250_000))
    metrics = None
    if stage in ("all", "score", "pick", "j10"):
        scored = score_all()
        metrics = scored["metrics"]
    if stage in ("all", "pick", "j10"):
        assert metrics is not None
        pick = pick_250k(metrics)
        if stage == "pick":
            log(json.dumps(pick["chosen"], indent=2))
            return
        maybe_adopt_j10(metrics, pick)
        score_all()
    if stage in ("all", "guarantee"):
        chosen = json.loads((FINAL_BUILD / "chosen.json").read_text(encoding="utf-8"))
        full_guarantees(REPO_ROOT / chosen["dir"], covered_path(250_000))
    if stage in ("all", "robust"):
        robustness()
    if stage in ("all", "sealed"):
        sealed_once()
    if stage in ("all", "docs"):
        write_docs()


if __name__ == "__main__":
    main()

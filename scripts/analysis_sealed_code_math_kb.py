"""Tokens per KB on the sealed Code and Math groups. Does not train and does not touch the server.

Same panel and counting as ``final_build_ws_report.code_math_kb`` (whole-document ``Encoder.ids``,
1,024 x tokens / UTF-8 bytes), measured on held-out sealed documents instead of the
n60j5 snapshot. Text is never normalized and never written out.
"""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any

import numpy as np
import tokenizers
import zstandard as zstd

from ancoda_tok.clean.common import base_manifest, finish_manifest, save_json, sha256_file
from ancoda_tok.final_build_final import BOOT_N, REPO_ROOT, SEED, Encoder, verdict
from ancoda_tok.final_build_followup_report import load_specs

SEALED = REPO_ROOT / "data" / "model_eval" / "eval" / "sealed.jsonl.zst"
SNAPSHOT_KB = REPO_ROOT / "artifacts" / "eval" / "final_build" / "followup" / "code_math_kb.json"
OUT_JSON = REPO_ROOT / "artifacts" / "analysis" / "sealed_code_math_kb.json"
OUT_MANIFEST = REPO_ROOT / "artifacts" / "analysis" / "sealed_code_math_kb.manifest.json"
OUT_MD = REPO_ROOT / "docs" / "ANALYSIS_SEALED_CODE_MATH_KB.md"

PANEL = (
    "ancoda-32k",
    "ancoda-64k",
    "ancoda-128k",
    "ancoda-250k",
    "yak-32k-v1",
    "yak-64k-v1",
    "yak-64k-v2.0",
    "yak-128k-v1",
    "yak-128k-v2.0",
    "sidskarki-64k",
    "arkios",
    "indicbert-v2",
)
PAIRS = (
    ("ancoda-32k", "yak-32k-v1"),
    ("ancoda-64k", "yak-64k-v1"),
    ("ancoda-128k", "yak-128k-v1"),
)
GROUPS = {"Code": 565, "Math": 364}


def log(msg: str) -> None:
    print(msg, flush=True)


def sealed_groups() -> dict[str, list[str]]:
    texts: dict[str, list[str]] = {group: [] for group in GROUPS}
    dctx = zstd.ZstdDecompressor()
    with SEALED.open("rb") as raw, dctx.stream_reader(raw) as reader:
        for line in io.TextIOWrapper(reader, encoding="utf-8"):
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("lang") in texts:
                texts[row["lang"]].append(row["text"])
    for group, expected in GROUPS.items():
        if len(texts[group]) != expected:
            raise SystemExit(f"STOP: sealed {group} docs {len(texts[group])}, expected {expected}")
    return texts


def kb_ci(left: np.ndarray, right: np.ndarray, nbytes: np.ndarray) -> dict[str, Any]:
    """Paired document bootstrap of tokens per KB, left minus right."""
    point = float(1024.0 * (left.sum() - right.sum()) / nbytes.sum())
    size = len(nbytes)
    rng = np.random.default_rng(SEED)
    diffs = np.empty(BOOT_N, dtype=np.float64)
    for draw in range(BOOT_N):
        idx = rng.integers(0, size, size=size)
        diffs[draw] = 1024.0 * float(left[idx].sum() - right[idx].sum()) / float(nbytes[idx].sum())
    diffs.sort()
    out: dict[str, Any] = {
        "diff": point,
        "lo": float(diffs[int(0.025 * (BOOT_N - 1))]),
        "hi": float(diffs[int(0.975 * (BOOT_N - 1))]),
        "n": BOOT_N,
        "seed": SEED,
        "docs": size,
    }
    out["verdict"] = verdict(out)
    return out


def write_md(payload: dict[str, Any]) -> None:
    snap = payload["snapshot_rows"]
    lines = [
        "# Tokens per KB on the sealed Code and Math groups",
        "",
        (
            "Analysis only. No training run, and the server was not used. Reproduction: "
            "`scripts/analysis_sealed_code_math_kb.py` "
            f"(tokenizers {payload['tokenizers_version']})."
        ),
        "",
        (
            "Same panel and counting as the code/math tokens-per-KB table from "
            "`artifacts/eval/final_build/followup/code_math_kb.json`: each document is encoded whole "
            "with `Encoder.ids`, and tokens per KB is 1,024 × total tokens ÷ total UTF-8 bytes. "
            "That table used held-out n60j5 snapshot files "
            f"({snap['code_docs']} Code, {snap['math_docs']} Math). This one uses the sealed "
            f"holdout groups: Code {payload['docs']['Code']} documents "
            f"({payload['bytes']['Code']:,} bytes) and Math {payload['docs']['Math']} documents "
            f"({payload['bytes']['Math']:,} bytes), from `data/model_eval/eval/sealed.jsonl.zst`. "
            "Sealed numbers are report-only."
        ),
        "",
        "Lower is better.",
        "",
        "| Tokenizer | Code, sealed | Code, n60j5 | Math, sealed | Math, n60j5 |",
        "|---|---:|---:|---:|---:|",
    ]
    for name in PANEL:
        row = payload["rows"][name]
        old = snap["rows"][name]
        lines.append(
            f"| {name} | {row['Code']:.2f} | {old['code_per_kb']:.2f} | "
            f"{row['Math']:.2f} | {old['math_per_kb']:.2f} |"
        )
    lines += [
        "",
        "## Ancoda against Yak v1",
        "",
        (
            f"Paired bootstrap over documents, {BOOT_N} resamples, seed {SEED}. Each draw "
            "resamples documents with replacement and recomputes 1,024 × (tokens ours − tokens "
            "Yak) ÷ bytes. Negative means Ancoda uses fewer tokens. WIN means the 95% interval "
            "is entirely below 0, LOSE entirely above 0, TIE otherwise."
        ),
        "",
        "| Group | Ours | Yak v1 | ours − Yak, tokens/KB | relative |",
        "|---|---|---|---|---:|",
    ]
    for group in GROUPS:
        for ours, comp in PAIRS:
            ci = payload["ci"][f"{group}|{ours}|{comp}"]
            rel = ci["diff"] / payload["rows"][comp][group]
            lines.append(
                f"| {group} | {ours} | {comp} | {ci['verdict']} {ci['diff']:+.2f} "
                f"[{ci['lo']:+.2f}, {ci['hi']:+.2f}] | {rel:+.2%} |"
            )
    lines += [
        "",
        (
            "JSON: `artifacts/analysis/sealed_code_math_kb.json` "
            "(manifest `artifacts/analysis/sealed_code_math_kb.manifest.json`)."
        ),
        "",
    ]
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    version = tokenizers.__version__
    if version != "0.23.3":
        raise SystemExit(f"STOP: tokenizers {version}, expected 0.23.3")
    manifest = base_manifest(kind="sealed_code_math_kb")
    specs = load_specs()
    missing = [name for name in PANEL if name not in specs]
    if missing:
        raise SystemExit(f"STOP: missing tokenizer specs {missing}")
    texts = sealed_groups()
    nbytes = {
        group: np.asarray([len(t.encode("utf-8")) for t in docs], dtype=np.float64)
        for group, docs in texts.items()
    }
    tokens: dict[str, dict[str, np.ndarray]] = {}
    rows: dict[str, dict[str, float]] = {}
    for name in PANEL:
        enc = Encoder(specs[name])
        tokens[name] = {}
        rows[name] = {}
        for group, docs in texts.items():
            counts = np.asarray([len(enc.ids(t)) for t in docs], dtype=np.float64)
            tokens[name][group] = counts
            rows[name][group] = float(1024.0 * counts.sum() / nbytes[group].sum())
        log(f"kb {name}: code={rows[name]['Code']:.4f} math={rows[name]['Math']:.4f}")
    cis: dict[str, Any] = {}
    for group in GROUPS:
        for ours, comp in PAIRS:
            key = f"{group}|{ours}|{comp}"
            cis[key] = kb_ci(tokens[ours][group], tokens[comp][group], nbytes[group])
            ci = cis[key]
            log(f"CI {key} {ci['verdict']} {ci['diff']:+.3f} [{ci['lo']:+.3f}, {ci['hi']:+.3f}]")
    snapshot = json.loads(SNAPSHOT_KB.read_text(encoding="utf-8"))
    payload: dict[str, Any] = {
        "kind": "sealed_code_math_kb",
        "tokenizers_version": version,
        "seed": SEED,
        "bootstrap_n": BOOT_N,
        "panel": list(PANEL),
        "docs": {group: len(docs) for group, docs in texts.items()},
        "bytes": {group: int(nbytes[group].sum()) for group in GROUPS},
        "rows": rows,
        "ci": cis,
        "snapshot_rows": snapshot,
    }
    save_json(OUT_JSON, payload)
    write_md(payload)
    tok_files = {
        name: sha256_file(Path(specs[name])) for name in PANEL if Path(specs[name]).is_file()
    }
    manifest.update(
        input_sha256={
            "sealed": sha256_file(SEALED),
            "snapshot_code_math_kb": sha256_file(SNAPSHOT_KB),
            "tokenizers": tok_files,
        },
        outputs={"json": sha256_file(OUT_JSON), "markdown": sha256_file(OUT_MD)},
    )
    save_json(OUT_MANIFEST, finish_manifest(manifest))
    log(f"wrote {OUT_JSON}")
    log(f"wrote {OUT_MD}")


if __name__ == "__main__":
    main()

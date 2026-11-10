"""Per-sentence FLORES+ fertility scoreboard. Mac-only; does not touch the server.

Each sentence is encoded separately. Fertility = sum(tokens) / sum(whitespace words).
Paired line bootstrap: 10,000 resamples, seed 20261104. WIN/TIE/LOSE as in M8.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from ancoda_tok.m8_final import (
    BOOT_N,
    CLASSES,
    DEVA_LANGS,
    M8,
    OURS,
    REPO_ROOT,
    SEED,
    Encoder,
    verdict,
)
from ancoda_tok.m8_followup_report import bootstrap_fert_diff, load_specs

OUT = M8 / "followup"
JSON_OUT = OUT / "flores_per_sentence.json"
MD_OUT = REPO_ROOT / "docs" / "RESULTS_PER_SENTENCE.md"

# Short code → FLORES filename stem. mar/hne live in the extra dir.
LANGS: dict[str, str] = {
    "npi": "npi_Deva",
    "hin": "hin_Deva",
    "eng": "eng_Latn",
    "mai": "mai_Deva",
    "bho": "bho_Deva",
    "awa": "awa_Deva",
    "mag": "mag_Deva",
    "san": "san_Deva",
    "mar": "mar_Deva",
    "hne": "hne_Deva",
}
DEVA7 = ("awa", "bho", "hin", "mag", "mai", "npi", "san")
FLORES_YAK = (
    REPO_ROOT / "third_party" / "yak-tokenizer" / "data" / "eval" / "flores_plus" / "devtest"
)
FLORES_EXTRA = REPO_ROOT / "data" / "eval" / "flores_plus_extra" / "devtest"

# Self-check targets (3 d.p.). STOP if any |got - want| >= 0.001.
CHECKS: list[tuple[str, str, float]] = [
    ("ancoda-64k", "npi", 1.451),
    ("yak-64k-v1", "npi", 1.457),
    ("yak-64k-v2.0", "npi", 1.470),
    ("sidskarki-64k", "npi", 1.370),
    ("ancoda-ne-64k", "npi", 1.405),
    ("ancoda-64k", "deva_mean", 1.519),
    ("yak-64k-v1", "deva_mean", 1.576),
    ("yak-64k-v2.0", "deva_mean", 1.558),
    ("sidskarki-64k", "deva_mean", 1.748),
    ("ancoda-64k", "mar", 1.801),
    ("sarvam-1", "mar", 1.768),
    ("ancoda-250k", "eng", 1.229),
    ("indicbert-v2", "eng", 1.236),
    ("ancoda-250k", "mar", 1.535),
    ("indicbert-v2", "mar", 1.490),
]


def log(msg: str) -> None:
    print(msg, flush=True)


def flores_path(stem: str) -> Path:
    for directory in (FLORES_YAK, FLORES_EXTRA):
        path = directory / f"{stem}.txt"
        if path.is_file():
            return path
    raise SystemExit(f"STOP: missing FLORES file {stem}.txt")


def read_sentences(stem: str) -> list[str]:
    # Keep empty lines so parallel indices stay aligned across languages.
    return flores_path(stem).read_text(encoding="utf-8").splitlines()


def per_sentence_pairs(enc: Encoder, lines: list[str]) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for line in lines:
        words = len(line.split())
        if words == 0:
            out.append((0, 0))
            continue
        out.append((len(enc.ids(line)), words))
    return out


def corpus_fert(pairs: list[tuple[int, int]]) -> float:
    tokens = sum(t for t, w in pairs if w)
    words = sum(w for _t, w in pairs if w)
    if words == 0:
        raise SystemExit("STOP: zero words in fertility corpus")
    return tokens / words


def align_pairs(
    left: list[tuple[int, int]], right: list[tuple[int, int]]
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    a: list[tuple[int, int]] = []
    b: list[tuple[int, int]] = []
    for x, y in zip(left, right, strict=True):
        if x[1] > 0 and y[1] > 0:
            a.append(x)
            b.append(y)
    return a, b


def bootstrap_deva_mean(
    per_lang: dict[str, list[tuple[int, int]]],
    left: str,
    right: str,
) -> dict[str, Any]:
    """Shared line-index bootstrap across the seven parallel Devanagari languages."""
    langs = list(DEVA7)
    # All seven FLORES files have the same line count; use the first for length.
    n = len(per_lang[f"{left}:{langs[0]}"])
    for code in langs:
        if len(per_lang[f"{left}:{code}"]) != n or len(per_lang[f"{right}:{code}"]) != n:
            raise SystemExit(f"STOP: line-count mismatch for Deva mean {left} vs {right}")

    def mean_at(name: str, indices: list[int] | None) -> float:
        ferts = []
        for code in langs:
            seq = per_lang[f"{name}:{code}"]
            chosen = seq if indices is None else [seq[i] for i in indices]
            tokens = sum(t for t, w in chosen if w)
            words = sum(w for _t, w in chosen if w)
            ferts.append(tokens / words if words else 0.0)
        return sum(ferts) / len(ferts)

    # Restrict to lines that have words in every language for both arms.
    usable = [
        i
        for i in range(n)
        if all(
            per_lang[f"{left}:{c}"][i][1] > 0 and per_lang[f"{right}:{c}"][i][1] > 0 for c in langs
        )
    ]
    if not usable:
        raise SystemExit(f"STOP: no usable parallel lines for {left} vs {right}")

    point = mean_at(left, usable) - mean_at(right, usable)
    rng = random.Random(SEED)
    m = len(usable)
    diffs: list[float] = []
    for _ in range(BOOT_N):
        pick = [usable[rng.randrange(m)] for _draw in range(m)]
        diffs.append(mean_at(left, pick) - mean_at(right, pick))
    diffs.sort()
    out = {
        "diff": point,
        "lo": diffs[int(0.025 * (len(diffs) - 1))],
        "hi": diffs[min(len(diffs) - 1, int(0.975 * (len(diffs) - 1)))],
        "n": BOOT_N,
        "seed": SEED,
        "method": "per-sentence",
        "langs": list(langs),
    }
    out["verdict"] = verdict(out)
    return out


def self_check(fertility: dict[str, dict[str, float]]) -> None:
    mismatches = []
    for name, metric, want in CHECKS:
        got = fertility[name][metric]
        if abs(got - want) >= 0.001:
            mismatches.append((name, metric, got, want))
    if mismatches:
        lines = [
            f"  {name} {metric}: got {got:.6f} want {want:.3f} Δ={got - want:+.6f}"
            for name, metric, got, want in mismatches
        ]
        raise SystemExit("STOP: self-check mismatch\n" + "\n".join(lines))
    # Explicit TIE check for eng 250k.
    a = fertility["ancoda-250k"]["eng"]
    b = fertility["indicbert-v2"]["eng"]
    # Recompute CI quickly for the reported TIE.
    log(f"self-check ok; eng 250k fertilities {a:.6f} vs {b:.6f}")


def fmt_fert(x: float) -> str:
    return f"{x:.4f}"


def fmt_cell(ci: dict[str, Any]) -> str:
    return f"{ci['verdict']} {ci['diff']:+.4f} [{ci['lo']:+.4f}, {ci['hi']:+.4f}]"


def write_markdown(payload: dict[str, Any]) -> None:
    fert: dict[str, dict[str, float]] = payload["fertility"]
    cells: list[dict[str, Any]] = payload["cells"]
    by_class: dict[str, list[str]] = payload["classes"]
    langs = payload["languages"]
    lines: list[str] = [
        "# FLORES+ per-sentence fertility",
        "",
        (
            "Generated from `artifacts/eval/m8/followup/flores_per_sentence.json`. "
            "Do not hand-edit the tables."
        ),
        "",
        (
            "Method: each FLORES+ devtest sentence is encoded separately; "
            "fertility = total tokens / total whitespace-separated words. "
            f"Paired line bootstrap, {BOOT_N} resamples, seed {SEED}. "
            "WIN = 95% CI entirely below 0 (ours − competitor). "
            "Devanagari mean is the unweighted mean of " + ", ".join(DEVA7) + "."
        ),
        "",
        (
            "This scoreboard supersedes the whole-file first-byte tables in "
            "`docs/GATE_A.md` for all comparisons. Gate A pass/fail bars stay as "
            "originally written."
        ),
        "",
    ]
    # Absolute fertilities per class.
    for cls, members in by_class.items():
        present = [m for m in members if m in fert]
        if not present:
            continue
        lines.append(f"## Absolute fertilities — {cls}")
        lines.append("")
        header = "| Model | " + " | ".join(langs) + " | Deva mean |"
        sep = "|---|" + "---:|" * (len(langs) + 1)
        lines.extend([header, sep])
        for name in present:
            vals = " | ".join(fmt_fert(fert[name][code]) for code in langs)
            lines.append(f"| {name} | {vals} | {fmt_fert(fert[name]['deva_mean'])} |")
        lines.append("")

    # Pairwise scoreboard.
    lines.append("## Pairwise scoreboard (ours − competitor)")
    lines.append("")
    for cls, members in by_class.items():
        ours = [n for n in members if n in OURS and n in fert]
        panel = [n for n in members if n not in OURS and n in fert]
        if not ours or not panel:
            continue
        lines.append(f"### {cls}")
        lines.append("")
        metrics = list(langs) + ["deva_mean"]
        header = "| Ours | Competitor | " + " | ".join(metrics) + " |"
        sep = "|---|---|" + "---|" * len(metrics)
        lines.extend([header, sep])
        for left in ours:
            for right in panel:
                row = [left, right]
                for metric in metrics:
                    hit = next(
                        (
                            c
                            for c in cells
                            if c["ours"] == left and c["base"] == right and c["metric"] == metric
                        ),
                        None,
                    )
                    row.append(fmt_cell(hit) if hit else "—")
                lines.append("| " + " | ".join(row) + " |")
        lines.append("")

    lines.extend(
        [
            "## Self-check",
            "",
            "Independently recounted targets (must match within 0.001):",
            "",
            "- npi 64k: Ancoda 1.451, Yak v1 1.457, Yak v2 1.470, sidskarki 1.370, NE-64k 1.405",
            "- Devanagari mean 64k: Ancoda 1.519, Yak v1 1.576, Yak v2 1.558, sidskarki 1.748",
            "- mar 64k: Ancoda 1.801, Sarvam-1 1.768",
            "- eng 250k: Ancoda 1.229, IndicBERT v2 1.236 (TIE)",
            "- mar 250k: Ancoda 1.535, IndicBERT v2 1.490",
            "",
            f"Result: **passed** (seed {SEED}).",
            "",
        ]
    )
    MD_OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    specs = {
        name: path
        for name, path in load_specs().items()
        if name != "o200k" and path != "skipped: license" and Path(path).is_file()
    }

    log(f"tokenizers={len(specs)} langs={list(LANGS)}")
    sentences: dict[str, list[str]] = {}
    for code, stem in LANGS.items():
        sentences[code] = read_sentences(stem)
        log(f"  {code}: {len(sentences[code])} lines from {flores_path(stem)}")

    # Encode every tokenizer on every language.
    pairs: dict[str, dict[str, list[tuple[int, int]]]] = {}
    fertility: dict[str, dict[str, float]] = {}
    for name, spec in specs.items():
        log(f"encode {name}")
        enc = Encoder(spec)
        pairs[name] = {}
        fertility[name] = {}
        for code, lines in sentences.items():
            got = per_sentence_pairs(enc, lines)
            pairs[name][code] = got
            fertility[name][code] = corpus_fert(got)
        fertility[name]["deva_mean"] = sum(fertility[name][c] for c in DEVA7) / len(DEVA7)
        log(f"  npi={fertility[name]['npi']:.6f} deva_mean={fertility[name]['deva_mean']:.6f}")

    self_check(fertility)

    # Flat lookup for Deva-mean bootstrap.
    flat: dict[str, list[tuple[int, int]]] = {}
    for name in pairs:
        for code in DEVA7:
            flat[f"{name}:{code}"] = pairs[name][code]

    cells: list[dict[str, Any]] = []
    for cls, members in CLASSES.items():
        ours = [n for n in members if n in OURS and n in pairs]
        panel = [n for n in members if n not in OURS and n in pairs]
        for left in ours:
            for right in panel:
                log(f"CI {left} vs {right}")
                for code in LANGS:
                    a, b = align_pairs(pairs[left][code], pairs[right][code])
                    ci = bootstrap_fert_diff(a, b)
                    cells.append(
                        {
                            "class": cls,
                            "ours": left,
                            "base": right,
                            "metric": code,
                            "method": "per-sentence",
                            **ci,
                        }
                    )
                ci_m = bootstrap_deva_mean(flat, left, right)
                cells.append(
                    {
                        "class": cls,
                        "ours": left,
                        "base": right,
                        "metric": "deva_mean",
                        **ci_m,
                    }
                )

    # Confirm the eng 250k TIE claim.
    eng_cell = next(
        c
        for c in cells
        if c["ours"] == "ancoda-250k" and c["base"] == "indicbert-v2" and c["metric"] == "eng"
    )
    if eng_cell["verdict"] != "TIE":
        raise SystemExit(
            f"STOP: eng 250k expected TIE, got {eng_cell['verdict']} "
            f"{eng_cell['diff']:+.6f} [{eng_cell['lo']:+.6f}, {eng_cell['hi']:+.6f}]"
        )

    payload = {
        "kind": "m8_flores_per_sentence",
        "seed": SEED,
        "bootstrap_n": BOOT_N,
        "method": (
            "Each FLORES+ sentence encoded separately; fertility = sum(tokens) / "
            "sum(whitespace words). Paired line bootstrap."
        ),
        "languages": list(LANGS),
        "deva_mean_langs": list(DEVA7),
        "classes": {k: list(v) for k, v in CLASSES.items()},
        "fertility": fertility,
        "cells": cells,
        "n_lines": {code: len(lines) for code, lines in sentences.items()},
        "self_check": [{"name": n, "metric": m, "want": w} for n, m, w in CHECKS],
        "self_check_passed": True,
        "note": (
            "Supersedes whole-file first-byte FLORES comparisons in docs/GATE_A.md. "
            "Gate A pass/fail bars are unchanged."
        ),
    }
    if DEVA_LANGS != (
        "awa_Deva",
        "bho_Deva",
        "hin_Deva",
        "mag_Deva",
        "mai_Deva",
        "npi_Deva",
        "san_Deva",
    ):
        raise SystemExit(f"STOP: DEVA_LANGS changed: {DEVA_LANGS}")

    JSON_OUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    log(f"wrote {JSON_OUT}")
    write_markdown(payload)
    log(f"wrote {MD_OUT}")


if __name__ == "__main__":
    main()

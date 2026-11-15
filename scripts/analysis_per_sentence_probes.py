"""Per-sentence recount of M8 robustness probes. Analysis only; no config changes.

Recomputes on FLORES+ Nepali with each sentence encoded separately:
  - mark-initial (orphan) share for sidskarki-64k and arkios
  - ZWJ / ZWNJ cost for every CLASSES tokenizer
  - misspelled Nepali fertility (and misspell token ratio) for every CLASSES tokenizer
  - Ancoda-250k − Sarvam-30B on Devanagari mean (FLORES+ and sealed), 95% CIs

Old values are read from artifacts/eval/m8/scoreboard.json and robustness.json.
"""

from __future__ import annotations

import json
import random
import unicodedata
from pathlib import Path
from typing import Any

import ancoda_seg

from ancoda_tok.m8_final import (
    BOOT_N,
    CLASSES,
    DEVA_LANGS,
    M8,
    REPO_ROOT,
    SEED,
    Encoder,
    _joiner_text,
    misspell,
    panel_paths,
    read_lines,
)
from ancoda_tok.m8_followup_report import bootstrap_fert_diff, load_specs
from ancoda_tok.m8_flores_per_sentence import (
    corpus_fert,
    flores_path,
    per_sentence_pairs,
)

OUT_JSON = REPO_ROOT / "artifacts" / "analysis" / "per_sentence_probes.json"
OUT_MD = REPO_ROOT / "docs" / "ANALYSIS_PER_SENTENCE_PROBES.md"
SCOREBOARD = M8 / "scoreboard.json"
ROBUSTNESS = M8 / "robustness.json"
FLORES_PS = M8 / "followup" / "flores_per_sentence.json"
SEALED_CI = M8 / "followup" / "sealed_nojjoiners_remeasure.json"


def log(msg: str) -> None:
    print(msg, flush=True)


def all_class_names() -> list[str]:
    names: list[str] = []
    for members in CLASSES.values():
        names.extend(members)
    return names


def is_mark_initial_piece(piece: bytes) -> bool:
    if not piece:
        return False
    try:
        ch = piece.decode("utf-8")[0]
    except UnicodeDecodeError:
        return False
    return unicodedata.category(ch) in ("Mn", "Mc", "Me")


def orphan_share_per_sentence(enc: Encoder, lines: list[str]) -> dict[str, Any]:
    """Mark-initial token share with each sentence encoded alone."""
    orphan = 0
    total = 0
    for line in lines:
        pieces = enc.pieces(line)
        if pieces is None:
            # Fallback: decode each id (HF/SP without byte pieces).
            ids = enc.ids(line)
            total += len(ids)
            # Without pieces we cannot classify; count 0 orphans.
            continue
        total += len(pieces)
        orphan += sum(1 for p in pieces if is_mark_initial_piece(p))
    return {
        "orphan_tokens": orphan,
        "total_tokens": total,
        "share": (orphan / total) if total else None,
        "method": "per-sentence",
    }


def orphan_share_whole_file(enc: Encoder, lines: list[str]) -> dict[str, Any]:
    joined = "\n".join(lines)
    pieces = enc.pieces(joined)
    if pieces is None:
        return {"orphan_tokens": None, "total_tokens": None, "share": None, "method": "whole-file"}
    orphan = sum(1 for p in pieces if is_mark_initial_piece(p))
    return {
        "orphan_tokens": orphan,
        "total_tokens": len(pieces),
        "share": orphan / len(pieces) if pieces else None,
        "method": "whole-file",
    }


def ratio_per_sentence(
    enc: Encoder, original: list[str], rewritten: list[str]
) -> dict[str, Any]:
    assert len(original) == len(rewritten)
    po = [(len(enc.ids(line)), len(line.split())) for line in original]
    pr = [(len(enc.ids(line)), len(line.split())) for line in rewritten]
    den = sum(t for t, _ in po)
    num = sum(t for t, _ in pr)
    point = (num / den) if den else None
    rng = random.Random(SEED)
    m = len(po)
    ratios: list[float] = []
    for _ in range(BOOT_N):
        n = d = 0
        for _draw in range(m):
            i = rng.randrange(m)
            n += pr[i][0]
            d += po[i][0]
        if d:
            ratios.append(n / d)
    ratios.sort()
    return {
        "ratio": point,
        "lo": ratios[int(0.025 * (len(ratios) - 1))] if ratios else None,
        "hi": ratios[min(len(ratios) - 1, int(0.975 * (len(ratios) - 1)))] if ratios else None,
        "method": "per-sentence",
        "n": BOOT_N,
        "seed": SEED,
        "fertility_original": corpus_fert(po),
        "fertility_rewritten": corpus_fert(pr),
    }


def main() -> None:
    letters = frozenset(int(x) for x in ancoda_seg.devanagari_letters())
    npi_lines = read_lines("npi_Deva")
    zwj_lines = [_joiner_text(line, "\u200d", letters) for line in npi_lines]
    zwnj_lines = [_joiner_text(line, "\u200c", letters) for line in npi_lines]
    rng = random.Random(SEED)
    miss_lines = [misspell(line, rng) for line in npi_lines]

    old_sb = json.loads(SCOREBOARD.read_text(encoding="utf-8"))
    old_rob = json.loads(ROBUSTNESS.read_text(encoding="utf-8"))
    specs = load_specs()

    names = all_class_names()
    results: dict[str, Any] = {}
    for name in names:
        if name not in specs:
            log(f"skip missing spec {name}")
            continue
        log(f"measure {name}")
        enc = Encoder(specs[name])
        old_m = old_sb["metrics"].get(name, {})
        old_r = old_rob["rows"].get(name, {}) if isinstance(old_rob["rows"], dict) else {}
        orphan_ps = orphan_share_per_sentence(enc, npi_lines)
        orphan_wf = orphan_share_whole_file(enc, npi_lines)
        zwj = ratio_per_sentence(enc, npi_lines, zwj_lines)
        zwnj = ratio_per_sentence(enc, npi_lines, zwnj_lines)
        miss = ratio_per_sentence(enc, npi_lines, miss_lines)
        results[name] = {
            "orphan": {
                "old_scoreboard_share": old_m.get("langs", {}).get("orphan_npi"),
                "whole_file_recompute": orphan_wf,
                "per_sentence": orphan_ps,
            },
            "zwj": {
                "old_ratio": (old_m.get("zwj") or {}).get("ratio"),
                "old_method": (old_m.get("zwj") or {}).get("method"),
                "per_sentence": zwj,
            },
            "zwnj": {
                "old_ratio": (old_m.get("zwnj") or {}).get("ratio"),
                "old_method": (old_m.get("zwnj") or {}).get("method"),
                "per_sentence": zwnj,
            },
            "misspell": {
                "old_ratio": (old_r.get("misspell_10pct") or {}).get("ratio"),
                "old_method": (old_r.get("misspell_10pct") or {}).get("method"),
                "old_fertility_note": (
                    "Old published misspell column is token ratio, not fertility. "
                    "Per-sentence fertility of the misspelled text is fertility_rewritten."
                ),
                "per_sentence": miss,
            },
        }

    # Sarvam-30B CI gaps: FLORES+ already in flores_per_sentence; sealed in followup.
    flores = json.loads(FLORES_PS.read_text(encoding="utf-8"))
    sealed = json.loads(SEALED_CI.read_text(encoding="utf-8"))
    flores_cell = None
    for cell in flores.get("cells", []):
        if cell.get("left") == "ancoda-250k" and cell.get("right") == "sarvam-30b":
            flores_cell = cell
            break
    sealed_row = None
    for row in sealed.get("rows", []):
        if "sarvam-30b" in str(row.get("comparison", row.get("name", ""))):
            sealed_row = row
            break
    # sealed file structure may use nested comparisons
    if sealed_row is None:
        comps = sealed.get("comparisons") or sealed.get("deva_mean") or {}
        if isinstance(comps, dict):
            sealed_row = comps.get("ancoda-250k_vs_sarvam-30b") or comps.get(
                "ancoda-250k vs sarvam-30b"
            )

    # Recompute FLORES+ Deva-mean CI from per-sentence fertility file if needed.
    fert = flores["fertility"]
    flores_deva = {
        "ancoda-250k": fert["ancoda-250k"]["deva_mean"],
        "sarvam-30b": fert["sarvam-30b"]["deva_mean"],
        "diff": fert["ancoda-250k"]["deva_mean"] - fert["sarvam-30b"]["deva_mean"],
    }
    if flores_cell and "deva_mean" in flores_cell:
        flores_deva["ci"] = flores_cell["deva_mean"]
    elif flores_cell:
        # cells store per-lang; take summary if present
        flores_deva["cell"] = {
            k: flores_cell[k] for k in flores_cell if k not in ("left", "right")
        }

    payload = {
        "kind": "analysis_per_sentence_probes",
        "n_lines": len(npi_lines),
        "bootstrap_n": BOOT_N,
        "seed": SEED,
        "tokenizers": results,
        "sarvam_30b_ci": {
            "flores_deva_mean": flores_deva,
            "flores_cell": flores_cell,
            "sealed": sealed_row,
            "sealed_gate_a_line": (
                "ancoda-250k vs sarvam-30b | WIN −0.5548 [−0.5942,−0.5285] "
                "(docs/GATE_A.md sealed no-joiners remeasure)"
            ),
            "note": (
                "Sarvam-30B is in the 250k class (5 tokenizers there; 21 across CLASSES). "
                "FLORES+ per-sentence CIs are in RESULTS_PER_SENTENCE.md; sealed CI is in "
                "GATE_A.md / sealed_nojjoiners_remeasure.json. Report both here for the appendix gap."
            ),
        },
        "whole_file_numbers_elsewhere": [
            "GATE_A.md column table npi/eng/hin/deva_mean fertilities (whole-file Yak definition)",
            "GATE_A.md ZWJ/ZWNJ columns (scoreboard ratio = whole-file for most HF tokenizers)",
            "GATE_A.md Robustness misspell/nukta/NFC/NFD ratios marked 'whole-file first-byte'",
            "docs/RESULTS.md leaderboard 'npi' column (whole-file); 'npi per line' is already per-sentence",
            "docs/RESULTS.md Robustness probes ZWJ/ZWNJ/typo ratios (Yak harness; often whole-file)",
            "Orphan-mark share in RESULTS.md leaderboard for sidskarki (1.7%) and arkios (14.6%)",
        ],
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "# Per-sentence recount of robustness probes",
        "",
        "Analysis only. Each FLORES+ Nepali sentence is encoded separately. "
        "Old values come from `artifacts/eval/m8/scoreboard.json` and `robustness.json` "
        "(many marked `whole-file first-byte`). Reproduction: "
        "`scripts/analysis_per_sentence_probes.py`.",
        "",
        "## Mark-initial (orphan) share — sidskarki & Arkios",
        "",
        "| Tokenizer | Old (scoreboard) | Whole-file recompute | Per-sentence |",
        "|---|---:|---:|---:|",
    ]
    for name in ("sidskarki-64k", "arkios"):
        r = results[name]["orphan"]
        lines.append(
            f"| {name} | {r['old_scoreboard_share']} | "
            f"{r['whole_file_recompute']['share']} | {r['per_sentence']['share']} |"
        )
    lines += [
        "",
        "## ZWJ / ZWNJ cost (Table 4 / GATE_A columns)",
        "",
        "| Tokenizer | Old ZWJ | New ZWJ (per-sentence) | Old ZWNJ | New ZWNJ (per-sentence) |",
        "|---|---:|---:|---:|---:|",
    ]
    for name in names:
        if name not in results:
            continue
        r = results[name]
        lines.append(
            f"| {name} | {r['zwj']['old_ratio']} | {r['zwj']['per_sentence']['ratio']} | "
            f"{r['zwnj']['old_ratio']} | {r['zwnj']['per_sentence']['ratio']} |"
        )
    lines += [
        "",
        "## Misspelled Nepali (10% M8 misspell)",
        "",
        "Old column is token ratio. New columns: per-sentence ratio and fertility of the "
        "misspelled text (tokens/words).",
        "",
        "| Tokenizer | Old ratio | New ratio | Fertility (misspelled) | Fertility (clean) |",
        "|---|---:|---:|---:|---:|",
    ]
    for name in names:
        if name not in results:
            continue
        m = results[name]["misspell"]
        ps = m["per_sentence"]
        lines.append(
            f"| {name} | {m['old_ratio']} | {ps['ratio']} | "
            f"{ps['fertility_rewritten']:.6f} | {ps['fertility_original']:.6f} |"
        )
    lines += [
        "",
        "## Sarvam-30B appendix CI (Ancoda-250k − Sarvam-30B, Devanagari mean)",
        "",
        f"- FLORES+ per-sentence Deva mean: Ancoda-250k={flores_deva['ancoda-250k']:.6f}, "
        f"Sarvam-30B={flores_deva['sarvam-30b']:.6f}, diff={flores_deva['diff']:.6f}.",
        f"- FLORES+ cell: `{flores_cell}`.",
        f"- Sealed (GATE_A): {payload['sarvam_30b_ci']['sealed_gate_a_line']}.",
        "",
        "## Other paper numbers from whole-file counts",
        "",
    ]
    for item in payload["whole_file_numbers_elsewhere"]:
        lines.append(f"- {item}")
    lines += ["", f"JSON: `{OUT_JSON.relative_to(REPO_ROOT)}`.", ""]
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    log(f"wrote {OUT_JSON}")
    log(f"wrote {OUT_MD}")


if __name__ == "__main__":
    main()

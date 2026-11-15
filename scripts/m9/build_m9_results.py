#!/usr/bin/env python3
"""Build docs/M9_RESULTS.md, artifacts/m9/results.json, and plots from eval JSONs."""

from __future__ import annotations

import json
import math
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
EVAL = REPO / "artifacts/m9/server_copy/evals"
OUT_JSON = REPO / "artifacts/m9/results.json"
OUT_MD = REPO / "docs/M9_RESULTS.md"
PLOT_DIR = REPO / "artifacts/m9/plots"
VERIFY = REPO / "artifacts/m9/verify"

DEV11 = (
    "Hindi",
    "Marathi",
    "Nepali",
    "Sanskrit",
    "anp_Deva",
    "awa_Deva",
    "bho_Deva",
    "gom_Deva",
    "hne_Deva",
    "mai_Deva",
    "new_Deva",
)
ARMS = {"A": "ancoda-64k", "B": "yak-64k-v1", "C": "sidskarki-64k", "E": "yak-64k-v2.0"}
SECONDARY_SEALED = (
    "English",
    "Hindi",
    "Marathi",
    "Code",
    "Math",
    "anp_Deva",
    "awa_Deva",
    "bho_Deva",
    "gom_Deva",
    "hne_Deva",
    "mai_Deva",
    "new_Deva",
    "und_Deva",
    "Sanskrit",
    "Nepali",
)
FLORES = ("npi", "hin", "eng", "mai", "bho", "awa", "mag", "san", "mar", "hne")
PERTURB = (
    "flores_npi_misspell.jsonl:npi",
    "flores_npi_zwj.jsonl:npi",
    "flores_npi_zwnj.jsonl:npi",
    "sealed_npi_misspell.jsonl:Nepali",
    "sealed_npi_zwj.jsonl:Nepali",
    "sealed_npi_zwnj.jsonl:Nepali",
)


def load(arm: str, seed: int, kind: str) -> dict:
    return json.loads((EVAL / f"{arm}-s{seed}-{kind}.json").read_text(encoding="utf-8"))


def sealed(payload: dict, lang: str) -> float:
    return float(payload["groups"][f"sealed.jsonl:{lang}"]["bpb"])


def group(payload: dict, key: str) -> float | None:
    g = payload["groups"].get(key)
    return None if g is None else g["bpb"]


def flores(payload: dict, code: str) -> float:
    return float(payload["groups"][f"{code}.jsonl:{code}"]["bpb"])


def npi_dev(payload: dict) -> tuple[float, float]:
    return sealed(payload, "Nepali"), statistics.fmean(sealed(payload, L) for L in DEV11)


def paired_call(diffs: list[float]) -> str:
    if len(diffs) != 3:
        raise ValueError(diffs)
    if not all(d < 0 for d in diffs):
        return "no measurable difference"
    mean = sum(diffs) / 3.0
    std = statistics.stdev(diffs)  # sample SD, divisor n-1
    if abs(mean) > std:
        return "A better"
    return "no measurable difference"


def fmt(x: float | None, digits: int = 6) -> str:
    if x is None:
        return "—"
    return f"{x:.{digits}f}"


def primary_block() -> dict:
    out: dict = {"equal_tokens": {}, "equal_bytes": {}}
    for kind, label in (("final", "equal_tokens"), ("bstar", "equal_bytes")):
        block: dict = {"per_seed": {}, "comparisons": {}}
        for seed in (1, 2, 3):
            vals = {}
            for letter, arm in (("A", ARMS["A"]), ("B", ARMS["B"]), ("C", ARMS["C"])):
                npi, dev = npi_dev(load(arm, seed, kind))
                vals[letter] = {"Nepali": npi, "Dev11": dev}
            block["per_seed"][str(seed)] = vals
        for other, name in (("B", "A_vs_B"), ("C", "A_vs_C")):
            comps = {}
            for metric in ("Nepali", "Dev11"):
                diffs = [
                    block["per_seed"][str(s)]["A"][metric] - block["per_seed"][str(s)][other][metric]
                    for s in (1, 2, 3)
                ]
                mean = statistics.fmean(diffs)
                std = statistics.stdev(diffs)
                comps[metric] = {
                    "diffs": diffs,
                    "mean": mean,
                    "sample_sd": std,
                    "verdict": paired_call(diffs),
                }
            block["comparisons"][name] = comps
        # means
        means = {L: {"Nepali": [], "Dev11": []} for L in ("A", "B", "C")}
        for seed in (1, 2, 3):
            for L in ("A", "B", "C"):
                means[L]["Nepali"].append(block["per_seed"][str(seed)][L]["Nepali"])
                means[L]["Dev11"].append(block["per_seed"][str(seed)][L]["Dev11"])
        block["seed_mean"] = {
            L: {m: statistics.fmean(vs) for m, vs in metrics.items()} for L, metrics in means.items()
        }
        out[label] = block
    return out


def lossless_block() -> dict:
    out: dict = {"equal_tokens": {}, "equal_bytes": {}, "roundtrip_rates": {}}
    # rates from any C report (same docs)
    c1 = load(ARMS["C"], 1, "final")["sidskarki_roundtrip"]
    out["roundtrip_rates"] = {
        "per_eval_set": c1["per_eval_set"],
        "per_language": c1["per_language"],
        "note": c1["note"],
    }
    for kind, label in (("final", "equal_tokens"), ("bstar", "equal_bytes")):
        per_seed = {}
        for seed in (1, 2, 3):
            a = load(ARMS["A"], seed, kind)["sidskarki_roundtrip"]["bpb_on_lossless_docs"]
            c = load(ARMS["C"], seed, kind)["sidskarki_roundtrip"]["bpb_on_lossless_docs"]

            def agg(groups: dict, langs: tuple[str, ...]) -> float:
                return statistics.fmean(groups[f"sealed.jsonl:{L}"]["bpb"] for L in langs)

            per_seed[str(seed)] = {
                "Nepali": {
                    "A": a["sealed.jsonl:Nepali"]["bpb"],
                    "C": c["sealed.jsonl:Nepali"]["bpb"],
                    "diff_A_minus_C": a["sealed.jsonl:Nepali"]["bpb"] - c["sealed.jsonl:Nepali"]["bpb"],
                    "docs": a["sealed.jsonl:Nepali"]["docs"],
                },
                "Dev11": {
                    "A": agg(a, DEV11),
                    "C": agg(c, DEV11),
                    "diff_A_minus_C": agg(a, DEV11) - agg(c, DEV11),
                },
            }
        out[label] = {"per_seed": per_seed}
    return out


def secondary_block() -> dict:
    out: dict = {"sealed_languages": {}, "flores": {}, "perturb": {}, "arm_E": {}}
    for kind, label in (("final", "equal_tokens"), ("bstar", "equal_bytes")):
        sealed_tbl = {}
        for seed in (1, 2, 3):
            row = {}
            for letter, arm in (("A", ARMS["A"]), ("B", ARMS["B"]), ("C", ARMS["C"])):
                p = load(arm, seed, kind)
                row[letter] = {lang: sealed(p, lang) for lang in SECONDARY_SEALED}
            sealed_tbl[str(seed)] = row
        # losses: where A is worse than B or C
        losses = []
        for seed, row in sealed_tbl.items():
            for lang in SECONDARY_SEALED:
                for other in ("B", "C"):
                    diff = row["A"][lang] - row[other][lang]
                    if diff > 0:
                        losses.append(
                            {
                                "checkpoint": label,
                                "seed": int(seed),
                                "language": lang,
                                "vs": other,
                                "A": row["A"][lang],
                                "other": row[other][lang],
                                "diff": diff,
                            }
                        )
        out["sealed_languages"][label] = {"per_seed": sealed_tbl, "A_losses": losses}

        flores_tbl = {}
        for seed in (1, 2, 3):
            row = {}
            for letter, arm in (("A", ARMS["A"]), ("B", ARMS["B"]), ("C", ARMS["C"])):
                p = load(arm, seed, kind)
                row[letter] = {code: flores(p, code) for code in FLORES}
            flores_tbl[str(seed)] = row
        flosses = []
        for seed, row in flores_tbl.items():
            for code in FLORES:
                for other in ("B", "C"):
                    diff = row["A"][code] - row[other][code]
                    if diff > 0:
                        flosses.append(
                            {
                                "checkpoint": label,
                                "seed": int(seed),
                                "language": code,
                                "vs": other,
                                "A": row["A"][code],
                                "other": row[other][code],
                                "diff": diff,
                            }
                        )
        out["flores"][label] = {"per_seed": flores_tbl, "A_losses": flosses}

        pert = {}
        for seed in (1, 2, 3):
            row = {}
            for letter, arm in (("A", ARMS["A"]), ("B", ARMS["B"]), ("C", ARMS["C"])):
                p = load(arm, seed, kind)
                row[letter] = {k: group(p, k) for k in PERTURB}
            pert[str(seed)] = row
        plosses = []
        for seed, row in pert.items():
            for key in PERTURB:
                for other in ("B", "C"):
                    if row["A"][key] is None or row[other][key] is None:
                        continue
                    diff = row["A"][key] - row[other][key]
                    if diff > 0:
                        plosses.append(
                            {
                                "checkpoint": label,
                                "seed": int(seed),
                                "group": key,
                                "vs": other,
                                "A": row["A"][key],
                                "other": row[other][key],
                                "diff": diff,
                            }
                        )
        out["perturb"][label] = {"per_seed": pert, "A_losses": plosses}

    # Arm E seed 1 only
    for kind, label in (("final", "equal_tokens"), ("bstar", "equal_bytes")):
        e = load(ARMS["E"], 1, kind)
        a = load(ARMS["A"], 1, kind)
        b = load(ARMS["B"], 1, kind)
        en, ed = npi_dev(e)
        an, ad = npi_dev(a)
        bn, bd = npi_dev(b)
        out["arm_E"][label] = {
            "label": "single seed, no verdict",
            "seed": 1,
            "Nepali": {"A": an, "B": bn, "E": en, "A_minus_E": an - en, "B_minus_E": bn - en},
            "Dev11": {"A": ad, "B": bd, "E": ed, "A_minus_E": ad - ed, "B_minus_E": bd - ed},
            "sealed_languages": {lang: sealed(e, lang) for lang in SECONDARY_SEALED},
            "flores": {code: flores(e, code) for code in FLORES},
        }
    return out


def curves_block() -> dict:
    """BPB vs bytes seen from training logs + eval BPB at final and B*."""
    runs = REPO / "docs/m9_server_copy/runs"
    # Prefer server_copy artifacts logs
    alt = REPO / "artifacts/m9/server_copy/runs"
    root = alt if alt.exists() else runs
    curves = {}
    for letter, arm in ARMS.items():
        seeds = (1, 2, 3) if letter != "E" else (1,)
        for seed in seeds:
            key = f"{arm}-s{seed}"
            log = root / key / "log.jsonl"
            if not log.exists():
                continue
            rows = [json.loads(l) for l in log.read_text().splitlines() if l.strip()]
            # Logs are already sparse (~every 50 steps). Keep every row; do not
            # filter on step%500 (those exact steps are never present).
            pts = [
                {
                    "step": int(r["completed_steps"]),
                    "bytes_seen": int(r["bytes_seen"]),
                    "train_loss": float(r["loss"]),
                }
                for r in rows
            ]
            evals = {}
            for kind in ("final", "bstar"):
                path = EVAL / f"{arm}-s{seed}-{kind}.json"
                if path.exists():
                    npi, dev = npi_dev(json.loads(path.read_text()))
                    evals[kind] = {"Nepali": npi, "Dev11": dev}
            curves[key] = {"letter": letter, "train": pts, "eval_bpb": evals}
    return curves


def make_plots(primary: dict, curves: dict) -> list[str]:
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    paths = []
    # paired diffs
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharey=False)
    seeds = [1, 2, 3]
    for ax, (ckpt, other, metric) in zip(
        axes.flat,
        [
            ("equal_tokens", "A_vs_B", "Nepali"),
            ("equal_tokens", "A_vs_B", "Dev11"),
            ("equal_tokens", "A_vs_C", "Nepali"),
            ("equal_tokens", "A_vs_C", "Dev11"),
        ],
        strict=True,
    ):
        diffs = primary[ckpt]["comparisons"][other][metric]["diffs"]
        ax.bar([str(s) for s in seeds], diffs, color="#2f6fed")
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_title(f"{ckpt.replace('_', ' ')} {other.replace('_', ' ')} {metric}")
        ax.set_ylabel("BPB(A) − BPB(other)")
        ax.set_xlabel("seed")
    fig.tight_layout()
    p = PLOT_DIR / "paired_diffs_equal_tokens.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)
    paths.append(str(p.relative_to(REPO)))

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    for ax, (ckpt, other, metric) in zip(
        axes.flat,
        [
            ("equal_bytes", "A_vs_B", "Nepali"),
            ("equal_bytes", "A_vs_B", "Dev11"),
            ("equal_bytes", "A_vs_C", "Nepali"),
            ("equal_bytes", "A_vs_C", "Dev11"),
        ],
        strict=True,
    ):
        diffs = primary[ckpt]["comparisons"][other][metric]["diffs"]
        ax.bar([str(s) for s in seeds], diffs, color="#c45c26")
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_title(f"{ckpt.replace('_', ' ')} {other.replace('_', ' ')} {metric}")
        ax.set_ylabel("BPB(A) − BPB(other)")
        ax.set_xlabel("seed")
    fig.tight_layout()
    p = PLOT_DIR / "paired_diffs_equal_bytes.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)
    paths.append(str(p.relative_to(REPO)))

    # BPB vs bytes: plot train loss vs bytes, and mark eval BPB on secondary axis conceptually
    # Primary request: BPB vs bytes seen for each arm — use sealed Nepali eval at B* and final
    # against bytes_seen at those checkpoints from logs.
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = {"A": "#2f6fed", "B": "#888888", "C": "#c45c26", "E": "#2a9d8f"}
    for key, curve in sorted(curves.items()):
        letter = curve["letter"]
        if letter not in ("A", "B", "C", "E"):
            continue
        # Only seed 1 for clarity on one plot, plus note
        if not key.endswith("-s1"):
            continue
        xs, ys = [], []
        train = {int(p["step"]): p for p in curve["train"]}
        # B* step from checkpoint name is not here; use eval points approximately:
        # final at 7630
        if 7630 in train and "final" in curve["eval_bpb"]:
            xs.append(train[7630]["bytes_seen"])
            ys.append(curve["eval_bpb"]["final"]["Nepali"])
        # find bstar step near log: use closest step with eval bstar if available
        if "bstar" in curve["eval_bpb"]:
            # pick step with bytes closest below final among logged
            # Prefer steps that look like bstar (~5400)
            candidates = [s for s in train if 5000 <= s <= 6000] or list(train)
            if candidates:
                s = min(candidates, key=lambda s: abs(s - 5400))
                xs.append(train[s]["bytes_seen"])
                ys.append(curve["eval_bpb"]["bstar"]["Nepali"])
        if xs:
            order = sorted(range(len(xs)), key=lambda i: xs[i])
            ax.plot(
                [xs[i] for i in order],
                [ys[i] for i in order],
                marker="o",
                label=f"{letter} {key}",
                color=colors.get(letter, "black"),
            )
    ax.set_xlabel("text bytes seen (approx. at checkpoint)")
    ax.set_ylabel("sealed Nepali BPB")
    ax.set_title("Sealed Nepali BPB vs bytes seen (seed 1)")
    ax.legend()
    fig.tight_layout()
    p = PLOT_DIR / "bpb_vs_bytes_seed1.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)
    paths.append(str(p.relative_to(REPO)))

    # Per-arm train loss vs bytes (all seeds) as supporting curve
    fig, ax = plt.subplots(figsize=(9, 5))
    for key, curve in sorted(curves.items()):
        letter = curve["letter"]
        if letter not in ("A", "B", "C"):
            continue
        xs = [p["bytes_seen"] for p in curve["train"]]
        ys = [p["train_loss"] for p in curve["train"]]
        ax.plot(xs, ys, linewidth=1.0, alpha=0.85, label=key, color=colors[letter])
    ax.set_xlabel("text bytes seen")
    ax.set_ylabel("train loss (nats/token)")
    ax.set_title("Training loss vs bytes seen")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    p = PLOT_DIR / "train_loss_vs_bytes.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)
    paths.append(str(p.relative_to(REPO)))
    return paths


def write_md(payload: dict) -> None:
    p = payload["primary"]
    lines: list[str] = []
    lines.append("# M9 results")
    lines.append("")
    lines.append(
        "Generated from `artifacts/m9/server_copy/evals/*.json` by "
        "`scripts/m9/build_m9_results.py`. Do not hand-edit the tables."
    )
    lines.append("")
    lines.append("Tokenizers frozen. Pre-registration: `docs/M9_PREREG.md`.")
    lines.append("")
    lines.append("## Verification")
    v = payload["verification"]
    lines.append("")
    lines.append(f"- BPB denominator: {v['denominator']}")
    lines.append(f"- EOS scoring: {v['eos_policy']}")
    lines.append(
        f"- Same eval documents across arms: {v['same_eval_docs']['status']} "
        f"({v['same_eval_docs']['groups']} groups; sealed doc_id list sha256 "
        f"`{v['same_eval_docs']['sealed_doc_id_list_sha256']}`)."
    )
    lines.append(
        f"- Independent 300-doc recompute: {v['independent_recompute']['status']} "
        f"(tolerance 0.001 BPB)."
    )
    for row in v["independent_recompute"]["arms"]:
        lines.append(
            f"  - {row['arm']}: pipeline {row['pipeline_bpb']:.6f}, "
            f"standalone {row['standalone_bpb']:.6f}, "
            f"|Δ|={row['abs_diff']:.6f}"
        )
    lines.append("")
    lines.append("## Primary verdicts (pre-registered rule)")
    lines.append("")
    lines.append(
        "Label is **A better** only when every seed difference BPB(A)−BPB(other) "
        "is strictly negative and |mean| > sample SD (n−1). Otherwise "
        "**no measurable difference**."
    )
    lines.append("")
    for ckpt in ("equal_tokens", "equal_bytes"):
        lines.append(f"### {ckpt.replace('_', ' ')}")
        lines.append("")
        lines.append("| Comparison | Metric | d1 | d2 | d3 | mean | sample SD | verdict |")
        lines.append("|---|---|---:|---:|---:|---:|---:|---|")
        for comp in ("A_vs_B", "A_vs_C"):
            for metric in ("Nepali", "Dev11"):
                c = p[ckpt]["comparisons"][comp][metric]
                d = c["diffs"]
                lines.append(
                    f"| {comp.replace('_', ' ')} | {metric} | {d[0]:+.6f} | {d[1]:+.6f} | "
                    f"{d[2]:+.6f} | {c['mean']:+.6f} | {c['sample_sd']:.6f} | **{c['verdict']}** |"
                )
        lines.append("")
        lines.append("| Seed | A Nepali | B Nepali | C Nepali | A Dev11 | B Dev11 | C Dev11 |")
        lines.append("|---:|---:|---:|---:|---:|---:|---:|")
        for seed in (1, 2, 3):
            row = p[ckpt]["per_seed"][str(seed)]
            lines.append(
                f"| {seed} | {row['A']['Nepali']:.6f} | {row['B']['Nepali']:.6f} | "
                f"{row['C']['Nepali']:.6f} | {row['A']['Dev11']:.6f} | "
                f"{row['B']['Dev11']:.6f} | {row['C']['Dev11']:.6f} |"
            )
        m = p[ckpt]["seed_mean"]
        lines.append(
            f"| mean | {m['A']['Nepali']:.6f} | {m['B']['Nepali']:.6f} | {m['C']['Nepali']:.6f} | "
            f"{m['A']['Dev11']:.6f} | {m['B']['Dev11']:.6f} | {m['C']['Dev11']:.6f} |"
        )
        lines.append("")

    lines.append("## Amended comparison: A vs C on sidskarki-lossless docs")
    lines.append("")
    lines.append(
        "Restricted to documents where `sidskarki.decode(sidskarki.encode(x)) == x`. "
        "Does not replace the primary label."
    )
    lines.append("")
    L = payload["lossless"]
    for ckpt in ("equal_tokens", "equal_bytes"):
        lines.append(f"### {ckpt.replace('_', ' ')}")
        lines.append("")
        lines.append("| Seed | Nepali A | Nepali C | A−C | lossless Nepali docs | Dev11 A | Dev11 C | A−C |")
        lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|")
        for seed in (1, 2, 3):
            row = L[ckpt]["per_seed"][str(seed)]
            lines.append(
                f"| {seed} | {row['Nepali']['A']:.6f} | {row['Nepali']['C']:.6f} | "
                f"{row['Nepali']['diff_A_minus_C']:+.6f} | {row['Nepali']['docs']} | "
                f"{row['Dev11']['A']:.6f} | {row['Dev11']['C']:.6f} | "
                f"{row['Dev11']['diff_A_minus_C']:+.6f} |"
            )
        lines.append("")
    lines.append("### sidskarki round-trip rates")
    lines.append("")
    lines.append("| Eval set | docs | ok | rate |")
    lines.append("|---|---:|---:|---:|")
    for name, agg in sorted(L["roundtrip_rates"]["per_eval_set"].items()):
        lines.append(f"| `{name}` | {agg['docs']} | {agg['ok']} | {agg['rate']:.4f} |")
    lines.append("")
    lines.append("| Language | docs | ok | rate |")
    lines.append("|---|---:|---:|---:|")
    for name, agg in sorted(L["roundtrip_rates"]["per_language"].items()):
        lines.append(f"| {name} | {agg['docs']} | {agg['ok']} | {agg['rate']:.4f} |")
    lines.append("")

    S = payload["secondary"]
    lines.append("## Secondary")
    lines.append("")
    lines.append("### Sealed languages (equal tokens, seed mean of A/B/C)")
    lines.append("")
    # mean over seeds for A/B/C equal tokens
    lines.append("| Language | A | B | C | A−B | A−C |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    et = S["sealed_languages"]["equal_tokens"]["per_seed"]
    for lang in SECONDARY_SEALED:
        a = statistics.fmean(et[str(s)]["A"][lang] for s in (1, 2, 3))
        b = statistics.fmean(et[str(s)]["B"][lang] for s in (1, 2, 3))
        c = statistics.fmean(et[str(s)]["C"][lang] for s in (1, 2, 3))
        lines.append(f"| {lang} | {a:.6f} | {b:.6f} | {c:.6f} | {a-b:+.6f} | {a-c:+.6f} |")
    lines.append("")
    lines.append("#### Every A loss on sealed languages")
    lines.append("")
    losses = S["sealed_languages"]["equal_tokens"]["A_losses"] + S["sealed_languages"]["equal_bytes"]["A_losses"]
    if not losses:
        lines.append("None: A was never higher BPB than B or C on these rows.")
    else:
        lines.append("| Checkpoint | Seed | Language | vs | A | other | diff |")
        lines.append("|---|---:|---|---|---:|---:|---:|")
        for row in losses:
            lines.append(
                f"| {row['checkpoint']} | {row['seed']} | {row['language']} | {row['vs']} | "
                f"{row['A']:.6f} | {row['other']:.6f} | {row['diff']:+.6f} |"
            )
    lines.append("")
    lines.append("### FLORES+ devtest (equal tokens, seed mean)")
    lines.append("")
    lines.append("| Lang | A | B | C | A−B | A−C |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    ft = S["flores"]["equal_tokens"]["per_seed"]
    for code in FLORES:
        a = statistics.fmean(ft[str(s)]["A"][code] for s in (1, 2, 3))
        b = statistics.fmean(ft[str(s)]["B"][code] for s in (1, 2, 3))
        c = statistics.fmean(ft[str(s)]["C"][code] for s in (1, 2, 3))
        lines.append(f"| {code} | {a:.6f} | {b:.6f} | {c:.6f} | {a-b:+.6f} | {a-c:+.6f} |")
    lines.append("")
    flosses = S["flores"]["equal_tokens"]["A_losses"] + S["flores"]["equal_bytes"]["A_losses"]
    lines.append("#### Every A loss on FLORES")
    lines.append("")
    if not flosses:
        lines.append("None.")
    else:
        lines.append("| Checkpoint | Seed | Lang | vs | A | other | diff |")
        lines.append("|---|---:|---|---|---:|---:|---:|")
        for row in flosses:
            lines.append(
                f"| {row['checkpoint']} | {row['seed']} | {row['language']} | {row['vs']} | "
                f"{row['A']:.6f} | {row['other']:.6f} | {row['diff']:+.6f} |"
            )
    lines.append("")
    lines.append("### Perturbed text (equal tokens, seed mean)")
    lines.append("")
    lines.append("Denominator is UTF-8 bytes of the unmodified text.")
    lines.append("")
    lines.append("| Group | A | B | C | A−B | A−C |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    pt = S["perturb"]["equal_tokens"]["per_seed"]
    for key in PERTURB:
        a = statistics.fmean(pt[str(s)]["A"][key] for s in (1, 2, 3))
        b = statistics.fmean(pt[str(s)]["B"][key] for s in (1, 2, 3))
        c = statistics.fmean(pt[str(s)]["C"][key] for s in (1, 2, 3))
        lines.append(f"| `{key}` | {a:.6f} | {b:.6f} | {c:.6f} | {a-b:+.6f} | {a-c:+.6f} |")
    lines.append("")
    plosses = S["perturb"]["equal_tokens"]["A_losses"] + S["perturb"]["equal_bytes"]["A_losses"]
    lines.append("#### Every A loss on perturbed sets")
    lines.append("")
    if not plosses:
        lines.append("None.")
    else:
        lines.append("| Checkpoint | Seed | Group | vs | A | other | diff |")
        lines.append("|---|---:|---|---|---:|---:|---:|")
        for row in plosses:
            lines.append(
                f"| {row['checkpoint']} | {row['seed']} | `{row['group']}` | {row['vs']} | "
                f"{row['A']:.6f} | {row['other']:.6f} | {row['diff']:+.6f} |"
            )
    lines.append("")
    lines.append("### Arm E (yak-64k-v2.0) — single seed, no verdict")
    lines.append("")
    e = S["arm_E"]["equal_tokens"]
    lines.append("| Checkpoint | Metric | A | B | E | A−E |")
    lines.append("|---|---|---:|---:|---:|---:|")
    for ckpt in ("equal_tokens", "equal_bytes"):
        e = S["arm_E"][ckpt]
        for metric in ("Nepali", "Dev11"):
            row = e[metric]
            lines.append(
                f"| {ckpt.replace('_', ' ')} | {metric} | {row['A']:.6f} | {row['B']:.6f} | "
                f"{row['E']:.6f} | {row['A_minus_E']:+.6f} |"
            )
    lines.append("")
    lines.append("## Plots")
    lines.append("")
    lines.append(
        "Sealed BPB vs bytes uses only the two scored checkpoints per run "
        "(B* and final). No intermediate eval series exists (one ancoda seed-1 "
        "pilot at step 301 only). Train-loss curves use the sparse training logs."
    )
    lines.append("")
    for path in payload["plots"]:
        lines.append(f"- `{path}`")
    lines.append("")
    lines.append("## Limitations (from the pre-registration)")
    lines.append("")
    lines.append(
        "(a) At equal-bytes, a more efficient arm has seen fewer tokens and is earlier "
        "in the schedule. (b) sidskarki is not lossless on whitespace; round-trip rates "
        "are above. (c) Vocabulary sizes differ; sidskarki embedding is extended by one."
    )
    lines.append("")
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    verify_path = VERIFY / "summary.json"
    if not verify_path.exists():
        raise SystemExit(f"STOP: missing {verify_path}; run verification first")
    verification = json.loads(verify_path.read_text(encoding="utf-8"))
    primary = primary_block()
    lossless = lossless_block()
    secondary = secondary_block()
    curves = curves_block()
    plots = make_plots(primary, curves)
    payload = {
        "kind": "m9_results",
        "prereg": "docs/M9_PREREG.md",
        "verification": verification,
        "primary": primary,
        "lossless": lossless,
        "secondary": secondary,
        "curves": curves,
        "plots": plots,
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_md(payload)
    print(f"wrote {OUT_JSON}")
    print(f"wrote {OUT_MD}")
    for p in plots:
        print(f"plot {p}")


if __name__ == "__main__":
    main()

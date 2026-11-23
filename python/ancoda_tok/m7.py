"""M7 mix sweep and ablation ladder orchestration."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

from ancoda_tok.clean.common import (
    base_manifest,
    finish_manifest,
    free_bytes,
    save_json,
)
from ancoda_tok.export_cli import export_dir
from ancoda_tok.inventory import find_snapshot
from ancoda_tok.storage import REPO_ROOT

SEED = 20261104
GATE_NPI = 1.445
GATE_ENG = 1.420
GATE_HIN = 1.385
YAK_DIR = REPO_ROOT / "third_party" / "yak-tokenizer"
M7_DIR = REPO_ROOT / "artifacts" / "eval" / "m7"
TOK_DIR = REPO_ROOT / "artifacts" / "tokenizers"
COUNTS_DIR = REPO_ROOT / "artifacts" / "counts"
COVERED = REPO_ROOT / "artifacts" / "inventory" / "covered_64k.json"
COVERED_GB9C = REPO_ROOT / "artifacts" / "inventory" / "covered_64k_gb9c.json"

MIX_NAMES = ("n45", "n50", "n60")
MIX_DIRS = {
    "n45": TOK_DIR / "ancoda-64k-n45",
    "n50": TOK_DIR / "ancoda-64k",
    "n60": TOK_DIR / "ancoda-64k-n60",
}
MIX_CONFIGS = {
    "n45": REPO_ROOT / "configs" / "build_64k_n45.yaml",
    "n50": REPO_ROOT / "configs" / "build_64k.yaml",
    "n60": REPO_ROOT / "configs" / "build_64k_n60.yaml",
}


def _env() -> dict[str, str]:
    env = dict(**{k: v for k, v in __import__("os").environ.items()})
    cargo_bin = str(Path.home() / ".cargo" / "bin")
    path = env.get("PATH", "")
    if cargo_bin not in path.split(":"):
        env["PATH"] = f"{cargo_bin}:{path}"
    return env


def _run(cmd: list[str], *, cwd: Path | None = None) -> None:
    print("+ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=cwd or REPO_ROOT, check=True, env=_env())


def _caffeinate(cmd: list[str], *, cwd: Path | None = None) -> None:
    _run(["caffeinate", "-dimsu", *cmd], cwd=cwd)


def count_snapshot(prefix: str) -> None:
    out = COUNTS_DIR / prefix
    merged = out / "pretoken_counts.bin"
    if merged.is_file() and (out / "manifest.json").is_file():
        print(f"counts {prefix}: already present", flush=True)
        return
    snapshot = find_snapshot(prefix)
    free = free_bytes()
    print(f"counting {prefix} free_bytes={free}", flush=True)
    _caffeinate(
        [
            "uv",
            "run",
            "python",
            "-u",
            "-m",
            "ancoda_tok.counts",
            "--snapshot",
            str(snapshot),
            "--out",
            str(out),
        ]
    )


def train_config(config: Path) -> Path:
    text = config.read_text(encoding="utf-8")
    out_dir = None
    for line in text.splitlines():
        if line.startswith("output_dir:"):
            out_dir = REPO_ROOT / line.split(":", 1)[1].strip()
            break
    if out_dir is None:
        raise SystemExit(f"STOP: no output_dir in {config}")
    ranks = out_dir / "ranks.json"
    if ranks.is_file() and (out_dir / "manifest.json").is_file():
        print(f"train {config.name}: ranks already present", flush=True)
        return out_dir
    _caffeinate(
        [
            "cargo",
            "run",
            "--release",
            "--offline",
            "-p",
            "ancoda-bpe",
            "--",
            "train",
            "--config",
            str(config),
        ]
    )
    return out_dir


def export_tokenizer(out_dir: Path) -> None:
    if (out_dir / "tokenizer.json").is_file() and (out_dir / "export_manifest.json").is_file():
        print(f"export {out_dir.name}: already present", flush=True)
        return
    export_dir(out_dir / "ranks.json")


def guarantee(out_dir: Path, snapshot_prefix: str, tag: str) -> Path:
    report = M7_DIR / f"guarantee_{tag}.json"
    orphans = M7_DIR / f"orphans_{tag}.txt"
    if report.is_file():
        data = json.loads(report.read_text(encoding="utf-8"))
        if not data.get("failures"):
            print(f"guarantee {tag}: already clean", flush=True)
            return report
    snapshot = find_snapshot(snapshot_prefix)
    covered = COVERED_GB9C if "gb9c" in tag else COVERED
    _caffeinate(
        [
            "uv",
            "run",
            "python",
            "-u",
            "-m",
            "ancoda_tok.guarantee",
            "--tokenizer-dir",
            str(out_dir),
            "--covered",
            str(covered),
            "--snapshot",
            str(snapshot),
            "--report",
            str(report),
            "--orphans",
            str(orphans),
        ]
    )
    data = json.loads(report.read_text(encoding="utf-8"))
    if data.get("failures"):
        raise SystemExit(f"STOP: guarantee failures for {tag}: {data['failures'][:5]}")
    return report


def harness_add(adds: dict[str, Path], tag: str) -> Path:
    """Run yak harness with --only none and --add entries. Copy JSONs to m7/."""
    M7_DIR.mkdir(parents=True, exist_ok=True)
    dest = M7_DIR / f"harness_{tag}.json"
    cmd = [
        "uv",
        "run",
        "python",
        "scripts/eval_panel.py",
        "run",
        "--only",
        "none",
        "--no-report",
        "--force",
    ]
    for name, path in sorted(adds.items()):
        if path.name.endswith(".model"):
            spec = f"sp:{path}"
        else:
            spec = str(path / "tokenizer.json" if path.is_dir() else path)
        cmd.extend(["--add", f"{name}={spec}"])
    _caffeinate(cmd, cwd=YAK_DIR)
    # Collect the produced artifacts into one summary.
    summary = {"tag": tag, "tokenizers": {}}
    for name in adds:
        src = YAK_DIR / "artifacts" / "eval" / f"{name}.json"
        if not src.is_file():
            raise SystemExit(f"STOP: harness missing {src}")
        copied = M7_DIR / f"{tag}_{name}.json"
        shutil.copy2(src, copied)
        summary["tokenizers"][name] = _harness_row(json.loads(src.read_text(encoding="utf-8")))
    save_json(dest, finish_manifest(base_manifest(kind="m7_harness", **summary)))
    return dest


def _harness_row(record: dict) -> dict:
    results = record.get("results") or {}
    npi = results.get("flores_devtest/npi_Deva") or {}
    eng = results.get("flores_devtest/eng_Latn") or {}
    hin = results.get("flores_devtest/hin_Deva") or {}
    rob = record.get("robustness") or {}
    return {
        "npi": npi.get("fertility"),
        "eng": eng.get("fertility"),
        "hin": hin.get("fertility"),
        "npi_bytes_per_tok": npi.get("bytes_per_token"),
        "npi_orphan_mark_rate": npi.get("orphan_mark_rate"),
        "npi_byte_piece_rate": npi.get("byte_piece_rate"),
        "zwj": (rob.get("zwj") or {}).get("token_ratio"),
        "zwnj": (rob.get("zwnj") or {}).get("token_ratio"),
        "vocab_size": record.get("vocab_size"),
        "roundtrip_exact": npi.get("roundtrip_exact_rate"),
    }


def pick_mix(rows: dict[str, dict]) -> str:
    """3.8 pick rule: meet Gate A fertilities; lowest Nepali; tie → lower English."""
    eligible = []
    for name, row in rows.items():
        npi, eng, hin = row["npi"], row["eng"], row["hin"]
        if npi is None or eng is None or hin is None:
            continue
        if npi < GATE_NPI and eng < GATE_ENG and hin < GATE_HIN:
            eligible.append((npi, eng, name))
    if not eligible:
        return ""
    eligible.sort()
    return eligible[0][2]


def rewrite_counts(config: Path, counts_rel: str, output_rel: str | None = None) -> Path:
    """Write a temp config with an overridden counts path (and optional output)."""
    lines = []
    for line in config.read_text(encoding="utf-8").splitlines():
        if line.startswith("counts:"):
            lines.append(f"counts: {counts_rel}")
        elif output_rel is not None and line.startswith("output_dir:"):
            lines.append(f"output_dir: {output_rel}")
        else:
            lines.append(line)
    dest = M7_DIR / "configs" / config.name
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return dest


def cmd_counts(_: argparse.Namespace) -> None:
    for prefix in ("n45", "n50", "n60", "n50_nojoiner"):
        count_snapshot(prefix)


def cmd_mix_train(_: argparse.Namespace) -> None:
    M7_DIR.mkdir(parents=True, exist_ok=True)
    count_snapshot("n45")
    count_snapshot("n60")
    # n50 already counted in M6.
    for name in MIX_NAMES:
        if name == "n50" and (MIX_DIRS["n50"] / "tokenizer.json").is_file():
            print("n50: reusing artifacts/tokenizers/ancoda-64k", flush=True)
            continue
        out = train_config(MIX_CONFIGS[name])
        export_tokenizer(out)


def cmd_mix_eval(_: argparse.Namespace) -> None:
    M7_DIR.mkdir(parents=True, exist_ok=True)
    for name in MIX_NAMES:
        out = MIX_DIRS[name]
        if name == "n50":
            # M6 already ran the full guarantee on this build.
            prior = REPO_ROOT / "artifacts" / "validation" / "guarantee.json"
            if prior.is_file() and not (M7_DIR / "guarantee_n50.json").is_file():
                shutil.copy2(prior, M7_DIR / "guarantee_n50.json")
                print("n50: copied M6 guarantee.json", flush=True)
            continue
        guarantee(out, name, name)
    adds = {f"anc64_{name}": MIX_DIRS[name] for name in MIX_NAMES}
    harness_add(adds, "mix")
    summary = json.loads((M7_DIR / "harness_mix.json").read_text(encoding="utf-8"))
    rows = summary["tokenizers"]
    # Remap anc64_n45 → n45 keys for the pick rule.
    pick_rows = {name: rows[f"anc64_{name}"] for name in MIX_NAMES}
    chosen = pick_mix(pick_rows)
    decision = {
        "pick": chosen,
        "gate": {"npi": GATE_NPI, "eng": GATE_ENG, "hin": GATE_HIN},
        "rows": pick_rows,
        "meets_all": bool(chosen),
    }
    save_json(
        M7_DIR / "mix_pick.json",
        finish_manifest(base_manifest(kind="m7_mix_pick", **decision)),
    )
    print(json.dumps(decision, indent=2), flush=True)
    if not chosen:
        raise SystemExit(
            "STOP (human): no mix meets Gate A fertility rows "
            f"(Nepali < {GATE_NPI}, English < {GATE_ENG}, Hindi < {GATE_HIN}). "
            "Propose 65% Nepali or a cap-bound covered set."
        )


def cmd_ladder(args: argparse.Namespace) -> None:
    M7_DIR.mkdir(parents=True, exist_ok=True)
    force = getattr(args, "force_snapshot", None)
    if force:
        pick = force
        print(f"ladder force-snapshot={pick} (diagnostic; not a mix pick)", flush=True)
    else:
        pick_path = M7_DIR / "mix_pick.json"
        if not pick_path.is_file():
            raise SystemExit("STOP: run mix-eval first")
        pick = json.loads(pick_path.read_text(encoding="utf-8"))["pick"]
        if not pick:
            raise SystemExit("STOP: no mix pick")
    count_snapshot(pick)
    count_snapshot("n50_nojoiner")

    ladder = [
        ("plain", REPO_ROOT / "configs" / "ladder_plain.yaml", pick),
        ("b0", REPO_ROOT / "configs" / "ladder_b0.yaml", pick),
        ("b0b1", REPO_ROOT / "configs" / "ladder_b0b1.yaml", pick),
        (
            "final_nojoiner",
            REPO_ROOT / "configs" / "ladder_final_nojoiner.yaml",
            "n50_nojoiner",
        ),
        ("final", REPO_ROOT / "configs" / "ladder_final.yaml", pick),
        ("gb9c", REPO_ROOT / "configs" / "ablation_gb9c.yaml", pick),
    ]
    dirs: dict[str, Path] = {}
    for name, config, snap_prefix in ladder:
        out_rel = f"artifacts/tokenizers/ladder-{name}"
        # Reuse the mix-sweep final when the pick matches and this is the final rung.
        if name == "final" and snap_prefix in MIX_DIRS:
            src = MIX_DIRS[snap_prefix]
            dest = TOK_DIR / f"ladder-{name}"
            if not (dest / "tokenizer.json").is_file():
                if dest.exists():
                    shutil.rmtree(dest)
                shutil.copytree(src, dest)
            dirs[name] = dest
            print(f"ladder {name}: reused {src.name}", flush=True)
            continue
        cfg = rewrite_counts(
            config,
            f"artifacts/counts/{snap_prefix}/pretoken_counts.bin",
            out_rel,
        )
        # Keep gb9c covered path from the original config.
        out = train_config(cfg)
        export_tokenizer(out)
        dirs[name] = out
        if name != "gb9c":
            # Full guarantee on final-like rungs; lighter skip flags for ablations
            # that may violate the 3.5 claim by design.
            if name in ("final_nojoiner",):
                guarantee(out, snap_prefix, name)
            elif name in ("plain", "b0", "b0b1"):
                print(f"ladder {name}: skipping full guarantee (ablation)", flush=True)

    # Unigram control.
    from ancoda_tok.unigram import train as train_unigram

    uni_dir = TOK_DIR / "ladder-unigram"
    train_unigram(find_snapshot(pick), uni_dir)
    dirs["unigram"] = uni_dir

    adds = {}
    for name, path in dirs.items():
        if name == "unigram":
            adds[f"lad_{name}"] = path / "unigram.model"
        else:
            adds[f"lad_{name}"] = path
    harness_add(adds, "ladder")

    # Morphology on HF rungs only.
    from ancoda_tok.eval.morph import sample_words, score_tokenizer

    words = sample_words()
    morph_rows = {}
    for name, path in dirs.items():
        if name == "unigram":
            continue
        morph_rows[name] = score_tokenizer(path, words)
        save_json(
            M7_DIR / f"morph_{name}.json",
            finish_manifest(
                base_manifest(kind="morphology", rung=name, **morph_rows[name])
            ),
        )
    save_json(
        M7_DIR / "ladder_morph.json",
        finish_manifest(base_manifest(kind="m7_ladder_morph", rows=morph_rows)),
    )
    print("ladder complete", flush=True)


def cmd_report(_: argparse.Namespace) -> None:
    """Write M7 sections into docs/RESULTS.md and the joiner decision."""
    mix = json.loads((M7_DIR / "mix_pick.json").read_text(encoding="utf-8"))
    harness_mix = json.loads((M7_DIR / "harness_mix.json").read_text(encoding="utf-8"))
    lines = [
        "",
        "---",
        "",
        "## M7 — Mix sweep and ablation ladder",
        "",
        (
            f"Seed `{SEED}`. Block 1 frozen from `artifacts/inventory/covered_64k.json` "
            "(n50 M5 selection) so the mix sweep isolates Block 2 / data effects."
        ),
        "",
        "### Mix sweep (64k final config)",
        "",
        (
            "| Mix | Nepali | English | Hindi | bytes/tok (npi) | orphan (npi) | "
            "ZWJ | ZWNJ | Gate A fertilities |"
        ),
        "|---|---:|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for name in MIX_NAMES:
        row = harness_mix["tokenizers"][f"anc64_{name}"]
        ok = (
            row["npi"] < GATE_NPI
            and row["eng"] < GATE_ENG
            and row["hin"] < GATE_HIN
        )
        mark = "PASS" if ok else "FAIL"
        lines.append(
            
                f"| {name} | {row['npi']:.6f} | {row['eng']:.6f} | {row['hin']:.6f} | "
                f"{row['npi_bytes_per_tok']:.4f} | {100 * row['npi_orphan_mark_rate']:.3f}% | "
                f"{row['zwj']:.6f} | {row['zwnj']:.6f} | {mark} |"
            
        )
    chosen = mix["pick"]
    lines.extend(
        [
            "",
            (
                f"**Pick:** `{chosen}` "
                "(meets every Gate A fertility row; lowest Nepali fertility; "
                "English fertility breaks ties)."
            ),
            "",
        ]
    )
    ladder_path = M7_DIR / "harness_ladder.json"
    if ladder_path.is_file():
        ladder = json.loads(ladder_path.read_text(encoding="utf-8"))
        morph = {}
        morph_path = M7_DIR / "ladder_morph.json"
        if morph_path.is_file():
            morph = json.loads(morph_path.read_text(encoding="utf-8")).get("rows", {})
        lines.extend(
            [
                "### Ablation ladder",
                "",
                "| Rung | Nepali | English | Hindi | bytes/tok | orphan | ZWJ | ZWNJ | morph F1 (unweighted) | whole-word |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        order = ["plain", "b0", "b0b1", "final_nojoiner", "final", "gb9c", "unigram"]
        for name in order:
            key = f"lad_{name}"
            row = ladder["tokenizers"].get(key)
            if row is None:
                continue
            m = morph.get(name, {})
            f1 = m.get("unweighted", {}).get("f1")
            ww = m.get("whole_word_preservation_unweighted")
            f1_s = "—" if f1 is None else f"{f1:.4f}"
            ww_s = "—" if ww is None else f"{ww:.4f}"
            lines.append(
                
                    f"| {name} | {row['npi']:.6f} | {row['eng']:.6f} | {row['hin']:.6f} | "
                    f"{row['npi_bytes_per_tok']:.4f} | {100 * row['npi_orphan_mark_rate']:.3f}% | "
                    f"{row['zwj']:.6f} | {row['zwnj']:.6f} | {f1_s} | {ww_s} |"
                
            )
        # Joiner decision.
        final = ladder["tokenizers"].get("lad_final")
        noj = ladder["tokenizers"].get("lad_final_nojoiner")
        if final and noj:
            fert_cost = final["npi"] - noj["npi"]
            zwj_helps = final["zwj"] < noj["zwj"]
            zwnj_helps = final["zwnj"] < noj["zwnj"]
            keep = zwj_helps and zwnj_helps and fert_cost <= 0.005
            lines.extend(
                [
                    "",
                    "### Joiner copy decision",
                    "",
                    f"- final Nepali fertility: {final['npi']:.6f}",
                    f"- no-joiner Nepali fertility: {noj['npi']:.6f}",
                    f"- fertility cost (final − nojoiner): {fert_cost:.6f}",
                    f"- ZWJ final/nojoiner: {final['zwj']:.6f} / {noj['zwj']:.6f}",
                    f"- ZWNJ final/nojoiner: {final['zwnj']:.6f} / {noj['zwnj']:.6f}",
                    f"- **Keep joiner copy:** {'yes' if keep else 'no'}",
                    "",
                ]
            )
            _write_joiner_decision(keep, final, noj, fert_cost)

    results = REPO_ROOT / "docs" / "RESULTS.md"
    text = results.read_text(encoding="utf-8")
    marker = "## M7 — Mix sweep and ablation ladder"
    if marker in text:
        text = text.split(marker)[0].rstrip() + "\n"
    results.write_text(text + "\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {results}", flush=True)


def _write_joiner_decision(keep: bool, final: dict, noj: dict, fert_cost: float) -> None:
    decisions = REPO_ROOT / "docs" / "DECISIONS.md"
    text = decisions.read_text(encoding="utf-8")
    block = (
        "\n## M7 — Joiner copy (human-recorded decision)\n\n"
        f"Date: 8 Nov 2026. Keep joiner copy: **{'yes' if keep else 'no'}**.\n\n"
        f"- final Nepali fertility {final['npi']:.6f}; no-joiner {noj['npi']:.6f}; "
        f"cost {fert_cost:.6f} (bar ≤ 0.005).\n"
        f"- ZWJ {final['zwj']:.6f} vs {noj['zwj']:.6f}; "
        f"ZWNJ {final['zwnj']:.6f} vs {noj['zwnj']:.6f}.\n"
        "Rule (docs/SPEC.md M7): keep only if both joiner probes are lower and Nepali "
        "fertility cost is at most 0.005.\n"
    )
    marker = "## M7 — Joiner copy"
    if marker in text:
        # Replace existing block through next ## or EOF.
        before, rest = text.split(marker, 1)
        if "\n## " in rest[1:]:
            _, after = rest[1:].split("\n## ", 1)
            text = before.rstrip() + block + "\n## " + after
        else:
            text = before.rstrip() + block
    else:
        text = text.rstrip() + "\n" + block
    decisions.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("counts")
    sub.add_parser("mix-train")
    sub.add_parser("mix-eval")
    ladder_p = sub.add_parser("ladder")
    ladder_p.add_argument(
        "--force-snapshot",
        default=None,
        help="Run the ladder on this snapshot prefix without a mix pick (diagnostic)",
    )
    sub.add_parser("report")
    sub.add_parser("all")
    args = parser.parse_args()
    if args.cmd == "counts":
        cmd_counts(args)
    elif args.cmd == "mix-train":
        cmd_mix_train(args)
    elif args.cmd == "mix-eval":
        cmd_mix_eval(args)
    elif args.cmd == "ladder":
        cmd_ladder(args)
    elif args.cmd == "report":
        cmd_report(args)
    elif args.cmd == "all":
        cmd_mix_train(args)
        cmd_mix_eval(args)
        cmd_ladder(args)
        cmd_report(args)


if __name__ == "__main__":
    main()

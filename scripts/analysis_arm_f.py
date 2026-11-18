"""Arm F minus arm A, and M9 training hours. Analysis only.

Reads eval JSON already pulled to artifacts/m9/server_copy. Does not train and
does not touch the server queue. The decision rule is the 16 Nov 2026 amendment
in docs/M9_PREREG.md: difference is BPB(F) − BPB(A).
"""

from __future__ import annotations

import json
import re
import statistics
from datetime import datetime
from pathlib import Path

from ancoda_tok.m9.common import DEVA11, FULL_STEPS, FULL_STREAM_TOKENS

REPO = Path(__file__).resolve().parents[1]
EVAL = REPO / "artifacts" / "m9" / "server_copy" / "evals"
RUNS = REPO / "artifacts" / "m9" / "server_copy" / "runs"
BSTAR = REPO / "artifacts" / "m9" / "server_copy" / "bstar.json"
OUT_JSON = REPO / "artifacts" / "analysis" / "arm_f.json"
OUT_MD = REPO / "docs" / "ANALYSIS_ARM_F.md"

ARM_A = "ancoda-64k"
ARM_F = "ancoda-64k-nojoiner"
SEEDS = (1, 2, 3)
HOURS_ARMS = (
    ("A", "ancoda-64k"),
    ("B", "yak-64k-v1"),
    ("C", "sidskarki-64k"),
    ("D", "plain-64k"),
    ("E", "yak-64k-v2.0"),
    ("F", "ancoda-64k-nojoiner"),
)
LOG_BSTAR = re.compile(r"^B\*=(\d+) first_step=(\d+)$")
MAIN = (
    ("sealed_npi_zwj", "sealed_npi_zwj.jsonl:Nepali"),
    ("sealed_npi_zwnj", "sealed_npi_zwnj.jsonl:Nepali"),
)
CHECKPOINTS = (("final", "equal tokens"), ("bstar", "equal bytes"))


def _self_check() -> None:
    assert verdict([-0.2, -0.2, -0.2]) == "F better"
    assert verdict([0.2, 0.2, 0.2]) == "F worse"
    assert verdict([-0.2, 0.2, -0.2]) == "no measurable difference"
    assert verdict([0.0, -0.2, -0.2]) == "no measurable difference"
    # All negative, but the mean is not larger than the sample SD.
    assert verdict([-1.0, -0.01, -0.01]) == "no measurable difference"


def verdict(diffs: list[float]) -> str:
    """Two-sided F−A rule. Not the one-sided paired_call used for arm A."""
    if len(diffs) != 3:
        raise ValueError(diffs)
    mean = statistics.fmean(diffs)
    sd = statistics.stdev(diffs)
    if all(diff < 0 for diff in diffs) and abs(mean) > sd:
        return "F better"
    if all(diff > 0 for diff in diffs) and abs(mean) > sd:
        return "F worse"
    return "no measurable difference"


def load_json(path: Path) -> dict:
    if not path.is_file():
        raise SystemExit(f"STOP: missing {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def parse_time(text: str) -> datetime:
    return datetime.fromisoformat(text)


def bpb_at(payload: dict, key: str) -> float:
    try:
        return float(payload["groups"][key]["bpb"])
    except KeyError as exc:
        raise SystemExit(f"STOP: {payload.get('checkpoint')} has no group {key}") from exc


def dev11(payload: dict) -> float:
    return statistics.fmean(bpb_at(payload, f"sealed.jsonl:{lang}") for lang in DEVA11)


def log_bstar(arm: str, seed: int) -> tuple[int, int]:
    path = RUNS / f"{arm}-s{seed}" / "train.log"
    if not path.is_file():
        raise SystemExit(f"STOP: missing {path}")
    line = path.read_text(encoding="utf-8").splitlines()[0]
    match = LOG_BSTAR.match(line)
    if not match:
        raise SystemExit(f"STOP: {path} does not start with a B* line: {line!r}")
    return int(match.group(1)), int(match.group(2))


def check_eval(arm: str, seed: int, kind: str, expected_step: int) -> dict:
    path = EVAL / f"{arm}-s{seed}-{kind}.json"
    payload = load_json(path)
    step = int(payload["completed_steps"])
    checkpoint = str(payload["checkpoint"])
    if step != expected_step:
        raise SystemExit(
            f"STOP: {path.name} completed_steps={step}, train log expects {expected_step}"
        )
    if kind == "final":
        name = "checkpoint-007630.pt"
    else:
        name = f"checkpoint-bstar-{expected_step:06d}.pt"
    if not checkpoint.endswith(f"/{arm}-s{seed}/{name}"):
        raise SystemExit(f"STOP: {path.name} checkpoint is {checkpoint}, expected {name}")
    if payload.get("arm") != arm:
        raise SystemExit(f"STOP: {path.name} arm is {payload.get('arm')}")
    return payload


def check_pair(seed: int, registered_bstar: int) -> dict:
    """F and A evals against each run's own train.log. B* bytes are not recomputed."""
    f_bytes, f_step = log_bstar(ARM_F, seed)
    a_bytes, a_step = log_bstar(ARM_A, seed)
    if f_bytes != registered_bstar or a_bytes != registered_bstar:
        raise SystemExit(
            f"STOP: seed {seed} train.log B* F={f_bytes} A={a_bytes}, registered {registered_bstar}"
        )
    f_manifest = load_json(RUNS / f"{ARM_F}-s{seed}" / "manifest.json")
    a_manifest = load_json(RUNS / f"{ARM_A}-s{seed}" / "manifest.json")
    for label, manifest in ((ARM_F, f_manifest), (ARM_A, a_manifest)):
        if int(manifest["completed_steps"]) != FULL_STEPS:
            raise SystemExit(f"STOP: {label}-s{seed} manifest steps {manifest['completed_steps']}")
        if int(manifest["tokens_seen"]) != FULL_STREAM_TOKENS:
            raise SystemExit(
                f"STOP: {label}-s{seed} tokens_seen {manifest['tokens_seen']} != {FULL_STREAM_TOKENS}"
            )
    out = {"seed": seed, "bstar_bytes": registered_bstar, "F": {}, "A": {}}
    for arm, bstar_step, slot in ((ARM_F, f_step, "F"), (ARM_A, a_step, "A")):
        final = check_eval(arm, seed, "final", FULL_STEPS)
        bstar = check_eval(arm, seed, "bstar", bstar_step)
        manifest = f_manifest if arm == ARM_F else a_manifest
        if parse_time(final["finished_at"]) < parse_time(manifest["finished_at"]):
            raise SystemExit(f"STOP: {arm}-s{seed} final eval finished before training")
        out[slot] = {
            "bstar_step": bstar_step,
            "final_git": final.get("git_commit"),
            "bstar_git": bstar.get("git_commit"),
            "manifest_git": manifest.get("git_commit"),
            "train_finished_at": manifest["finished_at"],
            "final_finished_at": final["finished_at"],
            "final": final,
            "bstar": bstar,
        }
    return out


def metric_value(payload: dict, name: str) -> float:
    if name == "Dev11":
        return dev11(payload)
    key = {"Nepali": "sealed.jsonl:Nepali", **dict(MAIN)}[name]
    return bpb_at(payload, key)


def compare(pairs: list[dict], kind: str, metric: str) -> dict:
    diffs = []
    per_seed = []
    for pair in pairs:
        f_bpb = metric_value(pair["F"][kind], metric)
        a_bpb = metric_value(pair["A"][kind], metric)
        diff = f_bpb - a_bpb
        diffs.append(diff)
        per_seed.append({"F": f_bpb, "A": a_bpb, "diff": diff})
    return {
        "metric": metric,
        "diffs": diffs,
        "per_seed": per_seed,
        "mean": statistics.fmean(diffs),
        "sample_sd": statistics.stdev(diffs),
        "verdict": verdict(diffs),
    }


def training_hours() -> list[dict]:
    rows = []
    for letter, arm in HOURS_ARMS:
        for seed in SEEDS:
            manifest = load_json(RUNS / f"{arm}-s{seed}" / "manifest.json")
            start = parse_time(manifest["started_at"])
            end = parse_time(manifest["finished_at"])
            seconds = (end - start).total_seconds()
            if seconds <= 0:
                raise SystemExit(f"STOP: {arm}-s{seed} finished before it started")
            rows.append(
                {
                    "letter": letter,
                    "arm": arm,
                    "seed": seed,
                    "started_at": manifest["started_at"],
                    "finished_at": manifest["finished_at"],
                    "hours": seconds / 3600.0,
                }
            )
    return rows


def fmt(value: float) -> str:
    return f"{value:.6f}"


def render(checks: list[dict], comparisons: dict, hours: list[dict]) -> str:
    lines = [
        "# Arm F, seed 3",
        "",
        "Analysis only. The server queue was not changed. Eval JSON and run",
        "manifests were pulled with rsync. The decision rule is the 16 Nov 2026",
        "amendment in `docs/M9_PREREG.md`. Difference is BPB(F) − BPB(A). Lower BPB",
        "is better. **F better** requires every seed difference strictly negative and",
        "|mean| > sample SD (divisor n−1). **F worse** requires every difference",
        "strictly positive and the same mean test. Otherwise **no measurable",
        "difference**. This is not the one-sided arm-A rule.",
        "",
        "Equal tokens is step 7630. Equal bytes is each seed's existing B* from",
        "`bstar.json` (the core A/B/C byte target). B* was not recomputed. The F",
        "checkpoint step is the first step whose covered bytes reach that target,",
        "read from the first line of that run's `train.log` and checked against the",
        "eval JSON.",
        "",
        "## Log check",
        "",
        "| Seed | Registered B* bytes | F step | A step | F final eval | F manifest |",
        "|---:|---:|---:|---:|---|---|",
    ]
    for pair in checks:
        slot = pair["F"]
        lines.append(
            f"| {pair['seed']} | {pair['bstar_bytes']} | {slot['bstar_step']} | "
            f"{pair['A']['bstar_step']} | {slot['final_finished_at']} | "
            f"{slot['train_finished_at']} |"
        )
    lines += [
        "",
        "Each final eval is `checkpoint-007630.pt` with `completed_steps` 7630.",
        "Each B* eval's checkpoint name and `completed_steps` match that run's",
        "`first_step`. Both arms' train logs carry the registered B* byte count.",
        "Manifest `tokens_seen` is 1,000,079,360 and `completed_steps` is 7630.",
        "Each final eval's `finished_at` is after that run's manifest `finished_at`.",
        "",
        "## Main comparison",
        "",
        "Pre-registered prediction: F is better than A on both joiner sets. If it",
        "is not, the joiner-copy explanation is weakened.",
        "",
    ]
    for kind, label in CHECKPOINTS:
        lines += [
            f"### {label}",
            "",
            "| Set | d1 | d2 | d3 | mean | sample SD | verdict |",
            "|---|---:|---:|---:|---:|---:|---|",
        ]
        for metric, _key in MAIN:
            row = comparisons[kind][metric]
            lines.append(
                f"| `{metric}` | {fmt(row['diffs'][0])} | {fmt(row['diffs'][1])} | "
                f"{fmt(row['diffs'][2])} | {fmt(row['mean'])} | {fmt(row['sample_sd'])} | "
                f"**{row['verdict']}** |"
            )
        lines.append("")
        lines += [
            "| Seed | F ZWJ | A ZWJ | F ZWNJ | A ZWNJ |",
            "|---:|---:|---:|---:|---:|",
        ]
        for index, pair in enumerate(checks):
            zwj = comparisons[kind]["sealed_npi_zwj"]["per_seed"][index]
            zwnj = comparisons[kind]["sealed_npi_zwnj"]["per_seed"][index]
            lines.append(
                f"| {pair['seed']} | {fmt(zwj['F'])} | {fmt(zwj['A'])} | "
                f"{fmt(zwnj['F'])} | {fmt(zwnj['A'])} |"
            )
        lines.append("")
    met = all(
        comparisons[kind][metric]["verdict"] == "F better"
        for kind, _label in CHECKPOINTS
        for metric, _key in MAIN
    )
    lines += [
        (
            "The prediction is met: F is better on both joiner sets at both checkpoints."
            if met
            else "The prediction is not met, so the joiner-copy explanation is weakened."
        ),
        "",
        "## Secondary comparison",
        "",
        "F minus A on sealed Nepali and the unweighted mean of the 11 sealed",
        "Devanagari groups, at both checkpoints. The same decision rule is applied.",
        "No prediction is attached.",
        "",
    ]
    for kind, label in CHECKPOINTS:
        lines += [
            f"### {label}",
            "",
            "| Set | d1 | d2 | d3 | mean | sample SD | verdict |",
            "|---|---:|---:|---:|---:|---:|---|",
        ]
        for metric in ("Nepali", "Dev11"):
            row = comparisons[kind][metric]
            lines.append(
                f"| {metric} | {fmt(row['diffs'][0])} | {fmt(row['diffs'][1])} | "
                f"{fmt(row['diffs'][2])} | {fmt(row['mean'])} | {fmt(row['sample_sd'])} | "
                f"**{row['verdict']}** |"
            )
        lines.append("")
    lines += [
        "## Training hours",
        "",
        "Elapsed time from `started_at` to `finished_at` in each run manifest.",
        "This is wall time for that training process, including whatever the",
        "manifest recorded between those two timestamps.",
        "",
        "| Arm | Seed | Started | Finished | Hours |",
        "|---|---:|---|---|---:|",
    ]
    total = 0.0
    for row in hours:
        total += row["hours"]
        lines.append(
            f"| {row['letter']} {row['arm']} | {row['seed']} | {row['started_at']} | "
            f"{row['finished_at']} | {row['hours']:.2f} |"
        )
    lines.append(f"| sum of the 18 runs |  |  |  | {total:.2f} |")
    lines += ["", "JSON: `artifacts/analysis/arm_f.json`.", ""]
    return "\n".join(lines)


def main() -> None:
    _self_check()
    registered = load_json(BSTAR)
    pairs = [check_pair(seed, int(registered[str(seed)]["bstar"])) for seed in SEEDS]
    comparisons: dict[str, dict] = {}
    for kind, _label in CHECKPOINTS:
        comparisons[kind] = {}
        for metric, _key in MAIN:
            comparisons[kind][metric] = compare(pairs, kind, metric)
        for metric in ("Nepali", "Dev11"):
            comparisons[kind][metric] = compare(pairs, kind, metric)
    hours = training_hours()
    # Drop the full eval payloads before writing the JSON.
    public_checks = []
    for pair in pairs:
        public_checks.append(
            {
                "seed": pair["seed"],
                "bstar_bytes": pair["bstar_bytes"],
                "F": {
                    key: value for key, value in pair["F"].items() if key not in ("final", "bstar")
                },
                "A": {
                    key: value for key, value in pair["A"].items() if key not in ("final", "bstar")
                },
            }
        )
    payload = {
        "kind": "analysis_arm_f",
        "rule": "BPB(F)-BPB(A); F better / F worse / no measurable difference",
        "checks": public_checks,
        "comparisons": comparisons,
        "training_hours": hours,
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    OUT_MD.write_text(render(pairs, comparisons, hours), encoding="utf-8")
    print(f"wrote {OUT_MD}")


if __name__ == "__main__":
    main()

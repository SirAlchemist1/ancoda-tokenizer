"""Gate A printer (M1).

Reads Yak harness artifact JSON for a named panel tokenizer and prints:
- docs/SPEC.md 3.10 rows with PASS/FAIL (harness-backed rows only in M1)
- Yak scoreboard for yak-64k-v1 and yak-64k-v2.0 (WIN/TIE/LOSE)

Ancoda-only metrics (covered violations, mark-initial vocab, equality, segmenter
agreement, encoding-variant cost, etc.) are filled in M8.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_EVAL_YAML = REPO_ROOT / "configs" / "eval.yaml"
DEFAULT_ARTIFACTS = REPO_ROOT / "third_party" / "yak-tokenizer" / "artifacts" / "eval"


def _load_yaml(path: Path) -> dict:
    with path.open() as f:
        return yaml.safe_load(f)


def _load_artifact(artifacts_dir: Path, name: str) -> dict:
    path = artifacts_dir / f"{name}.json"
    if not path.is_file():
        raise FileNotFoundError(f"missing harness artifact: {path}")
    data = json.loads(path.read_text())
    if data.get("status") != "ok":
        raise RuntimeError(f"artifact {name} status={data.get('status')!r}: {data.get('error')}")
    return data


def _fert(data: dict, key: str) -> float:
    return float(data["results"][key]["fertility"])


def _orphan(data: dict, key: str) -> float:
    return float(data["results"][key]["orphan_mark_rate"])


def _byte_pieces(data: dict, key: str) -> float:
    return float(data["results"][key]["byte_piece_rate"])


def _roundtrip(data: dict, key: str) -> float:
    return float(data["results"][key]["roundtrip_rate"])


def _coverage_128(data: dict) -> int:
    block = data.get("coverage", {}).get("devanagari_block", {})
    share = block.get("covered_share")
    if share is None:
        return -1
    return round(float(share) * 128)


def _probe(data: dict, which: str) -> float:
    return float(data["robustness"][which]["token_ratio"])


def _cmp(label: str, ours: float, bar: float, mode: str) -> str:
    if mode == "below":
        ok = ours < bar
    elif mode == "at_most":
        ok = ours <= bar
    elif mode == "eq":
        ok = ours == bar
    else:
        raise ValueError(mode)
    status = "PASS" if ok else "FAIL"
    return f"{status}\t{label}\tours={ours:.6f}\tbar={bar:.6f}\tmode={mode}"


def _scoreboard_verdict(ours: float, yak: float, lower_is_better: bool = True) -> str:
    # Exact float compare after shared harness path; treat tiny noise as TIE at 1e-12
    if abs(ours - yak) <= 1e-12:
        return "TIE"
    if lower_is_better:
        return "WIN" if ours < yak else "LOSE"
    return "WIN" if ours > yak else "LOSE"


def print_gate_a(name: str, eval_yaml: Path, artifacts_dir: Path) -> int:
    cfg = _load_yaml(eval_yaml)
    bars = cfg["gate_a_bars"]
    ours = _load_artifact(artifacts_dir, name)
    yak_v1 = _load_artifact(artifacts_dir, "yak-64k-v1")
    yak_v2 = _load_artifact(artifacts_dir, "yak-64k-v2.0")

    npi = "flores_devtest/npi_Deva"
    eng = "flores_devtest/eng_Latn"
    hin = "flores_devtest/hin_Deva"
    aya = "npi_everyday_aya"

    print(f"# Gate A — tokenizer={name}")
    print("## 3.10 rows (harness-backed; Ancoda-only rows PENDING until M8)")
    rows = [
        _cmp(
            "Nepali fertility, whole-file",
            _fert(ours, npi),
            float(bars["nepali_fertility_below"]),
            "below",
        ),
        _cmp(
            "English fertility",
            _fert(ours, eng),
            float(bars["english_fertility_below"]),
            "below",
        ),
        _cmp(
            "Hindi fertility",
            _fert(ours, hin),
            float(bars["hindi_fertility_below"]),
            "below",
        ),
        _cmp(
            "Orphan-mark share, Nepali FLORES+",
            _orphan(ours, npi),
            float(bars["orphan_mark_npi_below"]),
            "below",
        ),
        (
            f"REPORT\tOrphan-mark share, everyday Nepali (Aya)\tours={_orphan(ours, aya):.6f}"
        ),
        _cmp(
            "Exact round-trip (npi)",
            _roundtrip(ours, npi),
            1.0,
            "eq",
        ),
        (
            f"{'PASS' if _coverage_128(ours) == 128 else 'FAIL'}\t"
            f"Whole-character coverage U+0900–U+097F\tours={_coverage_128(ours)}/128\tbar=128/128"
        ),
        _cmp(
            "Nepali byte pieces",
            _byte_pieces(ours, npi),
            float(bars["byte_pieces_npi_at_most"]),
            "at_most",
        ),
        _cmp(
            "ZWJ probe",
            _probe(ours, "zwj"),
            float(bars["zwj_probe_below"]),
            "below",
        ),
        _cmp(
            "ZWNJ probe",
            _probe(ours, "zwnj"),
            float(bars["zwnj_probe_below"]),
            "below",
        ),
        "PENDING\tCovered-akshara violations\t(M6/M8)",
        "PENDING\tMark-initial vocab entries\t(M6/M8)",
        "PENDING\tCovered-unit integrity per export\t(M6/M8)",
        "PENDING\tNewline gap\t(M8)",
        "PENDING\tHugging Face IDs == tiktoken IDs\t(M6/M8)",
        "PENDING\tSegmenter agreement\t(M3/M8)",
        "PENDING\tUnknown tokens == 0\t(M8)",
    ]
    for line in rows:
        print(line)

    # Stretch
    print("## Stretch (beat Yak v1 64k English/Hindi)")
    print(
        _cmp(
            "English fertility stretch",
            _fert(ours, eng),
            float(bars["stretch_english_below"]),
            "below",
        )
    )
    print(
        _cmp(
            "Hindi fertility stretch",
            _fert(ours, hin),
            float(bars["stretch_hindi_below"]),
            "below",
        )
    )

    print("## Yak scoreboard (lower fertility / rates better unless noted)")
    print(
        "metric\tours\tyak-64k-v1\tverdict_v1\tyak-64k-v2.0\tverdict_v2"
    )
    score_rows: list[tuple[str, float, float, float]] = [
        ("Nepali fertility", _fert(ours, npi), _fert(yak_v1, npi), _fert(yak_v2, npi)),
        ("English fertility", _fert(ours, eng), _fert(yak_v1, eng), _fert(yak_v2, eng)),
        ("Hindi fertility", _fert(ours, hin), _fert(yak_v1, hin), _fert(yak_v2, hin)),
        ("Aya fertility", _fert(ours, aya), _fert(yak_v1, aya), _fert(yak_v2, aya)),
        (
            "Orphan-mark npi",
            _orphan(ours, npi),
            _orphan(yak_v1, npi),
            _orphan(yak_v2, npi),
        ),
        ("ZWJ probe", _probe(ours, "zwj"), _probe(yak_v1, "zwj"), _probe(yak_v2, "zwj")),
        (
            "ZWNJ probe",
            _probe(ours, "zwnj"),
            _probe(yak_v1, "zwnj"),
            _probe(yak_v2, "zwnj"),
        ),
        (
            "Byte pieces npi",
            _byte_pieces(ours, npi),
            _byte_pieces(yak_v1, npi),
            _byte_pieces(yak_v2, npi),
        ),
    ]
    for metric, o, v1, v2 in score_rows:
        print(
            f"{metric}\t{o:.6f}\t{v1:.6f}\t{_scoreboard_verdict(o, v1)}\t"
            f"{v2:.6f}\t{_scoreboard_verdict(o, v2)}"
        )
    print(
        "Languages of Nepal mean\tnot comparable\t"
        "(Yak private 14-set; see configs/eval.yaml m1.languages_of_nepal_note)"
    )
    print(
        "Phrase builds\tseparate Gate P category\t"
        "(never compared against word-level Gate A)"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Print Gate A table and Yak scoreboard")
    parser.add_argument("--name", required=True, help="Panel tokenizer name (harness artifact)")
    parser.add_argument("--eval-yaml", type=Path, default=DEFAULT_EVAL_YAML)
    parser.add_argument("--artifacts-dir", type=Path, default=DEFAULT_ARTIFACTS)
    args = parser.parse_args(argv)
    return print_gate_a(args.name, args.eval_yaml, args.artifacts_dir)


if __name__ == "__main__":
    sys.exit(main())

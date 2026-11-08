"""Unit tests for PLAN.md 3.6 Block 1 selection."""

from __future__ import annotations

from collections import Counter

import ancoda_seg

from ancoda_tok.inventory import (
    CAPS,
    chain_prefixes,
    is_eligible,
    is_n0_unit,
    select_block1,
    write_coverage_svg,
)


def test_chain_prefixes() -> None:
    assert chain_prefixes("कु") == ["कु"]
    assert chain_prefixes("कुर") == ["कु", "कुर"]
    assert chain_prefixes("क") == []


def test_eligibility_and_n0() -> None:
    letters = frozenset(int(x) for x in ancoda_seg.devanagari_letters())
    assert is_n0_unit("क")
    assert is_n0_unit("्")
    assert is_eligible("कु", letters)
    assert is_eligible("क्", letters)
    # Mark-initial is not eligible.
    assert not is_eligible("्क", letters)
    # Non–Block 0 mark (combining acute) makes the unit ineligible.
    assert not is_eligible("क\u0301", letters)


def test_select_block1_cap_and_order(tmp_path) -> None:
    letters = frozenset(int(x) for x in ancoda_seg.devanagari_letters())
    # Synthetic inventory: high-count bigram, then a trigram that shares a prefix.
    counts = Counter(
        {
            "क": 1000,  # N0
            "कु": 50,
            "की": 40,
            "कुर": 30,  # chain कु, कुर — reuses कु once selected
            "xyzक": 5,  # Devanagari-bearing but not letter-initial → ineligible
        }
    )
    out = select_block1(counts, cap=2, letters=letters)
    assert out["N"] == 1000 + 50 + 40 + 30 + 5
    assert out["N0"] == 1000
    assert out["block1_merge_count"] <= 2
    assert out["stop_reason"] == "cap"
    # First by count: कु (cost 1), then की (cost 1) fills cap; कुर not added.
    selected = [u["akshara"] for u in out["units_selected"]]
    assert selected == ["कु", "की"]
    assert out["block1_merges"] == ["कु", "की"]

    # Larger cap: कुर reuses कु and costs 1 more.
    out2 = select_block1(counts, cap=3, letters=letters)
    assert [u["akshara"] for u in out2["units_selected"]] == ["कु", "की", "कुर"]
    assert out2["block1_merges"] == ["कु", "की", "कुर"]
    assert out2["block1_merge_count"] == 3


def test_select_never_skips_ahead() -> None:
    letters = frozenset(int(x) for x in ancoda_seg.devanagari_letters())
    # First unit costs 2; cap is 1 → stop without taking the cheaper second unit.
    counts = Counter({"क": 10, "कुर": 100, "की": 90})
    out = select_block1(counts, cap=1, letters=letters)
    assert out["units_selected"] == []
    assert out["block1_merge_count"] == 0
    assert out["stop_reason"] == "cap"


def test_caps_match_plan() -> None:
    assert CAPS == {"32k": 1638, "64k": 3276, "128k": 6553}


def test_svg_writer(tmp_path) -> None:
    path = tmp_path / "c.svg"
    curves = {
        "32k": [{"block1_merges": 1, "coverage": 0.9, "units_selected": 1, "covered_occurrences": 9}],
        "64k": [
            {"block1_merges": 1, "coverage": 0.9, "units_selected": 1, "covered_occurrences": 9},
            {"block1_merges": 2, "coverage": 0.95, "units_selected": 2, "covered_occurrences": 10},
        ],
        "128k": [
            {"block1_merges": 1, "coverage": 0.9, "units_selected": 1, "covered_occurrences": 9},
            {"block1_merges": 3, "coverage": 0.99, "units_selected": 3, "covered_occurrences": 11},
        ],
    }
    write_coverage_svg(curves, path, title="test")
    text = path.read_text(encoding="utf-8")
    assert text.startswith("<svg")
    assert "Block 1 merges" in text

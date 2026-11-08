"""Morphology helper checks (PLAN.md 3.9)."""

from __future__ import annotations

from ancoda_tok.eval.morph import load_suffixes, reference_boundaries


def test_suffixes_nonempty_longest_first() -> None:
    suffixes = load_suffixes()
    assert len(suffixes) > 10
    lengths = [len(s) for s in suffixes]
    assert lengths == sorted(lengths, reverse=True)


def test_reference_boundaries_longest_match() -> None:
    suffixes = sorted(["हरू", "को", "ा"], key=lambda s: (-len(s), s))
    cuts = reference_boundaries("नेपालको", suffixes)
    assert 5 in cuts  # नेपाल | को

"""Morphology helper checks (PLAN.md 3.9)."""

from __future__ import annotations

from ancoda_tok.eval.morph import (
    CASE_MARKERS,
    case_marker_exclusion,
    classify_marker_spans,
    load_suffixes,
    markers_missing_from_inventory,
    pretoken_surface,
    reference_boundaries,
    split_case_marker,
    word_spans_from_pieces,
)


def test_suffixes_nonempty_longest_first() -> None:
    suffixes = load_suffixes()
    assert len(suffixes) > 10
    lengths = [len(s) for s in suffixes]
    assert lengths == sorted(lengths, reverse=True)


def test_reference_boundaries_longest_match() -> None:
    suffixes = sorted(["हरू", "को", "ा"], key=lambda s: (-len(s), s))
    cuts = reference_boundaries("नेपालको", suffixes)
    assert 5 in cuts  # नेपाल | को


def test_case_markers_do_not_suffix_each_other() -> None:
    for left in CASE_MARKERS:
        for right in CASE_MARKERS:
            if left != right:
                assert not left.endswith(right)


def test_bhanda_missing_from_papaya() -> None:
    missing = markers_missing_from_inventory(load_suffixes())
    assert missing == ["भन्दा"]


def test_split_case_marker_nonempty_stem() -> None:
    assert split_case_marker("घरमा") == ("घर", "मा")
    assert split_case_marker("नेपालको") == ("नेपाल", "को")
    assert split_case_marker("भन्दा") is None
    assert split_case_marker("मा") is None
    assert split_case_marker("घर") is None


def test_case_marker_exclusion() -> None:
    def reason(word: str) -> str | None:
        split = split_case_marker(word)
        assert split is not None
        return case_marker_exclusion(word, *split)

    for word in ("भागेकी", "घटेमा", "बसेमा"):
        assert reason(word) == "e_sign_verb_form"
    for word in ("खुलाएका", "नआएको"):
        assert reason(word) == "independent_e_verb_form"
        split = split_case_marker(word)
        assert split is not None
        assert case_marker_exclusion(word, *split, independent_e=False) is None
    assert reason("बिकी") == "single_unit"
    assert reason("(बीमा") == "non_devanagari"
    assert reason("घर\u200cमा") == "non_devanagari"
    assert reason("लागिएकाले") is None
    assert reason("नेपालको") is None
    assert reason("घरमा") is None


def test_classify_marker_spans() -> None:
    assert classify_marker_spans([(0, 2), (2, 4)], 2, 4) == "respected"
    assert classify_marker_spans([(0, 4)], 2, 4) == "fused"
    assert classify_marker_spans([(0, 3), (3, 4)], 2, 4) == "crossed"


def test_word_spans_ignore_leading_space() -> None:
    word = "अब"
    space = b" "
    pieces = [space + "अ".encode(), "ब".encode()]
    assert word_spans_from_pieces(word, pieces) == [(0, 1), (1, 2)]
    assert classify_marker_spans(word_spans_from_pieces(word, pieces) or [], 1, 2) == "respected"
    fused = word_spans_from_pieces(word, [space + word.encode()])
    assert fused == [(0, 2)]
    assert classify_marker_spans(fused or [], 1, 2) == "fused"
    assert pretoken_surface("\tनेपाल") == "नेपाल"
    assert pretoken_surface(" नेपाल") == "नेपाल"

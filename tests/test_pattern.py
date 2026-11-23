"""docs/SPEC.md 3.1 pattern construction."""

import regex

from ancoda_tok.pattern import (
    MARK_NEW,
    NUMBER_OLD,
    build_pattern,
    o200k_pat_str,
    pattern_sha256,
    read_pattern,
    write_pattern,
)

EXAMPLE_1 = "नेपालको राजधानी काठमाडौं हो ।\n"
EXAMPLE_1_PARTS = ["नेपालको", " राजधानी", " काठमाडौं", " हो", " ।\n"]
EXAMPLE_2 = "र्\u200dय क्\u200cष १२३ abc 123"
EXAMPLE_2_PARTS = ["र्\u200dय", " क्\u200cष", " ", "१", "२", "३", " abc", " ", "1", "2", "3"]


def _pieces(pattern: str, text: str) -> list[str]:
    return [match.group(0) for match in regex.compile(pattern).finditer(text)]


def test_sources_agree_and_edits_are_exact() -> None:
    raw = o200k_pat_str()
    pattern = build_pattern(raw)
    assert pattern != raw
    assert pattern.count(MARK_NEW) == 4
    assert "\\p{M}]" not in pattern
    assert NUMBER_OLD not in pattern
    assert "\\u200C\\u200D" in pattern
    assert "\u200c" not in pattern
    assert "\u200d" not in pattern
    written = write_pattern()
    assert written == pattern
    assert read_pattern() == pattern
    assert pattern_sha256(pattern) == pattern_sha256(read_pattern())


def test_plan_examples_with_python_regex() -> None:
    pattern = read_pattern()
    assert _pieces(pattern, EXAMPLE_1) == EXAMPLE_1_PARTS
    assert _pieces(pattern, EXAMPLE_2) == EXAMPLE_2_PARTS
    assert "".join(_pieces(pattern, EXAMPLE_1)) == EXAMPLE_1
    assert "".join(_pieces(pattern, EXAMPLE_2)) == EXAMPLE_2

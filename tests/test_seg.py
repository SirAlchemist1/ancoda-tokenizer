"""Segmenter, pre-tokenizer equality, and the independent reference."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import ancoda_seg
import hypothesis.strategies as st
import regex
from hypothesis import given, settings

from ancoda_tok.eval import ref_segmenter
from ancoda_tok.eval.segmenter_diff import (
    iter_eval_words,
    sample_pretok_lines,
    sample_training_words,
    write_result,
)
from ancoda_tok.pattern import read_pattern, repo_root, write_pattern

CASES = repo_root() / "tests" / "data" / "akshara_cases.tsv"
ADVERSARIAL = repo_root() / "tests" / "data" / "pretok_adversarial.tsv"

EXAMPLE_1 = "नेपालको राजधानी काठमाडौं हो ।\n"
EXAMPLE_1_PARTS = ["नेपालको", " राजधानी", " काठमाडौं", " हो", " ।\n"]
EXAMPLE_2 = "र्\u200dय क्\u200cष १२३ abc 123"
EXAMPLE_2_PARTS = ["र्\u200dय", " क्\u200cष", " ", "१", "२", "३", " abc", " ", "1", "2", "3"]


def _cases():
    for line in CASES.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        case_id, raw_input, default, gb9c = line.split("\t")
        yield case_id, json.loads(raw_input), json.loads(default), json.loads(gb9c)


def _adversarial():
    for line in ADVERSARIAL.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        case_id, payload = line.split("\t", 1)
        yield case_id, json.loads(payload)


def _override(parts: list[str], letters: set[int]) -> list[str]:
    out: list[str] = []
    for part in parts:
        start = 0
        for index, ch in enumerate(part):
            if index and ord(ch) in letters:
                out.append(part[start:index])
                start = index
        out.append(part[start:])
    return out


def test_reference_does_not_call_rust() -> None:
    text = (repo_root() / "python" / "ancoda_tok" / "eval" / "ref_segmenter.py").read_text()
    assert "ancoda_seg" not in text
    assert "unicodedata" not in text
    assert "unicode_segmentation" not in text


def test_letter_sets_match() -> None:
    assert ancoda_seg.devanagari_letters() == ref_segmenter.devanagari_letters()
    letters = set(ref_segmenter.devanagari_letters())
    assert 0x0915 in letters
    assert 0x094D not in letters
    assert 0x200C not in letters
    assert 0x200D not in letters
    assert ref_segmenter.block_category(0x0915) == "Lo"
    assert ref_segmenter.block_category(0x094D) == "Mn"


def test_plan_required_aksharas() -> None:
    assert ancoda_seg.aksharas("स्कूल") == ["स्", "कू", "ल"]
    assert ancoda_seg.aksharas("गर्नु") == ["ग", "र्", "नु"]
    assert ancoda_seg.aksharas("क्ष") == ["क्", "ष"]
    assert ancoda_seg.aksharas("र्\u200dय") == ["र्\u200d", "य"]
    assert ancoda_seg.aksharas("\u0958") == ["\u0958"]
    assert ancoda_seg.aksharas("\u0915\u093c") == ["\u0915\u093c"]
    precomposed = "\u0958"
    decomposed = "\u0915\u093c"
    assert precomposed != decomposed
    assert len(precomposed) == 1
    assert len(decomposed) == 2
    assert ref_segmenter.aksharas("स्कूल") == ["स्", "कू", "ल"]
    assert ref_segmenter.aksharas("गर्नु") == ["ग", "र्", "नु"]
    assert ref_segmenter.aksharas("क्ष") == ["क्", "ष"]
    assert ref_segmenter.aksharas("र्\u200dय") == ["र्\u200d", "य"]


def test_checked_raw_clusters() -> None:
    pairs = {
        "क्ष": (["क्", "ष"], ["क्ष"]),
        "क्\u200dष": (["क्\u200d", "ष"], ["क्\u200dष"]),
        "क्\u093cष": (["क्\u093c", "ष"], ["क्\u093cष"]),
        "क्\u094dष": (["क्\u094d", "ष"], ["क्\u094dष"]),
        "क्\u200cष": (["क्\u200c", "ष"], ["क्\u200c", "ष"]),
        "र\u200d\u094dय": (["र\u200d\u094d", "य"], ["र\u200d\u094dय"]),
        "स्कूल": (["स्", "कू", "ल"], ["स्कू", "ल"]),
        "गर्नु": (["ग", "र्", "नु"], ["ग", "र्नु"]),
    }
    for text, (default, raw) in pairs.items():
        assert ancoda_seg.aksharas(text, False) == default
        assert ancoda_seg.aksharas(text, True) == raw
        assert ref_segmenter.aksharas(text, False) == default
        assert ref_segmenter.aksharas(text, True) == raw


def test_devanagari_block_joiners_are_marks() -> None:
    for cp in range(0x0900, 0x0980):
        ch = chr(cp)
        raw = ref_segmenter.aksharas(f"क{ch}", True)
        if len(raw) == 1:
            assert ref_segmenter.block_category(cp) in {"Mn", "Mc", "Me"}
        assert len(ref_segmenter.aksharas(f"{ch}a", True)) == 2
        assert len(ancoda_seg.aksharas(f"{ch}a", True)) == 2


def test_grapheme_break_test_file() -> None:
    count = 0
    letters = set(ref_segmenter.devanagari_letters())
    for text, expected in ref_segmenter.iter_grapheme_break_cases():
        assert ref_segmenter.aksharas(text, True) == expected
        rust_raw = ancoda_seg.aksharas(text, True)
        assert rust_raw == expected
        assert ancoda_seg.aksharas(text, False) == _override(rust_raw, letters)
        assert ref_segmenter.aksharas(text, False) == _override(
            ref_segmenter.aksharas(text, True), letters
        )
        count += 1
    assert count >= 500


def test_akshara_cases_file() -> None:
    count = 0
    letters = set(ref_segmenter.devanagari_letters())
    for case_id, text, default, gb9c in _cases():
        assert ancoda_seg.aksharas(text, False) == default, case_id
        assert ancoda_seg.aksharas(text, True) == gb9c, case_id
        assert ref_segmenter.aksharas(text, False) == default, case_id
        assert ref_segmenter.aksharas(text, True) == gb9c, case_id
        assert ancoda_seg.aksharas(text, False) == _override(gb9c, letters)
        count += 1
    assert count >= 200


@given(st.text(max_size=24))
@settings(max_examples=30, deadline=None)
def test_override_and_reference_agree_on_random_text(text: str) -> None:
    letters = set(ancoda_seg.devanagari_letters())
    raw = ancoda_seg.aksharas(text, True)
    assert "".join(raw) == text
    assert ancoda_seg.aksharas(text, False) == _override(raw, letters)
    assert ref_segmenter.aksharas(text, True) == raw
    assert ref_segmenter.aksharas(text, False) == ancoda_seg.aksharas(text, False)


def test_pretokenizer_examples_agree_across_engines() -> None:
    write_pattern()
    pattern = read_pattern()
    compiled = regex.compile(pattern)
    for text, expected in ((EXAMPLE_1, EXAMPLE_1_PARTS), (EXAMPLE_2, EXAMPLE_2_PARTS)):
        onig = ancoda_seg.pretokenize(text)
        fancy = ancoda_seg.pretokenize_fancy(text)
        py = [match.group(0) for match in compiled.finditer(text)]
        assert onig == expected
        assert fancy == expected
        assert py == expected


def test_pretokenizer_adversarial_and_sample_agree() -> None:
    started = datetime.now(UTC)
    pattern = read_pattern()
    compiled = regex.compile(pattern)
    opened: list[Path] = []
    rows = list(_adversarial())
    lines = [text for _, text in rows]
    sampled = sample_pretok_lines(opened)
    lines.extend(sampled)
    bad: list[str] = []
    step = 2048
    for start in range(0, len(lines), step):
        chunk = lines[start : start + step]
        onig = ancoda_seg.pretokenize_batch(chunk)
        fancy = ancoda_seg.pretokenize_fancy_batch(chunk)
        for text, left, right in zip(chunk, onig, fancy, strict=True):
            middle = [match.group(0) for match in compiled.finditer(text)]
            if left != middle or left != right:
                bad.append(repr(text[:120]))
                if len(bad) >= 20:
                    break
        if len(bad) >= 20:
            break
    assert bad == [], bad
    write_result(
        "pretokenizer_pretokenizer",
        {"lines": len(lines), "sampled_lines": len(sampled), "mismatches": 0},
        opened,
        started,
    )


def _disagreements(words: list[str], limit: int = 20) -> list[str]:
    bad: list[str] = []
    step = 4096
    for gb9c in (False, True):
        for start in range(0, len(words), step):
            chunk = words[start : start + step]
            rust = ancoda_seg.aksharas_batch(chunk, gb9c)
            for word, got in zip(chunk, rust, strict=True):
                if got != ref_segmenter.aksharas(word, gb9c):
                    bad.append(f"gb9c={gb9c} {word!r}")
                    if len(bad) >= limit:
                        return bad
    return bad


def test_segmenter_differential() -> None:
    started = datetime.now(UTC)
    opened: list[Path] = []
    training = sample_training_words(opened)
    assert len(training) == 1_000_000
    bad = _disagreements(training)
    assert bad == [], bad
    eval_words: list[str] = []
    seen: set[str] = set()
    for word in iter_eval_words(opened):
        if word in seen:
            continue
        seen.add(word)
        eval_words.append(word)
    bad = _disagreements(eval_words)
    assert bad == [], bad
    write_result(
        "pretokenizer_segmenter",
        {
            "training_words": len(training),
            "eval_words": len(eval_words),
            "disagreements": 0,
            "modes": ["default", "gb9c"],
            "sample": (
                "1e6 distinct words from training-role raw parquet shuffled with "
                "Random(20261104); the cleaning snapshot does not exist yet. Eval words are "
                "distinct words from role=eval sources and harness FLORES+ dev/devtest."
            ),
        },
        opened,
        started,
    )

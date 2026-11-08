"""Unit tests for M4 cleaning helpers."""

from __future__ import annotations

from ancoda_tok.clean.augment_joiners import insert_joiners, roundtrip_ok
from ancoda_tok.clean.common import is_latin_line, strip_controls
from ancoda_tok.clean.decontam import match_key, words
from ancoda_tok.clean.legacy_font import PreetiLexicon, line_is_legacy, should_drop_document
from ancoda_tok.clean.lid import decide_lid
from ancoda_tok.clean.preeti_convert import convert_unicode_to_preeti
from ancoda_tok.clean.utf8 import clean_utf8_text


def test_strip_controls_keeps_tab_and_newline() -> None:
    assert strip_controls("a\tb\nc\x00d\x7fe") == "a\tb\ncde"


def test_utf8_drops_empty_after_strip() -> None:
    assert clean_utf8_text("\x00\x01") is None
    assert clean_utf8_text("नेपाल") == "नेपाल"


def test_lid_reroute_and_drop() -> None:
    keep = {"npi_Deva"}
    buckets = {"npi_Deva": "Nepali", "hin_Deva": "Hindi", "eng_Latn": "English"}
    d = decide_lid(
        "npi_Deva",
        0.9,
        source_bucket="Nepali",
        keep_labels=keep,
        label_buckets=buckets,
        skip_lid=False,
    )
    assert d.action == "keep"
    d = decide_lid(
        "hin_Deva",
        0.9,
        source_bucket="Nepali",
        keep_labels=keep,
        label_buckets=buckets,
        skip_lid=False,
    )
    assert d.action == "reroute" and d.target_bucket == "Hindi"
    d = decide_lid(
        "tha_Thai",
        0.9,
        source_bucket="Nepali",
        keep_labels=keep,
        label_buckets=buckets,
        skip_lid=False,
    )
    assert d.action == "drop"


def test_preeti_roundtrip_smoke() -> None:
    text = "नेपाल"
    preeti = convert_unicode_to_preeti(text)
    assert isinstance(preeti, str) and preeti != text
    assert any(ch.isascii() for ch in preeti)


def test_dictionary_detector_flags_preeti_line() -> None:
    forms = {
        convert_unicode_to_preeti("नेपालको"),
        convert_unicode_to_preeti("राजधानी"),
        convert_unicode_to_preeti("काठमाडौं"),
        convert_unicode_to_preeti("हो"),
    }
    forms = {f for f in forms if len(f) >= 2}
    lex = PreetiLexicon(
        forms=frozenset(forms),
        path=__file__,  # unused
        sha256="test",
        n_source_words=len(forms),
        n_forms=len(forms),
    )
    line = " ".join(sorted(forms)[:4]) if len(forms) >= 3 else " ".join(list(forms) * 3)
    # Ensure ≥3 tokens
    toks = line.split()
    if len(toks) < 3:
        line = " ".join((toks * 3)[:3])
    assert line_is_legacy(line, lex)
    assert not line_is_legacy("The capital of Nepal is Kathmandu.", lex)
    assert not should_drop_document("नेपाल राम्रो छ\nThe capital is fine here today", lex)


def test_joiner_augmentation_roundtrip() -> None:
    original = "विद्यालयमा क्षति भयो"
    assert roundtrip_ok(original, "\u200d")
    assert roundtrip_ok(original, "\u200c")
    zwj = insert_joiners(original, "\u200d")
    assert "\u200d" in zwj
    assert zwj != original


def test_decontam_match_key_is_nfc_collapsed() -> None:
    a = "नेपाल\u0301   हो"
    assert "  " not in match_key(a)
    assert len(words(match_key("a b c d e f g"))) == 7


def test_latin_line_detection() -> None:
    assert is_latin_line("Hello world today")
    assert not is_latin_line("Hello नेपाल")
    assert not is_latin_line("नेपाल")
    assert not is_latin_line("https://example.com/path")

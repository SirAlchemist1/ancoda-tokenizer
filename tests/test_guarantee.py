"""docs/SPEC.md trainer guarantee: equality, covered integrity, marks, round-trip, coverage."""

from __future__ import annotations

from functools import lru_cache

from hypothesis import given, settings
from hypothesis import strategies as st

from ancoda_tok.guarantee import Engines, _check


@lru_cache(maxsize=1)
def _engines() -> Engines:
    return Engines()


def test_guarantee_report() -> None:
    from ancoda_tok.guarantee import run

    assert run() == []


@settings(max_examples=40, deadline=None)
@given(
    st.text(
        alphabet=st.characters(blacklist_categories=("Cs",)),
        min_size=0,
        max_size=32,
    )
)
def test_fuzz_equality_roundtrip_integrity(text: str) -> None:
    failures: list[str] = []
    stats = {
        "texts": 0,
        "equality": 0,
        "hf_round": 0,
        "tt_round": 0,
        "hf_violations": 0,
        "tt_violations": 0,
    }
    _check(_engines(), text, failures, "fuzz", stats)
    assert failures == []
    assert stats["equality"] == 0
    assert stats["hf_round"] == 0
    assert stats["tt_round"] == 0
    assert stats["hf_violations"] == 0
    assert stats["tt_violations"] == 0

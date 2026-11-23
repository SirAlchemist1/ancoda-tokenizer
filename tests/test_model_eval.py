"""Pure checks for the model-eval procedure. No GPU and no full corpus."""

from __future__ import annotations

import hashlib
import random

import numpy as np

from ancoda_tok.final_build_final import misspell
from ancoda_tok.model_eval.common import (
    CONTENT_TOKEN_FLOOR,
    WARMUP_STEPS,
    bpb,
    choose_total_bytes,
    eos_choice,
    group_targets,
    paired_call,
    storage_dtype,
    text_windows,
    update_list_hash,
)
from ancoda_tok.model_eval.cutoff import prefix_spec
from ancoda_tok.model_eval.perm import (
    byte_prefix,
    bytes_at,
    first_step_reaching,
    fisher_yates,
    stream_prefix,
)
from ancoda_tok.model_eval.perturb import ZWNJ, insert_joiner, misspell_text
from ancoda_tok.model_eval.schedule import learning_rate


def test_warmup_is_153_and_cosine_ends_at_min() -> None:
    assert WARMUP_STEPS == 153
    assert learning_rate(0) == 1e-3 / 153
    assert learning_rate(152) == 1e-3
    assert abs(learning_rate(7630) - 1e-4) < 1e-12


def test_group_targets_sum_and_shares() -> None:
    targets = group_targets(50_000_000)
    assert sum(targets.values()) == 50_000_000
    assert targets["nepali"] == 20_000_000
    assert targets["english"] == 17_500_000
    assert targets["hindi"] == 7_500_000


def test_choose_total_stops_when_supply_is_short() -> None:
    try:
        choose_total_bytes(
            0.01, {name: 1_000 for name in ("nepali", "hindi", "other_deva", "english", "code")}
        )
    except SystemExit as exc:
        assert "STOP" in str(exc)
    else:
        raise AssertionError("expected a stop")


def test_choose_total_pads_when_supply_allows() -> None:
    available = {
        "nepali": 10**12,
        "hindi": 10**12,
        "other_deva": 10**12,
        "english": 10**12,
        "code": 10**12,
    }
    total = choose_total_bytes(1.0, available)
    need = CONTENT_TOKEN_FLOOR
    assert total % 20 == 0
    assert total >= need
    assert total <= int(need * 1.03) + 20


def test_prefix_includes_the_crossing_document() -> None:
    ranks = np.asarray([5, 1, 3, 1], dtype=np.uint64)
    nbytes = np.asarray([10, 10, 10, 10], dtype=np.int64)
    spec = prefix_spec(ranks, nbytes, 25)
    assert spec["mode"] == "prefix"
    assert spec["cutoff"] == 3
    assert spec["bytes_below"] == 20
    assert spec["need_at_cutoff"] == 5


def test_bpb_uses_original_bytes() -> None:
    assert abs(bpb(np.log(2) * 8, 8) - 1.0) < 1e-12


def test_decision_rule() -> None:
    assert paired_call([-0.2, -0.2, -0.2]) == "A better"
    assert paired_call([-0.2, -0.2, 0.01]) == "no measurable difference"
    assert paired_call([0.2, 0.2, 0.2]) == "no measurable difference"
    assert paired_call([-0.01, -0.02, -0.5]) == "no measurable difference"


def test_windows_cover_every_text_token() -> None:
    spans = text_windows(2000)
    covered = [i for a, b in spans for i in range(a, b)]
    assert covered == list(range(2000))
    assert all(b - a <= 1023 for a, b in spans)
    assert spans[0] == (0, 1023)


def test_eos_choice() -> None:
    assert eos_choice({"<|eos|>": 64513}, 65536) == (64513, "<|eos|>", 65536)
    assert eos_choice({"<|eos|>": 2}, 65536) == (2, "<|eos|>", 65536)
    assert eos_choice({}, 64000) == (64000, "extended", 64001)
    assert storage_dtype(64513) == "uint16"
    assert storage_dtype(65536) == "uint32"


def test_misspell_matches_final_build() -> None:
    text = "शिशु सपना िीुू"
    left = random.Random(20261104)
    right = random.Random(20261104)
    assert misspell_text(text, left) == misspell(text, right)


def test_zwnj_after_virama_before_consonant() -> None:
    assert insert_joiner("प्र", ZWNJ) == f"प्{ZWNJ}र"
    assert insert_joiner("abc", ZWNJ) == "abc"


def test_permutation_is_deterministic() -> None:
    a = fisher_yates(20, 1)
    b = fisher_yates(20, 1)
    c = fisher_yates(20, 2)
    assert np.array_equal(a, b)
    assert sorted(a.tolist()) == list(range(20))
    assert not np.array_equal(a, c)


def test_bytes_charge_content_tokens_only() -> None:
    order = np.asarray([0, 1], dtype=np.int64)
    tok = np.asarray([4, 0], dtype=np.int64)
    byt = np.asarray([10, 3], dtype=np.int64)
    cum = stream_prefix(order, tok)
    bpre = byte_prefix(order, byt)
    # doc 0 has 4 content tokens + EOS. Halfway is 2 content tokens -> 5 bytes.
    assert bytes_at(2, order, tok, byt, cum, bpre) == 5
    # the EOS adds no further bytes
    assert bytes_at(5, order, tok, byt, cum, bpre) == 10
    # empty doc contributes its 3 bytes when its EOS is consumed
    assert bytes_at(6, order, tok, byt, cum, bpre) == 13
    step = first_step_reaching(lambda pos: bytes_at(pos, order, tok, byt, cum, bpre), 10, 5, 4)
    assert step == 1


def test_document_list_hash_is_doc_id_lines() -> None:
    hasher = hashlib.sha256()
    update_list_hash(hasher, "a\x1fb")
    update_list_hash(hasher, "c")
    expect = hashlib.sha256(b"a\x1fb\nc\n").hexdigest()
    assert hasher.hexdigest() == expect

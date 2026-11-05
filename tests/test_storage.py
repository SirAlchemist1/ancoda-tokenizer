"""Pure storage-plan arithmetic. No Hub calls."""

from ancoda_tok.storage import (
    approved_peak,
    balanced_train_caps,
    download_peak,
    per_path_stored,
    propose_train_caps,
    resident_peak,
    stored_bytes,
    transient_bytes,
)


def test_stored_all_and_cap() -> None:
    assert stored_bytes([10, 20], "all") == 30
    assert stored_bytes([10, 20], 15) == 15
    assert stored_bytes([10], 50) == 10


def test_transient_all_is_the_tree() -> None:
    assert transient_bytes([5, 7], "all") == 12


def test_transient_cap_includes_one_whole_shard() -> None:
    # cap 10, shards 8 and 6: cache bound is min(14, 10+8) = 14
    assert transient_bytes([8, 6], 10) == 14
    # large first shard, small cap
    assert transient_bytes([100, 5], 10) == 105
    assert transient_bytes([100], 10) == 100


def test_transient_keep_ratio_scales_the_scan() -> None:
    # keep 1 byte per 10 file bytes, cap 100 → need 1000 file bytes, plus the largest shard
    # need 1000 file bytes plus the largest shard, capped by the tree (1200)
    assert transient_bytes([400, 400, 400], 100, keep_ratio=0.1) == 1200
    assert transient_bytes([400], 100, keep_ratio=0.1) == 400


def test_per_path_cap_does_not_pool_unused_budget() -> None:
    assert per_path_stored([[100], [10]], 40) == 50


def test_resident_peak_counts_training_twice() -> None:
    assert resident_peak(train_stored=10, other_stored=3, snapshots=4, counts=1) == 28


def test_approved_peak_counts_training_once() -> None:
    assert approved_peak(train_stored=10, other_stored=3, snapshots=4, counts=1) == 18


def test_download_peak_adds_one_cache() -> None:
    assert download_peak(20, 5) == 25


def test_caps_cut_fw2_npi_before_larger_sources() -> None:
    # peak = 2 * (100 + 1000) = 2200; free 500; floor 100; deficit 1800; cut 900
    # fw2_npi is only 100, then big loses 800
    cuts = propose_train_caps(
        [("big", 1000, 1), ("fw2_npi", 100, 0)],
        other_stored=0,
        free_bytes=500,
        floor=100,
        snapshots=0,
        counts=0,
    )
    assert cuts == [("fw2_npi", 100, 0), ("big", 1000, 200)]
    new_train = 0 + 200
    assert 500 - resident_peak(new_train, 0, 0, 0) >= 100


def test_balanced_caps_keep_fw2_npi_above_zero() -> None:
    # peak = 2 * (10 + 30) = 80; free 50; floor 20; need to cut 25 bytes of training.
    cuts = balanced_train_caps(
        [("fw2_npi", 10), ("big", 30)],
        other_stored=0,
        free_bytes=50,
        floor=20,
        snapshots=0,
        counts=0,
        fw2_ceiling=3,
        large_ceiling=8,
    )
    assert {item[0]: item[2] for item in cuts} == {"fw2_npi": 3, "big": 8}
    assert 50 - resident_peak(3 + 8, 0, 0, 0) >= 20


def test_no_cuts_when_the_floor_holds() -> None:
    assert (
        propose_train_caps(
            [("fw2_npi", 10, 0)],
            other_stored=0,
            free_bytes=100,
            floor=50,
            snapshots=0,
            counts=0,
        )
        == []
    )

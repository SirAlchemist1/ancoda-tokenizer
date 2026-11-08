"""The Python count writer must match the Rust ANCDACT1 reader."""

from __future__ import annotations

import subprocess

from ancoda_tok.counts import iter_entries, merge_counts, write_counts
from ancoda_tok.storage import REPO_ROOT


def test_merge_and_rust_reader(tmp_path) -> None:
    left = tmp_path / "left.bin"
    right = tmp_path / "right.bin"
    write_counts(left, [(b"a", 2), (b"c", 1), ("क".encode(), 7)])
    write_counts(right, [(b"a", 3), (b"b", 4)])
    merged = tmp_path / "merged.bin"
    stats = merge_counts([left, right], merged)
    assert list(iter_entries(merged)) == [(b"a", 5), (b"b", 4), (b"c", 1), ("क".encode(), 7)]
    assert stats == {"unique": 4, "occurrences": 17, "key_bytes": 1 + 1 + 1 + 3}
    binary = REPO_ROOT / "target" / "debug" / "ancoda-bpe"
    if not binary.exists():
        subprocess.run(
            ["cargo", "build", "-p", "ancoda-bpe", "--offline"],
            cwd=REPO_ROOT,
            check=True,
        )
    out = subprocess.check_output(
        [str(binary), "check-counts", str(merged)],
        cwd=REPO_ROOT,
        text=True,
    )
    assert "entries=4" in out
    assert "occurrences=17" in out
    assert "key_bytes=6" in out

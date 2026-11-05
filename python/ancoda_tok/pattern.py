"""Build the Ancoda pre-tokenizer pattern from o200k (PLAN.md 3.1).

The pattern is read from tiktoken. It is never retyped. Edit 1 inserts ZWNJ and
ZWJ into each combining-mark class. Edit 2 makes every digit its own pre-token.
The joiner escapes are the six-character sequences ``\\u200C`` and ``\\u200D``.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

MARK_OLD = "\\p{M}]"
MARK_NEW = "\\p{M}\\u200C\\u200D]"
NUMBER_OLD = "\\p{N}{1,3}"
NUMBER_NEW = "\\p{N}"

KEPT = (
    "(?i:'s|'t|'re|'ve|'m|'ll|'d)?",
    "[\\r\\n/]*",
    "\\s*[\\r\\n]+",
    "\\s+(?!\\S)",
    "\\s+",
)


def repo_root() -> Path:
    for cand in Path(__file__).resolve().parents:
        if (cand / "PLAN.md").is_file():
            return cand
    raise RuntimeError("PLAN.md not found")


def pattern_path() -> Path:
    return repo_root() / "configs" / "pretokenizer_pattern.txt"


def o200k_pat_str() -> str:
    """Return o200k's pattern after checking both plan-specified sources."""
    import tiktoken
    from tiktoken_ext.openai_public import o200k_base

    from_encoding = tiktoken.get_encoding("o200k_base")._pat_str
    from_function = o200k_base()["pat_str"]
    if from_encoding != from_function:
        raise SystemExit("STOP: tiktoken o200k pat_str sources differ")
    return from_encoding


def build_pattern(raw: str) -> str:
    marks = raw.count(MARK_OLD)
    if marks != 4:
        raise SystemExit(f"STOP: expected 4 replacements of {MARK_OLD!r}, found {marks}")
    edited = raw.replace(MARK_OLD, MARK_NEW)
    if MARK_OLD in edited:
        raise SystemExit("STOP: a mark class survived edit 1")
    if edited.count(MARK_NEW) != 4:
        raise SystemExit("STOP: edit 1 did not produce exactly 4 joiner classes")
    numbers = raw.count(NUMBER_OLD)
    if numbers != 1 or edited.count(NUMBER_OLD) != 1:
        raise SystemExit(f"STOP: expected 1 replacement of {NUMBER_OLD!r}, found {numbers}")
    edited = edited.replace(NUMBER_OLD, NUMBER_NEW, 1)
    if NUMBER_OLD in edited:
        raise SystemExit("STOP: the digit quantifier survived edit 2")
    for anchor in KEPT:
        if anchor not in edited:
            raise SystemExit(f"STOP: edit removed a kept branch {anchor!r}")
    if edited.count(KEPT[0]) != 2:
        raise SystemExit("STOP: the contraction branch was not kept twice")
    if "\\u200C" not in edited or "\\u200D" not in edited:
        raise SystemExit("STOP: joiner escapes were not written as \\\\u200C and \\\\u200D")
    if "\u200c" in edited or "\u200d" in edited or "\\x{" in edited:
        raise SystemExit("STOP: joiners must be \\u escapes, not literals or \\x{}")
    return edited


def pattern_sha256(pattern: str) -> str:
    return hashlib.sha256(pattern.encode("utf-8")).hexdigest()


def write_pattern() -> str:
    pattern = build_pattern(o200k_pat_str())
    path = pattern_path()
    path.write_bytes(pattern.encode("utf-8"))
    return pattern


def read_pattern() -> str:
    data = pattern_path().read_bytes()
    if b"\n" in data or b"\r" in data:
        raise RuntimeError("pretokenizer pattern file must not contain a newline")
    return data.decode("utf-8")


def main() -> None:
    pattern = write_pattern()
    print(f"wrote {pattern_path()}")
    print(f"sha256={pattern_sha256(pattern)}")
    print(f"bytes={len(pattern.encode('utf-8'))}")


if __name__ == "__main__":
    main()

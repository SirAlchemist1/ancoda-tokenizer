"""Human corrections for trainer, unit-tested before any tokenizer training.

Ban 1 lives in ``ancoda-bpe`` (``cargo test -p ancoda-bpe``). This file checks
that literal special-token strings stay ordinary text.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import tiktoken
from tokenizers import Tokenizer

from ancoda_tok.export_hf import build_tokenizer, bytes_to_unicode, load_hf, read_pattern
from ancoda_tok.export_tiktoken import export_tiktoken, load_tiktoken
from ancoda_tok.specials import special_token_names
from ancoda_tok.storage import REPO_ROOT

BYTE_TOKENS = [bytes([index]) for index in range(256)]


def _rust_specials() -> list[str]:
    binary = REPO_ROOT / "target" / "debug" / "ancoda-bpe"
    if not binary.exists():
        subprocess.run(
            ["cargo", "build", "-p", "ancoda-bpe", "--offline"],
            cwd=REPO_ROOT,
            check=True,
        )
    out = subprocess.check_output([str(binary), "specials"], cwd=REPO_ROOT, text=True)
    return [line for line in out.splitlines() if line]


def _byte_tokenizer(pattern: str) -> Tokenizer:
    specials = [(name, 256 + index) for index, name in enumerate(special_token_names())]
    tokenizer = build_tokenizer(BYTE_TOKENS, specials, pattern)
    tokenizer.encode_special_tokens = True
    return tokenizer


def _tiktoken(pattern: str) -> tiktoken.Encoding:
    specials = {name: 256 + index for index, name in enumerate(special_token_names())}
    return tiktoken.Encoding(
        name="ancoda-corrections",
        pat_str=pattern,
        mergeable_ranks={bytes([index]): index for index in range(256)},
        special_tokens=specials,
    )


def test_merge_list_is_sorted_by_token_rank() -> None:
    from ancoda_tok.export_hf import all_splits

    tokens = [bytes([index]) for index in range(256)]
    tokens.append(b"os")
    tokens.append(b"bo")
    splits = all_splits(tokens)
    os_at = splits.index((256, ord("o"), ord("s")))
    bo_at = splits.index((257, ord("b"), ord("o")))
    assert os_at < bo_at
    assert splits == sorted(splits)


def test_rust_and_python_specials_match() -> None:
    names = special_token_names()
    assert len(names) == 1024
    assert names[0] == "<|bos|>"
    assert names[15] == "<|sep|>"
    assert names[16] == "<|reserved_16|>"
    assert names[1023] == "<|reserved_1023|>"
    assert len(set(names)) == 1024
    assert _rust_specials() == names


def test_literal_specials_match_encode_ordinary(tmp_path: Path) -> None:
    pattern = read_pattern()
    alphabet = set(bytes_to_unicode().values())
    from tokenizers.pre_tokenizers import ByteLevel

    assert alphabet == set(ByteLevel.alphabet())
    hf = _byte_tokenizer(pattern)
    tt = _tiktoken(pattern)
    names = special_token_names()
    special_ids = set(range(256, 256 + len(names)))
    texts = [name for name in names]
    texts.extend(f"नेपाली {name} 123\n" for name in names)
    texts.append(" ".join(names))
    for text in texts:
        hf_ids = hf.encode(text, add_special_tokens=False).ids
        tt_ids = tt.encode_ordinary(text)
        assert not (special_ids & set(hf_ids)), text[:80]
        assert hf_ids == tt_ids, text[:80]
        assert hf.decode(hf_ids, skip_special_tokens=False) == text
        assert tt.decode(tt_ids) == text
    # The flag is a load-time setting. Saving must not be required to keep it.
    dest = tmp_path / "tokenizer.json"
    hf.save(str(dest))
    reloaded = Tokenizer.from_file(str(dest))
    swallowed = reloaded.encode("<|bos|>", add_special_tokens=False).ids
    assert swallowed == [256]
    ordinary = load_hf(dest)
    assert ordinary.encode_special_tokens is True
    assert ordinary.encode("<|bos|>", add_special_tokens=False).ids == list(b"<|bos|>")


def test_tiktoken_json_pattern_round_trip(tmp_path: Path) -> None:
    pattern = read_pattern()
    specials = [(name, 256 + index) for index, name in enumerate(special_token_names())]
    export_tiktoken(BYTE_TOKENS, specials, tmp_path, pattern)
    loaded = load_tiktoken(tmp_path, name="ancoda-corrections")
    text = "x <|bos|> <|reserved_1023|>"
    assert loaded.encode_ordinary(text) == list(text.encode())
    assert loaded.decode(loaded.encode_ordinary(text)) == text

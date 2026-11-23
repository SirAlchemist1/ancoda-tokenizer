"""Hugging Face tokenizer.json export (docs/SPEC.md 3.5).

Ordinary-text callers must use :func:`load_hf`. ``encode_special_tokens`` is not
written into ``tokenizer.json``; the file reloads with the library default
(``False``), which swallows literal special-token strings.
"""

from __future__ import annotations

import json
from pathlib import Path

from tokenizers import AddedToken, Regex, Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel, Sequence, Split
from tokenizers.processors import TemplateProcessing

from ancoda_tok.specials import special_token_names
from ancoda_tok.storage import REPO_ROOT


def bytes_to_unicode() -> dict[int, str]:
    """GPT-2 ``bytes_to_unicode``. Ids in our rank table stay byte values."""
    printable = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    byte_values = list(printable)
    code_points = list(printable)
    next_code = 0
    seen = set(printable)
    for byte in range(256):
        if byte not in seen:
            byte_values.append(byte)
            code_points.append(256 + next_code)
            next_code += 1
    return {byte: chr(code) for byte, code in zip(byte_values, code_points, strict=True)}


def piece(raw: bytes, encoder: dict[int, str]) -> str:
    return "".join(encoder[byte] for byte in raw)


def read_pattern() -> str:
    path = REPO_ROOT / "configs" / "pretokenizer_pattern.txt"
    text = path.read_bytes()
    if text.endswith(b"\n"):
        raise RuntimeError(f"{path} has a trailing newline; the pattern file must not")
    return text.decode("utf-8")


def all_splits(tokens: list[bytes]) -> list[tuple[int, int, int]]:
    """Every ``(rank(T), rank(A), rank(B))`` where ``T = A + B`` and all three are tokens.

    Sorted by that triple so Hugging Face's merge-list order matches lowest-rank
    concatenation (docs/SPEC.md 3.5).
    """
    index = {raw: rank for rank, raw in enumerate(tokens)}
    splits: list[tuple[int, int, int]] = []
    for raw, rank in index.items():
        if len(raw) < 2:
            continue
        for cut in range(1, len(raw)):
            left, right = raw[:cut], raw[cut:]
            left_id = index.get(left)
            right_id = index.get(right)
            if left_id is not None and right_id is not None:
                splits.append((rank, left_id, right_id))
    splits.sort()
    return splits


def build_tokenizer(
    tokens: list[bytes],
    specials: list[tuple[str, int]],
    pattern: str,
) -> Tokenizer:
    """Build a byte-level BPE tokenizer. Special ids must follow the mergeable ids."""
    if specials:
        expected = len(tokens)
        for offset, (name, ident) in enumerate(specials):
            if ident != expected + offset:
                raise RuntimeError(f"special {name} id {ident} is not {expected + offset}")
    encoder = bytes_to_unicode()
    vocab: dict[str, int] = {}
    for rank, raw in enumerate(tokens):
        key = piece(raw, encoder)
        if key in vocab:
            raise RuntimeError(f"duplicate byte-level token at ranks {vocab[key]} and {rank}")
        vocab[key] = rank
    for name, _ident in specials:
        if name in vocab:
            raise RuntimeError(
                f"STOP: special token {name} collides with a mergeable token's byte-level spelling"
            )
    merges = [
        (piece(tokens[left], encoder), piece(tokens[right], encoder))
        for _rank, left, right in all_splits(tokens)
    ]
    model = BPE(
        vocab=vocab,
        merges=merges,
        ignore_merges=True,
        byte_fallback=False,
        dropout=None,
        unk_token=None,
        fuse_unk=False,
    )
    tokenizer = Tokenizer(model)
    if specials:
        tokenizer.add_special_tokens([AddedToken(name, special=True) for name, _ident in specials])
        for name, ident in specials:
            got = tokenizer.token_to_id(name)
            if got != ident:
                raise RuntimeError(f"STOP: {name} landed on id {got}, wanted {ident}")
        bos = tokenizer.token_to_id("<|bos|>")
        tokenizer.post_processor = TemplateProcessing(
            single="<|bos|> $A",
            pair="<|bos|> $A <|bos|> $B",
            special_tokens=[("<|bos|>", bos)],
        )
    tokenizer.pre_tokenizer = Sequence(
        [
            Split(Regex(pattern), behavior="isolated", invert=False),
            ByteLevel(add_prefix_space=False, trim_offsets=False, use_regex=False),
        ]
    )
    tokenizer.decoder = ByteLevelDecoder()
    return tokenizer


def load_hf(path: Path) -> Tokenizer:
    """Load ``tokenizer.json`` for ordinary text.

    Sets ``encode_special_tokens = True`` so a literal special-token string is
    plain text, matching tiktoken ``encode_ordinary``. ``add_special_tokens``
    stays the caller's choice; equality tests pass ``False``.
    """
    tokenizer = Tokenizer.from_file(str(path))
    tokenizer.encode_special_tokens = True
    return tokenizer


def write_sidecar(out_dir: Path, specials: list[tuple[str, int]]) -> None:
    names = [name for name, _ident in specials]
    bos = names[0] if names else None
    eos = names[1] if len(names) > 1 else None
    pad = names[2] if len(names) > 2 else None
    config = {
        "tokenizer_class": "PreTrainedTokenizerFast",
        "bos_token": bos,
        "eos_token": eos,
        "pad_token": pad,
        "unk_token": None,
        "clean_up_tokenization_spaces": False,
    }
    extra = names[3:] if len(names) > 3 else []
    special_map = {
        "bos_token": bos,
        "eos_token": eos,
        "pad_token": pad,
        "additional_special_tokens": extra,
    }
    (out_dir / "tokenizer_config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (out_dir / "special_tokens_map.json").write_text(
        json.dumps(special_map, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def export_from_ranks(ranks_path: Path, out_dir: Path) -> Path:
    ranks = json.loads(ranks_path.read_text(encoding="utf-8"))
    tokens = [bytes.fromhex(item) for item in ranks["tokens_hex"]]
    specials = [(item["name"], int(item["id"])) for item in ranks["special_tokens"]]
    if not specials:
        # Plain validation builds have no reserved block. Keep the 3.7 names
        # out of the file so they cannot be swallowed.
        specials = []
    elif [name for name, _ident in specials] != special_token_names():
        raise RuntimeError("ranks.json specials do not match the 1,024-name list")
    out_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = build_tokenizer(tokens, specials, read_pattern())
    dest = out_dir / "tokenizer.json"
    tokenizer.save(str(dest))
    write_sidecar(out_dir, specials)
    return dest

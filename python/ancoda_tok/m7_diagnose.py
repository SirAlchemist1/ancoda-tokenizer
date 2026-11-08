"""M7 option-3 diagnostics: Nepali-only ceiling, violations, word splits.

Does not change mix pick or DECISIONS.md.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import ancoda_seg
import sentencepiece as spm
from tokenizers import Tokenizer

from ancoda_tok.clean.common import base_manifest, finish_manifest, save_json
from ancoda_tok.export_hf import load_hf
from ancoda_tok.inventory import find_snapshot
from ancoda_tok.m7 import (
    COUNTS_DIR,
    COVERED,
    COVERED_GB9C,
    M7_DIR,
    TOK_DIR,
    _caffeinate,
    export_tokenizer,
    harness_add,
    rewrite_counts,
    train_config,
)
from ancoda_tok.storage import REPO_ROOT

YAK_V1 = (
    REPO_ROOT
    / "data"
    / "hf_cache"
    / "hub"
    / "models--ampixa--yak-tokenizer"
    / "snapshots"
    / "12b6d3010ec542b484ec3b673354852fe8bb85fb"
    / "v1.0"
    / "64k"
    / "tokenizer.json"
)
SIDSKARKI = (
    REPO_ROOT
    / "data"
    / "hf_cache"
    / "hub"
    / "models--sidskarki--nepali-bpe-tokenizer"
    / "snapshots"
    / "fc5773dcf7b9dc2272e8f595f99828969621a855"
    / "nepali_bpe_64k.model"
)
FLORES_NPI = (
    REPO_ROOT
    / "third_party"
    / "yak-tokenizer"
    / "data"
    / "eval"
    / "flores_plus"
    / "devtest"
    / "npi_Deva.txt"
)
FLORES_ENG = (
    REPO_ROOT
    / "third_party"
    / "yak-tokenizer"
    / "data"
    / "eval"
    / "flores_plus"
    / "devtest"
    / "eng_Latn.txt"
)
FLORES_HIN = (
    REPO_ROOT
    / "third_party"
    / "yak-tokenizer"
    / "data"
    / "eval"
    / "flores_plus"
    / "devtest"
    / "hin_Deva.txt"
)


def _covered_units(path: Path) -> set[str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return {item["akshara"] for item in data["units_selected"]}


def _bounds(pieces: list[bytes]) -> list[int]:
    total = 0
    out = []
    for piece in pieces:
        total += len(piece)
        out.append(total)
    return out


def _inside(start: int, end: int, bounds: list[int]) -> bool:
    import bisect

    index = bisect.bisect_right(bounds, start)
    return index < len(bounds) and bounds[index] < end


def covered_violations_hf(tokenizer_path: Path, text: str, covered: set[str]) -> int:
    tok = load_hf(tokenizer_path)
    ids = tok.encode(text, add_special_tokens=False).ids
    from ancoda_tok.export_hf import bytes_to_unicode

    inverse = {char: byte for byte, char in bytes_to_unicode().items()}
    pieces = []
    for ident in ids:
        token = tok.id_to_token(ident)
        pieces.append(bytes(inverse[ch] for ch in token))
    bounds = _bounds(pieces)
    raw = text.encode("utf-8")
    if b"".join(pieces) != raw:
        # Still count spans against best-effort offsets.
        pass
    offset = 0
    bad = 0
    for unit in ancoda_seg.aksharas(text, False):
        unit_raw = unit.encode("utf-8")
        start, end = offset, offset + len(unit_raw)
        offset = end
        if unit in covered and _inside(start, end, bounds):
            bad += 1
    return bad


def covered_violations_sp(model_path: Path, text: str, covered: set[str]) -> int:
    sp = spm.SentencePieceProcessor(model_file=str(model_path))
    # SentencePiece returns pieces as strings; rebuild UTF-8 via encode ids + id_to_piece.
    ids = sp.encode(text, out_type=int)
    pieces: list[bytes] = []
    for ident in ids:
        piece = sp.id_to_piece(ident)
        # byte_fallback pieces look like <0xNN>
        if piece.startswith("<0x") and piece.endswith(">"):
            pieces.append(bytes([int(piece[3:-1], 16)]))
        else:
            # SentencePiece uses ▁ for space
            pieces.append(piece.replace("\u2581", " ").encode("utf-8"))
    bounds = _bounds(pieces)
    offset = 0
    bad = 0
    for unit in ancoda_seg.aksharas(text, False):
        unit_raw = unit.encode("utf-8")
        start, end = offset, offset + len(unit_raw)
        offset = end
        if unit in covered and _inside(start, end, bounds):
            bad += 1
    return bad


def fertility_whole_and_per_line(encode_tokens, path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    # Yak whole-file: lines joined by \n, no trailing newline.
    lines = text.splitlines()
    whole = "\n".join(lines)
    words_whole = len(whole.split())
    toks_whole = encode_tokens(whole)
    toks_lines = 0
    words_lines = 0
    for line in lines:
        if not line:
            continue
        toks_lines += encode_tokens(line)
        words_lines += len(line.split())
    fert_whole = 0.0 if words_whole == 0 else toks_whole / words_whole
    fert_line = 0.0 if words_lines == 0 else toks_lines / words_lines
    return {
        "whole_file": fert_whole,
        "per_line": fert_line,
        "newline_gap": fert_whole - fert_line,
        "tokens_whole": toks_whole,
        "words_whole": words_whole,
        "tokens_per_line_sum": toks_lines,
        "words_per_line_sum": words_lines,
    }


def hf_token_count(path: Path):
    tok = load_hf(path)

    def encode(text: str) -> int:
        return len(tok.encode(text, add_special_tokens=False).ids)

    return encode


def sp_token_count(path: Path):
    sp = spm.SentencePieceProcessor(model_file=str(path))

    def encode(text: str) -> int:
        return len(sp.encode(text, out_type=int))

    return encode


def cmd_nepali_only(_: argparse.Namespace) -> None:
    """Count + train final 64k on Nepali bucket of n60 only (diagnostic, not a candidate)."""
    M7_DIR.mkdir(parents=True, exist_ok=True)
    snapshot = find_snapshot("n60")
    out_counts = COUNTS_DIR / "n60_nepali_only"
    if not (out_counts / "pretoken_counts.bin").is_file():
        _caffeinate(
            [
                "uv",
                "run",
                "python",
                "-u",
                "-m",
                "ancoda_tok.counts",
                "--snapshot",
                str(snapshot),
                "--out",
                str(out_counts),
                "--bucket",
                "Nepali",
            ]
        )
    else:
        print("nepali-only counts already present", flush=True)
    config = rewrite_counts(
        REPO_ROOT / "configs" / "build_64k_n60.yaml",
        "artifacts/counts/n60_nepali_only/pretoken_counts.bin",
        "artifacts/tokenizers/diag-n60-nepali-only",
    )
    out = train_config(config)
    export_tokenizer(out)
    harness_add({"diag_n60_nepali_only": out}, "diag_nepali_only")
    print("nepali-only diagnostic ready", flush=True)


def cmd_violations(_: argparse.Namespace) -> None:
    """Covered-akshara violations on FLORES+ npi for ladder + diagnostic dirs."""
    covered = _covered_units(COVERED)
    covered_gb9c = _covered_units(COVERED_GB9C)
    text = FLORES_NPI.read_text(encoding="utf-8")
    whole = "\n".join(text.splitlines())
    rows = {}
    targets = {
        "plain": (TOK_DIR / "ladder-plain" / "tokenizer.json", covered),
        "b0": (TOK_DIR / "ladder-b0" / "tokenizer.json", covered),
        "b0b1": (TOK_DIR / "ladder-b0b1" / "tokenizer.json", covered),
        "final_nojoiner": (TOK_DIR / "ladder-final_nojoiner" / "tokenizer.json", covered),
        "final": (TOK_DIR / "ladder-final" / "tokenizer.json", covered),
        "gb9c": (TOK_DIR / "ladder-gb9c" / "tokenizer.json", covered_gb9c),
        "unigram": (TOK_DIR / "ladder-unigram" / "unigram.model", covered),
        "n60": (TOK_DIR / "ancoda-64k-n60" / "tokenizer.json", covered),
        "diag_nepali_only": (TOK_DIR / "diag-n60-nepali-only" / "tokenizer.json", covered),
    }
    for name, (path, units) in targets.items():
        if not path.is_file():
            rows[name] = {"missing": str(path)}
            continue
        if path.suffix == ".model":
            bad = covered_violations_sp(path, whole, units)
        else:
            bad = covered_violations_hf(path, whole, units)
        rows[name] = {
            "covered_violations_flores_npi_whole": bad,
            "covered_units": len(units),
            "path": str(path.relative_to(REPO_ROOT)),
        }
        print(f"{name}: violations={bad}", flush=True)
    dest = M7_DIR / "diag_violations.json"
    save_json(dest, finish_manifest(base_manifest(kind="m7_diag_violations", rows=rows)))


def cmd_word_splits(_: argparse.Namespace) -> None:
    text = FLORES_NPI.read_text(encoding="utf-8")
    # Frequency over whitespace words (same as fertility word count).
    freq = Counter(text.split())
    top = freq.most_common(50)
    n60 = load_hf(TOK_DIR / "ancoda-64k-n60" / "tokenizer.json")
    yak = Tokenizer.from_file(str(YAK_V1))
    # Yak may need encode_special_tokens; for ordinary Nepali words it should not matter.
    if hasattr(yak, "encode_special_tokens"):
        yak.encode_special_tokens = True
    sp = spm.SentencePieceProcessor(model_file=str(SIDSKARKI))

    def split_hf(tok, word: str) -> list[str]:
        ids = tok.encode(word, add_special_tokens=False).ids
        return [tok.id_to_token(i) for i in ids]

    def split_sp(word: str) -> list[str]:
        return sp.encode(word, out_type=str)

    rows = []
    for word, count in top:
        rows.append(
            {
                "word": word,
                "count": count,
                "n60": split_hf(n60, word),
                "yak_v1_64k": split_hf(yak, word),
                "sidskarki_64k": split_sp(word),
            }
        )
    # Also with leading space (BPE often treats word-initial separately).
    rows_space = []
    for word, count in top:
        spaced = " " + word
        rows_space.append(
            {
                "word": spaced,
                "count": count,
                "n60": split_hf(n60, spaced),
                "yak_v1_64k": split_hf(yak, spaced),
                "sidskarki_64k": split_sp(spaced),
            }
        )
    # Per-line / newline gap for n60.
    fert = fertility_whole_and_per_line(
        hf_token_count(TOK_DIR / "ancoda-64k-n60" / "tokenizer.json"), FLORES_NPI
    )
    out = {
        "flores_npi_fertility_n60": fert,
        "top50_no_leading_space": rows,
        "top50_leading_space": rows_space,
    }
    dest = M7_DIR / "diag_word_splits.json"
    save_json(dest, finish_manifest(base_manifest(kind="m7_diag_word_splits", **out)))
    print(json.dumps(fert, indent=2), flush=True)
    print(f"wrote {dest}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("nepali-only")
    sub.add_parser("violations")
    sub.add_parser("word-splits")
    args = parser.parse_args()
    if args.cmd == "nepali-only":
        cmd_nepali_only(args)
    elif args.cmd == "violations":
        cmd_violations(args)
    elif args.cmd == "word-splits":
        cmd_word_splits(args)


if __name__ == "__main__":
    main()

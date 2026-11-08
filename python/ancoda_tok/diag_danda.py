"""Danda (purna biram) diagnostics for the paper (FLORES+ npi / hin)."""

from __future__ import annotations

import json
from pathlib import Path

import sentencepiece as spm
from tokenizers import Tokenizer

from ancoda_tok.clean.common import base_manifest, finish_manifest, save_json
from ancoda_tok.export_hf import bytes_to_unicode, load_hf
from ancoda_tok.storage import REPO_ROOT

FLORES = REPO_ROOT / "third_party" / "yak-tokenizer" / "data" / "eval" / "flores_plus" / "devtest"
YAK_V1 = (
    REPO_ROOT
    / "data/hf_cache/hub/models--ampixa--yak-tokenizer/snapshots"
    / "12b6d3010ec542b484ec3b673354852fe8bb85fb/v1.0/64k/tokenizer.json"
)
YAK_V2 = (
    REPO_ROOT
    / "data/hf_cache/hub/models--ampixa--yak-tokenizer/snapshots"
    / "12b6d3010ec542b484ec3b673354852fe8bb85fb/v2.0/64k/tokenizer.json"
)
SID = (
    REPO_ROOT
    / "data/hf_cache/hub/models--sidskarki--nepali-bpe-tokenizer/snapshots"
    / "fc5773dcf7b9dc2272e8f595f99828969621a855/nepali_bpe_64k.model"
)
PHRASES = ["छ।", "हो।", "थियो।", "गर्यो।", "छन्।"]
DANDA = "।"
DOUBLE = "॥"


def _hf_pieces(tok, text: str) -> list[str]:
    inverse = {c: b for b, c in bytes_to_unicode().items()}
    ids = tok.encode(text, add_special_tokens=False).ids
    out = []
    for i in ids:
        piece = tok.id_to_token(i)
        try:
            out.append(bytes(inverse[ch] for ch in piece).decode("utf-8"))
        except Exception:
            out.append(piece)
    return out


def _count_file(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    words = text.split()
    n_danda = text.count(DANDA)
    n_double = text.count(DOUBLE)
    # Attached to previous word: occurrences of non-space + danda as a whitespace word,
    # or danda glued inside a whitespace token.
    attached = 0
    standalone = 0
    for w in words:
        if DANDA in w or DOUBLE in w:
            # strip trailing/leading dandas for "attached"
            body = w
            # count dandas in this word
            dcount = body.count(DANDA) + body.count(DOUBLE)
            stripped = body.replace(DANDA, "").replace(DOUBLE, "")
            if stripped == "":
                standalone += dcount
            else:
                attached += dcount
    # Also dandas that appear only as line punctuation with spaces — already in words.
    # Tokens-per-word spent on danda-only tokens under n60.
    n60 = load_hf(REPO_ROOT / "artifacts/tokenizers/ancoda-64k-n60/tokenizer.json")
    whole = "\n".join(text.splitlines())
    ids = n60.encode(whole, add_special_tokens=False).ids
    inverse = {c: b for b, c in bytes_to_unicode().items()}
    danda_only_tokens = 0
    for i in ids:
        piece = n60.id_to_token(i)
        try:
            decoded = bytes(inverse[ch] for ch in piece).decode("utf-8")
        except Exception:
            continue
        if decoded in (DANDA, DOUBLE) or decoded.strip() in (DANDA, DOUBLE):
            danda_only_tokens += 1
    n_words = len(words)
    return {
        "path": str(path.relative_to(REPO_ROOT)),
        "danda_count": n_danda,
        "double_danda_count": n_double,
        "attached_to_previous_word": attached,
        "standalone_word_dandas": standalone,
        "words": n_words,
        "danda_only_token_count_n60": danda_only_tokens,
        "danda_only_tokens_per_word_n60": (
            0.0 if n_words == 0 else danda_only_tokens / n_words
        ),
    }


def main() -> None:
    npi = _count_file(FLORES / "npi_Deva.txt")
    hin = _count_file(FLORES / "hin_Deva.txt")
    yak1 = Tokenizer.from_file(str(YAK_V1))
    yak2 = Tokenizer.from_file(str(YAK_V2))
    for t in (yak1, yak2):
        if hasattr(t, "encode_special_tokens"):
            t.encode_special_tokens = True
    sid = spm.SentencePieceProcessor(model_file=str(SID))
    n60 = load_hf(REPO_ROOT / "artifacts/tokenizers/ancoda-64k-n60/tokenizer.json")
    splits = {}
    for phrase in PHRASES:
        splits[phrase] = {
            "n60": _hf_pieces(n60, phrase),
            "yak_v1_64k": _hf_pieces(yak1, phrase),
            "yak_v2_64k": _hf_pieces(yak2, phrase),
            "sidskarki_64k": sid.encode(phrase, out_type=str),
        }
    out = {
        "npi_Deva": npi,
        "hin_Deva": hin,
        "phrase_splits": splits,
        "decision": (
            "Danda (purna biram U+0964) and double danda (U+0965) stay separate "
            "tokens, like '.' in English. Recorded 8 Nov 2026."
        ),
    }
    dest = REPO_ROOT / "artifacts" / "eval" / "m7" / "diag_danda.json"
    save_json(dest, finish_manifest(base_manifest(kind="danda_diag", **out)))
    print(json.dumps(out, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

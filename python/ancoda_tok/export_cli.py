"""Write tokenizer.json, the tiktoken files, and an export manifest from ranks.json."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ancoda_tok.clean.common import base_manifest, finish_manifest, save_json, sha256_file
from ancoda_tok.export_hf import export_from_ranks, read_pattern
from ancoda_tok.export_tiktoken import export_tiktoken, load_tiktoken
from ancoda_tok.storage import REPO_ROOT


def export_dir(ranks_path: Path) -> Path:
    ranks = json.loads(ranks_path.read_text(encoding="utf-8"))
    tokens = [bytes.fromhex(item) for item in ranks["tokens_hex"]]
    specials = [(item["name"], int(item["id"])) for item in ranks["special_tokens"]]
    out_dir = ranks_path.parent
    export_from_ranks(ranks_path, out_dir)
    export_tiktoken(tokens, specials, out_dir, read_pattern())
    # Load both to prove the files parse. Equality is the guarantee test.
    from ancoda_tok.export_hf import load_hf

    load_hf(out_dir / "tokenizer.json")
    load_tiktoken(out_dir)
    manifest = base_manifest(
        kind="tokenizer_export",
        ranks_sha256=sha256_file(ranks_path),
        tokenizer_json_sha256=sha256_file(out_dir / "tokenizer.json"),
        tiktoken_sha256=sha256_file(out_dir / "tiktoken" / "ancoda.tiktoken"),
        pattern_sha256=sha256_file(REPO_ROOT / "configs" / "pretokenizer_pattern.txt"),
        encode_special_tokens="load_hf sets Tokenizer.encode_special_tokens = True; not stored in tokenizer.json",
    )
    save_json(out_dir / "export_manifest.json", finish_manifest(manifest))
    print(f"wrote {out_dir / 'tokenizer.json'}", flush=True)
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ranks",
        type=Path,
        default=REPO_ROOT / "artifacts" / "tokenizers" / "ancoda-64k" / "ranks.json",
    )
    args = parser.parse_args()
    export_dir(args.ranks)


if __name__ == "__main__":
    main()

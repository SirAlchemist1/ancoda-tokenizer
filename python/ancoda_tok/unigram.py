"""Train the mix-sweep SentencePiece Unigram control (docs/SPEC.md mix-sweep).

Tokenizer-level control only: same snapshot, vocab 64k, byte_fallback=true,
normalization_rule_name=identity. Not an algorithmic BPE-vs-Unigram comparison.

Memory note (8 Nov 2026 diagnostic): the first attempt with
input_sentence_size=10_000_000 ballooned past tens of GB. Retrain uses
input_sentence_size=1_000_000 with shuffle_input_sentence=true and records
that the corpus was subsampled for memory.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import pyarrow.parquet as pq
import sentencepiece as spm

from ancoda_tok.clean.common import (
    FREE_FLOOR,
    base_manifest,
    finish_manifest,
    free_bytes,
    save_json,
    sha256_file,
)
from ancoda_tok.inventory import find_snapshot
from ancoda_tok.storage import REPO_ROOT

SEED = 20261104
VOCAB = 65536
# Subsampled for memory after the full-corpus attempt (see module docstring).
INPUT_SENTENCE_SIZE = 1_000_000
SHUFFLE_INPUT_SENTENCE = True


def _write_corpus(snapshot: Path, corpus: Path) -> dict:
    corpus.parent.mkdir(parents=True, exist_ok=True)
    docs = 0
    text_bytes = 0
    with corpus.open("w", encoding="utf-8") as handle:
        for shard in sorted(snapshot.glob("shard-*.parquet")):
            parquet = pq.ParquetFile(shard)
            for batch in parquet.iter_batches(columns=["text"], batch_size=256):
                for value in batch.column(0).to_pylist():
                    if not value:
                        continue
                    handle.write(value.replace("\n", " "))
                    handle.write("\n")
                    docs += 1
                    text_bytes += len(value.encode("utf-8"))
    return {"docs": docs, "text_bytes": text_bytes, "corpus_sha256": sha256_file(corpus)}


def train(snapshot: Path, out_dir: Path, vocab_size: int = VOCAB) -> Path:
    free = free_bytes()
    if free < FREE_FLOOR:
        raise SystemExit(f"STOP: {free} bytes free, floor is {FREE_FLOOR}")
    out_dir.mkdir(parents=True, exist_ok=True)
    corpus = out_dir / "corpus.txt"
    model_prefix = out_dir / "unigram"
    model_path = Path(str(model_prefix) + ".model")
    if model_path.is_file() and (out_dir / "manifest.json").is_file():
        print(f"resume: {model_path} already exists", flush=True)
        return model_path
    if corpus.is_file() and corpus.stat().st_size > 0:
        print(f"reusing existing corpus {corpus} ({corpus.stat().st_size} bytes)", flush=True)
        corpus_stats = {
            "docs": None,
            "text_bytes": corpus.stat().st_size,
            "corpus_sha256": sha256_file(corpus),
            "corpus_reused": True,
        }
    else:
        print(f"writing corpus from {snapshot}", flush=True)
        corpus_stats = _write_corpus(snapshot, corpus)
        corpus_stats["corpus_reused"] = False
        print(
            f"corpus docs={corpus_stats['docs']} text_bytes={corpus_stats['text_bytes']}",
            flush=True,
        )
    print(
        f"SentencePiece Unigram train input_sentence_size={INPUT_SENTENCE_SIZE} "
        f"shuffle_input_sentence={SHUFFLE_INPUT_SENTENCE} "
        f"(subsampled for memory)",
        flush=True,
    )
    spm.SentencePieceTrainer.train(
        input=str(corpus),
        model_prefix=str(model_prefix),
        vocab_size=vocab_size,
        model_type="unigram",
        character_coverage=0.9995,
        byte_fallback=True,
        normalization_rule_name="identity",
        unk_id=0,
        bos_id=1,
        eos_id=2,
        pad_id=3,
        train_extremely_large_corpus=True,
        num_threads=int(os.environ.get("ANCODA_WORKERS", "5")),
        seed_sentencepiece_size=1_000_000,
        shuffle_input_sentence=SHUFFLE_INPUT_SENTENCE,
        input_sentence_size=INPUT_SENTENCE_SIZE,
        max_sentence_length=16_000,
        hard_vocab_limit=False,
    )
    # Remove the training corpus after the model exists (large, gitignored).
    if corpus.is_file():
        corpus.unlink()
    manifest = base_manifest(
        kind="unigram_control",
        snapshot=str(snapshot.relative_to(REPO_ROOT)),
        vocab_size=vocab_size,
        byte_fallback=True,
        normalization_rule_name="identity",
        seed=SEED,
        input_sentence_size=INPUT_SENTENCE_SIZE,
        shuffle_input_sentence=SHUFFLE_INPUT_SENTENCE,
        subsampled_for_memory=True,
        model_sha256=sha256_file(model_path),
        vocab_sha256=sha256_file(Path(str(model_prefix) + ".vocab")),
        **corpus_stats,
    )
    save_json(out_dir / "manifest.json", finish_manifest(manifest))
    print(f"wrote {model_path}", flush=True)
    return model_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot-prefix", default="n50")
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "artifacts" / "tokenizers" / "ladder-unigram",
    )
    parser.add_argument("--vocab-size", type=int, default=VOCAB)
    args = parser.parse_args()
    snapshot = find_snapshot(args.snapshot_prefix)
    train(snapshot, args.out, args.vocab_size)


if __name__ == "__main__":
    main()

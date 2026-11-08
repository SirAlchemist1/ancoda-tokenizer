"""Compare the plain trainer with Hugging Face ``BpeTrainer`` on the 1 GB slice.

The installed library is tokenizers 0.23.3. Word counts are fed by yielding each
pre-token ``count`` times; ``train_from_iterator`` is what records them.
``initial_alphabet`` is ``ByteLevel.alphabet()`` and the vocabulary size is 65536.
"""

from __future__ import annotations

import json
from pathlib import Path

from tokenizers import Regex, Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel, Sequence, Split
from tokenizers.trainers import BpeTrainer

from ancoda_tok.counts import iter_entries
from ancoda_tok.export_hf import bytes_to_unicode, export_from_ranks, load_hf, read_pattern
from ancoda_tok.export_tiktoken import export_tiktoken
from ancoda_tok.inventory import find_n50_snapshot
from ancoda_tok.storage import REPO_ROOT

PLAIN_RANKS = REPO_ROOT / "artifacts" / "tokenizers" / "plain-1gb" / "ranks.json"
COUNTS = REPO_ROOT / "artifacts" / "counts" / "n50_1gb" / "pretoken_counts.bin"
REPORT = REPO_ROOT / "artifacts" / "validation" / "plain_vs_hf.md"


def _inverse_encoder() -> dict[str, int]:
    return {char: byte for byte, char in bytes_to_unicode().items()}


def _decode_piece(text: str, inverse: dict[str, int]) -> bytes:
    return bytes(inverse[char] for char in text)


def our_merges(ranks_path: Path) -> list[tuple[bytes, bytes, int]]:
    ranks = json.loads(ranks_path.read_text(encoding="utf-8"))
    tokens = [bytes.fromhex(item) for item in ranks["tokens_hex"]]
    merges = []
    for row in ranks["block2_merges"]:
        merges.append((tokens[row["left"]], tokens[row["right"]], int(row["count"])))
    return merges


def _yield_counts(path: Path):
    for raw, count in iter_entries(path):
        word = raw.decode("utf-8")
        for _ in range(count):
            yield word


def train_hf(counts: Path, vocab_size: int = 65536) -> Tokenizer:
    tokenizer = Tokenizer(BPE(unk_token=None, fuse_unk=False, byte_fallback=False, dropout=None))
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False, trim_offsets=False, use_regex=False)
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=0,
        show_progress=True,
        special_tokens=[],
        initial_alphabet=ByteLevel.alphabet(),
    )
    tokenizer.train_from_iterator(_yield_counts(counts), trainer=trainer)
    return tokenizer


def hf_merges(tokenizer: Tokenizer, inverse: dict[str, int]) -> list[tuple[bytes, bytes]]:
    blob = json.loads(tokenizer.to_str())
    merges = []
    for row in blob["model"]["merges"]:
        if isinstance(row, str):
            left, right = row.split(" ")
        else:
            left, right = row
        merges.append((_decode_piece(left, inverse), _decode_piece(right, inverse)))
    return merges


def _lines(limit: int) -> list[str]:
    found: list[str] = []
    snapshot = find_n50_snapshot()
    import pyarrow.parquet as pq

    for shard in sorted(snapshot.glob("shard-*.parquet")):
        parquet = pq.ParquetFile(shard)
        for batch in parquet.iter_batches(columns=["text"], batch_size=64):
            for value in batch.column(0).to_pylist():
                text = "" if value is None else value
                for line in text.split("\n"):
                    found.append(line)
                    if len(found) >= limit:
                        return found
    return found


def compare() -> str:
    inverse = _inverse_encoder()
    ours = our_merges(PLAIN_RANKS)
    print("training Hugging Face BpeTrainer on the 1 GB counts", flush=True)
    hf_tok = train_hf(COUNTS)
    theirs = hf_merges(hf_tok, inverse)
    same = len(ours) == len(theirs) and all(
        left == other[0] and right == other[1]
        for (left, right, _count), other in zip(ours, theirs, strict=False)
    )
    first_diff = None
    tie = None
    if not same:
        limit = min(len(ours), len(theirs))
        for index in range(limit):
            left, right, count = ours[index]
            if (left, right) != theirs[index]:
                first_diff = index
                # A later merge of the HF pair at the same recorded count is a tie.
                wanted = theirs[index]
                for later, (l2, r2, count2) in enumerate(ours[index:], start=index):
                    if (l2, r2) == wanted and count2 == count:
                        tie = (index, count, later)
                        break
                break
        if first_diff is None and len(ours) != len(theirs):
            first_diff = limit
    pattern = read_pattern()
    hf_tok.pre_tokenizer = Sequence(
        [
            Split(Regex(pattern), behavior="isolated", invert=False),
            ByteLevel(add_prefix_space=False, trim_offsets=False, use_regex=False),
        ]
    )
    hf_tok.decoder = ByteLevelDecoder()
    export_from_ranks(PLAIN_RANKS, PLAIN_RANKS.parent)
    ranks = json.loads(PLAIN_RANKS.read_text(encoding="utf-8"))
    tokens = [bytes.fromhex(item) for item in ranks["tokens_hex"]]
    export_tiktoken(tokens, [], PLAIN_RANKS.parent, pattern)
    ours_tok = load_hf(PLAIN_RANKS.parent / "tokenizer.json")
    lines = _lines(100_000)
    seg_mismatch = 0
    decode_mismatch = 0
    examples: list[str] = []
    for line in lines:
        hf_ids = hf_tok.encode(line, add_special_tokens=False).ids
        our_ids = ours_tok.encode(line, add_special_tokens=False).ids
        hf_bytes = b"".join(_decode_piece(hf_tok.id_to_token(i), inverse) for i in hf_ids)
        our_bytes = b"".join(tokens[i] for i in our_ids)
        if hf_bytes != our_bytes:
            seg_mismatch += 1
            if len(examples) < 5:
                examples.append(repr(line[:80]))
        if hf_tok.decode(hf_ids, skip_special_tokens=False) != line:
            decode_mismatch += 1
        if ours_tok.decode(our_ids, skip_special_tokens=False) != line:
            decode_mismatch += 1
    status = "PASS" if same or tie is not None else "DIFF"
    if seg_mismatch and tie is None and not same:
        status = "FAIL"
    lines_out = [
        "# Plain trainer vs Hugging Face BpeTrainer",
        "",
        "Slice: `artifacts/counts/n50_1gb/pretoken_counts.bin`.",
        "Our trainer: `configs/plain_1gb.yaml` (no blocks, no bans, vocab 65536).",
        "Hugging Face: tokenizers 0.23.3 `BpeTrainer`, `initial_alphabet=ByteLevel.alphabet()`,",
        "`vocab_size=65536`, `min_frequency=0`, `special_tokens=[]`.",
        "Pre-tokens were yielded `count` times so the trainer sees the same frequencies.",
        "Hugging Face assigns the initial alphabet by code point, not by byte value,",
        "so raw token ids are not the same numbers. The comparison is the merge byte strings",
        "and the encoded byte pieces.",
        "",
        f"- our merges: {len(ours)}",
        f"- HF merges: {len(theirs)}",
        f"- lists identical: {same}",
        f"- first difference index: {first_diff}",
        f"- first difference is an equal-count tie in our merge list: {tie}",
        f"- 100k-line byte-segmentation mismatches: {seg_mismatch}",
        f"- decode mismatches: {decode_mismatch}",
        f"- status: {status}",
        "",
    ]
    if examples:
        lines_out.append("Example lines whose byte pieces differed:")
        lines_out.extend(f"- {item}" for item in examples)
        lines_out.append("")
    if first_diff is not None and first_diff < len(ours) and first_diff < len(theirs):
        left, right, count = ours[first_diff]
        lines_out.append(
            f"First pair: ours {left.hex()} + {right.hex()} count {count}; "
            f"HF {theirs[first_diff][0].hex()} + {theirs[first_diff][1].hex()}."
        )
        lines_out.append("")
    text = "\n".join(lines_out)
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(text, encoding="utf-8")
    print(text, flush=True)
    return status


def main() -> None:
    status = compare()
    if status == "FAIL":
        raise SystemExit(1)


if __name__ == "__main__":
    main()

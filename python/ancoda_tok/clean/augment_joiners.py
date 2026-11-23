"""M4 step 6: synthetic ZWJ/ZWNJ joiner-augmented Nepali copy (docs/SPEC.md 3.8)."""

from __future__ import annotations

import random
import re
from pathlib import Path

from ancoda_tok.clean.common import (
    CLEAN_SCHEMA,
    REPO_ROOT,
    SEED,
    ShardWriter,
    base_manifest,
    finish_manifest,
    save_json,
    utf8_len,
)

# Virama then a Devanagari letter (consonant site), no joiner already present.
_VIRAMA_BEFORE_LETTER = re.compile(
    "(\u094d)(?![\u200c\u200d])([\u0905-\u0939\u0958-\u095f\u0972-\u097f])"
)


def insert_joiners(text: str, joiner: str) -> str:
    """Insert ``joiner`` after every virama that precedes a Devanagari letter."""
    return _VIRAMA_BEFORE_LETTER.sub(rf"\1{joiner}\2", text)


def strip_inserted_joiners(synthetic: str, original: str) -> str:
    """Exact inverse of ``insert_joiners`` relative to ``original``.

    Only removes ZWJ/ZWNJ that ``insert_joiners`` would have placed after a
    virama that already preceded a Devanagari letter in ``original``. Pre-existing
    virama–joiner–letter sequences in the source are left alone.
    """
    chars = list(synthetic)
    insertions_before = 0
    to_delete: list[int] = []
    for match in _VIRAMA_BEFORE_LETTER.finditer(original):
        # Virama at match.start(); inserted joiner sits immediately after it,
        # shifted by prior insertions.
        joiner_pos = match.start() + 1 + insertions_before
        to_delete.append(joiner_pos)
        insertions_before += 1
    for pos in reversed(to_delete):
        if 0 <= pos < len(chars) and chars[pos] in "\u200c\u200d":
            del chars[pos]
    return "".join(chars)


def roundtrip_ok(original: str, joiner: str) -> bool:
    synthetic = insert_joiners(original, joiner)
    return strip_inserted_joiners(synthetic, original) == original


def build_joiner_copy(
    docs: list[dict],
    *,
    target_bytes: int,
    seed: int = SEED,
    out_dir: Path | None = None,
) -> dict:
    """Build a synthetic joiner corpus of about ``target_bytes`` from ``docs``.

    Half the selected documents get ZWJ, half get ZWNJ. Documents are taken in
    shuffled order and never repeated. Original text is never modified in place.
    """
    out_dir = out_dir or (REPO_ROOT / "data" / "clean" / "_synthetic_joiners")
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    order = list(range(len(docs)))
    rng.shuffle(order)

    writer = ShardWriter(out_dir, schema=CLEAN_SCHEMA)
    used = 0
    n_zwj = 0
    n_zwnj = 0
    for idx, di in enumerate(order):
        if used >= target_bytes:
            break
        row = docs[di]
        text = row["text"]
        joiner = "\u200d" if idx % 2 == 0 else "\u200c"
        synthetic = insert_joiners(text, joiner)
        if synthetic == text:
            continue  # no virama sites; skip
        assert strip_inserted_joiners(synthetic, text) == text
        new_row = {
            "doc_id": f"synthetic\x1f{row['doc_id']}",
            "text": synthetic,
            "source": row.get("source", "synthetic"),
            "license": row.get("license", ""),
            "bucket": "Nepali",
            "glotlid_label": "synthetic",
            "glotlid_prob": 1.0,
        }
        writer.add(new_row)
        used += utf8_len(synthetic)
        if joiner == "\u200d":
            n_zwj += 1
        else:
            n_zwnj += 1
    writer.close()
    manifest = finish_manifest(
        {
            **base_manifest(stage="joiner_augmentation", synthetic=True),
            "docs": writer.docs,
            "text_bytes": writer.text_bytes,
            "target_bytes": target_bytes,
            "n_zwj_docs": n_zwj,
            "n_zwnj_docs": n_zwnj,
            "shards": writer.shards,
            "seed": seed,
            "tag": "synthetic",
        }
    )
    save_json(out_dir / "manifest.json", manifest)
    return manifest

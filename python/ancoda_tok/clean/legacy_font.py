"""cleaning step 3: dictionary-based Preeti / Kantipur legacy-font detector.

Replaces the logistic classifier (failed 99% precision). Decision recorded in
docs/DECISIONS.md (5 Nov 2026).
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from ancoda_tok.clean.common import (
    REPO_ROOT,
    SEED,
    artifacts_clean,
    is_latin_line,
    list_shards,
    load_mix,
    raw_dir,
    save_json,
    sha256_file,
)
from ancoda_tok.clean.lid import GlotLID
from ancoda_tok.clean.preeti_convert import convert_unicode_to_preeti
from ancoda_tok.clean.utf8 import clean_utf8_text

_DEV_WORD = re.compile(r"[\u0900-\u097F]+")


def is_devanagari_word(token: str) -> bool:
    return bool(_DEV_WORD.search(token))


def tokenize(line: str) -> list[str]:
    return [t for t in line.split() if t]


@dataclass(frozen=True)
class PreetiLexicon:
    """Set of Preeti-converted forms (length ≥ 2) from frequent Devanagari words."""

    forms: frozenset[str]
    path: Path
    sha256: str
    n_source_words: int
    n_forms: int

    def contains(self, token: str) -> bool:
        return token in self.forms


def lexicon_path() -> Path:
    return artifacts_clean() / "preeti_lexicon.tsv"


def load_lexicon(path: Path | None = None) -> PreetiLexicon:
    path = path or lexicon_path()
    forms: set[str] = set()
    n_source = 0
    with path.open(encoding="utf-8") as handle:
        header = handle.readline()
        if not header.startswith("preeti_form"):
            # allow headerless
            handle.seek(0)
        for line in handle:
            line = line.rstrip("\n")
            if not line or line.startswith("preeti_form"):
                continue
            parts = line.split("\t")
            form = parts[0]
            if len(form) >= 2:
                forms.add(form)
            n_source += 1
    digest = sha256_file(path)
    return PreetiLexicon(
        forms=frozenset(forms),
        path=path,
        sha256=digest,
        n_source_words=n_source,
        n_forms=len(forms),
    )


def build_lexicon(
    *,
    top_n: int = 50_000,
    out_path: Path | None = None,
    lid: GlotLID | None = None,
) -> PreetiLexicon:
    """Count Devanagari words in Nepali sources after UTF-8 + GlotLID keep."""
    out_path = out_path or lexicon_path()
    mix = load_mix()
    sources = mix["buckets"]["Nepali"]["sources"]
    keep_labels = set(mix["buckets"]["Nepali"]["keep_labels"])
    if lid is None:
        print("loading GlotLID v3 for lexicon build", flush=True)
        lid = GlotLID()

    counts: Counter[str] = Counter()
    docs_kept = 0
    docs_seen = 0
    for source_id in sources:
        directory = raw_dir(source_id)
        if not directory.exists():
            print(f"skip missing raw {source_id}", flush=True)
            continue
        shards = list_shards(directory)
        print(f"lexicon scan {source_id} shards={len(shards)}", flush=True)
        for shard_i, shard in enumerate(shards):
            import pyarrow.parquet as pq

            table = pq.read_table(shard, columns=["text"])
            texts = table.column("text")
            for i in range(table.num_rows):
                docs_seen += 1
                cleaned = clean_utf8_text(texts[i].as_py())
                if cleaned is None:
                    continue
                label, _ = lid.predict(cleaned)
                if label not in keep_labels:
                    continue
                docs_kept += 1
                for tok in cleaned.split():
                    if is_devanagari_word(tok):
                        counts[tok] += 1
            if (shard_i + 1) % 10 == 0 or shard_i + 1 == len(shards):
                print(
                    f"  {source_id} shard {shard_i + 1}/{len(shards)} "
                    f"docs_kept={docs_kept} vocab={len(counts)}",
                    flush=True,
                )

    # Top N by count desc, then word ascending (deterministic).
    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:top_n]
    rows: list[str] = ["preeti_form\tdevanagari\tcount"]
    kept_forms: set[str] = set()
    for word, count in ranked:
        preeti = convert_unicode_to_preeti(word)
        if len(preeti) < 2:
            continue
        rows.append(f"{preeti}\t{word}\t{count}")
        kept_forms.add(preeti)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    digest = sha256_file(out_path)
    meta = {
        "path": str(out_path.relative_to(REPO_ROOT)),
        "sha256": digest,
        "top_n": top_n,
        "n_source_words": len(ranked),
        "n_forms": len(kept_forms),
        "docs_seen": docs_seen,
        "docs_kept_npi_Deva": docs_kept,
        "sources": sources,
        "converter": "globalpolicy/UnicodeToPreeti",
        "converter_commit": mix["legacy_font"]["converter_commit"],
        "converter_license": "MIT",
        "seed": SEED,
        "tie_break": "count descending, then Devanagari word ascending",
    }
    save_json(artifacts_clean() / "preeti_lexicon.manifest.json", meta)
    print(
        f"wrote {out_path} forms={len(kept_forms)} sha256={digest}",
        flush=True,
    )
    return PreetiLexicon(
        forms=frozenset(kept_forms),
        path=out_path,
        sha256=digest,
        n_source_words=len(ranked),
        n_forms=len(kept_forms),
    )


def line_is_legacy(line: str, lexicon: PreetiLexicon) -> bool:
    """Flag a Latin-script line if ≥3 lexicon hits and ≥50% of tokens hit."""
    if not is_latin_line(line):
        return False
    tokens = tokenize(line)
    if not tokens:
        return False
    hits = sum(1 for t in tokens if lexicon.contains(t))
    if hits < 3:
        return False
    return hits / len(tokens) >= 0.5


def document_legacy_line_share(
    text: str, lexicon: PreetiLexicon
) -> tuple[float, int, int, int]:
    """Return (flagged_share, flagged, nonempty_lines, latin_lines)."""
    flagged = 0
    nonempty = 0
    latin = 0
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        nonempty += 1
        if is_latin_line(stripped):
            latin += 1
            if line_is_legacy(stripped, lexicon):
                flagged += 1
    if nonempty == 0:
        return 0.0, 0, 0, 0
    return flagged / nonempty, flagged, nonempty, latin


def should_drop_document(
    text: str,
    lexicon: PreetiLexicon,
    threshold: float = 0.20,
) -> bool:
    share, _, nonempty, _ = document_legacy_line_share(text, lexicon)
    if nonempty == 0:
        return False
    return share > threshold


# Back-compat aliases used by older call sites / tests.
def should_drop_document_clf(text: str, clf: object, threshold: float = 0.20) -> bool:
    raise TypeError("logistic classifier removed; pass PreetiLexicon")

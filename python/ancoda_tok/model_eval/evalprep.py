"""Sealed, FLORES, and perturbed eval sets. Stored text is not normalized."""

from __future__ import annotations

import io
import json
from collections import defaultdict
from pathlib import Path

import pyarrow.parquet as pq
import zstandard as zstd

from ancoda_tok.clean.common import base_manifest, finish_manifest, sha256_file
from ancoda_tok.model_eval.common import REPO_ROOT, canonical_lang, doc_id_sha256
from ancoda_tok.model_eval.perturb import ZWJ, ZWNJ, insert_joiner, misspell_corpus
from ancoda_tok.model_eval.stream import MODEL_EVAL

NO_JOINERS = REPO_ROOT / "docs" / "sealed_test_set_no_joiners.json"
INDEX = REPO_ROOT / "data" / "clean" / "_snapshot_index.parquet"
EVAL = MODEL_EVAL / "eval"
FLORES = {
    "npi": "npi_Deva.txt",
    "hin": "hin_Deva.txt",
    "eng": "eng_Latn.txt",
    "mai": "mai_Deva.txt",
    "bho": "bho_Deva.txt",
    "awa": "awa_Deva.txt",
    "mag": "mag_Deva.txt",
    "san": "san_Deva.txt",
    "mar": "mar_Deva.txt",
    "hne": "hne_Deva.txt",
}
FLORES_DIRS = [
    REPO_ROOT / "third_party" / "yak-tokenizer" / "data" / "eval" / "flores_plus" / "devtest",
    REPO_ROOT / "data" / "eval" / "flores_plus_extra" / "devtest",
]


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cctx = zstd.ZstdCompressor(level=3)
    with path.open("wb") as raw, cctx.stream_writer(raw) as writer:
        text = io.TextIOWrapper(writer, encoding="utf-8")
        for row in rows:
            text.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
        text.flush()
        text.detach()


def _flores_path(name: str) -> Path:
    for directory in FLORES_DIRS:
        path = directory / name
        if path.is_file():
            return path
    raise SystemExit(f"STOP: missing FLORES file {name}")


def _sealed_rows(wanted: set[str]) -> list[dict]:
    found: list[dict] = []
    pf = pq.ParquetFile(INDEX)
    columns = ["doc_id", "bucket", "lang", "text_bytes", "shard", "row"]
    for batch in pf.iter_batches(columns=columns, batch_size=16384):
        data = {name: batch.column(name).to_pylist() for name in columns}
        for i, doc_id in enumerate(data["doc_id"]):
            digest = doc_id_sha256(doc_id)
            if digest not in wanted:
                continue
            found.append(
                {
                    "doc_id": doc_id,
                    "lang": canonical_lang(data["bucket"][i], data["lang"][i]),
                    "text_bytes": int(data["text_bytes"][i]),
                    "clean_shard": data["shard"][i],
                    "row": int(data["row"][i]),
                }
            )
    by_shard: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for i, row in enumerate(found):
        by_shard[row["clean_shard"]].append((i, row["row"]))
    texts = [""] * len(found)
    for shard, items in sorted(by_shard.items()):
        column = pq.read_table(REPO_ROOT / shard, columns=["text"]).column(0)
        for i, row_i in items:
            texts[i] = column[row_i].as_py() or ""
    rows = []
    for meta, text in zip(found, texts, strict=True):
        raw = len(text.encode("utf-8"))
        if raw != meta["text_bytes"]:
            raise SystemExit(f"STOP: sealed text_bytes mismatch for {meta['doc_id']}")
        rows.append(
            {
                "doc_id": meta["doc_id"],
                "lang": meta["lang"],
                "text": text,
                "original_bytes": raw,
                "variant": "orig",
            }
        )
    rows.sort(key=lambda row: row["doc_id"])
    return rows


def _variant_rows(rows: list[dict], name: str, texts: list[str]) -> list[dict]:
    out = []
    for row, text in zip(rows, texts, strict=True):
        out.append(
            {
                "doc_id": row["doc_id"],
                "lang": row["lang"],
                "text": text,
                "original_bytes": row["original_bytes"],
                "variant": name,
            }
        )
    return out


def prepare_eval() -> dict:
    payload = json.loads(NO_JOINERS.read_text(encoding="utf-8"))
    wanted = set(payload["doc_id_sha256"])
    print(f"recovering sealed no-joiner docs n={len(wanted)}", flush=True)
    sealed = _sealed_rows(wanted)
    if len(sealed) != int(payload["n_docs"]):
        raise SystemExit(f"STOP: recovered {len(sealed)} sealed docs != {payload['n_docs']}")
    _write_jsonl(EVAL / "sealed.jsonl.zst", sealed)
    nepali = [row for row in sealed if row["lang"] == "Nepali"]
    if len(nepali) < 2000:
        raise SystemExit(f"STOP: sealed Nepali has {len(nepali)} docs, need 2000")
    sample = nepali[:2000]
    _write_jsonl(EVAL / "sealed_npi_2000.jsonl.zst", sample)
    flores_rows: dict[str, list[dict]] = {}
    for code, filename in FLORES.items():
        lines = [
            line for line in _flores_path(filename).read_text(encoding="utf-8").splitlines() if line
        ]
        rows = [
            {
                "doc_id": f"flores\x1f{code}\x1f{i}",
                "lang": code,
                "text": line,
                "original_bytes": len(line.encode("utf-8")),
                "variant": "orig",
            }
            for i, line in enumerate(lines)
        ]
        flores_rows[code] = rows
        _write_jsonl(EVAL / "flores" / f"{code}.jsonl.zst", rows)
    npi = flores_rows["npi"]
    npi_text = [row["text"] for row in npi]
    sample_text = [row["text"] for row in sample]
    variants = {
        "misspell": (
            misspell_corpus(npi_text),
            misspell_corpus(sample_text),
        ),
        "zwj": (
            [insert_joiner(text, ZWJ) for text in npi_text],
            [insert_joiner(text, ZWJ) for text in sample_text],
        ),
        "zwnj": (
            [insert_joiner(text, ZWNJ) for text in npi_text],
            [insert_joiner(text, ZWNJ) for text in sample_text],
        ),
    }
    for name, (flores_text, sealed_text) in variants.items():
        _write_jsonl(
            EVAL / "perturb" / f"flores_npi_{name}.jsonl.zst",
            _variant_rows(npi, name, flores_text),
        )
        _write_jsonl(
            EVAL / "perturb" / f"sealed_npi_{name}.jsonl.zst",
            _variant_rows(sample, name, sealed_text),
        )
    langs: dict[str, int] = defaultdict(int)
    for row in sealed:
        langs[row["lang"]] += 1
    manifest = finish_manifest(
        base_manifest(
            kind="model_eval_eval",
            input_sha256={"sealed_no_joiners": sha256_file(NO_JOINERS)},
            n_sealed=len(sealed),
            n_nepali_sample=2000,
            lang_counts=dict(sorted(langs.items())),
            flores=list(FLORES),
        )
    )
    (EVAL / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"eval langs {dict(sorted(langs.items()))}", flush=True)
    return manifest

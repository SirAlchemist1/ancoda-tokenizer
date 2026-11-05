"""Download M2 sources into ``data/raw/<source_id>/``.

Text sources become zstd parquet with columns ``doc_id``, ``text``, ``source``
and ``license``. Tool files are copied as themselves. Each directory gets a
manifest with the Hugging Face revision (or the pinned Papaya commit), so a
later milestone can re-download training text after deleting raw parquet.

Nothing here normalizes Unicode. Caps keep a sorted prefix of the files and
stop before the document that would pass the UTF-8 budget.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
from huggingface_hub.errors import HfHubHTTPError

from ancoda_tok.storage import (
    REPO_ROOT,
    free_bytes_df,
    git_commit,
    list_files,
    load_spec,
    measure_source,
    repo_revision,
)

FREE_FLOOR = 50_000_000_000
SHARD_TEXT_BYTES = 64 * 1024 * 1024
CODE_LICENSES = frozenset({"mit", "apache-2.0", "bsd-2-clause", "bsd-3-clause"})
CODE_COLUMNS = ["code", "repo_name", "path", "language", "license", "size"]
PAPAYA_COMMIT = "b3498528daa3435f091792e0b44bb222a2f8cb68"
PAPAYA_FILES = {
    "data/affix_inventory.tsv": 81127,
    "data/affix_inventory.json": 491081,
    "data/LICENSE": None,
}
SCHEMA = pa.schema(
    [
        ("doc_id", pa.string()),
        ("text", pa.string()),
        ("source", pa.string()),
        ("license", pa.string()),
    ]
)
SAMPLING = (
    "Sorted path order, then file order. Stop before a document that would "
    "exceed the UTF-8 budget. If the first kept document alone exceeds it, "
    "keep that document and stop. Not a random sample."
)
REPO_LICENSE = {
    "HuggingFaceFW/fineweb-2": "odc-by-1.0",
    "HuggingFaceFW/fineweb-edu": "odc-by-1.0",
    "IRIIS-RESEARCH/Nepali-Text-Corpus": "mit",
    "ai4bharat/sangraha": "cc-by-4.0",
    "ai4bharat/IndicCorpV2": "cc0-1.0",
    "HPLT/HPLT2.0_cleaned": "cc0-1.0",
    "open-web-math/open-web-math": "odc-by-1.0",
    "allenai/c4": "odc-by",
    "facebook/belebele": "cc-by-sa-4.0",
    "CohereLabs/aya_dataset": "apache-2.0",
}
UNICODE_SHA256 = {
    "DerivedCoreProperties.txt": (
        "39d35161f2954497f69e08bdb9e701493f476a3d30222de20028feda36c1dabd"
    ),
    "Scripts.txt": "9e88f0a677df47311106340be8ede2ecdacd9c1c931831218d2be6d5508e0039",
    "UnicodeData.txt": "ff58e5823bd095166564a006e47d111130813dcf8bf234ef79fa51a870edb48f",
    "auxiliary/GraphemeBreakTest.txt": (
        "ee2b9354d270ac061b29f09662cafea06341d77e704b8cc6bd72aaeeda363cb5"
    ),
}
PACKAGES = (
    "huggingface-hub",
    "datasets",
    "pyarrow",
    "pyyaml",
    "numpy",
    "polars",
    "regex",
    "tokenizers",
    "tiktoken",
)
MANIFEST_KEYS = (
    "source_id",
    "status",
    "role",
    "repo",
    "hf_revision",
    "input_sha256",
    "config_sha256",
    "git_commit",
    "library_versions",
    "unicode_segmentation",
    "unicode",
    "seed",
    "sampling",
    "started_at",
    "finished_at",
    "docs",
    "text_bytes",
    "file_bytes",
    "shards",
    "raw_deletion",
)


class Stop(Exception):
    """Human stop. ``main`` exits 2."""


def _retry_errors() -> tuple[type[BaseException], ...]:
    found: list[type[BaseException]] = [OSError, HfHubHTTPError]
    for module_name in ("httpx2", "httpx"):
        try:
            module = __import__(module_name)
        except ImportError:
            continue
        for name in ("HTTPError", "TimeoutException", "TransportError"):
            cls = getattr(module, name, None)
            if isinstance(cls, type):
                found.append(cls)
        break
    return tuple(found)


RETRY_ERRORS = _retry_errors()


@dataclass
class Rec:
    index: int
    resume_at: int
    after: int
    payload: object
    invalid: bool = False


@dataclass
class RunContext:
    git_commit: str
    git_dirty: bool
    config_sha256: str
    library_versions: dict[str, str]
    unicode_segmentation: str
    unicode: dict[str, object]


def utf8_len(text: str) -> int:
    return len(text.encode("utf-8"))


def budget_decision(kept: int, nbytes: int, cap: int | None) -> str:
    """``take`` or ``stop``. The first document is kept even if it exceeds ``cap``."""
    if cap is None or kept + nbytes <= cap or kept == 0:
        return "take"
    return "stop"


def make_doc_id(source_id: str, relative_path: str, index: int) -> str:
    return f"{source_id}\x1f{relative_path}\x1f{index}"


def room_ok(free: int, incoming: int, floor: int = FREE_FLOOR) -> bool:
    return free - incoming >= floor


def devanagari_letter_share(text: str) -> float:
    letters = [ch for ch in text if ch.isalpha()]
    if not letters:
        return 0.0
    dev = sum(1 for ch in letters if "\u0900" <= ch <= "\u097f")
    return dev / len(letters)


def script_decision(mean: float | None) -> str:
    """``keep``, ``skip`` or ``stop`` from the mean Devanagari share of letters."""
    if mean is None:
        return "stop"
    if mean >= 0.5:
        return "keep"
    if mean < 0.2:
        return "skip"
    return "stop"


def code_license_kept(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    label = value.strip().lower()
    if label in CODE_LICENSES:
        return label
    return None


def aya_kept(row: dict) -> bool:
    if row.get("language_code") == "nep":
        return True
    language = row.get("language")
    return isinstance(language, str) and language.strip().lower() == "nepali"


def split_plain_documents(lines: list[str]) -> tuple[list[str], str]:
    """Same rule as the streaming reader. Lines have no trailing newline."""
    if any(line.strip() == "" for line in lines):
        docs: list[str] = []
        buf: list[str] = []
        for line in lines:
            if line.strip() == "":
                if buf:
                    docs.append("\n".join(buf))
                    buf = []
            else:
                buf.append(line)
        if buf:
            docs.append("\n".join(buf))
        return docs, "blank-lines"
    return [line for line in lines if line.strip() != ""], "lines"


def detect_blank_separated(path: Path) -> bool:
    with path.open("rb") as handle:
        for raw in handle:
            if raw.strip() == b"":
                return True
    return False


def _decode_line(raw: bytes) -> str:
    line = raw.decode("utf-8")
    line = line.removesuffix("\n")
    line = line.removesuffix("\r")
    return line


def iter_plain_documents(path: Path, start_cursor: int, start_index: int, blank_separated: bool):
    index = start_index
    with path.open("rb") as handle:
        handle.seek(start_cursor)
        if not blank_separated:
            while True:
                resume_at = handle.tell()
                raw = handle.readline()
                if not raw:
                    break
                after = handle.tell()
                try:
                    line = _decode_line(raw)
                except UnicodeDecodeError:
                    yield Rec(index, resume_at, after, None, invalid=True)
                    index += 1
                    continue
                if line.strip() == "":
                    continue
                yield Rec(index, resume_at, after, line)
                index += 1
            return
        buf: list[str] = []
        doc_start = handle.tell()
        while True:
            resume_at = handle.tell()
            raw = handle.readline()
            if not raw:
                break
            after = handle.tell()
            try:
                line = _decode_line(raw)
            except UnicodeDecodeError:
                if buf:
                    yield Rec(index, doc_start, resume_at, "\n".join(buf))
                    index += 1
                    buf = []
                yield Rec(index, resume_at, after, None, invalid=True)
                index += 1
                doc_start = after
                continue
            if line.strip() == "":
                if buf:
                    yield Rec(index, doc_start, after, "\n".join(buf))
                    index += 1
                    buf = []
                doc_start = after
                continue
            if not buf:
                doc_start = resume_at
            buf.append(line)
        if buf:
            yield Rec(index, doc_start, handle.tell(), "\n".join(buf))


def raw_deletion_policy(role: str) -> str:
    if role == "train":
        return (
            "Delete this raw parquet only after the cleaned parquet is written "
            "and verified against its manifest. The revision recorded here is "
            "the re-download key. Do not delete it during M2."
        )
    return "Never delete. Eval-only, decontamination-only and tool files stay."


def license_label(spec: dict) -> str:
    if spec["id"] == "code":
        return ""
    label = REPO_LICENSE.get(spec.get("repo", ""))
    if not label:
        raise Stop(f"no license label recorded for {spec['id']}")
    return label


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def utc_now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    text = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    tmp.write_text(text)
    os.replace(tmp, path)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def file_kind(relative: str) -> str:
    if relative.endswith(".parquet"):
        return "parquet"
    if relative.endswith(".json.gz"):
        return "json.gz"
    if relative.endswith(".jsonl"):
        return "jsonl"
    if relative.endswith(".txt"):
        return "plain"
    raise Stop(f"unsupported input file {relative}")


def write_table(path: Path, rows: list[dict]) -> None:
    tmp = path.with_name(path.name + ".tmp")
    table = pa.Table.from_pylist(rows, schema=SCHEMA)
    pq.write_table(table, tmp, compression="zstd")
    with tmp.open("rb") as handle:
        os.fsync(handle.fileno())
    os.replace(tmp, path)
    directory = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def shard_text_bytes(path: Path) -> tuple[int, int]:
    pf = pq.ParquetFile(path)
    if list(pf.schema_arrow.names) != ["doc_id", "text", "source", "license"]:
        raise Stop(f"schema mismatch for {path.name}: {pf.schema_arrow.names}")
    rows = pf.metadata.num_rows
    total = 0
    for batch in pf.iter_batches(columns=["text"], batch_size=1024):
        for value in batch.column(0).to_pylist():
            if not isinstance(value, str):
                raise Stop(f"non-string text in {path.name}")
            total += utf8_len(value)
    return rows, total


def verify_shard(path: Path, expected: dict, *, text: bool) -> None:
    if sha256_file(path) != expected["sha256"]:
        raise Stop(f"sha256 mismatch for {path.name}")
    pf = pq.ParquetFile(path)
    if list(pf.schema_arrow.names) != ["doc_id", "text", "source", "license"]:
        raise Stop(f"schema mismatch for {path.name}: {pf.schema_arrow.names}")
    if pf.metadata.num_rows != expected["rows"]:
        raise Stop(f"row mismatch for {path.name}")
    if text:
        rows, total = shard_text_bytes(path)
        if rows != expected["rows"] or total != expected["text_bytes"]:
            raise Stop(f"text byte mismatch for {path.name}")


def hub_repo_dir(repo: str, repo_type: str) -> Path:
    hf_home = Path(os.environ.get("HF_HOME", str(REPO_ROOT / "data" / "hf_cache")))
    kind = "models" if repo_type == "model" else "datasets"
    return hf_home / "hub" / (kind + "--" + repo.replace("/", "--"))


def release_download(path: Path, repo_dir: Path) -> None:
    """Remove a finished Hub file, including the shared blob if no snapshot needs it.

    The snapshot path is what ``hf_hub_download`` returns. The bytes live in
    ``HF_HOME/hub/blobs/<prefix>/<hash>``, and the repo's ``blobs/`` directory
    holds another link. Once no snapshot points at the bytes, both go away.
    """
    del repo_dir
    if not path.is_symlink() and not path.exists():
        return
    blob = path.resolve() if path.is_symlink() else None
    path.unlink()
    if blob is None or not blob.is_file():
        return
    hf_home = Path(os.environ.get("HF_HOME", str(REPO_ROOT / "data" / "hf_cache")))
    hub = (hf_home / "hub").resolve()
    resolved = blob.resolve()
    if hub not in resolved.parents:
        return
    holders = [
        other
        for other in hub.rglob("*")
        if other.is_symlink() and other.resolve() == resolved
    ]
    if any("snapshots" in holder.parts for holder in holders):
        return
    for holder in holders:
        holder.unlink()
    resolved.unlink()


def ensure_cached(repo: str, repo_type: str, revision: str, relative: str) -> Path:
    last: BaseException | None = None
    for attempt in (1, 2):
        try:
            return Path(
                hf_hub_download(
                    repo_id=repo,
                    filename=relative,
                    repo_type=repo_type,
                    revision=revision,
                )
            )
        except RETRY_ERRORS as exc:
            last = exc
            print(f"    download attempt {attempt} failed for {relative}: {exc}", flush=True)
    raise Stop(f"download failed twice for {relative}: {last}")


def iter_parquet_records(path: Path, columns: list[str], start_row: int):
    pf = pq.ParquetFile(path)
    names = list(pf.schema_arrow.names)
    missing = [column for column in columns if column not in names]
    if missing:
        raise Stop(f"{path.name} missing {missing}; columns={names}")
    seen = 0
    for group in range(pf.num_row_groups):
        group_rows = pf.metadata.row_group(group).num_rows
        if seen + group_rows <= start_row:
            seen += group_rows
            continue
        batches = pf.iter_batches(batch_size=128, columns=columns, row_groups=[group])
        for batch in batches:
            data = batch.to_pydict()
            for offset in range(batch.num_rows):
                index = seen + offset
                if index < start_row:
                    continue
                payload = {column: data[column][offset] for column in columns}
                yield Rec(index, index, index + 1, payload)
            seen += batch.num_rows


def iter_jsonl_records(path: Path, start_index: int):
    with path.open("rb") as handle:
        for index, raw in enumerate(handle):
            if index < start_index or not raw.strip():
                continue
            try:
                payload = json.loads(raw.decode("utf-8"))
            except UnicodeDecodeError:
                yield Rec(index, index, index + 1, None, invalid=True)
                continue
            if not isinstance(payload, dict):
                raise Stop(f"{path.name} line {index} is not an object")
            yield Rec(index, index, index + 1, payload)


def iter_json_gz_records(path: Path, start_index: int):
    import gzip

    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if index < start_index or not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise Stop(f"{path.name} line {index} is not an object")
            yield Rec(index, index, index + 1, payload)


def text_cell(value: object) -> str | None:
    if value is None or value == "":
        return None
    if isinstance(value, str):
        return value
    raise Stop(f"text cell is {type(value).__name__}")


def classify_row(spec: dict, row: dict, default_license: str) -> tuple[str | None, str | None, str]:
    source_id = spec["id"]
    if source_id == "code":
        kept = code_license_kept(row.get("license"))
        if kept is None:
            return None, None, "skip"
        text = text_cell(row.get("code"))
        if text is None:
            return None, None, "skip"
        return text, kept, "keep"
    if source_id == "aya_npi":
        if not aya_kept(row):
            return None, None, "skip"
        parts: list[str] = []
        for key in ("inputs", "targets"):
            value = row.get(key)
            if value is None or value == "":
                continue
            if not isinstance(value, str):
                raise Stop(f"aya_npi {key} is {type(value).__name__}")
            parts.append(value)
        if not parts:
            return None, None, "skip"
        return "\n".join(parts), default_license, "keep"
    if source_id == "belebele_npi":
        passage = row.get("flores_passage")
        question = row.get("question")
        if not isinstance(passage, str) or not isinstance(question, str):
            raise Stop("belebele row missing flores_passage or question")
        return passage + "\n" + question, default_license, "keep"
    column = spec["text_column"]
    text = text_cell(row.get(column))
    if text is None:
        return None, None, "skip"
    return text, default_license, "keep"


def read_columns(spec: dict) -> list[str]:
    if spec["id"] == "code":
        return ["code", "license"]
    if spec["id"] == "aya_npi":
        return ["inputs", "targets", "language", "language_code"]
    if spec["id"] == "belebele_npi":
        return ["flores_passage", "question"]
    column = spec["text_column"]
    if column == "plain text":
        return []
    return [column]


def required_columns(spec: dict) -> list[str]:
    if spec["id"] == "code":
        return CODE_COLUMNS
    return read_columns(spec)


def check_parquet_schema(path: Path, spec: dict) -> None:
    required = required_columns(spec)
    if not required:
        return
    names = list(pq.ParquetFile(path).schema_arrow.names)
    missing = [column for column in required if column not in names]
    if missing:
        raise Stop(f"{spec['id']} file {path.name} missing {missing}; columns={names}")


def group_cap(spec: dict) -> int | None:
    if "per_path_cap" in spec:
        return int(spec["per_path_cap"])
    amount = spec.get("amount", "all")
    if amount == "all":
        return None
    return int(amount)


def remote_groups(api: object, spec: dict) -> list[tuple[str, list]]:
    needle = spec.get("path_contains")
    groups = []
    for repo_path in spec["paths"]:
        found = list_files(api, spec["repo"], repo_path, spec.get("repo_type", "dataset"))
        if needle:
            found = [item for item in found if needle in Path(item.path).name]
        found.sort(key=lambda item: item.path)
        if not found:
            raise Stop(f"no files for {spec['repo']}:{repo_path}")
        groups.append((repo_path, found))
    return groups


def under_path(relative: str, repo_path: str) -> bool:
    prefix = repo_path.rstrip("/")
    return relative == prefix or relative.startswith(prefix + "/")


def bytes_under(progress: dict, repo_path: str | None) -> int:
    total = 0
    entries = list(progress["inputs"].items())
    opened = progress.get("open_input")
    if isinstance(opened, dict) and opened.get("path"):
        entries.append((opened["path"], opened))
    for relative, meta in entries:
        if repo_path is None or under_path(relative, repo_path):
            total += int(meta["text_bytes"])
    return total


def matching_prefix(groups: list[tuple[str, list]], relative: str) -> str | None:
    matches = [repo_path for repo_path, _files in groups if under_path(relative, repo_path)]
    if not matches:
        return None
    return max(matches, key=len)


def new_progress(revision: str, config_sha256: str) -> dict:
    return {
        "hf_revision": revision,
        "config_sha256": config_sha256,
        "started_at": utc_now(),
        "next_shard": 0,
        "inputs": {},
        "open_input": None,
        "pending_shard": None,
        "skipped_paths": [],
        "script_checks": [],
        "document_rule": None,
    }


def new_input(relative: str, digest: str) -> dict:
    return {
        "path": relative,
        "sha256": digest,
        "status": "open",
        "cursor": 0,
        "doc_ordinal": 0,
        "docs": 0,
        "text_bytes": 0,
        "dropped_invalid": 0,
        "skipped_rows": 0,
        "truncated": False,
        "shards": [],
        "document_rule": None,
    }


def progress_path(out: Path) -> Path:
    return out / "progress.json"


def manifest_path(out: Path) -> Path:
    return out / "manifest.json"


def load_progress(out: Path, revision: str, config_sha256: str) -> dict:
    path = progress_path(out)
    if not path.exists():
        progress = new_progress(revision, config_sha256)
        save_json(path, progress)
        return progress
    progress = load_json(path)
    if progress.get("hf_revision") != revision:
        raise Stop(
            f"revision changed for {out.name}: "
            f"progress has {progress.get('hf_revision')} and the Hub now has {revision}"
        )
    return progress


def recover_pending(out: Path, progress: dict) -> None:
    pending = progress.get("pending_shard")
    if not pending:
        return
    final = out / pending["name"]
    tmp = out / (pending["name"] + ".tmp")
    if tmp.exists():
        tmp.unlink()
    opened = progress.get("open_input")
    if not isinstance(opened, dict):
        raise Stop("pending shard without an open input")
    if final.exists():
        rows, total = shard_text_bytes(final)
        if rows == pending["rows"] and total == pending["text_bytes"]:
            opened["shards"].append(
                {
                    "name": pending["name"],
                    "sha256": sha256_file(final),
                    "rows": rows,
                    "text_bytes": total,
                }
            )
            opened["docs"] += rows
            opened["text_bytes"] += total
            opened["cursor"] = pending["cursor"]
            opened["doc_ordinal"] = pending["doc_ordinal"]
            progress["next_shard"] = int(progress["next_shard"]) + 1
            progress["pending_shard"] = None
            save_json(progress_path(out), progress)
            return
        final.unlink()
    progress["pending_shard"] = None
    save_json(progress_path(out), progress)


class ShardBuffer:
    def __init__(self, out: Path, progress: dict, source_id: str) -> None:
        self.out = out
        self.progress = progress
        self.source_id = source_id
        self.rows: list[dict] = []
        self.text_bytes = 0

    def add(self, doc_id: str, text: str, label: str) -> None:
        self.rows.append(
            {"doc_id": doc_id, "text": text, "source": self.source_id, "license": label}
        )
        self.text_bytes += utf8_len(text)

    def flush(self, cursor: int, doc_ordinal: int) -> None:
        opened = self.progress["open_input"]
        if not self.rows:
            opened["cursor"] = cursor
            opened["doc_ordinal"] = doc_ordinal
            save_json(progress_path(self.out), self.progress)
            return
        index = int(self.progress["next_shard"])
        name = f"shard-{index:05d}.parquet"
        pending = {
            "name": name,
            "rows": len(self.rows),
            "text_bytes": self.text_bytes,
            "cursor": cursor,
            "doc_ordinal": doc_ordinal,
        }
        self.progress["pending_shard"] = pending
        save_json(progress_path(self.out), self.progress)
        path = self.out / name
        write_table(path, self.rows)
        digest = sha256_file(path)
        opened["shards"].append(
            {
                "name": name,
                "sha256": digest,
                "rows": pending["rows"],
                "text_bytes": pending["text_bytes"],
            }
        )
        opened["docs"] += pending["rows"]
        opened["text_bytes"] += pending["text_bytes"]
        opened["cursor"] = cursor
        opened["doc_ordinal"] = doc_ordinal
        self.progress["next_shard"] = index + 1
        self.progress["pending_shard"] = None
        save_json(progress_path(self.out), self.progress)
        print(
            f"    shard {name} rows={pending['rows']} text_bytes={pending['text_bytes']}",
            flush=True,
        )
        self.rows = []
        self.text_bytes = 0


def iter_records(local: Path, relative: str, spec: dict, opened: dict):
    kind = file_kind(relative)
    cursor = int(opened["cursor"])
    if kind == "plain":
        rule = opened.get("document_rule")
        if rule not in ("blank-lines", "lines"):
            rule = "blank-lines" if detect_blank_separated(local) else "lines"
            opened["document_rule"] = rule
            print(f"    document rule for {relative}: {rule}", flush=True)
        yield from iter_plain_documents(
            local, cursor, int(opened["doc_ordinal"]), rule == "blank-lines"
        )
        return
    if kind == "parquet":
        check_parquet_schema(local, spec)
        yield from iter_parquet_records(local, read_columns(spec), cursor)
        return
    if kind == "jsonl":
        yield from iter_jsonl_records(local, cursor)
        return
    if kind == "json.gz":
        yield from iter_json_gz_records(local, cursor)
        return
    raise Stop(f"unsupported input file {relative}")


def consume_file(
    out: Path,
    progress: dict,
    spec: dict,
    relative: str,
    local: Path,
    cap: int | None,
    kept: int,
) -> tuple[int, bool]:
    opened = progress.get("open_input")
    digest = sha256_file(local)
    if not isinstance(opened, dict) or opened.get("path") != relative:
        if isinstance(opened, dict):
            raise Stop(f"open input {opened.get('path')} does not match {relative}")
        opened = new_input(relative, digest)
        progress["open_input"] = opened
        save_json(progress_path(out), progress)
    elif opened.get("sha256") not in (None, digest):
        raise Stop(f"sha256 changed for {relative} at the same revision")
    else:
        opened["sha256"] = digest
    if file_kind(relative) == "plain" and not opened.get("document_rule"):
        opened["document_rule"] = "blank-lines" if detect_blank_separated(local) else "lines"
        print(f"    document rule for {relative}: {opened['document_rule']}", flush=True)
        save_json(progress_path(out), progress)
    label = license_label(spec)
    buffer = ShardBuffer(out, progress, spec["id"])
    seen_since_save = 0
    stop_after = False
    for rec in iter_records(local, relative, spec, opened):
        if rec.invalid or rec.payload is None:
            opened["dropped_invalid"] += 1
            opened["cursor"] = rec.after
            opened["doc_ordinal"] = rec.index + 1
            seen_since_save += 1
            continue
        if isinstance(rec.payload, str):
            text, row_license, kind = rec.payload, label, "keep"
        else:
            if not isinstance(rec.payload, dict):
                raise Stop(f"bad record in {relative}")
            text, row_license, kind = classify_row(spec, rec.payload, label)
        if kind != "keep" or text is None or row_license is None:
            opened["skipped_rows"] += 1
            opened["cursor"] = rec.after
            opened["doc_ordinal"] = rec.index + 1
            seen_since_save += 1
            if seen_since_save >= 5000:
                buffer.flush(rec.after, int(opened["doc_ordinal"]))
                seen_since_save = 0
            continue
        nbytes = utf8_len(text)
        if budget_decision(kept, nbytes, cap) == "stop":
            buffer.flush(rec.resume_at, int(opened["doc_ordinal"]))
            stop_after = True
            break
        buffer.add(make_doc_id(spec["id"], relative, rec.index), text, row_license)
        kept += nbytes
        opened["cursor"] = rec.after
        opened["doc_ordinal"] = rec.index + 1
        seen_since_save += 1
        if buffer.text_bytes >= SHARD_TEXT_BYTES or (cap is not None and kept >= cap):
            buffer.flush(rec.after, int(opened["doc_ordinal"]))
            seen_since_save = 0
        if cap is not None and kept >= cap:
            stop_after = True
            break
    if not stop_after:
        buffer.flush(int(opened["cursor"]), int(opened["doc_ordinal"]))
    opened["status"] = "done"
    opened["truncated"] = stop_after
    progress["inputs"][relative] = opened
    progress["open_input"] = None
    save_json(progress_path(out), progress)
    print(
        f"  done {relative} docs={opened['docs']} text_bytes={opened['text_bytes']}",
        flush=True,
    )
    return kept, stop_after


def verify_finished_shards(out: Path, progress: dict, *, text: bool) -> int:
    names = []
    file_bytes = 0
    for meta in progress["inputs"].values():
        for shard in meta["shards"]:
            path = out / shard["name"]
            verify_shard(path, shard, text=text)
            names.append(shard["name"])
            file_bytes += path.stat().st_size
    found = sorted(path.name for path in out.glob("shard-*.parquet"))
    if found != sorted(names):
        raise Stop(f"shard set mismatch in {out.name}: disk={found} recorded={sorted(names)}")
    return file_bytes


def build_manifest(spec: dict, progress: dict, ctx: RunContext, file_bytes: int) -> dict:
    docs = 0
    text_bytes = 0
    shards = []
    dropped = 0
    skipped = 0
    rules = []
    for relative in sorted(progress["inputs"]):
        meta = progress["inputs"][relative]
        docs += int(meta["docs"])
        text_bytes += int(meta["text_bytes"])
        shards.extend(meta["shards"])
        dropped += int(meta.get("dropped_invalid", 0))
        skipped += int(meta.get("skipped_rows", 0))
        if meta.get("document_rule"):
            rules.append(meta["document_rule"])
    rule: object
    if not rules:
        rule = progress.get("document_rule")
    elif len(set(rules)) == 1:
        rule = rules[0]
    else:
        rule = rules
    manifest = {
        "source_id": spec["id"],
        "status": "complete",
        "role": spec["role"],
        "repo": spec["repo"],
        "hf_revision": progress["hf_revision"],
        "input_sha256": {
            relative: progress["inputs"][relative]["sha256"]
            for relative in sorted(progress["inputs"])
        },
        "config_sha256": ctx.config_sha256,
        "git_commit": ctx.git_commit,
        "git_dirty": ctx.git_dirty,
        "library_versions": ctx.library_versions,
        "unicode_segmentation": ctx.unicode_segmentation,
        "unicode": ctx.unicode,
        "seed": None,
        "sampling": SAMPLING,
        "started_at": progress["started_at"],
        "finished_at": utc_now(),
        "docs": docs,
        "text_bytes": text_bytes,
        "file_bytes": file_bytes,
        "shards": shards,
        "raw_deletion": raw_deletion_policy(spec["role"]),
        "document_rule": rule,
        "script_checks": progress.get("script_checks", []),
        "skipped_paths": progress.get("skipped_paths", []),
        "dropped_invalid_utf8": dropped,
        "skipped_rows": skipped,
    }
    missing = [key for key in MANIFEST_KEYS if key not in manifest]
    if missing:
        raise Stop(f"manifest missing {missing}")
    return manifest


def manifest_is_valid(out: Path) -> bool:
    path = manifest_path(out)
    if not path.exists():
        return False
    manifest = load_json(path)
    if manifest.get("status") != "complete":
        return False
    for shard in manifest.get("shards", []):
        verify_shard(out / shard["name"], shard, text=False)
    for relative, digest in manifest.get("input_sha256", {}).items():
        if manifest["role"] == "tool":
            file_path = out / relative
            if sha256_file(file_path) != digest:
                raise Stop(f"tool file hash mismatch for {relative}")
    return True


def finish_source(
    spec: dict, out: Path, progress: dict, ctx: RunContext, *, text_verify: bool
) -> None:
    file_bytes = verify_finished_shards(out, progress, text=text_verify)
    if spec["role"] != "tool" and sum(int(m["docs"]) for m in progress["inputs"].values()) == 0:
        raise Stop(f"{spec['id']} produced no documents")
    rules = [m.get("document_rule") for m in progress["inputs"].values() if m.get("document_rule")]
    if rules:
        progress["document_rule"] = rules[0] if len(set(rules)) == 1 else rules
    manifest = build_manifest(spec, progress, ctx, file_bytes)
    save_json(manifest_path(out), manifest)
    print(
        f"complete {spec['id']} docs={manifest['docs']} text_bytes={manifest['text_bytes']} "
        f"file_bytes={manifest['file_bytes']}",
        flush=True,
    )


def sample_devanagari(path: Path, column: str, limit: int = 20) -> float | None:
    shares: list[float] = []
    for rec in iter_parquet_records(path, [column], 0):
        if not isinstance(rec.payload, dict):
            continue
        text = rec.payload.get(column)
        if not isinstance(text, str) or not any(ch.isalpha() for ch in text):
            continue
        shares.append(devanagari_letter_share(text))
        if len(shares) >= limit:
            break
    if not shares:
        return None
    return sum(shares) / len(shares)


def precheck_scripts(
    spec: dict, groups: list[tuple[str, list]], revision: str
) -> tuple[list, list]:
    if spec["id"] != "deva_pool_sangraha":
        return [], []
    skipped = []
    checks = []
    repo_dir = hub_repo_dir(spec["repo"], spec.get("repo_type", "dataset"))
    for repo_path, files in groups:
        if Path(repo_path).name not in {"kas", "snd"}:
            continue
        local = ensure_cached(
            spec["repo"], spec.get("repo_type", "dataset"), revision, files[0].path
        )
        mean = sample_devanagari(local, spec["text_column"])
        decision = script_decision(mean)
        shown = None if mean is None else round(mean, 4)
        checks.append({"path": repo_path, "mean_devanagari": shown, "decision": decision})
        print(f"  script {repo_path} mean={shown} decision={decision}", flush=True)
        if decision == "stop":
            raise Stop(
                f"{repo_path} Devanagari letter share {shown} needs a human decision "
                "before those shards are kept"
            )
        if decision == "skip":
            skipped.append(repo_path)
            release_download(local, repo_dir)
    return skipped, checks


def convert_source(api: object, spec: dict, ctx: RunContext, revision: str) -> None:
    out = REPO_ROOT / "data" / "raw" / spec["id"]
    out.mkdir(parents=True, exist_ok=True)
    if manifest_is_valid(out):
        print(f"skip {spec['id']}: manifest verified", flush=True)
        return
    for tmp in out.glob("*.tmp"):
        tmp.unlink()
    progress = load_progress(out, revision, ctx.config_sha256)
    recover_pending(out, progress)
    groups = remote_groups(api, spec)
    if not progress.get("script_checks") and spec["id"] == "deva_pool_sangraha":
        skipped, checks = precheck_scripts(spec, groups, revision)
        progress["skipped_paths"] = skipped
        progress["script_checks"] = checks
        save_json(progress_path(out), progress)
    skipped = set(progress.get("skipped_paths") or [])
    repo_dir = hub_repo_dir(spec["repo"], spec.get("repo_type", "dataset"))
    per_path = "per_path_cap" in spec
    shared_kept = 0 if per_path else bytes_under(progress, None)
    opened = progress.get("open_input")
    if isinstance(opened, dict):
        relative = opened["path"]
        local = ensure_cached(spec["repo"], spec.get("repo_type", "dataset"), revision, relative)
        cap = group_cap(spec)
        prefix = matching_prefix(groups, relative)
        if per_path and prefix is None:
            raise Stop(f"{relative} is not under a configured path")
        kept_now = shared_kept if not per_path else bytes_under(progress, prefix)
        _, truncated = consume_file(out, progress, spec, relative, local, cap, kept_now)
        verify_input_shards(out, progress["inputs"][relative])
        release_download(local, repo_dir)
        if not per_path:
            shared_kept = bytes_under(progress, None)
        if truncated and not per_path:
            finish_source(spec, out, progress, ctx, text_verify=False)
            release_remaining(spec, progress, revision, repo_dir)
            return
        if truncated and prefix is not None:
            skipped.add(prefix)
            progress["skipped_paths"] = sorted(skipped)
            save_json(progress_path(out), progress)
    for repo_path, files in groups:
        if repo_path in skipped:
            print(f"  skip path {repo_path}", flush=True)
            continue
        cap = group_cap(spec)
        kept = bytes_under(progress, repo_path) if per_path else shared_kept
        if cap is not None and kept >= cap:
            continue
        truncated_path = False
        for remote in files:
            relative = remote.path
            if relative in progress["inputs"]:
                continue
            print(f"  file {relative} ({remote.size} bytes)", flush=True)
            local = ensure_cached(
                spec["repo"], spec.get("repo_type", "dataset"), revision, relative
            )
            kept, truncated = consume_file(out, progress, spec, relative, local, cap, kept)
            verify_input_shards(out, progress["inputs"][relative])
            release_download(local, repo_dir)
            if not per_path:
                shared_kept = kept
            if truncated:
                truncated_path = True
                break
        if truncated_path and not per_path:
            break
        if truncated_path:
            skipped.add(repo_path)
            progress["skipped_paths"] = sorted(skipped)
            save_json(progress_path(out), progress)
    finish_source(spec, out, progress, ctx, text_verify=False)
    release_remaining(spec, progress, revision, repo_dir)


def verify_input_shards(out: Path, meta: dict) -> None:
    for shard in meta["shards"]:
        verify_shard(out / shard["name"], shard, text=True)


def release_remaining(spec: dict, progress: dict, revision: str, repo_dir: Path) -> None:
    snapshot = repo_dir / "snapshots" / revision
    for relative in progress["inputs"]:
        candidate = snapshot / relative
        if candidate.exists() or candidate.is_symlink():
            release_download(candidate, repo_dir)


def fetch_bytes(url: str) -> bytes:
    last: BaseException | None = None
    for attempt in (1, 2):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "ancoda-tokenizer"})
            with urllib.request.urlopen(request, timeout=120) as response:
                return response.read()
        except (OSError, urllib.error.URLError) as exc:
            last = exc
            print(f"    fetch attempt {attempt} failed: {exc}", flush=True)
    raise Stop(f"fetch failed twice for {url}: {last}")


def download_papaya(spec: dict, ctx: RunContext, revision: str) -> None:
    if revision != PAPAYA_COMMIT:
        raise Stop(f"papaya revision {revision} is not the approved commit")
    out = REPO_ROOT / "data" / "raw" / spec["id"]
    out.mkdir(parents=True, exist_ok=True)
    if manifest_is_valid(out):
        print(f"skip {spec['id']}: manifest verified", flush=True)
        return
    progress = load_progress(out, revision, ctx.config_sha256)
    base = f"https://raw.githubusercontent.com/kalexrt/nepali-morphology-tokenizer/{revision}"
    for relative, expected in PAPAYA_FILES.items():
        dest = out / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        if relative not in progress["inputs"]:
            payload = fetch_bytes(f"{base}/{relative}")
            if expected is not None and len(payload) != expected:
                raise Stop(f"{relative} is {len(payload)} bytes, expected {expected}")
            tmp = dest.with_name(dest.name + ".tmp")
            tmp.write_bytes(payload)
            os.replace(tmp, dest)
            progress["inputs"][relative] = {
                "path": relative,
                "sha256": sha256_bytes(payload),
                "status": "done",
                "cursor": 0,
                "doc_ordinal": 0,
                "docs": 0,
                "text_bytes": 0,
                "dropped_invalid": 0,
                "skipped_rows": 0,
                "truncated": False,
                "shards": [],
                "document_rule": None,
            }
            save_json(progress_path(out), progress)
        elif sha256_file(dest) != progress["inputs"][relative]["sha256"]:
            raise Stop(f"papaya file changed: {relative}")
    file_bytes = sum((out / relative).stat().st_size for relative in PAPAYA_FILES)
    manifest = build_manifest(spec, progress, ctx, file_bytes)
    manifest["attribution"] = (
        "Affix inventory CC BY 4.0. Re-derived from Balaram Prasain, "
        '"A Computational Analysis of Nepali Morphology: A Model for Natural '
        'Language Processing", PhD dissertation, Tribhuvan University, 2011. '
        "Cite that work. Tokenizer vocab and stem lexicons were not downloaded."
    )
    manifest["docs"] = 0
    manifest["text_bytes"] = 0
    save_json(manifest_path(out), manifest)
    print(f"complete {spec['id']} file_bytes={file_bytes}", flush=True)


def download_glotlid(spec: dict, ctx: RunContext, revision: str, expected_size: int) -> None:
    if spec["paths"] != ["model.bin"]:
        raise Stop("glotlid source must be model.bin only")
    out = REPO_ROOT / "data" / "raw" / spec["id"]
    out.mkdir(parents=True, exist_ok=True)
    if manifest_is_valid(out):
        print(f"skip {spec['id']}: manifest verified", flush=True)
        return
    progress = load_progress(out, revision, ctx.config_sha256)
    relative = "model.bin"
    local = ensure_cached(spec["repo"], "model", revision, relative)
    size = local.stat().st_size
    if size != expected_size:
        raise Stop(f"model.bin is {size} bytes; the Hub listing says {expected_size}")
    dest = out / relative
    if relative not in progress["inputs"]:
        tmp = dest.with_name(dest.name + ".tmp")
        shutil.copyfile(local, tmp)
        os.replace(tmp, dest)
        progress["inputs"][relative] = {
            "path": relative,
            "sha256": sha256_file(dest),
            "status": "done",
            "cursor": 0,
            "doc_ordinal": 0,
            "docs": 0,
            "text_bytes": 0,
            "dropped_invalid": 0,
            "skipped_rows": 0,
            "truncated": False,
            "shards": [],
            "document_rule": None,
        }
        save_json(progress_path(out), progress)
    if sha256_file(dest) != sha256_file(local):
        raise Stop("glotlid copy does not match the Hub file")
    file_bytes = dest.stat().st_size
    manifest = build_manifest(spec, progress, ctx, file_bytes)
    manifest["docs"] = 0
    manifest["text_bytes"] = 0
    manifest["note"] = "model.bin is the stable v3 file. model_v4_alpha.bin was not downloaded."
    save_json(manifest_path(out), manifest)
    release_download(local, hub_repo_dir(spec["repo"], "model"))
    print(f"complete {spec['id']} file_bytes={file_bytes}", flush=True)


def ensure_room(incoming: int, reason: str) -> None:
    free, df_text = free_bytes_df(REPO_ROOT)
    print(df_text, flush=True)
    print(f"room check {reason}: incoming={incoming} free={free}", flush=True)
    if not room_ok(free, incoming):
        raise Stop(
            f"{reason}: free {free} minus incoming {incoming} leaves under {FREE_FLOOR} bytes"
        )


def unicode_record() -> dict[str, object]:
    root = REPO_ROOT / "third_party" / "unicode" / "16.0.0"
    files = {}
    for relative, expected in UNICODE_SHA256.items():
        path = root / relative
        if not path.is_file():
            raise Stop(f"missing Unicode data file {path}")
        digest = sha256_file(path)
        if digest != expected:
            raise Stop(f"Unicode hash mismatch for {relative}")
        files[relative] = digest
    return {"version": "16.0.0", "files": files}


def segmentation_pin() -> str:
    text = (REPO_ROOT / "crates" / "ancoda-seg" / "Cargo.toml").read_text()
    for line in text.splitlines():
        if line.strip().startswith("unicode-segmentation"):
            return line.split("=", 1)[1].strip().strip('"')
    raise Stop("unicode-segmentation pin not found")


def library_versions() -> dict[str, str]:
    found = {}
    for name in PACKAGES:
        found[name] = version(name)
    return found


def make_context(sources_path: Path) -> RunContext:
    raw = sources_path.read_bytes()
    dirty = bool(
        __import__("subprocess")
        .check_output(["git", "status", "--porcelain"], cwd=REPO_ROOT)
        .strip()
    )
    return RunContext(
        git_commit=git_commit(),
        git_dirty=dirty,
        config_sha256=sha256_bytes(raw),
        library_versions=library_versions(),
        unicode_segmentation=segmentation_pin(),
        unicode=unicode_record(),
    )


def source_rank(spec: dict, stored: int) -> tuple:
    if spec["role"] != "train":
        rank = 0
    elif spec["id"] == "deva_pool_sangraha":
        rank = 1
    else:
        rank = 2
    return (rank, stored, spec["id"])


def run_downloads(sources_path: Path, only: list[str] | None) -> int:
    spec = load_spec(sources_path)
    selected = [source for source in spec["sources"] if source.get("download", True)]
    if only:
        wanted = set(only)
        selected = [source for source in selected if source["id"] in wanted]
        missing = wanted - {source["id"] for source in selected}
        if missing:
            raise Stop(f"unknown source id {sorted(missing)}")
    ctx = make_context(sources_path)
    from huggingface_hub import HfApi

    api = HfApi()
    measured = {source["id"]: measure_source(api, source) for source in selected}
    revisions = {}
    ordered = sorted(
        selected, key=lambda source: source_rank(source, measured[source["id"]].stored_bytes)
    )
    print("download order: " + ", ".join(source["id"] for source in ordered), flush=True)
    for source in ordered:
        revision = revisions.get(source["repo"])
        if revision is None:
            revision = repo_revision(
                api,
                source["repo"],
                source.get("repo_type", "dataset"),
                source.get("revision"),
            )
            revisions[source["repo"]] = revision
        out = REPO_ROOT / "data" / "raw" / source["id"]
        if manifest_is_valid(out):
            print(f"skip {source['id']}: manifest verified", flush=True)
            continue
        incoming = measured[source["id"]].stored_bytes + measured[source["id"]].transient_bytes
        ensure_room(incoming, source["id"])
        print(f"source {source['id']} revision {revision}", flush=True)
        if source.get("repo_type") == "github":
            download_papaya(source, ctx, revision)
        elif source.get("repo_type") == "model":
            download_glotlid(source, ctx, revision, measured[source["id"]].upstream_bytes)
        else:
            convert_source(api, source, ctx, revision)
    free, df_text = free_bytes_df(REPO_ROOT)
    print(df_text, flush=True)
    print(f"free_bytes={free}", flush=True)
    if free < FREE_FLOOR:
        raise Stop(f"free {free} is under {FREE_FLOOR} after downloads")
    return 0


def write_report(sources_path: Path) -> int:
    spec = load_spec(sources_path)
    free, df_text = free_bytes_df(REPO_ROOT)
    print("source_id\trole\tdocs\ttext_bytes\tfile_bytes\trevision")
    missing = []
    for source in spec["sources"]:
        if not source.get("download", True):
            continue
        path = REPO_ROOT / "data" / "raw" / source["id"] / "manifest.json"
        if not path.exists():
            missing.append(source["id"])
            print(f"{source['id']}\t{source.get('role', '')}\tMISSING\t\t\t")
            continue
        manifest = load_json(path)
        print(
            f"{manifest['source_id']}\t{manifest['role']}\t{manifest['docs']}\t"
            f"{manifest['text_bytes']}\t{manifest['file_bytes']}\t{manifest['hf_revision']}"
        )
    print(df_text)
    print(f"free_bytes={free}")
    if missing:
        print("missing: " + ", ".join(missing))
        return 2
    if free < FREE_FLOOR:
        print("STOP (human): under 50 GB free")
        return 2
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Download M2 sources into data/raw.")
    parser.add_argument("--report", action="store_true")
    parser.add_argument("--only", action="append", default=None)
    parser.add_argument("--sources", type=Path, default=REPO_ROOT / "configs" / "sources.yaml")
    args = parser.parse_args(argv)
    os.environ.setdefault("HF_HOME", str(REPO_ROOT / "data" / "hf_cache"))
    try:
        if args.report:
            return write_report(args.sources)
        return run_downloads(args.sources, args.only)
    except Stop as exc:
        print(f"STOP (human): {exc}", flush=True)
        return 2


if __name__ == "__main__":
    sys.exit(main())

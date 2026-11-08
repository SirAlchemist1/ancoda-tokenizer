"""Shared helpers for the M4 cleaning pipeline."""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Iterable, Iterator
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from ancoda_tok.storage import (
    REPO_ROOT,  # re-exported
    free_bytes_df,
    git_commit,
)

FREE_FLOOR = 50_000_000_000
SHARD_TEXT_BYTES = 64 * 1024 * 1024
SEED = 20261104
SCHEMA = pa.schema(
    [
        ("doc_id", pa.string()),
        ("text", pa.string()),
        ("source", pa.string()),
        ("license", pa.string()),
    ]
)
CLEAN_SCHEMA = pa.schema(
    [
        ("doc_id", pa.string()),
        ("text", pa.string()),
        ("source", pa.string()),
        ("license", pa.string()),
        ("bucket", pa.string()),
        ("glotlid_label", pa.string()),
        ("glotlid_prob", pa.float64()),
    ]
)

# C0 and C1 controls except TAB (0x09) and LF (0x0A). CR is stripped.
_CONTROL_RE = re.compile(
    "["
    + "".join(chr(c) for c in range(0x09))
    + chr(0x0B)
    + chr(0x0C)
    + "".join(chr(c) for c in range(0x0E, 0x20))
    + "".join(chr(c) for c in range(0x7F, 0xA0))
    + "]"
)

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
    "fasttext-wheel",
    "datatrove",
    "pyahocorasick",
)


class Stop(Exception):
    """Human stop. CLI exits 2."""


def now_utc() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_mix(path: Path | None = None) -> dict:
    path = path or (REPO_ROOT / "configs" / "mix.yaml")
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def load_sources(path: Path | None = None) -> dict:
    path = path or (REPO_ROOT / "configs" / "sources.yaml")
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def config_sha256() -> dict[str, str]:
    return {
        "mix.yaml": sha256_file(REPO_ROOT / "configs" / "mix.yaml"),
        "sources.yaml": sha256_file(REPO_ROOT / "configs" / "sources.yaml"),
    }


def library_versions() -> dict[str, str]:
    out: dict[str, str] = {}
    for name in PACKAGES:
        try:
            out[name] = version(name)
        except PackageNotFoundError:
            out[name] = "missing"
    try:
        import ancoda_seg

        out["ancoda-seg"] = getattr(ancoda_seg, "__version__", "local")
        out["unicode_segmentation"] = ".".join(str(x) for x in ancoda_seg.UNICODE_VERSION)
    except (ImportError, AttributeError):
        out["ancoda-seg"] = "missing"
    return out


def unicode_manifest() -> dict[str, object]:
    path = REPO_ROOT / "docs" / "UNICODE_16.0.0_SHA256.md"
    files: dict[str, str] = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if "|" not in line or line.startswith("|---") or "File" in line:
                continue
            parts = [p.strip() for p in line.strip("|").split("|")]
            if len(parts) >= 2 and parts[0] and parts[1]:
                files[parts[0]] = parts[1]
    return {"version": "16.0.0", "files": files}


def git_dirty() -> bool:
    import subprocess

    try:
        out = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=REPO_ROOT, text=True
        )
        return bool(out.strip())
    except (subprocess.CalledProcessError, FileNotFoundError):
        return True


def base_manifest(**extra: Any) -> dict[str, Any]:
    manifest: dict[str, Any] = {
        "git_commit": git_commit(),
        "git_dirty": git_dirty(),
        "config_sha256": config_sha256(),
        "library_versions": library_versions(),
        "unicode": unicode_manifest(),
        "seed": SEED,
        "started_at": now_utc(),
    }
    try:
        import ancoda_seg

        manifest["unicode_segmentation"] = ".".join(str(x) for x in ancoda_seg.UNICODE_VERSION)
    except (ImportError, AttributeError):
        manifest["unicode_segmentation"] = "missing"
    manifest.update(extra)
    return manifest


def finish_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    manifest = dict(manifest)
    manifest["finished_at"] = now_utc()
    return manifest


def free_bytes() -> int:
    return free_bytes_df(REPO_ROOT)[0]


def ensure_room(incoming: int, *, label: str) -> None:
    free = free_bytes()
    if free - incoming < FREE_FLOOR:
        raise Stop(
            f"STOP: writing {label} (~{incoming} bytes) would leave "
            f"{free - incoming} bytes free; floor is {FREE_FLOOR}"
        )


def strip_controls(text: str) -> str:
    """Strip C0/C1 controls except TAB and LF. No Unicode normalization."""
    if "\r" in text:
        text = text.replace("\r\n", "\n").replace("\r", "\n")
    return _CONTROL_RE.sub("", text)


def utf8_len(text: str) -> int:
    return len(text.encode("utf-8"))


def list_shards(directory: Path) -> list[Path]:
    return sorted(directory.glob("shard-*.parquet"))


def iter_parquet_rows(directory: Path) -> Iterator[dict[str, Any]]:
    for path in list_shards(directory):
        table = pq.read_table(path)
        cols = {name: table.column(name) for name in table.column_names}
        for i in range(table.num_rows):
            yield {name: cols[name][i].as_py() for name in cols}


def write_rows(path: Path, rows: list[dict[str, Any]], schema: pa.Schema) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    table = pa.Table.from_pylist(rows, schema=schema)
    pq.write_table(table, tmp, compression="zstd")
    tmp.replace(path)
    text_bytes = sum(utf8_len(r["text"]) for r in rows)
    return {
        "path": str(path.relative_to(REPO_ROOT)) if path.is_relative_to(REPO_ROOT) else str(path),
        "docs": len(rows),
        "text_bytes": text_bytes,
        "file_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


class ShardWriter:
    """Buffer rows and flush zstd parquet shards under ``out``."""

    def __init__(self, out: Path, schema: pa.Schema = CLEAN_SCHEMA, start_index: int = 0) -> None:
        self.out = out
        self.schema = schema
        self.out.mkdir(parents=True, exist_ok=True)
        self.rows: list[dict[str, Any]] = []
        self.buf_bytes = 0
        self.index = start_index
        self.shards: list[dict[str, Any]] = []
        self.docs = 0
        self.text_bytes = 0

    def add(self, row: dict[str, Any]) -> None:
        nbytes = utf8_len(row["text"])
        self.rows.append(row)
        self.buf_bytes += nbytes
        self.docs += 1
        self.text_bytes += nbytes
        if self.buf_bytes >= SHARD_TEXT_BYTES:
            self.flush()

    def flush(self) -> None:
        if not self.rows:
            return
        path = self.out / f"shard-{self.index:05d}.parquet"
        meta = write_rows(path, self.rows, self.schema)
        self.shards.append(meta)
        self.index += 1
        self.rows = []
        self.buf_bytes = 0

    def close(self) -> list[dict[str, Any]]:
        self.flush()
        return self.shards


def source_bucket_map(mix: dict) -> dict[str, str]:
    out: dict[str, str] = {}
    for bucket, spec in mix["buckets"].items():
        for source_id in spec["sources"]:
            out[source_id] = bucket
    return out


def label_to_bucket(mix: dict) -> dict[str, str]:
    """GlotLID label (without __label__) → mix bucket name."""
    out: dict[str, str] = {}
    for bucket, spec in mix["buckets"].items():
        for label in spec.get("keep_labels") or []:
            out[label] = bucket
    return out


def train_source_ids(mix: dict) -> list[str]:
    ids: list[str] = []
    for spec in mix["buckets"].values():
        ids.extend(spec["sources"])
    return ids


def progress_path(stage_dir: Path) -> Path:
    return stage_dir / "progress.json"


def load_progress(stage_dir: Path) -> dict[str, Any]:
    path = progress_path(stage_dir)
    if path.exists():
        return load_json(path)
    return {"completed_shards": [], "stats": {}}


def save_progress(stage_dir: Path, progress: dict[str, Any]) -> None:
    save_json(progress_path(stage_dir), progress)


def workers() -> int:
    return int(os.environ.get("ANCODA_WORKERS", "5"))


def raw_dir(source_id: str) -> Path:
    return REPO_ROOT / "data" / "raw" / source_id


def clean_dir(source_id: str) -> Path:
    return REPO_ROOT / "data" / "clean" / source_id


def artifacts_clean() -> Path:
    path = REPO_ROOT / "artifacts" / "clean"
    path.mkdir(parents=True, exist_ok=True)
    return path


def is_latin_line(line: str) -> bool:
    """True for Latin prose lines that may be Preeti/Kantipur-encoded.

    Rejects Devanagari, non-Latin-majority scripts, URL/JS/HTML chrome, and
    lines without enough Latin letters. Those are still Latin text, but they
    are not legacy-font candidates.
    """
    lower = line.lower().strip()
    if not lower:
        return False
    url_markers = (
        "http://",
        "https://",
        "www.",
        ".html",
        ".php",
        ".pdf",
        "youtube.com",
        "facebook.com",
        "youtu.be",
        "docs.google",
        "iframe",
        "adsbygoogle",
        "getelementbyid",
        "getelementsbytagname",
        "<script",
        "</",
        "mailto:",
    )
    if any(m in lower for m in url_markers):
        return False
    if lower.startswith(("source:", "source :")):
        # Citation lines that are mostly a URL were filtered above; bare
        # "Source: Setopati" style lines may still be prose.
        pass
    latin = 0
    other_letters = 0
    for ch in line:
        o = ord(ch)
        if 0x0900 <= o <= 0x097F:
            return False
        if ch.isalpha():
            if ("A" <= ch <= "Z") or ("a" <= ch <= "z"):
                latin += 1
            else:
                other_letters += 1
    if latin < 8:
        return False
    if latin < 2 * max(1, other_letters):
        return False
    # Require at least one space so single tokens / part numbers are skipped.
    return " " in line


def script_outside_share(text: str, expected: Iterable[str]) -> float:
    """Share of characters (excluding whitespace) outside expected scripts.

    Expected values are Unicode script names: Devanagari, Latin. Characters
    with Common/Inherited script (punctuation, digits, marks) are allowed.
    Empty expected means no check (return 0).
    """
    expected_set = set(expected)
    if not expected_set:
        return 0.0
    total = 0
    outside = 0
    for ch in text:
        if ch.isspace():
            continue
        total += 1
        name = _script_name(ch)
        if name in expected_set or name in {"Common", "Inherited", "Unknown"}:
            continue
        outside += 1
    if total == 0:
        return 0.0
    return outside / total


_SCRIPT_CACHE: dict[int, str] | None = None


def _load_scripts() -> dict[int, str]:
    global _SCRIPT_CACHE
    if _SCRIPT_CACHE is not None:
        return _SCRIPT_CACHE
    path = REPO_ROOT / "third_party" / "unicode" / "16.0.0" / "Scripts.txt"
    mapping: dict[int, str] = {}
    if not path.exists():
        _SCRIPT_CACHE = mapping
        return mapping
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        body = line.split("#", 1)[0].strip()
        if ";" not in body:
            continue
        rng, script = body.split(";", 1)
        script = script.strip()
        rng = rng.strip()
        if ".." in rng:
            lo_s, hi_s = rng.split("..")
            lo, hi = int(lo_s, 16), int(hi_s, 16)
            for cp in range(lo, hi + 1):
                mapping[cp] = script
        else:
            mapping[int(rng, 16)] = script
    _SCRIPT_CACHE = mapping
    return mapping


def _script_name(ch: str) -> str:
    return _load_scripts().get(ord(ch), "Unknown")

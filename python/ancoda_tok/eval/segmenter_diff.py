"""pretokenizer corpus checks: cross-engine pre-tokens and the segmenter differential.

The cleaning training snapshot does not exist yet. The 1,000,000-word sample is the
first distinct whitespace-separated words from training-role parquet, after the
shard list is shuffled with ``random.Random(20261104)``. Eval words are every
distinct word in role=eval sources plus the harness FLORES+ dev and devtest
files, read in place and not copied.
"""

from __future__ import annotations

import hashlib
import json
import random
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import pyarrow.parquet as pq
import yaml

from ancoda_tok.pattern import repo_root

SEED = 20261104
TRAIN_WORDS = 1_000_000
PRETOK_LINES = 100_000


def _sources(role: str) -> list[str]:
    spec = yaml.safe_load((repo_root() / "configs" / "sources.yaml").read_text())
    return sorted(item["id"] for item in spec["sources"] if item["role"] == role)


def _parquet_files(source_ids: list[str]) -> list[Path]:
    root = repo_root() / "data" / "raw"
    files: list[Path] = []
    for source_id in source_ids:
        files.extend(sorted((root / source_id).glob("*.parquet")))
    return files


def _texts(path: Path, batch_size: int = 256):
    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(columns=["text"], batch_size=batch_size):
        for value in batch.column(0).to_pylist():
            if value:
                yield value


def sample_training_words(opened: list[Path]) -> list[str]:
    files = _parquet_files(_sources("train"))
    rng = random.Random(SEED)
    rng.shuffle(files)
    words: list[str] = []
    seen: set[str] = set()
    for path in files:
        opened.append(path)
        for text in _texts(path):
            for word in text.split():
                if word in seen:
                    continue
                seen.add(word)
                words.append(word)
                if len(words) >= TRAIN_WORDS:
                    return words
    raise RuntimeError(f"training raw produced only {len(words)} distinct words")


def iter_eval_words(opened: list[Path]):
    for path in _parquet_files(_sources("eval")):
        opened.append(path)
        for text in _texts(path):
            yield from text.split()
    flores = repo_root() / "third_party" / "yak-tokenizer" / "data" / "eval" / "flores_plus"
    for path in sorted(flores.rglob("*.txt")):
        opened.append(path)
        yield from path.read_text(encoding="utf-8").split()


def sample_pretok_lines(opened: list[Path]) -> list[str]:
    """Seeded reservoir sample, 33,334 / 33,333 / 33,333 lines.

    One ``random.Random(20261104)`` shuffles each language group's files, then
    Algorithm R draws the quota from every newline-separated line in those files.
    """
    groups = (
        ["fw2_npi", "iriis", "sangraha_nep", "indiccorp_npi", "hplt_npi"],
        ["fw2_hin", "sangraha_hin"],
        ["fwedu_en"],
    )
    quotas = (33_334, 33_333, 33_333)
    rng = random.Random(SEED)
    chosen: list[str] = []
    for source_ids, quota in zip(groups, quotas, strict=True):
        files = _parquet_files(source_ids)
        rng.shuffle(files)
        reservoir: list[str] = []
        seen = 0
        for path in files:
            opened.append(path)
            for text in _texts(path, batch_size=4096):
                for line in text.split("\n"):
                    seen += 1
                    if len(reservoir) < quota:
                        reservoir.append(line)
                    else:
                        pick = rng.randrange(seen)
                        if pick < quota:
                            reservoir[pick] = line
        if seen < quota:
            raise RuntimeError(f"{source_ids} produced {seen} lines, need {quota}")
        chosen.extend(reservoir)
    if len(chosen) != PRETOK_LINES:
        raise RuntimeError(f"pre-token sample is {len(chosen)}, not {PRETOK_LINES}")
    return chosen


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fancy_version() -> str:
    for line in (repo_root() / "crates" / "ancoda-seg" / "Cargo.toml").read_text().splitlines():
        stripped = line.strip()
        if stripped.startswith("fancy-regex"):
            value = stripped.split("=", 1)[1].strip().strip('"')
            return value.removeprefix("=")
    raise RuntimeError("fancy-regex pin not found")


def write_result(name: str, payload: dict, opened: list[Path], started: datetime) -> Path:
    root = repo_root()
    out_dir = root / "artifacts" / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    artifact = out_dir / f"{name}.json"
    finished = datetime.now(UTC)
    payload = dict(payload)
    payload["seed"] = SEED
    artifact.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    unicode_root = root / "third_party" / "unicode" / "16.0.0"
    unicode_files = [
        "DerivedCoreProperties.txt",
        "Scripts.txt",
        "UnicodeData.txt",
        "auxiliary/GraphemeBreakTest.txt",
        "auxiliary/GraphemeBreakProperty.txt",
        "emoji/emoji-data.txt",
    ]
    inputs = {str(path.relative_to(root)): sha256_file(path) for path in sorted(set(opened))}
    pattern = root / "configs" / "pretokenizer_pattern.txt"
    inputs[str(pattern.relative_to(root))] = sha256_file(pattern)
    manifest = {
        "artifact": str(artifact.relative_to(root)),
        "config_sha256": sha256_file(pattern),
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip(),
        "git_dirty": bool(
            subprocess.check_output(["git", "status", "--porcelain"], cwd=root).strip()
        ),
        "input_file_sha256s": inputs,
        "library_versions": _library_versions(),
        "unicode_data_sha256s": {
            relative: sha256_file(unicode_root / relative) for relative in unicode_files
        },
        "unicode_data_version": "16.0.0",
        "unicode_segmentation_version": "1.12.0",
        "fancy_regex_version": _fancy_version(),
        "seed": SEED,
        "start_time": started.isoformat(),
        "end_time": finished.isoformat(),
    }
    manifest_path = out_dir / f"{name}.manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return artifact


def _library_versions() -> dict[str, str]:
    from importlib.metadata import version

    found = {"python": __import__("platform").python_version()}
    for package in ("regex", "tiktoken", "tokenizers", "pyarrow"):
        found[package] = version(package)
    return found

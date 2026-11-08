"""Independent decontam audit over the four M4 snapshots.

Every audit count written to snapshots.json is taken from an on-disk audit
result file (never typed). Each run also writes
``artifacts/clean/audit_<name>.result.json``.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import re
from pathlib import Path
from typing import Any

from ancoda_tok.clean.common import (
    REPO_ROOT,
    Stop,
    artifacts_clean,
    base_manifest,
    finish_manifest,
    free_bytes,
    load_json,
    save_json,
    sha256_file,
)
from ancoda_tok.clean.decontam import audit_snapshot

_AUDIT_LINE_RE = re.compile(
    r"^audit (?P<name>n45|n50|n60|n50_nojoiner) (?P<counts>\{.*\})\s*$"
)


def find_snapshots(wanted: list[str]) -> dict[str, Path]:
    snap_root = REPO_ROOT / "data" / "snapshots"
    found: dict[str, Path] = {}
    for d in sorted(snap_root.iterdir()):
        if not d.is_dir():
            continue
        for name in wanted:
            if name == "n50" and d.name.startswith("n50_nojoiner"):
                continue
            if d.name.startswith(name + "_") and name not in found:
                found[name] = d
                break
    missing = [n for n in wanted if n not in found]
    if missing:
        raise SystemExit(f"missing snapshots: {missing}")
    return found


def result_path(name: str) -> Path:
    return artifacts_clean() / f"audit_{name}.result.json"


def write_result(
    name: str,
    *,
    snapshot_dir: Path,
    counts: dict[str, int],
    source_log: Path | None = None,
) -> Path:
    """Write a machine-readable audit result; return its path."""
    payload: dict[str, Any] = {
        **base_manifest(stage="audit"),
        "snapshot": name,
        "snapshot_path": str(snapshot_dir.relative_to(REPO_ROOT)),
        "counts": {k: int(v) for k, v in sorted(counts.items())},
    }
    if source_log is not None:
        payload["source_log"] = str(source_log.relative_to(REPO_ROOT))
        payload["source_log_sha256"] = sha256_file(source_log)
    path = result_path(name)
    save_json(path, finish_manifest(payload))
    return path


def load_counts_from_result(path: Path) -> tuple[dict[str, int], dict[str, str]]:
    data = load_json(path)
    counts = {k: int(v) for k, v in data["counts"].items()}
    provenance = {
        "path": str(path.relative_to(REPO_ROOT)),
        "sha256": sha256_file(path),
    }
    if "source_log" in data:
        provenance["source_log"] = data["source_log"]
        provenance["source_log_sha256"] = data["source_log_sha256"]
    return counts, provenance


def parse_audit_counts_from_log(
    log_path: Path, name: str, *, snapshot_dirname: str
) -> dict[str, int]:
    """Read the final ``audit <name> {...}`` line for this snapshot from a log.

    Requires the log to mention ``snapshot_dirname`` before that line so an
    old audit of a different snapshot build cannot be reused.
    """
    text = log_path.read_text(encoding="utf-8")
    if snapshot_dirname not in text:
        raise ValueError(
            f"{log_path} does not mention snapshot dir {snapshot_dirname}"
        )
    # Restrict to the section for this snapshot dir when possible.
    parts = text.split(f"auditing {name} at {snapshot_dirname}")
    if len(parts) < 2:
        # Some logs print the dir only on later audit lines.
        section = text
    else:
        section = parts[1]
        # Stop at the next "auditing " header if present.
        nxt = section.find("\nauditing ")
        if nxt != -1:
            section = section[:nxt]

    counts: dict[str, int] | None = None
    for line in section.splitlines():
        m = _AUDIT_LINE_RE.match(line.strip())
        if m and m.group("name") == name:
            parsed = ast.literal_eval(m.group("counts"))
            if not isinstance(parsed, dict):
                raise ValueError(f"bad audit counts in {log_path}: {line}")
            counts = {str(k): int(v) for k, v in parsed.items()}
    if counts is None:
        raise ValueError(f"no audit line for {name} in {log_path}")
    return counts


def rebuild_snapshots_json_from_results(
    wanted: list[str],
    *,
    found: dict[str, Path],
) -> dict[str, Any]:
    results: dict[str, Any] = {
        **base_manifest(stage="snapshots"),
        "snapshots": {},
        "audits": {},
        "audit_provenance": {},
    }
    for name in wanted:
        path = found[name]
        man = load_json(path / "manifest.json")
        results["snapshots"][name] = {
            "path": str(path.relative_to(REPO_ROOT)),
            "manifest": str((path / "manifest.json").relative_to(REPO_ROOT)),
            "text_bytes": man["text_bytes"],
            "per_bucket": man["per_bucket"],
        }
        rpath = result_path(name)
        if not rpath.exists():
            raise Stop(f"STOP: missing audit result file {rpath}")
        counts, provenance = load_counts_from_result(rpath)
        # Refuse results that point at a different snapshot build.
        data = load_json(rpath)
        if data.get("snapshot_path") != results["snapshots"][name]["path"]:
            raise Stop(
                f"STOP: {rpath} is for {data.get('snapshot_path')}, "
                f"not {results['snapshots'][name]['path']}"
            )
        results["audits"][name] = counts
        results["audit_provenance"][name] = provenance
        if any(v != 0 for v in counts.values()):
            raise Stop(f"STOP: snapshot {name} failed decontam audit: {counts}")
    return finish_manifest(results)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--only",
        nargs="*",
        default=["n45", "n50", "n60", "n50_nojoiner"],
        help="Snapshot names to audit",
    )
    parser.add_argument(
        "--from-logs",
        action="store_true",
        help=(
            "Do not re-run audits: parse existing logs into result JSON files "
            "for the current snapshot dirs, then rebuild snapshots.json"
        ),
    )
    parser.add_argument(
        "--merge",
        action="store_true",
        help="Ignored for compatibility; snapshots.json is always rebuilt from results",
    )
    args = parser.parse_args()
    wanted = list(args.only)
    found = find_snapshots(wanted)

    if args.from_logs:
        # Known log files produced by prior audit_snapshots runs for the
        # current snapshot digests. Shared log is OK if it names the digest.
        log_candidates: dict[str, list[Path]] = {
            "n45": [artifacts_clean() / "audit_run2.log"],
            "n50": [artifacts_clean() / "audit_run2.log"],
            "n60": [artifacts_clean() / "audit_n60.log"],
            "n50_nojoiner": [artifacts_clean() / "audit_n50_nojoiner.log"],
        }
        for name in wanted:
            snap = found[name]
            rpath = result_path(name)
            chosen: Path | None = None
            counts: dict[str, int] | None = None
            for log in log_candidates.get(name, []):
                if not log.exists():
                    continue
                try:
                    counts = parse_audit_counts_from_log(
                        log, name, snapshot_dirname=snap.name
                    )
                    chosen = log
                    break
                except ValueError as exc:
                    print(f"skip {log.name} for {name}: {exc}", flush=True)
            if chosen is None or counts is None:
                raise Stop(
                    f"STOP: no audit log for current {name} ({snap.name}); "
                    f"re-run: uv run python -m ancoda_tok.clean.audit_snapshots --only {name}"
                )
            write_result(name, snapshot_dir=snap, counts=counts, source_log=chosen)
            print(
                f"wrote {rpath.relative_to(REPO_ROOT)} from {chosen.relative_to(REPO_ROOT)} "
                f"counts={counts}",
                flush=True,
            )
    else:
        for name in wanted:
            path = found[name]
            print(f"auditing {name} at {path.name}", flush=True)
            counts = audit_snapshot(path)
            print(f"audit {name} {counts}", flush=True)
            rpath = write_result(name, snapshot_dir=path, counts=counts)
            print(f"wrote {rpath.relative_to(REPO_ROOT)}", flush=True)
            if any(v != 0 for v in counts.values()):
                raise Stop(f"STOP: snapshot {name} failed decontam audit: {counts}")

    results = rebuild_snapshots_json_from_results(wanted, found=found)
    # Preserve audits/provenance for snapshots not in --only when merging.
    out_path = artifacts_clean() / "snapshots.json"
    if args.merge and out_path.exists():
        prev = load_json(out_path)
        for name, info in prev.get("snapshots", {}).items():
            if name not in results["snapshots"]:
                results["snapshots"][name] = info
        for name, counts in prev.get("audits", {}).items():
            if name not in results["audits"]:
                # Only keep if a result file still backs them.
                rpath = result_path(name)
                if rpath.exists():
                    c, p = load_counts_from_result(rpath)
                    results["audits"][name] = c
                    results.setdefault("audit_provenance", {})[name] = p
                else:
                    print(
                        f"dropping prior hand/unbacked audit for {name}",
                        flush=True,
                    )
        results = finish_manifest(results)

    save_json(out_path, results)
    print(f"wrote {out_path.relative_to(REPO_ROOT)}", flush=True)
    for name in sorted(results["audits"]):
        prov = results["audit_provenance"][name]
        print(
            f"  {name} {results['audits'][name]} "
            f"from {prov['path']} sha256={prov['sha256'][:16]}…",
            flush=True,
        )
    print(f"all audits clean free_bytes={free_bytes()}", flush=True)
    if free_bytes() < 50_000_000_000:
        raise Stop(f"STOP: free_bytes={free_bytes()} under 50 GB after snapshots")
    print("done", flush=True)


if __name__ == "__main__":
    main()

"""M2 storage plan.

Sizes of files we would download come from ``HfApi.list_repo_tree``. A number
in PLAN.md written as GB is 10^9 bytes. The plan stops when the projected peak
would leave under 50 GB free.
"""

from __future__ import annotations

import argparse
import math
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from huggingface_hub.errors import RemoteEntryNotFoundError

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCES = REPO_ROOT / "configs" / "sources.yaml"
DEFAULT_DOC = REPO_ROOT / "docs" / "STORAGE.md"

GB = 1_000_000_000


@dataclass(frozen=True)
class RemoteFile:
    path: str
    size: int


@dataclass
class MeasuredSource:
    source_id: str
    bucket: str
    role: str
    repo: str
    download: bool
    upstream_bytes: int
    stored_bytes: int
    transient_bytes: int
    n_files: int
    note: str = ""
    card_license: str = ""
    paths: list[str] = field(default_factory=list)


def stored_bytes(file_sizes: list[int], amount: str | int) -> int:
    """Bytes we keep. ``all`` is the file sum. A cap is a UTF-8 budget.

    Zstd parquet of text is at most the text budget, so the cap is an upper
    bound on the stored file. File sums are exact Hub sizes.
    """
    total = sum(file_sizes)
    if amount == "all":
        return total
    return min(total, int(amount))


def transient_bytes(
    file_sizes: list[int],
    amount: str | int,
    *,
    keep_ratio: float | None = None,
) -> int:
    """Upper bound on the Hub cache while this source is downloading.

    Unfiltered text: compressed bytes are at most the text bytes, and the last
    shard is fetched whole, so the cache is at most ``cap + largest shard``.
    A keep ratio (kept UTF-8 bytes / shard bytes) scales the cap. ``all`` keeps
    the whole tree in cache until the source is verified.
    """
    if not file_sizes:
        return 0
    total = sum(file_sizes)
    if amount == "all":
        return total
    cap = int(amount)
    largest = max(file_sizes)
    if keep_ratio is not None:
        if keep_ratio <= 0:
            return total
        need = math.ceil(cap / keep_ratio)
        return min(total, need + largest)
    return min(total, cap + largest)


def per_path_stored(groups: list[list[int]], cap: int) -> int:
    return sum(stored_bytes(group, cap) for group in groups)


def per_path_transient(groups: list[list[int]], cap: int) -> int:
    """Cache holds one language at a time. The bound is the largest language."""
    if not groups:
        return 0
    return max(transient_bytes(group, cap) for group in groups)


def resident_peak(train_stored: int, other_stored: int, snapshots: int, counts: int) -> int:
    """Rejected peak: raw and cleaned training copies at the same time.

    Eval and tool files are not cleaned. Kept so the storage note can show the
    number that failed the floor before option 2 was approved.
    """
    return train_stored + other_stored + train_stored + snapshots + counts


def approved_peak(train_stored: int, other_stored: int, snapshots: int, counts: int) -> int:
    """Gated peak after the 4 Nov 2026 decision.

    One training copy: raw parquet until a verified cleaned parquet replaces it.
    Eval, decontamination and tool files stay. The Hub cache is not in this peak.
    """
    return train_stored + other_stored + snapshots + counts


def download_peak(stored_total: int, transient_max: int) -> int:
    """All raw outputs exist and one source's Hub cache is still present."""
    return stored_total + transient_max


def balanced_train_caps(
    train_items: list[tuple[str, int]],
    other_stored: int,
    free_bytes: int,
    floor: int,
    snapshots: int,
    counts: int,
    *,
    fw2_ceiling: int = 3 * GB,
    large_ceiling: int = 8 * GB,
) -> list[tuple[str, int, int]]:
    """Caps that start with fw2_npi and still leave Nepali text in every large source.

    fw2_npi is capped at 3 GB first. Other training sources above 8 GB are then
    capped at 8 GB, largest first. If the floor is still missed, the largest
    remaining source other than fw2_npi is cut, and fw2_npi is touched again
    only after those are exhausted.
    """
    stored = {source_id: size for source_id, size in train_items}
    original = dict(stored)

    def fits() -> bool:
        return (
            free_bytes - resident_peak(sum(stored.values()), other_stored, snapshots, counts)
            >= floor
        )

    def deficit() -> int:
        peak = resident_peak(sum(stored.values()), other_stored, snapshots, counts)
        return peak + floor - free_bytes

    if "fw2_npi" in stored and stored["fw2_npi"] > fw2_ceiling and not fits():
        stored["fw2_npi"] = fw2_ceiling
    while not fits():
        above = [source_id for source_id, size in stored.items() if size > large_ceiling]
        if not above:
            break
        source_id = max(above, key=lambda item: (stored[item], item))
        stored[source_id] = large_ceiling
    order = sorted(
        (source_id for source_id in stored if source_id != "fw2_npi"),
        key=lambda item: (-stored[item], item),
    )
    if "fw2_npi" in stored:
        order.append("fw2_npi")
    for source_id in order:
        if fits():
            break
        need = (deficit() + 1) // 2
        stored[source_id] = max(0, stored[source_id] - need)
    return [
        (source_id, original[source_id], stored[source_id])
        for source_id in original
        if stored[source_id] != original[source_id]
    ]


def propose_train_caps(
    train_items: list[tuple[str, int, int]],
    other_stored: int,
    free_bytes: int,
    floor: int,
    snapshots: int,
    counts: int,
) -> list[tuple[str, int, int]]:
    """Cut training sources until ``free - peak >= floor``.

    ``train_items`` is ``(id, stored_bytes, priority)``. Lower priority is cut
    first. Each byte removed from a training source removes two bytes from the
    peak (raw and cleaned). Returns ``(id, old, new)`` for sources that change.
    """
    ordered = sorted(train_items, key=lambda item: (item[2], -item[1], item[0]))
    stored = {source_id: size for source_id, size, _priority in ordered}

    def peak() -> int:
        return resident_peak(sum(stored.values()), other_stored, snapshots, counts)

    cuts: list[tuple[str, int, int]] = []
    for source_id, old, _priority in ordered:
        if free_bytes - peak() >= floor:
            break
        deficit = peak() + floor - free_bytes
        need_cut = (deficit + 1) // 2
        cut = min(stored[source_id], need_cut)
        if cut <= 0:
            continue
        stored[source_id] -= cut
        cuts.append((source_id, old, stored[source_id]))
    return cuts


def format_bytes(n: int) -> str:
    return f"{n / GB:.3f} GB ({n / 1024**3:.3f} GiB, {n} bytes)"


def free_bytes_df(path: Path = Path(".")) -> tuple[int, str]:
    out = subprocess.check_output(["df", "-k", str(path)], text=True)
    lines = [line for line in out.splitlines() if line.strip()]
    avail_kib = int(lines[-1].split()[3])
    return avail_kib * 1024, out.rstrip()


def memsize_bytes() -> int:
    raw = subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True).strip()
    return int(raw)


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def load_spec(path: Path) -> dict:
    with path.open() as handle:
        spec = yaml.safe_load(handle)
    if not isinstance(spec, dict) or "sources" not in spec:
        raise ValueError(f"{path} has no sources list")
    return spec


def list_files(api: object, repo: str, repo_path: str, repo_type: str) -> list[RemoteFile]:
    """File sizes under ``repo_path``. A path that is itself a file is one row."""
    files: list[RemoteFile] = []
    try:
        items = api.list_repo_tree(  # type: ignore[attr-defined]
            repo,
            repo_type=repo_type,
            path_in_repo=repo_path,
            recursive=True,
        )
        item_list = list(items)
    except RemoteEntryNotFoundError:
        item_list = []
    for item in item_list:
        size = getattr(item, "size", None)
        if size is None:
            continue
        files.append(RemoteFile(item.path, int(size)))
    if files:
        return files
    parent = str(Path(repo_path).parent)
    parent_arg = None if parent == "." else parent
    name = Path(repo_path).name
    items = api.list_repo_tree(  # type: ignore[attr-defined]
        repo,
        repo_type=repo_type,
        path_in_repo=parent_arg,
        recursive=False,
    )
    for item in items:
        size = getattr(item, "size", None)
        if size is None:
            continue
        if Path(item.path).name == name or item.path == repo_path:
            files.append(RemoteFile(item.path, int(size)))
    if not files:
        raise FileNotFoundError(f"no files for {repo}:{repo_path}")
    return files


def _filter_names(files: list[RemoteFile], needle: str | None) -> list[RemoteFile]:
    if not needle:
        return files
    return [item for item in files if needle in Path(item.path).name]


def measure_source(api: object, spec: dict) -> MeasuredSource:
    download = bool(spec.get("download", True))
    if spec.get("repo_type") == "github":
        fixed = int(spec.get("fixed_bytes", 0))
        return MeasuredSource(
            source_id=spec["id"],
            bucket=spec.get("bucket", ""),
            role=spec.get("role", ""),
            repo=spec.get("repo", ""),
            download=download,
            upstream_bytes=fixed,
            stored_bytes=fixed if download else 0,
            transient_bytes=fixed if download else 0,
            n_files=2,
            note=str(spec.get("note") or ""),
            card_license=str(spec.get("card_license") or ""),
            paths=list(spec.get("paths") or []),
        )
    groups: list[list[RemoteFile]] = []
    needle = spec.get("path_contains")
    for repo_path in spec["paths"]:
        found = _filter_names(
            list_files(api, spec["repo"], repo_path, spec.get("repo_type", "dataset")),
            needle,
        )
        if not found:
            raise FileNotFoundError(f"no files for {spec['repo']}:{repo_path}")
        groups.append(found)
    keep_ratio = spec.get("keep_ratio")
    keep = float(keep_ratio) if keep_ratio not in (None, "null") else None
    if "per_path_cap" in spec:
        cap = int(spec["per_path_cap"])
        stored = per_path_stored([[item.size for item in group] for group in groups], cap)
        transient = per_path_transient([[item.size for item in group] for group in groups], cap)
        upstream = sum(item.size for group in groups for item in group)
    else:
        sizes = [item.size for group in groups for item in group]
        amount = spec.get("amount", "all")
        stored = stored_bytes(sizes, amount)
        transient = transient_bytes(sizes, amount, keep_ratio=keep)
        upstream = sum(sizes)
    return MeasuredSource(
        source_id=spec["id"],
        bucket=spec.get("bucket", ""),
        role=spec.get("role", ""),
        repo=spec.get("repo", ""),
        download=download,
        upstream_bytes=upstream,
        stored_bytes=stored if download else 0,
        transient_bytes=transient if download else 0,
        n_files=sum(len(group) for group in groups),
        note=str(spec.get("note") or ""),
        card_license=str(spec.get("card_license") or ""),
        paths=list(spec.get("paths") or []),
    )


def repo_revision(api: object, repo: str, repo_type: str, pinned: str | None = None) -> str:
    if pinned:
        return str(pinned)
    if repo_type == "github":
        raise ValueError(f"{repo} needs a pinned revision")
    if repo_type == "model":
        info = api.model_info(repo)  # type: ignore[attr-defined]
    else:
        info = api.dataset_info(repo)  # type: ignore[attr-defined]
    return str(info.sha)


def render_markdown(
    spec: dict,
    rows: list[MeasuredSource],
    *,
    free: int,
    df_text: str,
    ram_bytes: int,
    commit: str,
    revisions: dict[str, str],
) -> str:
    floor = int(spec["free_floor_bytes"])
    snap = int(spec["snapshot_bytes"])
    snap_m4 = int(spec["snapshot_count_m4"])
    snap_m2 = int(spec["snapshot_count_m2_wording"])
    counts = int(spec["counts_allowance_bytes"])
    active = [row for row in rows if row.download]
    held = [row for row in rows if not row.download]
    train = sum(row.stored_bytes for row in active if row.role == "train")
    other = sum(row.stored_bytes for row in active if row.role != "train")
    raw = train + other
    transient = max((row.transient_bytes for row in active), default=0)
    snapshots_m4 = snap * snap_m4
    snapshots_m2 = snap * snap_m2
    peak_dl = download_peak(raw, transient)
    rejected = resident_peak(train, other, snapshots_m4, counts)
    peak = approved_peak(train, other, snapshots_m4, counts)
    peak_if_m2_count = approved_peak(train, other, snapshots_m2, counts)
    remain = free - peak
    remain_dl = free - peak_dl
    stop = remain < floor or remain_dl < floor
    lines: list[str] = []
    lines.append("# Storage plan (M2 step 1)")
    lines.append("")
    lines.append(
        "File sizes are Hub `list_repo_tree` sizes (exact). "
        "A bracketed PLAN.md budget is a UTF-8 cap, and 1 GB = 10^9 bytes. "
        "Stored parquet of a capped source is at most that cap. "
        f"Git commit `{commit}`."
    )
    lines.append("")
    lines.append("## Machine now")
    lines.append("")
    lines.append(f"- RAM: {ram_bytes / 1024**3:.1f} GiB (`sysctl -n hw.memsize` = {ram_bytes}).")
    lines.append("- Under 32 GB RAM, PLAN.md 3.8 sets each snapshot to 5 GB, not 10 GB.")
    lines.append(f"- Free space: {format_bytes(free)}.")
    lines.append("")
    lines.append("```")
    lines.append(df_text)
    lines.append("```")
    lines.append("")
    lines.append("## What would be stored")
    lines.append("")
    lines.append(
        "Upstream is the sum of the files under the paths in `configs/sources.yaml`. "
        "Stored is upstream for `all`, or the minimum of upstream and the cap. "
        "Transient is the Hub cache bound for that source alone; the cache is deleted "
        "after the source is verified."
    )
    lines.append("")
    lines.append("| source | role | files | upstream | stored upper bound | cache upper bound |")
    lines.append("|---|---|---:|---:|---:|---:|")
    for row in active:
        lines.append(
            f"| `{row.source_id}` | {row.role} | {row.n_files} | "
            f"{row.upstream_bytes} | {row.stored_bytes} | {row.transient_bytes} |"
        )
    lines.append("")
    lines.append("## Components")
    lines.append("")
    lines.append("| component | bytes | notes |")
    lines.append("|---|---:|---|")
    lines.append(f"| raw parquet (stored upper bound) | {raw} | {format_bytes(raw)} |")
    lines.append(
        f"| transient Hugging Face cache | {transient} | "
        f"largest single source, deleted before the next source. {format_bytes(transient)} |"
    )
    lines.append(
        f"| cleaned parquet | {train} | replaces training raw after it is verified. "
        f"Not a second resident copy. {format_bytes(train)} |"
    )
    lines.append(
        f"| snapshots | {snapshots_m4} | M4 writes {snap_m4} × {snap} bytes "
        f"(5 GB because RAM < 32 GB). M2's wording of 3 snapshots would be {snapshots_m2}. |"
    )
    lines.append(
        f"| counts, tokenizers, eval | {counts} | planning allowance, not an API figure. "
        f"{format_bytes(counts)} |"
    )
    lines.append("")
    lines.append("## Approved retention (4 Nov 2026)")
    lines.append("")
    lines.append(
        "Option 2 is approved. For each training source, delete its raw parquet only "
        "after its cleaned parquet is written and verified against the manifest, and "
        "only after the exact Hugging Face revision (commit hash) is in that manifest. "
        "Never delete eval-only or decontamination-only sources. The 50 GB rule still "
        "applies before every large write. Caps and an external drive were not chosen."
    )
    lines.append("")
    lines.append("## Peak and the 50 GB floor")
    lines.append("")
    lines.append(
        "The floor is 50 × 10^9 bytes. The gated peak is one copy of the stored files "
        "(training raw until the cleaned parquet replaces it), plus four snapshots, "
        "plus the counts allowance. Download peak is every raw output plus one source's "
        "Hub cache. The cache is deleted after that source's shards are verified."
    )
    lines.append("")
    lines.append(f"- Approved peak (stored files + 4 snapshots + counts): {format_bytes(peak)}.")
    lines.append(
        f"- Same peak with the 3 snapshots in the M2 wording: {format_bytes(peak_if_m2_count)}."
    )
    lines.append(f"- Download peak (raw + one cache): {format_bytes(peak_dl)}.")
    lines.append(f"- Free now minus approved peak: {format_bytes(remain)}.")
    lines.append(f"- Free now minus download peak: {format_bytes(remain_dl)}.")
    lines.append(
        "- Rejected peak if raw and cleaned were both kept: "
        f"{format_bytes(rejected)}, leaving {format_bytes(free - rejected)}."
    )
    lines.append("")
    if stop:
        lines.append(
            f"**STOP (human).** The approved peak leaves {format_bytes(remain)} and the "
            f"download peak leaves {format_bytes(remain_dl)}. One of those is under 50 GB. "
            "No further download should start."
        )
    else:
        lines.append("The approved peak and the download peak each leave at least 50 GB free.")
    lines.append("")
    lines.append("## Held, not in the total")
    lines.append("")
    if not held:
        lines.append("None. Every row in `configs/sources.yaml` is in the total above.")
        lines.append("")
    else:
        for row in held:
            lines.append(
                f"- `{row.source_id}` upstream {format_bytes(row.upstream_bytes)} "
                f"across {row.n_files} files. Not downloaded. {row.note}"
            )
        lines.append("")
    lines.append("## Revisions queried with the sizes")
    lines.append("")
    lines.append("| repo | revision |")
    lines.append("|---|---|")
    for repo, sha in revisions.items():
        lines.append(f"| `{repo}` | `{sha}` |")
    lines.append("")
    lines.append("## Notes that are not size estimates")
    lines.append("")
    lines.append(
        "- `und_Deva` is a real FineWeb-2 config. The card calls it undetermined "
        "language and says it was uploaded without deduplication or filtering. "
        "It is inside `deva_pool_fw2` because M2 says every other `*_Deva` config. "
        "Its file size is exact; the decision to train on it is the plan's."
    )
    lines.append(
        "- Sangraha `verified/kas` and `verified/snd` have no script tag on the card. "
        "They are in the pool because FineWeb-2 has `kas_Deva` and `snd_Deva`. "
        "The first rows should be checked for Devanagari before those shards are kept."
    )
    lines.append(
        "- IndicCorp v2 `npi_Deva` is the file `data/ne.txt` (the README data_files map). "
        "It is plain text. Blank-line versus one-line documents is decided while reading "
        "the file, not guessed here."
    )
    lines.append(
        "- GlotLID `model.bin` is v3. The model card says `model.bin` stays at stable v3 "
        "while `model_v4_alpha.bin` is a pre-release."
    )
    lines.append(
        "- Papaya affix inventory is approved for evaluation only under CC BY 4.0 at "
        "commit `b3498528daa3435f091792e0b44bb222a2f8cb68`. Attribution: re-derived from "
        'Balaram Prasain, "A Computational Analysis of Nepali Morphology: A Model for '
        'Natural Language Processing", PhD dissertation, Tribhuvan University, 2011. '
        "The files stay under `data/` and are not committed. The byte total is the sum "
        "of the two inventory blobs (81127 + 491081), not a Hub tree size."
    )
    lines.append(
        "- `nepal_pool` is the ten approved FineWeb-2 train splits: `new_Deva`, "
        "`thl_Deva`, `taj_Deva`, `lif_Deva`, `rav_Deva`, `suz_Deva`, `xsr_Deva`, "
        "`rjs_Deva`, `kle_Deva`, `nwx_Deva`. No upsampling. NepTam and GurungJasmika "
        "are not included. FineWeb-2 `npi_Latn` test is eval-only; its train split "
        "is not downloaded."
    )
    lines.append(
        "- Code parquet shard `data/train-00000-of-00880.parquet` was opened. "
        "Columns are `code`, `repo_name`, `path`, `language`, `license`, `size`. "
        "Kept rows are mit, apache-2.0, bsd-2-clause and bsd-3-clause. "
        "On that shard, 550829383 kept UTF-8 bytes / 355122552 file bytes "
        "= keep_ratio 1.551095178. The cache bound uses that ratio. "
        "It is one shard, not a census of all 880 files."
    )
    lines.append("")
    return "\n".join(lines) + "\n"


def build(spec_path: Path, doc_path: Path) -> int:
    spec = load_spec(spec_path)
    from huggingface_hub import HfApi

    api = HfApi()
    rows = [measure_source(api, source) for source in spec["sources"]]
    revisions: dict[str, str] = {}
    for source in spec["sources"]:
        repo = source.get("repo")
        if not repo or repo in revisions:
            continue
        revisions[repo] = repo_revision(
            api,
            repo,
            source.get("repo_type", "dataset"),
            source.get("revision"),
        )
    free, df_text = free_bytes_df(REPO_ROOT)
    text = render_markdown(
        spec,
        rows,
        free=free,
        df_text=df_text,
        ram_bytes=memsize_bytes(),
        commit=git_commit(),
        revisions=revisions,
    )
    doc_path.write_text(text)
    floor = int(spec["free_floor_bytes"])
    snap = int(spec["snapshot_bytes"]) * int(spec["snapshot_count_m4"])
    counts = int(spec["counts_allowance_bytes"])
    active = [row for row in rows if row.download]
    train = sum(row.stored_bytes for row in active if row.role == "train")
    other = sum(row.stored_bytes for row in active if row.role != "train")
    transient = max((row.transient_bytes for row in active), default=0)
    peak = approved_peak(train, other, snap, counts)
    peak_dl = download_peak(train + other, transient)
    remain = free - peak
    print(f"wrote {doc_path}")
    print(f"free_bytes={free}")
    print(f"approved_peak={peak}")
    print(f"download_peak={peak_dl}")
    print(f"remain_bytes={remain}")
    print(f"download_remain_bytes={free - peak_dl}")
    if remain < floor or free - peak_dl < floor:
        print("STOP (human): projected peak leaves under 50 GB free.")
        return 2
    print("storage floor ok")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Write docs/STORAGE.md from Hub file sizes.")
    parser.add_argument("--sources", type=Path, default=DEFAULT_SOURCES)
    parser.add_argument("--doc", type=Path, default=DEFAULT_DOC)
    args = parser.parse_args(argv)
    return build(args.sources, args.doc)


if __name__ == "__main__":
    sys.exit(main())

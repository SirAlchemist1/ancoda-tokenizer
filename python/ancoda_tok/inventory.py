"""M5: akshara inventory and Block 1 covered-set selection (PLAN.md 3.6).

Counts every unit containing at least one Devanagari-block code point
(U+0900–U+097F) over the 50% training snapshot, then runs the exact 3.6
selection for 32k / 64k / 128k. Repeats with gb9c=true for the ablation.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from ancoda_tok.clean.common import (
    SEED,
    Stop,
    base_manifest,
    finish_manifest,
    free_bytes,
    list_shards,
    save_json,
    sha256_file,
)
from ancoda_tok.storage import REPO_ROOT

# Caps = 5% of vocab (PLAN.md 3.6 step 6).
CAPS: dict[str, int] = {
    "32k": 1_638,
    "64k": 3_276,
    "128k": 6_553,
}
VOCAB_TOTALS: dict[str, int] = {
    "32k": 32_768,
    "64k": 65_536,
    "128k": 131_072,
}
COVERAGE_TARGET = 0.9999
BLOCK0_CPS = frozenset(range(0x0900, 0x0980)) | frozenset({0x200C, 0x200D})
DEVA_BLOCK = frozenset(range(0x0900, 0x0980))

INVENTORY_DIR = REPO_ROOT / "artifacts" / "inventory"
FIGURES_DIR = REPO_ROOT / "docs" / "figures"


def find_snapshot(prefix: str) -> Path:
    """Return the unique snapshot directory whose name starts with ``prefix``.

    For ``n50``, directories starting with ``n50_nojoiner`` are excluded so the
    joiner and no-joiner 50% snapshots stay distinct.
    """
    root = REPO_ROOT / "data" / "snapshots"
    if not root.is_dir():
        raise Stop(f"STOP: missing snapshots dir {root}")
    matches = []
    for path in sorted(root.iterdir()):
        if not path.is_dir() or not path.name.startswith(prefix):
            continue
        if prefix == "n50" and path.name.startswith("n50_nojoiner"):
            continue
        matches.append(path)
    if len(matches) != 1:
        raise Stop(
            f"STOP: expected exactly one {prefix} snapshot, found {[p.name for p in matches]}"
        )
    return matches[0]


def find_n50_snapshot() -> Path:
    return find_snapshot("n50")


def code_points_field(unit: str) -> str:
    return " ".join(f"U+{ord(c):04X}" for c in unit)


def unit_has_deva(unit: str) -> bool:
    return any(ord(c) in DEVA_BLOCK for c in unit)


def is_block0_cp(cp: int) -> bool:
    return cp in BLOCK0_CPS


def is_eligible(unit: str, letters: frozenset[int]) -> bool:
    if not unit:
        return False
    cps = [ord(c) for c in unit]
    if cps[0] not in letters:
        return False
    return all(is_block0_cp(cp) for cp in cps)


def is_n0_unit(unit: str) -> bool:
    return len(unit) == 1 and is_block0_cp(ord(unit[0]))


def chain_prefixes(unit: str) -> list[str]:
    """Block 1 merge tokens for a multi-code-point unit: c1+c2, c1c2+c3, …"""
    chars = list(unit)
    if len(chars) < 2:
        return []
    out: list[str] = []
    acc = chars[0]
    for ch in chars[1:]:
        acc = acc + ch
        out.append(acc)
    return out


def _count_shard(args: tuple[str, bool]) -> tuple[str, dict[str, int], int, int]:
    """Worker: count Devanagari-bearing units in one parquet shard."""
    path_str, gb9c = args
    import ancoda_seg

    path = Path(path_str)
    table = pq.read_table(path, columns=["text"])
    texts = table.column("text").to_pylist()
    counts: Counter[str] = Counter()
    n_docs = 0
    n_units = 0
    # Batch to keep Python↔Rust traffic reasonable without holding huge lists.
    batch_size = 256
    for start in range(0, len(texts), batch_size):
        batch = [t if isinstance(t, str) else "" for t in texts[start : start + batch_size]]
        n_docs += len(batch)
        segmented = ancoda_seg.aksharas_batch(batch, gb9c)
        for units in segmented:
            for unit in units:
                if unit_has_deva(unit):
                    counts[unit] += 1
                    n_units += 1
    return path.name, dict(counts), n_docs, n_units


def count_snapshot(
    snapshot: Path,
    *,
    gb9c: bool,
    out_dir: Path,
    workers: int,
) -> Counter[str]:
    """Stream snapshot shards; resume from completed per-shard count files."""
    shards = list_shards(snapshot)
    if not shards:
        raise Stop(f"STOP: no shards in {snapshot}")
    shard_dir = out_dir / ("count_shards_gb9c" if gb9c else "count_shards")
    shard_dir.mkdir(parents=True, exist_ok=True)

    pending: list[Path] = []
    merged: Counter[str] = Counter()
    docs_total = 0
    units_total = 0
    for shard in shards:
        dest = shard_dir / f"{shard.stem}.json"
        if dest.exists():
            data = json.loads(dest.read_text(encoding="utf-8"))
            merged.update(data["counts"])
            docs_total += int(data["docs"])
            units_total += int(data["units"])
        else:
            pending.append(shard)

    print(
        f"count mode={'gb9c' if gb9c else 'default'} "
        f"done={len(shards) - len(pending)}/{len(shards)} pending={len(pending)} "
        f"workers={workers}",
        flush=True,
    )
    if pending:
        tasks = [(str(p), gb9c) for p in pending]
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_count_shard, t): t[0] for t in tasks}
            for fut in as_completed(futures):
                name, counts, docs, units = fut.result()
                dest = shard_dir / f"{Path(name).stem}.json"
                payload = {
                    "shard": name,
                    "gb9c": gb9c,
                    "docs": docs,
                    "units": units,
                    "unique": len(counts),
                    "counts": counts,
                }
                dest.write_text(
                    json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
                merged.update(counts)
                docs_total += docs
                units_total += units
                print(
                    f"  shard {name} docs={docs} units={units} unique={len(counts)}",
                    flush=True,
                )

    print(
        f"count complete docs={docs_total} units={units_total} unique={len(merged)}",
        flush=True,
    )
    return merged


def write_counts_tsv(counts: Counter[str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0].encode("utf-8")))
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("akshara\tcode_points\tcount\n")
        for unit, count in rows:
            # Escape tabs/newlines in the rare malformed unit.
            ak = unit.replace("\\", "\\\\").replace("\t", "\\t").replace("\n", "\\n")
            handle.write(f"{ak}\t{code_points_field(unit)}\t{count}\n")


def select_block1(
    counts: Counter[str],
    *,
    cap: int,
    letters: frozenset[int],
) -> dict[str, Any]:
    """PLAN.md 3.6 exact selection."""
    n = 0
    n0 = 0
    ineligible_occ = 0
    eligible_multi: list[tuple[str, int]] = []

    for unit, count in counts.items():
        n += count
        eligible = is_eligible(unit, letters)
        if is_n0_unit(unit):
            n0 += count
        if not eligible:
            ineligible_occ += count
            continue
        if len(unit) >= 2:
            eligible_multi.append((unit, count))

    eligible_multi.sort(key=lambda kv: (-kv[1], kv[0].encode("utf-8")))

    selected_merges: list[str] = []
    selected_set: set[str] = set()
    selected_units: list[dict[str, Any]] = []
    covered_selected_occ = 0
    curve: list[dict[str, float | int]] = []
    stop_reason = "exhausted"

    for unit, count in eligible_multi:
        prefixes = chain_prefixes(unit)
        cost = sum(1 for p in prefixes if p not in selected_set)
        if len(selected_merges) + cost > cap:
            stop_reason = "cap"
            break
        for p in prefixes:
            if p not in selected_set:
                selected_set.add(p)
                selected_merges.append(p)
        selected_units.append(
            {
                "akshara": unit,
                "code_points": code_points_field(unit),
                "count": count,
                "cost": cost,
                "merges_after": len(selected_merges),
            }
        )
        covered_selected_occ += count
        coverage = (n0 + covered_selected_occ) / n if n else 0.0
        curve.append(
            {
                "block1_merges": len(selected_merges),
                "units_selected": len(selected_units),
                "coverage": coverage,
                "covered_occurrences": n0 + covered_selected_occ,
            }
        )
        if coverage >= COVERAGE_TARGET:
            stop_reason = "coverage"
            break

    coverage = (n0 + covered_selected_occ) / n if n else 0.0
    uncovered_occ = n - n0 - covered_selected_occ

    selected_unit_set = {u["akshara"] for u in selected_units}
    uncovered: list[tuple[str, int]] = []
    for unit, count in counts.items():
        if is_n0_unit(unit):
            continue
        if unit in selected_unit_set:
            continue
        uncovered.append((unit, count))
    uncovered.sort(key=lambda kv: (-kv[1], kv[0].encode("utf-8")))
    top50 = [
        {
            "akshara": u,
            "code_points": code_points_field(u),
            "count": c,
            "eligible": is_eligible(u, letters),
        }
        for u, c in uncovered[:50]
    ]

    return {
        "N": n,
        "N0": n0,
        "ineligible_occurrences": ineligible_occ,
        "eligible_multi_types": len(eligible_multi),
        "unit_types_selected": len(selected_units),
        "units_selected": selected_units,
        "block1_merge_count": len(selected_merges),
        "block1_merges": selected_merges,
        "coverage": coverage,
        "uncovered_occurrences": uncovered_occ,
        "top_50_uncovered_aksharas": top50,
        "stop_reason": stop_reason,
        "cap": cap,
        "coverage_curve": curve,
    }


def write_coverage_svg(
    curves: dict[str, list[dict[str, float | int]]],
    path: Path,
    *,
    title: str,
) -> None:
    """Deterministic SVG: coverage vs Block 1 merges for 32k/64k/128k."""
    path.parent.mkdir(parents=True, exist_ok=True)
    width, height = 840, 480
    margin_l, margin_r, margin_t, margin_b = 64, 24, 40, 48
    plot_w = width - margin_l - margin_r
    plot_h = height - margin_t - margin_b

    max_merges = 1
    for curve in curves.values():
        for pt in curve:
            max_merges = max(max_merges, int(pt["block1_merges"]))
    # Y axis from a floor just below the N0-only coverage up to 1.0.
    y_min = 1.0
    for curve in curves.values():
        if curve:
            y_min = min(y_min, float(curve[0]["coverage"]))
    y_min = max(0.0, y_min - 0.01)
    y_max = 1.0
    y_span = max(y_max - y_min, 1e-9)

    colors = {"32k": "#1b9e77", "64k": "#d95f02", "128k": "#7570b3"}

    def x_of(merges: int) -> float:
        return margin_l + plot_w * (merges / max_merges)

    def y_of(cov: float) -> float:
        return margin_t + plot_h * (1.0 - (cov - y_min) / y_span)

    parts: list[str] = [
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
            f'viewBox="0 0 {width} {height}">'
        ),
        "<rect width='100%' height='100%' fill='white'/>",
        (
            f"<text x='{width / 2}' y='24' text-anchor='middle' "
            f"font-family='Helvetica,Arial,sans-serif' font-size='16'>{title}</text>"
        ),
        (
            f"<line x1='{margin_l}' y1='{margin_t}' x2='{margin_l}' y2='{margin_t + plot_h}' "
            f"stroke='#333' stroke-width='1'/>"
        ),
        (
            f"<line x1='{margin_l}' y1='{margin_t + plot_h}' x2='{margin_l + plot_w}' "
            f"y2='{margin_t + plot_h}' stroke='#333' stroke-width='1'/>"
        ),
        (
            f"<text x='{width / 2}' y='{height - 12}' text-anchor='middle' "
            f"font-family='Helvetica,Arial,sans-serif' font-size='12'>"
            f"Block 1 merges</text>"
        ),
        (
            f"<text x='16' y='{height / 2}' text-anchor='middle' "
            f"font-family='Helvetica,Arial,sans-serif' font-size='12' "
            f"transform='rotate(-90 16 {height / 2})'>coverage</text>"
        ),
    ]

    # Y ticks
    for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
        cov = y_min + y_span * frac
        y = y_of(cov)
        parts.append(
            f"<line x1='{margin_l - 4}' y1='{y}' x2='{margin_l}' y2='{y}' stroke='#333'/>"
        )
        parts.append(
            f"<text x='{margin_l - 8}' y='{y + 4}' text-anchor='end' "
            f"font-family='Helvetica,Arial,sans-serif' font-size='10'>{cov:.4f}</text>"
        )

    # X ticks
    for merges in (0, max_merges // 4, max_merges // 2, 3 * max_merges // 4, max_merges):
        x = x_of(merges)
        parts.append(
            f"<line x1='{x}' y1='{margin_t + plot_h}' x2='{x}' "
            f"y2='{margin_t + plot_h + 4}' stroke='#333'/>"
        )
        parts.append(
            f"<text x='{x}' y='{margin_t + plot_h + 18}' text-anchor='middle' "
            f"font-family='Helvetica,Arial,sans-serif' font-size='10'>{merges}</text>"
        )

    legend_y = margin_t + 8
    for i, name in enumerate(("32k", "64k", "128k")):
        curve = curves.get(name) or []
        if not curve:
            continue
        color = colors[name]
        pts = " ".join(
            f"{x_of(int(pt['block1_merges'])):.2f},{y_of(float(pt['coverage'])):.2f}"
            for pt in curve
        )
        parts.append(
            f"<polyline fill='none' stroke='{color}' stroke-width='2' points='{pts}'/>"
        )
        lx = margin_l + plot_w - 120
        ly = legend_y + i * 18
        parts.append(f"<rect x='{lx}' y='{ly - 8}' width='12' height='12' fill='{color}'/>")
        parts.append(
            f"<text x='{lx + 18}' y='{ly + 2}' font-family='Helvetica,Arial,sans-serif' "
            f"font-size='12'>{name} (cap {CAPS[name]})</text>"
        )

    parts.append("</svg>")
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def run_selection_suite(
    counts: Counter[str],
    *,
    out_dir: Path,
    gb9c: bool,
    letters: frozenset[int],
    snapshot: Path,
    counts_path: Path,
) -> dict[str, Any]:
    curves: dict[str, list[dict[str, float | int]]] = {}
    results: dict[str, Any] = {}
    for name, cap in CAPS.items():
        selected = select_block1(counts, cap=cap, letters=letters)
        if selected["block1_merge_count"] > cap:
            raise Stop(
                f"STOP: Block 1 merge count {selected['block1_merge_count']} exceeds cap {cap}"
            )
        payload = {
            "vocab_name": name,
            "vocab_size": VOCAB_TOTALS[name],
            "gb9c": gb9c,
            "snapshot": str(snapshot.relative_to(REPO_ROOT)),
            "counts_tsv": str(counts_path.relative_to(REPO_ROOT)),
            "counts_tsv_sha256": sha256_file(counts_path),
            "seed": SEED,
            **{k: selected[k] for k in (
                "N",
                "N0",
                "ineligible_occurrences",
                "eligible_multi_types",
                "unit_types_selected",
                "block1_merge_count",
                "coverage",
                "uncovered_occurrences",
                "top_50_uncovered_aksharas",
                "stop_reason",
                "cap",
            )},
            "units_selected": selected["units_selected"],
            "block1_merges": selected["block1_merges"],
            "coverage_curve": selected["coverage_curve"],
        }
        out_name = f"covered_{name}_gb9c.json" if gb9c else f"covered_{name}.json"
        out_path = out_dir / out_name
        save_json(out_path, payload)
        curves[name] = selected["coverage_curve"]
        results[name] = {
            "path": str(out_path.relative_to(REPO_ROOT)),
            "sha256": sha256_file(out_path),
            "block1_merge_count": selected["block1_merge_count"],
            "coverage": selected["coverage"],
            "stop_reason": selected["stop_reason"],
            "unit_types_selected": selected["unit_types_selected"],
            "N": selected["N"],
            "N0": selected["N0"],
            "ineligible_occurrences": selected["ineligible_occurrences"],
            "uncovered_occurrences": selected["uncovered_occurrences"],
        }
        print(
            f"covered {name} gb9c={gb9c} merges={selected['block1_merge_count']}/{cap} "
            f"coverage={selected['coverage']:.10f} stop={selected['stop_reason']}",
            flush=True,
        )

    svg_name = "coverage_vs_block1_gb9c.svg" if gb9c else "coverage_vs_block1.svg"
    title = (
        "Coverage vs Block 1 merges (gb9c=true)"
        if gb9c
        else "Coverage vs Block 1 merges (default)"
    )
    art_svg = out_dir / svg_name
    write_coverage_svg(curves, art_svg, title=title)
    fig_svg = FIGURES_DIR / ("m5_" + svg_name)
    write_coverage_svg(curves, fig_svg, title=title)
    results["plot"] = {
        "artifacts": str(art_svg.relative_to(REPO_ROOT)),
        "docs": str(fig_svg.relative_to(REPO_ROOT)),
        "artifacts_sha256": sha256_file(art_svg),
        "docs_sha256": sha256_file(fig_svg),
    }
    return results


def run_inventory(*, force: bool = False) -> dict[str, Any]:
    import ancoda_seg

    workers = max(1, int(os.environ.get("ANCODA_WORKERS", "5")))
    snapshot = find_n50_snapshot()
    out_dir = INVENTORY_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    letters = frozenset(int(x) for x in ancoda_seg.devanagari_letters())

    if force:
        for sub in ("count_shards", "count_shards_gb9c"):
            d = out_dir / sub
            if d.exists():
                for f in d.glob("*.json"):
                    f.unlink()

    free = free_bytes()
    if free < 50_000_000_000:
        raise Stop(f"STOP: free_bytes={free} under 50 GB before inventory")

    manifest = base_manifest(
        stage="inventory",
        snapshot=str(snapshot.relative_to(REPO_ROOT)),
        snapshot_manifest_sha256=sha256_file(snapshot / "manifest.json"),
        gb9c_modes=[False, True],
        workers=workers,
    )

    summary: dict[str, Any] = {"snapshot": str(snapshot.relative_to(REPO_ROOT))}

    for gb9c in (False, True):
        counts = count_snapshot(snapshot, gb9c=gb9c, out_dir=out_dir, workers=workers)
        tsv_name = "akshara_counts_gb9c.tsv" if gb9c else "akshara_counts.tsv"
        tsv_path = out_dir / tsv_name
        write_counts_tsv(counts, tsv_path)
        tsv_sha = sha256_file(tsv_path)
        print(f"wrote {tsv_path} sha256={tsv_sha} unique={len(counts)}", flush=True)
        key = "gb9c" if gb9c else "default"
        summary[key] = {
            "counts_tsv": str(tsv_path.relative_to(REPO_ROOT)),
            "counts_tsv_sha256": tsv_sha,
            "unique_units": len(counts),
            "total_occurrences": int(sum(counts.values())),
            "covered": run_selection_suite(
                counts,
                out_dir=out_dir,
                gb9c=gb9c,
                letters=letters,
                snapshot=snapshot,
                counts_path=tsv_path,
            ),
        }

    manifest = finish_manifest(manifest)
    manifest["summary"] = summary
    man_path = out_dir / "manifest.json"
    save_json(man_path, manifest)
    print(f"wrote {man_path}", flush=True)
    return summary


def verify_deterministic(*, gb9c: bool = False) -> None:
    """Re-run selection from the TSV and check covered JSON sha256s match."""
    import ancoda_seg

    letters = frozenset(int(x) for x in ancoda_seg.devanagari_letters())
    tsv = INVENTORY_DIR / ("akshara_counts_gb9c.tsv" if gb9c else "akshara_counts.tsv")
    if not tsv.exists():
        raise Stop(f"STOP: missing {tsv}")
    counts: Counter[str] = Counter()
    with tsv.open(encoding="utf-8") as handle:
        header = handle.readline()
        if not header.startswith("akshara\t"):
            raise Stop(f"STOP: bad TSV header in {tsv}")
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 3:
                continue
            ak, _cps, count_s = parts
            ak = (
                ak.replace("\\n", "\n")
                .replace("\\t", "\t")
                .replace("\\\\", "\\")
            )
            counts[ak] += int(count_s)

    for name, cap in CAPS.items():
        selected = select_block1(counts, cap=cap, letters=letters)
        out_name = f"covered_{name}_gb9c.json" if gb9c else f"covered_{name}.json"
        path = INVENTORY_DIR / out_name
        prev = json.loads(path.read_text(encoding="utf-8"))
        checks: list[tuple[str, object, object]] = [
            ("N", prev["N"], selected["N"]),
            ("N0", prev["N0"], selected["N0"]),
            ("ineligible_occurrences", prev["ineligible_occurrences"], selected["ineligible_occurrences"]),
            ("block1_merge_count", prev["block1_merge_count"], selected["block1_merge_count"]),
            ("uncovered_occurrences", prev["uncovered_occurrences"], selected["uncovered_occurrences"]),
            ("stop_reason", prev["stop_reason"], selected["stop_reason"]),
            ("block1_merges", prev["block1_merges"], selected["block1_merges"]),
            (
                "units_selected",
                [u["akshara"] for u in prev["units_selected"]],
                [u["akshara"] for u in selected["units_selected"]],
            ),
        ]
        for field, left, right in checks:
            if left != right:
                raise Stop(f"STOP: determinism fail {out_name} field {field}")
        # Re-write through the same serializer; sha256 must match the on-disk file.
        payload = {
            "vocab_name": name,
            "vocab_size": VOCAB_TOTALS[name],
            "gb9c": gb9c,
            "snapshot": prev["snapshot"],
            "counts_tsv": prev["counts_tsv"],
            "counts_tsv_sha256": prev["counts_tsv_sha256"],
            "seed": SEED,
            **{k: selected[k] for k in (
                "N",
                "N0",
                "ineligible_occurrences",
                "eligible_multi_types",
                "unit_types_selected",
                "block1_merge_count",
                "coverage",
                "uncovered_occurrences",
                "top_50_uncovered_aksharas",
                "stop_reason",
                "cap",
            )},
            "units_selected": selected["units_selected"],
            "block1_merges": selected["block1_merges"],
            "coverage_curve": selected["coverage_curve"],
        }
        import tempfile

        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", delete=False, suffix=".json"
        ) as tmp:
            tmp_path = Path(tmp.name)
        save_json(tmp_path, payload)
        new_sha = sha256_file(tmp_path)
        old_sha = sha256_file(path)
        tmp_path.unlink(missing_ok=True)
        if new_sha != old_sha:
            raise Stop(
                f"STOP: determinism sha mismatch for {out_name}: "
                f"old={old_sha} new={new_sha}"
            )
        print(f"determinism ok {out_name} sha256={old_sha}", flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="ancoda_tok.inventory")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Delete per-shard count caches and recompute",
    )
    parser.add_argument(
        "--verify-deterministic",
        action="store_true",
        help="Re-run 3.6 from the TSV and check covered JSON match",
    )
    args = parser.parse_args(argv)
    try:
        if args.verify_deterministic:
            verify_deterministic(gb9c=False)
            verify_deterministic(gb9c=True)
        else:
            run_inventory(force=args.force)
    except Stop as exc:
        print(exc, file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

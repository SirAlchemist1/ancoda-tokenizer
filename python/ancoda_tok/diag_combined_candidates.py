"""Combined candidates diagnostic (64k × 8 + 128k × 8). Final config unchanged."""

from __future__ import annotations

import argparse
import json
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import ancoda_seg
import sentencepiece as spm

from ancoda_tok.clean.augment_joiners import build_joiner_copy
from ancoda_tok.clean.audit_snapshots import audit_snapshot
from ancoda_tok.clean.common import (
    Stop,
    base_manifest,
    finish_manifest,
    free_bytes,
    iter_parquet_rows,
    load_mix,
    save_json,
    sha256_file,
    utf8_len,
)
from ancoda_tok.clean.snapshot import (
    SEED,
    build_index,
    fetch_texts,
    hashlib_int,
    load_index_rows,
    pool_sample_index,
    scale_shares,
    write_snapshot,
)
from ancoda_tok.counts import iter_entries, write_counts
from ancoda_tok.export_hf import load_hf
from ancoda_tok.frontier_common import (
    combine,
    enrich_from_harness,
    guarantee,
    held_out_code_math,
    is_deva_bytes,
    run_harness,
    tokens_per_kb,
    train_sides,
    write_side_config,
)
from ancoda_tok.frontier_metrics import attach_extra_metrics, fertility_on_texts
from ancoda_tok.inventory import INVENTORY_DIR, select_block1
from ancoda_tok.mix_sweep import _caffeinate
from ancoda_tok.storage import REPO_ROOT

OUT = REPO_ROOT / "artifacts" / "eval" / "mix_sweep" / "combined_candidates"
MIX_SWEEP = REPO_ROOT / "artifacts" / "eval" / "mix_sweep"
FLORES = (
    REPO_ROOT / "third_party" / "yak-tokenizer" / "data" / "eval" / "flores_plus" / "devtest"
)
YAK = (
    REPO_ROOT
    / "data/hf_cache/hub/models--ampixa--yak-tokenizer/snapshots"
    / "12b6d3010ec542b484ec3b673354852fe8bb85fb"
)
SID = (
    REPO_ROOT
    / "data/hf_cache/hub/models--sidskarki--nepali-bpe-tokenizer/snapshots"
    / "fc5773dcf7b9dc2272e8f595f99828969621a855/nepali_bpe_64k.model"
)
BLOCK0 = 133
RSS_CAP = 12 * 1024**3
BOOT_N = 10_000
BOOT_SEED = 20261104
N60_5GB = REPO_ROOT / "data" / "snapshots" / "n60_cf8066dbbdd0da93"

# Grid
B64 = (48000, 50000)
B128 = (94742, 100742)
CAPS = ("3pct", "2pct")
SCALES = (0.5, 0.25)
CAP_N = {
    "64k": {"3pct": 1966, "2pct": 1310},
    "128k": {"3pct": 3932, "2pct": 2621},
}


def _flores_lines(lang: str) -> list[str]:
    return [
        ln
        for ln in (FLORES / f"{lang}.txt").read_text(encoding="utf-8").splitlines()
        if ln.strip()
    ]


def _line_tokens_words(encode, lines: list[str]) -> list[tuple[int, int]]:
    out = []
    for ln in lines:
        w = len(ln.split())
        if w == 0:
            continue
        out.append((len(encode(ln)), w))
    return out


def paired_bootstrap_diff(
    cand_pairs: list[tuple[int, int]],
    base_pairs: list[tuple[int, int]],
    *,
    n: int = BOOT_N,
    seed: int = BOOT_SEED,
) -> dict:
    """95% CI of fertility(cand) − fertility(base) via paired line resample."""
    assert len(cand_pairs) == len(base_pairs)
    m = len(cand_pairs)
    if m == 0:
        return {"diff": None, "lo": None, "hi": None, "n_lines": 0}
    ct = sum(t for t, _ in cand_pairs)
    cw = sum(w for _, w in cand_pairs)
    bt = sum(t for t, _ in base_pairs)
    bw = sum(w for _, w in base_pairs)
    point = (ct / cw) - (bt / bw)
    rng = random.Random(seed)
    diffs = []
    for _ in range(n):
        ctok = cword = btok = bword = 0
        for _draw in range(m):
            i = rng.randrange(m)
            ct_i, cw_i = cand_pairs[i]
            bt_i, bw_i = base_pairs[i]
            ctok += ct_i
            cword += cw_i
            btok += bt_i
            bword += bw_i
        diffs.append((ctok / cword) - (btok / bword))
    diffs.sort()
    lo = diffs[int(0.025 * (n - 1))]
    hi = diffs[min(n - 1, int(0.975 * (n - 1)))]
    return {"diff": point, "lo": lo, "hi": hi, "n_lines": m, "n_resamples": n, "seed": seed}


def write_covered(vocab: str, pct: str, cap: int, dest: Path) -> Path:
    if dest.is_file():
        return dest
    letters = frozenset(int(x) for x in ancoda_seg.devanagari_letters())
    tsv = INVENTORY_DIR / "akshara_counts.tsv"
    counts: Counter[str] = Counter()
    with tsv.open(encoding="utf-8") as handle:
        handle.readline()
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 3:
                continue
            ak, _cps, count_s = parts
            ak = ak.replace("\\n", "\n").replace("\\t", "\t").replace("\\\\", "\\")
            counts[ak] += int(count_s)
    selected = select_block1(counts, cap=cap, letters=letters)
    payload = {
        "vocab_name": f"{vocab}_{pct}",
        "vocab_size": 65536 if vocab == "64k" else 131072,
        "gb9c": False,
        "diagnostic": True,
        "cap_pct": pct,
        "cap": cap,
        "seed": SEED,
        "counts_tsv": str(tsv.relative_to(REPO_ROOT)),
        "counts_tsv_sha256": sha256_file(tsv),
        **{
            k: selected[k]
            for k in (
                "N",
                "N0",
                "ineligible_occurrences",
                "eligible_multi_types",
                "unit_types_selected",
                "block1_merge_count",
                "coverage",
                "uncovered_occurrences",
                "stop_reason",
                "block1_merges",
                "units_selected",
                "coverage_curve",
                "top_50_uncovered_aksharas",
            )
        },
    }
    dest.parent.mkdir(parents=True, exist_ok=True)
    save_json(dest, payload)
    print(
        f"wrote {dest} merges={selected['block1_merge_count']} "
        f"coverage={selected['coverage']:.10f}",
        flush=True,
    )
    return dest


def build_5gb_n60_joiner() -> Path:
    name = "n60j5"
    existing = list((REPO_ROOT / "data" / "snapshots").glob(f"{name}_*"))
    if len(existing) == 1 and (existing[0] / "manifest.json").is_file():
        print(f"resume snapshot {existing[0]}", flush=True)
        return existing[0]
    budget = 5_000_000_000
    free = free_bytes()
    if free - budget < 50_000_000_000:
        raise Stop(
            f"STOP: fewer than 50 GB would remain after 5GB write (free={free})"
        )
    mix = load_mix()
    temperature = float(mix["temperature"])
    joiner_share = 0.01
    nepali_share = 0.60
    index_path = build_index(mix)
    index_rows = load_index_rows(index_path)
    by_bucket: dict[str, list] = defaultdict(list)
    for row in index_rows:
        by_bucket[row["bucket"]].append(row)
    shares = scale_shares(mix, nepali_share)
    natural_nepali = shares["Nepali"] - joiner_share
    targets = {b: int(budget * s) for b, s in shares.items()}
    targets["Nepali"] = int(budget * natural_nepali)
    selected_index = []
    used_ids: set[str] = set()
    for bucket, target in targets.items():
        picked = pool_sample_index(
            by_bucket.get(bucket, []),
            target,
            seed=SEED + int(nepali_share * 1000) + hashlib_int(bucket) + 20_000,
            temperature=temperature,
        )
        for p in picked:
            used_ids.add(p["doc_id"])
        selected_index.extend(picked)
    chosen = fetch_texts(selected_index)
    unused = [d for d in by_bucket.get("Nepali", []) if d["doc_id"] not in used_ids]
    joiner_target = int(budget * joiner_share)
    unused_sorted = sorted(unused, key=lambda d: d["doc_id"])
    rng = random.Random(SEED)
    rng.shuffle(unused_sorted)
    pool_rows = []
    pool_bytes = 0
    for item in unused_sorted:
        pool_rows.append(item)
        pool_bytes += item["text_bytes"]
        if pool_bytes >= joiner_target * 3:
            break
    unused_docs = fetch_texts(pool_rows)
    joiner_dir = REPO_ROOT / "data" / "clean" / f"_synthetic_joiners_{name}"
    build_joiner_copy(
        unused_docs, target_bytes=joiner_target, seed=SEED, out_dir=joiner_dir
    )
    for row in iter_parquet_rows(joiner_dir):
        chosen.append(
            {
                "source": row["source"],
                "doc_id": row["doc_id"],
                "license": row["license"],
                "bucket": "Nepali",
                "text_bytes": utf8_len(row["text"]),
                "text": row["text"],
                "glotlid_label": "synthetic",
                "glotlid_prob": 1.0,
                "synthetic": True,
            }
        )
    mix_meta = {
        "nepali_share": nepali_share,
        "shares": shares,
        "joiner_share": joiner_share,
        "budget": budget,
        "diagnostic": "combined_5gb_n60_joiner",
        "synthetic": True,
    }
    snap = write_snapshot(name, chosen, mix_meta=mix_meta)
    counts = audit_snapshot(snap)
    print(f"audit {name} {counts}", flush=True)
    if any(v != 0 for v in counts.values()):
        raise Stop(f"STOP: snapshot {name} failed decontam audit: {counts}")
    return snap


def choose_snapshot() -> dict:
    """Apply ΔNepali rule vs 5GB n60 frontier; return choice metadata + path."""
    t3_path = MIX_SWEEP / "t3_10gb_joiner" / "frontier.json"
    h = json.loads((MIX_SWEEP / "script_budget" / "frontier.json").read_text(encoding="utf-8"))
    h_by = {int(r["B_d"]): r for r in h["rows"]}
    comparisons = []
    if t3_path.is_file():
        t3 = json.loads(t3_path.read_text(encoding="utf-8"))
        for b in (44000, 46000, 48000):
            t3r = next((r for r in t3["rows"] if r.get("B_d") == b), None)
            h5 = h_by.get(b)
            if not t3r or t3r.get("npi") is None or not h5 or h5.get("npi") is None:
                comparisons.append({"B_d": b, "status": "missing"})
                continue
            d_npi = float(t3r["npi"]) - float(h5["npi"])
            d_eng = float(t3r["eng"]) - float(h5["eng"])
            comparisons.append(
                {
                    "B_d": b,
                    "status": "ok",
                    "npi_t3": float(t3r["npi"]),
                    "npi_5gb": float(h5["npi"]),
                    "eng_t3": float(t3r["eng"]),
                    "eng_5gb": float(h5["eng"]),
                    "delta_npi": d_npi,
                    "delta_eng": d_eng,
                    "npi_helps": d_npi < 0,
                    "eng_ok": d_eng <= 0.005,
                }
            )
    else:
        comparisons = [{"status": "t3_incomplete"}]

    helping = [c for c in comparisons if c.get("npi_helps") and c.get("eng_ok")]
    use_10gb = len(helping) >= 2
    reason_parts = []
    for c in comparisons:
        if c.get("status") == "ok":
            reason_parts.append(
                f"B_d={c['B_d']}: Δnpi={c['delta_npi']:+.6f} Δeng={c['delta_eng']:+.6f}"
            )
        else:
            reason_parts.append(f"B_d={c.get('B_d')}: {c.get('status')}")
    if use_10gb:
        snaps = list((REPO_ROOT / "data" / "snapshots").glob("n60j10_*"))
        if len(snaps) != 1:
            raise SystemExit("STOP: expected one n60j10_* snapshot")
        chosen = snaps[0]
        label = "10gb_joiner"
        reason = (
            f"chose 10GB+joiner: ΔNepali<0 at {len(helping)}/3 points with eng not "
            f">+0.005. " + "; ".join(reason_parts)
        )
    else:
        chosen = build_5gb_n60_joiner()
        label = "5gb_n60_joiner"
        reason = (
            "chose new 5GB n60+joiner: need ΔNepali<0 at ≥2 points with eng slack, "
            "or T3 incomplete/missing. " + "; ".join(reason_parts)
        )
    meta = {
        "chosen_label": label,
        "chosen_path": str(chosen.relative_to(REPO_ROOT)),
        "reason": reason,
        "comparisons": comparisons,
        "helping_points": len(helping),
    }
    print(json.dumps(meta, indent=2), flush=True)
    return meta | {"path": chosen}


def count_and_split(snap: Path) -> tuple[Path, Path, Path]:
    counts_dir = OUT / "counts"
    counts_dir.mkdir(parents=True, exist_ok=True)
    bin_path = counts_dir / "pretoken_counts.bin"
    if not bin_path.is_file():
        _caffeinate(
            [
                "uv",
                "run",
                "python",
                "-u",
                "-m",
                "ancoda_tok.counts",
                "--snapshot",
                str(snap),
                "--out",
                str(counts_dir),
            ]
        )
    dewa = OUT / "deva_pretoken_counts.bin"
    latin = OUT / "latin_pretoken_counts.bin"
    if not (dewa.is_file() and latin.is_file()):
        d_list: list[tuple[bytes, int]] = []
        l_list: list[tuple[bytes, int]] = []
        for raw, count in iter_entries(bin_path):
            if is_deva_bytes(raw):
                d_list.append((raw, count))
            else:
                l_list.append((raw, count))
        write_counts(dewa, d_list)
        write_counts(latin, l_list)
        print(f"split dewa={len(d_list)} latin={len(l_list)}", flush=True)
    return bin_path, dewa, latin


def count_bucket(snap: Path, bucket: str) -> Path:
    dest = OUT / f"counts_{bucket.lower()}"
    if (dest / "pretoken_counts.bin").is_file():
        return dest / "pretoken_counts.bin"
    dest.mkdir(parents=True, exist_ok=True)
    _caffeinate(
        [
            "uv",
            "run",
            "python",
            "-u",
            "-m",
            "ancoda_tok.counts",
            "--snapshot",
            str(snap),
            "--out",
            str(dest),
            "--bucket",
            bucket,
        ]
    )
    return dest / "pretoken_counts.bin"


def build_scaled_latin(scale: float) -> Path:
    tag = f"latin_cm{int(scale * 100)}"
    dest = OUT / f"{tag}_pretoken_counts.bin"
    if dest.is_file():
        return dest
    eng = dict(iter_entries(OUT / "counts_english" / "pretoken_counts.bin"))
    code = dict(iter_entries(OUT / "counts_code" / "pretoken_counts.bin"))
    math = dict(iter_entries(OUT / "counts_math" / "pretoken_counts.bin"))
    merged: dict[bytes, int] = {}
    for raw, c in eng.items():
        merged[raw] = merged.get(raw, 0) + c
    for src in (code, math):
        for raw, c in src.items():
            add = int(c * scale)
            if add > 0:
                merged[raw] = merged.get(raw, 0) + add
    entries = sorted(merged.items(), key=lambda kv: kv[0])
    write_counts(dest, entries)
    print(f"wrote {dest} unique={len(entries)} scale={scale}", flush=True)
    return dest


def train_one(cfg: Path, out: Path) -> int:
    if (out / "ranks.json").is_file():
        stats = json.loads((out / "ranks.json").read_text(encoding="utf-8")).get(
            "stats", {}
        )
        return int(stats.get("peak_rss_bytes") or 0)
    _caffeinate(
        [
            "cargo",
            "run",
            "--release",
            "--offline",
            "-p",
            "ancoda-bpe",
            "--",
            "train",
            "--config",
            str(cfg),
        ]
    )
    stats = json.loads((out / "ranks.json").read_text(encoding="utf-8")).get("stats", {})
    return int(stats.get("peak_rss_bytes") or 0)


def mark_initial_count(tok_dir: Path) -> int:
    from ancoda_tok.guarantee import _marks

    ranks = json.loads((tok_dir / "ranks.json").read_text(encoding="utf-8"))
    tokens = [bytes.fromhex(x) for x in ranks["tokens_hex"]]
    layout = ranks["layout"]
    marks = _marks()
    bad = 0
    for ident in range(layout["block0_end"], layout["reserved_start"]):
        raw = tokens[ident]
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            continue
        if text and (ord(text[0]) in marks or ord(text[0]) in (0x200C, 0x200D)):
            bad += 1
    return bad


def encode_fn_for(path: Path):
    if path.suffix == ".model" or path.name.endswith(".model"):
        sp = spm.SentencePieceProcessor(model_file=str(path))

        def enc(text: str) -> list[int]:
            return sp.encode(text, out_type=int)

        return enc
    tok = load_hf(path if path.name == "tokenizer.json" else path / "tokenizer.json")
    if hasattr(tok, "encode_special_tokens"):
        tok.encode_special_tokens = True

    def enc(text: str) -> list[int]:
        return tok.encode(text, add_special_tokens=False).ids

    return enc


def competitor_encode_specs() -> dict[str, Path]:
    return {
        "yak_v1_64k": YAK / "v1.0/64k/tokenizer.json",
        "yak_v2_64k": YAK / "v2.0/64k/tokenizer.json",
        "sidskarki_64k": SID,
        "yak_v1_128k": YAK / "v1.0/128k/tokenizer.json",
        "yak_v2_128k": YAK / "v2.0/128k/tokenizer.json",
    }


def run_point(
    *,
    vocab: str,
    b_d: int,
    pct: str,
    scale: float,
    mergeable: int,
    covered: Path,
    dewa_counts_rel: str,
    snap: Path,
    snap_label: str,
    fallback_5gb_joiner: Path | None,
    code_texts: list[str],
    math_texts: list[str],
    skip_guarantee: bool,
) -> dict:
    tag = f"{vocab}_bd{b_d}_{pct}_cm{int(scale * 100)}"
    print(f"=== {tag} ===", flush=True)
    block1 = int(json.loads(covered.read_text(encoding="utf-8"))["block1_merge_count"])
    coverage = json.loads(covered.read_text(encoding="utf-8"))["coverage"]
    latin_counts = build_scaled_latin(scale)
    used_snap_label = snap_label
    used_snap = snap

    def _train(counts_deva: str, counts_latin: str, out_root: Path) -> tuple[Path, Path, int]:
        dewa_dir, latin_dir = train_sides(
            out_root=out_root,
            tag=tag,
            b_d=b_d,
            mergeable=mergeable,
            block0=BLOCK0,
            block1=block1,
            block1_path=str(covered.relative_to(REPO_ROOT)),
            dewa_counts_rel=counts_deva,
            latin_counts_rel=counts_latin,
        )
        # Override latin with scaled counts (train_sides used latin_counts_rel already).
        peak = 0
        for side in (dewa_dir, latin_dir):
            stats = json.loads((side / "ranks.json").read_text(encoding="utf-8")).get(
                "stats", {}
            )
            peak = max(peak, int(stats.get("peak_rss_bytes") or 0))
        return dewa_dir, latin_dir, peak

    # First attempt on chosen snapshot counts.
    # Re-train latin side explicitly with scaled file (deva from train_sides).
    dewa_tag_root = OUT
    latin_b2 = mergeable - 256 - b_d
    dewa_vocab = 256 + b_d + 1024
    latin_vocab = 256 + latin_b2 + 1024
    dewa_out = OUT / tag / "side-deva"
    latin_out = OUT / tag / "side-latin"
    dewa_cfg = write_side_config(
        path=OUT / "configs" / f"{tag}_deva.yaml",
        counts_rel=dewa_counts_rel,
        output_rel=str(dewa_out.relative_to(REPO_ROOT)),
        vocab_size=dewa_vocab,
        block0=True,
        block1=True,
        block1_path=str(covered.relative_to(REPO_ROOT)),
    )
    latin_cfg = write_side_config(
        path=OUT / "configs" / f"{tag}_latin.yaml",
        counts_rel=str(latin_counts.relative_to(REPO_ROOT)),
        output_rel=str(latin_out.relative_to(REPO_ROOT)),
        vocab_size=latin_vocab,
        block0=False,
        block1=False,
        block1_path=str(covered.relative_to(REPO_ROOT)),
    )
    peak = max(train_one(dewa_cfg, dewa_out), train_one(latin_cfg, latin_out))
    if peak > RSS_CAP and vocab == "128k" and fallback_5gb_joiner is not None:
        print(
            f"RSS {peak} > {RSS_CAP}; rerunning {tag} on 5GB n60+joiner",
            flush=True,
        )
        # Fall back: need counts from 5gb joiner — expect prepared under OUT/fallback_*
        fb = OUT / "fallback_5gb"
        used_snap = fallback_5gb_joiner
        used_snap_label = "5gb_n60_joiner_rss_fallback"
        # Clear sides for retrain
        shutil.rmtree(dewa_out, ignore_errors=True)
        shutil.rmtree(latin_out, ignore_errors=True)
        dewa_cfg = write_side_config(
            path=OUT / "configs" / f"{tag}_deva_fb.yaml",
            counts_rel=str((fb / "deva_pretoken_counts.bin").relative_to(REPO_ROOT)),
            output_rel=str(dewa_out.relative_to(REPO_ROOT)),
            vocab_size=dewa_vocab,
            block0=True,
            block1=True,
            block1_path=str(covered.relative_to(REPO_ROOT)),
        )
        latin_cfg = write_side_config(
            path=OUT / "configs" / f"{tag}_latin_fb.yaml",
            counts_rel=str(
                (fb / f"latin_cm{int(scale * 100)}_pretoken_counts.bin").relative_to(
                    REPO_ROOT
                )
            ),
            output_rel=str(latin_out.relative_to(REPO_ROOT)),
            vocab_size=latin_vocab,
            block0=False,
            block1=False,
            block1_path=str(covered.relative_to(REPO_ROOT)),
        )
        peak = max(train_one(dewa_cfg, dewa_out), train_one(latin_cfg, latin_out))
        if peak > RSS_CAP:
            return {
                "arm": "combined",
                "label": tag,
                "stopped": "rss_cap",
                "peak_rss_bytes": peak,
                "snapshot": used_snap_label,
            }

    dest = OUT / tag / "combined"
    combine(dewa_out, latin_out, dest, mergeable=mergeable)
    if not skip_guarantee:
        # Prefix = name before digest suffix (n60j10 / n60j5 / n60).
        prefix = used_snap.name.rsplit("_", 1)[0]
        g = guarantee(
            dest,
            report=OUT / f"guarantee_{tag}.json",
            orphans=OUT / f"orphans_{tag}.txt",
            covered=covered,
            snapshot_prefix=prefix,
            snapshot_line_limit=1_000_000,
        )
        # Prefer exact prefix: n60j10 / n60j5 / n60
        viol = g.get("hf_covered_violations")
        fails = g.get("failures")
        tail = g.get("tail_hf_rate_lo_hi")
        orphan_occ = g.get("orphan_occurrences_npi_aya")
        snap_lines = g.get("snapshot_lines")
    else:
        viol = fails = tail = orphan_occ = snap_lines = None
    row = {
        "arm": "combined",
        "label": tag,
        "vocab": vocab,
        "B_d": b_d,
        "block1_cap": pct,
        "block1": block1,
        "block1_coverage": coverage,
        "code_math_scale": scale,
        "dir": str(dest.relative_to(REPO_ROOT)),
        "snapshot": used_snap_label,
        "peak_rss_bytes": peak,
        "hf_covered_violations": viol,
        "guarantee_failures": fails,
        "tail_hf_rate_lo_hi": tail,
        "orphan_occurrences_npi_aya": orphan_occ,
        "guarantee_snapshot_lines": snap_lines,
        "mark_initial": mark_initial_count(dest),
        "code_tokens_per_kb": tokens_per_kb(dest, code_texts),
        "math_tokens_per_kb": tokens_per_kb(dest, math_texts),
    }
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-guarantee", action="store_true")
    parser.add_argument(
        "--choice-only",
        action="store_true",
        help="Only apply snapshot choice rule and exit",
    )
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    # Ensure T3 frontier exists for choice rule (Prompt 1).
    if not (MIX_SWEEP / "t3_10gb_joiner" / "frontier.json").is_file():
        raise SystemExit(
            "STOP: Prompt 1 T3 frontier missing — finish diag_t3 first"
        )

    choice_path = OUT / "snapshot_choice.json"
    if choice_path.is_file() and not args.choice_only:
        choice = json.loads(choice_path.read_text(encoding="utf-8"))
        snap = REPO_ROOT / choice["chosen_path"]
        snap_label = choice["chosen_label"]
        print(f"resume snapshot choice: {snap_label} -> {snap}", flush=True)
    else:
        choice = choose_snapshot()
        save_json(
            choice_path,
            finish_manifest(
                base_manifest(
                    kind="combined_snapshot_choice",
                    **{
                        k: choice[k]
                        for k in (
                            "chosen_label",
                            "chosen_path",
                            "reason",
                            "comparisons",
                            "helping_points",
                        )
                    },
                )
            ),
        )
        if args.choice_only:
            return
        snap = choice["path"]
        snap_label = choice["chosen_label"]
    count_and_split(snap)
    print("counting English/Code/Math on chosen snapshot…", flush=True)
    count_bucket(snap, "English")
    count_bucket(snap, "Code")
    count_bucket(snap, "Math")
    for scale in SCALES:
        build_scaled_latin(scale)

    # Fallback 5GB joiner counts if we chose 10GB (for RSS overflow on 128k).
    fallback_snap = None
    if snap_label.startswith("10gb"):
        fallback_snap = build_5gb_n60_joiner()
        fb = OUT / "fallback_5gb"
        fb.mkdir(parents=True, exist_ok=True)
        if not (fb / "pretoken_counts.bin").is_file():
            _caffeinate(
                [
                    "uv",
                    "run",
                    "python",
                    "-u",
                    "-m",
                    "ancoda_tok.counts",
                    "--snapshot",
                    str(fallback_snap),
                    "--out",
                    str(fb),
                ]
            )
        if not (fb / "deva_pretoken_counts.bin").is_file():
            d_list, l_list = [], []
            for raw, count in iter_entries(fb / "pretoken_counts.bin"):
                (d_list if is_deva_bytes(raw) else l_list).append((raw, count))
            write_counts(fb / "deva_pretoken_counts.bin", d_list)
            write_counts(fb / "latin_pretoken_counts.bin", l_list)
        # bucket counts for scaled latin on fallback
        for bucket in ("English", "Code", "Math"):
            dest = fb / f"counts_{bucket.lower()}"
            if not (dest / "pretoken_counts.bin").is_file():
                dest.mkdir(parents=True, exist_ok=True)
                _caffeinate(
                    [
                        "uv",
                        "run",
                        "python",
                        "-u",
                        "-m",
                        "ancoda_tok.counts",
                        "--snapshot",
                        str(fallback_snap),
                        "--out",
                        str(dest),
                        "--bucket",
                        bucket,
                    ]
                )
        for scale in SCALES:
            dest = fb / f"latin_cm{int(scale * 100)}_pretoken_counts.bin"
            if dest.is_file():
                continue
            eng = dict(iter_entries(fb / "counts_english" / "pretoken_counts.bin"))
            code = dict(iter_entries(fb / "counts_code" / "pretoken_counts.bin"))
            math = dict(iter_entries(fb / "counts_math" / "pretoken_counts.bin"))
            merged: dict[bytes, int] = dict(eng)
            for src in (code, math):
                for raw, c in src.items():
                    add = int(c * scale)
                    if add > 0:
                        merged[raw] = merged.get(raw, 0) + add
            write_counts(dest, sorted(merged.items(), key=lambda kv: kv[0]))

    # Covered files
    covered_64 = {}
    covered_128 = {}
    for pct in CAPS:
        covered_64[pct] = write_covered(
            "64k",
            pct,
            CAP_N["64k"][pct],
            OUT / f"covered_64k_{pct}.json",
        )
        # Prefer T1 copies if identical
        t1c = MIX_SWEEP / "t1_block1_cap" / f"covered_64k_{pct}.json"
        if t1c.is_file():
            covered_64[pct] = t1c
        covered_128[pct] = write_covered(
            "128k",
            pct,
            CAP_N["128k"][pct],
            OUT / f"covered_128k_{pct}.json",
        )

    code_texts, math_texts = held_out_code_math(snap)
    rows: list[dict] = []
    adds: dict[str, Path] = {}

    dewa_rel = str((OUT / "deva_pretoken_counts.bin").relative_to(REPO_ROOT))
    for b_d in B64:
        for pct in CAPS:
            for scale in SCALES:
                row = run_point(
                    vocab="64k",
                    b_d=b_d,
                    pct=pct,
                    scale=scale,
                    mergeable=64512,
                    covered=covered_64[pct],
                    dewa_counts_rel=dewa_rel,
                    snap=snap,
                    snap_label=snap_label,
                    fallback_5gb_joiner=None,
                    code_texts=code_texts,
                    math_texts=math_texts,
                    skip_guarantee=args.skip_guarantee,
                )
                rows.append(row)
                if "dir" in row:
                    adds[row["label"]] = REPO_ROOT / row["dir"]

    for b_d in B128:
        for pct in CAPS:
            for scale in SCALES:
                row = run_point(
                    vocab="128k",
                    b_d=b_d,
                    pct=pct,
                    scale=scale,
                    mergeable=130048,
                    covered=covered_128[pct],
                    dewa_counts_rel=dewa_rel,
                    snap=snap,
                    snap_label=snap_label,
                    fallback_5gb_joiner=fallback_snap,
                    code_texts=code_texts,
                    math_texts=math_texts,
                    skip_guarantee=args.skip_guarantee,
                )
                rows.append(row)
                if "dir" in row:
                    adds[row["label"]] = REPO_ROOT / row["dir"]

    if adds:
        run_harness(adds, "combined_cand")
        for row in rows:
            if "dir" not in row:
                continue
            enrich_from_harness(row, "combined_cand", row["label"])
            attach_extra_metrics(
                row,
                REPO_ROOT / row["dir"],
                snap=snap,
                harness_aya=row.get("aya_fertility"),
            )

    # Paired bootstrap CIs vs competitors
    npi_lines = _flores_lines("npi_Deva")
    eng_lines = _flores_lines("eng_Latn")
    hin_lines = _flores_lines("hin_Deva")
    comps = competitor_encode_specs()
    comp_pairs = {}
    for name, path in comps.items():
        enc = encode_fn_for(path)
        comp_pairs[name] = {
            "npi": _line_tokens_words(enc, npi_lines),
            "eng": _line_tokens_words(enc, eng_lines),
            "hin": _line_tokens_words(enc, hin_lines),
        }

    for row in rows:
        if "dir" not in row:
            continue
        enc = encode_fn_for(REPO_ROOT / row["dir"])
        cand = {
            "npi": _line_tokens_words(enc, npi_lines),
            "eng": _line_tokens_words(enc, eng_lines),
            "hin": _line_tokens_words(enc, hin_lines),
        }
        cis = {}
        if row["vocab"] == "64k":
            for base in ("yak_v1_64k", "yak_v2_64k", "sidskarki_64k"):
                cis[base] = {
                    lang: paired_bootstrap_diff(cand[lang], comp_pairs[base][lang])
                    for lang in ("npi", "eng", "hin")
                }
        else:
            for base in ("yak_v1_128k", "yak_v2_128k"):
                cis[base] = {
                    lang: paired_bootstrap_diff(cand[lang], comp_pairs[base][lang])
                    for lang in ("npi", "eng", "hin")
                }
        row["bootstrap_ci_vs"] = cis

    save_json(
        OUT / "frontier.json",
        finish_manifest(
            base_manifest(
                kind="combined_candidates",
                snapshot_choice=choice["reason"],
                snapshot=str(snap.relative_to(REPO_ROOT)),
                rows=rows,
            )
        ),
    )
    # Compact table
    print("COMBINED_CANDIDATES_TABLE", flush=True)
    print(
        "label | npi | eng | hin | aya | fw2 | DevaMean | code/KB | math/KB | "
        "B1cov | tail | zwj | zwnj | morphF1 | whole | viol | orphan | markInit | snap",
        flush=True,
    )
    for r in rows:
        if "npi" not in r:
            print(f"{r.get('label')} STOPPED {r.get('stopped')}", flush=True)
            continue
        tail = r.get("tail_hf_rate_lo_hi") or [None]
        print(
            f"{r['label']} | {r['npi']:.6f} | {r['eng']:.6f} | {r['hin']:.6f} | "
            f"{(r.get('aya_fertility') or 0):.6f} | {(r.get('fw2_npi_test_fertility') or 0):.6f} | "
            f"{(r.get('flores_deva_mean') or 0):.6f} | {r.get('code_tokens_per_kb'):.3f} | "
            f"{r.get('math_tokens_per_kb'):.3f} | {r.get('block1_coverage'):.10f} | "
            f"{tail[0]} | {r.get('zwj')} | {r.get('zwnj')} | "
            f"{r.get('morph_f1_unweighted')} | {r.get('whole_word_unweighted')} | "
            f"{r.get('hf_covered_violations')} | {r.get('orphan')} | "
            f"{r.get('mark_initial')} | {r.get('snapshot')}",
            flush=True,
        )


if __name__ == "__main__":
    main()

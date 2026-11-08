"""M4 steps 7–8: mix sampling, four snapshots, independent decontam audit.

Sampling is two-pass and does not hold corpus text in RAM: an index of
(doc_id, bytes, shard, row) is shuffled with the fixed seed, then only the
selected documents are re-read into the snapshot.
"""

from __future__ import annotations

import hashlib
import random
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from ancoda_tok.clean.augment_joiners import build_joiner_copy
from ancoda_tok.clean.common import (
    CLEAN_SCHEMA,
    REPO_ROOT,
    SEED,
    ShardWriter,
    Stop,
    artifacts_clean,
    base_manifest,
    clean_dir,
    ensure_room,
    finish_manifest,
    free_bytes,
    iter_parquet_rows,
    list_shards,
    load_json,
    load_mix,
    load_sources,
    save_json,
    sha256_file,
    utf8_len,
)
from ancoda_tok.clean.decontam import audit_snapshot

DOC_IDS_SCHEMA = pa.schema(
    [
        ("source", pa.string()),
        ("doc_id", pa.string()),
        ("license", pa.string()),
        ("bucket", pa.string()),
        ("synthetic", pa.bool_()),
    ]
)

INDEX_SCHEMA = pa.schema(
    [
        ("bucket", pa.string()),
        ("lang", pa.string()),
        ("source", pa.string()),
        ("doc_id", pa.string()),
        ("license", pa.string()),
        ("text_bytes", pa.int64()),
        ("shard", pa.string()),
        ("row", pa.int64()),
    ]
)


def snapshot_budget_bytes() -> int:
    return int(load_sources()["snapshot_bytes"])


def hashlib_int(text: str) -> int:
    return int(hashlib.sha256(text.encode()).hexdigest()[:8], 16)


def build_index(mix: dict) -> Path:
    """Write a compact doc index for all clean training docs + reroutes."""
    out = REPO_ROOT / "data" / "clean" / "_snapshot_index.parquet"
    rows: list[dict[str, Any]] = []
    for bucket, bspec in mix["buckets"].items():
        for source_id in bspec["sources"]:
            directory = clean_dir(source_id)
            if not directory.exists():
                continue
            for shard in list_shards(directory):
                table = pq.read_table(
                    shard,
                    columns=["doc_id", "text", "source", "license", "glotlid_label"],
                )
                rel = str(shard.relative_to(REPO_ROOT))
                for i in range(table.num_rows):
                    text = table.column("text")[i].as_py()
                    label = table.column("glotlid_label")[i].as_py() or source_id
                    rows.append(
                        {
                            "bucket": bucket,
                            "lang": label if bucket in {
                                "Smaller Devanagari",
                                "Languages of Nepal",
                            } else bucket,
                            "source": table.column("source")[i].as_py(),
                            "doc_id": table.column("doc_id")[i].as_py(),
                            "license": table.column("license")[i].as_py(),
                            "text_bytes": utf8_len(text),
                            "shard": rel,
                            "row": i,
                        }
                    )
        reroute = REPO_ROOT / "data" / "clean" / "_reroute" / bucket
        if reroute.exists():
            for src_dir in sorted(p for p in reroute.iterdir() if p.is_dir()):
                for shard in list_shards(src_dir):
                    table = pq.read_table(shard)
                    rel = str(shard.relative_to(REPO_ROOT))
                    for i in range(table.num_rows):
                        text = table.column("text")[i].as_py()
                        label = table.column("glotlid_label")[i].as_py() or "reroute"
                        rows.append(
                            {
                                "bucket": bucket,
                                "lang": label,
                                "source": table.column("source")[i].as_py(),
                                "doc_id": table.column("doc_id")[i].as_py(),
                                "license": table.column("license")[i].as_py(),
                                "text_bytes": utf8_len(text),
                                "shard": rel,
                                "row": i,
                            }
                        )
        print(f"index bucket={bucket} rows_so_far={len(rows)}", flush=True)
    table = pa.Table.from_pylist(rows, schema=INDEX_SCHEMA)
    pq.write_table(table, out, compression="zstd")
    print(f"wrote snapshot index docs={len(rows)} path={out}", flush=True)
    return out


def load_index_rows(index_path: Path) -> list[dict[str, Any]]:
    table = pq.read_table(index_path)
    cols = {name: table.column(name) for name in table.column_names}
    rows = []
    for i in range(table.num_rows):
        rows.append({name: cols[name][i].as_py() for name in cols})
    return rows


def pool_sample_index(
    items: list[dict[str, Any]],
    target_bytes: int,
    *,
    seed: int,
    temperature: float,
) -> list[dict[str, Any]]:
    if target_bytes <= 0 or not items:
        return []
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for it in items:
        groups[it.get("lang") or "all"].append(it)
    avail = {lang: sum(x["text_bytes"] for x in docs) for lang, docs in groups.items()}
    if len(groups) == 1:
        shares = {next(iter(groups)): 1.0}
    else:
        weights = {lang: (max(b, 1) ** temperature) for lang, b in avail.items()}
        total_w = sum(weights.values())
        shares = {lang: w / total_w for lang, w in weights.items()}

    rng = random.Random(seed)
    chosen: list[dict[str, Any]] = []
    used = 0
    remaining_langs = set(groups)
    targets = {lang: int(target_bytes * shares[lang]) for lang in groups}
    # Working copies so we can remove taken docs.
    pools = {lang: list(docs) for lang, docs in groups.items()}

    while remaining_langs and used < target_bytes:
        progressed = False
        for lang in sorted(remaining_langs):
            need = min(targets[lang], target_bytes - used)
            if need <= 0:
                remaining_langs.discard(lang)
                continue
            docs = pools[lang]
            order = list(range(len(docs)))
            rng.shuffle(order)
            taken_bytes = 0
            take_idx: list[int] = []
            for i in order:
                if taken_bytes >= need:
                    break
                take_idx.append(i)
                taken_bytes += docs[i]["text_bytes"]
            take_set = set(take_idx)
            for i in sorted(take_idx):
                chosen.append(docs[i])
                used += docs[i]["text_bytes"]
            pools[lang] = [docs[i] for i in range(len(docs)) if i not in take_set]
            if not pools[lang] or taken_bytes < need:
                remaining_langs.discard(lang)
                leftover = max(0, need - taken_bytes)
                if leftover and remaining_langs:
                    extra = leftover // len(remaining_langs)
                    for other in remaining_langs:
                        targets[other] += extra
            else:
                targets[lang] = max(0, targets[lang] - taken_bytes)
            progressed = True
        if not progressed:
            break
    return chosen


def scale_shares(mix: dict, nepali_share: float) -> dict[str, float]:
    base = {b: float(s["share"]) for b, s in mix["buckets"].items()}
    others = {b: s for b, s in base.items() if b != "Nepali"}
    other_total = sum(others.values())
    remain = 1.0 - nepali_share
    out = {"Nepali": nepali_share}
    for b, s in others.items():
        out[b] = remain * (s / other_total) if other_total else 0.0
    return out


def fetch_texts(selected: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Re-read full text for selected index rows, grouped by shard."""
    by_shard: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in selected:
        by_shard[item["shard"]].append(item)
    out: list[dict[str, Any]] = []
    for shard_rel, items in sorted(by_shard.items()):
        path = REPO_ROOT / shard_rel
        table = pq.read_table(path)
        for item in items:
            i = int(item["row"])
            out.append(
                {
                    "source": item["source"],
                    "doc_id": item["doc_id"],
                    "license": item["license"],
                    "bucket": item["bucket"],
                    "text_bytes": item["text_bytes"],
                    "text": table.column("text")[i].as_py(),
                    "glotlid_label": (
                        table.column("glotlid_label")[i].as_py()
                        if "glotlid_label" in table.column_names
                        else ""
                    ),
                    "glotlid_prob": (
                        float(table.column("glotlid_prob")[i].as_py() or 0.0)
                        if "glotlid_prob" in table.column_names
                        else 0.0
                    ),
                }
            )
    return out


def write_snapshot(
    name: str,
    docs: list[dict[str, Any]],
    *,
    mix_meta: dict[str, Any],
) -> Path:
    payload = sum(d["text_bytes"] for d in docs)
    ensure_room(payload + 64_000_000, label=f"snapshot/{name}")
    digest = hashlib.sha256(
        f"{name}:{sorted(mix_meta.items())}:{SEED}:{payload}:{len(docs)}".encode()
    ).hexdigest()[:16]
    out = REPO_ROOT / "data" / "snapshots" / f"{name}_{digest}"
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    writer = ShardWriter(out, schema=CLEAN_SCHEMA)
    doc_rows: list[dict[str, Any]] = []
    per_source: dict[str, dict[str, int]] = defaultdict(lambda: {"docs": 0, "bytes": 0})
    per_bucket: dict[str, dict[str, int]] = defaultdict(lambda: {"docs": 0, "bytes": 0})

    docs_sorted = sorted(docs, key=lambda d: (d["bucket"], d["source"], d["doc_id"]))
    for d in docs_sorted:
        synthetic = bool(d.get("synthetic")) or str(d["doc_id"]).startswith("synthetic\x1f")
        writer.add(
            {
                "doc_id": d["doc_id"],
                "text": d["text"],
                "source": d["source"],
                "license": d["license"],
                "bucket": d["bucket"],
                "glotlid_label": d.get("glotlid_label", ""),
                "glotlid_prob": float(d.get("glotlid_prob") or 0.0),
            }
        )
        doc_rows.append(
            {
                "source": d["source"],
                "doc_id": d["doc_id"],
                "license": d["license"],
                "bucket": d["bucket"],
                "synthetic": synthetic,
            }
        )
        per_source[d["source"]]["docs"] += 1
        per_source[d["source"]]["bytes"] += d["text_bytes"]
        per_bucket[d["bucket"]]["docs"] += 1
        per_bucket[d["bucket"]]["bytes"] += d["text_bytes"]
    writer.close()

    ids_path = out / "doc_ids.parquet"
    pq.write_table(
        pa.Table.from_pylist(doc_rows, schema=DOC_IDS_SCHEMA),
        ids_path,
        compression="zstd",
    )
    manifest = finish_manifest(
        {
            **base_manifest(stage="snapshot", snapshot_name=name),
            "snapshot_name": name,
            "docs": writer.docs,
            "text_bytes": writer.text_bytes,
            "file_bytes": sum(s["file_bytes"] for s in writer.shards) + ids_path.stat().st_size,
            "shards": writer.shards,
            "doc_ids_sha256": sha256_file(ids_path),
            "mix": mix_meta,
            "per_source": dict(per_source),
            "per_bucket": dict(per_bucket),
            "seed": SEED,
        }
    )
    save_json(out / "manifest.json", manifest)
    print(
        f"snapshot {name} dir={out.name} docs={writer.docs} "
        f"text_bytes={writer.text_bytes} free_bytes={free_bytes()}",
        flush=True,
    )
    return out


def build_snapshots() -> dict[str, Any]:
    mix = load_mix()
    budget = snapshot_budget_bytes()
    temperature = float(mix["temperature"])
    joiner_share = float(mix["joiner_share"])

    index_path = build_index(mix)
    index_rows = load_index_rows(index_path)
    by_bucket: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in index_rows:
        by_bucket[row["bucket"]].append(row)

    results: dict[str, Any] = {"snapshots": {}, "audits": {}}

    for nepali_share in mix["nepali_sweep"]:
        shares = scale_shares(mix, float(nepali_share))
        with_joiners = abs(float(nepali_share) - 0.50) < 1e-9
        natural_nepali_share = (
            shares["Nepali"] - joiner_share if with_joiners else shares["Nepali"]
        )
        targets = {b: int(budget * s) for b, s in shares.items()}
        targets["Nepali"] = int(budget * natural_nepali_share)

        selected_index: list[dict[str, Any]] = []
        used_ids: set[str] = set()
        for bucket, target in targets.items():
            picked = pool_sample_index(
                by_bucket.get(bucket, []),
                target,
                seed=SEED + int(float(nepali_share) * 1000) + hashlib_int(bucket),
                temperature=temperature,
            )
            for p in picked:
                used_ids.add(p["doc_id"])
            selected_index.extend(picked)

        chosen = fetch_texts(selected_index)
        name = f"n{int(float(nepali_share) * 100)}"

        if with_joiners:
            unused = [d for d in by_bucket.get("Nepali", []) if d["doc_id"] not in used_ids]
            # Fetch a pool of unused Nepali docs for joiner synthesis (cap by 3× target).
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
            joiner_manifest = build_joiner_copy(
                unused_docs,
                target_bytes=joiner_target,
                seed=SEED,
                out_dir=joiner_dir,
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
                "joiner_docs": joiner_manifest["docs"],
                "joiner_bytes": joiner_manifest["text_bytes"],
                "budget": budget,
            }
        else:
            mix_meta = {
                "nepali_share": nepali_share,
                "shares": shares,
                "joiner_share": 0.0,
                "budget": budget,
            }

        path = write_snapshot(name, chosen, mix_meta=mix_meta)
        man = load_json(path / "manifest.json")
        results["snapshots"][name] = {
            "path": str(path.relative_to(REPO_ROOT)),
            "manifest": str((path / "manifest.json").relative_to(REPO_ROOT)),
            "text_bytes": man["text_bytes"],
            "per_bucket": man["per_bucket"],
        }

    # 50% without joiner.
    shares = scale_shares(mix, 0.50)
    targets = {b: int(budget * s) for b, s in shares.items()}
    selected_index = []
    for bucket, target in targets.items():
        selected_index.extend(
            pool_sample_index(
                by_bucket.get(bucket, []),
                target,
                seed=SEED + 5000 + hashlib_int(bucket),
                temperature=temperature,
            )
        )
    chosen = fetch_texts(selected_index)
    path = write_snapshot(
        "n50_nojoiner",
        chosen,
        mix_meta={
            "nepali_share": 0.50,
            "shares": shares,
            "joiner_share": 0.0,
            "budget": budget,
            "note": "50% snapshot without joiner copy for the M7 ladder",
        },
    )
    man = load_json(path / "manifest.json")
    results["snapshots"]["n50_nojoiner"] = {
        "path": str(path.relative_to(REPO_ROOT)),
        "manifest": str((path / "manifest.json").relative_to(REPO_ROOT)),
        "text_bytes": man["text_bytes"],
        "per_bucket": man["per_bucket"],
    }

    print("auditing snapshots for residual contamination", flush=True)
    for name, info in results["snapshots"].items():
        snap = REPO_ROOT / info["path"]
        counts = audit_snapshot(snap)
        results["audits"][name] = counts
        print(f"audit {name} {counts}", flush=True)
        if any(v != 0 for v in counts.values()):
            raise Stop(f"STOP: snapshot {name} failed decontam audit: {counts}")

    results = finish_manifest({**base_manifest(stage="snapshots"), **results})
    save_json(artifacts_clean() / "snapshots.json", results)
    if free_bytes() < 50_000_000_000:
        raise Stop(f"STOP: free_bytes={free_bytes()} under 50 GB after snapshots")
    return results

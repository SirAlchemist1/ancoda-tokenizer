"""M4 steps 1–3: per-source UTF-8, GlotLID, and legacy-font filtering."""

from __future__ import annotations

import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

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
    label_to_bucket,
    list_shards,
    load_json,
    load_mix,
    load_progress,
    load_sources,
    raw_dir,
    save_json,
    save_progress,
    sha256_file,
    source_bucket_map,
    train_source_ids,
    utf8_len,
)
from ancoda_tok.clean.legacy_font import (
    PreetiLexicon,
    load_lexicon,
    should_drop_document,
)
from ancoda_tok.clean.lid import GlotLID, audit_script_share, decide_lid, empty_stats
from ancoda_tok.clean.utf8 import clean_utf8_text


def _bucket_spec(mix: dict, bucket: str) -> dict:
    return mix["buckets"][bucket]


def _prob_bucket(prob: float) -> str:
    # Fixed bins for the confidence distribution.
    edges = [0.5, 0.7, 0.8, 0.9, 0.95, 0.99, 1.01]
    for edge in edges:
        if prob < edge:
            return f"<{edge}"
    return ">=1"


def process_source(
    source_id: str,
    *,
    mix: dict,
    lid: GlotLID | None,
    lexicon: PreetiLexicon | None,
    label_buckets: dict[str, str],
    source_buckets: dict[str, str],
    force: bool = False,
) -> dict[str, Any]:
    out = clean_dir(source_id)
    manifest_path = out / "manifest.json"
    if manifest_path.exists() and not force:
        existing = load_json(manifest_path)
        if existing.get("status") == "complete":
            print(f"skip {source_id}: clean manifest verified", flush=True)
            return existing

    bucket = source_buckets[source_id]
    bspec = _bucket_spec(mix, bucket)
    keep_labels = set(bspec.get("keep_labels") or [])
    skip_lid = bool(bspec.get("skip_lid"))
    expected_scripts = list(bspec.get("expected_scripts") or [])
    legacy_threshold = float(mix["legacy_font"]["line_flag_threshold"])
    legacy_enabled = bool(mix["legacy_font"].get("enabled", True))
    apply_legacy = bucket == "Nepali" and lexicon is not None and legacy_enabled

    src = raw_dir(source_id)
    if not src.exists():
        raise Stop(f"missing raw source {source_id}")
    raw_manifest = load_json(src / "manifest.json")
    shards = list_shards(src)
    ensure_room(sum(p.stat().st_size for p in shards), label=f"clean/{source_id}")

    out.mkdir(parents=True, exist_ok=True)
    progress = load_progress(out)
    completed = set(progress.get("completed_shards", []))
    stats = progress.get("stats") or empty_stats()
    kept_samples: list[str] = list(progress.get("kept_samples", []))
    dropped_samples: list[str] = list(progress.get("dropped_samples", []))
    script_texts: list[str] = list(progress.get("script_texts", []))
    rng = random.Random(SEED ^ hashlib_stable(source_id))

    # Resume writer index from existing clean shards.
    existing_clean = list_shards(out)
    start_index = 0
    if existing_clean:
        start_index = max(int(p.stem.split("-")[1]) for p in existing_clean) + 1
    writer = ShardWriter(out, schema=CLEAN_SCHEMA, start_index=start_index)
    # Re-load existing shard metas if resuming.
    writer.shards = []
    writer.docs = int(stats.get("docs_kept", 0))
    writer.text_bytes = int(stats.get("bytes_kept", 0))
    for p in existing_clean:
        writer.shards.append(
            {
                "path": str(p.relative_to(REPO_ROOT)),
                "docs": pq.read_metadata(p).num_rows,
                "file_bytes": p.stat().st_size,
                "sha256": sha256_file(p),
            }
        )

    reroute_root = REPO_ROOT / "data" / "clean" / "_reroute"
    reroute_writers: dict[str, ShardWriter] = {}

    def reroute_writer(target: str) -> ShardWriter:
        if target not in reroute_writers:
            dest = reroute_root / target / source_id
            dest.mkdir(parents=True, exist_ok=True)
            # Continue after any existing reroute shards for this pair.
            existing = list_shards(dest)
            idx = 0
            if existing:
                idx = max(int(p.stem.split("-")[1]) for p in existing) + 1
            reroute_writers[target] = ShardWriter(dest, schema=CLEAN_SCHEMA, start_index=idx)
        return reroute_writers[target]

    for shard_path in shards:
        rel = shard_path.name
        if rel in completed:
            continue
        table = pq.read_table(shard_path)
        cols = {name: table.column(name) for name in table.column_names}
        for i in range(table.num_rows):
            doc_id = cols["doc_id"][i].as_py()
            text = cols["text"][i].as_py()
            license_ = cols["license"][i].as_py()
            nbytes = utf8_len(text) if isinstance(text, str) else 0
            stats["docs_in"] += 1
            stats["bytes_in"] += nbytes

            cleaned = clean_utf8_text(text)
            if cleaned is None:
                stats["docs_utf8_dropped"] += 1
                stats["docs_dropped"] += 1
                stats["bytes_dropped"] += nbytes
                _maybe_sample(dropped_samples, text if isinstance(text, str) else "", rng)
                continue

            if skip_lid or lid is None:
                label, prob = ("skip", 1.0)
                decision = decide_lid(
                    label,
                    prob,
                    source_bucket=bucket,
                    keep_labels=keep_labels,
                    label_buckets=label_buckets,
                    skip_lid=True,
                )
            else:
                label, prob = lid.predict(cleaned)
                decision = decide_lid(
                    label,
                    prob,
                    source_bucket=bucket,
                    keep_labels=keep_labels,
                    label_buckets=label_buckets,
                    skip_lid=False,
                )
            key = f"{bucket}\t{label}"
            stats["confusion"][key] = stats["confusion"].get(key, 0) + 1
            pb = _prob_bucket(prob)
            stats["prob_hist"][pb] = stats["prob_hist"].get(pb, 0) + 1

            if decision.action == "drop":
                stats["docs_dropped"] += 1
                stats["bytes_dropped"] += utf8_len(cleaned)
                _maybe_sample(dropped_samples, cleaned, rng)
                continue

            if apply_legacy and should_drop_document(cleaned, lexicon, legacy_threshold):
                stats["docs_legacy_dropped"] += 1
                stats["docs_dropped"] += 1
                stats["bytes_dropped"] += utf8_len(cleaned)
                _maybe_sample(dropped_samples, cleaned, rng)
                continue
            if (
                bucket == "Nepali"
                and lexicon is not None
                and not legacy_enabled
            ):
                # Limitation report when filter is disabled: count would-flag lines.
                from ancoda_tok.clean.legacy_font import document_legacy_line_share

                _, flagged_lines, _, _ = document_legacy_line_share(cleaned, lexicon)
                stats["legacy_would_flag_lines"] = (
                    int(stats.get("legacy_would_flag_lines", 0)) + flagged_lines
                )

            row = {
                "doc_id": doc_id,
                "text": cleaned,
                "source": source_id,
                "license": license_,
                "bucket": decision.target_bucket or bucket,
                "glotlid_label": label,
                "glotlid_prob": float(prob),
            }
            if decision.action == "reroute":
                stats["docs_rerouted"] += 1
                stats["bytes_rerouted"] += utf8_len(cleaned)
                reroute_writer(decision.target_bucket).add(row)
            else:
                writer.add(row)
                stats["docs_kept"] += 1
                stats["bytes_kept"] += utf8_len(cleaned)
                _maybe_sample(kept_samples, cleaned, rng)
                if len(script_texts) < 2000:
                    script_texts.append(cleaned)

        completed.add(rel)
        progress = {
            "completed_shards": sorted(completed),
            "stats": stats,
            "kept_samples": kept_samples[:50],
            "dropped_samples": dropped_samples[:50],
            "script_texts": script_texts[:2000],
        }
        save_progress(out, progress)
        print(
            f"{source_id} {rel} kept={stats['docs_kept']} dropped={stats['docs_dropped']} "
            f"rerouted={stats['docs_rerouted']}",
            flush=True,
        )

    writer.close()
    for rw in reroute_writers.values():
        rw.close()

    # Script contamination audit on a sample of kept docs.
    share = audit_script_share(script_texts, expected_scripts) if expected_scripts else 0.0
    stats["script_outside_share"] = share
    if expected_scripts and share > float(mix["script_outside_stop"]):
        raise Stop(
            f"STOP (human): {source_id} / {bucket} has {share:.4f} of characters "
            f"outside expected scripts {expected_scripts} (bar 0.20)"
        )

    # Write samples for optional inspection.
    sample_dir = artifacts_clean() / "samples" / bucket
    sample_dir.mkdir(parents=True, exist_ok=True)
    _write_samples(sample_dir / f"{source_id}_kept.txt", kept_samples[:50])
    _write_samples(sample_dir / f"{source_id}_dropped.txt", dropped_samples[:50])

    manifest = base_manifest(
        source_id=source_id,
        status="complete",
        role="train",
        bucket=bucket,
        stage="utf8_lid_legacy",
        input_sha256={"raw_manifest": sha256_file(src / "manifest.json")},
        hf_revision=raw_manifest.get("hf_revision"),
        docs=writer.docs,
        text_bytes=writer.text_bytes,
        file_bytes=sum(s.get("file_bytes", 0) for s in writer.shards),
        shards=writer.shards,
        stats=stats,
        raw_deletion=(
            "Delete this source's raw parquet only after this clean manifest is verified "
            "and the HF revision is recorded (Option 2)."
        ),
    )
    manifest = finish_manifest(manifest)
    save_json(manifest_path, manifest)
    # Clear progress after success.
    progress_path = out / "progress.json"
    if progress_path.exists():
        progress_path.unlink()
    print(
        f"complete {source_id} docs={manifest['docs']} text_bytes={manifest['text_bytes']} "
        f"script_outside={share:.6f} free_bytes={free_bytes()}",
        flush=True,
    )
    return manifest


def hashlib_stable(text: str) -> int:
    import hashlib

    return int(hashlib.sha256(text.encode()).hexdigest()[:8], 16)


def _maybe_sample(bucket: list[str], text: str, rng: random.Random, cap: int = 50) -> None:
    if not text:
        return
    snippet = text if len(text) <= 2000 else text[:2000]
    if len(bucket) < cap:
        bucket.append(snippet)
        return
    # Reservoir among candidates seen via random replace.
    if rng.random() < 0.01:
        bucket[rng.randrange(cap)] = snippet


def _write_samples(path: Path, samples: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n----\n".join(samples) + ("\n" if samples else ""), encoding="utf-8")


def delete_raw_if_allowed(source_id: str) -> None:
    """Option 2: delete raw parquet after clean is verified."""
    out = clean_dir(source_id)
    manifest_path = out / "manifest.json"
    if not manifest_path.exists():
        return
    manifest = load_json(manifest_path)
    if manifest.get("status") != "complete":
        return
    if not manifest.get("hf_revision"):
        print(f"keep raw {source_id}: no hf_revision in clean manifest", flush=True)
        return
    src = raw_dir(source_id)
    removed = 0
    for shard in list_shards(src):
        shard.unlink()
        removed += 1
    print(f"deleted {removed} raw shards for {source_id}", flush=True)


def run_clean_phase(*, force: bool = False, delete_raw: bool = True) -> dict[str, Any]:
    mix = load_mix()
    sources_spec = load_sources()
    source_buckets = source_bucket_map(mix)
    label_buckets = label_to_bucket(mix)
    report: dict[str, Any] = base_manifest(stage="clean_phase", sources={})

    from ancoda_tok.clean.legacy_font import build_lexicon, lexicon_path

    lex_path = lexicon_path()
    if lex_path.exists() and not force:
        print(f"loading Preeti lexicon {lex_path}", flush=True)
        lexicon = load_lexicon(lex_path)
    else:
        lexicon = build_lexicon(out_path=lex_path)

    print("loading GlotLID v3", flush=True)
    lid = GlotLID()

    for source_id in train_source_ids(mix):
        # Confirm the source is a training row in sources.yaml.
        row = next((s for s in sources_spec["sources"] if s["id"] == source_id), None)
        if row is None or row.get("role") != "train":
            continue
        manifest = process_source(
            source_id,
            mix=mix,
            lid=lid,
            lexicon=lexicon,
            label_buckets=label_buckets,
            source_buckets=source_buckets,
            force=force,
        )
        report["sources"][source_id] = {
            "docs": manifest["docs"],
            "text_bytes": manifest["text_bytes"],
            "stats": manifest["stats"],
            "script_outside_share": manifest["stats"].get("script_outside_share"),
        }
        if delete_raw:
            delete_raw_if_allowed(source_id)

    # Aggregate per-language script shares; STOP if any language > 20%.
    by_bucket: dict[str, list[float]] = defaultdict(list)
    for source_id, info in report["sources"].items():
        bucket = source_buckets[source_id]
        share = info.get("script_outside_share")
        if share is not None:
            by_bucket[bucket].append(share)
    report["script_outside_by_bucket"] = {
        b: (sum(v) / len(v) if v else 0.0) for b, v in sorted(by_bucket.items())
    }
    for bucket, share in report["script_outside_by_bucket"].items():
        expected = mix["buckets"][bucket].get("expected_scripts") or []
        if expected and share > float(mix["script_outside_stop"]):
            raise Stop(
                f"STOP (human): language {bucket} characters outside scripts "
                f"share={share:.4f} > 0.20"
            )

    report = finish_manifest(report)
    save_json(artifacts_clean() / "report.json", report)
    print(f"wrote artifacts/clean/report.json free_bytes={free_bytes()}", flush=True)
    return report

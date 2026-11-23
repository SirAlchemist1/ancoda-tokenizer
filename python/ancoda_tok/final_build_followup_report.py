"""final-build follow-up reporting only. No training, no config changes, no sealed retune."""

from __future__ import annotations

import hashlib
import json
import random
from collections import Counter, defaultdict
from typing import Any

import pyarrow.parquet as pq

import ancoda_tok.guarantee as G
from ancoda_tok.clean.lid import GlotLID
from ancoda_tok.clean.snapshot import fetch_texts
from ancoda_tok.guarantee import (
    _SINGLE,
    Engines,
    _aksharas,
    _bootstrap,
    _has_deva,
    _tail,
)
from ancoda_tok.final_build_final import (
    BOOT_N,
    CLASSES,
    FINAL_BUILD,
    OURS,
    REPO_ROOT,
    SEED,
    TOK,
    Encoder,
    covered_path,
    panel_paths,
    verdict,
)
from ancoda_tok.storage import REPO_ROOT as _R

assert _R == REPO_ROOT

OUT = FINAL_BUILD / "followup"
SNAP = REPO_ROOT / "data" / "snapshots" / "n60j5_20af1fffbec0fe86"
FLORES_EXTRA = REPO_ROOT / "data" / "eval" / "flores_plus_extra" / "devtest"
FLORES_YAK = (
    REPO_ROOT / "third_party" / "yak-tokenizer" / "data" / "eval" / "flores_plus" / "devtest"
)

BUCKET_TO_GLOTLID = {
    "Nepali": "npi_Deva",
    "Hindi": "hin_Deva",
    "Marathi": "mar_Deva",
    "Sanskrit": "san_Deva",
    "English": "eng_Latn",
}
GLOTLID_TO_BUCKET = {v: k for k, v in BUCKET_TO_GLOTLID.items()}


def log(msg: str) -> None:
    print(msg, flush=True)


def recover_sealed() -> list[dict[str, Any]]:
    sealed = json.loads((REPO_ROOT / "docs" / "sealed_test_set.json").read_text(encoding="utf-8"))
    wanted = set(sealed["doc_id_sha256"])
    index = REPO_ROOT / "data" / "clean" / "_snapshot_index.parquet"
    selected: list[dict[str, Any]] = []
    pf = pq.ParquetFile(index)
    need = ["doc_id", "source", "license", "text_bytes", "shard", "row", "bucket", "lang"]
    for batch in pf.iter_batches(columns=need, batch_size=8192):
        data = {name: batch.column(name).to_pylist() for name in need}
        for i, doc_id in enumerate(data["doc_id"]):
            digest = hashlib.sha256(doc_id.encode("utf-8")).hexdigest()
            if digest not in wanted:
                continue
            selected.append({key: data[key][i] for key in need})
            selected[-1]["doc_id_sha256"] = digest
    if len(selected) != sealed["n_docs"]:
        raise SystemExit(f"STOP: recovered {len(selected)} != {sealed['n_docs']}")
    # fetch_texts returns shard-sorted order, not selected order — join on doc_id.
    fetched = {doc["doc_id"]: doc for doc in fetch_texts(selected)}
    rows = []
    for meta in selected:
        doc = fetched[meta["doc_id"]]
        rows.append(
            {
                **meta,
                "text": doc.get("text") or "",
                "stored_glotlid": doc.get("glotlid_label") or "",
                "stored_glotlid_prob": float(doc.get("glotlid_prob") or 0.0),
            }
        )
    return rows


def map_display(glot: str, bucket: str) -> str:
    if bucket in ("Code", "Math"):
        return bucket
    if glot in GLOTLID_TO_BUCKET:
        return GLOTLID_TO_BUCKET[glot]
    return glot


def correct_label(bucket: str, glot: str) -> str:
    """Source bucket + GlotLID: Code/Math keep bucket; else GlotLID (display-mapped)."""
    if bucket in ("Code", "Math"):
        return bucket
    return map_display(glot, bucket)


def sample_indices(n: int, k: int, seed: int, key: str) -> list[int]:
    rng = random.Random(f"{seed}:{key}")
    idx = list(range(n))
    rng.shuffle(idx)
    return sorted(idx[: min(k, n)])


def fertility_tokens_words(enc: Encoder, text: str) -> tuple[int, int]:
    words = len(text.split())
    if words == 0:
        return 0, 0
    return len(enc.ids(text)), words


def corpus_fert(pairs: list[tuple[int, int]]) -> float | None:
    t = sum(a for a, b in pairs if b)
    w = sum(b for a, b in pairs if b)
    return None if w == 0 else t / w


def bootstrap_fert_diff(
    left: list[tuple[int, int]],
    right: list[tuple[int, int]],
    n: int = BOOT_N,
    seed: int = SEED,
) -> dict[str, Any]:
    import numpy as np

    assert len(left) == len(right)
    lt, lw, rt, rw = [], [], [], []
    for a, b in zip(left, right, strict=True):
        if a[1] > 0 and b[1] > 0:
            lt.append(a[0])
            lw.append(a[1])
            rt.append(b[0])
            rw.append(b[1])
    if not lw:
        return {"diff": None, "lo": None, "hi": None, "n": 0, "verdict": "TIE"}
    lt_a = np.asarray(lt, dtype=np.int64)
    lw_a = np.asarray(lw, dtype=np.int64)
    rt_a = np.asarray(rt, dtype=np.int64)
    rw_a = np.asarray(rw, dtype=np.int64)
    point = float(lt_a.sum() / lw_a.sum() - rt_a.sum() / rw_a.sum())
    m = len(lw)
    rng = np.random.default_rng(seed)
    diffs = np.empty(n, dtype=np.float64)
    for i in range(n):
        idx = rng.integers(0, m, size=m)
        diffs[i] = lt_a[idx].sum() / lw_a[idx].sum() - rt_a[idx].sum() / rw_a[idx].sum()
    diffs.sort()
    lo = float(diffs[int(0.025 * (n - 1))])
    hi = float(diffs[int(0.975 * (n - 1))])
    out = {"diff": point, "lo": lo, "hi": hi, "n": n, "seed": seed}
    out["verdict"] = verdict(out)
    return out


def load_specs() -> dict[str, str]:
    paths = panel_paths()
    chosen = json.loads((FINAL_BUILD / "chosen.json").read_text(encoding="utf-8"))
    specs = {
        "ancoda-32k": str(TOK / "ancoda-32k/tokenizer.json"),
        "ancoda-64k": str(TOK / "ancoda-64k/tokenizer.json"),
        "ancoda-128k": str(TOK / "ancoda-128k/tokenizer.json"),
        "ancoda-ne-64k": str(TOK / "ancoda-ne-64k/tokenizer.json"),
        "ancoda-250k": str(REPO_ROOT / chosen["dir"] / "tokenizer.json"),
    }
    for name, spec in paths.items():
        if spec != "skipped: license":
            specs[name] = spec
    return specs


def encode_docs(enc: Encoder, texts: list[str]) -> list[tuple[int, int]]:
    return [fertility_tokens_words(enc, t) for t in texts]


def sealed_audit_and_remeasure(rows: list[dict[str, Any]]) -> dict[str, Any]:
    log("loading GlotLID v3")
    lid = GlotLID()
    by_label: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_label[str(row["lang"] or "unknown")].append(row)

    label_audit = {}
    for label, docs in sorted(by_label.items(), key=lambda kv: -len(kv[1])):
        idxs = sample_indices(len(docs), 200, SEED, f"glot:{label}")
        counts: Counter[str] = Counter()
        probs = []
        for i in idxs:
            glot, prob = lid.predict(docs[i]["text"])
            counts[glot] += 1
            probs.append(prob)
        top, top_n = counts.most_common(1)[0]
        label_audit[label] = {
            "n_docs": len(docs),
            "sample_n": len(idxs),
            "top_glotlid": top,
            "top_count": top_n,
            "top_share": top_n / len(idxs),
            "mean_prob": sum(probs) / len(probs) if probs else None,
            "distribution": dict(counts.most_common(12)),
            "expected_glotlid": BUCKET_TO_GLOTLID.get(label),
            "index_rule": (
                "glotlid_label at index time"
                if label.endswith(("_Deva", "_Latn"))
                else "source bucket name"
            ),
        }
        log(
            f"label {label}: n={len(docs)} sample_top={top} "
            f"{top_n}/{len(idxs)}={top_n/len(idxs):.3f}"
        )

    # Corrected labels: Code/Math keep bucket; else use stored clean-time GlotLID
    # (re-predict only if the clean shard has no label). The 200-doc audit above is
    # a fresh GlotLID check on the sealed text.
    log("assigning corrected labels from stored GlotLID + bucket")
    for i, row in enumerate(rows):
        if row["bucket"] in ("Code", "Math"):
            row["glotlid"] = row["bucket"]
            row["glotlid_prob"] = 1.0
            row["glotlid_source"] = "bucket_skip_lid"
        elif row.get("stored_glotlid"):
            row["glotlid"] = row["stored_glotlid"]
            row["glotlid_prob"] = row.get("stored_glotlid_prob") or 0.0
            row["glotlid_source"] = "clean_shard"
        else:
            glot, prob = lid.predict(row["text"])
            row["glotlid"] = glot
            row["glotlid_prob"] = prob
            row["glotlid_source"] = "repredict"
        row["correct_lang"] = correct_label(str(row["bucket"]), str(row["glotlid"]))
        if (i + 1) % 10000 == 0:
            log(f"  correct_lang {i+1}/{len(rows)}")

    agreement = sum(
        1
        for r in rows
        if (
            (r["bucket"] in ("Code", "Math") and r["correct_lang"] == r["bucket"])
            or (
                r["lang"] == r["correct_lang"]
                or BUCKET_TO_GLOTLID.get(str(r["lang"])) == r["glotlid"]
                or str(r["lang"]) == r["glotlid"]
            )
        )
    )
    corrected_counts = Counter(r["correct_lang"] for r in rows)
    index_vs_correct = Counter((str(r["lang"]), r["correct_lang"]) for r in rows)

    specs = load_specs()
    # Encode once per tokenizer.
    log("encoding sealed docs with frozen tokenizers")
    encoders = {name: Encoder(spec) for name, spec in specs.items()}
    per_tok: dict[str, list[tuple[int, int]]] = {}
    for name, enc in encoders.items():
        log(f"  encode {name}")
        per_tok[name] = encode_docs(enc, [r["text"] for r in rows])

    # Group fertilities under index labels and corrected labels.
    def group_payload(key_field: str) -> dict[str, Any]:
        groups: dict[str, list[int]] = defaultdict(list)
        for i, r in enumerate(rows):
            groups[str(r[key_field])].append(i)
        out = {}
        for label, idxs in sorted(groups.items(), key=lambda kv: -len(kv[1])):
            ferts = {}
            for name, pairs in per_tok.items():
                subset = [pairs[i] for i in idxs]
                ferts[name] = {
                    "docs": len(idxs),
                    "fertility": corpus_fert(subset),
                    "tokens": sum(a for a, b in subset),
                    "words": sum(b for a, b in subset),
                }
            # CIs: each ours vs each class competitor, on this language group.
            cis = []
            for cls, members in CLASSES.items():
                ours = [n for n in members if n in OURS and n in per_tok]
                panel = [n for n in members if n not in OURS and n in per_tok]
                for left in ours:
                    for right in panel:
                        left_pairs = [per_tok[left][i] for i in idxs]
                        right_pairs = [per_tok[right][i] for i in idxs]
                        ci = bootstrap_fert_diff(left_pairs, right_pairs)
                        cis.append(
                            {
                                "class": cls,
                                "ours": left,
                                "base": right,
                                **ci,
                            }
                        )
            out[label] = {"docs": len(idxs), "fertility": ferts, "cis": cis}
        return out

    index_groups = group_payload("lang")
    corrected_groups = group_payload("correct_lang")

    # Item 2: Nepali correctly identified.
    nep_idxs = [i for i, r in enumerate(rows) if r["correct_lang"] == "Nepali"]
    nep_compare_names = [
        "ancoda-64k",
        "ancoda-ne-64k",
        "sidskarki-64k",
        "yak-64k-v1",
        "yak-64k-v2.0",
    ]
    nepali = {"docs": len(nep_idxs), "fertility": {}, "cis": []}
    for name in nep_compare_names:
        subset = [per_tok[name][i] for i in nep_idxs]
        nepali["fertility"][name] = {
            "fertility": corpus_fert(subset),
            "tokens": sum(a for a, b in subset),
            "words": sum(b for a, b in subset),
        }
    for left in ("ancoda-64k", "ancoda-ne-64k"):
        for right in nep_compare_names:
            if left == right:
                continue
            ci = bootstrap_fert_diff(
                [per_tok[left][i] for i in nep_idxs],
                [per_tok[right][i] for i in nep_idxs],
            )
            nepali["cis"].append({"ours": left, "base": right, **ci})

    return {
        "label_rule": (
            "Index lang = source bucket name for Nepali/English/Hindi/Marathi/Sanskrit/"
            "Code/Math; for Smaller Devanagari and Languages of Nepal pools, index lang = "
            "GlotLID label stored at cleaning clean time. Corrected lang = bucket for Code/Math "
            "(skip_lid), else the clean-shard GlotLID label (display-mapped npi_Deva→Nepali, "
            "hin_Deva→Hindi, mar_Deva→Marathi, san_Deva→Sanskrit, eng_Latn→English). "
            "The 200-doc audit re-runs GlotLID on sealed text as a check. "
            "Texts are joined to metadata by doc_id (fetch_texts is shard-sorted)."
        ),
        "n_docs": len(rows),
        "agreement_index_vs_correct": agreement,
        "agreement_rate": agreement / len(rows),
        "label_audit_200": label_audit,
        "corrected_counts": dict(corrected_counts),
        "index_vs_correct_top": [
            {"index": a, "correct": b, "n": n}
            for (a, b), n in index_vs_correct.most_common(40)
        ],
        "index_groups": index_groups,
        "corrected_groups": corrected_groups,
        "sealed_nepali_correct": nepali,
    }


def flores_mar_hne() -> dict[str, Any]:
    yak_langs = sorted(p.stem for p in FLORES_YAK.glob("*.txt"))
    note = (
        "FLORES+ on Hugging Face includes mar_Deva and hne_Deva, but the Yak panel "
        "checkout only materialized a subset of FLORES+ languages under "
        "third_party/yak-tokenizer/data/eval/flores_plus/devtest "
        f"({yak_langs}). FINAL_BUILD scoreboards used that harness list, so mar/hne were absent. "
        "This report downloads them once into data/eval/flores_plus_extra/devtest for "
        "measurement only (not committed)."
    )
    specs = load_specs()
    results = {"note": note, "yak_harness_langs": yak_langs, "langs": {}}
    for lang in ("mar_Deva", "hne_Deva"):
        path = FLORES_EXTRA / f"{lang}.txt"
        lines = path.read_text(encoding="utf-8").splitlines()
        # as-stored, no NFC writeback
        joined = "\n".join(lines)
        words = len(joined.split())
        ferts = {}
        line_pairs: dict[str, list[tuple[int, int]]] = {}
        for name, spec in specs.items():
            enc = Encoder(spec)
            tokens = len(enc.ids(joined))
            ferts[name] = {"fertility": tokens / words, "tokens": tokens, "words": words}
            line_pairs[name] = [
                (len(enc.ids(line)), len(line.split())) for line in lines if line.split()
            ]
        cis = []
        for cls, members in CLASSES.items():
            ours = [n for n in members if n in OURS and n in ferts]
            panel = [n for n in members if n not in OURS and n in ferts]
            for left in ours:
                for right in panel:
                    # line bootstrap on per-line token/word pairs via fertility of resampled lines
                    # reuse document-style: treat each line as a doc
                    ci = bootstrap_fert_diff(line_pairs[left], line_pairs[right])
                    cis.append({"class": cls, "ours": left, "base": right, **ci})
        results["langs"][lang] = {"fertility": ferts, "cis": cis, "n_lines": len(lines)}
        log(f"{lang} measured lines={len(lines)}")
    return results


def stratified_tail() -> dict[str, Any]:
    """1M Devanagari-containing newline lines, seed 20261104, then PLAN 3.9 tail per build."""
    log("collecting Devanagari lines for stratified tail")
    # Reservoir of Devanagari lines across all shards, deterministic.
    target = 1_000_000
    pool: list[str] = []
    seen = 0
    rng = random.Random(SEED)
    for shard in sorted(SNAP.glob("shard-*.parquet")):
        pf = pq.ParquetFile(shard)
        for batch in pf.iter_batches(columns=["text"], batch_size=64):
            for text in batch.column(0).to_pylist():
                if not text:
                    continue
                for line in text.splitlines():
                    if not line or not _has_deva(line):
                        continue
                    seen += 1
                    if len(pool) < target:
                        pool.append(line)
                    else:
                        j = rng.randrange(seen)
                        if j < target:
                            pool[j] = line
        log(f"  shard {shard.name} pool={len(pool)} seen_deva_lines={seen}")
        if seen > target * 20 and len(pool) == target:
            # enough mixing; can stop early once we have full reservoir and lots of seen
            # Keep scanning a bit more for mix — actually Algorithm R needs full stream for exactness.
            # For report speed with determinism: continue through all shards.
            pass
    # Final shuffle with same seed for stable order in bootstrap docs
    order = list(range(len(pool)))
    random.Random(SEED).shuffle(order)
    lines = [pool[i] for i in order[:target]]
    log(f"stratified pool size={len(lines)} from seen_deva_lines={seen}")

    jobs = [
        ("ancoda-32k", TOK / "ancoda-32k", covered_path(32_768)),
        ("ancoda-64k", TOK / "ancoda-64k", covered_path(65_536)),
        ("ancoda-128k", TOK / "ancoda-128k", covered_path(131_072)),
        ("ancoda-ne-64k", TOK / "ancoda-ne-64k", covered_path(65_536)),
        ("ancoda-250k", TOK / "ancoda-250k", covered_path(250_000)),
    ]
    rows = {}
    for name, dest, covered in jobs:
        log(f"tail {name}")
        G.OUT = dest
        G.COVERED = covered
        G.SNAPSHOT = SNAP
        engines = Engines()
        doc_hf: list[tuple[int, int]] = []
        doc_tt: list[tuple[int, int]] = []
        uncovered = broken = 0
        unit_counts: Counter[str] = Counter()
        unit_broken: Counter[str] = Counter()
        for line in lines:
            pieces = engines.pieces(line)
            hf_one, tt_one = _tail(engines, line, pieces)
            b, t = hf_one
            if t:
                doc_hf.append(hf_one)
                doc_tt.append(tt_one)
                uncovered += t
                broken += b
                spans = _aksharas(line)
                bounds = G._bounds(pieces[1])
                for unit, start, end in spans:
                    if unit in engines.multi or unit in _SINGLE:
                        continue
                    if not any("\u0900" <= ch <= "\u097f" for ch in unit):
                        continue
                    unit_counts[unit] += 1
                    if G._inside(start, end, bounds):
                        unit_broken[unit] += 1
        hf_ci = _bootstrap(doc_hf)
        tt_ci = _bootstrap(doc_tt)
        rows[name] = {
            "lines": len(lines),
            "lines_with_uncovered_multi": len(doc_hf),
            "uncovered_multi": uncovered,
            "broken_multi": broken,
            "rate": None if uncovered == 0 else broken / uncovered,
            "tail_hf_rate_lo_hi": list(hf_ci),
            "tail_tt_rate_lo_hi": list(tt_ci),
            "top_uncovered": unit_counts.most_common(15),
            "top_broken": unit_broken.most_common(15),
            "covered_path": str(covered.relative_to(REPO_ROOT)),
        }
        log(f"  {name} uncovered={uncovered} broken={broken} rate={rows[name]['rate']} ci={hf_ci}")
    return {
        "definition": (
            "PLAN 3.9 tail on a Devanagari-stratified sample: Algorithm R reservoir of "
            f"{target} newline-separated lines that contain at least one Devanagari character, "
            f"drawn from all n60j5 shards with seed {SEED}. Not the code-first prefix."
        ),
        "seen_deva_lines": seen,
        "sample_lines": len(lines),
        "rows": rows,
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    # Always recover texts for encoding; meta cache optional.
    log("recovering sealed documents")
    rows = recover_sealed()
    log(f"recovered {len(rows)}")

    sealed_report = sealed_audit_and_remeasure(rows)
    (OUT / "sealed_relabel.json").write_text(
        json.dumps(sealed_report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    log("wrote sealed_relabel.json")

    flores = flores_mar_hne()
    (OUT / "flores_mar_hne.json").write_text(
        json.dumps(flores, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    log("wrote flores_mar_hne.json")

    tail = stratified_tail()
    (OUT / "tail_stratified_1m.json").write_text(
        json.dumps(tail, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    log("wrote tail_stratified_1m.json")

    summary = {
        "seed": SEED,
        "sealed_agreement_rate": sealed_report["agreement_rate"],
        "corrected_counts": sealed_report["corrected_counts"],
        "sealed_nepali_docs": sealed_report["sealed_nepali_correct"]["docs"],
        "flores_note": flores["note"],
        "tail_sample_lines": tail["sample_lines"],
        "misspell_correction": (
            "At 64k, absolute misspelled FLORES npi fertility: sidskarki 1.5282 < "
            "Ancoda-64k 1.5332. The earlier note that Ancoda kept the lowest absolute "
            "misspelled fertility vs sidskarki was wrong."
        ),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log("done")


if __name__ == "__main__":
    main()

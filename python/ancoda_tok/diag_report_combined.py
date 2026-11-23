"""NO-TRAINING combined STOP report for T4/T1/T2/T3 + competitor rows."""

from __future__ import annotations

import json
from pathlib import Path

from ancoda_tok.clean.common import base_manifest, finish_manifest, save_json, sha256_file
from ancoda_tok.export_hf import load_hf
from ancoda_tok.frontier_metrics import fertility_on_texts, fw2_npi_test_texts
from ancoda_tok.storage import REPO_ROOT

M7 = REPO_ROOT / "artifacts" / "eval" / "m7"
FLORES_HIN = (
    REPO_ROOT
    / "third_party/yak-tokenizer/data/eval/flores_plus/devtest/hin_Deva.txt"
)
YAK = (
    REPO_ROOT
    / "data/hf_cache/hub/models--ampixa--yak-tokenizer/snapshots"
    / "12b6d3010ec542b484ec3b673354852fe8bb85fb"
)
OUT = M7 / "combined_frontier_report.json"
FERT_KEYS = (
    "npi",
    "eng",
    "hin",
    "aya_fertility",
    "fw2_npi_test_fertility",
    "morph_f1_unweighted",
    "whole_word_unweighted",
    "orphan",
    "zwj",
    "zwnj",
    "flores_deva_mean",
    "block1_coverage",
    "tail_hf_rate",
)


def _load(rel: str | Path) -> dict:
    path = Path(rel)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return json.loads(path.read_text(encoding="utf-8"))


def _fmt6(x) -> str:
    if x is None:
        return ""
    return f"{float(x):.6f}"


def _harness_fert(res: dict, key: str) -> float | None:
    block = res.get(key) or {}
    return block.get("fertility")


def competitor_rows() -> list[dict]:
    rows: list[dict] = []
    t4 = _load(M7 / "script_budget_128k/frontier.json")
    t4_by = {r["label"]: r for r in t4["rows"]}

    specs = [
        (
            "64k",
            "yak_v1_64k",
            YAK / "v1.0/64k/tokenizer.json",
            "third_party/yak-tokenizer/artifacts/eval/yak-64k-v1.json",
        ),
        (
            "64k",
            "yak_v2_64k",
            YAK / "v2.0/64k/tokenizer.json",
            "third_party/yak-tokenizer/artifacts/eval/yak-64k-v2.0.json",
        ),
        (
            "64k",
            "sidskarki_64k",
            None,
            "third_party/yak-tokenizer/artifacts/eval/sidskarki-64k.json",
        ),
        (
            "128k",
            "yak_v1_128k",
            YAK / "v1.0/128k/tokenizer.json",
            "artifacts/eval/m7/t4_128k_yak_v1_128k.json",
        ),
        (
            "128k",
            "yak_v2_128k",
            YAK / "v2.0/128k/tokenizer.json",
            "artifacts/eval/m7/t4_128k_yak_v2_128k.json",
        ),
    ]

    for size, label, tok_path, harness_rel in specs:
        if label in t4_by and size == "128k":
            r = t4_by[label]
            rows.append(
                {
                    "arm": "competitor",
                    "label": label,
                    "vocab": size,
                    "npi": r.get("npi"),
                    "eng": r.get("eng"),
                    "hin": r.get("hin"),
                    "aya_fertility": r.get("aya_fertility"),
                    "fw2_npi_test_fertility": r.get("fw2_npi_test_fertility"),
                    "morph_f1_unweighted": r.get("morph_f1_unweighted"),
                    "path": r.get("dir"),
                    "path_sha256": r.get("path_sha256"),
                }
            )
            continue

        harness = _load(harness_rel)
        res = harness.get("results") or {}
        row: dict = {
            "arm": "competitor",
            "label": label,
            "vocab": size,
            "npi": _harness_fert(res, "flores_devtest/npi_Deva"),
            "eng": _harness_fert(res, "flores_devtest/eng_Latn"),
            "hin": _harness_fert(res, "flores_devtest/hin_Deva"),
            "aya_fertility": _harness_fert(res, "npi_everyday_aya"),
            "harness": harness_rel,
            "harness_sha256": sha256_file(REPO_ROOT / harness_rel),
        }

        if label == "sidskarki_64k":
            import sentencepiece as spm

            model = (
                REPO_ROOT
                / "data/hf_cache/hub/models--sidskarki--nepali-bpe-tokenizer/snapshots"
                / "fc5773dcf7b9dc2272e8f595f99828969621a855/nepali_bpe_64k.model"
            )
            if model.is_file():
                sp = spm.SentencePieceProcessor(model_file=str(model))
                texts = fw2_npi_test_texts()
                toks = words = 0
                for t in texts:
                    w = t.split()
                    if not w:
                        continue
                    words += len(w)
                    toks += len(sp.encode(t, out_type=int))
                row["fw2_npi_test_fertility"] = 0.0 if words == 0 else toks / words
                row["path"] = str(model.relative_to(REPO_ROOT))
                row["path_sha256"] = sha256_file(model)
            row["morph_f1_unweighted"] = None
            row["morph_note"] = "sidskarki is SentencePiece; HF morph skipped"
        elif tok_path is not None and tok_path.is_file():
            row["path"] = str(tok_path.relative_to(REPO_ROOT))
            row["path_sha256"] = sha256_file(tok_path)
            row["fw2_npi_test_fertility"] = fertility_on_texts(
                tok_path, fw2_npi_test_texts()
            )
            from ancoda_tok.eval.morph import score_tokenizer

            morph = score_tokenizer(tok_path.parent)
            row["morph_f1_unweighted"] = morph["unweighted"]["f1"]
        rows.append(row)
    return rows


def hindi_token_check() -> dict:
    anc = REPO_ROOT / "artifacts/tokenizers/ancoda-128k-n60/tokenizer.json"
    yak = YAK / "v1.0/128k/tokenizer.json"
    text = FLORES_HIN.read_text(encoding="utf-8")
    lines = text.splitlines()
    words = len(text.split())
    a = load_hf(anc)
    y = load_hf(yak)
    if hasattr(y, "encode_special_tokens"):
        y.encode_special_tokens = True
    a_ids = a.encode(text, add_special_tokens=False).ids
    y_ids = y.encode(text, add_special_tokens=False).ids
    differ = 0
    nonempty = 0
    for line in lines:
        if not line.strip():
            continue
        nonempty += 1
        if a.encode(line, add_special_tokens=False).ids != y.encode(
            line, add_special_tokens=False
        ).ids:
            differ += 1
    return {
        "flores_hin_path": str(FLORES_HIN.relative_to(REPO_ROOT)),
        "flores_hin_sha256": sha256_file(FLORES_HIN),
        "words": words,
        "lines_nonempty": nonempty,
        "anc128_n60": {
            "path": str(anc.relative_to(REPO_ROOT)),
            "sha256": sha256_file(anc),
            "total_tokens": len(a_ids),
            "fertility": None if words == 0 else len(a_ids) / words,
        },
        "yak_v1_128k": {
            "path": str(yak.relative_to(REPO_ROOT)),
            "sha256": sha256_file(yak),
            "total_tokens": len(y_ids),
            "fertility": None if words == 0 else len(y_ids) / words,
        },
        "lines_with_differing_token_sequences": differ,
        "token_totals_equal": len(a_ids) == len(y_ids),
        "hin_fertility_equal": (
            abs((len(a_ids) / words) - (len(y_ids) / words)) < 1e-12 if words else None
        ),
    }


def enrich_t1(rows: list[dict]) -> list[dict]:
    out = []
    for r in rows:
        rr = dict(r)
        g_path = M7 / "t1_block1_cap" / f"guarantee_bd{r['B_d']}_{r['block1_cap']}.json"
        if g_path.is_file():
            g = _load(g_path)
            rr["tail_hf_rate"] = (g.get("tail_hf_rate_lo_hi") or [None])[0]
            rr["tail_hf_ci95"] = g.get("tail_hf_rate_lo_hi")
        else:
            ci = r.get("tail_hf_rate_lo_hi")
            rr["tail_hf_rate"] = (ci or [None])[0]
            rr["tail_hf_ci95"] = ci
        covered = M7 / "t1_block1_cap" / f"covered_64k_{r['block1_cap']}.json"
        if covered.is_file():
            rr["block1_coverage"] = _load(covered)["coverage"]
        out.append(rr)
    return out


def t3_vs_5gb(t3_rows: list[dict], h_rows: list[dict]) -> list[dict]:
    h_by = {r["B_d"]: r for r in h_rows}
    out = []
    for b in (44000, 46000, 48000):
        t3 = next((r for r in t3_rows if r.get("B_d") == b), None)
        h = h_by.get(b)
        out.append(
            {
                "B_d": b,
                "t3_10gb_joiner": t3,
                "n60_5gb": h if h is not None else "missing",
            }
        )
    return out


def _print_row(arm: str, label: str, r: dict) -> None:
    cols = [
        arm,
        label,
        _fmt6(r.get("npi")),
        _fmt6(r.get("eng")),
        _fmt6(r.get("hin")),
        _fmt6(r.get("aya_fertility")),
        _fmt6(r.get("fw2_npi_test_fertility")),
        _fmt6(r.get("morph_f1_unweighted")),
        _fmt6(r.get("whole_word_unweighted")),
        "" if r.get("rare_tokens_lt_100") is None else str(r.get("rare_tokens_lt_100")),
        _fmt6(r.get("orphan")),
        ""
        if r.get("hf_covered_violations") is None
        else str(r.get("hf_covered_violations")),
        ""
        if r.get("code_tokens_per_kb") is None
        else f"{float(r['code_tokens_per_kb']):.6f}",
        ""
        if r.get("math_tokens_per_kb") is None
        else f"{float(r['math_tokens_per_kb']):.6f}",
        _fmt6(r.get("zwj")),
        _fmt6(r.get("zwnj")),
        _fmt6(r.get("block1_coverage")),
        _fmt6(r.get("tail_hf_rate")),
        (
            ""
            if not r.get("tail_hf_ci95")
            else f"[{_fmt6(r['tail_hf_ci95'][1])},{_fmt6(r['tail_hf_ci95'][2])}]"
        ),
        _fmt6(r.get("flores_deva_mean")),
        "" if r.get("B_d") is None else str(r.get("B_d")),
        "" if r.get("block1_cap") is None else str(r.get("block1_cap")),
        "" if r.get("code_math_scale") is None else str(r.get("code_math_scale")),
    ]
    print(" | ".join(cols))


def main() -> None:
    t4 = _load(M7 / "script_budget_128k/frontier.json")
    t1 = _load(M7 / "t1_block1_cap/frontier.json")
    t2 = _load(M7 / "t2_latin_rebalance/frontier.json")
    t3 = _load(M7 / "t3_10gb_joiner/frontier.json")
    h = _load(M7 / "script_budget/frontier.json")

    t1_rows = enrich_t1(t1["rows"])
    # T2 already has code/math tokens_per_kb on held-out samples from frontier H path
    comps = competitor_rows()
    hindi = hindi_token_check()
    compare = t3_vs_5gb(t3["rows"], h["rows"])

    payload = finish_manifest(
        base_manifest(
            kind="combined_frontier_report",
            t4_current_128k=t4.get("current_128k"),
            f_verify_hindi=t4.get("f_verify_hindi"),
            f_verify_equal=t4.get("f_verify_equal"),
            t4_rows=t4["rows"],
            t1_rows=t1_rows,
            t2_rows=t2["rows"],
            t3_rows=t3["rows"],
            competitors=comps,
            hindi_token_check=hindi,
            t3_vs_5gb=compare,
            notes={
                "fertilities": "printed to 6 decimals",
                "t1_coverage": "docs/SPEC.md 3.6 from covered_64k_{5,3,2}pct.json",
                "t1_tail": "guarantee tail_hf_rate_lo_hi = [point, lo, hi]",
                "t2_code_math": "same held-out samples as frontier H via held_out_code_math",
                "5gb_bd46000": "missing from script_budget H frontier",
                "final_shipping_config": "unchanged",
            },
        )
    )
    save_json(OUT, payload)

    print("COMBINED_TABLE")
    print(
        "arm | label | npi | eng | hin | aya | fw2 | morphF1 | wholeWord | rare | orphan | viol | code/KB | math/KB | zwj | zwnj | B1cov | tail | tailCI95 | DevaMean | B_d | block1_cap | cm_scale"
    )
    for r in t4["rows"]:
        _print_row("T4", r.get("label", ""), r)
    for r in t1_rows:
        _print_row("T1", r.get("label", ""), r)
    for r in t2["rows"]:
        _print_row("T2", r.get("label", ""), r)
    for r in t3["rows"]:
        _print_row("T3", r.get("label", ""), r)
    for r in comps:
        _print_row("CMP", r.get("label", ""), r)

    print("\n=== ITEM1 T1 Block1 coverage + tail break 95% CI ===")
    for r in t1_rows:
        ci = r.get("tail_hf_ci95") or [None, None, None]
        print(
            f"{r['label']}: coverage={_fmt6(r.get('block1_coverage'))} "
            f"tail={_fmt6(ci[0])} CI95=[{_fmt6(ci[1])},{_fmt6(ci[2])}]"
        )

    print("\n=== ITEM2 T2 code/math tokens per KB (held-out H samples) ===")
    for r in t2["rows"]:
        print(
            f"{r['label']}: code={float(r['code_tokens_per_kb']):.6f} "
            f"math={float(r['math_tokens_per_kb']):.6f}"
        )

    print("\n=== ITEM5 Hindi token check ===")
    print(json.dumps(hindi, indent=2))

    print("\n=== ITEM6 T3 (10GB+joiner) vs 5GB n60 frontier H ===")
    for c in compare:
        t3r = c["t3_10gb_joiner"] or {}
        h5 = c["n60_5gb"]
        if h5 == "missing":
            print(
                f"B_d={c['B_d']}: T3 npi={_fmt6(t3r.get('npi'))} eng={_fmt6(t3r.get('eng'))} "
                f"hin={_fmt6(t3r.get('hin'))} zwj={_fmt6(t3r.get('zwj'))} | 5GB=missing"
            )
        else:
            print(
                f"B_d={c['B_d']}: T3 npi={_fmt6(t3r.get('npi'))} eng={_fmt6(t3r.get('eng'))} "
                f"hin={_fmt6(t3r.get('hin'))} aya={_fmt6(t3r.get('aya_fertility'))} "
                f"zwj={_fmt6(t3r.get('zwj'))} | "
                f"5GB npi={_fmt6(h5.get('npi'))} eng={_fmt6(h5.get('eng'))} "
                f"hin={_fmt6(h5.get('hin'))} aya={_fmt6(h5.get('aya_fertility'))} "
                f"zwj={_fmt6(h5.get('zwj'))}"
            )
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()

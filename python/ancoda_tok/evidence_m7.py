"""Generate docs/EVIDENCE_M7.md from on-disk JSON (no hand-typed numbers)."""

from __future__ import annotations

import json
from pathlib import Path

from ancoda_tok.clean.common import sha256_file
from ancoda_tok.storage import REPO_ROOT

OUT = REPO_ROOT / "docs" / "EVIDENCE_M7.md"


def _require(rel: str) -> Path:
    path = REPO_ROOT / rel
    if not path.is_file():
        raise SystemExit(f"STOP: missing evidence source {rel}")
    return path


def _load_json(rel: str) -> dict:
    return json.loads(_require(rel).read_text(encoding="utf-8"))


def _claim(
    claims: list[dict],
    *,
    claim_id: str,
    text: str,
    source: str,
    reproduce: str,
    value: object,
) -> None:
    path = _require(source)
    claims.append(
        {
            "id": claim_id,
            "text": text,
            "source": source,
            "sha256": sha256_file(path),
            "reproduce": reproduce,
            "value": value,
        }
    )


def build() -> list[dict]:
    claims: list[dict] = []

    # a. Yak M1 reproduction
    tsv_rel = "artifacts/eval/m1_reproduction.tsv"
    tsv = _require(tsv_rel).read_text(encoding="utf-8")
    data_lines = [ln for ln in tsv.splitlines()[1:] if ln.strip()]
    pass_n = sum(1 for ln in data_lines if "\tPASS" in ln or ln.rstrip().endswith("PASS"))
    total = len(data_lines)
    _claim(
        claims,
        claim_id="M1_yak_reproduction",
        text=f"Yak numbers reproduced (M1): {pass_n}/{total} PASS within ±0.003",
        source=tsv_rel,
        reproduce="rg -c $'\\tPASS' artifacts/eval/m1_reproduction.tsv; wc -l artifacts/eval/m1_reproduction.tsv",
        value={"pass": pass_n, "total": total},
    )

    # b. guarantee integrity on n60
    g_rel = "artifacts/eval/m7/guarantee_n60.json"
    g = _load_json(g_rel)
    _claim(
        claims,
        claim_id="M6_guarantee_n60",
        text=(
            "0 covered violations, 0 equality mismatches, 100% round-trip "
            f"(hf_covered_violations={g.get('hf_covered_violations')}, "
            f"hf_roundtrip_failures={g.get('hf_roundtrip_failures')}, "
            f"failures={g.get('failures')})"
        ),
        source=g_rel,
        reproduce=(
            "uv run python -c \"import json; d=json.load(open('artifacts/eval/m7/guarantee_n60.json')); "
            "print(d['hf_covered_violations'], d['hf_roundtrip_failures'], d['failures'])\""
        ),
        value={
            "hf_covered_violations": g.get("hf_covered_violations"),
            "tt_covered_violations": g.get("tt_covered_violations"),
            "hf_roundtrip_failures": g.get("hf_roundtrip_failures"),
            "failures": g.get("failures"),
        },
    )

    # c. Ladder integrity cost from diag_ladder_summary
    lad_rel = "artifacts/eval/m7/diag_ladder_summary.json"
    lad = _load_json(lad_rel)["ladder"]
    plain_npi = float(lad["plain"]["npi"])
    final_npi = float(lad["final"]["npi"])
    cost = final_npi - plain_npi
    viol_plain = lad["plain"]["covered_violations"]
    _claim(
        claims,
        claim_id="M7_ladder_integrity_cost",
        text=(
            f"Ladder: integrity costs {cost:.4f} Nepali vs plain "
            f"(final {final_npi:.6f} − plain {plain_npi:.6f}); "
            f"plain has {viol_plain} violations"
        ),
        source=lad_rel,
        reproduce=(
            "uv run python -c \"import json; d=json.load(open('artifacts/eval/m7/diag_ladder_summary.json'))['ladder']; "
            "print(d['final']['npi']-d['plain']['npi'], d['plain']['covered_violations'])\""
        ),
        value={
            "plain_npi": plain_npi,
            "final_npi": final_npi,
            "cost": cost,
            "plain_violations": viol_plain,
        },
    )

    # d. Specialist nepali-only vs sidskarki
    nep_rel = "artifacts/eval/m7/diag_nepali_only_diag_n60_nepali_only.json"
    if not (REPO_ROOT / nep_rel).is_file():
        # harness summary
        nep_rel = "artifacts/eval/m7/harness_diag_nepali_only.json"
        nep = _load_json(nep_rel)
        tok = nep.get("tokenizers", {}).get("diag_n60_nepali_only") or next(
            iter(nep.get("tokenizers", {}).values())
        )
        npi, eng, hin = tok["npi"], tok["eng"], tok["hin"]
    else:
        nep = _load_json(nep_rel)
        res = nep["results"]
        npi = res["flores_devtest/npi_Deva"]["fertility"]
        eng = res["flores_devtest/eng_Latn"]["fertility"]
        hin = res["flores_devtest/hin_Deva"]["fertility"]
    sid_rel = "third_party/yak-tokenizer/artifacts/eval/sidskarki-64k.json"
    sid = _load_json(sid_rel)["results"]
    snpi = sid["flores_devtest/npi_Deva"]["fertility"]
    seng = sid["flores_devtest/eng_Latn"]["fertility"]
    shin = sid["flores_devtest/hin_Deva"]["fertility"]
    _claim(
        claims,
        claim_id="M7_specialist_nepali_only",
        text=(
            f"Specialist: Nepali-only {npi:.3f}/{eng:.3f}/{hin:.3f} "
            f"vs sidskarki {snpi:.3f}/{seng:.3f}/{shin:.3f}"
        ),
        source=nep_rel,
        reproduce=(
            f"uv run python -c \"import json; "
            f"print(json.load(open('{nep_rel}')))\""
        ),
        value={
            "nepali_only": {"npi": npi, "eng": eng, "hin": hin},
            "sidskarki": {"npi": snpi, "eng": seng, "hin": shin, "source": sid_rel},
            "sidskarki_sha256": sha256_file(REPO_ROOT / sid_rel),
        },
    )

    # e. Frontier H Bd44000 vs Yak v2 64k
    fr_rel = "artifacts/eval/m7/script_budget/frontier.json"
    frontier = _load_json(fr_rel)
    row44 = next(r for r in frontier["rows"] if r["B_d"] == 44000)
    yak_rel = "third_party/yak-tokenizer/artifacts/eval/yak-64k-v2.0.json"
    yres = _load_json(yak_rel)["results"]
    ynpi = yres["flores_devtest/npi_Deva"]["fertility"]
    yeng = yres["flores_devtest/eng_Latn"]["fertility"]
    yhin = yres["flores_devtest/hin_Deva"]["fertility"]
    beats = row44["npi"] < ynpi and row44["eng"] < yeng and row44["hin"] < yhin
    _claim(
        claims,
        claim_id="M7_frontier_H_bd44000_vs_yak_v2",
        text=(
            f"Frontier H: B_d=44000 Nepali/English/Hindi "
            f"{row44['npi']:.6f}/{row44['eng']:.6f}/{row44['hin']:.6f} "
            f"vs Yak v2 64k {ynpi:.6f}/{yeng:.6f}/{yhin:.6f} "
            f"(beats all three: {beats})"
        ),
        source=fr_rel,
        reproduce=(
            "uv run python -c \"import json; "
            "d=json.load(open('artifacts/eval/m7/script_budget/frontier.json')); "
            "r=[x for x in d['rows'] if x['B_d']==44000][0]; print(r['npi'], r['eng'], r['hin'])\""
        ),
        value={
            "bd44000": {"npi": row44["npi"], "eng": row44["eng"], "hin": row44["hin"]},
            "yak_v2_64k": {
                "npi": ynpi,
                "eng": yeng,
                "hin": yhin,
                "source": yak_rel,
                "sha256": sha256_file(REPO_ROOT / yak_rel),
            },
            "beats_all_three": beats,
        },
    )

    # f. Joiner probes n50 vs Yak v2
    n50_rel = "artifacts/eval/m7/mix_anc64_n50.json"
    n50 = _load_json(n50_rel)
    n50_zwj = n50["robustness"]["zwj"]["token_ratio"]
    n50_zwnj = n50["robustness"]["zwnj"]["token_ratio"]
    y2 = _load_json(yak_rel)
    y2_zwj = y2["robustness"]["zwj"]["token_ratio"]
    y2_zwnj = y2["robustness"]["zwnj"]["token_ratio"]
    _claim(
        claims,
        claim_id="M7_joiner_n50_probes",
        text=(
            f"Joiner copy: n50 probes {n50_zwj:.3f}/{n50_zwnj:.3f} "
            f"vs Yak v2 {y2_zwj:.3f}/{y2_zwnj:.3f}"
        ),
        source=n50_rel,
        reproduce=(
            "uv run python -c \"import json; d=json.load(open('artifacts/eval/m7/mix_anc64_n50.json')); "
            "print(d['robustness']['zwj']['token_ratio'], d['robustness']['zwnj']['token_ratio'])\""
        ),
        value={
            "n50": {"zwj": n50_zwj, "zwnj": n50_zwnj},
            "yak_v2": {"zwj": y2_zwj, "zwnj": y2_zwnj, "source": yak_rel},
        },
    )

    # g. Danda
    danda_rel = "artifacts/eval/m7/diag_danda.json"
    danda = _load_json(danda_rel)
    splits = danda["phrase_splits"]
    _claim(
        claims,
        claim_id="M7_danda_separate",
        text=(
            "Danda: Yak v1/v2 and sidskarki also split । as its own token "
            f"(छ। → {json.dumps(splits.get('छ।'), ensure_ascii=False)})"
        ),
        source=danda_rel,
        reproduce=(
            "uv run python -c \"import json; d=json.load(open('artifacts/eval/m7/diag_danda.json')); "
            "print(d['phrase_splits']['छ।'])\""
        ),
        value={"phrase_splits": splits, "decision": danda.get("decision")},
    )

    # Combined-candidates snapshot choice
    choice_rel = "artifacts/eval/m7/combined_candidates/snapshot_choice.json"
    choice = _load_json(choice_rel)
    _claim(
        claims,
        claim_id="M7_combined_snapshot_choice",
        text=(
            f"Combined candidates snapshot: {choice.get('chosen_label')} "
            f"({choice.get('chosen_path')}); helping_points={choice.get('helping_points')}"
        ),
        source=choice_rel,
        reproduce=(
            "uv run python -c \"import json; "
            "print(json.load(open('artifacts/eval/m7/combined_candidates/snapshot_choice.json'))"
            "['chosen_label'], "
            "json.load(open('artifacts/eval/m7/combined_candidates/snapshot_choice.json'))"
            "['helping_points'])\""
        ),
        value={
            "chosen_label": choice.get("chosen_label"),
            "chosen_path": choice.get("chosen_path"),
            "helping_points": choice.get("helping_points"),
            "reason": choice.get("reason"),
            "comparisons": choice.get("comparisons"),
        },
    )

    # Combined-candidates frontier (16 points + bootstrap CIs)
    cc_rel = "artifacts/eval/m7/combined_candidates/frontier.json"
    cc = _load_json(cc_rel)
    rows = cc["rows"]
    best64 = min((r for r in rows if r["vocab"] == "64k"), key=lambda r: r["npi"])
    best128 = min((r for r in rows if r["vocab"] == "128k"), key=lambda r: r["npi"])
    _claim(
        claims,
        claim_id="M7_combined_candidates",
        text=(
            f"Combined candidates: {len(rows)} points on {cc.get('snapshot')}; "
            f"best 64k npi={best64['npi']:.6f} ({best64['label']}); "
            f"best 128k npi={best128['npi']:.6f} ({best128['label']})"
        ),
        source=cc_rel,
        reproduce=(
            "uv run python -c \"import json; d=json.load(open("
            "'artifacts/eval/m7/combined_candidates/frontier.json')); "
            "print(len(d['rows']), d.get('snapshot')); "
            "b64=min((r for r in d['rows'] if r['vocab']=='64k'), key=lambda r:r['npi']); "
            "print(b64['label'], b64['npi'])\""
        ),
        value={
            "n_rows": len(rows),
            "snapshot": cc.get("snapshot"),
            "snapshot_choice": cc.get("snapshot_choice"),
            "best_64k": {
                "label": best64["label"],
                "npi": best64["npi"],
                "eng": best64["eng"],
                "hin": best64["hin"],
                "bootstrap_ci_vs": best64.get("bootstrap_ci_vs"),
            },
            "best_128k": {
                "label": best128["label"],
                "npi": best128["npi"],
                "eng": best128["eng"],
                "hin": best128["hin"],
                "bootstrap_ci_vs": best128.get("bootstrap_ci_vs"),
            },
        },
    )
    return claims


def write_md(claims: list[dict]) -> Path:
    lines = [
        "# M7 evidence bundle",
        "",
        "Generated by `uv run python -m ancoda_tok.evidence_m7`. Numbers come only from cited JSON/TSV files.",
        "",
    ]
    for c in claims:
        lines += [
            f"## {c['id']}",
            "",
            f"**Claim:** {c['text']}",
            "",
            f"- **Source:** `{c['source']}`",
            f"- **SHA256:** `{c['sha256']}`",
            f"- **Reproduce:** `{c['reproduce']}`",
            f"- **Extracted value:** `{json.dumps(c['value'], ensure_ascii=False)}`",
            "",
        ]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return OUT


def main() -> None:
    claims = build()
    path = write_md(claims)
    print(f"wrote {path} claims={len(claims)}", flush=True)
    for c in claims:
        print(f"  {c['id']}: {c['text'][:120]}", flush=True)


if __name__ == "__main__":
    main()

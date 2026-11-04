## M0 — Audit, environment and repo — done
- Commits: `dfff33e` (M0), `265df71` (PLAN v2.6 sync)

## M1 — Baselines on Yak's harness — done
- [x] 1. `eval_panel.py fetch` then `run` inside `third_party/yak-tokenizer` (caffeinate)
- [x] 2. Harness validity gate **5/5** (±0.03 Arkios-paper check)
- [x] 3. `docs/LICENSES.md` — Yak CC BY-NC; sidskarki community-use-1.0 allows eval+proxy under ≤$100k revenue + attribution (terms clear; no STOP); Himalaya license not stated; Arkios Apache-2.0
- [x] 4. `artifacts/eval/baseline_RESULTS.md` + `baseline_RESULTS.manifest.json`; copy also in `docs/RESULTS.md` with M1 notes
- [x] 5. Gate A comparison values → `configs/eval.yaml` (incl. orphan/ZWJ/ZWNJ from M1 harness)
- [x] 6. `python/ancoda_tok/eval/gate_a.py` — 3.10 harness rows + Yak scoreboard; Ancoda-only rows PENDING until M8
- [x] Acceptance: `artifacts/eval/m1_reproduction.tsv` all PASS within ±0.003; `uv run python -m ancoda_tok.eval.gate_a --name yak-64k-v2.0` runs

Acceptance:
- Harness validity gate → **5/5 reproduced** (`docs/RESULTS.md`)
- `artifacts/eval/m1_reproduction.tsv` → **18/18 PASS** (abs diff ≤ 0.003); FAIL count `0`
- `uv run python -m ancoda_tok.eval.gate_a --name yak-64k-v2.0` → prints 3.10 rows + Yak scoreboard (see `artifacts/eval/m1_gate_a_yak64v2.txt`)
- Panel mapping: PLAN “Yak v1 128k” → `yak-128k-v1`; “Yak v2 128k” → `yak-128k-v2.0`; “Yak v1 128k-phrase” → `yak-128k-v1-phrase` (recorded in `configs/eval.yaml`)
- Extra recorded: Yak v1 64k-phrase Nepali ≈ 1.373937; Sanskrit harness values in eval.yaml; languages-of-Nepal **not comparable**
- `airavata` failed gated 403 (not in acceptance table)

Needs the human:
- Confirm Ancoda Labs annual revenue ≤ USD $100,000 before using sidskarki-64k as M9 proxy arm (license allows eval/proxy only under that threshold). Not an M1 blocker.
- HimalayaAI Hub repos state no license; comparison-only; do not redistribute.

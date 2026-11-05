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

## M2 — Storage plan, downloads, license log — done
- [x] 1. Storage plan from Hub `list_repo_tree` sizes → `docs/STORAGE.md`. Option 2 approved; gated peak leaves 65.419 GB
- [x] 2. `configs/sources.yaml` and `python/ancoda_tok/download.py`
- [x] 3. Downloads under `caffeinate`, one source at a time, shard resume, Hub cache deleted after each source's shards verify. Training raw is still on disk (deletion is M4, after a verified clean copy)
- [x] 4. `docs/LICENSES.md` rows match the manifest revisions. Papaya CC BY 4.0 attribution recorded. Files under `data/`, not committed
- [x] 5. `deva_pool` from the API. Cap 0.3 GB per path. Angika (`anp_Deva`) stopped at 299996567 text bytes
- [x] 6. `nepal_pool`: ten FineWeb-2 train splits, ODC-By, no upsampling. NepTam and Gurung audio excluded
- [x] 7. Romanized Nepali is not a training source. FineWeb-2 `npi_Latn` test is eval-only (24 docs, 94370 text bytes)
- [x] Acceptance

Human decisions recorded in `docs/DECISIONS.md` and `docs/STORAGE.md`:
- Option 2. Delete a training source's raw parquet only after its cleaned parquet is written and verified, and only after the Hugging Face revision is in the manifest. Never delete eval-only or decontamination-only sources. The 50 GB rule stays.
- `nepal_pool`: `new_Deva`, `thl_Deva`, `taj_Deva`, `lif_Deva`, `rav_Deva`, `suz_Deva`, `xsr_Deva`, `rjs_Deva`, `kle_Deva`, `nwx_Deva`.
- Papaya affix inventory, evaluation only, CC BY 4.0, commit `b3498528`. Cite Balaram Prasain, "A Computational Analysis of Nepali Morphology: A Model for Natural Language Processing", PhD dissertation, Tribhuvan University, 2011.

Measured while downloading:
- IndicCorp `data/ne.txt` uses blank-line documents (`document_rule=blank-lines` in the manifest). 25263288 documents.
- Sangraha `verified/kas` mean Devanagari letter share 0.1994, skipped. `verified/snd` share 0.0002, skipped. Both checks are in the `deva_pool_sangraha` manifest. FineWeb-2 `kas_Deva` and `snd_Deva` were kept.
- Code rows kept only `mit`, `apache-2.0`, `bsd-2-clause`, `bsd-3-clause`. 999991787 text bytes.
- GlotLID `model.bin` is the stable v3 file, 1687094687 bytes. `model_v4_alpha.bin` was not downloaded.

Acceptance:
- `make storage-plan` → exit 0.

```
wrote /Users/jarvis/Downloads/ancoda-tokenizer/docs/STORAGE.md
free_bytes=162796093440
approved_peak=97376647800
download_peak=84240596096
remain_bytes=65419445640
download_remain_bytes=78555497344
storage floor ok
```

- `make download` → exit 0. Closing lines:

```
complete sangraha_nep docs=8588860 text_bytes=44194985577 file_bytes=10487075578
free_bytes=126736797696
```

- `uv run python -m ancoda_tok.download --report` → exit 0.

```
source_id	role	docs	text_bytes	file_bytes	revision
fw2_npi	train	4888163	26982905157	6368079561	af9c13333eb981300149d5ca60a8e9d659b276b9
fw2_npi_test	eval	49462	274887739	65064803	af9c13333eb981300149d5ca60a8e9d659b276b9
fw2_npi_latn_test	eval	24	94370	47201	af9c13333eb981300149d5ca60a8e9d659b276b9
iriis	train	6391117	27558449602	6252610322	167b56b327996ca2c5fe7f9a6b638aecbb2ae13b
sangraha_nep	train	8588860	44194985577	10487075578	8b813c3f62d37b2fa174d68c31e8b35ae2fe85e8
indiccorp_npi	train	25263288	14722609810	3966527602	2d7285e6ce14fdb3fb2449c9f89427b9f582ac3f
hplt_npi	train	2777574	19229744005	4489415257	d1324a5283f762ee62c2a5c81de08fc6450ea540
fwedu_en	train	628310	2999997165	1252389320	87f09149ef4734204d70ed1d046ddc9ca3f2b8f9
fw2_hin	train	228083	1499999812	374824191	af9c13333eb981300149d5ca60a8e9d659b276b9
fw2_hin_test	eval	55242	325389065	81486219	af9c13333eb981300149d5ca60a8e9d659b276b9
sangraha_hin	train	88742	499994647	123342921	8b813c3f62d37b2fa174d68c31e8b35ae2fe85e8
fw2_mar	train	66457	499998842	123905273	af9c13333eb981300149d5ca60a8e9d659b276b9
sangraha_mar	train	87445	499997960	122235903	8b813c3f62d37b2fa174d68c31e8b35ae2fe85e8
fw2_san	train	21453	350803610	86178704	af9c13333eb981300149d5ca60a8e9d659b276b9
sangraha_san	train	43243	499941093	129264534	8b813c3f62d37b2fa174d68c31e8b35ae2fe85e8
deva_pool_fw2	train	181364	993647062	217308827	af9c13333eb981300149d5ca60a8e9d659b276b9
deva_pool_sangraha	train	37654	201536484	56120740	8b813c3f62d37b2fa174d68c31e8b35ae2fe85e8
deva_pool_hplt	train	64143	524927043	127649283	d1324a5283f762ee62c2a5c81de08fc6450ea540
owm	train	130188	999892378	378229220	fde8ef8de2300f5e778f56261843dab89f230815
code	train	144967	999991787	238075921	c48d40f9e70f0196f8236901ee35807f7d6c44c0
c4_val	eval	23235	49999383	21769675	1588ec454efa1a09f29cd18ddd04fe05fc8653a2
belebele_npi	decontam	900	1264981	208485	7899cdfa4e1e0d733fd77c848e2c273cb1d32be2
aya_npi	eval	4002	3911016	1252763	f9ea04583f02a8f86404ff6c58bf75fe637df8a2
glotlid	tool	0	0	1687094687	baa20d139151a26a71a20eddabf30d4de7329fc7
papaya_affixes	tool	0	0	573882	b3498528daa3435f091792e0b44bb222a2f8cb68
nepal_pool	train	18729	136203187	31547096	af9c13333eb981300149d5ca60a8e9d659b276b9
free_bytes=126733250560
```

- Free space after the report: 126.733 GB. Floor is 50 GB.
- `ruff check` passed. `pytest`: 21 passed.

## setup — Audit, environment and repo — done
- Commits: `02fe1ab` (setup), `0d1deee` (PLAN v2.6 sync)

## baselines — Baselines on Yak's harness — done
- [x] 1. `eval_panel.py fetch` then `run` inside `third_party/yak-tokenizer` (caffeinate)
- [x] 2. Harness validity gate **5/5** (±0.03 Arkios-paper check)
- [x] 3. `docs/LICENSES.md` — Yak CC BY-NC; sidskarki community-use-1.0 allows eval+proxy under ≤$100k revenue + attribution (terms clear; no STOP); Himalaya license not stated; Arkios Apache-2.0
- [x] 4. `artifacts/eval/baseline_RESULTS.md` + `baseline_RESULTS.manifest.json`; copy also in `docs/RESULTS.md` with baselines notes
- [x] 5. Gate A comparison values → `configs/eval.yaml` (incl. orphan/ZWJ/ZWNJ from baselines harness)
- [x] 6. `python/ancoda_tok/eval/gate_a.py` — 3.10 harness rows + Yak scoreboard; Ancoda-only rows PENDING until final-build
- [x] Acceptance: `artifacts/eval/baselines_reproduction.tsv` all PASS within ±0.003; `uv run python -m ancoda_tok.eval.gate_a --name yak-64k-v2.0` runs

Acceptance:
- Harness validity gate → **5/5 reproduced** (`docs/RESULTS.md`)
- `artifacts/eval/baselines_reproduction.tsv` → **18/18 PASS** (abs diff ≤ 0.003); FAIL count `0`
- `uv run python -m ancoda_tok.eval.gate_a --name yak-64k-v2.0` → prints 3.10 rows + Yak scoreboard (see `artifacts/eval/baselines_gate_a_yak64v2.txt`)
- Panel mapping: PLAN “Yak v1 128k” → `yak-128k-v1`; “Yak v2 128k” → `yak-128k-v2.0`; “Yak v1 128k-phrase” → `yak-128k-v1-phrase` (recorded in `configs/eval.yaml`)
- Extra recorded: Yak v1 64k-phrase Nepali ≈ 1.373937; Sanskrit harness values in eval.yaml; languages-of-Nepal **not comparable**
- `airavata` failed gated 403 (not in acceptance table)

Needs the human:
- Confirm Ancoda Labs annual revenue ≤ USD $100,000 before using sidskarki-64k as model-eval proxy arm (license allows eval/proxy only under that threshold). Not an baselines blocker.
- HimalayaAI Hub repos state no license; comparison-only; do not redistribute.

## data-plan — Storage plan, downloads, license log — done
- [x] 1. Storage plan from Hub `list_repo_tree` sizes → `docs/STORAGE.md`. Option 2 approved; gated peak leaves 65.419 GB
- [x] 2. `configs/sources.yaml` and `python/ancoda_tok/download.py`
- [x] 3. Downloads under `caffeinate`, one source at a time, shard resume, Hub cache deleted after each source's shards verify. Training raw is still on disk (deletion is cleaning, after a verified clean copy)
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
wrote $REPO_ROOT/docs/STORAGE.md
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

## pretokenizer — Pre-tokenizer, segmenter, export check — done
- [x] 1. `pattern.py` builds the 3.1 pattern (both o200k sources, asserted replacement counts) and writes `configs/pretokenizer_pattern.txt`
- [x] 2. `ancoda-seg`: `pretokenize` (tokenizers Split / onig), `aksharas` (3.3), maturin bindings
- [x] 3. Segmenter tests: Unicode 16 `GraphemeBreakTest.txt`, override property, 3.3 table, `tests/data/akshara_cases.tsv` (200+)
- [x] 4. Pre-tokenizer tests: 3.1 examples; onig, Python `regex`, and fancy-regex 0.19.0 (tiktoken 0.14.0's version) agree on 100k raw lines and `pretok_adversarial.tsv`
- [x] 5. Export check: toy vocab plus 1,000 random vocabularies × 200 words; full 3.5 merges mismatch count 0; training-merges-only at least 1
- [x] 6a. Independent Python reference `eval/ref_segmenter.py` from Unicode 16.0.0 property files
- [x] 6b. Python reference `gb9c=true` passes `GraphemeBreakTest.txt`
- [x] 6c. Differential test, 0 disagreements, both modes. The cleaning training snapshot does not exist yet; 1M distinct words are sampled from training-role raw parquet (seed 20261104), plus every distinct word in role=eval sources and the harness FLORES+ dev/devtest files
- [x] 6d. ICU, if `icu4c` and PyICU install
- [x] Acceptance commands pasted below

Measured:
- `configs/pretokenizer_pattern.txt` is 317 bytes, sha256 `01bacb48139cdf6f483799bcec4202907b5e902fb1c3e837c76ef0bc9566306b`, no trailing newline. Both o200k sources matched. The edit counts are 4 (`\p{M}]` → `\p{M}\u200C\u200D]`) and 1 (`\p{N}{1,3}` → `\p{N}`).
- Devanagari letters (Lo ∩ Script=Devanagari): 89. `GraphemeBreakTest.txt`: 1093 cases, 100% in Rust `gb9c=true` and in the Python reference. `tests/data/akshara_cases.tsv`: 328 cases. `tests/data/pretok_adversarial.tsv`: 39 rows.
- Cross-engine pre-tokens: 100039 lines (100000 reservoir-sampled + 39 adversarial), mismatches 0. The sample is Algorithm R with `random.Random(20261104)`: 33334 Nepali lines (`fw2_npi`, `iriis`, `sangraha_nep`, `indiccorp_npi`, `hplt_npi`), 33333 Hindi (`fw2_hin`, `sangraha_hin`), 33333 English (`fwedu_en`). fancy-regex 0.19.0 is the crate in tiktoken 0.14.0's Cargo.toml. Engines: tokenizers 0.21.4 onig Split, Python regex 2026.9.29, fancy-regex 0.19.0.
- Export check, seed 20261104: 200201 words (1 toy word `abcd`, 200 toy random words, 1000 vocabularies × 200 words). Full-split mismatches 0. Training-merges-only mismatches 2151.
- Segmenter differential, both modes: 1000000 training words + 1833695 distinct eval words, disagreements 0. Training words are the first 1000000 distinct whitespace-separated words after shuffling training-role parquet paths with `random.Random(20261104)`. Eval words are every distinct word in role=eval sources plus harness FLORES+ dev and devtest, read in place. Results: `artifacts/eval/pretokenizer_pretokenizer.json`, `artifacts/eval/pretokenizer_segmenter.json`, and their manifests.
- ICU: `brew install icu4c` installed icu4c@79 79.1. PyICU 2.17 built against it. Character break iterator compared with `gb9c=true` on the same 2833695 words: 2831807 agree, 1888 disagree. ICU 79.1 reports Unicode 18.0. The disagreements are consonant + vowel sign + virama + consonant sequences. In Unicode 16.0.0, U+093F is InCB None (`DerivedCoreProperties.txt` `@missing` InCB is None; U+093E..U+0940 are Grapheme_Base), so GB9c does not join across it. Example: `दि्छ।` is `['दि्', 'छ', '।']` on Unicode 16 and `['दि्छ', '।']` in ICU 79.1. PyICU was uninstalled after the comparison. `uv sync` keeps `ancoda-seg` and does not build PyICU. Loading PyICU required preloading the keg-only dylibs `libicudata.79`, `libicuuc.79`, and `libicui18n.79`.
- `cargo fmt --all -- --check` passed. `cargo clippy --workspace --all-targets -- -D warnings` passed. `ruff check python tests` passed. The other pytest files passed: 24 passed in 2.63s (`tests/test_seg.py` excluded from that run because it is in the acceptance command below).

Acceptance:

`cargo test -p ancoda-seg`

```
    Finished `test` profile [unoptimized + debuginfo] target(s) in 0.18s
     Running unittests src/lib.rs (target/debug/deps/ancoda_seg-b6bb99b0068279b4)

running 11 tests
test tests::unicode_version_is_16_0_0 ... ok
test pretok::tests::pattern_file_uses_the_required_joiner_escapes ... ok
test pretok::tests::plan_31_examples ... ok
test pretok::tests::adversarial_file_engines_agree ... ok
test letters::tests::letter_set_matches_the_spec_examples ... ok
test segment::tests::default_mode_only_adds_devanagari_letter_boundaries ... ok
test segment::tests::plan_required_examples ... ok
test segment::tests::plan_boundary_table_and_checked_raw_clusters ... ok
test segment::tests::devanagari_block_joiners_are_marks_and_none_prepend ... ok
test segment::tests::akshara_cases_file ... ok
test segment::tests::grapheme_break_test_passes ... ok

test result: ok. 11 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.04s

   Doc-tests ancoda_seg

running 0 tests

test result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s
```

`uv run pytest tests/test_seg.py tests/test_pattern.py tests/test_export_equivalence.py`

```
============================= test session starts ==============================
platform darwin -- Python 3.11.18, pytest-9.2.0, pluggy-1.7.0
rootdir: $REPO_ROOT
configfile: pyproject.toml
plugins: hypothesis-6.170.14, anyio-4.16.0
collected 14 items

tests/test_seg.py ...........                                            [ 78%]
tests/test_pattern.py ..                                                 [ 92%]
tests/test_export_equivalence.py .                                       [100%]

======================== 14 passed in 364.59s (0:06:04) ========================
```

Cross-engine mismatches = 0.

## cleaning — Cleaning, decontamination, snapshots — done

- [x] 0. Task list; `configs/mix.yaml`; clean package CLI
- [x] 1–2. UTF-8 + GlotLID; max script-outside share 0.041 (deva_pool_sangraha) < 0.20 STOP bar
- [x] 3. Legacy fonts (dictionary detector)
  - Converter MIT approved; lexicon `artifacts/clean/preeti_lexicon.tsv` (49,830 forms, sha256 `2761f296d51b367f0418d2c6a2d6754c4540a88a0910f521e6628f78a04fd454`)
  - Logistic classifier v1 precision 0.05 → replaced by dictionary (DECISIONS.md 5 Nov 2026)
  - Dictionary v2 spot-check: precision **0.89**; known true Preeti **3/5** → below 0.99 → **filter disabled**; continue cleaning
- [x] 4. Dedup (exact + MinHash 5-gram 14×8 across Nepali; preference sangraha→iriis→indiccorp→fw2→hplt)
- [x] 5. Decontam (human-approved 7 Nov 2026 — DECISIONS.md): exact on all eval; 6–12 + 13-gram on FLORES+/Belebele/aya only; MinHash near-dup vs large web test sets; reroutes decontaminated (fixes n45 segment=200)
- [x] 6. Joiner augmentation on n50 only (1% of snapshot bytes from unused Nepali)
- [x] 7. Four snapshots under `data/snapshots/` (rebuilt after MinHash near-dup)
- [x] 8. Independent audit: exact / segment / 13-gram / minhash_eval all 0
- [x] Acceptance

### Spot-check v2 (exact)

```
flagged_reviewed=100
unflagged_reviewed=100
true_positives=89
false_positives=11
precision=0.89
known_true_preeti_caught=3/5
```

### Decision

legacy-font filter disabled: no detector reached 99% precision. Would-flag line counts recorded per source as a limitation (fw2_npi 35, iriis 0, sangraha_nep 104, indiccorp_npi 0, hplt_npi 426; total 565).

Large-set n-gram rules disabled (false-overlap ~58% on fw2_npi sample); exact-only + MinHash near-dup kept.

### MinHash near-dup vs large web test sets (docs dropped)

Against `fw2_npi_test` / `fw2_hin_test` / `c4_val` (same MinHash settings as step 4). Source: `artifacts/clean/minhash_eval_near_dup.json` + `minhash_eval_run.log`.

| source | dropped docs | scanned |
|---|---:|---:|
| sangraha_nep | 11,692 | 4,297,173 |
| iriis | 4,631 | 3,212,707 |
| hplt_npi | 2,331 | 883,600 |
| fw2_npi | 1,020 | 2,937,040 |
| indiccorp_npi | 236 | 21,578,116 |
| fw2_hin | 183 | 219,026 |
| sangraha_hin | 29 | 80,398 |
| deva_pool_fw2 | 4 | 178,425 |
| code | 3 | 144,907 |
| fwedu_en | 2 | 614,846 |
| owm | 1 | 128,504 |
| all other train + `_reroute/*` (82 dirs) | 0 | (fully scanned) |
| **total** | **20,132** | |

**Shard coverage:** every training source in `mix.yaml` and every non-empty `_reroute/` dir was processed; for each, `scanned = kept + dropped`. Log shows all pending shards completed (e.g. fw2_npi 227/227, iriis 194/194, sangraha_nep 331 pending → done, …). No source missing from the report; no scanned/kept+dropped mismatch.

### Decontam bytes removed by rule (share of input)

Input = kept + dropped after text decontam (pre-MinHash). Text rule bytes from `decontam_report.json` `per_eval`; MinHash from `minhash_eval_near_dup.json`. 13-gram drops: **0** bytes everywhere.

| source | exact | segment | 13-gram | MinHash | total drop | input | share |
|---|---:|---:|---:|---:|---:|---:|---:|
| code | 0 | 3,460,145 | 0 | 8,607 | 3,468,752 | 998,261,389 | 0.35% |
| deva_pool_fw2 | 0 | 18,415,865 | 0 | 13,650 | 18,429,515 | 964,462,702 | 1.91% |
| deva_pool_hplt | 0 | 4,390,001 | 0 | 0 | 4,390,001 | 504,986,458 | 0.87% |
| deva_pool_sangraha | 0 | 876,493 | 0 | 0 | 876,493 | 174,145,735 | 0.50% |
| fw2_hin | 0 | 134,286,991 | 0 | 592,394 | 134,879,385 | 1,485,712,545 | 9.08% |
| fw2_mar | 0 | 310,812 | 0 | 0 | 310,812 | 498,947,556 | 0.06% |
| fw2_npi | 0 | 74,311,674 | 0 | 5,074,838 | 79,386,512 | 15,279,926,039 | 0.52% |
| fw2_san | 0 | 2,517,196 | 0 | 0 | 2,517,196 | 348,213,007 | 0.72% |
| fwedu_en | 0 | 183,832,533 | 0 | 4,282 | 183,836,815 | 2,998,563,161 | 6.13% |
| hplt_npi | 23,899,670 | 53,985,656 | 0 | 16,879,648 | 94,764,974 | 6,412,988,278 | 1.48% |
| indiccorp_npi | 90,664 | 6,395,854 | 0 | 951,199 | 7,437,717 | 12,621,064,938 | 0.06% |
| iriis | 262,294 | 30,992,954 | 0 | 25,147,770 | 56,403,018 | 13,019,540,491 | 0.43% |
| nepal_pool | 0 | 42,486 | 0 | 0 | 42,486 | 134,371,824 | 0.03% |
| owm | 0 | 71,948,535 | 0 | 2,936 | 71,951,471 | 999,608,469 | 7.20% |
| sangraha_hin | 4,211 | 39,574,790 | 0 | 111,077 | 39,690,078 | 456,571,246 | 8.69% |
| sangraha_mar | 0 | 125,407 | 0 | 0 | 125,407 | 499,655,938 | 0.03% |
| sangraha_nep | 4,261,996 | 74,617,905 | 0 | 73,924,881 | 152,804,782 | 22,233,931,930 | 0.69% |
| sangraha_san | 0 | 820,025 | 0 | 0 | 820,025 | 479,330,094 | 0.17% |
| **total** | **28,518,835** | **700,905,322** | **0** | **122,711,282** | **852,135,439** | **80,110,281,800** | **1.06%** |

### n50 snapshot — bytes per language

Path: `data/snapshots/n50_32fd1b6a9c068400` (text_bytes **5,000,398,763**).

| bucket | bytes | docs |
|---|---:|---:|
| Nepali | 2,500,028,405 | 1,185,391 |
| English | 1,000,002,558 | 211,442 |
| Hindi | 500,005,528 | 520,979 |
| Code | 250,126,946 | 35,025 |
| Smaller Devanagari | 200,016,985 | 46,345 |
| Marathi | 150,142,261 | 87,741 |
| Sanskrit | 150,000,246 | 36,536 |
| Math | 150,013,422 | 21,320 |
| Languages of Nepal | 100,062,412 | 22,509 |

### Acceptance

`make snapshot` (rebuild after MinHash) + independent audit → four snapshots, free disk ≥ 50 GB:

```
snapshot n45 dir=n45_868dd1887992c328 docs=2099708 text_bytes=5000068588
snapshot n50 dir=n50_32fd1b6a9c068400 docs=2167288 text_bytes=5000398763
snapshot n60 dir=n60_cf8066dbbdd0da93 docs=2313669 text_bytes=5000710160
snapshot n50_nojoiner dir=n50_nojoiner_3e785f409c11d076 docs=2168971 text_bytes=5000095538
audit n45 {'exact': 0, 'segment': 0, '13gram': 0, 'minhash_eval': 0}
audit n50 {'exact': 0, 'segment': 0, '13gram': 0, 'minhash_eval': 0}
audit n60 {'exact': 0, 'segment': 0, '13gram': 0, 'minhash_eval': 0}
audit n50_nojoiner {'exact': 0, 'segment': 0, '13gram': 0, 'minhash_eval': 0}
all audits clean free_bytes=127512088576
```

`artifacts/clean/snapshots.json` audits are taken only from on-disk audit output (never hand-typed). n45/n50 were re-bound from `artifacts/clean/audit_run2.log` (current digests `n45_868dd1887992c328` / `n50_32fd1b6a9c068400`); n60 from `audit_n60.log`; n50_nojoiner from `audit_n50_nojoiner.log`. Each snapshot has `artifacts/clean/audit_<name>.result.json` with path+sha256 recorded under `audit_provenance`.

`docs/RESULTS.md` — section **cleaning — Training snapshots (Ancoda)** lists bytes per language per snapshot and per-source decontam byte shares.

`uv run pytest tests/test_clean.py`

```
........                                                                 [100%]
8 passed in 0.19s
```

### Ops note (human, 7 Nov 2026) — RAM / workers

While unattended: if any step uses **> 10 GB RAM** or is **killed for memory**, lower `ANCODA_WORKERS` from **5 → 3**, then **3 → 2** if needed, and **resume from the last finished shard**. Record each worker-count change here. Do **not** change dedup or decontamination settings. No worker reduction applied (default remains 5; MinHash near-dup peaked under ~6 GB with 5 workers).

## akshara-inventory — Akshara inventory and covered set — done

Pre-check (8 Nov 2026): unique `doc_id` == row count in every snapshot and every bucket, including Languages of Nepal (n45 25867, n50 22509, n60 15894, n50_nojoiner 22374). No document repeats.

- [x] 0. Task list; `python/ancoda_tok/inventory.py`; `make inventory`
- [x] 1. Count `aksharas()` over n50 → `artifacts/inventory/akshara_counts.tsv` (N=708613647, unique=14229)
- [x] 2. 3.6 selection for 32k / 64k / 128k → `covered_{32k,64k,128k}.json` (all stop at cap; merges = cap)
- [x] 3. Plot → `artifacts/inventory/coverage_vs_block1.svg` and `docs/figures/akshara_inventory_coverage_vs_block1.svg`
- [x] 4. `gb9c=true` → `akshara_counts_gb9c.tsv` + `covered_*_gb9c.json` + gb9c plot
- [x] Acceptance

Acceptance:

`make inventory` → exit 0.

```
count mode=default done=75/75 pending=0 workers=5
count complete docs=2167288 units=708613647 unique=14229
wrote .../akshara_counts.tsv sha256=35f33fe79bfc5d429db8a28a1ae757652e4c49db92520d0813844ad99e66d77e unique=14229
covered 32k gb9c=False merges=1638/1638 coverage=0.9996016503 stop=cap
covered 64k gb9c=False merges=3276/3276 coverage=0.9997618985 stop=cap
covered 128k gb9c=False merges=6553/6553 coverage=0.9998031057 stop=cap
count mode=gb9c done=75/75 pending=0 workers=5
count complete docs=2167288 units=630265342 unique=57131
wrote .../akshara_counts_gb9c.tsv sha256=8662a0e7254f6d93a90444c64cc4d8ba9de595027fdb7a5d3dcbb9590580b697 unique=57131
covered 32k gb9c=True merges=1638/1638 coverage=0.9903944853 stop=cap
covered 64k gb9c=True merges=3276/3276 coverage=0.9968672877 stop=cap
covered 128k gb9c=True merges=6553/6553 coverage=0.9989564205 stop=cap
wrote .../manifest.json
```

`uv run python -m ancoda_tok.inventory --verify-deterministic` → exit 0; all six covered JSON sha256s matched.

`uv run pytest tests/test_inventory.py` → 6 passed.

`docs/RESULTS.md` — section **akshara-inventory — Akshara inventory and Block 1 covered set** with tables and both plots.

## trainer — Trainer, exporter and the guarantee — accepted

- [x] 0. Task list
- [x] 0b. Human corrections (8 Nov 2026) recorded, implemented, and unit-tested before any tokenizer training
- [x] 1. `ancoda-seg` / count CLI: stream the n50 snapshot, pre-tokenize, write a sorted `(bytes, u64 count)` table and a manifest
- [x] 2. `ancoda-bpe` trainer: canonical encoder, Blocks 0 and 1, Block 2 with bans, tie-break, duplicate rule, config switches
- [x] 3. 1 GB slice benchmark; record peak RAM and time in `docs/MACHINE.md`; STOP if full-size peak would exceed 14 GB
- [x] 4. Train the 64k build on the 50% snapshot; write `ranks.json` and a manifest
- [x] 5. Exporters (Hugging Face tokenizer.json and tiktoken) per 3.5
- [x] 6. `--plain` validation vs Hugging Face `BpeTrainer` on the 1 GB slice → `artifacts/validation/plain_vs_hf.md`
- [x] 7. Guarantee tests (`tests/test_guarantee.py`) on the 64k build
- [x] 8. Acceptance commands recorded below

### Human corrections (8 Nov 2026) — recorded before training

The human corrected two readings. Section 3 of docs/SPEC.md is unchanged. The same note is in `docs/DECISIONS.md` and `docs/MODEL_CARD_NOTES.md`. No tokenizer training runs until both are unit-tested.

1. **Ban 1 (PLAN 3.4), second clause.** "A non-empty prefix (proper or complete) of the UTF-8 encoding of exactly one Unicode scalar value, starting at its lead byte" means `M = L+R` lies inside **one** character's encoding: `M` starts with a valid lead byte, every following byte is a valid continuation at that position, and `len(M)` is at most that lead's sequence length. Proper prefixes such as `E6 97` are allowed (`E6+97`, then `E6 97+A5` completes the character). Banned: a merge that joins part of one character to bytes outside it, starts at a continuation byte, or is invalid. The first clause is unchanged: both `L` and `R` may be complete valid UTF-8 strings. This is **not** "exactly one scalar has an encoding that starts with M" (that reading wrongly bans `E6 97`).

2. **Special tokens must not swallow ordinary text.** Equality and guarantee tests set tokenizers 0.23.3 `Tokenizer.encode_special_tokens = True` before `encode(..., add_special_tokens=False)`. With that flag, a literal special-token string such as `<|bos|>` is ordinary text, matching tiktoken `encode_ordinary`. The flag defaults to `False` and is **not** stored in `tokenizer.json`; `False` extracts added special tokens from the text. `add_special_tokens=False` still keeps the post-processor from inserting BOS. `load_hf` sets the flag. Documented in `docs/MODEL_CARD_NOTES.md`.

Counts (`uv run python -u -m ancoda_tok.counts`, caffeinate, exit 0, 113.8 s). Headers `ANCDACT1`.

| slice | unique | occurrences | key_bytes | text_bytes |
|---|---:|---:|---:|---:|
| n50 | 9041738 | 549487984 | 227794543 | 5000398763 |
| n50 1 GB prefix | 1950607 | 208066814 | 20046751 | 1000000996 |

1 GB prefix is whole documents in shard order, stopping at 1000000996 text bytes (195858 docs). Manifests: `artifacts/counts/n50/manifest.json`, `artifacts/counts/n50_1gb/manifest.json`.

Unit tests, before any tokenizer training:

```
cargo test -p ancoda-bpe --offline --lib
test result: ok. 12 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 11.08s

uv run pytest tests/test_trainer_corrections.py tests/test_counts_format.py -q
4 passed
```

Ban 1 cases in `ban::tests::human_cases`: E6+97 allowed, E697+A5 allowed, A5+20 banned, E697+41 banned, F0+9F allowed, complete+complete allowed. The 256×256 table matches that predicate. `tests/test_trainer_corrections.py` encodes every special-token name (alone, inside Nepali text, and all names in one string) and requires Hugging Face ids to equal tiktoken `encode_ordinary`. Reloading `tokenizer.json` without `load_hf` swallows `<|bos|>` to the special id; `load_hf` does not.

1 GB benchmark (`configs/bench_64k_1gb.yaml`): peak RSS 1,303,707,648 bytes (1.214 GiB), trainer 22 s. Key-byte ratio 11.363165. Projected full-size peak 14,814,245,355 bytes (13.797 GiB), under 14 × 1024³ = 15,032,385,536 by 218,140,181 bytes. Full n50 train then peaked at 4,078,977,024 bytes (3.800 GiB) in 68 s. Recorded in `docs/MACHINE.md`. Training was allowed.

Full 64k train (`configs/build_64k.yaml`, release, exit 0):

```
peak_rss_bytes=4078977024
learned=60847
duplicates=0
banned=93
stopped=filled
tokens=64512
elapsed_secs=68
```

Exports: `artifacts/tokenizers/ancoda-64k/` (`tokenizer.json`, tiktoken rank file, manifests). `load_hf` sets `encode_special_tokens = True`.

Plain vs Hugging Face (`artifacts/validation/plain_vs_hf.md`, status PASS). Both lists have 65,280 merges. They are not identical. The first difference is index 894, an equal-count tie (88816): ours ` I`+`f`, Hugging Face `t`+`he`. Every equal-count run before index 7761 is the same pair set. The first run that is not a permutation starts at index 7761, count 5842 (` qu`+`ot` versus `ot`+`ed`); 87 pairs then appear on only one side, from overlapping ties. 100,000 snapshot lines: 0 byte-segmentation mismatches, 0 decode mismatches.

Guarantee (`caffeinate -dimsu uv run pytest -s tests/test_guarantee.py`, exit 0):

```
======================== 2 passed in 431.88s (0:07:11) =========================
```

`artifacts/validation/guarantee.json`:

```
texts_checked_in_equality_pass: 1234979
equality_mismatches: 0
hf_roundtrip_failures: 0
tt_roundtrip_failures: 0
hf_covered_violations: 0
tt_covered_violations: 0
mark_initial_block1_or_block2: 0
coverage_failures: 0
unit_context_failures: 0
orphan_occurrences_npi_aya: 70
orphan_tokens_npi_aya: 392935
orphan_mark_share_npi_aya: 0.00017814651278201228
orphan_inside_covered: 0
tail_hf_rate_lo_hi: [0.8, 0.7624466571834992, 0.8353115727002968]
tail_tt_rate_lo_hi: [0.8, 0.7624466571834992, 0.8353115727002968]
snapshot_lines: 10000000
eval_texts: 132924
special_names: 1024
ban_predicate_exit: 0
failures: []
```

The tail interval is the 10M snapshot lines that contain an uncovered Devanagari akshara. Covered-akshara violations on those lines and on all eval texts are 0. Orphan list: `artifacts/validation/orphans.txt` (70 rows, none inside a selected multi-character unit). FLORES+ npi plus Aya orphan-mark share is 70 / 392935.

Acceptance: passed. mix-sweep has not been started.

Needs the human:
- Choose the model-eval tier: reduced, full, or none.

## mix-sweep — Mix sweep and ablation ladder — blocked (STOP human, option-3 diagnostics done)

- [x] 0. Task list; ladder/build configs; counts/guarantee accept non-n50 paths
- [x] 1. Count pre-tokens for n45, n60, n50_nojoiner
- [x] 2. Train final 64k on n45 / n50 / n60; export each
- [x] 3. Guarantee tests on n45 and n60 (n50 reused trainer); harness `--add anc64_n45/n50/n60`
- [x] 4. Apply 3.8 pick rule → **no mix eligible** (unchanged; no pick recorded)
- [x] 5. Ladder on n60 (diagnostic force) — all rungs including Unigram (subsampled)
- [x] 6. Harness + covered-violation metric + word splits
- [ ] 7. Joiner keep/drop decision — deferred (no pick)
- [x] 8. STOP (human): diagnostics reported; awaiting fix choice

### Mix sweep fertilities (Yak harness, FLORES+ whole-file)

| Mix | Nepali | English | Hindi | Gate A |
|---|---:|---:|---:|:---:|
| n45 | 1.526282 | 1.318798 | 1.362945 | FAIL (npi) |
| n50 | 1.515748 | 1.324962 | 1.365714 | FAIL (npi) |
| n60 | 1.492487 | 1.337564 | 1.368093 | FAIL (npi) |

### Option-3 diagnostics (8 Nov 2026)

- Unigram first attempt killed (PID 68728); BPE ladder + nepali-only + guarantees intact.
- Unigram rerun: `input_sentence_size=1000000`, shuffle on, peak RSS 7468384256 (~6.96 GiB) < 12 GiB cap. Marked `subsampled_for_memory`.
- Nepali-only ceiling: npi **1.404577** (under Gate A 1.445) but eng 2.244 / hin 1.591.
- Full tables in `docs/RESULTS.md` (mix-sweep Option 3) and `artifacts/eval/mix_sweep/diag_ladder_summary.json`.

Acceptance:
- STOP (human). No mix pick. No further training until reply.

Needs the human:
- Read option-3 tables. Choose a fix path (65% Nepali, cap-bound covered set, or other). Decisions unchanged until you say so.

## final-build — Final builds, Gate A, efficiency — done (Gate A written; STOP human for review)

- [x] Builds and exports: ancoda-32k, ancoda-64k (flagship), ancoda-ne-64k, ancoda-128k, ancoda-250k under `artifacts/tokenizers/final_build/`
- [x] Guarantees: covered violations 0, mark-initial 0, HF==tiktoken, round-trip, Block 1 coverage, orphan-inside-covered 0 (see `docs/GATE_A.md` and `docs/EVIDENCE_FINAL_BUILD.md`)
- [x] Harness + multilang + morph + efficiency rows; `make gate-a` wrote `docs/GATE_A.md`
- [x] Frozen scoreboard also in `README.md` (WIN/LOSE vs Yak / sidskarki / arkios / sarvam / …)
- [ ] STOP (human): Gate A review. Nepali fertility on ancoda-64k is 1.4509 (FAIL vs 1.445); English stretch also FAIL. Scoreboard still WIN on Nepali/Hindi/Devanagari mean vs Yak 64k-v1 and v2.0.

Acceptance:
- Gate A document and evidence hashes are recorded. Flagship files are frozen for model-eval comparison (arm A).

Needs the human:
- Gate A review before treating final-build as closed for product decisions. model-eval proxy runs proceed under the separate model-eval pre-registration.

## model-eval — Proxy pretraining — core A/B/C done (E-s1 still running)

Protocol: `docs/MODEL_EVAL_PREREG.md` (amended 10 Nov 2026 in commit `49a0437`). This does not change `docs/DECISIONS.md`. Primary decision rule applied on seeds 1–3; secondary arm E (yak-64k-v2.0 seed 1) is still training on $MODEL_EVAL_HOST.

- [x] Licence. Human confirmed Ancoda Labs' total annual revenue is at most USD $100,000. Yak (CC BY-NC 4.0) and sidskarki (community-use-1.0) allow a non-released research proxy and a private copy of the tokenizer file on the compute server. model-eval weights and third-party tokenizer files are not released.
- [x] `docs/MODEL_EVAL_PREREG.md` written before any training run; amendment records sidskarki 640/1000 round-trip, secondary lossless A vs C, Arm D skipped, eager training.
- [x] Shared stream and server tokenize. Stream `10fddf2e2d95ec356188958ce03bd1cb34636d6e3db1f5181dedb2182edd1bcd`, 2,477,651 docs, 7,189,336,358 bytes, overlap 0. Content tokens: ancoda-64k 1,085,350,393 (0.150967/byte), yak-64k-v1 1,086,527,299 (0.151130), sidskarki-64k 1,551,158,477 (0.215758, round-trip 640/1000), yak-64k-v2.0 1,103,061,378 (0.153430). Mac and server 10,000-doc counts match. Arm D skipped: no 64k plain tokenizer on n60j5.
- [x] Arm D stream tokenize (added 18 Nov 2026 under the 15 Nov amendment). Same stream `10fddf2e…dd1bcd`, 2,477,651 docs, 7,189,336,358 bytes. Content tokens: plain-64k 1,052,664,192 (0.146420/byte); stream tokens 1,055,141,843; uint32, end-of-text id 65536 (extended). Source: server `~/model_eval/tokens/plain-64k/manifest.json` (sha256 `e18b9acda33472ffaf1ad3118d1a2e0cb78f8a97aa1fa0d225c6fca6997cf6d4`, finished 2026-11-15T10:41:36Z, git `edb46d7`).
- [x] Pilot (eager). A-s1 step 0 loss 11.183 → step 300 loss 6.190 (~4.68 s/step). Resume proven from `latest.pt` at completed_steps=301. C-s1 50 steps: 11.172 → 7.636. BPB wrote `pilot-ancoda-64k-s1-step301.json` (32 groups; sealed Nepali BPB 0.716). `torch.compile` unavailable (no C compiler). Peak CPU temp during pilot/full A-s1 probe: 53°C. A-s1 full run passed step 100 with loss 7.662 (~28.1k tok/s).
- [x] Full runs A/B/C × seeds 1–3 (final+bstar BPB). Driver `screen m9full` / `~/model_eval/run_full.sh`. B*: seed1 4813230411, seed2 4795818967, seed3 4805817305. E-s1 (yak-64k-v2.0) still running after the core trio. Crontab `@reboot` installed. Mac-side SSH intermittent; queue survived via screen + resume.
- 11 Nov 2026 (Mac sleep prep). Active: ancoda-64k-s1 at step 7100 / 7630, loss ≈ 3.77, ~4.66 s/step. `latest.pt` fresh; free disk ≈ 56 GB; `bstar.json` present. `@reboot` crontab was present but inert (marker `# model-eval-full-driver` commented out the `screen` start); fixed to a trailing-comment form without interrupting the run. Stop conditions: non-finite loss and step-500 loss ≥ 8.0 exit train (`train.log`); disk < 10 GB and 3 crashes write `STOP:` to `queue.log` and exit the driver. Snapshot under `docs/model_eval_logs/`; live rsync target `artifacts/model_eval/server_copy/`. Watcher: `scripts/model_eval/watch_model_eval_while_away.sh` (30 min poll; never restarts the server).
- [x] model-eval DONE ancoda-64k-s1 at 2026-11-11T04:15:39Z (final+bstar evals written). Seed1 A finished.
- [x] model-eval DONE yak-64k-v1-s1 at 2026-11-11T14:32:22Z (final+bstar evals written).
- [x] model-eval DONE sidskarki-64k-s1 at 2026-11-12T00:49:51Z (final+bstar evals written). Seed1 A/B/C complete.
- [x] model-eval DONE ancoda-64k-s2 at 2026-11-12T11:06:08Z (final+bstar evals written).
- 12 Nov 2026 SSH restored after ~24h blackout. Active: yak-64k-v1-s2 (started 2026-11-12T11:06:08Z). Queue untouched. Disk ~49 GB free. Snapshot refreshed in docs/model_eval_logs/.
- Seed 1 BPB (sealed.jsonl). Equal tokens (final): Nepali A=0.466263 B=0.468417 C=0.485965 (A−B=−0.002153, A−C=−0.019702); Dev11 mean A=0.748445 B=0.750504 C=0.781543 (A−B=−0.002059, A−C=−0.033098). Equal bytes (B*): Nepali A=0.476757 B=0.479207 C=0.485965 (A−B=−0.002450, A−C=−0.009208); Dev11 A=0.767661 B=0.770013 C=0.781543 (A−B=−0.002352, A−C=−0.013882). Secondary lossless A vs C: equal-tokens Nepali A−C=−0.020597, Dev11 A−C=−0.033150; equal-bytes Nepali A−C=−0.009917, Dev11 A−C=−0.013940. Sealed Nepali sidskarki round-trip 30572/36094 (0.847).
- [x] model-eval DONE yak-64k-v1-s2 at 2026-11-12T21:23:07Z; sidskarki-64k-s2 at 2026-11-13T07:40:06Z (final+bstar). Seed2 A/B/C complete.
- Seed 2 BPB (sealed.jsonl). Equal tokens: Nepali A=0.464589 B=0.466887 C=0.484460 (A−B=−0.002298, A−C=−0.019871); Dev11 A=0.744314 B=0.747091 C=0.777994 (A−B=−0.002777, A−C=−0.033680). Equal bytes: Nepali A=0.475415 B=0.477750 C=0.484460 (A−B=−0.002335, A−C=−0.009045); Dev11 A=0.760261 B=0.763923 C=0.777994 (A−B=−0.003662, A−C=−0.017733). Secondary lossless A vs C: equal-tokens Nepali A−C=−0.020918, Dev11 A−C=−0.033435; equal-bytes Nepali A−C=−0.009787, Dev11 A−C=−0.017404. Nepali sidskarki round-trip 30572/36094 (0.847).
- [x] model-eval DONE ancoda-64k-s3 2026-11-13T17:55:55Z; yak-64k-v1-s3 2026-11-14T04:13:11Z; sidskarki-64k-s3 2026-11-14T14:29:38Z. Seed3 A/B/C complete. E-s1 (yak-64k-v2.0) started 2026-11-14T14:29:38Z.
- Seed 3 BPB (sealed.jsonl). Equal tokens: Nepali A=0.465655 B=0.467030 C=0.484676 (A−B=−0.001375, A−C=−0.019021); Dev11 A=0.745996 B=0.747682 C=0.778413 (A−B=−0.001687, A−C=−0.032418). Equal bytes: Nepali A=0.476057 B=0.478175 C=0.484676 (A−B=−0.002118, A−C=−0.008619); Dev11 A=0.765584 B=0.770476 C=0.778413 (A−B=−0.004892, A−C=−0.012829). Secondary lossless A vs C: equal-tokens Nepali A−C=−0.019980, Dev11 A−C=−0.032305; equal-bytes Nepali A−C=−0.009390, Dev11 A−C=−0.012856.
- [x] Pre-registered decision rule (`docs/MODEL_EVAL_PREREG.md` + amendment `49a0437`) on seeds 1–3. All eight primary labels are **A better** (every paired diff A−other strictly negative and |mean| > sample sd). Equal-tokens Nepali A−B diffs [−0.002153, −0.002298, −0.001375] mean −0.001942 sd 0.000497; Nepali A−C [−0.019702, −0.019871, −0.019021] mean −0.019531 sd 0.000450; Dev11 A−B [−0.002059, −0.002777, −0.001687] mean −0.002174 sd 0.000554; Dev11 A−C [−0.033098, −0.033680, −0.032418] mean −0.033065 sd 0.000632. Equal-bytes Nepali A−B [−0.002450, −0.002335, −0.002118] mean −0.002301 sd 0.000168; Nepali A−C [−0.009208, −0.009045, −0.008619] mean −0.008958 sd 0.000304; Dev11 A−B [−0.002352, −0.003662, −0.004892] mean −0.003635 sd 0.001270; Dev11 A−C [−0.013882, −0.017733, −0.012829] mean −0.014815 sd 0.002581. Secondary lossless A−C (not primary): equal-tokens Nepali mean −0.020498, Dev11 mean −0.032963; equal-bytes Nepali mean −0.009698, Dev11 mean −0.014733. Sealed Nepali sidskarki round-trip 30572/36094 (0.847) on all seeds.
- E-s1 status (secondary arm): training in progress at check (~step 2801/7630, ~28.1k tok/s, ~6.3 h left); not part of the primary decision.
- [x] model-eval evaluation writeup 15 Nov 2026. All 10 jobs DONE (`DRIVER_ALL_OK` 2026-11-15T00:46:11Z). Verification PASS (denominator=original UTF-8 bytes; EOS context-only; same eval docs; independent 300-doc recompute within 0.001). Pre-registered rule: A better vs B and vs C on sealed Nepali and Dev11 at equal-tokens and equal-bytes. Results: `docs/MODEL_EVAL_RESULTS.md`, `artifacts/model_eval/results.json`.
- [x] Mac re-verification 15 Nov 2026 (MPS). Read-only rsync of `checkpoint-007630.pt` for ancoda-64k-s1, yak-64k-v1-s1, and sidskarki-64k-s1. Server queue untouched. First 300 sealed Nepali documents by `doc_id` (1,652,418 bytes). Pipeline vs standalone BPB: ancoda 0.487752 vs 0.487738 (|Δ|=0.000014); yak-64k-v1 0.490893 vs 0.490874 (|Δ|=0.000019); sidskarki 0.511574 vs 0.511571 (|Δ|=0.000003). PASS within 0.001.

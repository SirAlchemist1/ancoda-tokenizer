# Licenses (baselines comparison tokenizers)

Columns: dataset/model ID, revision (commit), license as read on the card, card URL, attribution required, share-alike, redistribution permitted, training use permitted, notes for releasing our tokenizer.

Yak files are **evaluation / proxy-model comparison only** (CC BY-NC 4.0). Never copy Yak, sidskarki, HimalayaAI or Arkios vocab, merges, regex or settings into our build.

| ID | Kind (eval_panel) | Revision | License (as read) | Card URL | Attribution | Share-alike | Redistribution | Training use | Notes |
|---|---|---|---|---|---|---|---|---|---|
| `ampixa/yak-tokenizer` (all Yak panel builds) | `hf_json` (`v1.0/…`, `v2.0/…`) | `12b6d3010ec542b484ec3b673354852fe8bb85fb` | CC BY-NC 4.0 | https://huggingface.co/ampixa/yak-tokenizer | Yes | No (NC) | Non-commercial only | **No** — evaluation / research comparison only | Panel names: `yak-32k-v1`, `yak-64k-v1`, `yak-128k-v1`, `yak-64k-v1-phrase`, `yak-128k-v1-phrase`, `yak-64k-v2.0`, `yak-128k-v2.0` |
| `sidskarki/nepali-bpe-tokenizer` (`nepali_bpe_64k.model`) | `sentencepiece` | `fc5773dcf7b9dc2272e8f595f99828969621a855` | other / **community-use-1.0** (`LICENSE.md`) | https://huggingface.co/sidskarki/nepali-bpe-tokenizer | Yes (credit + link + note changes) | Yes (derivatives under same/compatible terms) | Yes under license conditions | Not for our training corpus; comparison arm only | **Evaluation and proxy-model comparison are allowed** as “use” under community-use-1.0 for Licensees with total annual revenue ≤ USD $100,000, with attribution. Above-threshold entities need a separate written license. Confirm Ancoda Labs revenue compliance before model-eval arm 6. |
| `himalaya-ai/nepali-tokenizer-bpe-64k-5b` | `sentencepiece` (`…-64000.model`) | `1c39f7a61b78211e3c5bce64e29b5ffd10b438d5` | **Not stated** on Hub card or repo files (baselines) | https://huggingface.co/himalaya-ai/nepali-tokenizer-bpe-64k-5b | Unknown | Unknown | Unknown | No (comparison only) | No LICENSE file in repo. Comparison arm only; do not copy into build; do not redistribute. |
| `himalaya-ai/nepali-tokenizer-bpe-100k-5b` | `sentencepiece` (`…-100000.model`) | `83beb6d9e7a9af6f8449399aa5e16b438f7cf396` | **Not stated** (baselines) | https://huggingface.co/himalaya-ai/nepali-tokenizer-bpe-100k-5b | Unknown | Unknown | Unknown | No | Same as 64k. |
| `himalaya-ai/nepali-tokenizer-bpe-128k-5b` | `sentencepiece` (`…-128000.model`) | `c525ebe0b8f72705e549e6e62d36fb1b6a495df3` | **Not stated** (baselines) | https://huggingface.co/himalaya-ai/nepali-tokenizer-bpe-128k-5b | Unknown | Unknown | Unknown | No | Same as 64k. |
| `sajalregmi4/arkios-tokenizer` | `hf_json` | `aef562fcaf8747bbd9461cdae61b61f99e97e048` | Apache-2.0 | https://huggingface.co/sajalregmi4/arkios-tokenizer | Yes (NOTICE if present) | No | Yes (Apache-2.0) | Not used for our training; comparison arm | Panel name `arkios`. |

## sidskarki community-use-1.0 — evaluation / proxy verdict (baselines)

Read from Hub `LICENSE.md` at revision above:

- Grant covers use, copy, modify, merge, publish, and distribute, subject to attribution, ≤ $100k annual revenue, no sublicensing to above-threshold entities, and share-alike for derivatives.
- **Evaluation on this harness** and **proxy-model comparison (model-eval)** are forms of use → **permitted** when the revenue condition holds and attribution is given.
- Terms are clear; no baselines STOP for ambiguity. Human must confirm revenue ≤ $100k before relying on model-eval arm 6.

## Harness

| ID | Revision | Notes |
|---|---|---|
| `Ampixa/yak-tokenizer` git clone (`third_party/yak-tokenizer`) | `9fe03a991dd7c44e078d574e4ffcd51a7606dd6a` | Eval harness only; not a training source. |

## data-plan training and eval sources

The human approved option 2, the ten FineWeb-2 `nepal_pool` configs, Romanized Nepali as eval-only, and the Papaya affix inventory as CC BY 4.0 evaluation data on 4 Nov 2026. See `docs/DECISIONS.md`. Each `data/raw/<source_id>/manifest.json` records the revision below. Text stays under `data/` and is not committed.

| ID | Revision | License (as read) | Card URL | Attribution | Share-alike | Redistribution | Training use | Notes |
|---|---|---|---|---|---|---|---|---|
| `HuggingFaceFW/fineweb-2` | `af9c13333eb981300149d5ca60a8e9d659b276b9` | ODC-By 1.0, plus Common Crawl terms of use | https://huggingface.co/datasets/HuggingFaceFW/fineweb-2 | Yes | No | Yes, with attribution | Yes for train configs; `npi_Latn` test is eval only | Configs: `npi_Deva`, `hin_Deva`, `mar_Deva`, `san_Deva`, the smaller `*_Deva` pool, and nepal_pool train splits `new_Deva`, `thl_Deva`, `taj_Deva`, `lif_Deva`, `rav_Deva`, `suz_Deva`, `xsr_Deva`, `rjs_Deva`, `kle_Deva`, `nwx_Deva`. `npi_Latn` test only. `_removed` folders and `npi_Latn` train are not used. |
| `HuggingFaceFW/fineweb-edu` | `87f09149ef4734204d70ed1d046ddc9ca3f2b8f9` | ODC-By 1.0, plus Common Crawl terms of use | https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu | Yes | No | Yes, with attribution | Yes | Config `sample-10BT` only. Cap 3 GB of text. |
| `IRIIS-RESEARCH/Nepali-Text-Corpus` | `167b56b327996ca2c5fe7f9a6b638aecbb2ae13b` | MIT (card tag and `LICENSE` file) | https://huggingface.co/datasets/IRIIS-RESEARCH/Nepali-Text-Corpus | Yes (copyright notice) | No | Yes | Yes | Card says the text was collected from Nepali news sites and blogs. Column `Article`. |
| `ai4bharat/sangraha` | `8b813c3f62d37b2fa174d68c31e8b35ae2fe85e8` | CC BY 4.0 | https://huggingface.co/datasets/ai4bharat/sangraha | Yes | No | Yes | Yes | Config `verified` only. Splits `nep` (all), `hin`/`mar`/`san` (capped), and pool splits `brx`, `doi`, `gom`, `mai`, `kas`, `snd`. |
| `ai4bharat/IndicCorpV2` | `2d7285e6ce14fdb3fb2449c9f89427b9f582ac3f` | README license section: CC-0. No `license` tag in the Hub YAML | https://huggingface.co/datasets/ai4bharat/IndicCorpV2 | No (CC-0 waiver, as written) | No | Yes | Yes | Config `indiccorp_v2`, split `npi_Deva` = `data/ne.txt`. Only that file. |
| `HPLT/HPLT2.0_cleaned` | `d1324a5283f762ee62c2a5c81de08fc6450ea540` | Card tag CC0-1.0. README: packaging is CC0; the project does not own the underlying text; takedown policy | https://huggingface.co/datasets/HPLT/HPLT2.0_cleaned | No for the packaging | No | Yes | Yes for the released corpus, with the README notice that source copyright is not transferred | Official HPLT org repo. Folders `npi_Deva` (all) and pool `awa_Deva`, `bho_Deva`, `hne_Deva`, `kas_Deva`, `mag_Deva`, `mai_Deva`. `hin_Deva`, `mar_Deva`, `san_Deva` are not in the data-plan source rows. |
| `open-web-math/open-web-math` | `fde8ef8de2300f5e778f56261843dab89f230815` | README: ODC-By 1.0, plus Common Crawl terms of use. Hub `card_data` had no license field | https://huggingface.co/datasets/open-web-math/open-web-math | Yes | No | Yes, with attribution | Yes | Cap 1 GB of text. |
| `codeparrot/github-code-clean` | `c48d40f9e70f0196f8236901ee35807f7d6c44c0` | Card tag Apache-2.0. Each row has its own `license` | https://huggingface.co/datasets/codeparrot/github-code-clean | Per row | No for the four kept licenses | Yes under the row license | Yes, only `mit`, `apache-2.0`, `bsd-2-clause`, `bsd-3-clause` | Shard `data/train-00000-of-00880.parquet` columns: `code`, `repo_name`, `path`, `language`, `license`, `size`. |
| `allenai/c4` | `1588ec454efa1a09f29cd18ddd04fe05fc8653a2` | ODC-By | https://huggingface.co/datasets/allenai/c4 | Yes | No | Yes, with attribution | No — eval only (`en` validation, 50 MB) | Do not download the train shards. |
| `facebook/belebele` | `7899cdfa4e1e0d733fd77c848e2c273cb1d32be2` | CC BY-SA 4.0 | https://huggingface.co/datasets/facebook/belebele | Yes | Yes | Yes, under CC BY-SA | No — decontamination only | File `data/npi_Deva.jsonl`. Do not commit the text. |
| `CohereLabs/aya_dataset` | `f9ea04583f02a8f86404ff6c58bf75fe637df8a2` | Apache-2.0 | https://huggingface.co/datasets/CohereLabs/aya_dataset | Yes | No | Yes | No as a training corpus — eval and decontamination. Keep rows with `language_code` `nep` | Card language table: `` `nep` `` is Nepali. |
| `cis-lmu/glotlid` (model) | `baa20d139151a26a71a20eddabf30d4de7329fc7` | other / Apache-2.0 plus notices (`license_name` on the card) | https://huggingface.co/cis-lmu/glotlid | Yes | No | Yes, under Apache-2.0 plus the LICENSE notices | No — language-ID tool | Card: `model.bin` is stable v3. `model_v4_alpha.bin` is a pre-release and is not used. |
| `kalexrt/nepali-morphology-tokenizer` | `b3498528daa3435f091792e0b44bb222a2f8cb68` | Code: MIT (`LICENSE`). Affix inventory: CC BY 4.0 (`data/LICENSE`) | https://github.com/kalexrt/nepali-morphology-tokenizer/tree/b3498528daa3435f091792e0b44bb222a2f8cb68 | Yes | No | Yes | No — eval reference (3.9), approved 4 Nov 2026 | Commit `b3498528daa3435f091792e0b44bb222a2f8cb68`. Files kept under `data/`, not committed: `data/affix_inventory.tsv`, `data/affix_inventory.json`. Attribution, from that repo's `data/LICENSE`: the affix dataset is re-derived from Balaram Prasain, "A Computational Analysis of Nepali Morphology: A Model for Natural Language Processing", PhD dissertation, Tribhuvan University, 2011. Cite that work. Do not copy the repo's tokenizer vocab, merges or stem lexicons. |

### nepal_pool and Romanized Nepali

Approved 4 Nov 2026. `nepal_pool` is the ten FineWeb-2 train splits named above, ODC-By, with no upsampling. Unused pool share goes to Nepali (docs/SPEC.md 3.8). `ilprl-docse/NepTam-A-Nepali-Tamang-Parallel-Corpus` is CC BY-NC 4.0 and is excluded. `eduhk-compling/GurungJasmika` is CC BY 4.0 English audio and is excluded. No separate Magar corpus with a training license was found.

Romanized Nepali is out of scope for training in v1. FineWeb-2 `npi_Latn` has a test split (`data/npi_Latn/test/000_00000.parquet`, 54981 bytes at the size query); that split is eval-only. The train split is not downloaded. `dipeshch71/nepaliflow-romanized-nepali-to-devanagari-dataset` and `sabin1234/NEPALI_TO_ROMANIZED_WIKIPEDIA` are not used.

## cleaning tools

| ID | Revision | License (as read) | Card/URL | Attribution | Share-alike | Redistribution | Training use | Notes |
|---|---|---|---|---|---|---|---|---|
| `globalpolicy/UnicodeToPreeti` | `c1d6e3080976d7113c24ab5b59d06f937a20949d` | **MIT** | https://github.com/globalpolicy/UnicodeToPreeti | Yes (copyright notice) | No | Yes | No — conversion table only, to synthesize Preeti positives for the legacy-font classifier | Human advance approval (cleaning): MIT is in the allowed set (MIT, Apache-2.0, BSD, ISC, CC0, CC BY). Mapping table and convert logic vendored under `python/ancoda_tok/clean/preeti_convert.py` with the MIT notice. Never used as training text. |

## Loading (from `configs/eval_panel.yaml`)

| Panel name | Kind | How loaded |
|---|---|---|
| sidskarki-64k | sentencepiece | `sidskarki/nepali-bpe-tokenizer` file `nepali_bpe_64k.model` |
| himalaya-64k / 100k / 128k | sentencepiece | `himalaya-ai/nepali-tokenizer-bpe-{64k,100k,128k}-5b` files `himalaya-ai_nepali-tokenizer-corpus-{64000,100000,128000}.model` |
| arkios | hf_json | `sajalregmi4/arkios-tokenizer` |
| Yak builds | hf_json | `ampixa/yak-tokenizer` paths `v1.0/…` and `v2.0/…` as in eval_panel.yaml |

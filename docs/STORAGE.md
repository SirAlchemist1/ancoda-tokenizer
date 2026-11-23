# Storage plan (M2 step 1)

File sizes are Hub `list_repo_tree` sizes (exact). A bracketed docs/SPEC.md budget is a UTF-8 cap, and 1 GB = 10^9 bytes. Stored parquet of a capped source is at most that cap. Git commit `be29daf4cff030a2f71dcd8650aca204fd63859c`.

## Machine now

- RAM: 18.0 GiB (`sysctl -n hw.memsize` = 19327352832).
- Under 32 GB RAM, docs/SPEC.md 3.8 sets each snapshot to 5 GB, not 10 GB.
- Free space: 162.796 GB (151.616 GiB, 162796093440 bytes).

```
Filesystem   1024-blocks      Used Available Capacity iused      ifree %iused  Mounted on
/dev/disk3s5   482797652 298338612 158980560    66% 3117290 1589805600    0%   /System/Volumes/Data
```

## What would be stored

Upstream is the sum of the files under the paths in `configs/sources.yaml`. Stored is upstream for `all`, or the minimum of upstream and the cap. Transient is the Hub cache bound for that source alone; the cache is deleted after the source is verified.

| source | role | files | upstream | stored upper bound | cache upper bound |
|---|---|---:|---:|---:|---:|
| `fw2_npi` | train | 2 | 6683397712 | 6683397712 | 6683397712 |
| `fw2_npi_test` | eval | 1 | 69460662 | 69460662 | 69460662 |
| `fw2_npi_latn_test` | eval | 1 | 54981 | 54981 | 54981 |
| `iriis` | train | 57 | 10129505538 | 10129505538 | 10129505538 |
| `sangraha_nep` | train | 25 | 16863948296 | 16863948296 | 16863948296 |
| `indiccorp_npi` | train | 1 | 14773136386 | 14773136386 | 14773136386 |
| `hplt_npi` | train | 42 | 7439462246 | 7439462246 | 7439462246 |
| `fwedu_en` | train | 14 | 28518193415 | 3000000000 | 5152819114 |
| `fw2_hin` | train | 8 | 34272968716 | 1500000000 | 6344220906 |
| `fw2_hin_test` | eval | 1 | 88669788 | 88669788 | 88669788 |
| `sangraha_hin` | train | 100 | 37340748989 | 500000000 | 881744745 |
| `fw2_mar` | train | 2 | 6278522796 | 500000000 | 5342568266 |
| `sangraha_mar` | train | 34 | 12781930636 | 500000000 | 883328789 |
| `fw2_san` | train | 1 | 88005869 | 88005869 | 88005869 |
| `sangraha_san` | train | 11 | 4182068434 | 500000000 | 886344759 |
| `deva_pool_fw2` | train | 17 | 238244972 | 238244972 | 84196283 |
| `deva_pool_sangraha` | train | 7 | 1496140653 | 384849409 | 1240871338 |
| `deva_pool_hplt` | train | 6 | 205882794 | 205882794 | 96229209 |
| `owm` | train | 114 | 27431041597 | 1000000000 | 1245889392 |
| `code` | train | 880 | 313663761086 | 1000000000 | 1012632270 |
| `c4_val` | eval | 8 | 325887478 | 50000000 | 91175078 |
| `belebele_npi` | decontam | 1 | 3153357 | 3153357 | 3153357 |
| `aya_npi` | eval | 2 | 138173317 | 138173317 | 138173317 |
| `glotlid` | tool | 1 | 1687094687 | 1687094687 | 1687094687 |
| `papaya_affixes` | tool | 2 | 572208 | 572208 | 572208 |
| `nepal_pool` | train | 10 | 33035578 | 33035578 | 33035578 |

## Components

| component | bytes | notes |
|---|---:|---|
| raw parquet (stored upper bound) | 67376647800 | 67.377 GB (62.749 GiB, 67376647800 bytes) |
| transient Hugging Face cache | 16863948296 | largest single source, deleted before the next source. 16.864 GB (15.706 GiB, 16863948296 bytes) |
| cleaned parquet | 65339468800 | replaces training raw after it is verified. Not a second resident copy. 65.339 GB (60.852 GiB, 65339468800 bytes) |
| snapshots | 20000000000 | M4 writes 4 × 5000000000 bytes (5 GB because RAM < 32 GB). M2's wording of 3 snapshots would be 15000000000. |
| counts, tokenizers, eval | 10000000000 | planning allowance, not an API figure. 10.000 GB (9.313 GiB, 10000000000 bytes) |

## Approved retention (4 Nov 2026)

Option 2 is approved. For each training source, delete its raw parquet only after its cleaned parquet is written and verified against the manifest, and only after the exact Hugging Face revision (commit hash) is in that manifest. Never delete eval-only or decontamination-only sources. The 50 GB rule still applies before every large write. Caps and an external drive were not chosen.

## Peak and the 50 GB floor

The floor is 50 × 10^9 bytes. The gated peak is one copy of the stored files (training raw until the cleaned parquet replaces it), plus four snapshots, plus the counts allowance. Download peak is every raw output plus one source's Hub cache. The cache is deleted after that source's shards are verified.

- Approved peak (stored files + 4 snapshots + counts): 97.377 GB (90.689 GiB, 97376647800 bytes).
- Same peak with the 3 snapshots in the M2 wording: 92.377 GB (86.032 GiB, 92376647800 bytes).
- Download peak (raw + one cache): 84.241 GB (78.455 GiB, 84240596096 bytes).
- Free now minus approved peak: 65.419 GB (60.927 GiB, 65419445640 bytes).
- Free now minus download peak: 78.555 GB (73.161 GiB, 78555497344 bytes).
- Rejected peak if raw and cleaned were both kept: 162.716 GB (151.541 GiB, 162716116600 bytes), leaving 0.080 GB (0.074 GiB, 79976840 bytes).

The approved peak and the download peak each leave at least 50 GB free.

## Held, not in the total

None. Every row in `configs/sources.yaml` is in the total above.

## Revisions queried with the sizes

| repo | revision |
|---|---|
| `HuggingFaceFW/fineweb-2` | `af9c13333eb981300149d5ca60a8e9d659b276b9` |
| `IRIIS-RESEARCH/Nepali-Text-Corpus` | `167b56b327996ca2c5fe7f9a6b638aecbb2ae13b` |
| `ai4bharat/sangraha` | `8b813c3f62d37b2fa174d68c31e8b35ae2fe85e8` |
| `ai4bharat/IndicCorpV2` | `2d7285e6ce14fdb3fb2449c9f89427b9f582ac3f` |
| `HPLT/HPLT2.0_cleaned` | `d1324a5283f762ee62c2a5c81de08fc6450ea540` |
| `HuggingFaceFW/fineweb-edu` | `87f09149ef4734204d70ed1d046ddc9ca3f2b8f9` |
| `open-web-math/open-web-math` | `fde8ef8de2300f5e778f56261843dab89f230815` |
| `codeparrot/github-code-clean` | `c48d40f9e70f0196f8236901ee35807f7d6c44c0` |
| `allenai/c4` | `1588ec454efa1a09f29cd18ddd04fe05fc8653a2` |
| `facebook/belebele` | `7899cdfa4e1e0d733fd77c848e2c273cb1d32be2` |
| `CohereLabs/aya_dataset` | `f9ea04583f02a8f86404ff6c58bf75fe637df8a2` |
| `cis-lmu/glotlid` | `baa20d139151a26a71a20eddabf30d4de7329fc7` |
| `kalexrt/nepali-morphology-tokenizer` | `b3498528daa3435f091792e0b44bb222a2f8cb68` |

## Notes that are not size estimates

- `und_Deva` is a real FineWeb-2 config. The card calls it undetermined language and says it was uploaded without deduplication or filtering. It is inside `deva_pool_fw2` because M2 says every other `*_Deva` config. Its file size is exact; the decision to train on it is the plan's.
- Sangraha `verified/kas` and `verified/snd` have no script tag on the card. They are in the pool because FineWeb-2 has `kas_Deva` and `snd_Deva`. The first rows should be checked for Devanagari before those shards are kept.
- IndicCorp v2 `npi_Deva` is the file `data/ne.txt` (the README data_files map). It is plain text. Blank-line versus one-line documents is decided while reading the file, not guessed here.
- GlotLID `model.bin` is v3. The model card says `model.bin` stays at stable v3 while `model_v4_alpha.bin` is a pre-release.
- Papaya affix inventory is approved for evaluation only under CC BY 4.0 at commit `b3498528daa3435f091792e0b44bb222a2f8cb68`. Attribution: re-derived from Balaram Prasain, "A Computational Analysis of Nepali Morphology: A Model for Natural Language Processing", PhD dissertation, Tribhuvan University, 2011. The files stay under `data/` and are not committed. The byte total is the sum of the two inventory blobs (81127 + 491081), not a Hub tree size.
- `nepal_pool` is the ten approved FineWeb-2 train splits: `new_Deva`, `thl_Deva`, `taj_Deva`, `lif_Deva`, `rav_Deva`, `suz_Deva`, `xsr_Deva`, `rjs_Deva`, `kle_Deva`, `nwx_Deva`. No upsampling. NepTam and GurungJasmika are not included. FineWeb-2 `npi_Latn` test is eval-only; its train split is not downloaded.
- Code parquet shard `data/train-00000-of-00880.parquet` was opened. Columns are `code`, `repo_name`, `path`, `language`, `license`, `size`. Kept rows are mit, apache-2.0, bsd-2-clause and bsd-3-clause. On that shard, 550829383 kept UTF-8 bytes / 355122552 file bytes = keep_ratio 1.551095178. The cache bound uses that ratio. It is one shard, not a census of all 880 files.


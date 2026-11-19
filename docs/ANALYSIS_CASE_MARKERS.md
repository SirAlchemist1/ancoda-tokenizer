# Nepali case-marker boundaries

Analysis only. No training run, and the server was not used. Reproduction: `scripts/analysis_case_markers.py` (tokenizers 0.23.3). The morphology scorer in `python/ancoda_tok/eval/morph.py` is unchanged; this metric and its exclusion filter are separate functions in that module.

Words are whitespace-separated tokens from sealed Nepali (28,094 documents in the no-joiner sealed set) and from FLORES+ Nepali devtest (1,012 lines that contain a word). A word is a candidate when it ends in one of: ले, लाई, मा, को, का, की, बाट, देखि, सँग, सम्म, भन्दा. Longest match wins, and the stem (the word minus the marker) must be nonempty. No Unicode normalization is applied.

A candidate is frequency-kept when its stem occurs as a standalone word at least 20 times in the n60j5 pre-token counts (`artifacts/counts/n60j5/pretoken_counts.bin`). A pre-token counts toward that stem when dropping its leading whitespace leaves exactly the stem. Count-table sha256 `2da2d84f18d5040c58bb47877533462b284e5f1915996048dcf14b7d4fe9c778`.

## Verb-form and noise filter

A native-speaker check of the first spot-check found verb forms counted as case markers: past participles (stem ending in े or ए before को/का/की, e.g. खुलाएका, नआएको, भागेकी) and conditionals (stem ending in े before मा, e.g. घटेमा, बसेमा), plus a one-unit stem (बिकी) and a word with punctuation ((बीमा). A frequency-kept candidate is now dropped when, checking in this order:

1. the word contains any code point outside the Devanagari block U+0900–U+097F (this includes ZWJ, ZWNJ, digits outside the block, and punctuation other than danda and double danda);
2. the stem is a single Ancoda letter unit (`ancoda_seg.aksharas`, DECISIONS.md: a consonant with its signs, or an independent vowel);
3. the marker is को, का, की or मा and the stem ends in the vowel sign े (U+0947);
4. the marker is को, का, की or मा and the stem ends in the independent vowel ए (U+090F).

Rule 4 is not in the written instruction, which named only U+0947, but two of its examples (खुलाएका, नआएको) end in ए (U+090F), and so do the most common participles (भएको, गएको). The **filtered** tables below use rules 1–4. The **literal** variant uses rules 1–3 only and is reported next to them. ले, लाई, बाट, सम्म, देखि, सँग and भन्दा after a participle stem are kept (e.g. लागिएकाले).

### What the filter removes

| Corpus | Variant | kept types before | removed types | share | kept tokens before | removed tokens | share |
|---|---|---:|---:|---:|---:|---:|---:|
| sealed | filtered | 69377 | 4180 | 0.0603 | 812335 | 192210 | 0.2366 |
| sealed | literal | 69377 | 3060 | 0.0441 | 812335 | 102664 | 0.1264 |
| flores | filtered | 2451 | 232 | 0.0947 | 4392 | 753 | 0.1714 |
| flores | literal | 2451 | 146 | 0.0596 | 4392 | 403 | 0.0918 |

Removed by reason (main filter; first matching reason wins):

| Reason | sealed types | sealed tokens | FLORES+ types | FLORES+ tokens |
|---|---:|---:|---:|---:|
| word has a character outside U+0900–U+097F | 582 | 1697 | 14 | 15 |
| stem is one letter unit | 527 | 10976 | 27 | 52 |
| stem ends in े (U+0947), marker को/का/की/मा | 1951 | 89991 | 105 | 336 |
| stem ends in ए (U+090F), marker को/का/की/मा | 1120 | 89546 | 86 | 350 |

Ten most frequent removed words, Sealed:

| Word | occurrences | reason |
|---|---:|---|
| भएको | 22314 | stem ends in ए (U+090F), marker को/का/की/मा |
| गरेको | 17899 | stem ends in े (U+0947), marker को/का/की/मा |
| रहेको | 9506 | stem ends in े (U+0947), marker को/का/की/मा |
| गरेका | 8286 | stem ends in े (U+0947), marker को/का/की/मा |
| गरिएको | 7873 | stem ends in ए (U+090F), marker को/का/की/मा |
| भएका | 5828 | stem ends in ए (U+090F), marker को/का/की/मा |
| आएको | 3837 | stem ends in ए (U+090F), marker को/का/की/मा |
| रहेका | 3466 | stem ends in े (U+0947), marker को/का/की/मा |
| जनाएको | 2906 | stem ends in ए (U+090F), marker को/का/की/मा |
| दिएको | 2829 | stem ends in ए (U+090F), marker को/का/की/मा |

Ten most frequent removed words, FLORES+:

| Word | occurrences | reason |
|---|---:|---|
| भएको | 82 | stem ends in ए (U+090F), marker को/का/की/मा |
| गरिएको | 64 | stem ends in ए (U+090F), marker को/का/की/मा |
| गरेको | 47 | stem ends in े (U+0947), marker को/का/की/मा |
| भएका | 39 | stem ends in ए (U+090F), marker को/का/की/मा |
| रहेको | 35 | stem ends in े (U+0947), marker को/का/की/मा |
| गरेका | 33 | stem ends in े (U+0947), marker को/का/की/मा |
| बारेमा | 29 | stem ends in े (U+0947), marker को/का/की/मा |
| भनेको | 25 | stem ends in े (U+0947), marker को/का/की/मा |
| गरिएका | 23 | stem ends in ए (U+090F), marker को/का/की/मा |
| रहेका | 16 | stem ends in े (U+0947), marker को/का/की/मा |

Most frequent removed words per reason, sealed:

| Reason | words (occurrences) |
|---|---|
| word has a character outside U+0900–U+097F | ‘हामीले (106), 'हामीले (51), ‘मलाई (45), पुर्‍याएको (41), ‘सरकारले (41), ‘नेपालको (32), 'मैले (31), 'मलाई (26) |
| stem is one letter unit | मैले (1667), मलाई (1266), बाँकी (1229), सीमा (602), बीमा (501), आमा (499), मौका (306), निको (226) |
| stem ends in े (U+0947), marker को/का/की/मा | गरेको (17899), रहेको (9506), गरेका (8286), रहेका (3466), भइरहेको (1966), परेको (1886), लागेको (1859), पुगेको (1611) |
| stem ends in ए (U+090F), marker को/का/की/मा | भएको (22314), गरिएको (7873), भएका (5828), आएको (3837), जनाएको (2906), दिएको (2829), आएका (2129), बताएका (2075) |

The rules are applied as specified and are not tuned on these lists. Known costs: the one-unit rule also drops the case-marked pronouns मैले and मलाई (म + ले, म + लाई); the े + मा rule also drops the postposition बारेमा; and the non-Devanagari rule mostly drops case-marked words that carry an opening quote or bracket (‘हामीले), not verb forms.

## Papaya inventory

Affix file: `data/raw/papaya_affixes/data/affix_inventory.tsv`. A marker is present when the affix column, with one leading hyphen removed, equals the marker. Missing markers: **भन्दा**. The missing marker stays in the metric. The inventory was not edited.

## Words kept

Candidate counts are before the frequency filter, frequency-kept counts after it, and kept counts after the verb-form and noise filter.

| Marker | sealed candidate types | candidate tokens | frequency-kept types | frequency-kept tokens | kept types | kept tokens |
|---|---:|---:|---:|---:|---:|---:|
| ले | 16006 | 144514 | 10976 | 137687 | 10762 | 134400 |
| लाई | 9525 | 54544 | 7594 | 52325 | 7481 | 50471 |
| मा | 17312 | 175679 | 12006 | 166616 | 11312 | 159264 |
| को | 27149 | 296869 | 17656 | 282382 | 16116 | 155193 |
| का | 14691 | 130217 | 10137 | 120366 | 9026 | 73268 |
| की | 1791 | 11054 | 1198 | 8271 | 818 | 3259 |
| बाट | 5365 | 21413 | 4474 | 20409 | 4431 | 20318 |
| देखि | 1253 | 5577 | 1088 | 5404 | 1074 | 5390 |
| सँग | 2692 | 8040 | 2291 | 7604 | 2254 | 7404 |
| सम्म | 1267 | 6830 | 1088 | 6634 | 1073 | 6603 |
| भन्दा | 931 | 4700 | 869 | 4637 | 850 | 4555 |
| total | 97982 | 859437 | 69377 | 812335 | 65197 | 620125 |


| Marker | flores candidate types | candidate tokens | frequency-kept types | frequency-kept tokens | kept types | kept tokens |
|---|---:|---:|---:|---:|---:|---:|
| ले | 436 | 724 | 347 | 633 | 336 | 620 |
| लाई | 241 | 393 | 218 | 369 | 210 | 354 |
| मा | 548 | 1025 | 499 | 971 | 482 | 923 |
| को | 856 | 1546 | 748 | 1432 | 639 | 990 |
| का | 456 | 712 | 407 | 656 | 330 | 446 |
| की | 24 | 53 | 16 | 31 | 9 | 10 |
| बाट | 87 | 100 | 79 | 92 | 79 | 92 |
| देखि | 20 | 23 | 17 | 20 | 17 | 20 |
| सँग | 58 | 70 | 53 | 64 | 51 | 62 |
| सम्म | 25 | 33 | 23 | 31 | 23 | 31 |
| भन्दा | 50 | 99 | 44 | 93 | 43 | 91 |
| total | 2801 | 4778 | 2451 | 4392 | 2219 | 3639 |


Kept word types in total, counting a type once if it occurs in either corpus: **65640**. Sealed kept tokens: **620125** in 26974 of 28094 documents. FLORES+ kept tokens: **3639** in 980 of 1012 lines.

## Spot-check

`docs/case_marker_spotcheck.tsv` is 200 kept types (after the verb-form and noise filter) that occur in the sealed set, taken in order from a `random.Random(20261104)` shuffle of the sorted kept types and then sorted by marker and word for reading. Types that occur only in FLORES+ are left out of this file. It replaces the first spot-check and is for a second native-speaker check.

## Ancoda-64k against the 64k panel, before and after the filter

Crossed token share (occurrence-weighted). Intervals as in the section below. The last column is the literal U+0947-only filter.

### Sealed

| Competitor | crossed tokens, ancoda-64k / competitor, before | after | ours − competitor, before | after | U+0947-only filter |
|---|---|---|---|---|---|
| yak-64k-v1 | 0.1039 / 0.1141 | 0.1246 / 0.1363 | WIN -0.0102 [-0.0110, -0.0094] | WIN -0.0117 [-0.0127, -0.0107] | WIN -0.0099 [-0.0108, -0.0090] |
| yak-64k-v2.0 | 0.1039 / 0.1125 | 0.1246 / 0.1343 | WIN -0.0086 [-0.0090, -0.0082] | WIN -0.0097 [-0.0103, -0.0092] | WIN -0.0092 [-0.0097, -0.0088] |
| sidskarki-64k | 0.1039 / 0.0713 | 0.1246 / 0.0857 | LOSE +0.0326 [+0.0319, +0.0334] | LOSE +0.0389 [+0.0380, +0.0398] | LOSE +0.0360 [+0.0352, +0.0367] |
| plain-64k | 0.1039 / 0.1306 | 0.1246 / 0.1544 | WIN -0.0267 [-0.0276, -0.0259] | WIN -0.0299 [-0.0309, -0.0288] | WIN -0.0272 [-0.0282, -0.0263] |
| sarvam-1 | 0.1039 / 0.0458 | 0.1246 / 0.0589 | LOSE +0.0581 [+0.0566, +0.0596] | LOSE +0.0657 [+0.0639, +0.0676] | LOSE +0.0614 [+0.0597, +0.0630] |
| himalaya-64k | 0.1039 / 0.1037 | 0.1246 / 0.1217 | TIE +0.0002 [-0.0004, +0.0008] | LOSE +0.0029 [+0.0021, +0.0037] | LOSE +0.0013 [+0.0006, +0.0021] |

### FLORES+

| Competitor | crossed tokens, ancoda-64k / competitor, before | after | ours − competitor, before | after | U+0947-only filter |
|---|---|---|---|---|---|
| yak-64k-v1 | 0.2006 / 0.2033 | 0.2317 / 0.2344 | TIE -0.0027 [-0.0118, +0.0060] | TIE -0.0027 [-0.0134, +0.0079] | TIE -0.0020 [-0.0116, +0.0077] |
| yak-64k-v2.0 | 0.2006 / 0.2026 | 0.2317 / 0.2330 | TIE -0.0020 [-0.0064, +0.0023] | TIE -0.0014 [-0.0065, +0.0038] | TIE -0.0020 [-0.0068, +0.0028] |
| sidskarki-64k | 0.2006 / 0.1523 | 0.2317 / 0.1764 | LOSE +0.0483 [+0.0406, +0.0562] | LOSE +0.0552 [+0.0462, +0.0641] | LOSE +0.0516 [+0.0435, +0.0601] |
| plain-64k | 0.2006 / 0.2256 | 0.2317 / 0.2575 | WIN -0.0250 [-0.0349, -0.0153] | WIN -0.0258 [-0.0376, -0.0142] | WIN -0.0243 [-0.0352, -0.0138] |
| sarvam-1 | 0.2006 / 0.0594 | 0.2317 / 0.0703 | LOSE +0.1412 [+0.1271, +0.1552] | LOSE +0.1613 [+0.1442, +0.1778] | LOSE +0.1499 [+0.1350, +0.1651] |
| himalaya-64k | 0.2006 / 0.1942 | 0.2317 / 0.2218 | TIE +0.0064 [-0.0011, +0.0137] | LOSE +0.0099 [+0.0014, +0.0182] | TIE +0.0078 [-0.0002, +0.0157] |

### Ordering of the 64k class by crossed share

| Corpus | Share | before the filter (lowest crossed first) | after | changed |
|---|---|---|---|---|
| sealed | token | sarvam-1, sidskarki-64k, ancoda-ne-64k, himalaya-64k, ancoda-64k, yak-64k-v2.0, yak-64k-v1, plain-64k, arkios | sarvam-1, sidskarki-64k, ancoda-ne-64k, himalaya-64k, ancoda-64k, yak-64k-v2.0, yak-64k-v1, plain-64k, arkios | no |
| sealed | type | sarvam-1, sidskarki-64k, himalaya-64k, yak-64k-v2.0, ancoda-64k, ancoda-ne-64k, yak-64k-v1, plain-64k, arkios | sarvam-1, sidskarki-64k, himalaya-64k, yak-64k-v2.0, ancoda-64k, ancoda-ne-64k, yak-64k-v1, plain-64k, arkios | no |
| flores | token | sarvam-1, sidskarki-64k, ancoda-ne-64k, himalaya-64k, ancoda-64k, yak-64k-v2.0, yak-64k-v1, plain-64k, arkios | sarvam-1, sidskarki-64k, ancoda-ne-64k, himalaya-64k, ancoda-64k, yak-64k-v2.0, yak-64k-v1, plain-64k, arkios | no |
| flores | type | sarvam-1, sidskarki-64k, ancoda-ne-64k, himalaya-64k, ancoda-64k, yak-64k-v1, yak-64k-v2.0, plain-64k, arkios | sarvam-1, sidskarki-64k, ancoda-ne-64k, himalaya-64k, ancoda-64k, yak-64k-v2.0, yak-64k-v1, plain-64k, arkios | **yes** |

Verdict changes across all interval rows (before → after):

- sealed token ancoda-64k himalaya-64k: TIE → LOSE
- sealed token ancoda-250k gemma3: WIN → LOSE
- sealed token ancoda-250k sarvam-30b: WIN → LOSE
- sealed type ancoda-250k bloom: TIE → LOSE
- flores token ancoda-64k himalaya-64k: TIE → LOSE
- flores token ancoda-128k qwen2.5: WIN → TIE
- flores token ancoda-128k qwen3: WIN → TIE
- flores type ancoda-128k qwen2.5: TIE → LOSE
- flores type ancoda-128k qwen3: TIE → LOSE

## Boundary classes

Each kept occurrence is classified from the tokenization of a leading space plus the word. Offsets are character offsets inside the word; the leading space is ignored.

- **respected**: a token boundary falls exactly at the stem|marker point.
- **fused**: the whole word is one token.
- **crossed**: some token covers the last character of the stem and the first character of the marker, and the word is more than one token.
- **other**: none of those three. A round-trip failure is recorded here.

Type shares give each kept word type equal weight. Token shares weight by occurrences in that corpus. The panel is the 16 per-sentence comparison tokenizers, the five Ancoda builds in those tables, himalaya-64k, himalaya-100k, himalaya-128k, o200k, and plain-64k. himalaya-100k is paired with ancoda-128k, as in the fertility tables. Tables in this section are after the filter.

### Sealed Nepali

| Tokenizer | respected types | fused types | crossed types | other types | respected tokens | fused tokens | crossed tokens | other tokens |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ancoda-32k | 0.4720 | 0.0393 | 0.4887 | 0.0000 | 0.1844 | 0.6427 | 0.1730 | 0.0000 |
| yak-32k-v1 | 0.4185 | 0.0386 | 0.5430 | 0.0000 | 0.1611 | 0.6395 | 0.1994 | 0.0000 |
| ancoda-64k | 0.4123 | 0.0950 | 0.4927 | 0.0000 | 0.1104 | 0.7650 | 0.1246 | 0.0000 |
| ancoda-ne-64k | 0.3033 | 0.1880 | 0.5087 | 0.0000 | 0.0590 | 0.8422 | 0.0988 | 0.0000 |
| yak-64k-v1 | 0.3685 | 0.0972 | 0.5344 | 0.0000 | 0.0966 | 0.7671 | 0.1363 | 0.0000 |
| yak-64k-v2.0 | 0.4320 | 0.0801 | 0.4879 | 0.0000 | 0.1247 | 0.7410 | 0.1343 | 0.0000 |
| sidskarki-64k | 0.3761 | 0.1876 | 0.4363 | 0.0000 | 0.0756 | 0.8387 | 0.0857 | 0.0000 |
| arkios | 0.4040 | 0.0271 | 0.5689 | 0.0000 | 0.1810 | 0.5828 | 0.2362 | 0.0000 |
| sarvam-1 | 0.9183 | 0.0006 | 0.0571 | 0.0240 | 0.9148 | 0.0151 | 0.0589 | 0.0112 |
| himalaya-64k | 0.4855 | 0.0733 | 0.4378 | 0.0034 | 0.1509 | 0.7259 | 0.1217 | 0.0015 |
| plain-64k | 0.3950 | 0.0694 | 0.5356 | 0.0000 | 0.1211 | 0.7244 | 0.1544 | 0.0000 |
| ancoda-128k | 0.3416 | 0.2034 | 0.4549 | 0.0000 | 0.0646 | 0.8502 | 0.0852 | 0.0000 |
| yak-128k-v1 | 0.2985 | 0.2245 | 0.4770 | 0.0000 | 0.0541 | 0.8588 | 0.0871 | 0.0000 |
| yak-128k-v2.0 | 0.3632 | 0.1817 | 0.4551 | 0.0000 | 0.0727 | 0.8367 | 0.0906 | 0.0000 |
| llama3 | 0.4559 | 0.0000 | 0.5441 | 0.0000 | 0.4869 | 0.0000 | 0.5131 | 0.0000 |
| sarvam-m | 0.7475 | 0.0001 | 0.2524 | 0.0000 | 0.7597 | 0.0021 | 0.2381 | 0.0000 |
| qwen2.5 | 0.7956 | 0.0000 | 0.2044 | 0.0000 | 0.7857 | 0.0000 | 0.2143 | 0.0000 |
| qwen3 | 0.7956 | 0.0000 | 0.2044 | 0.0000 | 0.7857 | 0.0000 | 0.2143 | 0.0000 |
| himalaya-100k | 0.4283 | 0.1304 | 0.4379 | 0.0034 | 0.1027 | 0.7974 | 0.0984 | 0.0015 |
| himalaya-128k | 0.3922 | 0.1761 | 0.4283 | 0.0034 | 0.0825 | 0.8295 | 0.0865 | 0.0015 |
| ancoda-250k | 0.2705 | 0.3434 | 0.3861 | 0.0000 | 0.0412 | 0.8988 | 0.0601 | 0.0000 |
| indicbert-v2 | 0.6880 | 0.0082 | 0.3038 | 0.0000 | 0.5158 | 0.3813 | 0.1029 | 0.0000 |
| bloom | 0.6155 | 0.0068 | 0.3777 | 0.0000 | 0.4101 | 0.3697 | 0.2202 | 0.0000 |
| gemma3 | 0.9348 | 0.0007 | 0.0645 | 0.0000 | 0.9285 | 0.0230 | 0.0484 | 0.0000 |
| sarvam-30b | 0.9348 | 0.0007 | 0.0645 | 0.0000 | 0.9285 | 0.0230 | 0.0484 | 0.0000 |
| o200k | 0.7385 | 0.0004 | 0.2610 | 0.0000 | 0.6852 | 0.0896 | 0.2252 | 0.0000 |

### FLORES+ Nepali

| Tokenizer | respected types | fused types | crossed types | other types | respected tokens | fused tokens | crossed tokens | other tokens |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ancoda-32k | 0.3114 | 0.2898 | 0.3988 | 0.0000 | 0.2204 | 0.4974 | 0.2822 | 0.0000 |
| yak-32k-v1 | 0.2853 | 0.2916 | 0.4232 | 0.0000 | 0.2001 | 0.5034 | 0.2965 | 0.0000 |
| ancoda-64k | 0.2068 | 0.4525 | 0.3407 | 0.0000 | 0.1355 | 0.6329 | 0.2317 | 0.0000 |
| ancoda-ne-64k | 0.1230 | 0.5813 | 0.2956 | 0.0000 | 0.0783 | 0.7277 | 0.1940 | 0.0000 |
| yak-64k-v1 | 0.1933 | 0.4624 | 0.3443 | 0.0000 | 0.1264 | 0.6392 | 0.2344 | 0.0000 |
| yak-64k-v2.0 | 0.2307 | 0.4254 | 0.3438 | 0.0000 | 0.1558 | 0.6112 | 0.2330 | 0.0000 |
| sidskarki-64k | 0.1676 | 0.5678 | 0.2645 | 0.0000 | 0.1080 | 0.7156 | 0.1764 | 0.0000 |
| arkios | 0.2952 | 0.2357 | 0.4691 | 0.0000 | 0.2179 | 0.4400 | 0.3421 | 0.0000 |
| sarvam-1 | 0.9175 | 0.0036 | 0.0536 | 0.0252 | 0.9016 | 0.0099 | 0.0703 | 0.0181 |
| himalaya-64k | 0.2826 | 0.3970 | 0.3186 | 0.0018 | 0.1918 | 0.5853 | 0.2218 | 0.0011 |
| plain-64k | 0.2330 | 0.3925 | 0.3745 | 0.0000 | 0.1586 | 0.5840 | 0.2575 | 0.0000 |
| ancoda-128k | 0.1352 | 0.5980 | 0.2668 | 0.0000 | 0.0863 | 0.7387 | 0.1750 | 0.0000 |
| yak-128k-v1 | 0.1118 | 0.6287 | 0.2596 | 0.0000 | 0.0701 | 0.7604 | 0.1696 | 0.0000 |
| yak-128k-v2.0 | 0.1487 | 0.5827 | 0.2686 | 0.0000 | 0.0959 | 0.7290 | 0.1750 | 0.0000 |
| llama3 | 0.4038 | 0.0000 | 0.5962 | 0.0000 | 0.4163 | 0.0000 | 0.5837 | 0.0000 |
| sarvam-m | 0.7485 | 0.0014 | 0.2501 | 0.0000 | 0.7744 | 0.0011 | 0.2245 | 0.0000 |
| qwen2.5 | 0.8017 | 0.0000 | 0.1983 | 0.0000 | 0.8252 | 0.0000 | 0.1748 | 0.0000 |
| qwen3 | 0.8017 | 0.0000 | 0.1983 | 0.0000 | 0.8252 | 0.0000 | 0.1748 | 0.0000 |
| himalaya-100k | 0.2046 | 0.5011 | 0.2925 | 0.0018 | 0.1341 | 0.6667 | 0.1981 | 0.0011 |
| himalaya-128k | 0.1672 | 0.5606 | 0.2704 | 0.0018 | 0.1083 | 0.7109 | 0.1797 | 0.0011 |
| ancoda-250k | 0.0888 | 0.7008 | 0.2105 | 0.0000 | 0.0563 | 0.8071 | 0.1366 | 0.0000 |
| indicbert-v2 | 0.6503 | 0.1023 | 0.2474 | 0.0000 | 0.5048 | 0.3089 | 0.1863 | 0.0000 |
| bloom | 0.5336 | 0.0937 | 0.3727 | 0.0000 | 0.4147 | 0.2883 | 0.2971 | 0.0000 |
| gemma3 | 0.9108 | 0.0050 | 0.0843 | 0.0000 | 0.9008 | 0.0148 | 0.0844 | 0.0000 |
| sarvam-30b | 0.9108 | 0.0050 | 0.0843 | 0.0000 | 0.9008 | 0.0148 | 0.0844 | 0.0000 |
| o200k | 0.7161 | 0.0099 | 0.2740 | 0.0000 | 0.6752 | 0.0844 | 0.2405 | 0.0000 |

## Crossed-share intervals

Paired bootstrap, 10000 resamples, seed 20261104. The difference is Ancoda minus the competitor, so a negative value means Ancoda crosses the stem|marker point less often. WIN means the 95% interval is entirely below 0. The token interval resamples documents (sealed) or sentences (FLORES+) that contain at least one kept word, and recomputes the occurrence-weighted share. The type interval resamples kept word types in that corpus, each with weight 1. Each comparison uses its own generator started at the same seed. The last column is the same interval on the frequency-kept words before the verb-form and noise filter.

### Sealed, by token

| Ours | Competitor | Crossed share, ours − competitor | Before the filter |
|---|---|---|---|
| ancoda-32k | yak-32k-v1 | WIN -0.0265 [-0.0278, -0.0251] | WIN -0.0232 [-0.0242, -0.0221] |
| ancoda-64k | yak-64k-v1 | WIN -0.0117 [-0.0127, -0.0107] | WIN -0.0102 [-0.0110, -0.0094] |
| ancoda-64k | yak-64k-v2.0 | WIN -0.0097 [-0.0103, -0.0092] | WIN -0.0086 [-0.0090, -0.0082] |
| ancoda-64k | sidskarki-64k | LOSE +0.0389 [+0.0380, +0.0398] | LOSE +0.0326 [+0.0319, +0.0334] |
| ancoda-64k | arkios | WIN -0.1116 [-0.1131, -0.1101] | WIN -0.0997 [-0.1009, -0.0985] |
| ancoda-64k | sarvam-1 | LOSE +0.0657 [+0.0639, +0.0676] | LOSE +0.0581 [+0.0566, +0.0596] |
| ancoda-64k | himalaya-64k | LOSE +0.0029 [+0.0021, +0.0037] | TIE +0.0002 [-0.0004, +0.0008] |
| ancoda-64k | plain-64k | WIN -0.0299 [-0.0309, -0.0288] | WIN -0.0267 [-0.0276, -0.0259] |
| ancoda-ne-64k | yak-64k-v1 | WIN -0.0375 [-0.0386, -0.0364] | WIN -0.0332 [-0.0340, -0.0323] |
| ancoda-ne-64k | yak-64k-v2.0 | WIN -0.0356 [-0.0364, -0.0347] | WIN -0.0316 [-0.0323, -0.0309] |
| ancoda-ne-64k | sidskarki-64k | LOSE +0.0131 [+0.0124, +0.0138] | LOSE +0.0096 [+0.0091, +0.0101] |
| ancoda-ne-64k | arkios | WIN -0.1374 [-0.1389, -0.1359] | WIN -0.1227 [-0.1240, -0.1215] |
| ancoda-ne-64k | sarvam-1 | LOSE +0.0399 [+0.0382, +0.0417] | LOSE +0.0350 [+0.0337, +0.0364] |
| ancoda-ne-64k | himalaya-64k | WIN -0.0229 [-0.0239, -0.0220] | WIN -0.0228 [-0.0236, -0.0221] |
| ancoda-ne-64k | plain-64k | WIN -0.0557 [-0.0569, -0.0545] | WIN -0.0497 [-0.0507, -0.0488] |
| ancoda-128k | yak-128k-v1 | WIN -0.0019 [-0.0027, -0.0011] | WIN -0.0021 [-0.0027, -0.0014] |
| ancoda-128k | yak-128k-v2.0 | WIN -0.0054 [-0.0059, -0.0050] | WIN -0.0047 [-0.0051, -0.0044] |
| ancoda-128k | llama3 | WIN -0.4279 [-0.4304, -0.4254] | WIN -0.4977 [-0.4999, -0.4956] |
| ancoda-128k | sarvam-m | WIN -0.1530 [-0.1553, -0.1507] | WIN -0.1148 [-0.1165, -0.1130] |
| ancoda-128k | qwen2.5 | WIN -0.1291 [-0.1315, -0.1267] | WIN -0.2073 [-0.2095, -0.2051] |
| ancoda-128k | qwen3 | WIN -0.1291 [-0.1315, -0.1267] | WIN -0.2073 [-0.2095, -0.2051] |
| ancoda-128k | himalaya-100k | WIN -0.0132 [-0.0140, -0.0125] | WIN -0.0115 [-0.0121, -0.0109] |
| ancoda-128k | himalaya-128k | WIN -0.0013 [-0.0019, -0.0007] | WIN -0.0010 [-0.0015, -0.0005] |
| ancoda-250k | indicbert-v2 | WIN -0.0428 [-0.0441, -0.0415] | WIN -0.0583 [-0.0594, -0.0573] |
| ancoda-250k | bloom | WIN -0.1601 [-0.1619, -0.1583] | WIN -0.1524 [-0.1539, -0.1510] |
| ancoda-250k | gemma3 | LOSE +0.0116 [+0.0104, +0.0129] | WIN -0.0109 [-0.0119, -0.0097] |
| ancoda-250k | sarvam-30b | LOSE +0.0116 [+0.0104, +0.0129] | WIN -0.0109 [-0.0119, -0.0097] |
| ancoda-250k | o200k | WIN -0.1652 [-0.1672, -0.1632] | WIN -0.1912 [-0.1928, -0.1895] |

### Sealed, by type

| Ours | Competitor | Crossed share, ours − competitor | Before the filter |
|---|---|---|---|
| ancoda-32k | yak-32k-v1 | WIN -0.0543 [-0.0586, -0.0499] | WIN -0.0595 [-0.0637, -0.0553] |
| ancoda-64k | yak-64k-v1 | WIN -0.0417 [-0.0458, -0.0375] | WIN -0.0461 [-0.0501, -0.0421] |
| ancoda-64k | yak-64k-v2.0 | LOSE +0.0048 [+0.0028, +0.0067] | LOSE +0.0040 [+0.0022, +0.0058] |
| ancoda-64k | sidskarki-64k | LOSE +0.0564 [+0.0532, +0.0595] | LOSE +0.0546 [+0.0515, +0.0577] |
| ancoda-64k | arkios | WIN -0.0762 [-0.0807, -0.0718] | WIN -0.0835 [-0.0878, -0.0792] |
| ancoda-64k | sarvam-1 | LOSE +0.4356 [+0.4313, +0.4398] | LOSE +0.4391 [+0.4351, +0.4432] |
| ancoda-64k | himalaya-64k | LOSE +0.0549 [+0.0519, +0.0578] | LOSE +0.0495 [+0.0466, +0.0523] |
| ancoda-64k | plain-64k | WIN -0.0429 [-0.0471, -0.0387] | WIN -0.0485 [-0.0525, -0.0444] |
| ancoda-ne-64k | yak-64k-v1 | WIN -0.0257 [-0.0300, -0.0214] | WIN -0.0341 [-0.0382, -0.0299] |
| ancoda-ne-64k | yak-64k-v2.0 | LOSE +0.0208 [+0.0179, +0.0237] | LOSE +0.0161 [+0.0132, +0.0188] |
| ancoda-ne-64k | sidskarki-64k | LOSE +0.0724 [+0.0695, +0.0753] | LOSE +0.0666 [+0.0638, +0.0694] |
| ancoda-ne-64k | arkios | WIN -0.0602 [-0.0648, -0.0556] | WIN -0.0715 [-0.0760, -0.0671] |
| ancoda-ne-64k | sarvam-1 | LOSE +0.4516 [+0.4473, +0.4559] | LOSE +0.4511 [+0.4470, +0.4553] |
| ancoda-ne-64k | himalaya-64k | LOSE +0.0709 [+0.0674, +0.0743] | LOSE +0.0615 [+0.0580, +0.0648] |
| ancoda-ne-64k | plain-64k | WIN -0.0269 [-0.0313, -0.0225] | WIN -0.0365 [-0.0407, -0.0322] |
| ancoda-128k | yak-128k-v1 | WIN -0.0220 [-0.0259, -0.0182] | WIN -0.0257 [-0.0294, -0.0220] |
| ancoda-128k | yak-128k-v2.0 | TIE -0.0001 [-0.0020, +0.0018] | TIE -0.0016 [-0.0035, +0.0002] |
| ancoda-128k | llama3 | WIN -0.0892 [-0.0944, -0.0838] | WIN -0.0977 [-0.1028, -0.0924] |
| ancoda-128k | sarvam-m | LOSE +0.2025 [+0.1974, +0.2077] | LOSE +0.2131 [+0.2081, +0.2179] |
| ancoda-128k | qwen2.5 | LOSE +0.2505 [+0.2455, +0.2556] | LOSE +0.2301 [+0.2251, +0.2352] |
| ancoda-128k | qwen3 | LOSE +0.2505 [+0.2455, +0.2556] | LOSE +0.2301 [+0.2251, +0.2352] |
| ancoda-128k | himalaya-100k | LOSE +0.0170 [+0.0140, +0.0199] | LOSE +0.0132 [+0.0103, +0.0162] |
| ancoda-128k | himalaya-128k | LOSE +0.0266 [+0.0238, +0.0294] | LOSE +0.0243 [+0.0216, +0.0271] |
| ancoda-250k | indicbert-v2 | LOSE +0.0823 [+0.0779, +0.0866] | LOSE +0.0638 [+0.0595, +0.0681] |
| ancoda-250k | bloom | LOSE +0.0084 [+0.0033, +0.0134] | TIE -0.0021 [-0.0070, +0.0028] |
| ancoda-250k | gemma3 | LOSE +0.3216 [+0.3175, +0.3257] | LOSE +0.3161 [+0.3121, +0.3202] |
| ancoda-250k | sarvam-30b | LOSE +0.3216 [+0.3175, +0.3257] | LOSE +0.3161 [+0.3121, +0.3202] |
| ancoda-250k | o200k | LOSE +0.1251 [+0.1200, +0.1301] | LOSE +0.1125 [+0.1074, +0.1174] |

### FLORES+, by token

| Ours | Competitor | Crossed share, ours − competitor | Before the filter |
|---|---|---|---|
| ancoda-32k | yak-32k-v1 | WIN -0.0143 [-0.0274, -0.0014] | WIN -0.0157 [-0.0268, -0.0048] |
| ancoda-64k | yak-64k-v1 | TIE -0.0027 [-0.0134, +0.0079] | TIE -0.0027 [-0.0118, +0.0060] |
| ancoda-64k | yak-64k-v2.0 | TIE -0.0014 [-0.0065, +0.0038] | TIE -0.0020 [-0.0064, +0.0023] |
| ancoda-64k | sidskarki-64k | LOSE +0.0552 [+0.0462, +0.0641] | LOSE +0.0483 [+0.0406, +0.0562] |
| ancoda-64k | arkios | WIN -0.1105 [-0.1254, -0.0960] | WIN -0.1045 [-0.1174, -0.0919] |
| ancoda-64k | sarvam-1 | LOSE +0.1613 [+0.1442, +0.1778] | LOSE +0.1412 [+0.1271, +0.1552] |
| ancoda-64k | himalaya-64k | LOSE +0.0099 [+0.0014, +0.0182] | TIE +0.0064 [-0.0011, +0.0137] |
| ancoda-64k | plain-64k | WIN -0.0258 [-0.0376, -0.0142] | WIN -0.0250 [-0.0349, -0.0153] |
| ancoda-ne-64k | yak-64k-v1 | WIN -0.0404 [-0.0514, -0.0294] | WIN -0.0369 [-0.0461, -0.0278] |
| ancoda-ne-64k | yak-64k-v2.0 | WIN -0.0390 [-0.0480, -0.0300] | WIN -0.0362 [-0.0440, -0.0284] |
| ancoda-ne-64k | sidskarki-64k | LOSE +0.0176 [+0.0104, +0.0250] | LOSE +0.0141 [+0.0079, +0.0202] |
| ancoda-ne-64k | arkios | WIN -0.1481 [-0.1632, -0.1331] | WIN -0.1387 [-0.1515, -0.1258] |
| ancoda-ne-64k | sarvam-1 | LOSE +0.1237 [+0.1077, +0.1395] | LOSE +0.1070 [+0.0937, +0.1202] |
| ancoda-ne-64k | himalaya-64k | WIN -0.0278 [-0.0381, -0.0176] | WIN -0.0278 [-0.0372, -0.0187] |
| ancoda-ne-64k | plain-64k | WIN -0.0635 [-0.0757, -0.0513] | WIN -0.0592 [-0.0695, -0.0490] |
| ancoda-128k | yak-128k-v1 | TIE +0.0055 [-0.0027, +0.0137] | TIE +0.0048 [-0.0023, +0.0119] |
| ancoda-128k | yak-128k-v2.0 | TIE +0.0000 [-0.0050, +0.0049] | TIE -0.0009 [-0.0050, +0.0032] |
| ancoda-128k | llama3 | WIN -0.4086 [-0.4278, -0.3894] | WIN -0.4590 [-0.4765, -0.4415] |
| ancoda-128k | sarvam-m | WIN -0.0495 [-0.0684, -0.0304] | WIN -0.0405 [-0.0567, -0.0245] |
| ancoda-128k | qwen2.5 | TIE +0.0003 [-0.0180, +0.0187] | WIN -0.0740 [-0.0914, -0.0561] |
| ancoda-128k | qwen3 | TIE +0.0003 [-0.0180, +0.0187] | WIN -0.0740 [-0.0914, -0.0561] |
| ancoda-128k | himalaya-100k | WIN -0.0231 [-0.0312, -0.0150] | WIN -0.0209 [-0.0282, -0.0138] |
| ancoda-128k | himalaya-128k | TIE -0.0047 [-0.0117, +0.0025] | TIE -0.0039 [-0.0100, +0.0024] |
| ancoda-250k | indicbert-v2 | WIN -0.0497 [-0.0628, -0.0371] | WIN -0.0612 [-0.0729, -0.0499] |
| ancoda-250k | bloom | WIN -0.1605 [-0.1773, -0.1434] | WIN -0.1564 [-0.1713, -0.1417] |
| ancoda-250k | gemma3 | LOSE +0.0522 [+0.0386, +0.0656] | LOSE +0.0323 [+0.0206, +0.0439] |
| ancoda-250k | sarvam-30b | LOSE +0.0522 [+0.0386, +0.0656] | LOSE +0.0323 [+0.0206, +0.0439] |
| ancoda-250k | o200k | WIN -0.1039 [-0.1218, -0.0858] | WIN -0.1314 [-0.1468, -0.1157] |

### FLORES+, by type

| Ours | Competitor | Crossed share, ours − competitor | Before the filter |
|---|---|---|---|
| ancoda-32k | yak-32k-v1 | WIN -0.0243 [-0.0437, -0.0050] | WIN -0.0253 [-0.0432, -0.0073] |
| ancoda-64k | yak-64k-v1 | TIE -0.0036 [-0.0198, +0.0126] | TIE -0.0041 [-0.0192, +0.0106] |
| ancoda-64k | yak-64k-v2.0 | TIE -0.0032 [-0.0113, +0.0050] | TIE -0.0045 [-0.0122, +0.0029] |
| ancoda-64k | sidskarki-64k | LOSE +0.0762 [+0.0631, +0.0897] | LOSE +0.0734 [+0.0608, +0.0861] |
| ancoda-64k | arkios | WIN -0.1284 [-0.1496, -0.1077] | WIN -0.1346 [-0.1542, -0.1151] |
| ancoda-64k | sarvam-1 | LOSE +0.2871 [+0.2654, +0.3091] | LOSE +0.2729 [+0.2525, +0.2933] |
| ancoda-64k | himalaya-64k | LOSE +0.0221 [+0.0095, +0.0342] | LOSE +0.0171 [+0.0049, +0.0294] |
| ancoda-64k | plain-64k | WIN -0.0338 [-0.0514, -0.0162] | WIN -0.0363 [-0.0526, -0.0200] |
| ancoda-ne-64k | yak-64k-v1 | WIN -0.0487 [-0.0662, -0.0315] | WIN -0.0502 [-0.0665, -0.0343] |
| ancoda-ne-64k | yak-64k-v2.0 | WIN -0.0482 [-0.0622, -0.0342] | WIN -0.0506 [-0.0632, -0.0379] |
| ancoda-ne-64k | sidskarki-64k | LOSE +0.0311 [+0.0203, +0.0419] | LOSE +0.0273 [+0.0171, +0.0375] |
| ancoda-ne-64k | arkios | WIN -0.1735 [-0.1951, -0.1519] | WIN -0.1807 [-0.2011, -0.1603] |
| ancoda-ne-64k | sarvam-1 | LOSE +0.2420 [+0.2204, +0.2636] | LOSE +0.2268 [+0.2077, +0.2464] |
| ancoda-ne-64k | himalaya-64k | WIN -0.0230 [-0.0383, -0.0077] | WIN -0.0290 [-0.0437, -0.0143] |
| ancoda-ne-64k | plain-64k | WIN -0.0789 [-0.0978, -0.0604] | WIN -0.0824 [-0.1000, -0.0653] |
| ancoda-128k | yak-128k-v1 | TIE +0.0072 [-0.0063, +0.0207] | TIE +0.0069 [-0.0057, +0.0196] |
| ancoda-128k | yak-128k-v2.0 | TIE -0.0018 [-0.0090, +0.0054] | TIE -0.0033 [-0.0102, +0.0033] |
| ancoda-128k | llama3 | WIN -0.3294 [-0.3547, -0.3037] | WIN -0.3525 [-0.3762, -0.3280] |
| ancoda-128k | sarvam-m | TIE +0.0167 [-0.0099, +0.0424] | TIE +0.0196 [-0.0045, +0.0432] |
| ancoda-128k | qwen2.5 | LOSE +0.0685 [+0.0428, +0.0942] | TIE +0.0245 [-0.0012, +0.0502] |
| ancoda-128k | qwen3 | LOSE +0.0685 [+0.0428, +0.0942] | TIE +0.0245 [-0.0012, +0.0502] |
| ancoda-128k | himalaya-100k | WIN -0.0257 [-0.0379, -0.0140] | WIN -0.0265 [-0.0384, -0.0147] |
| ancoda-128k | himalaya-128k | TIE -0.0036 [-0.0144, +0.0072] | TIE -0.0033 [-0.0131, +0.0069] |
| ancoda-250k | indicbert-v2 | WIN -0.0370 [-0.0545, -0.0194] | WIN -0.0624 [-0.0800, -0.0449] |
| ancoda-250k | bloom | WIN -0.1622 [-0.1857, -0.1393] | WIN -0.1795 [-0.2011, -0.1579] |
| ancoda-250k | gemma3 | LOSE +0.1262 [+0.1068, +0.1456] | LOSE +0.1040 [+0.0853, +0.1228] |
| ancoda-250k | sarvam-30b | LOSE +0.1262 [+0.1068, +0.1456] | LOSE +0.1040 [+0.0853, +0.1228] |
| ancoda-250k | o200k | WIN -0.0635 [-0.0883, -0.0388] | WIN -0.1024 [-0.1257, -0.0783] |

JSON: `artifacts/analysis/case_markers.json` (manifest `artifacts/analysis/case_markers.manifest.json`).

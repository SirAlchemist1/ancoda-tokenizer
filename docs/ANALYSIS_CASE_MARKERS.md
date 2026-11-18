# Nepali case-marker boundaries

Analysis only. No training run, and the server was not used. Reproduction: `scripts/analysis_case_markers.py` (tokenizers 0.23.3). The morphology scorer in `python/ancoda_tok/eval/morph.py` is unchanged; this metric is a separate function in that module.

Words are whitespace-separated tokens from sealed Nepali (28,094 documents in the no-joiner sealed set) and from FLORES+ Nepali devtest (1,012 lines that contain a word). A word is a candidate when it ends in one of: ले, लाई, मा, को, का, की, बाट, देखि, सँग, सम्म, भन्दा. Longest match wins, and the stem (the word minus the marker) must be nonempty. No Unicode normalization is applied.

A candidate is kept when its stem occurs as a standalone word at least 20 times in the n60j5 pre-token counts (`artifacts/counts/n60j5/pretoken_counts.bin`). A pre-token counts toward that stem when dropping its leading whitespace leaves exactly the stem. Count-table sha256 `2da2d84f18d5040c58bb47877533462b284e5f1915996048dcf14b7d4fe9c778`.

## Papaya inventory

Affix file: `data/raw/papaya_affixes/data/affix_inventory.tsv`. A marker is present when the affix column, with one leading hyphen removed, equals the marker. Missing markers: **भन्दा**. The missing marker stays in the metric. The inventory was not edited, and the spot-check below was not used to add or drop a marker.

## Words kept

Candidate counts are before the frequency filter. Kept counts are after it.

| Marker | sealed candidate types | candidate tokens | kept types | kept tokens |
|---|---:|---:|---:|---:|
| ले | 16006 | 144514 | 10976 | 137687 |
| लाई | 9525 | 54544 | 7594 | 52325 |
| मा | 17312 | 175679 | 12006 | 166616 |
| को | 27149 | 296869 | 17656 | 282382 |
| का | 14691 | 130217 | 10137 | 120366 |
| की | 1791 | 11054 | 1198 | 8271 |
| बाट | 5365 | 21413 | 4474 | 20409 |
| देखि | 1253 | 5577 | 1088 | 5404 |
| सँग | 2692 | 8040 | 2291 | 7604 |
| सम्म | 1267 | 6830 | 1088 | 6634 |
| भन्दा | 931 | 4700 | 869 | 4637 |
| total | 97982 | 859437 | 69377 | 812335 |


| Marker | flores candidate types | candidate tokens | kept types | kept tokens |
|---|---:|---:|---:|---:|
| ले | 436 | 724 | 347 | 633 |
| लाई | 241 | 393 | 218 | 369 |
| मा | 548 | 1025 | 499 | 971 |
| को | 856 | 1546 | 748 | 1432 |
| का | 456 | 712 | 407 | 656 |
| की | 24 | 53 | 16 | 31 |
| बाट | 87 | 100 | 79 | 92 |
| देखि | 20 | 23 | 17 | 20 |
| सँग | 58 | 70 | 53 | 64 |
| सम्म | 25 | 33 | 23 | 31 |
| भन्दा | 50 | 99 | 44 | 93 |
| total | 2801 | 4778 | 2451 | 4392 |


Kept word types in total, counting a type once if it occurs in either corpus: **69840**. Sealed kept tokens: **812335** in 27239 of 28094 documents. FLORES+ kept tokens: **4392** in 993 of 1012 lines.

## Spot-check

`docs/case_marker_spotcheck.tsv` is 200 kept types that occur in the sealed set, taken in order from a `random.Random(20261104)` shuffle of the kept types and then sorted by marker and word for reading. Types that occur only in FLORES+ are left out of this file. It is for a native-speaker check of the stem|marker split. It was not used to change the marker list, the count threshold, or the classifier.

## Boundary classes

Each kept occurrence is classified from the tokenization of a leading space plus the word. Offsets are character offsets inside the word; the leading space is ignored.

- **respected**: a token boundary falls exactly at the stem|marker point.
- **fused**: the whole word is one token.
- **crossed**: some token covers the last character of the stem and the first character of the marker, and the word is more than one token.
- **other**: none of those three. A round-trip failure is recorded here.

Type shares give each kept word type equal weight. Token shares weight by occurrences in that corpus. The panel is the 16 per-sentence comparison tokenizers, the five Ancoda builds in those tables, himalaya-64k, himalaya-100k, himalaya-128k, o200k, and plain-64k. himalaya-100k is paired with ancoda-128k, as in the fertility tables.

### Sealed Nepali

| Tokenizer | respected types | fused types | crossed types | other types | respected tokens | fused tokens | crossed tokens | other tokens |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ancoda-32k | 0.4640 | 0.0438 | 0.4922 | 0.0000 | 0.1463 | 0.7057 | 0.1479 | 0.0000 |
| yak-32k-v1 | 0.4060 | 0.0423 | 0.5518 | 0.0000 | 0.1265 | 0.7024 | 0.1711 | 0.0000 |
| ancoda-64k | 0.4039 | 0.1019 | 0.4942 | 0.0000 | 0.0874 | 0.8087 | 0.1039 | 0.0000 |
| ancoda-ne-64k | 0.2977 | 0.1961 | 0.5062 | 0.0000 | 0.0470 | 0.8722 | 0.0809 | 0.0000 |
| yak-64k-v1 | 0.3563 | 0.1034 | 0.5403 | 0.0000 | 0.0756 | 0.8104 | 0.1141 | 0.0000 |
| yak-64k-v2.0 | 0.4232 | 0.0866 | 0.4901 | 0.0000 | 0.0988 | 0.7887 | 0.1125 | 0.0000 |
| sidskarki-64k | 0.3644 | 0.1954 | 0.4396 | 0.0006 | 0.0594 | 0.8692 | 0.0713 | 0.0002 |
| arkios | 0.3922 | 0.0301 | 0.5777 | 0.0000 | 0.1422 | 0.6543 | 0.2036 | 0.0000 |
| sarvam-1 | 0.9206 | 0.0013 | 0.0551 | 0.0231 | 0.9307 | 0.0149 | 0.0458 | 0.0086 |
| himalaya-64k | 0.4681 | 0.0789 | 0.4447 | 0.0083 | 0.1182 | 0.7755 | 0.1037 | 0.0025 |
| plain-64k | 0.3825 | 0.0749 | 0.5426 | 0.0000 | 0.0948 | 0.7746 | 0.1306 | 0.0000 |
| ancoda-128k | 0.3336 | 0.2122 | 0.4543 | 0.0000 | 0.0511 | 0.8787 | 0.0701 | 0.0000 |
| yak-128k-v1 | 0.2877 | 0.2324 | 0.4800 | 0.0000 | 0.0422 | 0.8856 | 0.0722 | 0.0000 |
| yak-128k-v2.0 | 0.3538 | 0.1903 | 0.4559 | 0.0000 | 0.0574 | 0.8678 | 0.0749 | 0.0000 |
| llama3 | 0.4481 | 0.0000 | 0.5519 | 0.0000 | 0.4321 | 0.0000 | 0.5679 | 0.0000 |
| sarvam-m | 0.7587 | 0.0001 | 0.2412 | 0.0000 | 0.8134 | 0.0017 | 0.1849 | 0.0000 |
| qwen2.5 | 0.7759 | 0.0000 | 0.2241 | 0.0000 | 0.7225 | 0.0000 | 0.2775 | 0.0000 |
| qwen3 | 0.7759 | 0.0000 | 0.2241 | 0.0000 | 0.7225 | 0.0000 | 0.2775 | 0.0000 |
| himalaya-100k | 0.4130 | 0.1377 | 0.4411 | 0.0082 | 0.0806 | 0.8352 | 0.0817 | 0.0025 |
| himalaya-128k | 0.3778 | 0.1840 | 0.4299 | 0.0082 | 0.0646 | 0.8617 | 0.0711 | 0.0025 |
| ancoda-250k | 0.2633 | 0.3521 | 0.3846 | 0.0000 | 0.0325 | 0.9180 | 0.0495 | 0.0000 |
| indicbert-v2 | 0.6681 | 0.0112 | 0.3207 | 0.0000 | 0.4108 | 0.4814 | 0.1078 | 0.0000 |
| bloom | 0.6047 | 0.0086 | 0.3867 | 0.0000 | 0.3280 | 0.4700 | 0.2019 | 0.0000 |
| gemma3 | 0.9299 | 0.0017 | 0.0685 | 0.0000 | 0.8284 | 0.1112 | 0.0603 | 0.0000 |
| sarvam-30b | 0.9299 | 0.0017 | 0.0685 | 0.0000 | 0.8284 | 0.1112 | 0.0603 | 0.0000 |
| o200k | 0.7270 | 0.0009 | 0.2721 | 0.0000 | 0.5650 | 0.1944 | 0.2406 | 0.0000 |

### FLORES+ Nepali

| Tokenizer | respected types | fused types | crossed types | other types | respected tokens | fused tokens | crossed tokens | other tokens |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ancoda-32k | 0.2942 | 0.3207 | 0.3851 | 0.0000 | 0.1906 | 0.5608 | 0.2486 | 0.0000 |
| yak-32k-v1 | 0.2681 | 0.3215 | 0.4104 | 0.0000 | 0.1719 | 0.5638 | 0.2643 | 0.0000 |
| ancoda-64k | 0.1954 | 0.4814 | 0.3231 | 0.0000 | 0.1170 | 0.6824 | 0.2006 | 0.0000 |
| ancoda-ne-64k | 0.1171 | 0.6059 | 0.2770 | 0.0000 | 0.0683 | 0.7653 | 0.1664 | 0.0000 |
| yak-64k-v1 | 0.1824 | 0.4904 | 0.3272 | 0.0000 | 0.1091 | 0.6876 | 0.2033 | 0.0000 |
| yak-64k-v2.0 | 0.2171 | 0.4553 | 0.3276 | 0.0000 | 0.1339 | 0.6635 | 0.2026 | 0.0000 |
| sidskarki-64k | 0.1571 | 0.5928 | 0.2497 | 0.0004 | 0.0927 | 0.7548 | 0.1523 | 0.0002 |
| arkios | 0.2770 | 0.2652 | 0.4578 | 0.0000 | 0.1872 | 0.5077 | 0.3051 | 0.0000 |
| sarvam-1 | 0.9200 | 0.0069 | 0.0502 | 0.0228 | 0.9133 | 0.0123 | 0.0594 | 0.0150 |
| himalaya-64k | 0.2640 | 0.4264 | 0.3060 | 0.0037 | 0.1639 | 0.6398 | 0.1942 | 0.0020 |
| plain-64k | 0.2183 | 0.4223 | 0.3594 | 0.0000 | 0.1357 | 0.6387 | 0.2256 | 0.0000 |
| ancoda-128k | 0.1281 | 0.6218 | 0.2501 | 0.0000 | 0.0749 | 0.7748 | 0.1503 | 0.0000 |
| yak-128k-v1 | 0.1065 | 0.6503 | 0.2432 | 0.0000 | 0.0612 | 0.7933 | 0.1455 | 0.0000 |
| yak-128k-v2.0 | 0.1404 | 0.6063 | 0.2534 | 0.0000 | 0.0829 | 0.7659 | 0.1512 | 0.0000 |
| llama3 | 0.3974 | 0.0000 | 0.6026 | 0.0000 | 0.3907 | 0.0000 | 0.6093 | 0.0000 |
| sarvam-m | 0.7683 | 0.0012 | 0.2305 | 0.0000 | 0.8083 | 0.0009 | 0.1908 | 0.0000 |
| qwen2.5 | 0.7744 | 0.0000 | 0.2256 | 0.0000 | 0.7757 | 0.0000 | 0.2243 | 0.0000 |
| qwen3 | 0.7744 | 0.0000 | 0.2256 | 0.0000 | 0.7757 | 0.0000 | 0.2243 | 0.0000 |
| himalaya-100k | 0.1909 | 0.5288 | 0.2766 | 0.0037 | 0.1145 | 0.7122 | 0.1712 | 0.0020 |
| himalaya-128k | 0.1567 | 0.5863 | 0.2534 | 0.0037 | 0.0929 | 0.7509 | 0.1541 | 0.0020 |
| ancoda-250k | 0.0857 | 0.7185 | 0.1958 | 0.0000 | 0.0499 | 0.8336 | 0.1166 | 0.0000 |
| indicbert-v2 | 0.6095 | 0.1322 | 0.2583 | 0.0000 | 0.4310 | 0.3912 | 0.1778 | 0.0000 |
| bloom | 0.5047 | 0.1200 | 0.3754 | 0.0000 | 0.3582 | 0.3689 | 0.2730 | 0.0000 |
| gemma3 | 0.8976 | 0.0106 | 0.0918 | 0.0000 | 0.8308 | 0.0849 | 0.0842 | 0.0000 |
| sarvam-30b | 0.8976 | 0.0106 | 0.0918 | 0.0000 | 0.8308 | 0.0849 | 0.0842 | 0.0000 |
| o200k | 0.6838 | 0.0180 | 0.2982 | 0.0000 | 0.5995 | 0.1526 | 0.2480 | 0.0000 |

## Crossed-share intervals

Paired bootstrap, 10000 resamples, seed 20261104. The difference is Ancoda minus the competitor, so a negative value means Ancoda crosses the stem|marker point less often. WIN means the 95% interval is entirely below 0. The token interval resamples documents (sealed) or sentences (FLORES+) that contain at least one kept word, and recomputes the occurrence-weighted share. The type interval resamples kept word types in that corpus, each with weight 1. Each comparison uses its own generator started at the same seed.

### Sealed, by token

| Ours | Competitor | Crossed share, ours − competitor |
|---|---|---|
| ancoda-32k | yak-32k-v1 | WIN -0.0232 [-0.0242, -0.0221] |
| ancoda-64k | yak-64k-v1 | WIN -0.0102 [-0.0110, -0.0094] |
| ancoda-64k | yak-64k-v2.0 | WIN -0.0086 [-0.0090, -0.0082] |
| ancoda-64k | sidskarki-64k | LOSE +0.0326 [+0.0319, +0.0334] |
| ancoda-64k | arkios | WIN -0.0997 [-0.1009, -0.0985] |
| ancoda-64k | sarvam-1 | LOSE +0.0581 [+0.0566, +0.0596] |
| ancoda-64k | himalaya-64k | TIE +0.0002 [-0.0004, +0.0008] |
| ancoda-64k | plain-64k | WIN -0.0267 [-0.0276, -0.0259] |
| ancoda-ne-64k | yak-64k-v1 | WIN -0.0332 [-0.0340, -0.0323] |
| ancoda-ne-64k | yak-64k-v2.0 | WIN -0.0316 [-0.0323, -0.0309] |
| ancoda-ne-64k | sidskarki-64k | LOSE +0.0096 [+0.0091, +0.0101] |
| ancoda-ne-64k | arkios | WIN -0.1227 [-0.1240, -0.1215] |
| ancoda-ne-64k | sarvam-1 | LOSE +0.0350 [+0.0337, +0.0364] |
| ancoda-ne-64k | himalaya-64k | WIN -0.0228 [-0.0236, -0.0221] |
| ancoda-ne-64k | plain-64k | WIN -0.0497 [-0.0507, -0.0488] |
| ancoda-128k | yak-128k-v1 | WIN -0.0021 [-0.0027, -0.0014] |
| ancoda-128k | yak-128k-v2.0 | WIN -0.0047 [-0.0051, -0.0044] |
| ancoda-128k | llama3 | WIN -0.4977 [-0.4999, -0.4956] |
| ancoda-128k | sarvam-m | WIN -0.1148 [-0.1165, -0.1130] |
| ancoda-128k | qwen2.5 | WIN -0.2073 [-0.2095, -0.2051] |
| ancoda-128k | qwen3 | WIN -0.2073 [-0.2095, -0.2051] |
| ancoda-128k | himalaya-100k | WIN -0.0115 [-0.0121, -0.0109] |
| ancoda-128k | himalaya-128k | WIN -0.0010 [-0.0015, -0.0005] |
| ancoda-250k | indicbert-v2 | WIN -0.0583 [-0.0594, -0.0573] |
| ancoda-250k | bloom | WIN -0.1524 [-0.1539, -0.1510] |
| ancoda-250k | gemma3 | WIN -0.0109 [-0.0119, -0.0097] |
| ancoda-250k | sarvam-30b | WIN -0.0109 [-0.0119, -0.0097] |
| ancoda-250k | o200k | WIN -0.1912 [-0.1928, -0.1895] |

### Sealed, by type

| Ours | Competitor | Crossed share, ours − competitor |
|---|---|---|
| ancoda-32k | yak-32k-v1 | WIN -0.0595 [-0.0637, -0.0553] |
| ancoda-64k | yak-64k-v1 | WIN -0.0461 [-0.0501, -0.0421] |
| ancoda-64k | yak-64k-v2.0 | LOSE +0.0040 [+0.0022, +0.0058] |
| ancoda-64k | sidskarki-64k | LOSE +0.0546 [+0.0515, +0.0577] |
| ancoda-64k | arkios | WIN -0.0835 [-0.0878, -0.0792] |
| ancoda-64k | sarvam-1 | LOSE +0.4391 [+0.4351, +0.4432] |
| ancoda-64k | himalaya-64k | LOSE +0.0495 [+0.0466, +0.0523] |
| ancoda-64k | plain-64k | WIN -0.0485 [-0.0525, -0.0444] |
| ancoda-ne-64k | yak-64k-v1 | WIN -0.0341 [-0.0382, -0.0299] |
| ancoda-ne-64k | yak-64k-v2.0 | LOSE +0.0161 [+0.0132, +0.0188] |
| ancoda-ne-64k | sidskarki-64k | LOSE +0.0666 [+0.0638, +0.0694] |
| ancoda-ne-64k | arkios | WIN -0.0715 [-0.0760, -0.0671] |
| ancoda-ne-64k | sarvam-1 | LOSE +0.4511 [+0.4470, +0.4553] |
| ancoda-ne-64k | himalaya-64k | LOSE +0.0615 [+0.0580, +0.0648] |
| ancoda-ne-64k | plain-64k | WIN -0.0365 [-0.0407, -0.0322] |
| ancoda-128k | yak-128k-v1 | WIN -0.0257 [-0.0294, -0.0220] |
| ancoda-128k | yak-128k-v2.0 | TIE -0.0016 [-0.0035, +0.0002] |
| ancoda-128k | llama3 | WIN -0.0977 [-0.1028, -0.0924] |
| ancoda-128k | sarvam-m | LOSE +0.2131 [+0.2081, +0.2179] |
| ancoda-128k | qwen2.5 | LOSE +0.2301 [+0.2251, +0.2352] |
| ancoda-128k | qwen3 | LOSE +0.2301 [+0.2251, +0.2352] |
| ancoda-128k | himalaya-100k | LOSE +0.0132 [+0.0103, +0.0162] |
| ancoda-128k | himalaya-128k | LOSE +0.0243 [+0.0216, +0.0271] |
| ancoda-250k | indicbert-v2 | LOSE +0.0638 [+0.0595, +0.0681] |
| ancoda-250k | bloom | TIE -0.0021 [-0.0070, +0.0028] |
| ancoda-250k | gemma3 | LOSE +0.3161 [+0.3121, +0.3202] |
| ancoda-250k | sarvam-30b | LOSE +0.3161 [+0.3121, +0.3202] |
| ancoda-250k | o200k | LOSE +0.1125 [+0.1074, +0.1174] |

### FLORES+, by token

| Ours | Competitor | Crossed share, ours − competitor |
|---|---|---|
| ancoda-32k | yak-32k-v1 | WIN -0.0157 [-0.0268, -0.0048] |
| ancoda-64k | yak-64k-v1 | TIE -0.0027 [-0.0118, +0.0060] |
| ancoda-64k | yak-64k-v2.0 | TIE -0.0020 [-0.0064, +0.0023] |
| ancoda-64k | sidskarki-64k | LOSE +0.0483 [+0.0406, +0.0562] |
| ancoda-64k | arkios | WIN -0.1045 [-0.1174, -0.0919] |
| ancoda-64k | sarvam-1 | LOSE +0.1412 [+0.1271, +0.1552] |
| ancoda-64k | himalaya-64k | TIE +0.0064 [-0.0011, +0.0137] |
| ancoda-64k | plain-64k | WIN -0.0250 [-0.0349, -0.0153] |
| ancoda-ne-64k | yak-64k-v1 | WIN -0.0369 [-0.0461, -0.0278] |
| ancoda-ne-64k | yak-64k-v2.0 | WIN -0.0362 [-0.0440, -0.0284] |
| ancoda-ne-64k | sidskarki-64k | LOSE +0.0141 [+0.0079, +0.0202] |
| ancoda-ne-64k | arkios | WIN -0.1387 [-0.1515, -0.1258] |
| ancoda-ne-64k | sarvam-1 | LOSE +0.1070 [+0.0937, +0.1202] |
| ancoda-ne-64k | himalaya-64k | WIN -0.0278 [-0.0372, -0.0187] |
| ancoda-ne-64k | plain-64k | WIN -0.0592 [-0.0695, -0.0490] |
| ancoda-128k | yak-128k-v1 | TIE +0.0048 [-0.0023, +0.0119] |
| ancoda-128k | yak-128k-v2.0 | TIE -0.0009 [-0.0050, +0.0032] |
| ancoda-128k | llama3 | WIN -0.4590 [-0.4765, -0.4415] |
| ancoda-128k | sarvam-m | WIN -0.0405 [-0.0567, -0.0245] |
| ancoda-128k | qwen2.5 | WIN -0.0740 [-0.0914, -0.0561] |
| ancoda-128k | qwen3 | WIN -0.0740 [-0.0914, -0.0561] |
| ancoda-128k | himalaya-100k | WIN -0.0209 [-0.0282, -0.0138] |
| ancoda-128k | himalaya-128k | TIE -0.0039 [-0.0100, +0.0024] |
| ancoda-250k | indicbert-v2 | WIN -0.0612 [-0.0729, -0.0499] |
| ancoda-250k | bloom | WIN -0.1564 [-0.1713, -0.1417] |
| ancoda-250k | gemma3 | LOSE +0.0323 [+0.0206, +0.0439] |
| ancoda-250k | sarvam-30b | LOSE +0.0323 [+0.0206, +0.0439] |
| ancoda-250k | o200k | WIN -0.1314 [-0.1468, -0.1157] |

### FLORES+, by type

| Ours | Competitor | Crossed share, ours − competitor |
|---|---|---|
| ancoda-32k | yak-32k-v1 | WIN -0.0253 [-0.0432, -0.0073] |
| ancoda-64k | yak-64k-v1 | TIE -0.0041 [-0.0192, +0.0106] |
| ancoda-64k | yak-64k-v2.0 | TIE -0.0045 [-0.0122, +0.0029] |
| ancoda-64k | sidskarki-64k | LOSE +0.0734 [+0.0608, +0.0861] |
| ancoda-64k | arkios | WIN -0.1346 [-0.1542, -0.1151] |
| ancoda-64k | sarvam-1 | LOSE +0.2729 [+0.2525, +0.2933] |
| ancoda-64k | himalaya-64k | LOSE +0.0171 [+0.0049, +0.0294] |
| ancoda-64k | plain-64k | WIN -0.0363 [-0.0526, -0.0200] |
| ancoda-ne-64k | yak-64k-v1 | WIN -0.0502 [-0.0665, -0.0343] |
| ancoda-ne-64k | yak-64k-v2.0 | WIN -0.0506 [-0.0632, -0.0379] |
| ancoda-ne-64k | sidskarki-64k | LOSE +0.0273 [+0.0171, +0.0375] |
| ancoda-ne-64k | arkios | WIN -0.1807 [-0.2011, -0.1603] |
| ancoda-ne-64k | sarvam-1 | LOSE +0.2268 [+0.2077, +0.2464] |
| ancoda-ne-64k | himalaya-64k | WIN -0.0290 [-0.0437, -0.0143] |
| ancoda-ne-64k | plain-64k | WIN -0.0824 [-0.1000, -0.0653] |
| ancoda-128k | yak-128k-v1 | TIE +0.0069 [-0.0057, +0.0196] |
| ancoda-128k | yak-128k-v2.0 | TIE -0.0033 [-0.0102, +0.0033] |
| ancoda-128k | llama3 | WIN -0.3525 [-0.3762, -0.3280] |
| ancoda-128k | sarvam-m | TIE +0.0196 [-0.0045, +0.0432] |
| ancoda-128k | qwen2.5 | TIE +0.0245 [-0.0012, +0.0502] |
| ancoda-128k | qwen3 | TIE +0.0245 [-0.0012, +0.0502] |
| ancoda-128k | himalaya-100k | WIN -0.0265 [-0.0384, -0.0147] |
| ancoda-128k | himalaya-128k | TIE -0.0033 [-0.0131, +0.0069] |
| ancoda-250k | indicbert-v2 | WIN -0.0624 [-0.0800, -0.0449] |
| ancoda-250k | bloom | WIN -0.1795 [-0.2011, -0.1579] |
| ancoda-250k | gemma3 | LOSE +0.1040 [+0.0853, +0.1228] |
| ancoda-250k | sarvam-30b | LOSE +0.1040 [+0.0853, +0.1228] |
| ancoda-250k | o200k | WIN -0.1024 [-0.1257, -0.0783] |

JSON: `artifacts/analysis/case_markers.json`.

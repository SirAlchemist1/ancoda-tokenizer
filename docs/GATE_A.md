# Gate A

> **11 Nov 2026.** The whole-file first-byte FLORES scoreboard in this document is superseded by [`docs/RESULTS_PER_SENTENCE.md`](RESULTS_PER_SENTENCE.md) (from `artifacts/eval/m8/followup/flores_per_sentence.json`) for all tokenizer comparisons. Each FLORES sentence is encoded separately. Gate A pass/fail bars below stay as originally written; the old tables are not edited.

Ancoda-64k is the flagship. Selection is the mechanical lookup in `docs/DECISIONS.md`.

## PLAN 3.10

| Check | Result | Detail |
|---|---|---|
| Nepali fertility | FAIL | 1.450938 < 1.445 |
| English fertility | PASS | 1.414365 < 1.42 |
| Hindi fertility | PASS | 1.340132 < 1.385 |
| Covered violations | PASS | 0 |
| Mark-initial | PASS | 0 |
| Orphan-mark share, Nepali | PASS | 7.370826269624825e-05 < 0.0015639775951116607 |
| Orphan-mark share, Aya | PASS | reported 0.0002355313613977331 |
| Round-trip | PASS | hf 0 tt 0 |
| HF ids == tiktoken ids | PASS | 0 |
| Nepali byte pieces | PASS | 0.0001474165253924965 |
| Newline gap | PASS | 0.000160 <= 0.01 |
| ZWJ probe | PASS | 1.347350 < 1.5035649327028011 |
| ZWNJ probe | PASS | 1.345286 < 1.691269552564569 |
| Stretch English | FAIL | 1.414365 < 1.388 |
| Stretch Hindi | PASS | 1.340132 < 1.376 |
| 32k Nepali below Yak v1 | PASS | 1.571520 vs 1.585904 |
| 32k English within 2% of Yak v1 | FAIL | 1.572074 vs 1.538286 |
| 128k Nepali below Yak v1 | FAIL | 1.374472 vs 1.369980 |
| 128k English within 2% of Yak v1 | PASS | 1.301904 vs 1.291402 |
| Whole-character coverage | PASS | coverage_failures=0 on U+0900–U+097F plus ZWJ/ZWNJ, so 128/128 |
| Covered-unit integrity, both exports | PASS | same counts as covered violations |
| Orphan marks inside a covered akshara | PASS | orphan_inside_covered=0 |
| Segmenter agreement | PASS | M3 differential, 0 disagreements; segmenter unchanged in M8 |
| Languages of Nepal | not comparable | Yak's 14 held-out sets are private |

## Tail break rate

PLAN 3.9 tail break rate: token boundaries falling strictly inside an uncovered Devanagari akshara, divided by uncovered Devanagari akshara occurrences. Single Devanagari code points are not in the denominator (ancoda_tok.guarantee._tail). The 95% interval is a document-level bootstrap of 1,000 resamples with seed 20261104 over newline-separated snapshot lines that contain at least one such akshara (ancoda_tok.guarantee._bootstrap). The scan is the first 10,000,000 newline-separated lines of n60j5 (1,000,000 for Ancoda-32k). Eval texts are in the violation, round-trip and equality checks, and are not in this tail interval.

## Guarantees

| Model | Covered viol. HF/TT | Mark-initial | Round-trip HF/TT | HF==TT mismatches | Block 1 coverage | Orphan share | Tail HF (point, lo, hi) |
|---|---:|---:|---:|---:|---:|---:|---|
| ancoda-128k | 0/0 | 0 | 0/0 | 0 | 0.999778 | 0.00017405267620026164 | [0.5180096051227321, 0.4688995215311005, 0.573348749819338] |
| ancoda-250k | 0/0 | 0 | 0/0 | 0 | 0.999806 | 0.00015353774131497832 | [0.32047477744807124, 0.27473219911783237, 0.37311330247534713] |
| ancoda-32k | 0/0 | 0 | 0/0 | 0 | 0.999099 | 0.0004967345210916414 | [0.0, 0.0, 0.0] |
| ancoda-64k | 0/0 | 0 | 0/0 | 0 | 0.999670 | 0.0002355313613977331 | [0.7347278517660684, 0.6925559675868699, 0.7758553503234354] |
| ancoda-ne-64k | 0/0 | 0 | 0/0 | 0 | 0.999670 | 0.00024320328788984258 | [0.7526780544296467, 0.7137069083917045, 0.7912265823760697] |

## 250k pick

Rule: no point qualified: most WIN rows, then lowest Nepali

Block 1 coverage 0.999806 (gate ≥ 0.9995: True)

| Point | Qualifies | Hindi | English | Nepali | code/KB | WIN rows | Integrity |
|---|---|---:|---:|---:|---:|---:|---|
| ancoda-250k-bd144000-cm100 | False | 1.2494247942908396 | 1.2266106570476234 | 1.336024811507406 | 264.45910111358404 | 15 | True |
| ancoda-250k-bd144000-cm50 | False | 1.249307803299146 | 1.2223642756038537 | 1.3357574461258757 | 269.12770231970404 | 15 | True |
| ancoda-250k-bd160000-cm100 | False | 1.2417813828335218 | 1.2344185197022968 | 1.328110796214106 | 267.12545645076284 | 14 | True |
| ancoda-250k-bd160000-cm50 | False | 1.2417423858362906 | 1.2285740377151728 | 1.3276830116036575 | 272.58655320749494 | 15 | True |
| ancoda-250k-bd175000-cm100 | False | 1.2371017431657763 | 1.2435961828227022 | 1.3231912731939468 | 270.67067943610624 | 12 | True |
| ancoda-250k-bd175000-cm50 | False | 1.2371017431657763 | 1.2385279210994933 | 1.3228704347361104 | 276.71726721737474 | 12 | True |
| ancoda-250k-bd190000-cm100 | False | 1.2332800374371173 | 1.2589379480389022 | 1.3191807924709908 | 275.57838168889555 | 11 | True |
| ancoda-250k-bd190000-cm50 | False | 1.2333190344343485 | 1.251678005570522 | 1.318699534784236 | 282.3988923150394 | 11 | True |

n60j10: Hindi or English improved with CI excluding 0 (use_j10=True)

## Scoreboards

HimalayaAI BPE-128k: skipped: license (Hub card states no license).

CIs are paired line bootstraps (10,000 resamples, seed 20261104). Diff is ours minus the panel tokenizer. WIN means the interval is entirely below 0.

### ~128k

| Ours | Base | Metric | Result |
|---|---|---|---|
| ancoda-128k | yak-128k-v1 | npi | LOSE +0.0045 [+0.0015, +0.0075] whole-file first-byte |
| ancoda-128k | yak-128k-v1 | eng | LOSE +0.0105 [+0.0082, +0.0128] whole-file first-byte |
| ancoda-128k | yak-128k-v1 | hin | WIN -0.0193 [-0.0223, -0.0165] whole-file first-byte |
| ancoda-128k | yak-128k-v1 | deva_mean | WIN -0.0392 [-0.0415, -0.0369] whole-file first-byte |
| ancoda-128k | yak-128k-v2.0 | npi | WIN -0.0079 [-0.0102, -0.0056] whole-file first-byte |
| ancoda-128k | yak-128k-v2.0 | eng | WIN -0.0108 [-0.0133, -0.0083] whole-file first-byte |
| ancoda-128k | yak-128k-v2.0 | hin | WIN -0.0287 [-0.0314, -0.0261] whole-file first-byte |
| ancoda-128k | yak-128k-v2.0 | deva_mean | WIN -0.0232 [-0.0252, -0.0213] whole-file first-byte |
| ancoda-128k | llama3 | npi | WIN -2.3897 [-2.4156, -2.3640] whole-file first-byte |
| ancoda-128k | llama3 | eng | LOSE +0.0610 [+0.0559, +0.0664] whole-file first-byte |
| ancoda-128k | llama3 | hin | WIN -1.3990 [-1.4143, -1.3836] whole-file first-byte |
| ancoda-128k | llama3 | deva_mean | WIN -1.6774 [-1.6917, -1.6626] whole-file first-byte |
| ancoda-128k | sarvam-m | npi | WIN -1.7944 [-1.8155, -1.7736] whole-file first-byte |
| ancoda-128k | sarvam-m | eng | LOSE +0.0311 [+0.0273, +0.0350] whole-file first-byte |
| ancoda-128k | sarvam-m | hin | WIN -0.6724 [-0.6858, -0.6589] whole-file first-byte |
| ancoda-128k | sarvam-m | deva_mean | WIN -1.1653 [-1.1764, -1.1546] whole-file first-byte |
| ancoda-128k | qwen2.5 | npi | WIN -5.2001 [-5.2482, -5.1530] whole-file first-byte |
| ancoda-128k | qwen2.5 | eng | LOSE +0.0405 [+0.0367, +0.0444] whole-file first-byte |
| ancoda-128k | qwen2.5 | hin | WIN -3.4790 [-3.5065, -3.4511] per-line |
| ancoda-128k | qwen2.5 | deva_mean | WIN -3.9216 [-3.9486, -3.8951] per-line |
| ancoda-128k | qwen3 | npi | WIN -5.2001 [-5.2482, -5.1530] whole-file first-byte |
| ancoda-128k | qwen3 | eng | LOSE +0.0405 [+0.0367, +0.0444] whole-file first-byte |
| ancoda-128k | qwen3 | hin | WIN -3.4790 [-3.5065, -3.4511] per-line |
| ancoda-128k | qwen3 | deva_mean | WIN -3.9216 [-3.9486, -3.8951] per-line |

### ~250k

| Ours | Base | Metric | Result |
|---|---|---|---|
| ancoda-250k | indicbert-v2 | npi | WIN -0.2500 [-0.2614, -0.2389] per-line |
| ancoda-250k | indicbert-v2 | eng | WIN -0.0079 [-0.0154, -0.0001] per-line |
| ancoda-250k | indicbert-v2 | hin | TIE +0.0034 [-0.0025, +0.0094] per-line |
| ancoda-250k | indicbert-v2 | deva_mean | WIN -0.1216 [-0.1281, -0.1149] per-line |
| ancoda-250k | bloom | npi | WIN -0.3954 [-0.4090, -0.3821] whole-file first-byte |
| ancoda-250k | bloom | eng | WIN -0.0244 [-0.0320, -0.0166] whole-file first-byte |
| ancoda-250k | bloom | hin | WIN -0.1307 [-0.1389, -0.1226] whole-file first-byte |
| ancoda-250k | bloom | deva_mean | WIN -0.3694 [-0.3780, -0.3606] whole-file first-byte |
| ancoda-250k | gemma3 | npi | WIN -0.8882 [-0.9038, -0.8723] per-line |
| ancoda-250k | gemma3 | eng | WIN -0.0095 [-0.0129, -0.0062] per-line |
| ancoda-250k | gemma3 | hin | WIN -0.1450 [-0.1514, -0.1389] per-line |
| ancoda-250k | gemma3 | deva_mean | WIN -0.4904 [-0.4966, -0.4842] per-line |
| ancoda-250k | sarvam-30b | npi | WIN -0.8884 [-0.9041, -0.8725] per-line |
| ancoda-250k | sarvam-30b | eng | WIN -0.0099 [-0.0133, -0.0066] per-line |
| ancoda-250k | sarvam-30b | hin | WIN -0.1450 [-0.1514, -0.1389] per-line |
| ancoda-250k | sarvam-30b | deva_mean | WIN -0.4904 [-0.4967, -0.4842] per-line |

### ~32k

| Ours | Base | Metric | Result |
|---|---|---|---|
| ancoda-32k | yak-32k-v1 | npi | WIN -0.0144 [-0.0190, -0.0100] whole-file first-byte |
| ancoda-32k | yak-32k-v1 | eng | LOSE +0.0338 [+0.0305, +0.0371] whole-file first-byte |
| ancoda-32k | yak-32k-v1 | hin | WIN -0.0713 [-0.0759, -0.0668] whole-file first-byte |
| ancoda-32k | yak-32k-v1 | deva_mean | WIN -0.0857 [-0.0891, -0.0823] whole-file first-byte |

### ~64k

| Ours | Base | Metric | Result |
|---|---|---|---|
| ancoda-64k | yak-64k-v1 | npi | WIN -0.0059 [-0.0095, -0.0023] whole-file first-byte |
| ancoda-64k | yak-64k-v1 | eng | LOSE +0.0263 [+0.0234, +0.0292] whole-file first-byte |
| ancoda-64k | yak-64k-v1 | hin | WIN -0.0406 [-0.0445, -0.0369] whole-file first-byte |
| ancoda-64k | yak-64k-v1 | deva_mean | WIN -0.0571 [-0.0600, -0.0543] whole-file first-byte |
| ancoda-64k | yak-64k-v2.0 | npi | WIN -0.0193 [-0.0222, -0.0164] whole-file first-byte |
| ancoda-64k | yak-64k-v2.0 | eng | WIN -0.0059 [-0.0088, -0.0029] whole-file first-byte |
| ancoda-64k | yak-64k-v2.0 | hin | WIN -0.0509 [-0.0543, -0.0476] whole-file first-byte |
| ancoda-64k | yak-64k-v2.0 | deva_mean | WIN -0.0389 [-0.0413, -0.0367] whole-file first-byte |
| ancoda-64k | sidskarki-64k | npi | TIE +0.0061 [-0.0013, +0.0139] whole-file first-byte |
| ancoda-64k | sidskarki-64k | eng | WIN -0.9942 [-1.0118, -0.9771] whole-file first-byte |
| ancoda-64k | sidskarki-64k | hin | WIN -0.3264 [-0.3360, -0.3164] whole-file first-byte |
| ancoda-64k | sidskarki-64k | deva_mean | WIN -0.2862 [-0.2936, -0.2786] whole-file first-byte |
| ancoda-64k | arkios | npi | WIN -0.2369 [-0.2462, -0.2277] whole-file first-byte |
| ancoda-64k | arkios | eng | LOSE +0.1174 [+0.1098, +0.1252] whole-file first-byte |
| ancoda-64k | arkios | hin | WIN -0.6324 [-0.6441, -0.6209] per-line |
| ancoda-64k | arkios | deva_mean | WIN -0.5238 [-0.5315, -0.5159] per-line |
| ancoda-64k | sarvam-1 | npi | WIN -1.1384 [-1.1583, -1.1184] per-line |
| ancoda-64k | sarvam-1 | eng | WIN -0.0206 [-0.0240, -0.0173] per-line |
| ancoda-64k | sarvam-1 | hin | WIN -0.0616 [-0.0670, -0.0561] per-line |
| ancoda-64k | sarvam-1 | deva_mean | WIN -0.4509 [-0.4575, -0.4441] per-line |
| ancoda-ne-64k | yak-64k-v1 | npi | WIN -0.0515 [-0.0563, -0.0467] whole-file first-byte |
| ancoda-ne-64k | yak-64k-v1 | eng | LOSE +0.8629 [+0.8490, +0.8768] whole-file first-byte |
| ancoda-ne-64k | yak-64k-v1 | hin | LOSE +0.2117 [+0.2043, +0.2190] whole-file first-byte |
| ancoda-ne-64k | yak-64k-v1 | deva_mean | LOSE +0.1637 [+0.1593, +0.1681] whole-file first-byte |
| ancoda-ne-64k | yak-64k-v2.0 | npi | WIN -0.0649 [-0.0697, -0.0600] whole-file first-byte |
| ancoda-ne-64k | yak-64k-v2.0 | eng | LOSE +0.8308 [+0.8170, +0.8445] whole-file first-byte |
| ancoda-ne-64k | yak-64k-v2.0 | hin | LOSE +0.2014 [+0.1941, +0.2086] whole-file first-byte |
| ancoda-ne-64k | yak-64k-v2.0 | deva_mean | LOSE +0.1819 [+0.1773, +0.1864] whole-file first-byte |
| ancoda-ne-64k | sidskarki-64k | npi | WIN -0.0395 [-0.0466, -0.0321] whole-file first-byte |
| ancoda-ne-64k | sidskarki-64k | eng | WIN -0.1576 [-0.1683, -0.1469] whole-file first-byte |
| ancoda-ne-64k | sidskarki-64k | hin | WIN -0.0741 [-0.0806, -0.0674] whole-file first-byte |
| ancoda-ne-64k | sidskarki-64k | deva_mean | WIN -0.0654 [-0.0711, -0.0596] whole-file first-byte |
| ancoda-ne-64k | arkios | npi | WIN -0.2825 [-0.2934, -0.2718] whole-file first-byte |
| ancoda-ne-64k | arkios | eng | LOSE +0.9541 [+0.9383, +0.9697] whole-file first-byte |
| ancoda-ne-64k | arkios | hin | WIN -0.3804 [-0.3899, -0.3710] per-line |
| ancoda-ne-64k | arkios | deva_mean | WIN -0.3031 [-0.3101, -0.2961] per-line |
| ancoda-ne-64k | sarvam-1 | npi | WIN -1.1841 [-1.2047, -1.1632] per-line |
| ancoda-ne-64k | sarvam-1 | eng | LOSE +0.8149 [+0.8013, +0.8285] per-line |
| ancoda-ne-64k | sarvam-1 | hin | LOSE +0.1904 [+0.1822, +0.1988] per-line |
| ancoda-ne-64k | sarvam-1 | deva_mean | WIN -0.2302 [-0.2362, -0.2240] per-line |

### Columns

| Model | npi | eng | hin | deva mean | Aya | orphan npi | ZWJ | ZWNJ | code/KB | math/KB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ancoda-32k | 1.5715 | 1.5721 | 1.4373 | 1.6547 | 1.5590 | 0.00027221069107489197 | 1.3970 | 1.3963 | 358.0 | 329.4 |
| yak-32k-v1 | 1.5859 | 1.5383 | 1.5086 | 1.7404 | 1.5691 | 0.12816103580821364 | 1.6621 | 1.9211 | 350.7 | 326.4 |
| ancoda-64k | 1.4509 | 1.4144 | 1.3401 | 1.5188 | 1.4407 | 7.370826269624825e-05 | 1.3474 | 1.3453 | 320.7 | 301.5 |
| ancoda-ne-64k | 1.4053 | 2.2510 | 1.5924 | 1.7396 | 1.3962 | 7.610060499980975e-05 | 1.2700 | 1.2705 | 520.8 | 448.8 |
| yak-64k-v1 | 1.4568 | 1.3881 | 1.3808 | 1.5759 | 1.4459 | 0.09227719864924387 | 1.6611 | 1.9651 | 316.4 | 301.1 |
| yak-64k-v2.0 | 1.4702 | 1.4203 | 1.3910 | 1.5577 | 1.4546 | 0.0015639775951116607 | 1.5035 | 1.6912 | 324.7 | 306.2 |
| sidskarki-64k | 1.4448 | 2.4086 | 1.6665 | 1.8050 | 1.4068 | 0.016506902550057368 | 2.4737 | 2.4737 | 501.9 | 445.9 |
| arkios | 1.6878 | 1.2970 | 2.0117 | 2.0872 | 1.6478 | 0.14617919148396907 | 2.1674 | 2.1674 | 322.8 | 276.2 |
| sarvam-1 | 2.6593 | 1.4983 | 1.4632 | 2.0341 | 2.4825 | None | 1.3188 | 1.7303 | 388.2 | 332.8 |
| ancoda-128k | 1.3745 | 1.3019 | 1.2773 | 1.4317 | 1.3582 | 3.890445066915655e-05 | 1.2698 | 1.2737 | 289.2 | 281.7 |
| yak-128k-v1 | 1.3700 | 1.2914 | 1.2966 | 1.4710 | 1.3602 | 0.06717408274785323 | 1.6830 | 1.8481 | 291.7 | 284.9 |
| yak-128k-v2.0 | 1.3823 | 1.3127 | 1.3060 | 1.4550 | 1.3678 | 0.0007349812386368032 | 1.5139 | 1.7434 | 297.0 | 288.6 |
| llama3 | 3.7641 | 1.2409 | 2.6763 | 3.1091 | 3.6203 | 0.5475828562499112 | 1.1388 | 1.2644 | 258.7 | 262.0 |
| sarvam-m | 3.1689 | 1.2708 | 1.9497 | 2.5971 | 3.0392 | 0.3209868210121328 | 1.3700 | 1.3700 | 281.9 | 273.9 |
| qwen2.5 | 6.5746 | 1.2614 | 4.7682 | 5.3883 | 6.2844 | 0.12565168237753252 | 1.1567 | 1.0847 | 264.3 | 274.2 |
| qwen3 | 6.5746 | 1.2614 | 4.7682 | 5.3883 | 6.2844 | 0.12565168237753252 | 1.1567 | 1.0847 | 264.3 | 274.2 |
| ancoda-250k | 1.3284 | 1.2287 | 1.2404 | 1.3767 | 1.3170 | 0.0 | 1.2226 | 1.2242 | 275.3 | 269.0 |
| indicbert-v2 | 1.5783 | 1.2364 | 1.2368 | 1.4981 | 1.5422 | None | 1.4366 | 1.6399 | 334.5 | 271.2 |
| bloom | 1.7238 | 1.2530 | 1.3711 | 1.7461 | 1.6897 | 0.16729224183391755 | 1.4933 | 1.6110 | 271.3 | 248.7 |
| gemma3 | 2.2705 | 1.2842 | 1.4246 | 1.9117 | 2.1327 | None | 1.3183 | 1.4472 | 311.2 | 277.5 |
| sarvam-30b | 2.2707 | 1.2846 | 1.4246 | 1.9118 | 2.1328 | None | 1.3182 | 1.4472 | 312.9 | 277.8 |
| o200k | 2.3208 | 1.2272 | 1.6493 | 2.0999 | 2.2704 | 0.2566471591170914 | 1.5250 | 1.5559 | 261.0 | 260.2 |

### Devanagari languages

| Model | awa_Deva | bho_Deva | hin_Deva | mag_Deva | mai_Deva | npi_Deva | san_Deva |
|---|---:|---:|---:|---:|---:|---:|---:|
| ancoda-32k | 1.4964 | 1.5586 | 1.4373 | 1.5395 | 1.5559 | 1.5715 | 2.4237 |
| yak-32k-v1 | 1.6155 | 1.6337 | 1.5086 | 1.6374 | 1.6122 | 1.5859 | 2.5894 |
| ancoda-64k | 1.3790 | 1.4287 | 1.3401 | 1.4060 | 1.4266 | 1.4509 | 2.2002 |
| ancoda-ne-64k | 1.6149 | 1.6743 | 1.5924 | 1.6454 | 1.7431 | 1.4053 | 2.5016 |
| yak-64k-v1 | 1.4704 | 1.4781 | 1.3808 | 1.4455 | 1.4522 | 1.4568 | 2.3475 |
| yak-64k-v2.0 | 1.4212 | 1.4348 | 1.3910 | 1.4331 | 1.4226 | 1.4702 | 2.3311 |
| sidskarki-64k | 1.6796 | 1.7394 | 1.6665 | 1.7101 | 1.8044 | 1.4448 | 2.5899 |
| arkios | 1.9418 | 2.0267 | 2.0117 | 1.9946 | 2.0639 | 1.6878 | 2.8841 |
| sarvam-1 | 1.6509 | 1.7741 | 1.4632 | 1.6884 | 1.9503 | 2.6593 | 3.0525 |
| ancoda-128k | 1.3169 | 1.3465 | 1.2773 | 1.3384 | 1.3514 | 1.3745 | 2.0172 |
| yak-128k-v1 | 1.3894 | 1.3733 | 1.2966 | 1.3626 | 1.3597 | 1.3700 | 2.1452 |
| yak-128k-v2.0 | 1.3314 | 1.3419 | 1.3060 | 1.3565 | 1.3390 | 1.3823 | 2.1278 |
| llama3 | 2.7123 | 2.7575 | 2.6763 | 2.7778 | 2.8579 | 3.7641 | 4.2181 |
| sarvam-m | 2.2269 | 2.2631 | 1.9497 | 2.2706 | 2.5567 | 3.1689 | 3.7438 |
| qwen2.5 | 4.7181 | 4.6456 | 4.7682 | 4.6602 | 4.9619 | 6.5746 | 7.3897 |
| qwen3 | 4.7181 | 4.6456 | 4.7682 | 4.6602 | 4.9619 | 6.5746 | 7.3897 |
| ancoda-250k | 1.2693 | 1.2978 | 1.2404 | 1.2944 | 1.3055 | 1.3284 | 1.9008 |
| indicbert-v2 | 1.3747 | 1.4691 | 1.2368 | 1.4053 | 1.3547 | 1.5783 | 2.0679 |
| bloom | 1.5389 | 1.6432 | 1.3711 | 1.5871 | 1.7353 | 1.7238 | 2.6233 |
| gemma3 | 1.6065 | 1.7068 | 1.4246 | 1.6397 | 1.8470 | 2.2705 | 2.8870 |
| sarvam-30b | 1.6065 | 1.7068 | 1.4246 | 1.6397 | 1.8470 | 2.2707 | 2.8870 |
| o200k | 1.8041 | 1.8594 | 1.6493 | 1.8432 | 1.9206 | 2.3208 | 3.3018 |

o200k is reported in the column table and is not placed in a size class.

## Robustness (FLORES npi)

tokens(rewritten)/tokens(original) on FLORES npi devtest. Nukta swaps every U+0958–U+095F letter between precomposed and decomposed. NFC and NFD are temporary rewrites and are not written back. Misspellings swap ि↔ी, ु↔ू, and cycle श→स→ष→श at 10% of eligible sites with random.Random(20261104) walked in file order. CIs resample lines (10k, seed 20261104). SentencePiece uses offset mapping.

| Model | nukta | NFC | NFD | misspell 10% |
|---|---|---|---|---|
| ancoda-128k | 1.0000 [1.0000, 1.0001] whole-file first-byte | 1.0000 [1.0000, 1.0000] whole-file first-byte | 1.0000 [1.0000, 1.0000] whole-file first-byte | 1.0540 [1.0503, 1.0577] whole-file first-byte |
| ancoda-250k | 1.0000 [1.0000, 1.0000] whole-file first-byte | 1.0000 [1.0000, 1.0000] whole-file first-byte | 1.0000 [1.0000, 1.0001] whole-file first-byte | 1.0524 [1.0487, 1.0561] whole-file first-byte |
| ancoda-32k | 1.0000 [1.0000, 1.0001] whole-file first-byte | 1.0000 [1.0000, 1.0000] whole-file first-byte | 1.0000 [1.0000, 1.0001] whole-file first-byte | 1.0544 [1.0507, 1.0581] whole-file first-byte |
| ancoda-64k | 1.0000 [1.0000, 1.0001] whole-file first-byte | 1.0000 [1.0000, 1.0000] whole-file first-byte | 1.0000 [1.0000, 1.0001] whole-file first-byte | 1.0567 [1.0530, 1.0606] whole-file first-byte |
| ancoda-ne-64k | 1.0000 [1.0000, 1.0001] whole-file first-byte | 1.0000 [1.0000, 1.0000] whole-file first-byte | 1.0000 [1.0000, 1.0000] whole-file first-byte | 1.0608 [1.0567, 1.0649] whole-file first-byte |
| arkios | 1.0000 [1.0000, 1.0000] per-line | 1.0000 [1.0000, 1.0000] whole-file first-byte | 1.0000 [1.0000, 1.0000] per-line | 1.0548 [1.0512, 1.0585] whole-file first-byte |
| bloom | 1.0000 [1.0000, 1.0000] whole-file first-byte | 1.0000 [1.0000, 1.0000] whole-file first-byte | 1.0000 [1.0000, 1.0000] whole-file first-byte | 1.0478 [1.0445, 1.0512] whole-file first-byte |
| gemma3 | 1.0000 [0.9999, 1.0000] per-line | 1.0000 [1.0000, 1.0000] per-line | 1.0001 [1.0000, 1.0002] per-line | 1.0300 [1.0283, 1.0332] per-line |
| indicbert-v2 | 1.0000 [0.9999, 1.0000] per-line | 1.0000 [1.0000, 1.0000] per-line | 1.0000 [1.0000, 1.0000] per-line | 1.0513 [1.0478, 1.0547] per-line |
| llama3 | 1.0000 [1.0000, 1.0000] whole-file first-byte | 1.0000 [1.0000, 1.0000] whole-file first-byte | 1.0000 [1.0000, 1.0001] whole-file first-byte | 1.0059 [1.0052, 1.0065] whole-file first-byte |
| o200k | 1.0000 [1.0000, 1.0000] whole-file first-byte | 1.0000 [1.0000, 1.0000] whole-file first-byte | 1.0000 [1.0000, 1.0001] whole-file first-byte | 1.0320 [1.0296, 1.0346] whole-file first-byte |
| qwen2.5 | 1.0000 [1.0000, 1.0000] per-line | 1.0000 [1.0000, 1.0000] whole-file first-byte | 1.0000 [1.0000, 1.0000] per-line | 1.0017 [1.0013, 1.0021] whole-file first-byte |
| qwen3 | 1.0000 [1.0000, 1.0000] per-line | 1.0000 [1.0000, 1.0000] whole-file first-byte | 1.0000 [1.0000, 1.0000] per-line | 1.0017 [1.0013, 1.0021] whole-file first-byte |
| sarvam-1 | 1.0000 [1.0000, 1.0001] per-line | 1.0000 [1.0000, 1.0000] per-line | 1.0000 [1.0000, 1.0001] per-line | 1.0251 [1.0240, 1.0283] per-line |
| sarvam-30b | 1.0000 [0.9999, 1.0000] per-line | 1.0000 [1.0000, 1.0000] per-line | 1.0000 [0.9999, 1.0000] per-line | 1.0300 [1.0283, 1.0332] per-line |
| sarvam-m | 1.0000 [1.0000, 1.0000] whole-file first-byte | 1.0000 [1.0000, 1.0000] whole-file first-byte | 1.0001 [1.0000, 1.0002] whole-file first-byte | 1.0130 [1.0118, 1.0143] whole-file first-byte |
| sidskarki-64k | 1.0000 [1.0000, 1.0001] whole-file first-byte | 1.0000 [1.0000, 1.0000] whole-file first-byte | 1.0000 [1.0000, 1.0001] whole-file first-byte | 1.0577 [1.0538, 1.0616] whole-file first-byte |
| yak-128k-v1 | 1.0000 [0.9999, 1.0000] whole-file first-byte | 1.0000 [1.0000, 1.0000] whole-file first-byte | 1.0000 [1.0000, 1.0001] whole-file first-byte | 1.0630 [1.0587, 1.0673] whole-file first-byte |
| yak-128k-v2.0 | 1.0000 [0.9999, 1.0000] whole-file first-byte | 1.0000 [1.0000, 1.0000] whole-file first-byte | 1.0000 [1.0000, 1.0001] whole-file first-byte | 1.0585 [1.0545, 1.0624] whole-file first-byte |
| yak-32k-v1 | 1.0000 [0.9999, 1.0000] whole-file first-byte | 1.0000 [1.0000, 1.0000] whole-file first-byte | 1.0000 [1.0000, 1.0000] whole-file first-byte | 1.0587 [1.0549, 1.0626] whole-file first-byte |
| yak-64k-v1 | 1.0000 [0.9999, 1.0000] whole-file first-byte | 1.0000 [1.0000, 1.0000] whole-file first-byte | 1.0000 [1.0000, 1.0001] whole-file first-byte | 1.0626 [1.0585, 1.0668] whole-file first-byte |
| yak-64k-v2.0 | 1.0000 [0.9999, 1.0000] whole-file first-byte | 1.0000 [1.0000, 1.0000] whole-file first-byte | 1.0000 [1.0000, 1.0001] whole-file first-byte | 1.0586 [1.0548, 1.0626] whole-file first-byte |

HimalayaAI BPE-128k: skipped: license.

## Sealed test (once)

n=35688 seed=20261104 list_sha256=`71f744cea9e958351494fea6d70e35597daa833b4875bf00878caf64951c3da0`

Language field: index lang (bucket name, or glotlid label inside a pool). Sealed results are reported only; they do not drive tokenizer or config changes.

### Superseded table (zip misalignment)

The first sealed measurement (`artifacts/eval/m8/sealed.json`) zipped `fetch_texts` output (shard-sorted) with index metadata in scan order, so many documents were scored under the wrong language label. **That table is superseded and must not be used.** Kept below for audit only.

| Model | Code | English | Hindi | Marathi | Math | Nepali | Sanskrit | anp_Deva | awa_Deva | bho_Deva | gom_Deva | hne_Deva | mai_Deva | new_Deva | und_Deva |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ancoda-128k | 1.235 | 1.388 | 1.271 | 1.266 | 1.844 | 1.428 | 1.237 | 1.246 | 1.230 | 1.270 | 1.256 | 1.274 | 1.255 | 1.250 | 1.280 |
| ancoda-250k | 1.205 | 1.336 | 1.234 | 1.233 | 1.755 | 1.362 | 1.206 | 1.217 | 1.202 | 1.238 | 1.215 | 1.237 | 1.220 | 1.217 | 1.235 |
| ancoda-32k | 1.381 | 1.589 | 1.441 | 1.419 | 2.160 | 1.671 | 1.386 | 1.389 | 1.371 | 1.425 | 1.440 | 1.450 | 1.421 | 1.400 | 1.451 |
| ancoda-64k | 1.292 | 1.468 | 1.338 | 1.326 | 1.984 | 1.529 | 1.293 | 1.301 | 1.278 | 1.331 | 1.329 | 1.337 | 1.323 | 1.309 | 1.346 |
| ancoda-ne-64k | 1.249 | 1.793 | 1.345 | 1.285 | 2.147 | 2.083 | 1.250 | 1.260 | 1.240 | 1.291 | 1.279 | 1.294 | 1.271 | 1.265 | 1.294 |
| arkios | 1.443 | 1.648 | 1.558 | 1.485 | 2.436 | 1.597 | 1.449 | 1.458 | 1.423 | 1.489 | 1.504 | 1.493 | 1.493 | 1.470 | 1.540 |
| bloom | 1.526 | 1.541 | 1.559 | 1.562 | 2.355 | 1.509 | 1.527 | 1.534 | 1.491 | 1.562 | 1.583 | 1.613 | 1.588 | 1.547 | 1.647 |
| gemma3 | 2.046 | 1.843 | 2.051 | 2.069 | 2.724 | 1.709 | 2.038 | 2.058 | 2.021 | 2.074 | 2.036 | 2.127 | 2.117 | 2.061 | 2.145 |
| indicbert-v2 | 1.437 | 1.499 | 1.454 | 1.467 | 2.000 | 1.481 | 1.434 | 1.435 | 1.396 | 1.472 | 1.481 | 1.503 | 1.488 | 1.449 | 1.519 |
| llama3 | 3.555 | 2.831 | 3.584 | 3.571 | 4.218 | 2.325 | 3.519 | 3.532 | 3.627 | 3.560 | 3.461 | 3.606 | 3.584 | 3.563 | 3.581 |
| o200k | 2.108 | 1.929 | 2.152 | 2.145 | 3.012 | 1.722 | 2.101 | 2.103 | 2.114 | 2.134 | 2.127 | 2.183 | 2.173 | 2.127 | 2.209 |
| qwen2.5 | 6.191 | 4.553 | 6.213 | 6.213 | 7.235 | 3.473 | 6.127 | 6.160 | 6.317 | 6.161 | 6.023 | 6.232 | 6.218 | 6.192 | 6.240 |
| qwen3 | 6.191 | 4.553 | 6.213 | 6.213 | 7.235 | 3.473 | 6.127 | 6.160 | 6.317 | 6.161 | 6.023 | 6.232 | 6.218 | 6.192 | 6.240 |
| sarvam-1 | 2.365 | 2.112 | 2.348 | 2.385 | 3.067 | 1.979 | 2.358 | 2.377 | 2.329 | 2.399 | 2.349 | 2.469 | 2.441 | 2.380 | 2.470 |
| sarvam-30b | 2.046 | 1.844 | 2.051 | 2.069 | 2.724 | 1.710 | 2.038 | 2.058 | 2.021 | 2.074 | 2.036 | 2.127 | 2.117 | 2.061 | 2.145 |
| sarvam-m | 2.997 | 2.452 | 3.014 | 3.021 | 3.668 | 2.094 | 2.983 | 2.984 | 3.042 | 3.005 | 2.946 | 3.026 | 3.033 | 3.008 | 3.018 |
| sidskarki-64k | 1.222 | 1.784 | 1.324 | 1.259 | 2.140 | 2.118 | 1.220 | 1.239 | 1.205 | 1.262 | 1.269 | 1.272 | 1.261 | 1.242 | 1.281 |
| yak-128k-v1 | 1.236 | 1.417 | 1.320 | 1.267 | 2.030 | 1.439 | 1.236 | 1.245 | 1.228 | 1.270 | 1.251 | 1.277 | 1.255 | 1.250 | 1.279 |
| yak-128k-v2.0 | 1.246 | 1.422 | 1.326 | 1.277 | 2.005 | 1.454 | 1.247 | 1.255 | 1.240 | 1.281 | 1.266 | 1.284 | 1.269 | 1.260 | 1.286 |
| yak-32k-v1 | 1.395 | 1.624 | 1.497 | 1.435 | 2.325 | 1.678 | 1.398 | 1.400 | 1.374 | 1.440 | 1.454 | 1.460 | 1.436 | 1.417 | 1.483 |
| yak-64k-v1 | 1.299 | 1.499 | 1.390 | 1.332 | 2.162 | 1.534 | 1.299 | 1.307 | 1.284 | 1.337 | 1.338 | 1.352 | 1.328 | 1.316 | 1.359 |
| yak-64k-v2.0 | 1.311 | 1.507 | 1.402 | 1.345 | 2.141 | 1.555 | 1.313 | 1.320 | 1.298 | 1.351 | 1.357 | 1.358 | 1.343 | 1.328 | 1.370 |

### Corrected table (join by doc_id)

Re-measured with frozen M8 tokenizers; texts joined to metadata by `doc_id`. Source: `artifacts/eval/m8/followup/sealed_relabel.json`. Fertility is whitespace words. Document-level bootstrap CIs (10k, seed 20261104) for each Ancoda build vs its size-class competitors are in that JSON under `corrected_groups[*].cis`.

| Model | Code | English | Hindi | Marathi | Math | Nepali | Sanskrit | anp_Deva | awa_Deva | bho_Deva | gom_Deva | hne_Deva | mai_Deva | new_Deva | und_Deva |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ancoda-128k | 3.0511 | 1.3640 | 1.2512 | 1.6093 | 1.6770 | 1.2480 | 2.4784 | 1.2872 | 1.5302 | 1.3748 | 1.8275 | 1.4548 | 1.4797 | 1.3379 | 1.1040 |
| ancoda-250k | 2.8658 | 1.2868 | 1.2148 | 1.5011 | 1.5993 | 1.2152 | 2.3244 | 1.2477 | 1.4859 | 1.3165 | 1.6535 | 1.3950 | 1.4190 | 1.2507 | 1.0975 |
| ancoda-32k | 3.7195 | 1.6209 | 1.4126 | 2.0075 | 1.9590 | 1.4027 | 2.9760 | 1.4501 | 1.7496 | 1.5910 | 2.3359 | 1.6990 | 1.7254 | 1.5713 | 1.1413 |
| ancoda-64k | 3.3541 | 1.4708 | 1.3146 | 1.7726 | 1.7930 | 1.3077 | 2.7098 | 1.3487 | 1.6185 | 1.4694 | 2.0287 | 1.5551 | 1.5806 | 1.4358 | 1.1240 |
| ancoda-ne-64k | 5.4433 | 2.3120 | 1.5911 | 2.4684 | 2.6910 | 1.2631 | 3.0831 | 1.5797 | 1.8796 | 1.6790 | 2.5176 | 1.8700 | 1.8333 | 1.7131 | 1.2066 |
| arkios | 3.3536 | 1.3283 | 1.9078 | 2.8177 | 1.6591 | 1.4604 | 3.4975 | 1.8694 | 2.0275 | 1.9448 | 2.9944 | 1.9519 | 2.1285 | 2.0759 | 1.1960 |
| bloom | 2.8062 | 1.3077 | 1.3341 | 1.7251 | 1.5124 | 1.5547 | 3.2281 | 1.3749 | 1.6152 | 1.6284 | 2.3243 | 1.6685 | 1.8833 | 1.9292 | 1.1191 |
| gemma3 | 3.1963 | 1.3204 | 1.3562 | 1.9575 | 1.6614 | 2.0629 | 3.4568 | 1.4177 | 1.8111 | 1.6990 | 2.6344 | 1.7461 | 2.0093 | 1.9750 | 1.1978 |
| indicbert-v2 | 3.4059 | 1.3032 | 1.2062 | 1.4606 | 1.6586 | 1.4518 | 2.5731 | 1.2540 | 1.4571 | 1.4488 | 1.8586 | 1.5709 | 1.5277 | 1.7084 | 1.0997 |
| llama3 | 2.6783 | 1.2965 | 2.5802 | 3.8141 | 1.5774 | 3.5784 | 4.8829 | 2.5276 | 2.7544 | 2.8065 | 3.9052 | 2.4877 | 3.0560 | 2.9389 | 1.2043 |
| o200k | 2.7020 | 1.2834 | 1.5783 | 2.5094 | 1.5666 | 2.1384 | 3.9701 | 1.6155 | 1.9761 | 1.8927 | 2.9421 | 1.8952 | 2.1362 | 2.3363 | 1.2085 |
| qwen2.5 | 2.7294 | 1.3246 | 4.5337 | 6.3856 | 1.6469 | 6.2070 | 8.2276 | 4.3872 | 4.7427 | 4.5885 | 6.6030 | 3.9744 | 5.1873 | 4.7306 | 2.1511 |
| qwen3 | 2.7294 | 1.3246 | 4.5337 | 6.3856 | 1.6469 | 6.2070 | 8.2276 | 4.3872 | 4.7427 | 4.5885 | 6.6030 | 3.9744 | 5.1873 | 4.7306 | 2.1511 |
| sarvam-1 | 4.0384 | 1.5294 | 1.4527 | 1.9400 | 1.9790 | 2.3864 | 3.8490 | 1.5211 | 1.9605 | 1.8065 | 2.6474 | 1.8868 | 2.1768 | 2.1102 | 1.2397 |
| sarvam-30b | 3.1983 | 1.3210 | 1.3562 | 1.9576 | 1.6663 | 2.0629 | 3.4575 | 1.4177 | 1.8111 | 1.6981 | 2.6344 | 1.7461 | 2.0094 | 1.9754 | 1.1982 |
| sarvam-m | 2.9085 | 1.3335 | 1.9038 | 3.1085 | 1.6383 | 3.0205 | 4.3713 | 1.9424 | 2.4966 | 2.3200 | 3.4204 | 2.2491 | 2.7019 | 2.6261 | 1.2745 |
| sidskarki-64k | 5.2542 | 2.4264 | 1.6105 | 2.4875 | 2.7132 | 1.2323 | 3.1240 | 1.5894 | 1.7088 | 1.6762 | 2.5775 | 1.8484 | 1.8478 | 1.7299 | 1.1291 |
| yak-128k-v1 | 3.0364 | 1.3563 | 1.2837 | 2.3159 | 1.6741 | 1.2494 | 2.8286 | 1.3308 | 1.5986 | 1.3828 | 2.4008 | 1.6206 | 1.4940 | 1.3834 | 1.1290 |
| yak-128k-v2.0 | 3.0874 | 1.3774 | 1.2902 | 2.2483 | 1.6958 | 1.2596 | 2.7762 | 1.3268 | 1.5279 | 1.3512 | 2.3986 | 1.5533 | 1.4445 | 1.3114 | 1.1404 |
| yak-32k-v1 | 3.6176 | 1.5960 | 1.4920 | 2.6272 | 1.9220 | 1.4192 | 3.2927 | 1.5242 | 1.8567 | 1.6468 | 2.7469 | 1.8208 | 1.7804 | 1.6853 | 1.1768 |
| yak-64k-v1 | 3.2842 | 1.4479 | 1.3653 | 2.4474 | 1.7709 | 1.3161 | 3.0466 | 1.4056 | 1.7098 | 1.4983 | 2.5471 | 1.7041 | 1.6121 | 1.5149 | 1.1439 |
| yak-64k-v2.0 | 3.3469 | 1.4792 | 1.3749 | 2.4282 | 1.8014 | 1.3285 | 2.9905 | 1.4071 | 1.6304 | 1.4505 | 2.5833 | 1.6454 | 1.5506 | 1.4120 | 1.1548 |

### sidskarki FLORES path (1.4448 vs 1.3699)

Hub ships only `nepali_bpe_64k.model` (no `tokenizer.json`). Both paths use `sentencepiece.SentencePieceProcessor` on that file. The model has no BOS/EOS (`bos_id=eos_id=-1`).

| Path | Mode | tokens | BOS | EOS | unk | `<0x0A>` | fertility | round-trip |
|---|---|---:|---:|---:|---:|---:|---:|---|
| whole-file | `'\n'.join(lines)` then one `encode()` | 27019 | 0 | 0 | 0 | 1011 | 1.4448 | exact False (content OK) |
| per-line | `encode(line)` per line, sum | 25619 | 0 | 0 | 0 | 0 | 1.3699 | exact rate 0.9951 |

M8 scoreboard / misspell absolute used whole-file (1.4448 = Yak `fertility`). The whitespace follow-up FLORES number used per-line (1.3699 = Yak `fertility_segments` / npi-per-line). Sealed, Aya and fw2 already encode each document alone via `Encoder` → SentencePiece `.model`.

**Chosen path:** per-unit SentencePiece `.model` (FLORES: per line; Aya/fw2/sealed: per doc). No special tokens added; closest to lossless; no cross-unit newline byte tokens. Recomputed fertilities (chosen path):

| Set | ancoda-64k | ancoda-ne-64k | sidskarki | 64k vs sid | ne vs sid |
|---|---:|---:|---:|---|---|
| FLORES npi | 1.4508 | 1.4052 | 1.3699 | LOSE +0.0809 [+0.0733,+0.0884] | LOSE +0.0352 [+0.0283,+0.0428] |
| Aya npi | 1.4407 | 1.3962 | 1.4068 | LOSE +0.0340 [+0.0210,+0.0433] | WIN −0.0106 [−0.0162,−0.0050] |
| fw2_npi_test | 1.2928 | 1.2475 | 1.2361 | LOSE +0.0567 [+0.0560,+0.0574] | LOSE +0.0114 [+0.0108,+0.0119] |
| sealed Nepali (full) | 1.3077 | 1.2631 | 1.2323 | LOSE +0.0754 [+0.0738,+0.0769] | LOSE +0.0307 [+0.0293,+0.0322] |

Detail: `artifacts/eval/m8/followup/sidskarki_flores_path_audit.json`, `sidskarki_chosen_path_recompute.json`.

### Sealed list without joiner sources

Dropped 65 sealed docs that were synthetic ZWJ/ZWNJ sources in n60j5∪n60j10. Original `docs/sealed_test_set.json` unchanged. Reduced list: `docs/sealed_test_set_no_joiners.json` (n=35623, list_sha256=`820f385638336be6b904c1f3b10c56f380ef73515f76375eebd1f1e258156336`).

Sealed Nepali headline on the reduced list (n=28094), doc-level CIs:

| Pair | fertility ours / base | verdict |
|---|---|---|
| ancoda-64k vs sidskarki | 1.3077 / 1.2323 | LOSE +0.0754 [+0.0738,+0.0770] |
| ancoda-64k vs yak-64k-v1 | 1.3077 / 1.3162 | WIN −0.0084 [−0.0088,−0.0080] |
| ancoda-64k vs yak-64k-v2.0 | 1.3077 / 1.3285 | WIN −0.0208 [−0.0212,−0.0204] |
| ancoda-ne-64k vs sidskarki | 1.2631 / 1.2323 | LOSE +0.0308 [+0.0293,+0.0322] |
| ancoda-ne-64k vs yak-64k-v1 | 1.2631 / 1.3162 | WIN −0.0531 [−0.0539,−0.0522] |
| ancoda-ne-64k vs yak-64k-v2.0 | 1.2631 / 1.3285 | WIN −0.0654 [−0.0663,−0.0646] |

Selected extra CIs (reduced sealed):

| Pair | lang | docs | verdict |
|---|---|---:|---|
| ancoda-250k vs indicbert-v2 | Hindi | 1352 | LOSE +0.0086 [+0.0037,+0.0141] |
| ancoda-250k vs indicbert-v2 | Marathi | 401 | LOSE +0.0405 [+0.0309,+0.0509] |
| ancoda-250k vs indicbert-v2 | awa_Deva | 15 | TIE +0.0288 [−0.3215,+0.2417] |
| ancoda-128k vs yak-128k-v1 | Nepali | 28094 | WIN −0.0015 [−0.0018,−0.0012] |
| ancoda-64k vs yak-64k-v2.0 | bho_Deva | 172 | LOSE +0.0190 [+0.0051,+0.0425] |
| ancoda-64k vs yak-64k-v2.0 | mai_Deva | 258 | LOSE +0.0300 [+0.0235,+0.0362] |
| ancoda-64k vs yak-64k-v2.0 | new_Deva | 450 | LOSE +0.0238 [+0.0150,+0.0369] |

Unweighted mean of 11 Devanagari groups (Hin, Mar, Nep, San, anp, awa, bho, gom, hne, mai, new), doc-level bootstrap on the reduced sealed set:

| Pair | verdict |
|---|---|
| ancoda-32k vs yak-32k-v1 | WIN −0.1792 [−0.1937,−0.1696] |
| ancoda-64k vs yak-64k-v1 | WIN −0.1841 [−0.1988,−0.1734] |
| ancoda-64k vs yak-64k-v2.0 | WIN −0.1509 [−0.1585,−0.1425] |
| ancoda-64k vs sidskarki | WIN −0.2991 [−0.3261,−0.2808] |
| ancoda-64k vs arkios | WIN −0.5940 [−0.6239,−0.5728] |
| ancoda-64k vs sarvam-1 | WIN −0.5087 [−0.5386,−0.4885] |
| ancoda-ne-64k vs yak-64k-v1 | LOSE +0.1192 [+0.1016,+0.1323] |
| ancoda-ne-64k vs yak-64k-v2.0 | LOSE +0.1524 [+0.1368,+0.1672] |
| ancoda-ne-64k vs sidskarki | TIE +0.0042 [−0.0159,+0.0177] |
| ancoda-ne-64k vs arkios | WIN −0.2907 [−0.3197,−0.2694] |
| ancoda-ne-64k vs sarvam-1 | WIN −0.2054 [−0.2334,−0.1858] |
| ancoda-128k vs yak-128k-v1 | WIN −0.1827 [−0.1967,−0.1715] |
| ancoda-128k vs yak-128k-v2.0 | WIN −0.1463 [−0.1547,−0.1375] |
| ancoda-128k vs llama3 | WIN −1.6775 [−1.7319,−1.6359] |
| ancoda-128k vs sarvam-m | WIN −1.2074 [−1.2489,−1.1770] |
| ancoda-128k vs qwen2.5 | WIN −3.8808 [−3.9882,−3.7950] |
| ancoda-128k vs qwen3 | WIN −3.8808 [−3.9882,−3.7950] |
| ancoda-250k vs indicbert-v2 | WIN −0.1358 [−0.1739,−0.1113] |
| ancoda-250k vs bloom | WIN −0.3856 [−0.4293,−0.3563] |
| ancoda-250k vs gemma3 | WIN −0.5547 [−0.5942,−0.5284] |
| ancoda-250k vs sarvam-30b | WIN −0.5548 [−0.5942,−0.5285] |

Full JSON: `artifacts/eval/m8/followup/sealed_nojjoiners_remeasure.json`. Sealed remains report-only.


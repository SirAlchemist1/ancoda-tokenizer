# Tokens per KB on the sealed Code and Math groups

Analysis only. No training run, and the server was not used. Reproduction: `scripts/analysis_sealed_code_math_kb.py` (tokenizers 0.23.3).

Same panel and counting as the code/math tokens-per-KB table from `artifacts/eval/final_build/followup/code_math_kb.json`: each document is encoded whole with `Encoder.ids`, and tokens per KB is 1,024 × total tokens ÷ total UTF-8 bytes. That table used held-out n60j5 snapshot files (778 Code, 551 Math). This one uses the sealed holdout groups: Code 565 documents (3,589,870 bytes) and Math 364 documents (2,402,282 bytes), from `data/model_eval/eval/sealed.jsonl.zst`. Sealed numbers are report-only.

Lower is better.

| Tokenizer | Code, sealed | Code, n60j5 | Math, sealed | Math, n60j5 |
|---|---:|---:|---:|---:|
| ancoda-32k | 345.53 | 357.95 | 323.13 | 329.44 |
| ancoda-64k | 311.58 | 320.68 | 295.76 | 301.51 |
| ancoda-128k | 283.44 | 289.20 | 276.61 | 281.66 |
| ancoda-250k | 266.22 | 275.35 | 263.80 | 269.01 |
| yak-32k-v1 | 336.07 | 350.71 | 317.03 | 326.39 |
| yak-64k-v1 | 305.09 | 316.44 | 292.11 | 301.06 |
| yak-64k-v2.0 | 310.91 | 324.68 | 297.14 | 306.21 |
| yak-128k-v1 | 282.07 | 291.75 | 276.14 | 284.95 |
| yak-128k-v2.0 | 286.82 | 297.04 | 279.71 | 288.58 |
| sidskarki-64k | 488.10 | 501.88 | 447.53 | 445.92 |
| arkios | 311.54 | 322.76 | 273.67 | 276.19 |
| indicbert-v2 | 316.40 | 334.47 | 273.57 | 271.16 |

## Ancoda against Yak v1

Paired bootstrap over documents, 10000 resamples, seed 20261104. Each draw resamples documents with replacement and recomputes 1,024 × (tokens ours − tokens Yak) ÷ bytes. Negative means Ancoda uses fewer tokens. WIN means the 95% interval is entirely below 0, LOSE entirely above 0, TIE otherwise.

| Group | Ours | Yak v1 | ours − Yak, tokens/KB | relative |
|---|---|---|---|---:|
| Code | ancoda-32k | yak-32k-v1 | LOSE +9.46 [+8.11, +10.71] | +2.82% |
| Code | ancoda-64k | yak-64k-v1 | LOSE +6.49 [+4.45, +8.25] | +2.13% |
| Code | ancoda-128k | yak-128k-v1 | TIE +1.36 [-1.65, +3.79] | +0.48% |
| Math | ancoda-32k | yak-32k-v1 | LOSE +6.11 [+4.97, +7.26] | +1.93% |
| Math | ancoda-64k | yak-64k-v1 | LOSE +3.65 [+2.32, +4.80] | +1.25% |
| Math | ancoda-128k | yak-128k-v1 | TIE +0.47 [-0.75, +1.53] | +0.17% |

JSON: `artifacts/analysis/sealed_code_math_kb.json` (manifest `artifacts/analysis/sealed_code_math_kb.manifest.json`).

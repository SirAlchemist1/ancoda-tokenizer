# Joiner loss split

Analysis only. No training run, and the server queue was not touched. Final checkpoints (step 7,630), seeds 1–3, arms A/B/C. Device: `mps`. Reproduction: `scripts/analysis_joiner_loss.py`.

BPB denominator: UTF-8 bytes of the original unmodified text (`original_bytes`). Bits = NLL nats / ln 2. Scoring matches `ancoda_tok.m9.bpb` (EOS context-only, non-overlapping 1,023-token windows).

## (a)(b) Total bits, BPB, joiner vs other

| Arm | Seed | Set | Docs | Bytes | Total bits | BPB | Bits on joiner tokens | Bits on other |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| A (ancoda-64k) | 1 | `sealed_npi_2000` | 2000 | 10625570 | 5089288.385 | 0.478966 | 0.000 | 5089288.385 |
| A (ancoda-64k) | 1 | `sealed_npi_zwj` | 2000 | 10625570 | 11107512.679 | 1.045357 | 4552476.153 | 6555036.526 |
| A (ancoda-64k) | 1 | `sealed_npi_zwnj` | 2000 | 10625570 | 11847896.550 | 1.115036 | 4928818.065 | 6919078.485 |
| A (ancoda-64k) | 2 | `sealed_npi_2000` | 2000 | 10625570 | 5072284.095 | 0.477366 | 0.000 | 5072284.095 |
| A (ancoda-64k) | 2 | `sealed_npi_zwj` | 2000 | 10625570 | 11112677.067 | 1.045843 | 4419971.421 | 6692705.646 |
| A (ancoda-64k) | 2 | `sealed_npi_zwnj` | 2000 | 10625570 | 11986137.758 | 1.128047 | 4804290.738 | 7181847.021 |
| A (ancoda-64k) | 3 | `sealed_npi_2000` | 2000 | 10625570 | 5083505.551 | 0.478422 | 0.000 | 5083505.551 |
| A (ancoda-64k) | 3 | `sealed_npi_zwj` | 2000 | 10625570 | 11249906.641 | 1.058758 | 4666991.508 | 6582915.133 |
| A (ancoda-64k) | 3 | `sealed_npi_zwnj` | 2000 | 10625570 | 12121686.677 | 1.140803 | 5134800.194 | 6986886.483 |
| B (yak-64k-v1) | 1 | `sealed_npi_2000` | 2000 | 10625570 | 5118528.585 | 0.481718 | 0.000 | 5118528.585 |
| B (yak-64k-v1) | 1 | `sealed_npi_zwj` | 2000 | 10625570 | 10054791.058 | 0.946283 | 1681705.299 | 8373085.759 |
| B (yak-64k-v1) | 1 | `sealed_npi_zwnj` | 2000 | 10625570 | 10213787.245 | 0.961246 | 161914.567 | 10051872.678 |
| B (yak-64k-v1) | 2 | `sealed_npi_2000` | 2000 | 10625570 | 5099687.161 | 0.479945 | 0.000 | 5099687.161 |
| B (yak-64k-v1) | 2 | `sealed_npi_zwj` | 2000 | 10625570 | 10079256.121 | 0.948585 | 1665139.158 | 8414116.963 |
| B (yak-64k-v1) | 2 | `sealed_npi_zwnj` | 2000 | 10625570 | 10208957.917 | 0.960792 | 160310.996 | 10048646.921 |
| B (yak-64k-v1) | 3 | `sealed_npi_2000` | 2000 | 10625570 | 5103730.377 | 0.480325 | 0.000 | 5103730.377 |
| B (yak-64k-v1) | 3 | `sealed_npi_zwj` | 2000 | 10625570 | 10133692.991 | 0.953708 | 1810966.753 | 8322726.238 |
| B (yak-64k-v1) | 3 | `sealed_npi_zwnj` | 2000 | 10625570 | 10261077.004 | 0.965697 | 181941.545 | 10079135.459 |
| C (sidskarki-64k) | 1 | `sealed_npi_2000` | 2000 | 10625570 | 5329346.762 | 0.501559 | 0.000 | 5329346.762 |
| C (sidskarki-64k) | 1 | `sealed_npi_zwj` | 2000 | 10625570 | 10201119.310 | 0.960054 | 0.000 | 10201119.310 |
| C (sidskarki-64k) | 1 | `sealed_npi_zwnj` | 2000 | 10625570 | 10457391.714 | 0.984172 | 0.000 | 10457391.714 |
| C (sidskarki-64k) | 2 | `sealed_npi_2000` | 2000 | 10625570 | 5314363.711 | 0.500149 | 0.000 | 5314363.711 |
| C (sidskarki-64k) | 2 | `sealed_npi_zwj` | 2000 | 10625570 | 10234448.916 | 0.963191 | 0.000 | 10234448.916 |
| C (sidskarki-64k) | 2 | `sealed_npi_zwnj` | 2000 | 10625570 | 10519193.754 | 0.989989 | 0.000 | 10519193.754 |
| C (sidskarki-64k) | 3 | `sealed_npi_2000` | 2000 | 10625570 | 5316854.629 | 0.500383 | 0.000 | 5316854.629 |
| C (sidskarki-64k) | 3 | `sealed_npi_zwj` | 2000 | 10625570 | 10166217.211 | 0.956769 | 0.000 | 10166217.211 |
| C (sidskarki-64k) | 3 | `sealed_npi_zwnj` | 2000 | 10625570 | 10450260.276 | 0.983501 | 0.000 | 10450260.276 |

## (c) Extra bits over clean (`sealed_npi_2000`), by doc_id

Extra = bits(perturbed) − bits(clean), paired by `doc_id`. Class columns are bits_class(pert) − bits_class(clean) and sum to Extra.

| Arm | Seed | Set | Extra bits | Extra on joiner toks | On neighbors | On rest |
|---|---:|---|---:|---:|---:|---:|
| A (ancoda-64k) | 1 | `sealed_npi_zwj` | 6018224.294 | 4552476.153 | 3407209.172 | -1941461.031 |
| A (ancoda-64k) | 1 | `sealed_npi_zwnj` | 6758608.166 | 4928818.065 | 3725515.962 | -1895725.861 |
| A (ancoda-64k) | 2 | `sealed_npi_zwj` | 6040392.973 | 4419971.421 | 3466667.393 | -1846245.842 |
| A (ancoda-64k) | 2 | `sealed_npi_zwnj` | 6913853.664 | 4804290.738 | 3865112.687 | -1755549.761 |
| A (ancoda-64k) | 3 | `sealed_npi_zwj` | 6166401.090 | 4666991.508 | 3396436.058 | -1897026.476 |
| A (ancoda-64k) | 3 | `sealed_npi_zwnj` | 7038181.127 | 5134800.194 | 3743679.737 | -1840298.805 |
| B (yak-64k-v1) | 1 | `sealed_npi_zwj` | 4936262.472 | 1681705.299 | 3811484.660 | -556927.487 |
| B (yak-64k-v1) | 1 | `sealed_npi_zwnj` | 5095258.659 | 161914.567 | 3180942.389 | 1752401.703 |
| B (yak-64k-v1) | 2 | `sealed_npi_zwj` | 4979568.960 | 1665139.158 | 3863489.478 | -549059.676 |
| B (yak-64k-v1) | 2 | `sealed_npi_zwnj` | 5109270.757 | 160310.996 | 3238728.424 | 1710231.337 |
| B (yak-64k-v1) | 3 | `sealed_npi_zwj` | 5029962.614 | 1810966.753 | 3805445.161 | -586449.300 |
| B (yak-64k-v1) | 3 | `sealed_npi_zwnj` | 5157346.627 | 181941.545 | 3219862.575 | 1755542.507 |
| C (sidskarki-64k) | 1 | `sealed_npi_zwj` | 4871772.548 | 0.000 | 0.000 | 4871772.548 |
| C (sidskarki-64k) | 1 | `sealed_npi_zwnj` | 5128044.953 | 0.000 | 0.000 | 5128044.953 |
| C (sidskarki-64k) | 2 | `sealed_npi_zwj` | 4920085.205 | 0.000 | 0.000 | 4920085.205 |
| C (sidskarki-64k) | 2 | `sealed_npi_zwnj` | 5204830.043 | 0.000 | 0.000 | 5204830.043 |
| C (sidskarki-64k) | 3 | `sealed_npi_zwj` | 4849362.582 | 0.000 | 0.000 | 4849362.582 |
| C (sidskarki-64k) | 3 | `sealed_npi_zwnj` | 5133405.647 | 0.000 | 0.000 | 5133405.647 |

## (d) Arm A: joiner token types — stream count vs mean loss

Joiner-containing vocabulary entries with ≥1 occurrence on the joiner eval sets: **633**. Spearman rank correlation (stream count vs mean bits on joiner sets): **-0.4646840442493009** (n=633).

Top 15 by mean bits on the joiner sets:

| Token id | Stream count | Mean bits (joiner sets) | Occurrences scored |
|---:|---:|---:|---:|
| 37546 | 0 | 27.0822 | 3465 |
| 34500 | 0 | 26.0699 | 3972 |
| 45342 | 0 | 25.9299 | 1992 |
| 60384 | 0 | 25.9291 | 1479 |
| 46219 | 0 | 25.9199 | 2016 |
| 58626 | 0 | 25.8931 | 1074 |
| 46562 | 20 | 25.8340 | 2016 |
| 42717 | 0 | 25.8196 | 2172 |
| 61596 | 0 | 25.7705 | 648 |
| 56041 | 0 | 25.7136 | 1365 |
| 56894 | 0 | 25.6525 | 1083 |
| 50923 | 0 | 25.5892 | 1299 |
| 55520 | 0 | 25.5752 | 972 |
| 19136 | 0 | 25.5348 | 2598 |
| 62700 | 0 | 25.4944 | 699 |

JSON: `artifacts/analysis/joiner_loss.json`.

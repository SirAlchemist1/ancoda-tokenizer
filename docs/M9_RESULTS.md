# M9 results

Generated from `artifacts/m9/server_copy/evals/*.json` by `scripts/m9/build_m9_results.py`. Do not hand-edit the tables.

Tokenizers frozen. Pre-registration: `docs/M9_PREREG.md`.

## Verification

- BPB denominator: UTF-8 bytes of the ORIGINAL text via row['original_bytes']; for perturbed sets this is the unmodified text length (evalprep sets original_bytes from the original).
- EOS scoring: Identical for every arm: each window is [EOS]+span; cross-entropy scores only the text-token positions; EOS is context and is never a target (docs/M9_PREREG.md Evaluation windows; python/ancoda_tok/m9/bpb.py _score_spans).
- Same eval documents across arms: PASS (32 groups; sealed doc_id list sha256 `8501f43daeaf37dae8e729d66bed18c6803d26ce7d692792822975a15c65ff6a`).
- Independent 300-doc recompute: PASS (tolerance 0.001 BPB).
  - ancoda-64k: pipeline 0.487752, standalone 0.487738, |Δ|=0.000014
  - yak-64k-v1: pipeline 0.490893, standalone 0.490874, |Δ|=0.000019
  - sidskarki-64k: pipeline 0.511574, standalone 0.511571, |Δ|=0.000003

## Primary verdicts (pre-registered rule)

Label is **A better** only when every seed difference BPB(A)−BPB(other) is strictly negative and |mean| > sample SD (n−1). Otherwise **no measurable difference**.

### equal tokens

| Comparison | Metric | d1 | d2 | d3 | mean | sample SD | verdict |
|---|---|---:|---:|---:|---:|---:|---|
| A vs B | Nepali | -0.002153 | -0.002298 | -0.001375 | -0.001942 | 0.000497 | **A better** |
| A vs B | Dev11 | -0.002059 | -0.002777 | -0.001687 | -0.002174 | 0.000554 | **A better** |
| A vs C | Nepali | -0.019702 | -0.019871 | -0.019021 | -0.019531 | 0.000450 | **A better** |
| A vs C | Dev11 | -0.033098 | -0.033680 | -0.032418 | -0.033065 | 0.000632 | **A better** |

| Seed | A Nepali | B Nepali | C Nepali | A Dev11 | B Dev11 | C Dev11 |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.466263 | 0.468417 | 0.485965 | 0.748445 | 0.750504 | 0.781543 |
| 2 | 0.464589 | 0.466887 | 0.484460 | 0.744314 | 0.747091 | 0.777994 |
| 3 | 0.465655 | 0.467030 | 0.484676 | 0.745996 | 0.747682 | 0.778413 |
| mean | 0.465503 | 0.467445 | 0.485034 | 0.746251 | 0.748426 | 0.779317 |

### equal bytes

| Comparison | Metric | d1 | d2 | d3 | mean | sample SD | verdict |
|---|---|---:|---:|---:|---:|---:|---|
| A vs B | Nepali | -0.002450 | -0.002335 | -0.002118 | -0.002301 | 0.000168 | **A better** |
| A vs B | Dev11 | -0.002352 | -0.003662 | -0.004892 | -0.003635 | 0.001270 | **A better** |
| A vs C | Nepali | -0.009208 | -0.009045 | -0.008619 | -0.008958 | 0.000304 | **A better** |
| A vs C | Dev11 | -0.013882 | -0.017733 | -0.012829 | -0.014815 | 0.002581 | **A better** |

| Seed | A Nepali | B Nepali | C Nepali | A Dev11 | B Dev11 | C Dev11 |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.476757 | 0.479207 | 0.485965 | 0.767661 | 0.770013 | 0.781543 |
| 2 | 0.475415 | 0.477750 | 0.484460 | 0.760261 | 0.763923 | 0.777994 |
| 3 | 0.476057 | 0.478175 | 0.484676 | 0.765584 | 0.770476 | 0.778413 |
| mean | 0.476076 | 0.478377 | 0.485034 | 0.764502 | 0.768137 | 0.779317 |

## Amended comparison: A vs C on sidskarki-lossless docs

Restricted to documents where `sidskarki.decode(sidskarki.encode(x)) == x`. Does not replace the primary label.

### equal tokens

| Seed | Nepali A | Nepali C | A−C | lossless Nepali docs | Dev11 A | Dev11 C | A−C |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.485166 | 0.505763 | -0.020597 | 22572 | 0.758387 | 0.791536 | -0.033150 |
| 2 | 0.483420 | 0.504338 | -0.020918 | 22572 | 0.754471 | 0.787906 | -0.033435 |
| 3 | 0.484483 | 0.504463 | -0.019980 | 22572 | 0.755833 | 0.788139 | -0.032305 |

### equal bytes

| Seed | Nepali A | Nepali C | A−C | lossless Nepali docs | Dev11 A | Dev11 C | A−C |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.495846 | 0.505763 | -0.009917 | 22572 | 0.777597 | 0.791536 | -0.013940 |
| 2 | 0.494550 | 0.504338 | -0.009787 | 22572 | 0.770502 | 0.787906 | -0.017404 |
| 3 | 0.495073 | 0.504463 | -0.009390 | 22572 | 0.775283 | 0.788139 | -0.012856 |

### sidskarki round-trip rates

| Eval set | docs | ok | rate |
|---|---:|---:|---:|
| `flores/awa.jsonl.zst` | 1012 | 930 | 0.9190 |
| `flores/bho.jsonl.zst` | 1012 | 1012 | 1.0000 |
| `flores/eng.jsonl.zst` | 1012 | 1012 | 1.0000 |
| `flores/hin.jsonl.zst` | 1012 | 1006 | 0.9941 |
| `flores/hne.jsonl.zst` | 1012 | 982 | 0.9704 |
| `flores/mag.jsonl.zst` | 1012 | 1012 | 1.0000 |
| `flores/mai.jsonl.zst` | 1012 | 1012 | 1.0000 |
| `flores/mar.jsonl.zst` | 1012 | 955 | 0.9437 |
| `flores/npi.jsonl.zst` | 1012 | 1007 | 0.9951 |
| `flores/san.jsonl.zst` | 1012 | 997 | 0.9852 |
| `perturb/flores_npi_misspell.jsonl.zst` | 1012 | 1007 | 0.9951 |
| `perturb/flores_npi_zwj.jsonl.zst` | 1012 | 1007 | 0.9951 |
| `perturb/flores_npi_zwnj.jsonl.zst` | 1012 | 1007 | 0.9951 |
| `perturb/sealed_npi_misspell.jsonl.zst` | 2000 | 2000 | 1.0000 |
| `perturb/sealed_npi_zwj.jsonl.zst` | 2000 | 2000 | 1.0000 |
| `perturb/sealed_npi_zwnj.jsonl.zst` | 2000 | 2000 | 1.0000 |
| `sealed.jsonl.zst` | 35623 | 29067 | 0.8160 |
| `sealed_npi_2000.jsonl.zst` | 2000 | 2000 | 1.0000 |

| Language | docs | ok | rate |
|---|---:|---:|---:|
| Code | 565 | 97 | 0.1717 |
| English | 3576 | 3551 | 0.9930 |
| Hindi | 1352 | 1082 | 0.8003 |
| Marathi | 401 | 237 | 0.5910 |
| Math | 364 | 322 | 0.8846 |
| Nepali | 36094 | 30572 | 0.8470 |
| Sanskrit | 155 | 141 | 0.9097 |
| anp_Deva | 138 | 128 | 0.9275 |
| awa | 1012 | 930 | 0.9190 |
| awa_Deva | 15 | 15 | 1.0000 |
| bho | 1012 | 1012 | 1.0000 |
| bho_Deva | 172 | 169 | 0.9826 |
| eng | 1012 | 1012 | 1.0000 |
| gom_Deva | 17 | 8 | 0.4706 |
| hin | 1012 | 1006 | 0.9941 |
| hne | 1012 | 982 | 0.9704 |
| hne_Deva | 46 | 44 | 0.9565 |
| mag | 1012 | 1012 | 1.0000 |
| mai | 1012 | 1012 | 1.0000 |
| mai_Deva | 258 | 234 | 0.9070 |
| mar | 1012 | 955 | 0.9437 |
| new_Deva | 450 | 449 | 0.9978 |
| npi | 4048 | 4028 | 0.9951 |
| san | 1012 | 997 | 0.9852 |
| und_Deva | 20 | 18 | 0.9000 |

## Secondary

### Sealed languages (equal tokens, seed mean of A/B/C)

| Language | A | B | C | A−B | A−C |
|---|---:|---:|---:|---:|---:|
| English | 1.195976 | 1.195253 | 1.273680 | +0.000723 | -0.077704 |
| Hindi | 0.570930 | 0.574712 | 0.597466 | -0.003782 | -0.026536 |
| Marathi | 0.682259 | 0.692891 | 0.720390 | -0.010632 | -0.038132 |
| Code | 0.954438 | 0.960804 | 0.996146 | -0.006365 | -0.041708 |
| Math | 1.538859 | 1.548482 | 1.563964 | -0.009622 | -0.025104 |
| anp_Deva | 0.633306 | 0.630963 | 0.652706 | +0.002342 | -0.019400 |
| awa_Deva | 0.763602 | 0.758110 | 0.810800 | +0.005492 | -0.047197 |
| bho_Deva | 0.784229 | 0.784953 | 0.814789 | -0.000724 | -0.030560 |
| gom_Deva | 0.762793 | 0.756369 | 0.803977 | +0.006424 | -0.041184 |
| hne_Deva | 0.946568 | 0.964581 | 0.997770 | -0.018013 | -0.051201 |
| mai_Deva | 0.786354 | 0.791035 | 0.823241 | -0.004681 | -0.036887 |
| new_Deva | 0.833755 | 0.848480 | 0.869082 | -0.014725 | -0.035327 |
| und_Deva | 1.137799 | 1.131623 | 1.165147 | +0.006177 | -0.027347 |
| Sanskrit | 0.979467 | 0.963145 | 0.997231 | +0.016322 | -0.017764 |
| Nepali | 0.465503 | 0.467445 | 0.485034 | -0.001942 | -0.019531 |

#### Every A loss on sealed languages

| Checkpoint | Seed | Language | vs | A | other | diff |
|---|---:|---|---|---:|---:|---:|
| equal_tokens | 1 | English | B | 1.197136 | 1.195908 | +0.001229 |
| equal_tokens | 1 | anp_Deva | B | 0.634544 | 0.631190 | +0.003354 |
| equal_tokens | 1 | awa_Deva | B | 0.768071 | 0.761673 | +0.006397 |
| equal_tokens | 1 | gom_Deva | B | 0.765453 | 0.759184 | +0.006269 |
| equal_tokens | 1 | und_Deva | B | 1.151721 | 1.141697 | +0.010025 |
| equal_tokens | 1 | Sanskrit | B | 0.982854 | 0.965558 | +0.017296 |
| equal_tokens | 2 | anp_Deva | B | 0.630453 | 0.629816 | +0.000636 |
| equal_tokens | 2 | awa_Deva | B | 0.761168 | 0.757006 | +0.004162 |
| equal_tokens | 2 | gom_Deva | B | 0.758213 | 0.750839 | +0.007373 |
| equal_tokens | 2 | und_Deva | B | 1.133048 | 1.123025 | +0.010023 |
| equal_tokens | 2 | Sanskrit | B | 0.976954 | 0.962009 | +0.014945 |
| equal_tokens | 3 | English | B | 1.195922 | 1.194045 | +0.001876 |
| equal_tokens | 3 | anp_Deva | B | 0.634921 | 0.631884 | +0.003038 |
| equal_tokens | 3 | awa_Deva | B | 0.761569 | 0.755652 | +0.005916 |
| equal_tokens | 3 | bho_Deva | B | 0.784404 | 0.783937 | +0.000467 |
| equal_tokens | 3 | gom_Deva | B | 0.764712 | 0.759083 | +0.005630 |
| equal_tokens | 3 | Sanskrit | B | 0.978593 | 0.961867 | +0.016726 |
| equal_bytes | 1 | English | B | 1.228872 | 1.226710 | +0.002163 |
| equal_bytes | 1 | Code | C | 1.004257 | 0.997089 | +0.007168 |
| equal_bytes | 1 | Math | C | 1.575752 | 1.565287 | +0.010465 |
| equal_bytes | 1 | anp_Deva | B | 0.652396 | 0.647812 | +0.004584 |
| equal_bytes | 1 | awa_Deva | B | 0.792192 | 0.787521 | +0.004671 |
| equal_bytes | 1 | gom_Deva | B | 0.786958 | 0.779305 | +0.007653 |
| equal_bytes | 1 | und_Deva | B | 1.175549 | 1.169270 | +0.006279 |
| equal_bytes | 1 | und_Deva | C | 1.175549 | 1.166777 | +0.008772 |
| equal_bytes | 1 | Sanskrit | B | 0.998839 | 0.980192 | +0.018647 |
| equal_bytes | 1 | Sanskrit | C | 0.998839 | 0.996603 | +0.002235 |
| equal_bytes | 2 | English | B | 1.228358 | 1.228060 | +0.000298 |
| equal_bytes | 2 | Code | C | 1.000001 | 0.997178 | +0.002823 |
| equal_bytes | 2 | Math | C | 1.574777 | 1.562299 | +0.012478 |
| equal_bytes | 2 | anp_Deva | B | 0.644762 | 0.644141 | +0.000621 |
| equal_bytes | 2 | awa_Deva | B | 0.780545 | 0.775722 | +0.004824 |
| equal_bytes | 2 | gom_Deva | B | 0.776948 | 0.771688 | +0.005259 |
| equal_bytes | 2 | und_Deva | B | 1.167616 | 1.161098 | +0.006518 |
| equal_bytes | 2 | und_Deva | C | 1.167616 | 1.167244 | +0.000372 |
| equal_bytes | 2 | Sanskrit | B | 0.997885 | 0.981547 | +0.016339 |
| equal_bytes | 2 | Sanskrit | C | 0.997885 | 0.995633 | +0.002253 |
| equal_bytes | 3 | Math | C | 1.574356 | 1.564305 | +0.010051 |
| equal_bytes | 3 | anp_Deva | B | 0.645605 | 0.643544 | +0.002061 |
| equal_bytes | 3 | gom_Deva | B | 0.785806 | 0.783952 | +0.001854 |
| equal_bytes | 3 | und_Deva | B | 1.160360 | 1.152514 | +0.007847 |
| equal_bytes | 3 | Sanskrit | B | 1.008629 | 0.996685 | +0.011943 |
| equal_bytes | 3 | Sanskrit | C | 1.008629 | 0.999456 | +0.009173 |

### FLORES+ devtest (equal tokens, seed mean)

| Lang | A | B | C | A−B | A−C |
|---|---:|---:|---:|---:|---:|
| npi | 0.658083 | 0.657272 | 0.669070 | +0.000810 | -0.010987 |
| hin | 0.682352 | 0.684022 | 0.699886 | -0.001670 | -0.017535 |
| eng | 1.376737 | 1.370203 | 1.439715 | +0.006534 | -0.062978 |
| mai | 0.876791 | 0.885206 | 0.897113 | -0.008415 | -0.020322 |
| bho | 0.956653 | 0.956213 | 0.971991 | +0.000440 | -0.015338 |
| awa | 0.978251 | 0.972001 | 0.983140 | +0.006250 | -0.004888 |
| mag | 0.986140 | 0.979679 | 0.977372 | +0.006461 | +0.008767 |
| san | 1.044593 | 1.015382 | 1.055329 | +0.029211 | -0.010736 |
| mar | 0.816842 | 0.822785 | 0.843830 | -0.005943 | -0.026989 |
| hne | 1.008591 | 1.007934 | 1.028049 | +0.000657 | -0.019458 |

#### Every A loss on FLORES

| Checkpoint | Seed | Lang | vs | A | other | diff |
|---|---:|---|---|---:|---:|---:|
| equal_tokens | 1 | npi | B | 0.657310 | 0.656344 | +0.000966 |
| equal_tokens | 1 | eng | B | 1.377919 | 1.367238 | +0.010681 |
| equal_tokens | 1 | bho | B | 0.956020 | 0.955393 | +0.000627 |
| equal_tokens | 1 | awa | B | 0.983053 | 0.971489 | +0.011564 |
| equal_tokens | 1 | awa | C | 0.983053 | 0.982335 | +0.000718 |
| equal_tokens | 1 | mag | B | 0.989194 | 0.976935 | +0.012259 |
| equal_tokens | 1 | mag | C | 0.989194 | 0.976060 | +0.013134 |
| equal_tokens | 1 | san | B | 1.048459 | 1.014307 | +0.034153 |
| equal_tokens | 1 | hne | B | 1.013767 | 1.009532 | +0.004236 |
| equal_tokens | 2 | npi | B | 0.659255 | 0.656252 | +0.003003 |
| equal_tokens | 2 | hin | B | 0.682523 | 0.681279 | +0.001244 |
| equal_tokens | 2 | eng | B | 1.375002 | 1.371022 | +0.003981 |
| equal_tokens | 2 | bho | B | 0.955671 | 0.953098 | +0.002573 |
| equal_tokens | 2 | awa | B | 0.971928 | 0.970238 | +0.001690 |
| equal_tokens | 2 | mag | B | 0.982034 | 0.980022 | +0.002011 |
| equal_tokens | 2 | mag | C | 0.982034 | 0.977432 | +0.004602 |
| equal_tokens | 2 | san | B | 1.036784 | 1.011765 | +0.025020 |
| equal_tokens | 2 | hne | B | 1.007906 | 1.004302 | +0.003604 |
| equal_tokens | 3 | eng | B | 1.377288 | 1.372347 | +0.004940 |
| equal_tokens | 3 | awa | B | 0.979773 | 0.974275 | +0.005498 |
| equal_tokens | 3 | mag | B | 0.987191 | 0.982079 | +0.005112 |
| equal_tokens | 3 | mag | C | 0.987191 | 0.978625 | +0.008567 |
| equal_tokens | 3 | san | B | 1.048536 | 1.020075 | +0.028461 |
| equal_bytes | 1 | eng | B | 1.403892 | 1.395689 | +0.008203 |
| equal_bytes | 1 | bho | B | 0.966352 | 0.964539 | +0.001813 |
| equal_bytes | 1 | awa | B | 0.992340 | 0.981773 | +0.010567 |
| equal_bytes | 1 | awa | C | 0.992340 | 0.982335 | +0.010005 |
| equal_bytes | 1 | mag | B | 0.992901 | 0.985840 | +0.007061 |
| equal_bytes | 1 | mag | C | 0.992901 | 0.976060 | +0.016841 |
| equal_bytes | 1 | san | B | 1.077006 | 1.033620 | +0.043386 |
| equal_bytes | 1 | san | C | 1.077006 | 1.058496 | +0.018510 |
| equal_bytes | 1 | hne | B | 1.028184 | 1.022548 | +0.005636 |
| equal_bytes | 2 | npi | B | 0.668130 | 0.664430 | +0.003700 |
| equal_bytes | 2 | npi | C | 0.668130 | 0.667766 | +0.000364 |
| equal_bytes | 2 | mag | C | 0.990436 | 0.977432 | +0.013004 |
| equal_bytes | 2 | san | B | 1.066051 | 1.033430 | +0.032620 |
| equal_bytes | 2 | san | C | 1.066051 | 1.053677 | +0.012373 |
| equal_bytes | 2 | hne | B | 1.014688 | 1.013674 | +0.001014 |
| equal_bytes | 3 | eng | B | 1.406100 | 1.393560 | +0.012540 |
| equal_bytes | 3 | bho | B | 0.971486 | 0.969468 | +0.002019 |
| equal_bytes | 3 | bho | C | 0.971486 | 0.969949 | +0.001537 |
| equal_bytes | 3 | awa | B | 0.986140 | 0.982111 | +0.004029 |
| equal_bytes | 3 | mag | B | 0.998444 | 0.987929 | +0.010514 |
| equal_bytes | 3 | mag | C | 0.998444 | 0.978625 | +0.019819 |
| equal_bytes | 3 | san | B | 1.065651 | 1.034644 | +0.031007 |
| equal_bytes | 3 | san | C | 1.065651 | 1.053814 | +0.011837 |

### Perturbed text (equal tokens, seed mean)

Denominator is UTF-8 bytes of the unmodified text.

| Group | A | B | C | A−B | A−C |
|---|---:|---:|---:|---:|---:|
| `flores_npi_misspell.jsonl:npi` | 0.736032 | 0.737590 | 0.748141 | -0.001558 | -0.012109 |
| `flores_npi_zwj.jsonl:npi` | 1.342372 | 1.310008 | 1.246359 | +0.032364 | +0.096013 |
| `flores_npi_zwnj.jsonl:npi` | 1.430827 | 1.325634 | 1.293672 | +0.105192 | +0.137155 |
| `sealed_npi_misspell.jsonl:Nepali` | 0.563694 | 0.568431 | 0.588597 | -0.004737 | -0.024902 |
| `sealed_npi_zwj.jsonl:Nepali` | 1.049981 | 0.949535 | 0.960006 | +0.100446 | +0.089975 |
| `sealed_npi_zwnj.jsonl:Nepali` | 1.127954 | 0.962587 | 0.985889 | +0.165367 | +0.142065 |

#### Every A loss on perturbed sets

| Checkpoint | Seed | Group | vs | A | other | diff |
|---|---:|---|---|---:|---:|---:|
| equal_tokens | 1 | `flores_npi_zwj.jsonl:npi` | B | 1.333517 | 1.296878 | +0.036639 |
| equal_tokens | 1 | `flores_npi_zwj.jsonl:npi` | C | 1.333517 | 1.249643 | +0.083875 |
| equal_tokens | 1 | `flores_npi_zwnj.jsonl:npi` | B | 1.407327 | 1.310479 | +0.096848 |
| equal_tokens | 1 | `flores_npi_zwnj.jsonl:npi` | C | 1.407327 | 1.285816 | +0.121510 |
| equal_tokens | 1 | `sealed_npi_zwj.jsonl:Nepali` | B | 1.045344 | 0.946289 | +0.099055 |
| equal_tokens | 1 | `sealed_npi_zwj.jsonl:Nepali` | C | 1.045344 | 0.960056 | +0.085288 |
| equal_tokens | 1 | `sealed_npi_zwnj.jsonl:Nepali` | B | 1.115021 | 0.961246 | +0.153775 |
| equal_tokens | 1 | `sealed_npi_zwnj.jsonl:Nepali` | C | 1.115021 | 0.984179 | +0.130842 |
| equal_tokens | 2 | `flores_npi_zwj.jsonl:npi` | B | 1.350118 | 1.309134 | +0.040984 |
| equal_tokens | 2 | `flores_npi_zwj.jsonl:npi` | C | 1.350118 | 1.236016 | +0.114103 |
| equal_tokens | 2 | `flores_npi_zwnj.jsonl:npi` | B | 1.448962 | 1.322714 | +0.126247 |
| equal_tokens | 2 | `flores_npi_zwnj.jsonl:npi` | C | 1.448962 | 1.297068 | +0.151893 |
| equal_tokens | 2 | `sealed_npi_zwj.jsonl:Nepali` | B | 1.045845 | 0.948604 | +0.097241 |
| equal_tokens | 2 | `sealed_npi_zwj.jsonl:Nepali` | C | 1.045845 | 0.963192 | +0.082653 |
| equal_tokens | 2 | `sealed_npi_zwnj.jsonl:Nepali` | B | 1.128041 | 0.960819 | +0.167222 |
| equal_tokens | 2 | `sealed_npi_zwnj.jsonl:Nepali` | C | 1.128041 | 0.989987 | +0.138054 |
| equal_tokens | 3 | `flores_npi_zwj.jsonl:npi` | B | 1.343480 | 1.324011 | +0.019469 |
| equal_tokens | 3 | `flores_npi_zwj.jsonl:npi` | C | 1.343480 | 1.253418 | +0.090062 |
| equal_tokens | 3 | `flores_npi_zwnj.jsonl:npi` | B | 1.436192 | 1.343710 | +0.092482 |
| equal_tokens | 3 | `flores_npi_zwnj.jsonl:npi` | C | 1.436192 | 1.298131 | +0.138060 |
| equal_tokens | 3 | `sealed_npi_zwj.jsonl:Nepali` | B | 1.058755 | 0.953713 | +0.105042 |
| equal_tokens | 3 | `sealed_npi_zwj.jsonl:Nepali` | C | 1.058755 | 0.956772 | +0.101984 |
| equal_tokens | 3 | `sealed_npi_zwnj.jsonl:Nepali` | B | 1.140800 | 0.965695 | +0.175105 |
| equal_tokens | 3 | `sealed_npi_zwnj.jsonl:Nepali` | C | 1.140800 | 0.983502 | +0.157298 |
| equal_bytes | 1 | `flores_npi_zwj.jsonl:npi` | B | 1.358463 | 1.313729 | +0.044734 |
| equal_bytes | 1 | `flores_npi_zwj.jsonl:npi` | C | 1.358463 | 1.249643 | +0.108820 |
| equal_bytes | 1 | `flores_npi_zwnj.jsonl:npi` | B | 1.434884 | 1.329406 | +0.105478 |
| equal_bytes | 1 | `flores_npi_zwnj.jsonl:npi` | C | 1.434884 | 1.285816 | +0.149068 |
| equal_bytes | 1 | `sealed_npi_zwj.jsonl:Nepali` | B | 1.064013 | 0.962785 | +0.101227 |
| equal_bytes | 1 | `sealed_npi_zwj.jsonl:Nepali` | C | 1.064013 | 0.960056 | +0.103957 |
| equal_bytes | 1 | `sealed_npi_zwnj.jsonl:Nepali` | B | 1.134270 | 0.983749 | +0.150521 |
| equal_bytes | 1 | `sealed_npi_zwnj.jsonl:Nepali` | C | 1.134270 | 0.984179 | +0.150091 |
| equal_bytes | 2 | `flores_npi_misspell.jsonl:npi` | B | 0.745799 | 0.745758 | +0.000041 |
| equal_bytes | 2 | `flores_npi_zwj.jsonl:npi` | B | 1.351702 | 1.333652 | +0.018050 |
| equal_bytes | 2 | `flores_npi_zwj.jsonl:npi` | C | 1.351702 | 1.236016 | +0.115687 |
| equal_bytes | 2 | `flores_npi_zwnj.jsonl:npi` | B | 1.440102 | 1.333715 | +0.106387 |
| equal_bytes | 2 | `flores_npi_zwnj.jsonl:npi` | C | 1.440102 | 1.297068 | +0.143034 |
| equal_bytes | 2 | `sealed_npi_zwj.jsonl:Nepali` | B | 1.062990 | 0.975938 | +0.087052 |
| equal_bytes | 2 | `sealed_npi_zwj.jsonl:Nepali` | C | 1.062990 | 0.963192 | +0.099798 |
| equal_bytes | 2 | `sealed_npi_zwnj.jsonl:Nepali` | B | 1.136925 | 0.979755 | +0.157170 |
| equal_bytes | 2 | `sealed_npi_zwnj.jsonl:Nepali` | C | 1.136925 | 0.989987 | +0.146939 |
| equal_bytes | 3 | `flores_npi_zwj.jsonl:npi` | B | 1.356902 | 1.353120 | +0.003781 |
| equal_bytes | 3 | `flores_npi_zwj.jsonl:npi` | C | 1.356902 | 1.253418 | +0.103484 |
| equal_bytes | 3 | `flores_npi_zwnj.jsonl:npi` | B | 1.453817 | 1.356952 | +0.096864 |
| equal_bytes | 3 | `flores_npi_zwnj.jsonl:npi` | C | 1.453817 | 1.298131 | +0.155685 |
| equal_bytes | 3 | `sealed_npi_zwj.jsonl:Nepali` | B | 1.077135 | 0.987277 | +0.089858 |
| equal_bytes | 3 | `sealed_npi_zwj.jsonl:Nepali` | C | 1.077135 | 0.956772 | +0.120363 |
| equal_bytes | 3 | `sealed_npi_zwnj.jsonl:Nepali` | B | 1.165260 | 0.991112 | +0.174148 |
| equal_bytes | 3 | `sealed_npi_zwnj.jsonl:Nepali` | C | 1.165260 | 0.983502 | +0.181758 |

### Arm E (yak-64k-v2.0) — single seed, no verdict

| Checkpoint | Metric | A | B | E | A−E |
|---|---|---:|---:|---:|---:|
| equal tokens | Nepali | 0.466263 | 0.468417 | 0.467183 | -0.000920 |
| equal tokens | Dev11 | 0.748445 | 0.750504 | 0.750021 | -0.001576 |
| equal bytes | Nepali | 0.476757 | 0.479207 | 0.477234 | -0.000477 |
| equal bytes | Dev11 | 0.767661 | 0.770013 | 0.769650 | -0.001989 |

## Plots

- `artifacts/m9/plots/paired_diffs_equal_tokens.png`
- `artifacts/m9/plots/paired_diffs_equal_bytes.png`
- `artifacts/m9/plots/bpb_vs_bytes_seed1.png`
- `artifacts/m9/plots/train_loss_vs_bytes.png`

## Limitations (from the pre-registration)

(a) At equal-bytes, a more efficient arm has seen fewer tokens and is earlier in the schedule. (b) sidskarki is not lossless on whitespace; round-trip rates are above. (c) Vocabulary sizes differ; sidskarki embedding is extended by one.


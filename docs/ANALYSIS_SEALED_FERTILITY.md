# Sealed sample, fertility spread, joiner units

Analysis only. No training run, and the server queue was not touched. Counts are from the local sealed eval files, the n60j5 snapshot, and the model-eval text stream. Reproduction: `scripts/analysis_sealed_fertility.py`.

Percentiles are Hyndman–Fan type 7 (position `q * (n − 1)`, linear interpolation).

## 1. How `sealed_npi_2000` was sampled

The 2,000-document file is a prefix of the no-joiner sealed list. Recovered rows are sorted by `doc_id`, Nepali rows are kept, and the first 2,000 are written:

```99:99:python/ancoda_tok/model_eval/evalprep.py
    rows.sort(key=lambda row: row["doc_id"])
```

```126:130:python/ancoda_tok/model_eval/evalprep.py
    nepali = [row for row in sealed if row["lang"] == "Nepali"]
    if len(nepali) < 2000:
        raise SystemExit(f"STOP: sealed Nepali has {len(nepali)} docs, need 2000")
    sample = nepali[:2000]
    _write_jsonl(EVAL / "sealed_npi_2000.jsonl.zst", sample)
```

The same rule is in `docs/MODEL_EVAL_PREREG.md` (Nepali rows of the no-joiner sealed list, `doc_id` ascending, first 2,000). The file on disk matches that prefix: 2,000 ids, all Nepali, equal to the first 2,000 Nepali `doc_id`s in `sealed.jsonl.zst`.

That cut does not test sidskarki round-trip and does not apply a length minimum or maximum. The only check on length is that the recovered UTF-8 size equals the index `text_bytes` (`evalprep.py` lines 87–89); a mismatch stops the build.

The list those rows are drawn from was built earlier, still without a round-trip test or a per-document length cutoff:

- Drop cleaned documents whose `doc_id` is already in any snapshot (`diag_prompt2_gaps.py` lines 169–170).
- Within each mix bucket, keep the lowest `sha256(f"{seed}:{doc_id}")` values and take them in that order until the bucket's byte budget is met (`diag_prompt2_gaps.py` lines 155–158 and 199–207). The budget is 100 MB with Nepali scaled to 60%. The `2_000` in `keep_n` is an assumed average document size in bytes, used only to size the candidate heap.
- Drop MinHash near-duplicates of snapshot text (`diag_prompt2_gaps.py` line 256).
- Drop 65 documents whose `doc_id` was a synthetic joiner source (`final_build_sid_sealed_fix.py` lines 374–378).

### Length in bytes

| Set | n | median | p90 |
|---|---:|---:|---:|
| `sealed_npi_2000` | 2,000 | 3,498 | 10,275.7 |
| All sealed Nepali | 28,094 | 741 | 5,321 |

The sample is longer because `doc_id` order groups documents by source, and `fw2_npi` sorts first. All 2,000 sample documents are `fw2_npi` (2,510 sealed Nepali documents from that source; median 3,459 bytes, p90 9,975.3). Most sealed Nepali is `indiccorp_npi` (18,251 documents; median 458 bytes, p90 1,105). The other sealed Nepali sources are `sangraha_nep` 3,749 (median 3,233), `iriis` 2,808 (median 2,807), and `hplt_npi` 776 (median 4,599).

## 2. Fertility spread

A word is a whitespace-separated span (`str.split`). Each FLORES+ devtest line and each sealed document is encoded on its own, with no BOS/EOS. Every token is counted on exactly one word: the word with the greatest character overlap, ties to the earlier word, and a token that overlaps no word counted on the following word. No token overlapped two words, so the mean equals total tokens divided by total words.

FLORES+ Nepali is `npi_Deva` devtest, 1,012 lines, 18,701 words. The means match the per-sentence checks at 3 decimal places (Ancoda-64k 1.451, Yak v1 1.457, Yak v2 1.470, sidskarki 1.370). Sealed Nepali is all 28,094 Nepali documents in the no-joiner sealed set, 3,475,674 words. Arm D is `plain-64k`.

| Tokenizer | Corpus | mean | median | p90 | p95 | % of words with ≥4 tokens |
|---|---|---:|---:|---:|---:|---:|
| Ancoda-64k | FLORES+ Nepali | 1.4508 | 1 | 2 | 3 | 3.21 |
| Yak v1 64k | FLORES+ Nepali | 1.4566 | 1 | 2 | 3 | 3.56 |
| Yak v2 64k | FLORES+ Nepali | 1.4700 | 1 | 2 | 3 | 3.54 |
| sidskarki-64k | FLORES+ Nepali | 1.3699 | 1 | 2 | 3 | 2.45 |
| Arm D | FLORES+ Nepali | 1.4902 | 1 | 2 | 3 | 3.80 |
| Ancoda-64k | sealed Nepali | 1.3077 | 1 | 2 | 3 | 1.91 |
| Yak v1 64k | sealed Nepali | 1.3162 | 1 | 2 | 3 | 2.19 |
| Yak v2 64k | sealed Nepali | 1.3285 | 1 | 2 | 3 | 2.17 |
| sidskarki-64k | sealed Nepali | 1.2323 | 1 | 2 | 2 | 1.39 |
| Arm D | sealed Nepali | 1.3376 | 1 | 2 | 3 | 2.25 |

## 3. Joiner units in Ancoda-64k

A vocabulary entry is a joiner unit when the UTF-8 decoding of its piece contains U+200D (ZWJ) or U+200C (ZWNJ). The rate is how many emitted tokens have such an id, per million content tokens. `add_special_tokens=False`. End-of-text is not in the denominator. A token is counted once even if its piece contains more than one joiner.

Of 65,536 entries, 644 contain ZWJ or ZWNJ: 325 contain ZWJ and 319 contain ZWNJ. None contain both. One entry is the single character ZWJ and one is the single character ZWNJ.

| Corpus | Content tokens | Joiner-unit tokens | Per million | ZWJ units / million | ZWNJ units / million |
|---|---:|---:|---:|---:|---:|
| Tokenizer training text (n60j5) | 614,195,559 | 1,059,571 | 1,725.14 | 885.91 | 839.22 |
| n60j5, natural documents only | 609,533,026 | 53,716 | 88.13 |  |  |
| n60j5, synthetic joiner copy | 4,662,533 | 1,005,855 | 215,731.45 |  |  |
| Model training stream | 1,085,350,393 | 52,771 | 48.62 | 41.97 | 6.65 |
| FLORES+ Nepali, ZWJ rewrite | 36,556 | 8,506 | 232,684.10 | 232,137.00 | 547.11 |
| FLORES+ Nepali, ZWNJ rewrite | 36,500 | 8,509 | 233,123.29 | 931.51 | 232,191.78 |
| Sealed Nepali 2,000, ZWJ rewrite | 1,053,750 | 229,382 | 217,681.61 | 217,681.61 | 0 |
| Sealed Nepali 2,000, ZWNJ rewrite | 1,052,622 | 229,108 | 217,654.58 | 0 | 217,654.58 |

The tokenizer training text is the n60j5 snapshot (2,314,323 documents, 5,000,811,952 bytes), including its synthetic joiner copy (22,102 documents). The model stream is the model-eval text stream (2,477,651 documents, 7,189,336,358 bytes, 1,085,350,393 Ancoda-64k content tokens). It contains no `synthetic\x1f` documents. The small ZWNJ count on the FLORES ZWJ rewrite, and the small ZWJ count on the FLORES ZWNJ rewrite, are joiners already present in the original lines.

## 4. Anonymous release

The folder is `artifacts/anonymous_release/`. It is not committed (`/artifacts/` is gitignored). `CONTENTS.txt` in that folder lists every file with its size and sha256.

The sealed list is the no-joiner list: 35,623 lines of `sha256(doc_id)`, no trailing newline. The sha256 of that byte string is `820f385638336be6b904c1f3b10c56f380ef73515f76375eebd1f1e258156336`, the same value as `list_sha256` in `docs/sealed_test_set_no_joiners.json`.

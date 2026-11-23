# Matched no-joiner Ancoda-64k (arm F prep)

Preparation only. Tokenizer trained; **not** queued for model-eval. Marked in `docs/MODEL_EVAL_PREREG.md` as not yet registered.

## Config diff vs final_build/ancoda-64k

```json
{
  "changed": {
    "deva_counts": "artifacts/eval/final_build/n60j5_nojoiner_counts/deva_pretoken_counts.bin",
    "latin_counts": "artifacts/eval/final_build/n60j5_nojoiner_counts/latin_cm100_pretoken_counts.bin",
    "snapshot": {
      "arm_A": "data/snapshots/n60j5_20af1fffbec0fe86",
      "arm_F": "data/snapshots/n60j5_nojoiner_745e14157088dc30",
      "joiner_share": {
        "arm_A": 0.01,
        "arm_F": 0.0
      },
      "note": "1% joiner copy replaced by natural Nepali (DECISIONS 3.8)"
    }
  },
  "ladder_final_nojoiner_not_matched": {
    "path": "artifacts/tokenizers/ladder-final_nojoiner",
    "reason": "n50_nojoiner snapshot; greets tokenize differently from final_build/ancoda-64k"
  },
  "not_used": "configs/build_64k.yaml (points at n50; stale for this arm)",
  "unchanged": {
    "B_d": 48000,
    "bans": true,
    "block0": true,
    "block1": true,
    "code_math_scale": 1.0,
    "covered": "artifacts/inventory/covered_64k_3pct.json",
    "gb9c": false,
    "plain": false,
    "seed": 20261104,
    "slack": 4096,
    "vocab_size": 65536
  }
}
```

Joiner-containing vocab entries: **95** (zwj=56, zwnj=39).

## FLORES+ per-sentence fertility

| Lang | final_build/ancoda-64k | nojoiner | Δ (nojoiner−final-build) | Verdict |
|---|---:|---:|---:|---|
| npi | 1.450778 | 1.449227 | -0.001551 | WIN |
| hin | 1.339898 | 1.339313 | -0.000585 | TIE |
| eng | 1.414136 | 1.414502 | +0.000365 | TIE |

## ZWJ / ZWNJ cost (FLORES npi rewrite ÷ original)

- ZWJ final-build=1.347350, nojoiner=1.634791
- ZWNJ final-build=1.345286, nojoiner=1.658845

Round-trip 1,000 stream docs: 1000/1000 (rate 1.000000).

Tokens/byte on the same first 10,000 model-eval stream documents (shard order): final_build/ancoda-64k=0.077341, nojoiner=0.077247. The published full-stream rate for final-build is 0.150967 (different document mix over 2.48M docs); the 10k prefix is not that rate.

Tokenizer: `artifacts/tokenizers/final_build/ancoda-64k-nojoiner` sha256=`7c9676fd60a829a507d7d23f647765f5ec26c11d1b61612630f75ad7b8605221`.

## Snapshot match check (n60 vs n60j5)

Existing `data/snapshots/n60_cf8066dbbdd0da93` is **not** matched to n60j5: only ~685k of n60j5's natural docs appear in n60; ~1.6M docs differ each way. Built `data/snapshots/n60j5_nojoiner_745e14157088dc30` with the same selection seeds as `build_5gb_n60_joiner` and Nepali target = full 60% natural (`joiner_share=0`).

## Joiner vocab note

Expecting ~0 multi-character joiner merges is wrong for this recipe: Block 0 always keeps single-character U+200C and U+200D, and natural joiners in Devanagari text still yield some multi-character joiner pieces (95 total here vs 644 on final_build/ancoda-64k).


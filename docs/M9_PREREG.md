# M9 pre-registration

Locked on 10 Nov 2026, before any training run. Setup and a pilot only. Full runs do not start until the pilot report is reviewed. Tokenizers are frozen. M9 weights and third-party tokenizer files are not released.

## Licence

The human confirmed on 10 Nov 2026 that Ancoda Labs' total annual revenue is at most USD $100,000.

`docs/LICENSES.md` records both files as usable for this comparison:

- Yak (`ampixa/yak-tokenizer`, CC BY-NC 4.0) is evaluation and proxy-model comparison only. Copying the tokenizer file to a private server for that comparison is non-commercial reproduction. The trained weights are not released.
- sidskarki (`nepali_bpe_64k.model`, community-use-1.0) permits use and copy, including M9 proxy-model comparison, when annual revenue is at most USD $100,000 and attribution is given. Attribute: sidskarki, `sidskarki/nepali-bpe-tokenizer`, revision `fc5773dcf7b9dc2272e8f595f99828969621a855`. The trained weights and the `.model` file are not released.

## Arms

| Letter | Name | File | Seeds |
|---|---|---|---|
| A | ancoda-64k | `artifacts/tokenizers/m8/ancoda-64k/tokenizer.json` | 1, 2, 3 |
| B | yak-64k-v1 | `ampixa/yak-tokenizer` `v1.0/64k` at revision `12b6d3010ec542b484ec3b673354852fe8bb85fb` | 1, 2, 3 |
| C | sidskarki-64k | `nepali_bpe_64k.model` at revision `fc5773dcf7b9dc2272e8f595f99828969621a855` | 1, 2, 3 |
| E | yak-64k-v2.0 | `ampixa/yak-tokenizer` `v2.0/64k`, same revision | 1 only, after the core pilot, if time allows |

Core comparisons are A, B, and C. E is secondary.

Arm D (plain BPE 64k, same data, no Block 0, no Block 1, no bans) is skipped. No 64k plain build was trained on n60j5. `configs/ladder_plain.yaml` points at `artifacts/counts/n50/pretoken_counts.bin`. `configs/plain_1gb.yaml` is the 1 GB plain build. Neither is n60j5.

## Primary metric

BPB = (sum of token NLL in nats) / (ln 2 × UTF-8 bytes of the original, unmodified text).

Reported per language. Never pooled into one number by concatenating languages or weighting them by bytes.

NLL is the sum of natural-log next-token losses. PyTorch cross-entropy is that quantity. The separator token is context and is not scored. Every text token is scored once.

## Primary comparisons

A versus B, and A versus C, each reported separately:

- sealed Nepali
- the unweighted mean of the 11 sealed Devanagari groups: Hindi, Marathi, Nepali, Sanskrit, anp_Deva, awa_Deva, bho_Deva, gom_Deva, hne_Deva, mai_Deva, new_Deva

Each comparison is made at two checkpoints:

- equal tokens: the step-7630 checkpoint (1,000,079,360 stream tokens)
- equal bytes: the B* checkpoint defined below

The sealed list is `docs/sealed_test_set_no_joiners.json`, grouped with the corrected language labels from the M8 follow-up (bucket display names, with `npi_Deva`→Nepali, `hin_Deva`→Hindi, `mar_Deva`→Marathi, `san_Deva`→Sanskrit, `eng_Latn`→English; smaller pools keep their GlotLID label).

## Decision rule

For one comparison, one language aggregate, and one checkpoint type, the paired difference of a seed is BPB(A) − BPB(other). Lower BPB is better.

Report all three per-seed differences. The label is **A better** only when every difference is strictly negative and the absolute value of the mean exceeds the sample standard deviation across the three seeds (divisor n−1). Otherwise the label is **no measurable difference**.

A consistent loss for A is visible in the three differences. It is not labelled "A better". Every result is reported, including losses. Seeds are not dropped.

## Secondary

Reported, not used for the primary label:

- sealed English, Hindi, Marathi, code, math, and the smaller sealed groups
- perturbed text (below)
- FLORES devtest: npi, hin, eng, mai, bho, awa, mag, san, mar, hne

Perturbed sets, each scored with the unmodified text's UTF-8 length in the denominator:

- FLORES npi devtest, in file order
- a fixed 2,000-document sealed Nepali sample: Nepali rows of the no-joiner sealed list, ordered by `doc_id`, first 2,000 after sorting `doc_id` ascending

Perturbations, applied independently:

- 10% misspelling, `random.Random(20261104)` walked in that file order, same character rules as M8 (`python/ancoda_tok/m8_final.py` `misspell`): swap ि↔ी and ु↔ू, and cycle श→स→ष→श, each eligible character with probability 0.1
- ZWJ after every virama that precedes a consonant
- ZWNJ in the same sites

The joiner rule is the Yak probe: virama U+094D before a character in `[क-हक़-य़ॸ-ॿ]`.

## Stated limitations

(a) At the equal-bytes checkpoint, a more efficient arm has seen fewer tokens and is earlier in the schedule, so its learning rate is higher. That is conservative against the more efficient arm.

(b) sidskarki is not lossless on whitespace, so it predicts slightly less content than the other arms. Its exact round-trip rate on the 1,000-document check is reported with the results.

(c) Vocabulary sizes differ slightly. sidskarki has 64,000 pieces. The others have 65,536 tokenizer ids. sidskarki has no end-of-text id, so the model embedding is extended by one.

## Locked procedure

These choices are part of the pre-registration. They are not tuned after the pilot losses are seen.

### Data

One document stream for every arm. Mix by UTF-8 bytes of the original text: Nepali 40%, Hindi 15%, other Devanagari 5%, English 35%, code 5%. Source: `data/clean` pools, via `data/clean/_snapshot_index.parquet`.

- Nepali, Hindi, English, and Code are those clean buckets.
- Other Devanagari is one pool: buckets Marathi, Sanskrit, Smaller Devanagari, and Languages of Nepal. No further reweighting inside the 5%.
- Math is not in the mix. Documents whose `doc_id` starts with `synthetic\x1f` are not in the mix.

Excluded before selection: `sha256(doc_id)` in `docs/sealed_test_set.json` (the full original list), and `sha256(doc_id)` of every document in raw eval or decontamination sources (`fw2_npi_test`, `fw2_npi_latn_test`, `fw2_hin_test`, `c4_val`, `aya_npi`, `belebele_npi`). Exact text matches to FLORES devtest lines and to those eval documents are dropped at write time. The final stream's overlap with that exclusion set is reported and must be 0.

Selection rank is the big-endian integer of the first 8 bytes of `sha256(b"m9-v1\n" + doc_id.encode("utf-8"))`. Within a group, documents are ordered by `(rank, doc_id)`. The prefix is the shortest prefix whose index byte counts reach the group target. The document that crosses the target is included.

Group targets are exact fifths of the total (the total is a multiple of 20). The total is the smallest such multiple that is at least `ceil(1.05e9 / min_sample_tokens_per_byte)`, increased by 3% when every group still has the bytes. Otherwise it is the largest multiple of 20 that preserves the shares. Stop if that total cannot put the lowest-rate arm at 1.05e9 content tokens. The 50 MB sample uses the same shares and the same rank order. Tokens per byte are content tokens only (the added end-of-text id is not in the rate). The built stream is checked again after tokenization; stop if the fewest arm is under 1.05e9 content tokens.

Document order in the shards is that global `(rank, doc_id)` order. The document-list sha256 is SHA-256 over `doc_id.encode("utf-8") + b"\n"` in that order. Stored text is copied from the clean shards with no Unicode normalization.

### Tokens

No special tokens inside a document (`add_special_tokens=False`; SentencePiece without BOS or EOS).

One end-of-text id after each document. It is the tokenizer's own special token if the tokenizer file defines `<|eos|>`, `<|endoftext|>`, or `</s>` as a special token. Otherwise the id is `vocab_size` and the embedding has size `vocab_size + 1`.

Resolved from the files before this run: ancoda-64k uses `<|eos|>` id 64513 inside vocab 65536. Both Yak 64k files use `<|eos|>` id 2 inside vocab 65536. sidskarki has no end-of-text piece (`eos_id == -1`), so the id is 64000 and the embedding size is 64001.

Ids are stored as little-endian uint16 when every id including end-of-text is ≤ 65535, otherwise uint32. The stream above fits in uint16.

Training reads a permutation of documents, then the end-of-text-joined token stream, in contiguous non-overlapping windows of 1,024 tokens. Seed `s` is both the model-init seed and the permutation seed. The permutation is Fisher–Yates with the LCG `state = state * 6364136223846793005 + 1` (mod 2^64), starting from `state = seed`, and `j = state % (i+1)`. The same permutation is used for every arm with that seed. The document set does not depend on the seed.

### Model

Llama-style decoder, identical except vocabulary size: hidden 512, 8 layers, 8 heads, head dim 64, SwiGLU intermediate 1408, RoPE with theta 10,000 and Llama rotate-half, RMSNorm epsilon 1e-5, no bias, no dropout, tied input and output embeddings, context 1,024. Matrix and embedding weights are drawn `Normal(0, 0.02)` under the seed. RMSNorm weights start at 1.

AdamW betas 0.9 and 0.95, weight decay 0.1 on matrices and 0 on norms and on the tied embedding, peak learning rate 1e-3, linear warmup over `round(0.02 * 7630) = 153` steps (learning rate at 0-indexed step `i < 153` is `1e-3 * (i+1) / 153`), then cosine decay to 1e-4. Gradient clip 1.0. bf16 autocast on XPU. `scaled_dot_product_attention` with a causal mask.

1,000,079,360 stream tokens per full run: 128 sequences × 1,024 tokens = 131,072 tokens per step, for 7,630 steps. The loss at a step is the mean NLL of the 128 × 1,023 predicted positions (the first token of each window is context). Tokens seen counts the 131,072 stream tokens. Gradient accumulation is used if a microbatch is required to fit memory; the effective batch stays 131,072.

`torch.compile` is tried on a short run. If it errors or its median step time is not faster than eager, the pilot and the training code path use eager.

### Bytes and checkpoints

Cumulative text bytes charge a document's UTF-8 length across its content tokens. The end-of-text id carries no bytes. A partial document contributes `(bytes * content_tokens_consumed) // content_token_count`. A finished document contributes its full byte count. Empty documents contribute their bytes when the end-of-text id is consumed.

For each seed, B* is the minimum, over arms A, B, and C, of the text bytes that arm has covered at 1,000,079,360 stream tokens. Each arm saves a checkpoint at the first step whose cumulative bytes reach that seed's B*.

Checkpoints go to `~/m9/runs/<arm>-s<seed>/` every 500 steps on a full run. A run keeps the latest periodic checkpoint, the B* checkpoint, and the final checkpoint. Resume restores the model, the optimizer, the completed step, and therefore the data position `step * 131072`. The pilot uses a checkpoint every 100 steps so a resume can be shown before step 500. That interval is only for the pilot.

### Evaluation windows

Each document or FLORES sentence is scored on its own. Text tokens are split into non-overlapping spans of at most 1,023 tokens. A forward pass is `[EOS] + span`, which fits in context 1,024. The next span starts again at EOS and does not see the previous span. The end-of-text id is not scored.

### Pilot, then stop

Arm A, seed 1, 300 steps. Arm C, seed 1, 50 steps. One kill and resume of arm A. BPB on arm A's 300-step checkpoint, to prove the eval path. No full run.

## Amendment — 10 Nov 2026 (after tokenize, before full runs)

This section amends the pre-registration. The original text above is unchanged.

- **sidskarki round-trip is larger than limitation (b) said.** On the shared training stream, sidskarki round-trips exactly on 640 of 1,000 documents (rate 0.64). Limitation (b) called the whitespace loss "slight". That understates the measured rate. The primary A vs C comparisons still use every eval document, and the exact round-trip rate is reported with every result.
- **Secondary A vs C on lossless docs.** In addition to the primary A vs C comparisons, report BPB(A) − BPB(C) restricted to eval documents where `sidskarki.decode(sidskarki.encode(x)) == x` exactly. Report the sidskarki round-trip rate per eval set and per language. This secondary comparison does not replace the primary decision rule; it is reported beside it.
- **Arm D** remains skipped: no 64k plain tokenizer trained on n60j5.
- **`torch.compile`** is unavailable on the compute server (no C compiler). Training is eager for the pilot and the full runs.

## Amendment — 15 Nov 2026 (M9 extension: Arms D and E)

This section amends the pre-registration. The original text and the 10 Nov 2026 amendment above are unchanged. Full extension runs do not start until this amendment is committed.

### Arm D (new)

| Letter | Name | File | Seeds |
|---|---|---|---|
| D | plain-64k | `artifacts/tokenizers/m9/plain-64k/tokenizer.json` | 1, 2, 3 |

Plain byte-level BPE, vocab 65,536, trained on the same n60j5 snapshot and pre-tokenizer as Ancoda-64k, with no protected units, no bans, no script budget, and no joiner copy. Purpose: separate design from data.

Plain mode reserves no special-token ids. End-of-text follows the locked rule for a tokenizer without `<|eos|>` / `<|endoftext|>` / `</s>`: id `vocab_size` (65536) and model embedding size `vocab_size + 1` (65537). Token ids including end-of-text exceed 65535, so the arm-D stream uses uint32 storage.

### Arm E (extended)

| Letter | Name | File | Seeds |
|---|---|---|---|
| E | yak-64k-v2.0 | `ampixa/yak-tokenizer` `v2.0/64k`, revision `12b6d3010ec542b484ec3b673354852fe8bb85fb` | 1, 2, 3 |

Seeds 2 and 3 are added. Seed 1 remains as already run under the core queue.

### Shared procedure

Same model, shared document stream, schedule, evaluation sets, and decision rule as the core A/B/C runs. Equal-tokens and equal-bytes checkpoints use the locked step-7630 and B* definitions from the core procedure (B* already fixed per seed from arms A, B, and C).

### Primary new comparisons

- **A versus D**, and **A versus E**, each on sealed Nepali and on the unweighted Dev11 mean, at equal tokens and at equal bytes.
- Report every result, including losses. Seeds are not dropped. The decision rule label is applied as locked.

### Note — 18 Nov 2026

This note is added under the 15 Nov 2026 amendment. The amendment text above is unchanged.

Arm D's tokenizer was trained on the n60j5 counts (`artifacts/counts/n60j5`), which include the 1% joiner copy. The builder adds no copy of its own. D has 405 joiner vocabulary entries (206 ZWJ, 199 ZWNJ).

## Amendment — 16 Nov 2026 (Arm F: matched no-joiner Ancoda-64k)

Registered **2026-11-16T03:40:28Z**. This amendment was written **after** the core M9 results, the joiner-set result (A loses to B/C on ZWJ/ZWNJ), and the token-level joiner-loss analysis in `docs/ANALYSIS_JOINER_LOSS.md`. It tests the hypothesis those suggested: that arm A's joiner-set loss comes from the synthetic joiner-copy recipe, not from some other Ancoda-64k design choice.

The original text and earlier amendments above are unchanged.

### Arm F (new)

| Letter | Name | File | Seeds |
|---|---|---|---|
| F | ancoda-64k-nojoiner | `artifacts/tokenizers/m8/ancoda-64k-nojoiner/tokenizer.json` (sha256 `7c9676fd60a829a507d7d23f647765f5ec26c11d1b61612630f75ad7b8605221`) | 1, 2, 3 |

Same M8 64k budget build as arm A (`B_d` 48,000, `covered_64k_3pct.json`, `code_math_scale` 1.0, seed 20261104), trained on the matched snapshot `data/snapshots/n60j5_nojoiner_745e14157088dc30`: identical mix and selection seeds as `n60j5`, with `joiner_share=0` so the 1% synthetic ZWJ/ZWNJ joiner copy is replaced by natural Nepali (DECISIONS 3.8). Prep: `docs/ANALYSIS_NOJOINER_ARM.md`.

**What F changes.** The joiner copy is replaced by natural Nepali, and the **whole merge list is retrained** on that snapshot. F therefore tests the joiner-copy recipe, not only the 644 joiner vocabulary entries of arm A. F still has **95** joiner-containing vocabulary entries, learned from natural text (56 with ZWJ, 39 with ZWNJ).

### Shared procedure

Same model, shared document stream, steps (7,630), optimizer, seeds 1–3, and evaluation sets as arm A. Equal-tokens checkpoint is step 7,630. Equal-bytes checkpoint uses **each seed's existing B\* value from the core A/B/C runs**; do not recompute B\*.

### Main comparison (prediction written now)

**F minus A** on `sealed_npi_zwj` and `sealed_npi_zwnj`, at both checkpoints (equal tokens and equal bytes).

For one set and one checkpoint, the paired difference of a seed is BPB(F) − BPB(A). Lower BPB is better. Report all three per-seed differences, the mean, and the sample SD (divisor n−1). The label is **F better** only when every difference is strictly negative and |mean| > sample SD. The label is **F worse** only when every difference is strictly positive and |mean| > sample SD. Otherwise **no measurable difference**.

**Prediction (written at registration):** F is better than A on both joiner sets. If it is not, the joiner-copy explanation is weakened, and we report that.

### Secondary (no prediction)

**F minus A** on sealed Nepali and on the unweighted Dev11 mean, at both checkpoints, reported with the same decision rule. No prediction is attached.

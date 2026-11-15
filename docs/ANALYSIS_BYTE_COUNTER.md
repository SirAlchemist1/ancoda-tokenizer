# Byte counter, seed 1

The training byte counter is the bytes of the documents whose tokens were fed. It does not drop, skip, or truncate tokens of a document and then still charge that document's full length. Recomputed coverage for arms A and C, seed 1, matches the logged counter exactly. The 4.81 GB versus ~4.6 GB gap is the document order: this permutation reads the byte-rich head of the stream earlier than a uniform order would. No recorded result was changed.

## Where the counter is computed

`TokenStream.bytes_covered` in `python/ancoda_tok/m9/reader.py` (lines 51–52) calls `bytes_at` in `python/ancoda_tok/m9/perm.py` (lines 38–67). Training logs and checkpoints store that value at `completed_steps * 131072` (`python/ancoda_tok/m9/train.py` lines 110 and 195).

`bytes_at` charges a document's original UTF-8 length across its content tokens only:

- A finished document, every content token inside the prefix, contributes its full byte length. The end-of-text id adds nothing. An empty document contributes its bytes when that id is consumed.
- A document only partly inside the prefix contributes `(byte_length * content_tokens_consumed) // content_token_count`.

`TokenStream.read` (`reader.py` lines 54–77) emits every content token of the current document, then one end-of-text id. The same document index, the same token count, and the same byte length feed both the reader and the counter. A partial document is not charged its full length.

The integer division on the single open document is less than one byte below the exact proportion. At a document boundary the sum is exact.

## Seed 1, arms A and C

"Step" here is completed steps. Tokens fed are `completed_steps * 131072` stream tokens (content tokens plus one end-of-text id per finished document). Bytes covered are the original UTF-8 lengths under the rule above. The counter column is `bytes_at` on the same arrays. The two byte columns are equal at every row. Completed step 5401 is the log row whose `"step"` field is 5400; its counter matches `bytes_seen` in the seed-1 logs, as does step 7630.

| Arm | Completed steps | Tokens fed | Content tokens | End-of-text ids | Bytes covered | Counter |
|---|---:|---:|---:|---:|---:|---:|
| A | 5,400 | 707,788,800 | 706,106,378 | 1,682,422 | 4,816,358,983 | 4,816,358,983 |
| A | 7,630 | 1,000,079,360 | 997,791,754 | 2,287,606 | 6,628,017,935 | 6,628,017,935 |
| C | 5,400 | 707,788,800 | 706,537,119 | 1,251,681 | 3,531,851,242 | 3,531,851,242 |
| C | 7,630 | 1,000,079,360 | 998,397,997 | 1,681,363 | 4,813,230,411 | 4,813,230,411 |

Logged checks: A at completed step 5401 is 4,817,238,944 and at 7,630 is 6,628,017,935. C at 5401 is 3,532,414,714 and at 7,630 is 4,813,230,411.

Corpus-rate estimates, `stream_tokens * 7,189,336,358 / stream_length`, are lower because they assume the training prefix has the corpus bytes-per-token. Stream lengths are 1,087,828,044 (A) and 1,553,636,128 (C).

| Arm | Completed steps | Counter | Corpus stream-rate | Counter minus that rate |
|---|---:|---:|---:|---:|
| A | 5,400 | 4,816,358,983 | 4,677,698,632 | +138,660,351 |
| A | 7,630 | 6,628,017,935 | 6,609,414,919 | +18,603,016 |
| C | 5,400 | 3,531,851,242 | 3,275,240,362 | +256,610,880 |
| C | 7,630 | 4,813,230,411 | 4,627,793,326 | +185,437,085 |

Three uniform permutations of the same per-document counts land within about 2 MB of those corpus rates. The logged counter does not. The LCG Fisher–Yates used for training (`perm.py` lines 8–17, `j = state % (i+1)`) is a bijection, and its remainders are marginally even, but the early positions are biased toward low document indices. For seed 1 the mean document index in the step-5400 prefix is 1,208,541 (A) and 1,167,876 (C), against 1,238,825 for a uniform draw.

The stream itself is rank order, and that order is not exchangeable. The first half of the documents is about 13 bytes per content token (the head of shard 0 is almost entirely Nepali). The second half is about 5 bytes per content token for A and about 3.4 for C (the tail is English). Low indices are the byte-rich head, so this permutation covers more bytes per token than the corpus average until late in the run. By step 7,630, A has read 92% of its stream and the excess is down to 19 MB. C has read only 64% of its longer stream, so its full-run counter stays 185 MB above the corpus rate. That full-run figure, 4,813,230,411, is seed 1's B*.

## B* checkpoints

The counter matches bytes covered, so the checkpoints are at equal true bytes aside from whole-step rounding.

B* for seed 1 is 4,813,230,411, which is C's counter at completed step 7,630. C's B* checkpoint is that step, so it sits on the mark.

A's B* checkpoint is the first completed step whose counter reaches that mark: step 5,397, counter 4,814,058,292. That is 827,881 bytes past B*. Step 5,396 is still short of the mark; the checkpoint rule saves the first whole step that reaches it.

## Plot

`scripts/m9/plot_training_stream_bpb.py` now places each logged step at that row's `bytes_seen`. The approximate training-stream BPB on the y-axis is unchanged (`loss * tokens_per_byte / ln 2`).

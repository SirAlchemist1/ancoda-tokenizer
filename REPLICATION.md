# Replication

The five final exports (`tokenizer.json` and the tiktoken rank file) are on the [tokenizers-v1 release](https://github.com/SirAlchemist1/ancoda-tokenizer/releases/tag/tokenizers-v1). The load that prints a token count is in [Using the tokenizer](README.md#using-the-tokenizer).

## `make verify`

`make verify` runs, on the Mac, with no GPU:

```
uv run --frozen pytest tests/test_guarantee.py tests/test_export_equivalence.py tests/test_model_eval.py -q
```

Those tests check:

- lossless round-trip and protected-unit integrity (covered units stay whole, mark-initial tokens, equality of the two exports)
- export equivalence (Hugging Face `tokenizer.json` and the tiktoken rank file match the canonical encoder in [docs/SPEC.md](docs/SPEC.md) §3.5)
- the model-eval scoring helpers, including the bits-per-byte definition used by the independent recompute (tolerance 0.001 BPB)

They do not retrain a model and they do not need the training parquet. The full `pytest` suite also needs the downloaded corpus; `make verify` does not.

## Model-level test

The frozen record is [`docs/model_eval_logs/`](docs/model_eval_logs/): train logs, manifests, queue log, and eval JSON. The pre-registration is [`docs/MODEL_EVAL_PREREG.md`](docs/MODEL_EVAL_PREREG.md). Called M9 in the accompanying paper.

The independent check that does not retrain is [`scripts/model_eval/independent_bpb_recompute.py`](scripts/model_eval/independent_bpb_recompute.py). It uses only torch plus tokenizers or SentencePiece. On the first 300 sealed Nepali documents by `doc_id` (1,652,418 bytes), the Mac recompute matched the pipeline within 0.001 BPB (ancoda-64k |Δ|=0.000014, yak-64k-v1 |Δ|=0.000019, sidskarki-64k |Δ|=0.000003).

### What was trained

One shared document stream for every arm. Selection rank is `sha256(b"m9-v1\n" + doc_id)` as locked in the pre-registration; that prefix is part of the hash and is not a display name. Mix by original UTF-8 bytes: Nepali 40%, Hindi 15%, other Devanagari 5%, English 35%, code 5%. Stream list sha256 `10fddf2e2d95ec356188958ce03bd1cb34636d6e3db1f5181dedb2182edd1bcd` (2,477,651 documents, 7,189,336,358 bytes).

Model, identical except vocabulary size: Llama-style decoder, hidden 512, 8 layers, 8 heads, head dim 64, SwiGLU intermediate 1408, RoPE theta 10,000, RMSNorm epsilon 1e-5, tied embeddings, context 1,024. Weights `Normal(0, 0.02)` under the seed. AdamW betas 0.9 and 0.95, weight decay 0.1 on matrices, peak learning rate 1e-3, 153-step linear warmup, cosine to 1e-4, grad clip 1.0, bf16, eager (`torch.compile` was unavailable). 7,630 steps, 128 sequences × 1,024 tokens = 131,072 tokens per step (1,000,079,360 stream tokens). Micro-batch 4. Seed `s` is both the init seed and the Fisher–Yates permutation seed.

Plain-64k, the tokenizer for arm D, is `configs/model_eval_plain_64k.yaml`: vocab 65,536, no blocks and no bans, counts `artifacts/counts/n60j5/pretoken_counts.bin`, seed `20261104`.

### Arms, seeds, and the driver command

`scripts/model_eval/run_full.sh` runs one job at a time. Each job is:

```
python -m ancoda_tok.model_eval train \
  --tokens "$HOME/model_eval/tokens/<arm>" \
  --arm <arm> --seed <seed> --vocab <vocab> \
  --steps 7630 --run-dir "$HOME/model_eval/runs/<arm>-s<seed>" \
  --micro-batch 4 --checkpoint-every 500 \
  --bstar <B*> --resume --build "$HOME/model_eval/code/BUILD.json"
```

Vocabulary is 65536, except sidskarki-64k (64001) and plain-64k (65537). B* is fixed per seed from arms A, B, and C: seed 1 `4813230411`, seed 2 `4795818967`, seed 3 `4805817305`.

| Arm | Name | Seeds |
|---|---|---|
| A | ancoda-64k | 1, 2, 3 |
| B | yak-64k-v1 | 1, 2, 3 |
| C | sidskarki-64k | 1, 2, 3 |
| D | plain-64k | 1, 2, 3 |
| E | yak-64k-v2.0 | 1, 2, 3 |
| F | ancoda-64k-nojoiner | 1, 2, 3 |

The driver order is A/B/C for seeds 1, 2, 3, then E seed 1, D seed 1, E seed 2, D seed 2, E seed 3, D seed 3, then F seeds 1, 2, 3.

### Hardware and runtime

Training was eager bf16 on an XPU, 4 torch threads. The frozen ancoda-64k seed-1 log records about 4.66 s/step and a peak device allocation of 6,202,934,784 bytes. That run started `2026-11-10T18:01:08Z` and finished training `2026-11-11T04:00:53Z`, about 10 hours. The same step time is a realistic estimate for each other 7,630-step job, so the full queue is on the order of several days on that one device. The device name is not in the frozen logs. A later read-only check of three seed-1 final checkpoints was run on the Mac with MPS; it did not retrain.

Do not retrain to confirm the tables. Read [`docs/model_eval_logs/`](docs/model_eval_logs/) and, if the checkpoints and the 300-document sample are on disk, run `scripts/model_eval/independent_bpb_recompute.py`.

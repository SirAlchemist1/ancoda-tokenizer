# Ancoda tokenizer

Lossless, Nepali-first byte-level BPE with Devanagari akshara base units. Flagship: **ancoda-64k**. Companion sizes: 32k, ne-64k, 128k, 250k.

Frozen M8 results live in [`docs/GATE_A.md`](docs/GATE_A.md). Spec: [`PLAN.md`](PLAN.md). Decisions: [`docs/DECISIONS.md`](docs/DECISIONS.md). Evidence hashes: [`docs/EVIDENCE_M8.md`](docs/EVIDENCE_M8.md).

Lower fertility is better. Diff = ours − competitor. **WIN** = 95% CI entirely below 0 (seed `20261104`, 10k resamples). Sealed numbers are report-only and never drive training or config changes.

## Headline (FLORES+, whole-file first-byte unless noted)

| Ours | Competitor | Nepali | English | Hindi | Devanagari mean |
|---|---|---|---|---|---|
| ancoda-64k | yak-64k-v1 | **WIN** −0.0059 | LOSE +0.0263 | **WIN** −0.0406 | **WIN** −0.0571 |
| ancoda-64k | yak-64k-v2.0 | **WIN** −0.0193 | **WIN** −0.0059 | **WIN** −0.0509 | **WIN** −0.0389 |
| ancoda-64k | sidskarki-64k† | TIE +0.0061 | **WIN** −0.9942 | **WIN** −0.3264 | **WIN** −0.2862 |
| ancoda-64k | arkios | **WIN** −0.2369 | LOSE +0.1174 | **WIN** −0.6324‡ | **WIN** −0.5238‡ |
| ancoda-64k | sarvam-1 | **WIN** −1.1384‡ | **WIN** −0.0206‡ | **WIN** −0.0616‡ | **WIN** −0.4509‡ |
| ancoda-ne-64k | yak-64k-v1 | **WIN** −0.0515 | LOSE +0.8629 | LOSE +0.2117 | LOSE +0.1637 |
| ancoda-ne-64k | sidskarki-64k† | **WIN** −0.0395 | **WIN** −0.1576 | **WIN** −0.0741 | **WIN** −0.0654 |
| ancoda-128k | yak-128k-v1 | LOSE +0.0045 | LOSE +0.0105 | **WIN** −0.0193 | **WIN** −0.0392 |
| ancoda-128k | yak-128k-v2.0 | **WIN** −0.0079 | **WIN** −0.0108 | **WIN** −0.0287 | **WIN** −0.0232 |
| ancoda-128k | llama3 / sarvam-m / qwen2.5 / qwen3 | **WIN** (large) | LOSE on eng | **WIN** | **WIN** |
| ancoda-250k | indicbert-v2 | **WIN** −0.2500‡ | **WIN** −0.0079‡ | TIE +0.0034‡ | **WIN** −0.1216‡ |
| ancoda-250k | bloom / gemma3 / sarvam-30b | **WIN** | **WIN** | **WIN** | **WIN** |
| ancoda-32k | yak-32k-v1 | **WIN** −0.0144 | LOSE +0.0338 | **WIN** −0.0713 | **WIN** −0.0857 |

† Scoreboard used sidskarki **whole-file** FLORES (1.4448). Chosen reporting path is **per-line** SentencePiece (1.3699); under that path ancoda-64k and ne-64k **LOSE** on FLORES Nepali vs sidskarki. See Gate A.  
‡ Per-line encoding for that baseline.

### Absolute fertilities (FLORES columns)

| Model | npi | eng | hin | Deva mean |
|---:|---:|---:|---:|---:|
| ancoda-64k | 1.4509 | 1.4144 | 1.3401 | 1.5188 |
| ancoda-ne-64k | 1.4053 | 2.2510 | 1.5924 | 1.7396 |
| yak-64k-v1 | 1.4568 | 1.3881 | 1.3808 | 1.5759 |
| yak-64k-v2.0 | 1.4702 | 1.4203 | 1.3910 | 1.5577 |
| sidskarki-64k (whole-file) | 1.4448 | 2.4086 | 1.6665 | 1.8050 |
| ancoda-128k | 1.3745 | 1.3019 | 1.2773 | 1.4317 |
| ancoda-250k | 1.3284 | 1.2287 | 1.2404 | 1.3767 |
| ancoda-32k | 1.5715 | 1.5721 | 1.4373 | 1.6547 |

## Where we win (summary)

- **ancoda-64k vs Yak 64k-v1 / v2.0:** WIN on Nepali, Hindi, and Devanagari mean; WIN on English vs v2.0; LOSE English vs v1.
- **ancoda-64k vs sidskarki / arkios / sarvam-1:** WIN on Hindi and Devanagari mean (and English vs sidskarki / sarvam-1).
- **ancoda-128k vs Yak 128k-v2.0:** WIN on Nepali, English, Hindi, Deva mean. vs v1: WIN Hindi + Deva mean; LOSE Nepali + English by small margins.
- **ancoda-128k / 250k vs large multilingual baselines** (Llama-3, Qwen, Sarvam-M, Bloom, Gemma3, …): WIN on Nepali / Hindi / Deva mean (English mixed).
- **ancoda-32k vs Yak 32k-v1:** WIN Nepali, Hindi, Deva mean; LOSE English.
- **Integrity:** all Ancoda builds: 0 covered violations, 0 mark-initial, exact HF↔tiktoken round-trip, orphan-mark share ≪ Yak v1.

## Sealed holdout (report-only)

Reduced list without joiner-source docs: `docs/sealed_test_set_no_joiners.json` (n=35623). Document-level CIs.

**Nepali headline**

| Pair | Verdict |
|---|---|
| ancoda-64k vs yak-64k-v1 / v2.0 | **WIN** −0.0084 / −0.0208 |
| ancoda-64k vs sidskarki (per-doc SP) | LOSE +0.0754 |
| ancoda-ne-64k vs yak-64k-v1 / v2.0 | **WIN** −0.0531 / −0.0654 |
| ancoda-ne-64k vs sidskarki | LOSE +0.0308 |
| ancoda-128k vs yak-128k-v1 | **WIN** −0.0015 |

**Unweighted mean of 11 Devanagari groups** (Hin, Mar, Nep, San, anp, awa, bho, gom, hne, mai, new)

| Pair | Verdict |
|---|---|
| ancoda-64k vs yak-v1 / v2.0 / sidskarki / arkios / sarvam-1 | **WIN** (incl. sidskarki −0.2991) |
| ancoda-128k vs yak-128k-v1 / v2.0 / llama3 / sarvam-m / qwen | **WIN** |
| ancoda-250k vs indicbert / bloom / gemma3 / sarvam-30b | **WIN** |
| ancoda-32k vs yak-32k-v1 | **WIN** −0.1792 |
| ancoda-ne-64k vs yak-v1 / v2.0 | LOSE |
| ancoda-ne-64k vs sidskarki | TIE |

Full tables, robustness, and PLAN 3.10 gate checks: [`docs/GATE_A.md`](docs/GATE_A.md).

## Builds

| Name | Role |
|---|---|
| ancoda-64k | Flagship multilingual Devanagari + English |
| ancoda-ne-64k | Nepali-heavy 64k (trades English/Hindi fertility) |
| ancoda-128k | Larger general build |
| ancoda-250k | Largest build (frontier pick among 250k points) |
| ancoda-32k | Small footprint |

Exports: Hugging Face `tokenizer.json` and tiktoken ranks; both must match the canonical encoder in PLAN 3.5.

## License / comparison arms

Yak, sidskarki-64k, HimalayaAI, and Arkios are evaluation-only within their licenses. Do not copy their vocab, merges, or regex into Ancoda. HimalayaAI BPE-128k was skipped (Hub card: no license). FLORES+ text is never re-hosted in this repo.

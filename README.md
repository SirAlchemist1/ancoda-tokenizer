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
| ancoda-128k | himalaya-128k§ | **WIN** −0.0873 | TIE +0.0034 | **WIN** −0.2406 | **WIN** −0.3755 |
| ancoda-128k | llama3 / sarvam-m / qwen2.5 / qwen3 | **WIN** (large) | LOSE on eng | **WIN** | **WIN** |
| ancoda-250k | indicbert-v2 | **WIN** −0.2500‡ | **WIN** −0.0079‡ | TIE +0.0034‡ | **WIN** −0.1216‡ |
| ancoda-250k | bloom / gemma3 / sarvam-30b | **WIN** | **WIN** | **WIN** | **WIN** |
| ancoda-32k | yak-32k-v1 | **WIN** −0.0144 | LOSE +0.0338 | **WIN** −0.0713 | **WIN** −0.0857 |

† Scoreboard used sidskarki **whole-file** FLORES (1.4448). Chosen reporting path is **per-line** SentencePiece (1.3699); under that path ancoda-64k and ne-64k **LOSE** on FLORES Nepali vs sidskarki. See Gate A.  
‡ Per-line encoding for that baseline.  
§ HimalayaAI BPE-128k, 128,000 pieces. Same whole-file first-byte bootstrap as the other rows (seed `20261104`, 10k). Off the frozen Gate A scoreboard: the Hub card states no license. See [vs HimalayaAI BPE-128k](#vs-himalayaai-bpe-128k).

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

## vs HimalayaAI BPE-128k

FLORES+ devtest, whole file joined by newlines, tokens / `str.split()` words. Diff = ours − HimalayaAI. **WIN** = 95% CI entirely below 0 (seed `20261104`, 10,000 line resamples, whole-file first-byte). ancoda-128k has 131,072 ids; HimalayaAI BPE-128k has 128,000 pieces.

| Ours | Nepali | English | Hindi | Devanagari mean |
|---|---|---|---|---|
| ancoda-128k | **WIN** −0.0873 | TIE +0.0034 | **WIN** −0.2406 | **WIN** −0.3755 |
| ancoda-64k | TIE −0.0109 | LOSE +0.1159 | **WIN** −0.1777 | **WIN** −0.2885 |
| ancoda-250k | **WIN** −0.1334 | **WIN** −0.0698 | **WIN** −0.2775 | **WIN** −0.4306 |

On the matched size, ancoda-128k uses **6.0%** fewer Nepali tokens (1.3745 vs 1.4618), **15.9%** fewer Hindi tokens (1.2773 vs 1.5179), and **20.8%** fewer tokens on the seven-language Devanagari mean (1.4317 vs 1.8072). English is a tie (1.3019 vs 1.2985).

| | ancoda-128k | HimalayaAI BPE-128k |
|---|---:|---:|
| Nepali fertility | 1.3745 | 1.4618 |
| English fertility | 1.3019 | 1.2985 |
| Hindi fertility | 1.2773 | 1.5179 |
| Devanagari mean | 1.4317 | 1.8072 |
| Nepali lines, exact round-trip | 1,012 / 1,012 | 1,004 / 1,012 |
| Nepali lines that keep every character | 1,012 / 1,012 | 1,010 / 1,012 |
| Orphan-mark share, Nepali | 0.0039% | 1.89% |
| ZWJ token ratio | 1.2698 | 2.4457 |
| ZWNJ token ratio | 1.2737 | 2.4457 |

HimalayaAI's vocabulary has no piece containing ऊ, ऋ, ऐ, ः, ़, ऽ, ॥, or ॐ. Two of 1,012 Nepali lines lose characters, so its fertility is slightly flattered. The gaps above are much larger than that. The ZWJ and ZWNJ ratios are the Gate A probe: a joiner after every virama before a Devanagari letter, tokens(rewritten) / tokens(original) on the Nepali devtest.

ancoda-32k loses Nepali (+0.1097) and English (+0.2736) to this 128k model and wins Hindi (−0.0806) and the Devanagari mean (−0.1525). ancoda-ne-64k wins Nepali (−0.0565) and the Devanagari mean (−0.0677) and loses English (+0.9526) and Hindi (+0.0746).

## Where we win (summary)

- **ancoda-64k vs Yak 64k-v1 / v2.0:** WIN on Nepali, Hindi, and Devanagari mean; WIN on English vs v2.0; LOSE English vs v1.
- **ancoda-64k vs sidskarki / arkios / sarvam-1:** WIN on Hindi and Devanagari mean (and English vs sidskarki / sarvam-1).
- **ancoda-128k vs Yak 128k-v2.0:** WIN on Nepali, English, Hindi, Deva mean. vs v1: WIN Hindi + Deva mean; LOSE Nepali + English by small margins.
- **ancoda-128k vs HimalayaAI BPE-128k:** WIN on Nepali (−6.0% tokens), Hindi (−15.9%), and the Devanagari mean (−20.8%). English is a tie. Exact Nepali round-trip is 1,012/1,012 for Ancoda and 1,004/1,012 for HimalayaAI.
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

## License

Copyright (c) 2026 Ancoda Labs Pvt. Ltd. Both licences below are noncommercial. **Commercial use of any part of this project needs a separate licence from Ancoda Labs.**

| What | Licence |
|---|---|
| Code: `crates/`, `python/`, `scripts/`, `tests/`, `configs/`, `Makefile`, `Cargo.toml`, `Cargo.lock`, `pyproject.toml`, `uv.lock`, `rust-toolchain.toml` | [`LICENSE`](LICENSE), PolyForm Noncommercial 1.0.0 |
| Ancoda tokenizer files (`tokenizer.json` and tiktoken ranks for every Ancoda build, distributed separately from this repo) | [`LICENSE-DATA`](LICENSE-DATA), CC BY-NC 4.0 |
| All of `docs/` (including pre-registration, evidence, and sealed document-ID lists) | [`LICENSE-DATA`](LICENSE-DATA), CC BY-NC 4.0 |
| Result and snapshot files: `artifacts/` (local, not committed), `docs/m9_server_copy/` | [`LICENSE-DATA`](LICENSE-DATA), CC BY-NC 4.0 |

Third-party exception: the Unicode-to-Preeti table in `python/ancoda_tok/clean/preeti_convert.py` is from [globalpolicy/UnicodeToPreeti](https://github.com/globalpolicy/UnicodeToPreeti) and stays under its MIT licence. Third-party tokenizers, datasets and their licences are listed in [`docs/LICENSES.md`](docs/LICENSES.md); none of their files are covered by either licence above.

## Comparison arms

Yak, sidskarki-64k, HimalayaAI, and Arkios are evaluation-only within their licenses. Do not copy their vocab, merges, or regex into Ancoda. HimalayaAI's Hub card states no license: the section above is a local comparison only, and the model is not redistributed. FLORES+ text is never re-hosted in this repo.

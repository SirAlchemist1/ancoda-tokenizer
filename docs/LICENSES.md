# Licenses (M1 comparison tokenizers)

Columns: dataset/model ID, revision (commit), license as read on the card, card URL, attribution required, share-alike, redistribution permitted, training use permitted, notes for releasing our tokenizer.

Yak files are **evaluation / proxy-model comparison only** (CC BY-NC 4.0). Never copy Yak, sidskarki, HimalayaAI or Arkios vocab, merges, regex or settings into our build.

| ID | Kind (eval_panel) | Revision | License (as read) | Card URL | Attribution | Share-alike | Redistribution | Training use | Notes |
|---|---|---|---|---|---|---|---|---|---|
| `ampixa/yak-tokenizer` (all Yak panel builds) | `hf_json` (`v1.0/…`, `v2.0/…`) | `12b6d3010ec542b484ec3b673354852fe8bb85fb` | CC BY-NC 4.0 | https://huggingface.co/ampixa/yak-tokenizer | Yes | No (NC) | Non-commercial only | **No** — evaluation / research comparison only | Panel names: `yak-32k-v1`, `yak-64k-v1`, `yak-128k-v1`, `yak-64k-v1-phrase`, `yak-128k-v1-phrase`, `yak-64k-v2.0`, `yak-128k-v2.0` |
| `sidskarki/nepali-bpe-tokenizer` (`nepali_bpe_64k.model`) | `sentencepiece` | `fc5773dcf7b9dc2272e8f595f99828969621a855` | other / **community-use-1.0** (`LICENSE.md`) | https://huggingface.co/sidskarki/nepali-bpe-tokenizer | Yes (credit + link + note changes) | Yes (derivatives under same/compatible terms) | Yes under license conditions | Not for our training corpus; comparison arm only | **Evaluation and proxy-model comparison are allowed** as “use” under community-use-1.0 for Licensees with total annual revenue ≤ USD $100,000, with attribution. Above-threshold entities need a separate written license. Confirm Ancoda Labs revenue compliance before M9 arm 6. |
| `himalaya-ai/nepali-tokenizer-bpe-64k-5b` | `sentencepiece` (`…-64000.model`) | `1c39f7a61b78211e3c5bce64e29b5ffd10b438d5` | **Not stated** on Hub card or repo files (M1) | https://huggingface.co/himalaya-ai/nepali-tokenizer-bpe-64k-5b | Unknown | Unknown | Unknown | No (comparison only) | No LICENSE file in repo. Comparison arm only; do not copy into build; do not redistribute. |
| `himalaya-ai/nepali-tokenizer-bpe-100k-5b` | `sentencepiece` (`…-100000.model`) | `83beb6d9e7a9af6f8449399aa5e16b438f7cf396` | **Not stated** (M1) | https://huggingface.co/himalaya-ai/nepali-tokenizer-bpe-100k-5b | Unknown | Unknown | Unknown | No | Same as 64k. |
| `himalaya-ai/nepali-tokenizer-bpe-128k-5b` | `sentencepiece` (`…-128000.model`) | `c525ebe0b8f72705e549e6e62d36fb1b6a495df3` | **Not stated** (M1) | https://huggingface.co/himalaya-ai/nepali-tokenizer-bpe-128k-5b | Unknown | Unknown | Unknown | No | Same as 64k. |
| `sajalregmi4/arkios-tokenizer` | `hf_json` | `aef562fcaf8747bbd9461cdae61b61f99e97e048` | Apache-2.0 | https://huggingface.co/sajalregmi4/arkios-tokenizer | Yes (NOTICE if present) | No | Yes (Apache-2.0) | Not used for our training; comparison arm | Panel name `arkios`. |

## sidskarki community-use-1.0 — evaluation / proxy verdict (M1)

Read from Hub `LICENSE.md` at revision above:

- Grant covers use, copy, modify, merge, publish, and distribute, subject to attribution, ≤ $100k annual revenue, no sublicensing to above-threshold entities, and share-alike for derivatives.
- **Evaluation on this harness** and **proxy-model comparison (M9)** are forms of use → **permitted** when the revenue condition holds and attribution is given.
- Terms are clear; no M1 STOP for ambiguity. Human must confirm revenue ≤ $100k before relying on M9 arm 6.

## Harness

| ID | Revision | Notes |
|---|---|---|
| `Ampixa/yak-tokenizer` git clone (`third_party/yak-tokenizer`) | `9fe03a991dd7c44e078d574e4ffcd51a7606dd6a` | Eval harness only; not a training source. |

## Loading (from `configs/eval_panel.yaml`)

| Panel name | Kind | How loaded |
|---|---|---|
| sidskarki-64k | sentencepiece | `sidskarki/nepali-bpe-tokenizer` file `nepali_bpe_64k.model` |
| himalaya-64k / 100k / 128k | sentencepiece | `himalaya-ai/nepali-tokenizer-bpe-{64k,100k,128k}-5b` files `himalaya-ai_nepali-tokenizer-corpus-{64000,100000,128000}.model` |
| arkios | hf_json | `sajalregmi4/arkios-tokenizer` |
| Yak builds | hf_json | `ampixa/yak-tokenizer` paths `v1.0/…` and `v2.0/…` as in eval_panel.yaml |

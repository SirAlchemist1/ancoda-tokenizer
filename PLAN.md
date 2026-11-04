# Ancoda Nepali Tokenizer — Build Plan for the Cursor Agent (v2.3, 4 Nov 2026)

> **For the human (read this first, 2 minutes)**
> 1. On your Mac, create an empty folder `ancoda-tokenizer` on a drive with **at least 100 GB free** (your Mac's own disk is fine; an APFS-formatted external drive also works). Run `git init` in it and put this file at its root as `PLAN.md`.
> 2. Open the folder in Cursor and switch the chat to Agent mode. Milestones M0–M8 and M10 run on the Mac; no GPU is needed. Only M9 (the model check) needs a rented Linux GPU machine, reached over SSH.
> 3. Keep the Mac plugged in, lid open, on stable internet. Long jobs run under `caffeinate` and resume if interrupted.
> 4. Paste this into the agent chat:
>    `Read PLAN.md fully. Execute milestone M0 only, starting with the specification audit in M0 step 1. Follow every rule in section 2. Do not change any decision in section 3. Stop when M0's acceptance checks pass or at a STOP.`
> 5. After each milestone: read `docs/PROGRESS.md`, tell the agent `Commit this milestone.`, start a **new** chat, and paste:
>    `Read PLAN.md and docs/PROGRESS.md. Execute milestone M<n> only. Follow section 2. Stop when the acceptance checks pass or at a STOP.`
> 6. A **STOP** means the agent needs something only you can do. Do it, then tell it to continue.
>
> **What changed in v2.3 (4 Nov 2026): built from scratch, aimed at Yak**
> - Disk: no fixed 250 GB budget. M0 needs 100 GB free; M2 measures the real need and stops only if under 50 GB would remain.
> - Nothing from earlier internal work is used: no earlier code, lexicon, root list or test run. The `roots.py` check and its `sandhi_roots_290.tsv` file are removed.
> - sidskarki-64k is an outside tokenizer (Hugging Face account `sidskarki`, license community-use-1.0). Like HimalayaAI's and Arkios's, it is a comparison arm only.
> - Gate A is re-anchored on Yak. The 64k build must beat Yak v2 64k (the newest Yak build on its 3 Nov 2026 leaderboard) on every word-level row and have the lowest Nepali count of any 64k tokenizer on Yak's leaderboard. Beating Yak v1 64k on English and Hindi too is the stretch. A Yak scoreboard prints WIN, TIE or LOSE on every row where Yak's harness reports a number, so no loss is hidden.
>
> **What changed in v2 (after the external review, all verified 4 Nov 2026):**
> - `unicode-segmentation` is pinned to exactly **1.12.0**. Version 1.11.0 (the review's suggestion) ships Unicode 15.1 data but does **not** implement the conjunct rule GB9c (checked in its source), which would have silently broken the conjunct ablation.
> - The joiners in the pre-tokenizer pattern are written `\u200C\u200D`. The old `\x{200C}` form is rejected by Python's `regex` module (tested).
> - The tiktoken and Hugging Face exports now use one canonical algorithm: "lowest-rank concatenation" (tiktoken's). The Hugging Face file lists every split of every token as a merge. Tested: 0 mismatches in 200,000 words over 1,000 random vocabularies. The old export (`ignore_merges` alone) mismatched (tested).
> - Block 0 is fixed at 133 tokens. The Block 1 selection and the Block 2 ban are now exact algorithms.
> - Every Gate A metric is defined. The orphan-mark gate was impossible as written (0.00%); it now compares against the best competitor.
> - Decontamination adds exact-segment matching. Word lists are no longer used for it.
> - Wikipedia (share-alike license) is dropped from training.
> - Comparison tokenizer sources are confirmed (sidskarki-64k and HimalayaAI's are SentencePiece models). The code dataset is read file by file, because its loader is a script.
> - v2.2: the gold-set check is now blind, with a written guide, an integrity check, agreement statistics and a three-way mismatch rule (M3 step 6). Encoding-variant costs are measured and reported (3.9). Normalization stays off; tested: NFC/NFKC do not unify the two eyelash-ra spellings.
> - A human-annotated akshara test set, a plain-BPE control arm, a GPU pilot run, separator and scoring rules for the model check, an efficiency report and slip rules are added.

---

## 1. Mission and how you (the agent) work

**Mission.** Build a lossless, Nepali-first byte-level BPE tokenizer whose Devanagari base units are whole **aksharas** (3.3). It ships as a standard `tokenizer.json` and tiktoken file, and **no token boundary may ever fall inside a covered akshara**. Then measure, with small trained models, whether it beats Yak, Arkios and sidskarki-64k.

**The main target is Yak** (Ampixa). Everything is built from zero in this repo. No earlier internal code, data or results are used. Yak, sidskarki-64k, HimalayaAI and Arkios are outside tokenizers, used only for comparison.

**Hard dates.** The release candidate is due **22 Dec 2026**. The paper goes to ARR on **4 Jan 2027**.

**How you work.**

- Do **one milestone at a time**, in order. Never start a milestone until the previous one's acceptance checks pass.
- Before writing code for a milestone, write its task list in `docs/PROGRESS.md`, and tick items as you finish them.
- Every acceptance check is a command. Run it, paste the exact result into `docs/PROGRESS.md`, and only then mark the milestone done.
- If a check fails and two fix attempts do not solve it, **STOP** and report: what failed, the exact error, and what you tried.
- Never change a decision in section 3. If evidence says one is wrong, **STOP** and explain.
- Report numbers exactly as computed. Never round to make a target look met.

---

## 2. Non-negotiable rules

In M0, create `.cursor/rules/ancoda-tokenizer.mdc` with exactly this content:

```markdown
---
description: Non-negotiable rules for the Ancoda Nepali tokenizer project
globs:
alwaysApply: true
---
- Never apply Unicode normalization (NFC, NFD, NFKC, NFKD) to stored training text or inside the tokenizer. Normalized copies may be made only as temporary match keys for decontamination, never written back. The Yak harness may NFC its own inputs.
- Never commit anything under data/ or artifacts/. Never commit FLORES+ text anywhere, because its terms forbid re-hosting.
- Never retype the o200k regex. Read it in code and apply the edits in PLAN.md 3.1 with asserted replacement counts. Write the joiners as \u200C and \u200D, never \x{...} and never as invisible literal characters.
- The canonical encoder is defined in PLAN.md 3.5. Both exports must reproduce it exactly.
- Every produced artifact (snapshot, count table, tokenizer, eval result, model run) gets a manifest.json with: input file sha256s, config sha256, git commit, library versions, unicode-segmentation version, Unicode data version and file sha256s, seed, start/end time.
- Determinism: fixed seeds, sorted iteration, documented tie-breaking. The same inputs must give byte-identical outputs.
- Commit Cargo.lock, uv.lock and rust-toolchain.toml. Never upgrade a pinned dependency without the human's approval.
- Yak tokenizer files are CC BY-NC 4.0. Use them only as an evaluation or proxy-model comparison arm. Never copy their vocab, merges or regex into our build.
- sidskarki-64k (community-use-1.0), HimalayaAI and Arkios files are comparison arms only, within their licenses. Never copy their vocab, merges, data or settings into our build.
- No training corpus may contain evaluation text. Decontamination (PLAN.md M4 step 5) must run before any tokenizer is trained.
- A milestone is done only when its acceptance commands pass, with outputs recorded in docs/PROGRESS.md.
- Do not change decisions in docs/DECISIONS.md without the human's explicit approval.
- Rust: cargo fmt, cargo clippy -D warnings, cargo test must pass. Python: ruff and pytest must pass.
- This project runs on macOS. Any job expected to take more than 10 minutes runs under `caffeinate -dimsu`, writes output shard by shard, and skips completed shards, so it resumes after interruption.
- All Hugging Face downloads go to HF_HOME=./data/hf_cache on the repo drive.
- Before any step that writes more than 20 GB, check `df -h .`. STOP if fewer than 50 GB would remain.
```

---

## 3. Fixed decisions (the spec)

Copy this section into `docs/DECISIONS.md` in M0. It is the source of truth.

### 3.1 Pre-tokenizer

- **Source:** the o200k pattern, read in code. Either use `tiktoken.get_encoding("o200k_base")._pat_str` (this downloads the rank file once), or read `pat_str` from the `o200k_base()` function in the installed `tiktoken_ext/openai_public.py`. Both must give the same string; assert it.
- **Edit 1:** replace every `\p{M}]` with `\p{M}\u200C\u200D]`, so ZWNJ and ZWJ stay inside words. Assert exactly **4** replacements.
- **Edit 2:** replace `\p{N}{1,3}` with `\p{N}`, so every digit (ASCII and Devanagari) is its own pre-token. Assert exactly **1** replacement.
- **Keep everything else**, including the punctuation branch that absorbs trailing newlines, so ` ।\n` is one pre-token.
- Save to `configs/pretokenizer_pattern.txt` with its sha256.
- **Expected behavior** (verified on 4 Nov 2026 with Python `regex`, Hugging Face onig and tiktoken; all three agree):
  - `नेपालको राजधानी काठमाडौं हो ।\n` → `नेपालको` · ` राजधानी` · ` काठमाडौं` · ` हो` · ` ।\n`
  - `"र्\u200dय क्\u200cष १२३ abc 123"` (Python literal) → `"र्\u200dय"` · `" क्\u200cष"` · ` ` · `१` · `२` · `३` · ` abc` · ` ` · `1` · `2` · `3`
- **In tokenizer.json:** `Sequence([Split(Regex(pattern), behavior="isolated", invert=False), ByteLevel(add_prefix_space=False, use_regex=False)])`. `use_regex=False` is mandatory; the default re-applies GPT-2's letters-only regex, which splits Nepali at every vowel sign.

### 3.2 Normalization

None, in the tokenizer or the stored corpus.

### 3.3 Akshara (the Block 1 unit)

**Definition.** An akshara is a UAX #29 extended grapheme cluster computed **with rule GB9c disabled**. Each consonant keeps its vowel signs, virama, nukta, anusvara, chandrabindu, visarga and any joiner after a virama. A conjunct such as क्ष is two aksharas, क् and ष, which BPE may merge later.

**Implementation.**

1. Use the Rust crate `unicode-segmentation = "=1.12.0"` (Unicode 16.0.0).
   - This is the first release that implements GB9c.
   - 1.11.0 has Unicode 15.1 data but no GB9c code. 1.13.0 and 1.13.1 were yanked.
   - Record `unicode_segmentation::UNICODE_VERSION` in every manifest.
2. Compute extended grapheme clusters.
3. Inside any cluster that contains Devanagari, split after every U+094D virama (plus an immediately following U+200C or U+200D, if present) when the next code point is a Devanagari consonant. This undoes GB9c for Devanagari only.
4. The Devanagari consonant set = `InCB=Consonant ∩ Script=Devanagari`. Read it from Unicode **16.0.0** `DerivedCoreProperties.txt` and `Scripts.txt`. Never hand-code it.
5. **Mode `gb9c=true`:** return the raw clusters from step 2 (conjuncts joined). This is used only for the M7 and M9 ablation.

**Required unit tests (default mode):**

| Input | Aksharas |
|---|---|
| `स्कूल` | `स्` · `कू` · `ल` |
| `गर्नु` | `ग` · `र्` · `नु` |
| `क्ष` | `क्` · `ष` |
| `"र्\u200dय"` (र, virama, ZWJ, य) | `"र्\u200d"` · `"य"` |
| `"\u0958"` (precomposed क़) and `"\u0915\u093c"` (क + nukta) | one akshara each; the two byte strings stay distinct (never normalized) |

**Uncovered aksharas** (those not selected in 3.6):

- Their code points are still Block 0 tokens, and round-trip stays exact.
- Block 2 may join their pieces with neighbors.
- **The integrity guarantee applies only to covered aksharas.** The tail is always measured and reported, never hidden.

### 3.4 Merge blocks

| Block | Builds | Size | Order inside block |
|---|---|---|---|
| bytes | 256 single bytes | 256 | ID = byte value |
| Block 0 | Whole characters from UTF-8 bytes | **exactly 133** | 3 prefix tokens `E0 A4`, `E0 A5`, `E2 80`; then the 128 code points U+0900–U+097F in code-point order; then U+200C, U+200D |
| Block 1 | Covered aksharas from characters | ≤ 5% of vocab (3.6) | per 3.6 |
| Block 2 | Learned merges | the remainder (3.7) | learning order |
| reserved | special tokens | 1,024 | 3.7 |

**Block 2 bans.** A candidate pair `(L, R)` with `M = L + R` (byte strings) may be merged only if all three hold:

1. **Character integrity:** either both `L` and `R` decode as complete, valid UTF-8 strings; or `M` is a non-empty prefix (proper or complete) of the UTF-8 encoding of exactly one Unicode scalar value, starting at its lead byte. A merge that completes a multi-byte character is allowed; a merge that joins part of one character to anything outside it is not. Unit-test this predicate on valid, partial, continuation-only and invalid byte strings.
2. **No mark-initial tokens:** `M` decoded does not start with a code point of General_Category Mn, Mc or Me, or with U+200C or U+200D.
3. **Pre-token boundaries:** no merge crosses a pre-token (standard).

**No duplicates.** If the best pair's concatenation already exists as a token, re-segment those occurrences to the existing token without allocating a new ID, and continue. Every byte string has exactly one ID.

**Rank order = ID order:** bytes < Block 0 < Block 1 < Block 2 < reserved.

### 3.5 Canonical encoder and exports

**Canonical algorithm:** tiktoken's "lowest-rank concatenation" BPE.

- For each pre-token: if the whole byte string is a token, emit it.
- Otherwise start from single bytes. Repeatedly merge the adjacent pair whose concatenation is a token with the lowest rank (leftmost on ties). Stop when no adjacent concatenation is a token.

**The guarantee follows from this:**

- All Block 1 tokens rank below all Block 2 tokens, so every covered akshara is completed before any Block 2 merge can act.
- A string that spans two aksharas contains two base characters, so it can never be a Block 1 token.
- **The M6 property tests are the proof of record.**

**Hugging Face export:**

- `BPE(vocab, merges, ignore_merges=True, byte_fallback=False, dropout=None)`.
- `merges` = **every** split `(A, B)` of **every** multi-byte token `T` where `A`, `B` and `T = A+B` are all tokens, **whatever the ranks of A and B**.
- Sort merges by `(rank(T), rank(A), rank(B))`.
- Decoder `decoders.ByteLevel()`. Post-processor `TemplateProcessing(single="<|bos|> $A", pair="<|bos|> $A <|bos|> $B")`.
- Also write `tokenizer_config.json` and `special_tokens_map.json`.

Verified 4 Nov 2026 with tokenizers 0.23.3 and tiktoken 0.14.0: **0 mismatches in 200,000 random words over 1,000 random vocabularies**. Exporting only the training merges, or only splits with lower-ranked parts, produced mismatches. M3 repeats this test; M6 repeats it on the real vocabulary.

**tiktoken export:** a `.tiktoken` rank file (base64 token bytes, a space, the rank, one per line), plus `tiktoken.json` with the pattern and special tokens.

**llama.cpp:** a pre-tokenizer patch, after release only. It is not a gate.

### 3.6 Block 1 selection (exact algorithm)

1. Count every akshara containing at least one Devanagari code point over the Devanagari text of the training snapshot. `N` = total occurrences, single-code-point aksharas included (Block 0 already covers those).
2. Sort aksharas of 2 or more code points by count descending, ties broken by UTF-8 bytes ascending.
3. Walk the list. Each akshara's chain is `c1+c2`, then `c1c2+c3`, and so on. Its **incremental cost** = the chain prefixes not already selected.
4. Add the akshara if `selected merges + cost ≤ cap`. Otherwise **stop**; never skip ahead.
5. Also stop when covered occurrences reach **99.99% of N**.
6. **Caps** (5% of vocab): 1,638 at 32,768; 3,276 at 65,536; 6,553 at 131,072.
7. The coverage target never overrides the cap. If the cap binds at, say, 99.4%, report 99.4%.
8. Record:
   - N, the akshara types selected, the Block 1 merge count and the coverage achieved
   - uncovered occurrences, and the top 50 uncovered aksharas
9. Block 1 order = selection order, each chain's prefixes before its extensions.

### 3.7 Vocabulary layout

| Total IDs | Fixed (bytes + Block 0 + reserved) | Block 1 cap | Block 2 minimum |
|---|---|---|---|
| 32,768 | 1,413 | 1,638 | 29,717 |
| **65,536 (flagship)** | 1,413 | 3,276 | 60,847 |
| 131,072 | 1,413 | 6,553 | 123,106 |

- **Block 2 size** = total − 1,413 − |Block 1|.
- **The top 1,024 IDs are reserved.** From the first reserved ID, in order: `<|bos|> <|eos|> <|pad|> <|start_turn|> <|end_turn|> <|think|> <|/think|> <|tool_call|> <|/tool_call|> <|tool_result|> <|/tool_result|> <|fim_prefix|> <|fim_middle|> <|fim_suffix|> <|mask|> <|sep|>`, then `<|reserved_16|>` … `<|reserved_1023|>`. All are registered as special tokens.

### 3.8 Training mix (by UTF-8 bytes)

| Nepali | English | Hindi | Other Devanagari (Maithili, Sanskrit) | Code | Math |
|---|---|---|---|---|---|
| 60% | 22% | 6% | 2% | 6% | 4% |

- **Size:** 10 GB total. If the Mac has less than 32 GB of RAM, use 5 GB with the same shares. Never mix sizes within a sweep.
- **Sweep (M7):** Nepali at 50%, 60% and 70%, others scaled pro rata. **Pick rule:** the point that meets every Gate A fertility row with the lowest Nepali fertility; on a tie, the lower English fertility.
- **Excluded from training:**
  - Wikipedia: its CC BY-SA share-alike license complicates releasing the vocabulary, and it is about 0.5% of public Nepali text.
  - All evaluation sets.
  - Yak's files.

### 3.9 Metric definitions

All counts use `add_special_tokens=False`.

| Metric | Definition |
|---|---|
| Fertility, whole-file | tokens of the whole FLORES+ devtest file joined by `\n` (no trailing newline), encoded once ÷ `len(text.split())`. Yak's definition. |
| Fertility, per line | Σ tokens of each line encoded separately ÷ Σ words. A corpus ratio, **not** a mean of per-line ratios. |
| Newline gap | whole-file − per-line fertility, Nepali |
| Parity | tokens(language) ÷ tokens(English) over the 1,012 line-aligned devtest sentences, each encoded separately |
| Bytes per token; tokens per KB | UTF-8 bytes ÷ tokens; 1,024 × tokens ÷ UTF-8 bytes |
| Relative token saving vs X | (tokens_X − tokens_ours) ÷ tokens_X, same text |
| Orphan-mark share | tokens whose first character is Mn, Mc or Me ÷ all tokens, on encoded text. Yak's "orphan marks". |
| Mark-initial vocab entries | Block 1 or Block 2 entries whose first character is Mn, Mc, Me, ZWJ or ZWNJ. Block 0 single-character tokens are excluded by design. |
| Covered-akshara violations | token boundaries falling strictly inside the byte span of a covered akshara |
| Tail break rate | boundaries strictly inside uncovered aksharas ÷ uncovered akshara occurrences, with a document-level bootstrap 95% CI |
| Byte pieces | tokens that are not whole UTF-8 characters or character sequences ÷ all tokens |
| ZWJ / ZWNJ probe | tokens of the devtest rewritten with a ZWJ (or ZWNJ) after every virama that precedes a consonant ÷ tokens of the original. Yak's `robustness.py`. |
| Round-trip | share of segments with `decode(encode(x)) == x` byte-exactly |
| Akshara gold agreement | share of words in `tests/data/akshara_gold.tsv` whose segmentation equals the adjudicated gold |
| Encoding-variant cost (reported, not gated) | on FLORES+ npi and Aya, tokens of a rewritten copy ÷ tokens of the original, for three rewrites: every nukta letter swapped between its precomposed (U+0958–U+095F) and decomposed form; every eyelash ra written in the older order (र, ZWJ, virama) instead of र, virama, ZWJ; and the whole text NFC-normalized. A value near 1.00 means variant spellings cost almost nothing. |

### 3.10 Gate A targets (65,536 build, FLORES+ devtest via Yak's harness, unless noted)

**What "beat Yak" means here.** Better than Yak v2 64k (the newest Yak build on its 3 Nov 2026 leaderboard) on every word-level row below, and a lower Nepali count than every 64k tokenizer on Yak's leaderboard. References are from Yak's public `docs/RESULTS.md` (generated 3 Nov 2026). M1 re-measures them; the M1 values are the ones the gate uses.

| Check | Pass bar | Reference (64k) |
|---|---|---|
| Nepali fertility, whole-file | below 1.445 | sidskarki-64k 1.445; Yak v1 1.457; Yak v2 1.470 |
| English fertility | below 1.420 | Yak v2 1.420; Yak v1 1.388 |
| Hindi fertility | below 1.385 | Yak v2 1.385; Yak v1 1.376 |
| Covered-akshara violations (all eval texts plus 10M snapshot lines) | 0 | Yak measures but does not guarantee this |
| Mark-initial vocab entries | 0 | — |
| Orphan-mark share, Nepali FLORES+ | below Yak v2 64k's exact value measured in M1 (reported as 0.2%), and every occurrence inside an uncovered akshara or malformed sequence | Yak v2 0.2%; Yak v1 9.2%; sidskarki-64k 1.7% |
| Orphan-mark share, everyday Nepali (Aya) | reported; same explanation rule | — |
| Exact round-trip; unknown tokens | 100%; 0 | Yak 100%; 0. sidskarki-64k 99.1% |
| Devanagari code points with no token | 0 of 128 | Yak 0; sidskarki-64k 26 |
| Nepali byte pieces | ≤ 0.05% | Yak v2 0.02%; Yak v1 0.07% |
| Newline gap | ≤ 0.010 | Yak 0.000; sidskarki-64k 0.075 |
| Hugging Face IDs == tiktoken IDs | 100% of eval texts plus 1M lines | — |
| ZWJ and ZWNJ probes | below 2.560 and 2.560 (sidskarki-64k). Beating Yak here is not required; the result goes on the scoreboard | Yak v2 1.504 and 1.691; Yak v1 1.661 and 1.965 |
| Akshara gold agreement | 100%, or every difference documented and approved by the human | — |

- **Stretch (beat both Yak 64k builds):** English below 1.388 and Hindi below 1.376.
- **Yak scoreboard** (in `docs/GATE_A.md`): for every metric the harness reports for `yak-64k-v1` and `yak-64k-v2.0`, print ours, both Yak values and WIN, TIE or LOSE. This includes everyday Nepali on Aya (Yak v1 1.446, Yak v2 1.455) and the mean over the languages of Nepal (Yak v2 1.843). No row is hidden or dropped.
- **Different category:** Yak's phrase builds let one token span up to 4 words (64k phrase: Nepali 1.374; 128k phrase: 1.267). Report them in a separate, labelled table. Never claim to beat them unless that table shows it.
- **32k and 128k:** Nepali below Yak v1 at the same size (1.586 and 1.370), with English no more than 2% above Yak's.
- **Always reported, never gated:** Romanized Nepali word-list fertility. Romanized Nepali is out of scope for v1.

### 3.11 Claims discipline (for the paper and model card)

- **Tokenizer-level claims** ("fewer tokens", "never splits a covered akshara") require Gate A.
- **Model-level claims** ("trains a better model") require Gate B. A reduced-tier (single-seed) result is labeled "single seed, not significance-tested".
- **"Beats Yak"** may be written only for scoreboard rows marked WIN, naming the Yak build and size (for example "fewer Nepali tokens than Yak v2 64k").

### 3.12 Slip rules

If the schedule slips, cut in this order:

1. The 128k build.
2. The 32k build.
3. The full GPU tier, falling back to reduced.
4. The model-level conjunct ablation arm; it stays at the tokenizer level.
5. The efficiency extras.

**Never cut:** the guarantee tests, decontamination, the equality test, the gold agreement or the core Gate A rows.

---

## 4. Repository layout (create in M0)

```
ancoda-tokenizer/
  PLAN.md  README.md  Makefile
  pyproject.toml  uv.lock            # uv, Python 3.11
  Cargo.toml  Cargo.lock  rust-toolchain.toml
  .gitignore                         # data/, artifacts/, third_party/, .venv/, target/
  .cursor/rules/ancoda-tokenizer.mdc
  configs/
    sources.yaml  mix.yaml  eval.yaml  proxy.yaml
    build_{32k,64k,128k}.yaml
    ladder_{plain,b0,b0b1,final}.yaml  ablation_gb9c.yaml
    pretokenizer_pattern.txt         # generated in M3
  crates/
    ancoda-seg/                      # pre-tokenizer, akshara segmenter, counting; PyO3 via maturin
    ancoda-bpe/                      # trainer CLI: Blocks 0/1/2, bans, canonical encoder
  python/ancoda_tok/
    pattern.py  download.py  storage.py  mix.py  inventory.py
    clean/{utf8.py,lid.py,legacy_font.py,dedup.py,decontam.py,snapshot.py}
    export_hf.py  export_tiktoken.py
    eval/{akshara.py,equality.py,efficiency.py,gate_a.py}
    proxy/{prepare.py,train.py,bpb.py,gate_b.py}
  tests/
    data/{akshara_cases.tsv,akshara_gold.tsv,pretok_adversarial.tsv}
  docs/{DECISIONS.md,PROGRESS.md,LICENSES.md,RESULTS.md,SPEC_AUDIT.md,MACHINE.md,STORAGE.md}
  data/        # gitignored: raw/, clean/, snapshots/, eval/, hf_cache/
  artifacts/   # gitignored: counts/, inventory/, tokenizers/, eval/, runs/
  third_party/ # gitignored: yak-tokenizer clone, unicode/16.0.0/
```

---

## 5. Milestones

### M0 — Audit, environment and repo (target: 5 Nov)

1. **Specification audit, before writing any code.** Read sections 1–5 critically and write `docs/SPEC_AUDIT.md` covering:
   - contradictions
   - undefined quantities
   - wrong assumptions about third-party libraries
   - missing acceptance criteria

   Mark each item **blocking** or **non-blocking**. Change nothing in section 3. **STOP (human)** if any item is blocking.
2. **Machine check.** Write to `docs/MACHINE.md`:
   - `sw_vers` and `uname -m`
   - `sysctl -n hw.memsize` (in GB)
   - `sysctl -n hw.perflevel0.physicalcpu` (on an Intel Mac, `hw.physicalcpu`)
   - `df -h .`

   **STOP** if free disk is under 100 GB or RAM is under 16 GB. Space is managed as the build goes: M2 step 1 measures the real need before anything is downloaded.
3. **Toolchain (macOS).** Skip any line whose tool is already installed.
   ```bash
   xcode-select --install
   /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
   brew install git cmake pkg-config
   curl -LsSf https://astral.sh/uv/install.sh | sh
   curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
   source "$HOME/.cargo/env"
   uv python install 3.11
   ```
   Then:
   - Write `rust-toolchain.toml` pinned to the exact stable version now installed (`rustc --version`).
   - Export `HF_HOME="$PWD/data/hf_cache"` from the Makefile.
   - Set rayon and datatrove worker counts to the performance-core count.
4. Create the layout in section 4. Write the rules file (section 2) and `docs/DECISIONS.md` (section 3).
5. **Python dependencies:**
   ```bash
   uv init --python 3.11 --no-readme
   uv add tokenizers transformers tiktoken datasets "huggingface_hub>=0.34" pyarrow polars regex numpy pyyaml tqdm pyahocorasick sentencepiece
   uv add datatrove             # core; add an extra only if a MinHash block reports it missing
   uv add fasttext-wheel        # if this fails on Apple Silicon, try `fasttext` (needs cmake); if both fail, STOP
   uv add --dev pytest hypothesis ruff maturin
   ```
   Torch is **not** installed on the Mac.
6. **Rust workspace:**
   - `ancoda-seg`: `unicode-segmentation = "=1.12.0"`, `tokenizers` (default `onig` feature, so counting uses the same Split engine as tokenizer.json), `pyo3`, `rayon`, `serde`, `serde_json`, `sha2`.
   - `ancoda-bpe`: `rayon`, `serde`, `serde_json`, `clap`, `sha2`, `ahash`.
7. **Clone the harness:**
   ```bash
   git clone https://github.com/Ampixa/yak-tokenizer third_party/yak-tokenizer
   cd third_party/yak-tokenizer && uv sync --extra eval && cd -
   ```
   Record its commit hash.
8. **Unicode data.** Download into `third_party/unicode/16.0.0/` from `https://www.unicode.org/Public/16.0.0/ucd/`: `DerivedCoreProperties.txt`, `Scripts.txt`, `UnicodeData.txt` and `auxiliary/GraphemeBreakTest.txt`. Record their sha256. Assert the crate's `UNICODE_VERSION` is (16, 0, 0).
9. **Makefile targets** (stubs for now): `setup, storage-plan, download, clean-data, snapshot, inventory, counts, train, export, eval, gate-a, efficiency, proxy-pilot, proxy-train, gate-b, release`.

**STOP (human):** run `uv run hf auth login` and accept the `openlanguagedata/flores_plus` terms in the browser.

**Acceptance:**
```bash
cat docs/MACHINE.md docs/SPEC_AUDIT.md
uv run python -c "import tokenizers, tiktoken, datasets, fasttext, ahocorasick, sentencepiece; print('ok')"
cargo build --workspace && cargo test --workspace
uv run hf auth whoami
git ls-files | grep -E "Cargo.lock|uv.lock|rust-toolchain.toml"
```

---

### M1 — Baselines on Yak's harness (target: 7 Nov)

1. Run `uv run python scripts/eval_panel.py fetch`, then `... run`, inside `third_party/yak-tokenizer`.
2. Confirm its validity gate reproduces **5 of 5** published numbers within ±0.03.
3. From `configs/eval_panel.yaml`, record how each comparison tokenizer is loaded, and write each source, the resolved commit and the license to `docs/LICENSES.md`. Confirm these entries (read on 4 Nov 2026):

   | Name | Kind | Source |
   |---|---|---|
   | sidskarki-64k | SentencePiece BPE | `sidskarki/nepali-bpe-tokenizer`, file `nepali_bpe_64k.model` |
   | himalaya-64k / 100k / 128k | SentencePiece | `himalaya-ai/nepali-tokenizer-bpe-{64k,100k,128k}-5b`, files `himalaya-ai_nepali-tokenizer-corpus-{64000,100000,128000}.model` |
   | arkios | tokenizer.json | `sajalregmi4/arkios-tokenizer` |
   | Yak builds | tokenizer.json | `ampixa/yak-tokenizer`, files `v1.0/64k/tokenizer.json` etc. |

   The sidskarki-64k model card reports its own fertility (1.26 tokens/word at 64k) on a different evaluation set. **Only numbers from this harness are ever compared.**

   sidskarki-64k's license is community-use-1.0. Read its terms and record in `docs/LICENSES.md` whether evaluation and proxy-model comparison are allowed. **STOP (human)** if they are not, or if the terms are unclear.
4. Save the generated `docs/RESULTS.md` as `artifacts/eval/baseline_RESULTS.md`, with a manifest.
5. Record the exact values Gate A compares against: every harness metric for `yak-64k-v1` and `yak-64k-v2.0` (including the orphan-mark share and ZWJ/ZWNJ probes), sidskarki-64k's Nepali fertility and probes, and Yak v1 32k and 128k Nepali and English. Write them to `configs/eval.yaml`.
6. Write `eval/gate_a.py`. It reads harness output for a named tokenizer and prints the 3.10 rows with PASS/FAIL, plus the Yak scoreboard (3.10); our own metrics are added in M8.

**Acceptance:** the gate shows 5/5. Each value matches within ±0.003 (Nepali, English, Hindi):

| Panel name | Nepali | English | Hindi |
|---|---|---|---|
| `yak-64k-v1` | 1.457 | 1.388 | 1.376 |
| `yak-64k-v2.0` | 1.470 | 1.420 | 1.385 |
| `sidskarki-64k` | 1.445 | 2.409 | 1.656 |

If the panel names differ, use the names in `configs/eval_panel.yaml` and record the mapping. `uv run python -m ancoda_tok.eval.gate_a --name yak-64k-v2.0` prints the table and scoreboard.

---

### M2 — Storage plan, downloads, license log (target: 10 Nov)

1. **Storage plan first.**
   - Using the Hugging Face Hub API (`HfApi().list_repo_tree(..., recursive=True)` file sizes, or the dataset's parquet listing), sum the exact on-disk size of every file you will download. Never estimate when the API gives a number.
   - Write `docs/STORAGE.md`: projected size per component (raw parquet, the transient Hugging Face cache, cleaned parquet, the 3 snapshots, and counts/tokenizers/eval), plus the total and the free space `df -h .` reports now. There is no fixed budget.
   - **STOP (human)** if the projected total would leave less than 50 GB free on the disk. Propose caps, with `fw2_npi` first, or moving `data/` to an APFS-formatted external drive.
2. **Downloads.** Write `configs/sources.yaml` and `download.py`.
   - Save each source to `data/raw/<source_id>/` as zstd parquet with columns `doc_id, text, source, license`, plus a manifest.
   - One source at a time, under `caffeinate`.
   - Delete its Hugging Face cache after verifying the shards.
   - Samples stop at the bracketed byte budget.

| source_id | How to load | Role | Amount |
|---|---|---|---|
| fw2_npi | `load_dataset("HuggingFaceFW/fineweb-2", name="npi_Deva", split="train")` | train | all (or cap per step 1) |
| fw2_npi_test | same, `split="test"` | **eval only** | all |
| iriis | `load_dataset("IRIIS-RESEARCH/Nepali-Text-Corpus")`, train and test, column `Article` | train | all |
| sangraha_nep | `load_dataset("ai4bharat/sangraha", "verified", split="nep")` | train | all |
| indiccorp_npi | `load_dataset("ai4bharat/IndicCorpV2", "indiccorp_v2", split="npi_Deva")` | train | all |
| fwedu_en | `load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True)` | train | [5 GB] |
| fw2_hin | fineweb-2 `hin_Deva`, train, streaming | train | [2 GB] |
| fw2_hin_test | fineweb-2 `hin_Deva`, `split="test"` | **eval only** | all |
| fw2_mai, fw2_san | fineweb-2 `mai_Deva`, `san_Deva`, train | train | all |
| sangraha_mai, sangraha_san | sangraha `"verified"`, splits `mai`, `san` | train | all |
| owm | `load_dataset("open-web-math/open-web-math", split="train", streaming=True)` | train | [2 GB] |
| code | `codeparrot/github-code-clean`. Its loader is a Python script (`github-code-clean.py`), which current `datasets` may refuse, so **read the files in its `data/` folder directly**. First list them with `HfApi().list_repo_files(..., repo_type="dataset")` and open one file to confirm its format and its license and language columns. Then stream files, keeping only mit, apache-2.0, bsd-2-clause and bsd-3-clause rows. **If the files are not readable this way, or the columns differ, STOP and report what you found.** | train | [2 GB] |
| c4_val | `load_dataset("allenai/c4", "en", split="validation", streaming=True)` | **eval only** | [50 MB] |
| belebele_npi | `load_dataset("facebook/belebele", "npi_Deva")` | **decontam only** | all |
| aya_npi | `CohereLabs/aya_dataset`, rows whose language is Nepali | **eval and decontam** | all |
| glotlid | `hf_hub_download("cis-lmu/glotlid", "model.bin")`; confirm v3 on the model card | tool | 1 file |

- **Not downloaded for training:** Wikipedia (3.8).
- **IndicCorp:** a plain text file. Blank lines separate documents if present; otherwise each line is a document. Record the choice.
- **Never invent a config or split name.** If a load fails, list what exists (`get_dataset_config_names`, `list_repo_files`) and pick from the real list, or STOP.
3. **License log.** `docs/LICENSES.md` gets one row per source, with columns: dataset ID, revision (commit hash), license as read on the card, card URL, attribution required, share-alike, redistribution permitted, training use permitted, notes for releasing the tokenizer. Yak's CC BY-NC is listed explicitly as "evaluation only".

**STOP (human):** is there a licensed Romanized Nepali source? If not, it is out of scope for v1.

**Acceptance:** `make storage-plan` and `make download` complete. `docs/LICENSES.md` is complete. `uv run python -m ancoda_tok.download --report` prints docs and bytes per source, and at least 50 GB is still free.

---

### M3 — Pre-tokenizer, segmenter, export check (target: 14 Nov)

1. `pattern.py` builds the pattern exactly per 3.1, with the asserted counts and both sources compared, and writes `configs/pretokenizer_pattern.txt`.
2. **`crates/ancoda-seg`:**
   - `pretokenize(text)`, using the `tokenizers` crate's Split pre-tokenizer.
   - `aksharas(text, gb9c=false)`, per 3.3.
   - Python bindings via maturin.
3. **Segmenter tests:**
   - `gb9c=true` passes 100% of Unicode 16.0.0 `GraphemeBreakTest.txt`.
   - Default mode equals `gb9c=true` everywhere, except at the Devanagari virama-consonant joins it splits.
   - All 3.3 table cases pass.
   - 200+ cases in `tests/data/akshara_cases.tsv` pass, including:
     - ZWJ and ZWNJ after a virama
     - the eyelash ra (`"र्\u200d"`: र, virama, ZWJ)
     - U+0958–U+095F in precomposed and decomposed forms, kept byte-distinct
     - several marks on one consonant
     - marks after a nukta
     - क्ष, त्र, ज्ञ, श्र, द्ध
     - anusvara, chandrabindu and visarga combinations
     - standalone vowel signs and stray marks after a space or newline
     - malformed sequences (a mark at text start, a double virama)
     - Devanagari digits, danda and double danda
4. **Pre-tokenizer tests:**
   - The 3.1 examples must match exactly.
   - **Cross-engine equality:** pre-tokens from the `tokenizers` crate (onig), Python `regex` and tiktoken (fancy-regex) must be identical on:
     - 100k random lines from `data/raw` (Nepali, Hindi and English)
     - `tests/data/pretok_adversarial.tsv`: consecutive newlines, trailing whitespace, punctuation plus newlines, joiners at word start, middle and end, Devanagari and ASCII digits, mixed Nepali and English, emoji, CJK, and lone combining marks

     To read tiktoken's pre-tokens, compile the pattern with the same `fancy-regex` version tiktoken uses, in a small Rust test, or with an equivalent method; document which. Any mismatch: **STOP** and report the first 20.
5. **Export check (`tests/test_export_equivalence.py`).**
   - Reproduce the 3.5 experiment: the toy vocab `{a, b, c, d, bc, ab, abc}` plus 1,000 random vocabularies, each tested on 200 random words.
   - The 3.5 export must give **0** HF-vs-tiktoken mismatches.
   - The training-merges-only export must give at least one, which proves the test can detect the bug.
6. **Gold set (an independent check on the segmenter).**

   **What it tests.** Whether `aksharas()` cuts real words exactly where the 3.3 rule says. It tests the rule as written, not native intuition: under 3.3, क्ष is two units, क् | ष, even though a reader may see one letter.

   **a. Sample.** Write `tests/data/akshara_gold.tsv` with 500 words:
   - 300 from Aya Nepali (Apache-2.0)
   - 200 from IRIIS (MIT)

   The words must contain **no invisible characters** (ZWJ, ZWNJ) and no precomposed nukta letters U+0958–U+095F, because a human cannot see those differences. Those cases are covered at code-point level by `tests/data/akshara_cases.tsv` instead. At least 150 of the 500 must contain a conjunct. Use a fixed seed, never use FLORES words (the file is committed), and record each word's source and license.

   **b. Guide.** Write `docs/ANNOTATION_GUIDE.md`, one page:
   - the 3.3 rule in plain Nepali and English
   - 12 worked examples, including स्कूल → स्|कू|ल, गर्नु → ग|र्|नु, क्षेत्र → क्|षे|त्|र, विद्यालय → वि|द्|या|ल|य, and words with ँ, ं and ः
   - the instruction: insert `|` between aksharas and change nothing else

   **c. Blind sheets.** Write two copies, `annot_A.tsv` and `annot_B.tsv`, holding only the words. Never show the segmenter's output to annotators. Do not run the comparison until both sheets come back.

   **STOP (human):** two Nepali speakers fill one sheet each, independently, following the guide.

   **d. Integrity check.** For every row, removing `|` must give back the original word byte-for-byte. Rows that fail go back to the annotator.

   **e. Agreement.** Report word-level exact agreement and boundary-level Cohen's kappa between A and B, and list every disagreement. Keep these pre-adjudication numbers for the paper.

   **f. Adjudication.** The human resolves each disagreement against the guide, with a one-line reason, giving the adjudicated gold `akshara_gold.tsv`.

   **g. Comparison.** Run `aksharas()` on every word and compare with the gold. Sort every mismatch into one of three kinds:
   - **segmenter bug:** fix the code, add the word to `akshara_cases.tsv`, re-run all tests
   - **gold error:** correct the gold, with a note
   - **rule ambiguity:** **STOP (human)**; the decision goes into `docs/DECISIONS.md`, then both are re-run

   **Pass:** 100% agreement after this, with zero unexplained mismatches. Also report the gold compared against `gb9c=true` mode, for the paper.

   M3 may close before the sheets return; step g must pass before Gate A (M8).

**Acceptance:** `cargo test -p ancoda-seg` and `uv run pytest tests/test_seg.py tests/test_pattern.py tests/test_export_equivalence.py` pass, with cross-engine mismatches = 0.

---

### M4 — Cleaning, decontamination, snapshots (target: 17 Nov)

Run these steps in order, on training sources only. Write `data/clean/<source_id>/` and log counts before and after each step to `artifacts/clean/report.json`.

1. **UTF-8:** drop invalid documents. Strip C0 and C1 controls except `\t` and `\n`. No normalization.
2. **Language ID** (GlotLID v3, document level):
   - Nepali sources keep `npi_Deva`.
   - Re-route `hin_Deva` and `mai_Deva` to those buckets.
   - Drop everything else.
   - Record the confusion matrix.
3. **Legacy fonts** (Preeti and Kantipur):
   - **Classifier:** char 1–3-gram logistic regression, in numpy.
   - **Positives:** 5,000 Unicode Nepali sentences mapped through a Unicode→Preeti table from a published, permissively licensed converter. Cite it in `docs/LICENSES.md`.
   - **Negatives:** English lines and Romanized words.
   - Apply it to Latin-script lines in Nepali documents, and drop documents with more than 20% of lines flagged.

   **STOP (human):** approve the converter's license if unclear. Spot-check 100 flagged and 100 unflagged lines; the pass bar is 99% precision.
4. **Dedup** (datatrove local executor, performance-core workers, intermediates on the repo drive):
   - exact sha256 first
   - then MinHash with 5-gram word shingles and 112 hashes in 14 bands × 8 rows (FineWeb's setting)
   - across **all Nepali sources together**, keeping one per cluster in source preference: sangraha, iriis, indiccorp, fw2
5. **Decontamination.**
   - **Eval sets:**
     - FLORES+ dev and devtest for every harness language, read from the harness's fetched copy and never copied elsewhere
     - Belebele npi_Deva passages and questions
     - aya_npi
     - fw2_npi_test and fw2_hin_test
     - c4_val
   - **Not word lists.** Single-word lists are not contamination.
   - **Match keys:** NFC plus collapsed whitespace, on **temporary copies only**. Stored text never changes.
   - **Drop a training document if any of these holds:**
     - its full match key equals an eval document's
     - it contains, as a substring, any eval segment of 6–12 words (Aho-Corasick)
     - it shares any 13-word n-gram with an eval text
   - Eval segments under 6 words are counted, reported and not used.
   - Report drops per source and per rule.
6. **Snapshots:**
   - Sample the 3.8 mix at the 3.8 size, by bytes, at document level, seed 20261104.
   - Write zstd parquet to `data/snapshots/<sha>/`, with `manifest.json` (per-shard sha256, per-source docs and bytes, mix, seed) and `doc_ids.parquet` (source, doc_id, license).
   - Build the 60% snapshot, plus 50% and 70% for M7.
7. **Independent audit:** re-run all three decontamination rules on each finished snapshot. All three must report 0.

**Acceptance:** `make snapshot` builds three snapshots with at least 50 GB still free. The audit shows 0 on all rules. `uv run pytest tests/test_clean.py` passes.

---

### M5 — Akshara inventory and covered set (target: 19 Nov)

1. Run `aksharas()` over the 60% snapshot's Devanagari text. Write `artifacts/inventory/akshara_counts.tsv` (akshara, code points, count).
2. Run the 3.6 algorithm for 32k, 64k and 128k. Write `covered_{32k,64k,128k}.json` with every 3.6 step 8 field.
3. Plot coverage against Block 1 merges.
4. Repeat steps 1–3 with `gb9c=true` for the ablation.

**Acceptance:** the files exist, each Block 1 is within its cap, the selection is deterministic (re-running gives the same sha256), and `docs/RESULTS.md` has the table and plot.

---

### M6 — Trainer, exporter and the guarantee (target: 24 Nov)

**Counting** (`ancoda-seg counts`): stream a snapshot, pre-tokenize, and count unique pre-tokens as byte strings. Output a sorted `(bytes, u64 count)` table, plus a manifest.

**Trainer** (`ancoda-bpe train --config ...`):

1. Starting symbols per pre-token = bytes, with Blocks 0 and 1 applied using the **3.5 canonical encoder**.
2. Learn Block 2 merges with incremental pair counts and a priority queue:
   - Candidates are filtered by the 3.4 bans.
   - Ties break by count descending, then left bytes, then right bytes ascending.
   - Duplicates follow the 3.4 rule.
   - Stop at the 3.7 Block 2 size.
3. Write `ranks.json` (byte string to rank, which equals the ID) and a manifest.
4. **Config switches:** `block0`, `block1`, `bans`, `gb9c`. These are needed for the M7 ladder.

**Validation (`--plain`):** no blocks, no bans.

- On a 1 GB slice, compare with the Hugging Face `BpeTrainer` (`initial_alphabet=ByteLevel.alphabet()`, the same pattern and size).
- The merge lists must be identical, or every difference must be a tie at equal count.
- Write `artifacts/validation/plain_vs_hf.md`.

**Exporters:** per 3.5. The byte-level alphabet is `ByteLevel.alphabet()` via GPT-2's `bytes_to_unicode`.

**Guarantee tests (`tests/test_guarantee.py`)** on the 64k build from the 60% snapshot:

1. **Equality:** Hugging Face `encode(add_special_tokens=False)` == tiktoken `encode_ordinary`, on all eval texts plus 1M snapshot lines.
2. **Covered integrity:** across 10M sampled lines plus all eval texts, covered-akshara violations = 0. The tail break rate is reported with a CI.
3. **Vocab:** mark-initial entries in Block 1 or Block 2 = 0, scanning the whole vocabulary.
4. **Encoded-text orphans:** compute the orphan-mark share on FLORES+ npi and Aya. List every occurrence with its context, and confirm each lies in an uncovered akshara or a malformed sequence.
5. **Round-trip:** byte-exact on every test text.
6. **Coverage:** U+0900–U+097F, U+200C and U+200D are each a single token.
7. **Ban predicate:** unit tests per 3.4.
8. **Hypothesis fuzzing:** strings mixing Devanagari, Latin, digits, joiners, whitespace, emoji and stray marks. Checks:
   - round-trip
   - equality between the two exports
   - integrity on well-formed input

**Acceptance:** all of the above pass, and `plain_vs_hf.md` is written.

**STOP (human):** choose the M9 tier (see M9: reduced, full or none) before continuing.

---

### M7 — Mix sweep and ablation ladder (target: 1 Dec)

1. Train the final config at 64k on the 50%, 60% and 70% snapshots.
2. Export each, run the guarantee tests, and run the harness:
   `uv run python scripts/eval_panel.py run --only none --add anc64_n50=<dir>/tokenizer.json --add anc64_n60=... --add anc64_n70=...`
3. Apply the 3.8 pick rule and record the result in `docs/RESULTS.md`.
4. **Ladder** on the chosen mix, at 64k, with everything else identical:

| Rung | block0 | block1 | bans | gb9c |
|---|---|---|---|---|
| plain | off | off | off | — |
| b0 | on | off | on | — |
| b0b1 (no bans) | on | on | off | off |
| final | on | on | on | off |
| gb9c | on | on | on | on |

   Run the harness and our metrics on every rung. Report fertility, bytes per token, orphan share, violations and tail per rung in `docs/RESULTS.md`.

**STOP (human):** if no mix meets all three fertility rows of 3.10 (Nepali below 1.445, English below 1.420, Hindi below 1.385), report all three mixes and propose 75% Nepali, or a cap-bound covered set.

---

### M8 — Final builds, Gate A, efficiency (target: 8 Dec)

1. Build 32k, 64k and 128k on the chosen mix, and export each.
2. Run the harness on all of them plus the panel. Add our metrics:
   - `akshara.py`: violations, tail, and matra-detach, virama-detach and conjunct-split rates
   - `equality.py`
   - gold agreement (M3 step 6g)
   - encoding-variant cost (3.9), reported only
3. **`efficiency.py`**, on the same Mac, best of 3, for ours, Yak v1 64k, sidskarki-64k and o200k:
   - encode MB/s, single-thread and batched, on 200 MB of mixed text
   - peak RSS while encoding
   - tokenizer load time
   - embedding parameters (vocab × 1,024 and vocab × 4,096)
   - token counts on FLORES+ npi and Aya, and the sequence-length reduction against each

   Label derived attention and KV-cache savings as **estimates from token counts**: full-attention cost scales with the square of length, and KV-cache with length.
4. Loading: `AutoTokenizer.from_pretrained(<dir>)` must work. vLLM loading is marked "pending (GPU box)".
5. `make gate-a` writes `docs/GATE_A.md`: every 3.10 row PASS/FAIL, the stretch, 32k and 128k rows, the Yak scoreboard and the separate phrase-build table.

**STOP (human):** Gate A review. M9 needs approval.

---

### M9 — Proxy pretraining and Gate B (target: 18 Dec, on a rented Linux GPU box)

This is the only milestone that does not run on the Mac.

**STOP (human):** provide SSH access to a Linux machine with H100 (or A100 80GB) GPUs, and confirm the tier in `configs/proxy.yaml`.

**Arms** (tokenizer sources come from M1's records):

1. Ancoda 64k final
2. ladder `plain` (same data, pre-tokenizer and size; no blocks or bans)
3. ladder `b0` (no Block 1)
4. `gb9c`
5. Yak v1 64k (`ampixa/yak-tokenizer`, `subfolder="v1.0/64k"`; research use only)
6. sidskarki-64k (outside tokenizer; only if M1 recorded that its license allows this)
7. Arkios (`sajalregmi4/arkios-tokenizer`)

**Tiers:**

- **reduced:** arms 1, 2 and 5, seed 1 only.
- **full:** all 7 arms, seeds 1 and 2, plus the equal-bytes control (arm 1 against arm 5).
- **none:** skip M9 entirely. Gate B is recorded as "not run", and every model-level claim is dropped (3.11).

The budget is set **after the pilot** (step 3). Planning figures only: Arkios's comparable runs took about 5.3 H100-hours per 4B tokens. That suggests roughly 16 H100-hours for reduced and roughly 80 for full (15 runs).

**Setup on the GPU box:**

- In Cursor, use Remote-SSH and clone the repo.
- `rsync` across only:
  - `artifacts/tokenizers/`
  - the cleaned training parquet for fw2_npi, iriis, sangraha_nep, indiccorp_npi and fwedu_en
  - the eval sets
- Verify sha256 against the manifests.
- Run `nvidia-smi`. Install the matching CUDA build of PyTorch with the exact `pip` command from pytorch.org's "Get Started" selector, run through `uv pip`. Then `uv add accelerate`.
- Verify with `uv run python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"`.

**Steps:**

1. **Data (`prepare.py`):**
   - One fixed document stream, 50% Nepali and 50% English by bytes, from the cleaned training sources only, shuffled once with seed 7.
   - Every arm tokenizes the same stream in the same order into uint32 memmaps. SentencePiece arms (sidskarki-64k) use the `sentencepiece` library with `.model` files; tokenizer.json arms use `tokenizers`.
   - **Separator policy:** exactly one separator ID before every document. Use the arm's BOS if it has one; otherwise its end-of-text token; otherwise append one new ID after its vocabulary. Never substitute silently; record the ID and rule per arm in `docs/GATE_B.md`.
   - Record the bytes each arm consumed.
2. **Model and training** (identical except vocab size):
   - **Model:** `LlamaConfig` with hidden 1,024, intermediate 2,816, 16 layers, 16 heads, 16 KV heads, max position 2,048, rope theta 10,000, RMSNorm eps 1e-5, tied embeddings.
   - **Vocab:** padded to a multiple of 128. Record parameter counts.
   - **Run length:** sequence 2,048, global batch 256, **7,630 steps = 4,000,317,440 tokens** (at least 4.0B). Report the exact count.
   - **Optimizer:** AdamW (0.9, 0.95), weight decay 0.1, peak LR 6e-4, 300 warmup steps, cosine to 6e-5, grad clip 1.0.
   - **Precision:** bf16, SDPA attention, `torch.compile` if stable.
   - **Seed** changes initialization only; data order is fixed. Log loss every 50 steps.
3. **Pilot:**
   - Run 200 steps of arm 1 and arm 5 with the exact production config, including checkpointing.
   - Measure tokens per second and project GPU-hours for the chosen tier.
   - **STOP (human):** confirm the budget against the projection before full runs.
4. **Equal-bytes control (full tier only):** whichever of arms 1 and 5 consumed fewer bytes at 4.0B tokens gets one extra seed-1 run, trained until it has consumed the other arm's bytes.
5. **bits per byte (`bpb.py`):**
   - **Eval sets:**
     - fw2_npi_test (in-distribution)
     - aya_npi (out of distribution; prompt + `\n` + answer as one document)
     - fw2_hin_test
     - c4_val
   - Each eval set is the first 2,000 documents or 5 MB, whichever comes first. Freeze and hash the lists.
   - **Per document:** ids = [separator] + encode(text).
   - **Scoring:** NLL (nats) is summed over **text tokens only**. The separator is context and is never scored.
   - **Windows:** length 2,048, stride 1,024. The first window scores all its text tokens; each later window scores only positions not scored before. Every text token is scored exactly once.
   - **bpb** = total NLL ÷ (ln 2 × UTF-8 bytes of the texts). Never compare per-token perplexity across arms.
   - Report each arm's exact round-trip rate on these texts. Any arm below 100% is flagged in the table.
6. **Gate B (`docs/GATE_B.md`):** bpb per arm and eval set, giving the mean, the seed spread and the equal-bytes row.
   - **Noise floor:**
     - full tier: the larger seed spread of the two arms compared
     - reduced tier: 1.0% relative, fixed now (Arkios judged a 0.65% single-seed gap too thin)
   - **Pass:** arm 1 has the lowest Nepali bpb on fw2_npi_test, by more than the noise floor over every other arm. It is not worse on aya_npi beyond the noise floor. Its English bpb is within 1% of the best arm.
   - **Arm 2 (plain) better than arm 1 beyond noise:** recommend shipping plain BPE. Within noise: keep arm 1, since the guarantee costs nothing.
   - **Arm 3 better beyond noise:** recommend dropping Block 1.
   - **Arm 4 better beyond noise:** recommend GB9c units.
   - State the result against arms 5–7 separately, using 3.11 wording.

**STOP (human):** Gate B review.

---

### M10 — Release candidate and Gate C (target: 22 Dec)

1. **Vocab audit:** decode every token. Write Devanagari tokens of 6 or more characters to `artifacts/audit/long_tokens.tsv` for human review of personal names.
2. **Model card:**
   - all Gate A and Gate B numbers, with 3.11 wording
   - the layout, special tokens, pattern and canonical encoder
   - the efficiency report
   - sources, licenses and the snapshot hash
   - **limits:** the covered-set guarantee only; minority languages, Romanized Nepali and phrase-level tokens out of scope; llama.cpp pending
3. **Package:** `tokenizer.json`, `tokenizer_config.json`, `special_tokens_map.json`, `tiktoken/`, `manifest.json`, and `LICENSE`. The license is Apache-2.0, **pending human and legal sign-off** against the `docs/LICENSES.md` release-notes column.
4. `make release` pushes to a **private** Hugging Face repo.

**STOP (human):** license sign-off and audit review before anything goes public.

**Acceptance:** `docs/GATE_C.md` lists every item checked.

---

## 6. Progress file format (`docs/PROGRESS.md`)

```
## M<n> — <name> — <not started | in progress | done | blocked>
- [x] task …
- [ ] task …
Acceptance:
- <command> → <one-line result>
Needs the human:
- …
```

## 7. Never

- Never invent a dataset ID, config, file name, version or license.
- Never train a tokenizer before decontamination and its audit have passed.
- Never publish, push to a public repo, or delete data without the human's explicit go-ahead.
- Never weaken a test or a pass bar to make it pass.
- Never upgrade a pinned dependency on your own.

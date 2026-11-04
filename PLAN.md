# Ancoda Nepali Tokenizer — Build Plan for the Cursor Agent (v2.6, 4 Nov 2026)

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
> **What changed in v2.6 (4 Nov 2026): after a third external review (targeted fixes, no redesign)**
> - **Segmentation rule made exact (3.3):** start from Unicode 16.0.0 grapheme clusters, then apply one override: **every Devanagari letter begins a new unit.** This is described as an override, not as "undoing GB9c". It covers every virama pattern (including Extend characters between virama and consonant) and Prepend characters.
> - **Eligibility for protection (3.6):** only units that start with a Devanagari letter and contain only Block 0 characters can be covered. This is what makes the guarantee provable.
> - **Written proof of the guarantee (3.5),** with the conditions it needs, and exhaustive tests per covered unit (M6). The `gb9c=true` ablation has no guarantee; its violations are measured.
> - **Block 1 coverage formula** written out exactly (3.6).
> - **Joiner copy accounting:** 1% of the snapshot, taken out of Nepali's 50%, so totals stay 100%.
> - **Morphology metric:** a whole-word token no longer counts as a matched boundary; whole-word preservation is reported separately.
> - **Gate B fairness:** fixed-token and fixed-byte comparisons both run (for arms 1, 2 and 5 in the full tier); bytes, tokens, compute and documents are reported for every run; 3 seeds for arms 1 and 2; two-seed results are labelled preliminary.
> - **Tolerances clarified (M1):** ±0.03 is the harness's own check against papers; ±0.003 is our reproduction of Yak's table. M1 prints published, reproduced, absolute and relative difference.
> - **Equality corpus** gains random-Unicode and adversarial strings and is a release condition; phrase-pattern tests get their own corpus; the Unigram control's limits are stated.
>
> **What changed in v2.5 (4 Nov 2026): after a second external review; no annotators**
> - The protected unit is defined by an explicit boundary table (3.3), and the plan says plainly that it is not the conventional orthographic akshara: conjuncts are split.
> - The mechanism is stated (3.4): byte-level BPE whose rank order protects units; there are no special atoms outside the merge table. Integrity is tested separately on each export, not only through their equality.
> - No human annotators. The 500-word human gold set is replaced by automated checks: Unicode's test file, a second, independent implementation of 3.3 written in Python, and ICU where it installs (M3 step 6).
> - Coverage is reported in four separate rows: byte-level representability, whole-character coverage of U+0900–U+097F, coverage of the extended Devanagari blocks, and covered-unit integrity.
> - Morphology alignment becomes precision, recall and F1, also frequency-weighted and on unseen words (3.9).
> - Phrase builds report whitespace-crossing share, superword frequency and unused vocabulary (3.13).
> - Data quality is audited automatically per language: language-ID confidence, script contamination, bytes before and after cleaning. Nothing is ever upsampled (3.8, M4).
> - A tokenizer-level Unigram control is added to the M7 ladder, so BPE is not assumed to win.
>
> **What changed in v2.4 (4 Nov 2026): best Devanagari tokenizer overall, three builds**
> - Yak's numbers are now verified at the source (its GitHub README and blog, read 4 Nov 2026). Yak v2 128k's languages-of-Nepal mean is 1.629 (v2 64k: 1.843). Those 14 test sets are private and cannot be rerun, so that row is reported as "not comparable", never claimed.
> - Three builds ship: **Ancoda-64k** (flagship), **Ancoda-128k**, and **Ancoda-Phrase-128k**, whose tokens may span up to 4 words, like Yak's phrase build (3.13, M8b).
> - The training mix now covers every Devanagari language with open data: Nepali, Hindi, Marathi and Sanskrit, plus a pooled share for the smaller ones (Maithili, Bhojpuri, Awadhi, Magahi, Konkani, Bodo, Dogri, Doteli, Chhattisgarhi, Angika) and a languages-of-Nepal pool where licensed data exists (3.8).
> - Joiner robustness: a small synthetic copy of Nepali training text with ZWJ or ZWNJ inserted after viramas (3.8). It is kept only if the M7 ladder shows it helps. Target: beat Yak v2's joiner probes.
> - New reported metrics: fertility on every FLORES+ Devanagari language, morphology alignment against Papaya's open Nepali affix list, and cross-language token reuse (3.9).
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

**The goal** is the best Devanagari tokenizer overall: Nepali first, then Hindi, English, Marathi, Sanskrit and the other Devanagari languages. Three builds ship: Ancoda-64k (flagship), Ancoda-128k and Ancoda-Phrase-128k (3.13).

**The main competitor is Yak** (Ampixa). Everything is built from zero in this repo. No earlier internal code, data or results are used. Yak, sidskarki-64k, HimalayaAI and Arkios are outside tokenizers, used only for comparison.

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
- Synthetic joiner-augmented text (PLAN.md 3.8) is generated only from training documents, tagged `synthetic` in every manifest, and stored separately; original text is never modified.
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

**Definition.** In this project an "akshara" is the **Ancoda protected unit**: a UAX #29 extended grapheme cluster (Unicode 16.0.0), then split by one override: **every Devanagari letter begins a new unit.** A Devanagari letter is a code point with General_Category Lo and Script Devanagari (consonants, nukta letters, independent vowels, avagraha, om). This is an explicit Ancoda override, not an exact inverse of rule GB9c, and is described that way everywhere. Each consonant keeps its vowel signs, virama, nukta, anusvara, chandrabindu, visarga and any joiner after a virama. **This is not the conventional orthographic akshara:** a conjunct such as क्ष is two units, क् and ष, which BPE may merge later. The paper must say so wherever it uses the word.

**Boundary table (default mode; `|` marks a unit boundary):**

| Sequence | Units | Rule |
|---|---|---|
| क + ् + ष (क्ष) | क् \| ष | split after a virama that precedes a consonant |
| क + ् + ZWJ + ष | क्+ZWJ \| ष | the joiner stays with the virama before it |
| क + ् + ZWNJ + ष | क्+ZWNJ \| ष | same |
| र + ् + ZWJ + य (eyelash ra) | र्+ZWJ \| य | same |
| र + ZWJ + ् (older eyelash order) | one unit (ZWJ and virama are Extend); no special case | recorded in `akshara_cases.tsv` |
| क + ् + nukta + ष (Extend between virama and letter) | क्+nukta \| ष | the override splits before every letter |
| Prepend character + Devanagari letter | Prepend \| letter | override |
| consonant + nukta + vowel sign | one unit | marks extend the base |
| U+0958 (precomposed क़) vs क + U+093C | one unit each; byte strings stay different | no normalization |
| consonant + vowel sign + anusvara / chandrabindu / visarga | one unit | marks extend the base |
| independent vowel (+ marks) | one unit | |
| vowel sign or virama with no consonant before it | at text start: a unit on its own; after a space, digit or other character: attached to that character (UAX #29 rule GB9) | malformed; counted in the tail, never "fixed" |
| double virama | one unit up to the next letter (both viramas are Extend) | recorded in `akshara_cases.tsv` |
| Devanagari letter followed by a non-Devanagari mark (for example U+0301 or a Vedic sign) | one unit, but **not eligible** for protection (3.6) | |
| digits, danda, double danda | one unit each | |

**Checked on 4 Nov 2026 with `unicode-segmentation` 1.12.0 (Unicode 16.0.0):** the only Devanagari-block characters that join a letter's cluster are marks (Mn/Mc) and ZWJ/ZWNJ; no Devanagari-block character acts as Prepend; क्ष, क्+ZWJ+ष, क्+nukta+ष and क्+्+ष are each one raw cluster (so the override is needed for all of them); क्+ZWNJ+ष is already two raw clusters; र+ZWJ+्+य is one raw cluster. M3 repeats these as unit tests.

**Implementation.**

1. Use the Rust crate `unicode-segmentation = "=1.12.0"` (Unicode 16.0.0).
   - This is the first release that implements GB9c.
   - 1.11.0 has Unicode 15.1 data but no GB9c code. 1.13.0 and 1.13.1 were yanked.
   - Record `unicode_segmentation::UNICODE_VERSION` in every manifest.
2. Compute extended grapheme clusters.
3. **Override:** inside every cluster, insert a boundary before each Devanagari letter that is not the cluster's first code point. Everything between the virama and that letter (ZWJ, ZWNJ, nukta or any other Extend character) stays with the unit on the left. This removes every conjunct join GB9c makes, and any join made by a Prepend character before a Devanagari letter.
4. The Devanagari letter set = `General_Category=Lo ∩ Script=Devanagari`, read from Unicode **16.0.0** `UnicodeData.txt` and `Scripts.txt`. Never hand-code it.
5. **Mode `gb9c=true`:** return the raw clusters from step 2 (conjuncts joined). This is used only for the M7 and M9 ablation. **The 3.5 guarantee does not hold in this mode** (a unit can hold two letters); its violations are measured and reported, never gated.

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

**Mechanism (stated so it is never confused):** the vocabulary is an ordinary byte-level BPE rank table. There are no special atom symbols outside it. Every input, valid or not, encodes from bytes, so nothing ever fails to encode and decoding is byte-exact. A covered unit stays whole only because its own merges (Blocks 0 and 1) all rank below every Block 2 merge (3.5). Both exports carry this rank table unchanged, and M6 tests integrity on each export separately.

### 3.5 Canonical encoder and exports

**Canonical algorithm:** tiktoken's "lowest-rank concatenation" BPE.

- For each pre-token: if the whole byte string is a token, emit it.
- Otherwise start from single bytes. Repeatedly merge the adjacent pair whose concatenation is a token with the lowest rank (leftmost on ties). Stop when no adjacent concatenation is a token.

**The guarantee and its proof.** Claim: in default mode, no token boundary ever falls strictly inside a covered unit U, in either export.

Conditions it relies on (all enforced elsewhere in this plan):
- C1. U starts with a Devanagari letter and every code point of U is a Block 0 character (3.6 eligibility).
- C2. Every Devanagari letter begins a unit (3.3), and every later code point of U has Grapheme_Cluster_Break Extend or SpacingMark. So an occurrence of U's first letter always starts a unit, and no unit boundary can fall inside U.
- C3. All of U's code points are in the pre-tokenizer's letter classes (Devanagari letters, \p{M}, ZWJ, ZWNJ), so U never spans two pre-tokens.
- C4. Ranks: bytes < Block 0 < Block 1 < Block 2 (3.4), and the encoder is lowest-rank concatenation (above).
- C5. The Block 2 ban 1 forbids any token that joins part of one character to anything outside it, or starts at a continuation byte.

Proof:
1. While U is not a single symbol, it contains an adjacent pair whose concatenation is a Block 0 or Block 1 token: each character's bytes complete through Block 0 (lead byte + second byte = prefix token, + last byte = character), and U's characters then complete through its Block 1 chain `c1+c2`, `c1c2+c3`, … (3.6 adds every chain prefix).
2. Every Block 0 and Block 1 token ranks below every Block 2 token (C4), so no Block 2 merge happens anywhere in the pre-token until no Block 0 or Block 1 merge remains, so not before U is one symbol.
3. No Block 0 or Block 1 merge can cross U's edges. A Block 0 token lies inside one character. A Block 1 token starts with a Devanagari letter followed only by Extend or SpacingMark characters, so by C2 every occurrence of it lies inside one unit.
4. No other merge can happen inside U first. A Block 1 token cannot start at U's second or later character, because those are marks. Pieces of a character cannot join other material (C5).
5. Once U is one symbol, later merges only join whole symbols, so U stays whole inside any larger token.

Both exports reproduce this encoder exactly (tested), so the claim holds for tokenizer.json and tiktoken alike. **The M6 tests check it independently on each export**; if a test ever fails, the proof's conditions are wrong and the agent **STOPS**.

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

1. Count every unit containing at least one Devanagari code point over the training snapshot.
   - `N` = occurrences of all such units, eligible or not.
   - **Eligible** units: start with a Devanagari letter, and every code point is a Block 0 character (U+0900–U+097F, U+200C, U+200D).
   - `N0` = occurrences of single-code-point units whose code point is in Block 0 (already covered).
2. Sort eligible units of 2 or more code points by count descending, ties broken by UTF-8 bytes ascending.
3. Walk the list. Each akshara's chain is `c1+c2`, then `c1c2+c3`, and so on. Its **incremental cost** = the chain prefixes not already selected.
4. Add the akshara if `selected merges + cost ≤ cap`. Otherwise **stop**; never skip ahead.
5. Also stop when coverage reaches **99.99%**. **Coverage = (N0 + occurrences of selected Block 1 units) ÷ N.** Ineligible units count in N and can never be covered, so 99.99% may be unreachable; then the cap decides.
6. **Caps** (5% of vocab): 1,638 at 32,768; 3,276 at 65,536; 6,553 at 131,072.
7. The coverage target never overrides the cap. If the cap binds at, say, 99.4%, report 99.4%.
8. Record:
   - N, N0, the ineligible occurrences, the unit types selected, the Block 1 merge count and the coverage achieved
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

| Bucket | Share | Sources (M2) |
|---|---|---|
| Nepali | 50% | fw2_npi, iriis, sangraha_nep, indiccorp_npi, hplt_npi |
| English | 20% | fwedu_en |
| Hindi | 10% | fw2_hin, sangraha_hin |
| Marathi | 3% | fw2_mar, sangraha_mar |
| Sanskrit | 3% | fw2_san, sangraha_san |
| Smaller Devanagari languages (pool) | 4% | Maithili, Bhojpuri, Awadhi, Magahi, Konkani, Bodo, Dogri, Doteli, Chhattisgarhi, Angika: every `*_Deva` config that exists in FineWeb-2, Sangraha verified and HPLT v2 (M2) |
| Languages of Nepal (pool) | 2% | only sources found and licence-checked in M2 (for example Newar, Tharu, Tamang, Magar, Gurung, Limbu). If none are found, this share goes to Nepali |
| Code | 5% | code |
| Math | 3% | owm |

- **This is a language-balanced training mixture, not a vocabulary budget:** shares set how much text each language contributes; BPE then decides which merges each language gets.
- **Inside each pool,** sample languages with temperature τ = 0.3 on their available bytes: share ∝ bytes^0.3. A language with less data than its share contributes everything it has; the remainder is redistributed inside the pool. **No document is ever repeated (no upsampling).** A language's support is claimed only from its own measured results.
- **Joiner augmentation (on by default, removed if the M7 ladder says so):** 1% of the snapshot's bytes, **taken out of Nepali's share** (so Nepali is 49% natural + 1% synthetic, and the total stays 100%), made from Nepali training documents not otherwise sampled. In the ladder's no-joiner snapshot that 1% is natural Nepali. Half the documents get U+200D (ZWJ) after every virama that precedes a consonant; the other half get U+200C (ZWNJ) at the same sites. Exactly the sites Yak's probe uses (3.9). Tagged `synthetic`.
- **Size:** 10 GB total. If the Mac has less than 32 GB of RAM, use 5 GB with the same shares. Never mix sizes within a sweep.
- **Sweep (M7):** Nepali at 45%, 50% and 60%, others scaled pro rata. **Pick rule:** the point that meets every Gate A fertility row with the lowest Nepali fertility; on a tie, the lower English fertility.
- **Excluded from training:**
  - Wikipedia: its CC BY-SA share-alike license complicates releasing the vocabulary, and it is about 0.5% of public Nepali text.
  - All evaluation sets, including FLORES+ dev and devtest for every language.
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
| Segmenter agreement | share of words on which `aksharas()` equals the independent Python reference (M3 step 6) on 1M distinct training-snapshot words plus all eval words |
| Byte-level representability | share of eval segments that encode and decode byte-exactly (always 100% by construction; reported, not claimed as coverage) |
| Whole-character coverage | of the 128 code points U+0900–U+097F, how many are single tokens |
| Extended-block coverage (reported) | for Devanagari Extended (U+A8E0–U+A8FF), Vedic Extensions (U+1CD0–U+1CFF) and Devanagari Extended-A (U+11B00–U+11B5F): assigned code points that are single tokens, and their frequency in the training snapshot |
| Fertility per FLORES+ Devanagari language | whole-file fertility (Yak's definition) on the FLORES+ devtest of every `*_Deva` language FLORES+ has, for ours and every panel tokenizer; also the unweighted mean |
| Morphology alignment (reported) | on a fixed sample of 20,000 Nepali words from fw2_npi_test. Reference boundaries: suffix boundaries found by longest-match against Papaya's affix inventory (a reference list, not ground truth). **Recall:** reference boundaries that are token boundaries (a word kept as one token matches none of its reference boundaries). **Precision:** word-internal token boundaries that are reference boundaries; words with no internal boundary contribute nothing. **F1.** Each unweighted and frequency-weighted, and separately on words absent from the training snapshot |
| Whole-word preservation (reported) | share of the same 20,000 words encoded as a single token (with its leading space), unweighted and frequency-weighted. Reported beside morphology F1, never combined with it |
| Cross-language reuse (reported) | share of non-special tokens used at least once in 3 or more FLORES+ Devanagari languages; and median frequency of used tokens on FLORES+ npi |
| Encoding-variant cost (reported, not gated) | on FLORES+ npi and Aya, tokens of a rewritten copy ÷ tokens of the original, for three rewrites: every nukta letter swapped between its precomposed (U+0958–U+095F) and decomposed form; every eyelash ra written in the older order (र, ZWJ, virama) instead of र, virama, ZWJ; and the whole text NFC-normalized. A value near 1.00 means variant spellings cost almost nothing. |

### 3.10 Gate A targets (65,536 build, FLORES+ devtest via Yak's harness, unless noted)

**What "beat Yak" means here.** Better than Yak v2 64k (the newest Yak build on its 3 Nov 2026 leaderboard) on every word-level row below, and a lower Nepali count than every 64k tokenizer on Yak's leaderboard. References are from Yak's public `docs/RESULTS.md` (generated 3 Nov 2026) and its blog, both read 4 Nov 2026. M1 re-measures them; the M1 values are the ones the gate uses.

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
| Whole-character coverage, U+0900–U+097F | 128 of 128 | Yak 128; sidskarki-64k 102 |
| Covered-unit integrity, checked on the tokenizer.json and the tiktoken outputs separately | 0 violations in each | — |
| Nepali byte pieces | ≤ 0.05% | Yak v2 0.02%; Yak v1 0.07% |
| Newline gap | ≤ 0.010 | Yak 0.000; sidskarki-64k 0.075 |
| Hugging Face IDs == tiktoken IDs | 100% of eval texts plus 1M lines | — |
| ZWJ and ZWNJ probes | below Yak v2 64k's exact values measured in M1 (reported as 1.504 and 1.691). If missed, **STOP (human)** with the numbers | Yak v1 1.661 and 1.965; sidskarki-64k 2.560 and 2.560 |
| Segmenter agreement (M3 step 6) | 100% | — |

- **Stretch (beat both Yak 64k builds):** English below 1.388 and Hindi below 1.376.
- **Yak scoreboard** (in `docs/GATE_A.md`): for every metric the harness reports for `yak-64k-v1` and `yak-64k-v2.0`, print ours, both Yak values and WIN, TIE or LOSE. This includes everyday Nepali on Aya (Yak v1 1.446, Yak v2 1.455) and the mean over the languages of Nepal (Yak v2 1.843). No row is hidden or dropped.
- **Also on the scoreboard (reported, not gated):** Sanskrit (Yak 2.126, IndicBERT v2 2.067 per Yak's blog); fertility on every FLORES+ Devanagari language and their mean, measured by us for ours and every panel tokenizer.
- **Languages of Nepal (Yak v2 128k 1.629, v2 64k 1.843):** Yak's 14 held-out sets are private and cannot be rerun. Report this row as "not comparable". Never claim a win or loss on it.
- **Phrase builds are a separate category,** gated by Gate P (3.13). Word-level builds are never compared against phrase builds.
- **32k and 128k:** Nepali below Yak v1 at the same size (1.586 and 1.370), with English no more than 2% above Yak's.
- **Always reported, never gated:** Romanized Nepali word-list fertility. Romanized Nepali is out of scope for v1.

### 3.11 Claims discipline (for the paper and model card)

- **Tokenizer-level claims** ("fewer tokens", "never splits a covered akshara") require Gate A, or Gate P for the phrase build.
- **Model-level claims** ("trains a better model") require Gate B. A reduced-tier (single-seed) result is labeled "single seed, not significance-tested".
- **"Beats Yak"** may be written only for scoreboard rows marked WIN, naming the Yak build and size (for example "fewer Nepali tokens than Yak v2 64k").

### 3.12 Slip rules

If the schedule slips, cut in this order:

1. The 128k build.
2. The 32k build.
3. Ancoda-Phrase-128k (keep the 64k phrase result from the M8b sweep as the phrase entry).
4. The full GPU tier, falling back to reduced.
5. The model-level conjunct ablation arm; it stays at the tokenizer level.
6. The efficiency extras.

**Never cut:** the guarantee tests, decontamination, the equality test, segmenter agreement or the core Gate A rows.

### 3.13 Phrase build (Ancoda-Phrase)

A two-stage curriculum, after SuperBPE (arXiv 2503.13423): word-level merges first, then merges that cross spaces.

1. **Stage 1:** exactly the word-level pipeline (3.1–3.7) on the chosen mix, stopped early so that `t` Block 2 IDs remain.
2. **Stage 2:** continue learning merges with the **phrase pattern**, built in code from `configs/pretokenizer_pattern.txt` (never retyped): its first two branches (the letter branches) are wrapped as one group repeated `{1,4}`, so up to 4 consecutive words become one pre-token. All other branches (digits, punctuation, newlines, whitespace runs) are unchanged, so numbers and line breaks still split. Assert that on the 3.1 examples, splitting each phrase pre-token with the 3.1 pattern gives back exactly the 3.1 pre-tokens.
3. **Stage-2 bans,** on top of all 3.4 bans: a token holds at most 4 words; it never ends in a space; it never contains a newline, digit or punctuation character.
4. **Ranks:** stage-2 merges rank above every stage-1 merge, so the 3.5 guarantee and both exports carry over unchanged. Only the exported pre-tokenizer pattern changes, to the phrase pattern.
5. **Phrase metrics (reported, M8b):** tokens per word, bytes per token, share of tokens that cross a space, the frequency distribution of stage-2 tokens, and the share of vocabulary entries used fewer than 10 times on a held-out 100 MB slice.
6. **Word-level consistency (reported):** on FLORES+ npi, the share of phrase-build tokens that are either stage-2 tokens or identical to the stage-1 build's tokens over the same byte span. Covered-akshara violations must still be 0.
7. **Sweep (M8b), at 64k:** `t` = 10%, 25% and 50% of Block 2. **Pick rule:** the lowest Nepali fertility whose English fertility is within 2% of the 64k word-level build's English fertility. Then build 128k with the chosen fraction.

**Gate P (phrase builds, FLORES+ devtest via Yak's harness):**

| Check | Pass bar | Reference |
|---|---|---|
| Nepali, Phrase-128k | below 1.267 | Yak v1 128k-phrase 1.267 |
| English, Phrase-128k | below 1.232 | Yak v1 128k-phrase 1.232 |
| Hindi, Phrase-128k | below 1.247 | Yak v1 128k-phrase 1.247 |
| Nepali, Phrase-64k (sweep pick) | below 1.374 | Yak 64k-phrase 1.374 |
| All Gate A integrity rows (violations, mark-initial, round-trip, coverage, equality) | as Gate A | — |

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
    build_{32k,64k,128k}.yaml  phrase_{64k,128k}.yaml
    ladder_{plain,b0,b0b1,final}.yaml  ablation_gb9c.yaml
    pretokenizer_pattern.txt         # generated in M3
  crates/
    ancoda-seg/                      # pre-tokenizer, akshara segmenter, counting; PyO3 via maturin
    ancoda-bpe/                      # trainer CLI: Blocks 0/1/2, bans, canonical encoder
  python/ancoda_tok/
    pattern.py  download.py  storage.py  mix.py  inventory.py
    clean/{utf8.py,lid.py,legacy_font.py,dedup.py,decontam.py,augment_joiners.py,snapshot.py}
    export_hf.py  export_tiktoken.py
    eval/{akshara.py,ref_segmenter.py,equality.py,efficiency.py,multilang.py,morph.py,gate_a.py,gate_p.py}
    proxy/{prepare.py,train.py,bpb.py,gate_b.py}
  tests/
    data/{akshara_cases.tsv,pretok_adversarial.tsv}
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
2. Confirm the harness's own validity gate passes **5 of 5**. That gate is the harness's check against numbers published in papers, at its built-in tolerance of ±0.03; it is separate from the ±0.003 reproduction of Yak's table in the acceptance below.
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

**Acceptance:** the gate shows 5/5. Write `artifacts/eval/m1_reproduction.tsv` with, for every row below: published value, reproduced value, absolute difference, relative difference, PASS/FAIL. Each value must match within ±0.003. **STOP (human)** on any FAIL; never continue with an unexplained difference.

| Panel name | Nepali | English | Hindi |
|---|---|---|---|
| `yak-64k-v1` | 1.457 | 1.388 | 1.376 |
| `yak-64k-v2.0` | 1.470 | 1.420 | 1.385 |
| `sidskarki-64k` | 1.445 | 2.409 | 1.656 |
| Yak v1 128k | 1.370 | 1.291 | 1.290 |
| Yak v2 128k | 1.382 | 1.313 | 1.298 |
| Yak v1 128k-phrase | 1.267 | 1.232 | 1.247 |

Also record from the harness output: Yak v1 64k-phrase Nepali (reported 1.374), Sanskrit for Yak and IndicBERT v2 (reported 2.126 and 2.067), and which 5 languages-of-Nepal rows the harness can rerun on FLORES+. The 14 held-out languages cannot be rerun; note this in `docs/RESULTS.md`.

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
| fwedu_en | `load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True)` | train | [3 GB] |
| hplt_npi | HPLT v2 cleaned, Nepali (`npi_Deva`); find the official HPLT v2 repo and config via the API | train | all |
| fw2_hin | fineweb-2 `hin_Deva`, train, streaming | train | [1.5 GB] |
| sangraha_hin | sangraha `"verified"`, split for Hindi | train | [0.5 GB] |
| fw2_mar, sangraha_mar | fineweb-2 `mar_Deva` (streaming); sangraha verified Marathi split | train | [0.5 GB] each |
| fw2_hin_test | fineweb-2 `hin_Deva`, `split="test"` | **eval only** | all |
| fw2_san, sangraha_san | fineweb-2 `san_Deva`; sangraha verified `san` | train | [0.5 GB] each |
| deva_pool | every other `*_Deva` config in fineweb-2, every other Devanagari-language split in sangraha verified, and the matching HPLT v2 configs (expected among them: mai, bho, awa, mag, gom, brx, doi, dty, hne, anp). **List the real configs with the API first;** never guess a name | train | all, up to [0.3 GB] per language |
| nepal_pool | openly licensed text for languages of Nepal not covered above (for example Newar, Tharu, Tamang, Magar, Gurung, Limbu), searched in fineweb-2, HPLT v2 and Hugging Face datasets. Include only sources whose card states a license that permits training; record each in `docs/LICENSES.md`. **STOP (human)** with the list before downloading | train | all |
| owm | `load_dataset("open-web-math/open-web-math", split="train", streaming=True)` | train | [1 GB] |
| code | `codeparrot/github-code-clean`. Its loader is a Python script (`github-code-clean.py`), which current `datasets` may refuse, so **read the files in its `data/` folder directly**. First list them with `HfApi().list_repo_files(..., repo_type="dataset")` and open one file to confirm its format and its license and language columns. Then stream files, keeping only mit, apache-2.0, bsd-2-clause and bsd-3-clause rows. **If the files are not readable this way, or the columns differ, STOP and report what you found.** | train | [1 GB] |
| c4_val | `load_dataset("allenai/c4", "en", split="validation", streaming=True)` | **eval only** | [50 MB] |
| belebele_npi | `load_dataset("facebook/belebele", "npi_Deva")` | **decontam only** | all |
| aya_npi | `CohereLabs/aya_dataset`, rows whose language is Nepali | **eval and decontam** | all |
| glotlid | `hf_hub_download("cis-lmu/glotlid", "model.bin")`; confirm v3 on the model card | tool | 1 file |
| papaya_affixes | the affix inventory file in GitHub `kalexrt/nepali-morphology-tokenizer`; confirm its MIT license and record the commit | eval only (3.9) | 1 file |

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
   - Default mode equals `gb9c=true` everywhere, except at boundaries the 3.3 override inserts before Devanagari letters.
   - Override cases: virama with Extend characters between it and the letter; ZWJ; ZWNJ; several viramas; malformed sequences; nukta; vowel signs; Prepend characters; and Devanagari next to Bengali, Gurmukhi and Latin text.
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
6. **Independent segmenter checks (no human annotators).**
   - **a. Second implementation.** Write `python/ancoda_tok/eval/ref_segmenter.py`, a pure-Python implementation of 3.3 written from the Unicode 16.0.0 property files (`GraphemeBreakProperty`, `DerivedCoreProperties.txt` InCB, `emoji-data` as UAX #29 requires) and the 3.3 boundary table. It must not import or call the Rust code. Download any extra Unicode 16.0.0 file it needs into `third_party/unicode/16.0.0/` and record its sha256.
   - **b. Unicode test file.** The Python reference in `gb9c=true` mode passes 100% of `GraphemeBreakTest.txt`, like the Rust code.
   - **c. Differential test.** Compare `aksharas()` with the reference on 1M distinct words sampled from the training snapshot (seed 20261104) plus every word in the eval sets, in both modes. Every disagreement is a bug in one of them: fix it, add the word to `akshara_cases.tsv`, and rerun. **Pass:** 0 disagreements.
   - **d. ICU, if available.** Try `brew install icu4c` and `uv add PyICU`. If both install, compare `gb9c=true` mode with ICU's character break iterator on the same words and report agreement and ICU's version. If they do not install, record that and continue; this check is optional.

   M3 may close once a–c pass.

**Acceptance:** `cargo test -p ancoda-seg` and `uv run pytest tests/test_seg.py tests/test_pattern.py tests/test_export_equivalence.py` pass, with cross-engine mismatches = 0.

---

### M4 — Cleaning, decontamination, snapshots (target: 17 Nov)

Run these steps in order, on training sources only. Write `data/clean/<source_id>/` and log counts before and after each step to `artifacts/clean/report.json`.

1. **UTF-8:** drop invalid documents. Strip C0 and C1 controls except `\t` and `\n`. No normalization.
2. **Language ID** (GlotLID v3, document level):
   - Each source keeps the GlotLID labels of its own language(s).
   - Documents labelled as another language in the mix are re-routed to that language's bucket.
   - Drop everything else.
   - Record the confusion matrix.
   - **Per-language audit (automated):** bytes and documents before and after, the GlotLID confidence distribution, and the share of characters outside the language's expected scripts. Write 50 random kept and 50 random dropped documents per language to `artifacts/clean/samples/` for optional inspection. **STOP (human)** if more than 20% of a language's characters are outside its scripts.
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
     - FLORES+ dev and devtest for every harness language and every `*_Deva` language FLORES+ has, read from the harness's fetched copy or the gated dataset and never copied elsewhere
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
   - Report, per eval set and per source: candidate matches, documents and bytes removed, match-length distribution, the match keys used, and whether each drop was exact, segment or 13-gram, and write every 13-gram-only drop to `artifacts/clean/decontam_borderline.tsv` for optional inspection.
6. **Joiner augmentation (`augment_joiners.py`):** build the 3.8 synthetic copy from decontaminated Nepali documents not used elsewhere in the snapshot, seed 20261104. Unit test: removing every inserted joiner gives back the source document byte-for-byte.
7. **Snapshots:**
   - Sample the 3.8 mix at the 3.8 size, by bytes, at document level, seed 20261104. Pools use τ = 0.3 (3.8).
   - Write zstd parquet to `data/snapshots/<sha>/`, with `manifest.json` (per-shard sha256, per-source docs and bytes, mix, seed) and `doc_ids.parquet` (source, doc_id, license).
   - Build the 50% snapshot, plus 45% and 60% for M7, and one 50% snapshot **without** the joiner copy for the ladder.
8. **Independent audit:** re-run all three decontamination rules on each finished snapshot. All three must report 0.

**Acceptance:** `make snapshot` builds four snapshots with at least 50 GB still free. `docs/RESULTS.md` lists bytes per language per snapshot. The audit shows 0 on all rules. `uv run pytest tests/test_clean.py` passes.

---

### M5 — Akshara inventory and covered set (target: 19 Nov)

1. Run `aksharas()` over the 50% snapshot's Devanagari text. Write `artifacts/inventory/akshara_counts.tsv` (akshara, code points, count).
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

**Memory:** counting and training stream from disk and never hold raw text in RAM. Before the full build, benchmark the trainer on a 1 GB slice and record peak RAM and time in `docs/MACHINE.md`; **STOP** if peak RAM would exceed 14 GB at full size.

**Validation (`--plain`):** no blocks, no bans.

- On a 1 GB slice, compare with the Hugging Face `BpeTrainer` (`initial_alphabet=ByteLevel.alphabet()`, the same pattern and size).
- The merge lists must be identical, or every difference must be a tie at equal count.
- Also encode 100k snapshot lines with both and compare token IDs and decoded bytes exactly; any difference not caused by a documented tie is a bug.
- Write `artifacts/validation/plain_vs_hf.md`.

**Exporters:** per 3.5. The byte-level alphabet is `ByteLevel.alphabet()` via GPT-2's `bytes_to_unicode`.

**Guarantee tests (`tests/test_guarantee.py`)** on the 64k build from the 50% snapshot:

1. **Equality** (a Gate A and release condition): Hugging Face `encode(add_special_tokens=False)` == tiktoken `encode_ordinary` and both decode to the input bytes, on all eval texts, 1M snapshot lines, `pretok_adversarial.tsv`, 100k random Unicode strings (all planes, including unassigned and private-use code points), long digit strings, joiner sequences, mixed scripts, punctuation with trailing newlines, and lone surrogate-free malformed mark sequences.
1b. **Per-unit exhaustive check:** for every covered unit U, encode U alone and inside 100 contexts drawn from the snapshot (seed fixed), with both exports; no boundary may fall inside U. Also enumerate every byte string of length 1–4 built from the Block 0 characters' UTF-8 bytes and check the 3.4 ban-1 predicate against a brute-force reference.
2. **Covered integrity, per export:** across 10M sampled lines plus all eval texts, covered-akshara violations = 0, computed separately from the tokenizer.json output and from the tiktoken output. The tail break rate is reported with a CI.
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

1. Train the final config at 64k on the 45%, 50% and 60% snapshots.
2. Export each, run the guarantee tests, and run the harness:
   `uv run python scripts/eval_panel.py run --only none --add anc64_n45=<dir>/tokenizer.json --add anc64_n50=... --add anc64_n60=...`
3. Apply the 3.8 pick rule and record the result in `docs/RESULTS.md`.
4. **Ladder** on the chosen mix, at 64k, with everything else identical:

| Rung | block0 | block1 | bans | gb9c |
|---|---|---|---|---|
| plain | off | off | off | — |
| b0 | on | off | on | — |
| b0b1 (no bans) | on | on | off | off |
| final, no joiner copy | on | on | on | off |
| final | on | on | on | off |
| gb9c | on | on | on | on |
| unigram (control) | — | — | — | — |

   The `unigram` rung is a tokenizer-level control: SentencePiece Unigram at 64k on the same snapshot, with `byte_fallback=true` and no normalization (`normalization_rule_name=identity`), evaluated with the same harness and metrics. It differs from our trainer in pre-tokenization and has no protected units, so it compares tokenizers, not BPE versus Unigram as algorithms; the paper says so.

   The two "final" rungs differ only in the joiner copy (the M4 snapshot without it). **Keep the joiner copy** only if it lowers both joiner probes and costs no more than 0.005 Nepali fertility; record the decision in `docs/DECISIONS.md`.

   Run the harness and our metrics on every rung. Report fertility, bytes per token, orphan share, violations, tail, joiner probes and morphology alignment per rung in `docs/RESULTS.md`.

**STOP (human):** if no mix meets all three fertility rows of 3.10 (Nepali below 1.445, English below 1.420, Hindi below 1.385), report all three mixes and propose 65% Nepali, or a cap-bound covered set.

---

### M8 — Final builds, Gate A, efficiency (target: 8 Dec)

1. Build 32k, 64k and 128k on the chosen mix (with or without the joiner copy, per M7), and export each.
2. Run the harness on all of them plus the panel. Add our metrics:
   - `akshara.py`: violations, tail, and matra-detach, virama-detach and conjunct-split rates
   - `equality.py`
   - `multilang.py`: fertility on every FLORES+ Devanagari language and the mean, for ours and every panel tokenizer
   - `morph.py`: morphology alignment and cross-language reuse (3.9)
   - segmenter agreement (M3 step 6)
   - coverage rows (3.9): byte-level representability, whole-character coverage, extended-block coverage
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

### M8b — Phrase build and Gate P (target: 12 Dec)

1. Add the 3.13 stage 2 to `ancoda-bpe` (`--phrase-fraction`), with the phrase pattern and stage-2 bans. Write `tests/data/phrase_cases.tsv` and test that pre-tokens and tokens never exceed 4 words, never cross a newline, never absorb digits or punctuation, never end in a space, keep Devanagari marks and joiners inside words, and handle leading spaces and runs of spaces correctly.
2. Sweep `t` = 10%, 25%, 50% at 64k on the chosen mix; pick per 3.13.
3. Build Ancoda-Phrase-128k with the chosen fraction. Export both formats.
4. Rerun every M6 guarantee test on it (equality, covered integrity, mark-initial, round-trip, coverage, fuzzing).
5. Run the harness with `--add`, and `make gate-p` writes `docs/GATE_P.md` against Yak's phrase builds.

**Acceptance:** guarantee tests pass; `docs/GATE_P.md` is written. **STOP (human):** Gate P review.

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
8. Ancoda-Phrase (the 64k phrase pick from M8b, so vocab size matches arm 1)

**Tiers:**

- **reduced:** arms 1, 2, 5 and 8, seed 1 only.
- **full:** all 8 arms, seeds 1 and 2; seed 3 for arms 1 and 2; plus fixed-byte runs (step 4) for arms 1, 2 and 5.
- **none:** skip M9 entirely. Gate B is recorded as "not run", and every model-level claim is dropped (3.11).

The budget is set **after the pilot** (step 3). Planning figures only: Arkios's comparable runs took about 5.3 H100-hours per 4B tokens. That suggests roughly 21 H100-hours for reduced and roughly 110 for full (21 runs).

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
   - For every run record: raw UTF-8 bytes consumed, training tokens, documents seen and unique documents, compute (6 × total parameters × tokens), and whether the last batch overshoots the target.
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
4. **Fixed-byte runs (full tier only):** the main runs are fixed-token (4.0B tokens each). For arms 1, 2 and 5, also train one seed-1 run each to exactly the same number of raw bytes: the largest byte count any of the three reached in its fixed-token run. Report both comparisons; never present the fixed-token result alone as a pure measure of tokenizer quality.
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
   - **Downstream (reported, not gated):** Belebele `npi_Deva` accuracy, zero-shot, scoring each answer option by log-likelihood. Models this small may score near chance (25%); report it with that caveat.
6. **Gate B (`docs/GATE_B.md`):** bpb per arm and eval set: every seed, the mean, the seed spread (max − min), and the fixed-byte rows. Results with 2 seeds are labelled **preliminary**; no statistical-significance claim is made.
   - **Noise floor:**
     - full tier: the larger seed spread (max − min over seeds) of the two arms compared; a difference counts only if the mean gap exceeds it
     - reduced tier: 1.0% relative, fixed now (Arkios judged a 0.65% single-seed gap too thin)
   - **Pass:** arm 1 has the lowest Nepali bpb on fw2_npi_test, by more than the noise floor over every other arm. It is not worse on aya_npi beyond the noise floor. Its English bpb is within 1% of the best arm.
   - **Arm 2 (plain) better than arm 1 beyond noise:** recommend shipping plain BPE. Within noise: keep arm 1, since the guarantee costs nothing.
   - **Arm 3 better beyond noise:** recommend dropping Block 1.
   - **Arm 4 better beyond noise:** recommend GB9c units.
   - **Arm 8 (phrase) vs arm 1:** state which has lower bpb, beyond noise or not. This decides whether the phrase build is recommended for pretraining.
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
   - **limits:** the covered-set guarantee only; languages of Nepal measured only where open data exists; Romanized Nepali out of scope; phrase build not yet supported by llama.cpp; llama.cpp pending for all builds
3. **Package** each of Ancoda-64k, Ancoda-128k and Ancoda-Phrase-128k: `tokenizer.json`, `tokenizer_config.json`, `special_tokens_map.json`, `tiktoken/`, `manifest.json`, and `LICENSE`. The license is Apache-2.0, **pending human and legal sign-off** against the `docs/LICENSES.md` release-notes column.
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

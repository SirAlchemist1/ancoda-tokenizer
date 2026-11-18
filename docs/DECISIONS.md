# Fixed decisions (the spec)

Source of truth copied from PLAN.md section 3 (v2.6). Do not change without the human's explicit approval.

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

## M2 approvals (4 Nov 2026)

The human approved the following for milestone M2. Sections 3.1–3.13 are unchanged.

### Storage

Option 2 is approved. For each training source, delete its raw parquet only after its cleaned parquet is written and verified against the manifest, and only after the exact Hugging Face revision (commit hash) is recorded in that manifest so the text can be re-downloaded. Never delete eval-only or decontamination-only sources. The 50 GB rule from section 2 still applies at every step.

The storage-plan peak that is gated is one copy of each training source (the raw parquet until M4 replaces it with the cleaned parquet), plus eval, decontamination and tool files, plus the snapshots, plus the counts, tokenizers and eval allowance. Raw and cleaned training text are not kept together.

### Languages of Nepal

All 10 FineWeb-2 configs are approved for `nepal_pool`, ODC-By: `new_Deva`, `thl_Deva`, `taj_Deva`, `lif_Deva`, `rav_Deva`, `suz_Deva`, `xsr_Deva`, `rjs_Deva`, `kle_Deva`, `nwx_Deva`. No upsampling. Unused pool share goes to Nepali per 3.8. Train splits only. The CC BY-NC Tamang corpus (`ilprl-docse/NepTam-A-Nepali-Tamang-Parallel-Corpus`) and the Gurung audio (`eduhk-compling/GurungJasmika`) are excluded.

### Romanized Nepali

Out of scope for training in v1. FineWeb-2 `npi_Latn` has a test split; download that split only, as eval-only, for the reported Romanized metric. Do not download the `npi_Latn` train split. Do not use `dipeshch71/nepaliflow-romanized-nepali-to-devanagari-dataset` or `sabin1234/NEPALI_TO_ROMANIZED_WIKIPEDIA`.

### Papaya affix inventory

Approved for evaluation only under CC BY 4.0, at commit `b3498528daa3435f091792e0b44bb222a2f8cb68`. Attribution: the affix dataset is re-derived from Balaram Prasain, "A Computational Analysis of Nepali Morphology: A Model for Natural Language Processing", PhD dissertation, Tribhuvan University, 2011. Cite that work. The files stay under `data/` and are not committed. Do not copy the repository's tokenizer vocab, merges or stem lexicons.

## M4 legacy-font detector (5 Nov 2026)

The logistic-regression Preeti classifier failed the 99% precision spot-check (precision 0.05 after one retrain). The human replaced it with a **dictionary-based Preeti detector**:

1. Take the 50,000 most frequent Devanagari words from Nepali training sources after UTF-8 cleanup and GlotLID keep (`npi_Deva`). Convert each with the MIT `globalpolicy/UnicodeToPreeti` converter. Keep converted forms of length ≥ 2. Write `artifacts/clean/preeti_lexicon.tsv` with sha256.
2. A Latin-script line in a Nepali document is flagged if it has ≥ 3 whitespace tokens in that lexicon and those tokens are ≥ 50% of the line's tokens. Drop a document only if more than 20% of its lines are flagged.
3. Spot-check: 100 flagged + 100 unflagged → `artifacts/clean/preeti_spotcheck_v2.tsv`. If precision ≥ 99%, keep the filter. If below 99%, do not retry: disable the filter, record that here, count would-have-flagged lines per source as a limitation, and continue M4.

### Legacy-font filter status (after dictionary spot-check)

legacy-font filter disabled: no detector reached 99% precision.

Dictionary detector spot-check (`artifacts/clean/preeti_spotcheck_v2.tsv`): precision **0.89** (89 true Preeti / 100 flagged; 11 false positives). Known true Preeti lines caught: **3/5**. Lexicon: `artifacts/clean/preeti_lexicon.tsv` sha256 `2761f296d51b367f0418d2c6a2d6754c4540a88a0910f521e6628f78a04fd454` (49,830 forms from 50,000 Devanagari words).

**Limitation — would-have-flagged lines** (dictionary detector, filter not applied), from `artifacts/clean/report.json` `legacy_would_flag_lines`:

| source | would-flag lines |
| --- | ---: |
| fw2_npi | 35 |
| iriis | 0 |
| sangraha_nep | 104 |
| indiccorp_npi | 0 |
| hplt_npi | 426 |
| **total** | **565** |

## M4 decontamination n-gram scope (6 Nov 2026; human-approved 7 Nov 2026)

**Approved:** for the large same-distribution web test sets (`fw2_npi_test`, `fw2_hin_test`, `c4_val`), apply **exact-match only**. Their 6–12-word segment and 13-word n-gram rules produced ~40–60% false-overlap drops on Nepali training (ordinary n-gram overlap with the same web distribution, not eval-passage leakage) and the merged indexes (~339M segment hashes) were too heavy for an 18 GB Mac.

**Unchanged — full three rules** (exact, 6–12-word segments, 13-gram) still apply to:
- FLORES+ `dev` and `devtest` for every language present in the harness copy under `third_party/yak-tokenizer/data/eval/flores_plus/`
- Belebele `npi_Deva` (`belebele_npi`)
- `aya_npi`

Confirmed in code: `iter_eval_texts(ngram=True)` indexes FLORES+ + Belebele + aya for segments/13-grams; `_EXACT_ONLY_EVAL_SOURCES` contribute only to the exact-key map.

### Measured drop rates (fw2_npi sample / full corpus)

**Before** (large-set segment + 13-gram enabled), independent rates on the first **5,000** `fw2_npi` clean docs (`artifacts/clean/decontam_rate_probe.log`). A document may count in more than one cell:

| eval set | rule | hits / 5,000 | rate |
|---|---|---:|---:|
| fw2_npi_test | segment (6–12) | 2,887 | **0.5774** |
| fw2_npi_test | 13-gram | 539 | 0.1078 |
| fw2_hin_test | segment (6–12) | 30 | 0.0060 |
| fw2_hin_test | 13-gram | 2 | 0.0004 |
| c4_val | segment (6–12) | 32 | 0.0064 |
| c4_val | 13-gram | 0 | 0.0000 |
| short_eval (FLORES+Belebele+aya) | segment | 0 | 0.0000 |
| short_eval (FLORES+Belebele+aya) | 13-gram | 0 | 0.0000 |
| large-set segment **union** | segment | 2,897 | **0.5794** |

Earlier probes with the merged full index (including large sets): bloom bench 21/50 = **0.42**; first 50k `fw2_npi` docs in `decontam_run6.log` 30,318/50,000 = **0.6064**.

**After** (exact-only on large sets; full three rules on short eval), same 5,000-doc sample: **0** drops. Full-corpus after rates from `artifacts/clean/decontam_report.json` (34,652,585 scanned = kept + dropped over training sources in that report; overall drop rate **0.001556**):

| eval set | rule | dropped docs | rate |
|---|---|---:|---:|
| fw2_npi_test | exact | 5,821 | 0.000168 |
| fw2_hin_test | exact | 3 | ~0 |
| c4_val | exact | 0 | 0 |
| short_eval (FLORES+Belebele+aya) | exact | 0 | 0 |
| short_eval (FLORES+Belebele+aya) | segment (6–12) | 48,101 | 0.001388 |
| short_eval (FLORES+Belebele+aya) | 13-gram | 0 | 0 |
| fw2_npi_test / fw2_hin_test / c4_val | segment | **0** (disabled) | 0 |
| fw2_npi_test / fw2_hin_test / c4_val | 13-gram | **0** (disabled) | 0 |

On the 50k `fw2_npi` probe after the change (`decontam_run8.log`): 98/50,000 = **0.0020** (vs 0.6064 before).

### MinHash near-dup vs large test sets (7 Nov 2026)

In place of the disabled large-set n-gram rules: drop any training document whose MinHash signature (same settings as M4 step 4: 5-gram word shingles, 14 buckets × 8 hashes, seed `20261104`) collides in any band with a document in `fw2_npi_test`, `fw2_hin_test` or `c4_val`. Prior MinHash signature workdir was not on disk; eval bands are rebuilt. Counts go to `artifacts/clean/minhash_eval_near_dup.json`. The independent snapshot audit also checks this rule (`minhash_eval` must be 0).

**Result:** dropped **20,132** training documents (**122,711,282** bytes) across clean sources + `_reroute/`. Top sources by docs: sangraha_nep 11,692; iriis 4,631; hplt_npi 2,331; fw2_npi 1,020; indiccorp_npi 236; fw2_hin 183.

### n45 residual audit hits (200 segment) — cause and fix

The first n45 independent audit reported `{'exact': 0, 'segment': 200, '13gram': 0}`. Cause: LID-rerouted docs under `data/clean/_reroute/` were included in snapshots but had **not** been run through text decontamination. Fix: `decontam_reroutes()` in `run_decontam`, then rebuild snapshots. After that rebuild, text-rule audits were 0; MinHash near-dup then required another rebuild + full audit including `minhash_eval`.

## Human correction (8 Nov 2026) — Ban 1 reading and special-token encoding

Recorded at the human's instruction during M6. The section 3 text above is unchanged; this note is only the corrected reading.

1. **Ban 1, second clause.** `M` lies within one character's UTF-8 encoding: valid lead byte, valid continuation bytes for each later position, and `len(M)` no greater than that lead's width. Proper prefixes such as `E6 97` are allowed. The clause does not mean "exactly one scalar value has an encoding that starts with M". The first clause (both sides complete valid UTF-8) is unchanged.
2. **Ordinary-text encoding.** Tests set `Tokenizer.encode_special_tokens = True` (tokenizers 0.23.3) so literal special-token strings are plain text, like tiktoken `encode_ordinary`. See `docs/MODEL_CARD_NOTES.md`.

## M7 follow-up (8 Nov 2026) — Joiner copy only on n50 (deviation from 3.8)

PLAN.md 3.8 says the 1% joiner copy is on by default for the mix. The M4 snapshot builder applied joiners **only when Nepali share = 50%**, so only `n50` has the synthetic copy (`joiner_share=0.01`, 21348 docs / ~50 MB). `n45` and `n60` have `joiner_share=0.0`. The M7 ladder "final" rung forced onto n60 therefore has **no** joiner copy; `final_nojoiner` remains the n50_nojoiner arm.

**Recommendation:** keep the joiner copy in every future snapshot (not only n50). Evidence: n50 ZWJ/ZWNJ probes 1.368 / 1.369 vs n60 1.614 / 1.633. Nepali fertility cost on the matched n50 pair is about 0.002 (`mix_anc64_n50` 1.515748 vs `lad_final_nojoiner` on `n50_nojoiner` 1.513823).

## M7 follow-up (8 Nov 2026) — Danda stays a separate token

Danda (purna biram, U+0964) and double danda (U+0965) stay separate tokens, like "." in English. This is a product decision for the paper and shipping builds; pre-tokenizer / merge behaviour that keeps `।` as its own piece is intentional.

## Sealed test set (locked 9 Nov 2026, before M8)

Hash-selected held-out docs from cleaned training data that appear in no snapshot built so far (including n60j5 and the 10GB joiner snapshot), near-dup filtered against all snapshots, and excluded from future snapshot sampling via `sha256(doc_id)`.

- **n** = 35688
- **seed** = 20261104
- **list_sha256** = `71f744cea9e958351494fea6d70e35597daa833b4875bf00878caf64951c3da0`
- **Canonical copy (hashes + seed only, no document text):** `docs/sealed_test_set.json`

## M8 selection (lookup applied 9 Nov 2026)

Selection rules written in the research log at 12:45 on 9 Nov, before the 13:40 results; lookup applied mechanically.

Inputs:

- Yak v1 64k code/KB **316.4**
- Yak v1 128k code/KB **291.7**

Outputs:

- Snapshot **n60j5**, joiner copy on, code+math scale **1.0**
- Block 1 for every size = that size's M5 inventory ranking truncated to a **3%** cap. The 250k inventory is built the same way and saved as `artifacts/inventory/covered_250k.json`.
- **Ancoda-64k:** vocab 65,536, B_d 48,000
- **Ancoda-128k:** vocab 131,072, B_d 94,742
- **Ancoda-32k:** vocab 32,768, B_d 23,522 (same Devanagari share, 48,000/64,256, of non-byte non-reserved slots)
- **Ancoda-NE-64k:** vocab 65,536, standard single-list trainer (no script budget), Nepali-only text from n60j5 including the joiner copy
- **Ancoda-250k:** vocab 250,000. Frontier B_d 144,000 / 160,000 / 175,000 / 190,000 × code scale 1.0 and 0.5. Pick rule: integrity + Block 1 coverage ≥ 99.95%; prefer points that beat IndicBERT v2 on Hindi AND English (whole-file CI excluding 0), beat every ~250k tokenizer on Nepali, and have code/KB ≤ IndicBERT v2's; among those the lowest Hindi; if none qualifies, the most WIN rows in the ~250k class, then the lowest Nepali. Re-run the chosen point on n60j10; use it only if Hindi or English improves with CI excluding 0. Memory cap 12 GiB; if exceeded, stop that point and report.

## Tokenizer path naming (16 Nov 2026)

The paper numbers and the anonymous release use **`artifacts/tokenizers/m8/ancoda-64k`** (`tokenizer.json` sha256 `c9ee0986bcb249109371b696f7004e8b664a6533539150bc92f9babe2435013e`). The anonymous release ranks table matches that build’s `tokens_hex` and layout (only path strings inside `stats` are anonymized).

The older directory **`artifacts/tokenizers/ancoda-64k`** (sha256 `db6bd43e0e09ee221037d7d59642a6843179a5b45304c22ce191bb3a1f6781ed`) is **superseded**. It is an earlier n50-era build: the Figure 2a sentence `नमस्ते, तपाईंलाई कस्तो छ?` is 7 tokens there and 6 tokens under `m8/ancoda-64k`. Keep the directory on disk for provenance; do not use it for paper numbers, M9, or release.

## Ancoda-250k snapshot provenance (18 Nov 2026)

Provenance correction only. No decision changes and no tokenizer changes.

`artifacts/tokenizers/m8/ancoda-250k/m8_manifest.json` lists `snapshot: data/snapshots/n60j5_20af1fffbec0fe86`. That is wrong. The adopted 250k build is the n60j10 re-run (`data/snapshots/n60j10_502301959d423052`), as the M8 selection above allows:

- `artifacts/eval/m8/j10_decision.json`: `use_j10` true, reason "Hindi or English improved with CI excluding 0", adopted dir `artifacts/eval/m8/j10/ancoda-250k-bd160000-cm50-j10`. `artifacts/eval/m8/chosen.json` has `source: n60j10`.
- The manifest's `dewa_counts_sha256` `551b0a96b252c130f2891ac787437729bcba78dd50b4a325446505e20ba0e0ec` and `latin_counts_sha256` `a6eeeb998f041cc0f6f8f1af332d62a5e264cc7a7b49a73de9001a0fe4f223ad` equal the sha256 of `artifacts/eval/m8/n60j10_counts/deva_pretoken_counts.bin` and `latin_cm50_pretoken_counts.bin`. The n60j5 frontier point `bd160000-cm50` has different counts (`bfda24fc…b4b8a6`, `b32235a2…b8a749`).
- Cause: `train_budget` in `python/ancoda_tok/m8_final.py` writes `SNAP_J5` into every manifest, whatever counts it is given.

The manifest's existing fields are unchanged. A `snapshot_correction` field was added with the true snapshot and this evidence. The file is under `artifacts/` and is not committed; the identical original copy at `artifacts/eval/m8/j10/ancoda-250k-bd160000-cm50-j10/m8_manifest.json` keeps the sha256 `3f8ceeb2…e559a2` recorded in `docs/EVIDENCE_M8.md`.

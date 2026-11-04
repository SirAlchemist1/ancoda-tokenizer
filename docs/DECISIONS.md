# Fixed decisions (the spec)

Source of truth copied from PLAN.md section 3 in M0. Do not change without the human's explicit approval.

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

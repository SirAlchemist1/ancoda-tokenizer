# Specification audit (M0 step 1)

Audit of docs/SPEC.md sections 1–5 only. Section 3 decisions are not changed.
Each item is marked **blocking** or **non-blocking**.

---

## Summary

No **blocking** specification defects were found. Several **non-blocking** ambiguities, third-party risks, and missing or deferred acceptance details are recorded below. M0 may proceed past this step.

---

## Contradictions

### C1 — Akshara prose vs implementation scope — non-blocking
- **Where:** 3.3 opening definition vs implementation steps 2–3.
- **Issue:** The prose says an akshara is a UAX #29 extended grapheme cluster “computed **with rule GB9c disabled**.” The implementation computes clusters with `unicode-segmentation` 1.12.0 (GB9c on), then undoes GB9c **for Devanagari only** (split after virama(+joiner) before a Devanagari consonant). Non-Devanagari InCB conjunct behavior therefore still follows GB9c.
- **Why non-blocking:** Steps 1–5 are the executable rule; Nepali/Devanagari is the in-scope script. Mode `gb9c=true` is defined for ablation. No section 3 change required; treat the prose as summarizing the Devanagari intent.

### C2 — “Beat Yak v2 on every word-level row” vs Nepali pass bar — non-blocking
- **Where:** intro / 3.10 narrative vs Gate A table.
- **Issue:** Narrative requires beating Yak v2 64k on every word-level row **and** the lowest Nepali count among 64k tokenizers. The Nepali pass bar is “below 1.445” (sidskarki-64k), which is stricter than Yak v2’s 1.470; English/Hindi bars are strict inequalities vs Yak v2 (1.420 / 1.385).
- **Why non-blocking:** The table encodes the narrative; beating sidskarki on Nepali implies beating Yak v2 on Nepali. Stretch bars for Yak v1 English/Hindi are separate.

### C3 — Layout lists `pretokenizer_pattern.txt` under M0 tree; M3 generates it — non-blocking
- **Where:** section 4 vs M3 step 1 / section 4 comment “generated in M3”.
- **Issue:** The tree shows the file at repo creation time; generation is M3.
- **Why non-blocking:** Create `configs/` in M0; do not invent pattern content until M3.

### C4 — M3 milestone close vs gold agreement Gate A — non-blocking
- **Where:** M3 step 6 vs M3 acceptance vs 3.10 / M8.
- **Issue:** M3 may close before annotation sheets return; gold 100% is a Gate A / M8 requirement, not M3 acceptance.
- **Why non-blocking:** Explicitly stated; track gold as a cross-milestone dependency in PROGRESS.

---

## Undefined quantities

### U1 — Where worker counts are set — non-blocking
- **Where:** M0 step 3 (“Set rayon and datatrove worker counts to the performance-core count”).
- **Issue:** No file or env var name is specified.
- **Resolution for M0:** Export `ANCODA_WORKERS` (and document it) from the Makefile as the performance-core count; later code reads that. Not a section 3 change.

### U2 — Packaging layout for `python/ancoda_tok/` — non-blocking
- **Where:** section 4 vs M0 step 5 `uv init`.
- **Issue:** Exact `pyproject.toml` package discovery (`python/` layout, maturin binding path) is not specified.
- **Resolution:** Configure a src-style / `python/` package path in `pyproject.toml` so `ancoda_tok` imports work; keep torch off the Mac.

### U3 — “Languages of Nepal” mean on Yak scoreboard — non-blocking
- **Where:** 3.10 Yak scoreboard (Yak v2 1.843).
- **Issue:** Which languages enter the mean is not defined in docs/SPEC.md.
- **Why non-blocking:** Defined by Yak’s harness output; M1 records whatever the harness reports.

### U4 — Aya / everyday-Nepali fertility aggregation — non-blocking
- **Where:** 3.9–3.10.
- **Issue:** Whether Aya fertility uses whole-file, per-line, or another harness rule is not restated locally.
- **Why non-blocking:** “via Yak’s harness”; M1 captures the harness definition and numbers.

### U5 — Seeds outside the snapshot seed — non-blocking
- **Where:** section 2 determinism rule; snapshot seed `20261104` in M4.
- **Issue:** Other pipeline seeds (inventory, eval samples, proxy) are “fixed” but not all listed in section 3.
- **Why non-blocking:** Later milestones can pin and record seeds in manifests without changing section 3.

### U6 — Block 1 “5% of vocab” rounding — non-blocking
- **Where:** 3.6 step 6.
- **Issue:** Caps are given as integers (1,638 / 3,276 / 6,553) matching floor(0.05 × size); the rounding rule is not named.
- **Why non-blocking:** Exact caps are listed; use those integers only.

### U7 — README / stub file contents — non-blocking
- **Where:** section 4.
- **Issue:** `README.md` and many Python modules are named but have no M0 content spec.
- **Resolution:** Minimal stubs sufficient for layout and compile/import acceptance.

### U8 — First reserved ID indexing — non-blocking
- **Where:** 3.7.
- **Issue:** “From the first reserved ID” implies `total − 1024` but does not spell the formula.
- **Why non-blocking:** Fixed block sizes imply first reserved ID = total − 1,024; confirm in M6 tests.

---

## Third-party library / external assumptions

### T1 — o200k edit replacement counts (exactly 4 and 1) — non-blocking for M0
- **Where:** 3.1 Edit 1 and Edit 2.
- **Issue:** Counts are asserted against whatever string the installed `tiktoken` exposes. A tiktoken release that changes `o200k_base`’s pattern would fail the asserts.
- **Why non-blocking:** M3 asserts and STOPs on mismatch; do not change section 3. Pin tiktoken via `uv.lock` after M0 install; never upgrade without approval.

### T2 — `unicode-segmentation = "=1.12.0"` and Unicode 16.0.0 — non-blocking for audit
- **Where:** 3.3; M0 step 8.
- **Issue:** Plan states 1.12.0 is the first release with GB9c and Unicode 16.0.0 data; 1.11.0 lacks GB9c; 1.13.0/1.13.1 yanked. M0 must assert `UNICODE_VERSION == (16, 0, 0)` after the crate builds.
- **Why non-blocking:** Verification is an M0 acceptance/setup step, not a spec contradiction. If the assert fails → STOP (evidence vs section 3), do not retarget the pin without human approval.

### T3 — `tokenizers` crate default `onig` feature — non-blocking
- **Where:** M0 step 6 (`ancoda-seg`).
- **Issue:** Plan assumes default features include the onig Split engine matching Hugging Face `tokenizer.json`.
- **Why non-blocking:** Confirm in Cargo feature resolution during M0/M3; if defaults change, pin features explicitly without changing 3.1’s `use_regex=False` requirement.

### T4 — Cross-engine pretokenize equality (onig / Python `regex` / fancy-regex) — non-blocking
- **Where:** 3.1 “all three agree”; M3 step 4.
- **Issue:** Agreement was verified on 4 Nov 2026 for cited examples; not guaranteed for all adversarial strings forever.
- **Why non-blocking:** M3 STOPs on first 20 mismatches; that is the acceptance path.

### T5 — HF export equivalence (tokenizers 0.23.3, tiktoken 0.14.0) — non-blocking
- **Where:** 3.5.
- **Issue:** Verification was version-specific. Locked versions may differ slightly after `uv add`.
- **Why non-blocking:** M3/M6 re-run the equivalence tests on the locked versions; failures STOP.

### T6 — `fasttext-wheel` on Apple Silicon — non-blocking
- **Where:** M0 step 5.
- **Issue:** Plan already provides fallback to `fasttext` (needs cmake) and STOP if both fail.
- **Action:** Follow that order; no section 3 change.

### T7 — `codeparrot/github-code-clean` loader — non-blocking for M0
- **Where:** M2.
- **Issue:** Script loader may be refused by current `datasets`; plan requires reading `data/` files directly and STOP if format/columns differ.
- **Why non-blocking:** Deferred to M2 with an explicit STOP.

### T8 — Yak / competitor fertility reference table — non-blocking
- **Where:** 3.10 references from Yak `docs/RESULTS.md` (3 Nov 2026).
- **Issue:** Gate A uses **M1 re-measured** values, not the printed references, if they differ.
- **Why non-blocking:** Plan states M1 values are authoritative for the gate.

### T9 — sidskarki-64k license for proxy comparison — non-blocking for M0
- **Where:** M1 step 3; M9 arm 6.
- **Issue:** Evaluation/proxy use must be confirmed from community-use-1.0 terms in M1; STOP if disallowed or unclear.
- **Why non-blocking:** Explicit M1 STOP; does not block M0.

---

## Missing or weak acceptance criteria

### A1 — SPEC_AUDIT / DECISIONS / rules file not in M0 acceptance commands — non-blocking
- **Where:** M0 acceptance block.
- **Issue:** Acceptance does not `test -f` for `docs/SPEC_AUDIT.md`, `docs/DECISIONS.md`, or the local rules file.
- **Mitigation:** Still required by M0 steps 1 and 4; record paths in PROGRESS; `cat docs/MACHINE.md docs/SPEC_AUDIT.md` partially covers audit presence.

### A2 — `hf auth whoami` vs human login STOP — non-blocking
- **Where:** M0 STOP vs acceptance.
- **Issue:** Acceptance cannot pass until the human completes `uv run hf auth login` and FLORES+ terms.
- **Why non-blocking:** Intentional human gate; agent STOPs and waits.

### A3 — No command checks free-disk / RAM thresholds — non-blocking
- **Where:** M0 step 2 vs acceptance.
- **Issue:** Machine thresholds are enforced by process (STOP), not by a scripted acceptance line.
- **Mitigation:** Write exact `df` / `sysctl` output into `docs/MACHINE.md` and STOP in PROGRESS if violated.

### A4 — Empty crate tests satisfy `cargo test --workspace` — non-blocking
- **Where:** M0 acceptance.
- **Issue:** Stub crates with no tests still “pass.”
- **Why non-blocking:** Acceptable for M0; real tests arrive in M3+.

### A5 — Gate A orphan-mark “every occurrence inside uncovered akshara or malformed sequence” — non-blocking
- **Where:** 3.10.
- **Issue:** “Malformed sequence” is not formally defined beyond orphan-mark context listing in M6.
- **Why non-blocking:** M6 step 4 requires listing every occurrence with context; human/Gate A review can classify. Not required to redefine in section 3 for M0.

### A6 — Phrase-build table has no numeric pass bar — non-blocking
- **Where:** 3.10.
- **Issue:** Phrase builds are “different category”; report-only unless claiming a beat.
- **Why non-blocking:** Intentional; claims discipline in 3.11 covers misuse.

### A7 — M0 does not require `caffeinate` smoke test — non-blocking
- **Where:** section 2 long-job rule.
- **Issue:** No M0 check that `caffeinate` works.
- **Why non-blocking:** Used from M2 onward; trivial OS tool on macOS.

---

## Blocking items

**None.**

No specification item requires stopping M0 before the machine check, toolchain, or layout steps. Section 3 is left unchanged.

---

## Audit conclusion

| Category                         | Blocking | Non-blocking |
|----------------------------------|----------|--------------|
| Contradictions                   | 0        | 4            |
| Undefined quantities             | 0        | 8            |
| Third-party / external risks     | 0        | 9            |
| Missing / weak acceptance        | 0        | 7            |

**Proceed to M0 step 2 (machine check).** If free disk &lt; 250 GB or RAM &lt; 16 GB, STOP per M0 step 2 (environment), not because of this audit.

---

# Re-audit — PLAN v2.6 (4 Nov 2026)

Scope: changed parts **3.3–3.13** and milestones **M1–M10** (including new **M8b**). Prior M0 audit items that referred to the old “undo GB9c” wording or phrase-report-only Gate A are superseded where noted. Section 3 is not changed by this audit. Each item is **blocking** or **non-blocking**.

## Summary (v2.6)

No **blocking** defects. Proceed to commit and M1.

---

## 3.3–3.13

### R1 — Mission “akshara” vs 3.3 “not conventional orthographic akshara” — non-blocking
- **Where:** §1 mission; 3.3 definition.
- **Issue:** Mission still says “whole aksharas”; 3.3 defines an Ancoda protected unit and requires the paper to say it is not the conventional orthographic akshara.
- **Why non-blocking:** 3.3 is authoritative; mission uses the project term. Paper/model-card wording is gated by 3.11.

### R2 — Old audit C1 superseded — non-blocking (resolved)
- **Where:** M0 audit C1; 3.3 v2.6.
- **Issue:** v2.6 replaces “GB9c disabled / undo” with an explicit override (“every Devanagari letter begins a new unit”) and a boundary table.
- **Why non-blocking:** Implementation step 3 matches the definition; `gb9c=true` remains raw clusters without the guarantee.

### R3 — Guarantee proof C2 vs eligible units with non-Extend Block 0 chars — non-blocking
- **Where:** 3.5 proof C2; 3.6 eligibility.
- **Issue:** C2 assumes every later code point of U is Grapheme_Cluster_Break Extend or SpacingMark. Eligibility only requires Block 0 characters (U+0900–U+097F, ZWJ, ZWNJ), which also include Lo letters and digits.
- **Why non-blocking:** A second Lo letter inside one eligible multi-code-point unit cannot occur under the 3.3 override (letters start units). Digits/danda as “later” code points of an eligible unit that started with a letter are not produced by the segmenter for well-formed text; ineligible/malformed cases are excluded from coverage. M6 exhaustive per-unit tests are the backstop; STOP if they fail.

### R4 — Letter set source files — non-blocking
- **Where:** 3.3 step 4 (UnicodeData + Scripts); M0 still lists DerivedCoreProperties.
- **Issue:** Consonant/InCB reading is no longer the eligibility basis; Lo∩Devanagari is.
- **Why non-blocking:** UnicodeData.txt and Scripts.txt are already present; DerivedCoreProperties remains useful for the M3 Python reference (InCB) and is not contradictory.

### R5 — Block 1 coverage may be unreachable at 99.99% — non-blocking
- **Where:** 3.6 steps 5–7.
- **Issue:** Ineligible units count in N; 99.99% may be impossible.
- **Why non-blocking:** Plan states the cap then decides; report achieved coverage exactly.

### R6 — Joiner copy accounting (49%+1%) — non-blocking
- **Where:** 3.8; section 2 new synthetic-joiner rule.
- **Issue:** Nepali natural share becomes 49% when the 1% synthetic copy is on.
- **Why non-blocking:** Explicit; ladder has a no-joiner snapshot; M7 keep/drop rule is numeric.

### R7 — Morphology: whole-word no longer matches boundaries — non-blocking
- **Where:** 3.9 morphology vs whole-word preservation.
- **Issue:** Recall definition means a one-token word matches none of its reference boundaries; F1 and whole-word preservation are separate.
- **Why non-blocking:** Intentional (v2.6 changelog); both are reported, not Gate A.

### R8 — ZWJ/ZWNJ Gate A bar now beats Yak v2 (was sidskarki ceiling) — non-blocking
- **Where:** 3.10.
- **Issue:** Pass bar tightened to Yak v2 M1 values; miss → STOP (human).
- **Why non-blocking:** Explicit STOP path; joiner augmentation exists to pursue the bar.

### R9 — Languages-of-Nepal mean “not comparable” — non-blocking
- **Where:** 3.10; M1 acceptance notes.
- **Issue:** Private 14-set mean cannot be rerun; must not claim WIN/LOSE.
- **Why non-blocking:** Explicit reporting rule.

### R10 — Phrase builds moved to Gate P (3.13) — non-blocking (supersedes M0 A6)
- **Where:** 3.10, 3.13, M8b.
- **Issue:** M0 audit A6 said phrase rows had no pass bar; Gate P now defines bars.
- **Why non-blocking:** Updated by v2.4–v2.6; word-level vs phrase comparison forbidden on Gate A.

### R11 — Phrase pattern constructed from 3.1 file — non-blocking
- **Where:** 3.13 step 2.
- **Issue:** Exact regex surgery (“first two branches wrapped `{1,4}`”) depends on o200k structure remaining stable.
- **Why non-blocking:** Assert against 3.1 examples; M8b STOP on mismatch; do not retype the pattern.

---

## M1–M10 (incl. M8b)

### R12 — M1 step 6 path `eval/gate_a.py` vs package module — non-blocking
- **Where:** M1 step 6; acceptance `python -m ancoda_tok.eval.gate_a`.
- **Issue:** Shorthand path vs `python/ancoda_tok/eval/gate_a.py`.
- **Why non-blocking:** Acceptance command defines the module; implement there.

### R13 — M1 reproduction table vs harness panel names — non-blocking
- **Where:** M1 acceptance; `configs/eval_panel.yaml`.
- **Issue:** Table uses “Yak v1 128k” etc.; panel names are `yak-128k-v1`, `yak-128k-v2.0`, `yak-128k-v1-phrase`.
- **Why non-blocking:** Plan allows mapping from `eval_panel.yaml`; record mapping in PROGRESS/RESULTS.

### R14 — M1 ±0.03 vs ±0.003 — non-blocking
- **Where:** M1 steps 2 and acceptance.
- **Issue:** Two different tolerances for two different checks.
- **Why non-blocking:** Clarified in v2.6; both must be recorded in `m1_reproduction.tsv` for the ±0.003 rows.

### R15 — sidskarki community-use-1.0 for proxy — non-blocking (M1 STOP if unclear)
- **Where:** M1 step 3; M9 arm 6.
- **Issue:** Proxy use may be disallowed.
- **Why non-blocking:** Explicit STOP; evaluation-only remains possible.

### R16 — M2 HPLT / nepal_pool / papaya sources may be missing — non-blocking
- **Where:** M2 source table; 3.8 pools.
- **Issue:** Config names and licensed nepal_pool sources are discover-only; papaya GitHub path must be confirmed.
- **Why non-blocking:** Plan forbids inventing names and has STOP/list paths; does not block M1.

### R17 — M3 independent Python segmenter must not call Rust — non-blocking
- **Where:** M3 step 6.
- **Issue:** Dual implementation risk of shared bugs if both misread the same table.
- **Why non-blocking:** Differential test + Unicode test file; human STOP only if rule ambiguity remains (none identified as blocking now).

### R18 — M4 four snapshots vs older three — non-blocking
- **Where:** M4 acceptance.
- **Issue:** Adds no-joiner 50% snapshot.
- **Why non-blocking:** Explicit; storage plan in M2 absorbs size.

### R19 — M6 RAM STOP at 14 GB peak — non-blocking
- **Where:** M6 memory note; MACHINE.md has 18 GB RAM.
- **Issue:** Trainer must stream; 14 GB headroom on an 18 GB machine is tight with OS overhead.
- **Why non-blocking:** Explicit STOP before full build; 5 GB mix if RAM &lt; 32 GB already chosen by 3.8.

### R20 — M7 Unigram control not algorithmic BPE-vs-Unigram — non-blocking
- **Where:** M7 ladder note.
- **Issue:** Different pretokenization; paper must not overclaim.
- **Why non-blocking:** Plan states the limit.

### R21 — M8b / Gate P schedule vs slip rules — non-blocking
- **Where:** 3.12; M8b.
- **Issue:** Phrase-128k is cut before full GPU tier if slipping.
- **Why non-blocking:** Ordered; never cut guarantee/equality/segmenter/Gate A core.

### R22 — M9 fixed-byte + 3 seeds for arms 1–2 — non-blocking
- **Where:** M9 tiers; Gate B.
- **Issue:** Budget ~110 H100-hours planning figure; two-seed results labelled preliminary.
- **Why non-blocking:** Pilot STOP confirms budget; labelling rules are clear.

### R23 — M10 packages three builds — non-blocking
- **Where:** M10 step 3.
- **Issue:** Legal sign-off still required before public.
- **Why non-blocking:** Explicit human STOP.

---

## Blocking items (v2.6 re-audit)

**None.**

---

## Re-audit conclusion

| Category | Blocking | Non-blocking |
|---|---|---|
| 3.3–3.13 | 0 | 11 |
| M1–M10 / M8b | 0 | 12 |

**Proceed:** commit “Update to PLAN v2.6”, then execute M1.

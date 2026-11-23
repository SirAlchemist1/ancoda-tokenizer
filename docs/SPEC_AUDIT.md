# Specification audit (setup step 1)

Audit of docs/SPEC.md sections 1–5 only. Section 3 decisions are not changed.
Each item is marked **blocking** or **non-blocking**.

---

## Summary

No **blocking** specification defects were found. Several **non-blocking** ambiguities, third-party risks, and missing or deferred acceptance details are recorded below. setup may proceed past this step.

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

### C3 — Layout lists `pretokenizer_pattern.txt` under setup tree; pretokenizer generates it — non-blocking
- **Where:** section 4 vs pretokenizer step 1 / section 4 comment “generated in pretokenizer”.
- **Issue:** The tree shows the file at repo creation time; generation is pretokenizer.
- **Why non-blocking:** Create `configs/` in setup; do not invent pattern content until pretokenizer.

### C4 — pretokenizer milestone close vs gold agreement Gate A — non-blocking
- **Where:** pretokenizer step 6 vs pretokenizer acceptance vs 3.10 / final-build.
- **Issue:** pretokenizer may close before annotation sheets return; gold 100% is a Gate A / final-build requirement, not pretokenizer acceptance.
- **Why non-blocking:** Explicitly stated; track gold as a cross-milestone dependency in PROGRESS.

---

## Undefined quantities

### U1 — Where worker counts are set — non-blocking
- **Where:** setup step 3 (“Set rayon and datatrove worker counts to the performance-core count”).
- **Issue:** No file or env var name is specified.
- **Resolution for setup:** Export `ANCODA_WORKERS` (and document it) from the Makefile as the performance-core count; later code reads that. Not a section 3 change.

### U2 — Packaging layout for `python/ancoda_tok/` — non-blocking
- **Where:** section 4 vs setup step 5 `uv init`.
- **Issue:** Exact `pyproject.toml` package discovery (`python/` layout, maturin binding path) is not specified.
- **Resolution:** Configure a src-style / `python/` package path in `pyproject.toml` so `ancoda_tok` imports work; keep torch off the Mac.

### U3 — “Languages of Nepal” mean on Yak scoreboard — non-blocking
- **Where:** 3.10 Yak scoreboard (Yak v2 1.843).
- **Issue:** Which languages enter the mean is not defined in docs/SPEC.md.
- **Why non-blocking:** Defined by Yak’s harness output; baselines records whatever the harness reports.

### U4 — Aya / everyday-Nepali fertility aggregation — non-blocking
- **Where:** 3.9–3.10.
- **Issue:** Whether Aya fertility uses whole-file, per-line, or another harness rule is not restated locally.
- **Why non-blocking:** “via Yak’s harness”; baselines captures the harness definition and numbers.

### U5 — Seeds outside the snapshot seed — non-blocking
- **Where:** section 2 determinism rule; snapshot seed `20261104` in cleaning.
- **Issue:** Other pipeline seeds (inventory, eval samples, proxy) are “fixed” but not all listed in section 3.
- **Why non-blocking:** Later milestones can pin and record seeds in manifests without changing section 3.

### U6 — Block 1 “5% of vocab” rounding — non-blocking
- **Where:** 3.6 step 6.
- **Issue:** Caps are given as integers (1,638 / 3,276 / 6,553) matching floor(0.05 × size); the rounding rule is not named.
- **Why non-blocking:** Exact caps are listed; use those integers only.

### U7 — README / stub file contents — non-blocking
- **Where:** section 4.
- **Issue:** `README.md` and many Python modules are named but have no setup content spec.
- **Resolution:** Minimal stubs sufficient for layout and compile/import acceptance.

### U8 — First reserved ID indexing — non-blocking
- **Where:** 3.7.
- **Issue:** “From the first reserved ID” implies `total − 1024` but does not spell the formula.
- **Why non-blocking:** Fixed block sizes imply first reserved ID = total − 1,024; confirm in trainer tests.

---

## Third-party library / external assumptions

### T1 — o200k edit replacement counts (exactly 4 and 1) — non-blocking for setup
- **Where:** 3.1 Edit 1 and Edit 2.
- **Issue:** Counts are asserted against whatever string the installed `tiktoken` exposes. A tiktoken release that changes `o200k_base`’s pattern would fail the asserts.
- **Why non-blocking:** pretokenizer asserts and STOPs on mismatch; do not change section 3. Pin tiktoken via `uv.lock` after setup install; never upgrade without approval.

### T2 — `unicode-segmentation = "=1.12.0"` and Unicode 16.0.0 — non-blocking for audit
- **Where:** 3.3; setup step 8.
- **Issue:** Plan states 1.12.0 is the first release with GB9c and Unicode 16.0.0 data; 1.11.0 lacks GB9c; 1.13.0/1.13.1 yanked. setup must assert `UNICODE_VERSION == (16, 0, 0)` after the crate builds.
- **Why non-blocking:** Verification is an setup acceptance/setup step, not a spec contradiction. If the assert fails → STOP (evidence vs section 3), do not retarget the pin without human approval.

### T3 — `tokenizers` crate default `onig` feature — non-blocking
- **Where:** setup step 6 (`ancoda-seg`).
- **Issue:** Plan assumes default features include the onig Split engine matching Hugging Face `tokenizer.json`.
- **Why non-blocking:** Confirm in Cargo feature resolution during setup/pretokenizer; if defaults change, pin features explicitly without changing 3.1’s `use_regex=False` requirement.

### T4 — Cross-engine pretokenize equality (onig / Python `regex` / fancy-regex) — non-blocking
- **Where:** 3.1 “all three agree”; pretokenizer step 4.
- **Issue:** Agreement was verified on 4 Nov 2026 for cited examples; not guaranteed for all adversarial strings forever.
- **Why non-blocking:** pretokenizer STOPs on first 20 mismatches; that is the acceptance path.

### T5 — HF export equivalence (tokenizers 0.23.3, tiktoken 0.14.0) — non-blocking
- **Where:** 3.5.
- **Issue:** Verification was version-specific. Locked versions may differ slightly after `uv add`.
- **Why non-blocking:** pretokenizer/trainer re-run the equivalence tests on the locked versions; failures STOP.

### T6 — `fasttext-wheel` on Apple Silicon — non-blocking
- **Where:** setup step 5.
- **Issue:** Plan already provides fallback to `fasttext` (needs cmake) and STOP if both fail.
- **Action:** Follow that order; no section 3 change.

### T7 — `codeparrot/github-code-clean` loader — non-blocking for setup
- **Where:** data-plan.
- **Issue:** Script loader may be refused by current `datasets`; plan requires reading `data/` files directly and STOP if format/columns differ.
- **Why non-blocking:** Deferred to data-plan with an explicit STOP.

### T8 — Yak / competitor fertility reference table — non-blocking
- **Where:** 3.10 references from Yak `docs/RESULTS.md` (3 Nov 2026).
- **Issue:** Gate A uses **baselines re-measured** values, not the printed references, if they differ.
- **Why non-blocking:** Plan states baselines values are authoritative for the gate.

### T9 — sidskarki-64k license for proxy comparison — non-blocking for setup
- **Where:** baselines step 3; model-eval arm 6.
- **Issue:** Evaluation/proxy use must be confirmed from community-use-1.0 terms in baselines; STOP if disallowed or unclear.
- **Why non-blocking:** Explicit baselines STOP; does not block setup.

---

## Missing or weak acceptance criteria

### A1 — SPEC_AUDIT / DECISIONS / rules file not in setup acceptance commands — non-blocking
- **Where:** setup acceptance block.
- **Issue:** Acceptance does not `test -f` for `docs/SPEC_AUDIT.md`, `docs/DECISIONS.md`, or the local rules file.
- **Mitigation:** Still required by setup steps 1 and 4; record paths in PROGRESS; `cat docs/MACHINE.md docs/SPEC_AUDIT.md` partially covers audit presence.

### A2 — `hf auth whoami` vs human login STOP — non-blocking
- **Where:** setup STOP vs acceptance.
- **Issue:** Acceptance cannot pass until the maintainer completes `uv run hf auth login` and FLORES+ terms.
- **Why non-blocking:** Intentional human gate; agent STOPs and waits.

### A3 — No command checks free-disk / RAM thresholds — non-blocking
- **Where:** setup step 2 vs acceptance.
- **Issue:** Machine thresholds are enforced by process (STOP), not by a scripted acceptance line.
- **Mitigation:** Write exact `df` / `sysctl` output into `docs/MACHINE.md` and STOP in PROGRESS if violated.

### A4 — Empty crate tests satisfy `cargo test --workspace` — non-blocking
- **Where:** setup acceptance.
- **Issue:** Stub crates with no tests still “pass.”
- **Why non-blocking:** Acceptable for setup; real tests arrive in pretokenizer+.

### A5 — Gate A orphan-mark “every occurrence inside uncovered akshara or malformed sequence” — non-blocking
- **Where:** 3.10.
- **Issue:** “Malformed sequence” is not formally defined beyond orphan-mark context listing in trainer.
- **Why non-blocking:** trainer step 4 requires listing every occurrence with context; human/Gate A review can classify. Not required to redefine in section 3 for setup.

### A6 — Phrase-build table has no numeric pass bar — non-blocking
- **Where:** 3.10.
- **Issue:** Phrase builds are “different category”; report-only unless claiming a beat.
- **Why non-blocking:** Intentional; claims discipline in 3.11 covers misuse.

### A7 — setup does not require `caffeinate` smoke test — non-blocking
- **Where:** section 2 long-job rule.
- **Issue:** No setup check that `caffeinate` works.
- **Why non-blocking:** Used from data-plan onward; trivial OS tool on macOS.

---

## Blocking items

**None.**

No specification item requires stopping setup before the machine check, toolchain, or layout steps. Section 3 is left unchanged.

---

## Audit conclusion

| Category                         | Blocking | Non-blocking |
|----------------------------------|----------|--------------|
| Contradictions                   | 0        | 4            |
| Undefined quantities             | 0        | 8            |
| Third-party / external risks     | 0        | 9            |
| Missing / weak acceptance        | 0        | 7            |

**Proceed to setup step 2 (machine check).** If free disk &lt; 250 GB or RAM &lt; 16 GB, STOP per setup step 2 (environment), not because of this audit.

---

# Re-audit — PLAN v2.6 (4 Nov 2026)

Scope: changed parts **3.3–3.13** and milestones **baselines–release-candidate** (including new **phrase-build**). Prior setup audit items that referred to the old “undo GB9c” wording or phrase-report-only Gate A are superseded where noted. Section 3 is not changed by this audit. Each item is **blocking** or **non-blocking**.

## Summary (v2.6)

No **blocking** defects. Proceed to commit and baselines.

---

## 3.3–3.13

### R1 — Mission “akshara” vs 3.3 “not conventional orthographic akshara” — non-blocking
- **Where:** §1 mission; 3.3 definition.
- **Issue:** Mission still says “whole aksharas”; 3.3 defines an Ancoda protected unit and requires the paper to say it is not the conventional orthographic akshara.
- **Why non-blocking:** 3.3 is authoritative; mission uses the project term. Paper/model-card wording is gated by 3.11.

### R2 — Old audit C1 superseded — non-blocking (resolved)
- **Where:** setup audit C1; 3.3 v2.6.
- **Issue:** v2.6 replaces “GB9c disabled / undo” with an explicit override (“every Devanagari letter begins a new unit”) and a boundary table.
- **Why non-blocking:** Implementation step 3 matches the definition; `gb9c=true` remains raw clusters without the guarantee.

### R3 — Guarantee proof C2 vs eligible units with non-Extend Block 0 chars — non-blocking
- **Where:** 3.5 proof C2; 3.6 eligibility.
- **Issue:** C2 assumes every later code point of U is Grapheme_Cluster_Break Extend or SpacingMark. Eligibility only requires Block 0 characters (U+0900–U+097F, ZWJ, ZWNJ), which also include Lo letters and digits.
- **Why non-blocking:** A second Lo letter inside one eligible multi-code-point unit cannot occur under the 3.3 override (letters start units). Digits/danda as “later” code points of an eligible unit that started with a letter are not produced by the segmenter for well-formed text; ineligible/malformed cases are excluded from coverage. trainer exhaustive per-unit tests are the backstop; STOP if they fail.

### R4 — Letter set source files — non-blocking
- **Where:** 3.3 step 4 (UnicodeData + Scripts); setup still lists DerivedCoreProperties.
- **Issue:** Consonant/InCB reading is no longer the eligibility basis; Lo∩Devanagari is.
- **Why non-blocking:** UnicodeData.txt and Scripts.txt are already present; DerivedCoreProperties remains useful for the pretokenizer Python reference (InCB) and is not contradictory.

### R5 — Block 1 coverage may be unreachable at 99.99% — non-blocking
- **Where:** 3.6 steps 5–7.
- **Issue:** Ineligible units count in N; 99.99% may be impossible.
- **Why non-blocking:** Plan states the cap then decides; report achieved coverage exactly.

### R6 — Joiner copy accounting (49%+1%) — non-blocking
- **Where:** 3.8; section 2 new synthetic-joiner rule.
- **Issue:** Nepali natural share becomes 49% when the 1% synthetic copy is on.
- **Why non-blocking:** Explicit; ladder has a no-joiner snapshot; mix-sweep keep/drop rule is numeric.

### R7 — Morphology: whole-word no longer matches boundaries — non-blocking
- **Where:** 3.9 morphology vs whole-word preservation.
- **Issue:** Recall definition means a one-token word matches none of its reference boundaries; F1 and whole-word preservation are separate.
- **Why non-blocking:** Intentional (v2.6 changelog); both are reported, not Gate A.

### R8 — ZWJ/ZWNJ Gate A bar now beats Yak v2 (was sidskarki ceiling) — non-blocking
- **Where:** 3.10.
- **Issue:** Pass bar tightened to Yak v2 baselines values; miss → STOP (human).
- **Why non-blocking:** Explicit STOP path; joiner augmentation exists to pursue the bar.

### R9 — Languages-of-Nepal mean “not comparable” — non-blocking
- **Where:** 3.10; baselines acceptance notes.
- **Issue:** Private 14-set mean cannot be rerun; must not claim WIN/LOSE.
- **Why non-blocking:** Explicit reporting rule.

### R10 — Phrase builds moved to Gate P (3.13) — non-blocking (supersedes setup A6)
- **Where:** 3.10, 3.13, phrase-build.
- **Issue:** setup audit A6 said phrase rows had no pass bar; Gate P now defines bars.
- **Why non-blocking:** Updated by v2.4–v2.6; word-level vs phrase comparison forbidden on Gate A.

### R11 — Phrase pattern constructed from 3.1 file — non-blocking
- **Where:** 3.13 step 2.
- **Issue:** Exact regex surgery (“first two branches wrapped `{1,4}`”) depends on o200k structure remaining stable.
- **Why non-blocking:** Assert against 3.1 examples; phrase-build STOP on mismatch; do not retype the pattern.

---

## baselines–release-candidate (incl. phrase-build)

### R12 — baselines step 6 path `eval/gate_a.py` vs package module — non-blocking
- **Where:** baselines step 6; acceptance `python -m ancoda_tok.eval.gate_a`.
- **Issue:** Shorthand path vs `python/ancoda_tok/eval/gate_a.py`.
- **Why non-blocking:** Acceptance command defines the module; implement there.

### R13 — baselines reproduction table vs harness panel names — non-blocking
- **Where:** baselines acceptance; `configs/eval_panel.yaml`.
- **Issue:** Table uses “Yak v1 128k” etc.; panel names are `yak-128k-v1`, `yak-128k-v2.0`, `yak-128k-v1-phrase`.
- **Why non-blocking:** Plan allows mapping from `eval_panel.yaml`; record mapping in PROGRESS/RESULTS.

### R14 — baselines ±0.03 vs ±0.003 — non-blocking
- **Where:** baselines steps 2 and acceptance.
- **Issue:** Two different tolerances for two different checks.
- **Why non-blocking:** Clarified in v2.6; both must be recorded in `baselines_reproduction.tsv` for the ±0.003 rows.

### R15 — sidskarki community-use-1.0 for proxy — non-blocking (baselines STOP if unclear)
- **Where:** baselines step 3; model-eval arm 6.
- **Issue:** Proxy use may be disallowed.
- **Why non-blocking:** Explicit STOP; evaluation-only remains possible.

### R16 — data-plan HPLT / nepal_pool / papaya sources may be missing — non-blocking
- **Where:** data-plan source table; 3.8 pools.
- **Issue:** Config names and licensed nepal_pool sources are discover-only; papaya GitHub path must be confirmed.
- **Why non-blocking:** Plan forbids inventing names and has STOP/list paths; does not block baselines.

### R17 — pretokenizer independent Python segmenter must not call Rust — non-blocking
- **Where:** pretokenizer step 6.
- **Issue:** Dual implementation risk of shared bugs if both misread the same table.
- **Why non-blocking:** Differential test + Unicode test file; human STOP only if rule ambiguity remains (none identified as blocking now).

### R18 — cleaning four snapshots vs older three — non-blocking
- **Where:** cleaning acceptance.
- **Issue:** Adds no-joiner 50% snapshot.
- **Why non-blocking:** Explicit; storage plan in data-plan absorbs size.

### R19 — trainer RAM STOP at 14 GB peak — non-blocking
- **Where:** trainer memory note; MACHINE.md has 18 GB RAM.
- **Issue:** Trainer must stream; 14 GB headroom on an 18 GB machine is tight with OS overhead.
- **Why non-blocking:** Explicit STOP before full build; 5 GB mix if RAM &lt; 32 GB already chosen by 3.8.

### R20 — mix-sweep Unigram control not algorithmic BPE-vs-Unigram — non-blocking
- **Where:** mix-sweep ladder note.
- **Issue:** Different pretokenization; paper must not overclaim.
- **Why non-blocking:** Plan states the limit.

### R21 — phrase-build / Gate P schedule vs slip rules — non-blocking
- **Where:** 3.12; phrase-build.
- **Issue:** Phrase-128k is cut before full GPU tier if slipping.
- **Why non-blocking:** Ordered; never cut guarantee/equality/segmenter/Gate A core.

### R22 — model-eval fixed-byte + 3 seeds for arms 1–2 — non-blocking
- **Where:** model-eval tiers; Gate B.
- **Issue:** Budget ~110 H100-hours planning figure; two-seed results labelled preliminary.
- **Why non-blocking:** Pilot STOP confirms budget; labelling rules are clear.

### R23 — release-candidate packages three builds — non-blocking
- **Where:** release-candidate step 3.
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
| baselines–release-candidate / phrase-build | 0 | 12 |

**Proceed:** commit “Update to PLAN v2.6”, then execute baselines.

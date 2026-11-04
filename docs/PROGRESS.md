## M0 — Audit, environment and repo — done
- [x] 1. Specification audit → `docs/SPEC_AUDIT.md` (0 blocking)
- [x] 2. Machine check → `docs/MACHINE.md` (PASS: 156 GiB free ≥ 100 GB; 18.0 GB RAM; 5 perf cores)
- [x] 3. Toolchain: skipped brew/xcode/git/cmake/pkg-config (present); installed uv 0.13.7, rustup stable **1.99.1**, Python 3.11.18; wrote `rust-toolchain.toml`; Makefile exports `HF_HOME`, `ANCODA_WORKERS=5`, `RAYON_NUM_THREADS`, `DATATROVE_WORKERS`
- [x] 4. Section 4 layout; `.cursor/rules/ancoda-tokenizer.mdc` (exact section 2 text); `docs/DECISIONS.md` (section 3 copy)
- [x] 5. Python deps via uv (incl. `fasttext-wheel`); no torch
- [x] 6. Rust workspace `ancoda-seg` + `ancoda-bpe`; `unicode-segmentation = "=1.12.0"`; tokenizers brings `onig`
- [x] 7. Cloned `third_party/yak-tokenizer` @ `9fe03a991dd7c44e078d574e4ffcd51a7606dd6a`; `uv sync --extra eval` (see `docs/YAK_HARNESS.md`)
- [x] 8. Unicode 16.0.0 UCD files downloaded; sha256 in `docs/UNICODE_16.0.0_SHA256.md`; crate `UNICODE_VERSION` assert **(16, 0, 0)** passed
- [x] 9. Makefile stub targets listed in PLAN.md M0 step 9
- [x] HF auth: existing token wired into gitignored `data/hf_cache` via project `.env` (`HF_HOME`); FLORES+ streaming load succeeded (terms already accepted) — no human STOP required

Acceptance:
- `cat docs/MACHINE.md docs/SPEC_AUDIT.md` → both present (45 + 187 lines); machine PASS
- `uv run python -c "import tokenizers, tiktoken, datasets, fasttext, ahocorasick, sentencepiece; print('ok')"` → `ok`
- `cargo build --workspace && cargo test --workspace` → build ok; tests `ok. 1 passed` (`unicode_version_is_16_0_0`)
- `uv run hf auth whoami` → `user=SirAlchemist1 orgs=ancodalabs`
- `git ls-files | grep -E "Cargo.lock|uv.lock|rust-toolchain.toml"` →
  ```
  Cargo.lock
  rust-toolchain.toml
  uv.lock
  ```

Needs the human:
- None for M0. Next: commit this milestone, then start a new chat for M1.

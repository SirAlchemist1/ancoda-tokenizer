# Ancoda Nepali Tokenizer — Makefile
# HF downloads stay on the repo drive. Worker count = performance cores (MACHINE.md).

export HF_HOME := $(CURDIR)/data/hf_cache
export ANCODA_WORKERS ?= 5
export RAYON_NUM_THREADS ?= $(ANCODA_WORKERS)
export DATATROVE_WORKERS ?= $(ANCODA_WORKERS)

.PHONY: setup storage-plan download clean-data snapshot inventory counts train export eval gate-a efficiency proxy-pilot proxy-train gate-b release

setup:
	uv sync
	cargo build --workspace
	uv run maturin develop --manifest-path crates/ancoda-seg/Cargo.toml --features python

storage-plan:
	uv run python -m ancoda_tok.storage

download:
	caffeinate -dimsu uv run python -u -m ancoda_tok.download

clean-data:
	@echo "clean-data: stub — implemented in M4"

snapshot:
	@echo "snapshot: stub — implemented in M4"

inventory:
	@echo "inventory: stub — implemented in M5"

counts:
	@echo "counts: stub — implemented in M6"

train:
	@echo "train: stub — implemented in M6"

export:
	@echo "export: stub — implemented in M6"

eval:
	@echo "eval: stub — implemented in M1/M8"

gate-a:
	@echo "gate-a: stub — implemented in M8"

efficiency:
	@echo "efficiency: stub — implemented in M8"

proxy-pilot:
	@echo "proxy-pilot: stub — implemented in M9"

proxy-train:
	@echo "proxy-train: stub — implemented in M9"

gate-b:
	@echo "gate-b: stub — implemented in M9"

release:
	@echo "release: stub — implemented in M10"

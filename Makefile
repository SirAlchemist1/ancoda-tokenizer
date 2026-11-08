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
	caffeinate -dimsu uv run python -u -m ancoda_tok.clean all

snapshot:
	caffeinate -dimsu uv run python -u -m ancoda_tok.clean snapshot

inventory:
	caffeinate -dimsu uv run python -u -m ancoda_tok.inventory

counts:
	caffeinate -dimsu uv run python -u -m ancoda_tok.counts

train:
	caffeinate -dimsu cargo run --release --offline -p ancoda-bpe -- train --config configs/build_64k.yaml

export:
	uv run python -u -m ancoda_tok.export_cli

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

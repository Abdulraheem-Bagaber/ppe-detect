.DEFAULT_GOAL := help
UV ?= uv

# Which torch build this machine holds. Written by `make cpu` / `make gpu` and
# remembered in a gitignored file, so every target below selects the same
# extra and no command can silently revert the other one.
TORCH := $(shell cat .torch 2>/dev/null || echo cpu)
UVRUN := $(UV) run --extra $(TORCH)

.PHONY: help install hooks cpu gpu test test-fast lint format typecheck check \
        prepare-data train eval export benchmark serve viewer clean

help:  ## Show available targets
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'
	@echo "\n  torch build in use: $(TORCH)"

install:  ## Create the venv and install everything
	$(UV) sync --extra $(TORCH) --dev

hooks:  ## Install the pre-commit git hooks
	$(UVRUN) pre-commit install

gpu:  ## Switch to the CUDA torch build, and remember the choice
	@echo gpu > .torch
	$(UV) sync --extra gpu --dev
	@$(UV) run --extra gpu python -c \
		"import torch; print('torch', torch.__version__, '| cuda', torch.cuda.is_available())"

cpu:  ## Switch back to the CPU build, which is what the Docker image uses
	@echo cpu > .torch
	$(UV) sync --extra cpu --dev

test:  ## Run the full test suite with coverage
	$(UVRUN) pytest

test-fast:  ## Skip anything marked slow, gpu or model
	$(UVRUN) pytest -m "not slow and not gpu and not model"

lint:  ## Lint and check formatting
	$(UVRUN) ruff check .
	$(UVRUN) ruff format --check .

format:  ## Apply formatting and safe lint fixes
	$(UVRUN) ruff format .
	$(UVRUN) ruff check --fix .

typecheck:  ## Static type check
	$(UVRUN) mypy

check: lint typecheck test  ## Everything CI runs

prepare-data:  ## Phase 1: convert a source dataset to YOLO format
	$(UVRUN) python scripts/prepare_data.py

train:  ## Phase 2: fine-tune the baseline detector
	$(UVRUN) python scripts/train.py

eval:  ## Phase 5: violation-level precision, recall and error breakdown
	$(UVRUN) python scripts/evaluate.py

export:  ## Phase 6: export to ONNX and verify parity
	$(UVRUN) python scripts/export_onnx.py

benchmark:  ## Phase 7: latency and throughput across backends
	$(UVRUN) python scripts/benchmark.py

serve:  ## Run the API locally
	$(UVRUN) uvicorn ppe.api.app:create_app --factory --reload

viewer:  ## Run the local debug viewer at http://127.0.0.1:8000
	# Scoped to this command only; make does not export it to the caller.
	PPE_MODEL__WEIGHTS=runs/train/baseline/weights/best.pt \
	$(UVRUN) uvicorn ppe.api.app:create_app --factory --port 8000

clean:  ## Remove caches and build artefacts
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage coverage.xml
	find . -type d -name __pycache__ -prune -exec rm -rf {} +

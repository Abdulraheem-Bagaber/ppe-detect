.DEFAULT_GOAL := help
UV ?= uv

.PHONY: help install hooks test test-fast lint format typecheck check gpu \
        prepare-data train eval export benchmark serve clean

help:  ## Show available targets
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

install:  ## Create the venv and install everything, CPU torch by default
	$(UV) sync --dev

hooks:  ## Install the pre-commit git hooks
	$(UV) run pre-commit install

gpu:  ## Swap in CUDA torch for local training (see README)
	# NOTE: plain `uv run` re-syncs from uv.lock and silently reverts this,
	# which is why the ML targets below pass --no-sync.
	$(UV) pip install --python .venv --reinstall \
		--index-url https://download.pytorch.org/whl/cu126 torch torchvision

test:  ## Run the full test suite with coverage
	$(UV) run pytest

test-fast:  ## Skip anything marked slow, gpu or model
	$(UV) run pytest -m "not slow and not gpu and not model"

lint:  ## Lint and check formatting
	$(UV) run ruff check .
	$(UV) run ruff format --check .

format:  ## Apply formatting and safe lint fixes
	$(UV) run ruff format .
	$(UV) run ruff check --fix .

typecheck:  ## Static type check
	$(UV) run mypy

check: lint typecheck test  ## Everything CI runs

prepare-data:  ## Phase 1: convert a source dataset to YOLO format
	$(UV) run --no-sync python scripts/prepare_data.py

train:  ## Phase 2: fine-tune the baseline detector
	$(UV) run --no-sync python scripts/train.py

eval:  ## Phase 5: violation-level precision, recall and error breakdown
	$(UV) run --no-sync python scripts/evaluate.py

export:  ## Phase 6: export to ONNX and verify parity
	$(UV) run --no-sync python scripts/export_onnx.py

benchmark:  ## Phase 7: latency and throughput across backends
	$(UV) run --no-sync python scripts/benchmark.py

serve:  ## Phase 6: run the API locally
	$(UV) run uvicorn ppe.api.app:create_app --factory --reload

clean:  ## Remove caches and build artefacts
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage coverage.xml
	find . -type d -name __pycache__ -prune -exec rm -rf {} +

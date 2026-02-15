# ═══════════════════════════════════════════════════════════════════════
# StrandWeaver Makefile
# ═══════════════════════════════════════════════════════════════════════
PYTHON ?= python3
.PHONY: test lint format typecheck syntax install dev-install clean help

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*##' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

# ── Install ───────────────────────────────────────────────────────────
install: ## Install package
	$(PYTHON) -m pip install -e .

dev-install: ## Install package + dev dependencies
	$(PYTHON) -m pip install -r requirements-dev.txt
	$(PYTHON) -m pip install -e .

# ── Test ──────────────────────────────────────────────────────────────
test: ## Run pytest
	$(PYTHON) -m pytest tests/ -v --tb=short

test-cov: ## Run pytest with coverage
	$(PYTHON) -m pytest tests/ -v --tb=short --cov=strandweaver --cov-report=term-missing

# ── Lint ──────────────────────────────────────────────────────────────
lint: ## Run flake8 (fatal errors only)
	$(PYTHON) -m flake8 strandweaver/ --count --select=E9,F63,F7,F822 --show-source --statistics

lint-full: ## Run flake8 (all style checks)
	$(PYTHON) -m flake8 strandweaver/ --count --max-line-length=100 \
		--extend-ignore=E203,E501,W503,E741 --statistics

typecheck: ## Run mypy type checking
	$(PYTHON) -m mypy strandweaver/ --ignore-missing-imports --no-implicit-optional

syntax: ## Verify all .py files parse
	$(PYTHON) -c "import ast, pathlib; [ast.parse(f.read_text()) for f in pathlib.Path('strandweaver').rglob('*.py')]; print('All files parse OK')"

# ── Format ────────────────────────────────────────────────────────────
format: ## Auto-format with black + isort
	$(PYTHON) -m isort strandweaver/ tests/
	$(PYTHON) -m black strandweaver/ tests/

format-check: ## Check formatting without changing files
	$(PYTHON) -m isort --check-only --diff strandweaver/ tests/
	$(PYTHON) -m black --check --diff strandweaver/ tests/

# ── CI (runs what GitHub Actions runs) ────────────────────────────────
ci: lint syntax test ## Run full CI locally (lint + syntax + test)

# ── Cleanup ───────────────────────────────────────────────────────────
clean: ## Remove build artifacts
	rm -rf build/ dist/ *.egg-info strandweaver/*.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true

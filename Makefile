.DEFAULT_GOAL := help
.PHONY: help install install-dev install-observability test test-unit test-integration test-verbose test-coverage test-coverage-report lint format typecheck build clean dist-clean run shell

# Colors for output
BLUE := \033[0;34m
GREEN := \033[0;32m
YELLOW := \033[0;33m
NC := \033[0m # No Color

##@ Help

help: ## Display this help message
	@echo "$(BLUE)llm-conductor$(NC) - Development Makefile"
	@echo ""
	@awk 'BEGIN {FS = ":.*##"; printf "Usage:\n  make $(GREEN)<target>$(NC)\n"} /^[a-zA-Z_-]+:.*?##/ { printf "  $(GREEN)%-20s$(NC) %s\n", $$1, $$2 } /^##@/ { printf "\n$(YELLOW)%s$(NC)\n", substr($$0, 5) } ' $(MAKEFILE_LIST)

##@ Installation & Setup

install: ## Install package and dependencies
	@echo "$(BLUE)Installing llm-conductor...$(NC)"
	uv sync

install-dev: ## Install with dev dependencies
	@echo "$(BLUE)Installing llm-conductor with dev dependencies...$(NC)"
	uv sync --group dev

install-observability: ## Install with observability extras
	@echo "$(BLUE)Installing llm-conductor with observability support...$(NC)"
	uv sync --extra observability

##@ Testing

test: ## Run all tests
	@echo "$(BLUE)Running all tests...$(NC)"
	uv run pytest

test-unit: ## Run only unit tests
	@echo "$(BLUE)Running unit tests...$(NC)"
	uv run pytest tests/unit/

test-integration: ## Run only integration tests
	@echo "$(BLUE)Running integration tests...$(NC)"
	uv run pytest tests/integration/

test-verbose: ## Run tests with verbose output
	@echo "$(BLUE)Running tests (verbose)...$(NC)"
	uv run pytest -v

test-coverage: ## Run tests with coverage report
	@echo "$(BLUE)Running tests with coverage...$(NC)"
	uv run pytest --cov --cov-report=term --cov-report=html
	@echo "$(GREEN)✓ Coverage report generated in htmlcov/index.html$(NC)"

test-coverage-report: ## Open coverage report in browser
	@echo "$(BLUE)Opening coverage report...$(NC)"
	@open htmlcov/index.html || xdg-open htmlcov/index.html || echo "$(YELLOW)Please open htmlcov/index.html manually$(NC)"

##@ Code Quality

lint: ## Run linting checks
	@echo "$(BLUE)Running ruff linting...$(NC)"
	uv run ruff check src/ tests/

lint-fix: ## Run linting checks with auto-fix
	@echo "$(BLUE)Running ruff linting with auto-fix...$(NC)"
	uv run ruff check --fix src/ tests/

format: ## Auto-format code
	@echo "$(BLUE)Formatting code with ruff...$(NC)"
	uv run ruff format src/ tests/

format-check: ## Check code formatting without modifying files
	@echo "$(BLUE)Checking code formatting...$(NC)"
	uv run ruff format --check src/ tests/

typecheck: ## Run type checking
	@echo "$(YELLOW)Type checking not configured yet$(NC)"
	@echo "Consider adding: mypy or pyright"

pre-commit-install: ## Install pre-commit hooks
	@echo "$(BLUE)Installing pre-commit hooks...$(NC)"
	uv run pre-commit install
	@echo "$(GREEN)✓ Pre-commit hooks installed$(NC)"

pre-commit-run: ## Run pre-commit hooks on all files
	@echo "$(BLUE)Running pre-commit hooks...$(NC)"
	uv run pre-commit run --all-files

pre-commit-update: ## Update pre-commit hooks
	@echo "$(BLUE)Updating pre-commit hooks...$(NC)"
	uv run pre-commit autoupdate

##@ Building & Packaging

build: ## Build the package
	@echo "$(BLUE)Building package...$(NC)"
	uv build

clean: ## Remove build artifacts
	@echo "$(BLUE)Cleaning build artifacts...$(NC)"
	rm -rf dist/
	rm -rf build/
	rm -rf *.egg-info
	rm -rf .eggs/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name '*.pyc' -delete
	find . -type f -name '*.pyo' -delete

dist-clean: clean ## Deep clean (build + cache files)
	@echo "$(BLUE)Deep cleaning...$(NC)"
	rm -rf .pytest_cache/
	rm -rf .ruff_cache/
	rm -rf .mypy_cache/
	rm -rf .uv/
	rm -rf mlruns/
	rm -rf .local/
	rm -rf htmlcov/
	rm -rf .coverage

##@ Development

run: ## Run the CLI tool (use ARGS="..." for arguments)
	@echo "$(BLUE)Running llm-conductor...$(NC)"
	uv run llm-conductor $(ARGS)

shell: ## Start Python REPL with package available
	@echo "$(BLUE)Starting Python shell...$(NC)"
	uv run python

##@ Utility

version: ## Show package version
	@uv run python -c "from importlib.metadata import version; print(version('llm-conductor'))"

list-providers: ## List available providers
	@uv run llm-conductor list

watch: ## Watch for file changes and run tests
	@echo "$(YELLOW)Watch mode not configured yet$(NC)"
	@echo "Consider adding: pytest-watch or similar"

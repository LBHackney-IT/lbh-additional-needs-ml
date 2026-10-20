.PHONY: help install test test-cov lint fix build run-docker smoke-test clean

# Default variables
IMAGE_NAME  ?= housing-additional-needs-ml
AWS_PROFILE ?= data-platform-housing-staging
WHERE       ?=

help: ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}'

# ------------------------------------------------------------------------------
# LOCAL DEVELOPMENT & TESTING
# ------------------------------------------------------------------------------

install: ## Install package dependencies locally in editable mode with dev tools
	pip install -e ".[dev]"

test: ## Run unit and integration tests with pytest
	pytest -v tests/

test-cov: ## Run unit tests with coverage reporting
	pytest -v --cov=src --cov-report=term-missing tests/

lint: ## Run ruff formatting and lint checks
	ruff check .
	ruff format --check .

fix: ## Automatically fix ruff lint and formatting issues
	ruff check --fix .
	ruff format .

# ------------------------------------------------------------------------------
# DOCKER & CONTAINER EXECUTION
# ------------------------------------------------------------------------------

# Ensures local bind-mount directories exist before running Docker
data/models data/output ~/.aws/sso/cache:
	@mkdir -p $@

build: ## Build the Docker container image
	docker build -t $(IMAGE_NAME) .

run-docker: data/models data/output ~/.aws/sso/cache ## Run the Docker container with custom ARGS (e.g., make run-docker ARGS="--batch-size 10")
	docker run --rm \
		--user "$$(id -u):$$(id -g)" \
		-v "$$PWD/data/models:/opt/ml/processing/model:ro" \
		-v "$$PWD/data/output:/opt/ml/processing/output:rw" \
		-v "$$HOME/.aws:/root/.aws:ro" \
		-v "$$HOME/.aws/sso/cache:/root/.aws/sso/cache:rw" \
		-e AWS_PROFILE=$(AWS_PROFILE) \
		$(IMAGE_NAME) $(ARGS)

smoke-test: ## Run a container smoke test for a single note (override with WHERE="note_id = '123'")
	@test -n "$(WHERE)" || (echo "Usage: make smoke-test WHERE=\"note_id = '<note-id>'\"" && exit 1)
	@$(MAKE) run-docker ARGS="--where \"$(WHERE)\""

clean: ## Remove build artifacts, cache files, and test output
	rm -rf .pytest_cache .coverage htmlcov .ruff_cache build/ dist/ *.egg-info
	rm -rf data/output/*
	find . -type d -name "__pycache__" -exec rm -rf {} +
# Makefile for Project Automation

.PHONY: install lint type-check test docs serve-docs build all clean

# Variables
PACKAGE_NAME = langsmith_mcp_server
TEST_DIR = tests

# Default target
all: lint format type-check test docs

# Install project dependencies
install:
	uv sync

# Linting and Formatting Checks
lint:
	uv run ruff check $(PACKAGE_NAME) $(TEST_DIR)
	uv run black --check $(PACKAGE_NAME) $(TEST_DIR)
	uv run isort --check-only $(PACKAGE_NAME) $(TEST_DIR)

format:
	uv run ruff check $(PACKAGE_NAME) $(TEST_DIR) --fix
	uv run black $(PACKAGE_NAME) $(TEST_DIR)
	uv run isort $(PACKAGE_NAME) $(TEST_DIR)

# Type Checking with MyPy
type-check:
	uv run mypy $(PACKAGE_NAME) $(TEST_DIR)

# Run Tests with Coverage
test:
	uv run pytest --cov=$(PACKAGE_NAME) --cov-report=xml $(TEST_DIR)/

# Build Documentation using MkDocs
docs:
	uv run mkdocs build

# Serve Documentation Locally
serve-docs:
	uv run mkdocs serve

# Run Pre-Commit Hooks
pre-commit:
	uv run pre-commit run --all-files

# Clean Up Generated Files
clean:
	rm -rf dist/
	rm -rf build/
	rm -rf *.egg-info
	rm -rf htmlcov/
	rm -rf .mypy_cache/
	rm -rf .pytest_cache/
	rm -rf .ruff_cache/
	rm -rf site/

# Build the Package
build:
	uv build

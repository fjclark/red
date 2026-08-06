package_dir := "red"
test_args := "-v --cov=red --cov-report=term --cov-report=xml --junitxml=unit.xml --color=yes"

# List available recipes
default:
    @just --list

# Create the virtual environment and install the package + dev tools
install:
    uv sync
    uv run pre-commit install || true

# Lint the code with ruff
lint:
    uv run ruff check {{package_dir}}

# Autoformat the code and sort imports with ruff
format:
    uv run ruff format {{package_dir}}
    uv run ruff check --fix --select I {{package_dir}}

# Run the test suite with coverage
test:
    uv run pytest {{test_args}} {{package_dir}}/tests/

# Type check the code with mypy
type-check:
    uv run mypy --follow-imports=silent --ignore-missing-imports --strict {{package_dir}}

# Build the documentation
docs-build:
    uv run mkdocs build

# Deploy the documentation for the given version
docs-deploy version:
    uv run mike deploy --push --update-aliases {{version}}

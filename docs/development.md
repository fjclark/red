# Development

## Writing Code

To create a development environment, you must have [`uv` installed](https://docs.astral.sh/uv/getting-started/installation/) and [`just` installed](https://github.com/casey/just#installation).

A development environment can be created (and pre-commit hooks installed) with:

```shell
just install
```

This creates a virtual environment in `.venv`. Prefix commands with `uv run`, or activate the environment with `source .venv/bin/activate`.

Some handy `just` commands are available:
```shell
just lint # Lint the codebase with Ruff
just format # Format the codebase with Ruff
just type-check # Type-check the codebase with Mypy
just test # Run the unit tests with Pytest
```

To serve the documentation locally:

```shell
uv run mkdocs serve
```

## Publishing

### PyPI

There is a GitHub Actions workflow that will automatically publish to PyPI when a new tag is pushed:
```shell
git tag <new version>
git push origin <new version>
```

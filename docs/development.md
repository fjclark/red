# Development

To create a development environment, you must have [`mamba` installed](https://mamba.readthedocs.io/en/latest/installation/mamba-installation.html).

A development conda environment can be created and activated with:

```shell
make env
mamba activate red
```

Some handy `make` commands are available:
```shell
make lint # Lint the codebase with Ruff
make format # Format the codebase with Ruff
make type-check # Type-check the codebase with Mypy
make test # Run the unit tests with Pytest
```

To serve the documentation locally:

```shell
mkdocs serve
```

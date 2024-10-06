PACKAGE_NAME := red
PACKAGE_DIR  := red

# Set CONDA_ENV_RUN to empty if SKIP_CONDA_ENV_RUN is not set and GITHUB_ACTIONS is set
# For the CI github actions workflow, we skip "make env" and set up the environment manually. In this case,
# it's helpful to to set CONDA_ENV_RUN to be empty. However, for the documentation workflow, we want to override
# this and keep the normal behavior. We override this by setting KEEP_CONDA_ENV_RUN to true in the documentation workflow.
SKIP_CONDA_ENV = $(and $(GITHUB_ACTIONS),$(not $(KEEP_CONDA_ENV_RUN)))
CONDA_ENV_RUN = $(if $(SKIP_CONDA_ENV),,conda run --no-capture-output --name $(PACKAGE_NAME))

TEST_ARGS := -v --cov=$(PACKAGE_NAME) --cov-report=term --cov-report=xml --junitxml=unit.xml --color=yes

.PHONY: env lint format test type-check docs-build docs-deploy

env:
	mamba create     --name $(PACKAGE_NAME)
	mamba env update --name $(PACKAGE_NAME) --file devtools/conda-envs/test.yaml
	$(CONDA_ENV_RUN) pip install --no-deps -e .
	$(CONDA_ENV_RUN) pre-commit install || true

lint:
	$(CONDA_ENV_RUN) ruff check $(PACKAGE_DIR)

format:
	$(CONDA_ENV_RUN) ruff format $(PACKAGE_DIR)
	$(CONDA_ENV_RUN) ruff check --fix --select I $(PACKAGE_DIR)

test:
	$(CONDA_ENV_RUN) pytest -v $(TEST_ARGS) $(PACKAGE_DIR)/tests/

type-check:
	$(CONDA_ENV_RUN) mypy --follow-imports=silent --ignore-missing-imports --strict $(PACKAGE_DIR)

docs-build:
	$(CONDA_ENV_RUN) mkdocs build

docs-deploy:
ifndef VERSION
	$(error VERSION is not set)
endif
	$(CONDA_ENV_RUN) mike deploy --push --update-aliases $(VERSION)

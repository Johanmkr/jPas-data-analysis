#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = jpas_da
PYTHON_VERSION = 3.10
PYTHON_INTERPRETER = python

#################################################################################
# COMMANDS                                                                      #
#################################################################################


## Install Python dependencies
.PHONY: requirements
requirements:
	uv sync
	



## Delete all compiled Python files
.PHONY: clean
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete


## Lint using ruff (use `make format` to do formatting)
.PHONY: lint
lint:
	ruff format --check
	ruff check

## Format source code with ruff
.PHONY: format
format:
	ruff check --fix
	ruff format



## Run tests
.PHONY: test
test:
	python -m pytest tests


## Set up Python interpreter environment
.PHONY: create_environment
create_environment:
	uv venv --python $(PYTHON_VERSION)
	@echo ">>> New uv virtual environment created. Activate with:"
	@echo ">>> Windows: .\\\\.venv\\\\Scripts\\\\activate"
	@echo ">>> Unix/macOS: source ./.venv/bin/activate"
	



#################################################################################
# PROJECT RULES                                                                 #
#################################################################################


## Raw export -> interim object table (flux, sigma, mask) + band scales
.PHONY: data
data: requirements
	$(PYTHON_INTERPRETER) -m jpas_da.dataset


## Interim object table -> asinh representation (x, sigma_x, mask)
.PHONY: features
features: data
	$(PYTHON_INTERPRETER) -m jpas_da.features


## Leave-one-tile-out fold assignment
.PHONY: splits
splits: data
	$(PYTHON_INTERPRETER) -m jpas_da.splits


## Everything the models need: data, features, splits
.PHONY: datasets
datasets: features splits


## Open the marimo notebook inside the project environment
.PHONY: notebook
notebook: requirements
	uv run marimo edit $(NOTEBOOK)


#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

define PRINT_HELP_PYSCRIPT
import re, sys; \
lines = '\n'.join([line for line in sys.stdin]); \
matches = re.findall(r'\n## (.*)\n[\s\S]+?\n([a-zA-Z_-]+):', lines); \
print('Available rules:\n'); \
print('\n'.join(['{:25}{}'.format(*reversed(match)) for match in matches]))
endef
export PRINT_HELP_PYSCRIPT

help:
	@$(PYTHON_INTERPRETER) -c "${PRINT_HELP_PYSCRIPT}" < $(MAKEFILE_LIST)

#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = jpas_da
PYTHON_VERSION = 3.10
# `uv run` guarantees the project environment regardless of which shell starts
# make, and is a no-op inside the Docker image (UV_NO_SYNC=1 there).
PYTHON_INTERPRETER ?= uv run python

# Config file and per-run overrides for the VAE targets, e.g.
#   make train-debug SET="--set model.hidden=[64,32] --set loss.name=student_t"
CONFIG ?= configs/vae.yaml
SET ?=

# Cluster job parameters (see cluster/job.yaml.tmpl and `make job`).
IMAGE_TAG ?= $(shell git rev-parse --short HEAD 2>/dev/null || echo dev)
IMAGE ?= jpas-da:$(IMAGE_TAG)
TARGET ?= train
GPU_COUNT ?= 1

# `make train DEBUG=1` is the smoke path the objective names; `--debug` caps
# the run at train.debug_epochs on train.debug_objects.
DEBUG ?= 0
TRAIN_FLAGS = $(if $(filter 1 true yes,$(DEBUG)),--debug,)

#################################################################################
# COMMANDS                                                                      #
#################################################################################


## Install Python dependencies
.PHONY: requirements
requirements:
ifneq ($(UV_NO_SYNC),1)
	uv sync
endif
	



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
	$(PYTHON_INTERPRETER) -m pytest tests


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
.PHONY: interim
interim: requirements
	$(PYTHON_INTERPRETER) -m jpas_da.dataset


## Interim object table -> asinh representation (x, sigma_x, mask)
.PHONY: features
features: interim
	$(PYTHON_INTERPRETER) -m jpas_da.features


## Leave-one-tile-out fold assignment
.PHONY: splits
splits: interim
	$(PYTHON_INTERPRETER) -m jpas_da.splits


## Everything the models need: interim table, representation, folds
.PHONY: datasets
datasets: features splits


## Model-ready arrays: RMS shape/scale decomposition onto the fold assignment
.PHONY: model-data
model-data: datasets
	$(PYTHON_INTERPRETER) -m jpas_da.modeling.data


## Raw export -> the final model-ready dataset under data/processed
.PHONY: data
data: model-data
	@echo ">>> data/processed: $$(ls data/processed | tr '\n' ' ')"


## Train the VAE (CONFIG/SET override configs/vae.yaml; DEBUG=1 for a tiny run)
.PHONY: train
train: data
	$(PYTHON_INTERPRETER) -m jpas_da.modeling.train --config $(CONFIG) $(TRAIN_FLAGS) $(SET)


## Two-epoch run on a small subset, for smoke-testing the pipeline
.PHONY: train-debug
train-debug: data
	$(PYTHON_INTERPRETER) -m jpas_da.modeling.train --config $(CONFIG) --debug $(SET)


## Encode every object with the trained model (needs models/vae.pt already)
.PHONY: predict
predict: data
	$(PYTHON_INTERPRETER) -m jpas_da.modeling.predict


## Build the CPU training image
.PHONY: docker-build
docker-build:
	docker build -t jpas-da:dev .


## Run the debug training inside the image, with the host data mounted
.PHONY: docker-train
docker-train: docker-build
	docker run --rm \
		-v "$(CURDIR)/data:/app/data" \
		-v "$(CURDIR)/models:/app/models" \
		-v "$(CURDIR)/reports:/app/reports" \
		jpas-da:dev make train-debug


## Render the cluster Job YAML (writes cluster/jobs/<job>.yaml, not committed)
.PHONY: job
job:
	$(PYTHON_INTERPRETER) -m jpas_da.cluster \
		--image $(IMAGE) --target $(TARGET) --gpu-count $(GPU_COUNT) \
		--output cluster/jobs/$(TARGET)-$(IMAGE_TAG).yaml


## Submit the rendered Job. Runs on the cluster: only when you ask for it.
.PHONY: submit
submit: job
	kubectl apply -f cluster/jobs/$(TARGET)-$(IMAGE_TAG).yaml


## Job and pod state for the current TARGET/IMAGE_TAG
.PHONY: job-status
job-status:
	kubectl get jobs,pods -l app=jpas-da


## Follow the job's logs
.PHONY: job-logs
job-logs:
	kubectl logs -f job/$(TARGET)-$(IMAGE_TAG)


## Delete the job and its pod
.PHONY: job-delete
job-delete:
	kubectl delete job $(TARGET)-$(IMAGE_TAG)


## Run the smoke path: data, debug training, tests
.PHONY: smoke
smoke: train-debug test

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

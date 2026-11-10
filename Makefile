#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = jpas_da
PYTHON_VERSION = 3.10
# `uv run` guarantees the project environment whatever shell starts make. In the
# image UV_NO_SYNC=1 and /app/.venv/bin is on PATH, so the venv's `python` is
# used and nothing is re-resolved.
PYTHON_INTERPRETER ?= uv run python

# Config file and per-run overrides for the VAE targets, e.g.
#   make train-debug SET="--set model.hidden=[64,32] --set loss.name=student_t"
CONFIG ?= configs/vae.yaml
SET ?=

# `make train DEBUG=1` is the smoke path the objective names; `--debug` caps
# the run at train.debug_epochs on train.debug_objects.
DEBUG ?= 0
TRAIN_FLAGS = $(if $(filter 1 true yes,$(DEBUG)),--debug,)

#################################################################################
# WHERE A TARGET RUNS                                                           #
#################################################################################
# One target, three places. `make run TARGET=train` is the only entry point that
# differs: locally it is `make train`, inside the image it is
# `docker run ... make train`, on Springfield it is a Job whose command is
# ["make", "train"]. Paths under /app are identical in all three, and the
# cluster mounts the claim at the same subPaths (see cluster/job.yaml.tmpl).

# local -> this machine. docker -> the image, so the cluster path is rehearsed.
RUNNER ?= local
DOCKER ?= docker
# Pass --gpus all only on a host with the NVIDIA runtime; the container then
# sees what a Springfield pod sees (torch picks cuda in `train.device: auto`).
DOCKER_GPU ?=

REGISTRY ?= johanmkr
IMAGE_NAME ?= $(REGISTRY)/jpas-da
# The tag names the code in the image. `-dirty` is appended when the tree does
# not match HEAD (unstaged, staged or untracked changes), so an image built from
# uncommitted code can never masquerade as the commit's image.
GIT_STATE = $(shell git rev-parse --short HEAD 2>/dev/null || echo dev)$(shell git status --porcelain 2>/dev/null | grep -q . && echo -dirty)
IMAGE_TAG ?= $(GIT_STATE)
IMAGE ?= $(IMAGE_NAME):$(IMAGE_TAG)

# Springfield nodes are x86_64; build for that even from an ARM laptop.
PLATFORM ?= linux/amd64

# Mount the host tree the image expects: inputs in, artefacts out. Existing
# files stay owned by you (`--user`), HOME is writable for caches.
DOCKER_MOUNTS = -v "$(CURDIR)/data:/app/data" \
                -v "$(CURDIR)/models:/app/models" \
                -v "$(CURDIR)/reports:/app/reports"
DOCKER_FLAGS = --rm $(DOCKER_GPU) $(DOCKER_MOUNTS) \
               --user "$$(id -u):$$(id -g)" -e HOME=/tmp \
               -e CONFIG="$(CONFIG)" -e SET="$(SET)" -e DEBUG="$(DEBUG)" \
               $(if $(wildcard .env),--env-file .env,)

#################################################################################
# CLUSTER JOB PARAMETERS                                                        #
#################################################################################
# See cluster/job.yaml.tmpl and `make job`; every value is a make variable.
TARGET ?= train
# Job name = make target + the image's tag, so a job records which code it ran.
# Derived from IMAGE, not IMAGE_TAG: overriding IMAGE must rename the job too.
JOB_TAG = $(lastword $(subst :, ,$(IMAGE)))
JOB ?= $(TARGET)-$(JOB_TAG)
GPU_COUNT ?= 1
CPU ?= 4
MEMORY ?= 16Gi
DEADLINE ?= 86400
# Directory inside the claim (Springfield: the `storage` PVC, mounted at /storage).
STORAGE_ROOT ?= projects/jpas_da
SPRINGFIELD ?= springfield
FRINK ?= frink

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


## Raw export -> interim object table (flux, sigma, mask, holdout, quality)
.PHONY: interim
interim: requirements
	$(PYTHON_INTERPRETER) -m jpas_da.dataset


## Tile rotations (train/val/test) plus holdout and excluded rows
.PHONY: splits
splits: interim
	$(PYTHON_INTERPRETER) -m jpas_da.splits


## Interim table -> model-ready arrays (broad-band RMS normalisation)
.PHONY: features
features: interim
	$(PYTHON_INTERPRETER) -m jpas_da.features


## Raw export -> the final model-ready dataset under data/processed
.PHONY: data
data: features splits
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


## Weighted-PCA and asinh-PCA baselines on every rotation (sizes 4/8/16)
.PHONY: baselines
baselines: data
	$(PYTHON_INTERPRETER) -m jpas_da.modeling.baselines


## Local sweep: STAGE=size (latent 4/8/16 x 4 rotations) or STAGE=ablation DIM=<d>
STAGE ?= size
DIM ?= 8
JOBS ?= 3
.PHONY: sweep
sweep: data
	$(PYTHON_INTERPRETER) -m jpas_da.modeling.sweep --stage $(STAGE) --dim $(DIM) --jobs $(JOBS)


## Collect sweep runs and baselines into reports/sweep_v2.md
.PHONY: report
report:
	$(PYTHON_INTERPRETER) -m jpas_da.modeling.report


## Build the image (linux/amd64 by default: that is what Springfield runs)
.PHONY: image
image:
	$(DOCKER) build --platform $(PLATFORM) -t $(IMAGE) -t $(IMAGE_NAME):latest .


## Push the image the cluster will pull (Springfield pulls from Docker Hub)
# Both tags go up, so the manifest never references an unpushed image. Code
# that is not committed is not pushed: a `<sha>-dirty` image cannot be rebuilt
# from the repository, which defeats the point of tagging by commit. The check
# runs before the build, so a refused push costs no build time.
.PHONY: push
push:
	@case "$(IMAGE_TAG)" in *-dirty) \
		echo "refusing to push $(IMAGE): the tree has uncommitted changes; commit first" >&2; \
		exit 1;; esac
	$(MAKE) image
	$(DOCKER) push $(IMAGE)
	$(DOCKER) push $(IMAGE_NAME):latest


## Run any target wherever it belongs: local (default), docker, or cluster
#   make run TARGET=train-debug            # same shell, same data/
#   make run TARGET=train-debug RUNNER=docker
#   make run TARGET=train RUNNER=cluster   # render + submit the Job
.PHONY: run
run:
ifeq ($(filter $(RUNNER),local docker cluster),)
	$(error RUNNER must be local, docker or cluster; got '$(RUNNER)')
endif
ifeq ($(RUNNER),local)
	$(MAKE) $(TARGET)
else ifeq ($(RUNNER),docker)
	$(DOCKER) run $(DOCKER_FLAGS) $(IMAGE) make $(TARGET)
else
	$(MAKE) submit TARGET=$(TARGET)
endif


## Input data on the cluster: push the raw export to the claim
#   make sync-up                  # data/raw; the pod rebuilds the tables in seconds
#   make sync-up SYNC_SRC=data/   # mirror every derived table instead
#   make sync-up DRY=1            # show what would change, change nothing
.PHONY: sync-up
# `SYNC_SRC` is a directory whose contents go to <STORAGE_ROOT>/data/.
SYNC_SRC ?= data/raw/
SYNC_DIR = $(patsubst data/%,%,$(patsubst %/,%,$(SYNC_SRC)))
DRY ?=
DRY_RUN = $(filter 1 true yes,$(DRY))
SYNC_FLAGS = -az --info=progress2 $(if $(DRY_RUN),--dry-run -v,)
sync-up:
	@if [ -z "$(DRY_RUN)" ]; then ssh $(SPRINGFIELD) "mkdir -p $(STORAGE_ROOT)/data/$(SYNC_DIR)"; fi
	rsync $(SYNC_FLAGS) $(SYNC_SRC) $(SPRINGFIELD):$(STORAGE_ROOT)/data/$(SYNC_DIR)/


## Artefacts back from the cluster: <STORAGE_ROOT>/runs/<JOB>/{models,reports}
# This mirrors into models/ and reports/, so a same-named local file is
# replaced: the pulled vae.pt overwrites the local one. Back up first if the
# local checkpoint matters.
.PHONY: sync-down
sync-down:
	@test -n "$(JOB)" || { echo "JOB is empty"; exit 1; }
	mkdir -p models reports
	rsync $(SYNC_FLAGS) $(SPRINGFIELD):$(STORAGE_ROOT)/runs/$(JOB)/models/ models/
	rsync $(SYNC_FLAGS) $(SPRINGFIELD):$(STORAGE_ROOT)/runs/$(JOB)/reports/ reports/


## Render the cluster Job YAML (writes cluster/jobs/<job>.yaml, not committed)
.PHONY: job
job:
	$(PYTHON_INTERPRETER) -m jpas_da.cluster \
		--image $(IMAGE) --target $(TARGET) --job-name $(JOB) \
		--config $(CONFIG) --set '$(SET)' --debug $(DEBUG) \
		--gpu-count $(GPU_COUNT) --cpu $(CPU) --memory $(MEMORY) \
		--deadline $(DEADLINE) --storage-root $(STORAGE_ROOT) \
		--output cluster/jobs/$(JOB).yaml


## Submit the rendered Job. Contacting the cluster is only ever explicit.
.PHONY: submit
submit: job
	$(FRINK) run cluster/jobs/$(JOB).yaml


## Job and pod state for every JOB of this project
.PHONY: job-status
job-status:
	$(FRINK) ls


## Follow the logs of JOB
.PHONY: job-logs
job-logs:
	$(FRINK) logs $(JOB)


## Delete JOB and its pod
.PHONY: job-delete
job-delete:
	$(FRINK) rm $(JOB)


## Run the smoke path: debug training plus tests
.PHONY: smoke
smoke: train-debug test


## Same smoke path in the container, then the CPU smoke Job manifest
.PHONY: smoke-docker
smoke-docker: image
	$(DOCKER) run $(DOCKER_FLAGS) $(IMAGE) make smoke
	$(MAKE) job TARGET=train-debug GPU_COUNT=0


## Cluster-side smoke check: submit the CPU smoke Job and follow its logs
.PHONY: smoke-cluster
smoke-cluster:
	$(MAKE) submit TARGET=train-debug GPU_COUNT=0
	$(FRINK) logs train-debug-$(IMAGE_TAG)

## Open the marimo notebook inside the project environment
.PHONY: notebook
notebook: requirements
	uv run marimo edit $(NOTEBOOK)


## Serve the documentation site with live reload (http://127.0.0.1:8000)
.PHONY: docs
docs: requirements
	uv run mkdocs serve


## Build the documentation site into site/ (checks every internal link)
.PHONY: docs-build
docs-build: requirements
	uv run mkdocs build --strict


## Everything a reviewer should see pass: lint, tests, docs, data pipeline
.PHONY: check
check: lint test docs-build
	$(MAKE) data
	@echo ">>> check: lint, 113 tests, docs build and the full data pipeline all pass"

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

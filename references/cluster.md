# Running on a cluster

!!! warning "Springfield is one specific cluster, and this setup is tailored to it"
    The cluster path in this repository (`cluster/job.yaml.tmpl`,
    `make submit`, `make sync-up` / `make sync-down`, the `frink` CLI) was
    written for **Springfield**, the Kubernetes cluster of UiT The Arctic
    University of Norway (`springfield.uit.no`), and it encodes that cluster's
    conventions:

    - every namespace gets a PVC called `storage`, mounted at `/storage`;
    - jobs are submitted with `frink`, not raw `kubectl`;
    - the image is pulled from Docker Hub on `linux/amd64`;
    - GPUs are requested as `nvidia.com/gpu` and injected through CDI, which can
      leave `CUDA_VISIBLE_DEVICES` empty (the template sets it explicitly).

    **None of this is portable as-is.** On another cluster you will most likely
    have to change the PVC name (`PVC=storage`), the storage root
    (`STORAGE_ROOT=projects/jpas_da`), the namespace handling, and the submit
    command (`FRINK=`); if GPUs are scheduled differently, the resource limit and
    the device environment variable. The Makefile is parameterised for exactly
    this: every value below is a make variable, so no file needs editing.

    The **image** is the portable part: it is a plain CPU/GPU-agnostic OCI image
    that runs with any container runtime (`docker run`, `podman run`,
    `apptainer exec`, a Kubernetes pod, SLURM's `enroot`/`pyxis`). If your
    cluster does not use Kubernetes, use the image directly (see
    [the image](#the-image) below) and skip everything about Jobs.

## What the cluster setup promises, and what it does not

It promises that the *same* make target runs in three places with one command,
and that a Job is fully described by an image tag plus make variables:

```bash
make run TARGET=train-debug                 # local: uv run, host data/
make run TARGET=train-debug RUNNER=docker   # the image, host data/ mounted
make run TARGET=train RUNNER=cluster        # render + submit the Springfield Job
```

It does **not** promise:

- that the code has been run on another cluster (it has not);
- that the GPU path has been validated beyond a CPU smoke job;
- that the W&B secret exists (it is optional: the pipeline runs without a key);
- that a Job is cheap. A Job holds its GPU until the deadline
  (`DEADLINE=86400`, 24 h), so a stuck run is killed rather than left running.

## The image

`make image` builds `johanmkr/jpas-da:<tag>`, where `<tag>` is the current git
SHA (with `-dirty` appended when the tree is not clean). It contains Python 3.10,
the locked environment from `uv.lock`, `make`, and the code, configs, tests and
job template. It contains **no data, models or reports**: those are always
mounted at `/app/{data,models,reports}`.

| Property | Value |
|---|---|
| Base | `python:3.10-slim` |
| Platform | `linux/amd64` by default (`PLATFORM=`) — what Springfield runs, even when built on an ARM laptop |
| Size | ≈3.3 GB compressed, ≈9.6 GB on disk (the CUDA `torch` wheel dominates; measured on the built image) |
| Entry point | `make help`; a Job always names its target |
| Data | never baked in; `VOLUME ["/app/data", "/app/models", "/app/reports"]` |
| Runs as | any UID (`--user $(id -u):$(id -g)` works; `HOME=/tmp`) |
| Env at runtime | `UV_NO_SYNC=1`, `PYTHON_INTERPRETER=python`, `PYTEST_ADDOPTS=-p no:cacheprovider` |

`uv.lock` resolves `torch` from PyPI. On linux-x86_64 that wheel carries the CUDA
runtime and still imports on a CPU-only host, so one image serves both; the
driver is injected by the host or the pod, never installed in the image.

```bash
make image                          # build + tag with the git SHA
make push                           # build, then push <sha> and :latest (refuses a -dirty tree)
make push REGISTRY=<you> IMAGE_TAG=dev   # your own namespace on Docker Hub

docker run --rm -v "$PWD/data:/app/data" -v "$PWD/models:/app/models" \
           -v "$PWD/reports:/app/reports" --user "$(id -u):$(id -g)" \
           -e HOME=/tmp johanmkr/jpas-da:<tag> make train-debug
```

`make run TARGET=… RUNNER=docker` does the same thing with the mounts and
environment already set, and reads `.env` when one exists. Add `DOCKER_GPU=--gpus
all` on a host with the NVIDIA runtime if you want the container to see a GPU.

The image tag is the reproducibility handle: `<sha>` names the code, and the
config is inside the image, so image tag + make variables describes a run
completely. That is why `make push` refuses to push a `-dirty` build.

## Springfield: the first run

```bash
make sync-up                 # data/raw -> <PVC>/<STORAGE_ROOT>/data/raw
make push                    # image to Docker Hub
make submit TARGET=train     # render cluster/jobs/train-<sha>.yaml, then `frink run`
make job-logs JOB=train-<sha>
make sync-down JOB=train-<sha>   # models/ + reports/ back into the repo
```

Only the raw CSV needs to reach the claim: the pod rebuilds the interim table and
the arrays there in about five seconds. `make sync-up SYNC_SRC=data/` mirrors
prebuilt tables instead, and `DRY=1` shows what would change without changing it.

Rendering never contacts the cluster (`make job` writes
`cluster/jobs/<job>.yaml`, git-ignored); only `make submit`, `make job-logs`,
`make job-status` and `make job-delete` do.

## Parameters

Every value is a make variable; the defaults are Springfield's.

| Parameter | Default | Meaning |
|---|---|---|
| `TARGET` / `JOB` | `train` / `<target>-<image tag>` | make target inside the pod, and the Job name |
| `IMAGE` | `johanmkr/jpas-da:<git sha>` | image the pod runs; the tag records the code |
| `CONFIG` / `SET` / `DEBUG` | `configs/vae.yaml` / empty / `0` | passed as make variables, not baked into the manifest |
| `GPU_COUNT` / `GPU_DEVICES` | `1` / `0` | `GPU_COUNT=0` drops the `nvidia.com/gpu` limit; `GPU_DEVICES` sets `CUDA_VISIBLE_DEVICES` |
| `CPU` / `MEMORY` | `4` / `16Gi` | requests and limits |
| `DEADLINE` | `86400` | `activeDeadlineSeconds`: a stuck run cannot hold a GPU indefinitely |
| `PVC` / `STORAGE_ROOT` | `storage` / `projects/jpas_da` | the claim, and the project directory inside it |
| `SPRINGFIELD` | `springfield` | `~/.ssh/config` host used by `make sync-up` / `sync-down` |
| `FRINK` | `frink` | submission CLI |
| `REGISTRY` / `IMAGE_NAME` / `IMAGE_TAG` | `johanmkr` / `<registry>/jpas-da` / git SHA | image coordinates |
| `PLATFORM` | `linux/amd64` | build platform |

`WANDB_API_KEY` is read from the `wandb-secret`/`api-key` secret when it exists
(`kubectl create secret generic wandb-secret --from-literal=api-key=…`). Missing
secret: the env var is simply absent and the JSON metrics are still written.

An init container creates the `subPath` directories on the claim, because the
kubelet does not. To pin a GPU family (`gtx-1080-ti`, `rtx-2080-ti`, `rtx-3090`,
`rtx-A6000`) or to request the >24 GB A6000, uncomment the
`nodeSelector`/`tolerations` block at the bottom of `cluster/job.yaml.tmpl`: those
labels are Springfield's.

## Adapting it to another cluster

1. **No Kubernetes?** Use the image directly:

   ```bash
   apptainer exec docker://<registry>/jpas-da:<sha> make train   # SLURM/HPC
   docker run ... make train                                     # a single host
   ```

   Mount `data/`, `models/` and `reports/` at `/app/…` and you are done; nothing
   in the code knows about clusters.

2. **Kubernetes, other conventions?** Override `PVC`, `STORAGE_ROOT` and
   `FRINK`, and check the three assumptions in the warning at the top of this
   page. The template is one file (`cluster/job.yaml.tmpl`) and the renderer
   validates every value before writing it (`jpas_da/cluster.py`), so a wrong
   path or image reference fails locally instead of on the cluster.

3. **No SSH mount for storage?** Replace `make sync-up` / `make sync-down`
   (`rsync` over the `SPRINGFIELD` host) with whatever your cluster offers; they
   are the only two rules that assume SSH access to the claim.

4. **Private registry?** Set `REGISTRY` and `make push`; add an
   `imagePullSecrets` entry to `cluster/job.yaml.tmpl` if the namespace needs one.

## Smoke tests, in order of closeness to the cluster

```bash
make smoke           # train-debug + pytest on the host
make smoke-docker    # the same inside the image, then render a CPU Job manifest
make smoke-cluster    # submit the CPU smoke Job to Springfield and follow its logs
```

`make smoke-docker` exercises the whole local → image → manifest chain without
contacting any cluster, which is the right check before submitting anything.

# Portable image for jpas_da. The same image runs locally (`make run
# RUNNER=docker`), in CI and as a Kubernetes pod on Springfield (UiT) or any
# other cluster: it contains the code, the locked environment, `make` and the
# test suite, and nothing else. Data, models and reports are always mounted at
# /app/{data,models,reports} and never baked in, so the image is portable and
# small; see references/cluster.md for what is Springfield-specific and must be
# overridden elsewhere.
#
#   make image                             # tag = git sha
#   make push                              # build + push <sha> and :latest
#   docker run --rm -v "$PWD/data:/app/data" ... <image> make train
#
# uv.lock resolves torch from PyPI; on linux/amd64 that wheel carries the CUDA
# runtime and still imports on a CPU-only host, so one image serves both. The
# driver is injected by the host or the pod (CDI), never installed here.
FROM python:3.10-slim
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    MPLBACKEND=Agg \
    PATH="/app/.venv/bin:$PATH"

# make runs the pipeline targets; slim has no build tools. https sources so the
# build also works behind proxies that refuse plain http.
RUN sed -i 's|http://deb.debian.org|https://deb.debian.org|g' \
        /etc/apt/sources.list.d/debian.sources \
    && apt-get update \
    && apt-get install -y --no-install-recommends make \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.12 /uv /usr/local/bin/uv

WORKDIR /app

# flit reads these to build the package metadata; data never goes in.
COPY README.md LICENSE ./
COPY pyproject.toml uv.lock ./

# Dependency layer first, so editing code does not re-resolve the lock. uv's
# download cache (~5.6 GB with the CUDA torch wheel) is deleted in this same
# layer: a later layer cannot shrink an earlier one, and --no-cache would force
# a full re-resolve on every code change.
#
# `--no-dev --group test`: the runtime dependencies plus pytest, because
# `make smoke` runs the suite inside the image; docs and notebook tooling stay
# out.
RUN uv sync --frozen --no-dev --group test --no-install-project \
    && rm -rf /root/.cache/uv

COPY Makefile ./
COPY configs ./configs
COPY jpas_da ./jpas_da
# The cluster smoke path (`make smoke`) runs the suite in-image, and the job
# template is rendered host-side but kept here so the tree matches the repo.
COPY tests ./tests
COPY cluster ./cluster
# The second sync only installs the project, but uv still writes its download
# cache. Deleting it in this same layer keeps the ~5 GB out of the image while
# leaving the layer cacheable (unlike --no-cache, which forces a full re-resolve
# on every code change).
RUN uv sync --frozen --no-dev --group test && rm -rf /root/.cache/uv

# The image already has the exact environment: never re-resolve at run time, and
# call the venv's python directly instead of `uv run` (which wants the lock).
# `/app` is read-only to `--user` runs, so pytest must not write a cache there.
ENV UV_NO_SYNC=1 \
    PYTHON_INTERPRETER=python \
    PYTEST_ADDOPTS="-p no:cacheprovider"

# Mounts replace these; the directories only make a bare `docker run` obvious.
VOLUME ["/app/data", "/app/models", "/app/reports"]

# `make` with no target prints help; the job always names its target.
CMD ["make", "help"]

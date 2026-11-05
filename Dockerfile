# CPU image with the full pipeline. A CUDA build is a one-line change to the
# base image; the code selects the device at run time (train.device: auto).
FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_NO_SYNC=1 \
    MPLBACKEND=Agg

COPY --from=ghcr.io/astral-sh/uv:0.13 /uv /usr/local/bin/uv

WORKDIR /app
# flit reads these to build the package; the data directories never go in.
COPY README.md LICENSE ./

# Dependency layer first so edits to the code do not re-resolve the lock.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

COPY Makefile ./
COPY configs ./configs
COPY jpas_da ./jpas_da
RUN uv sync --frozen --no-dev

ENV PATH="/app/.venv/bin:$PATH"

# The raw export and the derived tables stay outside the image: mount the
# repository's data directory, e.g.
#   docker run --rm -v "$PWD/data:/app/data" jpas-da:dev make train-debug
VOLUME ["/app/data", "/app/models", "/app/reports"]

CMD ["make", "train-debug"]

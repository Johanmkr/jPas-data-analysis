"""Render the Springfield Job template.

One parametrised Job (``cluster/job.yaml.tmpl``) is filled in by make, so that
the target you run locally -- ``make train``, ``make sweep``, ``make test`` --
is the target the pod runs, inside the same image, against the same paths:

    local            docker run ... make train     make job/submit
    data/       <->  /app/data   <->  <PVC>/<STORAGE_ROOT>/data
    models/     <->  /app/models <->  <PVC>/<STORAGE_ROOT>/runs/<job>/models
    reports/    <->  /app/reports<->  <PVC>/<STORAGE_ROOT>/runs/<job>/reports

Rendering is deliberately strict: every ``${PLACEHOLDER}`` must be supplied and
every value is type-checked, because a silently empty image tag or a malformed
job name fails only after the pod is scheduled -- or worse, runs the wrong code.
Nothing here talks to the cluster; ``kubectl``/``frink`` is the user's call.

Run ``python -m jpas_da.cluster --help`` or use ``make job``.
"""

from __future__ import annotations

from pathlib import Path
import re
from typing import Annotated, Any

from loguru import logger
import typer
import yaml

from jpas_da.config import PROJ_ROOT

app = typer.Typer()

TEMPLATE = PROJ_ROOT / "cluster" / "job.yaml.tmpl"

PLACEHOLDER = re.compile(r"\$\{([A-Z_][A-Z0-9_]*)\}")

#: Springfield defaults: the claim every namespace gets is ``storage``, mounted
#: at ``/storage``; W&B is read from the ``wandb-secret``/``api-key`` pair that
#: ``kubectl create secret generic wandb-secret --from-literal=api-key=...``
#: creates. An empty NAMESPACE means "the context's namespace".
DEFAULTS: dict[str, str] = {
    "NAMESPACE": "",
    "PVC": "storage",
    "STORAGE_ROOT": "projects/jpas_da",
    "WANDB_SECRET": "wandb-secret",
    "WANDB_SECRET_KEY": "api-key",
    "TARGET": "train",
    "CONFIG": "configs/vae.yaml",
    "SET": "",
    "DEBUG": "0",
    "BACKOFF_LIMIT": "0",
    "DEADLINE": "86400",
    "TTL_SECONDS": "604800",
    "CPU": "4",
    "MEMORY": "16Gi",
    "GPU_COUNT": "1",
    "GPU_DEVICES": "0",
    "IMAGE": "jpas-da:dev",
}

#: Values that make a broken manifest, checked before anything is submitted.
_NAME = re.compile(r"^[a-z0-9]([-a-z0-9]*[a-z0-9])?$")
_PATH = re.compile(r"^[A-Za-z0-9._/-]+$")
_QUANTITY = re.compile(r"^[0-9]+(\.[0-9]+)?(m|Gi|Mi|G|M)?$")
_INT = re.compile(r"^[0-9]+$")
_IMAGE = re.compile(r"^[a-z0-9][a-z0-9._/-]*(:[A-Za-z0-9._-]+|@sha256:[a-f0-9]{64})$")
_DEVICES = re.compile(r"^[0-9]+(,[0-9]+)*$")
# Keep SET/CONFIG to characters that survive YAML quoting unchanged.
_SAFE_ARG = re.compile(r"^[A-Za-z0-9._=/,\[\]()<>@%+~: -]*$")
# `make` accepts single targets; a space would silently run `make train --set`.
_TARGET = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")


def job_name_for(target: str, image_tag: str) -> str:
    """Unique, RFC 1123 job name: the target plus a short image tag.

    The image tag is normally the git sha, so the job name records which code
    the pod ran without a separate lookup.
    """
    slug = re.sub(r"[^a-z0-9-]+", "-", f"{target}-{image_tag}".lower()).strip("-")
    if not slug:
        raise ValueError("job name is empty after sanitising target and image tag")
    if len(slug) > 63:
        raise ValueError(f"job name {slug!r} exceeds the 63-character DNS limit")
    if not _NAME.match(slug):
        raise ValueError(f"job name {slug!r} is not a valid DNS-1123 label")
    return slug


def parameters(**overrides: Any) -> dict[str, str]:
    """Defaults, overridden by non-empty values, all coerced to strings.

    ``TARGET_SAFE``, ``DATA_SUBPATH`` and ``JOB_SUBPATH`` are derived, not
    supplied: the label needs a slug, and the two subPaths are what makes the
    pod see the same ``data/``, ``models/``, ``reports/`` layout as a local run.
    """
    values = dict(DEFAULTS)
    for key, value in overrides.items():
        if value is None or value == "":
            continue
        values[key] = str(value)
    # Derived, not supplied; each is validated below, including the root they
    # are built from (a leading "/" or a ".." must fail, not be normalised).
    values["TARGET_SAFE"] = re.sub(r"[^a-z0-9-]+", "-", values["TARGET"].lower()).strip("-")
    root = values["STORAGE_ROOT"]
    values["DATA_SUBPATH"] = f"{root}/data"
    values["JOB_SUBPATH"] = f"{root}/runs/{values.get('JOB_NAME', '')}"
    # A moving tag ("latest", "dev") with IfNotPresent silently reuses whatever
    # the node happened to cache; only an immutable tag may skip the pull.
    tag = values["IMAGE"].rsplit(":", 1)[-1]
    values["PULL_POLICY"] = "Always" if tag in {"latest", "dev", "edge"} else "IfNotPresent"
    return values


def _check_subpath(key: str, value: str) -> None:
    if value.startswith("/") or ".." in value.split("/") or not _PATH.match(value):
        raise ValueError(f"{key} {value!r} must be a relative path inside the claim")


def validate(values: dict[str, str]) -> None:
    name = values["JOB_NAME"]
    if not _NAME.match(name) or len(name) > 63:
        raise ValueError(f"invalid JOB_NAME {name!r}: must be a DNS-1123 label")
    if not _IMAGE.match(values["IMAGE"]):
        raise ValueError(
            f"IMAGE {values['IMAGE']!r} must be a tagged reference, e.g. user/jpas-da:<sha>"
        )
    if values["NAMESPACE"] and not _NAME.match(values["NAMESPACE"]):
        raise ValueError(f"NAMESPACE {values['NAMESPACE']!r} must be a DNS-1123 label")

    for key in ("GPU_COUNT", "BACKOFF_LIMIT", "TTL_SECONDS", "DEADLINE"):
        if not _INT.match(values[key]):
            raise ValueError(f"{key} {values[key]!r} must be a non-negative integer")
    if values["GPU_COUNT"] != "0" and int(values["GPU_COUNT"]) != len(
        values["GPU_DEVICES"].split(",")
    ):
        raise ValueError(
            f"GPU_COUNT={values['GPU_COUNT']} needs the same number of GPU_DEVICES, "
            f"got {values['GPU_DEVICES']!r}"
        )
    if not _DEVICES.match(values["GPU_DEVICES"]):
        raise ValueError(f"GPU_DEVICES {values['GPU_DEVICES']!r} must be e.g. 0 or 0,1")

    for key in ("CPU", "MEMORY"):
        if not _QUANTITY.match(values[key]):
            raise ValueError(f"{key} {values[key]!r} is not a Kubernetes quantity")

    if not _TARGET.match(values["TARGET"]):
        raise ValueError(f"TARGET {values['TARGET']!r} must be a single make target")
    if not values["TARGET_SAFE"]:
        raise ValueError(f"TARGET {values['TARGET']!r} has no usable characters for a label")
    for key in ("CONFIG", "SET"):
        if not _SAFE_ARG.match(values[key]) or '"' in values[key]:
            raise ValueError(f"{key} {values[key]!r} contains characters that break the manifest")
    if values["DEBUG"] not in {"0", "1", "true", "false", "yes", "no"}:
        raise ValueError(f"DEBUG {values['DEBUG']!r} must be a make boolean, e.g. 0 or 1")
    for key in ("DATA_SUBPATH", "JOB_SUBPATH"):
        _check_subpath(key, values[key])


def render(template: Path = TEMPLATE, **overrides: Any) -> str:
    """Return the Job YAML with every placeholder substituted."""
    text = Path(template).read_text()
    required = set(PLACEHOLDER.findall(text))
    values = parameters(**overrides)
    missing = required - values.keys()
    if missing:
        raise ValueError(f"no value for template placeholders: {sorted(missing)}")
    validate(values)

    def substitute(match: re.Match[str]) -> str:
        return values[match.group(1)]

    rendered = PLACEHOLDER.sub(substitute, text)
    if "${" in rendered:
        raise ValueError("unsubstituted placeholder remains in the rendered template")

    # An empty value is not the same as "omit the key": drop the whole line (and
    # the list item, for SET) whose placeholder came out empty.
    rendered = re.sub(r"\n[ ]*namespace:[ ]*(?=\n)", "", rendered)
    rendered = re.sub(r"\n[ ]*- name: SET\n[ ]*value: \"\"", "", rendered)
    if values["GPU_COUNT"] == "0":
        rendered = re.sub(r"\n[ ]*nvidia\.com/gpu: \"0\"", "", rendered)

    manifest = yaml.safe_load(rendered)
    container = manifest["spec"]["template"]["spec"]["containers"][0]
    if container["image"] != values["IMAGE"]:
        raise ValueError("rendered image does not match the requested tag")
    if container["command"] != ["make", values["TARGET"]]:
        raise ValueError("rendered command does not match the requested target")
    limits = container["resources"].get("limits", {})
    if values["GPU_COUNT"] == "0" and "nvidia.com/gpu" in limits:
        raise ValueError("GPU_COUNT=0 must not publish an nvidia.com/gpu limit")
    if values["GPU_COUNT"] != "0" and "nvidia.com/gpu" not in limits:
        raise ValueError("a GPU request is missing its nvidia.com/gpu limit")
    mounts = {m["mountPath"]: m["subPath"] for m in container["volumeMounts"]}
    if mounts != {
        "/app/data": values["DATA_SUBPATH"],
        "/app/models": f"{values['JOB_SUBPATH']}/models",
        "/app/reports": f"{values['JOB_SUBPATH']}/reports",
    }:
        raise ValueError(f"unexpected volume mounts: {mounts}")
    if "namespace" in manifest["metadata"] and not values["NAMESPACE"]:
        raise ValueError("an empty NAMESPACE must not render a namespace key")
    return rendered


@app.command()
def main(
    template: Path = TEMPLATE,
    job_name: Annotated[str | None, typer.Option("--job-name", help="DNS-1123 job name")] = None,
    image: Annotated[str, typer.Option("--image", help="Image reference with a tag")] = DEFAULTS[
        "IMAGE"
    ],
    target: Annotated[
        str, typer.Option("--target", help="Make target to run inside the image")
    ] = DEFAULTS["TARGET"],
    config: Annotated[str, typer.Option("--config", help="CONFIG for the make target")] = DEFAULTS[
        "CONFIG"
    ],
    overrides: Annotated[
        str, typer.Option("--set", help="SET for the make target, e.g. --set data.rotation=2")
    ] = "",
    debug: Annotated[str, typer.Option("--debug", help="DEBUG for the make target")] = "0",
    gpu_count: Annotated[str, typer.Option("--gpu-count", help="GPUs to request")] = DEFAULTS[
        "GPU_COUNT"
    ],
    cpu: Annotated[str, typer.Option("--cpu")] = DEFAULTS["CPU"],
    memory: Annotated[str, typer.Option("--memory")] = DEFAULTS["MEMORY"],
    deadline: Annotated[str, typer.Option("--deadline", help="activeDeadlineSeconds")] = DEFAULTS[
        "DEADLINE"
    ],
    storage_root: Annotated[
        str, typer.Option("--storage-root", help="Project directory inside the claim")
    ] = DEFAULTS["STORAGE_ROOT"],
    output: Annotated[
        Path | None, typer.Option("--output", help="Write here instead of stdout")
    ] = None,
) -> None:
    """Render cluster/job.yaml.tmpl; never contacts the cluster."""
    name = job_name or job_name_for(target, image.rsplit(":", 1)[-1])
    rendered = render(
        template,
        JOB_NAME=name,
        IMAGE=image,
        TARGET=target,
        CONFIG=config,
        SET=overrides,
        DEBUG=debug,
        GPU_COUNT=gpu_count,
        CPU=cpu,
        MEMORY=memory,
        DEADLINE=deadline,
        STORAGE_ROOT=storage_root,
    )
    if output is None:
        typer.echo(rendered)
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered)
        logger.success(f"Wrote {output}")


if __name__ == "__main__":
    app()

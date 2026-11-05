"""Render the cluster Job template.

tavla task ``cluster-job-template``: one Job YAML, filled in by make, so
``make remote TARGET=<target>`` runs that make target inside the image on the
cluster. Generated YAML is not committed.

Rendering is deliberately strict: every ``${PLACEHOLDER}`` must be supplied and
every supplied value is type-checked, because a silently empty image tag or a
malformed job name fails only after the pod is scheduled -- or worse, runs the
wrong code. Nothing here talks to the cluster; ``kubectl`` is the user's call.

Run ``python -m jpas_da.cluster --help`` or use ``make job-render``.
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

DEFAULTS: dict[str, str] = {
    "NAMESPACE": "default",
    "PVC": "jpas-storage",
    "WANDB_SECRET": "wandb-api",
    "TARGET": "train",
    "BACKOFF_LIMIT": "0",
    "TTL_SECONDS": "86400",
    "CPU": "4",
    "MEMORY": "16Gi",
    "GPU_COUNT": "1",
    "IMAGE": "jpas-da:dev",
}

_NAME = re.compile(r"^[a-z0-9]([-a-z0-9]*[a-z0-9])?$")
_QUANTITY = re.compile(r"^[0-9]+(\.[0-9]+)?(m|Gi|Mi|G|M)?$")


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

    ``TARGET_SAFE`` is derived, not supplied: it is the target slug used for
    the job's ``target`` label, which Kubernetes requires to be a label value.
    """
    values = dict(DEFAULTS)
    for key, value in overrides.items():
        if value is None or value == "":
            continue
        values[key] = str(value)
    values["TARGET_SAFE"] = re.sub(r"[^a-z0-9-]+", "-", values["TARGET"].lower()).strip("-")
    return values


def validate(values: dict[str, str]) -> None:
    name = values["JOB_NAME"]
    if not _NAME.match(name) or len(name) > 63:
        raise ValueError(f"invalid JOB_NAME {name!r}: must be a DNS-1123 label")
    if not values["IMAGE"] or ":" not in values["IMAGE"]:
        raise ValueError(f"IMAGE {values['IMAGE']!r} must be tagged, e.g. jpas-da:<sha>")
    if values["GPU_COUNT"] not in {"0", "1"} and not values["GPU_COUNT"].isdigit():
        raise ValueError(f"GPU_COUNT {values['GPU_COUNT']!r} must be a non-negative integer")
    for key in ("CPU", "MEMORY"):
        if not _QUANTITY.match(values[key]):
            raise ValueError(f"{key} {values[key]!r} is not a Kubernetes quantity")
    if not values["TARGET"] or " " in values["TARGET"]:
        raise ValueError(f"TARGET {values['TARGET']!r} must be a single make target")
    if not values["TARGET_SAFE"]:
        raise ValueError(f"TARGET {values['TARGET']!r} has no usable characters for a label")


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
        key = match.group(1)
        if key not in values:
            raise ValueError(f"placeholder ${{{key}}} was not provided")
        return values[key]

    rendered = PLACEHOLDER.sub(substitute, text)
    if "${" in rendered:
        raise ValueError("unsubstituted placeholder remains in the rendered template")

    if values["GPU_COUNT"] == "0":
        # Requesting zero GPUs is not the same as requesting none: drop the key.
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
    return rendered


@app.command()
def main(
    template: Path = TEMPLATE,
    job_name: Annotated[str | None, typer.Option("--job-name", help="DNS-1123 job name")] = None,
    image: Annotated[str, typer.Option("--image", help="Image with tag")] = DEFAULTS["IMAGE"],
    target: Annotated[
        str, typer.Option("--target", help="Make target to run inside the image")
    ] = DEFAULTS["TARGET"],
    gpu_count: Annotated[str, typer.Option("--gpu-count", help="GPUs to request")] = (
        DEFAULTS["GPU_COUNT"]
    ),
    cpu: Annotated[str, typer.Option("--cpu")] = DEFAULTS["CPU"],
    memory: Annotated[str, typer.Option("--memory")] = DEFAULTS["MEMORY"],
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
        GPU_COUNT=gpu_count,
        CPU=cpu,
        MEMORY=memory,
    )
    if output is None:
        typer.echo(rendered)
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered)
        logger.success(f"Wrote {output}")


if __name__ == "__main__":
    app()

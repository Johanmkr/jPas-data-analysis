"""Invariants of the cluster Job renderer.

The rendered manifest is the only thing standing between a make target and the
cluster, so these tests pin the properties a wrong job would break: the image
tag is the one built, the command is one make target, the data lives on the PVC,
and a zero-GPU job requests no GPU at all.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from jpas_da.cluster import DEFAULTS, job_name_for, parameters, render

TEMPLATE = Path(__file__).resolve().parents[1] / "cluster" / "job.yaml.tmpl"


def _manifest(**overrides) -> dict:
    return yaml.safe_load(render(TEMPLATE, **overrides))


def _container(manifest: dict) -> dict:
    return manifest["spec"]["template"]["spec"]["containers"][0]


def test_rendered_job_matches_the_requested_image_and_target():
    manifest = _manifest(JOB_NAME="train-abc1234", IMAGE="jpas-da:abc1234", TARGET="train")
    container = _container(manifest)
    assert manifest["metadata"]["name"] == "train-abc1234"
    assert container["image"] == "jpas-da:abc1234"
    assert container["command"] == ["make", "train"]


def test_every_placeholder_is_substituted():
    rendered = render(TEMPLATE, JOB_NAME="train-abc1234")
    assert "${" not in rendered


def test_data_and_outputs_are_mounted_on_the_persistent_volume():
    manifest = _manifest(JOB_NAME="train-abc1234")
    spec = manifest["spec"]["template"]["spec"]
    mounts = {m["mountPath"]: m for m in _container(manifest)["volumeMounts"]}
    assert set(mounts) == {"/app/data", "/app/models", "/app/reports"}
    volume = spec["volumes"][0]
    assert volume["name"] == "storage"
    assert set(volume) == {"name", "persistentVolumeClaim"}
    assert volume["persistentVolumeClaim"]["claimName"] == DEFAULTS["PVC"]
    # The writable paths must not share a subPath with the mounted input data.
    assert mounts["/app/models"]["subPath"].startswith("models/")
    assert mounts["/app/reports"]["subPath"].startswith("reports/")
    assert mounts["/app/data"]["subPath"] == "data"


def test_zero_gpu_job_requests_no_gpu():
    manifest = _manifest(JOB_NAME="train-abc1234", GPU_COUNT="0")
    limits = _container(manifest)["resources"]["limits"]
    assert "nvidia.com/gpu" not in limits


def test_gpu_job_requests_one_gpu():
    manifest = _manifest(JOB_NAME="train-abc1234", GPU_COUNT="1")
    assert _container(manifest)["resources"]["limits"]["nvidia.com/gpu"] == "1"


def test_untagged_image_is_rejected():
    with pytest.raises(ValueError, match="must be tagged"):
        render(TEMPLATE, JOB_NAME="train-abc1234", IMAGE="jpas-da")


def test_multi_word_target_is_rejected():
    with pytest.raises(ValueError, match="single make target"):
        render(TEMPLATE, JOB_NAME="train-abc1234", TARGET="train --set x=1")


@pytest.mark.parametrize("name", ["Train_ABC", "-train", "train-", "with space", ""])
def test_invalid_job_name_is_rejected(name):
    with pytest.raises(ValueError, match="JOB_NAME"):
        render(TEMPLATE, JOB_NAME=name)


def test_malformed_quantity_is_rejected():
    with pytest.raises(ValueError, match="not a Kubernetes quantity"):
        render(TEMPLATE, JOB_NAME="train-abc1234", MEMORY="16 gigs")


def test_job_name_carries_the_target_and_the_image_tag():
    assert job_name_for("train-debug", "abc1234") == "train-debug-abc1234"
    assert job_name_for("train", "abc1234") == "train-abc1234"


def test_job_name_rejects_an_over_long_tag():
    with pytest.raises(ValueError, match="63-character"):
        job_name_for("train", "a" * 70)


def test_missing_placeholder_value_is_reported():
    template = TEMPLATE.with_name("job_extra_placeholder.yaml.tmpl")
    template.write_text(TEMPLATE.read_text() + "\n# ${NOT_A_REAL_PARAMETER}\n")
    try:
        with pytest.raises(ValueError, match="NOT_A_REAL_PARAMETER"):
            render(template, JOB_NAME="train-abc1234")
    finally:
        template.unlink()


def test_defaults_do_not_leak_into_explicit_values():
    values = parameters(CPU="8")
    assert values["CPU"] == "8"
    assert values["TARGET"] == DEFAULTS["TARGET"]
    assert values["TARGET_SAFE"] == DEFAULTS["TARGET"].lower()

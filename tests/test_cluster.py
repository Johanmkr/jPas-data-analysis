"""Invariants of the cluster Job renderer.

The rendered manifest is the only thing standing between a make target and the
cluster, so these tests pin the properties a wrong job would break: the image
tag is the one built, the command is one make target, ``make``'s own variables
are what configures the run, the claim is mounted so the pod sees the same
``data/``/``models/``/``reports/`` layout as a local run, and a zero-GPU job
requests no GPU at all.
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


def _env(manifest: dict) -> dict:
    return {item["name"]: item for item in _container(manifest)["env"]}


def test_rendered_job_matches_the_requested_image_and_target():
    manifest = _manifest(JOB_NAME="train-abc1234", IMAGE="johanmkr/jpas-da:abc1234")
    assert manifest["metadata"]["name"] == "train-abc1234"
    assert _container(manifest)["image"] == "johanmkr/jpas-da:abc1234"
    # The init container exists only to create subPath dirs; same image, no pull.
    init = manifest["spec"]["template"]["spec"]["initContainers"][0]
    assert init["image"] == "johanmkr/jpas-da:abc1234"
    assert _container(manifest)["command"] == ["make", "train"]


def test_every_placeholder_is_substituted():
    rendered = render(TEMPLATE, JOB_NAME="train-abc1234")
    assert "${" not in rendered


def test_make_variables_are_passed_through_to_the_pod():
    manifest = _manifest(
        JOB_NAME="train-abc1234",
        TARGET="train-debug",
        CONFIG="configs/other.yaml",
        SET="--set data.rotation=2 --set loss.beta=0.5",
        DEBUG="1",
    )
    env = _env(manifest)
    assert _container(manifest)["command"] == ["make", "train-debug"]
    assert env["CONFIG"]["value"] == "configs/other.yaml"
    assert env["SET"]["value"] == "--set data.rotation=2 --set loss.beta=0.5"
    assert env["DEBUG"]["value"] == "1"


def test_empty_set_is_omitted_rather_than_passed_as_an_empty_string():
    manifest = _manifest(JOB_NAME="train-abc1234", SET="")
    assert "SET" not in _env(manifest)


def test_pod_sees_the_same_layout_as_a_local_run():
    manifest = _manifest(JOB_NAME="train-abc1234")
    spec = manifest["spec"]["template"]["spec"]
    mounts = {m["mountPath"]: m["subPath"] for m in _container(manifest)["volumeMounts"]}
    assert mounts == {
        "/app/data": "projects/jpas_da/data",
        "/app/models": "projects/jpas_da/runs/train-abc1234/models",
        "/app/reports": "projects/jpas_da/runs/train-abc1234/reports",
    }
    init_mounts = spec["initContainers"][0]["volumeMounts"]
    assert [m["mountPath"] for m in init_mounts] == ["/storage"]
    volume = spec["volumes"][0]
    assert volume["name"] == "storage"
    assert set(volume) == {"name", "persistentVolumeClaim"}
    assert volume["persistentVolumeClaim"]["claimName"] == DEFAULTS["PVC"] == "storage"
    # Outputs must never share a subPath with the mounted input data.
    assert not mounts["/app/models"].startswith(mounts["/app/data"])


def test_rendered_manifest_is_a_kubernetes_job():
    manifest = _manifest(JOB_NAME="train-abc1234")
    assert manifest["apiVersion"] == "batch/v1"
    assert manifest["kind"] == "Job"


def test_init_container_creates_every_mounted_subpath():
    # `mkdir` runs at the image's working directory unless the paths are
    # absolute, so each one must be anchored on the /storage mount.
    manifest = _manifest(JOB_NAME="train-abc1234")
    script = manifest["spec"]["template"]["spec"]["initContainers"][0]["command"][-1]
    for subpath in ("projects/jpas_da/data", "projects/jpas_da/runs/train-abc1234/models"):
        assert f"/storage/{subpath}" in script
    assert "projects/jpas_da/runs/train-abc1234/reports" in script


def test_moving_tag_always_pulls_and_sha_tag_does_not():
    moving = _container(_manifest(JOB_NAME="train-abc1234", IMAGE="johanmkr/jpas-da:latest"))
    assert moving["imagePullPolicy"] == "Always"
    pinned = _container(_manifest(JOB_NAME="train-abc1234", IMAGE="johanmkr/jpas-da:abc1234"))
    assert pinned["imagePullPolicy"] == "IfNotPresent"


def test_an_empty_namespace_leaves_the_key_out():
    assert "namespace" not in _manifest(JOB_NAME="train-abc1234")["metadata"]
    assert (
        _manifest(JOB_NAME="train-abc1234", NAMESPACE="jkr669")["metadata"]["namespace"]
        == "jkr669"
    )


def test_zero_gpu_job_requests_no_gpu():
    manifest = _manifest(JOB_NAME="train-abc1234", GPU_COUNT="0")
    limits = _container(manifest)["resources"]["limits"]
    assert "nvidia.com/gpu" not in limits


def test_gpu_job_requests_one_gpu_and_pins_the_device():
    manifest = _manifest(JOB_NAME="train-abc1234", GPU_COUNT="1", GPU_DEVICES="0")
    limits = _container(manifest)["resources"]["limits"]
    assert limits["nvidia.com/gpu"] == "1"
    # Springfield's CDI injection may leave this empty; torch needs "0".
    assert _env(manifest)["CUDA_VISIBLE_DEVICES"]["value"] == "0"


def test_gpu_count_must_match_the_device_list():
    with pytest.raises(ValueError, match="GPU_DEVICES"):
        render(TEMPLATE, JOB_NAME="train-abc1234", GPU_COUNT="2")


def test_untagged_image_is_rejected():
    with pytest.raises(ValueError, match="must be a tagged reference"):
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


def test_storage_root_cannot_escape_the_claim():
    with pytest.raises(ValueError, match="relative path"):
        render(TEMPLATE, JOB_NAME="train-abc1234", STORAGE_ROOT="/abs/path")
    with pytest.raises(ValueError, match="relative path"):
        render(TEMPLATE, JOB_NAME="train-abc1234", STORAGE_ROOT="projects/../../etc")


def test_config_values_that_break_the_manifest_are_rejected():
    with pytest.raises(ValueError, match="break the manifest"):
        render(TEMPLATE, JOB_NAME="train-abc1234", SET='--set x="unterminated')


def test_job_name_carries_the_target_and_the_image_tag():
    assert job_name_for("train-debug", "abc1234") == "train-debug-abc1234"
    assert job_name_for("train", "abc1234") == "train-abc1234"


def test_job_name_rejects_an_over_long_tag():
    with pytest.raises(ValueError, match="63-character"):
        job_name_for("train", "a" * 70)


def test_missing_placeholder_value_is_reported(tmp_path):
    # tmp_path, not beside the template: the image mounts the tree read-only.
    template = tmp_path / "job_extra_placeholder.yaml.tmpl"
    template.write_text(TEMPLATE.read_text() + "\n# ${NOT_A_REAL_PARAMETER}\n")
    with pytest.raises(ValueError, match="NOT_A_REAL_PARAMETER"):
        render(template, JOB_NAME="train-abc1234")


def test_defaults_do_not_leak_into_explicit_values():
    values = parameters(CPU="8")
    assert values["CPU"] == "8"
    assert values["TARGET"] == DEFAULTS["TARGET"]
    assert values["TARGET_SAFE"] == DEFAULTS["TARGET"].lower()

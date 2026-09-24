"""Synthetic site configurations for the harness (neutral example values only; DESIGN 5.6).

The production launcher, worker and checkpoint code read their site values from a validated
``SiteConfig`` (S1a). The harness drives that code with synthetic sites built here: documentation
addresses (RFC 5737), ``example`` names and the placeholder bucket of ``examples/site.example.toml``,
plus the temporary run root, lock files and topology binding each exercise creates.

One value cannot be synthetic: G4's tiny pack writes its source URI into the manifest it
records, and that URI was derived at S0 from the model source pinned at ``181c013e``.
:func:`baseline_source_uri` reads it from the baseline commit (the harness is bound to that commit
anyway), so the recorded digests stay reproducible without the harness ever spelling the value.

Standard library only (the G6 controller stage imports this module and must stay JAX-free).
"""

from __future__ import annotations

import ast
import os
from pathlib import Path
from typing import Any

from .common import BASELINE_COMMIT, git

EXAMPLE_BUCKET = "gs://example-bucket/"
EXAMPLE_COORDINATOR = "203.0.113.10:8476"  # documentation address (RFC 5737), production port
EXAMPLE_HEX = "0" * 64


def site_mapping(base: Path, **overrides: dict[str, Any]) -> dict[str, Any]:
    """A complete site mapping whose paths live under ``base``; ``overrides`` replace keys per table."""
    mapping: dict[str, Any] = dict(
        schema="glm_tpu_site_v1",
        fleet=dict(
            tpu_name="example-vm",
            zone="example-zone1-a",
            project="",
            num_hosts=8,
            chips_per_host=4,
            host_rank_regex=r"-w-(\d+)$",
            coordinator_address=EXAMPLE_COORDINATOR,
            worker_python="/opt/example/bin/python3.12",
            worker_pythonpath=["/opt/example/site-packages"],
            helper_python="python3",
            known_hosts=str(base / "known_hosts"),
        ),
        paths=dict(
            repo="", run_root=str(base / "runs"), model_path=str(base / "model"), hlo_dump_root="/dev/shm/glm-tpu-hlo"
        ),
        checkpoint=dict(
            namespace=str(base / "checkpoints"),
            root=str(base / "checkpoints" / "pack"),
            inventory_namespace=str(base / "inventories"),
            source_inventory=str(base / "inventories" / "tag" / "source_inventory.json"),
            source_inventory_sha256=EXAMPLE_HEX,
            manifest_sha256=EXAMPLE_HEX,
            success_sha256=EXAMPLE_HEX,
            source_complete_sha256=EXAMPLE_HEX,
        ),
        topology=dict(
            binding_dir=str(base / "binding"),
            binding_sha256=EXAMPLE_HEX,
            capture_root=str(base / "captures"),
            topology_sha256=EXAMPLE_HEX,
            topology_fleet_sha256=EXAMPLE_HEX,
            mesh_sha256=EXAMPLE_HEX,
            slice_name="example-vm",
        ),
        storage=dict(source_uri=EXAMPLE_BUCKET + "models/GLM-5.3-FP8", allowed_source_uri_prefixes=[EXAMPLE_BUCKET]),
        locks=dict(
            workload=[str(base / "locks" / "lock0"), str(base / "locks" / "lock1")],
            sync=[str(base / "locks" / "lock2"), str(base / "locks" / "lock3")],
        ),
        launch=dict(
            allowed_branches=["main", "release/*"], expected_origin="", require_clean=True, require_pushed=True
        ),
    )
    for table, values in overrides.items():
        mapping[table] = dict(mapping[table], **values)
    return mapping


def site(base: Path, **overrides: dict[str, Any]) -> Any:
    from glm_tpu.config.site import SiteConfig

    return SiteConfig.from_mapping(site_mapping(base, **overrides))


def write_site(path: Path, mapping: dict[str, Any]) -> Path:
    """The mapping as an owner-only TOML site file (what an operator writes)."""
    from glm_tpu.config.site import to_toml

    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as stream:
        stream.write(to_toml(mapping))
    return path


def baseline_source_uri() -> str:
    """``SOURCE_URI`` of ``glm_tpu/optimized/model.py`` at the baseline commit (never printed)."""
    text = git("show", f"{BASELINE_COMMIT}:glm_tpu/optimized/model.py")
    for node in ast.parse(text).body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "SOURCE_URI" for t in node.targets):
            value = ast.literal_eval(node.value)
            if isinstance(value, str):
                return value
    raise RuntimeError("the baseline commit's model.py defines no SOURCE_URI")


def baseline_storage_site(base: Path) -> Any:
    """A synthetic site whose storage section admits the baseline model source's bucket (G4)."""
    uri = baseline_source_uri()
    prefix = "/".join(uri.split("/")[:3]) + "/"
    return site(base, storage=dict(source_uri=uri, allowed_source_uri_prefixes=[prefix]))

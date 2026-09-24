"""G5 completion: the CPU worker preflight on a locally staged bundle with the real site file.

Site tier (marker ``site``, excluded from G10/G11): runs only with ``GLM_TPU_TEST_SITE=<site file>``
on a fleet host (normally rank 0) while no TPU run is live, and reads the real assets read-only.
The run root is relocated to the test's temporary directory with ``GLM_TPU_RUN_ROOT``, exactly as
an operator overrides it for the controller; nothing else is written.

What runs is the launch path of one host, from the real code: the site file loaded and resolved as
the controller loads it (``SiteConfig.load``), the bundle built by the launcher's real
``stage_bundle`` (``git archive`` of this checkout's HEAD, ``source_manifest.json``, the resolved
``site.json``, the topology rebinding and captures), the real ``stage_bundle`` remote helper run
through ``bash -c`` on the exact command string the controller sends (``fleet.command``), and the
worker's ``--preflight-only`` run through ``bash -c`` on the exact ``fleet.preflight_command``
string with the launcher's argv (``--site-sha256`` of the staged ``site.json``). The worker then
re-hashes every staged source file, checks the staged site, the model template and assets, the
request digest, the site pins (``site_args``/``require_site``) and the topology binding before it
reports its environment (``jax == jaxlib == 0.10.1``). A staged file changed afterwards must be
refused. Output: hostnames, versions and refusal messages only (no site values, no private data).

Usage (rank 0, fleet idle)::

    GLM_TPU_TEST_SITE=~/.config/glm-tpu/site.toml JAX_PLATFORMS=cpu \\
        python -m pytest -p no:cacheprovider tests/worker/test_local_preflight.py -m site
"""

from __future__ import annotations

from datetime import UTC, datetime
from hashlib import sha256
import json
import os
from pathlib import Path
import socket
import subprocess

import pytest

from glm_tpu import envs
from glm_tpu.config.site import SiteConfig
from glm_tpu.engine import resident_protocol as protocol
from glm_tpu.executor import fleet as remote
from glm_tpu.executor import launch_policy
from glm_tpu.engine import request
from glm_tpu.utils.json_utils import canonical

pytestmark = pytest.mark.site

TIMEOUT = 900 * envs.GLM_TPU_TEST_TIMEOUT_SCALE
# A host shell's environment, not pytest's: the command strings set everything the worker needs.
SHELL_ENV = ("PATH", "HOME", "USER", "LOGNAME", "LANG", "LC_ALL", "TMPDIR")


def _site_file() -> Path:
    path = envs.GLM_TPU_TEST_SITE
    if path is None:
        pytest.skip("site tier: set GLM_TPU_TEST_SITE=<site file> on a fleet host")
    return path


def _shell(command: str, *, payload: bytes | None = None) -> subprocess.CompletedProcess[bytes]:
    env = {name: os.environ[name] for name in SHELL_ENV if name in os.environ}
    return subprocess.run(
        ["bash", "-c", command], input=payload, capture_output=True, env=env, timeout=TIMEOUT, check=False
    )


def _last_line(data: bytes) -> str:
    lines = data.decode(errors="replace").strip().splitlines()
    return lines[-1] if lines else ""


def test_the_cpu_worker_preflight_admits_a_locally_staged_bundle_and_refuses_a_changed_file(tmp_path, monkeypatch):
    site_file = _site_file()
    from glm_tpu.executor import staging
    from tools.equivalence.budget import live_tpu_run, refusal_reason

    monkeypatch.setenv("GLM_TPU_SITE_CONFIG", str(site_file))  # the budget rule reads its workload locks
    if live_tpu_run():
        pytest.skip(refusal_reason())
    runs = tmp_path / "runs"
    runs.mkdir(mode=0o700)
    monkeypatch.setenv("GLM_TPU_RUN_ROOT", str(runs))
    site = SiteConfig.load(site_file)  # as the controller loads it: file + controller-side overrides
    assert site.paths.run_root == runs
    fleet = site.fleet
    hostname = socket.gethostname()
    rank = fleet.host_rank(hostname)
    if rank is None or not 0 <= rank < fleet.num_hosts:
        pytest.skip("not a host of the site's fleet")

    repo = launch_policy.package_checkout()
    pin = subprocess.run(
        ["git", "rev-parse", "--verify", "HEAD^{commit}"], cwd=repo, check=True, capture_output=True, text=True
    ).stdout.strip()
    value = request.from_token_ids([30, 31, 32], request_id="site-preflight", max_new_tokens=2)
    raw = canonical(value) + b"\n"
    root = runs / ("optimized_request_" + datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ"))
    previous = os.umask(0o077)
    try:
        if rank == 0:  # the controller creates the run directory on rank 0; the helper elsewhere
            root.mkdir(mode=0o700)
        bundle, manifest_sha = staging.stage_bundle(repo, pin, root, raw, site)
    finally:
        os.umask(previous)
    hosts = [f"placeholder-host-{r}" for r in range(fleet.num_hosts)]
    hosts[rank] = hostname
    staged = _shell(
        remote.command(fleet, "stage_bundle", dict(root=str(root), digest=sha256(bundle).hexdigest(), hosts=hosts)),
        payload=bundle,
    )
    assert staged.returncode == 0, _last_line(staged.stderr)
    assert (root / "source" / "glm_tpu" / "worker" / "tpu_worker.py").is_file()
    assert sha256((root / "site.json").read_bytes()).hexdigest() == site.resolved_sha256()

    # The launcher's worker argv (launch_ws32_optimized_request.main), then its preflight string.
    command = [
        fleet.worker_python,
        "-m",
        protocol.WORKER_MODULE,
        "--output",
        str(root),
        "--code-hash",
        pin,
        "--source-manifest-sha256",
        manifest_sha,
        "--request-file-sha256",
        sha256(raw).hexdigest(),
        "--site-sha256",
        site.resolved_sha256(),
        "--topology-rebinding-sha256",
        site.topology.binding_sha256,
        "--coordinator-address",
        fleet.coordinator_address,
        "--wall-seconds",
        "60",
    ]
    preflight = remote.preflight_command(fleet, root, command)
    result = _shell(preflight)
    assert result.returncode == 0, _last_line(result.stderr)
    environment = json.loads(_last_line(result.stdout))
    assert environment["hostname"] == hostname
    assert environment["jax"] == environment["jaxlib"] == "0.10.1"
    assert set(environment["sha256"]) == {"python", "jax", "jaxlib", "libtpu"}
    assert sorted(p.name for p in root.iterdir()) == sorted(
        ["request.json", "site.json", "source", "source_manifest.json", "topology_capture", "topology_rebinding.json"]
    )  # the preflight wrote nothing

    # Sensitivity: one staged source byte changed after staging is refused before any device use.
    target = root / "source" / "README.md"
    target.write_bytes(target.read_bytes() + b"\n")
    refused = _shell(preflight)
    assert refused.returncode != 0
    assert "deployed source differs" in refused.stderr.decode(errors="replace")

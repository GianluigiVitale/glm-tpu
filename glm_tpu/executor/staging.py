"""The staged bundle of a launch: ``git archive`` of the pinned commit with its source manifest, the
request and the resolved site.
"""

from hashlib import sha256
import io
import json
import subprocess
import tarfile

from glm_tpu.utils import json_utils
from glm_tpu.executor.fleet import require


def stage_bundle(repo, pin, root, raw, site):
    archive = subprocess.check_output(["git", "archive", "--format=tar", pin], cwd=repo)
    files = {}
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:") as tar:
        for member in tar:
            require(member.isfile() or member.isdir(), "release archive contains a non-regular entry")
            if member.isfile():
                files["source/" + member.name] = tar.extractfile(member).read()
    manifest = {name.removeprefix("source/"): sha256(data).hexdigest() for name, data in files.items()}
    manifest_raw = json_utils.canonical(manifest) + b"\n"
    binding_dir = site.topology.binding_dir
    files.update(
        {
            "request.json": raw,
            "source_manifest.json": manifest_raw,
            "site.json": site.resolved_json(),
            "topology_rebinding.json": (binding_dir / "topology_rebinding.json").read_bytes(),
        }
    )
    require(
        sha256(files["topology_rebinding.json"]).hexdigest() == site.topology.binding_sha256, "site rebinding changed"
    )
    binding = json.loads(files["topology_rebinding.json"])
    for rank in range(8):
        name = f"topology.rank{rank}.json"
        data = (binding_dir / "captures" / name).read_bytes()
        require(sha256(data).hexdigest() == binding["capture_sha256"][name], "topology capture differs")
        files["topology_capture/" + name] = data
    result = io.BytesIO()
    with tarfile.open(fileobj=result, mode="w:gz") as tar:
        for name, data in files.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            info.mode = 0o600
            tar.addfile(info, io.BytesIO(data))
    return result.getvalue(), sha256(manifest_raw).hexdigest()

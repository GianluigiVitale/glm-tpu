"""Bounded original publication with fixture storage, never real cloud writes."""

from pathlib import Path
from types import SimpleNamespace as NS

from google.api_core.exceptions import PreconditionFailed
import pytest

from scripts.greenfield import ws32_prefill_frontier_publish as publisher

TAG = "greenfield_ws32_short_decoder_8k_numerical_c128_hrope_bp1_ps1_rp1_ep1_lm1_first128_20260909T120000000000000Z"
ROOT = Path("/home/gianl/glm-run") / TAG / "first_window.rank3"
REMOTE = "gs://driftbench-dsv4-uc/results/" + TAG


def test_bounded_completed_and_known_partial_originals(tmp_path):
    for name in ("wide_initial.npz", "wide_initial.json", "wide_final.npz.pending", ".wide_final.json.tmp.123"):
        (tmp_path / name).write_bytes(b"x")
    assert len(publisher.originals(tmp_path)) == 4
    (tmp_path / "unknown.bin").write_bytes(b"x")
    with pytest.raises(ValueError, match="unknown"):
        publisher.originals(tmp_path)


def test_symlink_and_budget_refused_before_read(tmp_path, monkeypatch):
    file = tmp_path / "runner.json"
    file.symlink_to(tmp_path / "absent")
    with pytest.raises(ValueError, match="nonregular"):
        publisher.originals(tmp_path)
    file.unlink()
    file.write_bytes(b"12345")
    monkeypatch.setattr(publisher, "LIMIT", 4)
    with pytest.raises(ValueError, match="budget"):
        publisher.originals(tmp_path)


@pytest.mark.parametrize("existing", [None, b"original", b"wrong"])
def test_append_only_exact_generation_readback(tmp_path, monkeypatch, existing):
    path = tmp_path / "runner.json"
    path.write_bytes(b"original")
    monkeypatch.setattr(publisher, "originals", lambda root: [path])
    objects = {} if existing is None else {"raw": existing}
    generation = 9007199254740993
    events = []
    class Blob:
        def __init__(self, version=None):
            self.generation = generation
            self.crc32c = "fixture-crc"
            self.version = version
        def upload_from_string(self, raw, **kwargs):
            assert kwargs == dict(if_generation_match=0, checksum="crc32c")
            events.append("upload")
            if "raw" in objects:
                raise PreconditionFailed("existing")
            objects["raw"] = raw
        def reload(self):
            events.append("reload")
        def download_as_bytes(self, **kwargs):
            assert self.version == generation
            assert kwargs == dict(if_generation_match=generation, checksum="crc32c")
            events.append("readback")
            return objects["raw"]
    def blob(name, generation=None):
        assert name == f"results/{TAG}/diagnostic_local/{TAG}/first_window.rank3/runner.json"
        return Blob(generation)
    client = NS(bucket=lambda name: NS(blob=blob) if name == "driftbench-dsv4-uc" else None)
    if existing == b"wrong":
        with pytest.raises(ValueError, match="differs"):
            publisher.publish(ROOT, REMOTE, client=client)
    else:
        receipts = publisher.publish(ROOT, REMOTE, client=client)
        assert receipts[0]["generation"] == str(generation)
        assert receipts[0]["bytes"] == 8
        assert events[-1] == "readback"


@pytest.mark.parametrize("root,remote", [
    (ROOT, REMOTE.replace("driftbench-dsv4-uc", "other-bucket")),
    (ROOT.with_name("first_window.rank8"), REMOTE),
    (ROOT, REMOTE + "/extra"),
    (ROOT.parent.with_name("another-tag") / ROOT.name, REMOTE),
])
def test_wrong_bucket_tag_or_owner_refuses_without_client(root, remote):
    with pytest.raises(ValueError):
        publisher.publish(root, remote, client=None)

from __future__ import annotations

from io import BytesIO
import json
from pathlib import Path

from glm_tpu.greenfield.checkpoint import (
    DestinationFilePlan,
    StreamedFileEvidence,
)
from scripts.greenfield import pack_checkpoint_streaming as script


class _BlobStream(BytesIO):
    def __init__(self, blob: "_Blob") -> None:
        super().__init__()
        self._blob = blob
        self.terminated = False

    def close(self) -> None:
        if not self.closed:
            self._blob.payload = self.getvalue()
        super().close()

    def terminate(self) -> None:
        self.terminated = True
        super().close()


class _Blob:
    def __init__(self, name: str) -> None:
        self.name = name
        self.payload: bytes | None = None
        self.metadata: dict[str, str] = {}
        self.content_type: str | None = None
        self.generation = 1
        self.size = 0
        self.crc32c = "fixture-crc32c"
        self.last_stream: _BlobStream | None = None

    def exists(self) -> bool:
        return self.payload is not None

    def open(self, *_args: object, **_kwargs: object) -> _BlobStream:
        self.last_stream = _BlobStream(self)
        return self.last_stream

    def reload(self) -> None:
        self.size = len(self.payload or b"")

    def patch(self, **_kwargs: object) -> None:
        return None

    def upload_from_string(self, value: str, **_kwargs: object) -> None:
        self.payload = value.encode()
        self.reload()


class _Bucket:
    def __init__(self) -> None:
        self.blobs: dict[str, _Blob] = {}

    def blob(self, name: str, **_kwargs: object) -> _Blob:
        return self.blobs.setdefault(name, _Blob(name))


def _plan(slot: int) -> DestinationFilePlan:
    return DestinationFilePlan(
        filename=f"base_decoder/stage_00/device_slot_{slot:02d}.safetensors",
        load_set="base_decoder",
        stage_id=0,
        device_slot=slot,
        device_id=slot,
        header=b"h",
        payload_bytes=2,
        tensors=(),
    )


def test_full_packer_is_fail_closed_to_exact_tpu_region() -> None:
    source = Path(script.__file__).read_text()
    assert 'APPROVED_LOCATION = "US-CENTRAL2"' in source
    assert "bucket.reload(timeout=300)" in source
    assert "bucket.location != APPROVED_LOCATION" in source
    assert '"bucket_location": APPROVED_LOCATION' in source


def test_pending_pp16_group_streams_both_owners_in_one_source_pass(
    tmp_path: Path,
    monkeypatch,
) -> None:
    plans = (_plan(0), _plan(1))
    calls: list[tuple[str, ...]] = []

    def fake_stream_pack_group(**kwargs: object):
        selected = tuple(kwargs["plans"])
        outputs = kwargs["outputs"]
        calls.append(tuple(plan.filename for plan in selected))
        result = []
        for plan in selected:
            outputs[plan.filename].write(b"hab")
            result.append(
                StreamedFileEvidence(
                    filename=plan.filename,
                    file_bytes=3,
                    sha256=f"{plan.device_slot + 1:064x}",
                )
            )
        return tuple(result)

    monkeypatch.setattr(script, "stream_pack_group", fake_stream_pack_group)
    bucket = _Bucket()
    evidence = script._stream_pending_group(
        bucket=bucket,
        prefix="checkpoint",
        layout={"manifest_sha256": "a" * 64, "plan_id": "PP16_LP2"},
        plans=plans,
        source_root=tmp_path,
        run_dir=tmp_path / "run",
        code_hash="b" * 40,
    )

    assert calls == [tuple(plan.filename for plan in plans)]
    assert [item["destination_filename"] for item in evidence] == [
        plan.filename for plan in plans
    ]
    for plan in plans:
        blob = bucket.blobs[f"checkpoint/{plan.filename}"]
        assert blob.payload == b"hab"
        sidecar = bucket.blobs[
            f"checkpoint/evidence/{plan.filename}.json"
        ]
        assert json.loads(sidecar.payload)["destination_filename"] == plan.filename


def test_pending_group_terminates_all_owner_streams_on_pack_failure(
    tmp_path: Path,
    monkeypatch,
) -> None:
    plans = (_plan(0), _plan(1))
    opened: list[_BlobStream] = []
    bucket = _Bucket()

    original_open = _Blob.open

    def tracking_open(blob: _Blob, *args: object, **kwargs: object) -> _BlobStream:
        stream = original_open(blob, *args, **kwargs)
        opened.append(stream)
        return stream

    monkeypatch.setattr(_Blob, "open", tracking_open)

    def fail_pack(**_kwargs: object):
        raise RuntimeError("fixture pack failure")

    monkeypatch.setattr(script, "stream_pack_group", fail_pack)
    try:
        script._stream_pending_group(
            bucket=bucket,
            prefix="checkpoint",
            layout={"manifest_sha256": "a" * 64, "plan_id": "PP16_LP2"},
            plans=plans,
            source_root=tmp_path,
            run_dir=tmp_path / "run",
            code_hash="b" * 40,
        )
    except RuntimeError as exc:
        assert str(exc) == "fixture pack failure"
    else:
        raise AssertionError("pack failure was not propagated")

    assert len(opened) == 2
    assert all(stream.terminated for stream in opened)


def test_pending_group_terminates_open_stream_if_later_owner_exists(
    tmp_path: Path,
) -> None:
    plans = (_plan(0), _plan(1))
    bucket = _Bucket()
    occupied = bucket.blob(f"checkpoint/{plans[1].filename}")
    occupied.payload = b"occupied"

    try:
        script._stream_pending_group(
            bucket=bucket,
            prefix="checkpoint",
            layout={"manifest_sha256": "a" * 64, "plan_id": "PP16_LP2"},
            plans=plans,
            source_root=tmp_path,
            run_dir=tmp_path / "run",
            code_hash="b" * 40,
        )
    except RuntimeError as exc:
        assert "destination appeared without accepted evidence" in str(exc)
    else:
        raise AssertionError("occupied later owner was not rejected")

    first = bucket.blobs[f"checkpoint/{plans[0].filename}"]
    assert first.payload is None
    assert first.last_stream is not None and first.last_stream.terminated

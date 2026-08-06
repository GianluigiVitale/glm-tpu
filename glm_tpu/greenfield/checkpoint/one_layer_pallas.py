"""Append-only final-MXU-layout derivative of one protected MoE layer.

The first bounded one-layer artifact already proves source identity and
topology-local ownership, but its routed experts retain checkpoint ``[N,K]``
orientation. Selected-expert Pallas kernels require persistent ``[K,N]``
order to avoid a complete-table transpose in every decode. This derivative
transposes only routed FP8 payloads offline; scales, shared-expert shards,
router state, ownership, and source identities remain unchanged.
"""

from __future__ import annotations

from contextlib import ExitStack
from dataclasses import dataclass
import gc
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Mapping

from .one_layer import inspect_one_layer_artifact


PALLAS_ONE_LAYER_FORMAT_VERSION = 1
PALLAS_ONE_LAYER_ARTIFACT_KIND = "greenfield_one_layer_moe_pallas_final"
PALLAS_ONE_LAYER_LAYOUT_ID = "selected_expert_kn_v1"
_ROUTED_WEIGHTS = frozenset(
    ("expert_gate", "expert_up", "expert_down")
)


def _canonical_json(value: Mapping[str, Any]) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def _manifest_hash(value: Mapping[str, Any]) -> str:
    without_hash = dict(value)
    without_hash.pop("manifest_sha256", None)
    return sha256(_canonical_json(without_hash).encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tensor_sha256(tensor: Any) -> str:
    import torch

    contiguous = tensor.contiguous()
    byte_view = contiguous.view(torch.uint8).numpy()
    return sha256(memoryview(byte_view).cast("B")).hexdigest()


def _tensor_nbytes(shape: list[int], dtype: str) -> int:
    item_size = {"BF16": 2, "F32": 4, "F8_E4M3": 1}.get(dtype)
    if item_size is None:
        raise ValueError(f"unsupported Pallas derivative dtype {dtype!r}")
    elements = 1
    for dimension in shape:
        if not isinstance(dimension, int) or isinstance(dimension, bool) or (
            dimension <= 0
        ):
            raise ValueError("Pallas derivative shapes must be positive")
        elements *= dimension
    return elements * item_size


@dataclass(frozen=True, slots=True)
class PallasOneLayerPackConfig:
    """Immutable identities for one final-layout derivative."""

    source_artifact_dir: Path
    source_artifact_uri: str
    source_manifest_sha256: str
    output_dir: Path
    code_hash: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "source_artifact_dir", Path(self.source_artifact_dir)
        )
        object.__setattr__(self, "output_dir", Path(self.output_dir))
        if not self.source_artifact_uri.startswith(
            "gs://driftbench-dsv4-uc/"
        ):
            raise ValueError(
                "source_artifact_uri must use the approved driftbench bucket"
            )
        if len(self.source_manifest_sha256) != 64 or any(
            character not in "0123456789abcdef"
            for character in self.source_manifest_sha256
        ):
            raise ValueError("source manifest must be a lowercase SHA-256")
        if len(self.code_hash) not in (40, 64) or any(
            character not in "0123456789abcdef"
            for character in self.code_hash
        ):
            raise ValueError("code_hash must be a lowercase Git object id")


def build_pallas_one_layer_layout(
    source_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """Return the complete semantic layout document bound by the manifest."""

    geometry = source_manifest["geometry"]
    hidden = int(geometry["hidden_size"])
    intermediate = int(geometry["intermediate_size"])
    local_experts = int(geometry["num_experts"]) // int(
        geometry["stage_size"]
    )
    layout: dict[str, Any] = {
        "layout_id": PALLAS_ONE_LAYER_LAYOUT_ID,
        "routed": {
            "expert_down": {
                "order": ["local_expert", "contraction", "output"],
                "shape": [local_experts, intermediate, hidden],
                "source_transform": "transpose_last_two",
            },
            "expert_gate": {
                "order": ["local_expert", "contraction", "output"],
                "shape": [local_experts, hidden, intermediate],
                "source_transform": "transpose_last_two",
            },
            "expert_up": {
                "order": ["local_expert", "contraction", "output"],
                "shape": [local_experts, hidden, intermediate],
                "source_transform": "transpose_last_two",
            },
            "scale_order": ["local_expert", "output_block", "input_block"],
        },
        "shared": {
            "order": "checkpoint_out_in",
            "source_transform": "identity",
        },
    }
    layout["layout_sha256"] = sha256(
        _canonical_json(layout).encode("utf-8")
    ).hexdigest()
    return layout


def _pack_file(
    source_dir: Path,
    output_dir: Path,
    source_file_record: Mapping[str, Any],
    *,
    source_manifest_sha256: str,
    layout_sha256: str,
    source_artifact_uri: str,
) -> dict[str, Any]:
    from safetensors import safe_open
    from safetensors.torch import save_file

    source_path = source_dir / str(source_file_record["filename"])
    tensors: dict[str, Any] = {}
    records = []
    source_records = {
        str(record["name"]): record
        for record in source_file_record["tensors"]
    }
    with safe_open(source_path, framework="pt", device="cpu") as source:
        if set(source.keys()) != set(source_records):
            raise ValueError(
                f"source tensor keys drifted for {source_path.name}"
            )
        for name in sorted(source_records):
            source_tensor = source.get_tensor(name)
            transform = (
                "transpose_last_two" if name in _ROUTED_WEIGHTS else "identity"
            )
            destination = (
                source_tensor.transpose(-2, -1).contiguous()
                if transform == "transpose_last_two"
                else source_tensor.contiguous()
            )
            source_record = source_records[name]
            shape = list(destination.shape)
            byte_count = _tensor_nbytes(shape, str(source_record["dtype"]))
            if byte_count != int(source_record["byte_count"]):
                raise ValueError(
                    f"layout transform changed payload bytes for {name!r}"
                )
            tensors[name] = destination
            records.append(
                {
                    "byte_count": byte_count,
                    "dtype": source_record["dtype"],
                    "name": name,
                    "ownership": source_record["ownership"],
                    "sha256": _tensor_sha256(destination),
                    "shape": shape,
                    "source_name": name,
                    "source_shape": source_record["shape"],
                    "source_tensor_sha256": _tensor_sha256(source_tensor),
                    "transform": transform,
                }
            )

    filename = str(source_file_record["filename"])
    path = output_dir / filename
    partial = path.with_suffix(path.suffix + ".partial")
    save_file(
        tensors,
        partial,
        metadata={
            "artifact_kind": PALLAS_ONE_LAYER_ARTIFACT_KIND,
            "device_slot": str(source_file_record["device_slot"]),
            "format_version": str(PALLAS_ONE_LAYER_FORMAT_VERSION),
            "layout_sha256": layout_sha256,
            "source_artifact_uri": source_artifact_uri,
            "source_manifest_sha256": source_manifest_sha256,
        },
    )
    partial.replace(path)
    result = {
        "device_slot": source_file_record["device_slot"],
        "file_byte_count": path.stat().st_size,
        "filename": filename,
        "payload_byte_count": sum(
            int(record["byte_count"]) for record in records
        ),
        "sha256": _sha256_file(path),
        "source_file_sha256": source_file_record["sha256"],
        "tensors": records,
    }
    del tensors
    gc.collect()
    return result


def pack_pallas_one_layer(
    config: PallasOneLayerPackConfig,
) -> dict[str, Any]:
    """Transpose routed FP8 tables once and seal the derivative manifest."""

    source = inspect_one_layer_artifact(config.source_artifact_dir)
    if source["manifest_sha256"] != config.source_manifest_sha256:
        raise ValueError("source one-layer manifest identity drifted")
    if config.output_dir.exists():
        raise FileExistsError(
            "append-only Pallas one-layer destination already exists: "
            f"{config.output_dir}"
        )
    config.output_dir.mkdir(parents=True)
    layout = build_pallas_one_layer_layout(source)
    files = [
        _pack_file(
            config.source_artifact_dir,
            config.output_dir,
            source_file,
            source_manifest_sha256=config.source_manifest_sha256,
            layout_sha256=layout["layout_sha256"],
            source_artifact_uri=config.source_artifact_uri.rstrip("/"),
        )
        for source_file in sorted(
            source["files"], key=lambda record: record["device_slot"]
        )
    ]
    manifest: dict[str, Any] = {
        "artifact_kind": PALLAS_ONE_LAYER_ARTIFACT_KIND,
        "code_hash": config.code_hash,
        "files": files,
        "format_version": PALLAS_ONE_LAYER_FORMAT_VERSION,
        "geometry": source["geometry"],
        "layer": source["layer"],
        "layout": layout,
        "model_id": source["model_id"],
        "packed_payload_byte_count": sum(
            int(record["payload_byte_count"]) for record in files
        ),
        "plan_group_hash": source["plan_group_hash"],
        "plan_id": source["plan_id"],
        "source_artifact_kind": source["artifact_kind"],
        "source_artifact_uri": config.source_artifact_uri.rstrip("/"),
        "source_code_hash": source["code_hash"],
        "source_manifest_sha256": source["manifest_sha256"],
        "source_packed_payload_byte_count": source[
            "packed_payload_byte_count"
        ],
        "source_revision": source["source_revision"],
        "topology_hash": source["topology_hash"],
    }
    if manifest["packed_payload_byte_count"] != manifest[
        "source_packed_payload_byte_count"
    ]:
        raise ValueError("Pallas derivative payload bytes do not reconcile")
    manifest["manifest_sha256"] = _manifest_hash(manifest)
    (config.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    inspect_pallas_one_layer_artifact(
        config.output_dir,
        source_artifact_dir=config.source_artifact_dir,
    )
    return manifest


def inspect_pallas_one_layer_artifact(
    output_dir: Path,
    *,
    source_artifact_dir: Path | None = None,
) -> dict[str, Any]:
    """Verify every file/tensor hash and optionally the source transform."""

    from safetensors import safe_open
    import torch

    output_dir = Path(output_dir)
    path = output_dir / "manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"missing Pallas one-layer manifest {path}")
    manifest = json.loads(path.read_text())
    if manifest.get("artifact_kind") != PALLAS_ONE_LAYER_ARTIFACT_KIND:
        raise ValueError("not a Pallas final-layout one-layer artifact")
    if manifest.get("format_version") != PALLAS_ONE_LAYER_FORMAT_VERSION:
        raise ValueError("unsupported Pallas one-layer format version")
    if manifest.get("manifest_sha256") != _manifest_hash(manifest):
        raise ValueError("Pallas one-layer manifest checksum mismatch")
    layout = build_pallas_one_layer_layout(manifest)
    if manifest.get("layout") != layout:
        raise ValueError("Pallas one-layer semantic layout drifted")
    files = manifest.get("files")
    stage_size = int(manifest["geometry"]["stage_size"])
    if not isinstance(files, list) or len(files) != stage_size:
        raise ValueError("Pallas one-layer artifact has incomplete device files")

    source = None
    source_files: dict[int, Mapping[str, Any]] = {}
    if source_artifact_dir is not None:
        source = inspect_one_layer_artifact(Path(source_artifact_dir))
        if source["manifest_sha256"] != manifest["source_manifest_sha256"]:
            raise ValueError("Pallas derivative names the wrong source artifact")
        source_files = {
            int(record["device_slot"]): record for record in source["files"]
        }

    payload_total = 0
    for expected_slot, file_record in enumerate(
        sorted(files, key=lambda record: record["device_slot"])
    ):
        if int(file_record["device_slot"]) != expected_slot:
            raise ValueError("Pallas device slots are not contiguous")
        packed_path = output_dir / str(file_record["filename"])
        if packed_path.stat().st_size != int(file_record["file_byte_count"]):
            raise ValueError(f"Pallas packed file size mismatch for {packed_path.name}")
        if _sha256_file(packed_path) != file_record["sha256"]:
            raise ValueError(
                f"Pallas packed file checksum mismatch for {packed_path.name}"
            )
        expected = {
            str(record["name"]): record for record in file_record["tensors"]
        }
        with ExitStack() as stack:
            source_handle = None
            if source is not None:
                source_record = source_files[expected_slot]
                if file_record["source_file_sha256"] != source_record["sha256"]:
                    raise ValueError(
                        "Pallas derivative source file identity drifted"
                    )
                source_handle = stack.enter_context(
                    safe_open(
                        Path(source_artifact_dir)
                        / str(source_record["filename"]),
                        framework="pt",
                        device="cpu",
                    )
                )
            packed = stack.enter_context(
                safe_open(packed_path, framework="pt", device="cpu")
            )
            metadata = packed.metadata()
            expected_metadata = {
                "artifact_kind": PALLAS_ONE_LAYER_ARTIFACT_KIND,
                "device_slot": str(expected_slot),
                "format_version": str(PALLAS_ONE_LAYER_FORMAT_VERSION),
                "layout_sha256": manifest["layout"]["layout_sha256"],
                "source_artifact_uri": manifest["source_artifact_uri"],
                "source_manifest_sha256": manifest[
                    "source_manifest_sha256"
                ],
            }
            if metadata != expected_metadata:
                raise ValueError(
                    f"Pallas safetensor metadata mismatch for {packed_path.name}"
                )
            if set(packed.keys()) != set(expected):
                raise ValueError(
                    f"Pallas tensor key mismatch for {packed_path.name}"
                )
            for name, record in expected.items():
                tensor_slice = packed.get_slice(name)
                if list(tensor_slice.get_shape()) != record["shape"] or (
                    tensor_slice.get_dtype() != record["dtype"]
                ):
                    raise ValueError(
                        f"Pallas tensor metadata mismatch for {name!r}"
                    )
                if _tensor_nbytes(
                    record["shape"], record["dtype"]
                ) != int(record["byte_count"]):
                    raise ValueError(
                        f"Pallas tensor byte mismatch for {name!r}"
                    )
                tensor = packed.get_tensor(name)
                if _tensor_sha256(tensor) != record["sha256"]:
                    raise ValueError(
                        f"Pallas tensor checksum mismatch for {name!r}"
                    )
                if source_handle is not None:
                    source_tensor = source_handle.get_tensor(
                        str(record["source_name"])
                    )
                    if _tensor_sha256(source_tensor) != record[
                        "source_tensor_sha256"
                    ]:
                        raise ValueError(
                            f"Pallas source tensor checksum drift for {name!r}"
                        )
                    transformed = (
                        source_tensor.transpose(-2, -1)
                        if record["transform"] == "transpose_last_two"
                        else source_tensor
                    )
                    if not torch.equal(tensor, transformed):
                        raise ValueError(
                            f"Pallas source transform mismatch for {name!r}"
                        )
        payload = sum(int(record["byte_count"]) for record in expected.values())
        if payload != int(file_record["payload_byte_count"]):
            raise ValueError(
                f"Pallas payload total mismatch for {packed_path.name}"
            )
        payload_total += payload
        gc.collect()
    if payload_total != int(manifest["packed_payload_byte_count"]):
        raise ValueError("Pallas manifest payload total mismatch")
    return manifest

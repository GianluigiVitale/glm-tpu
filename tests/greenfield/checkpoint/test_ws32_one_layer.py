from __future__ import annotations

import base64
from contextlib import ExitStack
from copy import deepcopy
from dataclasses import replace
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import subprocess
import sys

import numpy as np
import pytest

from glm_tpu.greenfield.checkpoint.one_layer import pack_one_layer_moe
from glm_tpu.greenfield.checkpoint.ws32_one_layer import (
    Ws32OneLayerPackConfig,
    inspect_ws32_one_layer,
    load_ws32_one_layer_slot,
    pack_ws32_one_layer,
)
from tests.greenfield.checkpoint.test_one_layer import (
    tiny_config,
    write_tiny_source,
)


REPO = Path(__file__).resolve().parents[3]
REAL_SOURCE_MANIFEST = Path(
    "/home/gianl/gcs-models/checkpoints/greenfield/glm52/layer3/PP8_LP4/"
    "greenfield_one_layer_pack_20260805T151828912346032Z/manifest.json"
)


def _source_and_config(tmp_path: Path) -> tuple[dict, Ws32OneLayerPackConfig]:
    source_input = tmp_path / "source_input"
    source_artifact = tmp_path / "pp8_artifact"
    pp8_config = tiny_config(source_input, source_artifact)
    write_tiny_source(pp8_config)
    source_manifest = pack_one_layer_moe(pp8_config)
    config = Ws32OneLayerPackConfig(
        source_manifest_path=source_artifact / "manifest.json",
        source_payload_dir=source_artifact,
        source_artifact_uri=(
            "gs://driftbench-dsv4-uc/checkpoints/test/pp8_one_layer"
        ),
        source_manifest_sha256=source_manifest["manifest_sha256"],
        output_dir=tmp_path / "ws32_artifact",
        code_hash="a" * 40,
        mesh_hash="d" * 64,
    )
    return source_manifest, config


def _rehash_manifest(manifest: dict) -> None:
    value = dict(manifest)
    value.pop("manifest_sha256", None)
    manifest["manifest_sha256"] = sha256(
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()


def test_ws32_one_layer_derivative_reconciles_exact_final_owners(
    tmp_path: Path,
) -> None:
    import torch
    from safetensors import safe_open

    source, config = _source_and_config(tmp_path)
    manifest = pack_ws32_one_layer(config)
    assert manifest["plan_id"] == "WS32_2D"
    assert manifest["geometry"]["expert_axis_size"] == 8
    assert manifest["geometry"]["feature_axis_size"] == 4
    assert len(manifest["files"]) == 32
    assert [record["device_slot"] for record in manifest["files"]] == list(
        range(32)
    )
    shared_bytes = sum(
        record["byte_count"]
        for record in source["source_leaves"]
        if ".shared_experts." in record["name"]
    )
    bias_bytes = next(
        record["byte_count"]
        for record in source["source_leaves"]
        if record["name"].endswith("e_score_correction_bias")
    )
    assert manifest["packed_payload_byte_count"] == (
        source["source_payload_byte_count"]
        + 7 * shared_bytes
        + 3 * bias_bytes
    )
    assert manifest["replication"] == {
        "correction_bias_extra_bytes": 3 * bias_bytes,
        "shared_expert_extra_bytes": 7 * shared_bytes,
    }
    assert inspect_ws32_one_layer(
        config.output_dir, verify_tensor_hashes=True
    ) == manifest

    # Expert coordinate five comes from the second expert in PP8 source slot
    # two; feature coordinate two owns hidden columns [4:6].
    target_slot = 5 * 4 + 2
    with safe_open(
        config.source_payload_dir / "device_slot_02.safetensors",
        framework="pt",
        device="cpu",
    ) as source_file, safe_open(
        config.output_dir / f"device_slot_{target_slot:02d}.safetensors",
        framework="pt",
        device="cpu",
    ) as target_file:
        expected_gate_bits = (
            source_file.get_tensor("expert_gate")[1:2, :, 4:6]
            .contiguous()
            .view(torch.uint8)
        )
        assert torch.equal(
            target_file.get_tensor("expert_gate_bits"), expected_gate_bits
        )
        assert target_file.get_slice("expert_gate_bits").get_shape() == [1, 4, 2]
        assert target_file.get_slice("expert_down_bits").get_shape() == [1, 2, 4]
        assert target_file.get_slice("shared_gate_bits").get_shape() == [4, 2]
        assert target_file.get_slice("shared_down_bits").get_shape() == [2, 4]
        assert target_file.get_slice("router_weight").get_shape() == [1, 2]
        assert target_file.get_slice("correction_bias").get_shape() == [1]

    with ExitStack() as stack:
        source_handles = [
            stack.enter_context(
                safe_open(
                    config.source_payload_dir
                    / f"device_slot_{slot:02d}.safetensors",
                    framework="pt",
                    device="cpu",
                )
            )
            for slot in range(4)
        ]
        shared_expected = {
            "shared_gate_bits": torch.cat(
                tuple(handle.get_tensor("shared_gate") for handle in source_handles),
                dim=0,
            ).view(torch.uint8),
            "shared_up_bits": torch.cat(
                tuple(handle.get_tensor("shared_up") for handle in source_handles),
                dim=0,
            ).view(torch.uint8),
            "shared_down_bits": torch.cat(
                tuple(handle.get_tensor("shared_down") for handle in source_handles),
                dim=1,
            ).view(torch.uint8),
            "shared_gate_scale": torch.cat(
                tuple(
                    handle.get_tensor("shared_gate_scale")
                    for handle in source_handles
                ),
                dim=0,
            ),
            "shared_up_scale": torch.cat(
                tuple(
                    handle.get_tensor("shared_up_scale")
                    for handle in source_handles
                ),
                dim=0,
            ),
            "shared_down_scale": torch.cat(
                tuple(
                    handle.get_tensor("shared_down_scale")
                    for handle in source_handles
                ),
                dim=1,
            ),
        }
        for expert_coordinate in range(8):
            target_tensors: dict[str, list[torch.Tensor]] = {}
            for feature_coordinate in range(4):
                slot = expert_coordinate * 4 + feature_coordinate
                with safe_open(
                    config.output_dir / f"device_slot_{slot:02d}.safetensors",
                    framework="pt",
                    device="cpu",
                ) as handle:
                    for name in handle.keys():
                        target_tensors.setdefault(name, []).append(
                            handle.get_tensor(name)
                        )
            source_slot = expert_coordinate // 2
            source_local_expert = expert_coordinate % 2
            source_handle = source_handles[source_slot]
            for base in ("expert_gate", "expert_up"):
                reconstructed = torch.cat(
                    tuple(target_tensors[f"{base}_bits"]), dim=-1
                )
                expected = source_handle.get_tensor(base)[
                    source_local_expert : source_local_expert + 1
                ].contiguous().view(torch.uint8)
                assert torch.equal(reconstructed, expected)
                reconstructed_scale = torch.cat(
                    tuple(target_tensors[f"{base}_scale"]), dim=-1
                )
                expected_scale = source_handle.get_tensor(f"{base}_scale")[
                    source_local_expert : source_local_expert + 1
                ]
                assert torch.equal(reconstructed_scale, expected_scale)
            reconstructed_down = torch.cat(
                tuple(target_tensors["expert_down_bits"]), dim=1
            )
            expected_down = source_handle.get_tensor("expert_down")[
                source_local_expert : source_local_expert + 1
            ].contiguous().view(torch.uint8)
            assert torch.equal(reconstructed_down, expected_down)
            reconstructed_down_scale = torch.cat(
                tuple(target_tensors["expert_down_scale"]), dim=1
            )
            expected_down_scale = source_handle.get_tensor("expert_down_scale")[
                source_local_expert : source_local_expert + 1
            ]
            assert torch.equal(reconstructed_down_scale, expected_down_scale)
            for name, expected in shared_expected.items():
                axis = 0 if name.startswith("shared_down") else 1
                assert torch.equal(
                    torch.cat(tuple(target_tensors[name]), dim=axis), expected
                )
            expected_router = source_handle.get_tensor("router_weight")[
                expert_coordinate : expert_coordinate + 1
            ]
            assert torch.equal(
                torch.cat(tuple(target_tensors["router_weight"]), dim=1),
                expected_router,
            )
            expected_bias = source_handle.get_tensor("correction_bias")[
                expert_coordinate : expert_coordinate + 1
            ]
            assert all(
                torch.equal(
                    expected_bias,
                    target_tensors["correction_bias"][feature],
                )
                for feature in range(4)
            )

            local_experts = source["geometry"]["num_experts"] // 8
            hidden_local = source["geometry"]["hidden_size"] // 4
            block_out, block_in = source["geometry"]["fp8_block_shape"]
            expert_start = expert_coordinate * local_experts
            expert_end = expert_start + local_experts
            for feature_coordinate in range(4):
                slot = expert_coordinate * 4 + feature_coordinate
                hidden_start = feature_coordinate * hidden_local
                hidden_end = hidden_start + hidden_local
                in_start, in_end = hidden_start // block_in, hidden_end // block_in
                out_start, out_end = (
                    hidden_start // block_out,
                    hidden_end // block_out,
                )
                tensor_records = {
                    item["name"]: item for item in manifest["files"][slot]["tensors"]
                }
                ownership = {
                    "device_slot": slot,
                    "expert_coordinate": expert_coordinate,
                    "feature_coordinate": feature_coordinate,
                    "kind": "ws32_final_owner",
                }
                assert all(
                    record["ownership"] == ownership
                    for record in tensor_records.values()
                )
                assert tensor_records["expert_gate_bits"]["source_slots"] == [
                    source_slot
                ]
                assert tensor_records["expert_gate_bits"]["source_slice"] == {
                    "expert": [expert_start, expert_end],
                    "hidden_input": [hidden_start, hidden_end],
                }
                assert tensor_records["expert_gate_scale"]["source_slice"] == {
                    "expert": [expert_start, expert_end],
                    "hidden_input_blocks": [in_start, in_end],
                }
                assert tensor_records["expert_down_bits"]["source_slice"] == {
                    "expert": [expert_start, expert_end],
                    "hidden_output": [hidden_start, hidden_end],
                }
                assert tensor_records["expert_down_scale"]["source_slice"] == {
                    "expert": [expert_start, expert_end],
                    "hidden_output_blocks": [out_start, out_end],
                }
                assert tensor_records["shared_gate_scale"]["source_slots"] == [
                    0,
                    1,
                    2,
                    3,
                ]
                assert tensor_records["shared_gate_scale"]["source_slice"] == {
                    "hidden_input_blocks": [in_start, in_end]
                }
                assert tensor_records["shared_down_scale"]["source_slice"] == {
                    "hidden_output_blocks": [out_start, out_end]
                }
                assert tensor_records["router_weight"]["source_slice"] == {
                    "expert": [expert_start, expert_end],
                    "hidden_input": [hidden_start, hidden_end],
                }
                assert tensor_records["correction_bias"]["source_slice"] == {
                    "expert": [expert_start, expert_end]
                }

    loaded = load_ws32_one_layer_slot(
        config.output_dir,
        expected_manifest_sha256=manifest["manifest_sha256"],
        device_slot=target_slot,
    )
    assert loaded.expert_coordinate == 5
    assert loaded.feature_coordinate == 2
    assert set(loaded.arrays) == {
        item["name"] for item in manifest["files"][target_slot]["tensors"]
    }
    assert loaded.arrays["expert_gate_bits"].dtype == np.uint8
    assert loaded.arrays["expert_gate_bits"].shape == (1, 4, 2)


def test_ws32_one_layer_refuses_identity_and_file_corruption(
    tmp_path: Path,
) -> None:
    _, config = _source_and_config(tmp_path)
    with pytest.raises(ValueError, match="identity drifted"):
        pack_ws32_one_layer(replace(config, source_manifest_sha256="e" * 64))
    manifest = pack_ws32_one_layer(config)
    with pytest.raises(FileExistsError, match="append-only"):
        pack_ws32_one_layer(config)
    first = config.output_dir / manifest["files"][0]["filename"]
    with first.open("ab") as stream:
        stream.write(b"corrupt")
    with pytest.raises(ValueError, match="size mismatch"):
        inspect_ws32_one_layer(config.output_dir)


def test_ws32_one_layer_refuses_duplicate_slot_file_and_header_drift(
    tmp_path: Path,
) -> None:
    from safetensors import safe_open
    from safetensors.torch import save_file

    _, config = _source_and_config(tmp_path)
    manifest = pack_ws32_one_layer(config)
    manifest_path = config.output_dir / "manifest.json"
    mutated = deepcopy(manifest)
    mutated["files"][1] = deepcopy(mutated["files"][0])
    mutated["files"][1]["device_slot"] = 1
    mutated["files"][1]["expert_coordinate"] = 0
    mutated["files"][1]["feature_coordinate"] = 1
    _rehash_manifest(mutated)
    manifest_path.write_text(json.dumps(mutated, sort_keys=True))
    with pytest.raises(ValueError, match="filenames are not unique"):
        inspect_ws32_one_layer(config.output_dir, verify_tensor_hashes=True)

    manifest_path.write_text(json.dumps(manifest, sort_keys=True))
    first_path = config.output_dir / manifest["files"][0]["filename"]
    with safe_open(first_path, framework="pt", device="cpu") as handle:
        tensors = {name: handle.get_tensor(name) for name in handle.keys()}
        metadata = dict(handle.metadata() or {})
    metadata["feature_coordinate"] = "1"
    save_file(tensors, first_path, metadata=metadata)
    file_record = manifest["files"][0]
    file_record["file_byte_count"] = first_path.stat().st_size
    file_record["sha256"] = sha256(first_path.read_bytes()).hexdigest()
    _rehash_manifest(manifest)
    manifest_path.write_text(json.dumps(manifest, sort_keys=True))
    with pytest.raises(ValueError, match="file metadata mismatch"):
        inspect_ws32_one_layer(config.output_dir, verify_tensor_hashes=True)


def test_ws32_one_layer_does_not_publish_manifest_before_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import glm_tpu.greenfield.checkpoint.ws32_one_layer as ws32_one_layer

    _, config = _source_and_config(tmp_path)

    def refuse_validation(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise ValueError("injected provisional validation failure")

    monkeypatch.setattr(ws32_one_layer, "_verify_ws32_file", refuse_validation)
    with pytest.raises(ValueError, match="injected provisional"):
        pack_ws32_one_layer(config)
    assert not (config.output_dir / "manifest.json").exists()


def test_ws32_one_layer_cli_and_protected_wrapper_are_wired(
    tmp_path: Path,
) -> None:
    source_manifest, config = _source_and_config(tmp_path)
    output = tmp_path / "cli_ws32_artifact"
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    completed = subprocess.run(
        [
            sys.executable,
            str(REPO / "scripts/greenfield/pack_ws32_one_layer.py"),
            "--source-manifest",
            str(config.source_manifest_path),
            "--source-payload-dir",
            str(config.source_payload_dir),
            "--source-artifact-uri",
            config.source_artifact_uri,
            "--source-manifest-sha256",
            source_manifest["manifest_sha256"],
            "--output",
            str(output),
            "--expected-code-hash",
            head,
            "--mesh-hash",
            "d" * 64,
        ],
        cwd=REPO,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    summary = json.loads(completed.stdout)
    manifest = inspect_ws32_one_layer(output, verify_tensor_hashes=True)
    assert summary == {
        "artifact_kind": "greenfield_ws32_one_layer_moe",
        "files": 32,
        "layer": 3,
        "manifest_sha256": manifest["manifest_sha256"],
        "mesh_hash": "d" * 64,
        "packed_payload_byte_count": manifest["packed_payload_byte_count"],
        "source_manifest_sha256": source_manifest["manifest_sha256"],
    }
    wrong_pin = subprocess.run(
        [
            *completed.args[:-4],
            "--expected-code-hash",
            "0" * 40,
            "--mesh-hash",
            "d" * 64,
        ],
        cwd=REPO,
        check=False,
        capture_output=True,
        text=True,
    )
    assert wrong_pin.returncode != 0
    assert "code identity drifted" in wrong_pin.stderr
    wrapper = REPO / "scripts/greenfield/run_pack_ws32_one_layer.sh"
    syntax = subprocess.run(
        ["bash", "-n", str(wrapper)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert syntax.returncode == 0, syntax.stdout + syntax.stderr
    source = wrapper.read_text()
    assert "EXPECTED_PACKED_BYTES=9971249152" in source
    assert "SOURCE_PACKED_PAYLOAD_BYTES=9716380672" in source
    assert "SOURCE_UNIQUE_PAYLOAD_BYTES=9706940416" in source
    assert "WS32_MESH_HASH=de5f59cb" in source
    assert "gcloud storage cp --no-clobber" in source
    assert '"$REMOTE_PREFIX/SUCCESS"' in source
    assert "terminal object set drifted" in source
    heredocs = re.findall(r"<<'PY'[^\n]*\n(.*?)\nPY\n", source, re.DOTALL)
    assert len(heredocs) == 6
    for index, body in enumerate(heredocs):
        compile(body, f"{wrapper}:heredoc-{index}", "exec")

    run_dir = tmp_path / "archive"
    run_dir.mkdir()
    ledger_path = run_dir / "remote_objects.json"
    ledger_path.write_text(json.dumps({"objects": []}))
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "manifest_sha256": "b" * 64,
                "mesh_hash": "d" * 64,
                "source": {"manifest_sha256": "c" * 64},
            }
        )
    )
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_gcloud = fake_bin / "gcloud"
    fake_gcloud.write_text(
        """#!/usr/bin/env python3
import json
import os
from pathlib import Path
import sys

args = sys.argv[1:]
if args[:3] == ["storage", "objects", "describe"]:
    path = Path(os.environ["FAKE_LEDGER"])
    print(json.dumps({
        "crc32c_hash": os.environ["FAKE_LEDGER_CRC32C"],
        "generation": "123",
        "size": path.stat().st_size,
    }))
elif args[:3] == ["storage", "objects", "list"]:
    prefix = os.environ["FAKE_BUCKET_PREFIX"].rstrip("/") + "/"
    for name in json.loads(os.environ["FAKE_OBJECTS"]):
        print(prefix + name)
else:
    raise SystemExit(f"unexpected fake gcloud arguments: {args}")
"""
    )
    fake_gcloud.chmod(0o755)
    expected_nonterminal = {
        "remote_objects.json",
        "source_preflight.json",
        "pack.log",
        "inspection.json",
        "orchestrator.log",
        "evidence.sha256",
        "manifest.json",
    } | {f"packed/device_slot_{slot:02d}.safetensors" for slot in range(32)}
    remote = "gs://driftbench-dsv4-uc/checkpoints/unit/ws32"
    import google_crc32c

    ledger_checksum = google_crc32c.Checksum(ledger_path.read_bytes())
    environment = {
        **os.environ,
        "FAKE_BUCKET_PREFIX": remote.removeprefix("gs://").split("/", 1)[1],
        "FAKE_LEDGER": str(ledger_path),
        "FAKE_LEDGER_CRC32C": base64.b64encode(
            ledger_checksum.digest()
        ).decode("ascii"),
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
    }
    command = [
        sys.executable,
        "-c",
        heredocs[4],
        str(run_dir),
        str(manifest_path),
        remote,
        "a" * 40,
        "greenfield_ws32_one_layer_pack_20260815T000000000000000Z",
    ]
    planted = subprocess.run(
        command,
        env={
            **environment,
            "FAKE_OBJECTS": json.dumps(
                sorted(expected_nonterminal | {"planted-extra"})
            ),
        },
        check=False,
        capture_output=True,
        text=True,
    )
    assert planted.returncode != 0
    assert "nonterminal object set drifted" in planted.stderr
    assert not (run_dir / "SUCCESS").exists()
    exact = subprocess.run(
        command,
        env={
            **environment,
            "FAKE_OBJECTS": json.dumps(sorted(expected_nonterminal)),
        },
        check=False,
        capture_output=True,
        text=True,
    )
    assert exact.returncode == 0, exact.stdout + exact.stderr
    success = json.loads((run_dir / "SUCCESS").read_text())
    assert success["nonterminal_object_count"] == 39
    nonterminal_check = source.index("remote WS32 nonterminal object set drifted")
    success_upload = source.index(
        'gcloud storage cp --no-clobber "$RUN_DIR/SUCCESS"'
    )
    assert nonterminal_check < success_upload


@pytest.mark.skipif(
    not REAL_SOURCE_MANIFEST.is_file(),
    reason="SHA-pinned real PP8 layer artifact is unavailable",
)
def test_ws32_wrapper_source_byte_pins_match_real_manifest() -> None:
    manifest = json.loads(REAL_SOURCE_MANIFEST.read_text())
    assert manifest["manifest_sha256"] == (
        "68ef82011892456409a194f6fa31697dd1e31d96fe1a3f0069228288f613f938"
    )
    assert manifest["packed_payload_byte_count"] == 9_716_380_672
    assert manifest["source_payload_byte_count"] == 9_706_940_416

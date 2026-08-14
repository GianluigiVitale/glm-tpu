from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import shutil

import numpy as np
import pytest

from glm_tpu.greenfield.benchmarking.association_fingerprint import (
    STRATEGY_ND_ALGORITHM,
    array_sha256,
    model_axis_to_physical_input_bits,
    replay_db533_strategy_nd_row0_bits,
    validate_strategy_nd_fingerprint_hlo,
)
from glm_tpu.greenfield.validation.strategy_nd_dense_replay import (
    EXPECTED_FLEET_LOCAL_DEVICE_IDS,
    EXPECTED_HOSTNAMES,
    EXPECTED_MODEL_AXIS_DEVICE_IDS,
    EXPECTED_SOURCE,
    EXPECTED_TOPOLOGY_HASH,
    _recompute_comparison,
    validate_strategy_nd_dense_replay,
)


REAL_DB550_PARTIALS = Path(
    "/home/gianl/glm-run/"
    "greenfield_legacy_layer0_dense_partials_p8155_20260814T100132090917640Z/"
    "dense_partials_capture/dense_partials.npz"
)
REAL_DB550_DIR = REAL_DB550_PARTIALS.parents[1]
REAL_DB533_DIR = Path(
    "/home/gianl/glm-run/greenfield_collective_association_20260811T213152133863450Z"
)
PIN = "1" * 40
TAG = "unit_strategy_nd_dense_replay"
REAL_SOURCES_AVAILABLE = REAL_DB550_PARTIALS.is_file() and REAL_DB533_DIR.is_dir()


def _file_sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _hlo() -> str:
    group = ",".join(map(str, range(32)))
    backend = json.dumps(
        {"collective_algorithm_config": STRATEGY_ND_ALGORITHM},
        separators=(",", ":"),
    )
    return f'''HloModule dense_replay, num_partitions=32, replica_count=1

add {{
  x = bf16[] parameter(0)
  y = bf16[] parameter(1)
  ROOT sum = bf16[] add(x, y)
}}

ENTRY main {{
  input = bf16[32,6144]{{1,0:T(8,128)(2,1)S(3)}} parameter(0)
  ROOT reduced = bf16[32,6144]{{1,0:T(8,128)(2,1)S(3)}} all-reduce(input), replica_groups={{{{{group}}}}}, use_global_device_ids=true, to_apply=add, metadata={{op_name="jit(replay)/shard_map/strategy_nd_association_fingerprint/psum"}}, backend_config={backend}
}}
'''


def _save_array(path: Path, value: np.ndarray) -> dict[str, object]:
    with path.open("wb") as stream:
        np.save(stream, value, allow_pickle=False)
    return {
        "array_sha256": array_sha256(value),
        "dtype": value.dtype.str,
        "file": path.name,
        "file_sha256": _file_sha256(path),
        "shape": list(value.shape),
    }


def _fixture(tmp_path: Path) -> tuple[Path, list[dict[str, object]]]:
    run_dir = tmp_path / TAG
    for name in ("host_records", "hlo", "replay", "source", "source_db533"):
        (run_dir / name).mkdir(parents=True, exist_ok=True)
    shutil.copy2(REAL_DB550_PARTIALS, run_dir / "source" / "dense_partials.npz")
    shutil.copy2(
        REAL_DB550_DIR / "dense_partials_capture" / "capture.json",
        run_dir / "source" / "capture.json",
    )
    shutil.copy2(
        REAL_DB550_DIR / "dense_partials_capture" / "comparison.json",
        run_dir / "source" / "comparison.json",
    )
    shutil.copy2(REAL_DB550_DIR / "SUCCESS", run_dir / "source" / "SUCCESS")
    shutil.copy2(
        REAL_DB550_DIR / "remote_objects.json",
        run_dir / "source" / "remote_objects.json",
    )
    shutil.copy2(
        REAL_DB533_DIR / "association" / "analysis.json",
        run_dir / "source_db533" / "analysis.json",
    )
    shutil.copy2(
        REAL_DB533_DIR / "summary.json",
        run_dir / "source_db533" / "summary.json",
    )
    shutil.copy2(
        REAL_DB533_DIR / "SUCCESS",
        run_dir / "source_db533" / "SUCCESS",
    )
    shutil.copy2(
        REAL_DB533_DIR / "hlo" /
        "strategy_nd_association_bfloat16_32x6144.hlo_contract.json",
        run_dir / "source_db533" / "hlo_contract.json",
    )
    with np.load(REAL_DB550_PARTIALS, allow_pickle=False) as payload:
        model_bits = np.ascontiguousarray(
            payload["accepted_dense_partials_bfloat16_bits"]
        ).reshape(32, 6144)
    physical_input = model_axis_to_physical_input_bits(
        model_bits, EXPECTED_MODEL_AXIS_DEVICE_IDS
    )[None, ...]
    software = replay_db533_strategy_nd_row0_bits(
        model_bits, EXPECTED_MODEL_AXIS_DEVICE_IDS
    )
    hardware = np.broadcast_to(software, (1, 32, 6144)).copy()
    manifest = {
        "physical_input_bits": _save_array(
            run_dir / "replay" / "physical_input_bits.npy", physical_input
        ),
        "hardware_output_bits": _save_array(
            run_dir / "replay" / "hardware_output_bits.npy", hardware
        ),
        "software_row0_bits": _save_array(
            run_dir / "replay" / "software_row0_bits.npy", software
        ),
    }
    (run_dir / "replay" / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    comparison = _recompute_comparison(hardware, software)
    (run_dir / "replay" / "comparison.json").write_text(
        json.dumps(comparison, indent=2, sort_keys=True) + "\n"
    )
    hlo = _hlo()
    hlo_path = (
        run_dir / "hlo" /
        "strategy_nd_dense_partials_bfloat16_32x6144.optimized_hlo.txt"
    )
    hlo_path.write_text(hlo)
    report, algorithm = validate_strategy_nd_fingerprint_hlo(
        hlo, tuple(range(32))
    )
    (
        run_dir / "hlo" /
        "strategy_nd_dense_partials_bfloat16_32x6144.hlo_contract.json"
    ).write_text(
        json.dumps(
            {
                "collective_algorithm": algorithm,
                "decode_shape_admissible": True,
                "hlo": report.to_dict(),
                "valid": True,
            },
            indent=2,
            sort_keys=True,
        ) + "\n"
    )
    fleet_hashes = {
        "input": [array_sha256(physical_input)] * 8,
        "output": [array_sha256(hardware)] * 8,
        "software": [array_sha256(software)] * 8,
        "source_file": [EXPECTED_SOURCE["npz_sha256"]] * 8,
        "source_raw": [EXPECTED_SOURCE["accepted_dense_partials_raw_sha256"]] * 8,
    }
    payload = {
        "accepted_model_axis_device_ids": list(EXPECTED_MODEL_AXIS_DEVICE_IDS),
        "accepted_model_axis_recipe": (
            "mesh_utils.create_device_mesh:shape=1,1,1,1,32,1:"
            "allow_split_physical_axes=true:model_axis=4"
        ),
        "artifact_manifest": {},
        "capture": {
            "compile_bucket_rows": 32,
            "determinism_repeat_invocations": 1,
            "input_bits_sha256": array_sha256(physical_input),
            "input_rows_replicated": True,
            "invocation_count": 2,
            "local_replica_output_sha256_by_trial": [[array_sha256(hardware[0])] * 4],
            "measured_trial_invocations": 1,
            "output_bits_sha256": array_sha256(hardware),
            "repeated_local_replica_output_sha256_by_trial": [[array_sha256(hardware[0])] * 4],
            "repeated_output_bits_sha256": array_sha256(hardware),
        },
        "collective_algorithm": algorithm,
        "collective_groups": [list(range(32))],
        "comparison": comparison,
        "config": {"seed": 1196575821, "trials": 1, "width": 6144},
        "diagnostic_only": True,
        "fleet_hashes": fleet_hashes,
        "fleet_hlo_hashes": [sha256(hlo.encode()).hexdigest()] * 8,
        "hlo": report.to_dict(),
        "member_device_ids": list(range(32)),
        "optimized_hlo_sha256": sha256(hlo.encode()).hexdigest(),
        "performance_claim": False,
        "source": dict(EXPECTED_SOURCE),
    }
    accepted_topology_record = json.loads(
        (REAL_DB533_DIR / "host_records" / "collective.rank0.json").read_text()
    )
    topology = accepted_topology_record["topology"]
    records = []
    for process_index in range(8):
        item = deepcopy(payload)
        item["artifact_manifest"] = manifest if process_index == 0 else {}
        record = {
            "association_dense_replay": item,
            "association_fingerprint": None,
            "captured_utc": "2026-08-14T12:00:00+00:00",
            "code_hash": PIN,
            "fleet_local_device_ids_in_runtime_order": (
                EXPECTED_FLEET_LOCAL_DEVICE_IDS
            ),
            "hostname": EXPECTED_HOSTNAMES[process_index],
            "jax_process_index": process_index,
            "jax_version": "0.6.2",
            "launch_process_id": process_index,
            "matrix": [],
            "mechanism_only": True,
            "mode": "strategy_nd_dense_replay",
            "run_tag": TAG,
            "schema_version": 3,
            "topology": topology,
            "topology_hash": EXPECTED_TOPOLOGY_HASH,
        }
        records.append(record)
        (run_dir / "host_records" / f"collective.rank{process_index}.json").write_text(
            json.dumps(record, indent=2, sort_keys=True) + "\n"
        )
    return run_dir, records


@pytest.mark.skipif(not REAL_SOURCES_AVAILABLE, reason="DB533/DB550 artifacts absent")
def test_terminal_recomputes_complete_dense_replay(tmp_path: Path) -> None:
    run_dir, _ = _fixture(tmp_path)
    summary = validate_strategy_nd_dense_replay(
        run_dir, expected_code_hash=PIN, expected_run_tag=TAG
    )
    assert summary["classification"] == "hardware_row0_exact_db533_software"
    assert summary["row0_exact"] is True
    assert summary["row0_mismatch_count"] == 0
    assert summary["hardware_hidden_2795_bfloat16_bits"] == 47808


@pytest.mark.skipif(not REAL_SOURCES_AVAILABLE, reason="DB533/DB550 artifacts absent")
def test_terminal_refuses_coherent_wrong_physical_input(tmp_path: Path) -> None:
    run_dir, records = _fixture(tmp_path)
    path = run_dir / "replay" / "physical_input_bits.npy"
    value = np.load(path, allow_pickle=False)
    value[:, [0, 1]] = value[:, [1, 0]]
    manifest = json.loads((run_dir / "replay" / "manifest.json").read_text())
    manifest["physical_input_bits"] = _save_array(path, value)
    (run_dir / "replay" / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    input_hash = array_sha256(value)
    for process_index, record in enumerate(records):
        item = record["association_dense_replay"]
        item["capture"]["input_bits_sha256"] = input_hash
        item["fleet_hashes"]["input"] = [input_hash] * 8
        item["artifact_manifest"] = manifest if process_index == 0 else {}
        (run_dir / "host_records" / f"collective.rank{process_index}.json").write_text(
            json.dumps(record, indent=2, sort_keys=True) + "\n"
        )
    with pytest.raises(ValueError, match="does not derive from DB550"):
        validate_strategy_nd_dense_replay(
            run_dir, expected_code_hash=PIN, expected_run_tag=TAG
        )


@pytest.mark.skipif(not REAL_SOURCES_AVAILABLE, reason="DB533/DB550 artifacts absent")
def test_terminal_refuses_unbound_fleet_topology_and_run(tmp_path: Path) -> None:
    run_dir, records = _fixture(tmp_path)
    records[0]["run_tag"] = "wrong"
    (run_dir / "host_records" / "collective.rank0.json").write_text(
        json.dumps(records[0], indent=2, sort_keys=True) + "\n"
    )
    with pytest.raises(ValueError, match="run/mechanism/software"):
        validate_strategy_nd_dense_replay(
            run_dir, expected_code_hash=PIN, expected_run_tag=TAG
        )

    run_dir, records = _fixture(tmp_path / "topology")
    records[0]["topology_hash"] = "a" * 64
    (run_dir / "host_records" / "collective.rank0.json").write_text(
        json.dumps(records[0], indent=2, sort_keys=True) + "\n"
    )
    with pytest.raises(ValueError, match="topology hash/local ordering"):
        validate_strategy_nd_dense_replay(
            run_dir, expected_code_hash=PIN, expected_run_tag=TAG
        )

    run_dir, records = _fixture(tmp_path / "hostnames")
    for process_index, record in enumerate(records):
        record["hostname"] = f"rogue-cluster-w-{process_index}"
        (run_dir / "host_records" / f"collective.rank{process_index}.json").write_text(
            json.dumps(record, indent=2, sort_keys=True) + "\n"
        )
    with pytest.raises(ValueError, match="filename/launch/hostname"):
        validate_strategy_nd_dense_replay(
            run_dir, expected_code_hash=PIN, expected_run_tag=TAG
        )

    run_dir, records = _fixture(tmp_path / "nested_schema")
    for process_index, record in enumerate(records):
        record["association_dense_replay"]["classification"] = (
            "forged-contradiction"
        )
        (run_dir / "host_records" / f"collective.rank{process_index}.json").write_text(
            json.dumps(record, indent=2, sort_keys=True) + "\n"
        )
    with pytest.raises(ValueError, match="nested payload schema/type"):
        validate_strategy_nd_dense_replay(
            run_dir, expected_code_hash=PIN, expected_run_tag=TAG
        )


def test_protected_wrapper_is_default_exact_and_success_last() -> None:
    wrapper = Path(
        "scripts/greenfield/run_strategy_nd_dense_replay.sh"
    ).read_text()
    assert "--mode strategy_nd_dense_replay" in wrapper
    assert "--association-trials 1" in wrapper
    assert "strict_census pre" in wrapper and "strict_census post" in wrapper
    assert "google_crc32c" in wrapper
    assert "collective.rank${idx}.sha256" not in wrapper
    assert '--output "$run/collective.rank${idx}.json"' in wrapper
    assert 'rm -f "$RUN_DIR/collective.rank0.json"' in wrapper
    assert "remote nonterminal object set drifted" in wrapper
    assert 'GLM_GREENFIELD_RUN_TAG="$tag"' in wrapper
    assert wrapper.index('remote_objects.json" >/dev/null') < wrapper.index(
        '"$REMOTE_PREFIX/SUCCESS" >/dev/null'
    )
    assert "results_db_role" in wrapper

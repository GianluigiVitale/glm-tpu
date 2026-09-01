from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "scripts/greenfield/adjudicate_gate_d_compensated_pp16_hlo.py"
RUN = Path(
    "/home/gianl/gate-d-runs/gate_d_compensated_pp16_hlo_20260901T070612366187759Z"
)
HLO = RUN / "hlo/compensated_pp16_stage0.optimized_hlo.txt"
STABLE = RUN / "hlo/compensated_pp16_stage0.stablehlo.mlir"

spec = importlib.util.spec_from_file_location("gate_d_hlo_adjudicator", SOURCE)
assert spec is not None and spec.loader is not None
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)


@pytest.fixture(scope="module")
def hlo() -> str:
    return HLO.read_text()


@pytest.fixture(scope="module")
def stable() -> str:
    return STABLE.read_text()


def test_exact_real_hlo_passes_narrow_structural_adjudication(
    hlo: str, stable: str
) -> None:
    optimized = module.validate_optimized_hlo(hlo)
    source = module.validate_stablehlo(stable)
    assert optimized["collective_count"] == 3
    assert optimized["rooted_compensated_witness"] is True
    assert optimized["primary_noninterference"] is True
    assert optimized["logical_row_axis"] == 1
    assert optimized["num_partitions"] == 2
    assert source == {
        "all_gather_count": 3,
        "compensated_slice_rooted": True,
        "logical_rows": 1,
        "manual_axes": ["feature"],
        "mesh_axis": "feature",
        "mesh_size": 2,
        "owner_output_sharding_count": 10,
    }


@pytest.mark.parametrize(
    ("old", "new", "reason"),
    (
        (
            "replica_groups={{0,1}}",
            "replica_groups={{0,1,2}}",
            "replica group",
        ),
        (
            "use_global_device_ids=true",
            "use_global_device_ids=false",
            "global ids",
        ),
        (
            'custom_call_target="ConcatBitcast"',
            'custom_call_target="xla_python_cpu_callback"',
            "forbidden host/transport token",
        ),
        (
            'custom_call_target="AllocateBuffer"',
            'custom_call_target="UnknownDeviceEffect"',
            "custom-call allowlist",
        ),
        ("num_partitions=2", "num_partitions=4", "two-partition"),
    ),
)
def test_optimized_surface_attacks_fail_closed(
    hlo: str, old: str, new: str, reason: str
) -> None:
    attacked = hlo.replace(old, new, 1)
    assert attacked != hlo
    with pytest.raises(module.Refusal, match=reason):
        module.validate_optimized_hlo(attacked)


def test_extra_collective_fails_closed(hlo: str) -> None:
    original = (
        "  %bitcast.148 = f32[1,1,6144]{2,1,0:T(1,128)} bitcast(%convert_add_fusion)"
    )
    injected = (
        "  %attack = f32[1,1,6144]{2,1,0:T(1,128)} "
        "all-reduce(%convert_add_fusion), replica_groups={{0,1}}\n" + original
    )
    attacked = hlo.replace(original, injected, 1)
    assert attacked != hlo
    with pytest.raises(module.Refusal, match="collective surface"):
        module.validate_optimized_hlo(attacked)


def test_auxiliary_cannot_escape_into_primary_root(hlo: str) -> None:
    original = "tuple(%bitcast.148, %bitcast.130, %bitcast.138"
    attacked = hlo.replace(
        original,
        "tuple(%bitcast.148, %bitcast.148, %bitcast.138",
        1,
    )
    assert attacked != hlo
    with pytest.raises(module.Refusal, match="root shapes|contaminates"):
        module.validate_optimized_hlo(attacked)


def test_auxiliary_must_remain_root_slot_zero(hlo: str) -> None:
    original = "tuple(%bitcast.148, %bitcast.130, %bitcast.138"
    attacked = hlo.replace(
        original,
        "tuple(%bitcast.130, %bitcast.148, %bitcast.138",
        1,
    )
    assert attacked != hlo
    with pytest.raises(module.Refusal, match="root shapes|root slot zero"):
        module.validate_optimized_hlo(attacked)


def test_compensation_subtract_body_is_semantic_not_metadata(hlo: str) -> None:
    original = (
        "ROOT %sub.45 = f32[1,6144]{1,0:T(1,128)S(3)} "
        "subtract(%param_0.102, %convert_element_type.96)"
    )
    attacked = hlo.replace(original, original.replace("subtract(", "add("), 1)
    assert attacked != hlo
    with pytest.raises(module.Refusal, match="compensation subtract"):
        module.validate_optimized_hlo(attacked)


def test_restore_body_cannot_reuse_correction_as_original(hlo: str) -> None:
    original = (
        "%convert_element_type.97 = bf16[1,6144]{1,0:T(2,128)(2,1)} "
        "convert(%param_1.134)"
    )
    attacked = hlo.replace(
        original,
        original.replace("%param_1.134", "%param_0.101"),
        1,
    )
    assert attacked != hlo
    with pytest.raises(module.Refusal, match="restored original source"):
        module.validate_optimized_hlo(attacked)


def test_restore_entry_operands_cannot_swap_semantic_roles(hlo: str) -> None:
    original = "fusion(%convert_subtract_fusion, %get-tuple-element.215), kind=kLoop"
    attacked = hlo.replace(
        original,
        "fusion(%get-tuple-element.215, %convert_subtract_fusion), kind=kLoop",
        1,
    )
    assert attacked != hlo
    with pytest.raises(module.Refusal, match="restore ENTRY operand binding"):
        module.validate_optimized_hlo(attacked)


def test_subtract_entry_cannot_gain_extra_operand(hlo: str) -> None:
    original = "fusion(%get-tuple-element.215), kind=kLoop"
    attacked = hlo.replace(
        original,
        "fusion(%get-tuple-element.215, %get-tuple-element.215), kind=kLoop",
        1,
    )
    assert attacked != hlo
    with pytest.raises(module.Refusal, match="compensation ENTRY operand binding"):
        module.validate_optimized_hlo(attacked)


def test_source_value_cannot_gain_unknown_consumer(hlo: str) -> None:
    original = (
        "  %get-tuple-element.214 = f32[]{:T(128)} "
        "get-tuple-element(%multiply_reduce_fusion.1), index=0"
    )
    injected = (
        "  %attack-copy = f32[1,6144]{1,0:T(1,128)} "
        "copy(%get-tuple-element.215)\n" + original
    )
    attacked = hlo.replace(original, injected, 1)
    assert attacked != hlo
    with pytest.raises(module.Refusal, match="consumer set"):
        module.validate_optimized_hlo(attacked)


@pytest.mark.parametrize(
    ("old", "new", "reason"),
    (
        (
            "mhlo.num_partitions = 2",
            "mhlo.num_partitions = 4",
            "mesh",
        ),
        (
            "%10 = stablehlo.subtract %7, %9",
            "%10 = stablehlo.add %7, %9",
            "causal slice",
        ),
        (
            "replica_groups = dense<[[0, 1]]>",
            "replica_groups = dense<[[0, 1, 2]]>",
            "collective locality",
        ),
    ),
)
def test_stablehlo_attacks_fail_closed(
    stable: str, old: str, new: str, reason: str
) -> None:
    attacked = stable.replace(old, new, 1)
    assert attacked != stable
    with pytest.raises(module.Refusal, match=reason):
        module.validate_stablehlo(attacked)


def test_stablehlo_owner_output_sharding_attack_fails_closed(stable: str) -> None:
    manual = next(
        line for line in stable.splitlines() if "sdy.manual_computation" in line
    )
    attacked_manual = manual.replace(
        'out_shardings=[<@mesh, [{"feature"}',
        "out_shardings=[<@mesh, [{}",
        1,
    )
    attacked = stable.replace(manual, attacked_manual, 1)
    assert attacked != stable
    with pytest.raises(module.Refusal, match="owner output shardings"):
        module.validate_stablehlo(attacked)


def test_crc32c_matches_every_local_ledger_receipt() -> None:
    ledger = json.loads((RUN / "remote_objects.json").read_text())
    for item in ledger["objects"]:
        assert module._crc32c((RUN / item["path"]).read_bytes()) == item["crc32c"]
    assert module._crc32c((RUN / "HLO_ACQUIRED").read_bytes()) == "HK0N6Q=="


def test_remote_replay_requires_exact_versions_and_no_soft_delete(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    files = module._inventory(RUN)
    ledger = json.loads(files["remote_objects.json"])
    records = [*ledger["objects"]]
    records.extend(
        (
            {
                "path": "remote_objects.json",
                "generation": str(module.LEDGER_GENERATION),
                "sha256": module.LEDGER_SHA256,
                "size": len(files["remote_objects.json"]),
                "crc32c": module._crc32c(files["remote_objects.json"]),
            },
            {
                "path": "HLO_ACQUIRED",
                "generation": str(module.TERMINAL_GENERATION),
                "sha256": module.TERMINAL_SHA256,
                "size": len(files["HLO_ACQUIRED"]),
                "crc32c": module._crc32c(files["HLO_ACQUIRED"]),
            },
        )
    )
    listing = [
        {
            "url": f"{module.REMOTE_PREFIX}/{item['path']}#{item['generation']}",
            "type": "cloud_object",
            "metadata": {
                "bucket": "driftbench-dsv4-uc",
                "name": f"results/greenfield/glm52/gate_d_pp16_hlo/{module.RUN_TAG}/{item['path']}",
                "generation": str(item["generation"]),
                "size": str(item["size"]),
                "crc32c": item["crc32c"],
            },
        }
        for item in records
    ]

    def fake_gcloud(
        _gcloud: Path,
        arguments: list[str],
        *,
        allow_exact_vacant: bool = False,
    ) -> bytes:
        del allow_exact_vacant
        if arguments[0] == "cat":
            url = arguments[1]
            relative = url.split(module.REMOTE_PREFIX + "/", 1)[1].rsplit("#", 1)[0]
            return files[relative]
        if "--soft-deleted" in arguments:
            return b"[]"
        return json.dumps(listing).encode()

    monkeypatch.setattr(module, "_gcloud", fake_gcloud)
    assert module.replay_remote(files, Path("unused"))["generation_replay_count"] == 17
    listing.append(listing[-1])
    with pytest.raises(module.Refusal, match="catalogue"):
        module.replay_remote(files, Path("unused"))
    listing.pop()
    listing[0]["metadata"], listing[1]["metadata"] = (
        listing[1]["metadata"],
        listing[0]["metadata"],
    )
    with pytest.raises(module.Refusal, match="URL/metadata"):
        module.replay_remote(files, Path("unused"))


def test_boolean_generation_is_not_an_integer() -> None:
    with pytest.raises(module.Refusal):
        module._positive_int(True, "generation")


@pytest.mark.parametrize("value", (True, 1, 1.0, "01", "0", "-1", "1.0"))
def test_generation_requires_canonical_positive_decimal_string(value: object) -> None:
    with pytest.raises(module.Refusal):
        module._decimal_generation(value, "generation")


def test_report_publication_is_exclusive_replayed_and_read_only(tmp_path: Path) -> None:
    output_root = tmp_path / "authority"
    output_root.mkdir(mode=0o700)
    payload = b'{"classification":"GATE_D_OPEN"}\n'
    output = module._write_exclusive(output_root, payload)
    assert output == output_root / "report.json"
    assert output.read_bytes() == payload
    assert output.stat().st_mode & 0o777 == 0o400
    with pytest.raises(module.Refusal, match="not vacant"):
        module._write_exclusive(output_root, payload)


def test_report_publication_rejects_unsafe_or_symlink_root(tmp_path: Path) -> None:
    unsafe = tmp_path / "unsafe"
    unsafe.mkdir(mode=0o700)
    unsafe.chmod(0o755)
    with pytest.raises(module.Refusal, match="ownership/mode"):
        module._write_exclusive(unsafe, b"payload")

    target = tmp_path / "target"
    target.mkdir(mode=0o700)
    link = tmp_path / "link"
    link.symlink_to(target, target_is_directory=True)
    with pytest.raises(OSError):
        module._write_exclusive(link, b"payload")

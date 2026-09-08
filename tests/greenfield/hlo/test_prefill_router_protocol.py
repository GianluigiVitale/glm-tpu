"""Diagnostic evidence refusal tests; no TPU or performance claims."""

from copy import deepcopy
from hashlib import sha256
import json

import numpy as np
import pytest

from scripts.greenfield import prefill_router_protocol as p
from scripts.greenfield.prefill_layer_evidence import encode_arrays
from scripts.greenfield.prefill_router_worker import stack


def fixture_arrays():
    from glm_tpu.greenfield.kernels.reference.rotary import build_rotary_table_host

    original = json.loads(p.ORIGINAL.read_text())
    host = p.host_case(
        "boundary", build_rotary_table_host(1024, rotary_dim=64, theta=8e6)
    )
    arrays = encode_arrays("input", host)
    weights = dict(
        router_weight=np.zeros((32, 1536), p.BF16),
        correction_bias=np.zeros(32, np.float32),
    )
    arrays.update(encode_arrays("weights_0", weights))
    ledger = {
        0: {
            "selected": {
                "model.layers.3.mlp.gate.weight": sha256(
                    weights["router_weight"].tobytes()
                ).hexdigest(),
                "model.layers.3.mlp.gate.e_score_correction_bias": sha256(
                    weights["correction_bias"].tobytes()
                ).hexdigest(),
            }
        }
    }
    for kind in p.KINDS:
        values = {}
        for name in p.fields_for(kind):
            dtype = (
                p.BF16
                if name in p.BF16_FIELDS
                else (
                    np.int32
                    if name == "routes"
                    else np.bool_ if name == "prefix_health" else np.float32
                )
            )
            values[name] = np.zeros(p.SHAPES[name], dtype)
        source = kind.split("_")[0]
        routes = np.asarray(original["original_routes"][source], np.int32)
        values["routes"] = routes
        np.put_along_axis(
            values["biased_scores"],
            routes,
            np.broadcast_to(np.arange(8, 0, -1), (17, 8)),
            axis=1,
        )
        if "prefix_health" in values:
            values["prefix_health"][:] = True
        arrays.update(encode_arrays(f"{kind}_0", values))
    return arrays, ledger


def test_original_arrays_replayed_without_numerical_promotion(tmp_path):
    arrays, ledger = fixture_arrays()
    path = tmp_path / "boundary.npz"
    np.savez_compressed(path, **arrays)
    result = p.replay_file(path, {0: 0}, ledger)
    assert result["evidence_complete"] and not result["numerical_admission"]
    assert not result["performance_claim"]
    assert result["owners"]["0"]["original_routes_reproduced"]


@pytest.mark.parametrize(
    "field", ["router_input", "logits", "biased_scores", "route_weights"]
)
def test_nonfinite_capture_refused(field):
    arrays, ledger = fixture_arrays()
    key = f"actual_0__{field}"
    if field in p.BF16_FIELDS:
        arrays[key].view(p.BF16).flat[0] = np.nan
    else:
        arrays[key].flat[0] = np.nan
    with pytest.raises(ValueError, match="nonfinite"):
        p.verify_prefix(arrays, {0: 0}, ledger)


def test_weight_hash_and_full_route_reproduction_refused():
    arrays, ledger = fixture_arrays()
    bad = deepcopy(ledger)
    bad[0]["selected"]["model.layers.3.mlp.gate.weight"] = "0" * 64
    with pytest.raises(ValueError, match="checkpoint"):
        p.verify_prefix(arrays, {0: 0}, bad)
    # Perturb a row OTHER than the original row4; keep its own-score ordering valid.
    routes = arrays["actual_0__routes"]
    routes[8, :2] = routes[8, 1::-1]
    scores = arrays["actual_0__biased_scores"]
    scores[8] = 0
    scores[8, routes[8]] = np.arange(8, 0, -1)
    with pytest.raises(ValueError, match="full17-row"):
        p.verify_prefix(arrays, {0: 0}, ledger)


@pytest.mark.parametrize("mutation", ["input", "bias", "extra", "tie"])
def test_replay_binding_inventory_and_ties(tmp_path, mutation):
    arrays, ledger = fixture_arrays()
    if mutation == "input":
        arrays["actual_scalar_0__router_input"].view(p.BF16)[0, 0] = 1
    elif mutation == "bias":
        arrays["actual_batch_0__bias"][0] = 1
    elif mutation == "extra":
        arrays["unregistered"] = np.zeros(1)
    else:
        arrays["actual_scalar_0__biased_scores"][:] = 0
    path = tmp_path / "boundary.npz"
    np.savez_compressed(path, **arrays)
    with pytest.raises(ValueError):
        p.replay_file(path, {0: 0}, ledger)


def test_fleet_replication_preserves_distinct_partial_owners(tmp_path):
    arrays, _ = fixture_arrays()
    records = []
    for rank in range(8):
        record = dict(launch_rank=rank, local_device_slots=[])
        local = {}
        for slot in range(rank * 4, rank * 4 + 4):
            record["local_device_slots"].append(dict(device_id=slot, device_slot=slot))
            for kind in p.KINDS:
                for name in p.fields_for(kind):
                    value = arrays[f"{kind}_0__{name}"].copy()
                    if name == "partial_logits":
                        value[:] = (
                            slot  # local partials must NOT be compared as replicas
                        )
                    local[f"{kind}_{slot}__{name}"] = value
        directory = tmp_path / f"rank{rank}"
        directory.mkdir()
        np.savez_compressed(directory / "boundary.npz", **local)
        records.append(record)
    p.verify_fleet_replicas(tmp_path, records)
    path = tmp_path / "rank7/boundary.npz"
    with np.load(path) as data:
        bad = {k: data[k] for k in data.files}
    bad["actual_31__logits"][0, 0] = 1
    np.savez_compressed(path, **bad)
    with pytest.raises(ValueError, match="replica disagreement"):
        p.verify_fleet_replicas(tmp_path, records)


def test_scalar_stack_keeps_final_cache_and_constant_bias():
    rows = [
        dict(
            kv=np.full((2, 64, 640), i),
            bias=np.zeros(256),
            routes=np.zeros((1, 8), np.int32),
        )
        for i in range(17)
    ]
    result = stack(rows)
    assert np.all(result["kv"] == 16) and result["routes"].shape == (17, 8)
    rows[8]["bias"][0] = 1
    with pytest.raises(ValueError, match="bias changed"):
        stack(rows)


def test_diagnostic_tag_and_evidence_are_distinct():
    from scripts.greenfield.ws32_prefill_layer_campaign import evidence_files

    assert p.is_router_tag(f"greenfield_fp8_{p.KERNEL}_l3_test")
    assert not p.is_router_tag(f"greenfield_fp8_{p.KERNEL}_l0_test")
    files = evidence_files(3, diagnostic=True)
    assert "router_scalar.optimized_hlo.txt" in files and "boundary.npz" in files
    assert "empty.npz" not in files and "tail.npz" not in files
    with pytest.raises(ValueError):
        evidence_files(0, diagnostic=True)


def test_worker_compiles_plain_shard_map_through_real_lowering(tmp_path):
    import jax
    import jax.numpy as jnp
    from jax.sharding import Mesh, PartitionSpec as P
    from scripts.greenfield.prefill_router_worker import compile_observer

    mesh = Mesh(np.asarray(jax.devices()[:1]), ("x",))
    fn = jax.shard_map(lambda a: a + 1, mesh=mesh, in_specs=P(), out_specs=P())
    assert not hasattr(fn, "lower")
    record = {}
    compiled = compile_observer(fn, (jnp.zeros((1,)),), "candidate", tmp_path, record)
    np.testing.assert_array_equal(compiled(jnp.zeros((1,))), [1])
    for form, field in (
        ("stablehlo.mlir", "stablehlo_sha256"),
        ("optimized_hlo.txt", "optimized_hlo_sha256"),
    ):
        assert (
            sha256((tmp_path / f"candidate.{form}").read_bytes()).hexdigest()
            == record["programs"]["candidate"][field]
        )


def router_hlo(kind):
    rows = 17 if kind in ("candidate", "router_batch") else 1
    prefix = kind in ("candidate", "reference")
    payload = [(4, rows * 32), (8, rows * 256), (8, 256)]
    if prefix:
        payload += [
            (4, rows),
            (4, rows),
            (4, rows * 2048),
            (4, rows * 576),
            (8, rows * 2048 * 640),
            (8, rows * 1536),
        ]
    lines = ["HloModule router", "ENTRY %main {"]
    for i, (group, size) in enumerate(payload):
        dtype = "bf16" if size == rows * 2048 * 640 else "f32"
        groups = (
            "{{"
            + "},{".join(
                ",".join(map(str, g)) for g in (p.FEATURE if group == 4 else p.EXPERT)
            )
            + "}}"
        )
        lines += [
            f"%p{i} = {dtype}[{size}] parameter({i})",
            f"%r{i} = {dtype}[{size}] all-reduce(%p{i}), replica_groups={groups}",
        ]
    if prefix:
        for i, name in enumerate(
            ["greenfield_fp8_block_matmul"] * 4
            + ["greenfield_fp8_structured_kv_b"] * 2
            + ["greenfield_pregathered_sparse_mla"]
        ):
            lines.append(
                f'%call{i} = f32[1] custom-call(%p0), custom_call_target="tpu_custom_call", metadata={{op_name="{name}"}}'
            )
    return "\n".join(lines + ["}"])


@pytest.mark.parametrize("kind", p.PROGRAMS)
def test_diagnostic_hlo_minimal_inventory_and_refusals(kind):
    hlo = router_hlo(kind)
    proof = p.check_hlo(hlo, kind)
    assert proof["passed"], proof["checks"]
    assert not p.check_hlo(hlo.replace("all-reduce", "all-to-all", 1), kind)["passed"]
    assert not p.check_hlo(
        (
            hlo.replace("f32[544]", "f32[545]")
            if "f32[544]" in hlo
            else hlo.replace("f32[32]", "f32[33]")
        ),
        kind,
    )["passed"]
    assert not p.check_hlo(
        hlo.replace("ENTRY %main {", "ENTRY %main {\n%bad = f32[1] infeed()"), kind
    )["passed"]
    bad = hlo.replace(
        "ENTRY %main {",
        'ENTRY %main {\n%bad = f32[1] custom-call(), custom_call_target="HostCallback"',
    )
    assert not p.check_hlo(bad, kind)["passed"]


def test_diagnostic_workers_cannot_promote_as_layer_admission():
    from scripts.greenfield import ws32_prefill_layer_campaign as c
    from tests.greenfield.hlo.test_prefill_layer_campaign import valid_workers

    records, pin, pins, ledger = valid_workers(3)
    for r in records:
        r.update(protocol=p.PROTOCOL, admission_only=False, diagnostic_only=True)
        program = next(iter(r["programs"].values()))
        r["programs"] = {n: deepcopy(program) for n in p.PROGRAMS}
        r["cases"] = {
            "boundary": dict(
                evidence_complete=True,
                input_sha256={"test": "x"},
                replay=dict(
                    evidence_complete=True,
                    numerical_admission=False,
                    owners={str(s["device_id"]): {} for s in r["local_device_slots"]},
                ),
            )
        }
    c.validate_workers(records, pin, layer=3, pins=pins, ledger=ledger, diagnostic=True)
    with pytest.raises((ValueError, KeyError)):
        c.validate_workers(records, pin, layer=3, pins=pins, ledger=ledger)
    records[0]["cases"]["boundary"]["replay"]["numerical_admission"] = True
    with pytest.raises(ValueError, match="misclassified"):
        c.validate_workers(
            records, pin, layer=3, pins=pins, ledger=ledger, diagnostic=True
        )

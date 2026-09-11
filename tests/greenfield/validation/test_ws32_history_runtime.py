"""Retained real owner identity; fixture loaders and CPU-only abstract jobs."""

from collections import namedtuple
from copy import deepcopy
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace as NS

import numpy as np
import pytest

from scripts.greenfield import ws32_history_runtime as runtime_module
from scripts.greenfield import ws32_history_protocol as protocol
from scripts.greenfield.ws32_prefill_budget_campaign import topology_bindings

REPO = Path(__file__).resolve().parents[3]
TAG = "greenfield_fp8_ws32_history_frontier_l06_20260911T200000000000000Z"


@pytest.fixture(scope="module")
def actual():
    return {b: json.loads((Path("/home/gianl/glm-run") / protocol.ORIGINAL_TAGS[b]
                          / "runner.rank0.json").read_text()) for b in protocol.BRANCHES}


@pytest.fixture
def fixture(tmp_path, monkeypatch, actual):
    import jax
    runners = deepcopy(actual)
    prior = runners["candidate"]
    physical, captures = topology_bindings()
    owners = {device: row["jax_process_index"] for row in captures for device in row["local_device_ids"]}
    devices = [NS(id=d, platform="tpu", process_index=owners[d]) for d in physical.flattened_device_ids]
    local = [d for d in devices if d.process_index == prior["jax_process_index"]]
    backend = NS(default_backend=lambda: "tpu", process_count=lambda: 8, device_count=lambda: 32,
        process_index=lambda: prior["jax_process_index"], local_devices=lambda: local, tree=jax.tree)
    mesh = NS(devices=np.asarray(devices, object).reshape(8, 4), axis_names=("expert", "feature"))
    runtime = (backend, mesh, physical, NS(topology_hash=prior["topology_sha256"]), prior["topology_fleet_sha256"])
    identity = dict(original_tags=protocol.ORIGINAL_TAGS, original_pins={"fixture": "a" * 64})
    originals = {b: {"positions": np.zeros((4, 1, 2048), np.int32)} for b in protocol.BRANCHES}
    record = dict(protocol=protocol.PROTOCOL, compile_only=False, diagnostic_only=True,
                  tag=TAG, code_hash="a" * 40, launch_rank=0, programs={})
    checkpoint_pins = dict(expected_success_sha256=prior["checkpoint_success_sha256"])
    retained = dict(protocol=protocol.PROTOCOL, tag=TAG, code_hash=record["code_hash"], launch_rank=0,
        hostname=prior["hostname"], selected_layer_ids=list(protocol.LAYERS), include_embedding=True,
        context_capacity=8192, host_main_rope_table=True, selected_leaf_count=201,
        payload_bytes_per_chip=protocol.PAYLOAD_BYTES, overlay_tensor_count=12,
        overlay_bytes_per_chip=protocol.OVERLAY_BYTES, prompt_ids_sha256=protocol.PROMPT_SHA,
        prompt_hash_source="AUTHENTICATED_ORIGINAL_TOKEN_ORACLE",
        original_oracle_pins={key: prior[key] for key in runtime_module.preflight_module.ORACLE_KEYS},
        main_rope_table=prior["main_rope_table"], strategy_nd_dense_overlay=prior["strategy_nd_dense_overlay"],
        local_device_slots=prior["local_device_slots"], checkpoint_pins=checkpoint_pins,
        overlay_manifest_sha256=prior["strategy_nd_dense_overlay"]["manifest_sha256"],
        headers=[dict(device_slot=v["device_slot"], filename=f"slot{v['device_slot']}.fixture",
                      file_bytes=123, header_sha256="b" * 64) for v in prior["local_device_slots"]],
        numerical_execution_available=False, numerical_admission=False,
        numerical_promotion=False, performance_claim=False, **identity)
    monkeypatch.setattr(runtime_module.socket, "gethostname", lambda: prior["hostname"])
    monkeypatch.setattr(runtime_module.preflight_module, "load_originals", lambda *a, **k: (runners, originals, identity))
    def save():
        (tmp_path / "retained_preflight.json").write_text(json.dumps(retained))
    save()
    return NS(root=tmp_path, runners=runners, prior=prior, originals=originals, identity=identity,
              record=record, retained=retained, runtime=runtime, local=local, save=save)


@pytest.mark.parametrize("mutation", [None, "protocol", "tag", "rank_bool", "pin", "compile_only",
    "preflight_pin", "original_pin", "preflight_admission", "backend", "process", "owners",
    "duplicate_owner", "device_platform", "device_process", "topology", "mesh", "axes", "header", "symlink"])
def test_real_nonidentity_owner_binding_and_refusal(fixture, mutation):
    f = fixture
    backend, mesh, physical, topology, fleet = f.runtime
    if mutation == "protocol": f.record["protocol"] = "compiler-only"
    elif mutation == "tag": f.record["tag"] = "other"
    elif mutation == "rank_bool": f.record["launch_rank"] = False
    elif mutation == "pin": f.record["code_hash"] = "unpublished"
    elif mutation == "compile_only": f.record["compile_only"] = True
    elif mutation == "preflight_pin": f.retained["code_hash"] = "b" * 40
    elif mutation == "original_pin": f.retained["original_pins"] = {"fixture": "b" * 64}
    elif mutation == "preflight_admission": f.retained["numerical_admission"] = True
    elif mutation == "backend": backend.default_backend = lambda: "cpu"
    elif mutation == "process": backend.process_index = lambda: 0
    elif mutation == "owners": backend.local_devices = lambda: tuple(mesh.devices[0])
    elif mutation == "duplicate_owner": backend.local_devices = lambda: (f.local[0],) * 4
    elif mutation == "device_platform": f.local[0].platform = "cpu"
    elif mutation == "device_process": f.local[0].process_index = 7
    elif mutation == "topology": topology.topology_hash = "b" * 64
    elif mutation == "mesh": mesh.devices = mesh.devices[::-1]
    elif mutation == "axes": mesh.axis_names = ("feature", "expert")
    elif mutation == "header": f.retained["headers"][0]["device_slot"] = 0
    f.save()
    if mutation == "symlink":
        original = f.root / "retained_preflight.json"
        original.rename(f.root / "other.json")
        original.symlink_to(f.root / "other.json")
    def run():
        return runtime_module.bind(root=f.root, record=f.record, repo=REPO, runtime=f.runtime)
    if mutation:
        with pytest.raises(ValueError): run()
    else:
        runners, originals, slots, retained = run()
        assert runners is f.runners and originals is f.originals
        assert slots == {12: 9, 14: 13, 13: 25, 15: 29}
        assert f.record["jax_process_index"] == 3
        assert f.record["retained_preflight_sha256"] == sha256((f.root / "retained_preflight.json").read_bytes()).hexdigest()
        assert not retained["numerical_admission"]


@pytest.mark.parametrize("failure", [None, "headers", "overlay_pin", "host_inputs", "raw_inventory",
    "raw_owner", "raw_digest", "raw_duplicate", "overlay_owner", "peer_raw", "peer_overlay"])
def test_voted_selected_overlay_loading_order_and_shared_views(fixture, monkeypatch, failure):
    f, events = fixture, []
    names = tuple(f"leaf{i}" for i in range(201))
    layers = tuple(tuple(names[1 + 28*i:1 + 28*(i+1)]) for i in range(6)) + (names[169:],)
    tree = (names[0], layers)
    rows = {h["device_slot"]: {**h, "sha256": v["file_sha256"], "tensor_sha256": ["d" * 64] * 201}
            for h, v in zip(f.retained["headers"], f.prior["local_device_slots"], strict=True)}
    manifest = dict(manifest_sha256=f.prior["checkpoint_manifest_sha256"], geometry={},
                    source={"inventory_sha256": f.prior["source_inventory_sha256"]})
    subset = NS(metadata=NS(records_by_slot=rows, manifest=manifest,
                           plans=[NS(tensors=[NS(name=n) for n in names])]), tensor_indices=tuple(range(201)))
    overlay = NS(manifest={"manifest_sha256": f.retained["overlay_manifest_sha256"]})
    config = NS(dsa_contract=object())
    tokens, rope = np.arange(8155, dtype=np.int32), np.zeros((8192, 64), np.uint16)
    prepared = NS(manifest_sha256=manifest["manifest_sha256"], source_inventory_sha256=manifest["source"]["inventory_sha256"],
                  inputs={"candidate_b128": (None,) * 5 + tree + (None, rope), "observer": (None,) * 8, "exact_decode": ()})
    jobs = tuple((name, object(), ()) for name in protocol.PROGRAMS)
    monkeypatch.setattr(runtime_module.preflight_module, "selected_metadata", lambda *a: (f.retained["checkpoint_pins"], subset))
    monkeypatch.setattr(runtime_module.preflight_module, "bind_metadata", lambda *a: overlay)
    monkeypatch.setattr(runtime_module.ModelGeometry, "from_dict", lambda _: object())
    monkeypatch.setattr(runtime_module, "Ws32DecoderConfig", lambda *a, **k: config)
    monkeypatch.setattr(runtime_module, "ws32_decoder_weight_names", lambda _: NS(embedding_local=tree[0], layers=tree[1]))
    def host(*args):
        events.append("host_inputs")
        if failure == "host_inputs": raise ValueError("fixture host refusal")
        return tokens, rope
    monkeypatch.setattr(runtime_module.preflight_module, "host_inputs", host)
    def compiler_jobs(*a, **k):
        events.append("prepare_once")
        assert k["prepared"] is prepared
        return prepared, jobs
    monkeypatch.setattr(runtime_module, "compiler_jobs", compiler_jobs)
    monkeypatch.setattr(runtime_module, "_tree_contract", lambda *a: None)  # Separate real-array shape/sharding tests below.
    selected_rows = [dict(device_id=v["device_id"], device_slot=v["device_slot"],
        expected_full_file_sha256_not_verified=v["file_sha256"], selected_payload_bytes=protocol.PAYLOAD_BYTES,
        observed_selected_tensor_sha256={n: "d" * 64 for n in names}) for v in f.prior["local_device_slots"]]
    if failure == "headers": rows[9]["header_sha256"] = "e" * 64
    elif failure == "overlay_pin": overlay.manifest["manifest_sha256"] = "e" * 64
    elif failure == "raw_owner": selected_rows[0]["device_id"] = 0
    elif failure == "raw_digest": selected_rows[0]["observed_selected_tensor_sha256"][names[-1]] = "e" * 64
    elif failure == "raw_duplicate": selected_rows[0] = deepcopy(selected_rows[1])
    raw_arrays = {n: object() for n in names}
    if failure == "raw_inventory": raw_arrays.pop(names[-1])
    def raw_load(*a, **k):
        events.append("raw")
        return NS(layer_ids=protocol.LAYERS, include_embedding=True, payload_bytes_per_chip=protocol.PAYLOAD_BYTES,
            arrays=raw_arrays, local_device_slots=selected_rows[::-1], device_memory_before=(),
            device_memory_after=(), integrity_scope="selected_layers_and_embedding_only_not_complete_checkpoint")
    monkeypatch.setattr(runtime_module, "load_ws32_layer_subset", raw_load)
    overlay_records = deepcopy(f.prior["strategy_nd_dense_overlay"]["local_records"])
    if failure == "overlay_owner": overlay_records.pop()
    def overlay_load(*a, **k):
        events.append("overlay")
        return NS(arrays={f"overlay{i}": object() for i in range(12)}, local_records=overlay_records[::-1])
    monkeypatch.setattr(runtime_module, "load_ws32_strategy_nd_dense_overlay", overlay_load)
    observer_views = {}
    def observe(config_, embedding, raw_layers, overlay_arrays):
        observer_views.update(embedding=embedding, layers=raw_layers)
        return config_, raw_layers, tuple(raw_layers[i] for i in protocol.PRODUCERS)
    monkeypatch.setattr(runtime_module.observer, "bind_selected_views", observe)
    import jax.sharding
    monkeypatch.setattr(jax.sharding, "NamedSharding", lambda *a: None)
    f.runtime[0].make_array_from_callback = lambda shape, s, callback: callback((slice(None), slice(None)))
    f.runtime[0].block_until_ready = lambda v: events.append("rope")
    def vote(ok):
        return ok and not (failure == "peer_raw" and "raw" in events or failure == "peer_overlay" and "overlay" in events)
    def run():
        return runtime_module.prepare_bound(root=f.root, record=f.record, repo=REPO,
            runtime=f.runtime, consensus=vote, prepared=prepared)
    if failure:
        with pytest.raises((ValueError, RuntimeError)): run()
        assert "rope" not in events
        if failure in ("headers", "overlay_pin", "host_inputs"): assert "raw" not in events
        if failure not in ("overlay_owner", "peer_overlay"): assert "overlay" not in events
    else:
        bound = run()
        assert events == ["host_inputs", "prepare_once", "raw", "overlay", "rope"]
        assert bound.jobs is jobs and bound.prepared is prepared and bound.prompt_tokens is tokens
        assert bound.embedding is observer_views["embedding"] is raw_arrays[names[0]]
        assert bound.layers is bound.observer_layers and bound.raw_exact[3] is bound.layers[6]
        assert bound.resident_roots()["raw_layers"] is bound.layers
        assert f.record["selected_layer_ids"] == list(range(7))
        assert f.record["selected_leaf_count"] == 201 and f.record["overlay_tensor_count"] == 12
        assert "complete_checkpoint" in f.record["integrity_scope"]
        assert "numerical_admission" not in f.record


@pytest.mark.parametrize("mutation", [None, "abstract", "shape", "dtype", "tree", "sharding"])
def test_loaded_tree_requires_real_arrays_and_exact_contract(mutation):
    import jax
    from jax.sharding import NamedSharding, Mesh, PartitionSpec as P
    mesh = Mesh(np.asarray(jax.devices()[:1], object), ("feature",))
    sharding = NamedSharding(mesh, P())
    value = jax.device_put(np.zeros((2, 3), np.float32), sharding)
    expected = jax.ShapeDtypeStruct((2, 3), np.float32, sharding=sharding)
    actual, abstract = (value,), (expected,)
    if mutation == "abstract": actual = (expected,)
    elif mutation == "shape": abstract = (jax.ShapeDtypeStruct((3, 2), np.float32, sharding=sharding),)
    elif mutation == "dtype": abstract = (jax.ShapeDtypeStruct((2, 3), np.int32, sharding=sharding),)
    elif mutation == "tree": actual = [value]
    elif mutation == "sharding": abstract = (NS(shape=(2, 3), dtype=value.dtype, sharding=None),)
    if mutation:
        with pytest.raises((ValueError, TypeError)): runtime_module._tree_contract(actual, abstract)
    else:
        runtime_module._tree_contract(actual, abstract)


@pytest.mark.parametrize("mutation", [None, "program", "input", "manifest", "source", "concrete",
                                      "wk_shape", "wk_dtype", "wk_spec", "producer6"])
def test_reused_abstract_preparation_and_all_four_wk_interfaces(monkeypatch, mutation):
    import jax
    from jax.sharding import NamedSharding, Mesh, PartitionSpec as P
    from scripts.greenfield import probe_ws32_prefill_layer as probe
    mesh = Mesh(np.asarray(jax.devices()[:1], object), ("feature",))
    def abstract(shape, dtype, spec=P()):
        return jax.ShapeDtypeStruct(shape, dtype, sharding=NamedSharding(mesh, spec))
    Dsa, Layer = namedtuple("Dsa", "wk_bits_local wk_scale_local"), namedtuple("Layer", "dsa")
    bits, scales = abstract((128, 6144), np.uint8, P(None, "feature")), abstract((1, 48), np.float32, P(None, "feature"))
    layers = [Layer(Dsa(bits, scales)) for _ in range(7)]
    if mutation == "wk_shape": layers[0] = Layer(Dsa(abstract((64, 6144), np.uint8, P(None, "feature")), scales))
    elif mutation == "wk_dtype": layers[0] = Layer(Dsa(abstract((128, 6144), np.int8, P(None, "feature")), scales))
    elif mutation == "wk_spec": layers[0] = Layer(Dsa(abstract((128, 6144), np.uint8), scales))
    elif mutation == "producer6": layers[6] = Layer(Dsa(bits, abstract((1, 12), np.float32, P(None, "feature"))))
    inputs = (abstract((), np.int32),) * 6 + (tuple(layers),)
    prepared = NS(programs={n: NS(execute=object()) for n in runtime_module.compiler.PROGRAMS},
        inputs={n: inputs for n in runtime_module.compiler.PROGRAMS}, manifest_sha256="a"*64,
        source_inventory_sha256="b"*64)
    metadata = NS(manifest={"manifest_sha256":"a"*64, "source":{"inventory_sha256":"b"*64}})
    if mutation == "program": prepared.programs.pop("observer")
    elif mutation == "input": prepared.inputs.pop("observer")
    elif mutation == "manifest": prepared.manifest_sha256 = "c"*64
    elif mutation == "source": prepared.source_inventory_sha256 = "c"*64
    elif mutation == "concrete": prepared.inputs["observer"] = (np.zeros(1),)
    monkeypatch.setattr(runtime_module.compiler, "require_source", lambda repo: None)
    monkeypatch.setattr(runtime_module.compiler, "prepare", lambda *a, **k: pytest.fail("prepared twice"))
    seen = []
    monkeypatch.setattr(probe, "build_wk_programs", lambda *a, **k: (seen.append((a,k)), object()))
    def run(): return runtime_module.compiler_jobs(mesh, metadata, NS(dsa_contract="original"), repo=REPO, prepared=prepared)
    if mutation:
        with pytest.raises(ValueError): run()
        assert not seen
    else:
        reused, jobs = run()
        assert reused is prepared and tuple(n for n,_,_ in jobs) == protocol.PROGRAMS
        assert jobs[0][2] == (bits,scales) and len(seen) == 1
        assert jobs[1][2][0].shape == (128,6144)


def memory_fixture():
    from glm_tpu.greenfield.validation.ws32_prefill_memory import SCHEMA
    analyses = {name: dict(argument_size_in_bytes=300, output_size_in_bytes=70,
        temp_size_in_bytes=30, generated_code_size_in_bytes=100*(i+1), alias_size_in_bytes=0)
        for i, name in enumerate(protocol.PROGRAMS)}
    rows = [dict(device_id=d, platform="tpu", process_index=3,
        buffers=[dict(bytes=500)], accounted_resident_bytes=500,
        memory_stats=dict(bytes_in_use=600, peak_bytes_in_use=700, bytes_limit=protocol.RESERVE+10000))
        for d in (12, 14, 13, 15)]
    census = dict(schema_version=SCHEMA, includes_all_live_arrays=True, devices=rows)
    return census, analyses


@pytest.mark.parametrize("mutation", [None, "missing", "extra", "active", "alias_nonactive", "cap", "bool", "not_all_live", "oom"])
def test_nine_resident_programs_caps_and_all_live_memory(mutation):
    census, analyses = memory_fixture()
    active = "exact_promote"
    if mutation == "missing": analyses.pop("observer")
    elif mutation == "extra": analyses["unexpected"] = dict(analyses[active])
    elif mutation == "active": active = "unexpected"
    elif mutation == "alias_nonactive": analyses["wk_decode"]["alias_size_in_bytes"] = 1
    elif mutation == "cap": analyses["exact_promote"]["output_size_in_bytes"] = (256 << 20) + 1
    elif mutation == "bool": analyses["wk_decode"]["temp_size_in_bytes"] = False
    elif mutation == "not_all_live": census["includes_all_live_arrays"] = False
    elif mutation == "oom": census["devices"][0]["memory_stats"]["bytes_limit"] = protocol.RESERVE + 5199
    if mutation not in (None, "oom"):
        with pytest.raises(ValueError): runtime_module.memory_budget(census, analyses, active_graph=active)
    else:
        result = runtime_module.memory_budget(census, analyses, active_graph=active)
        assert result["resident_graphs"] == list(protocol.PROGRAMS)
        assert result["devices"][0]["resident_code_bytes"] == 4500
        assert result["devices"][0]["estimated_peak_bytes"] == 5200
        assert result["estimate_fits"] is (mutation is None)
        assert result["numerical_admission"] is False


@pytest.mark.parametrize("mutation", [None, "owners", "missing", "overwrite"])
def test_snapshot_uses_every_actual_analysis_and_all_live_roots(monkeypatch, mutation):
    census, analyses = memory_fixture()
    seen = []
    programs = {name: NS(memory_analysis=lambda name=name: (seen.append(name), NS(**analyses[name]))[1])
                for name in protocol.PROGRAMS}
    roots = {"shared": object()}
    bound = NS(resident_roots=lambda: dict(roots), local_slots={12: 9, 14: 13, 13: 25, 15: 29})
    extra = {"two_histories_and_promoted_exact": object()}
    def capture(values, *, devices):
        assert values == {**roots, **extra} and devices == ("fixture",)
        return census
    monkeypatch.setattr(runtime_module, "capture_resident_buffers", capture)
    if mutation == "owners": census["devices"][0]["process_index"] = 7
    elif mutation == "missing": programs.pop("observer")
    elif mutation == "overwrite": extra = {"shared": object()}
    def run(): return runtime_module.capture_memory(bound, programs, devices=("fixture",), process_index=3,
                                                  active_graph="observer", additional_roots=extra)
    if mutation:
        with pytest.raises(ValueError): run()
    else:
        result = run()
        assert seen == list(protocol.PROGRAMS)
        assert result["compiled_memory"] == analyses and result["budget"]["estimate_fits"]
        assert not result["runtime_peak_hbm_measured"]


def test_production_nine_abstract_jobs_without_payload_compile_or_dispatch():
    source = r'''
from pathlib import Path
from unittest.mock import patch
import jax
import numpy as np
from jax.sharding import Mesh
from jax._src.pallas.mosaic import tpu_info
from glm_tpu.greenfield.types import ModelGeometry
from glm_tpu.greenfield.runtime.ws32_decoder import Ws32DecoderConfig
from scripts.greenfield import ws32_history_runtime as runtime
from scripts.greenfield import ws32_history_compile as compiler
from scripts.greenfield import ws32_history_protocol as protocol
assert jax.default_backend() == 'cpu'
tpu_info.registry['cpu'] = lambda: tpu_info.get_tpu_info_for_chip(tpu_info.ChipVersion.TPU_V4, 1)
tpu_info.get_tpu_info.cache_clear()
mesh = Mesh(np.asarray(jax.devices(), object).reshape(8,4), ('expert','feature'))
original_open = Path.open
def checked(path, *args, **kwargs):
    assert path.suffix not in ('.safetensors', '.bin'), path
    if '/glm-ws32-runtime/' in str(path): assert path.name in ('manifest.json','SUCCESS'), path
    return original_open(path, *args, **kwargs)
with patch.object(Path, 'open', checked), \
     patch('jax.device_put', side_effect=AssertionError('payload placement')), \
     patch('jax.stages.Lowered.compile', side_effect=AssertionError('compilation')), \
     patch('jax.stages.Compiled.__call__', side_effect=AssertionError('dispatch')):
    metadata = compiler.read_metadata(Path.cwd())
    config = Ws32DecoderConfig(ModelGeometry.from_dict(metadata.manifest['geometry']),8192,host_main_rope_table=True)
    prepared, jobs = runtime.compiler_jobs(mesh, metadata, config, repo=Path.cwd())
    assert tuple(n for n,_,_ in jobs) == protocol.PROGRAMS
    assert all(isinstance(v,jax.ShapeDtypeStruct) and v.sharding is not None for _,_,x in jobs for v in jax.tree.leaves(x))
    assert len(jax.tree.leaves(prepared.inputs['candidate_b128'][5:7])) == 201
    assert [v.shape for v in jobs[0][2]] == [(128,6144),(1,48)]
    assert str(jobs[1][2][0].dtype) == 'bfloat16' and jobs[1][2][0].shape == (128,6144)
    assert len(prepared.inputs['exact_decode'][0]) == len(prepared.inputs['observer'][8]) == 4
    assert len(prepared.inputs['observer'][8][0].wq_b_weight_aliases) == 4
    with patch.object(compiler,'prepare',side_effect=AssertionError('duplicate preparation')):
        reused, second = runtime.compiler_jobs(mesh, metadata, config, repo=Path.cwd(), prepared=prepared)
    assert reused is prepared and all(a[1] is b[1] for a,b in zip(jobs[2:],second[2:],strict=True))
    assert prepared.inputs['exact_promote'][0][0].wq_b_weight_local.sharding.shard_shape((4096,2048)) == (1024,2048)
print('NINE_ABSTRACT_JOBS_NO_PAYLOAD_COMPILE_OR_DISPATCH',flush=True)
'''
    env = {**os.environ, "JAX_PLATFORMS": "cpu", "XLA_FLAGS": "--xla_force_host_platform_device_count=32"}
    result = subprocess.run([sys.executable, "-c", source], cwd=REPO, env=env, text=True,
                            capture_output=True, timeout=240)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "NINE_ABSTRACT_JOBS_NO_PAYLOAD_COMPILE_OR_DISPATCH" in result.stdout

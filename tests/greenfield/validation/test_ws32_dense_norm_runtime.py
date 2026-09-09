"""Real original/context binding; explicit fixture runtime, payload loader, compiler."""

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pytest

from scripts.greenfield import ws32_dense_frontier_runtime as runtime_module
from scripts.greenfield import ws32_dense_norm_originals as norm
from scripts.greenfield import ws32_dense_norm_prepare as preparation
from scripts.greenfield import ws32_dense_norm_protocol as protocol
from scripts.greenfield import ws32_dense_canonical as canonical
from tests.greenfield.validation.test_ws32_dense_frontier_runtime import original, setup

REPO = Path(__file__).resolve().parents[3]


@pytest.fixture(scope="module")
def bundle():
    return norm.load_bundle(
        Path("/home/gianl/glm-run") / norm.TAG / "fleet/rank0", repo=REPO, rank=0
    )


@pytest.mark.parametrize(
    "failure",
    [
        None,
        "identity",
        "context",
        "mesh",
        "selected",
        "duplicates",
        "peer_originals",
        "mixed_tag",
    ],
)
@pytest.mark.parametrize("canonical_mode", [False, True])
def test_norm_originals_before_payload_and_exact_selected_weight_join(
    tmp_path,
    monkeypatch,
    original,
    bundle,
    failure,
    canonical_mode,
):
    prior, preflight, record, runtime, save = setup(tmp_path, monkeypatch, original)
    norm_runner, arrays, identity = bundle
    norm_runner, identity = deepcopy(norm_runner), deepcopy(identity)
    tag = "greenfield_fp8_ws32_dense_norm_d01_20260909T170000000000000Z"
    selected = canonical if canonical_mode else protocol
    if canonical_mode:
        tag = tag.replace("dense_norm", "dense_canonical")
    for value in (record, preflight):
        value.update(protocol=selected.PROTOCOL, tag=tag)
    preflight["norm_originals"] = deepcopy(identity)
    if failure == "mixed_tag":
        record["tag"] = "greenfield_fp8_ws32_dense_frontier_d01_20260909T170000000000000Z"
    elif failure == "identity":
        identity["receipt_sha256"] = "0" * 64
    elif failure == "context":
        norm_runner["prompt_ids_sha256"] = "0" * 64
    elif failure == "mesh":
        norm_runner["physical_device_ids"][0].reverse()
    save()
    events = []

    def retained(root, *, repo, rank):
        events.append("originals")
        assert (
            root == tmp_path / "retained_norm_reference" and repo == REPO and rank == 0
        )
        return norm_runner, arrays, identity

    monkeypatch.setattr(norm, "load_bundle", retained)
    records = {
        v["device_slot"]: {**header, "sha256": v["file_sha256"]}
        for v, header in zip(
            prior["local_device_slots"], preflight["headers"], strict=True
        )
    }
    metadata = NS(
        records_by_slot=records,
        manifest={"source": {"inventory_sha256": prior["source_inventory_sha256"]}},
    )
    subset = NS(metadata=metadata)
    prepared = NS(
        config=object(),
        tensor_names=("embedding", "layer0", "layer1"),
        manifest_sha256=prior["checkpoint_manifest_sha256"],
        payload_bytes_per_chip=102589760,
    )
    def selected_metadata(*args, **kwargs):
        assert kwargs == (dict(canonical_dense=True) if canonical_mode else {})
        return preflight["checkpoint_pins"], subset

    monkeypatch.setattr(
        runtime_module.preflight_module,
        "selected_metadata",
        selected_metadata,
    )
    monkeypatch.setattr(canonical if canonical_mode else preparation, "prepare", lambda *args, **kw: prepared)
    monkeypatch.setattr(
        runtime_module.preparation,
        "prepare",
        lambda *args, **kw: pytest.fail("norm selected original builder"),
    )

    def host_inputs(*args):
        events.append("inputs")
        return np.arange(8155, dtype=np.int32), np.zeros((8192, 64), np.float32)

    monkeypatch.setattr(runtime_module.execution, "host_inputs", host_inputs)
    loaded_owners = deepcopy(bundle[0]["local_device_slots"])
    loaded_owners.reverse()  # Owner records compare byslot, not incidental list order.
    if failure == "selected":
        loaded_owners[0]["observed_selected_tensor_sha256"][
            "model.layers.0.post_attention_layernorm.weight"
        ] = ("0" * 64)
    elif failure == "duplicates":
        loaded_owners[-1] = deepcopy(loaded_owners[0])

    def load(*args, **kw):
        events.append("load")
        return NS(
            layer_ids=(0, 1),
            include_embedding=True,
            payload_bytes_per_chip=102589760,
            arrays={n: n for n in prepared.tensor_names},
            local_device_slots=loaded_owners,
            integrity_scope="fixture selected payload",
            device_memory_before=(),
            device_memory_after=(),
        )

    monkeypatch.setattr(runtime_module, "load_ws32_layer_subset", load)
    monkeypatch.setattr(
        runtime_module,
        "ws32_decoder_weight_names",
        lambda config: NS(embedding_local="embedding", layers=("layer0", "layer1")),
    )
    monkeypatch.setattr(
        runtime_module,
        "_bind_weight_name_tree",
        lambda names, values: ("embedding", ("layer0", "layer1")),
    )
    import jax.sharding

    monkeypatch.setattr(jax.sharding, "NamedSharding", lambda *args: None)
    runtime[0].make_array_from_callback = lambda shape, sharding, callback: callback(
        (slice(None), slice(None))
    )
    runtime[0].block_until_ready = lambda value: events.append("rope")

    def execute(**kw):
        events.append("execute")
        key = "canonical_originals" if canonical_mode else "norm_originals"
        assert kw[key] is arrays and kw["prepared"] is prepared
        assert ("norm_originals" if canonical_mode else "canonical_originals") not in kw
        assert kw["local_slots"] == {12: 9, 14: 13, 13: 25, 15: 29}

    monkeypatch.setattr(runtime_module.execution, "execute", execute)

    def consensus(ok):
        events.append(("vote", ok))
        return ok and not (
            failure == "peer_originals" and events[-2:-1] == ["originals"]
        )

    def run():
        runtime_module.execute_bound(
            root=tmp_path,
            record=record,
            repo=REPO,
            runtime=runtime,
            consensus=consensus,
            inspect_program=lambda *args: None,
        )

    if failure:
        with pytest.raises((ValueError, RuntimeError)):
            run()
        assert "execute" not in events and "rope" not in events
        if failure not in ("selected", "duplicates"):
            assert "load" not in events
    else:
        run()
        assert [v for v in events if isinstance(v, str)] == [
            "originals",
            "inputs",
            "load",
            "rope",
            "execute",
        ]
        assert record["norm_originals"] == identity

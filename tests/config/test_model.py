from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from glm_tpu.exceptions import GeometryValidationError, TopologyValidationError
from glm_tpu.config.model import ModelGeometry
from glm_tpu.distributed.topology import PhysicalDevice, PhysicalTopology
# The pinned GLM-5.3 config; its geometry equals the archived GLM-5.2 file's (model identity tests below).
from tools.equivalence.fixture import config_json
from glm_tpu.config import model
from glm_tpu.config.site import site_args


REPO = Path(__file__).resolve().parents[2]

# The retained GLM-5.2 config (archived at S2f; archive/research-20260922:configs/glm-5.2-fp8-config.json,
# sha256 22e49334...) as the tests below compared it: its canonical JSON without transformers_version
# (modules_to_not_convert sorted) and the geometry hash the frozen parser derives from it.
RETAINED_CONFIG_CANONICAL_SHA256='ba0b99f8cc6bcb0d17bd9607309a999e9f9d8067dc56aa04c54b6638a6baee74'
RETAINED_GEOMETRY_HASH='5e979eafb202e5e122081106062d11806dc3bbb82c4798cd2c80199f8cc0e91b'


@pytest.fixture(scope="module")
def geometry() -> ModelGeometry:
    config = config_json()
    return ModelGeometry.from_hf_config(config)


def topology(*, reverse: bool = False) -> PhysicalTopology:
    devices = []
    device_id = 0
    for x in range(2):
        for y in range(4):
            for z in range(4):
                process = device_id // 4
                devices.append(
                    PhysicalDevice(
                        device_id=device_id,
                        process_index=process,
                        local_device_id=device_id % 4,
                        coordinates=(x, y, z),
                        core_on_chip=0,
                        platform="tpu",
                        device_kind="TPU v4",
                    )
                )
                device_id += 1
    if reverse:
        devices.reverse()
    return PhysicalTopology(
        slice_name="example-slice",
        topology_shape=(2, 4, 4),
        devices=tuple(devices),
    )


def test_checked_in_glm_geometry_is_exact(geometry: ModelGeometry) -> None:
    assert geometry.model_id == "zai-org/GLM-5.2-FP8"
    assert geometry.num_layers == 78
    assert geometry.first_dense_layers == 3
    assert geometry.hidden_size == 6144
    assert geometry.num_routed_experts == 256
    assert geometry.routed_top_k == 8
    assert geometry.dsa_top_k == 2048
    assert geometry.dsa_indexer_heads == 32
    assert geometry.index_share_group_size == 4
    assert geometry.qk_nope_head_dim == 192
    assert geometry.qk_rope_head_dim == 64
    assert geometry.v_head_dim == 256
    assert geometry.num_nextn_predict_layers == 1
    assert geometry.fp8_block_shape == (128, 128)
    assert geometry.weight_storage_dtype == "fp8:e4m3"
    assert len(geometry.mlp_layer_types) == len(geometry.indexer_types) == 78


def test_geometry_refuses_inconsistent_layer_schedule(
    geometry: ModelGeometry,
) -> None:
    with pytest.raises(GeometryValidationError, match="first_dense_layers"):
        replace(
            geometry,
            mlp_layer_types=("sparse",) + geometry.mlp_layer_types[1:],
        )


def test_geometry_refuses_boolean_integer(geometry: ModelGeometry) -> None:
    with pytest.raises(GeometryValidationError, match="hidden_size"):
        replace(geometry, hidden_size=True)


def test_topology_is_canonical_and_content_addressed() -> None:
    ordered = topology()
    reversed_input = topology(reverse=True)
    assert ordered == reversed_input
    assert ordered.topology_hash == reversed_input.topology_hash
    assert len(ordered.topology_hash) == 64
    assert ordered.process_indices == tuple(range(8))


def test_topology_refuses_duplicate_coordinates() -> None:
    good = topology()
    duplicate = replace(good.devices[1], coordinates=good.devices[0].coordinates)
    with pytest.raises(TopologyValidationError, match="coordinates"):
        replace(good, devices=(good.devices[0], duplicate, *good.devices[2:]))


# Model identity: new model identities must not inherit old weights or a changed architecture.
def test_inventory_binds_model_revision_config_and_index():
    fields=dict(model_id=model.MODEL_ID,source_revision=model.REVISION,
                config_sha256=model.CONFIG_SHA,index_sha256=model.INDEX_SHA)
    model.require_inventory(SimpleNamespace(**fields))
    for key in fields:
        with pytest.raises(ValueError,match='GLM-5.3 source'):
            model.require_inventory(SimpleNamespace(**(fields|{key:'different'})))


def test_architecture_matches_retained_numerical_geometry():
    from hashlib import sha256
    from glm_tpu.config.model import ModelGeometry
    repo=Path(__file__).resolve().parents[2]
    new=json.loads((repo/'glm_tpu/models/glm_moe_dsa/hf_config/config.json').read_bytes())
    assert ModelGeometry.from_hf_config(new).geometry_hash==RETAINED_GEOMETRY_HASH
    new.pop('transformers_version')
    new['quantization_config']['modules_to_not_convert'].sort()
    canonical=json.dumps(new,sort_keys=True,separators=(',',':')).encode()
    assert sha256(canonical).hexdigest()==RETAINED_CONFIG_CANONICAL_SHA256


def test_geometry_binds_new_identity_preserving_all_dimensions():
    from dataclasses import replace
    from glm_tpu.config.model import ModelGeometry
    repo=Path(__file__).resolve().parents[2]
    # the retained geometry (equal to the GLM-5.2 config's, test above)
    old=ModelGeometry.from_hf_config(json.loads((repo/'glm_tpu/models/glm_moe_dsa/hf_config/config.json').read_bytes()))
    assert old.geometry_hash==RETAINED_GEOMETRY_HASH
    new=model.geometry(repo)
    assert new.model_id==model.MODEL_ID
    assert replace(new,model_id=old.model_id)==old
    assert new.geometry_hash!=old.geometry_hash


def test_geometry_rejects_changed_config(tmp_path):
    path=tmp_path/model.TEMPLATE_PATH.parent/'config.json'
    path.parent.mkdir(parents=True)
    path.write_text('{}')
    with pytest.raises(ValueError,match='geometry configuration'):
        model.geometry(tmp_path)


def test_template_and_config_checked_before_tokenizer_import(tmp_path):
    repo=tmp_path/'repo'
    path=repo/model.TEMPLATE_PATH
    path.parent.mkdir(parents=True)
    path.write_text('old or changed template')
    with pytest.raises(ValueError,match='chat template'):
        model.verified_template(repo,tmp_path)


def site_fixture(tmp_path):
    """An example site (tests/fixtures/site.py) whose model path holds a SOURCE_COMPLETE receipt."""
    from hashlib import sha256
    from glm_tpu.config.site import SiteConfig
    from tests.fixtures.site import example_mapping
    complete=dict(passed=True,repository=model.MODEL_ID,revision=model.REVISION,
                  verified_shards=141,verified_bytes=755632050320)
    raw=json.dumps(complete).encode()
    mapping=example_mapping(tmp_path,paths=dict(model_path=str(tmp_path)),checkpoint=dict(
        source_inventory_sha256='a'*64,manifest_sha256='b'*64,success_sha256='c'*64,
        source_complete_sha256=sha256(raw).hexdigest()))
    (tmp_path/'SOURCE_COMPLETE.json').write_bytes(raw)
    return SiteConfig.from_mapping(mapping),mapping,complete


def test_new_site_binds_complete_source_without_legacy_overlay(tmp_path):
    site,_,_=site_fixture(tmp_path)
    args=site_args(SimpleNamespace(),site)
    assert args.model_id==model.MODEL_ID and args.model_revision==model.REVISION
    assert args.source_inventory_sha256=='a'*64 and args.checkpoint_manifest_sha256=='b'*64
    assert args.checkpoint_success_sha256=='c'*64
    assert args.checkpoint_root==site.checkpoint.root and args.source_inventory==site.checkpoint.source_inventory
    assert args.num_processes==8 and args.checkpoint_transport=='shm'
    assert args.hlo_dump_root==site.paths.hlo_dump_root
    assert (args.topology_capture_root,args.slice_name,args.mesh_sha256)==(
        site.topology.capture_root,site.topology.slice_name,site.topology.mesh_sha256)
    assert not hasattr(args,'strategy_nd_dense_overlay_root')


@pytest.mark.parametrize('field,value',[
    ('passed',False),('verified_shards',17),('verified_bytes',91184236216),
    ('revision','0'*40),('repository','zai-org/GLM-5.2-FP8'),
])
def test_incomplete_or_other_model_source_refused_even_with_matching_digest(
        tmp_path,field,value):
    from hashlib import sha256
    from glm_tpu.config.site import SiteConfig
    _,mapping,complete=site_fixture(tmp_path)
    complete[field]=value;raw=json.dumps(complete).encode()
    mapping['checkpoint']['source_complete_sha256']=sha256(raw).hexdigest()
    (tmp_path/'SOURCE_COMPLETE.json').write_bytes(raw)
    with pytest.raises(ValueError,match='incomplete'):
        site_args(SimpleNamespace(),SiteConfig.from_mapping(mapping))


def test_changed_source_completion_receipt_refused(tmp_path):
    site,_,complete=site_fixture(tmp_path)
    (tmp_path/'SOURCE_COMPLETE.json').write_bytes(json.dumps(dict(complete,note='x')).encode())
    with pytest.raises(ValueError,match='source completion identity'):
        site_args(SimpleNamespace(),site)


def test_site_does_not_fall_back_when_missing(tmp_path):
    from glm_tpu.config.site import SiteConfig, SiteConfigError
    with pytest.raises(SiteConfigError,match='no site file'):SiteConfig.load(tmp_path/'site.toml')
    site,_,_=site_fixture(tmp_path)
    (tmp_path/'SOURCE_COMPLETE.json').unlink()
    with pytest.raises(FileNotFoundError):site_args(SimpleNamespace(),site)

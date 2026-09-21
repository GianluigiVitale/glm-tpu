"""New model identities must not inherit old weights or a changed architecture."""
from pathlib import Path
import json
from types import SimpleNamespace

import pytest

from glm_tpu.optimized import model


def test_inventory_binds_model_revision_config_and_index():
    fields=dict(model_id=model.MODEL_ID,source_revision=model.REVISION,
                config_sha256=model.CONFIG_SHA,index_sha256=model.INDEX_SHA)
    model.require_inventory(SimpleNamespace(**fields))
    for key in fields:
        with pytest.raises(ValueError,match='GLM-5.3 source'):
            model.require_inventory(SimpleNamespace(**(fields|{key:'different'})))


def test_architecture_matches_retained_numerical_geometry():
    from glm_tpu.greenfield.types import ModelGeometry
    repo=Path(__file__).resolve().parents[2]
    old=json.loads((repo/'configs/glm-5.2-fp8-config.json').read_bytes())
    new=json.loads((repo/'reference/hf-glm53/config.json').read_bytes())
    assert ModelGeometry.from_hf_config(new)==ModelGeometry.from_hf_config(old)
    for value in (old,new):
        value.pop('transformers_version')
        value['quantization_config']['modules_to_not_convert'].sort()
    assert new==old


def test_geometry_binds_new_identity_preserving_all_dimensions():
    from dataclasses import replace
    from glm_tpu.greenfield.types import ModelGeometry
    repo=Path(__file__).resolve().parents[2]
    old=ModelGeometry.from_hf_config(json.loads((repo/'configs/glm-5.2-fp8-config.json').read_bytes()))
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
    from hashlib import sha256
    complete=dict(passed=True,repository=model.MODEL_ID,revision=model.REVISION,
                  verified_shards=141,verified_bytes=755632050320)
    raw=json.dumps(complete).encode()
    config=dict(schema='glm_ws32_glm53_site_v1',model_id=model.MODEL_ID,
        model_revision=model.REVISION,source_inventory_sha256='a'*64,
        checkpoint_manifest_sha256='b'*64,checkpoint_success_sha256='c'*64,
        source_complete_sha256=sha256(raw).hexdigest(),
        source_inventory='/home/gianl/gcs-models/checkpoints/greenfield/glm53/plan/source_inventory.json',
        checkpoint_root='/dev/shm/glm-ws32-runtime/glm53-test')
    (tmp_path/'configs').mkdir()
    (tmp_path/'configs/glm53-site.json').write_text(json.dumps(config))
    (tmp_path/'SOURCE_COMPLETE.json').write_bytes(raw)
    return config,complete


def test_new_site_binds_complete_source_without_legacy_overlay(monkeypatch,tmp_path):
    config,_=site_fixture(tmp_path)
    monkeypatch.setattr(model,'TOKENIZER_ROOT',tmp_path)
    args=model.site_args(SimpleNamespace(),repo=tmp_path)
    assert args.model_revision==model.REVISION
    assert args.source_inventory_sha256==config['source_inventory_sha256']
    assert args.checkpoint_root==Path(config['checkpoint_root'])
    assert args.num_processes==8 and args.checkpoint_transport=='shm'
    assert not hasattr(args,'strategy_nd_dense_overlay_root')


@pytest.mark.parametrize('field,value',[
    ('passed',False),('verified_shards',17),('verified_bytes',91184236216),
    ('revision','0'*40),('repository','zai-org/GLM-5.2-FP8'),
])
def test_incomplete_or_other_model_source_refused_even_with_matching_digest(
        monkeypatch,tmp_path,field,value):
    from hashlib import sha256
    config,complete=site_fixture(tmp_path)
    complete[field]=value;raw=json.dumps(complete).encode()
    config['source_complete_sha256']=sha256(raw).hexdigest()
    (tmp_path/'configs/glm53-site.json').write_text(json.dumps(config))
    (tmp_path/'SOURCE_COMPLETE.json').write_bytes(raw)
    monkeypatch.setattr(model,'TOKENIZER_ROOT',tmp_path)
    with pytest.raises(ValueError,match='incomplete'):
        model.site_args(SimpleNamespace(),repo=tmp_path)


def test_site_does_not_fall_back_when_missing(tmp_path):
    with pytest.raises(FileNotFoundError):model.site_args(SimpleNamespace(),repo=tmp_path)

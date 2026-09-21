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


def test_template_and_config_checked_before_tokenizer_import(tmp_path):
    repo=tmp_path/'repo'
    path=repo/model.TEMPLATE_PATH
    path.parent.mkdir(parents=True)
    path.write_text('old or changed template')
    with pytest.raises(ValueError,match='chat template'):
        model.verified_template(repo,tmp_path)

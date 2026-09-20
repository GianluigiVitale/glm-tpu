from dataclasses import replace
import pytest
from glm_tpu.greenfield.runtime.ws32_decoder import Ws32DecoderConfig
from glm_tpu.perf.mtp_checkpoint import mtp_source_placements
from tests.greenfield.checkpoint.test_ws32_runtime import _geometry,_tensor


def config():return Ws32DecoderConfig(_geometry(),8192,host_main_rope_table=True)


def test_mtp_projection_output_features_and_whole_concatenation():
    source=_tensor('model.layers.78.eh_proj.weight','BF16',(6144,12288))
    placements=mtp_source_placements(source,config())
    assert len(placements)==32
    for p in placements:
        assert p.source_name==source.name and p.destination_name=='mtp.eh_proj.weight'
        assert p.partition_spec==('feature',None)
        assert p.source_starts==(p.feature_coordinate*1536,0)
        assert p.source_stops==((p.feature_coordinate+1)*1536,12288)
        assert p.destination_shape==(1536,12288)
    assert sum(p.byte_count for p in placements)==8*source.byte_count


@pytest.mark.parametrize('name,destination',[('enorm.weight','mtp.enorm.weight'),('hnorm.weight','mtp.hnorm.weight'),('shared_head.norm.weight','model.norm.weight')])
def test_native_norms_are_distinct_and_feature_sharded(name,destination):
    source=_tensor('model.layers.78.'+name,'BF16',(6144,))
    ps=mtp_source_placements(source,config())
    assert len(ps)==32 and all(p.source_name==source.name for p in ps)
    assert {p.destination_name for p in ps}=={destination}
    assert {p.partition_spec for p in ps}=={('feature',)}
    assert {p.destination_shape for p in ps}=={(1536,)}


def test_native_expert_preserves_original_source_and_exact_bits():
    source=_tensor('model.layers.78.mlp.experts.130.gate_proj.weight','F8_E4M3',(2048,6144))
    ps=mtp_source_placements(source,config())
    assert [p.slot for p in ps]==[16,17,18,19]
    for p in ps:
        assert p.source_name==source.name and p.destination_name=='model.layers.0.mlp.experts.gate_proj.weight_bits'
        assert p.destination_starts==(2,0,0) and p.transform=='fp8_bits'
    assert sum(p.byte_count for p in ps)==source.byte_count


@pytest.mark.parametrize('source',[
    _tensor('model.layers.74.enorm.weight','BF16',(6144,)),
    _tensor('model.embed_tokens.weight','BF16',(256,6144)),
    _tensor('model.layers.78.eh_proj.weight','BF16',(12288,6144)),
    _tensor('model.layers.78.enorm.weight','F32',(6144,)),
    _tensor('model.layers.78.unsupported.weight','BF16',(6144,)),
])
def test_wrong_layer_or_schema_is_refused(source):
    with pytest.raises(ValueError):mtp_source_placements(source,config())

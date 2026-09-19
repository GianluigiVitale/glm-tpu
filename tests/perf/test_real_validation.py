"""Fail-closed research HLO and physical-memory admission, CPU only."""
import pytest

from glm_tpu.perf.real_validation import inspect_research_hlo, memory_projection


def hlo(*, groups=None, dtype='f32', count=4, partitions=32, opcode='all-reduce'):
    if groups is None:
        groups = '{'+','.join('{'+','.join(map(str,range(i,i+4)))+'}' for i in range(0,32,4))+'}'
    return f'''HloModule test, num_partitions={partitions}
ENTRY main {{
  p = {dtype}[{count}] parameter(0)
  ROOT r = {dtype}[{count}] {opcode}(p), replica_groups={groups}, use_global_device_ids=true
}}
'''


def test_axis_exchange_admitted_and_censused():
    r=inspect_research_hlo(hlo())
    assert r['passed'] and r['maximum_collective_payload_bytes']==16
    assert r['collectives']=={'all-reduce':1}
    groups='{'+','.join('{'+','.join(map(str,range(i,32,4)))+'}' for i in range(4))+'}'
    assert inspect_research_hlo(hlo(groups=groups,opcode='all-gather'))['passed']


@pytest.mark.parametrize('options',[
    {'partitions':8}, {'groups':'{{0,1,2,3}}'}, {'dtype':'f8e4m3fn'},
    {'count':128*1024**2}, {'opcode':'all-to-all'},
    {'groups':'{{'+','.join(map(str,range(32)))+'}}','count':2048},
    {'groups':'{{'+','.join(map(str,range(32)))+'},{0}}'},
])
def test_graph_refusals(options):
    with pytest.raises(ValueError):inspect_research_hlo(hlo(**options))


def test_partition_local_group_ids_are_refused():
    with pytest.raises(ValueError):
        inspect_research_hlo(hlo().replace('use_global_device_ids=true','use_global_device_ids=false'))


def test_async_exchange_payload_is_checked():
    assert inspect_research_hlo(hlo(opcode='all-reduce-start'))['passed']
    with pytest.raises(ValueError):inspect_research_hlo(hlo(opcode='all-reduce-start',count=128*1024**2))


def test_memory_all_four_chips_and_resident_output_scratch_code():
    stats=[dict(device_id=i,bytes_in_use=100,bytes_limit=1000) for i in range(4)]
    mem=dict(output_size_in_bytes=200,temp_size_in_bytes=300,generated_code_size_in_bytes=50,alias_size_in_bytes=100)
    r=memory_projection(stats,mem,reserve_bytes=100)
    assert r['passed'] and {x['predicted_bytes'] for x in r['chips']}=={650}
    stats[3]['bytes_limit']=650
    assert not memory_projection(stats,mem,reserve_bytes=100)['passed']
    with pytest.raises(ValueError):memory_projection(stats[:3],mem)
    stats[3]['device_id']=0
    with pytest.raises(ValueError):memory_projection(stats,mem)
    with pytest.raises(ValueError):memory_projection(stats,dict(mem,temp_size_in_bytes=-1))


def test_db610_builder_accepts_greedy_without_uniform_before_loading_weights():
    import os
    import subprocess
    import sys
    code = '''
import json
from pathlib import Path
import jax, numpy as np
from jax.sharding import Mesh
from glm_tpu.greenfield.types import ModelGeometry
from glm_tpu.greenfield.runtime.ws32_decoder import Ws32DecoderConfig
from glm_tpu.perf.real_validation import build_db610_decoder
mesh=Mesh(np.asarray(jax.devices(),object).reshape(8,4),('expert','feature'))
config=Ws32DecoderConfig(ModelGeometry.from_hf_config(json.loads(Path('configs/glm-5.2-fp8-config.json').read_text())),8192,host_main_rope_table=True)
for lse,two_stage in ((True,True),(False,False),(True,False),(False,True)):
    program=build_db610_decoder(mesh,config,lse_attention=lse,dsa_two_stage=two_stage)
    assert program.takes_uniform is False
    assert program.options.lse_attention is lse and program.options.dsa_two_stage is two_stage
fixed=build_db610_decoder(mesh,config,write_empty_slot=True)
assert fixed.options.routed_projection.write_empty_slot is True
'''
    result=subprocess.run([sys.executable,'-c',code],capture_output=True,text=True,
        env=dict(os.environ,JAX_PLATFORMS='cpu',XLA_FLAGS='--xla_force_host_platform_device_count=32'),timeout=60)
    assert result.returncode==0,result.stdout+result.stderr


def fake_completed_fleet(root):
    import json
    controller=dict(code_hash='c'*40,source_manifest_sha256='s'*64,input_sha256='i'*64)
    (root/'controller_identity.json').write_text(json.dumps(controller))
    for i in range(8):
        phases={name:dict(passed=True) for name in ['verify_checkpoint','load_checkpoint','prefill_frontier',
            'prefill_cache_finite','final_cache_finite',*[f'prefill_block_{j}' for j in range(16)],
            *[f'decode_health_{j}' for j in range(1,29)]]}
        programs={name:dict(stablehlo_sha256='a'*64,optimized_hlo_sha256='b'*64,
            compiled_memory={},memory_admission=dict(passed=True),hlo_admission=dict(passed=True))
            for name in ('wk_decode','wk_promote','prefill_128','prefill_114','decode')}
        identity=dict(db_run_id=610,prompt_tokens=2034,reference_tokens=29,
            **{k:'x'*64 for k in ('run_tag','runner_sha256','ledger_sha256','prompt_sha256','reference_sha256','token_oracle_manifest_sha256')})
        row=dict(schema='glm_perf_real_validation_rank_v1',rank=i,jax_process_index=(i+3)%8,
            hostname='test-w-'+str(i),complete=True,devices=32,capacity=8192,jax='test',**controller,
            input_identity=identity,checkpoint=dict(manifest_sha256='m',success_sha256='s',inventory_sha256='i',verified_slots=list(range(i*4,i*4+4))),
            physical_identity=dict(mesh_sha256='m',topology_sha256='t',fleet_sha256='f',local_slots=list(range(i*4,i*4+4))),
            phases=phases,programs=programs,finite_cache_checks=dict(after_prefill=True,after_decode=True),
            token_comparison=dict(compared=29,matches=29,all_equal=True,first_mismatch_index=None,observed_sha256='o'*64),
            prefill=dict(prompt_tokens=2034,wall_seconds=20.,prompt_tokens_per_second=101.7),
            decode=dict(samples=28,p50_ms=70.,p99_ms=80.,after_five_warm_steps_p50_ms=70.,model_tokens_per_second=14.,
                        memory_after=[dict(device_id=i*4+j,peak_bytes_in_use=28,bytes_limit=33) for j in range(4)]))
        (root/f'validation.rank{i}.json').write_text(json.dumps(row))


def test_fleet_summary_requires_every_rank_and_every_block(tmp_path):
    import json
    from glm_tpu.perf.real_validation import summarize_real_validation
    fake_completed_fleet(tmp_path)
    assert summarize_real_validation(tmp_path)['db610_token_check_passed']
    p=tmp_path/'validation.rank7.json';r=json.loads(p.read_text())
    del r['phases']['prefill_block_15'];p.write_text(json.dumps(r))
    with pytest.raises(ValueError,match='execution phase'):summarize_real_validation(tmp_path)
    p.unlink()
    with pytest.raises(FileNotFoundError):summarize_real_validation(tmp_path)


def test_fleet_summary_token_mismatch_is_not_admission_and_payloads_stay_private(tmp_path):
    import json
    from glm_tpu.perf.real_validation import summarize_real_validation
    fake_completed_fleet(tmp_path)
    for p in tmp_path.glob('validation.rank*.json'):
        r=json.loads(p.read_text())
        r['token_comparison'].update(matches=28,all_equal=False,first_mismatch_index=9,private_tokens=['DO_NOT_PUBLISH'])
        r['input_identity']['private_prompt']='DO_NOT_PUBLISH'
        p.write_text(json.dumps(r))
    summary=summarize_real_validation(tmp_path)
    assert summary['complete_experiment'] and not summary['db610_token_check_passed']
    assert 'DO_NOT_PUBLISH' not in json.dumps(summary)


@pytest.mark.parametrize('failure',['duplicate_owner','different_graph','different_source','unfinished'])
def test_fleet_summary_rejects_incompatible_evidence(tmp_path,failure):
    import json
    from glm_tpu.perf.real_validation import summarize_real_validation
    fake_completed_fleet(tmp_path)
    p=tmp_path/'validation.rank7.json';r=json.loads(p.read_text())
    if failure=='duplicate_owner':
        r['physical_identity']['local_slots']=r['checkpoint']['verified_slots']=[0,1,2,3]
    elif failure=='different_graph':r['programs']['decode']['optimized_hlo_sha256']='z'*64
    elif failure=='different_source':r['source_manifest_sha256']='z'*64
    else:r['complete']=False
    p.write_text(json.dumps(r))
    with pytest.raises(ValueError):summarize_real_validation(tmp_path)


@pytest.mark.parametrize('fault',[None,'missing_host','missing_layer','graph_drift','missing_comparison'])
def test_layerwise_summary_requires_complete_diagnostics_and_omits_payload(tmp_path,fault):
    import json
    from glm_tpu.perf.real_validation import summarize_real_validation
    fake_completed_fleet(tmp_path)
    for i in range(8):
        p=tmp_path/f'validation.rank{i}.json';r=json.loads(p.read_text())
        stats=dict(finite=True,nonzero=10,elements=10,max_abs=1.,rms=.5,private_activation='DO_NOT_PUBLISH')
        d=dict(schema='glm_perf_layerwise_diagnostic_v1',compiler_boundary_changed=True,
               embedding=stats,final_residual=stats,whole_final_residual=stats,
               layers=[dict(layer=j,healthy=True,update=stats,carried=stats,normalized_input=stats) for j in range(78)],
               **{k:True for k in ('head_healthy','whole_healthy','whole_first_token_matches_db610',
                  'split_first_token_matches_db610','split_and_whole_token_equal','split_and_whole_residual_bitwise')})
        r['first_decode_diagnostic']=d
        r['phases']['layerwise_diagnostic']=dict(passed=True)
        r['diagnostic_programs']={n:dict(r['programs']['decode']) for n in
            ('diagnostic_embedding','diagnostic_layer_full_dense','diagnostic_layer_full_sparse','diagnostic_layer_shared_sparse','diagnostic_head')}
        if i==7:
            if fault=='missing_host':del r['first_decode_diagnostic']
            elif fault=='missing_layer':d['layers'].pop()
            elif fault=='graph_drift':r['diagnostic_programs']['diagnostic_head']['stablehlo_sha256']='z'*64
            elif fault=='missing_comparison':del d['whole_healthy']
        p.write_text(json.dumps(r))
    if fault:
        with pytest.raises(ValueError):summarize_real_validation(tmp_path)
    else:
        summary=summarize_real_validation(tmp_path)
        assert len(summary['first_decode_diagnostic']['layers'])==78
        assert 'DO_NOT_PUBLISH' not in json.dumps(summary)


@pytest.mark.parametrize('fault',[None,'missing_host','wrong_variant','graph_drift'])
def test_ablation_summary_scope_and_private_payload(tmp_path,fault):
    import json
    from glm_tpu.perf.real_validation import summarize_real_validation
    fake_completed_fleet(tmp_path)
    for i in range(8):
        p=tmp_path/f'validation.rank{i}.json';r=json.loads(p.read_text())
        r['ablation_programs']={};r['first_decode_ablations']={}
        for label,lse,two_stage in [('d1_d8',False,False),('d1_d8_d5',True,False),('d1_d8_d10',False,True)]:
            name='ablation_'+label
            r['phases'][name+'_execute']=dict(passed=True)
            r['ablation_programs'][name]=dict(r['programs']['decode'])
            r['first_decode_ablations'][label]=dict(lse_attention=lse,dsa_two_stage=two_stage,
                healthy=True,first_token_matches_db610=False,token_sha256='x'*64,
                final_residual=dict(finite=True,nonzero=0),private_token='DO_NOT_PUBLISH')
        if i==7:
            if fault=='missing_host':del r['first_decode_ablations']
            elif fault=='wrong_variant':r['first_decode_ablations']['d1_d8']['lse_attention']=True
            elif fault=='graph_drift':r['ablation_programs']['ablation_d1_d8']['stablehlo_sha256']='z'*64
        p.write_text(json.dumps(r))
    if fault:
        with pytest.raises(ValueError):summarize_real_validation(tmp_path)
    else:
        summary=summarize_real_validation(tmp_path)
        assert len(summary['first_decode_ablations'])==3
        assert 'DO_NOT_PUBLISH' not in json.dumps(summary)

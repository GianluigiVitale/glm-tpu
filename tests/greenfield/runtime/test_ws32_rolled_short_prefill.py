"""Reproduce the actual production-width rolled graph registration without weights."""

from pathlib import Path
import runpy


ROOT = Path(__file__).resolve().parents[3]


def test_production_raw_registration():
    # Reuse the existing 78-layer abstract fixture and CPU32 subprocess helper.
    helpers = runpy.run_path(
        str(ROOT / "tests/greenfield/runtime/test_ws32_paired_prefill.py")
    )
    source = helpers["_fixture_source"](
        "test_production_78_layer_schema_without_allocating_weights"
    )
    source = source.split("plan=adapter.BatchedPrefillPlan", 1)[0]
    source += r"""
from hashlib import sha256
from unittest.mock import patch
from glm_tpu.greenfield.validation import ws32_prefill_admission as admission
tpu_info.registry['cpu']=lambda:tpu_info.get_tpu_info_for_chip(tpu_info.ChipVersion.TPU_V4,1)
tpu_info.get_tpu_info.cache_clear()
profile=admission.ROLLED_SHORT_PROFILE
registration=admission.rolled_registration(Path.cwd())
plan=admission.short_plan(profile)
programs=adapter.build_graph_pair(mesh,config,plan,**admission.short_program_options(profile))
for graph,rows in plan.graph_rows:
    inputs=(abstract((rows,),jnp.int32),abstract((),jnp.int32),state,weights,wk,rope)
    with patch('jax._src.tpu_custom_call.get_ir_version',return_value=None):
        raw=str(programs[graph].execute.trace(*inputs).lower(lowering_platforms=('tpu',)).compiler_ir('stablehlo')).encode()
    expected=registration['graphs'][graph]
    assert len(raw)==expected['stablehlo_bytes'],(graph,len(raw))
    assert sha256(raw).hexdigest()==expected['stablehlo_sha256'],(graph,sha256(raw).hexdigest())
    print('ROLLED_REGISTERED_GRAPH_PASS',graph,rows,flush=True)
"""
    output = helpers["_cpu"](source, timeout=300)
    assert output.count("ROLLED_REGISTERED_GRAPH_PASS") == 2

"""P4 bitwise CPU interpretation at skewed/empty expert panels."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax._src.pallas.mosaic import tpu_info

from glm_tpu.greenfield.kernels.prefill_expert_panels import build_expert_panels
from glm_tpu.greenfield.kernels.pallas.prefill_panel_fp8 import prefill_panel_fp8_matmul
from glm_tpu.perf.prefill_panels import wide_prefill_panel_fp8_matmul


@pytest.mark.parametrize('counts,offset', [([0,0,0,0],0),([1,33,0,30],0),([32,0,1,31],2)])
@pytest.mark.parametrize('dtype',[jnp.float32,jnp.bfloat16])
def test_wide_output_panels_match_frozen_bitwise(counts,offset,dtype):
    tpu_info.registry['cpu']=lambda:tpu_info.get_tpu_info_for_chip(tpu_info.ChipVersion.TPU_V4,1)
    tpu_info.get_tpu_info.cache_clear()
    rows=max(1,sum(counts))
    if not sum(counts): counts=[1,0,0,0];offset=2
    plan=build_expert_panels(jnp.array(counts,jnp.int32),jnp.int32(offset),rows=rows,local_groups=2)
    rng=np.random.default_rng(44)
    lhs=jnp.asarray(rng.normal(size=(rows,384)),jnp.bfloat16)
    bits=jax.lax.bitcast_convert_type(jnp.asarray(rng.normal(0,.03,(2,512,384)),jnp.float8_e4m3fn),jnp.uint8)
    scales=jnp.asarray(rng.uniform(.2,1.5,(2,4,3)),jnp.float32)
    frozen=jax.jit(lambda x,w,s:prefill_panel_fp8_matmul(x,w,s,plan,result_dtype=dtype,interpret=True))(lhs,bits,scales)
    for width in (256,512):
        out=jax.jit(lambda x,w,s:wide_prefill_panel_fp8_matmul(x,w,s,plan,result_dtype=dtype,output_tile=width,interpret=True))(lhs,bits,scales)
        for a,b in zip(frozen,out):
            np.testing.assert_array_equal(np.asarray(a).view(np.uint8),np.asarray(b).view(np.uint8))

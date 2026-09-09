"""Actual production metadata, abstract output budget and TPU-target raw graph."""

import os
import subprocess
import sys


def test_production_norm_capture_without_weight_reads_or_device_allocation():
    code = r"""
from pathlib import Path
from hashlib import sha256
from unittest.mock import patch
import jax
import numpy as np
from jax.sharding import Mesh,NamedSharding,PartitionSpec as P
from jax._src.pallas.mosaic import tpu_info
from scripts.greenfield.ws32_dense_norm_prepare import prepare,compiler_programs
from scripts.greenfield.ws32_dense_norm_protocol import RAW
from scripts.greenfield.ws32_dense_norm_boundary import build_capture_program,build_completed_dense_suffix
assert jax.default_backend()=='cpu'
mesh=Mesh(np.asarray(jax.devices(),object).reshape(8,4),('expert','feature'))
opened=Path.open
def guarded(path,*args,**kwargs):
 assert path.suffix not in ('.safetensors','.bin'),path
 return opened(path,*args,**kwargs)
with patch.object(Path,'open',guarded),patch('jax.device_put',side_effect=AssertionError('concrete allocation')):
 prepared=prepare(mesh,repo=Path.cwd())
 fn=prepared.program
 assert all(isinstance(x,jax.ShapeDtypeStruct) for x in jax.tree.leaves(prepared.inputs))
 original,packet=jax.eval_shape(fn,*prepared.inputs)
 assert len(original)==2 and all(len(v)==12 for v in original)
 assert len(packet)==11 and all(v.shape[:2]==(8,4) for v in packet.values())
 assert packet['post_norm/summed'].shape==(8,4,128,1536)
 assert packet['post_norm/summed'].dtype==np.float32
 assert packet['post_norm/weight'].shape==(8,4,1536)
 assert packet['boundary/live'].shape==(8,4,128)
 amount=sum(v.size*v.dtype.itemsize//32 for v in packet.values())
 assert amount==2757248,amount
 suffix=build_completed_dense_suffix(mesh,prepared.config)
 suffix_args=(jax.ShapeDtypeStruct((128,6144),'bfloat16',sharding=NamedSharding(mesh,P(None,'feature'))),
              jax.ShapeDtypeStruct((128,),np.bool_,sharding=NamedSharding(mesh,P())),
              prepared.inputs[6][0].dense)
 assert jax.eval_shape(suffix,*suffix_args)[0].shape==(128,6144)
 jobs=compiler_programs(prepared,mesh)
 assert tuple(n for n,_,_ in jobs)==('wk_decode','wk_promote','dense01_norm','dense_suffix')
 owned_shape=jax.eval_shape(jobs[-1][1],*jobs[-1][2])
 assert owned_shape[0].shape==(8,4,128,1536) and owned_shape[1].shape==(8,4,128)
 assert all(isinstance(v,jax.ShapeDtypeStruct) for _,_,args in jobs for v in jax.tree.leaves(args))
tpu_info.registry['cpu']=lambda:tpu_info.get_tpu_info_for_chip(tpu_info.ChipVersion.TPU_V4,1)
tpu_info.get_tpu_info.cache_clear()
for name,program,inputs in jobs:
 with patch('jax._src.tpu_custom_call.get_ir_version',return_value=None):
  raw=str(program.trace(*inputs).lower(lowering_platforms=('tpu',)).compiler_ir('stablehlo')).encode()
 assert raw
 assert (len(raw),sha256(raw).hexdigest())==RAW[name],(name,len(raw),sha256(raw).hexdigest())
 print(name,len(raw),sha256(raw).hexdigest(),flush=True)
print('DENSE_NORM_PRODUCTION_ABSTRACT_PASS',amount,flush=True)
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        text=True,
        capture_output=True,
        timeout=180,
        env=dict(
            os.environ,
            JAX_PLATFORMS="cpu",
            XLA_FLAGS="--xla_force_host_platform_device_count=32",
        ),
    )
    assert result.returncode == 0, result.stdout + result.stderr
    print(result.stdout)

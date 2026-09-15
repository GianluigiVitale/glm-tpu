"""Independent FP64 CPU reference for GLM-5.2-FP8 layer 0 + layer-1 indexer inputs.

Written from the HF transformers GlmMoeDsa model definition and the GLM-5.2 config,
with vLLM/legacy conventions where HF and vLLM differ: interleaved indexer RoPE (config
indexer_rope_interleave=True; HF modeling uses half-split) and config rms_norm_eps for the q_a/kv_a
norms (HF modeling default 1e-6). Both engines under test use these conventions.
Shares no code with the greenfield engine.  All arithmetic is float64.
"""
import json, numpy as np, torch, ml_dtypes
from safetensors import safe_open
from safetensors.numpy import load_file

MODEL='/home/gianl/gcs-models/models/GLM-5.2-FP8'
SHARD=MODEL+'/model-00001-of-00141.safetensors'
CFG=json.load(open(MODEL+'/config.json'))
H=CFG['hidden_size']; ROPE_THETA=CFG['rope_parameters']['rope_theta']
RMS_EPS=CFG['rms_norm_eps']            # input/post layernorm (1e-5)
LORA_EPS=CFG['rms_norm_eps']           # vLLM/legacy convention (config 1e-5) for q_a/kv_a norms; HF modeling default is 1e-6.
                                        # NOT negligible here: residual mean-square ~2e-5, so eps is first-order (1e-5 -> 1e-6 shifts
                                        # the layer-1 query scale by ~+0.056%). Both conventions are adjudicated; see spec §21.5.
KNORM_EPS=1e-6                          # indexer k_norm LayerNorm eps (HF and vLLM)
TOK='/home/gianl/gcs-models/oracles/greenfield/glm52/short_context/8k/greenfield_short_context_oracle_8k_20260807T172307269147351Z/oracle/tokens.safetensors'

_f=None
def _handle():
    global _f
    if _f is None: _f=safe_open(SHARD, framework='pt')
    return _f

def weight(name):
    """Dequantized [out,in] float64 weight (fp8 e4m3 x block-128 scales, or bf16/f32)."""
    f=_handle(); w=f.get_tensor(name)
    if w.dtype==torch.float8_e4m3fn:
        s=f.get_tensor(name+'_scale_inv').to(torch.float64).numpy()
        w=w.to(torch.float64).numpy()
        bo,bi=CFG['quantization_config']['weight_block_size']
        so=np.repeat(np.repeat(s,bo,axis=0),bi,axis=1)[:w.shape[0],:w.shape[1]]
        return w*so
    return w.to(torch.float64).numpy()

def embed_rows(token_ids):
    f=_handle(); e=f.get_slice('model.embed_tokens.weight')
    out=np.empty((len(token_ids),H),dtype=np.float64)
    for i,t in enumerate(token_ids): out[i]=e[int(t):int(t)+1].to(torch.float64).numpy()[0]
    return out

def tokens():
    t=load_file(TOK); prompt=t['prompt_token_ids'].astype(np.int64); gen=t['generated_token_ids'].astype(np.int64)
    return np.concatenate([prompt, gen[:1]])   # positions 0..8155; position 8155 holds first generated token

def rmsnorm(x,w,eps):
    var=np.mean(x*x,axis=-1,keepdims=True); return w*(x/np.sqrt(var+eps))

def layernorm(x,w,b,eps):
    mu=x.mean(-1,keepdims=True); var=((x-mu)**2).mean(-1,keepdims=True); return (x-mu)/np.sqrt(var+eps)*w+b

def rope_angles(positions, dim):
    inv=1.0/(ROPE_THETA**(np.arange(0,dim,2,dtype=np.float64)/dim))   # [dim/2]
    ang=np.asarray(positions,dtype=np.float64)[:,None]*inv[None,:]          # [T, dim/2]
    return np.cos(ang), np.sin(ang)

def rope_interleaved(x, cos, sin):
    """x[..., T, dim] with pairs (x0,x1),(x2,x3)...; cos/sin [T, dim/2] broadcast over leading dims."""
    x1=x[...,0::2]; x2=x[...,1::2]
    out=np.empty_like(x); out[...,0::2]=x1*cos-x2*sin; out[...,1::2]=x2*cos+x1*sin
    return out

def rope_half(x, cos, sin):
    """HF rotate_half layout: first half / second half pairs; output layout [rot first half, rot second half]."""
    d=x.shape[-1]//2; x1=x[...,:d]; x2=x[...,d:]
    return np.concatenate([x1*cos-x2*sin, x2*cos+x1*sin],axis=-1)

def silu(x): return x/(1.0+np.exp(-x))

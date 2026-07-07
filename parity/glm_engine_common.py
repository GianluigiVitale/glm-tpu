"""Shared mini-GLM-5.2 config + synthetic-checkpoint machinery for parity tests.

Mirrors the DSV4 pattern (~/moe-tpu/parity/dsv4_engine_full_parity.py): generate
ONE set of logical fp32 weights, write them as a synthetic checkpoint in the
NATIVE GLM-5.2 key names (verified against the real model.safetensors.index.json),
and let the PRODUCTION vLLM/tpu-inference loader map + pack + shard them, while
the SAME logical weights load into the transformers `GlmMoeDsaForCausalLM`
reference. Structural knobs mirror the real config (config.json in
~/glm-tpu/configs/glm-5.2-fp8-config.json):

  - MLA: q_lora + kv_lora + separate nope/rope head dims, v_head_dim
  - MoE: sigmoid scoring, noaux_tc (n_group=topk_group=1), routed_scaling 2.5,
    norm_topk_prob, 1 shared expert, e_score_correction_bias
  - DSA indexer on `full` layers per the index_skip_topk_offset/freq formula
    (mini: index_topk >= any test T so DSA == dense, tie-free)
  - first_k_dense_replace dense MLPs, the rest MoE

FP8 mode (write_checkpoint(fp8=True)) quantizes every *_proj weight to
float8_e4m3fn with [BLK,BLK] block `weight_scale_inv` scales, exactly the real
checkpoint's quantization_config (fmt e4m3, dynamic activations, block 128 —
mini uses a smaller block so tiny dims still tile).
"""
from __future__ import annotations

import json
import os

import numpy as np

# ---- mini dims (tiny but real-ratio; kv_lora %128 for the mla.v2 kernel) ----
H = 256            # hidden_size
NH = 4             # num_attention_heads (== num_key_value_heads)
Q_LORA = 128
KV_LORA = 256
NOPE = 64          # qk_nope_head_dim
ROPE = 32          # qk_rope_head_dim
VHD = 64           # v_head_dim
IDX_HD = 64        # index_head_dim (> ROPE, like real 128 > 64)
IDX_NH = 4         # index_n_heads
IDX_TOPK = 128     # >= any test T -> DSA selects everything == dense
NEXP = 16          # n_routed_experts
TOPK = 8           # num_experts_per_tok (real value)
NSHARED = 1
MOE_INTER = 128
DENSE_INTER = 256
VOCAB = 512
MAX_POS = 512
EPS = 1e-5


def use_real_dims():
    """Switch the module to GLM-5.2's REAL per-layer dims (config.json) —
    the shape/padding class the mini aliases away (nope 192 != v 256,
    64 heads, q_lora 2048, kv_lora 512, 32x128 indexer, fp8 block 128).
    Kept small only where it does not change the shape class: n_experts 16
    (vs 256 — same GMM/topk math at top-8), vocab 4096, 3 layers typical."""
    g = globals()
    g.update(H=6144, NH=64, Q_LORA=2048, KV_LORA=512, NOPE=192, ROPE=64,
             VHD=256, IDX_HD=128, IDX_NH=32, IDX_TOPK=2048, NEXP=16, TOPK=8,
             NSHARED=1, MOE_INTER=2048, DENSE_INTER=12288, VOCAB=4096,
             MAX_POS=4096)


def make_mini_config(n_layers: int = 5,
                     first_k_dense: int = 1,
                     fp8: bool = False,
                     fp8_block: int = 64,
                     index_topk: int = IDX_TOPK) -> dict:
    """A config.json dict consumable by BOTH transformers (GlmMoeDsaConfig) and
    vLLM (deepseek_v2.py GlmMoeDsaForCausalLM). Field set mirrors the real
    config.json; only sizes shrink."""
    # the real schedule formula (HF configuration_glm_moe_dsa.py:142-146 ==
    # vLLM deepseek_v2.py:1023-1032): full iff max(i-offset+1,0) % freq == 0
    freq, offset = 4, 3
    indexer_types = [
        "full" if max(i - offset + 1, 0) % freq == 0 else "shared"
        for i in range(n_layers)
    ]
    cfg = {
        "architectures": ["GlmMoeDsaForCausalLM"],
        "model_type": "glm_moe_dsa",
        "attention_bias": False,
        "attention_dropout": 0.0,
        "dtype": "bfloat16",
        "eos_token_id": [1],
        "pad_token_id": 0,
        "ep_size": 1,
        "first_k_dense_replace": first_k_dense,
        "head_dim": NOPE,  # overridden to qk_rope by the HF config class
        "hidden_act": "silu",
        "hidden_size": H,
        "index_head_dim": IDX_HD,
        "index_n_heads": IDX_NH,
        "index_share_for_mtp_iteration": True,
        "index_skip_topk_offset": offset,
        "index_topk": index_topk,
        "index_topk_freq": freq,
        "index_topk_pattern": None,
        "indexer_rope_interleave": True,
        "indexer_types": indexer_types,
        "initializer_range": 0.02,
        "intermediate_size": DENSE_INTER,
        "kv_lora_rank": KV_LORA,
        "max_position_embeddings": MAX_POS,
        "mlp_layer_types": ["dense"] * first_k_dense +
                           ["sparse"] * (n_layers - first_k_dense),
        "moe_intermediate_size": MOE_INTER,
        "moe_layer_freq": 1,
        "moe_router_dtype": "float32",
        "n_group": 1,
        "n_routed_experts": NEXP,
        "n_shared_experts": NSHARED,
        "norm_topk_prob": True,
        "num_attention_heads": NH,
        "num_experts_per_tok": TOPK,
        "num_hidden_layers": n_layers,
        "num_key_value_heads": NH,
        "num_nextn_predict_layers": 0,
        "q_lora_rank": Q_LORA,
        "qk_head_dim": NOPE + ROPE,
        "qk_nope_head_dim": NOPE,
        "qk_rope_head_dim": ROPE,
        "rms_norm_eps": EPS,
        "rope_interleave": True,
        "rope_parameters": {"rope_theta": 8000000, "rope_type": "default"},
        "routed_scaling_factor": 2.5,
        "scoring_func": "sigmoid",
        "tie_word_embeddings": False,
        "topk_group": 1,
        "topk_method": "noaux_tc",
        "use_cache": True,
        "v_head_dim": VHD,
        "vocab_size": VOCAB,
    }
    if fp8:
        cfg["quantization_config"] = {
            "activation_scheme": "dynamic",
            "fmt": "e4m3",
            "quant_method": "fp8",
            "weight_block_size": [fp8_block, fp8_block],
        }
    return cfg


def _full_indexer_layers(cfg: dict) -> list[int]:
    return [i for i, t in enumerate(cfg["indexer_types"]) if t == "full"]


def gen_weights(seed: int, cfg: dict) -> dict:
    """One set of logical fp32 weights keyed by NATIVE checkpoint names
    (HF [out, in] Linear layout), matching the real index.json key set."""
    rng = np.random.default_rng(seed)
    L = cfg["num_hidden_layers"]

    def r(*shape, s=0.05):
        return (rng.standard_normal(shape) * s).astype(np.float32)

    def norm(*shape):
        return (1.0 + 0.1 * rng.standard_normal(shape)).astype(np.float32)

    w = {
        "model.embed_tokens.weight": r(VOCAB, H, s=0.02),
        "model.norm.weight": norm(H),
        "lm_head.weight": r(VOCAB, H, s=0.02),
    }
    full_idx = set(_full_indexer_layers(cfg))
    for i in range(L):
        p = f"model.layers.{i}."
        w[p + "input_layernorm.weight"] = norm(H)
        w[p + "post_attention_layernorm.weight"] = norm(H)
        # ---- MLA attention (native low-rank projections) ----
        a = p + "self_attn."
        w[a + "q_a_proj.weight"] = r(Q_LORA, H)
        w[a + "q_a_layernorm.weight"] = norm(Q_LORA)
        w[a + "q_b_proj.weight"] = r(NH * (NOPE + ROPE), Q_LORA)
        w[a + "kv_a_proj_with_mqa.weight"] = r(KV_LORA + ROPE, H)
        w[a + "kv_a_layernorm.weight"] = norm(KV_LORA)
        w[a + "kv_b_proj.weight"] = r(NH * (NOPE + VHD), KV_LORA)
        w[a + "o_proj.weight"] = r(H, NH * VHD)
        # ---- DSA indexer (only on `full` layers, like the real ckpt) ----
        if i in full_idx:
            x = a + "indexer."
            w[x + "wq_b.weight"] = r(IDX_NH * IDX_HD, Q_LORA)
            w[x + "wk.weight"] = r(IDX_HD, H)
            w[x + "k_norm.weight"] = norm(IDX_HD)
            w[x + "k_norm.bias"] = r(IDX_HD, s=0.02)
            w[x + "weights_proj.weight"] = r(IDX_NH, H)
        # ---- MLP ----
        if cfg["mlp_layer_types"][i] == "dense":
            m = p + "mlp."
            w[m + "gate_proj.weight"] = r(DENSE_INTER, H)
            w[m + "up_proj.weight"] = r(DENSE_INTER, H)
            w[m + "down_proj.weight"] = r(H, DENSE_INTER)
        else:
            m = p + "mlp."
            w[m + "gate.weight"] = r(NEXP, H, s=0.2)
            w[m + "gate.e_score_correction_bias"] = r(NEXP, s=0.5)
            for e in range(NEXP):
                pe = m + f"experts.{e}."
                w[pe + "gate_proj.weight"] = r(MOE_INTER, H)
                w[pe + "up_proj.weight"] = r(MOE_INTER, H)
                w[pe + "down_proj.weight"] = r(H, MOE_INTER)
            ps = m + "shared_experts."
            w[ps + "gate_proj.weight"] = r(MOE_INTER * NSHARED, H)
            w[ps + "up_proj.weight"] = r(MOE_INTER * NSHARED, H)
            w[ps + "down_proj.weight"] = r(H, MOE_INTER * NSHARED)
    return w


# Weights kept high-precision in the real FP8 checkpoint (modules_to_not_convert:
# every norm, mlp.gate + e_score_correction_bias, embed, lm_head, weights_proj).
_FP8_SKIP_SUBSTR = ("layernorm", "k_norm", ".gate.", "e_score_correction",
                    "embed_tokens", "lm_head", "model.norm", "weights_proj")


def _quantize_block_fp8(a: np.ndarray, blk: int):
    """Per-[blk,blk]-block e4m3 quantization -> (fp8 weight, fp32 scale_inv).

    scale_inv layout matches the real checkpoint: [ceil(out/blk), ceil(in/blk)],
    dequant = w.astype(f32) * scale_inv[block]. Amax-scaled per block."""
    import torch
    t = torch.from_numpy(a)
    out, inn = t.shape
    nbo, nbi = -(-out // blk), -(-inn // blk)
    fp8_max = torch.finfo(torch.float8_e4m3fn).max
    w_q = torch.empty(out, inn, dtype=torch.float8_e4m3fn)
    s = torch.empty(nbo, nbi, dtype=torch.float32)
    for bo in range(nbo):
        for bi in range(nbi):
            sl = t[bo * blk:(bo + 1) * blk, bi * blk:(bi + 1) * blk]
            amax = sl.abs().max().clamp(min=1e-12)
            scale = fp8_max / amax
            w_q[bo * blk:(bo + 1) * blk, bi * blk:(bi + 1) * blk] = \
                (sl * scale).clamp(-fp8_max, fp8_max).to(torch.float8_e4m3fn)
            s[bo, bi] = 1.0 / scale
    return w_q, s.numpy()


def write_checkpoint(d: str, w: dict, cfg: dict, fp8: bool = False,
                     roundtrip_fp8_block: int | None = None) -> None:
    """Write config.json + model.safetensors (bf16, or fp8+block scale_inv).

    roundtrip_fp8_block: for the bf16 TWIN of an fp8 engine checkpoint —
    quantize+dequantize every would-be-fp8 tensor at this block size so the
    reference sees the SAME effective weights as the fp8 engine (the parity
    diff then isolates the engine's dequant math, not quantization error).
    """
    import torch
    from safetensors.torch import save_file
    os.makedirs(d, exist_ok=True)
    blk = cfg.get("quantization_config", {}).get("weight_block_size",
                                                 [roundtrip_fp8_block or 64])[0]
    t = {}
    for k, v in w.items():
        is_proj = k.endswith(".weight") and v.ndim == 2 and \
            not any(s in k for s in _FP8_SKIP_SUBSTR)
        if fp8 and is_proj:
            w_q, s_inv = _quantize_block_fp8(v, blk)
            t[k] = w_q
            t[k + "_scale_inv"] = torch.from_numpy(s_inv)
        elif roundtrip_fp8_block and is_proj:
            w_q, s_inv = _quantize_block_fp8(v, roundtrip_fp8_block)
            deq = w_q.to(torch.float32).numpy()
            b = roundtrip_fp8_block
            for bo in range(s_inv.shape[0]):
                for bi in range(s_inv.shape[1]):
                    deq[bo * b:(bo + 1) * b, bi * b:(bi + 1) * b] *= \
                        s_inv[bo, bi]
            t[k] = torch.from_numpy(deq).to(torch.bfloat16)
        else:
            t[k] = torch.from_numpy(v).to(torch.bfloat16)
    save_file(t, os.path.join(d, "model.safetensors"))
    json.dump(cfg, open(os.path.join(d, "config.json"), "w"), indent=1)


def build_hf_reference(ckpt_dir: str, dtype):
    """transformers GlmMoeDsaForCausalLM from the same checkpoint (CPU).
    Requires a bf16 (non-fp8) checkpoint dir."""
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM
    cfg = AutoConfig.from_pretrained(ckpt_dir)
    model = AutoModelForCausalLM.from_pretrained(
        ckpt_dir, config=cfg, torch_dtype=dtype, attn_implementation="eager")
    model.eval()
    return model

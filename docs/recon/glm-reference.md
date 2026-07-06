# GLM-5.2 reference availability — recon summary

## (1) transformers — REFERENCE PRESENT LOCALLY, no fetch needed
- `~/vllm-env/bin/python`: **transformers 5.12.0** at `/home/gianl/vllm-env/lib/python3.12/site-packages/transformers/`
- `models/glm_moe_dsa/` **exists** with `modeling_glm_moe_dsa.py` (39,556 B), `configuration_glm_moe_dsa.py`, `modular_glm_moe_dsa.py`. All three **copied to `/home/gianl/glm-tpu/reference/`**.
- Key HF classes (`/home/gianl/glm-tpu/reference/modeling_glm_moe_dsa.py`): `GlmMoeDsaIndexer` :166, `.forward` :198 (pure bf16/fp32 DSA scoring math — skips Hadamard + fp8_index, documented :211-215; score = relu(q·k·scale) weighted per-head via `weights_proj·n_heads^-0.5`, returns int32 top-k indices); `GlmMoeDsaAttention` :347 (`skip_topk = indexer_types[layer_idx]=="shared"` :406, indexer=None on shared layers :407, cross-layer reuse via `prev_topk_indices` :353); `GlmMoeDsaTopkRouter` :510; MLA rope is `apply_rotary_pos_emb_interleave` :434/:302.
- **⚠ RoPE-interleave CONFLICT (parity landmine):** HF indexer applies **NON-interleaved** half-split RoPE (comment + call, modeling :238-241) and the HF config class never reads `indexer_rope_interleave`; but checkpoint config sets `"indexer_rope_interleave": true` and **vLLM honors it** (`is_neox_style=not indexer_rope_interleave` → interleaved, deepseek_v2.py:1007). One reference is wrong — resolve empirically in Stage 2 parity.

## (2) config.json — saved to `/home/gianl/glm-tpu/configs/glm-5.2-fp8-config.json` (HTTP 200, no token)
- **quantization_config** (:224-232): `"quant_method": "fp8"`, `"fmt": "e4m3"`, `"activation_scheme": "dynamic"`, `"weight_block_size": [128, 128]`, plus a ~540-entry `modules_to_not_convert` (all layernorms, all `mlp.gate` + `e_score_correction_bias`, `lm_head`, `model.embed_tokens`, `model.norm`).
- **indexer_types** (:26-105): 78 entries — `full,full,full`, then repeating `shared,shared,shared,full`; **21 full** (idx 0,1,2 then 6,10,…,74) / **57 shared**. Matches `index_topk_freq=4` (:23), `index_skip_topk_offset=3` (:21), `index_topk_pattern=null` (:24).
- **mlp_layer_types** (:110-188): 78 entries — 3× `dense` + 75× `sparse` (= `first_k_dense_replace=3` :14).
- **Fields NOT in the task summary** (verbatim): `first_k_dense_replace: 3`; `head_dim: 192` (HF config later overrides to qk_rope_head_dim=64, configuration :149-150); `qk_head_dim: 256`; `intermediate_size: 12288` (dense MLP); `moe_intermediate_size: 2048`; `n_group: 1`, `topk_group: 1`, `norm_topk_prob: true`, `moe_router_dtype: "float32"`; `vocab_size: 154880`; `max_position_embeddings: 1048576`; `rope_parameters: {"rope_theta": 8000000, "rope_type": "default"}` (**no YaRN** → no mscale correction); `rope_interleave: true` (main MLA); `index_share_for_mtp_iteration: true`; `num_key_value_heads: 64`; `rms_norm_eps: 1e-05`; `ep_size: 1`; `attention_bias: false`; `attention_dropout: 0.0`; `tie_word_embeddings: false`; `eos_token_id: [154820,154827,154829]`, `pad_token_id: 154820`; `dtype: "bfloat16"`; `transformers_version: "5.12.0"`.
- **Checkpoint-only modules revealed by `modules_to_not_convert`** (not in the HF modeling file, not handled anywhere in vLLM — grep `indexers_proj` over `/home/gianl/vllm-build/vllm/` returns nothing): **`self_attn.indexers_proj`** on exactly the full-indexer layers (0,1,2,6,…,74) **plus MTP layer 78** (e.g. config :258, :390, :594); **`indexer.k_norm.bias`** entries (LayerNorm with bias, HF uses `nn.LayerNorm` modeling :193 — consistent); MTP layer-78 modules `enorm` :365, `hnorm` :302, `eh_proj` :666, `shared_head.norm` :637. `indexers_proj` is an **unmodeled weight** — investigate before load (likely IndexShare-related projection used by zai's stack).

## (3) vLLM GPU reference (`/home/gianl/vllm-build`, version `0.1.dev1+ga30addc75`, commit a30addc75, 2026-06-12)
- Registry: `"GlmMoeDsaForCausalLM": ("deepseek_v2", "GlmMoeDsaForCausalLM")` — `vllm/model_executor/models/registry.py:124`; class is a bare subclass `GlmMoeDsaForCausalLM(DeepseekV2ForCausalLM)` — `vllm/model_executor/models/deepseek_v2.py:1750`.
- **DSA indexer forward — YES, usable reference math**:
  - `Indexer` class `deepseek_v2.py:603`; **`Indexer.forward` :678-742** — wq_b → split rope/nope, fused `wk_weights_proj` GEMM, k_norm, rope, per-token-group FP8 quant of q (ue8m0 scales, block 128, :728-735), `weights = weights_proj_out * q_scale * softmax_scale * n_head**-0.5` (:737-740), then `self.indexer_op(...)` :742.
  - `SparseAttnIndexer(CustomOp)` — `vllm/model_executor/layers/sparse_attn_indexer.py:407`; **`sparse_attn_indexer` :82** (full prefill/decode top-k logic incl. fp8 k-cache insert), `sparse_attn_indexer_fake` :376, `forward_native` :448, `forward_cuda` :465, `forward_xpu` :497, `forward_hip` :506.
  - GLM-specific wiring in `DeepseekV2MLAAttention.__init__` — `deepseek_v2.py:999-1075`: indexer rope `is_neox_style=not indexer_rope_interleave` :1007; IndexShare skip-topk derivation `_skip_topk = max(layer_id - index_skip_topk_offset + 1, 0) % index_topk_freq != 0` :1023-1032 (matches HF configuration_glm_moe_dsa.py:142-146); plumbed as `skip_topk=` into `MultiHeadLatentAttentionWrapper` :1061-1075.
  - Indexer KV backend: `vllm/v1/attention/backends/mla/indexer.py` — `DeepseekV32IndexerBackend` :118, metadata builder :232; fp8 k-cache (uint8, head_dim + head_dim/128×4 scale bytes) `deepseek_v2.py:655-661`.
  - FP8 wk load path (dequant-to-bf16 fusion) `_try_load_fp8_indexer_wk` `deepseek_v2.py:745`.
  - MTP: `glm_moe_dsa` remapped to `deepseek_mtp` — `vllm/config/speculative.py:301-313`; also listed in `vllm/transformers_utils/model_arch_config_convertor.py:262`.

## (4) SGLang
Not installed: `pip show sglang` → not found (system); `~/vllm-env/bin/python -c "import sglang"` → ModuleNotFoundError.

## Files written
- `/home/gianl/glm-tpu/configs/glm-5.2-fp8-config.json`
- `/home/gianl/glm-tpu/reference/{modeling,configuration,modular}_glm_moe_dsa.py`
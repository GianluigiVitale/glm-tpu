# ATTN_HEAD head-sharding scheme (PR-C) — summary from `~/moe-tpu/docs/12-prc-dense-weight-sharding.md`

## 1. What gets sharded, and how

Motivation: attention projections replicated ×32 cost ~8.1 GiB/chip — 72% of the replicated set — making head-sharding "unavoidable" for prefill HBM headroom (12-prc:15, 22-24).

Mesh: hybrid token-DP × head-TP. Axes `("data","attn_dp","attn_dp_expert","expert","model","dcp")`; `ATTN_HEAD = ('model','expert','dcp')`, `ATTN_DATA = ('data','attn_dp','attn_dp_expert')` (12-prc:29-38). Set via `additional_config.sharding.sharding_strategy.attn_dp_size`; `tensor_parallelism = TP // attn_dp` becomes `model` (12-prc:31-33; fork `tpu_inference/layers/common/sharding.py:243-251`). E.g. attn_dp=8 → model=4; EXPERT stays 32; tokens stay 8-way DP; KV cache `P(ATTN_DATA,'dcp')` is replicated ×model (cheap for prefill; a decode/long-context tradeoff) (12-prc:34-41).

Per-weight scheme (12-prc:64-74):
- `wqa`/`wkv` (shared latent), norms, compressor, indexer: **replicated** — the MLA latent is a single full-latent head, not head-decomposable (12-prc:52-53, 75-77).
- `wqb [n_heads*head_dim, q_lora]`: **column-parallel over heads** — 64 heads / model=4 → 16 heads/rank (12-prc:66-68).
- Each rank attends its local heads against the replicated latent kv (12-prc:69).
- Grouped o-proj `woa`/`wob`: shard the **GROUP dim** over `model`; each rank's partial `[T,hidden]` is reduced by an **explicit `jax.lax.psum` over the model axis** — required because the forward is a manual `jax.shard_map` (per-shard-local arrays), unlike V3's implicit nnx all-reduce (12-prc:56-59, 70-74).
- Load-time persistent sharding is mandatory: a gather-time `with_sharding_constraint` frees zero HBM; params must be re-`device_put` with a head `NamedSharding` at load, name-targeted (type-registration too broad) (12-prc:79-90). No value-preserving shortcut exists: the forward must consume weights sharded, else the all-gather re-inflates the peak (12-prc:99-103).

## 2. Env gates + fork code

- **`DSV4_SHARD_ATTN=1`** (default off = byte-identical replicated path; 12-prc:103, 106-109). Load-time: `vllm_model_wrapper._shard_ds_v4_attn_weights` (fork `tpu_inference/models/vllm/vllm_model_wrapper.py:239-300`, called at :520) — matches key suffixes `.wq_b./.wo_a./.wo_b.weight` owned by a `VllmDeepseekV4MLAAttention` (excludes the indexer's `wq_b`); stored `[in,out]` layout → `wq_b`/`wo_a` `P(None,AH)`, `wo_b` `P(AH,None)`.
- Forward: `deepseek_v4_attention.py` — `_head_sharded_extra_specs` (:1584-1600; gathered `[out,in]` layout → wqb/woa `P(AH,None)`, wob `P(None,AH)`, `attn_sink P(AH)`), dense `_paged_forward` psum at :1776, compressed `_compressed_prefill_jit` psum at :2500 (compressor+indexer stay replicated, :2429-2434).
- **`DSV4_ATTN_DP_SIZE`** — harness knob setting attn_dp (12-prc:94-95); requires `NEW_MODEL_DESIGN=1` (12-prc:42-43). Landed as fork commits `2f5c5476` (dense) + `60a21a29` (HCA/CSA) (12-prc:104, 117).

## 3. Constraints

- **Grouped-o-proj alignment**: `n_heads/model` must be a multiple of heads/group. DSV4: 64 heads, `o_groups=8` → 8 heads/group; holds for model∈{2,4} (16 or 32 heads/rank = 2 or 4 groups/rank) (12-prc:70-74, 148-149) and model=8 (1 group/rank); **model ≤ 8** because model=16/32 would cut inside an o-proj group (CLAUDE.md LATEST-10).
- Head divisibility: `n_heads % model == 0`.
- KV cache replicated ×model (12-prc:39-41); pure-DP (flag off) byte-identical (12-prc:109).
- Multi-host model>1 additionally needed the logits token-sharding fixes `1ad5a238`/`19e13a9a` (12-prc:136-144).

## 4. Re-verify for GLM (64 heads, qk_head_dim 256 = 192 nope + 64 rope, v_head_dim 256, hidden 6144)

- **Head count 64 = DSV4's** → head divisibility unchanged; model∈{2,4,8} splits 32/16/8 heads/rank.
- **o_groups=8 is DSV4-specific**: GLM MLA has a single `o_proj [n_heads*v_head_dim=16384, 6144]` — no `wo_a`/`wo_b` LoRA grouping. The group-alignment constraint vanishes, but the name-targeted suffix set (`.wo_a./.wo_b.`) and `_head_sharded_extra_specs` must be rewritten for a plain row-parallel `o_proj` `P(AH-on-heads-input, None)` + psum.
- **wq_b out-dim** = 64×(192+64)=16384 over `q_lora_rank=2048`; shard heads axis, 16384 % model fine.
- Per-head sink: GLM DSA has no `attn_sink` — drop `P(AH)` extra or re-verify absence.
- Indexer (32 heads, head_dim 128, topk 2048) replicated per the DSV4 precedent (12-prc:76-77) — check its HBM share is still small at GLM scale (78 layers).
- kv latent (`kv_lora_rank=512` + rope 64) stays replicated; verify HBM ledger: GLM hidden 6144 & 78 layers change the freed-GiB math (12-prc:13-24 numbers are DSV4-specific).
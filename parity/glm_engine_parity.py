"""Mini-GLM-5.2 in-engine forward parity vs the HF reference (v4 sub-cube).

Stage 1b (dense MLA, DSA indexer disabled): run the WHOLE mini GLM model
forward through the production vLLM/tpu-inference stack (VllmModelWrapper ->
vLLM DeepseekV2ForCausalLM(GlmMoeDsa) -> fork MLA wrapper -> mla.v2 Pallas
kernel + fused MoE GMM) on the local TPU sub-cube, and diff hidden states +
logits against transformers `GlmMoeDsaForCausalLM` (CPU, fp32 + bf16 controls).

The HF reference runs its DSA path, but the mini config sets index_topk >= T,
so top-k selects every token and DSA == dense — tie-free by construction.
The engine runs dense MLA with TPU_DISABLE_DSA_INDEXER=1 (Stage-1 config).

Identical weights on both sides via a synthetic native-key checkpoint
(glm_engine_common.write_checkpoint) loaded by each stack's production loader.

Run (from ~/glm-tpu). SINGLE-CHIP: this harness hand-builds the attention
metadata for ONE global sequence, which is only consistent with an unsharded
mesh — multi-chip validation goes through the real runner on the pod (the
model-axis shard_map would token-shard q/kv against global metadata: the
PR #2324 cross-shard class). Exits nonzero if the machine gate fails.
  TPU_CHIPS_PER_PROCESS_BOUNDS=1,1,1 TPU_PROCESS_BOUNDS=1,1,1 \
  TPU_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 \
  NEW_MODEL_DESIGN=1 MODEL_IMPL_TYPE=vllm TPU_DISABLE_DSA_INDEXER=1 \
  [DISABLE_WEIGHT_REQUANTIZATION=1] \
  ~/vllm-env/bin/python parity/glm_engine_parity.py [--fp8] [--layers 5] [--T 32] \
      [--two-step [--split-at N]]

--two-step (the DSV4 cached 2-step pattern, dsv4_engine_full_parity.py):
after the single full-prefill forward (kept as the reference), re-create
FRESH zeroed kv_caches and run the SAME wrapper/params twice — step 1
prefills tokens [0, split) (seq_lens=[split]), step 2 continues with
tokens [split, T) reading the paged latent KV cache step 1 wrote
(seq_lens=[T], query_start_loc=[0, T-split]). The cache persists because
VllmMLAAttention.forward writes vllm_model_wrapper_context.kv_caches[idx]
in place on the SAME list object the harness closes over. A REVERSED
block table exercises the paging indirection (GLM_IDENTITY_BT=1 disables).
Gate: concat(step1, step2) must match the full-prefill hidden states
within 1e-2 relative (bf16-exact expected modulo bucketing).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

import glm_engine_common as C


# ----------------------------------------------------------------------------
# Engine side
# ----------------------------------------------------------------------------
def run_engine(ckpt_dir: str, input_ids_np: np.ndarray,
               positions_np: np.ndarray, two_step: bool = False,
               split_at: int | None = None):
    import jax
    import torch
    import torchax
    from jax.sharding import Mesh
    from torchax.interop import jax_view, torch_view
    from vllm.config import set_current_vllm_config
    from vllm.distributed.parallel_state import (
        ensure_model_parallel_initialized, init_distributed_environment)
    from vllm.engine.arg_utils import EngineArgs
    from vllm.forward_context import set_forward_context

    from tpu_inference.distributed import jax_parallel_state
    from tpu_inference.layers.common.attention_metadata import \
        AttentionMetadata
    from tpu_inference.layers.common.sharding import MESH_AXIS_NAMES
    from tpu_inference.models.vllm.vllm_model_wrapper import VllmModelWrapper
    from tpu_inference.models.vllm.vllm_model_wrapper_context import \
        set_vllm_model_wrapper_context
    from tpu_inference.utils import device_array

    ea = EngineArgs(
        model=ckpt_dir, skip_tokenizer_init=True, trust_remote_code=True,
        max_model_len=64, max_num_batched_tokens=64, max_num_seqs=4,
        dtype="bfloat16", enforce_eager=True,
        additional_config={
            "sharding": {"sharding_strategy": {"enable_dp_attention": True}}})
    vc = ea.create_engine_config()
    tf = tempfile.mkstemp()[1]
    devs = np.array(jax.local_devices())
    # MESH_AXIS_NAMES = (data, attn_dp, attn_dp_expert, expert, model, dcp);
    # put the chips on the `model` axis (head-TP), dcp stays 1.
    mesh = Mesh(devs.reshape(1, 1, 1, 1, len(devs), 1), MESH_AXIS_NAMES)

    with set_current_vllm_config(vc):
        init_distributed_environment(1, 0, local_rank=0,
                                     distributed_init_method=f"file://{tf}",
                                     backend="gloo")
        ensure_model_parallel_initialized(1, 1)
        jax_parallel_state.init_pp_distributed_environment(
            ip="127.0.0.1", rank=0, world_size=1,
            device=jax.local_devices()[0], need_pp=False)
        wrapper = VllmModelWrapper(vllm_config=vc, rng=jax.random.PRNGKey(0),
                                   mesh=mesh)
        params, _ = wrapper.load_weights()

    vm = wrapper.model.vllm_model
    n_layers = len(vm.model.layers)

    # Per-layer + embedding capture hooks (forward-hook outputs).
    caps: dict[str, np.ndarray] = {}

    def to_np(t):
        try:
            return np.asarray(jax.device_get(jax_view(t))).astype(np.float32)
        except Exception:
            return t.detach().cpu().float().numpy()

    def _cap(name):
        def hook(mod, inp, out):
            try:
                if isinstance(out, tuple) and len(out) == 2 and \
                        out[1] is not None and out[0].shape == out[1].shape:
                    # vLLM decoder layers return (hidden_delta, residual);
                    # the transformers hidden state is their SUM.
                    caps[name] = to_np(out[0]) + to_np(out[1])
                else:
                    caps[name] = to_np(out[0] if isinstance(out, tuple)
                                       else out)
            except Exception as e:  # diagnostic only
                caps[name] = repr(e)
        return hook

    vm.model.embed_tokens.register_forward_hook(_cap("emb"))
    for i in range(n_layers):
        vm.model.layers[i].register_forward_hook(_cap(f"layer{i}"))
    vm.model.layers[0].self_attn.register_forward_hook(_cap("attn0"))
    vm.model.layers[0].mlp.register_forward_hook(_cap("mlp0"))
    if n_layers > 1:
        vm.model.layers[1].self_attn.register_forward_hook(_cap("attn1"))
        vm.model.layers[1].mlp.register_forward_hook(_cap("mlp1"))

    # MLA layers REQUIRE a paged latent KV cache (the mla.v2 kernel writes new
    # kv into it even on pure prefill). Build per-layer caches with the
    # kernel's own shape helper, page_size from the backend (1024 -> use a
    # small page for the mini: the kernel needs page_size % 128 == 0).
    from tpu_inference.kernels.mla.v2.kernel import get_kv_cache_shape
    import jax.numpy as jnp

    def align_to(x, m):
        return -(-x // m) * m

    PAGE = 128
    T = int(input_ids_np.shape[0])
    num_pages = max(2, -(-T // PAGE) + 1)
    lkv_al = align_to(C.KV_LORA, 128)
    r_al = align_to(C.ROPE, 128)
    cache_shape = get_kv_cache_shape(num_pages, PAGE, lkv_al + r_al,
                                     jnp.bfloat16)
    kv_caches = [device_array(mesh, np.zeros(cache_shape, np.float32))
                 .astype(jnp.bfloat16) for _ in range(n_layers)]
    mapping = {}
    for i in range(n_layers):
        # self_attn.mla_attn = VllmMultiHeadLatentAttentionWrapper;
        # the inner VllmMLAAttention (which owns layer_name) is one deeper.
        attn = vm.model.layers[i].self_attn.mla_attn
        inner = getattr(attn, "mla_attn", attn)
        mapping[inner.layer_name] = i

    input_ids = device_array(mesh, input_ids_np.astype(np.int32))
    input_positions = device_array(mesh, positions_np.astype(np.int32))
    block_table_np = np.arange(num_pages, dtype=np.int32)
    attn_metadata = AttentionMetadata(
        input_positions=input_positions,
        block_tables=device_array(mesh, block_table_np),
        seq_lens=device_array(mesh, np.array([T], np.int32)),
        query_start_loc=device_array(mesh, np.array([0, T], np.int32)),
        request_distribution=device_array(
            mesh, np.array([0, 0, 1], np.int32)),
        padded_num_reqs=1)

    from vllm.ir import enable_torch_wrap

    def _forward(ids_j, pos_j, am):
        with torchax.default_env(), enable_torch_wrap(False), \
                set_vllm_model_wrapper_context(
                    kv_caches=kv_caches, mesh=mesh,
                    layer_name_to_kvcache_index=mapping,
                    vllm_config=vc), set_forward_context(
                        attn_metadata=am, vllm_config=vc):
            out = torch.func.functional_call(
                wrapper.model, torch_view(params),
                kwargs={"input_ids": torch_view(ids_j),
                        "positions": torch_view(pos_j),
                        "intermediate_tensors": None, "inputs_embeds": None},
                tie_weights=False)
            return to_np(out)

    hidden = _forward(input_ids, input_positions, attn_metadata)

    # NaN localization: report health of every captured site, in graph order.
    print("  [probe] captured sites (nan%, |max|):")
    for k in sorted(caps):
        v = caps[k]
        if isinstance(v, np.ndarray):
            nanp = float(np.isnan(v).mean()) * 100
            amax = float(np.nanmax(np.abs(v))) if not np.isnan(v).all() \
                else float("nan")
            print(f"    {k:10s} shape={v.shape} nan={nanp:5.1f}% "
                  f"absmax={amax:.4f}")
        else:
            print(f"    {k:10s} {v}")

    # Logits: hidden @ lm_head with the loaded (possibly resharded) weight.
    # tpu-inference linear methods store weights [in, out] or [out, in]
    # depending on the method — disambiguate by the vocab axis.
    logits = None
    try:
        wt = to_np(vm.lm_head.weight)
        logits = hidden @ (wt.T if wt.shape[0] == C.VOCAB else wt)
    except Exception as e:
        print(f"  (logits capture skipped: {e!r})")

    # ---- cached 2-step (prefill + continuation decode chunk) ----
    # Mirrors ~/moe-tpu/parity/dsv4_engine_full_parity.py: step 1 prefills
    # [0, split_at), step 2 runs [split_at, T) reading the paged latent KV
    # cache written by step 1. VllmMLAAttention.forward writes
    # vllm_model_wrapper_context.kv_caches[idx] = new_kv_cache, i.e. item
    # assignment on the SAME list object _forward closes over — so the
    # update propagates between steps as long as we mutate `kv_caches` in
    # place (slice assignment below) and never rebind a fresh list into
    # the closure.
    hidden_two_step = None
    if two_step:
        if split_at is None:
            split_at = T // 2
        assert 0 < split_at < T, f"--split-at must be in (0, {T})"
        # Snapshot the full-prefill hook captures — the two extra forwards
        # would overwrite them (restored after, so the caller's per-layer
        # diffs still compare the reference full-prefill forward).
        caps_snapshot = {k: (v.copy() if isinstance(v, np.ndarray) else v)
                         for k, v in caps.items()}
        # FRESH zeroed caches, in place (same list object → closure sees
        # them, and step 2 sees step 1's writes).
        kv_caches[:] = [device_array(mesh, np.zeros(cache_shape, np.float32))
                        .astype(jnp.bfloat16) for _ in range(n_layers)]
        # Non-identity (reversed) block table to exercise the paging
        # indirection, exactly like DSV4; GLM_IDENTITY_BT=1 disables.
        if os.environ.get("GLM_IDENTITY_BT"):
            ts_block_table_np = np.arange(num_pages, dtype=np.int32)
        else:
            ts_block_table_np = np.array(
                [num_pages - 1 - i for i in range(num_pages)], np.int32)

        def _step_am(pos_slice_np, total_len_after):
            # seq_lens = the sequence length in the cache AFTER this step;
            # query_start_loc covers only the new tokens. distribution
            # [0,0,1] = 1 mixed sequence (mla.v2 kernel docstring:
            # sequences[0:i] decode-only, [i:j] chunked-prefill-only,
            # [j:k] mixed; k = total) — correct for both the fresh prefill
            # chunk and the continuation chunk.
            n_new = int(pos_slice_np.shape[0])
            return AttentionMetadata(
                input_positions=device_array(
                    mesh, pos_slice_np.astype(np.int32)),
                block_tables=device_array(mesh, ts_block_table_np),
                seq_lens=device_array(
                    mesh, np.array([total_len_after], np.int32)),
                query_start_loc=device_array(
                    mesh, np.array([0, n_new], np.int32)),
                request_distribution=device_array(
                    mesh, np.array([0, 0, 1], np.int32)),
                padded_num_reqs=1)

        outs = []
        for (a, b) in [(0, split_at), (split_at, T)]:
            ids_j = device_array(mesh, input_ids_np[a:b].astype(np.int32))
            pos_j = device_array(mesh, positions_np[a:b].astype(np.int32))
            # positions stay ABSOLUTE for both steps (RoPE + cache slots).
            outs.append(_forward(ids_j, pos_j, _step_am(positions_np[a:b],
                                                        b)))
        hidden_two_step = np.concatenate(outs, axis=0)
        bt_kind = "identity" if os.environ.get("GLM_IDENTITY_BT") \
            else "reversed"
        print(f"  [two-step] split_at={split_at} block_table={bt_kind} "
              f"step1={outs[0].shape} step2={outs[1].shape}")
        caps.clear()
        caps.update(caps_snapshot)

    return hidden, logits, caps, hidden_two_step


# ----------------------------------------------------------------------------
# HF reference side
# ----------------------------------------------------------------------------
def run_hf(ckpt_dir: str, input_ids_np, dtype):
    import torch
    model = C.build_hf_reference(ckpt_dir, dtype)
    ids = torch.from_numpy(input_ids_np.astype(np.int64))[None, :]
    with torch.no_grad():
        out = model(ids, use_cache=False, output_hidden_states=True)
    hs = [h[0].float().numpy() for h in out.hidden_states]
    logits = out.logits[0].float().numpy()
    return hs, logits


def _diff(name, a, b):
    if a is None or b is None or not isinstance(a, np.ndarray):
        print(f"  {name:12s}  MISSING ({type(a).__name__}/{type(b).__name__})")
        return None
    if a.shape != b.shape:
        print(f"  {name:12s}  SHAPE MISMATCH {a.shape} vs {b.shape}")
        return None
    d = np.abs(a - b)
    rel = d.max() / (np.abs(b).max() + 1e-9)
    print(f"  {name:12s}  max|Δ|={d.max():.6f}  mean|Δ|={d.mean():.6f} "
          f" rel={rel:.6f}  (|ref|max={np.abs(b).max():.3f})")
    return d.max()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, default=5)
    ap.add_argument("--T", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fp8", action="store_true",
                    help="write an FP8 block-quantized checkpoint for the "
                         "ENGINE (HF ref still loads the bf16 twin)")
    ap.add_argument("--real-dims", action="store_true",
                    help="GLM-5.2's real per-layer dims (nope 192/v 256/"
                         "64 heads/2048+512 lora/32x128 indexer, block 128)")
    ap.add_argument("--two-step", action="store_true",
                    help="ALSO run the cached 2-step path (prefill [0,split)"
                         " + continuation [split,T) through the paged KV "
                         "cache, reversed block table) and gate it against "
                         "the single full-prefill hidden states")
    ap.add_argument("--split-at", type=int, default=None,
                    help="two-step split point (default T//2)")
    ap.add_argument("--keep", action="store_true")
    args = ap.parse_args()
    if args.real_dims:
        C.use_real_dims()

    assert args.T <= C.IDX_TOPK, "index_topk must cover T (tie-free DSA)"
    if args.two_step:
        split = args.split_at if args.split_at is not None else args.T // 2
        assert 0 < split < args.T, \
            f"--split-at must be in (0, {args.T}), got {split}"

    fp8_block = 128 if args.real_dims else 64
    cfg = C.make_mini_config(n_layers=args.layers, fp8=args.fp8,
                             fp8_block=fp8_block)
    w = C.gen_weights(args.seed, cfg)
    rng = np.random.default_rng(100 + args.seed)
    input_ids = rng.integers(0, C.VOCAB, size=args.T).astype(np.int32)
    positions = np.arange(args.T).astype(np.int32)

    tmp = tempfile.mkdtemp(prefix="glm_mini_")
    eng_dir = os.path.join(tmp, "engine_ckpt")
    hf_dir = os.path.join(tmp, "hf_ckpt")
    C.write_checkpoint(eng_dir, w, cfg, fp8=args.fp8)
    if args.fp8:
        # HF twin: SAME effective weights (fp8 quant->dequant roundtrip) so
        # the diff isolates the engine's fp8 handling, not quantization error.
        blk = cfg["quantization_config"]["weight_block_size"][0]
        cfg_bf16 = C.make_mini_config(n_layers=args.layers, fp8=False)
        C.write_checkpoint(hf_dir, w, cfg_bf16, fp8=False,
                           roundtrip_fp8_block=blk)
    else:
        hf_dir = eng_dir
    print(f"[parity] checkpoint(s) at {tmp} (fp8={args.fp8})")

    # HF reference FIRST (CPU; fails fast if the mini config is malformed).
    import torch
    hs32, logits32 = run_hf(hf_dir, input_ids, torch.float32)
    hs16, logits16 = run_hf(hf_dir, input_ids, torch.bfloat16)
    print(f"[parity] HF reference done: {len(hs32)} hidden states, "
          f"logits {logits32.shape}")

    hidden_eng, logits_eng, caps, hidden_ts = run_engine(
        eng_dir, input_ids, positions, two_step=args.two_step,
        split_at=args.split_at)
    print(f"[parity] engine done: hidden {hidden_eng.shape}")

    # bf16 noise floor: |HF bf16 - HF fp32| per checkpoint site.
    print("\n== control: HF bf16 vs HF fp32 (the bf16 noise floor) ==")
    _diff("final_hs", hs16[-1], hs32[-1])
    _diff("logits", logits16, logits32)

    print("\n== engine (bf16 TPU) vs HF fp32 ==")
    # transformers hidden_states[i] = INPUT of layer i; [-1] = final (post-norm)
    _diff("emb", caps.get("emb"), hs32[0])
    for i in range(args.layers):
        _diff(f"layer{i}", caps.get(f"layer{i}"), hs32[i + 1])
    d_final = _diff("final_hs", hidden_eng, hs32[-1])
    d_logits = _diff("logits", logits_eng, logits32) \
        if logits_eng is not None else None

    print("\n== engine vs HF bf16 (like-for-like) ==")
    _diff("final_hs", hidden_eng, hs16[-1])
    t1_ok = None
    if logits_eng is not None:
        _diff("logits", logits_eng, logits16)
        # top-1 agreement on next-token prediction (the operative signal)
        t1_eng = logits_eng.argmax(-1)
        t1_32 = logits32.argmax(-1)
        t1_16 = logits16.argmax(-1)
        ref_t1 = float(np.mean(t1_16 == t1_32))
        eng_t1 = float(np.mean(t1_eng == t1_32))
        print(f"  top1 agree: eng-vs-fp32 {eng_t1:.3f}, "
              f"eng-vs-bf16 {np.mean(t1_eng == t1_16):.3f}, "
              f"bf16-vs-fp32 {ref_t1:.3f}")
        # the engine may not agree with fp32 less often than the bf16
        # reference itself does (allow 1 flip of slack at small T)
        t1_ok = eng_t1 >= ref_t1 - (1.0 / len(t1_eng))

    # two-step vs single full-prefill: same math through the paged cache
    # modulo bucketing → bf16-exact expected; gate at 1e-2 relative.
    rel_ts = None
    if args.two_step:
        print("\n== two-step (prefill+decode via the paged KV cache) vs "
              "single full-prefill ==")
        d_ts = np.abs(hidden_ts - hidden_eng).max()
        rel_ts = float(d_ts / (np.abs(hidden_eng).max() + 1e-9))
        exact_ts = bool(np.array_equal(hidden_ts, hidden_eng))
        print(f"  two_step      max|Δ|={d_ts:.6f}  rel={rel_ts:.6f}  "
              f"exact_bit_match={exact_ts}")

    # ---- machine gate (exit nonzero on failure) ----
    floor_hs = np.abs(hs16[-1] - hs32[-1]).max()
    floor_lg = np.abs(logits16 - logits32).max()
    d_hs = np.abs(hidden_eng - hs32[-1]).max()
    d_lg = np.abs(logits_eng - logits32).max() \
        if logits_eng is not None else None
    K = 1.5   # engine must sit within 1.5x the bf16 noise floor
    checks = {
        "no_nan": not np.isnan(hidden_eng).any(),
        f"final_hs {d_hs:.4f} <= {K}x floor {floor_hs:.4f}":
            d_hs <= K * floor_hs,
        f"logits {d_lg:.4f} <= {K}x floor {floor_lg:.4f}":
            (d_lg is not None and d_lg <= K * floor_lg),
        "top1": bool(t1_ok),
    }
    if args.two_step:
        checks[f"two_step rel {rel_ts:.6f} <= 0.01"] = rel_ts <= 1e-2
    failed = [k for k, v in checks.items() if not v]
    print(f"\n[GATE] {'PASS' if not failed else 'FAIL: ' + '; '.join(failed)}")

    if not args.keep:
        import shutil
        shutil.rmtree(tmp, ignore_errors=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

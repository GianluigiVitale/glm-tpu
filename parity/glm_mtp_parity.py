"""Stage-3 M1: MTP draft-forward parity vs the COMPOSED HF reference (CPU).

HF's ``modeling_glm_moe_dsa.py`` has NO MTP module (layer-78 weights are
ignored), so the reference is COMPOSED (docs/08-mtp-design.md §7 M1):

  reference = hand-transcribed MTP glue (embed -> mask-position-0 -> enorm |
              hnorm(prev_hidden) -> concat -> eh_proj, numpy fp32)
            + HF ``GlmMoeDsaDecoderLayer`` instantiated at the MTP layer index
              from a TWIN config (num_hidden_layers extended by 1; the MTP
              layer's indexer/mlp schedule derived by the SAME fall-through
              formulas the engine uses)
            + shared_head.norm -> shared (target) lm_head for logits.

  candidate = the production draft build: EngineArgs --speculative-config
              '{"method":"mtp",...}' -> vLLM config surgery (DeepSeekMTPModel)
              -> VllmModelWrapper(is_draft_model=True) with the G2 shared-
              weight map (build_mtp_shared_params off the loaded TARGET's
              state) -> the wrapper's REAL jitted ``draft_step_fun`` +
              ``compute_logits``.

CPU-only (JAX_PLATFORMS=cpu), mini dims, fp32 both sides. Attention runs
GLM_DSA_MODE=xla_ref with index_topk >= T, which is BIT-EXACT equal to dense
(the Stage-2a D0 gate) and pure-XLA (the mla.v2 Pallas kernel cannot run on
CPU); the MTP block's MLP is dense (first_k_dense > MTP layer id) for the
same reason — the MoE draft path is exercised by the separate LOAD AUDIT
below (full loader-name coverage incl. all experts; forward is TPU-only).

Also runs (exit nonzero if any gate fails):
  * M1 load audit — through the production loader, the ONLY draft params not
    loaded from the checkpoint are exactly {embed_tokens, shared_head.head}
    (then hard-error-shared from the target): "zero unexpected/missing params
    after sharing". Run twice: dense-MTP mini and MoE-MTP mini (expert keys).
  * V1 — the glm_moe_dsa load patch engages for the draft (indexer built on
    the full MTP layer; topk buffer constructed on CPU).
  * Byte-identity hash test — the TARGET forward with the MTP speculative
    config ATTACHED equals (sha256 of output bytes) a separate build with
    speculative_config=None: enabling MTP leaves the non-spec path unchanged.

Run (from ~/glm-tpu; PYTHONPATH selects the fork worktree under test):
  JAX_PLATFORMS=cpu NEW_MODEL_DESIGN=1 MODEL_IMPL_TYPE=vllm \
  GLM_DSA_MODE=xla_ref PYTHONPATH=$HOME/tpu-inference-mtp \
  ~/vllm-env/bin/python parity/glm_mtp_parity.py [--T 16] [--layers 2]
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys
import tempfile

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("NEW_MODEL_DESIGN", "1")
os.environ.setdefault("MODEL_IMPL_TYPE", "vllm")
os.environ.setdefault("GLM_DSA_MODE", "xla_ref")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

import glm_engine_common as C

_DIST_INITIALIZED = False


def _init_dist_once(vc):
    global _DIST_INITIALIZED
    if _DIST_INITIALIZED:
        return
    import jax
    import vllm.ir
    from vllm.config import set_current_vllm_config
    from vllm.distributed.parallel_state import (
        ensure_model_parallel_initialized, init_distributed_environment)

    from tpu_inference.distributed import jax_parallel_state
    # What the real worker does at init (worker_base.py): dispatch vLLM IR
    # ops directly (no torch custom-op wrap — required under torchax).
    vllm.ir.set_default_torch_wrap(
        vc.compilation_config.ir_enable_torch_wrap)
    tf = tempfile.mkstemp()[1]
    with set_current_vllm_config(vc):
        init_distributed_environment(1, 0, local_rank=0,
                                     distributed_init_method=f"file://{tf}",
                                     backend="gloo")
        ensure_model_parallel_initialized(1, 1)
        jax_parallel_state.init_pp_distributed_environment(
            ip="127.0.0.1", rank=0, world_size=1,
            device=jax.local_devices()[0], need_pp=False)
    _DIST_INITIALIZED = True


def _cpu_mesh():
    import jax
    from jax.sharding import Mesh

    from tpu_inference.layers.common.sharding import MESH_AXIS_NAMES
    devs = np.array(jax.devices("cpu")[:1])
    return Mesh(devs.reshape(1, 1, 1, 1, 1, 1), MESH_AXIS_NAMES)


def _engine_config(ckpt_dir: str, speculative: dict | None, T_cap: int):
    from vllm.engine.arg_utils import EngineArgs
    ea = EngineArgs(
        model=ckpt_dir, skip_tokenizer_init=True, trust_remote_code=True,
        max_model_len=T_cap, max_num_batched_tokens=T_cap, max_num_seqs=4,
        dtype="float32", enforce_eager=True,
        speculative_config=speculative,
        # The mini checkpoint is unquantized, so vLLM's DefaultModelLoader
        # weights-tracking would (correctly!) flag the draft's deliberately
        # absent embed/shared_head.head. The real GLM-5.2-FP8 is quantized
        # (tracking off by default); our load AUDIT below checks the same
        # invariant more precisely.
        model_loader_extra_config={"enable_weights_track": False},
        additional_config={
            "sharding": {"sharding_strategy": {"enable_dp_attention": True}}})
    return ea.create_engine_config()


def _load_wrapper(vc, mesh, is_draft: bool = False, shared_params=None):
    import jax
    from vllm.config import set_current_vllm_config

    from tpu_inference.models.vllm.vllm_model_wrapper import VllmModelWrapper
    with set_current_vllm_config(vc):
        wrapper = VllmModelWrapper(vllm_config=vc, rng=jax.random.PRNGKey(0),
                                   mesh=mesh, is_draft_model=is_draft)
        params, _ = wrapper.load_weights(shared_params=shared_params)
    return wrapper, params


def _to_np(x):
    import jax
    return np.asarray(jax.device_get(x)).astype(np.float32)


def _attn_metadata(mesh, positions_np: np.ndarray):
    from tpu_inference.layers.common.attention_metadata import \
        AttentionMetadata
    from tpu_inference.utils import device_array
    T = int(positions_np.shape[0])
    return AttentionMetadata(
        input_positions=device_array(mesh, positions_np.astype(np.int32)),
        block_tables=device_array(mesh, np.zeros(4, np.int32)),
        seq_lens=device_array(mesh, np.array([T], np.int32)),
        query_start_loc=device_array(mesh, np.array([0, T], np.int32)),
        request_distribution=device_array(mesh, np.array([0, 0, 1],
                                                         np.int32)),
        padded_num_reqs=1)


# ----------------------------------------------------------------------------
# Load audit (M0-style, on the REAL built model): the only draft params not
# fed by the checkpoint are exactly the two shared ones.
# ----------------------------------------------------------------------------
class _MTPLoadSpy:
    """Captures DeepSeekMTP.load_weights' returned loaded_params set."""

    def __init__(self):
        self.loaded: set[str] | None = None
        self._orig = None

    def __enter__(self):
        from vllm.model_executor.models import deepseek_mtp as mtp_mod
        self._mod = mtp_mod
        self._orig = mtp_mod.DeepSeekMTP.load_weights
        spy = self

        def wrapped(model_self, weights):
            out = spy._orig(model_self, weights)
            spy.loaded = set(out)
            return out

        mtp_mod.DeepSeekMTP.load_weights = wrapped
        return self

    def __exit__(self, *exc):
        self._mod.DeepSeekMTP.load_weights = self._orig
        return False


def _audit_draft_load(draft_wrapper, spy: _MTPLoadSpy, spec_layer: int):
    vm = draft_wrapper.model.vllm_model
    all_params = {n for n, _ in vm.named_parameters()}
    assert spy.loaded is not None, "DeepSeekMTP.load_weights never ran"
    # The absorbed MLA projections (W_UK_T / W_UV + scales) are DERIVED from
    # kv_b_proj by process_weights_after_loading, never present in any
    # checkpoint — exclude them from the "must come from the checkpoint or
    # the shared set" audit.
    _DERIVED_SUFFIXES = (".W_UK_T", ".W_UK_T_scale", ".W_UV", ".W_UV_scale")
    all_params = {
        n for n in all_params if not n.endswith(_DERIVED_SUFFIXES)
    }
    not_loaded = all_params - spy.loaded
    # vLLM's DeepseekV2MoE re-registers the GATE's e_score_correction_bias on
    # the RoutedExperts module (same tensor, second name): it is fed by the
    # gate's checkpoint load, which we assert.
    _ALIAS = ".experts.routed_experts.e_score_correction_bias"
    for name in sorted(n for n in not_loaded if n.endswith(_ALIAS)):
        gate_name = name.replace(_ALIAS, ".gate.e_score_correction_bias")
        assert gate_name in spy.loaded, (
            f"alias {name} present but its source {gate_name} was never "
            "loaded from the checkpoint")
        not_loaded.discard(name)
    expected_shared = {
        "model.embed_tokens.weight",
        f"model.layers.{spec_layer}.shared_head.head.weight",
    }
    assert not_loaded == expected_shared, (
        f"load audit: params without checkpoint tensors {sorted(not_loaded)} "
        f"!= the expected shared set {sorted(expected_shared)}")
    # V1: the glm_moe_dsa load patch engaged for the draft — the MTP layer is
    # a FULL indexer layer (indexer module built), and the is_v32 topk buffer
    # was constructed (on CPU — current_platform.device_type patched).
    mtp_layer = vm.model.layers[str(spec_layer)]
    assert mtp_layer.mtp_block.self_attn.mla_attn.indexer is not None, (
        "V1: draft MTP layer lost its indexer (glm_moe_dsa patch mis-keyed?)")
    assert mtp_layer.is_v32
    print(f"  [audit] {len(all_params)} draft params; checkpoint fed "
          f"{len(all_params) - len(expected_shared)}; shared-only set == "
          f"{sorted(expected_shared)}; V1 indexer/is_v32 OK")


# ----------------------------------------------------------------------------
# Composed HF reference
# ----------------------------------------------------------------------------
def _rms(x: np.ndarray, g: np.ndarray, eps: float) -> np.ndarray:
    x = x.astype(np.float32)
    return x / np.sqrt((x * x).mean(-1, keepdims=True) + eps) * g


def _twin_config(cfg: dict, spec_layer: int) -> dict:
    """The HF twin: the MTP layer materialized as decoder layer L, with its
    indexer/mlp schedule derived by the same fall-throughs the engine uses."""
    twin = dict(cfg)
    twin["num_hidden_layers"] = spec_layer + 1
    twin["num_nextn_predict_layers"] = 0
    twin["indexer_types"] = list(cfg["indexer_types"]) + [
        "full" if C.mtp_layer_indexer_is_full(cfg, spec_layer) else "shared"
    ]
    twin["mlp_layer_types"] = list(cfg["mlp_layer_types"]) + [
        "sparse" if spec_layer >= cfg["first_k_dense_replace"] else "dense"
    ]
    return twin


def _reference_draft_forward(twin_dir: str, w: dict, cfg: dict,
                             spec_layer: int, ids: np.ndarray,
                             prev_hidden: np.ndarray, positions: np.ndarray,
                             torch_dtype):
    """Composed reference: numpy glue + HF GlmMoeDsaDecoderLayer[L] (torch)."""
    import torch
    eps = cfg["rms_norm_eps"]
    p = f"model.layers.{spec_layer}."

    emb = w["model.embed_tokens.weight"][ids].copy()
    emb[positions == 0] = 0.0  # DeepSeekMTP masks inputs at position 0
    glue_in = np.concatenate([
        _rms(emb, w[p + "enorm.weight"], eps),
        _rms(prev_hidden, w[p + "hnorm.weight"], eps),
    ], axis=-1)
    h_in = glue_in @ w[p + "eh_proj.weight"].T  # HF Linear: [out, in]

    model = C.build_hf_reference(twin_dir, torch_dtype)
    layer = model.model.layers[spec_layer]
    rotary = model.model.rotary_emb
    h_t = torch.from_numpy(h_in.astype(np.float32)).to(torch_dtype)[None]
    pos_ids = torch.from_numpy(positions.astype(np.int64))[None]
    with torch.no_grad():
        cos_sin = rotary(h_t, pos_ids)
        hidden, _ = layer(h_t, attention_mask=None, position_ids=pos_ids,
                          position_embeddings=cos_sin, use_cache=False)
    hidden = hidden[0].float().numpy()

    normed = _rms(hidden, w[p + "shared_head.norm.weight"], eps)
    logits = normed @ w["lm_head.weight"].T
    return hidden, logits


# ----------------------------------------------------------------------------
# Engine candidate
# ----------------------------------------------------------------------------
def _engine_draft_forward(vc, mesh, target_params, spec_layer: int,
                          ids: np.ndarray, prev_hidden: np.ndarray,
                          positions: np.ndarray):
    import jax.numpy as jnp

    from tpu_inference.models.vllm.glm_mtp.shared_weights import \
        build_mtp_shared_params
    from tpu_inference.utils import device_array

    n_layers = vc.model_config.hf_config.num_hidden_layers
    shared = build_mtp_shared_params(target_params,
                                     num_hidden_layers=n_layers)
    with _MTPLoadSpy() as spy:
        draft_wrapper, draft_params = _load_wrapper(vc, mesh, is_draft=True,
                                                    shared_params=shared)
    _audit_draft_load(draft_wrapper, spy, spec_layer)

    vm = draft_wrapper.model.vllm_model
    inner = vm.model.layers[str(spec_layer)].mtp_block.self_attn.mla_attn
    inner = getattr(inner, "mla_attn", inner)
    mapping = ((inner.layer_name, 0), )

    draft_fn = draft_wrapper.jit_step_func()
    logits_fn = draft_wrapper.jit_compute_logits_func()

    kv_caches = [device_array(mesh, np.zeros((4, 8), np.float32))]
    ids_j = device_array(mesh, ids.astype(np.int32))
    prev_j = device_array(mesh, prev_hidden.astype(np.float32))
    attn_md = _attn_metadata(mesh, positions)

    _, hidden, residual, _ = draft_fn(draft_params, kv_caches, ids_j, prev_j,
                                      attn_md, mapping, spec_step_idx=0)
    # MTP contract: the recursion input (residual[0]) IS the pre-norm hidden.
    assert np.array_equal(_to_np(hidden), _to_np(residual[0])), (
        "draft_step_fun MTP branch: hidden_prenorm != hidden_states")
    logits = logits_fn(draft_params, hidden, None)
    return _to_np(hidden), _to_np(logits), jnp.asarray(hidden).dtype


def _target_forward_hash(vc, mesh, wrapper, params, ids: np.ndarray,
                         positions: np.ndarray) -> str:
    """One target forward (functional_call, xla_ref attention) -> sha256."""
    import torch
    import torchax
    from torchax.interop import torch_view
    from vllm.forward_context import set_forward_context
    from vllm.ir import enable_torch_wrap

    from tpu_inference.models.vllm.vllm_model_wrapper_context import \
        set_vllm_model_wrapper_context
    from tpu_inference.utils import device_array

    vm = wrapper.model.vllm_model
    mapping = {}
    for i in range(len(vm.model.layers)):
        attn = vm.model.layers[i].self_attn.mla_attn
        inner = getattr(attn, "mla_attn", attn)
        mapping[inner.layer_name] = i
    kv_caches = [
        device_array(mesh, np.zeros((4, 8), np.float32))
        for _ in range(len(vm.model.layers))
    ]
    ids_j = device_array(mesh, ids.astype(np.int32))
    pos_j = device_array(mesh, positions.astype(np.int32))
    attn_md = _attn_metadata(mesh, positions)
    with torchax.default_env(), enable_torch_wrap(False), \
            set_vllm_model_wrapper_context(
                kv_caches=kv_caches, mesh=mesh,
                layer_name_to_kvcache_index=mapping,
                vllm_config=vc), set_forward_context(attn_metadata=attn_md,
                                                     vllm_config=vc):
        out = torch.func.functional_call(
            wrapper.model, torch_view(params),
            kwargs={"input_ids": torch_view(ids_j),
                    "positions": torch_view(pos_j),
                    "intermediate_tensors": None, "inputs_embeds": None},
            tie_weights=False)
        out_np = _to_np(out)
    return hashlib.sha256(out_np.tobytes()).hexdigest(), out_np


def _diff(name, a, b):
    d = np.abs(a - b)
    rel = float(d.max() / (np.abs(b).max() + 1e-9))
    print(f"  {name:12s} max|Δ|={d.max():.7f} mean|Δ|={d.mean():.7f} "
          f"rel={rel:.7f} (|ref|max={np.abs(b).max():.3f})")
    return rel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--T", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--keep", action="store_true")
    args = ap.parse_args()

    import torch

    L = args.layers
    T = args.T
    T_cap = max(64, T + 32)
    tmp = tempfile.mkdtemp(prefix="glm_mtp_mini_")
    failures: list[str] = []

    # ---------------- forward-parity checkpoint (ALL-DENSE mlp: the MoE GMM
    # kernel is TPU-only; MoE-MTP is covered by the load audit below) -------
    cfg_fwd = C.make_mini_config(n_layers=L, first_k_dense=L + 2,
                                 num_mtp_layers=1)
    assert C.mtp_layer_indexer_is_full(cfg_fwd, L), (
        f"mini MTP layer {L} derives SHARED — pick --layers so the MTP "
        "layer is a FULL indexer layer (mirrors real layer 78)")
    assert cfg_fwd["index_topk"] >= T, "index_topk must cover T (== dense)"
    # The harness copy of the indexer-schedule formula must not drift from
    # the fork's authoritative derivation (docs/08 §1.2).
    from types import SimpleNamespace

    from tpu_inference.models.vllm.glm_mtp.draft_config import \
        mtp_layer_indexer_is_full as fork_indexer_is_full
    _cfg_ns = SimpleNamespace(**cfg_fwd)
    for _lid in range(L + 2):
        assert fork_indexer_is_full(_cfg_ns, _lid) == \
            C.mtp_layer_indexer_is_full(cfg_fwd, _lid), (
                f"indexer-schedule formula drift at layer {_lid}")
    w = C.gen_weights(args.seed, cfg_fwd)
    eng_dir = os.path.join(tmp, "engine_ckpt")
    twin_dir = os.path.join(tmp, "hf_twin_ckpt")
    C.write_checkpoint(eng_dir, w, cfg_fwd, dtype="float32")
    C.write_checkpoint(twin_dir, w, _twin_config(cfg_fwd, L),
                       dtype="float32")

    rng = np.random.default_rng(100 + args.seed)
    ids = rng.integers(0, C.VOCAB, size=T).astype(np.int32)
    positions = np.arange(T).astype(np.int32)  # includes the pos-0 mask row
    prev_hidden = (rng.standard_normal((T, C.H)) * 0.5).astype(np.float32)

    # ---------------- engine candidate ------------------------------------
    vc = _engine_config(eng_dir, {"method": "mtp",
                                  "num_speculative_tokens": 1}, T_cap)
    assert vc.speculative_config.draft_model_config.hf_config.architectures \
        == ["DeepSeekMTPModel"]
    _init_dist_once(vc)
    mesh = _cpu_mesh()
    print("[mtp-parity] loading TARGET (fp32, CPU)...")
    target_wrapper, target_params = _load_wrapper(vc, mesh)
    print("[mtp-parity] loading DRAFT through the production spec path...")
    eng_hidden, eng_logits, eng_dtype = _engine_draft_forward(
        vc, mesh, target_params, L, ids, prev_hidden, positions)
    print(f"[mtp-parity] engine draft forward done: hidden "
          f"{eng_hidden.shape} ({eng_dtype}), logits {eng_logits.shape}")

    # ---------------- composed HF reference (fp32 + bf16 control) ---------
    ref_hidden, ref_logits = _reference_draft_forward(
        twin_dir, w, cfg_fwd, L, ids, prev_hidden, positions, torch.float32)
    ref16_hidden, ref16_logits = _reference_draft_forward(
        twin_dir, w, cfg_fwd, L, ids, prev_hidden, positions, torch.bfloat16)

    print("\n== control: composed HF bf16 vs fp32 (the bf16 noise floor) ==")
    floor_hs = _diff("final_hs", ref16_hidden, ref_hidden)
    floor_lg = _diff("logits", ref16_logits, ref_logits)

    print("\n== M1: engine draft (fp32) vs composed HF reference (fp32) ==")
    rel_hs = _diff("final_hs", eng_hidden, ref_hidden)
    rel_lg = _diff("logits", eng_logits, ref_logits)
    t1_eng = eng_logits.argmax(-1)
    t1_ref = ref_logits.argmax(-1)
    t1_agree = float(np.mean(t1_eng == t1_ref))
    t1_floor = float(np.mean(ref16_logits.argmax(-1) == t1_ref))
    print(f"  top1 agree: eng-vs-fp32 {t1_agree:.3f} "
          f"(bf16 control {t1_floor:.3f})")

    # fp32-vs-fp32 with identical weights: only op-order rounding remains.
    FP32_REL = 1e-4
    if not np.isfinite(eng_hidden).all():
        failures.append("NaN/Inf in engine draft hidden states")
    if rel_hs > FP32_REL:
        failures.append(f"hidden rel {rel_hs:.2e} > {FP32_REL:.0e}")
    if rel_lg > FP32_REL:
        failures.append(f"logits rel {rel_lg:.2e} > {FP32_REL:.0e}")
    if rel_hs > floor_hs or rel_lg > floor_lg:
        failures.append("engine fp32 exceeds the bf16 control floor")
    if t1_agree < 1.0:
        failures.append(f"top-1 agreement {t1_agree:.3f} < 1.0")

    # ---------------- byte-identity hash test (spec off vs spec on) -------
    print("\n== hash test: TARGET forward, speculative on vs off ==")
    h_on, out_on = _target_forward_hash(vc, mesh, target_wrapper,
                                        target_params, ids, positions)
    vc_off = _engine_config(eng_dir, None, T_cap)
    assert vc_off.speculative_config is None
    target_off_wrapper, target_off_params = _load_wrapper(vc_off, mesh)
    h_off, _ = _target_forward_hash(vc_off, mesh, target_off_wrapper,
                                    target_off_params, ids, positions)
    print(f"  sha256(spec-on)  = {h_on}")
    print(f"  sha256(spec-off) = {h_off}")
    if h_on != h_off:
        failures.append("target forward NOT byte-identical with "
                        "speculative config attached")
    if not np.isfinite(out_on).all():
        failures.append("NaN/Inf in target hidden states")

    # ---------------- MoE-MTP load audit (no forward: GMM is TPU-only) ----
    print("\n== MoE-MTP draft LOAD audit (expert keys through the loader) ==")
    cfg_moe = C.make_mini_config(n_layers=L, first_k_dense=1,
                                 num_mtp_layers=1)
    moe_dir = os.path.join(tmp, "engine_moe_ckpt")
    C.write_checkpoint(moe_dir, C.gen_weights(args.seed + 1, cfg_moe),
                       cfg_moe, dtype="float32")
    vc_moe = _engine_config(moe_dir, {"method": "mtp",
                                      "num_speculative_tokens": 1}, T_cap)
    _, moe_target_params = _load_wrapper(vc_moe, mesh)
    from tpu_inference.models.vllm.glm_mtp.shared_weights import \
        build_mtp_shared_params
    shared = build_mtp_shared_params(moe_target_params, num_hidden_layers=L)
    with _MTPLoadSpy() as spy:
        moe_draft_wrapper, _ = _load_wrapper(vc_moe, mesh, is_draft=True,
                                             shared_params=shared)
    try:
        _audit_draft_load(moe_draft_wrapper, spy, L)
        moe_vm = moe_draft_wrapper.model.vllm_model
        n_expert_params = sum(1 for n, _ in moe_vm.named_parameters()
                              if ".experts." in n)
        assert n_expert_params > 0, "MoE draft has no expert params"
        print(f"  [audit] MoE draft OK ({n_expert_params} fused expert "
              "param(s) fed by layer-MTP checkpoint keys)")
    except AssertionError as e:
        failures.append(f"MoE load audit: {e}")

    print(f"\n[GATE] {'PASS' if not failures else 'FAIL: ' + '; '.join(failures)}")
    if not args.keep:
        import shutil
        shutil.rmtree(tmp, ignore_errors=True)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())

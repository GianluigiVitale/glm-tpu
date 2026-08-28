# Owner audit — PR 2 exact TorchAX/vLLM DSA bridge

This document covers PR 2 only. Official upstream is not authorized and has not
been mutated. Review PR 1 first because this patch is stacked on it.

## Immutable review range

- Parent: `650b5fccb890b5a872871af489b50fc4c584e8ad` (PR 1)
- Head: `d837832ab41f947ee9ff759e65ea8417ba1bd5c9`
- Branch: private `pr/glm-dsa-bridge-v3`
- Dependency: PR 1. PR 2 contains no model registration or Buildkite entry.
- Size: 11 files, +1,344/-34; four DCO-signed commits.

Inspect the exact patch:

```bash
git -C /home/gianl/tpu-inference-glm-baseline \
  diff --find-renames 650b5fccb890b5a872871af489b50fc4c584e8ad..d837832ab41f947ee9ff759e65ea8417ba1bd5c9
```

## Production-file checklist

| File | Change | Fail-closed boundary | Direct proof |
|---|---|---|---|
| `tpu_inference/layers/common/paged_cache.py` | Maps logical tokens to physical paged-cache slots and scatters latent/RoPE cache records | Invalid or padding tokens map to `-1`; no Python/host cache update | Fragmented slots, padding, and exact cache-byte tests |
| `tpu_inference/layers/vllm/custom_ops/sparse_attn_indexer.py` | Registers the vLLM V3.2 indexer out of tree and calls PR 1 scorer/cache primitives | Only FP8 + UE8M0 block-128 cache; DCP/PCP rejected; vLLM keeps metadata/buffer ownership | Constructor, unsupported-mode, decode, prefill, cache, and exact top-k tests |
| `tpu_inference/layers/vllm/backends/flash_attn_mla.py` | Consumes the shared selected positions in sparse MLA for prefill/decode and reconstructs complete TP query rows | Quantized/transposed main cache rejected; decode uses one live row; absent buffer rejected upstream | Real sparse kernel, TP2, phase-metadata, padding, and cache-write tests |
| `tpu_inference/layers/vllm/custom_ops/mla_attention.py` | Threads the model-owned top-k buffer into the backend | Requires `max_num_seqs=1`, DP1, continue-decode, and a non-null shared buffer | Missing-buffer regression proves sparse mode cannot silently become dense |
| `tpu_inference/runner/kv_cache_manager.py` | Sizes the physical padded V3.2 indexer cache from the pinned vLLM cache class | Only the recognized cache contract receives special sizing | Manager size/shape regression |
| `tpu_inference/layers/vllm/custom_ops/__init__.py` | Imports registration side effect | No model constructor monkeypatch | Import/constructor test |

The matching five test files exercise each production boundary directly:
`test_paged_cache.py`, `test_sparse_attn_indexer.py`,
`test_flash_attn_mla.py`, `test_mla_attention.py`, and
`test_kv_cache_manager.py`.

## Semantic ownership and review focus

- Current vLLM remains authoritative for `GlmMoeDsaForCausalLM`, request
  metadata, IndexShare scheduling, and shared-buffer lifetime.
- PR 2 adds no parallel scheduler, model fork, constructor monkeypatch,
  DSA-disable flag, host dispatch, or CPU fallback.
- Top-k is actually consumed by paged sparse MLA; it is not computed and then
  discarded into dense attention.
- TPU bucket rows beyond vLLM's allocation receive `-1` sentinels. Decode
  performs model work for one live row and zero-pads the seven TPU-only rows.
- TP2 reconstructs complete query rows before pairing them with global
  top-k/cache metadata. Request metadata, not row count, selects prefill versus
  decode.

CODEOWNER surfaces are model layers (`@kyuyeunk @lk-chen @jrplatin @gxd3`),
vLLM layers (`@kyuyeunk @vanbasten23 @gxd3 @jrplatin`), runner owners, and the
corresponding layer/runner test owners.

## Evidence to accept or reject

- Exact-head protected suite: 57/57 in 99.99 seconds; local/remote tag
  `upstream_glm_dsa_pr2_rebase_20260828T085714Z`; evidence-manifest SHA-256
  `7731e8397b58b323b1f68e82509005dce24dc121bd2e235c194d95200a0383de`.
- Coverage includes scorer/top-k, cache bytes, sparse MLA, shared buffer,
  metadata phase, missing-buffer rejection, and TP2 behavior on real TPU v4.
- The run used four local v4 devices and authenticated 8/8 pre/post fleet
  cleanup. All-file pre-commit and focused two-device CPU regressions pass.
- A broader run separately reproduces three pre-existing dense FP8-cache v4
  Mosaic failures; none exercises this sparse bridge contract.
- These are bridge correctness results, not full-model quality, HBM, serving
  latency, or cross-generation performance claims.

## Compatibility and rollback decision

- The path activates only when vLLM supplies the V3.2 sparse-indexer contract;
  other MLA callers retain the existing behavior.
- Unsupported layouts and parallel modes raise instead of falling back.
- Rollback is the four PR2 commits; no checkpoint/state migration exists.

Owner decision: approve the stacked PR 2, request a named correction, or ask
for a smaller bridge/backend split. PR 3 should not influence this decision.

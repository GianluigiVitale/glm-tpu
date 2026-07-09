# Adversarial review — DCP KV-cache persistence fix

**Scope:** branch `glm-5.2-v4-dcppersist`, commits `124a732f1` + `4211041d2`
(worktree `~/tpu-inference-dcppersist`). Base = `origin/glm-5.2-v4-next`
(merge-base `886eaceb4`). Independent reviewer; did not author the code.
Read-only. All checks run `JAX_PLATFORMS=cpu` — **TPU never touched.**

**Change under review:** `tpu_inference/models/common/model_loader.py` adds
`get_kv_cache_out_sharding(mesh, use_mla)` and uses it for the `out_shardings`
of the donated (`donate_argnums=1`) KV-cache buffer of the jitted step fns
`run_model` (L449-464) and `run_draft_model` (L466-479), replacing the base's
hardcoded `P(ATTN_DATA, None, ATTN_HEAD)` (base `model_loader.py` L363-365, used
by both step fns). New test `tests/layers/common/test_mla_dcp_cache_persistence.py`.

---

## VERDICT: SHIP. Fix is real, root-cause, PR-quality. No masking / reward-hacking.

The change makes the persisted (donated-round-trip) KV-cache layout of the
flax step fn **byte-exactly equal** to the layout `mla_attention` actually
returns under the DCP gate, so the donated buffer aliases with no cross-step
reshard — the true root cause of the multi-chunk-prefill stripe drop. The
"works" path (pure-TP, dcp=1, gate off, non-MLA) is provably byte-identical to
base. Every consumer inherits the corrected sharding through the same jitted
fns. All required tests pass. One **low-severity latent fragility** (Finding 3);
two design notes (Findings 6-7). Nothing blocks the fix.

Tests run (vllm-env, py3.12, JAX_PLATFORMS=cpu, 8-32 sim CPU devices):
- `test_mla_dcp_cache_persistence.py` — **10/10 pass**
- `test_mla_dcp.py` + `test_mla_head_sharded.py` — **48/48 pass**

---

## Finding 1 — [CONFIRMED / POSITIVE] Concern (1): out-sharding == the EXACT out_spec `mla_attention` produces under DCP

`get_kv_cache_out_sharding` returns, under `dcp_active and use_mla`,
`P(BATCH, CONTEXT)` (`model_loader.py:365-368`). I did not merely read that this
*should* equal `mla_attention`'s out_spec — I instrumented the real
`mla_attention` (monkeypatched `jax.shard_map` to capture the `out_specs`
tuple it constructs) on a real `model=8, dcp=4` mesh with `GLM_MLA_DCP=1`:

| branch | captured cache `out_spec` | captured cache `in_spec` |
|---|---|---|
| plain DCP (`attention_interface.py:764-765`) | `(('data','attn_dp','attn_dp_expert'), 'dcp')` | same |
| head-sharded + DCP (`attention_interface.py:718-719`) | `(('data','attn_dp','attn_dp_expert'), 'dcp')` | same |

`get_kv_cache_out_sharding(mesh, use_mla=True)` resolves to the identical tuple
`(('data','attn_dp','attn_dp_expert'), 'dcp')`, and `create_kv_caches(...,
use_mla=True)` (`runner/kv_cache.py:132-135`) allocates with the same spec.
**Byte-exact on every axis** — dim-0 = `BATCH`, dim-1 = `CONTEXT`(=`'dcp'`),
trailing dims replicated. `_cache_spec` in `mla_attention` is
`P(BATCH, CONTEXT) if _dcp else P(BATCH)` (`attention_interface.py:662-663`) and
both DCP out_spec sites (L719, L765) use `_cache_spec`; the transposed layout is
hard-refused under DCP (`attention_interface.py:652-655`), so there is no other
DCP cache out_spec. No axis mismatch → no reshard → no stripe drop.

*Refutation attempted:* Could the `dcp` axis land on a different array dim than
the shard_map's? No — both are dim-1 (`CONTEXT`='dcp'); I confirmed the
addressable shard of a `(16,8,8,4)` cache under `P(BATCH,CONTEXT)` is
`(16,2,8,4)` (dim-1 split 4-way, dim-2 `kv_packing` untouched) — the opposite of
the legacy spec which put `dcp` on dim-2 via `ATTN_HEAD=('model','expert','dcp')`.
Could a `with_sharding_constraint` after the shard_map change the returned
layout? The out_spec governs the value at the shard_map boundary and the step
fn's `out_shardings` governs the jit boundary; both are `P(BATCH,CONTEXT)`, so
the *persisted* layout is unambiguous regardless of any intra-step constraint.

## Finding 2 — [CONFIRMED / POSITIVE] Concern (2): untouched paths are byte-identical to base

The base forced `PartitionSpec(ShardingAxisName.ATTN_DATA, None,
ShardingAxisName.ATTN_HEAD)` for the kv-cache out-sharding of **both** step fns
(`git show origin/glm-5.2-v4-next:.../model_loader.py` L363-365, referenced at
L385 & L402). The fix's fall-through returns the *literally identical*
expression (`model_loader.py:374-377`). I enumerated the branch on a pure-TP
`model=32, dcp=1` mesh and resolved concrete tuples:

| gate | use_mla | returned spec | == base legacy? |
|---|---|---|---|
| off | True  | `(('data','attn_dp','attn_dp_expert'), None, ('model','expert','dcp'))` | ✅ |
| off | False | same | ✅ |
| on  | True  | same | ✅ (dcp_size==1 → gate inactive) |
| on  | False | same | ✅ (non-MLA never overridden) |

So the 32K/GSM8K path (pure TP, dcp=1) is byte-identical; and even with
`GLM_MLA_DCP=1` accidentally set on a dcp=1 mesh the `dcp_size > 1` guard
(`model_loader.py:362-364`) keeps it legacy. The step fn's `out_shardings`
tuple, `donate_argnums`, and `static_argnums` are otherwise unchanged, so the
traced program on the working path is unchanged. Commit `4211041d2` is the
*correct* half of this: it reverted the first commit's non-MLA DCP branch (which
had returned the `create_kv_caches` allocation `P(BATCH,CONTEXT,KV_CACHE_HEAD)`)
back to legacy, because non-MLA attention (`sharded_ragged_paged_attention`,
`attention_interface.py:368-369`) always returns `P(ATTN_DATA, None, ATTN_HEAD,
None, None)` regardless of dcp — the historical 3-tuple spec already matches it
(trailing `None`s are equivalent). Forcing the allocation spec there would have
*introduced* a per-step reshard for a non-MLA model under DCP. Good catch,
correctly resolved.

## Finding 3 — [PLAUSIBLE / LOW — latent fragility, not a bug for the GLM target] Concern (3): draft `use_mla` is read from the MAIN model config

`get_flax_model` derives `use_mla = getattr(vllm_config.model_config, "use_mla",
False)` (`model_loader.py:431`) **unconditionally**, and passes it to
`get_kv_cache_out_sharding` for *both* `run_model` and the draft's
`run_draft_model`. But `get_flax_model` is invoked with `is_draft_model=True`
for the MTP/EAGLE draft (`spec_decode/jax/eagle3.py:140`; `get_model` →
`get_flax_model` L701), and elsewhere in this very file the draft correctly
switches config: `_get_nnx_model` uses `vllm_config.speculative_config.
draft_model_config if is_draft_model else vllm_config.model_config`
(`model_loader.py:150-151`). The new code does **not** apply that switch.

*Failure scenario:* an MLA main model paired with a **non-MLA** draft under
`GLM_MLA_DCP=1` → `use_mla` reads the main's `True` → the draft step fn forces
`P(BATCH,CONTEXT)`, but the draft's non-MLA attention returns
`P(ATTN_DATA,None,ATTN_HEAD)` → reintroduces exactly the per-step reshard the
fix removes, on the draft cache.

*Refutation / why it is LOW:* GLM's MTP draft is dense-MTP sharing the main
MLA backbone (both MLA), so `draft_model_config.use_mla == model_config.use_mla
== True` and the value is coincidentally correct; `GLM_MLA_DCP` is an
MLA-specific gate, so the only configs where this branch fires are MLA anyway.
The draft under MLA+DCP *legitimately wants* `P(BATCH,CONTEXT)` (it runs the
same `mla_attention` under the same env+mesh gate — verified the gate logic is
env+`dcp_size>1` identical). So for the target workload the draft "inherits"
correctly. But the **source** of `use_mla` is wrong-by-construction. Recommend
`model_cfg = vllm_config.speculative_config.draft_model_config if is_draft_model
else vllm_config.model_config; use_mla = bool(getattr(model_cfg, "use_mla",
False))` to remove the latent trap. Non-blocking.

## Finding 4 — [CONFIRMED / POSITIVE] Concern (4): the donation is now a true no-reshard alias

With in-sharding == out-sharding == `P(BATCH,CONTEXT)` and `donate_argnums=1`,
XLA can alias the donated input buffer to the output. Empirical confirmation
from the test run:
- The **legacy negative control** (`test_legacy_out_sharding_is_the_reshard_bug`,
  forces `P(ATTN_DATA,None,ATTN_HEAD)`) emits JAX's
  `UserWarning: Some donated buffers were not usable: float32[2,2,8,4]` — the
  reshard between in- and out-sharding prevents donation aliasing (a copy is
  forced; on TPU this is the aliasing that drops a stripe).
- **Every fixed test** (matching in/out sharding) emits **no such warning** →
  the donated buffer is reused in place, no reshard. This is exactly the
  intended fix, corroborated on CPU (layout match ⇒ aliasing succeeds).

*Refutation attempted:* Could XLA still copy on TPU despite matching
NamedShardings (physical-layout/tiling mismatch)? On steps ≥2 the input *is* a
prior invocation of the same jit's output, so its physical layout is identical
to the output by construction — no copy. A one-time first-step relayout vs the
`create_kv_caches` allocation is possible but harmless (warmup already runs one
step and aligns layouts); it is not a cross-step correctness effect.

## Finding 5 — [CONFIRMED / POSITIVE] Fix is complete; no missed step-fn sites

The only place the flax step fn forced the kv-cache out-sharding was
`model_loader.py`. The vLLM-wrapper path passes `None` for kv_caches
out-sharding ("keep original sharding", `vllm_model_wrapper.py:864, 877`), so it
was never subject to this bug and needs no change. GLM is a registered flax
architecture → routes to `get_flax_model` (`model_loader.py:701`), the fixed
path, for both main and draft. Other `P(ATTN_DATA, None, ...)` hits are
unrelated (non-MLA attention out_spec L368; GDN/Mamba layers
`gdn_attention.py`; logits/hidden specs) and are not the donated KV-cache
carry. Warmup (`compilation_manager.py:490-503`) and the on-device decode
`while_loop` (`decode_loop.py:_decode_core`, `donate_argnames=("kv_caches",)`,
calls the same out-sharded `model_fn` at L167) both thread `self.kv_caches`
through the same jitted fns, so the while_loop carry sharding is consistent
`P(BATCH,CONTEXT)` init→body→out (init from `create_kv_caches`, body constrained
by the inner jit's out_shardings) — well-formed, and it is the layout the DCP
decode legitimately wants (per-shard local-slice kernel). eagle3
(`spec_decode/jax/eagle3.py:144,677,747`) likewise reuses `model.model_fn`.

## Finding 6 — [INFO / not a regression] head-sharded WITHOUT DCP keeps a pre-existing spec mismatch

`GLM_MLA_HEAD_SHARDED=1` with the DCP gate off makes `mla_attention` return the
cache as `P(BATCH)` (`attention_interface.py:746`), while the step fn (both base
and fix) forces the legacy `P(ATTN_DATA,None,ATTN_HEAD)`. This is a pre-existing
per-step reshard, **unchanged by this fix** (the fix only overrides MLA-under-
DCP). On pure-TP (head-sharded refuses any data/attn_dp axis,
`attention_interface.py:690-695`) `P(BATCH)` is replicated so values stay
consistent; head-sharded is documented as composed *with* DCP in production
(the composed branch is covered by Finding 1). Flagged only for completeness —
not introduced here.

## Finding 7 — [INFO / optional hardening] the flax path pins the spec; the vLLM path uses `None`

Because the flax step fn *pins* the kv-cache out-sharding rather than passing
`None` (as the vLLM path does), `get_kv_cache_out_sharding` must be kept
in lockstep with `mla_attention`'s `_cache_spec` forever; a future change to the
MLA out_spec that is not mirrored here silently reintroduces the reshard. The
commit documents this invariant thoroughly (and notes it deliberately omits
`mla_attention`'s test-only `_no_explicit_shardings` clause, which is always
`None` on served paths — verified: `_no_explicit_shardings` gates only the
explicit q/k/v/o sharding args, unused in GLM serving). Passing `None` (donate +
keep-original) would auto-track the out_spec and be strictly more robust;
the explicit-and-documented approach is legitimate PR-quality either way.
Optional.

---

## Positive assurance summary

- **Root cause is real and correctly located.** Base forced dim-1 replicated +
  `dcp` on dim-2 (`ATTN_HEAD` tuple ends in `'dcp'`) while the cache is born and
  returned dim-1-`dcp`-striped; the donation then aliased pre- into post-reshard
  and dropped a stripe across the multi-chunk-prefill scheduler boundary. The
  fix aligns the out-sharding to the real out_spec — not a symptom mask.
- **Concern 1 (match):** empirically byte-exact vs the real `mla_attention` DCP
  out_spec on both the plain and head-sharded+DCP branches, and vs
  `create_kv_caches`. ✅
- **Concern 2 (byte-identity):** all four gate×use_mla combinations on a pure-TP
  dcp=1 mesh return the exact base legacy spec; base used the identical hardcoded
  expression. ✅
- **Concern 3 (consumers):** execute_model reassign, decode `while_loop` carry,
  MTP `run_draft_model`, warmup, eagle3 all inherit the corrected sharding via
  the same jitted fns and all legitimately want `P(BATCH,CONTEXT)` under DCP.
  One low-severity latent fragility (Finding 3: draft `use_mla` source).
- **Concern 4 (donation):** empirically a true no-reshard alias under the fix
  (no "donated buffers not usable" warning; the legacy control emits it). ✅
- **Tests:** 10/10 persistence + 48/48 dcp/head-sharded pass on CPU.

**No reward-hacking / no masking detected.** The negative-control test honestly
documents that CPU cannot reproduce the TPU stripe-drop (XLA safely
replicate-then-repartitions on CPU), and pivots to a deterministic,
CPU-faithful discriminator (persisted sharding ≠ striped spec == the reshard).
The tests assert the real property (out-sharding == allocation == attention
out_spec, and two-step all-stripe value equality), not a tautology.

## Recommendation
Ship. Optionally address Finding 3 (draft `use_mla` from `draft_model_config`)
as a one-line hardening before/after merge; Findings 6-7 are informational.

# Adversarial review — DCP KV-cache out-sharding fix on GLM's REAL serving path

**Reviewer:** independent (did not author the change). READ-ONLY on the worktree; CPU-only (`JAX_PLATFORMS=cpu`); TPU never touched.
**Target:** branch `glm-5.2-v4-dcppersist`, worktree `~/tpu-inference-dcppersist`
- `c6da472e5` — DCP: fix KV-cache out-sharding on GLM's REAL path (VllmModelWrapper step fns)
- `ce51b85ef` — DCP: pin cache out-sharding on the MTP `_propose` jit + wiring-guard tests
- base / merge-base: `886eaceb4` (origin/glm-5.2-v4-next)

## VERDICT: **SHIP** (off-gate byte-identity on the working path is PROVEN)

The one risk that could sink the whole project — the fix silently perturbing the
validated serving path (GSM8K n=32 96.9%, sparse passkey 100% @32K, smoke
byte-identity) — **does not exist**. Off the DCP gate, the changed
`out_shardings[0]` is the Python object `None`, the exact literal the base fed,
on every donated step fn. I confirmed this three ways: by proof-reading the gate,
by capturing the out_shardings the REAL wrapper/proposer feed to `jax.jit`, and by
diffing the actual StableHLO lowering base-vs-fix. All identical off-gate;
provably different (non-vacuous) on-gate.

---

## What the change actually does

`model_loader.dcp_mla_cache_sharding(mesh, use_mla)` is the single shared gate:

```python
dcp_axes = (CONTEXT,) if not tuple else CONTEXT     # CONTEXT == 'dcp'
dcp_size = prod(mesh.shape.get(ax, 1) for ax in dcp_axes)
dcp_active = os.environ.get("GLM_MLA_DCP","0").lower() in ("1","true") and dcp_size > 1
if dcp_active and use_mla:
    return NamedSharding(mesh, P(BATCH, CONTEXT))    # striped
return None                                          # <-- every other case
```

It is wired into `out_shardings[0]` (the donated `kv_caches`) at exactly the
sites that persist the cache across `execute_model` steps:
- `vllm_model_wrapper.py` `jit_step_func`: `draft_step_fun`, `step_fun_no_options`,
  `step_fun_with_options` (all three build via the same `kv_cache_out_sharding`).
- `eagle3.py` `_propose`: the outer donated MTP jit, previously a static
  `@jax.jit(out_shardings=(None,None))` decorator, now lazily built in
  `_get_propose_fn` so the mesh is available to compute `cache_out`.

No other jit in either file donates the persistent KV cache — `compute_logits_func`,
`combine_hidden_states_func`, and `_prepare_inputs` all run with `kv_caches=None`.
So the fix is both complete (no persistence jit left un-pinned) and narrow.

---

## Attack 1 — Is off-gate PROVABLY `None`?  YES

`dcp_mla_cache_sharding` returns `None` unless **all three** hold: `GLM_MLA_DCP ∈
{1,true}` (case-insensitive), a real dcp axis (`dcp_size > 1`), and `use_mla`.
Any GLM run that does not opt into the env flag is `None`; and even with the flag
set, a pure-TP / `dcp=1` mesh or a non-MLA model is still `None`.

Executed (`offgate_proxy.py`, 32 simulated CPU devices, `NEW_MODEL_DESIGN=1` so
`ShardingAxisName == ShardingAxisNameBase`, `CONTEXT=='dcp'`, matching GLM prod):

| case | result |
|---|---|
| gate unset, MLA, TP mesh (model=32,dcp=1) | `is None` ✓ |
| gate unset, MLA, dcp=2 mesh | `is None` ✓ |
| gate unset, non-MLA | `is None` ✓ |
| gate ∈ {`0`,`false`,`FALSE`,`no`,``,`2`}, MLA, dcp=2 | `is None` ✓ |
| gate ∈ {`1`,`true`,`TRUE`,`True`}, MLA, **dcp=1** (pure TP) | `is None` ✓ |
| gate ∈ {`1`,`true`,`TRUE`,`True`}, **non-MLA**, dcp=2 | `is None` ✓ |
| gate=`1`, MLA, dcp=2 | non-None, `spec == P(BATCH,CONTEXT)` ✓ (non-vacuous) |

Checked with `is None`, not truthiness. **Consequence for the working path:** the
validated runs came through with `out_shardings=None`; whether they had
`GLM_MLA_DCP` unset *or* set on a `dcp=1` mesh, the fix reproduces `None`. There is
**no served, non-DCP GLM configuration in which a step fn now receives a non-None
cache out-sharding.** The only configurations that change are exactly
`{gate on} ∧ {dcp>1} ∧ {MLA}` — the intended target.

## Attack 2 — CPU trace/jaxpr comparison, base vs fix, gate-off. IDENTICAL

Built a donating, 4-output, `donate_argnums=(0,)` step fn mirroring the wrapper
(`out_shardings=(cache_out, P(ATTN_DATA,None), None, None)`, striped-`shard_map`
body) and lowered it on the pure-TP mesh (model=32, dcp=1):

- **off-gate, model=32/dcp=1:** `step.lower(...).as_text()` for `cache_out=None`
  (base) **==** `cache_out=dcp_mla_cache_sharding(...)` (fix, which *is* `None`).
  Byte-identical StableHLO.
- **off-gate, dcp=2 (flag unset):** also byte-identical.
- **on-gate, dcp=2:** base(`None`) **!=** fix(striped) — the lowering differs, so
  the pin is real and the discriminator is not vacuous.

This is the cleanest possible proof: off-gate the two arg tuples handed to
`jax.jit` are the same object (`None`), and jit lowering is a pure function of
`(fn, static args, in/out shardings, donation)`, so the compiled program cannot
differ.

## Attack 3 — REAL wiring (not a proxy) via a `jax.jit` spy

`VllmModelWrapper.__new__` + stub, run `jit_step_func` under a spy that records
every `out_shardings` tuple:
- **off-gate (model=32/dcp=1, MLA):** 3 step fns wired; `out_shardings[0] is None`
  on **all three** (== base); `[1]==P(ATTN_DATA,None)`, `[2]==None`, `[3]==None`
  unchanged from the base literals.
- **on-gate dcp=2 MLA:** `[0].spec == P(BATCH,CONTEXT)` on all three.
- **on-gate non-MLA:** `[0] is None` (== base). **on-gate pure-TP dcp=1:** `[0] is
  None` (== base).

Same spy on `Eagle3Proposer._get_propose_fn`:
- **off-gate:** `out_shardings == (None, None)` — identical to the historical
  decorator; `donate_argnames==('kv_caches',)`; `static_argnames == {num_speculative
  _tokens, layer_name_to_kvcache_index}`; both `post_spmd_conservative` compiler
  options preserved.
- **on-gate dcp=2:** `out_shardings[0].spec == P(BATCH,CONTEXT)`, `[1] is None`.

## Attack 4 — eagle3 `_propose` → `_propose_impl` / `_get_propose_fn`

Concern: does the lazy build preserve exact prior behavior when spec/DCP is off
(MTP is used in Stage-3)?

- **Body byte-identical:** `diff` of the base `_propose` body vs the new
  `_propose_impl` body (post-signature) is empty (94 lines each). Only the
  decorator was replaced by the lazy builder and the function renamed.
- **Signature identical:** same 8 params in the same order. Old
  `static_argnums=(0,7,8)` = `(self, num_speculative_tokens,
  layer_name_to_kvcache_index)`; new bound-method jit drops `self` (bound, not an
  arg) and uses `static_argnames=("num_speculative_tokens",
  "layer_name_to_kvcache_index")` — the exact same two args, and `propose()` passes
  both by keyword.
- **Semantics:** old = self static arg (captured as a compile-time constant,
  retraced per instance); new = self captured in the bound method closure, jit
  cached per instance on `self._propose_jit`. Same jaxpr, same constants → same
  compiled program off-gate (out_shardings `(None,None)`). Off-gate byte-identical
  to the historical decorator; on-gate pins `[0]` striped.
- **No dangling refs:** no live code references the removed `_propose`; `propose()`
  routes through `_get_propose_fn()`. Only stale docstring/comment mentions remain
  (harmless — see nit).

## Attack 5 — did the 3 index-share tests get weakened?  NO

The three tests changed exactly one token — the eager call target
`Eagle3Proposer._propose.__wrapped__(...)` → `Eagle3Proposer._propose_impl(...)`.
`jax.jit` sets `__wrapped__` to the raw function via `functools.wraps`; the new
`_propose_impl` *is* that same raw function. Same positional args, same
`num_speculative_tokens=3/4`, same `layer_name_to_kvcache_index=tuple()`. Every
assertion below the call line is untouched:
- `test_propose_reseeds_last_token_rows_and_collapses`: still asserts step-0
  `shared_topk_indices is None`, exactly-3 records, collapsed `spec_step_idx==1`,
  and row-exact reseed `== emit[last_token_indices]`.
- `test_propose_dense_loop_identical_to_historical`: still asserts
  `spec_step_idx==[0,1,2]` and `extra_keys==[]` (no seed kwarg on the dense path).
- `test_propose_traces_two_draft_signatures`: still `make_jaxpr`s the impl and
  asserts the `[(0,F),(1,T),(1,T),(1,T)]` two-trace signature.

Not weakened.

---

## Tests executed (worktree, `JAX_PLATFORMS=cpu`, `PYTHONPATH` pinned to worktree)

> Note: the installed `tpu_inference` resolves to `~/tpu-inference` (main
> checkout), so all runs pin `PYTHONPATH=/home/gianl/tpu-inference-dcppersist`.

| suite | result |
|---|---|
| `tests/layers/common/test_mla_dcp_cache_persistence.py` | **19 passed** (incl. both new wiring-guard tests) |
| `tests/layers/vllm/test_glm_dsa_mtp_index_share.py` | **14 passed** |
| `tests/layers/vllm/test_mla_attention.py` + `test_glm_dsa_indexer.py` | **40 passed** |
| `test_glm_dsa_sparse_prefill.py` + `test_glm_dsa_pallas_decode.py` + `test_dsv4_csa_nwin0.py` | **41 passed** |
| `offgate_proxy.py` (this review) | **38/38 checks** |
| `tests/spec_decode/test_eagle3.py` | 8 failed — **pre-existing, unrelated** |

The `test_eagle3.py` failures are `OSError`/HTTP `401 Unauthorized` raised inside
`_create_proposer` (`tests/spec_decode/test_eagle3.py:28`, `hf_hub_download` /
`snapshot_download`) during fixture setup — a gated-model download with no HF token
in this sandbox. It fails *before* any `_get_propose_fn`/`_propose_impl` runs, and
neither commit touches `test_eagle3.py`. Also affects `test_prepare_inputs` and
`test_update_inputs_for_loop_speculation_mrope`, which never call propose. Matches
the commit-message claim ("identical on base").

---

## Non-blocking nits (no ship impact)

1. **Env parsing is not stripped:** `GLM_MLA_DCP="1 "` (trailing space) `.lower()`
   is not in `{"1","true"}` → gate reads as OFF. Fail-safe (defaults to the
   historical `None`), so no regression risk, but a surprising foot-gun for whoever
   turns DCP on. A `.strip()` would harden it.
2. **Stale doc references:** `glm_mtp/index_share.py`, `glm_mtp/__init__.py`, and a
   `test_glm_dsa_mtp_index_share.py` docstring still say `Eagle3Proposer._propose`;
   the symbol is now `_propose_impl`. Documentation-only.
3. **On-gate correctness (that `P(BATCH,CONTEXT)` is the *right* pin matching
   `mla_attention`'s `_cache_spec`) is a TPU-layout property** outside this review's
   off-gate scope; it is covered by the persistence test's `shard_map` proxy and the
   prior correctness review (`dcp-persist-fix-correctness.md`). This review certifies
   the regression axis (off-gate byte-identity), which is clean.

## Bottom line

Off-gate the working path is byte-identical to base — proven by gate proof-reading,
real-wiring capture on all three wrapper step fns + the MTP proposer, and a
StableHLO lowering diff on the pure-TP (model=32/dcp=1) mesh. The eagle3 refactor
preserves the historical `(None,None)` decorator behavior exactly (body + wiring +
static-arg mapping), and the 3 retargeted tests keep every assertion. **SHIP.**

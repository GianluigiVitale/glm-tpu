# On-gate adversarial review — DCP vLLM step-fn KV-cache out-sharding fix

**Reviewer:** independent (did not write the code). Read-only; no TPU touched (`JAX_PLATFORMS=cpu` for the two test runs).
**Target:** branch `glm-5.2-v4-dcppersist`, commits `c6da472e5` + `ce51b85ef` (worktree `~/tpu-inference-dcppersist`).
**Base:** `origin/glm-5.2-v4-next`.
**Question:** does pinning the vLLM step-fn cache `out_shardings` actually fix GLM's dcp=2 multi-chunk-prefill "sees 1/dcp of context" drop — on GLM's REAL serving path, targeting the measured one-stripe signature?

---

## Verdict

**The fix is correct, minimal, on GLM's real serving path, and targets exactly the measured failure signature. SHIP the code — but do NOT trust any passkey number until a post-fix on-pod cache-dump MATCH is observed first.** The four on-gate checks pass; the code is byte-identical off-gate on both paths. The one hard limitation is intrinsic and honestly disclosed by the authors: the CPU cannot exhibit or refute the metal stripe-drop (on CPU the pre-fix `None` already yields correct values), so the CPU suite proves the spec is *pinned*, not that the *drop is gone*. The sole proof is a pod re-run of the 2-chunk cache-dump probe requiring **DIFFER → MATCH** before the passkey. Three residual items (one blocking the falsifier itself) are listed in §6.

No masking / reward-hacking found. The authors' own commit messages disclose the CPU limitation, that the value-persistence test was vacuous w.r.t. the wiring (which is why the MED-2 wiring guards were added), and that the MTP `_propose` fix is not exercised by the immediate passkey. Those disclosures check out against the code.

---

## 1. Target spec — is `P(BATCH, CONTEXT)` EXACTLY what `mla_attention` returns as the cache out_spec ON THE vLLM PATH under the DCP gate?

**YES, on the exact path the passkey runs.** Traced end to end:

- The dcp=2 128K passkey (and the 5d cache-dump debug that pinned the bug) runs with **`TPU_DISABLE_DSA_INDEXER=1`** (`docs/11-pod-runbook.md:490`), i.e. **dense MLA** — so the live attention is `layers/vllm/backends/flash_attn_mla.py:205`, not the DSA custom op.
- `flash_attn_mla.py:205-223` calls `attention_interface.mla_attention(...)` passing **`query_nth_sharding=None, query_tnh_sharding=None, keyvalue_skh_sharding=None, attn_o_nth_sharding=None`** (lines 215-218). Hence in `attention_interface.py:620-624` `_no_explicit_shardings = True`.
- The DSA path (`layers/vllm/custom_ops/mla_attention.py:1211`, the `_dense_fallback_branch`) also calls `attention_interface.mla_attention(...)` with **no** sharding kwargs (lines 1211-1221) → they default to `None` → `_no_explicit_shardings = True` there too.
- With `_no_explicit_shardings=True`, `GLM_MLA_DCP=1`, and a real `dcp` axis (`_dcp_size>1`), `attention_interface.py:650-651` sets `_dcp=True`, and `:662-663`:
  ```
  _cache_spec = P(ShardingAxisName.BATCH, ShardingAxisName.CONTEXT) if _dcp else P(ShardingAxisName.BATCH)
  ```
  The `shard_map` `out_specs[0]` (the cache) is `_cache_spec` in every served branch: default `:764-765`, and head-sharded+dcp `:718-719`. So the cache the vLLM forward returns is striped **`P(BATCH, CONTEXT)`**.
- The fix pins the step-fn `out_shardings[0]` to `model_loader.dcp_mla_cache_sharding(mesh, use_mla)`, which returns **`NamedSharding(mesh, P(BATCH, CONTEXT))`** under the same gate (`model_loader.py:356-362`). **Byte-for-byte the same spec.**

**Independent confirmation of the donated INPUT layout:** `runner/kv_cache.py:132-135` allocates the MLA cache `P(BATCH, CONTEXT)` (for `use_mla`, path-agnostic). So donated-input == pinned-output == attention-out_spec == allocation: **all four agree on `P(BATCH, CONTEXT)`.** This is the exact condition for a reshard-free donation round-trip.

**Gate-alignment refutation attempt (the subtle divergence the task flagged).** `dcp_mla_cache_sharding` mirrors `mla_attention`'s gate *except* it omits the `_no_explicit_shardings` clause. I tried to break this on a served path:
- vLLM dense backend (`flash_attn_mla.py`): shardings hard-coded `None` → `_no_explicit_shardings=True`. Aligned. ✔
- vLLM DSA dense-fallback (`custom_ops/mla_attention.py:1211`): no sharding kwargs → `None` → aligned. ✔
- The only caller that passes non-`None` shardings is the **FLAX** path `models/jax/deepseek_v3.py:702-714` (`query_nth_sharding=self.query_nth`, a `PartitionSpec`, not `None`). GLM does **not** use the flax path (see §2), so it does not affect GLM. But note this makes the commits' NOTE ("explicit q/k/v/o shardings are a test-only hook, always `None` on a served path") **inaccurate for the flax `deepseek_v3` served path** — see finding R2 (§6). It does not change the GLM verdict.

**Conclusion:** on GLM's live passkey path the pinned spec is exactly the attention's cache out_spec. A mismatch on any axis (which would re-introduce the reshard) does **not** occur.

---

## 2. The 4-output donating step — is the cache genuinely `output[0]`, and are the other 3 outputs' shardings unchanged from base?

**YES to both, on all three touched jits.**

`VllmModelWrapper.jit_step_func` (`models/vllm/vllm_model_wrapper.py`):
- `step_fun_impl` returns `new_kv_caches, output, aux_hidden_states, expert_indices` (`:795`) → cache is `[0]`. ✔
- `draft_step_fun_impl` returns `new_kv_caches, hidden_states, [hidden_prenorm], emitted_topk` (`:857-858`) → cache is `[0]`. ✔
- Both `out_shardings` tuples changed ONLY element `[0]` (`None → kv_cache_out_sharding`, `:881` and `:894`). Elements `[1]=P(ATTN_DATA,None)`, `[2]=None`, `[3]=None` are **verbatim identical** to `origin/glm-5.2-v4-next` (diffed the base file: base `[1..3]` = `P(ATTN_DATA,None), None, None` on both step fns). ✔
- Both step fns are the ones actually served: `jit_step_func` returns `step_fun_with_options` (target) / `draft_step_fun` (draft) and stores `step_fn_no_options` (`:914-920`), which the runner invokes (`runner/tpu_runner.py:1490`, `runner/compilation_manager.py:2143-2168`). ✔

`Eagle3Proposer` (`spec_decode/jax/eagle3.py`, commit 2):
- `_propose_impl` returns `kv_caches, draft_token_ids` (`:710-711`) → cache is `[0]`. ✔
- `_get_propose_fn` pins `out_shardings=(cache_out, None)` (`:665-668`) — `[1]` (draft_token_ids) stays `None` as in base. ✔
- `static_argnums=(0,7,8)` → `static_argnames=("num_speculative_tokens","layer_name_to_kvcache_index")` is a correct 1:1 remap: base arg-0 was `self` (dropped because the jit now wraps the **bound** method `self._propose_impl`), args 7/8 are exactly those two names in the signature (`:690-691`). Call site passes everything by keyword (`:631-640`), so donation-by-name and static-by-name resolve correctly. Instance-level caching (`self._propose_jit`, `:654-656`) is safe: `self.mesh` is fixed at construction. ✔

`donate_argnames=("kv_caches",)` is unchanged on all three; the donated buffer is `kv_caches`, whose output slot is exactly the pinned `[0]`. So the pin governs precisely the donated round-trip.

---

## 3. Mechanism → evidence — does pinning the vLLM step-fn cache out_sharding specifically eliminate the ONE-stripe drop (the measured signature)?

**The mechanism is sound and matches the measured signature; the causal chain is:**

1. Measured signature (`glm-tpu/docs/RESEARCH_LOG.md`, 2026-07-08 21:10): the gated 2-chunk-vs-1-chunk cache-dump probe returned **DIFFER, max|Δ|=5.44, exactly 1024/2048 rows stale = precisely ONE dcp stripe (half at dcp=2)**. Root cause: chunk-1's writes to the `P(BATCH,CONTEXT)`-striped cache are **not carried into chunk-2's `execute_model` step** — one dcp shard lost across the scheduler-step boundary. Explicitly *not* the kernel (single-chunk prefill is correct to ≥5000 tok; RESEARCH_LOG 19:30).
2. The persistent cache is **donated** (`donate_argnames="kv_caches"`) into the step fn every step. Pre-fix `out_shardings[0]=None` leaves the output layout **unconstrained** — XLA is free to choose a layout for the whole TPU program that need not equal the striped donated input. When input-layout ≠ output-layout under donation, the alias cannot be honored cleanly and the buffer is resharded across the step boundary; empirically that reshard loses one dcp shard.
3. The fix forces **output-layout == input-layout == `P(BATCH,CONTEXT)`**, so `input_output_aliases` can be honored in place: the striped buffer round-trips with no cross-step reshard. That is precisely the "preserve the DCP-striped kv_caches buffer across chunked-prefill `execute_model` calls" remedy the RESEARCH_LOG named as the fix target.
4. **On the right code path this time.** The prior attempt fixed `get_flax_model.run_model`; independent review (RESEARCH_LOG 21:55) caught that `GlmMoeDsaForCausalLM` is in `_VLLM_PREFERRED_ARCHITECTURES` (`model_loader.py:54-70`) → served via `get_vllm_model → VllmModelWrapper`, so the flax fix was a **no-op for GLM**. Commit `c6da472e5` moves the identical mechanism to `VllmModelWrapper`'s two step fns — GLM's actual path. ✔

I could not refute the mechanism on logic. Whether XLA *in fact* chose a mismatching layout pre-fix (vs. propagating `P(BATCH,CONTEXT)` for free) is a metal-only fact — that is exactly what the pre-fix probe measured (DIFFER), and exactly what the post-fix probe must re-measure.

---

## 4. The honest CPU caveat — what the CPU suite does and does NOT prove

**The CPU cannot exhibit the value-drop, so it cannot prove the metal drop is gone.** Verified, and it is honestly disclosed.

- On CPU, `out_shardings=None` already coincidentally yields correct values: GSPMD services the mismatched donated reshard by a safe replicate-then-repartition, so **no** token goes stale. The prior CPU-proxy review (`docs/reviews/dcp-persist-fix-cpu-proxy.md`, Finding 2) measured this directly: *"stale/missing tokens under legacy on CPU: 0 (of 64)."* My own run reproduces the tell-tale: `test_legacy_out_sharding_is_the_reshard_bug` emits JAX's `"Some donated buffers were not usable: float32[2,…]"` warning — the donation is *refused* on CPU yet values still survive. The drop is **TPU-only**.
- Consequence: the value-persistence test `test_vllm_wrapper_step_persists_all_stripes` passes with **both** `None` and `P(BATCH,CONTEXT)` on CPU — it is **vacuous w.r.t. the wiring**. The authors state this and added the MED-2 **wiring guards** (`test_vllm_wrapper_wires_cache_out_sharding`, `test_eagle3_propose_wires_cache_out_sharding`) that spy on `jax.jit` and assert `out_shardings[0].spec == P(BATCH,CONTEXT)` under the gate / `None` off-gate. I confirmed these are **non-vacuous by construction**: reverting `[0]` to `None` makes `cs is None` and the `all(cs is not None and cs.spec == striped)` assertion fails on the gate-on case.
- **So the CPU suite proves: (a) `dcp_mla_cache_sharding` returns `P(BATCH,CONTEXT)` on-gate / `None` off-gate; (b) both step fns + the MTP `_propose` jit actually feed that into `out_shardings[0]`; (c) the striped round-trip preserves all stripes *on CPU* (where it already did). It does NOT prove the TPU stripe-drop is cured.**

CPU results I ran (read-only, `JAX_PLATFORMS=cpu`):
- `tests/layers/common/test_mla_dcp_cache_persistence.py`: **19 passed**.
- `tests/layers/vllm/test_glm_dsa_mtp_index_share.py`: **14 passed** (confirms the `_propose.__wrapped__ → _propose_impl` rename did not break the index-share callers).

Off-gate byte-identity (the safety contract) holds by construction: `dcp_mla_cache_sharding` returns `None` when the env is unset, `dcp_size==1`, or `use_mla=False`, and `None` == the base `out_shardings[0]` on all three jits — so gate-off / dcp=1 / non-MLA traces are unchanged. Confirmed by `test_gate_off_is_byte_identical`, `test_gate_on_dcp1_is_noop`, `test_non_mla_dcp_out_sharding_is_unchanged`.

---

## 5. The pod falsifier — the ONLY proof the metal drop is gone (run BEFORE the passkey)

**Mandatory order: cache-dump MATCH first, passkey second.** A passkey pass without the cache-dump MATCH is not evidence the fix worked — passkey is a noisy, thresholded, downstream signal that can pass or fail for unrelated reasons (protocol, capacity, sampling). The cache-dump is the *direct* falsifier of "the stripe is now intact."

### 5.1 Falsifier definition

Re-run the **exact 21:10 probe**: two one-shot runs of the SAME ~3200-token prompt differing ONLY in `--max-batched-tokens` (a 2-chunk prefill vs a 1-chunk prefill), dump layer-0 post-prefill KV, reassemble all shards offline, diff over the prompt's physical pages.

- **PRE-FIX (recorded):** `DIFFER, max|Δ|=5.44, n_diff_rows=1024 (of 2048)` = one full dcp stripe stale.
- **POST-FIX PASS CRITERION:** `MATCH` — `max|Δ| ≤ 1e-3` and **`n_diff_rows == 0`**. Anything else (still ~1 stripe, or a *different* stale count) = the fix did not land on metal; **do not proceed to passkey, do not trust it.**

### 5.2 Exact commands (from `docs/11-pod-runbook.md:484-543`, dense-MLA / DSA-disabled — the passkey path)

```bash
# Relaunch with DCP baked into the raylet env if the engine isn't up:
EXTRA_ENVS="GLM_MLA_DCP=1" GLM_FLIGHT_RECORDER=1 TPU_MIN_TOKEN_BUCKET=32 \
  bash ~/glm-tpu/scripts/launch_glm_32chip.sh
cd ~/glm-tpu/bench && set -a && . ~/glm-tpu/.env && set +a
_ENVS="NEW_MODEL_DESIGN=1 MODEL_IMPL_TYPE=vllm TPU_MULTIHOST_BACKEND=ray OMP_NUM_THREADS=1 \
HF_HUB_DISABLE_XET=1 TPU_DISABLE_DSA_INDEXER=1 DISABLE_WEIGHT_REQUANTIZATION=1 \
REQUANTIZE_WEIGHT_DTYPE=float8_e4m3fn TPU_MIN_TOKEN_BUCKET=32 GLM_TP=32 \
GLM_ASYNC_SCHED=0 GLM_MLA_DCP=1 GLM_DCP=2 GLM_DCP_CACHE_DUMP_LAYERS=0"

# (a) TWO-chunk prefill (mbt=2048 < N -> chunks 2048 + 1152)
env $_ENVS GLM_DCP_CACHE_DUMP=/tmp/dcp_2chunk.npz \
  ~/vllm-env/bin/python -u glm_longctx.py --lengths 3200 --depths 0.5 --trials 1 \
    --max-seqs 1 --max-batched-tokens 2048 --gmu 0.90 --note "DCP debug 2-chunk"

# (b) ONE-chunk prefill of the SAME prompt (mbt=6144 >= N -> single chunk)
env $_ENVS GLM_DCP_CACHE_DUMP=/tmp/dcp_1chunk.npz \
  ~/vllm-env/bin/python -u glm_longctx.py --lengths 3200 --depths 0.5 --trials 1 \
    --max-seqs 1 --max-batched-tokens 6144 --gmu 0.90 --note "DCP debug 1-chunk"
```
Then the offline numpy diff (`docs/11-pod-runbook.md:509-542`) prints
`VERDICT: MATCH -> write fine` (required) or `DIFFER -> ... persistence lost` (fix did not land).

Run the probe **3×** (the same DIFFER→MATCH each time) — multi-host is probabilistic; one pass ≠ done (the DSV4 worker-3 race discipline).

### 5.3 Only after 3/3 MATCH: the passkey

```
GLM_MLA_DCP=1 GLM_DCP=2 ... glm_longctx.py --lengths 131072 --depths ... (Stage-2 gate P = retrieval ≥95%)
```
A MATCH followed by a passkey ≥95% is the fix validated. A MATCH followed by a passkey *miss* points elsewhere (kernel/indexer/capacity), not at this fix. A DIFFER means stop.

---

## 6. Residual findings (do not block the code; do gate the validation)

**R1 — BLOCKING THE FALSIFIER: the cache-dump probe hook is not in this branch.** `tpu_inference/runner/dcp_cache_dump.py` and every `GLM_DCP_CACHE_DUMP*` reference are **absent** from the `glm-5.2-v4-dcppersist` worktree (`grep -rn GLM_DCP_CACHE_DUMP tpu_inference/` → nothing; the file does not exist). The hook lives on commit `af708bbca` (the obsguard line). **The §5 falsifier cannot be run until that hook is merged/cherry-picked into the serving branch.** Merge it (it is gated; unset = zero behavioral change) before the pod run, or the "proof" step is impossible.

**R2 — `_decode_core` is an untouched sibling of the same bug class (verify it is off the drop path).** `runner/decode_loop.py:93-121` is the fused multi-step decode jit; it `donate_argnames=("kv_caches",)` (`:112`), **returns** the updated `kv_caches` (`:247-248`, `:421`), and has **no `out_shardings`** → defaults to `None` for every output, including the donated cache. This is the identical "donated cache, unconstrained output layout under DCP" shape these commits fixed on the step fn, and it was **not** touched. Mitigating evidence it is benign: single-chunk prefill + generation reportedly passed at dcp=2 (RESEARCH_LOG 19:30), and decode now calls the *pinned* `step_fn_no_options` internally so the loop-carry inherits `P(BATCH,CONTEXT)`. But CPU cannot prove it and the multi-chunk drop is a prefill-boundary phenomenon, so this is not the passkey's critical drop site. **Action:** the cache-dump hook only dumps *prefill* steps (runbook `:506`); if the fused decode loop is on the passkey path, additionally confirm the post-*decode* cache is intact (or rely on the end-to-end passkey retrieval AFTER the prefill MATCH gates). If a future dcp config regresses, `_decode_core` is the first suspect.

**R3 — the commits' `_no_explicit_shardings` justification overclaims for the flax path.** The NOTE ("explicit q/k/v/o shardings are a test-only hook, always `None` on a served path") is **false for `models/jax/deepseek_v3.py:702-714`**, which passes real `PartitionSpec`s (`self.query_nth`, etc.). This does **not** affect GLM (vLLM path, shardings `None`) or the passkey. It only means: if a *flax* MLA model were ever served under `GLM_MLA_DCP=1` on a dcp>1 mesh, `mla_attention` would return `P(BATCH)` (gate off via `_no_explicit_shardings=False`) while `get_kv_cache_out_sharding` pins the flax step-fn output to `P(BATCH,CONTEXT)` — an internal reshard (still input==output so donation stays clean, but not the "byte-identical" the NOTE implies). Fix the wording, or gate `get_kv_cache_out_sharding` on `_no_explicit_shardings` too, to keep the flax claim honest. Out of scope for GLM.

---

## 7. Positive assurance

- **Right path:** GLM (`GlmMoeDsaForCausalLM`) and its `DeepSeekMTPModel` draft are both in `_VLLM_PREFERRED_ARCHITECTURES` (`model_loader.py:54-70`); the fix is on `VllmModelWrapper`'s two step fns + the MTP `_propose` jit — the code GLM actually executes. The prior flax-only miss is corrected.
- **Right spec:** pinned `out_shardings[0] = P(BATCH,CONTEXT)` == the vLLM attention cache out_spec (`attention_interface.py:662-663`, reached with `_no_explicit_shardings=True` from both `flash_attn_mla.py:205` and `custom_ops/mla_attention.py:1211`) == `create_kv_caches` allocation (`kv_cache.py:132-135`) == the donated input.
- **Minimal + additive:** only `out_shardings[0]` changes; the other three outputs are byte-identical to base on both step fns; off-gate returns `None` == base. Gate mirrors `mla_attention` on every served path.
- **Cache is `[0]`** on all three jits; `donate_argnames` unchanged; the `static_argnums→static_argnames` remap on `_propose` is a correct 1:1 (self dropped for the bound method).
- **Tests:** 19 + 14 pass on CPU; wiring guards are non-vacuous; the honest limits are disclosed in-commit, not hidden.
- **No masking / reward-hacking.**

**Bottom line:** the code is PR-quality and correct on-gate; merge it. But the metal cure is unproven off-pod by construction. Gate the passkey behind a 3/3 on-pod cache-dump **MATCH** (after merging the R1 probe hook). Until DIFFER→MATCH is observed, treat the fix as *plausible, not validated*.

# Adversarial review — DCP cross-step KV-cache persistence fix: is the CPU test a SOUND proxy for the metal stripe-drop?

Reviewer: independent adversarial reviewer (did NOT write this code; read-only; `JAX_PLATFORMS=cpu`, **TPU untouched** — no TPU jaxlib present, so I *cannot* touch it).
Date: 2026-07-08.
Target: worktree `~/tpu-inference-dcppersist`, branch `glm-5.2-v4-dcppersist`,
commits `124a732f1` (fix + test) and `4211041d2` (review follow-up).
Base: `886eaceb4` (= `git merge-base HEAD origin/glm-5.2-v4-next`).
Diff scope (verified): **only** `tpu_inference/models/common/model_loader.py` (+74) and the new
`tests/layers/common/test_mla_dcp_cache_persistence.py` (+304). No other file changes.

## Verdict up front

The change, **considered as a fix to the flax_nnx MLA step-fn (`get_flax_model`)**, is real,
correctly gated, and not reward-hacking. The test is honest about its own limitation. **But the
central epistemic claim I was asked to attack is worse than framed, for a reason that dominates
everything else:**

> **FINDING 1 (HIGH — the fix is not on GLM's serving path).** The bug was empirically observed on
> **GLM** (`GlmMoeDsaForCausalLM`, RESEARCH_LOG 2026-07-08 21:10, cache-dump on the pod). GLM has
> **no flax_nnx implementation** and is served through the **vLLM wrapper**
> (`VllmModelWrapper.jit_step_func`), whose KV-cache out-sharding is `None` and **was not touched by
> this fix**. The fix edits `get_flax_model.run_model/run_draft_model` — a code path GLM never
> takes. The CPU test exercises `get_kv_cache_out_sharding`, which is wired **only** into
> `get_flax_model`. So the CPU test is not merely "unable to show the stripe-drop" — it is
> validating code that is **dead for GLM**. On the GLM path the fix is a **no-op**, and the pod
> passkey run cannot be moved by it.

So the answer to "is *no reshard on CPU* a sound proxy for *no stripe-drop on TPU*" splits in two:
- For the **flax MLA path** (DeepSeek-V3-style, where the fix is live): it is a **NECESSARY, not
  sufficient** proxy; sufficiency is pod-only (Findings 2–4). But the pod runs GLM, not this path,
  so there is **no planned metal validation of the live fix at all.**
- For **GLM** (the model that actually fails, and the model the pod validates): the proxy is **moot**
  — the tested helper isn't on GLM's path, and the fix changes nothing GLM executes.

Recommendation (detailed in §6): the pod validation **must** include a post-fix cache-dump **MATCH**
check, not just passkey — and the sharpest prediction of Finding 1 is that on GLM that dump will show
**no change** (still ~1 stripe stale). The real fix for GLM belongs in
`VllmModelWrapper.jit_step_func` out-shardings (both `step_fun_jit` and `draft_step_fun`), not
`get_flax_model`.

---

## 0. The fix, and the mechanism it claims

Pre-fix, `get_flax_model` forced the donated KV-cache OUTPUT sharding of the jitted step fn to the
generic `P(ATTN_DATA, None, ATTN_HEAD)` (`model_loader.py`, historical). The MLA cache is allocated
`P(BATCH, CONTEXT)`-striped (`runner/kv_cache.py:131-136`) and `mla_attention` returns it
`P(BATCH, CONTEXT)` under the DCP gate (`attention_interface.py:662-663`). The mismatch + input
donation (`donate_argnums=1`) is claimed to alias a pre-reshard buffer into a post-reshard output on
TPU, dropping exactly one `dcp` stripe across the multi-chunk-prefill scheduler-step boundary.

Fix: `get_kv_cache_out_sharding(mesh, use_mla)` (`model_loader.py:320-378`) returns `P(BATCH, CONTEXT)`
**only** for MLA under the DCP gate (`GLM_MLA_DCP` truthy AND real `dcp` axis), else the exact
historical `P(ATTN_DATA, None, ATTN_HEAD)`. Wired at `model_loader.py:432`, feeding the `run_model`
(`:450-460`) and `run_draft_model` (`:467-473`) `out_shardings[0]`.

Empirical evidence the fix cites (RESEARCH_LOG 2026-07-08 21:10): a 2-chunk vs 1-chunk prefill
cache-dump on the pod (dcp=2, layer 0) returned **max|Δ|=5.44, exactly 1024/2048 rows stale = one
`dcp` stripe.**

---

## FINDING 1 (HIGH) — mistargeted: `get_flax_model` is not GLM's serving path

Chain of custody (all verified by reading source on this worktree):

1. `GlmMoeDsaForCausalLM` is in `_VLLM_PREFERRED_ARCHITECTURES` with the explicit comment
   *"GLM-5.x uses DeepSeek Sparse Attention (DSA), which has no flax_nnx implementation; route it
   through the vLLM path"* — `model_loader.py:52-68`. Its MTP draft (`DeepSeekMTPModel`) is in the
   same set (`:69-76`).
2. `resolve_model_architecture` returns `"vllm"` for any arch in that set
   (`model_loader.py:777-779`: `impl = "vllm" if arch in _VLLM_PREFERRED_ARCHITECTURES else "flax_nnx"`).
   The pod runbook (RESEARCH_LOG 21:10 command block) sets `MODEL_IMPL_TYPE=vllm` explicitly, so
   `get_model` dispatches **straight to `get_vllm_model`** (`model_loader.py:709-711`) — it never even
   attempts `get_flax_model`.
3. `get_vllm_model` returns `model_fn = VllmModelWrapper.jit_step_func()` (`model_loader.py:629`,
   `:657`). The runner calls exactly this `self.model_fn` (`tpu_runner.py:1333-1336`) and stores its
   output back (`:1517`). Persistence between scheduler steps is governed **solely** by this step-fn's
   out-sharding.
4. `VllmModelWrapper.jit_step_func` sets, for BOTH the main and draft step fns,
   `donate_argnames=("kv_caches",)` with `out_shardings=(None, ...)` — `None  # kv_caches - keep
   original sharding` (`vllm_model_wrapper.py:862-864` draft, `:875-877` main). **This is not the
   legacy `P(ATTN_DATA, None, ATTN_HEAD)` the fix removes, and it is not touched by the two commits**
   (`git log 886eaceb4..4211041d2 -- vllm_model_wrapper.py` is empty).
5. GLM's MLA still flows through the common DCP machinery — `flash_attn_mla.py:30,205` and
   `custom_ops/mla_attention.py:40,1211` call `attention_interface.mla_attention`, so the cache does
   come out `P(BATCH, CONTEXT)` under the gate. So GLM has a striped cache but an **unconstrained
   (`None`)** step-fn out-sharding, not the forced-legacy one.
6. `get_kv_cache_out_sharding` is referenced **only** in `model_loader.py` (def at `:320`, call at
   `:432`) — grep-confirmed nowhere in `vllm_model_wrapper.py` or the runner.

Consequence: the committed program that runs on the GLM pod is **byte-identical pre- and post-fix**
(get_flax_model is never compiled; the helper is never imported by the vLLM path). Therefore:
- The pod stripe-drop diagnosed at 21:10 is on the **`None`-out-sharding** vLLM path — which does
  **not** express the "forced `P(ATTN_DATA,None,ATTN_HEAD)`" mismatch the fix targets.
- Either (a) the `None`+donation path does NOT force a legacy reshard, so the fix's forced-reshard
  root-cause narrative does not describe GLM's failure at all, **or** (b) `None`+donation still
  reshards on metal (GSPMD picks an output layout it can't alias to the `P(BATCH,CONTEXT)` input), in
  which case the correct fix is to **pin `out_shardings[0]` to the striped spec in the vLLM wrapper**
  — which this change does not do. In both cases **the fix does not fix GLM.**

Refutation attempts (all failed to save the "touches the serving path" premise):
- *"Maybe get_flax_model is a fallback that still runs for GLM."* No — GLM is `_VLLM_PREFERRED`, and
  the pod pins `MODEL_IMPL_TYPE=vllm`; the flax attempt is skipped, and even under `auto` it resolves
  to `"vllm"`. `get_flax_model` would raise `UnsupportedArchitectureError` for GLM regardless.
- *"Maybe a companion vLLM-wrapper change is part of this work."* The branch tip is `4211041d2`; the
  vLLM wrapper is untouched by both DCP commits (its last change is an MTP commit `89e1d5b5a`).
- *"Maybe the author knows GLM is vLLM."* The follow-up commit's own note (`model_loader.py:353-356`)
  reasons about *"always None in GLM serving"* — but that "None" is `mla_attention`'s explicit
  q/k/v/o sharding args, a different thing from the step-fn `out_shardings=None`. The author correctly
  established that GLM activates the DCP gate inside the common `mla_attention`, then wired the
  out-sharding remedy into the **wrong wrapper.**

This does not make the code *wrong* — it is a correct fix **for flax MLA models (DeepSeek-V3 under
DCP)**. It is mistargeted for its **stated** goal (unblock GLM 128K/256K), and the CPU test locks in a
helper GLM never calls.

---

## FINDING 2 (the task's core) — the CPU discriminator is "reshard present vs absent," empirically confirmed; it is NECESSARY, not sufficient, for "stripe intact"

I confirmed the central claim empirically. Under the LEGACY out-sharding on 8 CPU devices, threading
two `execute_model`-like steps through the same donating jit:

```
dcp=2: legacy persisted spec == striped?  False    # the reshard IS present (the discriminator)
       VALUES two-step(legacy) == single-step ref?  True
       stale/missing tokens under legacy on CPU:  0 (of 64)
dcp=4: (identical) reshard present, VALUES match, 0 stale
```

and XLA printed the mechanism verbatim:

> `[SPMD] Involuntary full rematerialization. The compiler cannot go from sharding {...} to {...}
> efficiently ... As the last resort, SPMD will replicate the tensor and then partition it to obtain
> the target sharding` (spmd_partitioner.cc:668, seen twice — dcp=2 and dcp=4).

So the report's admission is **directly observed**: on CPU the mismatched donated reshard is serviced
by replicate-then-repartition, the values survive, and the buffer-donation warning fires
(`Some donated buffers were not usable: float32[2,2,8,4]` — legacy path only; the fixed path emits no
such warning). The CPU test therefore keys on **spec-invariance (reshard present vs absent)**, NOT on
**stripe stale vs intact.** (Probe: `scratchpad/prefix-repro/probe_values.py`, reusing the suite's own
geometry helpers.)

Is eliminating the reshard **sufficient** to guarantee the metal stripe-drop is gone? (Answer scoped
to the flax path where the fix is live.)

- **Necessary — yes.** A forced legacy out-sharding on a donated `P(BATCH,CONTEXT)` cache creates a
  cross-step reshard on the striped dim; removing it is necessary.
- **Merely necessary, not provably sufficient — three independent reasons:**
  1. **Spec-equality ≠ physical-layout-equality.** The test asserts `.spec == P(BATCH, CONTEXT)`
     (`test_...py:197,258-263`), not TPU tiling / memory-kind / device-order equality. XLA on TPU can
     still insert a **layout-only** reshard between the shard_map body's `P(BATCH,CONTEXT)` and the
     jit boundary's `P(BATCH,CONTEXT)` if it picks different physical layouts — re-arming the exact
     in-place-aliased-transform hazard the fix is meant to remove. CPU cannot see this.
  2. **Donation is best-effort, and the fix flips it.** Under legacy, CPU reported the donation
     "not usable" (declined → safe copy). The fix makes the layouts match, so post-fix the buffer IS
     aliased. A same-layout alias is the well-trodden safe case (every working non-DCP step relies on
     it), but it is a **behavior change on metal** — only the pod confirms the honored alias is clean.
  3. **The report's stated mechanism is overbroad.** "Mismatch + donation ⇒ drop a stripe" proves too
     much: **non-DCP MLA also reshards a donated cache every step** — its shard_map returns `P(BATCH)`
     (`attention_interface.py:663`) while the historical/fixed jit out-sharding is
     `P(ATTN_DATA,None,ATTN_HEAD)` (they differ on dim-2, sharded over `model/expert`), yet dcp=1 MLA
     is correct (GSM8K 96.9%, passkey 100% @8K/32K). So a donated-cache reshard is **demonstrably
     tolerated** on TPU on the non-dcp dims; the true failure predicate is narrower than "any reshard"
     — specifically the dim-1 all-gather over the `dcp` axis. The CPU test's discriminator ("any
     reshard") is therefore **coarser** than the metal failure predicate.

**Only the pod can confirm:** that the striped **physical** layout round-trips on TPU with the
donation honored and **no** stripe lost — i.e. a post-fix cache-dump **MATCH** (max|Δ|=0, 0 rows
stale). "No reshard on CPU" cannot establish that.

---

## FINDING 3 — the 6/10 pre-fix failures are non-vacuous and key on the sharding-invariance (reproduced)

Reproduced the disable-then-restore without modifying tracked files (scratchpad copy of the test +
a `conftest.py` autouse fixture that monkeypatches the test module's imported
`get_kv_cache_out_sharding` back to the unconditional legacy spec; enabled by `REPRO_PREFIX=1`).

- **Fixed (HEAD):** 10/10 pass.
- **Pre-fix (legacy forced):** exactly **6 failed / 4 passed**, matching the task's "6/10". The 6 are
  the DCP-property tests; the 4 passing are the byte-identity guards (`gate_off`, `gate_on_dcp1`,
  `non_mla_unchanged[2,4]`).

Every one of the 6 fails on a **sharding-spec** assert, not a setup error and not a values assert:
- `test_dcp_out_sharding_matches_allocation[2,4]` — `test_...py:197`
  `assert out.spec == alloc.sharding.spec` (legacy `P(ATTN_DATA,None,ATTN_HEAD)` ≠ striped
  `P(BATCH,CONTEXT)`).
- `test_two_step_cache_persistence_all_stripes[2,4]` — `test_...py:263` assert **(a)**
  `c.sharding.spec == P(BATCH,CONTEXT)`. **This is the crucial one: it trips on the sharding
  invariance at line 263, BEFORE the value comparison `assert_array_equal(two_np, ref_np)` at line
  269** — corroborating Finding 2 that on CPU the *values* would have matched. It is non-vacuous: the
  reference asserts all `TOTAL_TOKENS` tokens are actually present (`:266-267`).
- `test_legacy_out_sharding_is_the_reshard_bug[2,4]` — `test_...py:304`
  `assert get_kv_cache_out_sharding(...).spec != _LEGACY_SPEC`. Its first two asserts (`:300-301`,
  that the legacy persisted layout *is* the legacy reshard and *is not* the striped spec) still PASS
  pre-fix — correctly pinning the reshard.

Minor doc-staleness: commit `124a732f1`'s message says *"8/10 of these fail on the pre-fix
out-sharding."* That was accurate **at that commit** (the non-MLA test then asserted the
allocation spec and also failed pre-fix, 6+2=8); the follow-up `4211041d2` flipped that test to assert
the legacy spec is *unchanged*, dropping the pre-fix failure count to **6/10**. The current tree is
6/10; the commit message is now stale. Not a correctness issue.

---

## FINDING 4 — the mechanism does not DERIVE "exactly one stripe," and the "one stripe" was measured on the path the fix doesn't touch

- The empirical "exactly 1024/2048 rows stale = one `dcp` stripe" (RESEARCH_LOG 21:10) was measured on
  **GLM = the vLLM `None`-out-sharding path** (Finding 1). The mechanism the fix removes (forced
  `P(ATTN_DATA,None,ATTN_HEAD)` reshard) is a **`get_flax_model` property that is not present on the
  measured path.** So the correspondence "forced-legacy-reshard ⇒ one stale stripe" is a **cross-path
  inference**, never established on the path where the "one stripe" was observed.
- Even on the flax path, the fix's write-up does not *derive* multiplicity-one. The stale data lying
  on the `dcp`/dim-1 axis is *consistent* with a dim-1 (CONTEXT) reshard and *inconsistent* with a
  dim-2 (kv_packing) reshard — a good consistency argument — but "exactly one vs two vs scrambled" is
  an **empirical** observation about XLA's in-place-collective ordering, not a prediction of the
  stated model. A skeptic can hold that the reshard is a *correlate* of the true cause (e.g. an
  unconstrained-`None` layout choice on the vLLM path) rather than the cause. Removing all cross-step
  transformation is still the right structural move **where it is applied**; the gap is that it is
  applied off-path for GLM.

---

## 5. Positive assurance (what is genuinely right)

- **Not reward-hacking / not masking.** The test's own docstring explicitly states the CPU cannot
  exhibit the value drop (*"XLA on CPU falls back to a safe replicate-then-repartition ... the silent
  stripe loss is a TPU donation-aliasing effect"*) and names its discriminator as the reshard itself.
  I confirmed that admission empirically (Finding 2). No test asserts a metal-only property it cannot
  observe, and no gate is fudged.
- **Correct and well-gated for the flax MLA path.** `get_kv_cache_out_sharding` returns the exact
  historical spec for gate-off, dcp=1, and non-MLA (`model_loader.py:362-378`); the follow-up commit
  `4211041d2` correctly narrowed the override to MLA-only (the non-MLA attention shard_map already
  returns `P(ATTN_DATA,None,ATTN_HEAD)`, so touching it would have *introduced* a reshard — the
  review catch is sound). `test_gate_off_is_byte_identical` / `test_gate_on_dcp1_is_noop` /
  `test_non_mla_dcp_out_sharding_is_unchanged` pin byte-identity; all pass.
- **The DCP-gate computation matches `mla_attention`** (env `GLM_MLA_DCP` + real `dcp` axis via
  `mesh.shape`; `model_loader.py:358-364` vs `attention_interface.py:645-651`), modulo the documented
  `_no_explicit_shardings` clause — a latent coupling (see §6) but not a bug on the served path.
- Worktree left pristine (`git status --porcelain` empty); all repro artifacts are in the session
  scratchpad, TPU never contacted (no TPU jaxlib installed).

---

## 6. Recommendations — and the explicit answer on the cache-dump MATCH check

**Q (task): must the pod validation include a post-fix cache-dump MATCH check, not just passkey?**
**A: YES — mandatory, and for a sharper reason than originally intended.**

1. **Mandatory post-fix cache-dump MATCH.** Re-run the exact 21:10 probe (2-chunk vs 1-chunk prefill,
   dcp=2, layer 0, all shards reassembled) **after the fix** and assert **max|Δ| == 0 / 0 rows stale**
   — a direct falsifier of "the stripe is now intact." Passkey ≥95% is a noisy, threshold, downstream
   signal that can pass for the wrong reason (needle happens to land on a surviving stripe; fp8 masks
   it; recompilation nudges the `None` layout). Only the dump tests the mechanism.
2. **Finding-1 corollary — the dump is the decisive test of whether the fix does anything on GLM.**
   Because the committed change does not touch GLM's compiled program, the **prediction is that the
   GLM cache-dump is UNCHANGED from pre-fix (still ~1 stripe stale)** and passkey still fails. If
   passkey *passes*, that is a red flag for a spurious/unrelated cause (or a silent recompile side
   effect) and must be explained, not celebrated. Run the dump pre- AND post-fix on GLM and diff the
   two dumps.
3. **Where the real GLM fix belongs.** Pin `VllmModelWrapper.jit_step_func` `out_shardings[0]` (the
   `None` at `vllm_model_wrapper.py:864` for `draft_step_fun` and `:877` for `step_fun_jit`) to the
   DCP-striped spec `P(BATCH, CONTEXT)` for MLA-under-DCP (the same `get_kv_cache_out_sharding` rule),
   matching the `create_kv_caches` allocation and the `mla_attention` out-spec. Confirm on CPU that
   the vLLM step-fn's persisted `.spec` round-trips (the existing test only proves it for the flax
   helper). Then the same pod cache-dump MATCH is the gate.
4. **Close the `_no_explicit_shardings` coupling.** `get_kv_cache_out_sharding` intentionally omits
   `mla_attention`'s `_no_explicit_shardings` clause (`model_loader.py:353-356`,
   `attention_interface.py:651`). If any served path ever passed explicit q/k/v/o shardings,
   `mla_attention` would emit `P(BATCH)` while the helper emits `P(BATCH,CONTEXT)` → the reshard
   returns. Documented, not currently triggered, but it is a latent divergence — assert it (a test
   that both gates agree given explicit shardings) rather than relying on the prose invariant.

### Repro pointers (session scratchpad; delete freely)
- `scratchpad/prefix-repro/test_mla_dcp_cache_persistence.py` — copy of the suite.
- `scratchpad/prefix-repro/conftest.py` — `REPRO_PREFIX=1` monkeypatches the helper back to legacy →
  6/10 fail, all on the sharding-invariance asserts (Finding 3).
- `scratchpad/prefix-repro/probe_values.py` — under legacy on CPU: reshard present, **0/64 tokens
  stale, values byte-match** the single-step reference; XLA logs replicate-then-repartition (Finding 2).

# Gate D Resolution Research: Mechanism-Level Analysis of the Layer-1 One-ULP BF16 Divergence in the GLM-5.2-FP8 JAX/XLA/TPU-v4 Port

## TL;DR
- The single most probable root cause of the `27bd`→`26bd` one-ULP miss is **conversion elision under excess-precision**: XLA's `xla_allow_excess_precision` is set to `true` by default (source-confirmed: `opts.set_xla_allow_excess_precision(true);` in `xla/debug_options_flags.cc`), permitting XLA to *retain FP32 excess precision* rather than physically materialize the BF16 rounding the accepted HLO performs at the qkv-a/RMS frontier — so the highest-ranked mechanism (M1) pins that rounding with `stablehlo.reduce_precision`, the one op XLA's `AlgebraicSimplifier` will not elide.
- The same downstream symptom across PP8/PP16/WS32 points upstream of the scorer to a **physical reduction-association / conversion-placement** difference; the two next-ranked mechanisms (M2 barrier-fenced association, M3 Pallas/Mosaic explicit reduction+rounding over LP2/LP4 remote DMA) reproduce the accepted association bit-for-bit without a 32-chip collective.
- The two declared abstract mechanisms are weak as stated: `compensated_auxiliary_dependency` (M4) is implementable only as an explicit error-free transform whose cancellation is *proven*, and `auxiliary_device_tuple_dependency` (M5) should be **rejected** unless it is made physically causal via a `reduce_precision`+`optimization_barrier` pin — otherwise it is a no-consumer shadow that XLA will delete.

---

## Key Findings

1. **XLA does NOT silently elide a rounding `f32→bf16→f32` pair by default, but the excess-precision flag that governs this is ON by default.** The `xla.proto` comment for `xla_allow_excess_precision` (field 122) states the `FloatNormalization` pass inserts `f32 → bf16 → f32` conversion pairs and that "These are not removed by the `AlgebraicSimplifier`, as that will only simplify conversions that are no-ops, e.g. `bf16 → f32 → bf16`." [DOCUMENTED — openxla/xla `xla/xla.proto`; `xla/debug_options_flags.cc`: "Allow xla to increase the output precision of an instruction."] The default is `true` — `xla/debug_options_flags.cc` sets `opts.set_xla_allow_excess_precision(true);`. [DOCUMENTED — source-quoted]. This is the strongest single explanation for a one-bit normalized-state difference between a plan that materializes the round and one that carries excess FP32 precision into RMS.

2. **`reduce_precision` is a semantically-defined rounding op that models exactly the BF16 round the accepted contract requires.** XLA's operation semantics: ReducePrecision "Models the effect of converting floating-point values to a lower-precision format … and back … The input values are rounded to the nearest value representable with the given number of mantissa bits (using 'ties to even' semantics)." [DOCUMENTED — openxla.org/xla/operation_semantics; StableHLO spec `reduce_precision`; interpreter test `stablehlo/tests/interpret/reduce_precision.mlir` confirms materialized rounding]. BF16 = 8 exponent bits, 7 mantissa bits (format `e8m7`).

3. **`optimization_barrier` is an identity op that prevents code motion, fusion, and CSE across it — but it is a coarse boundary, not a fine-grained numeric-order guarantee.** Spec: "Ensures that the operations that produce the operand are executed before any operations that depend on the result and prevents compiler transformations from moving operations across the barrier. Other than that, the operation is an identity." [DOCUMENTED — StableHLO spec; openxla.org/xla/operation_semantics; mhlo docs]. JAX docs/changelog: it prevents CSE and controls scheduling. [DOCUMENTED — jax.lax.optimization_barrier]. However, real cases show it is insufficient against XLA-inserted copies and cannot force physical layout. [DOCUMENTED — jax-ml/jax issues #25399, #23471; pytorch/xla #3486].

4. **TPU MXU numerics and the FP32→BF16 conversion scheme are fixed and documented.** "By default, TPUs perform matrix multiplication operations with bfloat16 values and accumulations with IEEE float32 values." [DOCUMENTED — cloud.google.com/tpu/docs/bfloat16; TPU architecture doc: "All multiplies take bfloat16 inputs, but all accumulations are performed in FP32 number format"; corroborated by arXiv 1810.09868, 128×128 = 16,384 MAC units per MXU with bfloat16 multiplies and float32 accumulates]. Crucially, the conversion rounding scheme is specified: "The format conversion from float32 to bfloat16 is automatically inserted by the XLA compiler. On TPU, the rounding scheme in the conversion is round to nearest even and overflow to inf," and "the bfloat16 on Cloud TPU does not support subnormals, so all subnormals are flushed to zero during the conversion." [DOCUMENTED — docs.cloud.google.com/tpu/docs/bfloat16]. A one-bit normalized-state difference is therefore consistent with a *different placement of the FP32→BF16 conversion* or a *different FP32 accumulation association* — not with the MXU being nondeterministic on a fixed program. The subnormal-flush detail is a concrete falsification lever: if element 2,795's accepted value sits near the BF16 subnormal boundary, flush-to-zero timing (before vs. after the round) can itself produce the one-ULP delta.

5. **XLA reductions are explicitly reassociation-tolerant / order-unspecified.** "The evaluation order of the reduction function is arbitrary and may be non-deterministic. Therefore, the reduction function should not be overly sensitive to reassociation. Some reduction functions like addition are not strictly associative for floats." [DOCUMENTED — openxla.org/xla/operation_semantics, Reduce]. This is the compiler-sanctioned mechanism by which a 32-partial reduction and a local-group reduction can differ by one ULP.

6. **Deterministic/reproducible reductions can make a local group reproduce a global reduction bit-for-bit.** Ahrens, Nguyen & Demmel's reproducible summation (UCB/EECS-2015-229; ACM TOMS 46(3), Art. 22, 2020, ReproBLAS) "computes a reproducible sum of floating point numbers, independent of the order of summation … depends only on a subset of the IEEE Floating Point Standard 754-2008," and is reproducible "independent of the order of the summands, how they are assigned to processors, or how they are aligned in memory." [DOCUMENTED — peer-reviewed]. A fixed binary reduction tree independent of TP size also yields order-invariance. [DOCUMENTED — arXiv 2511.17826].

7. **Error-free transforms (TwoSum/FastTwoSum/Dekker/Knuth) are exact but fragile to compiler reassociation.** FastTwoSum EFT holds when `x−a ∈ F` (e.g. `|a|≥|b|` or exponent-ordered operands). [DOCUMENTED — Dekker 1971; arXiv 2601.17198 Properties 1–2; Ogita–Rump–Oishi TwoSum]. Critically: "An error-free transform is an algorithm over rounding events … A compiler that reassociates the expression can therefore erase the information the algorithm was designed to recover." [DOCUMENTED — luma.gl FP note]. This is why any compensated mechanism on XLA *must* be fenced with `optimization_barrier`/`reduce_precision`.

8. **Pallas/Mosaic gives explicit accumulator-dtype and conversion control, exposes 2-/4-chip local communication, and appears in HLO as an opaque `tpu_custom_call`.** `pltpu.make_async_remote_copy` performs remote DMA to a `device_id` computed from `lax.axis_index`; barrier semaphores synchronize. [DOCUMENTED — docs.jax.dev pallas/tpu/distributed.html]. "Pallas TPU lowering is aware of `jax.default_matmul_precision`." [DOCUMENTED — pallas/tpu/details.html]. Pallas "shows up in HLO as an opaque `tpu_custom_call`. XLA … can't see or rewrite the kernel body." [DOCUMENTED — RPA paper arXiv 2604.15464; patricktoulme substack]. This means a Mosaic kernel is a *reassociation-proof* container for the accepted association.

9. **`jax.lax.top_k` and `lax.sort` have documented, stable tie semantics — the correct basis for a DSA set/order equivalence proof.** `top_k`: "This is a stable algorithm: If two elements are equal, the lower-index element appears first." `lax.sort` default `is_stable=True`; "-0.0 and 0.0 are treated as equivalent, and NaN values are sorted to the end." [DOCUMENTED — docs.jax.dev top_k, sort]. `approx_max_k` is approximate and must NOT be used for exact cutoff-active proofs (tombstoned). [DOCUMENTED — approx_max_k is the approximate path; "As of this writing, jax.lax.top_k is the only exact Top-K implementation available for TPUs," arXiv 2506.04165].

10. **Verification tooling exists to seal and diff programs offline.** `stablehlo-translate --interpret` with `check.expect_eq` does bitwise equality; `run_hlo_module --reference_platform=Interpreter` compiles+runs+compares against the reference interpreter; `hlo-opt --list-stages` exposes `hlo`, `hlo-backend`, `buffer-assignment`, `llvm`, etc.; `--xla_dump_to` dumps before/after/optimized HLO and HloProto. [DOCUMENTED — openxla.org/stablehlo/reference, openxla.org/xla/tools, openxla.org/xla/hlo_dumps].

---

## Details: Ranked Mechanisms

### M1 — Conversion-placement pin at the qkv-a/RMS frontier via `reduce_precision` (RECOMMENDED: IMPLEMENT)

**Hypothesis.** The accepted HLO physically materializes the BF16 round of the recurrent state before it crosses the qkv-a boundary (DB485: "explicit `bf16[32,6144]` weighted row, bitcasts BF16"). The PP16 candidate, compiled with the default `xla_allow_excess_precision=true`, retains FP32 excess precision at the corresponding edge, so the RMS input differs by one ULP (`27bd` vs `26bd`). This is the "conversion placement" branch of the localization evidence and is directly supported by the `xla.proto` field-122 semantics plus the source-confirmed default-on flag.

**Source-level sketch.**
```python
import jax, jax.numpy as jnp
from jax import lax

def qkv_a_frontier(recurrent_fp32):  # recurrent_fp32 : f32[6144], the pre-round RMS operand
    # Force the accepted BF16 rounding to be physically materialized:
    rounded = lax.reduce_precision(recurrent_fp32, exponent_bits=8, mantissa_bits=7)  # bf16 RNE
    rounded = lax.optimization_barrier(rounded)  # fence so no producer fuses/reassociates across the round
    return rounded  # feeds RMS calc + bf16 normalized consumer
```
Compile with `--xla_allow_excess_precision=false` as belt-and-suspenders for this executable, OR rely on `reduce_precision` alone (it is not a no-op convert and is not elided by `AlgebraicSimplifier`). Note the TPU conversion is round-to-nearest-even with subnormal flush-to-zero; `reduce_precision(e8m7)` in the interpreter is RNE but does *not* flush subnormals, so if element 2,795 is subnormal in BF16 terms, prefer an explicit `convert` to `bfloat16` on TPU (which flushes) and confirm both against the accepted bit pattern.

**StableHLO signature (expected, present).**
```mlir
%r = stablehlo.reduce_precision %operand, format = e8m7 : tensor<6144xf32>
%b = stablehlo.optimization_barrier %r : tensor<6144xf32>
```
**Prohibited signature (falsifies).** Optimized HLO in which no `reduce-precision` / bf16 `convert` appears between the FP32 recurrent sum and the RMS reduce (i.e., the round was elided and excess FP32 flows into RMS), OR a `convert` that the layout/fusion pass has sunk into a producer so no rounding is materialized at the frontier.

**Why BF16 primary bits are preserved.** `reduce_precision(mantissa=7)` *is* the BF16 round-to-nearest-even; applying it makes the candidate's primary value equal the accepted rounded state — it does not add a shadow. One row, no collective, topology-agnostic.

**Distinct from tombstones.** Not "rounded-then-widened shadow with no physical distinction": `reduce_precision` is a spec-defined op with observable rounding (interpreter test) that the simplifier will not remove — it is the physical distinction. Not "direct persistent unrounded FP32 state": it forces *more* rounding, not less. Not a scorer/top-k family: it is strictly upstream of q-a.

**Tests.**
- *Offline (seconds):* `stablehlo-translate --interpret` a 1×6144 module with a hand-set FP32 value whose BF16 round is known; `check.expect_eq` the `reduce_precision` output equals `0x27bd` on the offending element.
- *Compile-only (seconds):* `hlo-opt --stage=hlo-backend` (or `--xla_dump_to`) and grep the optimized HLO for a `reduce-precision` (or bf16 `convert`) on the RMS-operand SSA edge; hash the module.
- *Bounded numerical (minutes):* run layers 0–1 for position 8,155 only; assert hidden element 2,795 == `0x27bd` and event-1 selected set/order match.

**Falsification.** If the optimized HLO already contains a materialized bf16 round at that edge AND the candidate still yields `26bd`, M1 is wrong and the cause is association/layout (→ M2/M3). If forcing the round changes element 2,795 to `27bd` and fixes event 1, M1 is confirmed.

**Risks.** Compiler collapse: low — `reduce_precision` is not elided by AlgebraicSimplifier (documented); the added `optimization_barrier` blocks cross-edge fusion. Reassociation: N/A (elementwise). Observer perturbation: **none** — no callback, no extra rooted output, no host path; the op is on the primary data edge. Mixed authority: none.

### M2 — Barrier-fenced reduction-association pin reproducing the 32-partial result from an LP2/LP4 local group (RECOMMENDED: IMPLEMENT / INVESTIGATE)

**Hypothesis.** The accepted state is the BF16 round of a *specific* FP32 accumulation association (the 32-partial tree). A 2-/4-chip local group computes a different association and rounds to `26bd`. Fixing the association tree to match the accepted one, then rounding once, reproduces `27bd`. This is the "reduction association / accumulation order" branch and is sanctioned by XLA's documented reduction-order freedom (Finding 5).

**Source-level sketch.** Compute the local group's partials, then combine them in the *exact accepted pairwise order* using explicit adds fenced against reassociation, then round once:
```python
def fixed_assoc_reduce(partials):  # partials: list of f32 partial sums in accepted leaf order
    partials = [lax.optimization_barrier(p) for p in partials]
    acc = partials[0]
    for p in partials[1:]:                 # or an explicit balanced binary tree matching accepted
        acc = lax.optimization_barrier(acc + p)
    return lax.reduce_precision(acc, exponent_bits=8, mantissa_bits=7)
```
Use a fixed binary reduction tree independent of shard count (order-invariance proof, arXiv 2511.17826; reproducible-summation guarantee, ReproBLAS) so LP2 and LP4 give the identical association.

**Expected HLO signature.** A chain/tree of `add` ops each separated by `optimization_barrier`, terminating in one `reduce-precision`; no `all-reduce`/`reduce-scatter` spanning 32 chips on this edge. **Prohibited:** a single fused `reduce` over 32 partials (compiler-chosen association), or any `all-reduce` with `replica_groups` of size 32 on the layer-boundary edge.

**Why bits preserved & topology-local.** The association is pinned to the accepted tree, so the rounded BF16 equals accepted for *every* input (not just the captured prompt), satisfying the "not merely for the captured prompt" requirement. LP2/LP4 repeated comm only; no full-pod collective; one live row.

**Distinct from tombstones.** Not "another uniform tree / scalar or global correction": the tree is the *accepted-specific* association, verified against accepted HLO, not an arbitrary uniform tree. Not "full-pod hidden reconstruction": partials stay local. Not PP8/PP16/WS32 retry: the association is explicitly fenced, which none of those plans do.

**Tests.** Offline: replay the exact 32 accepted partial FP32 values (from accepted HLO constants/inputs, not from a callback) through `fixed_assoc_reduce` in the StableHLO interpreter; `check.expect_eq` == `0x27bd`. Compile-only: dump optimized HLO, confirm the barrier-add chain survived and no 32-wide collective is present; hash. Bounded: layer-0→1 single-row run.

**Falsification.** If no ordering of the 2-/4-chip partials reproduces `27bd` at the interpreter level, the difference is not association but conversion placement (→ M1) or layout (→ M3). If the accepted association is reproducible offline but the barriers are stripped in optimized HLO (issue #25399 failure mode), M2 is not robust on TPU and must move into a Mosaic kernel (→ M3).

**Risks.** Reassociation/compiler collapse: **moderate** — `optimization_barrier` is documented to block cross-barrier motion but has failed against copy insertion (#25399) and cannot pin layout (#23471); must verify in optimized HLO, not just StableHLO. Observer perturbation: none. Mixed authority: none if partials are candidate-coherent.

### M3 — Pallas/Mosaic explicit reduction+rounding kernel over LP2/LP4 remote DMA (RECOMMENDED: INVESTIGATE, strongest robustness)

**Hypothesis/role.** Because Mosaic kernels appear to XLA as an opaque `tpu_custom_call` that XLA "can't see or rewrite," a Pallas kernel is the only container in which the accepted accumulation order *and* the single BF16 conversion placement are guaranteed immune to AlgebraicSimplifier/layout/fusion. It is the fallback that makes M1/M2 robust when barriers are insufficient on TPU.

**Source-level sketch.**
```python
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax import lax

def boundary_kernel(local_ref, out_ref, send_sem, recv_sem):
    me = lax.axis_index('lp')                 # 2- or 4-chip local axis
    peer = lax.rem(me + 1, pl.num_programs('lp'))
    bsem = pltpu.get_barrier_semaphore()
    pltpu.semaphore_signal(bsem, device_id=peer); pltpu.semaphore_wait(bsem, 1)
    pltpu.make_async_remote_copy(local_ref, out_ref, send_sem, recv_sem,
                                 device_id=(peer,)).start()
    # ... accumulate partials in FP32 in a fixed order in VMEM, then:
    acc = fixed_order_accumulate(local_ref, out_ref)          # f32 accumulator
    out_ref[...] = lax.reduce_precision(acc, exponent_bits=8, mantissa_bits=7)  # single bf16 round
```
Call with `dimension_semantics=("arbitrary", …)` on the reduction axis (output window does not vary ⇒ must be "arbitrary"/ordered, per Pallas TPU docs) so the accumulation order is fixed, not parallelized across cores.

**Expected HLO signature.** A single `tpu_custom_call` (opaque Mosaic payload) on the boundary edge; the reduction/rounding are *inside* the kernel and invisible to XLA. **Prohibited:** any XLA-level `reduce`, `all-reduce`, or `convert` on that edge outside the custom call (would mean the boundary leaked back into XLA's reassociating domain).

**Why bits preserved & topology-local.** Accumulator dtype and the single `reduce_precision` are explicit in Mosaic; "arbitrary" dimension semantics forbid core-parallel reassociation. `make_async_remote_copy` restricted to a 2-/4-chip `lp` axis satisfies LP2/LP4-only. One row via a `(1,·)`/`(8,128)`-tiled block.

**Distinct from tombstones.** Not "Pallas-layout-only" family: this is not a layout trick — it is an explicit accumulation-order + conversion-placement contract with a live consumer (the boundary output). Not "gather-before-weight / output-ownership / non-rooted fusion": the kernel consumes the explicit weighted row and roots the boundary output. Not a repeated 32-chip collective.

**Tests.** Offline: `pl.pallas_call(..., interpret=True)` on CPU to check the accumulation order and rounding logic (mark: this is a *logic* check only, NOT a TPU numerical proof — CPU-interpret success must not be presented as TPU success, per constraints). Compile-only: dump optimized HLO; confirm exactly one `tpu_custom_call` on the edge and no stray XLA `reduce`/`convert`; hash the Mosaic payload. Bounded: single-row TPU run of layers 0–1, assert `0x27bd` and event-1 set/order.

**Falsification.** If the single-row TPU kernel still yields `26bd` despite fixed order + single round, then neither association nor conversion placement is the cause and the remaining candidate is a genuine MXU/layout tiling effect (investigate `(8,128)` sublane layout / transpose / subnormal-flush timing). If it yields `27bd`, M3 confirms and supersedes M1/M2 as the sealed implementation.

**Risks.** Compiler collapse: **low** (opaque custom call). Observer perturbation: none. Correctness/complexity risk: **high** — Mosaic kernels are hard to get bit-exact and hard to audit; DMA/semaphore reuse hazards documented. Mixed authority: must ensure the kernel reads only candidate-coherent state.

### M4 — Compensated auxiliary dependency via an explicit error-free transform (INVESTIGATE, likely REJECT as declared)

**Assessment of the declared `compensated_auxiliary_dependency`.** As declared it is under-specified (blocker 4). To be admissible it must be an *explicit* EFT (TwoSum/FastTwoSum) whose low-order term is (a) computed, (b) kept live through a real device consumer, and (c) *proven* to cancel exactly for the primary BF16 output. The literature is unambiguous that the primary danger is compiler reassociation erasing the compensation term (luma.gl; FastTwoSum exactness needs operand ordering `|a|≥|b|`, arXiv 2601.17198). On XLA this means every EFT step must be individually fenced with `optimization_barrier`, and the final primary must be a `reduce_precision` of the *uncompensated* sum so the compensation is provably dead for the primary bits while still forcing the accepted association through the auxiliary path.

**Source-level sketch (only admissible form).**
```python
def two_sum(a, b):                       # requires |a|>=|b| for FastTwoSum; use full TwoSum otherwise
    s = lax.optimization_barrier(a + b)
    z = lax.optimization_barrier(s - a)
    e = (a - (s - z)) + (b - z)          # low-order error term, exact under TwoSum
    return s, lax.optimization_barrier(e)
```
The `e` term is consumed only to steer the accepted association; the primary is `reduce_precision(s, …, 7)`.

**Why it is weak.** Blocker 4 stands: merely finding a live `add`/`subtract` does not prove exact cancellation. A certificate must bind to the *specific* TwoSum SSA quadruple and prove `s + e == a + b` symbolically (e.g. via the StableHLO interpreter on adversarial inputs) — otherwise it is indistinguishable from a decorative shadow. Given M1/M2/M3 achieve the same effect more directly, M4 is **not recommended** unless the accepted state is provably an *accurately-rounded compensated sum* (which the DB485 contract — plain FP32 accumulate then BF16 round — does not indicate).

**Falsification.** If the accepted boundary is a plain FP32 accumulation rounded to BF16 (per DB485), then no compensation term is present in the accepted semantics and M4 changes semantics ⇒ reject. Confirm by checking the accepted HLO has no error-term SSA.

**Risks.** Reassociation erasure: **high** (documented). Mixed authority / non-causal shadow: high. Recommend **reject as declared; investigate only if accepted HLO shows a compensated-sum structure.**

### M5 — `auxiliary_device_tuple_dependency` (REJECT as declared; salvageable only if made causal)

**Assessment.** As declared — "carry the accepted BF16 primary unchanged while an FP32 shadow remains live through a device-only auxiliary result/dependency" — this is precisely the tombstoned "unused/no-consumer shadow / non-rooted direct fusion." XLA's dataflow/DCE will delete a value with no effect on a rooted output; `optimization_barrier` keeps a value *scheduled* but does not make it *causal* to the primary. Buffer donation/`input_output_aliases` are embedded in StableHLO but govern memory reuse, not numeric causality. There is **no documented device-only side-effect-free construct that both (i) leaves the primary bit-identical and (ii) forces a physical numeric distinction** — those two requirements are contradictory unless the "shadow" actually feeds the primary (in which case it is M1/M2, not a shadow).

**Only salvageable form.** Make the auxiliary the *actual* rounding pin: the FP32 quantity is consumed by a `reduce_precision` whose output *is* the primary (collapsing M5 into M1). Then it is causal, rooted, and one-row.

**Falsification.** Dump optimized HLO; if the "shadow" SSA has no path to a rooted output, it will be (or will have been) eliminated ⇒ mechanism void. This is the decisive compile-only test.

**Risk.** Compiler collapse: **certain** for the non-causal form. **Reject.**

---

## Addressing the Six Admission-Proof Blockers

1. **Mutable `site-packages` / same-version replacement & import races.** Version strings are insufficient (documented general risk). Harden by content-hashing the actually-loaded native artifacts at runtime: hash `jaxlib`/`libtpu`/MLIR `.so` bytes via their loaded file paths (`Module.__file__`, `/proc/self/maps`) and the imported Python module source, and seal those digests into the certificate; pin with a lockfile + hash-checking install and, ideally, an immutable/read-only mount or content-addressed store. Record `jax.print_environment_info()` and the `libtpu` build hash. [Engineering hardening; no single primary XLA source — flag as such.]

2. **Source AST and StableHLO not mechanically linked.** Bind them: lower with `jax.jit(...).lower()` and hash *both* the canonicalized source AST and the resulting StableHLO in the same sealing step, and embed the source digest as a `frontend_attributes`/module metadata string inside the StableHLO so a rehashed source cannot reuse old HLO authority. Verify the embedded digest matches at load. `input_output_aliases`/donation attributes are already embedded in StableHLO [DOCUMENTED — jax PR #21978], establishing that arbitrary attributes ride along with the module; use the same channel for the source digest.

3. **Certificate can name an arbitrary op as the auxiliary source.** Hard-bind to the exact pre-BF16-convert transient: require the certificate to reference the specific `reduce-precision` (e8m7) instruction on the RMS-operand edge, identified by its optimized-HLO instruction fingerprint (operand SSA + result shape) obtained from `--xla_dump_to`/`hlo-opt`. Because `reduce_precision` is a unique, non-elidable op (Findings 1–2), there is exactly one such frontier op to bind to; assert its presence and position in optimized HLO, not just StableHLO.

4. **Compensated survivor distinguished only by "some live add/subtract."** Require a *proof of exact cancellation*: run the TwoSum SSA quadruple through `stablehlo-translate --interpret` with `check.expect_eq` over an adversarial input battery (including `|a|<|b|`, subnormals, ties) confirming `s+e == a+b` and that the *primary* equals `reduce_precision(s)` independent of `e`. Bind the certificate to the canonical auxiliary slice (the specific `s,z,e` instructions), not to any add. If the accepted HLO lacks a compensated structure, this blocker is moot because M4 is rejected.

5. **Content-derived micro-plan self-declaring nonlocal ranks as "local."** Bind groups to a sealed runtime physical-topology authority: query the real device coordinates (`jax.devices()[i].coords` / TPU `(x,y,z)` from the runtime) and verify each LP2/LP4 group is physically adjacent on the ICI torus before admitting the plan; seal the coordinate map. TPU v4 topology is a 3D torus with `(x,y,z)` coordinates [DOCUMENTED — TPU v4 ISCA 2023 arXiv 2304.01433; TPU architecture doc]. The certificate must include the measured coordinate tuples, not self-declared ranks.

6. **Capsule roles owner0/owner1 not tied to owner-axis indices; cache owner axis lacks a map.** Mechanically derive role names from `lax.axis_index(owner_axis)` at kernel build time and assert `role == f"owner{axis_index}"`; refuse to compile if a supplied prefix disagrees with the axis index. Emit and seal an explicit cache-owner-axis map (axis name → physical coordinate → owned cache slice) and verify it against the runtime topology (blocker 5). Swapping prefixes then fails the axis-index assertion.

---

## Recommendations (staged)

**Stage 0 — Decisive offline triage (minutes, no TPU).** Before any TPU run, settle association-vs-conversion with the StableHLO interpreter. Extract the accepted FP32 partials/operand for position 8,155 element 2,795 *from accepted HLO constants/inputs* (never a callback). (a) Apply `reduce_precision(e8m7)` (and, separately, a TPU-style `convert` with subnormal flush) to the accepted pre-round operand → if it yields `0x27bd` and the candidate's operand differs only by a retained-excess-precision FP32, **M1 is confirmed as root cause.** (b) If the operands are bit-identical FP32 but round differently, the difference is association → replay partials through `fixed_assoc_reduce` to find the accepted tree (**M2**). Benchmark to change plan: whichever reproduces `0x27bd` wins.

**Stage 1 — Implement M1 (default path).** Insert `reduce_precision(e8m7)` + `optimization_barrier` at the qkv-a frontier; compile this executable with `--xla_allow_excess_precision=false` (recall the default is `true`, which is the likely originating condition). Verify in optimized HLO (blocker 3 binding). Gate: element 2,795 == `0x27bd` AND event-1 selected set + tie order exact, on a single-row layers-0→1 run. Keep the current PP16 as the **bit-exact fallback**, default-off for the new op.

**Stage 2 — If barriers strip in optimized HLO (issue #25399 failure mode) OR association is the cause, escalate to M3 (Mosaic).** Encapsulate the accepted accumulation order + single BF16 round in a `pl.pallas_call` over the LP2/LP4 axis with `make_async_remote_copy`; verify exactly one `tpu_custom_call` on the edge. Gate: same single-row `0x27bd` + event-1 criteria; seal the Mosaic payload hash.

**Stage 3 — Only after Stage 1/2 passes, widen** to full DSA sequence (positions up to 8,155), then WS32's seven event-1 swaps, confirming set equality/total order/ties via `lax.sort(is_stable=True)`/`lax.top_k` semantics. Do NOT run a full decoder first.

**Thresholds that change the plan.** If Stage-0(a) does not produce `0x27bd` under either RNE or TPU-flush rounding, drop M1 and pursue M2/M3. If Stage-0(b) finds no association reproducing `0x27bd`, the cause is layout/tiling — investigate `(8,128)` sublane layout, transposes, and subnormal-flush timing before writing more kernels. If M3's single-row kernel still yields `26bd`, escalate to an MXU/layout numerics investigation (not another scorer/top-k family — tombstoned).

**Reject now:** M5 as declared (non-causal shadow → certain DCE); M4 as declared (no compensated structure in DB485 contract). Do not re-run unchanged PP8/PP16/WS32.

---

## Caveats

- **`optimization_barrier` is a documented boundary but has documented failure modes** (XLA-inserted copies #25399, cannot pin layout #23471, gradient-flow surprises pytorch/xla #3486). M2's robustness on TPU is therefore *uncertain*; treat the optimized-HLO check as mandatory, and treat M3 as the robust fallback. [INFERENCE where noted.]
- **RNE vs subnormal-flush.** `reduce_precision(e8m7)` performs round-to-nearest-even but the interpreter does not model TPU's documented BF16 subnormal flush-to-zero; if element 2,795 is near the BF16 subnormal boundary, an interpreter `reduce_precision` and a TPU `convert` can disagree. Always cross-check both in Stage 0. [DOCUMENTED conversion semantics; INFERENCE on interpreter divergence.]
- **No documented device-only construct simultaneously preserves primary bits and forces a numeric distinction without being causal.** This is why M5 is rejected; if future XLA exposes such a construct it would need re-evaluation.
- **CPU/interpreter success is not a TPU numerical proof.** Per the task's own constraint and the Pallas docs' `interpret=True` caveat, Stage-0 interpreter results and `interpret=True` runs validate *logic/association*, not TPU rounding; the `0x27bd` gate must ultimately be met on a single-row TPU execution.
- **MXU internal numerics are documented only at the level "BF16 in, FP32 accumulate," with round-to-nearest-even/flush conversion.** The exact internal accumulation tree of a single MXU pass is fixed in silicon but not publicly specified to bit level; if M1/M2/M3 all fail, the residual MXU-tiling hypothesis cannot be resolved from public primary sources and would require Google-internal documentation.
- **DSA tie semantics** rely on `lax.top_k`/`lax.sort` stability being preserved through lowering; a JAX issue notes sorted-order for `top_k` is not fully documented (jax #27594), so the certificate should assert tie order empirically on the sealed executable rather than assume it.
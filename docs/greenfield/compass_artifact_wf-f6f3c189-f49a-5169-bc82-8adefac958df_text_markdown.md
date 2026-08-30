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

# Gate D: the scalar frontier — v2, corrected

**Status date:** 2026-08-30
**Supersedes:** `GATE_D_SCALAR_FRONTIER_ANALYSIS.md` (v1). v1 contains six substantive errors, listed in §E.
Do not cite v1.
**Type:** analysis input, not mechanism authority.
**Authorizes:** no compilation, no JAX work, no TPU run.

Marking: **[GIVEN]** = from your handoff/adjudication; **[DERIVED]** = IEEE-754 consequence, no hardware
assumption; **[MEASURED]** = numerically simulated here, method stated; **[INFERENCE]** = could be wrong.

---

## E. Errata against v1

| # | v1 claim | Status | Correction |
|---|---|---|---|
| E1 | f32(bf16)+f32(bf16) exact when exponent gap ≤ 15 | **off by one** | ≤ **16**. Exhaustive check over all 128×128 significand pairs: 0 inexact at g=16, 8192/16384 inexact at g=17. |
| E2 | flips ≈ 6144·ε·2⁸ | **wrong constant** | E[value/ulp] under log-uniform bf16 mantissa = 128/ln2 = **184.66 (2⁷·⁵³)**, not 256. v1 overestimates by 1.39×. |
| E3 | ε in variance propagates directly to the scale factor | **missing chain rule** | s = (var+eps)^(−1/2) ⟹ ε_s = **ε_v / 2**. v1 omitted the factor of two throughout §3. |
| E4 | "predicted 1.1 flips, observed 1" | **spurious fit** | Used a *worst-case bound* as a typical magnitude. Simulated realistic associations give **0.04–0.6** flips, not 1.1. |
| E5 | a row cause "would perturb many elements, predicting many flips, not one" | **wrong, and load-bearing** | One flipped bf16 element upstream perturbs `summed_fp32` at that index by ~2⁻⁸ relative and flips **exactly one** normalized element. The flip count does **not** discriminate row-cause from scalar-cause. If anything it slightly favours the row cause. **Claim 2 of v1 does not stand.** |
| E6 | interval intersection collapses "quite possibly to one" f32 value | **wrong by ~25×** | Simulated width **≈ 25 f32 ulps** (analytic estimate 21). A 1–8 ulp difference is not resolvable. v1's §4 discrimination table requires resolution the method cannot deliver and is void. |
| E7 | "exactly two loci of freedom" | **incomplete** | Also: FMA contraction in the sum-of-squares; `x·rsqrt(v)` vs `x/sqrt(v)`; `rsqrt(v)` vs `1/sqrt(v)`. |
| E8 | §3.1 suspect 3 text | **self-contradictory sentence** | v1 wrote "1/6144 is exact in binary (… not a power of two and is inexact)". The conclusion is right — 6144 = 2¹¹·3, 1/6144 is inexact in f32, and `sum/n` vs `sum·(1/n)` differ in ~34% of random cases — but the sentence as written is nonsense. |
| E9 | §6 "error-free transformation" for partial sums-of-squares | **incomplete** | TwoSum alone is insufficient: the *squares* are themselves inexact in f32. Exactness requires exact products (TwoProduct via FMA, or Dekker splitting), which may not be expressible without FMA availability. |

**What survives v1 intact:** the freedom audit (§1, once E7 is folded in) and Claim 1 (§2, with E1 fixed).
Claim 1 is the real contribution. Claim 2 (§3) should be deleted, not repaired — see §3 below.

---

## 1. Where freedom actually lives (corrected)

Accepted program **[GIVEN]** (your adjudication §2):

```python
summed_fp32           = float32(hidden_update_bf16) + float32(residual_bf16)
carried_residual_bf16 = bfloat16(summed_fp32)
variance_fp32         = mean(square(summed_fp32))
normalized_fp32       = summed_fp32 * rsqrt(variance_fp32 + epsilon)
weighted_output_bf16  = bfloat16(bfloat16(normalized_fp32) * weight_bf16)
```

| Step | Association freedom | Implementation freedom |
|---|---|---|
| `summed_fp32` | **none** — one correctly-rounded binary add | none (absent non-conforming contraction) |
| `carried_residual_bf16` | none | subnormal flush only |
| `variance_fp32` | **yes — full** | **FMA contraction** of `square`+accumulate; `sum/n` vs `sum·(1/n)` vs running mean |
| `+ epsilon` | none | order relative to the divide |
| `rsqrt` | n/a | **yes** — `rsqrt` vs `1/sqrt`, and approximation quality |
| `summed * s` | none | **yes** — `x·rsqrt(v)` vs `x/sqrt(v)` are different operations |
| both `bfloat16(...)` | none | subnormal flush |

**[DERIVED] Freedom is confined to the variance scalar and its consumption.** Every row-level step is
uniquely determined by its inputs. This still refutes M1: rounding `summed_fp32` perturbs a value that had no
latitude to begin with.

**[MEASURED] Magnitudes of each freedom**, expressed as relative error in `variance` (f32 ulps, u = 2⁻²⁴),
mean over 40 random 6144-rows; and the resulting expected count of bf16 flips in the normalized row using the
corrected formula `flips = 6144 · 184.66 · ε_v/2`:

| Freedom | Δ in variance | Expected flips |
|---|---|---|
| sequential vs binary tree | 16.7 ulps | 0.57 |
| binary tree vs 16 partials | 1.2 ulps | 0.04 |
| sequential vs 16 partials | 16.8 ulps | 0.57 |
| 2 partials vs 16 partials | 8.2 ulps | 0.28 |
| FMA-contracted vs separate-round squares | median 0, max 4 ulps | ≤ 0.13 |
| `sum/n` vs `sum·(1/n)` | ≤ 1 ulp | 0.03 |
| `rsqrt` off by 1–2 ulp | (direct on s) | 0.07–0.14 |

Note FMA contraction turns out to be a **weak** suspect, not a strong one: once the accumulator is large
relative to a single square, pre-rounding the square usually does not survive the accumulator's own rounding.
I expected the opposite before measuring.

---

## 2. Claim 1 — the missing FP32 operand is derivable (stands, with E1 fixed)

**[DERIVED]** `float32(bf16)` is exact — bf16 is bit-for-bit the high half of f32, same 8-bit exponent. The
f32 `add` of two f32 values is correctly rounded and uniquely defined (IEEE-754-2008 §5.1): one binary op, no
association question, no hardware latitude.

**[DERIVED, exhaustively verified]** The sum is **bit-exact** whenever the operands' exponents differ by
**≤ 16**; for larger gaps it is RNE-rounded and still unique. (Significands `Ma·2^g + Mb` with `Ma,Mb ∈ [128,255]`
span at most `g+8` bits; `g+8 ≤ 24`.)

**Consequence:** given accepted BF16 `hidden_update` and BF16 `residual` at layer 1 / position 8,155,
`accepted_summed_fp32[6144]` is computable offline in numpy with a correctness proof that does not depend on
TPU behaviour, XLA passes, or executable identity. It is a *derivation*, not an observation, and should be
sealed as a distinct authority class — never labelled as captured bytes.

Conditional on those two BF16 rows being held with sealed authority. If they are not, the ask shrinks from an
unobservable FP32 transient to two BF16 rows sitting at real program boundaries.

---

## 3. Claim 2 of v1 is withdrawn

v1 argued the single-element miss count fingerprints a scalar cause. **[MEASURED]** it does not:

- A scalar cause with a realistic association difference predicts **0.04–0.6** flips.
- A row cause — one bf16 element of `hidden_update` or `residual` differing by one bf16 ulp — perturbs
  `summed_fp32` at that index by ~2⁻⁸ relative and predicts **exactly 1** flip.
- Observed: 1. The likelihood ratio does not favour the scalar hypothesis; it mildly favours the row cause.

More importantly, the statistical argument was **unnecessary**. It was a workaround for not having
`summed_fp32` — but §2 says you can derive it. **Derive it on both sides and compare directly.** If accepted
and candidate `summed_fp32` are bit-identical, the row cause is dead outright and the cause is scalar. If they
differ, you have the index and the magnitude immediately. That is a decisive test, not an inference, and it
replaces all of v1 §3.

---

## 4. The inversion gives a specification, not a measurement (corrected, and better)

**[MEASURED]** Inverting the accepted bf16 row for the scale factor `s` — for each `i`, the set of `s` with
`bf16_RNE(f32(summed[i]·s)) = accepted_bf16[i]` is an interval; intersect over 6144 — yields an interval of
width **≈ 25 f32 ulps**, not a point. v1 claimed near-uniqueness; that was wrong.

But the limitation is the answer. Read it forward instead of backward:

> **Any variance computation whose `s` lands inside that ~25-ulp window reproduces the accepted normalized
> BF16 row bit-for-bit** — and therefore reproduces every downstream watchpoint that derives from it
> (q-a, query, head, current key, scorer, event-1 set and tie order).

So Gate D's layer-1 boundary does **not** require bit-equality of the variance. It requires landing in a
window whose location is recoverable **without observing accepted anything** — only from the accepted bf16
row you already hold plus derived `summed_fp32`. That is your §5.1 exit 2, stated as an engineering target
with a tolerance.

Note the scale: realistic association differences are 1–17 ulps against a ~25-ulp window. **Most association
choices already land inside it.** Which sharpens the real question — not "how do we match accepted's tree"
but "why is the candidate outside a window that most trees fall inside?" A 30-ulp-scale deviation is larger
than ordinary association drift and points at something structural (a different `rsqrt` form, `x/sqrt` vs
`x·rsqrt`, a divisor placement, or a genuine row difference), not at tree choice.

Two caveats on the inversion, both implementation-level:
- The accepted path double-rounds (`f32` multiply, then `bf16` convert). Interval boundaries must be computed
  through both roundings. Both are monotone, so the preimage is still an interval.
- Skip `summed[i] == 0`; handle `summed[i] < 0` by orientation flip.

**Falsifier, unchanged and still valuable:** if the intersection is **empty**, no single scalar explains the
accepted row given derived `summed_fp32` — so the row itself differs and the frontier is upstream.

---

## 5. Corrected test ladder

**T1 — derive and compare `summed_fp32` on both sides (minutes).** The decisive test. Compute
`float32(hidden_update_bf16) + float32(residual_bf16)` for accepted and for candidate at layer 1,
position 8,155. Record per-element exponent gaps (flag any > 16 as RNE-rounded rather than exact).
- **Identical ⟹ row cause dead, scalar cause confirmed, frontier is one f32 number.**
- **Differ ⟹ frontier is upstream of this function; report the differing indices and go there.**
This replaces v1's T0/T1/T2 and subsumes the whole statistical argument.

**T2 — invert the accepted row for the admissible `s` window (minutes).** Per §4. Report interval
endpoints and width in ulps. Empty ⟹ T1's premises are violated somewhere.

**T3 — locate the candidate's `s` relative to the window (minutes).** Invert the *candidate* row the same
way. The two windows are disjoint if the rows differ. The gap between them is a lower bound on the deviation
you must close; the sum of widths is an upper bound.

**T4 — forward-simulate the freedom space (minutes).** For each combination in §1's freedom table
(association × FMA contraction × divisor placement × `rsqrt` form × `x·rsqrt` vs `x/sqrt`), compute `s`, apply
to derived `summed_fp32`, round, and compare bit-for-bit to the accepted row. Enumerate which combinations
land inside the T2 window. Expect **many** to qualify — that set is the space of admissible implementations,
and any member of it closes the boundary.

**T5 — deleted.** v1's T5 asked whether association matters at all; §4's window analysis answers it directly
and T4 subsumes it.

---

## 6. Mechanism status

v1's `exact_partial_variance_recombination` is **downgraded from a proposal to a contingency**, for two
reasons: E9 (exact accumulation of squares needs exact products, not just TwoSum, and that may not be
expressible without an FMA primitive), and §4 (if the window is ~25 ulps wide and ordinary associations land
inside it, an exotic exactness mechanism is probably unnecessary — T4 will say). Do not develop it until T4
shows that no ordinary member of the freedom space lands in the window.

If T4 does show that, the fingerprint and the "why it is not M4" argument from v1 §6 still hold; add the
exact-product requirement to the sketch.

---

## 7. What is still unknown

- Whether the two accepted BF16 operands are held with sealed authority. All of §2 is conditional on it.
- The accepted `rsqrt`'s accuracy and form. Not resolvable from public TPU documentation; T4 resolves it
  empirically instead.
- Whether the layer-1 boundary is the *only* place this happens. Nothing here addresses layers 2–77.
- Everything in §1's magnitude table is measured on **synthetic Gaussian rows**, not your actual data. The
  ordering of the magnitudes should be robust; the absolute numbers should be recomputed on the real row.

---

## 8. Sequence

1. **T1.** It is decisive and costs minutes. Everything else is contingent on its outcome.
2. If T1 says scalar: **T2 → T3 → T4**. T4 likely ends with a set of admissible implementations.
3. If T1 says row: stop, discard §§3–6, and re-bisect upstream of `fused_add_rms_norm`.
4. No compilation, no TPU, either way.

**Bottom line, restated after correction:** v1's second claim was wrong and its inversion was over-resolved.
What survives is stronger than what I discarded — `summed_fp32` is derivable rather than unobservable, and the
accepted row specifies an admissible ~25-ulp window for the scale factor rather than demanding bit-equality of
the variance. Both are testable offline today. The flip-count statistics were a detour and should be dropped.

# GLM-5.2-FP8 → TPU-v4 Port, Gate D: Deep Research on Directions (A)–(H)

## TL;DR
- The one-ULP miss is most likely a **matmul-precision/algorithm mismatch** (Direction A): on TPU the default `dot_general` precision runs float32 inputs through a bf16 MXU path, and `HIGH`/`HIGHEST` map to documented multi-pass bf16 decompositions (bf16_3x / bf16_6x); a mismatch versus the GPU reference at the RMSNorm/QKV boundary can plausibly produce exactly a 1-ULP bf16 difference — this is the highest-value, cheapest lead to test offline.
- **Direction D likely invalidates a strict bit-exactness gate**: the accepted GPU oracle's DSA top-k is selected by operators whose tie ordering is *explicitly not guaranteed* (PyTorch `torch.topk` on CUDA) or that GLM-5's own report calls "non-deterministic" (SGLang's CUDA top-k). If the oracle itself is not run-to-run bit-stable at the selection boundary, bit-equality is the wrong gate; a **selection-set + tie-margin equivalence gate (Direction C)** is the defensible target.
- Non-perturbing observation (Direction B) is achievable via `--xla_dump_hlo_snapshots` + `run_hlo_module`/`hlo-opt` replay and `isolate_hlo` sub-module extraction, but sub-module bit-equivalence to the full module is **not guaranteed** across fusion/layout/scheduling boundaries — soundness must be established, not assumed.

## Key Findings
1. **(A)** JAX exposes both the legacy `precision` (DEFAULT/HIGH/HIGHEST) and the newer `DotAlgorithm`/`algorithm=` API with named presets (`BF16_BF16_F32`, `_X3`, `_X6`, `_X9`, `TF32_TF32_F32`, `F32_F32_F32`), lowering to StableHLO `dot_general`'s `algorithm` attribute. On TPU, DEFAULT=bf16, HIGH=bf16_3x, HIGHEST=bf16_6x. A precision mismatch is a documented, plausible source of a 1-ULP difference.
2. **(B)** `--xla_dump_hlo_snapshots` captures module + executable inputs/outputs (not arbitrary intermediates); replay via `run_hlo_module`/`multihost_hlo_runner` with `--run_xla_backend_only` or `--xla_disable_all_hlo_passes`. `isolate_hlo` extracts a sub-module ending at a chosen instruction. AOT via `jax.jit(...).lower().compile()` + `serialize_executable` gives a deterministic, re-runnable executable.
3. **(C)** There is a mature literature on top-k / margin certification and FP-sound certification (Lipschitz-based FP certification, certified top-k). If the k-th vs (k+1)-th score gap exceeds a rigorously computed error bound, the selected set is invariant — computable with interval/affine arithmetic (Gappa, FPTaylor, Daisy).
4. **(D)** Reference oracle is NOT guaranteed bit-stable: cuBLAS/cuBLASLt reproducibility requires fixed workspace and same-shape/algorithm; `torch.topk` CUDA tie order is explicitly "not guaranteed … may vary across different invocations"; GLM-5's report calls SGLang's CUDA DSA top-k "non-deterministic." Index scores are FP32; k=2048.
5. **(E)** Prior art (MaxText, DiFR, deterministic-inference work) frames cross-backend equivalence as *distributional/logit divergence* or *token-DiFR*, not bit-equality; bit-identical cross-backend results are generally not expected.
6. **(F)** TPU v4 uses `(8,128)` tiling with a `(2,1)` second-minor sub-tile for bf16 packing; convert placement and reduction tree shape can be affected by layout/fusion; ICI is a 3D torus (6 neighbors) with twisted-torus variants; physical coordinates are queryable from the runtime.
7. **(G)** SMT FP theory (Z3/CVC5), Alive2/MLIR-TV translation validation, and differential FP tooling (FLiT, Varity, pLiner, FPDiff, Herbie) can prove/refute equivalence of two reduction trees or error-free transforms — with caveats (Z3 FP is slow; Alive2 unrolls loops once).
8. **(H)** Concrete hardening exists: pip `--require-hashes`, Sigstore/cosign verification, `/proc/self/maps` + build-ID checks of loaded objects, `frontend_attributes`/metadata sealing of source digests into StableHLO, and runtime topology attestation via device coordinates.

## Details

### (A) Dot/matmul precision and algorithm control as the source of the 1-ULP miss

**DIRECTLY DOCUMENTED.** JAX's `jax.default_matmul_precision` / `jax.config.update("jax_default_matmul_precision", ...)` and the per-op `precision=` argument to `jax.lax.dot_general` control float32 matmul behavior on TPU. Per the JAX `jax.lax` docs, the enum semantics are device-dependent and TPU-specific:
- **DEFAULT / "fastest"**: "On TPU: performs float32 computations in bfloat16."
- **HIGH**: "On TPU: performs float32 computations in 3 bfloat16 passes."
- **HIGHEST**: "On TPU: performs float32 computations in 6 bfloat16" passes.

The documentation states this "only has an effect on float32 computations, and does not affect the input/output datatypes." Critically (GitHub discussion #18938): for **bf16 inputs**, `dot_general`'s `precision` argument has essentially no effect ("You're doing bf16 dot products, which will always be done at bf16 precision"); to accumulate in float32 you must pass `preferred_element_type='float32'`. The user reported no difference between HIGH and DEFAULT for bf16 inputs, and only a HIGHEST-vs-others difference for float32 dtype. **INFERENCE:** this means for a bf16 QKV/RMSNorm-consumer matmul, the lever that changes bits is `preferred_element_type` (accumulation type) and whether the *inputs* were upconverted to f32 before the dot — not `precision` on already-bf16 operands.

**DIRECTLY DOCUMENTED.** The newer API: `jax.lax.DotAlgorithm.Preset` enumerates `BF16_BF16_F32`, `BF16_BF16_F32_X3` (10), `BF16_BF16_F32_X6` (11), `BF16_BF16_F32_X9` (12), `TF32_TF32_F32` (13), `TF32_TF32_F32_X3` (14), `F32_F32_F32` (15), `F64_F64_F64` (16). Per the JAX docs, "The _X3 suffix indicates that the algorithm uses 3 operations to emulate higher precision"; X6 uses 6 and X9 uses 9. This is exposed through `jax_default_matmul_precision` too (enum values include all the above plus `'default'`, `'high'`, `'highest'`, `'bfloat16'`, `'tensorfloat32'`, `'float32'`).

**DIRECTLY DOCUMENTED (StableHLO spec).** `stablehlo.dot_general` carries an optional `DotAlgorithm` attribute with fields `lhs_precision_type`, `rhs_precision_type`, `accumulation_type`, `lhs_component_count`, `rhs_component_count`, `num_primitive_operations`, `allow_imprecise_accumulation`, plus `precision_config`. The spec states `algorithm` and `precision` are mutually exclusive, and that if an algorithm is not supported on hardware "an error should be raised as opposed to falling back to an alternative." The spec explicitly documents the decomposition structure: "bf16_6x: each input is decomposed to 3 bf16 components, then 6 dot operations are done on those components, and the result is accumulated in f32." Supported algorithm values are enumerated in `xla_data.proto > Algorithm`.

**DIRECTLY DOCUMENTED (history).** JAX PR #7859 changed the TPU default matmul precision to "highest" precision (multi-pass) after users reported the "low-quality-by-default behavior to be a footgun"; note the effective default has varied across JAX versions, so **the accepted GPU reference and the TPU candidate may silently disagree** unless precision is pinned. Issue #18934 documents that `Precision.DEFAULT` means FP32 on V100, TF32 on A100/H100, and BF16 on TPU — "a major user footgun for porting JAX functions between TPU/GPU."

**The 3-pass / 6-pass structure (INFERENCE from documented decomposition).** bf16_3x computes cross-products of two bf16 splits of each operand (A = A_hi + A_lo), accumulating A_hi·B_hi + A_hi·B_lo + A_lo·B_hi in f32 (dropping the A_lo·B_lo term); bf16_6x decomposes into 3 components with 6 products. The rounding/association structure differs from a native f32 FMA, so **HIGHEST on TPU v4 is not guaranteed to be bit-identical to a GPU f32 (or TF32) matmul**, and can differ by 1 ULP at an RMSNorm/QKV output boundary. This is consistent with a `27bd` vs `26bd` (one-ULP) bf16 difference.

**Could a precision mismatch produce exactly this 1-ULP difference? INFERENCE: yes, plausibly.** The post-RMSNorm normalized element is the product of (a) an f32 reduction (sum of squares over 6144) and (b) a matmul (the weighted row / QKV projection). If the reference computes the projection in one accumulation order/precision and the TPU candidate uses a different bf16-pass decomposition, the f32 intermediate can differ in its last bit and round to a different bf16 value. This is the **single cheapest and highest-probability lead.**

**XLA TPU flags (DIRECTLY DOCUMENTED existence; specifics require source inspection).** The `precision_config` `operand_precision` array appears in HLO backend configs (e.g., `"operand_precision":["DEFAULT","DEFAULT"],"algorithm":"ALG_DOT_..."`). Debug/determinism flags live in `xla/debug_options_flags.cc` and `xla.proto`/`xla/xla.proto`. **NOT FULLY SOURCED:** an exhaustive list of `xla_tpu_*` matmul/reduction-precision flags — these are best enumerated directly from `xla/debug_options_flags.cc` at your pinned XLA commit rather than from secondary docs.

**Smallest tests for (A):**
- *Offline (seconds):* Compile a single `dot_general` reproducing the QKV/consumer projection with each of DEFAULT/HIGH/HIGHEST and each `DotAlgorithm` preset; dump StableHLO with `.lower()` and diff the `algorithm`/`precision_config` attributes and the emitted TPU `convert`/tiling. **Falsification:** if every precision setting yields the identical accepted `27bd` bit (or none does), the miss is not precision-driven.
- *Compile-only:* `jax.jit(fn).lower(*avals)` and inspect the StableHLO `dot_general` attribute to confirm which algorithm the candidate is actually emitting versus what you intend.
- *Bounded numerical:* Feed the recorded FP32 RMS operand (from Direction B) into the isolated dot under each preset and compare the single element 2,795 against `27bd`. **Falsification:** a preset reproduces `27bd` exactly → fix is to pin that algorithm; default-off with bit-exact fallback becomes trivial.

### (B) Non-perturbing observability via snapshots, replay, and slicing

**DIRECTLY DOCUMENTED.** `--xla_dump_hlo_snapshots` (xla.proto field 118): "every time an HLO module is run, we will dump an HloSnapshot (essentially, a serialized module plus its inputs) to the --xla_dump_to directory." The `HloSnapshot` proto (xla/service/hlo.proto) "Encapsulates HloProto together with the arguments, result, and execution_platform … for purposes such as analysis/replay/file-storage." **Key limitation (DIRECTLY DOCUMENTED):** it captures **executable-level inputs and outputs**, not arbitrary internal intermediates. `hlo_module_loader.h` has `LoadInputFromFile` that "Loads an HLO snapshot from file, only for its inputs."

**Replay (DIRECTLY DOCUMENTED).** `run_hlo_module` "operates on pre-optimization HLO, and by default bundles compilation, running and comparison." For an already-optimized module you must use `--run_xla_backend_only` or `--xla_disable_all_hlo_passes` (otherwise "XLA will try to recompile the HLO and this isn't supported … it will give you many strange errors"). `multihost_hlo_runner`/`hlo_runner_main` replays multi-device modules and honors `sharding=` annotations; `--hlo_argument_mode=uninitialized` avoids large input generation.

**Sub-module slicing (DIRECTLY DOCUMENTED).** `isolate_hlo` (xla/tools) "extracts a single HLO instruction (and its necessary context) into a new, smaller HLO module … a minimal, compiler-level reproducer." There is also an `hlo_isolation_test` CLI "For automated debugging and verifying numeric stability or mismatches across compiled HLO modules." A `hlo_slicer` exists in the XLA tools tree.

**Soundness of observation-by-equivalent-sub-module — INFERENCE with strong basis.** A prefix module ending at the pre-bf16-convert f32 RMS operand is **NOT guaranteed to reproduce the full module's bits**, because XLA re-runs layout assignment, fusion, and scheduling on the smaller module. Documented mechanisms that break bit-equivalence: (1) **fusion boundaries** — the full module may fuse the RMS reduction with the producing matmul (public XLA walkthroughs show RMS sum-of-squares fused with matmul_1 into one `kOutput` fusion), changing accumulation; a sliced module may not reproduce that fusion; (2) **layout assignment** — the `(8,128)(2,1)` tiling and any transpose/`convert` placement can change when the surrounding ops change; (3) **scheduling** — reduction tree shape can change. **Soundness condition:** the sliced module reproduces the full module's bits only if you (a) compile with `--xla_disable_all_hlo_passes`/`--run_xla_backend_only` on the *already-optimized* full-module HLO (so the instruction's fusion/layout is frozen), and (b) extract the instruction *with its enclosing fusion intact*, not the pre-fusion op. This turns "observation by equivalent sub-module" into a defensible technique only against the post-optimization dump.

**Bisection tooling (DIRECTLY DOCUMENTED existence).** `hlo-opt` runs individual passes; `interactive_graphviz` visualizes subgraphs; `xla/tools/hlo_module_loader` loads modules; `--xla_dump_hlo_pass_re=regex` dumps after specific passes. **NOT SOURCED as first-class named tools:** a dedicated `hlo_diff` and `hlo_bisect` — these appear in the XLA tools tree but I could not confirm their exact CLIs from primary docs; verify at your pinned commit.

**Non-root intermediate device buffer — NOT SUPPORTED without codegen change (INFERENCE).** Exposing a non-root intermediate via donation/aliasing changes the module's output set, which changes DCE/fusion/scheduling → a different compiled executable, which violates the "candidate-coherent state only / no change to compiled executable" constraint. The snapshot-replay path is the correct non-perturbing alternative.

**AOT determinism (DIRECTLY DOCUMENTED).** `jax.jit(f).trace(*avals).lower().compile()` (jax.stages / aot.md) produces a `Compiled` object that is "staged out of Python." `jax.experimental.serialize_executable.serialize`/`deserialize_and_load` serialize the exact executable for re-run "in another process or machine without … repeat[ing] the staging-out and lowering." `jax.export`/`export.serialize` seals the StableHLO. This gives a byte-stable executable you can re-run deterministically.

**Smallest tests for (B):**
- *Offline (minutes):* Run the candidate once with `--xla_dump_hlo_snapshots --xla_dump_to=DIR`; confirm the snapshot `.pb` contains the layer-1 executable's inputs. **Falsification:** if the RMS f32 operand is an *intermediate* of a larger fused executable (not an executable input), the snapshot won't contain it — you must fall back to post-optimization `isolate_hlo` of the enclosing fusion.
- *Compile-only:* `isolate_hlo --input=optimized.hlo --instruction_name=<rms_convert>` then `run_hlo_module --run_xla_backend_only`; diff the isolated module's layout/fusion against the full module to establish soundness before trusting the value.

### (C) Prove the 1-ULP difference CANNOT flip the top-k selection (strategy-changing)

**DIRECTLY DOCUMENTED literature.** The standard result: the top-k selected set is invariant under a perturbation if, for every retained index i* and every competitor j, the score margin exceeds the worst-case error, i.e. m_{j,i*}(x) > L·ε (real-arithmetic margin certification). "Lipschitz-Based Robustness Certification Under Floating-Point Execution" (arXiv 2603.13334) proves an **FP-sound** version: passing the FP check *implies* the classical real-arithmetic margin certificate (their Corollary 6.3), and the FP error term is "instance-dependent — varying with the input point, perturbation radius, floating-point format, and network complexity — and much smaller for higher-precision formats." "Certified Robustness for Top-k Predictions" (arXiv 1912.09899; Duke) derives a tight top-k certificate requiring Pr(l) to exceed the max-min over any competing k-set. There is also explicit work on **stability of top-k selection under weight perturbation** via a linear program that "enforces a margin ξ between the weight vector and the cell boundary" (arXiv 2603.04689).

**Application to the DSA indexer (INFERENCE, well-grounded).** The indexer score is I_{t,s} = Σ_j w_{t,j}·ReLU(q_{t,j}·k_s) (DeepSeek-V3.2 report, arXiv 2512.02556; GLM-5 report). A 1-ULP bf16 perturbation δ in one hidden element (index 2,795) propagates: (1) through RMSNorm (bounded by the RMS condition number and the normalization denominator), (2) through the linear projections producing q/k (bounded by the projection row norms), (3) through the ReLU (1-Lipschitz), (4) through the weighted sum over heads (bounded by Σ|w_{t,j}|). The resulting per-score error bound Δ is computable; if the gap between the 2048th and 2049th index scores exceeds 2Δ for the affected query position, **the selected set of 2048 positions is provably invariant**. This directly explains the WS32 observation: "20/20 correct tokens and exact event 0, then seven event-1 selected-position swaps" is exactly the signature of near-ties at the top-k cutoff, not a systematic error.

**What falsifies the certificate (DIRECTLY DOCUMENTED failure modes):** exact ties (Δ→ the tie-break decides, and the reference tie-break is not defined — see D), near-ties within 2Δ, and sentinel/masking handling. The certificate is *per-position*; a single position with a sub-2Δ gap breaks it there.

**Standard way to compute the bound (DIRECTLY DOCUMENTED tools).** Interval or affine arithmetic through the score pipeline via **Gappa** (bounds "compound rounding errors of expressions"), **FPTaylor**, **Daisy**, **Satire**, **PRECiSA**, **Rosa**; the VCs can be discharged by dReal/MetiTarski or SMT. For a per-position scalar margin this is a seconds-scale computation.

**Smallest tests for (C):**
- *Offline (seconds):* From the recorded index scores at the diverging query position, sort and compute the (2048th − 2049th) gap. **Falsification / go-no-go:** if gap > 2Δ (with Δ from a cheap interval-arithmetic pass), bit-exactness is unnecessary at that position; if gap < 2Δ, you have a genuine near-tie and must either raise indexer precision or accept selection-set tolerance.
- *Bounded numerical:* Run Gappa/FPTaylor on the single-score expression q·k → ReLU → weighted sum to get a rigorous Δ. **Falsification:** if the certified Δ already exceeds observed gaps at multiple positions, the 1-ULP chase cannot guarantee selection stability and the strategy must move to (D)'s tolerance gate.

### (D) Is the accepted oracle itself bit-stable? (strategy-changing)

**DIRECTLY DOCUMENTED — this is the decisive finding.**

*cuBLAS/cuBLASLt.* PyTorch reproducibility docs: several CUDA ops are nondeterministic unless `CUBLAS_WORKSPACE_CONFIG=:4096:8` or `:16:8` is set (CUDA ≥ 10.2); cuBLAS reproducibility further requires the same architecture, same library version, and same-shape calls (algorithm/split-k heuristics otherwise vary). Independent report (Ingonyama, "Solving Reproducibility Challenges in Deep Learning and LLMs," published Sep 22, 2024): identical cuBLAS GEMM code "differed with an error margin of 1e-4" across three machines — "Ubuntu 20 with an NVIDIA RTX 3090," "Ubuntu 22.04 with an NVIDIA RTX 4080," and "Centos with an NVIDIA L4" (spanning Ampere and Ada Lovelace), with "All machines were running CUDA Toolkit 12.0" — and "in the context of LLMs, where each token depends on the previous one, such errors quickly accumulate."

*cuDNN / atomics.* Per NVIDIA/PyTorch docs, some routines "use atomic operations … in a way that introduces truly random floating point rounding errors," and "across different architectures, no cuDNN routines guarantee bit-wise reproducibility." `scaled_dot_product_attention` fused backends (Flash/Efficient/cuDNN) have differing determinism characteristics.

*DSA top-k specifically (subagent-verified primary sources).*
- **Index scores are FP32.** DeepSeek-V3.2-Exp reference `inference/model.py`: `index_score = fp8_index(...)` where the TileLang kernel output is FP32 — confirmed verbatim from the repo (GitHub issue #43, `kernel.py`): the TileLang prim_func declares output `o: T.Tensor[(b, m, n), FP32]`, with model.py computing `index_score = fp8_index(q_fp8.contiguous(), weights, self.k_cache...)`. The top-k is `topk_indices = index_score.topk(min(self.index_topk, end_pos), dim=-1)[1]` with `index_topk = 2048`. Per the HuggingFace Transformers DeepSeek-V3.2 model docs: `index_topk (int, optional, defaults to 2048) — Number of top tokens selected by the indexer`, `index_head_dim ... defaults to 128`, `index_n_heads ... defaults to 64`; the docs confirm the indexer runs "in FP8 with a Hadamard (rotate_activation) transform ... the transformers port computes the same scores directly in bf16/fp32."
- **`torch.topk` on CUDA has no guaranteed tie order.** PyTorch `torch.topk` docs (verbatim, current PyTorch 2.9–2.13): "When using torch.topk, the indices of tied elements are not guaranteed to be stable and may vary across different invocations." A `stable=` flag (lower-index-wins) was added **CPU-only** (PR #88619); CUDA has no such guarantee. `torch.topk` is **not** listed in `torch.use_deterministic_algorithms` (neither forced deterministic nor error-raising).
- **GLM-5's own report** (arXiv 2602.15763) states, verbatim: the top-k results are "critical for RL stability," and "adopting a deterministic top-k operator effectively resolves this issue. Compared with the non-deterministic CUDA-based top-k implementation used in SGLang's DSA Indexer, directly using the naive `torch.topk` is slightly slower but deterministic." SGLang's TopK-V2 kernel "treated TopK as a selection problem … boundary candidates undergo exact FP32 radix selection" (LMSYS GLM-5.2 blog).
- **DeepSeek reference enforces cross-rank agreement but not run-to-run reproducibility:** model.py broadcasts rank-0's topk and asserts all ranks match (`dist.broadcast(topk_indices_, src=0); assert torch.all(topk_indices == topk_indices_)`) — a cross-GPU coherence check, silent on single-device run-to-run stability.
- **NOT FOUND:** any statement in the DeepSeek-V3.2 report (2512.02556) guaranteeing a canonical accumulation order or numerical determinism for the indexer/top-k.

**Interpretation (INFERENCE, high confidence).** Whether the accepted `27bd` is itself reproducible depends on which reference kernel produced it. If the oracle used SGLang's CUDA top-k, GLM-5's authors call it non-deterministic → a 7-position swap at event 1 may be within the reference's *own* run-to-run variance, making a bit-exact TPU gate **unachievable in principle**. If the oracle used naive `torch.topk` over distinct FP32 scores (no exact ties), it is effectively reproducible, and the swap indicates genuine near-ties. Either way, **strict bit-equality of selected-position sets is the wrong gate**; the right gate is: identical selected *set* (order-insensitive) with a certified tie-margin (Direction C), plus a documented tie-break.

**Equivalence criteria used by others (DIRECTLY DOCUMENTED).** DiFR (arXiv 2511.20621) uses **Token-DiFR** with fixed-seed Gumbel-Max, so "any disagreement … arises solely from numerical differences in the logits" — a divergence-detection rather than bit-equality gate. Deterministic-inference work (arXiv 2511.17826) reports "Average Maximum Probability Divergence" where 0 means "bitwise identical … across all evaluated runtime configurations," achieved only with batch-invariant + tie-break-invariant kernels (BIO+TBIK) — evidence that bitwise identity requires *purpose-built* kernels and is not the default.

**Smallest tests for (D):**
- *Offline (seconds):* Re-run the reference oracle twice on the identical input with the exact kernel stack and diff event-1 selected sets. **Falsification of the bit-exact gate:** if the reference disagrees with *itself* at any of the seven positions, abandon bit-equality and adopt the (C) set+margin gate.
- *Offline:* Inspect which top-k kernel the oracle used (torch.topk vs SGLang CUDA). **Decision:** SGLang-CUDA → non-deterministic per GLM-5 → tolerance gate mandatory.

### (E) Prior art: GPU→TPU parity

**DIRECTLY DOCUMENTED.** MaxText is the reference "optimization-free" JAX/TPU LLM stack; its docs emphasize that "the compilation environment must match the execution environment" (same `XLA_FLAGS`) for reproducible compiled artifacts, and support CPU/single-VM **pre-compilation** to flag issues without the target hardware. This is the practical mechanism for the compile-only checks above.

**DIRECTLY DOCUMENTED methodology.** The parity literature converges on **divergence metrics, not bit-equality**: DiFR's token-level divergence under fixed seeds; "Maximum Probability Divergence" (0 = bitwise identical only with batch/tie-invariant kernels). **INFERENCE:** for a 78-layer decoder, first-divergence bisection is done by dumping per-layer activations on both backends and locating the first layer/element exceeding a tolerance — which is exactly the user's current layer-1/pos-8155/elem-2795 finding. No public writeup I found claims *bit-exact* GPU→TPU parity for a 70B+ decoder; the realistic target is index-level (selection-set) parity.

**NOT FOUND:** a published, quotable GLM/DeepSeek-specific GPU→TPU bit-exact port writeup. Levanter/EasyDeL/tpu-inference exist as JAX/TPU stacks but I found no primary "numerical parity" methodology doc specific to them at bit level.

### (F) TPU v4 layout, tiling, and convert placement

**DIRECTLY DOCUMENTED.** XLA `tiled_layout`: for bf16 the tile is `(8,128)(2,1)` — "one element from an even row and one element from an odd row are laid out together and put in one 32-bit element … because TPUs work with 32 bit values natively and it is much more efficient to move data across the second most minor dimension." The `(8,128)` matches "the VPU's 8 sublanes × 128 lanes." The Ragged Paged Attention paper (arXiv 2604.15464) documents that a `BF16(12,128)` tensor "is padded to BF16(16,128) by XLA to align with a T(8,128) tile," and that placing a packing dimension in the second-minor dimension forces the minimum tile — i.e., **reshape/transpose/layout choices change where padding and packing occur**, which changes where `convert` is materialized.

**Convert sinking/hoisting across RMSNorm (INFERENCE, grounded).** XLA's `AlgebraicSimplifier` and fusion passes can move elementwise ops (including `convert`) relative to reductions; public profiling walkthroughs show the RMS sum-of-squares fused with the producing matmul into one fusion. **Whether XLA sinks an f32→bf16 convert across the RMS reduction is layout/fusion-dependent and can change the last bit.** NOT DIRECTLY SOURCED: a specific documented bug of this exact convert-across-RMSNorm reassociation — treat as a hypothesis to test by diffing the optimized HLO around the RMS reduction.

**Reduction order (INFERENCE from documented mechanism).** Public walkthroughs describe the length-reduction as a cross-lane reduction using the XLU transpose unit followed by a log₂(n) sublane rotation tree (rotate by 4, 2, 1). For length 6144 = 48×128, XLA emits a fixed-shape tree per compiled module; **the tree shape is deterministic for a fixed module/layout but differs from a GPU's accumulation order** — a documented source of the f32 RMS-denominator differing in its last bit.

**Topology (DIRECTLY DOCUMENTED).** TPU v4/v5p connect to 6 nearest neighbors forming a **3D torus**; "twisted torus" wraps Möbius-like to reduce average distance (JAX Scaling book). Physical coordinates are queryable: `jax.make_mesh`/`mesh_utils.create_device_mesh` order devices by physical topology; the concrete `Mesh` "includes physical device objects with … precise coordinates"; PyTorch/XLA's `_get_physical_tpu_mesh`/`_create_device_mesh_for_nd_torus` expose the same. **This is the mechanism for Direction H's topology attestation** — read each device's coordinate and verify a 2-/4-chip group is contiguous along one torus axis (i.e., a ring slice), per the Pallas distributed docs ("taking a slice along an axis of the pod … we have a ring of devices").

**Smallest tests for (F):**
- *Compile-only (seconds):* Dump optimized HLO for the RMS + consumer subgraph; grep the `convert` and `reduce` instructions and their `{...:T(8,128)(2,1)}` layouts. **Falsification:** if the candidate's `convert` sits on a different side of the `reduce` than the reference's semantic contract (f32 RMS → bf16 boundary), that is the bug — fixable with an `optimization_barrier` or explicit `convert` placement.
- *Offline (seconds):* Query device coordinates for the intended LP2/LP4 group. **Falsification:** non-adjacent coordinates → the "local" group is not physically a ring slice.

### (G) Formal and differential FP verification (deeper)

**DIRECTLY DOCUMENTED.** *SMT FP theory:* Z3 and CVC5 implement IEEE-754 bit-precise FP; you can encode two reduction trees and ask for a counterexample input where they differ, or prove a TwoSum/Dekker error-free transform is exact. **Caveat (DIRECTLY DOCUMENTED):** Alive2's authors note "Floating point operations, wide integer divisions, and complex memory operations are common culprits" for Z3 timeouts — so all-input proofs are feasible only for small trees; expect to bound width. Gappa can be embedded in an SMT solver for FP-error reasoning (Conchon et al., "Built-in Treatment of an Axiomatic Floating-Point Theory for SMT Solvers").

*Translation validation:* **Alive2** does bounded translation validation of LLVM IR (proves refinement or yields a counterexample), and has verified FP optimizations, but "unrolls loops once" and does not do interprocedural optimizations. **MLIR-TV** is "designed for MLIR-level verification" (contrasted with Alive2 for LLVM IR) — the relevant tool for StableHLO/MLIR-level FP-preservation questions. **INFERENCE:** neither is turnkey for full StableHLO modules, but MLIR-TV/Alive2-style refinement is the right frame for proving a specific XLA rewrite (e.g., convert-sinking) preserves FP semantics on a small extracted region.

*Differential/compiler-induced FP tooling (DIRECTLY DOCUMENTED):* **FLiT** (bisection algorithms to "locate the root causes of variability" across compilers/flags/hardware, with a trusted baseline), **Varity** (randomized cross-platform FP variation, used to study NVIDIA-vs-AMD differences), **pLiner** ("isolate lines of floating-point code for compiler-induced variability"), **FPDiff** (equivalence classes of library "function synonyms," finds divergence points), **Herbie** (rewrites for accuracy). These are CPU/GPU-source-level; **INFERENCE:** the XLA analog is `hlo_isolation_test` + `run_hlo_module` differential replay (Direction B), which is the FLiT-equivalent at HLO granularity.

**NOT FOUND:** a published paper specifically verifying that XLA/StableHLO transformations preserve FP semantics end-to-end; this remains an open area, so rely on bounded per-region checks.

**Smallest tests for (G):**
- *Offline (seconds–minutes):* Encode the candidate vs reference length-6144 reduction (or a downsampled proxy) in Z3 FP and ask for a differing input. **Falsification:** a counterexample proves the trees are not universally equal (expected); absence within a width bound supports equivalence on the restricted domain.
- *Bounded:* Use Gappa to bound the RMS-denominator error between tree orders; feed that Δ into the (C) certificate.

### (H) Supply-chain and evidence sealing

**(1) Pin/verify native libs (DIRECTLY DOCUMENTED).** pip `--require-hashes` mode with a fully specified requirements file enforces hash-pinned wheels for `jaxlib`/`libtpu` (the Sigstore install example uses exactly this pattern). **Sigstore/cosign** sign and verify blobs/containers (`cosign verify-blob … --bundle …`), with keyless OIDC and offline verification; sign the digest, not the tag. **INFERENCE (well-grounded on OS mechanisms):** to defeat same-version replacement and import races at runtime, read `/proc/self/maps` to enumerate loaded `.so` objects and compare their GNU build-IDs (and/or hashes) against the signed manifest; mount the libs read-only (read-only bind mount) or load from a sealed `memfd_create` image so they cannot be swapped post-verification. TUF provides the update-framework layer above Sigstore. **NOT DIRECTLY SOURCED:** a single doc combining all of these for jaxlib specifically — assemble from the primitives.

**(2) Bind source AST digest into StableHLO (DIRECTLY DOCUMENTED primitives).** StableHLO modules and ops carry `frontend_attributes` and `metadata` (op_name/source_file/source_line are already emitted, as seen in HLO dumps). **INFERENCE:** compute a digest of the source AST and inject it as a module-level `frontend_attributes` entry before `jax.jit(fn).lower()`, then `jax.export`/`serialize_executable` to seal it; on load, verify the digest matches the current source AST so "a rehashed source cannot reuse old HLO authority." The AOT `Compiled`/`Exported` object is the sealing boundary.

**(3) Attest physical topology (DIRECTLY DOCUMENTED mechanism).** Read device coordinates from the concrete `Mesh` (physical device objects "with precise coordinates") or PyTorch/XLA `_get_physical_tpu_mesh`; verify the declared LP2/LP4 group's coordinates are adjacent along one 3D-torus axis (a ring slice). **Falsification:** any group whose coordinates are non-contiguous is not physically local and the "local communication" claim is false.

**Smallest tests for (H):**
- *Offline (seconds):* `cosign verify-blob` the pinned `libtpu`/`jaxlib` and cross-check `/proc/self/maps` build-IDs at runtime. **Falsification:** any loaded object whose build-ID is absent from the signed manifest.
- *Compile-only:* lower the sealed module and assert the source-AST-digest `frontend_attribute` is present and correct. **Falsification:** digest mismatch → stale/forged HLO.
- *Offline:* dump device coordinates. **Falsification:** non-adjacent LP group.

## Recommendations

**Stage 0 — decide whether bit-exactness is even the right gate (do FIRST, cost: seconds–minutes).**
Run the reference oracle twice on the identical input and diff the seven event-1 selected positions (Direction D). In parallel, compute the 2048th-vs-2049th index-score gap at the diverging query position and a cheap interval-arithmetic Δ (Direction C). **Threshold that changes everything:** if the oracle disagrees with itself, OR if any gap < 2Δ, **abandon strict bit-exactness** and adopt a selection-set + certified-tie-margin gate. This is the highest-leverage step and could invalidate the entire bit-exact chase.

**Stage 1 — cheapest offline falsification of the mechanism (cost: seconds–minutes).**
Isolate the QKV/consumer `dot_general` and the RMS reduction; sweep `precision` (DEFAULT/HIGH/HIGHEST), `preferred_element_type`, and every `DotAlgorithm` preset; feed the recorded f32 RMS operand (from Stage 2) and check for `27bd` (Direction A). **Threshold:** a preset reproduces `27bd` exactly → pin it (default-off with bit-exact fallback is then trivial). Because bf16-input dots ignore `precision`, focus on `preferred_element_type` and pre-dot upconvert placement.

**Stage 2 — obtain the accepted f32 RMS operand without perturbation (cost: minutes).**
Dump `--xla_dump_hlo_snapshots` and, if the operand is an executable input, replay with `run_hlo_module --run_xla_backend_only`. If it is an intermediate, `isolate_hlo` the *enclosing fusion* from the *optimized* HLO and establish sub-module soundness by diffing layout/fusion against the full module (Direction B). Do NOT add a rooted output (that changes the executable).

**Stage 3 — confirm the layout/convert hypothesis (cost: seconds, compile-only).**
Diff optimized-HLO `convert`/`reduce` layouts around the RMS boundary (Direction F). If the candidate's f32→bf16 convert sits on the wrong side of the reduction versus the semantic contract, insert an `optimization_barrier`/explicit convert to enforce the contract.

**Stage 4 — certify or fix (cost: minutes).**
If Stage 0 permits a set+margin gate: run Gappa/FPTaylor to certify per-position tie margins (Direction C) and ship with a documented tie-break. If bit-exactness is required and Stage 1 found the preset, pin it and add the bit-exact fallback. Use Z3/MLIR-TV only for the specific small reduction-tree/error-free-transform equivalence you must prove (Direction G).

**Stage 5 — seal the evidence (cost: minutes, do before any long TPU run).**
Hash-pin + cosign-verify `jaxlib`/`libtpu`, verify `/proc/self/maps` build-IDs, seal the source-AST digest into the StableHLO `frontend_attributes`, and attest LP2/LP4 adjacency from device coordinates (Direction H).

**Do NOT** run the full 78-layer decoder as a first step; every stage above is an isolated offline/compile-only/bounded test.

## Ranked list of highest-value NEW leads (by expected information gain per unit effort)

1. **(D) Test the oracle against itself + identify its top-k kernel — STRATEGY-INVALIDATING.** Seconds of effort; can prove bit-exactness is impossible in principle (SGLang CUDA top-k is "non-deterministic" per GLM-5's own report; `torch.topk` CUDA tie order "may vary across different invocations"). This should be done before anything else because a positive result retires the whole bit-exact program.
2. **(C) Compute the k-th/(k+1)-th score gap vs a certified Δ — STRATEGY-INVALIDATING/REDIRECTING.** Seconds–minutes; if the gap exceeds 2Δ the 1-ULP miss provably cannot flip the selection, converting the problem from "eliminate the ULP" to "certify the margin." The 20/20-correct-tokens + 7-swaps signature strongly suggests near-ties, i.e., this is the likely resolution.
3. **(A) Sweep precision/algorithm presets on the isolated dot.** Seconds; highest probability of pinpointing the actual bit-level mechanism and yielding a one-line fix (pin `DotAlgorithm`/`preferred_element_type`).
4. **(B) Snapshot + `run_hlo_module --run_xla_backend_only` replay for the f32 RMS operand.** Minutes; unblocks Stages 1/3 by supplying the ground-truth operand without perturbing the executable — but only sound against optimized HLO with the enclosing fusion preserved.
5. **(F) Diff optimized-HLO convert/reduce layouts around RMSNorm.** Seconds, compile-only; directly tests the convert-sinking hypothesis and is fixable with `optimization_barrier`.
6. **(H) Topology attestation from device coordinates + lib build-ID verification.** Minutes; necessary for the admission-proof blockers and cheap, though it does not resolve the numerical question.
7. **(G) Z3/Gappa equivalence or bound on the specific reduction tree.** Minutes–hours; rigorous but narrow, best reserved for the one equivalence you must formally prove after (A)/(C) narrow the target.
8. **(E) Prior-art methodology.** Confirms the framing (divergence gates, not bit-equality) but yields no direct fix; lowest incremental gain.

**The two leads that can invalidate the current bit-exactness strategy are (D) and (C).** (D) attacks the *premise* that the oracle is a bit-stable target; (C) attacks the *necessity* of eliminating the ULP at all. Either alone is sufficient to justify replacing "bit-exact hidden state" with "identical DSA selected set under a certified tie-margin, with a documented tie-break."

## Caveats
- The claim that a precision mismatch produces *exactly* `27bd`→`26bd` is an INFERENCE consistent with documented multi-pass rounding, not a proven root cause; Stage 1 is designed to confirm or refute it.
- Sub-module bit-equivalence (B) is only sound against the *optimized* HLO with passes disabled and the enclosing fusion preserved; a naive prefix-module slice will generally NOT reproduce the full module's bits.
- Whether XLA sinks a convert across the RMS reduction, and the exact `xla_tpu_*` precision/determinism flags, must be confirmed at your pinned XLA commit from `debug_options_flags.cc`; I did not find an exhaustive primary list.
- The oracle-determinism finding (D) is the load-bearing result: if the reference used SGLang's CUDA top-k, GLM-5's authors call it non-deterministic, and a bit-exact gate is unachievable in principle. Confirm which reference kernel produced `27bd`.
- `hlo_diff`/`hlo_bisect` as named CLIs, and bit-level GPU→TPU parity writeups for GLM/DeepSeek specifically, were NOT found in primary sources.
- Several GLM version identifiers surfaced in results (GLM-4.5/4.6/4.7, GLM-5, GLM-5.2) with future-dated arXiv/blog stamps; DSA adoption and k=2048 are consistent across the GLM-5 line and DeepSeek-V3.2, but treat any single version-specific hyperparameter (e.g., `index_n_heads`) as requiring confirmation against your exact GLM-5.2-FP8 config.
- Performance characteristics were deliberately not inferred from papers, per scope.
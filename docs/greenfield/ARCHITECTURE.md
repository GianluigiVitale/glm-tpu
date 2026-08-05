# Greenfield architecture state

The binding design is `../glm-tpu-revolution.md`. This file records implemented structure, not a
replacement specification.

## Isolation

- Branch: `rewrite/topology-first-decode`
- Worktree: `/home/gianl/glm-tpu-topology-rewrite`
- Starting harness pin: `a4a17ac4e90b15f1994bd8b26917ef62daa52660`
- Starting legacy-oracle pin: `b3c25df47ac98783912dc658878181ec0a8ae16d`
- Greenfield modules: `glm_tpu/greenfield`; legacy execution is never imported.
- Branch-local `AGENTS.md`, root `goal.md`, and `docs/glm-tpu-revolution.md` supersede inherited TP32
  instructions. The owner's untracked main-worktree `AGENTS.md` remains untouched and has no authority
  in this worktree.

## Implemented contracts

`ModelGeometry`, `PhysicalDevice`, `PhysicalTopology`, `StageAssignment`, and `ExecutionPlan` are
frozen, typed, canonical-JSON serializable, round-trippable, and content-addressed with SHA-256.
Validation refuses incomplete schedules, topology ambiguity, layer gaps, device reuse, cross-host
local groups, wrong named-plan geometry, and incomplete memory classes.

Runtime discovery reads device id, JAX process, physical coordinates, core-on-chip, platform, kind,
and observed local order. TPU v4 exposes no usable `local_hardware_id`, so every process contributes
its actual `jax.local_devices()` order and the fleet gathers a complete mapping. No device ordering
is inferred from ids.

PP8 and PP16 groups are derived from physical coordinates. Stage order is a deterministic
Hamiltonian ring. Every stage boundary, including last-to-first token return, requires a distinct
physical-neighbor match for every transfer lane. PP16 first chooses adjacent two-chip pairs along
the physical length-two axis where available.

The HLO contract parser reads optimized textual XLA HLO and records physical replica groups,
source-target pairs, result/operand shapes, source metadata, counts, channels, and global-id
semantics. Policies reject missing/wrong groups, repeated full-pod residual synchronization, dead
batch-32 rows, and count drift. Full-pod collective diagnostics require an explicit non-promotable
policy; XLA all-to-all's absence of a global-id flag is a narrow named exemption while its exact
replica groups remain mandatory. Because XLA numbers groups in executable partition order, every
policy records the exact partition-to-physical-device assignment and reports both logical and
physical groups/pairs.

The dependent collective benchmark supports control, all-reduce, reduce-scatter, all-gather,
collective-permute, all-to-all, and fused tuple all-reduce over physical groups of 2/4/8/32 chips.
Rank-dependent state, nonlinear cross-iteration feedback, optimization barriers, bitwise output
checksums, and optimized-HLO count assertions prevent elision. Protected configs require a chain of
75, at least 200 warmups, at least 1,000 measured samples, and report complete distributions. TPU
optimized HLO is authoritative: small decode reduce-scatter currently fails closed because TPU-v4
XLA rewrites it to all-reduce, while the other supported operations preserve the exact contract.

## Transport mechanism

The model-free executable maps PP8 into four closed eight-stage physical lanes and PP16 into two
closed sixteen-stage lanes. One compiled global `shard_map` advances the rank-dependent payload
with exact neighbor `ppermute` pairs once per stage and returns it to its origin. Protected HLO and
fresh 64-core XPlanes prove exact 8/16 operations and no other collective, host/Ray/Python stage
dispatch, or model-equivalent compute. For `bf16[1,6144]`, the whole ring is `0.030 ms` PP8 and
`0.076 ms` PP16 above matched control at fleet-max p50.

## Exact sparse-layer implementation

An independent exactness-first batch-one MoE reference now exists under
`kernels/reference/moe.py`. It pins FP8 block dequantization, FP32 sigmoid/noaux_tc routing,
correction-bias selection-only semantics, lowest-expert-id ties, top-8 normalization, expert
ownership, shared-expert sharding, and post-reduction routed scale. Forced four-device CPU tests
prove distributed and adversarial single-chip expert concentration against the unsharded fallback.
The optimized CPU HLO has one four-rank stacked routed/shared all-reduce and no other collective.
Protected PP8 DB 417 and PP16 DB 418 prove the required `bf16[2,1,6144]` payload over exact
four-/two-rank groups, no other collective, real checkpoint load, exact routes, bounded tensors,
and per-chip HBM. PP16 initializes the known-good four-chip host subcube but arrays/executable live
only on its selected adjacent pair; its two-partition HLO and four active XPlane cores prove this.

A versioned one-layer-only packer now validates the exact layer-3 source leaf set and writes final
PP8 or PP16 identity ownership with byte/hash reconciliation. Its tiny-fixture tests pass. Real
PP8 artifact `greenfield_one_layer_pack_20260805T151828912346032Z` proves 1,544 leaves /
9,706,940,416 unique payload bytes across only shards 38–40, four independently hashed final files,
exact manifest reconciliation, and approved-bucket `SUCCESS`. PP16 artifact
`greenfield_one_layer_pack_pp16_20260805T172003732526347Z` writes two independently hashed final
files with the same source reconciliation and remote `SUCCESS`. These bounded artifacts are not
Gate B.
A separate raw-source PyTorch oracle now captures normal routes spanning all four PP8 slots and a
bias-forced all-eight-on-slot-2 case. It records decomposed expert, routed, shared, and final outputs
plus exact source/legacy hashes without importing greenfield JAX, packed weights, a model class, or
legacy execution.

No full checkpoint packer/loader, decoder, or serving path exists yet.
Protected TPU dependent-chain matrices now cover the dominant payload, required bf16 live-
residual/intermediate shapes, `f32[1,6144]`, and small `int32` routing metadata. FP8 is a checkpoint
weight-storage format here, not a numerically valid residual/reduction or stage-transfer payload;
the numerical contract uses bf16/f32 for those paths. The matrices show that 75 full-pod
all-reduces have a `3.941 ms` fleet-max
p50 and 75 full-ring nearest-neighbor permutes `0.791 ms`, versus `106.495 ms/token` attributed to
the legacy MoE combine region. PP8 DB 417 additionally measures the exact normal real sparse layer
at `0.696 ms` p50 and a single-chip-concentrated adversary at `1.134 ms` p50. This isolates legacy
arrival/layout/barrier behavior rather than raw small-payload ICI or local sparse compute as the
dominant loss. It is one-layer evidence, not token throughput. Both mandatory local-layer forms
now pass. PP8 normal p50 is `0.696 ms` versus PP16
`0.901 ms`; PP16 concentrated p50 is `1.076 ms` versus PP8 `1.134 ms`.
PP8 therefore remains the provisional leader. The mandatory local-layer prerequisite is complete;
full-checkpoint work is now authorized only through Gate B's ordered plan/manifest/pack/load gates.

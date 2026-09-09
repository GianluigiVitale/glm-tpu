# Expert-relative FP8 panels — next throughput candidate

2026-09-08. Main-agent source/byte inspection and independent reviewer
`/root/observer_identity_review` agree. This is a design, not a TPU performance result. DB597 has sealed; source freeze is over.

DB597 paired own2K is sealed:30.9739prompt tok/s. See
`../artifacts/prefill-paired-short-sealed-20260909.json`.

## Evidence and priority

Source `pallas/prefill_grouped_fp8.py` uses global row-aligned Megablox metadata,
row_tile8 and grid(Ntile,group_row_tile,Ktile). Experts sharing global row tiles
cause masked revisits. Weight dequantization executes per active route tile.
This confirms scheduled work, not measured HBM reload or utilization.

DB596 local original `fleet/rank0/phase_first.npz` matches archived ledger:
generation1788901524314158, size47677909, CRC32C QGGd7A==,
SHA bcaac31ae51c27bf245ba0de4f8c9dde746b56fda5c1eee65c9a0a76ea13db7b.
wide_12/13/14/15 routes are equal. B128 has1024 routes,117active experts,
maximum33 rows/expert; owner totals150,90,135,120,151,123,109,146.
This is real-weight/synthetic-state layer6 evidence, NOT real-prompt routing.

| M | Global visits | Expert-relative panels | Global max/owner | Relative max/owner |
|---|---:|---:|---:|---:|
|8|234|192|35|29|
|16|172|140|27|20|
|32|144|118|22|18|
|64|131|117|20|18|
|128|123|117|19|18|

Counts derive from global ceil(end/M)-floor(start/M) versus ceil(count/M), with
zero-count experts skipped. No speed forecast follows from these counts.
M32 is a useful first discriminator; M128 quadruples relative padded rows versus
M32 (14976 versus3776) while eliminating only one panel in this fixture.

## Proposed bounded candidate

Default-off expert-relative panels, M32/N256/K128 arithmetic. Preserve stable
expert/route sorting and original route-slot restore/FP32 route sum. Pack only
owned expert rows into aligned [Pmax,32,K] with descriptors(expert, sorted_start,
live_count). Ordinary BlockSpec((32,K)) cannot express arbitrary expert starts.
Exclusive aligned panel outputs prevent overlapping expert boundary writes;
mask padding, unpack to original sorted rows with unowned output zeros.
Pmax=ceil(m/32)+local_groups-1; m=8B, G32 gives63panels/2016rows atB128.
Reuse descriptors gate/up/down; initially preserve existing intermediate shapes,
feature4 reductions, BF16/SwiGLU boundaries and restoration. Include all packing
and unpacking in measured wall. Do not allocate per-expert full-B workspaces.

Grid(N/256,active_panels), full-K rawU8 [256,K] panel resident in VMEM; internally
two independent N128 outputs, each using its correct FP8 scale row and original
increasing K128 accumulation. RawFP8->F32scale->BF16 boundary unchanged. Do NOT
broadcast one scale across N256. Logical double-buffer allowance approximately
1.08MiB K1536 and1.39MiB K2048, not actual compiled VMEM/HBM admission.

Smallest test: reuse interpreted grouped-projection tests and the authenticated
DB594/596 completed-prefix B128 suffix operands; no new scalar taps or baseline.
Test metadata validity, tails/empty/unowned experts, exclusive writes, every live
row retained, source scale geometry, original K order; then actual changed suffix
arithmetic/memory/inclusive wall under existing protected harness.

Larger full-model token windows still needed afterward. Current window code
unrolls four B32 prefixes atB128. Scaling to512/2048 by duplicating prefix graphs
would multiply code; a rolled bounded causal prefix loop is a separate candidate.
Keep per-layer KV, unrepaired/repaired index lifecycle and own-score DSA contract.
Neither panel proposal nor prior shared-prefix agreement proves own8K/L7/L8.
No B32 full-model intermediate, no new checkpoint, no numerical tolerance change,
and no10Ktok/s promise. DB597 now supplies accepted short-model comparison evidence.

## Independent design review

Astra /root/observer_identity_review: no P0 blocker. Preserve independent N128
scale selection; `_scale_value` wraps ki modulo8 and must receive the correct slab.
Test N512/K1152 with distinct K0/K8 and four N128 scales. Guard dynamic count overflow,
invalid/extreme offsets, empty owners and inactive indices before gathers. Prove
exclusive panel writes, inverse mapping, every live row once and masked padding.
Reuse metadata once for all projections. Actual TPU allocation and inclusive
packing/unpacking wall remain required; CPU interpretation cannot prove either.

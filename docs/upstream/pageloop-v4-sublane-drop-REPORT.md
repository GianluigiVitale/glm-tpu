# [TPU v4] Per-page masked read-modify-write loop into a donated, shard_map-resident paged KV cache silently drops sublane row-stripes (never stored); identical writes as a flat 1-D scatter are byte-complete

*(Staged bug-report package — for submission by the repo owner. Do not file as-is
without re-checking versions against the submission-day environment.)*

## Summary

On TPU v4, a `lax.fori_loop` of per-page
`{dynamic_index_in_dim(page) -> jnp.where masked merge -> dynamic_update_index_in_dim(page)}`
read-modify-write steps into a **donated**, **shard_map-resident**, tiled 4-D
paged KV cache (`[num_pages, 16, 32, 128]` bf16; 512 tokens/page) **silently
drops contiguous sublane row-stripes of some pages**: the store for those rows
never happens, and the region retains whatever the previous occupant left in
HBM. The identical traced logic is position-exact on CPU (sentinel-proven, all
formulations), and expressing the **same writes** as a flat 1-D-index scatter
over the flattened cache view is **byte-complete on v4** across engine
instances under a byte-diff protocol. This looks like an XLA:TPU
lowering/aliasing defect for the masked full-page-tile RMW pattern against a
donated sharded buffer, not a source-logic bug — it may ultimately belong in
the JAX/XLA tracker, but the affected pattern is this repo's per-page
("pageloop") scatter formulation used for decode-context-parallel cache
writes, so it is reported here first.

**Why this matters**: the corruption is silent (no crash, no NaN), its
expression depends on stale HBM contents (launch-state lottery, so CI and
re-runs can be green by luck), and in our runs it **passed end-to-end
long-context retrieval accuracy checks while corrupting a cache**.

## Environment

| Component | Version |
|---|---|
| Hardware | TPU v4-64 pod (8 hosts x 4 chips), defect observed on all 8 hosts |
| jax / jaxlib | 0.10.1 / 0.10.1 |
| libtpu (PyPI `libtpu`) | 0.0.41 |
| Python | 3.12.13 |
| Serving stack | vLLM tpu-inference-based engine; cache donated into the jitted model step (`donate_argnames=("kv_caches",)`), resident under `jax.shard_map` with a `P(BATCH, CONTEXT)`-sharded page layout |

## The failing pattern

A paged-cache "owner-scatter" under decode context parallelism: each mesh
shard writes only the token rows it owns into its local cache slice
`[num_pages, sub_l=16, packing=32, width=128]` (bf16). The per-page ("pageloop")
formulation makes every cache mutation a single dim-0 page-tile
dynamic-update-slice:

```python
def write_one_page(i, cache):                       # cache: [P, 16, 32, 128] bf16
    p    = pages_target[i]                          # from jnp.unique over touched pages
    old  = lax.dynamic_index_in_dim(cache, p, axis=0, keepdims=False)
    buf  = jnp.zeros_like(old).at[row_p, sub_t, :].set(val, mode="drop")
    mask = jnp.zeros((16, 32), bool).at[row_p, sub_t].set(True, mode="drop")
    new  = jnp.where(mask[:, :, None], buf, old)    # masked merge (RMW)
    return lax.dynamic_update_index_in_dim(cache, new, p, axis=0)

cache = lax.fori_loop(0, n_pages_touched, write_one_page, cache)
# `cache` is a donated jit argument and lives inside jax.shard_map
# (in/out spec P(None, 'dcp', None, None) on the sub-row dim); the loop runs
# once per model step, cache round-tripping donated across steps.
```

## Observed defect

After a 2034-token prefill chunk that should fully populate 4 pages, one
cache buffer ends the step with **never-written sublane row-stripes**:

- **Hole geometry (byte-diff-proven instance)**: sublane rows `{0,1} u {8,9}`
  of 16 — i.e. **the first 2 sublanes of each 8-sublane tile** of the
  `(16, 32)`-token page layout — in two physical pages; identical coordinates
  on **all 8 hosts** of the pod.
- In token space that is in-page offsets `[0,64) u [256,320)` of alternating
  logical pages.
- The affected buffer, rows, stripe width/phase **vary per compiled
  executable / engine instance** (another instance expressed 128-token-wide
  stripes at in-page offsets 64 and 320), but the structure is always
  "leading sublanes of an 8-sublane tile, periodic across the page".

Because the cache is donated, the dropped region is **not zeros** — it holds
whatever the previous engine instance left at that HBM address. Downstream
expression is therefore a lottery:

| Stale HBM contents | Downstream symptom |
|---|---|
| zeros | constant attention-indexer scores -> degenerate `arange` top-k rows |
| foreign-layout garbage | structured wrong scores (we saw exact x~1/4-depressed 128-wide score stripes) |
| same-layout leftovers from an identical previous run | **invisible** — run is "clean" by luck |

## Evidence chain (how we know it is the lowering, not the logic)

1. **CPU exoneration (sentinel suite).** A deterministic CPU suite rebuilds
   the exact serving geometry (sentinel-initialized `[8,16,32,128]` cache,
   T=2048 chunk / 2034 live tokens, non-sequential logical->physical page
   map, garbage-valued padding rows, the verbatim serving `shard_map`
   in/out specs) and asserts position-exact coverage plus full-array
   equality. **All four scatter formulations x dcp={1,2} meshes pass** —
   including the pageloop formulation. 16/16 coverage tests, 28/28 full
   suite, 0 failures. The traced write logic covers every sublane row.

2. **Never-written proof on metal (scrambler byte-diff protocol).** Prefill
   compute is deterministic, so two identically-configured engine instances
   must produce byte-identical caches at every coordinate the program
   writes. To defeat the stale-HBM coincidence (two same-config runs can
   inherit each other's bytes), each probe run was preceded by a
   **scrambler** run with a *different* cache layout to repaint HBM.
   Byte-diffing the dumped caches across the scrambled pair: coordinates
   that differ across runs were **never written**. Result for pageloop:
   exactly one cache buffer differs, on all 8 hosts, at identical
   coordinates — sublane rows `{0,1} u {8,9}` of 16 in two physical pages
   (32732/32768 elements differing inside the hole region).

3. **Flat control (same writes, different op sequence).** The same owner
   -scatter expressed as a 1-D-index scatter over the flattened
   `[num_pages*512, 128]` view, run through the same scrambled-pair
   byte-diff protocol: **byte-identical everywhere, across all 12 dumped
   cache buffers**. The write set is identical by construction (bit-identical
   to pageloop on CPU); only the lowering differs.

4. **Geometry cross-check.** The never-written coordinates from (2) map
   exactly onto the corrupted-score windows observed at the model level
   (in-page offsets `[0,64) u [256,320)` of alternating logical pages, after
   accounting for the non-sequential block table) — the two independent
   instruments agree on the hole.

5. **Launch-state lottery confirmed directly.** Same executable, same
   inputs, back-to-back runs in one session: run A striped, run B clean.
   The only difference is inherited HBM state. So "clean" runs of the
   pageloop formulation are clean by luck, and any CI that reuses device
   state cannot be trusted to catch this.

6. **Accuracy-test invisibility.** Corrupted runs passed 6/6 end-to-end
   long-context needle retrievals. Only score-payload instrumentation
   (dumping selection scores, not just selected indices) exposed the
   corruption.

## Why this is dangerous

- **Silent**: no crash, no NaN, no shape error; the cache simply contains
  stale HBM in the holes.
- **Donation makes it worse**: with buffer donation (the standard serving
  pattern) the holes alias arbitrary prior contents, so the same binary can
  be correct-looking or subtly wrong depending on what ran before it.
- **Accuracy-check-invisible**: top-k/attention consumers degrade gracefully;
  end-to-end retrieval tests pass while a cache is corrupt.
- **Formulation-dependent in both directions**: on a *different* cache/path
  in the same stack we have the mirror-image history — a plain 4-D indexed
  scatter mislowered on v4 and the pageloop RMW was the *fix* there. So no
  single formulation is safe by inspection; each buffer/pattern needs metal
  validation, which strongly suggests the defect class lives in the
  XLA:TPU lowering of masked/partial stores into donated sharded tiled
  buffers.

## Workaround

Express the owner-scatter as a **flat 1-D-index scatter over the flattened
local cache view** (`cache.reshape(P*512, 128).at[slot, :].set(val,
mode="drop")`). This is bit-identical to the pageloop formulation on CPU and
was byte-complete on v4 across scrambled engine instances in our evidence.
We shipped it as the default for this write path, keeping the pageloop
formulation selectable only for on-metal isolation diffs. A startup
write-probe (sentinel scratch cache through the compiled scatter; refuse to
serve on any hole) is a cheap permanent guard.

## Repro

`pageloop-v4-repro.py` (attached alongside this report; self-contained,
jax+numpy only):

- builds the `[8,16,32,128]` bf16 sentinel-initialized cache, donated and
  `shard_map`-resident on a dcp mesh axis;
- runs the pageloop-formulation RMW write of 2034 position-tagged rows
  (padded 2048-token chunk, non-sequential page map, garbage pads), plus the
  flat control;
- reads back and reports never-written / corrupt coordinates and the
  per-page sublane-row hole geometry.

On CPU (`JAX_PLATFORMS=cpu`) it prints `COMPLETE` for every impl x dcp
combination (verified: jax 0.10.1). On a v4, per our evidence, the pageloop
run is expected to report never-written sublane rows while flat stays
complete. **Caveat**: the defect was characterized in-situ inside a full
serving step program; a lowering/allocation-level bug may not express in a
minimal program if XLA makes different layout/donation decisions — if the
minimal repro comes back clean on v4, the authoritative protocol is the
scrambler byte-diff across engine instances described above (and the script's
`--steps` / `--dcp` / `--no-donate` knobs widen the search).

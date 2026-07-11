#!/usr/bin/env python
"""Minimal repro: paged-cache owner-scatter "pageloop" formulation drops
sublane row-stripes on TPU v4 (silent never-written holes in a donated,
shard_map-resident, tiled 4-D cache).

WHAT THIS BUILDS (no external deps beyond jax/numpy):

  * a [8, 16, 32, 128] bf16 paged cache (= 8 physical pages of 512 tokens,
    each page tiled [sublane_rows=16, packing=32, width=128]), initialized
    to a sentinel value, laid out under jax.shard_map with the sub-row dim
    sharded on a decode-context-parallel ("dcp") mesh axis, and DONATED
    into the jitted write step (exactly how a serving engine round-trips
    its KV cache);
  * one padded 2048-token prefill chunk carrying 2034 live position-tagged
    rows for a single request, per-token block-table rows with a
    NON-sequential logical->physical page map, and garbage-valued padding
    rows with valid=False;
  * the "pageloop" owner-scatter formulation: a lax.fori_loop over touched
    pages of { dynamic_index_in_dim(page) -> masked jnp.where merge ->
    dynamic_update_index_in_dim(page) } read-modify-write, i.e. every cache
    mutation is a single full-page-tile dynamic-update-slice;
  * the "flat" control formulation: the SAME writes expressed as a 1-D-index
    scatter over the flattened [pages*512, 128] view.

EXPECTED:

  * CPU (JAX_PLATFORMS=cpu): every impl x dcp combination prints COMPLETE —
    all 2034 rows position-exact, zero sentinel below kv_len, untouched
    region intact.  The traced logic is correct (this mirrors a 16-test
    sentinel suite that exonerated it).
  * TPU v4: per our on-pod evidence, "pageloop" leaves contiguous sublane
    row-stripes of some pages NEVER WRITTEN (observed: rows {0,1} and {8,9}
    of 16 = the first 2 sublanes of each 8-sublane tile; the affected
    pages/rows vary per compiled executable / engine instance), while
    "flat" is byte-complete.  The sentinel initialization stands in for
    stale HBM: any coordinate this script reports as NEVER-WRITTEN is a
    coordinate that, in a real engine, silently retains whatever a previous
    process left in HBM.

CAVEAT for v4 runs: the defect was characterized in-situ inside a full
serving step program (donated cache round-tripping across steps, many live
buffers); expression of a lowering/allocation-level bug can depend on
layout/donation decisions the minimal program does not force.  If this
script comes back COMPLETE on your v4, the authoritative in-situ evidence
protocol is described in the accompanying report (scrambler byte-diff
across engine instances).  Knobs to widen the search: --dcp, --steps
(chained donated calls), --no-donate.

Usage:
  JAX_PLATFORMS=cpu python pageloop-v4-repro.py           # expect COMPLETE
  python pageloop-v4-repro.py                             # on a v4 host
  python pageloop-v4-repro.py --impl pageloop --dcp 2 --steps 4

Exit code 0 iff every run printed COMPLETE.
"""

import argparse
import os
import sys

# Simulated host devices for the dcp=2 mesh on CPU (must precede jax import;
# harmless for TPU runs, where the flag only affects the host platform).
os.environ["XLA_FLAGS"] = (os.environ.get("XLA_FLAGS", "") +
                           " --xla_force_host_platform_device_count=8").strip()

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from jax import lax  # noqa: E402
from jax.sharding import Mesh, NamedSharding  # noqa: E402
from jax.sharding import PartitionSpec as P  # noqa: E402

# ---------------------------------------------------------------------------
# Geometry: the exact shape the defect was observed on (a 512-token page
# stored as [sublane_rows=16, packing=32, width=128] bf16 tiles).
# ---------------------------------------------------------------------------
TOTAL_PAGES = 8
SUB_L = 16          # global sublane rows per page (cache dim 1)
KV_PACK = 32        # tokens packed per sublane row (cache dim 2)
WIDTH = 128         # row payload (cache dim 3)
P_G = SUB_L * KV_PACK  # 512 tokens per logical block
T_CHUNK = 2048      # padded prefill chunk
LIVE = 2034         # live tokens of the single request
MAX_BLOCKS = 5
# Non-sequential logical block -> physical page map (what a real block-table
# allocator produces; sequential maps can mask geometry bugs).
PERM = [0, 2, 1, 4, 3]
NUM_SEQS = 8        # static bound input, the serving-typical value
SENTINEL = -1.0     # bf16-exact; impossible as a written value (writes >= 0)
DTYPE = jnp.bfloat16


# ---------------------------------------------------------------------------
# The two owner-scatter formulations.  Both are called INSIDE jax.shard_map
# on the shard's LOCAL cache slice [pages, SUB_L/dcp, KV_PACK, WIDTH]; each
# token is written only by the mesh shard that owns its global position
# (owner = (pos % P_g) // P_l).  They are bit-identical on CPU.
# ---------------------------------------------------------------------------
def owner_scatter_pageloop(cache4, val_rows, positions, bt_tok, valid, shard,
                           *, dcp_size, num_seqs):
    """Per-physical-page read-modify-write: every cache mutation is a single
    dim-0 page-tile dynamic-update-slice (never a multi-page scatter).  THIS
    is the formulation whose v4 lowering drops sublane row-stripes."""
    total_pages, sub_l, kv_pack, kv_dim = cache4.shape
    p_l = sub_l * kv_pack           # local tokens per logical block
    p_g = p_l * dcp_size            # global tokens per logical block
    num_tokens = val_rows.shape[0]
    max_blocks = bt_tok.shape[1]

    pos = positions.astype(jnp.int32)
    owner_t = (pos % p_g) // p_l
    mine = jnp.logical_and(valid, owner_t == shard)
    blk = jnp.clip(pos // p_g, 0, max_blocks - 1)
    page_t = jnp.take_along_axis(bt_tok, blk[:, None], axis=1)[:, 0]
    in_page = jnp.clip(pos, 0, None) % p_l   # owner's local in-page row
    row_t = in_page // kv_pack
    sub_t = in_page % kv_pack
    val = val_rows.astype(cache4.dtype)
    # Non-owned / padding rows get the out-of-bounds page sentinel and are
    # dropped by both formulations.
    page_safe = jnp.where(mine, page_t, total_pages)

    # Static bound on distinct owned pages of one step (+ the OOB sentinel);
    # at these shapes it saturates total_pages+1, so nothing can be dropped.
    n_pages_touched = min((num_tokens + p_l - 1) // p_l + 2 * num_seqs + 2,
                          total_pages + 1)
    pages_target = jnp.unique(page_safe, size=n_pages_touched,
                              fill_value=total_pages)

    def _write_one_page(i, cache):
        p = pages_target[i]
        # Clamp so read/write index a real page even for the OOB sentinel
        # (that iteration is then a verified no-op merge).
        p_clamp = jnp.minimum(p, total_pages - 1)
        hit = jnp.logical_and(mine, page_t == p)
        # OOB row for any token not writing this page -> dropped by the
        # page-local scatter (mode="drop").
        row_p = jnp.where(hit, row_t, sub_l)
        old = lax.dynamic_index_in_dim(cache, p_clamp, axis=0,
                                       keepdims=False)
        # Page-local scatter into a FRESH tile, then a masked merge so only
        # owned slots change (read-modify-write; no arithmetic on cache
        # values -> narrow-dtype safe).
        buf = jnp.zeros_like(old)
        buf = buf.at[row_p, sub_t, :].set(val, mode="drop")
        wmask = jnp.zeros((sub_l, kv_pack), jnp.bool_)
        wmask = wmask.at[row_p, sub_t].set(True, mode="drop")
        new_page = jnp.where(wmask[:, :, None], buf, old)
        return lax.dynamic_update_index_in_dim(cache, new_page, p_clamp,
                                               axis=0)

    return lax.fori_loop(0, n_pages_touched, _write_one_page, cache4)


def owner_scatter_flat(cache4, val_rows, positions, bt_tok, valid, shard, *,
                       dcp_size, num_seqs):
    """Control: the SAME writes as a 1-D-index scatter over the flattened
    local slice.  Byte-complete on v4 in our evidence."""
    total_pages, sub_l, kv_pack, kv_dim = cache4.shape
    p_l = sub_l * kv_pack
    p_g = p_l * dcp_size
    max_blocks = bt_tok.shape[1]

    pos = positions.astype(jnp.int32)
    owner_t = (pos % p_g) // p_l
    mine = jnp.logical_and(valid, owner_t == shard)
    blk = jnp.clip(pos // p_g, 0, max_blocks - 1)
    page_t = jnp.take_along_axis(bt_tok, blk[:, None], axis=1)[:, 0]
    in_page = jnp.clip(pos, 0, None) % p_l
    row_t = in_page // kv_pack
    sub_t = in_page % kv_pack
    val = val_rows.astype(cache4.dtype)
    page_safe = jnp.where(mine, page_t, total_pages)

    n_slots = total_pages * p_l
    slot = page_safe * p_l + row_t * kv_pack + sub_t
    slot = jnp.where(mine, slot, n_slots)   # OOB -> dropped
    flat = cache4.reshape(n_slots, kv_dim)
    flat = flat.at[slot, :].set(val, mode="drop")
    return flat.reshape(cache4.shape)


IMPLS = {"pageloop": owner_scatter_pageloop, "flat": owner_scatter_flat}


# ---------------------------------------------------------------------------
# Inputs: 2034 live position-tagged rows, garbage pads, non-sequential pages.
# Row tag for position p (bf16-exact: all components are small integers):
#   channel 0 = p // 256, channel 1 = p % 256, channels 2.. = 1.0
# ---------------------------------------------------------------------------
def make_inputs():
    positions = np.zeros(T_CHUNK, np.int32)
    positions[:LIVE] = np.arange(LIVE, dtype=np.int32)
    garbage = np.array([0, 511, 2033, 2549, 3000, -3], np.int32)
    positions[LIVE:] = np.resize(garbage, T_CHUNK - LIVE)

    bt_row = np.zeros(MAX_BLOCKS, np.int32)
    bt_row[:len(PERM)] = PERM
    bt_tok = np.broadcast_to(bt_row, (T_CHUNK, MAX_BLOCKS)).copy()
    valid = np.zeros(T_CHUNK, bool)
    valid[:LIVE] = True

    keys = np.ones((T_CHUNK, WIDTH), np.float32)
    p = np.arange(LIVE, dtype=np.float32)
    keys[:LIVE, 0] = np.floor(p / 256.0)
    keys[:LIVE, 1] = np.mod(p, 256.0)
    keys[LIVE:] = 200.0  # pad garbage; must never land (valid=False)
    return positions, bt_tok, valid, keys


def expected_cache_f32():
    exp = np.full((TOTAL_PAGES, SUB_L, KV_PACK, WIDTH), SENTINEL, np.float32)
    for p in range(LIVE):
        phys = PERM[p // P_G]
        row = (p % P_G) // KV_PACK
        sub = (p % P_G) % KV_PACK
        exp[phys, row, sub, :] = 1.0
        exp[phys, row, sub, 0] = float(p // 256)
        exp[phys, row, sub, 1] = float(p % 256)
    return exp


# ---------------------------------------------------------------------------
# The write step: shard_map (sub-row dim on the dcp axis) + jit with the
# cache DONATED, matching how a serving engine round-trips its KV cache.
# ---------------------------------------------------------------------------
def run_write(mesh, impl, cache, keys, positions, bt_tok, valid, *, dcp_size,
              donate, steps):
    cache_spec = P(None, "dcp", None, None)
    fn = IMPLS[impl]

    def _write(cache_l, keys_b, pos_b, bt_b, valid_b):
        shard = lax.axis_index("dcp")
        return fn(cache_l, keys_b, pos_b, bt_b, valid_b, shard,
                  dcp_size=dcp_size, num_seqs=NUM_SEQS)

    shmapped = jax.shard_map(_write, mesh=mesh,
                             in_specs=(cache_spec, P(), P(), P(), P()),
                             out_specs=cache_spec, check_vma=False)
    jitted = jax.jit(shmapped, donate_argnums=(0,) if donate else ())

    cache_dev = jax.device_put(cache, NamedSharding(mesh, cache_spec))
    for _ in range(steps):  # chained donated round-trips (serving pattern)
        cache_dev = jitted(cache_dev, keys, positions, bt_tok, valid)
    return np.asarray(jax.device_get(cache_dev)).astype(np.float32)


# ---------------------------------------------------------------------------
# Readback adjudication: report never-written / corrupt coordinates and the
# hole geometry (sublane-row sets per page — the defect's fingerprint).
# ---------------------------------------------------------------------------
def adjudicate(got, tag):
    exp = expected_cache_f32()
    never, wrong = [], []
    for p in range(LIVE):
        phys = PERM[p // P_G]
        row = (p % P_G) // KV_PACK
        sub = (p % P_G) % KV_PACK
        cell = got[phys, row, sub]
        if np.array_equal(cell, exp[phys, row, sub]):
            continue
        if np.all(cell == np.float32(SENTINEL)):
            never.append((p, phys, row, sub))
        else:
            wrong.append((p, phys, row, sub, float(cell[0]), float(cell[1])))
    stray = int(np.sum((got != exp) & (exp == np.float32(SENTINEL))))

    if not never and not wrong and stray == 0:
        print(f"[{tag}] COMPLETE: {LIVE}/{LIVE} rows position-exact; "
              "0 never-written; 0 corrupt; untouched region intact")
        return True

    print(f"[{tag}] HOLES/CORRUPTION: {len(never)} never-written, "
          f"{len(wrong)} wrong-value coordinates, "
          f"{stray} stray-written cells")
    by_page = {}
    for (_p, phys, row, _sub) in never:
        by_page.setdefault(phys, set()).add(row)
    for phys in sorted(by_page):
        rows = sorted(by_page[phys])
        n = sum(1 for x in never if x[1] == phys)
        print(f"  page {phys}: never-written sublane rows {rows} "
              f"({n} token slots)  <- compare the observed v4 signature: "
              "rows {0,1} u {8,9} of 16")
    for w in wrong[:10]:
        print(f"  wrong value: pos {w[0]} -> page {w[1]} row {w[2]} "
              f"sub {w[3]}: got tag ({w[4]}, {w[5]})")
    return False


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--impl", choices=("pageloop", "flat", "both"),
                    default="both")
    ap.add_argument("--dcp", type=int, default=0,
                    help="dcp mesh size (0 = run 1 and, if devices allow, 2)")
    ap.add_argument("--steps", type=int, default=1,
                    help="chained donated write calls (serving round-trip)")
    ap.add_argument("--no-donate", action="store_true",
                    help="disable cache donation into the jitted step")
    args = ap.parse_args()

    print(f"jax {jax.__version__} | backend {jax.default_backend()} | "
          f"devices {jax.device_count()} ({jax.devices()[0].platform})")

    dcp_sizes = ([args.dcp] if args.dcp else
                 [1, 2] if jax.device_count() >= 2 else [1])
    impls = ["pageloop", "flat"] if args.impl == "both" else [args.impl]

    positions, bt_tok, valid, keys = make_inputs()
    ok = True
    for dcp in dcp_sizes:
        if SUB_L % dcp:
            print(f"skip dcp={dcp}: sub-row dim {SUB_L} not divisible")
            continue
        devices = np.asarray(jax.devices()[:dcp]).reshape(dcp)
        mesh = Mesh(devices, ("dcp",))
        for impl in impls:
            cache = np.full((TOTAL_PAGES, SUB_L, KV_PACK, WIDTH), SENTINEL,
                            np.float32).astype(DTYPE)
            got = run_write(mesh, impl, cache, keys, positions, bt_tok,
                            valid, dcp_size=dcp, donate=not args.no_donate,
                            steps=args.steps)
            ok &= adjudicate(got, f"impl={impl} dcp={dcp} "
                             f"donate={not args.no_donate} "
                             f"steps={args.steps}")
    print("VERDICT:", "COMPLETE" if ok else "HOLES DETECTED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

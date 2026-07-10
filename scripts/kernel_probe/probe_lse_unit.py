#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Sparse-ladder RUNG 1 — single-chip metal unit for dsa_sparse_decode emit_lse.

Runbook: ~/glm-tpu/docs/11-pod-runbook.md §8 rung 1. Clears the top Stage-A/C
metal risk before any engine run: the [1, H, 128] lane-broadcast lse out-block
has only ever run in interpret mode; v4 Mosaic must (a) COMPILE it, (b) leave
the attention output BITWISE unchanged vs emit_lse=False, (c) produce lse
matching the XLA oracle's m + log(l).

Three gates, numbers printed VERBATIM (the deliverable), artifact written under
docs/artifacts/:

  GATE L1 — byte-identity: dsa_sparse_decode(emit_lse=False) output BITWISE ==
      the default call (no kwarg), same seeds, fp32 AND bf16.
  GATE L2 — out invariance: emit_lse=True's `out` BITWISE == emit_lse=False's
      `out` (the lse ref must not perturb the attention path), fp32 AND bf16.
  GATE L3 — lse parity vs dsa_sparse_decode_xla's lse:
      * fp32 max-abs <= 5e-5  (log-domain; interpreter measured ~1e-6 — the
        MXU single-pass bars re-measured here, the §2b discipline)
      * bf16 max-abs <= 1e-2  (log of a bf16-accumulated sum; provisional bar,
        the printed number is the record)
      * empty rows (seg_valid=0): lse <= -1e37 on BOTH kernel and oracle
        (below _DCP_LSE_FLOOR -> combine weight exactly 0).

Geometries: the Stage-A test set (R=8/64, H=8, lkv=512, r=64, seg 2048 with
seg_valid full/partial/one/zero, seg_block default and 768).

FROZEN KERNELS: test-harness code — MUST NOT modify kernel logic.

Metal invocation (owner/main-thread, EXCLUSIVE single chip, no engine running):
    TPU_VISIBLE_DEVICES=0 ~/vllm-env/bin/python \
        scripts/kernel_probe/probe_lse_unit.py --tag <date>

CPU wiring validation (never touches the TPU):
    ~/vllm-env/bin/python scripts/kernel_probe/probe_lse_unit.py --interpret

Exit codes: 0 = all gates PASS   1 = a gate FAILED   2 = misconfiguration.
"""
from __future__ import annotations

import argparse
import datetime
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]  # ~/glm-tpu
_ARTIFACTS = _ROOT / "docs" / "artifacts"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--interpret", action="store_true",
                    help="CPU wiring validation (interpret mode; no TPU)")
    ap.add_argument("--tag", default=None, help="artifact tag")
    args = ap.parse_args()

    if args.interpret:
        os.environ.setdefault("JAX_PLATFORMS", "cpu")

    import jax
    import jax.numpy as jnp
    import numpy as np

    from tpu_inference.kernels.dsa.sparse_mla_kernel import (
        dsa_sparse_decode, dsa_sparse_decode_xla)

    if not args.interpret:
        kind = jax.devices()[0].device_kind
        if "TPU" not in kind.upper():
            print(f"MISCONFIG: expected a TPU device, got {kind!r}")
            return 2
        print(f"device: {kind} x{jax.device_count()} (expect 1 chip / 2 cores)")

    interpret = bool(args.interpret)
    LKV, RD, SEG = 512, 64, 2048
    results, all_pass = [], True

    def rec(line: str, ok: bool | None = None):
        nonlocal all_pass
        print(line, flush=True)
        results.append(line)
        if ok is False:
            all_pass = False

    for dtype, name, l3_bar in ((jnp.float32, "fp32", 5e-5),
                                (jnp.bfloat16, "bf16", 1e-2)):
        for R, seg_valid, seg_block in ((8, SEG, None), (8, 700, None),
                                        (64, 1, None), (8, 0, None),
                                        (8, SEG, 768)):
            rng = np.random.default_rng(hash((name, R, seg_valid)) % 2**31)
            q_nope = jnp.asarray(rng.standard_normal((R, 8, LKV)) * 0.3,
                                 dtype)
            q_pe = jnp.asarray(rng.standard_normal((R, 8, RD)) * 0.3, dtype)
            seg = jnp.asarray(rng.standard_normal((R, SEG, LKV + 128)) * 0.3,
                              dtype)
            sv = jnp.full((R,), seg_valid, jnp.int32)
            kw = dict(sm_scale=0.11, interpret=interpret)
            if seg_block is not None:
                kw["seg_block"] = seg_block

            out_default = dsa_sparse_decode(q_nope, q_pe, seg, sv, **kw)
            out_false = dsa_sparse_decode(q_nope, q_pe, seg, sv,
                                          emit_lse=False, **kw)
            out_true, lse = dsa_sparse_decode(q_nope, q_pe, seg, sv,
                                              emit_lse=True, **kw)
            _, lse_oracle = dsa_sparse_decode_xla(q_nope, q_pe, seg, sv,
                                                  sm_scale=0.11,
                                                  emit_lse=True)

            cfg = (f"{name} R={R} sv={seg_valid}"
                   f"{f' blk={seg_block}' if seg_block else ''}")
            l1 = bool(
                (np.asarray(out_default).view(np.uint8) ==
                 np.asarray(out_false).view(np.uint8)).all())
            rec(f"GATE L1 [{cfg}] byte-identity(default vs False): "
                f"{'PASS' if l1 else 'FAIL'}", l1)
            l2 = bool(
                (np.asarray(out_true).view(np.uint8) ==
                 np.asarray(out_false).view(np.uint8)).all())
            rec(f"GATE L2 [{cfg}] out invariance(True vs False): "
                f"{'PASS' if l2 else 'FAIL'}", l2)
            lse_np = np.asarray(lse, np.float64)
            ora_np = np.asarray(lse_oracle, np.float64)
            if seg_valid == 0:
                l3 = bool((lse_np <= -1e37).all() and (ora_np <= -1e37).all())
                rec(f"GATE L3 [{cfg}] empty-row floor: kernel_max="
                    f"{lse_np.max():.3e} oracle_max={ora_np.max():.3e} "
                    f"{'PASS' if l3 else 'FAIL'}", l3)
            else:
                live = ora_np > -1e37
                diff = float(np.abs(lse_np[live] - ora_np[live]).max()) \
                    if live.any() else 0.0
                l3 = diff <= l3_bar
                rec(f"GATE L3 [{cfg}] lse max-abs vs oracle = {diff:.6e} "
                    f"(bar {l3_bar:.0e}): {'PASS' if l3 else 'FAIL'}", l3)

    tag = args.tag or datetime.datetime.utcnow().strftime("%Y%m%d-%H%M")
    mode = "interpret" if interpret else "metal"
    _ARTIFACTS.mkdir(parents=True, exist_ok=True)
    art = _ARTIFACTS / f"rung1-lse-unit-{mode}-{tag}.txt"
    art.write_text("\n".join(results) + "\n")
    print(f"\nartifact: {art}")
    print("RUNG 1 VERDICT:", "PASS" if all_pass else "FAIL")
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())

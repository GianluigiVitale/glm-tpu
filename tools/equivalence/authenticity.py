"""Adapter authenticity (S0, best effort, read-only): does the v0 adapter build what production
compiled on the TPU fleet?

Production writes the StableHLO it compiled to ``<hlo root>/<run>/native.rank0/<name>.stablehlo.mlir``
(``str(lowered.compiler_ir("stablehlo"))``, the same printer as N2). Those originals embed
source locations only inside the base64 Mosaic kernel bodies. This module lowers the matching
production-tier program on the CPU host and compares:

* ``raw``: byte equality of the two texts (expected to differ only in kernel bodies);
* ``masked``: equality after replacing every kernel body by ``<mosaic>`` -- kernel count, kernel
  names, operand and result types and everything else stay byte-compared.

Never writes next to the originals and never deletes them. Prints hashes and verdicts only.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
from pathlib import Path
from typing import Any

from .common import emit, environment, require_cpu, source_record

# Original file name -> production-tier record key (32K session: donated; concurrent: n=4).
KEYS = {
    "wk_decode": "wk_decode",
    "wk_promote": "wk_promote",
    "cache_init": "cache_init@32768",
    "prefill_128": "prefill_128@32768+donated",
    "prefill_114": "prefill_114@32768+donated",
    "decode": "decode@32768+donated",
    "batch_cache_init": "batch_cache_init@32768#n4",
    "batch_insert": "batch_insert@32768#n4",
    "batch_decode": "batch_decode@32768#n4",
}


def _first_difference(left: str, right: str) -> dict[str, Any] | None:
    if left == right:
        return None
    a, b = left.split("\n"), right.split("\n")
    for index, (x, y) in enumerate(zip(a, b, strict=False)):
        if x != y:
            column = next((i for i, (p, q) in enumerate(zip(x, y, strict=False)) if p != q), min(len(x), len(y)))
            return dict(line=index + 1, column=column, lines=[len(a), len(b)])
    return dict(line=min(len(a), len(b)) + 1, column=0, lines=[len(a), len(b)])


def compare(directories: list[Path]) -> dict[str, Any]:
    from . import fixture, lowering, normalize, programs

    require_cpu()
    originals: dict[str, Path] = {}
    for directory in directories:
        for path in sorted(directory.glob("*.stablehlo.mlir")):
            name = path.name[: -len(".stablehlo.mlir")]
            if name in KEYS:
                originals.setdefault(name, path)
    wanted = {KEYS[name] for name in originals}
    mesh = fixture.cpu_mesh()
    with lowering.tpu_v4_info():
        specs = programs.program_specs("production", mesh, only=wanted)
    rows: dict[str, Any] = {}
    for name, path in sorted(originals.items()):
        spec = specs[KEYS[name]]
        with lowering.location_free():
            text = normalize.stablehlo_text(lowering.lower_for_tpu(spec.fn, spec.args))
        original = path.read_text()
        mine_masked, mine_kernels = normalize.mosaic_bodies(text, full_mask=True)
        theirs_masked, their_kernels = normalize.mosaic_bodies(original, full_mask=True)
        rows[name] = dict(
            key=KEYS[name],
            original_run=path.parent.parent.name,
            original_sha256=sha256(original.encode()).hexdigest(),
            harness_sha256=sha256(text.encode()).hexdigest(),
            raw_equal=text == original,
            masked_equal=mine_masked == theirs_masked,
            masked_sha256=sha256(mine_masked.encode()).hexdigest(),
            kernel_calls=[len(mine_kernels), len(their_kernels)],
            kernel_names_equal=[k for k, _ in mine_kernels] == [k for k, _ in their_kernels],
            first_masked_difference=_first_difference(mine_masked, theirs_masked),
            bytes=[len(text), len(original)],
        )
    verdict = bool(rows) and all(row["masked_equal"] for row in rows.values())
    return dict(gate="adapter-authenticity", status="pass" if verdict else ("fail" if rows else "no-originals"),
                environment=environment(), source=source_record(), programs=rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="compare the v0 adapter with TPU StableHLO originals")
    parser.add_argument("directories", nargs="+", type=Path,
                        help="native.rank0 directories of golden runs (read-only)")
    args = parser.parse_args(argv)
    emit(compare(args.directories))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

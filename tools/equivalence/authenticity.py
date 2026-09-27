"""Adapter authenticity (best effort, read-only): does the harness build what production compiled on
the TPU fleet? (Since the programs come from the real ``TPUModelRunner``, this checks the CPU-hosted
TPU lowering and the abstract production inputs.)

Production writes the StableHLO it compiled to ``<hlo root>/<run>/native.rank0/<name>.stablehlo.mlir``
(``str(lowered.compiler_ir("stablehlo"))``, the same printer as N2). Those originals embed source
locations only inside the base64 Mosaic kernel bodies. This module lowers the matching
production-tier program (built by the real runtime) on the CPU host and compares:

* ``raw``: byte equality of the two texts;
* ``masked``: equality after replacing every kernel body by ``<mosaic>`` -- kernel count, kernel
  names, operand and result types and everything else stay byte-compared;
* ``decoded``: every kernel body on both sides is decoded (base64 -> Mosaic MLIR bytecode, parsed
  with the TPU dialect registered and deserialized by ``mosaic-serde``, exactly as jax itself
  re-reads a body) and printed without locations, with the module's ``stable_mosaic.version``
  attribute taken out and reported separately; the whole module must then be equal with each body
  replaced by the SHA-256 of that text. The version is a serialization target, not kernel content:
  on a CPU host jax has no TPU backend and serializes at its forward-compatible version
  (``tpu_custom_call.get_ir_version`` -> ``_FWD_COMPAT_VERSION``, 11 in jax 0.10.1), while the fleet
  serializes at the current one (13); the deserialized IR is compared in full.

``--kernel-names`` chooses the Pallas kernel names the harness lowers with (``lowering.kernel_names``):
``recorded`` (default) patches them back to their 181c013e spellings through ``kernel_renames.toml``,
as in the records, for originals compiled at 181c013e (the B1/B2 baseline runs); ``public`` keeps the
names production lowers since S4.2b, for originals compiled from this tree. The mode applies to the
production build itself (the kept ``Lowered`` objects are built under it), and the report states it.
A kernel's name is both its custom call's ``kernel_name`` and the function symbol inside its Mosaic
body, so under the other mode every program with a Pallas kernel differs, masked and decoded.

``--out FILE`` writes the per-kernel table (program, index, kernel name, decoded equality, digest
prefixes) to a small JSON report outside Git. Never writes next to the originals and never
deletes them. Prints hashes and verdicts only.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
import re
from pathlib import Path
from typing import Any

from .common import emit, environment, require_cpu, source_record
from .lowering import KERNEL_NAME_MODES

# Original file name -> production-tier record key: the sequential 32K session (B1, donated) and,
# for the batch programs, the concurrent n=4 session (B2). The first directory holding a name wins.
KEYS = {
    "wk_decode": "wk_decode@32768+donated",
    "wk_promote": "wk_promote@32768+donated",
    "cache_init": "cache_init@32768+donated",
    "prefill_128": "prefill_128@32768+donated",
    "prefill_114": "prefill_114@32768+donated",
    "decode": "decode@32768+donated",
    "batch_cache_init": "batch_cache_init@32768+donated#n4",
    "batch_insert": "batch_insert@32768+donated#n4",
    "batch_decode": "batch_decode@32768+donated#n4",
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


_VERSION_ATTRIBUTE = re.compile(r"stable_mosaic\.version = ([0-9]+) : i64")


def decode_mosaic(raw: bytes) -> str:
    """Location-free MLIR text of one serialized Mosaic kernel body (as jax re-reads a body in
    ``tpu_custom_call.CustomCallBackendConfig.downgrade_lowered_module_asm``)."""
    from jax._src import tpu_custom_call
    from jax._src.interpreters import mlir
    from jax._src.lib import tpu
    from jaxlib.mlir import ir
    from jaxlib.mlir.passmanager import PassManager

    context = mlir.make_ir_context()
    tpu.register_dialect(context)
    for loader in tpu_custom_call._extra_dialect_loaders:
        loader(context)
    with context, ir.Location.unknown():
        context.allow_unregistered_dialects = True
        module = ir.Module.parse(raw)
        PassManager.parse("builtin.module(mosaic-serde{serialize=false})").run(module.operation)
        return module.operation.get_asm(enable_debug_info=False)


def decoded_bodies(text: str) -> tuple[str, list[list[str]], list[str]]:
    """``text`` with every kernel body replaced by ``<mosaic:decoded=sha256>`` of its decoded,
    location-free MLIR without the serialization-version attribute; ``[kernel_name,
    decoded_sha256]`` per custom call in order; the sorted serialization versions seen."""
    import base64

    from .normalize import _BODY, _KERNEL_NAME

    kernels: list[list[str]] = []
    versions: set[str] = set()
    lines = text.split("\n")
    for index, line in enumerate(lines):
        if "@tpu_custom_call" not in line:
            continue
        bodies, names = _BODY.findall(line), _KERNEL_NAME.findall(line)
        if len(bodies) != 1 or len(names) != 1:
            raise ValueError("unexpected tpu_custom_call form")
        decoded = decode_mosaic(base64.b64decode(bodies[0][1], validate=True))
        found = _VERSION_ATTRIBUTE.findall(decoded)
        if len(found) != 1:
            raise ValueError("kernel body without exactly one stable_mosaic.version attribute")
        versions.update(found)
        digest = sha256(_VERSION_ATTRIBUTE.sub("stable_mosaic.version = <serialized>", decoded).encode()).hexdigest()
        kernels.append([names[0][1:-1], digest])
        lines[index] = _BODY.sub(lambda m, d=digest: m.group(1) + f"<mosaic:decoded={d}>" + m.group(3), line)
    return "\n".join(lines), kernels, sorted(versions)


def compare(directories: list[Path], out: Path | None = None, kernel_names: str = "recorded") -> dict[str, Any]:
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
    with lowering.kernel_names(kernel_names), lowering.tpu_v4_info():  # the build lowers under the mode
        specs = programs.program_specs("production", mesh, only=wanted, keep_lowered=True)
    rows: dict[str, Any] = {}
    table: list[list[Any]] = []
    for name, path in sorted(originals.items()):
        spec = specs[KEYS[name]]
        # production's own Lowered (built location-free by compile_program)
        with lowering.kernel_names(kernel_names), lowering.location_free():
            lowered = spec.lowered if spec.lowered is not None else lowering.lower_for_tpu(spec.fn, spec.args)
            text = normalize.stablehlo_text(lowered)
        original = path.read_text()
        mine_masked, mine_kernels = normalize.mosaic_bodies(text, full_mask=True)
        theirs_masked, their_kernels = normalize.mosaic_bodies(original, full_mask=True)
        mine_decoded, mine_bodies, mine_versions = decoded_bodies(text)
        theirs_decoded, their_bodies, their_versions = decoded_bodies(original)
        pairs = list(zip(mine_bodies, their_bodies, strict=False))
        for index, (mine, theirs) in enumerate(pairs):
            table.append([name, index, mine[0], mine == theirs, mine[1][:16], theirs[1][:16]])
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
            decoded_equal=mine_decoded == theirs_decoded,
            serialization_versions=[mine_versions, their_versions],
            decoded_kernels_equal=sum(1 for mine, theirs in pairs if mine == theirs),
            raw_kernel_bodies_equal=sum(1 for a, b in zip(mine_kernels, their_kernels, strict=False) if a == b),
            first_decoded_difference=_first_difference(mine_decoded, theirs_decoded),
            bytes=[len(text), len(original)],
        )
    verdict = bool(rows) and all(row["masked_equal"] and row["decoded_equal"] for row in rows.values())
    if out is not None:
        out.write_text(
            json.dumps(
                dict(
                    columns=["program", "index", "kernel", "decoded_equal", "harness_sha16", "original_sha16"],
                    kernel_names=kernel_names,
                    kernels=table,
                ),
                indent=0,
            )
            + "\n"
        )
    return dict(
        gate="adapter-authenticity",
        status="pass" if verdict else ("fail" if rows else "no-originals"),
        kernel_names=kernel_names,
        environment=environment(),
        source=source_record(),
        programs=rows,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="compare the real runtime's programs with TPU StableHLO originals")
    parser.add_argument("directories", nargs="+", type=Path, help="native.rank0 directories of TPU runs (read-only)")
    parser.add_argument("--out", type=Path, help="write the per-kernel table here (outside Git)")
    parser.add_argument(
        "--kernel-names",
        choices=KERNEL_NAME_MODES,
        default="recorded",
        help="recorded: the 181c013e names (originals compiled at 181c013e); public: the names production "
        "lowers (originals compiled from this tree)",
    )
    args = parser.parse_args(argv)
    emit(compare(args.directories, args.out, kernel_names=args.kernel_names))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

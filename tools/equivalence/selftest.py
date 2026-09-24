"""G14: mutation self-test of the fingerprint procedure (N1-N8).

Invariance cases must leave the normalized digest and the N8 signature unchanged; sensitivity
cases must change at least one of them. Synthetic programs run in this process; the cases on
real production programs (the fixture-tier ``decode`` and ``batch_decode``) run in a CPU32 child;
the relocation case lowers the fixture ``decode`` from a copy of the source tree at another path
with blank lines prepended to a kernel and a layer module. A failing self-test blocks every gate.

Case ids follow DESIGN.md section 7.5.3. Findings recorded at S0 (jax 0.10.1):

* ``iv`` (named scope renamed *inside a Pallas kernel*) is **not** invariant: Mosaic lowering
  emits ``tpu.trace_start(message=<scope>)`` markers into the kernel body. It is recorded as the
  sensitivity case ``iv-pallas``; a named scope inside a kernel body is part of the device program
  and may not be renamed in a pure refactor. Plain-JAX scopes (``iv``) are invariant.
* ``vi``/``vii``/``j`` run on synthetic Pallas kernels now (the D5 procedure itself is exercised
  through a temporary names module and an explicit rename table).
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, NamedTuple

from .common import PRODUCTION_PATHS, REPO, emit, environment, run_child, source_record


def _fingerprint(fn: Any, *args: Any, renames: dict[str, Any] | None = None) -> dict[str, Any]:
    from . import lowering, normalize

    with lowering.location_free(renames):
        lowered = lowering.lower_for_tpu(fn, args)
        return normalize.fingerprint(lowered, args, summary=False)


def _same(a: dict[str, Any], b: dict[str, Any]) -> bool:
    """What G1/G2 compare for one program: normalized digest, N8 signature and the compiler options
    bound to the ``Lowered`` by ``jax.jit`` (the ``Lowered.compile`` arguments are recorded by the
    real compile path, outside the normalizer)."""
    return (
        a["digest"] == b["digest"]
        and a["signature_digest"] == b["signature_digest"]
        and a["jit_compiler_options"] == b["jit_compiler_options"]
    )


def _case(case: str, kind: str, description: str, a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    same = _same(a, b)
    return dict(
        case=case,
        kind=kind,
        description=description,
        passed=same if kind == "invariance" else not same,
        digest_equal=a["digest"] == b["digest"],
        signature_equal=a["signature_digest"] == b["signature_digest"],
        options_equal=a["jit_compiler_options"] == b["jit_compiler_options"],
    )


# ----------------------------------------------------------------------------- synthetic programs
def _pallas(
    const: float = 2.0, *, name: str | None = None, scope: str | None = None, body_name: str = "scale_kernel"
) -> Any:
    import jax
    import jax.numpy as jnp
    from jax.experimental import pallas as pl

    def kernel(x_ref: Any, o_ref: Any) -> None:
        if scope is None:
            o_ref[...] = x_ref[...] * const
        else:
            with jax.named_scope(scope):
                o_ref[...] = x_ref[...] * const

    kernel.__name__ = kernel.__qualname__ = body_name
    call = pl.pallas_call(kernel, out_shape=jax.ShapeDtypeStruct((8, 128), jnp.float32), name=name)
    return jax.jit(lambda x: call(x) + 1.0)


def synthetic_cases() -> list[dict[str, Any]]:
    import jax
    import jax.numpy as jnp
    from jax import lax

    x = jax.ShapeDtypeStruct((8, 128), jnp.float32)
    results = []

    # (iii) rename a result NamedTuple field
    class Result(NamedTuple):
        value: Any
        other: Any

    class Renamed(NamedTuple):
        renamed_value: Any
        other: Any

    results.append(
        _case(
            "iii",
            "invariance",
            "result NamedTuple field renamed",
            _fingerprint(jax.jit(lambda a: Result(a + 1, a * 2)), x),
            _fingerprint(jax.jit(lambda a: Renamed(a + 1, a * 2)), x),
        )
    )

    # (iv) named scope renamed in plain JAX
    def scoped(label: str) -> Any:
        def f(a: Any) -> Any:
            with jax.named_scope(label):
                return jnp.tanh(a) * 3

        return jax.jit(f)

    results.append(
        _case(
            "iv",
            "invariance",
            "jax.named_scope renamed (plain JAX)",
            _fingerprint(scoped("glm_scope_a"), x),
            _fingerprint(scoped("renamed_scope"), x),
        )
    )
    # (iv-pallas) the same inside a Pallas kernel body is a tpu.trace_start marker: must be detected
    results.append(
        _case(
            "iv-pallas",
            "sensitivity",
            "jax.named_scope renamed inside a Pallas kernel (tpu.trace_start marker; design expected invariance)",
            _fingerprint(_pallas(scope="glm_kernel_scope"), x),
            _fingerprint(_pallas(scope="renamed_scope"), x),
        )
    )

    # (v) rename the outer jitted function and a nested jitted function
    def nested(outer_name: str, inner_name: str) -> Any:
        def inner(a: Any) -> Any:
            return jnp.sin(a) * 3

        inner.__name__ = inner.__qualname__ = inner_name
        inner_jit = jax.jit(inner)

        def outer(a: Any) -> Any:
            return inner_jit(a) + inner_jit(a * 2)

        outer.__name__ = outer.__qualname__ = outer_name
        return jax.jit(outer)

    results.append(
        _case(
            "v",
            "invariance",
            "outer and nested jitted functions renamed",
            _fingerprint(nested("execute", "body"), x),
            _fingerprint(nested("run_step", "layer"), x),
        )
    )

    # (vi) explicit name= equal to the implicit Pallas name; (vii) explicit name=, body renamed
    implicit = _fingerprint(_pallas(), x)
    results.append(
        _case(
            "vi",
            "invariance",
            "explicit Pallas name= equal to the implicit name",
            implicit,
            _fingerprint(_pallas(name="scale_kernel"), x),
        )
    )
    results.append(
        _case(
            "vii",
            "invariance",
            "explicit Pallas name=, Python kernel body renamed",
            _fingerprint(_pallas(name="scale_kernel"), x),
            _fingerprint(_pallas(name="scale_kernel", body_name="renamed_body"), x),
        )
    )

    # N1 control: without the location patch, a kernel defined at another line differs
    results.append(n1_control())

    # (b) FP32 accumulation -> BF16
    a = jax.ShapeDtypeStruct((16, 64), jnp.bfloat16)
    b = jax.ShapeDtypeStruct((64, 32), jnp.bfloat16)

    def accumulate(dtype: Any) -> Any:
        return jax.jit(lambda p, q: lax.dot_general(p, q, (((1,), (0,)), ((), ())), preferred_element_type=dtype))

    results.append(
        _case(
            "b",
            "sensitivity",
            "FP32 accumulation switched to BF16",
            _fingerprint(accumulate(jnp.float32), a, b),
            _fingerprint(accumulate(jnp.bfloat16), a, b),
        )
    )

    # (c) precision default -> highest (as the prefill selector's dot)
    f32a = jax.ShapeDtypeStruct((16, 64), jnp.float32)
    f32b = jax.ShapeDtypeStruct((64, 32), jnp.float32)
    results.append(
        _case(
            "c",
            "sensitivity",
            "dot precision default -> highest",
            _fingerprint(jax.jit(lambda p, q: jnp.dot(p, q, precision="default")), f32a, f32b),
            _fingerprint(jax.jit(lambda p, q: jnp.dot(p, q, precision="highest")), f32a, f32b),
        )
    )

    # (d) two pytree fields swapped (signature)
    class Pair(NamedTuple):
        left: Any
        right: Any

    class Swapped(NamedTuple):
        right: Any
        left: Any

    left, right = jax.ShapeDtypeStruct((4, 8), jnp.float32), jax.ShapeDtypeStruct((8, 4), jnp.float32)
    results.append(
        _case(
            "d",
            "sensitivity",
            "two input pytree fields swapped",
            _fingerprint(jax.jit(lambda p: p.left @ p.right), Pair(left, right)),
            _fingerprint(jax.jit(lambda p: p.left @ p.right), Swapped(right, left)),
        )
    )

    # (n8-spec / d-spec) equivalent PartitionSpec spellings agree (N8 canonical spec); a different
    # placement is still detected by the signature
    results.extend(spec_spelling_cases())

    # (e) routed FP8 projection tile 256 -> 128 (the production Pallas kernel)
    results.append(routed_tile_case())

    # (f) donation removed
    def body(p: Any, q: Any) -> Any:
        return p * 2 + q

    results.append(
        _case(
            "f",
            "sensitivity",
            "donation removed",
            _fingerprint(jax.jit(body, donate_argnums=(0,)), x, x),
            _fingerprint(jax.jit(body), x, x),
        )
    )

    # (k) an XLA option bound to the jit: byte-identical StableHLO, but Lowered.compile hands it to
    # the compiler (G1/G2 compare jit_compiler_options)
    donated = jax.jit(body, donate_argnums=(0,))
    with_option = jax.jit(body, donate_argnums=(0,), compiler_options={"xla_allow_excess_precision": False})
    results.append(
        _case(
            "k",
            "sensitivity",
            "jit compiler_options added (identical StableHLO text)",
            _fingerprint(donated, x, x),
            _fingerprint(with_option, x, x),
        )
    )

    # (g) one constant inside a Pallas kernel body
    results.append(
        _case("g", "sensitivity", "constant inside a Pallas kernel body", implicit, _fingerprint(_pallas(3.0), x))
    )

    # (h) two independent ops reordered
    def ordered(first_sin: bool) -> Any:
        def f(p: Any, q: Any) -> Any:
            if first_sin:
                s = jnp.sin(p)
                c = jnp.cos(q)
            else:
                c = jnp.cos(q)
                s = jnp.sin(p)
            return s, c

        return jax.jit(f)

    results.append(
        _case(
            "h",
            "sensitivity",
            "two independent ops reordered",
            _fingerprint(ordered(True), x, x),
            _fingerprint(ordered(False), x, x),
        )
    )

    results.extend(kernel_rename_cases())
    return results


def n1_control() -> dict[str, Any]:
    """The same kernel source at two line offsets: identical under N1, different without it."""
    import jax
    import jax.numpy as jnp

    from . import normalize

    source = (
        "import jax\nfrom jax.experimental import pallas as pl\n"
        "def kernel(x_ref, o_ref):\n    o_ref[...] = x_ref[...] * 2.0\n"
        "def build():\n"
        "    call = pl.pallas_call(kernel, out_shape=jax.ShapeDtypeStruct((8, 128), jax.numpy.float32))\n"
        "    return jax.jit(lambda x: call(x))\n"
    )
    x = jax.ShapeDtypeStruct((8, 128), jnp.float32)
    built = []
    for offset in (0, 10):
        namespace: dict[str, Any] = {}
        exec(compile("\n" * offset + source, f"<selftest-kernel-{offset}>", "exec"), namespace)
        built.append(namespace["build"]())
    patched = [_fingerprint(fn, x) for fn in built]
    raw = []
    for fn in built:
        jax.clear_caches()
        text = normalize.stablehlo_text(fn.trace(x).lower(lowering_platforms=("tpu",)))
        raw.append(normalize.normalize(text)[0])
    return dict(
        case="n1-control",
        kind="invariance",
        description="kernel source at another line: equal under N1 "
        "(and different without the N1 patch, proving the patch is load-bearing)",
        passed=_same(*patched) and raw[0] != raw[1],
        digest_equal=patched[0]["digest"] == patched[1]["digest"],
        signature_equal=patched[0]["signature_digest"] == patched[1]["signature_digest"],
        unpatched_differs=raw[0] != raw[1],
    )


def spec_spelling_cases() -> list[dict[str, Any]]:
    """A trailing unsharded ``None`` in an input ``PartitionSpec`` lowers identically and must not
    change the N8 signature; moving the sharded axis must."""
    import jax
    import jax.numpy as jnp
    import numpy as np
    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

    mesh = Mesh(np.asarray(jax.devices()[:1], object).reshape(1, 1), ("expert", "feature"))

    def arg(*spec: Any) -> Any:
        return jax.ShapeDtypeStruct((8, 16, 128), jnp.float32, sharding=NamedSharding(mesh, P(*spec)))

    fn = jax.jit(lambda a: jnp.tanh(a) * 2)
    short = _fingerprint(fn, arg(None, "expert"))
    return [
        _case(
            "n8-spec",
            "invariance",
            "input PartitionSpec spelled with a trailing None (same placement)",
            short,
            _fingerprint(fn, arg(None, "expert", None)),
        ),
        _case(
            "d-spec",
            "sensitivity",
            "input PartitionSpec with the sharded axis moved (signature)",
            short,
            _fingerprint(fn, arg("expert", None)),
        ),
    ]


def routed_tile_case() -> dict[str, Any]:
    import jax
    import jax.numpy as jnp

    from glm_tpu.kernels.fp8_grouped_matmul.kernel import RoutedProjectionConfig, fp8_routed_projection

    lhs = jax.ShapeDtypeStruct((8, 512), jnp.bfloat16)
    bits = jax.ShapeDtypeStruct((4, 512, 512), jnp.uint8)
    scale = jax.ShapeDtypeStruct((4, 4, 4), jnp.float32)
    ids = jax.ShapeDtypeStruct((8,), jnp.int32)
    owned = jax.ShapeDtypeStruct((8,), jnp.bool_)

    def build(tile: int) -> Any:
        config = RoutedProjectionConfig(output_tile=tile, contraction_tile=tile)
        return jax.jit(lambda x, b, s, i, o: fp8_routed_projection(x, ((b, s),), i, o, config=config))

    return _case(
        "e",
        "sensitivity",
        "routed FP8 projection tiles 256 -> 128 (production Pallas kernel)",
        _fingerprint(build(256), lhs, bits, scale, ids, owned),
        _fingerprint(build(128), lhs, bits, scale, ids, owned),
    )


def kernel_rename_cases() -> list[dict[str, Any]]:
    """D5 on a temporary names module: a renamed kernel differs unless the rename table maps it back."""
    import types

    import jax
    import jax.numpy as jnp

    module = types.ModuleType("glm_equivalence_selftest_names")
    module.KERNEL_NAMES = {"scale": "greenfield_scale_kernel"}
    sys.modules[module.__name__] = module
    x = jax.ShapeDtypeStruct((8, 128), jnp.float32)

    def build() -> Any:
        from jax.experimental import pallas as pl

        def kernel(x_ref: Any, o_ref: Any) -> None:
            o_ref[...] = x_ref[...] * 2.0

        call = pl.pallas_call(
            kernel, out_shape=jax.ShapeDtypeStruct((8, 128), jnp.float32), name=module.KERNEL_NAMES["scale"]
        )
        return jax.jit(lambda a: call(a))

    try:
        baseline = _fingerprint(build(), x)
        module.KERNEL_NAMES["scale"] = "glm_scale"
        unpatched = _fingerprint(build(), x)
        table = dict(module=module.__name__, attribute="KERNEL_NAMES", names={"glm_scale": "greenfield_scale_kernel"})

        class Lazy:  # build inside the patch so the builder reads the patched table
            def trace(self, *args: Any) -> Any:
                return build().trace(*args)

        patched = _fingerprint(Lazy(), x, renames=table)
    finally:
        sys.modules.pop(module.__name__, None)
    return [
        _case("j", "sensitivity", "kernel name= changed without a kernel_renames entry", baseline, unpatched),
        _case(
            "j-mapped",
            "invariance",
            "kernel name= changed with a kernel_renames entry (patched back)",
            baseline,
            patched,
        ),
    ]


# ----------------------------------------------------------------------------- real programs (CPU32 child)
def fixture_cases() -> dict[str, Any]:
    from dataclasses import replace

    from . import fixture, lowering, programs

    mesh = fixture.cpu_mesh()
    out: dict[str, Any] = {}
    with lowering.tpu_v4_info():
        specs = programs.program_specs("fixture", mesh, only={"decode@1536", "batch_decode@1536#n4"})
        base_decode = programs.fingerprint_specs({"decode": specs["decode@1536"]}, summary=False)["decode"]
        base_batch = programs.fingerprint_specs({"batch": specs["batch_decode@1536#n4"]}, summary=False)["batch"]
        out["decode"] = base_decode

        # (a) RMS epsilon 1e-5 -> 1e-6 in the decoder config of the real decode program
        from glm_tpu.models.glm_moe_dsa.model import build_packed_decoder_program

        frozen = fixture.fixture_v1(panel_geometry=True)
        config = replace(frozen.config, rms_norm_epsilon=1e-6)
        spec = specs["decode@1536"]
        variant = programs.ProgramSpec("decode", build_packed_decoder_program(mesh, config).execute, spec.args)
        out["epsilon"] = programs.fingerprint_specs({"v": variant}, summary=False)["v"]

        # (i) one finished-lane jnp.where mask removed from the batched decode body (mutated copy
        # of the decode model module; the production file is never touched)
        import types

        from glm_tpu.models.glm_moe_dsa import model

        path = REPO / "glm_tpu" / "models" / "glm_moe_dsa" / "model.py"
        source = path.read_text()
        mutated = source.replace("next_token = jnp.where(active, next_token, token_ids)", "next_token = next_token")
        if mutated == source:
            raise RuntimeError("self-test mutation site (i) not found; update selftest.py")
        module = types.ModuleType("glm_tpu.models.glm_moe_dsa._equivalence_selftest_unmasked")
        module.__package__ = "glm_tpu.models.glm_moe_dsa"
        sys.modules[module.__name__] = module  # dataclasses resolve their defining module
        exec(compile(mutated, str(path), "exec"), module.__dict__)
        original = model.build_decoder_program
        model.build_decoder_program = module.build_decoder_program
        try:
            batch_spec = specs["batch_decode@1536#n4"]
            fn = model.build_batched_decoder_program(mesh, frozen.config, batch_size=4)
            variant = programs.ProgramSpec("batch_decode", fn, batch_spec.args)
            out["unmasked"] = programs.fingerprint_specs({"v": variant}, summary=False)["v"]
        finally:
            model.build_decoder_program = original
            sys.modules.pop(module.__name__, None)
        out["batch"] = base_batch
    return out


def relocated_decode(workdir: Path) -> dict[str, Any]:
    """(i)+(ii): the fixture decode lowered from a copy of the tree at another path, with ten blank
    lines prepended to a production kernel module and a layer module."""
    root = workdir / "relocated-source-tree-copy"
    for name in PRODUCTION_PATHS:  # the 181c013e production paths that exist in this tree
        if (REPO / name).is_dir():
            shutil.copytree(REPO / name, root / name, ignore=shutil.ignore_patterns("__pycache__"))
    for relative in ("glm_tpu/kernels/sparse_mla/kernel.py", "glm_tpu/layers/attention/mla.py"):
        path = root / relative
        path.write_text("\n" * 10 + path.read_text())
    return run_child(
        "tools.equivalence.programs", "--tier", "fixture", "--only", "decode@1536", source_root=root, timeout=1800
    )["programs"]["decode@1536"]


def run() -> dict[str, Any]:
    results = synthetic_cases()
    real = run_child("tools.equivalence.selftest", "--fixture-cases", timeout=1800)
    results.append(
        _case("a", "sensitivity", "RMS epsilon 1e-5 -> 1e-6 (fixture decode)", real["decode"], real["epsilon"])
    )
    results.append(
        _case(
            "i",
            "sensitivity",
            "finished-lane jnp.where mask removed (fixture batch_decode)",
            real["batch"],
            real["unmasked"],
        )
    )
    with tempfile.TemporaryDirectory(prefix="glm-equivalence-selftest-") as scratch:
        moved = relocated_decode(Path(scratch))
    results.append(
        _case(
            "i+ii",
            "invariance",
            "fixture decode from a relocated tree copy with 10 blank lines "
            "prepended to a kernel module and a layer module",
            real["decode"],
            moved,
        )
    )
    failed = [r["case"] for r in results if not r["passed"]]
    return dict(
        gate="G14",
        status="pass" if not failed else "fail",
        failed=failed,
        cases=results,
        environment=environment(),
        source=source_record(),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="G14 mutation self-test")
    parser.add_argument("--fixture-cases", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.fixture_cases:
        emit(fixture_cases())
        return 0
    result = run()
    emit(result)
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())

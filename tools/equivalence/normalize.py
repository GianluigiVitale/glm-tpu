"""N2-N8: normalization of location-free TPU StableHLO text, the signature record and a
structural summary. Nothing outside this list is rewritten; everything else is byte-compared.

N2  text = ``lowered.as_text(debug_info=False)``
N3  strip residual ``loc(...)`` suffixes and ``#loc`` lines (defensive; N1 leaves none)
N4  delete ``jax.result_info = "..."`` / ``jax.arg_info = "..."`` (pytree-path strings)
N5  ``module @<name>`` -> ``module @jit_main``; private functions -> ``@f0, @f1, ...`` in definition
    order, with every reference rewritten
N6  per ``stablehlo.custom_call @tpu_custom_call``: the base64 Mosaic ``body`` inside the escaped
    ``backend_config`` becomes ``<mosaic:sha256=...>`` (hash of the decoded bytes, so the body
    stays compared); ``kernel_name`` and every other field stay verbatim
N7  everything else (shardings, aliasing/donation attributes, num_partitions, frontend
    attributes, constants, op order) is compared byte for byte
N8  signature record, compared separately: flattened input avals in order (shape, dtype,
    canonical ``PartitionSpec`` -- trailing unsharded ``None`` entries dropped, so equivalent
    spellings such as ``P(None, 'expert')`` and ``P(None, 'expert', None)`` agree --, donated),
    output avals in order, counts

Besides the text and the signature, every record carries the XLA compiler options bound to the
``Lowered`` itself (``jax.jit(..., compiler_options=...)``: ``jit_compiler_options``). They never
reach the StableHLO text, but ``Lowered.compile`` merges them into what the TPU compiler receives,
so they are compared with the digest and the signature.
"""

from __future__ import annotations

import base64
import binascii
from collections import Counter
from hashlib import sha256
import json
import re
from typing import Any

_STRING = r'"(?:[^"\\]|\\.)*"'
_STRING_RE = re.compile(_STRING)
_NAME = r'(?:' + _STRING + r'|[A-Za-z_][\w$.\-]*)'
_SYMBOL = re.compile(r'@(' + _NAME + r')')
_PRIVATE = re.compile(r'func\.func private @(' + _NAME + r')')
_MODULE = re.compile(r'\A(module @)(' + _NAME + r')')
_PYTREE_NAME = re.compile(r'(, )?jax\.(?:result|arg)_info = ' + _STRING + r'(, )?')
_BODY = re.compile(r'(\\22body\\22: \\22)([A-Za-z0-9+/=]*)(\\22)')
_KERNEL_NAME = re.compile(r'kernel_name = (' + _STRING + r')')
_OP = re.compile(r'\b((?:stablehlo|sdy|mhlo|chlo|func|tpu)\.[a-z_]+)\b')
_COLLECTIVE = re.compile(r'stablehlo\.(all_reduce|all_gather|reduce_scatter|all_to_all|collective_permute|'
                         r'collective_broadcast)"?.*?'
                         r'(replica_groups = dense<[^>]*>|source_target_pairs = dense<[^>]*>)')
_ARG = re.compile(r'%arg(\d+): ')


def stablehlo_text(lowered: Any) -> str:
    """N2."""
    return lowered.as_text(debug_info=False)


def strip_locations(text: str) -> str:
    """N3: remove ``#loc`` definition lines and `` loc(...)`` suffixes outside string literals."""
    if "loc(" not in text and "#loc" not in text:
        return text
    text = "\n".join(line for line in text.split("\n") if not line.startswith("#loc"))
    out: list[str] = []
    i, n = 0, len(text)
    while i < n:
        ch = text[i]
        if ch == '"':
            match = _STRING_RE.match(text, i)
            end = match.end() if match else n
            out.append(text[i:end])
            i = end
            continue
        if text.startswith(" loc(", i):
            depth, j = 0, i + 4
            while j < n:
                c = text[j]
                if c == '"':
                    match = _STRING_RE.match(text, j)
                    j = match.end() if match else n
                    continue
                if c == "(":
                    depth += 1
                elif c == ")":
                    depth -= 1
                    if depth == 0:
                        j += 1
                        break
                j += 1
            i = j
            continue
        out.append(ch)
        i += 1
    return "".join(out)


def strip_pytree_names(text: str) -> str:
    """N4."""
    return _PYTREE_NAME.sub(lambda m: ", " if m.group(1) and m.group(2) else "", text)


def canonical_symbols(text: str) -> str:
    """N5."""
    text = _MODULE.sub(r"\1jit_main", text, count=1)
    names: dict[str, str] = {}
    for match in _PRIVATE.finditer(text):
        names.setdefault(match.group(1), f"f{len(names)}")
    if not names:
        return text
    return _SYMBOL.sub(lambda m: "@" + names.get(m.group(1), m.group(1)), text)


def mosaic_bodies(text: str, *, full_mask: bool = False) -> tuple[str, list[list[str]]]:
    """N6. Returns the rewritten text and ``[kernel_name, body_sha256]`` per custom call in order.

    ``full_mask`` replaces each body with ``<mosaic>`` (no hash), keeping kernel count, kernel
    names and operand/result types: used only by the adapter-authenticity comparison against
    TPU-host originals, whose bodies embed source locations.
    """
    kernels: list[list[str]] = []
    if "@tpu_custom_call" not in text:
        return text, kernels
    lines = text.split("\n")
    for index, line in enumerate(lines):
        if "@tpu_custom_call" not in line:
            continue
        bodies = _BODY.findall(line)
        names = _KERNEL_NAME.findall(line)
        if len(bodies) != 1 or len(names) != 1:
            raise ValueError(f"unexpected tpu_custom_call form: {len(bodies)} bodies, {len(names)} kernel names")
        try:
            raw = base64.b64decode(bodies[0][1], validate=True)
        except binascii.Error as exc:
            raise ValueError("tpu_custom_call body is not base64") from exc
        digest = sha256(raw).hexdigest()
        kernels.append([names[0][1:-1], digest])
        replacement = "<mosaic>" if full_mask else f"<mosaic:sha256={digest}>"
        lines[index] = _BODY.sub(lambda m, r=replacement: m.group(1) + r + m.group(3), line)
    return "\n".join(lines), kernels


def normalize(text: str, *, full_mask: bool = False) -> tuple[str, list[list[str]]]:
    """Apply N3-N6 to N2 text. Returns (normalized text, kernel list)."""
    text = strip_locations(text)
    text = strip_pytree_names(text)
    text = canonical_symbols(text)
    return mosaic_bodies(text, full_mask=full_mask)


# ----------------------------------------------------------------------------- N8 signature
def canonical_spec(spec: Any) -> Any:
    """``spec`` without trailing ``None`` entries (unsharded trailing dimensions). A
    ``PartitionSpec`` shorter than the array rank leaves the remaining dimensions unsharded, so
    ``P(None, 'expert')`` and ``P(None, 'expert', None)`` describe the same placement of a rank-3
    array (identical lowering); ``unreduced``/``reduced`` axes are kept."""
    entries = list(spec)
    while entries and entries[-1] is None:
        entries.pop()
    return spec.update(partitions=tuple(entries)) if hasattr(spec, "update") else type(spec)(*entries)


def _spec(sharding: Any) -> str | None:
    if sharding is None:
        return None
    spec = getattr(sharding, "spec", None)
    if spec is not None:
        return str(canonical_spec(spec))
    return type(sharding).__name__


def sharding_key(sharding: Any) -> Any:
    """Hashable, spelling-independent identity of a sharding (mesh, canonical spec, memory kind)
    for ``NamedSharding``; any other sharding is its own key."""
    spec = getattr(sharding, "spec", None)
    mesh = getattr(sharding, "mesh", None)
    if spec is None or mesh is None:
        return sharding
    return ("named", mesh, str(canonical_spec(spec)), getattr(sharding, "memory_kind", None))


def signature(abstract_args: tuple[Any, ...], lowered: Any) -> dict[str, Any]:
    import jax

    args = jax.tree.leaves(abstract_args)
    infos = jax.tree.leaves(lowered.args_info, is_leaf=lambda x: hasattr(x, "donated"))
    if len(args) != len(infos):
        raise ValueError("argument/args_info leaf counts differ")
    inputs = [
        [list(map(int, a.shape)), str(a.dtype), _spec(getattr(a, "sharding", None)), bool(info.donated)]
        for a, info in zip(args, infos, strict=True)
    ]
    outs = jax.tree.leaves(lowered.out_info, is_leaf=lambda x: hasattr(x, "shape") and hasattr(x, "dtype"))
    outputs = [[list(map(int, o.shape)), str(o.dtype)] for o in outs]
    return dict(inputs=inputs, outputs=outputs, input_count=len(inputs), output_count=len(outputs))


def _digest(value: Any) -> str:
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


# ----------------------------------------------------------------------------- structure
def structure(normalized: str, kernels: list[list[str]]) -> dict[str, Any]:
    """Diagnosis only (never the pass criterion)."""
    ops = Counter(m.group(1) for m in _OP.finditer(normalized))
    collectives: Counter[str] = Counter()
    for line in normalized.split("\n"):
        match = _COLLECTIVE.search(line)
        if match:
            collectives[f"{match.group(1)} {match.group(2)}"] += 1
    main = next((line for line in normalized.split("\n") if "func.func public @main" in line), "")
    kernel_table: dict[str, dict[str, int]] = {}
    for name, digest in kernels:
        bucket = kernel_table.setdefault(name, {})
        bucket[digest[:16]] = bucket.get(digest[:16], 0) + 1
    return dict(
        ops=dict(sorted(ops.items())),
        collectives=dict(sorted(collectives.items())),
        kernels=dict(sorted(kernel_table.items())),
        kernel_calls=len(kernels),
        parameters=len(_ARG.findall(main)),
        bytes=len(normalized.encode()),
    )


def _option_value(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return repr(value)
    if isinstance(value, (bytes, bytearray)):
        return "<bytes:sha256=" + sha256(bytes(value)).hexdigest() + ">"
    return f"<{type(value).__name__}:{value!r}>"


def jit_compiler_options(lowered: Any) -> list[list[Any]]:
    """The compiler options bound to ``lowered`` by ``jax.jit(..., compiler_options=...)``, in
    order (``Lowered.compile`` appends its own ``compiler_options`` argument to them and hands the
    merged options to the XLA compiler). The StableHLO text does not contain them. Read from a
    private jax attribute -- the records are bound to the jax version anyway -- and fail closed
    when it is missing."""
    lowering = getattr(lowered, "_lowering", None)
    kvs = getattr(lowering, "_compiler_options_kvs", None)
    if lowering is None or kvs is None:
        raise RuntimeError("jax internals changed: Lowered._lowering._compiler_options_kvs is missing; "
                           "update normalize.jit_compiler_options for this jax version")
    return [[str(key), _option_value(value)] for key, value in kvs]


def fingerprint(lowered: Any, abstract_args: tuple[Any, ...], *, summary: bool = True) -> dict[str, Any]:
    text = stablehlo_text(lowered)
    normalized, kernels = normalize(text)
    sig = signature(abstract_args, lowered)
    record = dict(
        digest=sha256(normalized.encode()).hexdigest(),
        signature_digest=_digest(sig),
        jit_compiler_options=jit_compiler_options(lowered),
        input_count=sig["input_count"],
        output_count=sig["output_count"],
        donated=[i for i, row in enumerate(sig["inputs"]) if row[3]],
    )
    if summary:
        record["summary"] = structure(normalized, kernels)
    return record

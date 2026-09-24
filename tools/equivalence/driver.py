"""Drive the real ``OrdinaryRuntime`` (``__init__``, ``_load``, ``compile``, ``generate``) on the CPU host.

The gates build their device programs and CPU goldens by running production's own runtime code,
so an edit to ``__init__``, ``_load``, ``compile``, ``compile_program``, ``generate`` or
``compile_batch`` (donation, options, config flags, sample shapes, the tail rule, what is lowered
and how it is compiled) reaches the fingerprints and goldens. Only what needs a TPU fleet,
private assets or the TPU compiler is replaced:

* checkpoint I/O: ``authenticated_inventory`` returns the synthetic GLM-5.3 inventory with the
  pinned digest string; ``verify_ws32_runtime_checkpoint`` returns the file plans (synthetic
  production plans, or placeholder plans for the fixture); ``load_ws32_runtime_checkpoint``
  returns the fixture arrays (concrete) or ``ShapeDtypeStruct`` values with the plans' shardings
  (abstract). Their keyword arguments are recorded. ``model.require_site`` and
  ``model.require_inventory`` run for real on pinned-identity inputs;
* the TPU compiler: the real ``OrdinaryRuntime.compile`` and ``compile_program`` run. While
  ``compile`` runs, ``jax.stages.Traced.lower`` lowers for the TPU platform (what ``fn.lower`` does
  on the fleet) and ``jax.stages.Lowered.compile`` records its arguments, fingerprints the
  ``Lowered`` production built (N1-N8, when requested) and returns a stand-in executable: it runs
  the jitted function on the CPU mesh (concrete) or returns ``ShapeDtypeStruct`` outputs with the
  CPU-compiled executable's shardings (abstract), reports zero compiler memory and a stand-in
  optimized-HLO text. The HLO admission parser (``inspect_research_hlo``) cannot read a TPU
  optimized module that does not exist here: it is replaced by a recorder that checks it was
  handed the stand-in text read back from the HLO directory;
* the fleet: votes are identity (``phase`` runs for real and records the phase sequence), and
  ``process_allgather`` stacks the local value eight times (payload sizes recorded); a probe
  proves the real graph-consensus phase rejects a divergent host;
* device memory: ``stats`` reports four idle synthetic chips (1 TiB), so the real ``admit`` /
  ``admit_memory`` / ``memory_projection`` run; every admission request is recorded;
* the host: every file production writes or reads under ``/dev/shm`` (the HLO-originals
  directory: ``mkdir``, the StableHLO and optimized-HLO originals, ``runner.json``) lives in an
  in-memory overlay (recorded relative to the runtime's ``hlo`` directory; nothing is ever written
  to ``/dev/shm``), any other ``pathlib`` write during the build is refused, and the free-space
  probe sees 1 TiB;
* abstract mode only: ``ShapeDtypeStruct.addressable_shards`` answers the per-shard byte probe
  ``_load`` uses for the BF16 preparation admission.

Fixture tier (frozen fixture v1): the one ``Ws32DecoderConfig`` construction in ``__init__`` is
adjusted at the class (``_config_injection`` wraps the class's ``__init__``, so every import
binding sees it): the pinned production geometry ``model.geometry`` returns becomes the fixture
geometry, and ``sparse_segment_block=128`` is added when ``__init__`` passes none (the fixture's
DSA top-k is 128, so the production default cannot validate); the recorded protocol keeps the call
``__init__`` made. No program-shaping constant is patched: the donated fixture run uses a capacity
above 8,192, so production's own rule donates wherever that rule lives. The only other fixture
relaxation is the concurrent guard (``request.CONCURRENT_CAPACITY`` set to the fixture capacity,
so a 1,536-slot runtime may be concurrent); it shapes no program, and if the guard moves the build
fails loudly. Production tier: no override; production geometry, capacities and donation rule as
they are.
"""

from __future__ import annotations

from contextlib import ExitStack, contextmanager
import copy
import dataclasses
from dataclasses import dataclass, field
from hashlib import sha256
import importlib
import inspect
import io
import pathlib
import re
import shutil
import sys
import tempfile
import time
from types import SimpleNamespace
from typing import Any, Callable, Iterator
from unittest import mock

import numpy as np

from .common import REPO

RUNTIME_MODULE = "glm_tpu.optimized.runtime"
RUNTIME_CLASS = "OrdinaryRuntime"
PROGRAMS_MODULE = "glm_tpu.runner.programs"  # S2c: the one production program builder
REQUEST_MODULE = "glm_tpu.optimized.request"
BATCHED_MODULE = "glm_tpu.optimized.batched_runtime"
DECODER_MODULE = "glm_tpu.optimized.ws32_decoder"
CONFIG_CLASS = "Ws32DecoderConfig"          # the config __init__ builds (adjusted at the class, fixture tier)
TMPFS = "/dev/shm"                         # production's HLO originals live here; the harness never writes it
FIXTURE_SEGMENT_BLOCK = 128                # the only fixture override of the config __init__ builds
INTERPRET = dict(sparse_attention_interpret=True, linear_interpret=True)
SYNTHETIC_HBM = 1 << 40                    # bytes_limit of the four synthetic chips ``stats`` reports
# Every module that defines a faked loader function is patched: the 181c013e home first, then the
# homes DESIGN.md S2a moves them to. A move elsewhere leaves the real function in place, which
# fails on the placeholder arguments (fail-closed), and ``stub never called`` names it.
HOMES = {
    "authenticated_inventory": ("scripts.greenfield.ws32_compile_originals",
                                "glm_tpu.optimized.source_inventory"),
    "verify_ws32_runtime_checkpoint": ("glm_tpu.optimized.runtime_checkpoint",),
    "load_ws32_runtime_checkpoint": ("glm_tpu.optimized.runtime_checkpoint",),
}
# Where ``compile`` looks up the HLO admission parser (181c013e: imported into the runtime module).
# A move elsewhere leaves the real parser in place, which refuses the stand-in text (fail-closed).
ADMISSION_HOMES = (RUNTIME_MODULE, "glm_tpu.optimized.admission")
# Builders the prefill and decode programs come from, by role, under every name a home has bound
# them (S2d c3: ``build_prefill_program``; before, ``build_ws32_prefill_challenger_program``), and
# the homes that look them up: ``build_program_set`` (S2c) or, at 181c013e, ``_load`` itself. G3 runs
# them in interpret mode; every binding in an existing home is patched and each role must be bound
# somewhere. A builder bound anywhere else runs TPU kernels on CPU and crashes (fail-closed).
INTERPRET_BUILDERS = {"prefill": ("build_prefill_program", "build_ws32_prefill_challenger_program"),
                      "decode": ("build_packed_decoder_program",)}
INTERPRET_HOMES = (PROGRAMS_MODULE, RUNTIME_MODULE)
# Where ``_load`` looks up ``build_program_set`` (S2c): the harness records every ProgramSet built
# while the runtime is constructed and checks the runtime compiled exactly its programs.
PROGRAM_SET_HOME = (RUNTIME_MODULE, "build_program_set")
MEMORY_FIELDS = ("argument_size_in_bytes", "output_size_in_bytes", "alias_size_in_bytes", "temp_size_in_bytes",
                 "generated_code_size_in_bytes")
STANDIN_HLO = "HloModule glm_equivalence_standin, program={name}\n"
_STANDIN_HLO = re.compile(r"\AHloModule glm_equivalence_standin, program=(.+)\n\Z")


@dataclass(frozen=True)
class ProgramSpec:
    name: str                          # production program name (HLO original file name)
    fn: Callable[..., Any]             # the jax.jit object production handed to ``compile`` (donation applied)
    args: tuple[Any, ...]              # arguments production passes (arrays or ShapeDtypeStructs)
    compile_calls: Any = None          # arguments of each ``Lowered.compile`` (None: compiled by jit dispatch)
    record: dict[str, Any] | None = None  # fingerprint of the ``Lowered`` production compiled (when requested)
    lowered: Any = None                # that ``Lowered`` (only when kept, e.g. for ``diff``/``authenticity``)


def leaf_signature(leaf: Any) -> tuple[Any, ...]:
    """(shape, dtype, sharding) with the sharding canonicalized (``normalize.sharding_key``), so
    equivalent ``PartitionSpec`` spellings of the same placement compare equal."""
    from .normalize import sharding_key

    sharding = getattr(leaf, "sharding", None)
    return (tuple(int(d) for d in leaf.shape), str(leaf.dtype), None if sharding is None else sharding_key(sharding))


def abstract_like(tree: Any) -> Any:
    import jax

    return jax.tree.map(lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype, sharding=x.sharding), tree)


def _cpu_compile(fn: Any, args: tuple[Any, ...]) -> Any:
    """The harness's own CPU compile (never the intercepted ``Lowered.compile``)."""
    from jax._src import stages

    lowered = fn.trace(*args).lower(lowering_platforms=("cpu",))
    return _REAL_LOWERED_COMPILE(lowered) if _REAL_LOWERED_COMPILE else stages.Lowered.compile(lowered)


_REAL_LOWERED_COMPILE: Any = None


class ProgramRecorder:
    """Records ``(name, fn, args)`` in call order (the real runtime compiles through
    ``ProductionCompile``). ``executor`` is what a compiled program does here: concrete, run
    ``fn`` on the CPU mesh (as ``_load`` executes producers on TPU); abstract, return
    ``ShapeDtypeStruct`` outputs with the CPU-compiled executable's shardings."""

    def __init__(self, *, concrete: bool, outputs: dict[Any, Any] | None = None):
        self.concrete = concrete
        self.context: Any = None
        self.specs: list[ProgramSpec] = []
        self._outputs = {} if outputs is None else outputs  # may be shared across runtimes of a tier

    def executor(self, fn: Any, name: str) -> Callable[..., Any]:
        return fn if self.concrete else self.abstract(fn, name)

    def abstract(self, fn: Any, name: str) -> Callable[..., Any]:
        import jax

        context = self.context

        def call(*args: Any) -> Any:
            leaves, treedef = jax.tree.flatten(args)
            key = (name, context, treedef, tuple(leaf_signature(x) for x in leaves))
            if key not in self._outputs:
                compiled = _cpu_compile(fn, args)
                infos = jax.tree.leaves(compiled.out_info)
                shardings = jax.tree.leaves(compiled.output_shardings)
                out = [jax.ShapeDtypeStruct(i.shape, i.dtype, sharding=s)
                       for i, s in zip(infos, shardings, strict=True)]
                self._outputs[key] = jax.tree.unflatten(jax.tree.structure(compiled.out_info), out)
            return self._outputs[key]

        return call

    def program(self, name: str) -> ProgramSpec:
        matches = [spec for spec in self.specs if spec.name == name]
        if len(matches) != 1:
            raise KeyError(f"{len(matches)} recorded programs named {name}")
        return matches[0]


def fp8_table_name(key: tuple[Any, ...]) -> str:
    bits, scale, spec, block = key
    render = ",".join("None" if axis is None else str(axis) for axis in spec)
    return (f"fp8_table[{'x'.join(map(str, bits))}/{'x'.join(map(str, scale))}/({render})"
            f"/{'x'.join(map(str, block))}]")


@contextmanager
def recording_tables(r: ProgramRecorder) -> Iterator[None]:
    """Capture every per-table decoder ``bf16_resident_weights`` builds, in call order (production
    compiles them implicitly on first call; a fresh process starts with an empty cache)."""
    from glm_tpu.optimized import bf16_resident

    original = bf16_resident._decode_program
    saved = dict(bf16_resident._DECODERS)
    bf16_resident._DECODERS.clear()
    seen: set[Any] = set()

    def recording(mesh: Any, bits: Any, scale: Any, spec: Any, block: tuple[int, int]) -> Any:
        program = original(mesh, bits, scale, spec, block)
        key = (tuple(bits.shape), tuple(scale.shape), tuple(spec), tuple(block))
        if key not in seen:
            seen.add(key)
            r.specs.append(ProgramSpec(fp8_table_name(key), program, (bits, scale)))
        return program if r.concrete else r.abstract(program, fp8_table_name(key))

    bf16_resident._decode_program = recording
    try:
        yield
    finally:
        bf16_resident._decode_program = original
        bf16_resident._DECODERS.clear()
        bf16_resident._DECODERS.update(saved)


# ----------------------------------------------------------------------------- the real compile path
class StandInCompiled:
    """What ``Lowered.compile`` returns while the harness builds a runtime (no TPU compiler on the
    CPU host): calling it runs the harness executor; memory analysis reports zeros; ``as_text``
    is a stand-in optimized-HLO module naming the program."""

    def __init__(self, name: str, call: Callable[..., Any]):
        self.name, self._call = name, call

    def __call__(self, *args: Any) -> Any:
        return self._call(*args)

    def memory_analysis(self) -> Any:
        return SimpleNamespace(**{name: 0 for name in MEMORY_FIELDS})

    def as_text(self) -> str:
        return STANDIN_HLO.format(name=self.name)


class ProductionCompile:
    """Runs the real ``OrdinaryRuntime.compile`` (and through it ``compile_program``, the
    graph-consensus phase, the HLO admission and the memory admission) and records what it did.

    ``wrap`` returns the ``compile`` the runtime instance uses: it sets the program context, calls
    the real class method, and records ``ProgramSpec(name, fn, values, compile_calls, record)``.
    ``patches`` intercepts ``Traced.lower`` (TPU platform while a program compiles, as on the
    fleet) and ``Lowered.compile`` (records its arguments, fingerprints the ``Lowered`` production
    built, returns a ``StandInCompiled``), fakes the HLO admission parser and ``process_allgather``.
    """

    def __init__(self, recorder: ProgramRecorder, *, fingerprint: bool, keep: Callable[[str], bool] | None):
        self.recorder = recorder
        self.fingerprint = fingerprint
        self.keep = keep or (lambda name: False)
        self.current: dict[str, Any] | None = None
        self.hlo_admissions: list[str] = []
        self.consensus: list[int] = []
        self.divergent = False

    def wrap(self, runtime: Any, cls: Any) -> Callable[..., Any]:
        real = cls.compile

        def compile(name: str, fn: Any, values: Any, *args: Any, **kwargs: Any) -> Any:
            self.current = dict(name=name, fn=fn, values=tuple(values), compiles=[], record=None, lowered=None)
            try:
                exe = real(runtime, name, fn, values, *args, **kwargs)
            finally:
                current, self.current = self.current, None
            self.recorder.specs.append(ProgramSpec(name, fn, tuple(values), compile_calls=current["compiles"],
                                                   record=current["record"], lowered=current["lowered"]))
            return exe

        return compile

    @contextmanager
    def patches(self) -> Iterator[None]:
        global _REAL_LOWERED_COMPILE
        from jax._src import stages
        from jax.experimental import multihost_utils

        real_lower, real_compile = stages.Traced.lower, stages.Lowered.compile
        _REAL_LOWERED_COMPILE = real_compile

        def lower(traced: Any, *, lowering_platforms: Any = None, **kwargs: Any) -> Any:
            if self.current is not None and lowering_platforms is None:
                lowering_platforms = ("tpu",)  # jax on the fleet lowers for its default backend
            return real_lower(traced, lowering_platforms=lowering_platforms, **kwargs)

        def compile(lowered: Any, compiler_options: Any = None, **kwargs: Any) -> Any:
            context = self.current
            if context is None:
                return real_compile(lowered, compiler_options, **kwargs)
            context["compiles"].append(dict(compiler_options=_describe_options(compiler_options),
                                            **{k: describe(v) for k, v in sorted(kwargs.items())}))
            if self.fingerprint:
                from . import normalize

                started = time.perf_counter()
                record = normalize.fingerprint(lowered, context["values"], summary=True)
                record["seconds"] = round(time.perf_counter() - started, 1)
                context["record"] = record
            if self.keep(context["name"]):
                context["lowered"] = lowered
            return StandInCompiled(context["name"], self.recorder.executor(context["fn"], context["name"]))

        def allgather(value: Any) -> Any:
            array = np.asarray(value)
            self.consensus.append(int(array.nbytes))
            rows = np.stack([array] * 8)
            if self.divergent:  # the consensus probe: one host reports a different graph
                rows = rows.copy()
                rows[3] = rows[3] ^ np.asarray(1, rows.dtype)
            return rows

        def admission(text: str) -> dict[str, Any]:
            match = _STANDIN_HLO.match(text)
            if match is None or self.current is None or match.group(1) != self.current["name"]:
                raise ValueError("HLO admission was not handed the stand-in module of the program being compiled")
            self.hlo_admissions.append(match.group(1))
            return dict(passed=True, profile="glm-equivalence stand-in (no TPU optimized HLO on the CPU host)")

        with ExitStack() as stack:
            stack.enter_context(mock.patch.object(stages.Traced, "lower", lower))
            stack.enter_context(mock.patch.object(stages.Lowered, "compile", compile))
            stack.enter_context(mock.patch.object(multihost_utils, "process_allgather", allgather))
            for module_name in ADMISSION_HOMES:
                module = importlib.import_module(module_name)
                if hasattr(module, "inspect_research_hlo"):
                    stack.enter_context(mock.patch.object(module, "inspect_research_hlo", admission))
            yield

    def consensus_probe(self, runtime: Any, cls: Any) -> str:
        """The real ``compile`` of a tiny program while one host reports a different graph digest:
        the graph-consensus phase must refuse it. ``runtime.record`` is restored afterwards."""
        import jax

        saved = copy.deepcopy(runtime.record)
        fn = jax.jit(lambda value: value + 1)
        values = (runtime.put(np.int32(0)),)
        self.current = dict(name="consensus_probe", fn=fn, values=values, compiles=[], record=None, lowered=None)
        self.divergent = True
        try:
            cls.compile(runtime, "consensus_probe", fn, values, model=False)
            outcome = "accepted a divergent host"
        except Exception as exc:  # the expected refusal
            outcome = f"{type(exc).__name__}: {exc}"
        finally:
            self.divergent, self.current = False, None
            runtime.record = saved
        return outcome


def _describe_options(options: Any) -> Any:
    if options is None:
        return None
    if isinstance(options, dict):
        return {str(k): describe(v) for k, v in sorted(options.items(), key=lambda kv: str(kv[0]))}
    return describe(options)


# ----------------------------------------------------------------------------- /dev/shm overlay
class TmpfsOverlay:
    """In-memory stand-in for everything production writes under ``/dev/shm`` while the runtime
    is built (the HLO originals directory). Reads of written files come from memory; large texts
    keep only their size and digest (reading one back fails closed). Any other ``pathlib`` write
    during the build is refused, so the harness never writes production files to disk."""

    KEEP_TEXT = 1 << 20

    def __init__(self) -> None:
        self.files: dict[str, tuple[int, str, str | None]] = {}
        self.mkdirs: list[dict[str, Any]] = []
        self.probes: list[str] = []

    @staticmethod
    def owns(path: Any) -> bool:
        text = str(path)
        return text == TMPFS or text.startswith(TMPFS + "/")

    def relative(self, text: str, hlo: pathlib.Path | None) -> str:
        if hlo is not None and (text == str(hlo) or text.startswith(str(hlo) + "/")):
            return "<hlo>" + text[len(str(hlo)):]
        return "<tmpfs>" + text[len(TMPFS):]

    @contextmanager
    def active(self) -> Iterator[None]:
        overlay = self
        real = {name: getattr(pathlib.Path, name) for name in
                ("mkdir", "write_text", "write_bytes", "read_text", "read_bytes", "replace", "rename", "exists",
                 "is_file", "open", "unlink", "touch")}

        def refuse(path: Any, operation: str) -> None:
            raise RuntimeError(f"the runtime build tried to {operation} outside {TMPFS} (the harness never writes "
                               "production files to disk); update driver.TmpfsOverlay if the HLO root moved")

        def mkdir(self: pathlib.Path, mode: int = 0o777, parents: bool = False, exist_ok: bool = False) -> None:
            if not overlay.owns(self):
                refuse(self, "mkdir")
            overlay.mkdirs.append(dict(path=str(self), mode=oct(mode), parents=parents, exist_ok=exist_ok))

        def write_text(self: pathlib.Path, data: str, *args: Any, **kwargs: Any) -> int:
            if not overlay.owns(self):
                refuse(self, "write")
            raw = data.encode()
            overlay.files[str(self)] = (len(raw), sha256(raw).hexdigest(), data if len(raw) <= overlay.KEEP_TEXT
                                        else None)
            return len(data)

        def write_bytes(self: pathlib.Path, data: bytes) -> int:
            if not overlay.owns(self):
                refuse(self, "write")
            overlay.files[str(self)] = (len(data), sha256(data).hexdigest(), None)
            return len(data)

        def stored(path: pathlib.Path) -> str:
            entry = overlay.files.get(str(path))
            if entry is None:
                raise FileNotFoundError(str(path))
            if entry[2] is None:
                raise RuntimeError(f"the harness kept only the digest of {path.name}; production read it back")
            return entry[2]

        def read_text(self: pathlib.Path, *args: Any, **kwargs: Any) -> str:
            return stored(self) if overlay.owns(self) else real["read_text"](self, *args, **kwargs)

        def read_bytes(self: pathlib.Path) -> bytes:
            return stored(self).encode() if overlay.owns(self) else real["read_bytes"](self)

        def replace(self: pathlib.Path, target: Any) -> pathlib.Path:
            if not overlay.owns(self) or not overlay.owns(target):
                refuse(self, "rename")
            overlay.files[str(target)] = overlay.files.pop(str(self))
            return pathlib.Path(target)

        def exists(self: pathlib.Path, *args: Any, **kwargs: Any) -> bool:
            if overlay.owns(self):
                text = str(self)
                return text in overlay.files or any(d["path"] == text for d in overlay.mkdirs)
            return real["exists"](self, *args, **kwargs)

        def is_file(self: pathlib.Path, *args: Any, **kwargs: Any) -> bool:
            return str(self) in overlay.files if overlay.owns(self) else real["is_file"](self, *args, **kwargs)

        def open_(self: pathlib.Path, mode: str = "r", *args: Any, **kwargs: Any) -> Any:
            if overlay.owns(self):
                if any(flag in mode for flag in "wax+"):
                    refuse(self, "open for writing")
                text = stored(self)
                return io.BytesIO(text.encode()) if "b" in mode else io.StringIO(text)
            if any(flag in mode for flag in "wax+"):
                refuse(self, "open for writing")
            return real["open"](self, mode, *args, **kwargs)

        def unlink(self: pathlib.Path, missing_ok: bool = False) -> None:
            if not overlay.owns(self):
                refuse(self, "unlink")
            if overlay.files.pop(str(self), None) is None and not missing_ok:
                raise FileNotFoundError(str(self))

        def touch(self: pathlib.Path, *args: Any, **kwargs: Any) -> None:
            if not overlay.owns(self):
                refuse(self, "touch")
            overlay.files.setdefault(str(self), (0, sha256(b"").hexdigest(), ""))

        def disk_usage(path: Any) -> Any:
            overlay.probes.append(str(path))
            return shutil._ntuple_diskusage(1 << 40, 0, 1 << 40)

        replacements = dict(mkdir=mkdir, write_text=write_text, write_bytes=write_bytes, read_text=read_text,
                            read_bytes=read_bytes, replace=replace, rename=replace, exists=exists, is_file=is_file,
                            open=open_, unlink=unlink, touch=touch)
        with ExitStack() as stack:
            for name, function in replacements.items():
                stack.enter_context(mock.patch.object(pathlib.Path, name, function))
            stack.enter_context(mock.patch.object(shutil, "disk_usage", disk_usage))
            yield

    def record(self, hlo: pathlib.Path | None) -> dict[str, Any]:
        """What production did under ``/dev/shm``, relative to the runtime's ``hlo`` directory."""
        mkdirs: list[dict[str, Any]] = []
        for request in self.mkdirs:
            entry = dict(request, path=self.relative(request["path"], hlo))
            if entry not in mkdirs:
                mkdirs.append(entry)
        return dict(mkdir=mkdirs, mkdir_calls=len(self.mkdirs),
                    files=sorted(self.relative(path, hlo) for path in self.files),
                    free_space_probes=[self.relative(p, hlo) if self.owns(p) else "<outside tmpfs>"
                                       for p in dict.fromkeys(self.probes)],
                    hlo_under_tmpfs=hlo is not None and self.owns(hlo))


# ----------------------------------------------------------------------------- recorded protocol
def describe(value: Any, known: dict[str, Any] | None = None) -> Any:
    """JSON-safe description of an argument production passed to a faked function."""
    for label, obj in (known or {}).items():
        if value is obj:
            return label
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, (tuple, list)):
        return [describe(v, known) for v in value]
    if hasattr(value, "geometry_hash"):
        return "geometry:" + value.geometry_hash
    return f"<{type(value).__name__}>"


def encode_default(value: Any) -> Any:
    """Structural encoding of a default value (dataclass instances by field, no class names)."""
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {f.name: encode_default(getattr(value, f.name)) for f in dataclasses.fields(value)}
    if isinstance(value, (tuple, list)):
        return [encode_default(v) for v in value]
    if isinstance(value, dict):
        return {str(k): encode_default(v) for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))}
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return repr(value)
    return f"<{type(value).__name__}>"


def config_record(config: Any) -> dict[str, Any]:
    return {f.name: describe(getattr(config, f.name)) for f in dataclasses.fields(config)}


# ----------------------------------------------------------------------------- stubs
class LoadStubs:
    """The faked checkpoint I/O; records how ``_load`` called it and every memory admission."""

    def __init__(self, arrays: dict[str, Any], plans: Any, inventory: Any):
        self.arrays, self.plans, self.inventory = arrays, plans, inventory
        self.checkpoint = SimpleNamespace(plans=plans, manifest=dict(manifest_sha256="<synthetic manifest>"),
                                          success=dict(success_sha256="<synthetic success>"))
        self.calls: dict[str, Any] = {}
        self.admissions: list[list[Any]] = []
        self.known: dict[str, Any] = {}

    def _record(self, name: str, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
        if name in self.calls:
            raise RuntimeError(f"_load called {name} twice")
        self.calls[name] = dict(args=[describe(a, self.known) for a in args],
                                kwargs={k: describe(v, self.known) for k, v in sorted(kwargs.items())})

    def authenticated_inventory(self, *args: Any, **kwargs: Any) -> Any:
        self._record("authenticated_inventory", args, kwargs)
        return self.inventory

    def verify_ws32_runtime_checkpoint(self, *args: Any, **kwargs: Any) -> Any:
        self._record("verify_ws32_runtime_checkpoint", args, kwargs)
        return self.checkpoint

    def load_ws32_runtime_checkpoint(self, *args: Any, **kwargs: Any) -> Any:
        self._record("load_ws32_runtime_checkpoint", args, kwargs)
        return SimpleNamespace(arrays=dict(self.arrays))

    def admission(self, name: str, memory: dict[str, Any]) -> None:
        self.admissions.append([name, {k: describe(v) for k, v in sorted(memory.items())}])


def fixture_plans(arrays: dict[str, Any]) -> list[Any]:
    """Placeholder file plan for the fixture (only feeds the checkpoint-load admission request)."""
    sizes = [int(np.prod(a.shape, dtype=np.int64)) * np.dtype(a.dtype).itemsize for a in arrays.values()]
    return [SimpleNamespace(payload_bytes=sum(sizes), tensors=[SimpleNamespace(byte_count=s) for s in sizes])]


def pinned_args() -> Any:
    """Worker arguments ``_load`` reads, as site_args would bind them (placeholders for paths and
    content pins the faked loader ignores; the model identity is the pinned one)."""
    from glm_tpu.optimized import model

    from .identities import INVENTORY_PIN

    # hlo_dump_root: the site default (S1a moved the dump root from a runtime literal to the site
    # file; paths under it are recorded relative to the runtime's ``hlo`` directory).
    from glm_tpu.config.site import DEFAULT_HLO_DUMP_ROOT

    return SimpleNamespace(model_id=model.MODEL_ID, model_revision=model.REVISION,
                           source_inventory="<source_inventory>", source_inventory_sha256=INVENTORY_PIN,
                           checkpoint_root="<checkpoint_root>", checkpoint_manifest_sha256="<manifest_sha256>",
                           checkpoint_success_sha256="<success_sha256>", mesh_sha256="<mesh_sha256>",
                           topology_sha256="<topology_sha256>", hlo_dump_root=pathlib.Path(DEFAULT_HLO_DUMP_ROOT))


def _patch_homes(stack: ExitStack, stubs: LoadStubs) -> None:
    for name, homes in HOMES.items():
        for module_name in homes:
            try:
                module = importlib.import_module(module_name)
            except ImportError:
                continue
            if hasattr(module, name):
                stack.enter_context(mock.patch.object(module, name, getattr(stubs, name)))


def _shard_probe(self: Any) -> list[Any]:
    """``x.addressable_shards[0].data.nbytes`` for an abstract value (bytes of one shard)."""
    shape = self.sharding.shard_shape(tuple(self.shape))
    nbytes = int(np.prod(shape, dtype=np.int64)) * np.dtype(self.dtype).itemsize
    return [SimpleNamespace(data=SimpleNamespace(nbytes=nbytes))]


def synthetic_stats() -> list[dict[str, Any]]:
    """Four idle local chips (``memory_projection`` requires exactly four); CPU devices report no
    allocator statistics."""
    import jax

    return [dict(device_id=int(d.id), bytes_in_use=0, bytes_limit=SYNTHETIC_HBM, peak_bytes_in_use=0)
            for d in jax.local_devices()[:4]]


@dataclass
class Built:
    runtime: Any
    recorder: ProgramRecorder
    protocol: dict[str, Any] = field(default_factory=dict)
    program_set: dict[str, Any] = field(default_factory=dict)  # ``programset_identity`` (S2c)


def programset_identity(sets: list[Any], specs: list[ProgramSpec]) -> dict[str, Any]:
    """Did the runtime compile exactly the programs of the one ``ProgramSet`` it built -- the same
    function objects, in the set's order? (The per-table FP8 decoders ``bf16_resident_weights``
    compiles by jit dispatch are not part of the set.) Before S2c no set exists: ``absent``."""
    compiled = [spec for spec in specs if not spec.name.startswith("fp8_table[")]
    if not sets:
        return dict(status="absent", identical=False, compiled=[spec.name for spec in compiled])
    if len(sets) != 1:
        return dict(status="built", identical=False, reason=f"{len(sets)} program sets built")
    expected = list(sets[0].specs())
    names = [spec.name for spec in expected] == [spec.name for spec in compiled]
    same = names and all(a.fn is b.fn for a, b in zip(expected, compiled, strict=True))
    return dict(status="built", identical=same, programs=[spec.name for spec in expected],
                compiled=[spec.name for spec in compiled], same_functions=same)


def runtime_class() -> Any:
    return getattr(importlib.import_module(RUNTIME_MODULE), RUNTIME_CLASS)


def build_runtime(mesh: Any, *, tier: str, capacity: int, concurrent_size: int, arrays: dict[str, Any], plans: Any,
                  concrete: bool, fixture_geometry: Any = None, interpret: bool = False,
                  outputs: dict[Any, Any] | None = None, fingerprint: bool = False,
                  keep: Callable[[str], bool] | None = None) -> Built:
    """Construct the real runtime with the real ``__init__`` (which calls the real ``_load``, whose
    ``compile`` calls run the real compile path). ``fingerprint``: fingerprint every ``Lowered``
    production compiles (inside ``location_free``); ``keep(name)``: keep that ``Lowered``."""
    import jax

    from . import lowering
    from .identities import synthetic_inventory

    request_module = importlib.import_module(REQUEST_MODULE)
    cls = runtime_class()
    recorder = ProgramRecorder(concrete=concrete, outputs=outputs)
    compiler = ProductionCompile(recorder, fingerprint=fingerprint, keep=keep)
    overlay = TmpfsOverlay()
    stubs = LoadStubs(arrays, plans, _pinned_inventory(synthetic_inventory))
    local = [int(d.id) for d in jax.local_devices()]
    physical = SimpleNamespace(mesh_hash="<synthetic mesh>",
                               flattened_device_ids=tuple(local[:4]) + tuple(100000 + i for i in range(28)))
    topology = SimpleNamespace(topology_hash="<synthetic topology>")
    phases: list[str] = []
    config_calls: list[Any] = []
    program_sets: list[Any] = []

    runtime = object.__new__(cls)
    # Instance attributes shadow these methods for the whole life of this runtime (generate
    # included); ``compile`` and ``admit_memory`` record and then run the real class methods.
    runtime.compile = compiler.wrap(runtime, cls)
    runtime.stats = synthetic_stats

    def admit_memory(name: str, memory: dict[str, Any]) -> Any:
        stubs.admission(name, memory)
        return cls.admit_memory(runtime, name, memory)

    runtime.admit_memory = admit_memory

    def vote(valid: Any) -> bool:
        return bool(valid)

    real_phase = runtime.phase

    def phase(name: str, action: Any) -> Any:
        phases.append(name)
        return real_phase(name, action)

    runtime.phase = phase
    with tempfile.TemporaryDirectory(prefix="glm-equivalence-runtime-") as scratch, ExitStack() as stack:
        root = pathlib.Path(scratch) / "run" / "native.rank0"
        root.mkdir(parents=True)

        def load(repo: Any, physical_mesh: Any) -> Any:
            recorder.context = runtime.config  # abstract outputs are cached per config across runtimes
            stubs.known.update({"<runtime mesh>": runtime.mesh, "<physical mesh>": physical_mesh,
                                "<authenticated inventory>": stubs.inventory,
                                "<verified checkpoint>": stubs.checkpoint})
            with recording_tables(recorder):
                return cls._load(runtime, repo, physical_mesh)

        runtime._load = load
        _patch_homes(stack, stubs)  # imports the stubs' homes before any other patch is active
        constructed: list[Any] = []
        stack.enter_context(_config_injection(tier, fixture_geometry, config_calls, constructed))
        if not concrete:
            stack.enter_context(mock.patch.object(jax.ShapeDtypeStruct, "addressable_shards", property(_shard_probe),
                                                  create=True))
        if tier == "fixture" and concurrent_size:  # the concurrent guard only (no program depends on it)
            stack.enter_context(mock.patch.object(request_module, "CONCURRENT_CAPACITY", capacity))
        if interpret:
            patched = set()
            for module_name in INTERPRET_HOMES:
                try:
                    module = importlib.import_module(module_name)
                except ImportError:
                    continue
                for role, names in INTERPRET_BUILDERS.items():
                    for name in names:
                        if hasattr(module, name):
                            stack.enter_context(mock.patch.object(module, name,
                                                                  _with_interpret(getattr(module, name))))
                            patched.add(role)
            if patched != set(INTERPRET_BUILDERS):
                raise RuntimeError("no home binds the interpret builders; update driver.INTERPRET_BUILDERS")
        home = importlib.import_module(PROGRAM_SET_HOME[0])
        if hasattr(home, PROGRAM_SET_HOME[1]):  # 181c013e has no ProgramSet (recorded as absent)
            stack.enter_context(mock.patch.object(home, PROGRAM_SET_HOME[1],
                                                  _recording_program_sets(getattr(home, PROGRAM_SET_HOME[1]),
                                                                          program_sets)))
        stack.enter_context(lowering.location_free())  # production's lowerings are fingerprinted location-free
        # jax's persistent compilation cache would write through pathlib while the overlay refuses
        # every write outside /dev/shm; the harness's CPU compiles never need it.
        from jax._src import config as jax_config

        stack.enter_context(jax_config.enable_compilation_cache(False))
        stack.enter_context(compiler.patches())
        stack.enter_context(overlay.active())
        cls.__init__(runtime, args=pinned_args(), repo=REPO, root=root, mesh=mesh, physical=physical,
                     topology=topology, fleet_sha="<synthetic fleet>", vote=vote, save=lambda record: None,
                     context_capacity=capacity, concurrent_size=concurrent_size)
        if len(config_calls) != 1 or runtime.config is not constructed[0]:
            raise RuntimeError(f"the runtime's config is not the first {CONFIG_CLASS} __init__ constructs "
                               f"({len(config_calls)} recorded); update driver._config_injection")
        missing = [name for name in HOMES if name not in stubs.calls]
        if missing:
            raise RuntimeError("_load no longer calls the faked " + ", ".join(missing) + "; update driver.HOMES")
        hlo = getattr(runtime, "hlo", None)
        record = runtime.record
        protocol = dict(
            config_call=config_calls[0],
            config=config_record(runtime.config),
            phases=list(phases),
            admissions=[list(admission) for admission in stubs.admissions],  # later admissions stay out
            loader=dict(stubs.calls),
            record=dict(keys=sorted(record), state_ownership=record.get("state_ownership"),
                        capacity=record.get("capacity"), profile=record.get("profile"), schema=record.get("schema"),
                        complete=record.get("complete"), checkpoint_keys=sorted(record.get("checkpoint") or {}),
                        local_slots=record.get("physical_identity", {}).get("local_slots"),
                        program_row_keys=sorted({k for row in record.get("programs", {}).values() for k in row})),
            programs=[spec.name for spec in recorder.specs],
            hlo_directory=overlay.record(pathlib.Path(hlo) if hlo is not None else None),
            compile=dict(hlo_admissions=list(compiler.hlo_admissions), consensus_calls=len(compiler.consensus),
                         consensus_payload_bytes=sorted(set(compiler.consensus))),
        )
        protocol["probes"] = dict(graph_consensus=compiler.consensus_probe(runtime, cls))
    del runtime.phase  # generate uses the class method
    return Built(runtime, recorder, protocol, programset_identity(program_sets, recorder.specs))


def _recording_program_sets(builder: Any, sets: list[Any]) -> Any:
    def build(*args: Any, **kwargs: Any) -> Any:
        program_set = builder(*args, **kwargs)
        sets.append(program_set)
        return program_set

    return build


@contextmanager
def _config_injection(tier: str, fixture_geometry: Any, calls: list[Any], constructed: list[Any]) -> Iterator[None]:
    """Record (and, on the fixture tier, adjust) the first ``Ws32DecoderConfig`` constructed while
    the runtime is built -- the one config ``__init__`` makes -- by wrapping the class's own
    ``__init__``, so every import binding of the class and of ``model.geometry`` sees it (a
    from-import of either is a pure refactor and must not change what the harness builds).

    Fixture tier only: a ``geometry`` argument equal to the pinned production geometry (what
    ``model.geometry`` returns) becomes the fixture geometry, and ``sparse_segment_block=128`` is
    added when the call passes none (the fixture's DSA top-k is 128). The call is recorded as made,
    with the geometry after that substitution and without the added segment block, so the load
    protocol does not depend on how ``__init__`` reaches the class or the geometry function.
    Later constructions (none at 181c013e) run unchanged."""
    cls = getattr(importlib.import_module(DECODER_MODULE), CONFIG_CLASS)
    real_init = cls.__init__
    signature = inspect.signature(real_init)
    names = list(signature.parameters)[1:]
    substitutes: set[str] = set()
    if tier == "fixture":
        from .identities import production_geometry

        substitutes = {production_geometry().geometry_hash, fixture_geometry.geometry_hash}

    def init(self: Any, *args: Any, **kwargs: Any) -> None:
        if constructed:
            real_init(self, *args, **kwargs)
            return
        bound = signature.bind(self, *args, **kwargs).arguments
        if tier == "fixture" and getattr(bound.get("geometry"), "geometry_hash", None) in substitutes:
            bound["geometry"] = fixture_geometry
        call_args = [bound[name] for name in names[:len(args)]]
        call_kwargs = {name: bound[name] for name in kwargs}
        calls.append(dict(args=[describe(a) for a in call_args],
                          kwargs={k: describe(v) for k, v in sorted(call_kwargs.items())}))
        if tier == "fixture" and "sparse_segment_block" not in bound:
            call_kwargs["sparse_segment_block"] = FIXTURE_SEGMENT_BLOCK
        constructed.append(self)
        real_init(self, *call_args, **call_kwargs)

    with mock.patch.object(cls, "__init__", init):
        yield


def _with_interpret(builder: Any) -> Any:
    def build(*args: Any, **kwargs: Any) -> Any:
        return builder(*args, **kwargs, **INTERPRET)

    return build


_INVENTORY: list[Any] = []


def _pinned_inventory(factory: Any) -> Any:
    """The synthetic GLM-5.3 inventory with the pinned digest string (built once per process)."""
    if not _INVENTORY:
        from .identities import INVENTORY_PIN

        _INVENTORY.append(factory(pinned=INVENTORY_PIN))
    return _INVENTORY[0]


def _relaxed_batch(values: Any, *, concurrent: bool = False) -> dict[str, Any]:
    """``request.batch`` for fixture requests: the lane count and one shared capacity only."""
    capacities = {value["context_capacity"] for value in values}
    if not 1 <= len(values) <= 4 or len(capacities) != 1 or concurrent is not True:
        raise ValueError("fixture batch: one to four concurrent requests of one capacity")
    return dict(context_capacity=capacities.pop())


@contextmanager
def serving_fakes(*, relaxed_validation: bool = False) -> Iterator[None]:
    """Fleet fakes for ``generate``/``generate_concurrent``: ``process_allgather`` stacks the local
    value eight times; ``relaxed_validation`` replaces ``runtime.validate`` and the batch binding
    ``batched_runtime.batch`` (the fixture's 1,536-slot, 256-token vocabulary requests are not a
    production profile)."""
    from jax.experimental import multihost_utils

    with ExitStack() as stack:
        stack.enter_context(mock.patch.object(multihost_utils, "process_allgather",
                                              lambda value: np.stack([np.asarray(value)] * 8)))
        if relaxed_validation:
            stack.enter_context(mock.patch.object(importlib.import_module(RUNTIME_MODULE), "validate",
                                                  lambda request: None))
            stack.enter_context(mock.patch.object(importlib.import_module(BATCHED_MODULE), "batch", _relaxed_batch))
        yield


# ----------------------------------------------------------------------------- defaults (G1-protocol)
DEFAULT_CLASSES = ("Ws32DecoderConfig", "Ws32PerfOptions", "RoutedProjectionConfig", "SparseMlaConfig")
DEFAULT_FUNCTIONS = ("build_prefill_program", "build_packed_decoder_program",
                     "build_ws32_challenger_decoder_program", "build_batched_decoder_program",
                     "build_cache_initializer", "build_wk_programs", "bf16_resident_weights")
ABSENT = "<absent>"


def _definition(name: str) -> Any:
    """The one object recorded as ``name`` (a 181c013e top-level name) defined in a loaded
    repository module (not re-exported), found under any current name ``closure_map.toml``
    ``[functions]`` maps to it. Returns ``ABSENT`` when there is none (e.g. S2d deletes a knob
    class) and ``<ambiguous: ...>`` for several: both are recorded, never raised."""
    from . import closure_map

    wanted = closure_map.load().current_names(name)
    root = str(REPO) + "/"
    found = {}
    for module_name, module in list(sys.modules.items()):
        path = getattr(module, "__file__", None)
        if not path or not str(path).startswith(root) or module_name.startswith("tools.equivalence"):
            continue
        for attribute in wanted:
            obj = getattr(module, attribute, None)
            if obj is not None and getattr(obj, "__module__", None) == module_name:
                found[f"{module_name}:{attribute}"] = obj
    if len(found) > 1:
        return f"<ambiguous: {len(found)} definitions>"
    return next(iter(found.values())) if found else ABSENT


def production_defaults() -> dict[str, Any]:
    """Defaults of the production option dataclasses and program-builder keywords (by field and
    parameter name, values structurally encoded), under their 181c013e names. Call after the
    runtime was built (modules loaded). Catches a changed default the fixture tier overrides or
    never exercises; a renamed class or builder is found through ``closure_map.toml``, a removed
    or ambiguous one is recorded as such (the characterization record is re-baselined with a
    reviewed reason, e.g. S2d deleting ``Ws32PerfOptions``)."""
    out: dict[str, Any] = {}
    for name in DEFAULT_CLASSES:
        cls = _definition(name)
        if isinstance(cls, str) or not dataclasses.is_dataclass(cls):
            out[name] = cls if isinstance(cls, str) else "<not a dataclass>"
            continue
        fields = {}
        for f in dataclasses.fields(cls):
            if f.default is not dataclasses.MISSING:
                fields[f.name] = encode_default(f.default)
            elif f.default_factory is not dataclasses.MISSING:
                fields[f.name] = encode_default(f.default_factory())
            else:
                fields[f.name] = "<required>"
        out[name] = fields
    for name in DEFAULT_FUNCTIONS:
        function = _definition(name)
        if isinstance(function, str):
            out[name] = function
            continue
        parameters = inspect.signature(function).parameters
        out[name] = {p.name: encode_default(p.default) for p in parameters.values()
                     if p.default is not inspect.Parameter.empty}
    return out

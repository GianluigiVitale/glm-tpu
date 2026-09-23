"""Drive the real ``OrdinaryRuntime`` (``__init__``, ``_load``, ``generate``) on the CPU host.

The gates build their device programs and CPU goldens by running production's own runtime code,
so an edit to ``__init__``, ``_load``, ``generate`` or ``compile_batch`` (donation, options,
config flags, sample shapes, the tail rule) reaches the fingerprints and goldens. Only what needs
a TPU fleet, private assets or the TPU compiler is replaced:

* checkpoint I/O: ``authenticated_inventory`` returns the synthetic GLM-5.3 inventory with the
  pinned digest string; ``verify_ws32_runtime_checkpoint`` returns the file plans (synthetic
  production plans, or placeholder plans for the fixture); ``load_ws32_runtime_checkpoint``
  returns the fixture arrays (concrete) or ``ShapeDtypeStruct`` values with the plans' shardings
  (abstract). Their keyword arguments are recorded. ``model.require_site`` and
  ``model.require_inventory`` run for real on pinned-identity inputs;
* ``compile``: the recorder keeps the ``(name, jitted function, arguments)`` production hands it
  (in abstract mode the result is a stand-in whose outputs carry the shardings the CPU-compiled
  executable reports); ``admit_memory`` records its request and admits; ``admit`` and ``stats``
  are inert (no device memory statistics on CPU);
* the fleet: votes are identity (``phase`` runs for real and records the phase sequence), and
  ``process_allgather`` stacks the local value eight times (``serving_fakes``);
* the host: ``__init__``'s ``mkdir`` of the HLO-originals directory under
  ``/dev/shm/glm-optimized-hlo`` is recorded and not performed, and its free-space probe sees
  1 TiB (nothing is ever written to ``/dev/shm``);
* abstract mode only: ``ShapeDtypeStruct.addressable_shards`` answers the per-shard byte probe
  ``_load`` uses for the BF16 preparation admission.

Fixture tier (frozen fixture v1, 1,536 slots): ``model.geometry`` returns the fixture geometry,
and the one ``Ws32DecoderConfig`` construction in ``__init__`` gets exactly one addition,
``sparse_segment_block=128`` when ``__init__`` passes none (the fixture's DSA top-k is 128, so the
production default cannot validate); the recorded protocol keeps the call ``__init__`` made. The donated
variant evaluates production's own rule ``capacity > CAPACITY`` with ``runtime.CAPACITY`` lowered
to 1,024, and the concurrent variants set ``request.CONCURRENT_CAPACITY`` to the fixture capacity.
Production tier: no override; production geometry, capacities and donation rule as they are.
"""

from __future__ import annotations

from contextlib import ExitStack, contextmanager
import dataclasses
from dataclasses import dataclass, field
import importlib
import inspect
import pathlib
import shutil
import sys
import tempfile
from types import SimpleNamespace
from typing import Any, Callable, Iterator
from unittest import mock

import numpy as np

from .common import REPO

RUNTIME_MODULE = "glm_tpu.optimized.runtime"
RUNTIME_CLASS = "OrdinaryRuntime"
REQUEST_MODULE = "glm_tpu.optimized.request"
DECODER_MODULE = "glm_tpu.greenfield.runtime.ws32_decoder"
HLO_PREFIX = "/dev/shm/glm-optimized-hlo"  # OrdinaryRuntime.__init__ (host RAM); never written by the harness
FIXTURE_SEGMENT_BLOCK = 128                # the only fixture override of the config __init__ builds
FIXTURE_DONATION_CAPACITY = 1024           # runtime.CAPACITY for the donated fixture variant (< 1,536)
INTERPRET = dict(sparse_attention_interpret=True, linear_interpret=True)
# Every module that defines a faked loader function is patched: the 181c013e home first, then the
# homes DESIGN.md S2a moves them to. A move elsewhere leaves the real function in place, which
# fails on the placeholder arguments (fail-closed), and ``stub never called`` names it.
HOMES = {
    "authenticated_inventory": ("scripts.greenfield.ws32_compile_originals",
                                "glm_tpu.greenfield.partitioning.source_inventory"),
    "verify_ws32_runtime_checkpoint": ("glm_tpu.greenfield.checkpoint.ws32_runtime_checkpoint",),
    "load_ws32_runtime_checkpoint": ("glm_tpu.greenfield.checkpoint.ws32_runtime_checkpoint",),
}
# Builders ``_load``/``compile_batch`` call; G3 runs the prefill and decode ones in interpret mode.
INTERPRET_BUILDERS = ((RUNTIME_MODULE, "build_ws32_prefill_challenger_program"),
                      (RUNTIME_MODULE, "build_packed_decoder_program"))


@dataclass(frozen=True)
class ProgramSpec:
    name: str                 # production program name (HLO original file name)
    fn: Callable[..., Any]    # the jax.jit object production lowers (donation already applied)
    args: tuple[Any, ...]     # arguments production passes (arrays or ShapeDtypeStructs)


def leaf_signature(leaf: Any) -> tuple[Any, ...]:
    """(shape, dtype, sharding) with the sharding canonicalized (``normalize.sharding_key``), so
    equivalent ``PartitionSpec`` spellings of the same placement compare equal."""
    from .normalize import sharding_key

    sharding = getattr(leaf, "sharding", None)
    return (tuple(int(d) for d in leaf.shape), str(leaf.dtype), None if sharding is None else sharding_key(sharding))


def abstract_like(tree: Any) -> Any:
    import jax

    return jax.tree.map(lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype, sharding=x.sharding), tree)


class ProgramRecorder:
    """Records ``(name, fn, args)`` in call order instead of compiling for TPU. Concrete: returns
    ``fn`` (executed on the CPU mesh, as ``_load`` executes producers on TPU). Abstract: returns a
    stand-in producing ``ShapeDtypeStruct`` outputs with the CPU-compiled executable's shardings."""

    def __init__(self, *, concrete: bool, outputs: dict[Any, Any] | None = None):
        self.concrete = concrete
        self.context: Any = None
        self.specs: list[ProgramSpec] = []
        self._outputs = {} if outputs is None else outputs  # may be shared across runtimes of a tier

    def compile(self, name: str, fn: Any, values: tuple[Any, ...], *, model: bool = True) -> Any:
        self.specs.append(ProgramSpec(name, fn, tuple(values)))
        return fn if self.concrete else self.abstract(fn, name)

    def abstract(self, fn: Any, name: str) -> Callable[..., Any]:
        import jax

        context = self.context

        def call(*args: Any) -> Any:
            leaves, treedef = jax.tree.flatten(args)
            key = (name, context, treedef, tuple(leaf_signature(x) for x in leaves))
            if key not in self._outputs:
                compiled = fn.trace(*args).lower().compile()
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
    """The faked checkpoint I/O and admission; records how ``_load`` called them."""

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

    def admit_memory(self, name: str, memory: dict[str, Any]) -> dict[str, Any]:
        self.admissions.append([name, {k: describe(v) for k, v in sorted(memory.items())}])
        return dict(passed=True, harness="admitted without device memory statistics")


def fixture_plans(arrays: dict[str, Any]) -> list[Any]:
    """Placeholder file plan for the fixture (only feeds the checkpoint-load admission request)."""
    sizes = [int(np.prod(a.shape, dtype=np.int64)) * np.dtype(a.dtype).itemsize for a in arrays.values()]
    return [SimpleNamespace(payload_bytes=sum(sizes), tensors=[SimpleNamespace(byte_count=s) for s in sizes])]


def pinned_args() -> Any:
    """Worker arguments ``_load`` reads, as site_args would bind them (placeholders for paths and
    content pins the faked loader ignores; the model identity is the pinned one)."""
    from glm_tpu.optimized import model

    from .identities import INVENTORY_PIN

    return SimpleNamespace(model_id=model.MODEL_ID, model_revision=model.REVISION,
                           source_inventory="<source_inventory>", source_inventory_sha256=INVENTORY_PIN,
                           checkpoint_root="<checkpoint_root>", checkpoint_manifest_sha256="<manifest_sha256>",
                           checkpoint_success_sha256="<success_sha256>", mesh_sha256="<mesh_sha256>",
                           topology_sha256="<topology_sha256>")


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


@contextmanager
def _divert_hlo_mkdir(requests: list[Any]) -> Iterator[None]:
    """``__init__`` creates ``/dev/shm/glm-optimized-hlo/<run>/<rank dir>`` for the compiler
    originals. While the runtime is built, a ``Path.mkdir`` under that prefix is recorded and
    not performed (``compile`` is faked, so nothing is ever written there); every other
    ``mkdir`` is the real one."""
    real = pathlib.Path.mkdir

    def mkdir(self: pathlib.Path, mode: int = 0o777, parents: bool = False, exist_ok: bool = False) -> None:
        text = str(self)
        if text == HLO_PREFIX or text.startswith(HLO_PREFIX + "/"):
            requests.append(dict(path="<hlo root>" + text[len(HLO_PREFIX):], mode=oct(mode), parents=parents,
                                 exist_ok=exist_ok))
            return None
        return real(self, mode, parents, exist_ok)

    pathlib.Path.mkdir = mkdir  # type: ignore[method-assign]
    try:
        yield
    finally:
        pathlib.Path.mkdir = real  # type: ignore[method-assign]


@dataclass
class Built:
    runtime: Any
    recorder: ProgramRecorder
    protocol: dict[str, Any] = field(default_factory=dict)


def runtime_class() -> Any:
    return getattr(importlib.import_module(RUNTIME_MODULE), RUNTIME_CLASS)


def build_runtime(mesh: Any, *, tier: str, capacity: int, concurrent_size: int, arrays: dict[str, Any], plans: Any,
                  concrete: bool, donated_fixture: bool = False, fixture_geometry: Any = None,
                  interpret: bool = False, outputs: dict[Any, Any] | None = None) -> Built:
    """Construct the real runtime with the real ``__init__`` (which calls the real ``_load``)."""
    import jax

    from .identities import synthetic_inventory

    runtime_module = importlib.import_module(RUNTIME_MODULE)
    request_module = importlib.import_module(REQUEST_MODULE)
    from glm_tpu.optimized import model

    cls = runtime_class()
    recorder = ProgramRecorder(concrete=concrete, outputs=outputs)
    stubs = LoadStubs(arrays, plans, _pinned_inventory(synthetic_inventory))
    local = [int(d.id) for d in jax.local_devices()]
    physical = SimpleNamespace(mesh_hash="<synthetic mesh>",
                               flattened_device_ids=tuple(local[:4]) + tuple(100000 + i for i in range(28)))
    topology = SimpleNamespace(topology_hash="<synthetic topology>")
    phases: list[str] = []
    config_calls: list[Any] = []

    runtime = object.__new__(cls)
    # Instance attributes shadow the methods for the whole life of this runtime (generate included).
    runtime.compile = recorder.compile
    runtime.admit_memory = stubs.admit_memory
    runtime.admit = lambda name: None
    runtime.stats = lambda: []

    def vote(valid: Any) -> bool:
        return bool(valid)

    real_phase = runtime.phase

    def phase(name: str, action: Any) -> Any:
        phases.append(name)
        return real_phase(name, action)

    runtime.phase = phase
    hlo_requests: list[Any] = []
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
        decoder_module = importlib.import_module(DECODER_MODULE)
        real_config = decoder_module.Ws32DecoderConfig

        def config_once(*args: Any, **kwargs: Any) -> Any:
            """The one config construction in ``__init__``: recorded as called; the fixture adds
            its segment block when ``__init__`` passes none (the only fixture override)."""
            decoder_module.Ws32DecoderConfig = real_config
            config_calls.append(dict(args=[describe(a) for a in args],
                                     kwargs={k: describe(v) for k, v in sorted(kwargs.items())}))
            if tier == "fixture":
                kwargs.setdefault("sparse_segment_block", FIXTURE_SEGMENT_BLOCK)
            return real_config(*args, **kwargs)

        stack.enter_context(mock.patch.object(decoder_module, "Ws32DecoderConfig", config_once))
        stack.enter_context(mock.patch.object(shutil, "disk_usage", lambda path: shutil._ntuple_diskusage(
            1 << 40, 0, 1 << 40)))
        stack.enter_context(_divert_hlo_mkdir(hlo_requests))
        if not concrete:
            stack.enter_context(mock.patch.object(jax.ShapeDtypeStruct, "addressable_shards", property(_shard_probe),
                                                  create=True))
        if tier == "fixture":
            stack.enter_context(mock.patch.object(model, "geometry", lambda repo=None: fixture_geometry))
            if donated_fixture:
                stack.enter_context(mock.patch.object(runtime_module, "CAPACITY", FIXTURE_DONATION_CAPACITY))
            if concurrent_size:
                stack.enter_context(mock.patch.object(request_module, "CONCURRENT_CAPACITY", capacity))
        if interpret:
            for module_name, name in INTERPRET_BUILDERS:
                module = importlib.import_module(module_name)
                original = getattr(module, name)
                stack.enter_context(mock.patch.object(module, name, _with_interpret(original)))
        cls.__init__(runtime, args=pinned_args(), repo=REPO, root=root, mesh=mesh, physical=physical,
                     topology=topology, fleet_sha="<synthetic fleet>", vote=vote, save=lambda record: None,
                     context_capacity=capacity, concurrent_size=concurrent_size)
    if len(config_calls) != 1:
        raise RuntimeError(f"__init__ built {len(config_calls)} decoder configs through {DECODER_MODULE}; "
                           "update driver.py")
    missing = [name for name in HOMES if name not in stubs.calls]
    if missing:
        raise RuntimeError("_load no longer calls the faked " + ", ".join(missing) + "; update driver.HOMES")
    del runtime.phase  # generate uses the class method
    record = runtime.record
    protocol = dict(
        config_call=config_calls[0],
        config=config_record(runtime.config),
        phases=phases,
        admissions=stubs.admissions,
        loader=stubs.calls,
        record=dict(keys=sorted(record), state_ownership=record.get("state_ownership"),
                    capacity=record.get("capacity"), profile=record.get("profile"), schema=record.get("schema"),
                    complete=record.get("complete"), checkpoint_keys=sorted(record.get("checkpoint") or {}),
                    local_slots=record.get("physical_identity", {}).get("local_slots")),
        programs=[spec.name for spec in recorder.specs],
        hlo_directory=hlo_requests,
    )
    return Built(runtime, recorder, protocol)


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


@contextmanager
def serving_fakes(*, relaxed_validation: bool = False) -> Iterator[None]:
    """Fleet fakes for ``generate``: ``process_allgather`` stacks the local value eight times;
    ``relaxed_validation`` replaces ``runtime.validate`` (the fixture's 1,536-slot, 256-token
    vocabulary requests are not a production profile)."""
    from jax.experimental import multihost_utils

    with ExitStack() as stack:
        stack.enter_context(mock.patch.object(multihost_utils, "process_allgather",
                                              lambda value: np.stack([np.asarray(value)] * 8)))
        if relaxed_validation:
            stack.enter_context(mock.patch.object(importlib.import_module(RUNTIME_MODULE), "validate",
                                                  lambda request: None))
        yield


# ----------------------------------------------------------------------------- defaults (G1)
DEFAULT_CLASSES = ("Ws32DecoderConfig", "Ws32PerfOptions", "RoutedProjectionConfig", "SparseMlaConfig")
DEFAULT_FUNCTIONS = ("build_ws32_prefill_challenger_program", "build_packed_decoder_program",
                     "build_ws32_challenger_decoder_program", "build_batched_decoder_program",
                     "build_cache_initializer", "build_wk_programs", "bf16_resident_weights")


def _definition(name: str) -> Any:
    """The one object named ``name`` defined in a loaded repository module (not re-exported)."""
    wanted = (name,)
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
    if len(found) != 1:
        raise RuntimeError(f"expected exactly one definition of {name}, found {sorted(found)}")
    return next(iter(found.values()))


def production_defaults() -> dict[str, Any]:
    """Defaults of the production option dataclasses and program-builder keywords (by field and
    parameter name, values structurally encoded). Call after the runtime was built (modules
    loaded). Catches a changed default the fixture tier overrides or never exercises."""
    out: dict[str, Any] = {}
    for name in DEFAULT_CLASSES:
        cls = _definition(name)
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
        parameters = inspect.signature(_definition(name)).parameters
        out[name] = {p.name: encode_default(p.default) for p in parameters.values()
                     if p.default is not inspect.Parameter.empty}
    return out

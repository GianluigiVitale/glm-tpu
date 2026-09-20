# Architecture and code map

WS32_2D uses 32 TPU v4 chips on 8 hosts. Physical feature-four/expert-eight
subgroups preserve sharded hidden state; this is not one 32-chip collective group
for every layer. PP8/PP16 code and measurements are research history.

`glm_tpu/greenfield/runtime/ws32_batched_prefill.py` constructs layer-major
prefill. B128/B114 are prompt-row shapes, not request batch sizes. Layer windows
preserve causality, per-row routing, IndexShare and separate unrepaired/repaired
index state. Decode has one live row.

`runtime/ws32_decoder.py` and `runtime/ws32_sampled_request.py` construct decode
and sampled-head programs. `runtime/ws32_request_session.py` owns delivery,
stop/cap policy, RNG frontier and live pause/resume. Its caller supplies admitted
compiled programs and resident weights; this class is not a standalone server.

`kernels/` holds JAX operations, `kernels/pallas/` TPU kernels. `checkpoint/`
owns payload/scale layout, integrity and direct placement. `topology/` and
`sharding/` make physical ownership/collectives explicit. `validation/` and
`benchmarking/` protect evidence; they do not replace execution measurements.

## Deployment boundary

The ordinary greedy candidate enters through
`scripts/release/launch_ws32_optimized_request.py`, then
`ws32_optimized_worker.py` and `glm_tpu/optimized/runtime.py`. The controller
authenticates a clean published source archive and the existing eight-host site,
holds both workload leases and uses both sync locks during staging. The worker
verifies checkpoint bytes, constructs resident BF16 non-routed weights, compiles
fresh B128/B114 prefill and packed decode graphs, checks graph agreement and
live memory, warms disposable state and generates from a fresh cache.

`glm_tpu/optimized/request.py` fixes greedy sampling and an 8,192-slot combined
prompt/output budget. `request_loop.py` reuses the frozen request policy and
delivery contract with compact decode metadata and fleet votes. Rank0 writes
and flushes token events locally, then decodes final text. Deadline, peer or
delivery failure poisons the request; it cannot automatically retry. The
controller authenticates cleanup on all eight hosts before releasing leases.
[STATUS](STATUS.md) determines whether this integration has passed admission;
the architecture description alone is not validation.

The historical benchmark campaign launcher is removed from the release tree;
its exact original is recoverable through the [curation ledger](../curation/README.md).
Separate `ws32_native_benchmark_*` modules still supply shared cold preparation,
observation, transport and validation. Their individual dependencies remain under
curation; their names do not make benchmark campaigns supported release commands.

Site configuration remains fixed to the supported installation; not every
research script is a supported entry point. Import reachability alone misses
subprocesses, dynamic imports, data files and source-hash registrations.

The legacy sampled user interface is
`scripts/release/launch_ws32_user_request.py`. Shared host admission, SSH and
original-process/publication checks live in `scripts/release/ws32_host_ops.py`;
the user controller no longer imports the campaign launcher for these helpers.
The nine extracted helpers preserve the original function ASTs, constants and
negative/recovery checks. This host-only separation changes no numerical code
or admitted HLO identity and is not a new hardware-validation claim.
The worker uses the retained native
loader and host request runtime, followed by user-only replay/DB/archive modules.
The user controller requires an explicitly reviewed published
owner branch (main by default), exact code and retained site/asset pins. This
replaces the old research-only branch selection; the physical site remains
explicitly limited to the existing pod. Real release deployment and an ordinary
user response passed DB621; this does not establish portable or persistent serving.

The benchmark registry dynamically loads only pinned `bench/benchmarks.py` and
`bench/extract.py`; provenance uses `bench/provenance.py`. Those files must stay
even though static import reachability misses the registry's dynamic loader.
Their presence does not authorize use of `bench/engine.py` or legacy execution.
Script roles and historical-tool boundaries: [scripts index](../../scripts/README.md).

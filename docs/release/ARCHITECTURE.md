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

`scripts/greenfield/launch_ws32_native_benchmark.py` is a protected site-specific
campaign launcher, not generic serving. It pins research branch, hosts, Python
path, private request capsule and regional assets, acquires both leases and
authenticates ownership. Separate `ws32_native_benchmark_*` modules handle
cold preparation, requests, observation, transport, replay, DB and archive.

The release must separate site configuration without weakening safeguards or
presenting every research script as supported. Import reachability alone misses
subprocesses, dynamic imports, data files and source-hash registrations.

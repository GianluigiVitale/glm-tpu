# GLM TPU

Native-JAX inference for GLM-5.2-FP8 on an existing eight-host, 32-chip TPU v4
slice. This private repository contains the engine, checkpoint tools, tests and
reproducible validation machinery.

The supported boundary is a **site-specific, single-request inference engine**,
not a public inference service. Protected long-context runs and the ordinary
user-response release smoke test (DB621) have passed. Full task-quality and
model-card parity are not established. See [release status](docs/release/STATUS.md).

## Implemented capabilities

- Complete 78-layer WS32_2D decoder with batched B128/B114 prefill.
- Explicit feature-four and expert-eight communication groups across 32 chips;
  no repeated full-pod hidden-state reconstruction inside transformer layers.
- Plan-aware FP8 checkpoint verification and direct local-shard loading.
- Causal KV caches, DSA selection/tie checks, IndexShare and routed/shared MoE.
- Sampled sequential requests with first-token delivery to a local output sink
  and pause/resume of the same live session.
- HLO, per-chip memory, timing, provenance and failure-recovery protections.

There is no supported HTTP service, concurrent-request batching, durable KV
recovery or speculative decoding in this release. A local flushed
token is not proof of network-delivered latency. The native model does not
execute through the legacy `tpu-inference` engine.

## Measured results

Retained protected results, not benchmarks of the release-cleanup diff.
Prefill excludes cold loading/compilation; E0 is capacity/throughput evidence,
not a model-quality evaluation.

| Workload | Prefill tokens/s | Decode wall tokens/s | Evidence |
|---|---:|---:|---|
| 2,034-token prompt | 62.761 | 7.660 | DB610 |
| 127,363-token passkey, depth 0.95 | 45.459 | 6.935 | DB619 |
| 262,144-token E0 | 32.157 | 6.148 | DB620 |

All four protected 128K passkey depths completed (DB616–619). Full 256K E0
completed (DB620), with 29.930 GB maximum measured HBM per chip and 3.084 GB
minimum headroom. Full GPQA/AIME accuracy and model-card parity are **not established**.
Sources and qualifications: [validation status](docs/release/STATUS.md).

## Navigation

- [Architecture and code map](docs/release/ARCHITECTURE.md)
- [Installation and environment checks](docs/release/INSTALLATION.md)
- [Checkpoint loading, retained weights and recovery](docs/release/CHECKPOINTS.md)
- [User prompt preparation and inference integration](docs/release/INFERENCE.md)
- [Release checklist and limitations](docs/release/STATUS.md)
- [Development and branch policy](CONTRIBUTING.md)
- [Operational constraints](docs/release/OPERATIONS.md)
- [Third-party notices and unresolved provenance](THIRD_PARTY_NOTICES.md)
- [Security, private data and release checks](SECURITY.md)
- [Release handoff](HANDOFF.md)

Packaging, fresh pinned-dependency installation and 431 CPU release tests pass;
the protected user-response path is validated by DB621. Do not
use historical campaign scripts as a generic installer or launch a second TPU
workflow alongside a running one.

## Repository policy

`main` is reserved for the supported release. Its preparation branch is
`release/production-20260914`; research remains on `rewrite/topology-first-decode`
and other preserved branches in this same repository. The repository stays
private. Cleanup does not remove history or change any upstream repository.
Weights, credentials, private benchmark questions and runtime databases stay out of Git.

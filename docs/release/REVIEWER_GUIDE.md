# Technical reviewer guide

This is a systems research implementation, not a novel foundation model or a
generally deployable serving product. Its design choices and measured claims
should be inspectable and connected to evidence.

## A short reading route

1. Read the [README](../../README.md) for the problem, scope and results, then the
   [architecture](ARCHITECTURE.md) for the execution boundary.
2. Inspect [batched prefill](../../glm_tpu/greenfield/runtime/ws32_batched_prefill.py)
   and [request state](../../glm_tpu/greenfield/runtime/ws32_request_session.py).
   These distinguish prompt-row batching from one live request.
3. Follow a result to its receipt. [DB620](../artifacts/prefill-delivery-db620-sealed-20260912.json)
   records prompt length, timing, HBM, source/recovery pins, validation scope and
   archive identity. `NO_CORRECTNESS_ORACLE` is a limitation, not a quality pass.
4. Inspect [request-state tests](../../tests/greenfield/runtime/test_ws32_request_session.py),
   [launch refusals](../../tests/release/test_user_launch.py) and
   [archive recovery](../../tests/release/test_user_archive.py). Failure boundaries
   matter alongside happy paths; fake math in CPU tests is not TPU validation.
5. Read [status](STATUS.md) and the [audit](READINESS_AUDIT.md) for quality,
   portability, self-review and provenance limitations.

## Separate the claims

| Claim | Evidence | Do not infer |
|---|---|---|
| Short numerical agreement | [DB610](../artifacts/prefill-canonical-short-db610-sealed-20260909.json), including `numerical_limitations` | Agreement at 8K, all state values, or every later build |
| Long-context retrieval | DB616–619; [DB619 example](../artifacts/prefill-delivery-db619-sealed-20260912.json) | General reasoning quality from four passkey prompts |
| Full 262,144-token execution | [DB620](../artifacts/prefill-delivery-db620-sealed-20260912.json) | Model-card parity or a sampled 256K endpoint |
| Ordinary user response | [DB621](user-response-db621-sealed-20260914.json) and [metric scope](STATUS.md#user-response-admission) | Persistent service readiness, network TTFT or task accuracy |
| Installation and host-side checks | [Fresh install](fresh-install-20260914.json), [CPU checks](final-cpu-check-20260914.json) | Hardware performance or universal deployment portability |

Database IDs are local experiment identifiers, not independent replications or
third-party certifications. A sealed result means the project's specified checks
and archive linkage completed. No universal exactness, priority, state-of-the-art
throughput or independent release-review claim is made.

## Reproducibility boundary

Compact receipts and source can be inspected in Git. Raw traces, checkpoint
payloads and some detailed results need access to the private regional archive.
An external reader cannot independently reproduce the hardware runs from this
checkout alone. The wheel excludes weights and the historical script/artifact tree.

CPU checks use the documented environment. Hardware reproduction needs the
existing site, retained assets and original admission checks; do not create
infrastructure or restart old benchmarks to review source.

## Questions worth examining

- How are physical groups and ownership represented, and what prevents repeated
  full-pod hidden-state reconstruction?
- Where do prefill, cache promotion and decode meet, and which invariants protect them?
- Which discrepancies are numerical behavior, which are state bugs, and what remains unproved?
- What do device traces, steady wall timing and cold startup each measure?
- Can a failed upload be recovered without recomputing or relabeling the response?

The [observability playbook](../greenfield/GATE_D_OBSERVABILITY_PLAYBOOK.md) records
historical diagnostic methods. It complements current source; it is not permission
to resume old campaigns. The scoped release audit is self-review, not an
independent assessment of research novelty or a universal security audit.

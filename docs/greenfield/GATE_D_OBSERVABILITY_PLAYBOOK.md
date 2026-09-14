# Observability lessons from the GLM TPU port

This guide explains the debugging methods worth keeping, not how to restart the
old Gate D campaigns. For supported execution and recovery use
[operations](../release/OPERATIONS.md), [inference](../release/INFERENCE.md) and
[release status](../release/STATUS.md). Repository curation authorizes no TPU runs.

## The useful result: distinguish the failure classes

A different output token does not identify its cause. Checkpoint corruption,
cache ownership, numerical rounding, a changed observer and an invalid comparison
can all produce similar symptoms. The research progressively separated these
questions by comparing captured state at producer/consumer boundaries.

Three concrete lessons explain the method:

- The accepted/DB518 comparison had a downstream BF16 mismatch but lacked the
  unperturbed FP32 input at the relevant RMS boundary. Computing that value on a
  CPU did not establish what the original TPU executable had consumed.
- Adding a callback made an internal value visible but changed the executable
  class and its outputs. Those observations could not explain the uninstrumented
  run without a separate noninterference argument.
- Comparing query/scorer arithmetic while borrowing another run's cache mixed
  histories. A locally plausible result was not a coherent decoder comparison.

These are historical findings, not claims that those problems remain open.
Original classifications, precise arrays, source pins and failed hypotheses are
recoverable from Git; see the recovery section below. Current release limitations
and measurements belong in STATUS, not a second changing experiment journal.

## Start with one question, then the smallest decisive check

1. **Establish identity.** Record code/runtime, plan, checkpoint, input, cache
   predecessor and executable identities. A filename or a `passed` flag alone
   is not authority.
2. **Locate the first missing or different value.** Compare in causal order:
   loaded state, residual operands, normalization, cache/query/key/head inputs,
   selected DSA entries, then output tokens. Stop at an evidence gap.
3. **State a falsifiable hypothesis.** Name the boundary, expected change and
   preserved invariants. Reuse a captured row or existing HLO when it can reject
   the hypothesis without compilation or model execution.
4. **Check observation effects.** Extra callbacks, roots and consumers can change
   fusion, materialization and scheduling. Match relevant executable identities
   and outputs before treating the instrumented run as evidence of the original.
5. **Separate evidence levels.** CPU reference arithmetic, logical StableHLO,
   optimized TPU HLO, actual device values and end-to-end quality answer different
   questions. Do not promote one into another.
6. **Preserve a useful failure.** Keep its exact inputs, classification and
   recovery location. A renamed hypothesis with unchanged association,
   consumer boundary, reduction, representation and transport is not a new test.

Any future hardware experiment needs explicit authority under the current goal.
A passing offline check is not that authority. Historical one-row/local-group
rules belonged to particular plans; inspect the actual supported plan contract
instead of imposing an old PP16 rule on every WS32 graph.

## What an array comparison must identify

For each watchpoint record semantic role, layer/token position, shape, storage
and semantic dtype, layout, physical owner, raw byte length and digest. Bind its
source file, array key/slice, code/plan/executable and cache history as well.

Snapshot/hash before parsing. Reject malformed shapes, duplicate archive members,
unexpected dtypes, missing owners and mixed histories. Retain descriptors or
revalidate identity when paths can change: a same-name replacement after preflight
must not silently become the input to replay.

Exact-bit comparisons can localize rounding boundaries. They do not by themselves
establish task quality, and a relaxed task-quality criterion does not make missing
or contradictory provenance acceptable. State each comparison's actual oracle
and tolerance; never relabel historical results to satisfy a new criterion.

For DSA, when context length does not exceed top-k, selected-set equality cannot
test truncation. A cutoff-active comparison must consider membership, order/ties,
positions and the cache/query/key state that produced the scores.

## Current tooling: choose the relevant evidence

| Question | Where to inspect | Limit |
|---|---|---|
| Which physical devices and groups are involved? | [Topology discovery](../../glm_tpu/greenfield/topology/discover.py), [HLO contract](../../glm_tpu/greenfield/sharding/hlo_contract.py) | A plan or matching operation name is not proof of live producer-to-output behavior. |
| What happened during the request? | [Native request instrumentation](../../scripts/greenfield/ws32_native_benchmark_observability.py) | Instrumented samples are not profiler-free latency. |
| How is the response validated? | [User result validation](../../scripts/release/ws32_user_result.py) | One ordinary-response smoke test is not broad quality or public-card parity. |
| How is evidence collected and preserved? | [User evidence handling](../../scripts/release/ws32_user_evidence.py), [WS32 evidence utilities](../../glm_tpu/greenfield/validation/ws32_evidence.py) | Collected bytes still require identity, completeness and claim checks. |
| What consumed device time versus wall time? | [XPlane reader](../../scripts/analysis/parse_xplane.py), [steady-decode reader](../../scripts/analysis/extract_steady_decode.py) | Device attribution, cold start, local first-token delivery and request wall are distinct metrics. |
| How do I recover an interrupted publication? | [Operations runbook](../release/OPERATIONS.md) | Follow the exact supported recovery path; do not relaunch the model or use an old campaign wrapper. |

These links locate implementation; they do not certify that every historical
branch inside a module executes for a user request. The supported dependency
boundary and every-file review are tracked in [curation](../curation/README.md).

## Avoid the expensive mistakes

- Test parser, ownership, publication and failure paths before an expensive run.
  Inspect actual sentinel formats and interpreter/runtime requirements first.
- Do not infer physical rounding/materialization from source types or HLO labels.
  Follow live operands, callees, roots, layouts and physical collective groups.
- Do not assume host0 owns all useful dumps. Check the declared host/device set
  and preserve divergent observations before selecting a canonical copy.
- Force `JAX_PLATFORMS=cpu` for local tests. A metadata reader should not discover
  a TPU merely because an import or installed environment permits it.
- Keep planned memory separate from measured per-chip peaks. Aggregate spare
  capacity does not show that every participating chip has headroom.
- Distinguish metadata location from payload residency. A small manifest does not
  prove the associated weights exist beneath that directory or mount.
- Preserve evidence before cleanup; verify the actual remote object set, bytes
  and applicable generation/CRC/SHA identities. Publish terminal success last.
- Use authenticated ownership and the supported leases. Timeout or a stale lock
  does not authorize killing/restarting a process or managing infrastructure.
- Use the installed locked same-region mirror, not an old backup script or a
  POSIX assumption about mounted object storage. Do not create full-size safety
  copies or modify retention merely to simplify a diagnostic.

## Minimal incident handoff

Keep enough to diagnose without repeating the run:

- question, expected outcome, first failing boundary and UTC run identity;
- exact command/environment and code/runtime/plan/checkpoint/input/cache identities;
- relevant raw arrays with schemas/digests, HLO and host/device ownership;
- observed output, state/load/cache/HBM evidence and separately scoped timings;
- failure logs, publication state, remote identities and cleanup evidence;
- a claims table separating observed facts, inferences, missing evidence and
  unsupported numerical/performance/quality conclusions;
- one next action that can change the diagnosis.

Only preserve data relevant to the question and existing proof contract; do not
manufacture a new large artifact to fill every field. Mark unavailable evidence
absent. Do not put model weights, private questions or raw response payloads in Git.

## Recover the historical campaign

The complete original playbook and tools remain at starting main
`b667f00f1ae48c8ff37e92500550c1395d74c66d`, also preserved in research history:

```bash
git show b667f00f1ae48c8ff37e92500550c1395d74c66d:docs/greenfield/GATE_D_OBSERVABILITY_PLAYBOOK.md
```

Use the same `git show <commit>:<path>` form for original experiment sources or
receipts; [the disposition ledger](../curation/README.md) records exact recovery.
Those archives explain the research, not current launch instructions. Preserve
their original failed and successful claims rather than rewriting them as proof
of today's source.

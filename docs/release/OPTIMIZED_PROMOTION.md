# Optimized ordinary release promotion

The owner requested a usable optimized engine on main on 2026-09-20. This
supersedes the documentation-only publication scope; optimization and speculative
experiments remain stopped. This candidate is not yet a promoted release.

## Precedent inspected

The earlier release connected the protected user controller, loader and request
executor, admitted the actual release source and graphs, and validated an
ordinary response (DB621). The subsequent curation at `493b67de` reduced main
from 1,971 to 616 files, preserving 1,380 removed originals with exact recovery.
`c7f6d42f` extracted shared native helpers before removing campaign modules;
`17690311` trimmed admission tooling to its supported consumers. Numerical
identities and original evidence were preserved, not silently re-registered.
See [curation](../curation/README.md) and [deployment order](OPERATIONS.md).

## Supported candidate boundary

Promote the measured ordinary greedy path: D1 grouped experts with the empty-slot
fix, D8 resident BF16 non-routed tables, D10 selection, D4 packed request loop,
and D8/P1/P2 prefill. Its measured profile uses 8,192 total prompt/output slots.
The historical ~14.3 tok/s suite and 138.85 prompt tok/s short result do not
qualify different sampling, context capacities or a changed executable.

The release must expose private request preparation and an actual protected
user command using this engine. Preserve the older sampled/long-context path
and evidence explicitly; do not silently reduce a requested context or alter
sampling. Reject unsupported optimized profiles before allocating devices.
MTP, decode D5, fused reductions, global-max and fused-EP experiments stay on
research branches. Historical research is recoverable at `e3290fd8`.

## Completion gates

- Trace required files from the usable command through loading, prefill, decode,
  token delivery, stop policy and cleanup; import only the necessary dependencies.
- Preserve numerical bodies where practical; prove extracted interfaces and
  failure behavior on CPU with JAX_PLATFORMS=cpu.
- Validate the exact release integration, real-weight token agreement, fresh
  graph/memory admission, delivery timing and authenticated eight-host cleanup.
  This is bounded release validation, not a resumed optimization campaign.
- Update file dispositions, packaging and release checks. Resolve review findings;
  self-review is not independent review.
- Publish to private main only after verified regional backup. README describes
  the usable release, installation, usage and measured scope. Future work is at
  most two sentences at its end; detailed failures belong in historical records.

Promotion remains pending until these gates and the actual main update finish.

## Candidate implementation

`glm_tpu/optimized` extracts numerical helpers from trained short-request pin
`5ff7b01e7a6520c652e7ce6dcc4a7012363e0ff8`. The fixed profile retains routed
256x256 projection tiles and removes the rejected decode-LSE/fused-reduction
bodies. Prefill still uses its separately qualified partial-attention/LSE path.
The prefill window differs from frozen source only in dependency binding; CPU
AST checks preserve its control flow. No frozen numerical source is edited.

The candidate adds a private request schema, protected controller/worker and cold
runtime. The controller stages a published Git archive; workers verify its whole
manifest before opening devices, authenticate the existing topology rebinding,
verify checkpoint bytes, and admit fresh graphs and live memory. It preserves
local token delivery and poisons failed requests rather than retrying. Deadlines
participate in existing delivery votes, avoiding divergent per-host dispatch.

Completed CPU checks include tied/skewed DSA selection, B114/B128 dense placement,
complete prefill and atomic append refusal, retained decoder tokens/selections and
cache bounds, and host privacy/failure/SSH gates. The
[complete CPU release check](optimized-cpu-check-20260920.json) passed 571 tests
with one skip. Trained entry validation is the existing
`optimized_request_20260920T170223095076Z` at `4f551e6b`; its result must be
collected before admission. These CPU checks are not throughput results.
See [candidate usage](OPTIMIZED_INFERENCE.md). Review is self-review in this chat.

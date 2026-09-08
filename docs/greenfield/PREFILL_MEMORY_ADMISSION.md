# Batched-prefill resident memory admission

2026-09-08. CPU-tested accounting and host-adapter integration; NOT a new TPU
memory result, full HLO admission, or long-context feasibility proof.
Starting pin `00ced366667831ce36c0f0ed03bde439117ce948`.

## What is implemented

`validation/ws32_prefill_memory.py` inventories explicitly named roots plus
`jax.live_arrays()` once before the first model dispatch. Roots/array leaves
stay alive through the snapshot, preventing allocator address reuse during
the census. Each local shard contributes its on-device allocated byte count;
global logical array sizes are not multiplied into each chip's budget.
No tensor payload is downloaded to the host.

Equal nonzero buffer pointers on the same device identify shared allocations.
The largest size at that pointer is counted, not the smallest view. Distinct
pointers are never merged using guessed overlap. If the backend cannot expose
a pointer, distinct array objects count separately; this is explicitly a
conservative upper count, not a physical-alias proof. Raw/decode/repair views
of the same array object are therefore not blindly added several times. Real
CPU-array tests exercise pointer deduplication; TPU pointer support and the
actual live census remain unmeasured. Pointer/object addresses are not archived.

The all-live census also catches retained arrays that are NOT arguments of the
active executable, including decode-only overlays or compile placeholders.
Every declared device requires coherent current/peak/limit counters. Missing
counters, deleted arrays or backend/process/device drift refuse the snapshot.

For each chip and each acquired prefill graph:

```
resident = max(accounted live buffers, allocator current, compiled argument bytes)
estimate = resident + all declared resident model executable code bytes
                    + active graph output bytes + active graph temporary bytes
headroom = device limit - max(estimate, previously measured peak)
```

Both prefill programs remain resident and contribute code bytes. Only the active
program contributes outputs and scratch; seven graph argument trees are NEVER
summed. The acquired prefill has zero aliases, and any alias change refuses.
No donation/alias subtraction is assumed. Code bytes can overlap allocator
current bytes; conservative double counting there is intentional because the
acquisition did not establish code residency from counter changes.

`make_prefill_memory_record` reads both actual compiled analyses and accepts an
explicit mapping of additional resident model executables. Their analyses are
included in BOTH budgets. Replay recomputes every estimate and requires every
declared executable; a stored pass is not trusted. Completeness of this mapping
still depends on the audited caller lifecycle—JAX live arrays do not enumerate
executable code. Compiler/runtime allocations not reported by these APIs are
not claimed bounded by the estimate. Actual numerical peak measurement remains
mandatory. The protected workload must preregister an explicit positive reserve;
there is no default reserve or new production admission in this change.

## Host-loop integration and lifetime

`ws32_batched_prefill_runner.execute_graph_pair` now requires that reserve.
After creating fresh state and transferring the first narrow input block, it
takes ONE census and budgets both programs. Every host votes on the outcome,
including a local capture/replay exception; a peer refusal prevents the first
dispatch everywhere. The census is not repeated per block/layer/token. Its wall
time is recorded separately and included in request-prefill wall. The
projection treats this as a one-time cost, not a cost multiplied by remaining
prompt blocks (a slow-census regression prevents that false early refusal).
Existing per-block fleet health and wall/logging votes remain, including final block.
The execution record includes the census/analyses/budgets and the shared record
validator rederives them. Numerical worker and sealer admission remain disabled.

The adapter makes its own fresh current state. It retains no separate initial
decoder argument; inputs are replaced before each next dispatch, and
current/result/state/final_result reference the current generation. A weakref
regression with three newly allocated cache generations proves the initial cache
is released before dispatch2. One current plus one proposed output is the model
state allocation covered by the formula. Review initially inferred a retained
decoder argument incorrectly; reading the actual signature/body and the lifetime
test resolved that hypothesis without adding an unjustified extra-state reserve.

REAL wiring hazard: the worker currently holds compile placeholders in `state`,
`repaired_buffer` and `batched_state`. Those are distinct from fresh adapter
state and would be included in the all-live census if retained. Release all
placeholder aliases before invoking the numerical adapter; do not allocate a
second long-context state merely because each function has a convenient factory.
After prefill, also release raw-only weights/repair state/executables when the
promoted decoder no longer needs them. No such worker lifetime change is claimed
here: numerical wiring is still the next integration step.

## Existing acquisition: orientation, not admission

Original receipt:
`../artifacts/prefill-batched-seven-graph-acquisition-20260908.json`.
At capacity8192, the recorded allocator current is25,896,472,064B/chip, main
output113,267,200B and scratch855,351,808B, main+tail generated code268,781,568B.
Using ONLY that allocator current gives27,133,872,640B, leaving5,880,526,336B of
the33,014,398,976B limit. This is arithmetic on compile-only observations, not
the new actual all-live census and not an executed-prefill peak. Additional
fresh-state allocation or nonargument views may increase it. Do not extrapolate
this headroom to128K/256K; their cache/output sizes and scratch must be acquired
and measured under their own workload.

## Remaining path to numerical admission

1. Finish the saved-graph integrated replay of the bounded operand-health delta.
   Helper/payload/repair/cache/writer checks already pass; the assembly document
   fixes the health scope. Do not expand symbolic arithmetic proofs before2K.
2. Wire numerical mode using the intended lifetime, register the reserve and
   preserve/replay each host's actual census and compiled analyses.
3. Require all32 unique physical devices, actual execution peaks, own short
   §21 numerical/state/cache proofs, trace/wall/provenance/DB/archive/cleanup.
4. Register prefill/TTFT targets and useful larger-row reuse; then efficient
   four-depth128K and full256K evidence. No unchanged acquisition or serial long run.

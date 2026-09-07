# WS32 long-run recovery after controller loss

Current case: `greenfield_ws32_short_decoder_128k_d0_0_numerical_cap131072_hrope_20260907T064941550123130Z`,
source pin `679e2392b76caf1acb0a98ff87962c5b5c908e14`. The original workers are progressing;
their detached timeout processes remain responsible for the original wall limit. Never restart them.

## Observe the original workers

`scripts/greenfield/watch_ws32_run.py` holds the workload and user rsync leases. Its first receipt
fixes the eight original PID/start-time/boot/executable/argv identities. It resumes that baseline after
a watcher restart. SSH failure is UNKNOWN, not completion; process replacement or reboot refuses.
Monitor PID at 2026-09-07 07:48Z: 2248787 (recheck before relying on it). Durable files:

```
/home/gianl/glm-run/<tag>/watch.jsonl
/home/gianl/glm-run/<tag>/watch.stdout.log
```

Two complete observations without original runners or libtpu holders yield `READY_FOR_CENSUS` and
release its leases. This is not a numerical verdict. `.ended` is not authoritative: the lost worker
shell was supposed to create it and cannot do so now. Historical exit status may be unavailable.

## Collect only after the fleet finishes

Use `scripts/greenfield/collect_ws32_worker_evidence.py`, shipped as reviewed source through the
existing authenticated SSH channel. It imports no model code and performs no TPU work. Keep the
controller's `JAX_PLATFORMS=cpu`; do not change worker repositories or the live enforcement surface.

1. Recheck monitor process and latest receipt. After it has terminated, acquire the canonical user
   workload and rsync leases, then run the existing wrapper's authenticated strict census. Require
   eight unique `CENSUS_OK` hosts. A free lease by itself proves nothing about surviving workers.
2. Read `configs/greenfield-ws32-l7-d0-recovery.json`. It preserves the original worker argv (SHA
   checked against the first monitor receipt), original source pin, all 36 wrapper recovery environment
   settings and eight topology-file SHA-256s. Its prerequisites are mandatory.
3. On every host invoke the collector with `--tag`, `--code-hash` and that host's
   `--topology-capture-sha256`, omitting `--publish-inventory-sha256`. Save all eight `WS32_COLLECT`
   JSON lines locally. The local live guard must pass too. The inventory binds original SUCCESS
   JSON, NPZ, log, trace and fourteen raw HLO files. Host rank and JAX process index are different;
   the pinned topology maps them (host 0 is JAX process 3).
4. Parse only the eight `WS32_COLLECT ` lines and call `require_fleet_inventories(rows, tag, pin)`
   from the collector module. It requires unique ranks, intact inventory digests and identical shared
   HLO sizes/SHA-256s. **Do not publish any host before this whole-fleet check passes.**
5. Invoke the same worker command again, adding its own exact inventory digest as
   `--publish-inventory-sha256`. The worker re-inventories and refuses changed inputs. Absent payloads
   are created with generation-zero preconditions; every existing/new object is generation-readback
   checked for size/CRC32C and original SHA-256. Shared gzip objects are compared by their bounded
   inflated raw HLO SHA, so original GNU gzip containers remain valid and are never overwritten.
6. Require all eight publication receipts and another authenticated eight-host clean census. Keep
   receipts local in a collection-specific directory under the run root. Release the collection
   leases; the existing recovery wrapper reacquires its workload lease and repeats its census.

The remote primary set is exactly 46 objects:

| Objects | Count |
|---|---:|
| `host_records/runner.rankN.{json,npz,log}` | 24 |
| `traces/trace.rankN.xplane.pb` | 8 |
| `hlo/GRAPH.{stablehlo.mlir,optimized_hlo.txt}.gz` | 14 |

Graphs: exact_materialize, exact_promote, prefill_chunk, prefill_tail, observer, decode, cache_probe.
Do not publish extra collector receipts, raw/per-rank HLO, fabricated completion markers, modified
JSON, or a SUCCESS object. Preserve existing `diagnostic_local/<tag>/` objects. Keep local temporary
files out of `fleet/`, `fleet_hlo/` and `traces/`; the materializer checks those exact layouts.

## Seal through the existing recovery path

Preserve the original `remote_vacancy.txt`, sync/launch/census files and exact-DSA source summary and
SUCCESS. A failed original launch log is historical evidence; do not fabricate `WS32_SHORT_OK`.
Use the capsule's `recovery_environment` with the clean published checkout and invoke the existing
`run_short_decoder_ws32.sh`. Confirm all run pins and the same-region mirror first. `RECOVER=1` skips
numerical launch but performs rollback/archive before its census, so never invoke it while workers
are live. Keep the user rsync lease around recovery; the wrapper owns the workload lease itself.
The original source pin remains `679e2392`; recovery pin is the published current enforcement code.
Only the original arrays plus unchanged sealer decide correctness, DB publication and terminal SUCCESS.

The collector's 28 combined unit tests pass; its inventory replays the prior sealed DB573 host-0
originals as 18 files, digest `a514136898bb92d41bb53e0d4ef0c90d22ea7c0ce807cd8c522054d3382c3f8a`.
An invocation while the actual worker remained live refused before inventory or cloud writes.
Independent Astra review approved persistence and later idle collection under the workflow above.

## Before subsequent runs

Apply the deferred surface-isolation, legacy dependency coverage and refusal-wording fixes after
the current run seals. Also enforce §23.5's 256-step E0 window in the sealer before any L8 launch.
The wrapper now selects 256 timed steps for E0, retaining 10 for L7/short context; the old unconditional
10 would have produced an insufficient E0 measurement after a ten-hour prefill. Its behavioral test
executes the actual shell selection and verifies the full 262,419-position requirement fits 262,656.
The diagnostic-token cardinality predicate is separate from this performance-window requirement.

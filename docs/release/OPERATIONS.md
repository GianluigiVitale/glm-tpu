# Operations and recovery

The engine runs on one existing TPU v4 slice of eight hosts with four chips each.
It never creates, resizes, restarts or deletes a TPU, VM or queued resource;
provisioning is outside this repository. Everything site-specific comes from the
site file ([INSTALLATION](INSTALLATION.md#the-site-file)).

## Entry points

| Entry | Role |
|---|---|
| `python -m glm_tpu` (`glm-tpu`) | `info`, `collect-env` (`doctor`), `prepare-request`, `checkpoint inventory`/`mark-source`/`verify` on local files; `ask` prepares requests and runs the controller |
| `python -m glm_tpu.executor.multihost_executor` | the rank-0 controller: one request (or a resident session) on the fleet |
| `python -m glm_tpu.worker.tpu_worker` | the per-host worker; the controller starts it on every host (it refuses to run without the controller's handshake) |
| `python -m glm_tpu.model_loader.pack_worker` | the per-host checkpoint packer, started by a packing driver outside this repository ([CHECKPOINTS](CHECKPOINTS.md#packing)) |
| `python -m glm_tpu.entrypoints.serve.server` | the loopback chat UI and `/v1` API attached to a resident session ([UI](../UI.md), [API](../API.md)) |
| `glm_tpu/executor/remote/*.py` | standard-library helper programs the controller sends to the hosts as `<interpreter> -c <text> <JSON>`, where the interpreter is the site's `fleet.helper_python` (`fleet.worker_python` for `stage_bundle.py`); never run by hand |
| `python -m tools.equivalence` | the CPU graph-equivalence gates ([README](../../tools/equivalence/README.md)); never touches a TPU |

The command line and the controller are not alternate ways around the controller's
checks: a request reaches the TPU only through the controller.

## Before a launch

The controller runs on rank 0 (host 0 of the slice), from a checkout the site's
`[launch]` policy admits (`glm_tpu/executor/launch_policy.py`): a branch matching
`allowed_branches` (a detached HEAD is refused), an origin equal to
`expected_origin` when set, a clean worktree and a HEAD equal to the origin's
branch head when `require_clean` and `require_pushed` are set. The staged source
is `git archive` of that commit, so an edit of the checkout during a run changes
nothing that is sent.

Keep the checkpoint, tokenizer files and topology binding in place
([CHECKPOINTS](CHECKPOINTS.md)). JAX runs with `JAX_PLATFORMS=cpu` in every
command except inside the workers, which the controller starts with the TPU
platform.

## What a run does

1. Validates the site file and the prepared request (outside the checkout,
   owner-only), resolves the launch commit and reads the remote helper texts
   of that commit (their SHA-256s go to the run's `helpers.json`).
2. Creates an owner-only run directory under the site's `paths.run_root` and
   prints `RUN <directory>`.
3. Takes the site's locks: every `locks.workload` lock without waiting (a live
   owner is a refusal) and every `locks.sync` lock, waiting (a scheduled backup
   only delays staging).
4. Checks over SSH that all eight hosts are idle and that it runs on rank 0.
5. Stages the source archive, the request and the resolved site configuration
   (`site.json`, bound by its SHA-256) to every host and runs a CPU-only worker
   preflight on each; the eight environments must agree. It then releases the
   sync locks.
6. Starts the eight workers. Each verifies the staged source, the site, the
   topology binding and the checkpoint, loads its owner files, compiles the
   programs, checks the compiled graphs' collectives and the live memory, and
   serves the request with an all-host vote at every phase and token.
7. After the workers end (completion, stop or failure), checks all hosts idle
   again, collects every host's records once (nothing is overwritten; a
   divergent record is kept under `final/`), writes `controller_terminal.json`
   and releases the workload locks.

There is no attach, resume or automatic retry. `--wall-seconds` bounds the run
(for a resident session: startup plus the first request group, then each later
group).

## Resident sessions

`--keep-loaded` keeps the workers, the loaded model and the workload locks after
the first answer and serves later prepared requests from the run directory's
`inbox/` in sequence order ([ordinary inference](OPTIMIZED_INFERENCE.md)). A
resident session ends only when `inbox/stop.json` holds `{"stop":true}`, after
outstanding work, or on a failure; either way the controller checks the hosts
idle and collects the records before it releases the locks. A successful
resident record says `all_hosts_idle_after=false`: the workers were meant to stay.
Idle time has no timeout. The chat UI and the `/v1` API use the same inbox through
one producer lock; run only one producer at a time.

## Failures

A deadline, a peer or delivery failure, or a worker exit poisons the request: it
is never retried under a new sequence, and a resident session with a failed
request ends. On failure the controller keeps the controller and worker
identities, the per-rank logs, the partial tokens and the runtime records in the
run directory. If it cannot verify that every host is idle after cleanup, it
prints `Cleanup unresolved; workload leases retained` and waits, holding the
workload locks, until an operator has authenticated the cleanup and ends it;
inspect the run directory before anything else. Authenticate
a process by its PID, start time, boot and command line (the records hold them)
before stopping it; an elapsed deadline alone is not permission to stop or
relaunch anything. Recover by preserving the original run directory, not by
rerunning the model to regenerate evidence.

| Observation | Action |
|---|---|
| Launch-policy or source refusal | Publish the intended commit on an allowed branch and launch from a clean checkout of it; do not bypass the policy. |
| Workload lock busy | Another model owns the fleet: use its inbox or wait for it; never remove a lock. |
| Hosts not idle, or preflight environments differ | Find and authenticate the owner; do not launch beside it. |
| Checkpoint or memory refusal | Diagnose the named pin or resource ([CHECKPOINTS](CHECKPOINTS.md)); do not lower a floor or repack an intact checkpoint. |
| Output stops at the context limit during reasoning | Report the terminal reason; it is not a completed answer. |
| Collection failed on a host | The other hosts' records are kept (`uncollected_ranks`); fetch that host's run directory by hand after it is idle. |

## Diagnosing a run

A different token does not identify its cause: checkpoint corruption, cache
ownership, rounding, an instrumented executable and an invalid comparison can all
look alike. Work from one question to the smallest check that can decide it:

1. **Establish identity.** Record the code, site, checkpoint, request, cache
   history and executable identities (the run directory's records hold them). A
   file name or a `passed` flag alone is not authority.
2. **Find the first value that differs**, in causal order: loaded state, residual
   operands, normalization, cache, query and key inputs, DSA selections, then the
   output tokens. Stop at an evidence gap.
3. **State a falsifiable hypothesis**: the boundary, the expected change and what
   stays invariant. Prefer a recorded value or an existing graph over a new run.
4. **Check observation effects.** A callback or an extra output can change
   fusion and scheduling; an instrumented run is evidence about the original only
   when its executable and outputs match.
5. **Keep evidence levels apart.** CPU reference arithmetic, lowered StableHLO,
   optimized TPU HLO, device values and end-to-end answers answer different
   questions.
6. **Preserve a useful failure**: its inputs, classification and location.

| Question | Where to look |
|---|---|
| Which devices and groups take part? | `glm_tpu/distributed/mesh.py` (the physical mesh), `glm_tpu/distributed/topology.py` (the topology binding), `glm_tpu/runner/admission.py` and `glm_tpu/runner/hlo_utils.py` (the collective check of each compiled graph) |
| What happened during the run? | the run directory: `controller_identity.json`, `helpers.json`, the per-rank logs and records, the token event files, `resident-measurement.json`, `controller_terminal.json` |
| Did the tokens change after a code change? | `python -m tools.equivalence compare-run RUN --golden DIR` (token equivalence against earlier run directories, matched by request) |
| Did a code change alter the device programs? | `python -m tools.equivalence check` on the CPU ([README](../../tools/equivalence/README.md)) |

An incident note should hold: the question and expected outcome, the first failing
boundary and the UTC run identity; the exact command, environment and identities;
the relevant values with their schema and digest; observed output, memory and
timings, each with its scope; failure logs and cleanup evidence; which statements
are observed, inferred or missing; and one next step that can change the
diagnosis. Keep weights, private questions and raw answers out of Git.

## Weights and storage

Weights are external and never in Git. The packed checkpoint lives in tmpfs on
every host, which a host restart loses; mounted weights alone do not prove that a
cold start will succeed. Keep private questions, answers and raw databases out of
Git and keep compact receipts instead. Do not make full-size safety copies of the
weights.

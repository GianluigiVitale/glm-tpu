# Handoff

The state of the repository after the public-structure release, for whoever
continues the work. The working rules are in [AGENTS](AGENTS.md), the goal and
its acceptance criteria in [goal.md](goal.md), the measured results in the
[release status](docs/release/STATUS.md).

## Current state (2026-09-28)

- **Branch.** `release/public-structure-20260922` holds the finished release;
  stage S9 (`62e4e615`) rewrote these governance files, and one later unit
  (H17: the chat UI shows the session's context capacity) followed. It is not
  merged: `main` still holds the pre-refactor GLM-5.3 release (tag
  `pre-refactor-main-20260922`). Merging is the owner's decision after review.
- **Code.** After `ab4c6582`, the commit the TPU comparison ran (below), only
  the chat UI's context display changed under `glm_tpu/` (H17: the workspace
  state reports the resident session's capacity and the page footer shows it,
  where both said 32K before); nothing in `pyproject.toml` changed. The device
  programs and the request path are those the comparison ran.
- **Records.** `tests/golden/data/` last changed at the H17 re-baseline
  (`http.json`: the served page and script), before that at `e6fdd113`;
  `tools/equivalence/closure_map.toml` is empty, as it must be between units;
  `tools/equivalence/kernel_renames.toml` keeps its four permanent rows (the
  public Pallas kernel names mapped back to their `181c013e` spelling).
- **Fleet** (as found at the close of the TPU comparison on 2026-09-27; not
  probed again since). The eight hosts were idle and no workload lock was held.
  No resident session ran: the 166,912-slot session of 2026-09-22 ended when the
  TPU slice was recreated on fresh disks on 2026-09-24. The verified packed
  checkpoint made for the TPU comparison (32 owner files, `checkpoint verify`
  passed) was kept in tmpfs on every host; a host restart loses it. The
  operator's site file and its site-check baseline are on rank 0, outside Git.
- **Host setting.** systemd-logind on these hosts removes a user's `/dev/shm`
  files when that user's last session ends (RemoveIPC), which deleted a first
  pack on six hosts. Lingering was enabled for the operator account on all eight
  hosts (checked at the same close), so the checkpoint survives the end of a
  session; `loginctl disable-linger` undoes it and would expose the checkpoint
  again. A recreated host needs this before a tmpfs pack.

## What the release did

The commit messages state each unit's scope, its deviations with reasons and the
gates run (with the exceptions that [goal.md](goal.md#acceptance) names). After
the harness was finished, every change of a record is a separate commit titled
`[Equivalence] Re-baseline <gates> for <token>`.

| Stage | Content | Main commits |
|---|---|---|
| S0 | the graph-equivalence harness and the records of `181c013e` | `7fa1e0f3` … `187ecb33`, records last at `d54572a3` |
| S1 | the site file, the launch policy, remote helpers as real modules, the resident-stop fix | `2257b93f`, `42205429`, `41009388`, `79e9b39e`, `3aa04d95`, `0439c399` |
| S2 | production out of the research modules, one program builder, explicit prefill calls, the research layer removed (kept at the tag); an unsharded reference model validated against the FP8 oracle | `f68388ba` … `450f6aba`; `59102426` |
| S3 | the vLLM/tpu-inference package layout, in one move commit | `90c9deae` |
| S4 | definitions in their final modules, public names, public Pallas kernel names, ruff and docstrings | `f1509d19`, `48b32282`, `27b1b411`, `4c453a0e`, `42efb433`, `a1417f0f`, `a3c6e279` |
| S5 | work units: runner and engine split, request sessions merged, serving and command-line splits, `checkpoint inventory`/`verify`, `collect-env`, admission names, bounded file input, the wording of help and messages | `9323a4fc` … `f865f782` |
| S6 | the documentation rewritten for the layout; the Apache License 2.0 | `1c255211`, `3b59b87d`, `a6f1b0e0` |
| S7 | the migration tables and checkers removed (kept in history) | `536deee1` |
| Tests | the test gaps left by S5 to S7 | `bb428063` |
| S8 | the TPU run comparison: harness preparation, then the results | `ab4c6582`, `60b68b4a` |
| S9 | these governance files, the data-contract scan of them and of the release receipts | `62e4e615` |
| H17 | after the release: the chat UI shows the resident session's context capacity instead of a fixed 32K | the H17 code commit and `[Equivalence] Re-baseline G9 for H17` |

## Evidence

**On the CPU**, commit by commit: the gates of the
[equivalence harness](tools/equivalence/README.md) against these records (a few
early commits deferred the heavy gates to their stage's close, as their messages
state).

| Record | Gate | Last written | Kind |
|---|---|---|---|
| `fingerprints_fixture.json`, `fingerprints_production.json`, `cpu_digests.json`, `checkpoint_identity.json` | G1, G2, G3, G4 | `d54572a3` | frozen: device programs, numerics, identities of `181c013e` |
| `fixture.json` | fixture | `08826aa1` | frozen |
| `load_protocol_fixture.json`, `load_protocol_production.json` | G1-protocol, G2-protocol | `cad90e6e` | reviewed, H11 (the HLO admission profile renamed) |
| `import_closure.json` | G6 | `e6fdd113` | rename-only, WU-C |
| `trace_closure.json` | G7 | `93147c55` | rename-only, WU-E |
| `wire.json` | G9 | `bfd4345c` | reviewed, S3 (module-derived fields only) |
| `http.json` | G9-http | the H17 re-baseline | reviewed, H17 (the chat UI's context display) |

For the S9 commit every CPU gate passed with every record unchanged:
`check` (G1, G1-protocol, G3, G4, G6, G7, G9), `check --tier production` (G2,
G2-protocol), `check --gates G6-static`, `selftest` (G14, 23 cases), the helper
contract tests (G8), the full test suite and the `cpu32` golden wrappers; the
counts are in the [release status](docs/release/STATUS.md#verification-and-review).
The site check (G5) and the site-bound test need the fleet; they last ran in the
TPU comparison below.

**On the TPU**, on 2026-09-27 at `ab4c6582`: new golden runs of `181c013e` and
runs of the tree on the same fleet, with the same prepared requests. B1 (10
sequential requests) 10/10, B2 (4 concurrent) 4/4 and R3 (resident: a first
request and two inbox rounds, then stop) 3/3 token streams identical; slowest-host decode speed
0.994–1.024 of the baseline; the programs the fleet compiled equal the ones the
harness lowers (9/9 programs, 1,062/1,062 Pallas kernels, for both trees); the
site check passed before and after; a dirty checkout was refused; the chat UI
and the `/v1` API answered against the resident session. Details:
[release status](docs/release/STATUS.md#tpu-comparison-of-the-current-tree) and
[harness README](tools/equivalence/README.md#tpu-comparison-compare_runpy).

**The resident-stop defect** of the GLM-5.3 release (after `inbox/stop.json`
the controller crashed with `FileExistsError` while writing its final records)
is fixed in `79e9b39e`: the stop keeps every collected record. In the TPU
comparison the tree's resident stop exited 0, where `181c013e` still ends in
that error.

The raw run directories, gate logs and the unit-by-unit design notes are kept
privately by the owner; the commit messages and these pages are the public
record.

## Resuming work

Use the pinned environment of [INSTALLATION](docs/release/INSTALLATION.md)
(Python 3.12, jax and jaxlib 0.10.1, numpy 2.3.5, ml-dtypes 0.5.4, and for G4
torch 2.10.0 CPU and safetensors 0.7.0): the records are bound to these versions,
and another version makes a gate skip (fail with `GLM_EQUIVALENCE_STRICT=1`).
From the repository root:

```bash
export JAX_PLATFORMS=cpu PYTHONDONTWRITEBYTECODE=1
python -m tools.equivalence budget
python -m tools.equivalence check --gates G4,G6-static,G9
GLM_EQUIVALENCE_STRICT=1 python -m pytest -q -p no:cacheprovider tests/golden -m "not cpu32"
```

`budget` must report `"live": false` before any heavy gate. Then follow the gate
table of [AGENTS](AGENTS.md#changing-the-code) for the change at hand.

A hardware run needs the owner's authorization, the site file
([template](examples/site.example.toml)), the site check
(`python -m tools.equivalence site-check`, on rank 0 with the fleet idle), a
packed checkpoint that passes `python -m glm_tpu checkpoint verify`, lingering
(or RemoveIPC off) on every host before a tmpfs pack, and both workload locks
free ([OPERATIONS](docs/release/OPERATIONS.md),
[CHECKPOINTS](docs/release/CHECKPOINTS.md)).

## Post-release backlog

None of these blocks the release. Each is its own unit with its own gates.

- **Harness.** `authenticity` compares only the first original it finds for each
  program and ignores later duplicates; it should report which file it used and
  compare or refuse disagreeing duplicates. Its report key `kernel_names` holds
  the mode and would read better as `kernel_name_mode`. A failing `selftest`
  exits 1 with a traceback instead of the list of failed cases. The
  sensitivity half of case `j-public` compares a fresh relowering with the kept
  program instead of kept with kept. A public-names `location_free()` nested in
  a recorded-names one keeps the recorded names (no path nests today), and a new
  Pallas kernel without a `kernel_renames.toml` row fails `j-public` with a cause
  that is not obvious.
- **Behaviour.** The chat UI and `/v1` server attach only to a controller started
  as `python -m glm_tpu.executor.multihost_executor`, identified by a dispatch
  receipt that no command of the repository writes; a session started with
  `glm-tpu ask --keep-loaded` cannot be attached. The operator workflows that
  write the source completion marker, capture the topology binding and drive
  packing are not in the repository
  ([quickstart](README.md#quickstart-from-the-weights-to-an-answer)).
  `collect-env` run from an uninstalled checkout can read stale
  in-tree build metadata. `checkpoint verify` should say
  in its help and report what it does not check, and a malformed manifest (a
  JSON list) escapes it as a traceback. The site-file reader should check the
  opened descriptor and close it on every path, as `read_bounded` now does. One
  sparse-MLA refusal message lost a space ("for1..32"). `prepare-request
  --profile` offers 8k and 128k while `ask --context` offers 8k, 32k, 128k and
  256k. The FP8 weight-decode programs are cached by table shape and sharding
  spec but not by mesh: correct while a process uses one mesh, as production
  does.
- **Tests.** The documentation test accepts a tracked file deleted from the work
  tree without `git rm` as a link target, stops the suite at import where git
  cannot run, and would read a footnote definition as a link. The sampler test
  requires an all-gather and writes the vocabulary size out. The `Resident`
  tests in the HTTP handler's test module belong with the resident client's
  tests, and the engine test imports helpers from another test module instead of
  `tests/fixtures/`.
- **Text.** Docstrings and comments in `glm_tpu/`, `tests/` and `tools/` cite
  section numbers of the private design document (for example "DESIGN 5.6"); a
  few comments and one `info` value keep research wording; some docstrings and
  documentation sentences are stale in detail (a top-k merge docstring, the
  prefill-semantics test docstring, a side-by-side sentence in TESTING, the API
  page's offline test list). OPERATIONS and CHECKPOINTS should state the
  lingering requirement above.
- **Publication (owner).** The history keeps private infrastructure literals in
  old trees and commit messages; a whole-tree private-literal scan should exist
  before any publication. The data-contract scan of the governance files and the
  release receipts catches home paths, private addresses, run and pack names,
  bucket URIs and zones, but not node, host or account names, and it accepts any
  value under a receipt's `run` or `run_id` key. Decide whether the wheel keeps
  the GLM-5.2 MIT text and the Hugging Face Apache text among its license files.

## Limitations

- The CPU proof lowers the TPU programs but never runs XLA's TPU compilation; a
  compiler or libtpu change is caught only by a TPU comparison. The failure paths
  of the fleet votes are proven with CPU fakes only; the TPU comparison induced no
  failure.
- The TPU comparison covered 17 requests at 32,768 slots. The 8,192 and 166,912
  slot programs are covered by the CPU fingerprints, the 262,144-slot capacity by
  a unit test only. Four kernel components have no stand-alone CPU golden; the
  prefill and decode goldens and the program fingerprints cover them.
- The site check reads rank 0's assets only; the other hosts are covered by each
  worker's own load-time verification.
- The golden runs of the TPU comparison were made again at `181c013e` on the
  recreated fleet (the first ones were lost with the earlier slice), with a
  checkpoint re-packed from the pinned source whose slot records equal the
  sealed manifest.
- The GLM-5.3 release's own limits stand: a partial, owner-stopped GSM8K
  evaluation, short prompts, sequential resident mode, no durable recovery
  ([release status](docs/release/STATUS.md#limitations)).
- Review: the restructure was implemented by assistant sessions. From S4.2b on,
  each unit's code commit was checked with its raw gate evidence by a separate
  adversarial verifier session before it was pushed (in all units but one, on
  the local commit); fixes for its findings were amended into the unpushed
  commit, the other findings were recorded as follow-ups, and a re-baseline
  commit, where the unit needed one, was recorded after that pass. Before S4.2b
  there was no standing verifier step: some commits cite review rounds or a
  verifier, many cite none. The owner reviews afterwards. None of this is
  independent human review.

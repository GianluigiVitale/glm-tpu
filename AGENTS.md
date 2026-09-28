# Working in this repository

This file is for anyone, person or agent, who explores or changes this
repository. The working rules come first; then the
[repository map](#repository-map) lists every directory and file with one line
on what it does, and [where to start](#where-to-start) follows a request from the
command line down to the TPU program. The current state is in the
[handoff](HANDOFF.md), the [goal](goal.md) and the
[release status](docs/release/STATUS.md); the development policy is in
[CONTRIBUTING](CONTRIBUTING.md) and the exact test commands in
[TESTING](docs/release/TESTING.md). Instructions written for earlier campaigns
(the GLM-5.2 and GLM-5.3 releases, the research tree) are history, not
instructions to restart that work; the older version of this file is
`git show archive/research-20260922:AGENTS.md`.

## Roles and review

- The owner is the integrator and the only one who approves a re-baseline of the
  equivalence records, a deviation from these rules, a run on the TPU fleet, a
  merge into `main`, a publication and any change of repository visibility or
  upstream.
- An approval given in advance for a class of work covers that work's
  re-baselines, hardware runs and decisions; record every decision taken under it
  where the owner will review it. It never covers a merge into `main`, a
  publication, a change of visibility or upstream, or a deviation from a hard rule:
  each needs the owner's explicit approval for that instance.
- Merge into `main` only after the release checks and the review have passed and
  a backup of the work has been verified.
- Work in units, one at a time: write down the scope, expected record changes and
  risks first; implement; run every gate the change requires (below) on the tree
  you commit; commit locally; have a separate verifier pass review the commit and
  the raw gate evidence adversarially before anything is pushed; amend blocking
  fixes into the unpushed commit and re-run the gates they affect; record the other
  findings as follow-ups; push. The owner reviews each unit afterwards. A verifier
  pass by another assistant session is not independent human review; describe
  review as it was done, and resolve material findings before a merge.

## Hard rules

- Tests and source audits run on the CPU (`JAX_PLATFORMS=cpu`). Never initialize
  a TPU outside a hardware run the owner authorized.
- Never create, delete, resize or reconfigure compute (TPU VMs, nodes, queued
  resources), write instance metadata or run provisioning scripts. A hardware run
  holds both site workload locks, runs one workload at a time and is waited for;
  staging or repacking a checkpoint also holds the site's sync locks
  ([OPERATIONS](docs/release/OPERATIONS.md),
  [CHECKPOINTS](docs/release/CHECKPOINTS.md)). Never launch beside a live run or
  interrupt a run's collection of its evidence.
- Before stopping or removing a process or a file, prove that it is yours: PID,
  start time, boot and command line for a process; an inventory of what the unit
  created for a path. An elapsed deadline is not permission to stop, relaunch or
  delete anything.
- Storage: only the project's bucket in the fleet's region, as the operator's
  site file names it, within its existing storage bounds. No full-size safety
  copies of the weights.
- Keep out of Git: weights, credentials, private questions and answers, raw run
  directories, token logs, caches, databases, large generated artifacts, and
  private infrastructure literals (home directories, private addresses, host,
  VM, zone and bucket names, timestamped run or pack directory names; the
  committed release receipts and the migration history keep the run names they
  recorded). Cite compact receipts, pins and digests instead. Never print a
  suspected secret.
- **No force-push**, no history rewriting, no change of visibility or upstream;
  push only this private repository, fast-forward. Preserve other people's
  changes, every tag (among them `glm-5.2`, `glm-5.3`, `archive/research-20260922`,
  `pre-refactor-main-20260922` and `freeze-correct-128k-20260712`), the research
  branches, the receipts and the original evidence.
- Before removing a file, establish what depends on it and that a preserved
  commit (a tag or a research branch) keeps it; static import reachability alone
  is not deletion authority.
- Do not resume the owner-stopped GSM8K evaluation or the research optimization
  campaigns.
- Shell work: absolute paths, `git -C <path>`, `set -euo pipefail`, and a scratch
  directory of your own for each task. Do not edit a checkout that someone else
  or a live run is using. Never run a mutating git command (checkout, switch,
  reset, stash, clean, restore, worktree removal) in a checkout you did not
  create, and never with `--force`.
- Wait for long jobs by completion notification, or check them at least ten
  minutes apart unless you are diagnosing a known failure or answering an
  explicit request for status.

## Changing the code

Code lives in `glm_tpu/` and its tests in the mirrored path under `tests/`. A
change of behaviour comes with a test that fails without it and is declared in
its commit message.

### Tests and gates on the CPU

Every command runs from the repository root with
`JAX_PLATFORMS=cpu` (the tests force it; `tests/conftest.py` refuses any other
platform):

```bash
export JAX_PLATFORMS=cpu PYTHONDONTWRITEBYTECODE=1
python -m pytest -q -p no:cacheprovider -rs tests --ignore=tests/golden                   # the suite, cpu32 included (~20 min)
GLM_EQUIVALENCE_STRICT=1 python -m pytest -q -p no:cacheprovider -rs tests/golden -m "not cpu32"   # data contracts, light wrappers (~1 min)
python -m pytest -q -p no:cacheprovider tests/test_documentation.py                       # links, documented test paths, this map
python -m tools.equivalence check --gates G4,G6-static,G9                                 # the light gates (under a minute)
python -m tools.equivalence check                                                         # G1 G1-protocol G3 G4 G6 G7 G9 (~20 min)
python -m tools.equivalence check --tier production                                       # G2 G2-protocol (~12 min)
python -m tools.equivalence selftest                                                      # G14 (~4 min)
GLM_EQUIVALENCE_PRODUCTION=1 GLM_EQUIVALENCE_STRICT=1 python -m pytest -p no:cacheprovider -rs tests/golden -m cpu32   # heavy wrappers (~30 min)
GLM_TPU_TEST_HELPER_PYTHON=python3 python -m pytest -q -p no:cacheprovider tests/engine/test_resident_protocol.py tests/executor/test_remote_helpers.py tests/executor/test_remote_loopback.py   # G8 helper contracts
```

Durations are those of a 240-core host. The heavy gates and the `cpu32` tests
refuse or skip while a TPU run is live on the host
(`python -m tools.equivalence budget`): run them before or after a hardware run,
never beside it. Run the gates of every row that applies, on the tree you commit:

| Change | Gates |
|---|---|
| code, tests or data (anything but Markdown) under `glm_tpu/`, `tests/` or `tools/` | `python -m tools.equivalence check` (G1, G1-protocol, G3, G4, G6, G7, G9), `check --tier production` (G2, G2-protocol), `check --gates G6-static`, `pytest tests --ignore=tests/golden`, `GLM_EQUIVALENCE_STRICT=1 pytest tests/golden -m "not cpu32"`, and the helper contract tests (G8) |
| `pyproject.toml`, dependency pins, `examples/` or other configuration | the gates of the first row, and for a dependency or pin change also the `cpu32` golden wrappers (below) |
| anything under `tools/equivalence/` | `python -m tools.equivalence selftest` (G14) |
| anything under `tests/golden/`, and before a release | the `cpu32` golden wrappers, which run the heavy gates through pytest: `GLM_EQUIVALENCE_PRODUCTION=1 GLM_EQUIVALENCE_STRICT=1 pytest tests/golden -m cpu32` |
| Markdown only | `pytest tests/test_documentation.py` and `pytest tests/golden -m "not cpu32"` (the data contracts) |
| every change | `ruff check .`, `ruff format --check .`, codespell and `git diff --check` ([CONTRIBUTING](CONTRIBUTING.md#formatting-and-lint)) |

### Equivalence rules

The [graph-equivalence harness](tools/equivalence/README.md) proves that a change
leaves the lowered TPU device programs, the CPU numerics, the checkpoint
identities, the import closures, the executed functions and the wire formats
equal to the records in `tests/golden/data/`. A recorded difference is never
absorbed silently ([re-baselining](tools/equivalence/README.md#re-baselining)):

- G1-G4 and the fixture are frozen: recorded only from production paths equal to
  the baseline `181c013e`, never changed by a restructuring commit. A change meant
  to alter them (numerical, kernel or model) needs the owner's decision on a new
  baseline before it starts.
- A move or rename that changes G6 or G7 carries reviewed entries in
  `tools/equivalence/closure_map.toml` in its code commit; a separate commit,
  `[Equivalence] Re-baseline <gates> for <token>`, then records from the clean
  committed code commit (`record --rename-only --reason <token>`) and empties the
  table again.
- A reviewed change of a characterization record (G1-protocol, G2-protocol, G6,
  G7, G9 with G9-http) that is not a pure rename is a separate commit of the same
  title, recorded from the clean committed code commit with exactly one reason
  token: an H number for a sanctioned host change (`H1`, `H11`, ...), a stage or
  a work unit (`WU-E`, ...).
- A normalizer change re-records nothing and re-runs G14.
- A change the CPU records cannot hold (the compiler, libtpu, host behaviour)
  needs a TPU comparison: golden runs of the previous tree and runs of the change
  on the same fleet, compared with `python -m tools.equivalence compare-run`, in
  hardware time the owner authorized.

Commit messages: a title prefix (`[Refactor]`, `[Tests]`, `[Docs]`,
`[Equivalence]`, `[Launch]`, `[Config]`, `[Packaging]`) and the unit's token where
there is one; what changed and why; deviations with their reasons; the gates run
on the committed tree with results and durations, and what was not run and why;
the expected re-baseline, or "none". No tool-attribution trailers.

## Repository map

Every tracked directory and file, with what it does. The layout of `glm_tpu/`
follows vLLM and tpu-inference; `tests/` mirrors it.
`tests/test_documentation.py` fails when a tracked file under `glm_tpu/` or
`tools/` is missing here or when an entry names something that does not exist:
add a line when you add a file. Each block below is a tree: two spaces of
indentation per level, a directory ends in `/`, and the text after the name says
what the entry does.

### Root

```text
README.md                    Overview, measured results, layout, CPU checks, how to run inference, documentation index
AGENTS.md                    This file: working rules, the repository map, where to start
HANDOFF.md                   State of the repository, evidence, how to resume, the backlog and the limitations
goal.md                      Goal of the public-structure release and its acceptance criteria
CONTRIBUTING.md              Development policy: branches, CPU-only tests, equivalence gates, formatting and lint
SECURITY.md                  Private data, the loopback-only serving surface, the release checks
LICENSE                      Apache License 2.0 of the project's own work
THIRD_PARTY_NOTICES.md       What keeps its own license (the GLM-5.3 configuration files), history-only material, data
pyproject.toml               Package metadata, exact dependency pins and extras, the glm-tpu script, package data, ruff
.gitignore                   Keeps weights, results, secrets, private request files and build outputs out of Git
```

### The engine package: `glm_tpu/`

```text
glm_tpu/                     The engine (python -m glm_tpu, console command glm-tpu)
  __init__.py                Package docstring only
  __main__.py                python -m glm_tpu: runs the command line of entrypoints/cli/main.py
  envs.py                    Registry of the GLM_TPU_* environment variables, read lazily at attribute access
  exceptions.py              Fail-closed configuration errors and ApiError, the refusal of an OpenAI-compatible request
```

```text
glm_tpu/config/              Configuration; importing it imports no JAX
  __init__.py                Package docstring: what each module configures
  model.py                   Pinned GLM-5.3 identity (MODEL_ID, REVISION), exact hashed geometry, hf_config and template checks
  cache.py                   CacheConfig (geometry x context capacity) and the MLA, KV-layout, DSA and MoE numerical contracts
  site.py                    The untracked site file: fail-closed TOML loading and validation, SiteConfig, site_args
  parallel.py                The mesh axis names "expert" and "feature" (serialized into hashes and shardings)
```

```text
glm_tpu/engine/              The request path shared by the controller, the worker, the CLI and serving
  __init__.py                Package docstring; importing it imports nothing
  request.py                 Request schemas and preparation: pinned template, tokenizer, capacities, validation, batches
  llm_engine.py              LLMEngine: generate (one fresh request) and generate_concurrent (a batch) over a loaded runner
  request_session.py         RequestSession and BatchedSession: the per-request host loop, token delivery, fleet votes
  outputs.py                 TokenEvent (one line of the token stream) and final_channel (reasoning versus final answer)
  resident_protocol.py       Every file name, module name, flag and byte string the processes exchange
  resident_client.py         Resident: the inbox client of a running resident controller, used by the UI and /v1
```

```text
glm_tpu/entrypoints/         What a user runs or connects to
  __init__.py                Package docstring
  cli/                       The glm-tpu command line
    __init__.py              Package docstring
    main.py                  The parser and subcommand dispatch; the info and ask subcommands
    types.py                 CLISubcommand, the base class of every subcommand
    ask.py                   ask: prepare one to ten questions under the run root, then run the controller in-process
    prepare.py               prepare-request: tokenize a private chat file (8k or 128k profile); launches nothing
    collect_env.py           collect-env (alias doctor): installed versions against the declared pins of a profile
    checkpoint.py            checkpoint inventory and checkpoint verify on local files (no JAX, no other host, no lock)
  openai/                    The OpenAI-compatible /v1 surface over a resident session
    __init__.py              Package docstring
    serving_chat.py          OpenAIServingChat: stateless /v1/chat/completions, buffered or streamed
    serving_models.py        OpenAIServingModels: /v1/models with the session's window and output ceilings
    chat_utils.py            OpenAI messages and tools mapped onto what the pinned chat template renders
    tool_parser.py           GLM <tool_call> output parsed into OpenAI tool calls; final_channel
    protocol.py              Model ids and aliases, reasoning efforts, request caps, ApiError
  serve/                     The loopback HTTP server of the chat UI and /v1
    __init__.py              Package docstring
    server.py                python -m glm_tpu.entrypoints.serve.server: attach to a resident controller and serve
    http_handler.py          The handler: the static page, /api/* behind Host, Origin and X-GLM-UI checks, /v1 behind a key
    job_queue.py             JobQueue: persistent chat jobs handed one at a time to the resident controller
    security.py              api_token: the owner-only API key file, created on first use
  ui/                        The browser chat workspace
    __init__.py              Package docstring
    conversations.py         ConversationStore: saved conversations, kept in the job queue's document
    static/                  Package data served by http_handler
      index.html             The page
      app.js                 The script: conversations, the job stream, rendering, the context footer
      style.css              The styling (the owner's own design)
```

```text
glm_tpu/executor/            The rank-0 controller and its launch machinery; importing it imports no JAX
  __init__.py                Package docstring
  multihost_executor.py      The controller: locks, idle checks, staging, the eight workers, resident inbox, cleanup, records
  launch_policy.py           Which checkout may launch (the site's [launch] table) and the commit it stages
  staging.py                 stage_bundle: git archive of the pinned commit, its source manifest, the request, the site
  fleet.py                   The exact SSH command strings; HelperTexts, the pinned snapshot of the helper texts
  remote/                    Standard-library helper programs sent as <interpreter> -c <text> <JSON>; never imported on a host
    __init__.py              The rules every helper follows (Python 3.10, one JSON argument, main(argv))
    idle_probe.py            This host is idle: no libtpu holder and no live worker of this run
    stage_bundle.py          Receive the staged bundle on stdin, check its digest, extract it
    start_worker.py          Write this host's worker start marker, then exec the worker
    fetch.py                 Print named records of one directory as base64 JSON
    cleanup.py               Kill only this run's authenticated worker on this host
```

```text
glm_tpu/worker/              The per-host worker process
  __init__.py                Package docstring
  tpu_worker.py              python -m glm_tpu.worker.tpu_worker: preflight, load, then one request, a resident loop or a batch
```

```text
glm_tpu/runner/              Device-program preparation for one worker
  __init__.py                Package docstring
  tpu_runner.py              TPUModelRunner: verify and load the checkpoint, compile and admit the program set; compile_batch
  programs.py                build_program_set: every program a runtime compiles, in compile order, with the donation rule
  compilation_manager.py     compile_program: lower, keep the StableHLO and optimized HLO, compile, record memory and time
  kv_cache_manager.py        build_cache_initializer: the program that allocates one request's fresh caches
  admission.py               check_hlo_collectives and project_memory: collective and live-memory admission of a graph
  hlo_utils.py               The textual XLA HLO parser the collective check uses
```

```text
glm_tpu/model_loader/        The checkpoint pipeline
  __init__.py                Package docstring
  source_inventory.py        Payload-free, content-addressed inventory of a safetensors source (index and headers only)
  placement.py               Which slice of every source tensor goes to which of the 32 device slots
  pack_worker.py             python -m glm_tpu.model_loader.pack_worker: one host's packing, started by an outside driver
  sharded_state/             The packed runtime checkpoint: one file per device slot
    __init__.py              Package docstring
    format.py                The packed format: file and tensor plans, headers, digests, RuntimePackConfig
    writer.py                pack_runtime_slots and finalize_runtime_checkpoint: owner files, manifest, SUCCESS seal
    manifest.py              assemble_owner_manifest: the manifest from the eight hosts' owner receipts
    verify.py                verify_runtime_checkpoint: metadata, manifest and SUCCESS seals, file hashes
    loader.py                load_runtime_checkpoint: place verified owner files on the device mesh
```

```text
glm_tpu/models/              Model definitions
  __init__.py                Package docstring
  glm_moe_dsa/               GLM-5.3 (GlmMoeDsaForCausalLM)
    __init__.py              Package docstring: what each module holds
    model.py                 The greedy decode step programs (single, batched and packed)
    prefill.py               The B128/B114 layer-major prefill block program
    decoder_layer.py         The decoder layer bodies for prefill and decode (attention, DSA, dense MLP or MoE)
    state.py                 Device state of prefill and decode, their result types and partition specs
    weights.py               FP8 checkpoint-layout weight trees, their specs and names, the resident BF16 tables
    hf_config/               Pinned zai-org/GLM-5.3 assets (package data under Z.AI's GLM-5.3 license; not code)
      LICENSE                Z.AI's GLM-5.3 license, verbatim
      config.json            The model configuration, source of the pinned geometry
      generation_config.json  The generation configuration
      tokenizer_config.json  The tokenizer configuration
      chat_template.jinja    The chat template every request is rendered with
```

```text
glm_tpu/layers/              Per-shard JAX layer bodies, run inside shard_map on the expert-8 x feature-4 mesh
  __init__.py                Package docstring
  contracts.py               The numerical contracts (from config.cache), shape validators, the BF16 weight groups
  embed.py                   Owner-masked token embedding
  norm.py                    Sharded RMSNorm, fused add + RMSNorm, the DSA key LayerNorms
  rope.py                    Rotary tables and rotation with explicit pairing
  linear.py                  Resident BF16 projections and their feature-4 or expert-8 reductions
  mlp.py                     The dense MLP bodies of prefill and decode
  fp8.py                     FP8 E4M3FN decoding and block scales on the device
  sampler.py                 Final logits and the greedy token, without a full-vocabulary gather
  attention/                 Attention layers
    __init__.py              Package docstring
    mla.py                   Absorbed multi-head latent attention for prefill and decode
    dsa_indexer.py           The DSA indexer: key scoring and exact top-k position selection
    kv_cache.py              Stage-local KV cache writes and selected-row gathers
  moe/                       Mixture of experts
    __init__.py              Package docstring
    router.py                noaux_tc expert selection with normalized sigmoid weights
    routed_experts.py        Routed experts (FP8 panels in prefill, route-grouped FP8 projections in decode), shared expert
```

```text
glm_tpu/kernels/             Pallas TPU kernels, one package per operation
  __init__.py                Package docstring
  names.py                   KERNEL_NAMES: the pallas_call name of every kernel, looked up at trace time
  sparse_mla/                Sparse MLA attention
    __init__.py              Package docstring
    kernel.py                Decode: the fused selected-KV gather and sparse MLA
    partial_kernel.py        Prefill: owner-local partial attention with log-sum-exp outputs
  fp8_grouped_matmul/        Routed-expert FP8 matmuls with 128x128 block scales
    __init__.py              Package docstring
    kernel.py                Decode: fp8_routed_projection over scalar-prefetched routes
    panel_kernel.py          Prefill: the M32/N256 expert-panel FP8 matmul
    panels.py                Prefill: build, pack and unpack the expert panels
```

```text
glm_tpu/distributed/         Multi-host runtime state
  __init__.py                Package docstring
  parallel_state.py          initialize_runtime (JAX distributed, topology check, mesh) and the all-host vote
  mesh.py                    MeshContract and the physical expert=8 x feature=4 mesh
  topology.py                The eight hosts' topology captures and the authenticated topology binding
```

```text
glm_tpu/utils/               Shared utilities; standard library only
  __init__.py                Package docstring
  io_utils.py                Owner-only, create-once writes, bounded reads, private-file checks
  json_utils.py              Canonical JSON: the hash contract and the wire bytes
```

### Tests: `tests/`

```text
tests/                       CPU tests, laid out like glm_tpu/ (not in the wheel)
  __init__.py                Package marker
  conftest.py                Session guard: JAX_PLATFORMS=cpu only, no operator site file, cpu32 skips during a TPU run
  test_documentation.py      Documentation links and documented test paths resolve; this repository map is complete
  test_envs.py               The GLM_TPU_* registry: lazy, documented, one definition per variable
  test_error_messages.py     Error messages carry no development-phase labels
  test_import_boundaries.py  No vLLM or tpu-inference import; layers never import models, engine never entrypoints, config no JAX
  config/                    Tests of glm_tpu/config
    __init__.py              Package marker
    test_model.py            The pinned geometry, identity, inventory binding and template checks
    test_site.py             The site file is fail-closed; the worker's site binding refuses incomplete sites
  engine/                    Tests of glm_tpu/engine
    __init__.py              Package marker
    fleet_fakes.py           Fleet and device fakes for the vote-schedule tests
    test_llm_engine.py       LLMEngine.generate and generate_concurrent over synthetic device results
    test_request.py          Fixed-profile, privacy and budget checks of request preparation
    test_request_session.py  Request-control failures and timing of RequestSession and BatchedSession
    test_resident_protocol.py  G8: every resident-protocol constant names what the processes use
    test_vote_schedule.py    The fleet-vote order and failure paths through the worker entry points
  entrypoints/               Tests of glm_tpu/entrypoints
    __init__.py              Package docstring
    cli/                     The command line
      __init__.py            Package docstring
      test_ask.py            Question preparation and dispatch boundaries, without a model call
      test_checkpoint.py     checkpoint inventory and verify on the tiny checkpoint
      test_collect_env.py    The environment report against the declared pins
      test_main.py           Help text, info output and every argument pinned; no command module loaded early
      test_prepare.py        prepare-request, without a tokenizer
    openai/                  The /v1 surface
      __init__.py            Package docstring
      test_serving_chat.py   /v1 chat: messages, tools, streaming, statelessness, capacity, keys, failures
      test_serving_models.py  The /v1/models listing
    serve/                   The loopback server
      __init__.py            Package docstring
      test_http_handler.py   UI boundaries: CSRF and rebinding checks, receipts, the capacity footer, stream errors
      test_job_queue.py      The persistent queue of chat jobs
      test_server.py         The server's command line
    ui/                      The chat workspace
      __init__.py            Package docstring
      test_conversations.py  Saved conversations and the reported session capacity
  executor/                  Tests of glm_tpu/executor
    __init__.py              Package marker
    test_fleet.py            SSH command discovery and fan-out, offline
    test_idle_probe.py       The deployed libtpu-holder guard of idle_probe
    test_launch_policy.py    The launch policy on real local git checkouts
    test_multihost_executor.py  The controller's summary, lease order, resident stop and failure paths on a fake fleet
    test_remote_helpers.py   Helpers are self-contained standard-library programs, never templated
    test_remote_loopback.py  G8: runs the exact remote command strings on this host
  fixtures/                  Shared synthetic fixtures (neutral example values only)
    __init__.py              Package docstring
    serving.py               FakeResident and open_store for the serving tests
    site.py                  Example site configurations (the harness's synthetic site)
    tiny_checkpoint.py       A one-tensor source checkpoint for the checkpoint tests
    tiny_model.py            The frozen tiny GLM checkpoint (fixture v1) of the CPU tests
    prefill_layer_schema.json  Tensor schema of the 78-layer checkpoint, for shape evaluation only
  golden/                    Thin pytest wrappers around python -m tools.equivalence
    __init__.py              Package docstring
    conftest.py              Golden markers and the recorded-version rule (GLM_EQUIVALENCE_STRICT)
    test_budget.py           Live-run detection is fail-closed about the site file
    test_checkpoint_identity.py  G4: checkpoint and format identities
    test_closure_map.py      G6/G7 comparison through the rename table
    test_cpu_digests.py      G3: CPU32 execution goldens
    test_data_contract.py    Golden data compact and public-safe; no private literal in harness, governance or receipts
    test_equivalence_cli.py  The harness command line: site-check exit, authenticity kernel names, record refusal
    test_equivalence_selftest.py  G14: the normalizer's mutation self-test
    test_fixture_equivalence.py   Fixture v1 equals the historical CPU fixture
    test_http_characterization.py  G9-http: the real UI and /v1 handler with a fake resident
    test_import_closure.py   G6: serving-stage import closures and the static layering scan
    test_program_fingerprints.py  G1/G2 program fingerprints and the G1-/G2-protocol records
    test_record_unchanged.py  record keeps an unchanged characterization file
    test_recorded_names.py   The harness's permanent recorded-name tables
    test_site_literals.py    G5 launcher constants: nothing derived from site literals is committed
    test_trace_closure.py    G7: the executed-function set
    test_wire_formats.py     G9: request bytes, token events, records and the resident protocol
    data/                    The recorded baselines: digests and small summaries, written only by record
      checkpoint_identity.json      G4
      cpu_digests.json              G3
      fingerprints_fixture.json     G1
      fingerprints_production.json  G2
      fixture.json                  Fixture v1 identity
      http.json                     G9-http
      import_closure.json           G6 and G6-static
      load_protocol_fixture.json    G1-protocol
      load_protocol_production.json  G2-protocol
      trace_closure.json            G7
      wire.json                     G9
  kernels/                   Tests of glm_tpu/kernels
    __init__.py              Package docstring
    test_fp8_panel_matmul.py  The prefill FP8 panel matmul on the CPU
    test_sparse_mla.py       The sparse MLA kernel in interpret mode against a reference
  layers/                    Tests of glm_tpu/layers
    __init__.py              Package docstring
    test_fp8.py              BF16-resident tables decoded exactly; the prefill repair key weight
    test_linear.py           Resident projection refusals, the multirow prefill projection, the dense row placement
    test_mlp.py              The canonical row placement of the dense prefill MLP and its refusals
    test_norm.py             RMSNorm and fused add + RMSNorm against GLM's rounding; mesh layout
    test_rope.py             Rotary frequencies, pairing and the FP32 final rounding
    test_sampler.py          Greedy sampling on 32 devices against a reference, without a vocabulary gather
    attention/               Attention layers
      __init__.py            Package docstring
      test_dsa_indexer.py    Bitwise top-k shortlists, including ties, skew and the forced fallback
      test_kv_cache.py       Cache addressing and causality
      test_mla.py            Multirow attention primitives and the causal prefill attention block
    moe/                     Mixture of experts
      __init__.py            Package docstring
      test_routed_experts.py  Lossless route grouping
      test_router.py         noaux_tc selection, bias and tie rules
  model_loader/              Tests of glm_tpu/model_loader
    __init__.py              Package docstring
    test_pack_worker.py      The pack worker's command line
    test_placement.py        Placement of every source tensor onto the 32 slots
    test_source_inventory.py  The inventory reconciles headers and index and refuses disagreement
    sharded_state/           The packed checkpoint
      __init__.py            Package docstring
      test_format.py         Pack, verify and load of the 32 owner files; a failure never commits a manifest
      test_manifest.py       Manifest assembly from real tiny owner files
      test_verify.py         The names the verification helpers answer to
  models/                    Tests of glm_tpu/models
    __init__.py              Package docstring
    glm_moe_dsa/             GLM-5.3
      __init__.py            Package docstring
      test_model.py          CPU32: a batched decode equals eight independent histories
      test_prefill.py        The prefill program: health, atomic refusal, donation, real-schema shapes
      test_prefill_semantics.py  Prefill blocks: handoff to decode, the layer window, the tail block
      test_weights.py        The 78-layer weight contract and names, the main RoPE table
      test_against_reference.py  CPU32: the reference model against the production composition
      floors.json            Error floors recorded from the archived FP8 oracle, read by test_against_reference.py
  reference/                 An unsharded single-device pure-JAX reference of GLM-5.3 (never production code)
    __init__.py              Package docstring
    VALIDATION.md            The acceptance receipt of the reference and its known limits
    model.py                 The reference forward, op by op
    attention.py             Absorbed MLA of the reference
    dsa.py                   The DSA indexer of the reference
    linear.py                FP8 block dequantization, projections, SwiGLU
    moe.py                   Router, routed and shared experts of the reference
    norm.py                  Normalizations of the reference
    independent.py           An FP64 NumPy restatement that shares no code with the reference
    oracle_run.py            CPU32 child: the reference against the production composition, one JSON report
    test_reference_consistency.py  The reference's unit semantics and self-consistency
  runner/                    Tests of glm_tpu/runner
    __init__.py              Package docstring
    test_admission.py        Memory and HLO admission boundaries and refusal messages
    test_hlo_utils.py        The HLO parser's collectives, shapes and operands
    test_programs.py         build_program_set: names, compile order, donation, admission flags
    test_tpu_runner.py       TPUModelRunner's refusals, voted phases and compile records
  utils_/                    Tests of glm_tpu/utils
    __init__.py              Package marker
    test_io_utils.py         Create-once writes, bounded reads, private inputs
    test_json_utils.py       Canonical JSON bytes and the hash contract
  worker/                    Tests of glm_tpu/worker
    __init__.py              Package marker
    test_tpu_worker.py       The worker's batch and resident paths and its help text
    test_local_preflight.py  Site tier: the worker preflight against the operator's real site file
```

### Tools: `tools/`

```text
tools/                       Developer tools (not in the wheel)
  equivalence/               The graph-equivalence harness, python -m tools.equivalence (CPU only)
    README.md                Method, gate catalogue, commands, re-baselining rules, known weaknesses
    __init__.py              Package docstring
    __main__.py              The command line: record, check, selftest, site-check, compare-run, diff, authenticity, budget
    gates.py                 Gate orchestration: record and check, statuses, data files, re-baseline reasons
    common.py                Shared helpers: paths, canonical JSON, the environment record, the child runner, leaf digests
    budget.py                Is a TPU run live on this host (processes, libtpu holders, workload locks)
    driver.py                Drives the real TPUModelRunner and LLMEngine on the CPU, faking only fleet and TPU parts
    programs.py              G1/G2: the device programs of the tree under test, built by production
    lowering.py              N1: location-free TPU lowering on a CPU host; the kernel-name modes
    normalize.py             N2-N8: StableHLO normalization, signatures and fingerprints
    fixture.py               Frozen fixture v1: the small real-schema checkpoint of the CPU gates
    golden_run.py            G3: CPU32 execution goldens of the production composition
    identities.py            G4 checkpoint and format identities; the G5 site check
    import_closure.py        G6: import closures of the serving stages and the static layering scan
    trace_closure.py         G7: the repository functions the production composition executes
    controller.py            The controller's launch path driven for real against a synthetic host (G9, G6)
    wire.py                  G9: serving wire formats and request identity, produced by the real code
    verdicts.py              G1/G2 safety verdicts: memory and HLO admission on synthetic inputs
    selftest.py              G14: the mutation self-test of the fingerprint procedure
    authenticity.py          Harness lowerings compared with the StableHLO a TPU run compiled
    compare_run.py           Token-equivalence comparison of real TPU runs against golden runs
    site_fixture.py          Synthetic site configurations with neutral example values
    closure_map.py           Loads and applies the G6/G7 rename table
    closure_map.toml         The reviewed G6/G7 rename table, empty between re-baselines
    kernel_renames.toml      Permanent map from the current Pallas kernel names to the recorded ones
```

### Documentation, examples and license texts

```text
docs/                        Documentation (not in the wheel)
  API.md                     The loopback OpenAI-compatible /v1 API
  UI.md                      The browser chat workspace
  release/                   Release documentation and the compact result receipts
    ARCHITECTURE.md          Module map, request path, parallelism, the model, numerical conventions
    CHECKPOINTS.md           The pinned source, the packed checkpoint, inventory and verify, packing, capacity
    CONCURRENT.md            Four concurrent conversations
    GLM53_MIGRATION.md       How the GLM-5.3 release was reached
    INSTALLATION.md          Environments, extras, collect-env, the wheel, the site file, environment variables
    OPERATIONS.md            Entry points, what a run does, resident sessions, failures and diagnosis
    OPTIMIZED_INFERENCE.md   ask, resident sessions, the inbox, stopping
    PROJECT_SUMMARY.md       The problem, approach, results and limits on one page
    REVIEWER_GUIDE.md        A reading order for reviewers
    STATUS.md                What was measured on the hardware and what the current tree is
    TESTING.md               Test layout, tiers, gates, exact commands and durations
    glm53-context-profiles-20260922.json  Receipt: admission results of the context profiles
    glm53-four-answers-20260921.json      Receipt: the four-chat test
    glm53-resident-results-20260922.json  Receipt: the resident GSM8K evaluation
examples/                    Examples of operator configuration
  site.example.toml          Template of the untracked site file; documents every key
licenses/                    License texts of material kept only in the repository history; not in the wheel
  Apache-2.0.txt             Apache 2.0 as distributed with Transformers 5.12.0, for the archived reference extracts
  GLM-5.2-MIT.txt            MIT license of the archived GLM-5.2-FP8 repository snapshot
```

## Where to start

### The request path, from the command line to the TPU

1. `python -m glm_tpu ask "..."` runs `glm_tpu/__main__.py`, then
   `entrypoints/cli/main.py` (`AskSubcommand`) and `entrypoints/cli/ask.py`. The
   question is rendered with the pinned template and tokenized
   (`engine/request.py`, `config/model.py`) into an owner-only `request.json`
   under the site's run root (`config/site.py`).
2. The controller, `executor/multihost_executor.py` `main`, runs in that process
   on rank 0: the launch policy picks the commit (`executor/launch_policy.py`),
   the helper texts are pinned (`executor/fleet.py`), the site locks are taken,
   the hosts are checked idle (`executor/remote/idle_probe.py`), the bundle is
   staged (`executor/staging.py`, `executor/remote/stage_bundle.py`) and a worker
   is started on each of the eight hosts (`executor/remote/start_worker.py`).
3. Each worker, `worker/tpu_worker.py`, checks its staged source and site
   (`config/site.py` `site_args`), joins the fleet and builds the mesh
   (`distributed/parallel_state.py`, `distributed/topology.py`,
   `distributed/mesh.py`) and creates the runner.
4. `runner/tpu_runner.py` `TPUModelRunner` verifies and loads the four owner files
   of its host (`model_loader/sharded_state/verify.py`, `loader.py`), builds the
   resident BF16 tables (`models/glm_moe_dsa/weights.py`) and the program set
   (`runner/programs.py`: the cache initializer of `runner/kv_cache_manager.py`,
   the prefill blocks of `models/glm_moe_dsa/prefill.py`, the decode step of
   `models/glm_moe_dsa/model.py`), compiles each program
   (`runner/compilation_manager.py`) and admits it (`runner/admission.py`,
   `runner/hlo_utils.py`).
5. The programs are `shard_map` bodies: `models/glm_moe_dsa/decoder_layer.py`
   composes `layers/` (norms, `attention/mla.py`, `attention/dsa_indexer.py`,
   `attention/kv_cache.py`, `moe/router.py`, `moe/routed_experts.py`, `mlp.py`,
   `sampler.py`), which call the Pallas kernels in `kernels/sparse_mla/` and
   `kernels/fp8_grouped_matmul/`.
6. `engine/llm_engine.py` `LLMEngine.generate` drives the loaded programs through
   `engine/request_session.py`: prefill, then one decode step per token, with an
   all-host vote at each phase and token; rank 0 writes `engine/outputs.py`
   `TokenEvent` lines. The controller collects every host's records and cleans up.
7. With `--keep-loaded` the controller keeps the workers and serves the run's
   `inbox/` in sequence order; the worker's `resident_loop` reuses the loaded
   runtime (`engine/resident_protocol.py` names every file and command).

The chat UI and `/v1` reach the same resident session from the side:
`entrypoints/serve/server.py` builds `engine/resident_client.py` `Resident` and
`entrypoints/serve/http_handler.py`; browser turns go through
`entrypoints/ui/conversations.py`, `/v1/chat/completions` through
`entrypoints/openai/serving_chat.py` (with `chat_utils.py` and `tool_parser.py`),
and both are queued by `entrypoints/serve/job_queue.py`, which publishes each job
into the session's inbox.

### Where to look

| Task | Start here |
|---|---|
| Add or change a CLI subcommand | `glm_tpu/entrypoints/cli/main.py`, a `CLISubcommand` of `types.py`; tests in `tests/entrypoints/cli/` (help and argument snapshots in `test_main.py`) |
| A site setting or environment variable | `glm_tpu/config/site.py` and `examples/site.example.toml`; `glm_tpu/envs.py` |
| Request preparation, capacities, the template | `glm_tpu/engine/request.py`, `glm_tpu/config/model.py` |
| The `/v1` API or the chat UI | `glm_tpu/entrypoints/openai/`, `glm_tpu/entrypoints/serve/`, `glm_tpu/entrypoints/ui/` ([API](docs/API.md), [UI](docs/UI.md)) |
| Launch, staging, SSH, locks and cleanup | `glm_tpu/executor/` ([OPERATIONS](docs/release/OPERATIONS.md)); helper texts are recorded byte for byte (G9) |
| Model numerics, a layer or a kernel | `glm_tpu/models/glm_moe_dsa/`, `glm_tpu/layers/`, `glm_tpu/kernels/`; any change here moves the frozen G1-G3 records and needs the owner's baseline decision first |
| Compilation, memory or collective admission | `glm_tpu/runner/` |
| The checkpoint format, inventory or verify | `glm_tpu/model_loader/` ([CHECKPOINTS](docs/release/CHECKPOINTS.md)) |
| Mesh, topology, fleet votes | `glm_tpu/distributed/` |
| Why a gate fails | `python -m tools.equivalence diff`, [the harness README](tools/equivalence/README.md) |
| Measured results and their limits | [STATUS](docs/release/STATUS.md), [PROJECT_SUMMARY](docs/release/PROJECT_SUMMARY.md) |

# Historical GLM-5.2 sampled inference

This document preserves the legacy sampled/long-context interface and evidence.
Its GLM-5.2 weight payloads were retired. These commands and historical results
do not qualify GLM-5.3. Use [ordinary GLM-5.3 inference](OPTIMIZED_INFERENCE.md)
for the current release; full32K inputs and larger profiles remain untested.

This document describes the retained sampled long-context entry. The ordinary
greedy candidate has [separate instructions](OPTIMIZED_INFERENCE.md) and a fixed
8K capacity; these profiles do not share performance or capacity admission.

The native resident runtime accepts user prompts, not only benchmark questions.
The release adds a separate user-request format and executor so a user response
cannot be mislabeled as a GPQA/AIME result. The outer controller and transport
and original-evidence/DB/archive sealing are implemented and CPU-tested.
The protected ordinary-response smoke passed as **DB621** at release execution
commit ca545aa4a47b5c00ecde179623f9e094bddfdb22. See [the sealed receipt](user-response-db621-sealed-20260914.json)
and [validation scope](STATUS.md). Never launch alongside another active workload.

Each controller invocation starts fresh worker processes, cold-loads/prepares the
model, executes one request and exits. It is not a persistent interactive server:
DB621 cold loading/compilation took about 38 minutes. Do not interpret short
decode latency as the startup time of this command. The runtime Python interface
and same-live-session continuation are distinct from a resident multi-request service.

## Prepare a prompt locally

Create a JSON file **outside Git** containing text-only messages, for example:

```json
[{"role":"user","content":"Explain why the sky is blue in one sentence."}]
```

Then use the existing local tokenizer, without downloading weights:

```bash
JAX_PLATFORMS=cpu python -m glm_tpu prepare-request \
  --profile legacy-sampled \
  --repo /home/gianl/glm-tpu-release \
  --tokenizer-root /home/gianl/gcs-models/models/GLM-5.2-FP8 \
  --messages /path/outside/git/messages.json \
  --output /path/outside/git/request.json \
  --request-id example-001 --seed 42 --max-new-tokens 256
```

Replace the input/output paths with existing local parent directories. Output is
created exclusively with owner-only permissions; an existing file, symlink or
output within the specified source checkout is refused. Prompt IDs are private
data too. The printed report contains hashes and counts, not prompt contents.

The command verifies the two tokenizer files and the model chat template against
the frozen source hashes. It uses thinking ON / effort Max, temperature1.0 and
top-p0.95. It performs **zero model executions**. Max-new-tokens includes reasoning
and the final answer: a short cap can end during reasoning, not produce a final
answer. No setting silently shortens the user's prompt or generation allowance.

The currently resident sampled graph has capacity166,912 tokens, shared by prompt
and generation, with at most163,840 generated tokens. This does not expose the
separately measured256K E0 graph as a256K sampled user service. Changing capacity
or sampling parameters requires appropriate graph/memory validation.

## Worker API

The protected worker can invoke
`scripts.release.ws32_user_request.execute_user_request` with:

- `loaded`: the real result of the existing native `load_runtime`, including
  admitted compiled programs and original memory authorization;
- `request`: the validated prepared JSON;
- a fresh `RequestStore`, the pinned tokenizer and `NativeObservability`;
- an operational deadline, while the outer controller holds both leases.

This reuses `NativeBenchmarkRuntime.start_request`, immediate local token delivery,
the same live-session decode loop, and terminal cache release. The historical
class name does not force benchmark scoring. The new executor never loads the
benchmark registry, accepts benchmark gold answers, or reports a quality score.
It preserves raw token output, answer text, original memory boundaries, observation
metadata, cold loading time, prefill phases and per-step decode durations. Trace-
instrumented samples remain explicitly identified; do not average them into a
profiler-free performance claim. Local JSONL flush is not network delivery.

Failure keeps the already-delivered prefix and a failure record. There is no
automatic regeneration, cache replay or retry after ambiguous delivery failure.
Pause/resume remains in-process only. Concurrency and HTTP serving are not added.

## Validation and deployment boundary

CPU tests exercise the **actual native host request loop with fake compiled math**:
fresh cache/memory boundaries, prefill then decode, terminal release, refusal and
partial-output preservation, without invoking benchmark validators or scoring.
An opt-in test also uses the real pinned local tokenizer through the CLI:

```bash
GLM_RELEASE_LOCAL_TOKENIZER=/home/gianl/gcs-models/models/GLM-5.2-FP8 \
JAX_PLATFORMS=cpu python -m pytest -q tests/release/test_user_request.py
```

These checks are not TPU execution or performance evidence for the new executor.
The default-off `scripts/release/ws32_user_worker.py` connects that executor
to the original topology initialization and `load_runtime`. Its fixed site
arguments match the benchmark loader's retained recipe; prompt identity and
one-sequence policy are separate from the benchmark registration. The existing
owner script dispatches `--user-request` to this entry so the process observer
still recognizes its original script name. It requires a fresh owner-only run
directory, owner-only request file, exact file hash, source/model guards and
pinned tokenizer/template bytes before initializing the backend. Unknown request
fields, replayed directories and changed sampling/capacity are refusals.

User cold evidence now receives the same original byte-budget enforcement, but
benchmark publication defaults still reject the user namespace. The explicit
user transport policy uses a separate schema and allows only item000, not the
228-item benchmark. No benchmark manifest
or quality score is reused to seal an ordinary response. CPU tests cover entry
wiring through the actual host executor, loader failure, privacy/path checks,
default-off process dispatch and unchanged benchmark transport. They do not
prove fresh compiler/HBM admission for this new entry stack on actual TPU.

The default-off `scripts/release/launch_ws32_user_request.py` now owns both leases,
the original clean/published source and idle-fleet checks,6GiB launch floors,
regional/live-storage admission and immutable private request transport. Its
worker supervisor uses the original observer-recognized script name. SSH dispatch
ambiguity enters same-tag observation without redispatch; an unknown child wait
never fabricates an ended marker. Attach refuses new input/deadline overrides.
Publication retains partial requests even if the independent cold channel fails.
Collectors use original generations, CRCs, inflated hashes and owner/request
binding, with owner-only local output and one shared physical HLO copy.

Inspect arguments without launching anything:

```bash
JAX_PLATFORMS=cpu python -m scripts.release.launch_ws32_user_request --help
```

The enabling variable is `GLM_GREENFIELD_USER_REQUEST=1`. Do not enable it before
deployment review, source cutover and the owner-cancelled benchmark's authenticated
worker/publication termination, partial-original preservation and idle cleanup.
Full benchmark completion or a success seal is not required.
Controller inputs are `--tag`, `--code-hash`, `--reviewed-branch`, `--request`
(a prepared private file) and `--wall-seconds` (1..86400, default3600, including
cold load/compile). This deadline does not shorten the request's token budget:
expiry leaves an incomplete response. `--attach` recovers the original tag/pin
and must not specify a different request, transport or deadline. Worker roles
are internal controller plumbing, not alternative ways to bypass admission.

### Run the prepared request on the supported installation

After release admission and the leased source cutover in [OPERATIONS](OPERATIONS.md),
run on **worker0** from the canonical checkout. The selected published branch and
clean checkout must identify the same reviewed commit. Use `main` only after the
release has actually merged; candidate validation uses its explicit `release/...`
branch instead. This command does not provision hardware or reconstruct weights.

```bash
cd /home/gianl/glm-tpu-topology-rewrite
GLM_RELEASE_REQUEST_TAG="greenfield_ws32_user_request_$(date -u +%Y%m%dT%H%M%S%NZ)"
GLM_GREENFIELD_USER_REQUEST=1 JAX_PLATFORMS=cpu \
  /home/gianl/vllm-env/bin/python -u -m scripts.release.launch_ws32_user_request \
  --tag "$GLM_RELEASE_REQUEST_TAG" \
  --code-hash "$(git rev-parse HEAD)" \
  --reviewed-branch main \
  --request /path/outside/git/request.json \
  --wall-seconds 3600
```

Use the private request produced by `prepare-request` above. The controller stays
CPU-only; it explicitly launches the protected TPU worker roles. Run just one
controller, retain its tag, and never repeat this command to repair an interrupted
upload. Follow the same-tag recovery instructions below instead.

Live rank0 output is under `/home/gianl/glm-run/<tag>/sessions.rank0/item000/`:
`tokens.jsonl` contains emitted token records; `answer.txt.gz` is written when the
request reaches its stop policy. These files are private, not Git artifacts.
An output file alone does not imply success: the controller exits successfully
only after original-result checks, DB linkage, regional archive and response seal.
Report token-cap termination honestly, even when reasoning has not produced a
final answer. Cold loading/compilation is separate from the capped generation.

Collection alone has `protected_result_sealed=false`. The outer now replays
original evidence and publishes `USER_RESPONSE_SEALED.json` only after DB linkage
and regional generation/CRC/SHA readback. This is one completed response under its
stop policy, not proof of task quality or authorization to merge main. User
inputs/responses are private runtime objects outside Git in the approved bucket;
they are not automatically deleted by these tools. Every new launch enforces
the storage cap and existing disabled soft-delete policy without changing it.

`scripts/release/ws32_user_result.py` replays all eight collected user responses:
strict input/source identities, raw token stop policy and rank agreement, answer
decoding, original memory/DSA/cache witnesses, trace byte identity and phase timings.
It excludes the first instrumented decode from ordinary decode statistics and
returns no rate when no ordinary samples exist. A one-token terminal response
has no decode/trace witness; it cannot establish fresh decode hardware admission.
CPU tests run the actual host executor and observability with fake math/counters;
they also reject corrupted originals despite a saved passing status. Fake trace
bytes are explicitly not physical XPlane proof. The replay returns no quality
score, protected seal, cold admission or authenticated-worker claim on its own.

The outer `ws32_user_evidence.py` binds the same source and tokenizer pins, cold
checkpoint/HLO/memory replay, authenticated worker PID/start/boot/argv, successful
worker exit and cleanup, and eight single-step XPlanes to their original hosts.
Single-step cycle/idle metrics remain unset. `ws32_user_database.py` stores this
run's hashes and timings without benchmark scores or extra raw prompt copies.
`ws32_user_archive.py` rechecks original generations, exports only this run's DB
rows and publishes the bounded regional ledger and response seal. Both leases
remain held throughout. An interrupted archive reuses the same originals/rows;
it never retries generation. Inputs/answers remain private and outside Git.

Actual site/branch/asset admission and the bounded ordinary user request are now
proved by DB621; final branch/mirror promotion is tracked in STATUS.md. Never route
a user request through the original228-item benchmark seal or weaken its registration.

## Recover a failed upload without repeating the response

An upload failure is not permission to regenerate a response. After diagnosing
the cause, a successfully ended original user run may use the same tag, code pin
and reviewed branch with `--attach --republish-originals`. Do not supply a new
request, deadline or transport identity. The source must still be the original
clean, published pin; changed-code recovery needs a separate reviewed procedure.

This explicit option waits for original ownership and idle-fleet confirmation,
holds both leases, and invokes **only the publish role on failed ranks**. It
requires successful original worker exits on all eight ranks. Conditional object
creation and generation/hash checks reuse existing bytes; changed originals are
refused. The original failed `published.rankN.json` markers remain unchanged.
A separate `publication_recovery.json` preserves those failures and the new
upload receipts, and is included in the bounded regional archive.

Collection, cold/request/trace replay, DB linkage and sealing still run normally;
the recovery receipt is not a model-quality or execution pass. Ambiguous upload
SSH failure preserves any uploaded objects and does not dispatch model workers.
A later explicit attach may repeat the immutable uploads. Missing worker exit
markers, a failed model process, changed owners or altered originals require
diagnosis; this option does not invent terminal state or repair model failures.
The recovery flow has CPU/fake-SSH coverage, not a live fault-injection result.

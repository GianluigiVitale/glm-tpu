# User inference integration

The native resident runtime accepts user prompts, not only benchmark questions.
The release adds a separate user-request format and executor so a user response
cannot be mislabeled as a GPQA/AIME result. The outer protected user launch and
publication integration is **not yet complete**. Do not use preparation success
as permission to launch alongside the active benchmark.

## Prepare a prompt locally

Create a JSON file **outside Git** containing text-only messages, for example:

```json
[{"role":"user","content":"Explain why the sky is blue in one sentence."}]
```

Then use the existing local tokenizer, without downloading weights:

```bash
JAX_PLATFORMS=cpu python -m glm_tpu prepare-request \
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

## Validation and remaining deployment work

CPU tests exercise the **actual native host request loop with fake compiled math**:
fresh cache/memory boundaries, prefill then decode, terminal release, refusal and
partial-output preservation, without invoking benchmark validators or scoring.
An opt-in test also uses the real pinned local tokenizer through the CLI:

```bash
GLM_RELEASE_LOCAL_TOKENIZER=/home/gianl/gcs-models/models/GLM-5.2-FP8 \
JAX_PLATFORMS=cpu python -m pytest -q tests/release/test_user_request.py
```

These checks are not TPU execution or performance evidence for the new executor.
Remaining: connect a reviewed user controller/worker and evidence publication,
prove its site/branch/asset admission, then run the smallest necessary real user
request after the current benchmark terminates and seals. Never route a user
request through the original228-item benchmark seal or weaken its registration.

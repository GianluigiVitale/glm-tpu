# GLM-5.3 ordinary inference

Use a clean, published full checkout on authenticated rank0 of the existing
8host/32chip site, with the [pinned environment](INSTALLATION.md) and verified
[checkpoint assets](CHECKPOINTS.md). The recommended command is:

```bash
JAX_PLATFORMS=cpu python -m glm_tpu ask --questions /private/questions.json \
  --context 32k --concurrent --wall-seconds 14400
```

The input is a private JSON array of one to four question strings. One model load
serves the fixed batch, with sequential prompt prefill and concurrent decoding.
Each conversation has32768combined prompt/history/reasoning/output slots.
Thinking is enabled/max. Omitting --max-new-tokens grants all remaining slots;
there is no1024/2048default or separate thinking cap. An explicit shorter cap is
still honored if requested. Input is not silently truncated. Context exhaustion,
timeout or failure is incomplete, even if a correct number appeared in reasoning.

Four fixed GSM8K questions completed correctly at normal EOS in the actual
[acceptance run](glm53-four-answers-20260921.json), with prompt sizes100,62,85,70.
The unchanged prepared request was executed with:

```bash
JAX_PLATFORMS=cpu python -m scripts.release.launch_ws32_optimized_request \
  --request /private/prepared/request.json --wall-seconds 14400
```

`ask --prepare-only` creates private inputs without launching hardware; the same
preparation and controller code compose the recommended command. Do not feed
reference answers into prepared requests. Prepared files bind model/template
identities and cannot be substituted with old GLM-5.2 requests.

The controller acquires both workload leases and sync locks, authenticates an
idle fleet, stages exact published source, and checks all8environments. It loads
and compiles, warms disposable state, and starts each measured conversation fresh.
It refuses dirty/unpublished source. A live invocation must not be duplicated.
Failures preserve partial evidence; authenticated cleanup precedes a repaired retry.

The private run directory contains summary.json, runner.rankN.json and separate
itemNNN/tokens.jsonl and answer.txt files. Token streams are written locally during
generation; ask prints decoded responses after cleanup. Raw answer files include
reasoning and model control markers; final-answer checks inspect the portion after
</think> and require normal stopping. These files stay outside Git/reviewer packages.

[Results](STATUS.md) separate cold startup, prefill, active decode and aggregate
throughput. There is no persistent server, network streaming endpoint, online
batch admission or durable cache recovery. Maximum effort does not guarantee a
correct answer. Four allocated32K caches with short inputs do not prove long-input
quality.

Other retained interfaces include sequential queues of up to10,8K combined slots,
and a128K-input/166912-total-slot profile. The latter has a163840output ceiling.
These are not current GLM-5.3 hardware acceptance claims. Explicitly use the32K
concurrent command above. [Legacy sampling](INFERENCE.md) remains5.2history;
old weights were retired. Prior instructions are recoverable at
`git show glm-5.2:docs/release/OPTIMIZED_INFERENCE.md`.

## Optional resident ordinary session

`ask` now defaults to a 32K combined input/history/thinking/output capacity.
`--keep-loaded` retains the ordinary single-chat runtime, compiled graphs and both
workload leases after answering. It does not apply to concurrent batches. The
existing release measurements above predate this option; solo speed requires
its own real-weight measurement. Successful retained sessions deliberately do not
claim cleanup: `resident-measurement.json` records `all_hosts_idle_after=false`.

For subsequent questions, prepare a same-capacity ordinary request with
`ask ... --prepare-only`, then atomically place that private `request.json` into
the printed run's `inbox/0001.json`, then `0002.json`, etc. Only this controller
owns the fleet; do not launch another `ask` workload. Each subsequent request
gets fresh conversation state using the same loaded weights and compiled graphs.
Include the full history in prepared chat messages when continuing a conversation.
Outputs and receipts are in `resident-0001/`, etc. Files must be owner-only.

To stop intentionally, atomically write `{"stop":true}` to `inbox/stop.json`.
The controller then verifies eight-host cleanup and releases the leases. Idle
residency has no automatic timeout; the inference deadline still applies to each
active request, and worker/controller failures retain the existing authenticated
failure cleanup. There is no HTTP endpoint or restart recovery of live model state.

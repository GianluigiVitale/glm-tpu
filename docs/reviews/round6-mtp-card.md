# Round-6 adversarial review — MTP M0 skeleton (fork 183f18ce) + card-protocol bench (glm-tpu cdbf0db)

**Reviewer:** adversarial, read-only (report write only). **Date:** 2026-07-07 (second pass).
**Method:** every claim re-derived against the installed vLLM (`~/vllm-build`,
0.1.dev1+ga30addc75.tpu), the fork at 183f18ce, `reference/hf-repo/*`,
`configs/glm-5.2-fp8-keyset.json`, and — new this pass — the REAL staged checkpoint metadata in
`gs://driftbench-dsv4-uc/models/GLM-5.2-FP8/` (index json + chat template, read-only download).
All executions CPU-only (`JAX_PLATFORMS=cpu`); the M0 self-check, the bench suites (at cdbf0db in
a scratch worktree AND at the current working tree), and a card-mode `--stub` run were re-run
independently. No TPU touched; `results.db` never written.

**Numbering note:** this pass independently re-derived and CONFIRMS all ten findings (F1–F10) of
the earlier round-6 draft at this path; their numbers are preserved because uncommitted
working-tree fixes already cite "round-6 F1/F10". F11–F15 are new to this pass.

**Working-tree status (matters for F1/F10):** `bench/run_bench.py` + `bench/test_bench.py` carry
UNCOMMITTED fixes responding to F1/F10 (`_per_request_seed_support` platform probe, seed omission
+ `items.seed=NULL` recording, `seed_passthrough` note, `test_seed_provenance_gating`). I reviewed
them (below, under F1/F10) — they are correct as far as CPU can verify, and `pytest bench/` on the
working tree passes 25/25. They are not yet committed; F1 remains live in every committed state.
The fork tree likewise carries uncommitted M1 work (`draft_load_filter.py`, eagle3/wrapper/loader
edits) — out of scope here; only 183f18ce was reviewed.

---

## Findings (most severe first)

### F1 — HIGH — at cdbf0db the card protocol CANNOT run on the pod: the TPU platform rejects every seeded request
`bench/run_bench.py` (cdbf0db `_sp`, `seed=seed` unconditionally; card always sends an int seed —
`sample_seed = seed + s`, CLI default `--seed 0`) / fork `tpu_inference/platforms/tpu_platform.py:418-428`.

With `temperature=1.0` + `seed is not None`, vLLM classifies the request `SamplingType.RANDOM_SEED`
and the fork's `TpuPlatform.validate_request` raises `ValueError("JAX does not support per-request
seed.")`, called per request from `vllm/v1/engine/input_processor.py:296` (driver side, before any
TPU work). **Re-proven live this pass** on this stack (CPU, platform resolves to `TpuPlatform`):
`SamplingParams(temperature=1.0, top_p=0.95, seed=0).sampling_type == RANDOM_SEED` →
`validate_request` raises; the same params with `seed=None` pass; `temperature=0.0 + seed=0`
classifies GREEDY and passes (so greedy runs were never at risk).

**Failure scenario:** any pod run of docs/07 §6's own commands (`--protocol card --seed 0`) builds
the engine (~45 min weight streaming) then dies on the FIRST `llm.generate`. The committed CPU
tests could not catch it: they assert "seeds reach the generator", never engine-side
`SamplingParams` validation.

**Also (same root):** even if accepted, the TPU sampler has no per-request RNG — one global key
chain from `model_config.seed` (`layers/jax/sample/sampling.py` takes a single `rng`) — so the
per-row `seed` column at cdbf0db implies (prompt, seed) reproducibility this backend cannot
deliver.

**Fix assessment (uncommitted working tree):** `_per_request_seed_support` probes
`current_platform.validate_request` once with a representative RANDOM_SEED request, fails safe on
any exception, omits `SamplingParams.seed` when unsupported, records `items.seed=NULL`, and stamps
`seed_passthrough=off(...)` into the summary note; `test_seed_provenance_gating` covers the
off/on/greedy matrix. Correct by inspection + tests; end-to-end acceptance of seedless card
requests is pod-verifiable only. **Needs committing.**

### F2 — MEDIUM — §2.3/§4 "rows stay request-aligned across steps" is unsubstantiated; the M4 carriage design leaves the row-gather unspecified
`docs/08-mtp-design.md` §2.3/§4; `glm_mtp/index_share.py` (docstring contract).

Re-derived independently this pass, same conclusion: the GPU "reference semantics" the doc claims
to replicate write the `topk_indices_buffer` **per flattened token** in the draft's first pass
(which in spec-decode steady state carries `accepted_r+1 ≥ 1` tokens per request —
`llm_base_proposer.py:494-520`), while steps 1+ read it **positionally**:
`topk_indices = self.topk_indices_buffer[:num_actual_toks]`
(`flashmla_sparse.py:863`, identical in xpu/rocm/flashinfer variants) with `num_actual_toks =
batch_size` (`llm_base_proposer.py:614`). Whenever any request contributed >1 token to step 0,
buffer row r at step 1 is the r-th flattened step-0 token, NOT request r's last token — so
"decode = 1 token/request/step ⇒ request-aligned" does not describe the pass that writes the
reused rows; the GPU reference itself is at best approximately aligned there.

TPU consequence: the step-0 stash is `[T0_padded, 2048]` per-token; §4.4 carries
"`[num_reqs, 2048]`" into the loop — SOMEONE must gather rows at `last_token_indices`, and neither
§4 nor `index_share.py` says who. Done naively (pass the stash straight back) it either shape-fails
loudly or silently misaligns when the padded bucket coincides. Because rejection sampling masks
draft errors, the blast radius is **acceptance/throughput, not output correctness** — but that is
exactly what gate M4(c) measures, so an unnoticed misalignment would read as "index-share costs
acceptance". Add to M4 gate (a): a parity case with *unequal accepted counts across requests*, and
an explicit decision (with GPU-side confirmation) whether the reused set should be the per-request
last-token rows. Also still open: reused sets never contain p+1..p+4, so under a pure top-k mask
draft tokens cannot attend to each other/themselves unless the kernel force-includes the current
token — unaddressed in §4.

### F3 — LOW — M0 self-check [d] is vacuous for `embed_tokens`
`glm_mtp/draft_config.py:232-238` (`EXPECTED_SHARED_ONLY_PARAMS` at :153).

For the template `model.embed_tokens.weight`, `p.split("model.layers.78.", 1)[-1]` doesn't split →
suffix = the whole string → can never match a `model.layers.78.*` key (`endswith` fails on the
prefix mismatch). A hypothetical layer-78 `embed_tokens` in the keyset would pass [d] silently
(and [c] classifies it happily as `direct → model.embed_tokens.weight`). The `shared_head.head`
half works. Real impact bounded — the committed keyset is sha-pinned and (new this pass) verified
against the actual checkpoint index (see Confirmed #4) — but half of check [d] can never fire.

### F4 — LOW — card fallback extraction scores answers out of truncated, unclosed `<think>` reasoning
`bench/extract.py:20-29` (`strip_think`) → `:134-166` / `benchmarks.py:82-94`.

Re-demonstrated: `"<think>draft: Exact Answer: 5 maybe"` (no `</think>` anywhere — the shape of a
cap-truncated reply) → `extract_exact_answer` returns `"5 maybe"` → `norm_math` strips the alpha
tail → `"5"` — an answer scored out of mid-reasoning draft text. Card runs at temperature 1.0
under a binding pod cap (docs/07 §6 suggests `--max-len 65536` vs the 163,840 card cap) WILL
truncate. Rows are flagged `truncated`, and "scored as-is" is documented — but this specific mode
contradicts the extractor chain's "never guess" framing. Suggest suppressing Exact-Answer
extraction when `finish_reason=='length'` and `<think>` never closes, or naming the mode in
docs/07 §4.

### F5 — LOW — `drop_confidence_lines` misses mid-line "Confidence:", so the fallback still grabs the percentage
`bench/extract.py:131` (`_CONFIDENCE_LINE` is `^`-anchored per line).

Re-demonstrated live: nonconforming single-line reply
`"After working through it, the result is 128. Confidence: 95%"` → `card_math_extract` returns
`"95"` — the exact trap the function exists to defuse, evaded by inline placement. Line-formatted
confidence (what the system prompt requests) is covered; nonconforming replies are precisely the
fallback's domain. False-positive-capable only when gold happens to equal the percentage.

### F6 — LOW — last-marker-wins backfires on template echoes after the answer
`bench/extract.py:150-153`.

Re-demonstrated: `"Exact Answer: 42\nConfidence: 90%\nRemember the format was 'Exact Answer:
{your succinct, final answer}'"` → extracts `"{your succinct, final answer}'"` (non-None → no
fallback) → scored wrong although a conforming answer exists earlier. Plausible at temperature
1.0. Consistent with never-guess (raw stored, `rescore.py` can re-score later), but a
placeholder-shaped field (`{your ...}`) could be skipped cheaply.

### F7 — LOW (latent) — `mtp_shared_param_map` drops all but the last MTP layer's head share when `num_mtp_layers > 1`
`glm_mtp/shared_weights.py` @183f18ce (`mapping["vllm_model.lm_head.weight"] = ...` in a loop).

At 183f18ce the map is TARGET-keyed, so per-layer writes collide on the one `lm_head` key: with
`num_mtp_layers=2` only `layers.79`'s head is mapped and layer 78's `shared_head.head` stays
unshared — the NaN-logits case the module's own docstring warns about. GLM-5.2
(`num_nextn_predict_layers=1`) unaffected. **The uncommitted working-tree revision already inverts
the orientation to draft-keyed (`{draft_name: target_key}`), which fixes the collision and matches
what the wrapper's matcher consumes** (`vllm_model_wrapper.py:562-576` looks draft names up in the
shared dict) — confirm that lands with M1.

### F8 — INFO — G1's "raises before any load" is wrong: the impl-equality failure fires AFTER a full draft load
`docs/08-mtp-design.md` §5 G1; fork `spec_decode/jax/eagle3.py` @183f18ce (`get_model` at ~:83,
equality raise at ~:107-110), `models/common/model_loader.py:645+`.

Verified order this pass: `Eagle3Proposer.load_model` calls `get_model(...)` FIRST; unfixed, the
draft's `resolve_model_architecture` returns `flax_nnx` (DeepSeekMTPModel not in the JAX registry
→ `UnsupportedArchitectureError` → default), `get_model` falls back to the vLLM path and
**streams the full draft load**, and only then the equality check raises (its own `resolve` call
still says `flax_nnx` ≠ target `vllm`). So the unfixed failure costs a full checkpoint pass, not
"before any load". The diagnosis and the one-line fix (add `"DeepSeekMTPModel"` to
`_VLLM_PREFERRED_ARCHITECTURES`, `model_loader.py:53`) are otherwise verified correct.

### F9 — INFO — M0 contracts have no test wiring; the vLLM cross-check covers only the name rewrite
`glm_mtp/draft_config.py` (`__main__`-only self-check at 183f18ce; no callers/tests elsewhere).

Self-check [e] cross-verifies `_rewrite_spec_layer_name` only; `_STACKED_PARAMS`/expert routing
transcribe function-local literals in `DeepSeekMTP.load_weights` (not introspectable) — a silent
vLLM change to e.g. the `wk_weights_proj` fusion or the `fused_qkv_a_proj` params_dict fallback
would surface only at M1's load audit. Two scope gaps, both verified non-diverging on all 39 key
classes of THIS keyset: the real loader also accepts a bare `layers.{i}.` prefix
(`get_spec_layer_idx_from_weight_name`, `deepseek_v2.py:1756`), and matches stacked names by raw
substring (`"wk" in name`) where the skeleton requires a `.wk.` segment. (The working tree has
since added `tests/models/vllm/test_glm_mtp_stage3.py` — uncommitted, not reviewed.)

### F10 — INFO — provenance nits (bench)
- At cdbf0db, greedy rows record `seed=0` (the unused `--seed` default) while `SamplingParams` got
  no seed — the column meant different things across protocols. **Fixed in the uncommitted
  working tree** (`record_seed`; greedy and TPU-card rows record NULL) with tests. Needs committing.
- `CARD_FOOTNOTE_REASONING` has no dedicated byte-pin test (only the system prompt is pinned, and
  to ANY backticked occurrence in the README rather than footnote 1 specifically). Verified
  byte-exact today (incl. the `evaluate  each` double space); pin it so it stays so.
- docs/08 §9's "39/39 layer-78 keyset keys" = 39 key **classes** (256 experts templated as
  `mlp.experts.N.*`); fine for the keyset as written, but the commit message reads stronger.

### F11 — NEW, LOW — the indexer-schedule "mirrors vLLM" claim is mechanically wrong (result-identical on this config)
`glm_mtp/draft_config.py:65-82` + docs/08 §1.2 vs `vllm-build/vllm/model_executor/models/deepseek_v2.py:1022-1034`.

Upstream vLLM **never reads `indexer_types`** (zero occurrences in the installed tree): its
per-layer `_skip_topk` uses `index_topk_pattern` when present, else the offset/freq formula — for
ALL layers, not as a beyond-range fall-through. The skeleton (and the fork's `_layer_is_shared`)
prioritize `indexer_types` first. On the real config this is harmless — `index_topk_pattern` is
null and `indexer_types` agrees with the formula on all 78 layers (0 mismatches, verified
programmatically) — but docs/08 §1.2's "indexer_types len 78 → layer 78 out of range → falls to
the formula" describes the FORK's derivation, not vLLM's, and the docstring's "Mirrors vLLM
deepseek_v2.py's per-layer `_skip_topk` derivation" is only accidentally true here. Two latent
divergences if configs ever change: (a) pattern present + layer beyond range → vLLM leaves
`_skip_topk=False` (full) by initialization, skeleton applies the formula; (b) list-form pattern
entries — vLLM tests `== "S"` (so a literal `"shared"` entry counts as FULL), skeleton tests
`not in ("S", "shared")`. Suggest a comment + an assert that `indexer_types`, pattern, and the
formula agree wherever more than one is defined.

### F12 — NEW, INFO — §4.5's "compile 2 draft programs per bucket instead of 5" mislabels the compile unit
`docs/08-mtp-design.md` §4 item 5 vs `spec_decode/jax/eagle3.py` @183f18ce (`_propose` is
`@jax.jit` with static `num_speculative_tokens`; `compilation_manager.py` precompiles
`drafter.propose` per bucket).

The compiled unit is the OUTER `_propose` — one executable per bucket — with the k
`model_fn(spec_step_idx=…)` calls traced inline (nested jit inlines at lowering). Static
`spec_step_idx ∈ {0..4}` therefore produces 5 inner traces embedded in ONE program, not 5 draft
programs; collapsing to `is_first_spec_step` cuts trace/lowering work and program size (a real
warmup saving), not program count. Directionally the optimization stands; the stated unit doesn't.

### F13 — NEW, LOW — Exact-Answer cut artifact: inline parenthesized confidence leaves junk; junk fields preempt a recoverable fallback
`bench/extract.py:153-166`.

Demonstrated this pass: (a) `"Exact Answer: 42 (Confidence: 90%)"` → the Confidence cut keeps the
opening paren → `"42 ("`, `norm_math` → `"42("` ≠ `"42"` → false negative (the end-strip regex
covers whitespace/`*`/`_`/backticks but not brackets); (b) `"Exact Answer: (see below)\n\boxed{42}\n
Confidence: 90%"` → field reduces to `"(see below)"` (non-None) → the fallback that WOULD find
`\boxed{42}` never runs. Both are under-crediting only (never credit a wrong answer) and re-scorable
from stored raw output; cheap hardening: also strip trailing `([` orphans, and treat a field that is
pure punctuation/placeholder as None so the fallback chain runs.

### F14 — NEW, INFO — commit-message test-count overclaim at cdbf0db
Commit cdbf0db says "pytest bench/ 22 passed"; a clean worktree at cdbf0db collects and passes
**20** (re-run this pass: `20 passed in 0.74s`). Not a code defect — but the repo's own standard is
that stated numbers be reproducible. (Working tree today: 25 passed, including the new
seed-gating/F1 tests.)

### F15 — NEW, INFO — GPQA "no judge" labeling is an interpretation stated as fact
`bench/benchmarks.py:169-180` (GPQA `substitutes` text) vs `reference/hf-repo/README.md:80`.

The GPQA substitutes string asserts the card "says nothing about GPQA's … answer extraction, or a
judge". The README's judge sentence ("We use GPT-5.5 (medium) as the judge model.") sits inside the
one bullet that covers "HLE & other reasoning tasks" — its scope (all reasoning tasks vs the
preceding AIME/HMMT/IMOAnswerBench sentence) is genuinely ambiguous, and docs/07 §1 says so
("most plausibly …") — but the provenance string recorded into every card run's `env_json` states
the stronger reading as fact. One clause ("judge-sentence scope ambiguous; we read it as scoped to
the free-form tasks") would make the stored provenance as honest as the doc.

---

## Confirmed (positive assurance — claims attacked and NOT broken)

**M0 / design doc (fork 183f18ce + docs/08):**
1. **Config surgery (§2.1) is real and exactly as described:** installed
   `vllm/config/speculative.py:301-313` maps `glm_moe_dsa → deepseek_mtp →
   architectures=["DeepSeekMTPModel"], n_predict=num_nextn_predict_layers`; with `method="mtp"` +
   no model, draft path = target checkpoint and quantization is inherited (`speculative.py:556-569`);
   `use_eagle()` includes `"mtp"` (`:1070-1071`). Self-check [a] reproduces it on the real
   `config.json` (`DeepSeekMTPModel`, n_predict=1).
2. **M0 self-check independently re-run:** exit 0 — [a] OK; [b] layer-78 FULL; [c] 39/39 classes
   (stacked=11 / expert=6 / direct=22, hand-audited: q_a/kv_a/wk/weights_proj/shared-expert
   gate+up stacked; experts templated; the rest direct incl. `k_norm.bias`, which exists —
   `Indexer.k_norm = LayerNorm` — and `e_score_correction_bias`); [e] rewrite transcription
   bit-identical to installed vLLM's `_rewrite_spec_layer_name` on all 39 keys.
3. **Indexer schedule is checkpoint-true:** formula ≡ `indexer_types` on all 78 layers (0
   mismatches), `index_topk_pattern` is null, layer 78 → FULL, matching the keyset's indexer
   weights on 78 (wq_b+scale, wk+scale, weights_proj, k_norm w+b).
4. **NEW — the share map is verified against the REAL checkpoint, not just the committed keyset:**
   downloaded `model.safetensors.index.json` from the staged GCS copy — 118,629 weight-map keys;
   sha256 of the sorted keyset == the committed `keyset_sha256`
   (`a5a5b3de…b29c`, byte-exact); layer-78 collapses to exactly the 39 committed templates;
   **no** `embed_tokens`/`shared_head.head` under layer 78; top level = exactly
   `{lm_head.weight, model.embed_tokens.weight, model.norm.weight}`. The loader skips everything
   outside `model.layers.78.` (`get_spec_layer_idx_from_weight_name` verified), so both shares are
   mandatory — the doc's central §1.1 claim is checkpoint-proven. (141 safetensors shards; the
   G3 "150 files" = the 150-object GCS dir incl. index/config; layer-78 keys live in just 3
   shards, so G3's file-filter mitigation is even cheaper than stated.)
5. **GPU reference semantics quoted accurately:** `_maybe_share_lm_head` (MTP → always share,
   explicit `shared_head.head` re-point, NaN comment), target→draft `topk_indices_buffer` sharing
   (module-level too), `set_skip_topk(False)` before / `(True)` after the step-0 pass,
   `_share_mtp_indices ← draft hf_config.index_share_for_mtp_iteration` — all present at the cited
   places in `llm_base_proposer.py`. Draft forward/compute_logits split, `eh_proj` plain
   `nn.Linear` (keyset: bf16, no scale — matches), embed-mask at position 0, `enorm⧺hnorm→eh_proj`
   ordering, ≥1-loaded-weight-per-MTP-layer validation — all verified in `deepseek_mtp.py`.
6. **"DSV4 MTP stub structurally absent" holds with one precision:** `_disable_ds_v4_mtp_buffer`
   keys on `DeepseekV4ForCausalLM` only and GLM has no `_mtp_hidden_buffer` anywhere (the DSV4
   buffer lives in `vllm/models/deepseek_v4/*/model.py`). The analogous GLM-side mutable buffer
   (`topk_indices_buffer`, written by vLLM's `Indexer.forward`/`SparseAttnIndexer`) is avoided
   OPERATIONALLY, not structurally: the legacy call is dead because `DISABLE_DSA_INDEXER` defaults
   True (`envs.py:44`) and `xla_ref` mode routes around it via JAX-array carriage
   (`glm_dsa_indexer.py:483-526`, stash key == `index_share.MTP_TOPK_STASH_KEY`, unseeded fetch
   raises loudly). Anyone flipping `TPU_DISABLE_DSA_INDEXER=0` on a sparse config re-enters the
   DSV4 failure class — the env's own comment says as much.
7. **Two-trace mechanics are jit-sound as specced** (modulo F2's row question and F12's unit nit):
   `spec_step_idx` is a static argname on `draft_step_fun` (`vllm_model_wrapper.py`, verified at
   183f18ce with the MTP kwargs branch and single-tensor output `hidden_prenorm=hidden_states`);
   the wrapper context is rebuilt per traced call; seed = traced input, emission = traced output;
   pytree structure is stable per trace-time mode.
8. **Runner/wrapper claims verified at the cited lines (183f18ce):** `Eagle3Proposer` for
   `method=="mtp"` (`tpu_runner.py:698`), MTP aux-hidden branch = `(hidden_states,)`
   (`speculative_decoding_manager.py:181-184`), `max_logits_per_req = k+1` (`tpu_runner.py:810`),
   `RejectionSampler` wiring (`tpu_runner.py:1686`), DP spec-stats aggregation
   (`dp_scheduler.py:997+`), draft `static_forward_context` merge (`vllm_model_wrapper.py:556`),
   shared-params exact-name match with **warn-and-skip on shape mismatch** (`:562-576` — G2's
   hard-error demand is justified), `shard_model_to_tpu` replicates only `_tensor_is_in_cpu`
   tensors (`process_weights/cleanup_sharding.py:61-91`) and `MODULE_TYPE_TO_SHARDING_FUNC`
   registers LoRA types only (`:200-208`) — the shared-view sharding-preservation argument holds
   statically; the `_logits_partition_spec` probe indeed reads `self.model.vllm_model.lm_head`
   and quietly assumes replicated when absent (`:972-994`) — V3 is a real, correctly-flagged risk.
   `Eagle3Proposer.load_model` today shares **embed only** (`eagle3.py:74-81`) — the head gap (G2)
   is real; the MTP `_select_inputs_for_loop_speculation` branch recurses on `residual[0]` and
   `constant_draft_positions` is Gemma4-only — §3's recursion/KV claims check out. HF
   `modeling_glm_moe_dsa.py` contains no MTP module — M1's composed-reference plan is necessary.

**Card protocol / bench (cdbf0db):**
9. **The card quote is verbatim and complete — no cherry-picking:** `CARD_FOOTNOTE_REASONING` is a
   byte-exact substring of `reference/hf-repo/README.md` (only the leading `* ` bullet marker
   excluded; the `evaluate  each` double space preserved), and it is the FULL first footnote —
   the text-only-subset sentence, the GPT-5.5-judge sentence, and the HLE-with-tools sentence are
   all included. It is genuinely the only footnote touching the reasoning rows (all other bullets
   cover coding/agentic benchmarks). The backtick pin test binds the exact backticked region.
   The trailing `.` inside the backticks is honestly carried into `CARD_REASONING_SYSTEM_PROMPT`.
10. **The chat-template claim is exactly right — and now checkpoint-pinned:**
    `reference/hf-repo/chat_template.jinja` is **byte-identical to the staged checkpoint's copy**
    (GCS, `cmp` clean) and `tokenizer_config.json` (both copies) has no overriding
    `chat_template` field. Rendering the template (jinja2, trim/lstrip like transformers) with the
    card system message yields byte-for-byte
    `[gMASK]<sop><|system|>Reasoning Effort: Max<|system|>{card prompt}<|user|>{q}<|assistant|><think>`
    and without it `…Max<|user|>{q}<|assistant|><think>` — the second `<|system|>` block is the
    template's own convention (its tools path emits one identically).
11. **Sampling plumbing (CPU-verifiable part):** card temperature/top_p/max_new reach
    `SamplingParams` per request (`_sp` override dict); effective cap = min(CLI `--max-new`,
    163840, window room) with binding caps surfaced (`finish_reason`/`n_truncated`); the TPU
    sampler does honor per-request temperature/top_p arrays (`layers/jax/sample/sampling.py:72-104`)
    — only seeds are the problem (F1); `EOS_IDS = [154820, 154827, 154829]` matches the
    checkpoint's `generation_config.json` exactly (which itself defaults temperature 1.0 /
    top_p 0.95 — consistent with the card).
12. **Refusal paths verified:** `--protocol card` refuses gsm8k/mmlu_pro at the CLI before any run
    and in `load_items`; unknown protocol strings raise; `--samples>1` under greedy refused at
    both layers; greedy card-silent GPQA keeps `build/extract/system_prompt = None` → greedy
    machinery unchanged (asserted by tests).
13. **Tests + stub:** worktree @cdbf0db: `pytest bench/` **20 passed**; working tree:
    **25 passed** (incl. the F1/F10 fix tests). Card-mode stub re-run in the scratch worktree
    (`--stub --limit 3 --samples 2 --seed 7`): 6 `#sN` rows, per-sample seeds 7/8 in the log,
    avg@N summary, pinned dataset revision resolved from the offline cache. Extractor probes
    beyond the shipped suite behaved correctly: nested boxed unwrapped, boxed-on-next-line,
    full-width colon, bolded markers, empty-last-marker rescued by the fallback (returns the
    earlier 128), "Confidence intervals" not treated as a cut, marker inside CLOSED think ignored.

---

## What only on-TPU runs can verify (CPU review cannot close these)

1. **F1 fix end-to-end:** seedless card `SamplingParams` accepted by the engine, and temp-1.0
   sampling actually varying across samples (global-RNG advance) — plus GPQA/AIME card runs
   completing under a real `--max-len`.
2. **V1:** `_maybe_patch_for_glm_moe_dsa` engaging during the real draft load (buffer built on
   CPU, Indexer present on draft layer 78). Structurally confirmed (the patch keys off the TARGET
   architectures, unchanged during the draft build), but load-order is runtime.
3. **V2/V3:** single-KV-group degeneracy of `prepare_inputs`; shared-param sharding preservation
   in vivo; the draft `_logits_partition_spec` probe (no `.lm_head` → replicated fallback → the
   1.77 GiB all-gather risk with a vocab-sharded shared head); dtype/shape of the target
   `lm_head` state entry vs the draft's `ParallelLMHead` param (FP8 quant-method could in
   principle give the draft head scale params; the hard-error contract would catch it).
4. **F2's row question:** GPU-side confirmation of the buffer row semantics in the mixed step-0
   pass, and the M4 parity case with unequal accepted counts.
5. **M2/M3 gates entirely** (greedy spec-equivalence across batch shapes; acceptance ≥ ~4.5 at
   k=5; 2–4× decode uplift) — the doc claims none of these yet, correctly.

---

*Report written (not committed) per review instructions. Fork reviewed at 183f18ce; bench at
cdbf0db (drift to HEAD 5af5f25 is additive: `--offset`, `GLM_LOG_STATS`, `rescore.py`); the
uncommitted F1/F10 bench fixes and the uncommitted fork M1 work noted but not part of the
reviewed commits.*

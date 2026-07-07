# 03 — Decode-throughput bottleneck analysis: GLM-5.2-FP8, Stage-1 XLA path, v4-64 (32 chips)

**2026-07-07. READ-ONLY analysis (no TPU touched; JAX_PLATFORMS=cpu session; a benchmark owns the pod).**
This doc owns the **roofline + device-side suspect ranking** for the measured ~0.95 tok/s single-stream /
4.3–5.9 tok/s batch-8 decode. It is the "FLOP/roofline side" that `docs/04-decode-hostpath-audit.md`
defers to; `docs/05-kv-scaling-design.md` owns KV capacity/batch scaling. Methodology follows the DSV4
discipline: floors first, then per-suspect evidence, then a discriminating probe *before* fixing
(`~/moe-tpu/bench/dsv4_decode_throughput.py`'s prefill-subtraction A/B + the techreport's two clean
nulls that localized DSV4's 61 ms floor to comm/MoE, not attention).

Code bases cited: fork `~/tpu-inference` @ `glm-5.2-v4` (761ea755); harness `~/glm-tpu`; logs
`~/glm-run/{smoke16,gsm8k_n32,gsm8k_n32_try1}.log`.

---

## 0. Executive summary

1. **Measured**: single-stream decode **1.09 s/step (0.92 tok/s)**; batch-8 **~1.37 s/step (5.85 tok/s
   aggregate)** (gsm8k_n32 pass-1 tqdm, 16 prompts, 29m32s, ~10.4k output tokens). Step time is nearly
   flat in batch size → the step is dominated by batch-independent device work.
2. **Roofline**: the fundamental per-step floor on this pod (weight-read + minimal collectives) is
   **≈ 4.7–7 ms single-stream** (~140–210 tok/s) and **≈ 9–12 ms at batch 8** (~700–860 tok/s aggregate).
   We sit **~150–230× above** the single-stream floor and **~120×** above the batch-8 floor — large, but
   not the guessed ~1000×. Even the most pessimal bound — re-reading *every* resident byte (23.6 GB/chip)
   every step — is only **19.2 ms**; we are **~70×** above *that*. Nothing about 753B-on-32-chips forces
   >20 ms/step: the gap is implementation, not physics.
3. **Prime suspect (a): `TPU_MIN_TOKEN_BUCKET=512`** — verified in code: every decode step (7–74 real
   tokens per the crash dumps) runs the **512-token compiled program** (single bucket; `compile_ranges_endpoints=[512]`
   in the logs). This inflates every dense matmul 7–64×, makes the MLA cross-shard gathers move
   **~3 GB/step/chip** over ICI, and — the sleeper — routes **~500 identical pad-token rows into the same
   8 experts every step**, concentrating ~40–120 ms/step of GMM compute on 1–4 chips that the per-layer
   psum then makes everyone wait for.
4. The **true constraint is divisibility by 32** (the token axis is sharded 32-wide inside the MLA
   shard_map). **Bucket 32 is legal** by every static constraint in the code; the PR's 512 is not
   load-bearing. Expected effect of `TPU_MIN_TOKEN_BUCKET=32`: **~10–30× aggregate**, converging toward
   the DSV4-calibrated ~60–100 ms/step regime.
5. Suspects ruled down: **(c) host/Ray** — docs/04's audit bounds the exposed host path at **2–7 ms/step**
   (async scheduling verified ON); **(f) recompiles** — zero mid-run compile events in any log (all 26 XLA
   compiles front-loaded); **(e) sampling/logits** — logits are computed on ≤8 selected rows, vocab-sharded
   (~30 µs weight read); **(b') MLA kernel FLOPs at decode** — the kernel only visits real tokens
   (grid over seqs), so the 32× redundancy is bytes (gathers), not decode FLOPs.
6. **Static budget at bucket 512 accounts for ~135–380 ms** of the 1,370 ms step → a **3–8× residual**
   that static analysis cannot attribute. Do not fix blind: run the two probes in §5 (bucket-32 A/B +
   20-step phased profiler) — one relaunch each, exact commands given.
7. **Fix ladder (speedup × risk)**: bucket 32 (big, low risk) → `max_num_seqs=16` (~2×, free: KV pool
   already covers 16×4096 exactly) → continue_decode / protect async (docs/04) → head-sharded decode
   attention specs (docs/05 S1; kills the q-gather + 32× redundant attention) → Stage-2 Pallas DSA +
   DCP KV sharding. With the first three we should land at **~15–25 tok/s/seq, 150–300+ tok/s aggregate**
   at batch 16 — before any new kernel is written.

---

## 1. Measured baseline (all numbers from the logs)

| Run | Config | Measured | Derived |
|---|---|---|---|
| smoke16 | max_len 4096, mbt 512, max_seqs 4, 128 blocks, gmu 0.90 | engine built 615.5 s; GSM8K n=4 acc 75.0 | — |
| gsm8k_n32 pass 1 | max_seqs 8, 16 prompts, `--max-new 1024` | 16/16 in **29m32s**; final tqdm `input 0.84 toks/s, output 5.85 toks/s` | ≈10,360 output tok; **≈1.37 s/step** at 8 live seqs |
| gsm8k_n32_try1 pass 1 | max_seqs 8, 8 prompts | 8/8 in 18m28s; output 4.34 toks/s | ≈4,810 output tok (drain-limited) |
| single-stream (both logs, 1 live seq) | — | tqdm at 1 prompt in flight: `output 0.92 toks/s` | **≈1.09 s/step** |

Supporting log facts (identical across runs unless noted):
- `compile_ranges_endpoints': [512]`, `max_num_batched_tokens=512`, block_size 512, `num_gpu_blocks_override=128`
  (auto-sizer wanted 3,103 blocks — the override is ours), `GPU KV cache size: 65,536 tokens`,
  `Maximum concurrency for 4,096 tokens per request: 16.00x`.
- HBM: `total_hbm_limit_gb=983.91GiB` (30.75 GiB/chip), used 737.77 GiB (23.06 GiB/chip = weights + runtime),
  avail 147.74 GiB (4.62 GiB/chip).
- Crash-time scheduler dumps (the only per-step visibility with `disable_log_stats=True`):
  a pure-decode step carried **`total_num_scheduled_tokens=7`**; a mixed step **74** (67-token prefill
  chunk + 7 decodes). Both execute the 512-token program → **1.4–14% useful rows**.
- **Zero mid-run recompiles**: all XLA compiles cluster at startup (26 events); generation windows are clean.
- Both gsm8k runs crashed on their *second* `llm.generate` pass with a device-fatal
  `TPU_EXECUTE_ERROR` (not OOM, `preempted_req_ids=[]`; try1 additionally logged
  `Should not schedule a request that does nothing!` ×8 at the pass boundary). **Open reliability item,
  tracked separately** — it kills long benches but does not explain throughput.

DSV4 anchor (same pod, same executor stack, TP-32 + attn-DP + EP, mbt=32, `enforce_eager=True`):
**61.2 ms/step single-seq** (16.3 tok/s), floor localized to MoE all-reduce/comm by two controlled nulls
(`~/moe-tpu/bench/README.md` §Decode throughput A/B; `docs/publication/paper/techreport_en.md` A.1).
GLM at bucket 512 is **17.8×** DSV4's step time on ~2.6× the total params and ~1.5× the active params.

---

## 2. Hardware and model constants

**TPU v4 chip**: 275 TFLOP/s bf16; 32 GiB HBM2 @ **1,228 GB/s**; ICI 6 links × ~50 GB/s/direction
(3D torus; v4-64 slice = 32 chips, 8 hosts × 4). Aggregate: 8.8 PFLOP/s, 39.3 TB/s HBM.
Effective 1-axis all-gather/all-reduce bandwidth per chip: ~100 GB/s (one bidirectional ring) to
~300 GB/s (torus-optimal); small-message collective latency across 32 chips ~5–20 µs.

**GLM-5.2-FP8 byte/FLOP accounting** (config-verified, `docs/00`):
- Routed expert: w1 `[6144, 2×2048]` + w2 `[2048, 6144]` = **37,748,736 B fp8 ≈ 37.75 MB** (=36 MiB).
- 75 MoE layers × 256 experts → **724.8 GB routed**; everything else (attention ≈165 MB/layer ×78,
  shared experts, 3 dense MLPs, embed + lm_head 2×0.95 GB) ≈ **28–31 GB**.
- Per chip (EP=32, 8 experts/chip/layer): 22.65 GB routed + ~0.97 GB dense = **≈23.6 GB resident**
  (matches the measured 23.06 GiB used).
- Active per token: 8×37.75 MB×75 = 22.65 GB routed + shared 2.8 GB + attn/dense ≈ **40 GB ≈ 40B params** ✓.
- MLA per-token latent KV: 640 dims (512+64 padded to 640) × 2 B × 78 layers = **97.5 KiB/token/chip,
  replicated ×32** (docs/05 §0.1). KV pool: 128 blocks × 512 tok × 48.75 MiB/block ≈ 6.1 GiB/chip.

---

## 3. Roofline: per-decode-step floors and where 1.09 s sits

A decode step must, at minimum, read every weight byte it touches from HBM once, and cross ICI for the
per-layer reductions. FLOPs are never binding at decode (batch-8: ~0.64 TFLOP/step aggregate → 73 µs at
peak) — decode on this machine is a **bytes** problem.

### F1 — single-stream (B=1) floor
| Component | Bytes (worst chip) | Time @1.228 TB/s |
|---|---|---|
| Routed experts: 8 distinct/layer over 32 chips → worst chip reads 1–2 experts ×75 layers | 2.8–5.7 GB | 2.3–4.6 ms |
| Dense/attention/lm_head (÷32) | 0.97 GB | 0.79 ms |
| KV read (1k-token ctx: 78 × 1.28 MB) | 0.1 GB | 0.08 ms |
| ICI: ~2 small all-reduces × 78 layers, latency-bound | — | 0.8–1.6 ms |
| **Floor** | | **≈ 4.7–7.0 ms → 140–210 tok/s** |

**Measured 1.087 s/step → 155–230× above F1.** (The task brief guessed ~1000×; the honest number is
~10^2.2. The floor itself refuses to go below ~4 ms because the top-8 experts of even one token are
302 MB that some chip must read.)

### F2 — batch-8 floor
64 routing draws → E[distinct experts] = 256·(1−(255/256)^64) ≈ **57/layer**; worst chip ≈ 3–4 experts
(113–151 MB/layer) → 8.5–11.5 ms + dense 0.79 + ICI ~1.6 → **≈ 9–12 ms/step → 690–860 tok/s aggregate**
(86–107 tok/s/seq). **Measured 1.37 s / 5.85 tok/s → ~120× above F2.**

### F3 — the pessimal "read the whole model every step" bound
23.6 GB/chip ÷ 1.228 TB/s = **19.2 ms/step**. This is what a *maximally naive* dense-read decode would
cost. **Measured is 57–71× above even F3** → the run is emphatically *not* weight-read bound; the missing
time is compute-on-padding, data movement, and inefficiency, all fixable in software.

### ICI floors, current design vs. minimal design
- *Minimal* (head-sharded attention, no token gather): 78 layers × {attn-out psum + MoE psum} of
  `B×6144×2 B` (98 KB @ B=8) → latency-dominated, **~1.6–3 ms/step**.
- *Current* (`mla_attention` cross-shard token gather, bucket 512): per layer, inside the shard_map
  (`attention_interface.py:561-587`), every chip materializes the FULL padded arrays:
  q_nope `[64,512,512]`bf16 = **33.55 MB** + q_pe `[512,64,64]` = 4.19 MB + kv_c `[512,512]` = 0.52 MB
  + k_pe = 0.07 MB ≈ **38.3 MB/layer** → ×78 = **2.99 GB/step/chip** of gathered bytes
  (~2.9 GB received over ICI). At 100–300 GB/s effective → **10–30 ms/step**, and the GSPMD region
  *around* the shard_map adds a head→token reshard first: q_nope is *born head-sharded* from the
  `W_UK_T` einsum (`flash_attn_mla.py:173-179`, W_UK_T is ATTN_HEAD-sharded), gets resharded to the
  token-sharded in_spec `P(None, MLP_TENSOR, None)` (`attention_interface.py:526-533`), then
  all-gathered back to full — and the output mirrors it (slice to token-shard at :644-656, reshard to
  head-shard for the W_UV einsum at flash_attn_mla.py:226-231). Bandwidth cost ~2× the gather alone:
  call it **15–60 ms/step, ~16× of which is pure padding**.

---

## 4. Where the 1.37 s plausibly goes (static budget @ bucket 512, batch 8)

| # | Component | Est. ms/step | Basis |
|---|---|---|---|
| 1 | MoE GMM compute on **pad-expert concentration** (suspect d) | 40–120 | §5(d): ~504 identical rows × 8 fixed experts; 38 GFLOP/expert-with-504-rows; worst chip owns 2–4 of them; psum barrier serializes everyone |
| 2 | MoE routed weight read (~57–75 distinct experts) | 9–15 | §3 F2 (padding barely changes *which* experts are read) |
| 3 | MLA gathers + resharding (suspect b) | 15–60 | §3 ICI-current |
| 4 | Router + permutation ops at 4,096 rows × 75 layers (top_k 512×256, 2× argsort(4096), one_hot(4096,256), ragged_gather — all replicated on every chip; `fused_moe_gmm.py:577-621`) | 15–75 | XLA sort/scatter on TPU is slow; **profiler must pin this** |
| 5 | Dense matmuls + f32 einsum intermediates at T=512 (`preferred_element_type=jnp.float32` → 2× `[64,512,512]`f32 = 134 MB HBM round-trip/layer; flash_attn_mla.py:177,230) | 25–45 | 169 GFLOP/layer global + intermediate traffic |
| 6 | MLA v2 kernel proper (decode grid visits only real seqs; `kernel.py:2373,2520-2560`; v4 blocks (1,1,1)/(1,8,8), s_dtype=f32) | 2–8 | KV-read tiny; launch/grid overhead ×78 |
| 7 | Collectives latency (≥4 shard_map boundaries + 2 psums per layer) | 3–8 | ~500 collectives × 5–20 µs |
| 8 | Logits + sampling + per-step host (docs/04: exposed 2–7 ms) | 4–10 | suspects (c),(e) |
| | **Total accounted** | **≈135–380** | vs **measured ≈1,370** |

**Honest residual: ×3–8 unattributed.** Static analysis cannot distinguish (i) XLA emitting the ops in
rows 1/3/4/5 far worse than estimated (layout transposes at the four shard_map boundaries per layer,
sort lowering, dequant expansion of the dense fp8 linears), from (ii) an ICI efficiency far below
100 GB/s on this 8-host torus for 78 back-to-back gathers, from (iii) something unmodeled. This is
exactly what the §5 profiler probe resolves — **run it before believing any fix beyond the bucket.**

---

## 5. Suspect ranking, with evidence

### (a) `TPU_MIN_TOKEN_BUCKET=512` pads every decode step to a 512-token forward — **CONFIRMED, #1**
Code: `tpu_runner.py:735-741` sets the *minimum* bucket to `max(env=512, 16, …)`; the per-step padding
`runner_utils.get_padded_token_len(self.num_tokens_paddings_per_dp, max_num_scheduled_tokens)` at
`tpu_runner.py:2155-2157` then has exactly ONE bucket to choose from — the logs show
`compile_ranges_endpoints': [512]` and the crash dumps show decode steps of 7 real tokens.
There is **no separate decode bucket**: decode and prefill share `num_tokens_paddings`.
Impact multipliers at batch 8: dense/einsum work ×64, gather bytes ×64 (vs. an 8-token step)
— realized as rows 1/3/4/5 of §4, i.e. **the majority of the modeled step**.
**The real constraint** (why the floor isn't 8 or 16): inside the MLA shard_map the token axis is
sharded `MLP_TENSOR`-wide = 32 on this mesh (`attention_interface.py:526-559`); a 16-token bucket fails
shard_map divisibility (16 % 32 ≠ 0) — a loud trace-time error, not garbage. Additional static
constraints, all satisfied by 32: power-of-2 ≥16 (`runner/utils.py:180-193` assert), `num_tokens×topk % 16 == 0`
(`fused_moe_gmm.py:528`, 32×8=256 ✓), kv-packing minimum (=2 for bf16, `utils.py:239`).
We could **not** find the "garbage output at >50% padding" claim anywhere in the recon docs
(`docs/recon/pr2324-diff.md` documents 512 only as "32-way token shard divisibility") or the code; the
kernel's pad-row NaN guards (m==−inf→0, l==0→out=0) exist and pad rows are never selected by
`logits_indices`. Treat bucket-32 numerics as *unvalidated-but-unthreatened*: gate on the §6 validation.

### (b) The `mla_attention` all-gather design — **CONFIRMED mechanism; bytes, not decode-FLOPs**
Code: `attention_interface.py:561-587` all-gathers q/q_rope/k/k_rope over the 32-way token shard so
every chip sees the full (padded) token set; each chip then runs the kernel over **all 64 heads** and
slices back its 1/32 of tokens (:644-656). Two corrections to the task brief's framing:
(i) at *decode* the kernel does NOT compute 512 queries — the grid iterates sequences and only real
tokens per `cu_q_lens` (kernel.py:2373; batched-decode dispatch :2520-2560), so the redundant compute
is small (~0.5 GFLOP/layer at B=8); the cost is the **38.3 MB/layer of gathered+resharded bytes** (§3);
(ii) at *prefill* the redundancy is real 32× attention FLOPs (every chip computes the full 512-token
chunk attention) — visible in the prompt-side rate (input 0.84 tok/s).
At bucket 32 this suspect shrinks 16× and stops mattering for decode (~1–4 ms); the design fix
(head-sharded specs) is docs/05 S1 and remains the right *prefill/latency* fix.

### (c) Per-step host/Ray overhead — **RULED DOWN (2–7 ms exposed)**
docs/04's hop-by-hop audit: compiled Ray DAG (shm channels) + async scheduling ON by default for this
stack; the exposed path is scheduler + DAG-int return + the `get_execute_model_output` RPC round +
`update_from_output` ≈ **2–7 ms/step**, hidden-prep 3–8 ms overlapped. Cross-check: the same executor
stack delivered 61 ms *total* steps for DSV4. Host cannot explain 1,370 ms. It becomes relevant again
only after the step drops under ~100 ms (then: continue_decode, docs/04 fix #1).

### (d) MoE GMM at padded token counts — **CONFIRMED, the sleeper inside (a)**
Code path: `moe.py:132` → GMM_EP → `fused_moe_gmm.py`. On this mesh (`MLP_DATA='data'`=1) every chip
redundantly processes ALL `512×8 = 4,096` routing rows (`:626-641` in_specs replicate tokens across
'model'), then `gmm_v2` computes only its 8 local experts' groups. Two distinct effects:
1. **Weight read is NOT "all 8 experts/chip regardless of routing"** — `gmm_v2` skips empty groups
   (`gmm_v2.py:830` "Skips over empty groups"), and a nonempty group reads that expert's full 37.75 MB
   once regardless of row count. At 512 padded tokens the distinct-expert set is ~57–75 (real tokens)
   + the pad set — NOT 256 — so the read is ≈F2's 9–15 ms. The dequant-in-VMEM path (`gmm_v2.py:205-224`)
   keeps it fp8-read (correct; no HBM bf16 round-trip).
2. **The pad rows are pathological for COMPUTE**: all ~504 pad rows carry the *same* embedding
   (token id 0) → identical router logits → the *same* top-8 experts get ~504 rows **each**, every
   layer, every step. A chip owning k of those 8 experts does k × ~38 GFLOP (504 rows × (gate/up+down))
   ≈ 0.6–1.6 ms/layer at 30–50% MXU → **45–120 ms/step**, concentrated on 1–4 chips, serialized into
   everyone's step by the closing `psum` (`fused_moe_gmm.py:278`). Plus the replicated argsort/one_hot
   machinery over 4,096 rows (row 4 of §4).
Fix = (a). Independent discriminator: `FORCE_MOE_RANDOM_ROUTING=1` (exists, `fused_moe_gmm.py:568-575`)
spreads the pad rows uniformly — if step time moves materially, (d) is confirmed live.

### (e) Sampling / logits at bucket 512 × vocab 154,880 — **RULED DOWN**
`_execute_model` selects hidden states down to `logits_indices` (≤ padded_num_reqs = 8) *before*
`compute_logits` (`tpu_runner.py:1380-1390`), which stays vocab-sharded (fork commits 87ace031/10efa393
— the 1.77 GiB all-gather was already found and killed). lm_head read: 0.95 GB/32 = 29.7 MB/chip ≈ 24 µs.
Greedy sample on [8, 4840]/chip + one device_get/step (docs/04 hop 6). Total ~2–5 ms. Not the story.

### (f) Per-step recompiles — **RULED OUT by the logs**
All 26 XLA module compiles and every `compilation_manager` event sit in the startup window
(e.g. 07:24:07–07:24:38, generation starts 07:24:39); the generation windows contain zero compile /
trace / cache-miss lines. `enforce_eager=False` note: the log's "Detected eager backend, disabling AOT
compile" only means compilation happens lazily at warmup — it did happen, once.

---

## 6. Measure FIRST: the two probes (exact commands)

Both are one-relaunch, ~30–40 min each including the 10–12 min cold start. Coordinate pod ownership
first (launcher SIGKILLs every vLLM on all 8 hosts — see the launcher's collision policy).

**Probe A — bucket-32 A/B (the (a)/(d) discriminator AND the candidate fix).**
The launcher already parameterizes the env (`TPU_MIN_TOKEN_BUCKET=${TPU_MIN_TOKEN_BUCKET:-512}`,
`scripts/launch_glm_32chip.sh:89`) and bakes it into every raylet:

```bash
TPU_MIN_TOKEN_BUCKET=32 bash ~/glm-tpu/scripts/launch_glm_32chip.sh
cd ~/glm-tpu/bench && set -a && . ~/glm-tpu/.env && set +a
NEW_MODEL_DESIGN=1 MODEL_IMPL_TYPE=vllm TPU_MULTIHOST_BACKEND=ray OMP_NUM_THREADS=1 \
HF_HUB_DISABLE_XET=1 TPU_DISABLE_DSA_INDEXER=1 DISABLE_WEIGHT_REQUANTIZATION=1 \
REQUANTIZE_WEIGHT_DTYPE=float8_e4m3fn TPU_MIN_TOKEN_BUCKET=32 GLM_TP=32 \
~/vllm-env/bin/python -u run_bench.py --benchmark gsm8k --limit 8 \
  --max-len 4096 --max-new 256 --num-gpu-blocks 128 --gmu 0.90 \
  --max-batched-tokens 512 --max-seqs 8 --batch-size 8 \
  --note "PROBE-A bucket32" > ~/glm-run/probe_bucket32.log 2>&1
```

Read out: the tqdm `output N toks/s` vs the 5.85 baseline, plus answer sanity (the n=8 extraction —
this doubles as the numerics gate). Expected if (a)+(d) dominate: **≥10× aggregate**. If the step stays
≳0.5 s, the residual of §4 is real and Probe B decides. Token buckets become [32..512]
(5 programs — startup compile grows a few minutes; prefills of ~93-token prompts drop to the 128 bucket,
so prefill speeds up ~4× too).

**Probe B — 20-step phased profiler at the CURRENT bucket (full attribution).**
The runner has the machinery built in (`tpu_runner.py:_init_phased_profiling`, `runner/utils.py:654`
PhasedBasedProfiler — captures decode-only / prefill-heavy / mixed phases separately;
`envs.py:23,249`). Env must be raylet-baked → use the launcher's `EXTRA_ENVS` hook:

```bash
EXTRA_ENVS="PHASED_PROFILING_DIR=/tmp/glm-prof PHASED_PROFILER_NUM_STEPS_TO_PROFILE_FOR=20" \
  bash ~/glm-tpu/scripts/launch_glm_32chip.sh
# then the same run_bench command as Probe A but WITHOUT TPU_MIN_TOKEN_BUCKET=32,
#   --limit 8 --max-new 64 --note "PROBE-B profile bucket512"
# collect from worker 0:  /tmp/glm-prof/*decode*/**/xplane.pb  → xprof/tensorboard
```

Read out, in order: (1) device-busy vs gap time per step (gap>20% ⇒ host/dispatch after all);
(2) top-10 device ops — expect `gmm`/fusion (suspect d), `all-gather`/`all-to-all` (b), `sort`
(row 4), the two f32 einsums (row 5); (3) the per-step wall time of `execute_model` spans across
workers (straggler = the pad-expert chip). Alternative live view: `USE_JAX_PROFILER_SERVER=1
JAX_PROFILER_SERVER_PORT=9999` (envs.py:40-41). Also flip `disable_log_stats=False` (one-line in
`bench/engine.py` `LLM(...)`) to get vLLM's per-iteration counters in every future run.

**Probe C (only if B shows GMM dominating)** — `EXTRA_ENVS="FORCE_MOE_RANDOM_ROUTING=1"`, perf-only
(outputs are garbage by construction; label the run note accordingly): spreads pad rows across all 256
experts. Step-time drop confirms concentration (d); a *rise* to ~19 ms-floor behavior confirms the
weight-read model of §3.

---

## 7. Fix list, ranked by expected speedup × risk

| # | Fix | Expected | Risk | Exact change | Validation |
|---|---|---|---|---|---|
| 1 | **`TPU_MIN_TOKEN_BUCKET=32`** | **~10–30× aggregate** (step 1.37 s → ~60–120 ms; kills §4 rows 1,3,4,5 by ~16×) | LOW-MED: numerics unvalidated below 512 (no identified mechanism; all static constraints pass at 32 — §5a) | env only (launcher already parameterized). Optionally sweep 32/64/128 | Probe A; then GSM8K n=16 acc ≥ baseline 75±noise, parity `--two-step` unchanged (it runs 1-chip, bucket-independent), 3/3 pod runs |
| 2 | **`--max-seqs 16`** | ~2× aggregate (16 real tokens still ≤ any bucket; step time ~flat) | LOW: KV exactly covers it — 4096/512 = 8 blocks/seq × 16 = 128 blocks (“Maximum concurrency … 16.00x” in the log); GSM8K seqs ≤~1.2k tok → real pressure ~48 blocks. Preemption only if all 16 approach 4096 — watch `Preempted` | `--max-seqs 16` in run_bench | aggregate tok/s ~2×; zero preemptions in log |
| 3 | **Per-step visibility permanently on** | measurement, not speed | none | `disable_log_stats=False` in `bench/engine.py`; keep `PHASED_PROFILING_DIR` in EXTRA_ENVS for probe runs | n/a |
| 4 | **continue_decode (on-device multi-step)** — docs/04 fix #1 | 1.1–2× *after* #1 (amortizes the 2–7 ms exposed host + both Ray rounds over ≤10 steps); irrelevant before #1 | MED: pod-untested; requires `async_scheduling=False` (sync mode trade-off), decode-only batches; scheduler monkeypatch accounting | `additional_config={"enable_continue_decode": true, "max_decode_steps": 10}` in `build_llm` | A/B step time at batch 1 and 8; token-identity vs non-CD run (greedy) |
| 5 | **Head-sharded decode attention specs** — docs/05 S1, what PR #2324 didn’t do | decode: kills the reshard+gather (~1–4 ms at bucket 32, more at long ctx); **prefill: removes the 32× redundant attention FLOPs**; prerequisite for DCP | MED-HIGH: kernel must accept 2 heads/chip (64/32; check MLA v2 head tiling + v4 blocks (1,8,8)); causal-mask fix becomes moot (each chip sees all tokens); KV writes stay replicated-identical | In `flash_attn_mla.py` pass `query_nth_sharding=P(MLP_TENSOR, None, None)`, `query_tnh_sharding=P(None, MLP_TENSOR, None)`, `keyvalue_skh_sharding=P(None, None)` (k is tiny — replicate), `attn_o_nth_sharding=P(MLP_TENSOR, None, None)`; delete the internal all-gather/slice for that spec combo (gate: env `TPU_MLA_HEAD_SHARDED=1`, default off, byte-identical off) | 1-chip regression (spec product=1 → no-op), sub-cube 4-chip parity, pod GSM8K acc, prefill tok/s A/B |
| 6 | **KV pool sizing for 8k+ ctx runs** | enables GPQA at batch >8 | LOW | after #1 shrinks the forward workspace, re-probe free HBM; ~+90 blocks available (4.62 GiB/chip ÷ 48.75 MiB) → `--num-gpu-blocks 192` @ max_len 8192 ⇒ 12 seqs | HBM stats line; no RuntimeProgramAllocationFailure |
| 7 | **Stage-2: Pallas DSA decode kernel + DCP latent-KV sharding** (docs/01 + docs/05 S2) | the real endgame: sparse top-2048 attention + 32× KV capacity | HIGH (weeks) | as designed in docs/01 / docs/05 | staged gates per PLAN.md |
| 8 | **Stage-3: MTP speculative decode** (5 drafts) | ~2–4× effective tok/s multiplier on top | HIGH | after Stage 2 | acceptance ~5 per card |

Non-fixes / de-prioritized: sampling path (§5e), recompile hunting (§5f), Ray transport tuning (docs/04
bounds it), fp8 KV (blocked on the PR #2324 NaN-under-EP root cause — docs/05 S3).
Also flag: the **second-pass `TPU_EXECUTE_ERROR` crash** (§1) needs a root-cause pass (first divergence
candidate: the pass-boundary empty-schedule step) before any 3/3 benchmark gate can pass.

---

## 8. Appendix — arithmetic details

- Expert bytes: 6144·4096 + 2048·6144 = 25,165,824 + 12,582,912 = 37,748,736 B.
- Distinct experts, n uniform draws of 8-sets ≈ 256·(1−(255/256)^(8B)): B=1→8, B=8→56.7, B=16→100.9.
- Gather bytes/layer @T=512 (bf16): q 64·512·512·2 = 33,554,432; q_pe 512·64·64·2 = 4,194,304;
  kv_c 512·512·2 = 524,288; k_pe 512·64·2 = 65,536 → 38.34 MB; ×78 = 2.99 GB/step.
- Ring all-gather time ≈ (31/32)·bytes ÷ eff_bw; all-reduce ≈ 2×.
- Pad-expert GMM: 504 rows → gmm1 504·6144·4096·2 = 25.4 GF, gmm2 504·2048·6144·2 = 12.7 GF
  → 38.1 GF/expert/layer; ÷(0.3–0.5 · 275 TF) ≈ 0.28–0.46 ms per owned pad-expert per layer.
- f32 einsum intermediates @T=512: [64,512,512]·4 B = 67.1 MB ×2 (write+read) ×2 einsums ≈ 268 MB/layer
  HBM traffic ≈ 218 µs/layer ≈ 17 ms/step (bucket 32: ~1 ms).
- Step-time back-out from tqdm: pass-1 16 prompts, 1,772 s, ≈10,360 output tok, ~8 live seqs steady →
  10,360/8 ≈ 1,295 steps → 1.37 s/step. Single-stream: 0.92 tok/s → 1.087 s/step.
- Floors vs measured: 1,087/4.7–7.0 = 155–231×; 1,370/9–12 = 114–152×; 1,370/19.2 = 71×.
- KV block: 512 tok · 640 · 2 B · 78 layers = 51.1 MB = 48.75 MiB/chip; 128 blocks = 6.1 GiB/chip.

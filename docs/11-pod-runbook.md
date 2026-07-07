# docs/11 — POST-GPQA POD RUNBOOK (the exact ordered sequence, zero-turnaround)

**Written 2026-07-07 (CPU-only prep session; GPQA n=198 was in flight — run 48,
`~/glm-run/gpqa198.log`, launched 19:40 UTC, ETA ~4–5 h).** Everything below runs on
**worker 0 of `db-v4-64-od`** (this VM), from `~/glm-tpu` unless stated. TPU access is
**serialized to one session** — nothing here runs concurrently with anything else on the pod,
and helper agents never run TPU code (they set `JAX_PLATFORMS=cpu` BEFORE python starts).

**House rules that apply to every step:**
- **3/3 rule**: any *gate* claim (byte-identity, D0, dcp bring-up, spec-decode equivalence)
  needs **3/3 clean pod runs** before it is "done" (probabilistic multi-host halts —
  `~/moe-tpu/docs/15`). A single benchmark *score* run is a data point, not a gate.
- **Relaunch after any crash** (`bash scripts/launch_glm_32chip.sh`) before re-running —
  leaked EngineCore / ~1800 s placement-group timeout. But **triage/fetch flight files FIRST**
  (step 0's recipe): the launcher's stop phase prunes old `/tmp/glm_flight_*` files.
- **Shared-pod collision policy** (launcher header): the stop phase SIGKILLs every vLLM/Ray
  proc on all 8 hosts as root, including the cohabiting ASPt stack. Check `docker ps` /
  `pgrep -af EngineCore` and coordinate before every launch.
- **`GLM_*`/`TPU_*` envs are trace-time and worker-side** — they must be baked into every
  raylet (launcher `ENVS` / `EXTRA_ENVS`) *and* exported on the driver. Only
  `TPU_DISABLE_DSA_INDEXER` / `DISABLE_WEIGHT_REQUANTIZATION` / `TPU_MIN_TOKEN_BUCKET`
  force-propagate driver→workers (`TpuPlatform.additional_env_vars`); **`GLM_DSA_MODE`,
  `GLM_MLA_DCP`, `GLM_FLIGHT_RECORDER` do NOT** — a host missing them traces a DIFFERENT
  SPMD program (hang/garbage) and, since round-6 (paged indexer), an **inconsistent KV-cache
  topology**. Always pass them through `EXTRA_ENVS`.
- **Backup after every milestone**: `bash scripts/backup_bundle.sh` (now bundles ALL fork
  branches). Commit + push first — bundles carry committed state only.

**The per-run driver env block** (used verbatim in every `run_bench`/`glm_longctx` command
below; it mirrors the raylet ENVS so driver-side traces match the workers):

```bash
cd ~/glm-tpu/bench && set -a && . ~/glm-tpu/.env && set +a   # HF token etc.
export NEW_MODEL_DESIGN=1 MODEL_IMPL_TYPE=vllm TPU_MULTIHOST_BACKEND=ray \
  OMP_NUM_THREADS=1 HF_HUB_DISABLE_XET=1 TPU_DISABLE_DSA_INDEXER=1 \
  DISABLE_WEIGHT_REQUANTIZATION=1 REQUANTIZE_WEIGHT_DTYPE=float8_e4m3fn GLM_TP=32 \
  GLM_ASYNC_SCHED=0 GLM_LOG_STATS=1 GLM_FLIGHT_RECORDER=1
```

(`GLM_ASYNC_SCHED=0` = sync scheduling — the proven run-47/48 config, makes flight-recorder
step attribution exact, ~1% cost. `TPU_MIN_TOKEN_BUCKET` is set per step.)

**Branch map (fork `~/tpu-inference`, worktrees `~/tpu-inference-<tag>`):**

| branch | tip (2026-07-07) | contents | on origin? |
|---|---|---|---|
| `glm-5.2-v4` | `02e44b36` | **mainline** — Stage-1 + obs + OOB fix; what the pod runs NOW | yes |
| `glm-5.2-v4-next` | `cda23c81` | **staging** — mainline + r5fix + 2a2 + 2int (pallas_decode) + dcp + r6fix, all gated off by default | **NO — push before step 2** |
| `glm-5.2-v4-2int` | `53c5e5ee` | Stage-2 DSA sparse-decode integration (merged into -next) | no |
| `glm-5.2-v4-dcp` | `70aa6825` | DCP MLA attention `GLM_MLA_DCP` (merged into -next) | no |
| `glm-5.2-v4-mtp` | `6beacb5d` | Stage-3 dense-MTP M1 (G1–G3) — **LACKS the OOB fix `02e44b36`** (step 7 merges it) | no |
| `glm-5.2-v4-sparse-prefill` | `53c5e5ee` | branch pointer for Stage-2 sparse prefill (work in flight) | no |
| `pr-g1..4-*` | — | upstream PR series slices (owner submits) | no |

**Review-round status:** rounds 1–6 reports are in `docs/reviews/`. **Round 7:**
`round7-2int.md` landed while this was written (integration review of `53c5e5ee`):
**correctness survives attack** (108/108 CPU tests, byte-identity hash gate re-verified), but
findings 1–3 are **blockers for ENABLING `pallas_decode`**, not for the gated merge — see
step 3, which now implements that review's ordered on-TPU gate list. `round7-pg2-oob.md`
(the OOB-fix review) was still pending — check `ls docs/reviews/round7-*` before step 3.

---

## Step 0 — GPQA outcome triage (do this first, whatever happened)

**0a. Determine the outcome:**

```bash
tail -40 ~/glm-run/gpqa198.log
python3 - <<'PY'
import sqlite3
db = sqlite3.connect('/home/gianl/glm-tpu/bench/results.db')
print("summary:", db.execute("SELECT benchmark,n,metric,value,card_value,delta,note FROM summary WHERE run_id=48").fetchall())
print("items recorded:", db.execute("SELECT COUNT(*), SUM(correct), SUM(truncated) FROM items WHERE run_id=48").fetchone())
PY
```

- **Completed** (summary row exists, n=198): log the score + signed Δ vs card 91.2 in
  RESEARCH_LOG + HANDOFF, run `bash scripts/backup_bundle.sh`, go to step 1.
  Note: `--batch-size 0` = one `llm.generate` over all 198 → items land in the DB only at
  the very end; a truncation count (`n_truncated`, cap 4096) must be reported with the Δ.
- **Crashed / wedged** (no summary row; log ends in a fatal or goes silent): → 0b.

**0b. Triage BEFORE relaunching** (the launcher prunes flight files; a pod restart wipes /tmp):

```bash
bash ~/glm-tpu/scripts/triage_crash.sh ~/glm-run/gpqa198.log | tee ~/glm-run/gpqa198.triage
```

Read per docs/10: run 48 was **sync** (`GLM_ASYNC_SCHED=0`) → halted step = each host's last
flight line exactly (lag 0). Expected crash classes: (i) the pre-OOB-fix core-halt class is
CLOSED (`02e44b36` validated by run 47) — if a core halt reappears at a kv page edge, the fix
is incomplete: capture kv_len_min/max from the flight lines and file it against
`docs/reviews/round7-pg2-oob.md`; (ii) KV/HBM pressure at max_len 8192 (`RESOURCE_EXHAUSTED`,
`RuntimeProgramAllocationFailure`) → drop `--max-seqs` to 4; (iii) the flaky FP8-load crash
(during build only) → plain relaunch+retry; 2 consecutive = full relaunch anyway.

**0c. Retry recipe** (exact run-48 reproduction; resume is automatic in spirit — provenance
appends a NEW run, never overwrites):

```bash
GLM_FLIGHT_RECORDER=1 TPU_MIN_TOKEN_BUCKET=32 bash ~/glm-tpu/scripts/launch_glm_32chip.sh
# expect: 8 nodes / 32 TPU in ray status
cd ~/glm-tpu/bench && set -a && . ~/glm-tpu/.env && set +a
NEW_MODEL_DESIGN=1 MODEL_IMPL_TYPE=vllm TPU_MULTIHOST_BACKEND=ray OMP_NUM_THREADS=1 \
HF_HUB_DISABLE_XET=1 TPU_DISABLE_DSA_INDEXER=1 DISABLE_WEIGHT_REQUANTIZATION=1 \
REQUANTIZE_WEIGHT_DTYPE=float8_e4m3fn TPU_MIN_TOKEN_BUCKET=32 GLM_TP=32 \
GLM_ASYNC_SCHED=0 GLM_LOG_STATS=1 GLM_FLIGHT_RECORDER=1 \
nohup ~/vllm-env/bin/python -u run_bench.py --benchmark gpqa_diamond --limit 198 \
  --max-len 8192 --max-new 4096 --max-seqs 8 --num-gpu-blocks 128 --gmu 0.90 \
  --max-batched-tokens 512 --batch-size 0 \
  --note "stage1 GPQA-Diamond n=198 bucket32 retry" > ~/glm-run/gpqa198_retry.log 2>&1 &
```

If the first attempt died mid-batch with 0 items recorded, consider `--batch-size 32`
(chunked: items land per chunk, a crash preserves finished chunks) at the cost of slightly
lower aggregate throughput. **Abort criterion:** 2 crashed attempts with the same signature →
stop, triage both, and fall back to `--max-seqs 4 --batch-size 32` before burning a third.

---

## Step 1 — GSM8K n=128, `--max-new 2048` (kill the truncation-miss artifact)

Why: run 47 (n=32) scored 87.5 with **6/32 misses = truncations at the 1024 cap** — the cap,
not the model, drives most misses (waveA2 n=16 at 93.75). n=128 at 2048 gives the clean
Stage-1 GSM8K number. Same proven config as run 47 otherwise (bucket 32, max_seqs 16,
KV 128 blocks × 512 = 65,536 tokens = exactly 16 × 4096).

```bash
# engine still up from step 0 (same launcher config) — otherwise:
# GLM_FLIGHT_RECORDER=1 TPU_MIN_TOKEN_BUCKET=32 bash ~/glm-tpu/scripts/launch_glm_32chip.sh
cd ~/glm-tpu/bench && set -a && . ~/glm-tpu/.env && set +a
NEW_MODEL_DESIGN=1 MODEL_IMPL_TYPE=vllm TPU_MULTIHOST_BACKEND=ray OMP_NUM_THREADS=1 \
HF_HUB_DISABLE_XET=1 TPU_DISABLE_DSA_INDEXER=1 DISABLE_WEIGHT_REQUANTIZATION=1 \
REQUANTIZE_WEIGHT_DTYPE=float8_e4m3fn TPU_MIN_TOKEN_BUCKET=32 GLM_TP=32 \
GLM_ASYNC_SCHED=0 GLM_LOG_STATS=1 GLM_FLIGHT_RECORDER=1 \
nohup ~/vllm-env/bin/python -u run_bench.py --benchmark gsm8k --limit 128 \
  --max-len 4096 --max-new 2048 --max-seqs 16 --num-gpu-blocks 128 --gmu 0.90 \
  --max-batched-tokens 512 --batch-size 0 \
  --note "stage1 GSM8K n=128 max-new 2048 bucket32" > ~/glm-run/gsm8k_n128.log 2>&1 &
tail -f ~/glm-run/gsm8k_n128.log   # GLM_LOG_STATS prints tok/s every ~10 s
```

**Expected:** ~10–12 min engine build + generation at ~30+ tok/s aggregate (run 47: 32.9);
budget **~25–45 min total**. Acc ≥ ~90 with `n_truncated` ≈ 0 (any remaining truncated item
at 2048 is real and gets reported, not re-capped). **Abort:** a core halt here reopens the
OOB class (see 0b-i) — triage, do NOT just retry-loop.

---

## Step 2 — Staging-branch switch: `glm-5.2-v4-next` + byte-identity smoke

`-next` = mainline + ALL Stage-2 code with every gate off (`GLM_DSA_MODE` default `off`,
`GLM_MLA_DCP` unset, recorder default off). The gate-off contract is **byte-identical
model behavior**; this step proves it on the pod before any DSA/dcp work runs.

**2a. Push + sync** (the workers pull from origin — `-next` is local-only until pushed):

```bash
cd ~/tpu-inference-next && git push origin glm-5.2-v4-next
TPU_INFERENCE_BRANCH=glm-5.2-v4-next bash ~/glm-tpu/scripts/sync_workers.sh
# ABORT unless all 8 lines show tpu-inference @ cda23c81 dirty=0 (same hash on every host)
```

**2b. Relaunch at the run-26 config** (bucket **512** — run 26/smoke16 predates bucket-32):

```bash
GLM_FLIGHT_RECORDER=1 bash ~/glm-tpu/scripts/launch_glm_32chip.sh   # bucket default 512
```

**2c. Smoke: GSM8K n=4, exact run-26 shape, sequential** (`--batch-size 1` = one generate per
item — run 26 predates batched generation):

```bash
cd ~/glm-tpu/bench && set -a && . ~/glm-tpu/.env && set +a
NEW_MODEL_DESIGN=1 MODEL_IMPL_TYPE=vllm TPU_MULTIHOST_BACKEND=ray OMP_NUM_THREADS=1 \
HF_HUB_DISABLE_XET=1 TPU_DISABLE_DSA_INDEXER=1 DISABLE_WEIGHT_REQUANTIZATION=1 \
REQUANTIZE_WEIGHT_DTYPE=float8_e4m3fn GLM_TP=32 GLM_DSA_MODE=off \
~/vllm-env/bin/python -u run_bench.py --benchmark gsm8k --limit 4 \
  --max-len 4096 --max-new 512 --max-seqs 4 --num-gpu-blocks 128 --gmu 0.90 \
  --max-batched-tokens 512 --batch-size 1 \
  --note "staging -next gate-off byte-identity smoke vs run 26" \
  > ~/glm-run/next_identity_smoke.log 2>&1
```

**2d. Verbatim comparison vs run 26 (smoke16):**

```bash
python3 - <<'PY'
import sqlite3
db = sqlite3.connect('/home/gianl/glm-tpu/bench/results.db')
ref = dict(db.execute("SELECT item_id, raw_output FROM items WHERE run_id=26"))
new = db.execute("SELECT MAX(run_id) FROM runs").fetchone()[0]
cand = dict(db.execute("SELECT item_id, raw_output FROM items WHERE run_id=?", (new,)))
assert set(ref) == set(cand), (sorted(ref), sorted(cand))
diff = [k for k in ref if ref[k] != cand[k]]
print(f"run {new} vs run 26: {len(ref)} items, {len(diff)} verbatim mismatches {diff}")
raise SystemExit(1 if diff else 0)
PY
```

**Expected: 0 mismatches.** HONESTY CAVEAT: run 26 ran at fork `10efa393`; mainline has since
added `761ea755` (guards) and `02e44b36` (the OOB *read* fix) — both no-ops at this config on
paper. **If mismatches appear, do NOT blame `-next` yet**: rerun 2b–2c once on mainline
(`TPU_INFERENCE_BRANCH=glm-5.2-v4 bash scripts/sync_workers.sh`, relaunch, same command) —
if mainline also mismatches run 26, the delta is `761ea755`/`02e44b36` (document it, compare
`-next` against the fresh mainline run instead); if mainline matches run 26 but `-next` does
not, `-next`'s gate-off contract is broken → **ABORT the staging switch**, bisect the merges.
Gate = 3/3 identical smokes before `-next` becomes the working pod branch.

---

## Step 3 — DSA compile probe (`GLM_DSA_MODE=pallas_decode`, single request)

First on-TPU Mosaic compile of the Stage-2 kernels (indexer scoring + hierarchical top-k +
gathered-segment sparse MLA + paged indexer k-cache spec). Goal: *observe the compile and
close the round-7 on-TPU gates*, not score anything.

**Round-7 (`docs/reviews/round7-2int.md`) verdict shapes this step:** correctness green on
CPU, but (F1) the sparse branch's work scales with the **padded token bucket** — at bucket
512 a decode step gathers `seg_kv [512, 2048, 640]` ≈ 1.25 GiB/layer ⇒ O(100 GiB)/chip/step,
slower than dense; (F2) `lax.cond` compiles BOTH branches into every bucket ⇒ the never-taken
sparse branch parks a ~1.25–3.9 GiB reservation in the T=512 bucket's executable; (F3) the
latent-cache donation through the cond may fail to alias → per-layer full-cache copies
(TPU-only verifiable). **Therefore the probe runs at bucket 32 / max_seqs 1** (decode steps
pad to T=32 → seg_kv 80 MiB/layer — survivable) **with a HALVED KV pool** (`--num-gpu-blocks
64` = 3 GiB/chip) to leave headroom for the F2 reservation. Do NOT run pallas_decode at
bucket 512, and do NOT expect Stage-1 speed — F1's row-slice fix is queued work on `-2int`.

```bash
# GLM_DSA_MODE must be RAYLET-BAKED (not in additional_env_vars!) AND on the driver:
EXTRA_ENVS="GLM_DSA_MODE=pallas_decode" GLM_FLIGHT_RECORDER=1 TPU_MIN_TOKEN_BUCKET=32 \
  bash ~/glm-tpu/scripts/launch_glm_32chip.sh
cd ~/glm-tpu/bench && set -a && . ~/glm-tpu/.env && set +a
NEW_MODEL_DESIGN=1 MODEL_IMPL_TYPE=vllm TPU_MULTIHOST_BACKEND=ray OMP_NUM_THREADS=1 \
HF_HUB_DISABLE_XET=1 TPU_DISABLE_DSA_INDEXER=1 DISABLE_WEIGHT_REQUANTIZATION=1 \
REQUANTIZE_WEIGHT_DTYPE=float8_e4m3fn TPU_MIN_TOKEN_BUCKET=32 GLM_TP=32 \
GLM_ASYNC_SCHED=0 GLM_LOG_STATS=1 GLM_FLIGHT_RECORDER=1 GLM_DSA_MODE=pallas_decode \
nohup ~/vllm-env/bin/python -u run_bench.py --benchmark gsm8k --limit 1 \
  --max-len 4096 --max-new 128 --max-seqs 1 --num-gpu-blocks 64 --gmu 0.90 \
  --max-batched-tokens 512 --batch-size 0 \
  --note "DSA pallas_decode compile probe n=1" > ~/glm-run/dsa_probe1.log 2>&1 &
```

Watch for: the KV-spec change (the indexer k-cache registers as a second spec, +5.4% bytes —
the sizing log lines WILL differ from Stage-1), engine build, warmup, then 128 decode steps
through the sparse path (prefill falls back to dense in-graph). While the engine is up,
close the round-7 TPU-only gates in its ordered list (`round7-2int.md` §"What only the TPU
can verify"): (gate 1) dense-write→sparse-read layout agreement = step 4's logit diff;
(gate 2, F3) HLO dump of the decode bucket → grep `copy` ops on the 640-wide cache buffer +
confirm `input_output_alias` spans the conditional; (gate 3, F2) per-bucket
`memory_analysis()` mode on vs off; (gate 4) Mosaic compile of the kernel-in-shard_map-in-
cond-in-donated-jit; (gate 5) bf16 numerics at real scale. Also note round-7 F4: MTP spec
decode classifies every step as mixed → dense — do NOT combine step 7 with `pallas_decode`.

**Expected Mosaic/on-TPU failure modes — enumerated from the round-5 reviews'
on-TPU checklists (`docs/reviews/round5-*.md`) + round-6 (`round6-{paged-indexer,dcp}.md`).
None of these has off-TPU evidence; hitting one is a FINDING, not a surprise:**

1. **Indexer kernel `(1, 32)` fp32 `w` tile used as a matmul LHS** (round-5 indexer 3a —
   the highest-risk single spec; no validated 1-sublane matmul-operand precedent in the
   repo). Documented fallback: broadcast-multiply `w[0,:,None] * s` + sublane reduction.
2. **`(1, page_size)` fp32 output tiles** — 1-sublane stores (3b); likely fine, unproven.
3. **The whole `interpret=False` branch is first-run-on-TPU** (3d): `CompilerParams` /
   `dimension_semantics` / vmem_limit — API names verified in both jax versions, compiled
   pipeline never exercised.
4. **SMEM scalar-prefetch table scaling** (3e): `bt_flat` i32[R×max_blocks] — 256 KiB at
   R=64/128K-ctx/page-128; DSV4 idiom but validated only ≤12K ctx. (At this probe's
   max_len 4096 it is tiny — this fires later, at the ladder lengths.)
5. **Sparse-MLA `seg_block` non-128 tile** (round-5 sparse F3): only reachable with
   tiny-topk debug configs; Mosaic may reject or silently mis-tile — the guard asserts at
   `interpret=False` since r5fix.
6. **`pack_new_kv` roll branch at `kv_packing=32`** (round-5 sparse R1 residual): the
   misaligned decode write has zero upstream test coverage on TPU (CPU-unrunnable). A wrong
   write shows up as garbage gathers — step 4's D0 catches it.
7. **`jax.debug.callback` guard semantics under 8-host Ray** (round-6 paged-indexer): a
   tripped validation guard may surface as a fatal `XlaRuntimeError` or a cross-host desync
   instead of a clean Python error.
8. **Gate-K numeric bars are interpreter artifacts** (round-5 sparse R4): re-measure
   fp32/bf16 kernel-vs-oracle deltas on real MXU before quoting them anywhere.
9. Mixed-dtype q/k dot and `page_size % 128` are already hard-asserted (r5fix) — those
   should raise loudly in Python, not reach Mosaic.

**Success:** engine builds, zero Mosaic errors, 1 item recorded with coherent text and (at
ctx ≪ topk=2048, all keys selected) an answer that MATCHES the dense greedy answer for item 0
(run 26 item gsm8k:0 — same extracted answer expected, ideally same text; small kernel-order
numeric drift is tolerable here, exact identity is step 4's job).
**Abort:** any Mosaic lowering error → capture the full error + HLO dump, map it to the list
above, fix on a worktree branch (`glm-5.2-v4-2int`), re-merge to `-next`; do NOT hand-patch
the main checkout.

---

## Step 4 — Gate D0 on TPU (ctx ≤ 2048: sparse == dense)

D0 (docs/01 §4): with every causally-valid key selected (topk ≥ ctx), the sparse path must
reproduce the dense path. Two instruments, run both:

**4a. Pod-level token identity (engine, cheap — uses the step-3 engine).** GSM8K prompts are
~60–100 tokens; at `--max-new 512` total ctx stays ≤ 2048 ≤ topk. Run n=4 sequential under
`pallas_decode`, then compare **verbatim** against the SAME config's dense outputs:

```bash
# (engine from step 3 is still up, GLM_DSA_MODE=pallas_decode baked)
cd ~/glm-tpu/bench && set -a && . ~/glm-tpu/.env && set +a
NEW_MODEL_DESIGN=1 MODEL_IMPL_TYPE=vllm TPU_MULTIHOST_BACKEND=ray OMP_NUM_THREADS=1 \
HF_HUB_DISABLE_XET=1 TPU_DISABLE_DSA_INDEXER=1 DISABLE_WEIGHT_REQUANTIZATION=1 \
REQUANTIZE_WEIGHT_DTYPE=float8_e4m3fn TPU_MIN_TOKEN_BUCKET=32 GLM_TP=32 \
GLM_ASYNC_SCHED=0 GLM_LOG_STATS=1 GLM_FLIGHT_RECORDER=1 GLM_DSA_MODE=pallas_decode \
~/vllm-env/bin/python -u run_bench.py --benchmark gsm8k --limit 4 \
  --max-len 4096 --max-new 512 --max-seqs 4 --num-gpu-blocks 64 --gmu 0.90 \
  --max-batched-tokens 512 --batch-size 1 --note "D0 sparse n=4 (vs dense twin)" \
  > ~/glm-run/d0_sparse.log 2>&1
# dense twin at the SAME config incl. --num-gpu-blocks 64 (relaunch WITHOUT the gate):
GLM_FLIGHT_RECORDER=1 TPU_MIN_TOKEN_BUCKET=32 bash ~/glm-tpu/scripts/launch_glm_32chip.sh
# rerun the identical run_bench command with GLM_DSA_MODE=off, note "D0 dense twin n=4"
# then compare the two newest runs:
python3 - <<'PY'
import sqlite3
db = sqlite3.connect('/home/gianl/glm-tpu/bench/results.db')
r2, r1 = [r[0] for r in db.execute("SELECT run_id FROM runs ORDER BY run_id DESC LIMIT 2")]
a = dict(db.execute("SELECT item_id, raw_output FROM items WHERE run_id=?", (r1,)))
b = dict(db.execute("SELECT item_id, raw_output FROM items WHERE run_id=?", (r2,)))
diff = [k for k in a if a.get(k) != b.get(k)]
print(f"D0 token-identity run{r1} (dense) vs run{r2} (sparse): {len(diff)} mismatches {diff}")
raise SystemExit(1 if diff else 0)
PY
```

Greedy token identity is the pod-level D0 bar. A single flipped token late in a long reply
means a logit-level delta — go to 4b to quantify.

**4b. 1-chip logit comparison (parity harness, exact — engine must be DOWN first**
(`ray stop`/launcher stop phase; the harness pins 1 chip and hand-builds single-shard
metadata — never "fix" it to multi-chip). `--two-step` drives real decode through the paged
cache; the same-run "vs full-prefill" line directly compares sparse decode (decode steps take
the pallas path) against dense prefill of the same tokens, and the dumped npz allows an
off/on bit comparison. Real-dims cfg asserts `index_topk >= T` → all-selected → D0 regime:

```bash
cd ~/glm-tpu
for MODE in off pallas_decode; do
TPU_CHIPS_PER_PROCESS_BOUNDS=1,1,1 TPU_PROCESS_BOUNDS=1,1,1 TPU_VISIBLE_DEVICES=0 \
OMP_NUM_THREADS=1 NEW_MODEL_DESIGN=1 MODEL_IMPL_TYPE=vllm TPU_DISABLE_DSA_INDEXER=1 \
PYTHONPATH=$HOME/tpu-inference-next GLM_DSA_MODE=$MODE GLM_DUMP_NPZ=/tmp/d0_$MODE.npz \
~/vllm-env/bin/python parity/glm_engine_parity.py --fp8 --real-dims --two-step \
  | tee ~/glm-run/d0_parity_$MODE.log; done
python3 - <<'PY'
import numpy as np
a = np.load('/tmp/d0_off.npz'); b = np.load('/tmp/d0_pallas_decode.npz')
for k in a.files:
    d = float(abs(a[k].astype('float64') - b[k].astype('float64')).max())
    print(f"{k}: max|off-sparse| = {d}")
PY
```

**Pass bar:** both runs print `[GATE] PASS` vs HF; the prefill dump (`eng`) is **bit-identical**
off-vs-sparse (prefill falls back to dense in-graph — any delta means the gate leaks); the
sparse run's two-step "vs full-prefill" max|Δ| is within the dense run's own bar (the 2c
kernel's accumulation order differs from mla.v2 — bit-for-bit was the *xla_ref* D0 bar; for
`pallas_decode` the honest bar is ≤ the dense two-step delta + Gate-K's re-measured MXU bound,
with 4a's token identity as the end-to-end check). Record the measured deltas in RESEARCH_LOG.
**3/3 on 4a before calling D0 closed.**

---

## Step 5 — Passkey ladder at DCP (the docs/05 S2 capacity unlock)

**Branch: `glm-5.2-v4-next`** (dcp merged: `GLM_MLA_DCP` + LSE combine + strided positions).
KV math (docs/05 §4.3, pool = 128-block baseline 65,536 tokens ≈ 6.1 GiB/chip):

| config | pool tokens | 128K/seq per chip | 128K seqs |
|---|---|---|---|
| baseline (replicated) | 65,536 | 11.9 GiB | **0 — impossible** |
| dcp=4 (model 8 × dcp 4) | 262K | 2.98 GiB | 2 (marginal) |
| **dcp=8** (model 4 × dcp 8) | 524K | 1.49 GiB | **4 — ladder runnable** |

So: **8K/32K cells can run at dcp=4; the 128K cell needs dcp=8.** 1M needs dcp≥16 + fp8 KV —
out of scope here.

**5a. Prereq — plumb the engine arg.** `build_llm` does not yet pass
`decode_context_parallel_size` (bench/ is owned by the bench session — land it there, or
apply this exact one-liner at run time in `bench/engine.py`, inside the `extra = {}` block):

```python
    if os.environ.get("GLM_DCP"):                       # docs/11 step 5: DCP
        extra["decode_context_parallel_size"] = int(os.environ["GLM_DCP"])
```

(vLLM's `ParallelConfig.decode_context_parallel_size` exists upstream — #2398; dcp must
divide TP=32. The fork requires `MLA_TRANSPOSE_KV_CACHE=0` — the default — with GLM_MLA_DCP.)

**5b. Bring-up gate first (never straight to 128K):** relaunch with dcp baked, then GSM8K
n=32 must reproduce the Stage-1 rows (≈87.5 at max-new 1024, or the step-1 number at 2048):

```bash
EXTRA_ENVS="GLM_MLA_DCP=1" GLM_FLIGHT_RECORDER=1 TPU_MIN_TOKEN_BUCKET=32 \
  bash ~/glm-tpu/scripts/launch_glm_32chip.sh
cd ~/glm-tpu/bench && set -a && . ~/glm-tpu/.env && set +a
NEW_MODEL_DESIGN=1 MODEL_IMPL_TYPE=vllm TPU_MULTIHOST_BACKEND=ray OMP_NUM_THREADS=1 \
HF_HUB_DISABLE_XET=1 TPU_DISABLE_DSA_INDEXER=1 DISABLE_WEIGHT_REQUANTIZATION=1 \
REQUANTIZE_WEIGHT_DTYPE=float8_e4m3fn TPU_MIN_TOKEN_BUCKET=32 GLM_TP=32 \
GLM_ASYNC_SCHED=0 GLM_LOG_STATS=1 GLM_FLIGHT_RECORDER=1 GLM_MLA_DCP=1 GLM_DCP=4 \
~/vllm-env/bin/python -u run_bench.py --benchmark gsm8k --limit 32 \
  --max-len 4096 --max-new 1024 --max-seqs 16 --num-gpu-blocks 0 --gmu 0.90 \
  --max-batched-tokens 512 --batch-size 0 --note "dcp4 bring-up GSM8K n=32" \
  > ~/glm-run/dcp4_gsm8k.log 2>&1
```

(`--num-gpu-blocks 0` — let vLLM re-size: the scheduler's logical block is now 512×dcp
tokens, the 128-override no longer means what it meant.) On-TPU risk list from
`docs/reviews/round6-dcp.md`: Mosaic compile of the strided-mask/LSE variant (**VMEM: the lse
scratch is 4 MiB at H=128 token-sharded — head-sharded H_local ≤ 16 is comfortable; a
compile-time fit check is mandatory**), zero-byte dma_start liveness on all-empty shards,
kv_packing>1 with emit_lse never executed packed, block tables arriving in P_g-token units.

**5c. The ladder** (dense long-context, generation-based passkey; Stage-2 gate P = ≥95%
per length; `--max-len 0` auto-derives `max(lengths)+256`):

```bash
# 8K + 32K at dcp=4 (same engine as 5b):
NEW_MODEL_DESIGN=1 MODEL_IMPL_TYPE=vllm TPU_MULTIHOST_BACKEND=ray OMP_NUM_THREADS=1 \
HF_HUB_DISABLE_XET=1 TPU_DISABLE_DSA_INDEXER=1 DISABLE_WEIGHT_REQUANTIZATION=1 \
REQUANTIZE_WEIGHT_DTYPE=float8_e4m3fn TPU_MIN_TOKEN_BUCKET=32 GLM_TP=32 \
GLM_ASYNC_SCHED=0 GLM_LOG_STATS=1 GLM_FLIGHT_RECORDER=1 GLM_MLA_DCP=1 GLM_DCP=4 \
nohup ~/vllm-env/bin/python -u glm_longctx.py --lengths 8192,32768 \
  --depths 0.25,0.5,0.75 --trials 8 --max-seqs 2 --gmu 0.90 \
  --note "passkey dcp4 8K/32K" > ~/glm-run/passkey_dcp4.log 2>&1 &
# 128K at dcp=8 (relaunch with GLM_DCP=8; max_seqs 1 first — 1.49 GiB/chip/seq):
EXTRA_ENVS="GLM_MLA_DCP=1" GLM_FLIGHT_RECORDER=1 TPU_MIN_TOKEN_BUCKET=32 \
  bash ~/glm-tpu/scripts/launch_glm_32chip.sh
# ... same env with GLM_DCP=8:
nohup ~/vllm-env/bin/python -u glm_longctx.py --lengths 131072 \
  --depths 0.25,0.5,0.75 --trials 8 --max-seqs 1 --gmu 0.90 \
  --note "passkey dcp8 128K" > ~/glm-run/passkey_dcp8_128k.log 2>&1 &
```

Raw-protocol prompts (`[gMASK]<sop>` ids prepended explicitly); if retrieval is erratic at
8K (protocol, not capacity), rerun that cell with `--protocol chat` before touching dcp.
128K prefill at 512-token chunks = 256 chunks — expect a LONG prefill (hours-scale at current
unoptimized step times; watch the flight recorder cadence, don't kill it for being slow).
**Abort:** dcp bring-up (5b) failing acc → stop the ladder; the LSE-combine/striping math is
wrong on hardware — collect logits evidence at 4K first (docs/05 §7's micro-parity plan).

---

## Step 6 — AIME-2026 card protocol, n=30 (docs/07 §6; seed fix landed)

Card protocol: temperature=1.0, top_p=0.95, the byte-pinned `Exact Answer:` system prompt,
bare-question user turn, 163,840-token cap min-ed with window room. The round-6 F1 fix makes
this runnable (per-request seeds are OMITTED — the platform rejects them; the loud
"per-request sampling seeds are UNSUPPORTED" banner at build is **the fix working, not an
error**; `--seed 0` is a provenance label only).

```bash
# dense mainline or -next gate-off engine, bucket 32, back at the Stage-1 recipe:
GLM_FLIGHT_RECORDER=1 TPU_MIN_TOKEN_BUCKET=32 bash ~/glm-tpu/scripts/launch_glm_32chip.sh
cd ~/glm-tpu/bench && set -a && . ~/glm-tpu/.env && set +a
NEW_MODEL_DESIGN=1 MODEL_IMPL_TYPE=vllm TPU_MULTIHOST_BACKEND=ray OMP_NUM_THREADS=1 \
HF_HUB_DISABLE_XET=1 TPU_DISABLE_DSA_INDEXER=1 DISABLE_WEIGHT_REQUANTIZATION=1 \
REQUANTIZE_WEIGHT_DTYPE=float8_e4m3fn TPU_MIN_TOKEN_BUCKET=32 GLM_TP=32 \
GLM_ASYNC_SCHED=0 GLM_LOG_STATS=1 GLM_FLIGHT_RECORDER=1 \
nohup ~/vllm-env/bin/python -u run_bench.py --benchmark aime_2026 --protocol card \
  --max-len 65536 --max-seqs 4 --batch-size 0 --seed 0 --gmu 0.90 \
  --max-batched-tokens 512 --note "card-protocol AIME 2026 n=30" \
  > ~/glm-run/aime30_card.log 2>&1 &
```

Notes: no `--max-new` — the effective cap is min(163,840, window room); `--max-len 65536`
is the fidelity-vs-KV knob (the card-faithful 165,888 does not fit Stage-1 KV; any binding
cap shows as `n_truncated` and MUST be reported with the Δ vs card 99.2). KV pressure is real
at 64K windows (128 blocks = one 64K seq): watch `Preempted` lines; thinking-mode traces are
LONG — budget hours, and Δ caveats per docs/07 §4 (judge substitute, n=30 ⇒ ±3.33 pts/item,
temperature-1.0 single-sample variance — consider `--samples 4` for avg@4, labeled as such).

---

## Step 7 — MTP M2 (spec-decode greedy == non-spec; k=1 then k=5)

From the M1 report (RESEARCH_LOG 2026-07-07 Stage-3 entry + docs/08 §M2). **Gate M2: for
N≥32 prompts, generated token sequences EQUAL the non-speculative greedy baseline exactly**
(argmax-match acceptance ⇒ distribution-identical; exact equality can only break on logit
ties — the tie-flip protocol in docs/08). M1's engine-scale checklist to verify during
bring-up: (V2) draft KV grouped with target MLA layers — eagle3 `prepare_inputs`'s last-group
assumption must degenerate correctly for a single unified group; (G3) the file-level draft
filter against the REAL gs:// safetensors index (expect ~1–2 of 150 files re-streamed, NOT
755.7 GB); FP8 draft load (the quantized path skips vLLM weights-tracking); per-bucket
precompile of the draft programs.

**7a. Prereq — the MTP branch LACKS the OOB fix** (`6beacb5d` branched pre-`02e44b36`;
bucket-32 decode would re-hit the fixed core-halt class). Merge + push + sync:

```bash
cd ~/tpu-inference-mtp && git merge --no-edit 02e44b36 && git push origin glm-5.2-v4-mtp
TPU_INFERENCE_BRANCH=glm-5.2-v4-mtp bash ~/glm-tpu/scripts/sync_workers.sh
GLM_FLIGHT_RECORDER=1 TPU_MIN_TOKEN_BUCKET=32 bash ~/glm-tpu/scripts/launch_glm_32chip.sh
```

**7b. Prereq — spec-config passthrough in `build_llm`** (bench-owned; exact one-liner, same
`extra = {}` block as 5a):

```python
    if os.environ.get("GLM_SPEC_CONFIG"):               # docs/11 step 7: MTP
        extra["speculative_config"] = json.loads(os.environ["GLM_SPEC_CONFIG"])
```

**7c. Baseline (spec OFF) then k=1, token-identity compare** (docs/08: start k=1 — single
trace, simplest batch shaping):

```bash
cd ~/glm-tpu/bench && set -a && . ~/glm-tpu/.env && set +a
BASE="NEW_MODEL_DESIGN=1 MODEL_IMPL_TYPE=vllm TPU_MULTIHOST_BACKEND=ray OMP_NUM_THREADS=1 \
HF_HUB_DISABLE_XET=1 TPU_DISABLE_DSA_INDEXER=1 DISABLE_WEIGHT_REQUANTIZATION=1 \
REQUANTIZE_WEIGHT_DTYPE=float8_e4m3fn TPU_MIN_TOKEN_BUCKET=32 GLM_TP=32 \
GLM_ASYNC_SCHED=0 GLM_LOG_STATS=1 GLM_FLIGHT_RECORDER=1"
env $BASE ~/vllm-env/bin/python -u run_bench.py --benchmark gsm8k --limit 32 \
  --max-len 4096 --max-new 1024 --max-seqs 16 --num-gpu-blocks 0 --gmu 0.90 \
  --max-batched-tokens 512 --batch-size 0 --note "M2 baseline non-spec n=32" \
  > ~/glm-run/m2_baseline.log 2>&1
env $BASE GLM_SPEC_CONFIG='{"method":"mtp","num_speculative_tokens":1}' \
  ~/vllm-env/bin/python -u run_bench.py --benchmark gsm8k --limit 32 \
  --max-len 4096 --max-new 1024 --max-seqs 16 --num-gpu-blocks 0 --gmu 0.90 \
  --max-batched-tokens 512 --batch-size 0 --note "M2 spec-decode mtp k=1 n=32" \
  > ~/glm-run/m2_k1.log 2>&1
python3 - <<'PY'
import sqlite3
db = sqlite3.connect('/home/gianl/glm-tpu/bench/results.db')
r2, r1 = [r[0] for r in db.execute("SELECT run_id FROM runs ORDER BY run_id DESC LIMIT 2")]
a = dict(db.execute("SELECT item_id, raw_output FROM items WHERE run_id=?", (r1,)))
b = dict(db.execute("SELECT item_id, raw_output FROM items WHERE run_id=?", (r2,)))
diff = [k for k in a if a.get(k) != b.get(k)]
print(f"M2 run{r1} (non-spec) vs run{r2} (k=1): {len(diff)}/{len(a)} mismatches {diff[:8]}")
raise SystemExit(1 if diff else 0)
PY
```

KV note: MTP adds a draft KV group — auto-size (`--num-gpu-blocks 0`); expect fewer blocks
than 128. Draft load: watch for the G3 filter log (only layer-78-bearing files streamed);
a full 755.7 GB re-stream = G3 failed open → fix before calling M2 (it fails OPEN by design).

**7d. k=5** (same pair with `"num_speculative_tokens":5`, note "M2 ... k=5") — same identity
bar, plus record the acceptance stats (M3's ≥~4.5 acceptance-length target starts here;
`SpecDecodingStats` aggregation is an M3 deliverable — at minimum capture tok/s A/B).
**Isolated mismatches**: apply the docs/08 tie-flip protocol (a logit tie at the mismatch
position exonerates spec-decode; anything else is a real bug). **3/3 for the k that passes.**

---

## After the sequence

Log every step in `docs/RESEARCH_LOG.md` (numbers + honest nulls), refresh `HANDOFF.md`,
commit + push, and run `bash scripts/backup_bundle.sh` again. The next frontier after this
runbook: sparse-prefill (branch exists), Gate D divergence ladder 4K→128K under
`pallas_decode`, DSA+DCP composition (docs/05 §6), and the M3 acceptance instrumentation.

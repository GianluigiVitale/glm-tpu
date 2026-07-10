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
| `glm-5.2-v4-next` | `cda8a707` (2026-07-08) | **staging** — mainline (incl. OOB fix `02e44b36`) + r5fix + 2a2 + 2int (pallas_decode) + dcp + r6fix + sparse-prefill + **mtp-g4 (dense-MTP M1, merge `534cd74d7`)** + det, all gated off by default | **partially — origin @ `886eaceb4` already has mtp-g4 + OOB (enough for step 7); push before step 2 for det** |
| `glm-5.2-v4-2int` | `53c5e5ee` | Stage-2 DSA sparse-decode integration (merged into -next) | no |
| `glm-5.2-v4-dcp` | `70aa6825` | DCP MLA attention `GLM_MLA_DCP` (merged into -next) | no |
| `glm-5.2-v4-mtp` | `6beacb5d` | Stage-3 dense-MTP M1 (G1–G3) — frozen; **SUPERSEDED for step 7 by the mtp-g4 merge into `-next`** (do not run M2 from here: lacks the OOB fix) | no |
| `glm-5.2-v4-sparse-prefill` | `53c5e5ee` | branch pointer for Stage-2 sparse prefill (work in flight) | no |
| `pr-g1..4-*` | — | upstream PR series slices (owner submits) | no |

**Review-round status:** rounds 1–6 reports are in `docs/reviews/`. **Round 7:**
`round7-2int.md` landed while this was written (integration review of `53c5e5ee`):
**correctness survives attack** (108/108 CPU tests, byte-identity hash gate re-verified), but
findings 1–3 are **blockers for ENABLING `pallas_decode`**, not for the gated merge — see
step 3, which now implements that review's ordered on-TPU gate list. `round7-pg2-oob.md`
(the OOB-fix review) was still pending — check `ls docs/reviews/round7-*` before step 3.

---


> **AUDIT REORDER (2026-07-07, independent audit adopted):** the Stage-2 items below run
> CHEAPEST-RISK-FIRST — the single-chip kernel gates (§2a/§2b, minutes each) come BEFORE any
> engine-level DSA work or staging switch. The kernel logic is FROZEN at c1456937 behavior
> until these gates run; only a gate failure reopens it. Every outcome (incl. compile
> rejections) goes to results.db or docs/artifacts/ — a number with no stored rows is not a
> result. Benchmarks state their attention_path (now recorded in run env_json); GPQA-198 is
> a DENSE-path Stage-1 result and must never be cited as a Stage-2/kernel result.

## §2a (NEW, runs FIRST after §0/§1): single-chip indexer-kernel compile probe (interpret=False)
One v4 chip on w-0 (pod idle — NEVER concurrent with an engine). Tiny shapes. The ONE question:
does Mosaic accept the (1, H) w-tile as a matmul LHS (round-5 flagged, no validated precedent)?
```
cd ~/tpu-inference-spre && TPU_VISIBLE_DEVICES=0 ~/vllm-env/bin/python -m pytest \
  tests/kernels/test_dsa_indexer_kernel.py -k "pallas" --interpret-off 2>&1 | tee ~/glm-run/kernelprobe_2b.log
```
(adapt: the suite runs interpret=True by default — flip via the test's env/param hook; if none
exists, a 10-line driver calling indexer_scores_pallas with interpret=False at R=2,ctx=512.)
- ACCEPT -> §2b. REJECT (Mosaic error) -> apply the documented fallback (broadcast-multiply
  w[0,:,None]*s + sublane reduction), re-run, commit the fallback as the gate-fix.
- Save the log either way: docs/artifacts/kernelprobe-2b-<date>.log.

## §2b (NEW): single-chip real-MXU parity
Same chip. interpret=False runs of: indexer kernel vs indexer_scores_xla + hierarchical_topk vs
the HF-math oracle (fp32: selected-set EXACT mod ties; bf16: S2 boundary-band); sparse-MLA kernel
vs dsa_sparse_decode_xla (fp32 ~1e-6-class, bf16 per round-5 bars re-measured on real MXU);
pack_new_kv OOB suite interpret=False. Save deltas verbatim to docs/artifacts/. Exit: parity
green on silicon. ONLY THEN proceed to the staging switch + engine-level DSA (§2/§3 below).

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

**5a. Prereq — plumb the engine arg. ✅ LANDED (bench session, 2026-07-08).**
`bench/engine.py build_llm` now reads `GLM_DCP`: set `GLM_DCP=N` →
`decode_context_parallel_size=N` in the `LLM(...)` kwargs; unset/`0`/empty → the kwarg is
ABSENT (engine args byte-identical to the pre-DCP harness — the `0`-off convention matches
`GLM_ASYNC_SCHED`). The engine-built log line records `dcp=N` when active, and `GLM_DCP`
lands in `runs.env_json.os_env` via the `GLM_*` provenance sweep in both harnesses
(run_bench + glm_longctx). Unit-tested with a monkeypatched `vllm.LLM` capture
(`bench/test_bench.py::test_dcp_engine_kwarg` — present/absent + byte-identity).

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

> **⛔ OBSOLETE (2026-07-09 21:30 — kept for history only).** Everything from here through 5d(iv) — the
> cache-dump DIFFER→MATCH protocol, the A/B/C localizer, the scatter-only probe, and the whole
> `GLM_DCP_SCATTER_IMPL` DIFFER→MATCH lever hunt — was chasing a bug that DID NOT EXIST in the
> write/scatter/read machinery. The real root cause was the **block-table granularity contract**: the fork's
> `get_kv_cache_spec` pre-multiplied block_size ×dcp AND vLLM's engine multiplied ×dcp again, so the engine
> allocated one block id per `512·dcp²` tokens while the TPU stack consumed the table at `512·dcp`
> tokens/entry — every table entry ≥1 dereferenced an unallocated/stale page. Fixed in `1f700c507`
> (convergent with upstream PR #3129, merged 3 h earlier — see
> `docs/upstream/dcp-block-granularity-report.md`). Post-fix: 900tok/4K/16K needles all correct (runs
> 119–121), 128K smoked 7/7. The scatter was never the bug: `pageloop`/`flat`/`no_donate` were all
> correctly-measured INERT (and that inertness was evidence FOR the granularity theory). `pageloop` is
> retained (CPU-bit-identical, the default for the sparse indexer-key scatter; debug-priced — per-layer
> sort+scan) and `dcp_cache_diff` now REFUSES partial dumps + asserts block-table coverage (`6f45e0944`),
> because under the granularity bug the diff itself could return a WRONG verdict. Do not run this section
> to debug DCP; go to §8. The text below is preserved verbatim as a record of the (wrong but disciplined)
> localization path.

**5d. DCP multi-chunk debug (the pinned failure: multi-chunk prefill).** The on-metal discriminator
showed L=3200 FAILS at `--max-batched-tokens 2048` (two prefill chunks) but PASSES at `--max-batched-tokens
6144` (one chunk). CPU tests prove the owner-scatter WRITE and its metadata are correct
(`tests/layers/common/test_mla_dcp.py::test_dcp_chunked_prefill_write_matches_single_chunk`), so the surviving
metal bug is either (i) chunk-1's `P(BATCH,CONTEXT)` scattered cache not surviving into chunk-2's SCHEDULER
STEP (write-back / persistence / `input_output_aliases`), or (ii) the chunk-2 read. This gated hook
(`GLM_DCP_CACHE_DUMP`, fork `tpu_inference/runner/dcp_cache_dump.py`, wired in `tpu_runner.execute_model`;
unset = zero behavioral change) dumps the post-prefill KV cache so an offline diff distinguishes them. Two
one-shot runs (same prompt, only `--max-batched-tokens` differs), then diff:

```bash
# Relaunch with DCP baked (same as 5b) BEFORE each run if the engine isn't up:
EXTRA_ENVS="GLM_MLA_DCP=1" GLM_FLIGHT_RECORDER=1 TPU_MIN_TOKEN_BUCKET=32 \
  bash ~/glm-tpu/scripts/launch_glm_32chip.sh
cd ~/glm-tpu/bench && set -a && . ~/glm-tpu/.env && set +a
_ENVS="NEW_MODEL_DESIGN=1 MODEL_IMPL_TYPE=vllm TPU_MULTIHOST_BACKEND=ray OMP_NUM_THREADS=1 \
HF_HUB_DISABLE_XET=1 TPU_DISABLE_DSA_INDEXER=1 DISABLE_WEIGHT_REQUANTIZATION=1 \
REQUANTIZE_WEIGHT_DTYPE=float8_e4m3fn TPU_MIN_TOKEN_BUCKET=32 GLM_TP=32 \
GLM_ASYNC_SCHED=0 GLM_MLA_DCP=1 GLM_DCP=2 GLM_DCP_CACHE_DUMP_LAYERS=0"

# (a) TWO-chunk prefill of an N≈3200 prompt: mbt=2048 (< N) -> chunks 2048 + 1152.
env $_ENVS GLM_DCP_CACHE_DUMP=/tmp/dcp_2chunk.npz \
  ~/vllm-env/bin/python -u glm_longctx.py --lengths 3200 --depths 0.5 --trials 1 \
    --max-seqs 1 --max-batched-tokens 2048 --gmu 0.90 \
    --note "DCP debug 2-chunk" > ~/glm-run/dcp_dbg_2chunk.log 2>&1

# (b) ONE-chunk prefill of the SAME prompt: mbt=6144 (>= N) -> a single chunk.
env $_ENVS GLM_DCP_CACHE_DUMP=/tmp/dcp_1chunk.npz \
  ~/vllm-env/bin/python -u glm_longctx.py --lengths 3200 --depths 0.5 --trials 1 \
    --max-seqs 1 --max-batched-tokens 6144 --gmu 0.90 \
    --note "DCP debug 1-chunk" > ~/glm-run/dcp_dbg_1chunk.log 2>&1
# (each writes /tmp/dcp_{2,1}chunk.step<NNNN>.proc<P>.npz per prefill step & host;
#  the HIGHEST-step file per proc is the post-prefill cache — decode steps don't dump.)
```

Offline diff — use the committed, CPU-tested tool (`tpu_inference/runner/dcp_cache_diff.py`). It reassembles
each run's global cache from its per-process shards and compares **by LOGICAL token position** via each run's
own block table, so it is correct even though the two runs allocate DIFFERENT physical pages (the naive
per-page diff is not — do not compare `fa[pages]` vs `fb[pages]`):

```bash
~/vllm-env/bin/python -m tpu_inference.runner.dcp_cache_diff /tmp/dcp_2chunk /tmp/dcp_1chunk
# seq_len=... max|Δ|=... n_diff_positions=... first=[...]
# VERDICT: DIFFER -> cache CONTENT wrong (chunk-2 write positions / chunk-1
#          corruption) -> fix the write/scatter
#      or  MATCH  -> cache CONTENT correct -> the multi-chunk failure is the
#          READ path (decode block_tables / kv_lens after a multi-chunk prefill)
```

This is the **decisive correctness check** the A/B/C temporal-consistency probe does NOT do (A/B/C only proved
chunk-1 survives the step boundary; it did not compare against a 1-chunk prefill of the same prompt). Geometry:
`P_g` (tokens/logical page) `= layer0__shape[1]*layer0__shape[2] = block_size*dcp`; global pos `p` →
`page = block_table[p // P_g]`, `row = (p % P_g)//kv_packing`, `sub = (p % P_g)%kv_packing`. CPU-covered
(reassembly-from-shards + logical alignment MATCH across different page maps + DIFFER on wrong write positions)
by `tests/runner/test_dcp_cache_diff.py`; the dump hook by `tests/runner/test_dcp_cache_dump.py`.

**5d(ii). Single-run A/B/C localizer (post out-sharding fix — pins WHERE chunk-1 dies).** The out-sharding
fix landed (step-fn/`_propose` cache now `P(BATCH,CONTEXT)`; `GLM_DCP_ASSERT_SHARDING` stays SILENT — the
striped cache round-trips with no reshard), but if a 2-chunk passkey still returns `pred=None`, chunk-1 is
lost by a mechanism BELOW the jit out-sharding. To separate "lost in the RUNNER CARRY between scheduler
steps" from "lost INSIDE chunk-2's forward (the read / block-table path)" in **one** run, set
`GLM_DCP_CACHE_DUMP_PREFWD=1` alongside `GLM_DCP_CACHE_DUMP`: the runner then also dumps `self.kv_caches`
**before** each `model_fn` call (the value carried in from the previous step), labeled `prefwd`, so a
2-chunk run writes the triple **A = `postfwd.step0001`** (after chunk-1's forward), **B = `prefwd.step0002`**
(carried into chunk-2's forward), **C = `postfwd.step0002`** (after chunk-2's forward):

```bash
# ONE 2-chunk run (mbt < prompt); PREFWD adds the pre-forward snapshot.
env $_ENVS GLM_DCP_CACHE_DUMP=/tmp/dcp_abc.npz GLM_DCP_CACHE_DUMP_PREFWD=1 \
  ~/vllm-env/bin/python -u glm_longctx.py --lengths 3200 --depths 0.5 --trials 1 \
    --max-seqs 1 --max-batched-tokens 2048 --gmu 0.90 \
    --note "DCP A/B/C localizer" > ~/glm-run/dcp_abc.log 2>&1
# writes /tmp/dcp_abc.{prefwd,postfwd}.step<NNNN>.proc<P>.npz
```

```python
import glob, numpy as np
def _load(phase, step):                        # reassemble global layer-0 over all procs
    fs = glob.glob(f"/tmp/dcp_abc.{phase}.step{step:04d}.proc*.npz")
    ds = [np.load(f) for f in fs]
    shp = tuple(int(x) for x in ds[0]["layer0__shape"]); a = np.zeros(shp, np.float32)
    for d in ds:
        n = int(d["layer0__nshards"])
        if n == 0: a[...] = d["layer0__full"]; continue
        for si in range(n):
            idx = eval(str(d[f"layer0__shard{si}__index"]), {"slice": slice, "__builtins__": {}})
            a[idx] = d[f"layer0__shard{si}__data"]
    return a
A, B, C = _load("postfwd", 1), _load("prefwd", 2), _load("postfwd", 2)
ab = np.abs(A.astype(np.float64) - B.astype(np.float64)).max()
print(f"A vs B max|Δ|={ab:.3e}")
print("VERDICT:",
      "A != B  -> chunk-1 dies in the RUNNER CARRY (self.kv_caches store-back / "
      "re-shard / stale ref between execute_model calls) — fix tpu_runner"
      if ab > 1e-3 else
      "A == B  -> carry preserves chunk-1; chunk-1 dies INSIDE chunk-2's forward "
      "(kernel read of chunk-1's blocks or the chunk-2 block-table/slot map) — fix the read path")
```

Run this probe at **`--max-seqs 1`** (as above): under interleaved multi-seq serving the absolute step
numbers advance past decode steps, so `stepNNNN` is not simply "the Nth prefill" — `prefwd`/`postfwd` still
pair *within* a call, but pick the A/B/C files by inspecting each file's `num_scheduled_tokens` rather than
assuming step 1/2 are the two chunks.

CPU note: the runner carry is a plain reference hand-off (`self.kv_caches = model_fn(...)[0]`; the next step
reads it back), so on CPU **A == B always** (`test_mla_dcp_cache_persistence.py::
test_runner_carry_preserves_chunk1_ABC`); the negative-control
`test_runner_carry_ABC_detects_broken_carry` proves the A-vs-B diff DOES flag a dropped carry (A != B), so
the localizer is a real discriminator — the metal probe is what decides which branch fired. Expectation from the CPU
analysis: **A == B on metal too** (the runner does not re-create/re-`device_put`/re-shard the cache between
steps), which would localize the residual loss to chunk-2's forward read.

**5d(iii). Scatter-vs-kernel isolation (`GLM_DCP_SCATTER_ONLY`) — when the correctness diff says DIFFER.** The
2chunk-vs-1chunk diff (5d) came back **DIFFER at exactly one logical page** (first_diff at P_g, contiguous
`[P_g, 2·P_g)` — the 2nd page of a 2-page chunk-1; page 0 and the chunk-2 region correct). That is a WRITE
corruption of a multi-page chunk's non-first page. **Note the CPU status:** the DCP owner-scatter is
CPU-correct at this exact geometry — 2-page chunk-1 across pack∈{1,2,4}, model∈{1,2,4}, dcp∈{2,4}, prompt = 4–5
pages (`test_mla_dcp.py::test_dcp_chunked_prefill_write_matches_single_chunk`, multi-page cases) — and the
kernel does not write under `history_only` (`kernel.py:2030-2034,2050`). So the corruption is a **metal-only**
effect. To pin whether it is the XLA owner-scatter *as executed on TPU* or the kernel's cache return, set
`GLM_DCP_SCATTER_ONLY=1`: the DCP path then returns the cache **straight after the owner-scatter, skipping the
attention kernel** (attention output is meaningless; debug only, default off → byte-identical). Dump both a
2-chunk and 1-chunk run under it and diff:

```bash
# same _ENVS as 5d, plus GLM_DCP_SCATTER_ONLY=1
env $_ENVS GLM_DCP_SCATTER_ONLY=1 GLM_DCP_CACHE_DUMP=/tmp/dcp_so_2chunk.npz \
  ~/vllm-env/bin/python -u glm_longctx.py --lengths 3200 --depths 0.5 --trials 1 \
    --max-seqs 1 --max-batched-tokens 2048 --gmu 0.90 --note "scatter-only 2chunk"
env $_ENVS GLM_DCP_SCATTER_ONLY=1 GLM_DCP_CACHE_DUMP=/tmp/dcp_so_1chunk.npz \
  ~/vllm-env/bin/python -u glm_longctx.py --lengths 3200 --depths 0.5 --trials 1 \
    --max-seqs 1 --max-batched-tokens 6144 --gmu 0.90 --note "scatter-only 1chunk"
PYTHONPATH=~/tpu-inference-dcppersist2 ~/vllm-env/bin/python -m \
  tpu_inference.runner.dcp_cache_diff /tmp/dcp_so_2chunk /tmp/dcp_so_1chunk
# DIFFER at page 1  -> the metal owner-scatter itself is wrong (XLA scatter execution
#                     on TPU for this shape) -> fix attention_interface.py:902-923
# MATCH             -> scatter is fine; the kernel's history_only cache RETURN corrupts
#                     the actively-processed boundary page -> fix kernel.py
```

CPU-covered: `test_mla_dcp.py::test_dcp_scatter_only_probe_returns_post_scatter_cache` asserts the probe returns
exactly the post-scatter cache (and zeros output); gate-off byte-identity by the existing
`test_gate_off_traced_jaxpr_byte_identity`.

**5d(iv). Fix — owner-scatter reformulations (`GLM_DCP_SCATTER_IMPL`), find the one that lowers correctly.**
The scatter-only diff (5d(iii)) came back **DIFFER @ page 1** (max|Δ|=5.44, first_diff=1024) with the kernel
skipped — so the pure owner-scatter output is wrong at the 2nd local page on TPU, while the CPU 264-case
ground truth (`test_mla_dcp_scatter_gt.py`) proves the arithmetic is correct. Diagnosis: the 4D
`.at[...].set(mode="drop")` (`attention_interface.py:927-965`) **mislowers on TPU when writing into the
DONATED, `P(BATCH,CONTEXT)`-sharded, TILED cache** — the 2nd local-page tile is not committed. `GLM_DCP_SCATTER_IMPL`
selects an alternative that is **bit-identical to the default on CPU** (all four verified) but lowers to a
DIFFERENT TPU op sequence; run the scatter-only diff under each until one flips **DIFFER→MATCH**:

**On-metal so far: `GLM_DCP_NO_DONATE=1` and `GLM_DCP_SCATTER_IMPL=flat` BOTH still `pred=None`** — so it is NOT
the donation and NOT the scatter's index form. The corruption is the **cross-tile GSPMD scatter itself**:
within one `dcp`-shard's local buffer, a single `.at[...].set` over all tokens writes MULTIPLE physical pages
(dim-0 tiles), and the 2nd page tile is not committed. The remaining candidate targets exactly that:

| value | what it does | reviewer / metal note |
|---|---|---|
| unset / `scatter` | current 4D indexed scatter (byte-identical) | baseline |
| **`shardlocal`** | split the local scatter into **one single-physical-page scatter per logical block** (`fori_loop` bounded to the step's active block range) so no scatter crosses a dim-0 page-tile boundary | **TOP candidate** — removes the cross-tile scatter that survived `no_donate`+`flat`; bit-identical on CPU incl. metal tiling + 32-way all_gather + multi-seq/decode (reviewer-verified). Caveats: single-tile is guaranteed only at **max_seqs=1** (the passkey; batched multi-seq decode reverts to multi-tile), and the per-iteration op is the same scatter primitive with data-dependent indices → this fixes the metal bug only if the mis-commit is sensitive to the runtime tile count (the observed "2nd tile not committed"), not a structural scatter bug. If it does NOT flip DIFFER→MATCH, the next lever is a dynamic_update_slice variant (genuinely different op). |
| `flat` | 1D-index scatter on a flattened cache | tried on metal → **still fails** (XLA re-tiles the 1D scatter identically) |
| `barrier` | `optimization_barrier` before the scatter | likely no-op (barrier on the input, not the donated output) |
| `onehot` | per-slot one-hot masked write, no scatter primitive at all (small caches only) | **diagnostic probe**: if onehot RETRIEVES → the scatter op is the culprit (shardlocal will work); if onehot ALSO fails → the values are wrong BEFORE the scatter (upstream — the token all_gather ordering or the new-KV latents), NOT the scatter |
| `GLM_DCP_NO_DONATE=1` | drop the step-fn kv_caches donation | tried on metal → **still fails** (not the donation) |

Plus a separate step-fn lever, `GLM_DCP_NO_DONATE=1` (the genuine option 1 — the scatter-input `barrier` likely does not achieve it): under the DCP gate it **drops the `kv_caches` donation** in `VllmModelWrapper.jit_step_func` so XLA must preserve the input cache buffer and the owner-scatter's OUTPUT can no longer be placed in-place in the donated, sharded, tiled buffer. Costs one transient cache copy/step (measure HBM at 128K; harmless at the L=3200 debug size). Compose it with any `GLM_DCP_SCATTER_IMPL` (or none):

```bash
# Try in this order (reviewer ranking). Each entry is the extra env for the two
# scatter-only dumps + diff. "no_donate" is GLM_DCP_NO_DONATE=1 (no IMPL).
for LEVER in "GLM_DCP_SCATTER_IMPL=shardlocal" \
             "GLM_DCP_SCATTER_IMPL=shardlocal GLM_DCP_NO_DONATE=1" \
             "GLM_DCP_SCATTER_IMPL=onehot"; do   # onehot = diagnostic (see table)
  TAG=$(echo "$LEVER" | tr -cd 'a-z_')
  env $_ENVS $LEVER GLM_DCP_SCATTER_ONLY=1 GLM_DCP_CACHE_DUMP=/tmp/so2_$TAG.npz \
    ~/vllm-env/bin/python -u glm_longctx.py --lengths 3200 --depths 0.5 --trials 1 \
      --max-seqs 1 --max-batched-tokens 2048 --gmu 0.90 --note "scatter $TAG 2chunk"
  env $_ENVS $LEVER GLM_DCP_SCATTER_ONLY=1 GLM_DCP_CACHE_DUMP=/tmp/so1_$TAG.npz \
    ~/vllm-env/bin/python -u glm_longctx.py --lengths 3200 --depths 0.5 --trials 1 \
      --max-seqs 1 --max-batched-tokens 6144 --gmu 0.90 --note "scatter $TAG 1chunk"
  echo "== $LEVER =="; PYTHONPATH=~/tpu-inference-dcppersist2 ~/vllm-env/bin/python -m \
    tpu_inference.runner.dcp_cache_diff /tmp/so2_$TAG /tmp/so1_$TAG
done
# The LEVER whose scatter-only diff is MATCH is the fix. Then re-run WITHOUT
# GLM_DCP_SCATTER_ONLY (full path) at that lever -> the 2chunk-vs-1chunk full
# diff must also MATCH, and the dcp=2 128K passkey must clear >=95%/depth.
# (onehot: run only if it's cheap enough at L=3200; it's a probe, not a fix.)
```

Once an IMPL wins, make it the DEFAULT under the DCP gate (drop the env, keep gate-off byte-identical) and
re-review. CPU-covered: `test_mla_dcp_scatter_gt.py` runs all four impls against the independent ground truth
(bit-identical); gate-off byte-identity unchanged.

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

*(§7 refreshed 2026-07-08 — the GLM_SPEC_K knob + `bench/mtp_m2_check.py` are LANDED and
CPU-tested; the old 7a merge prereq is OBSOLETE, see below.)*

From the M1 report (RESEARCH_LOG 2026-07-07 Stage-3 entry + docs/08 §M2). **Gate M2 (full
bar): for N≥32 prompts, generated token sequences EQUAL the non-speculative greedy baseline
exactly** (argmax-match acceptance ⇒ distribution-identical; exact equality can only break
on logit ties — the tie-flip protocol in docs/08). This step runs the **n=8 SEQUENTIAL
bring-up pair first** (`--batch-size 1`: one request in flight — single-sequence decode,
the simplest batch shaping and the cheapest first signal); scale to the N≥32 batched gate
only after n=8 is exact. M1's engine-scale checklist to verify during bring-up: (V2) draft
KV grouped with target MLA layers — eagle3 `prepare_inputs`'s last-group assumption must
degenerate correctly for a single unified group; (G3) the file-level draft filter against
the REAL gs:// safetensors index (expect ~1–2 of 150 files re-streamed, NOT 755.7 GB); FP8
draft load (the quantized path skips vLLM weights-tracking); per-bucket precompile of the
draft programs.

**7a. Branch state (verified 2026-07-08 via `git merge-base --is-ancestor`):**
`glm-5.2-v4-mtp-g4` (dense-MTP M1, G1–G4) **IS merged into `glm-5.2-v4-next`** (merge
commit `534cd74d7`) **and the OOB fix `02e44b36` is also an ancestor of `-next`** — both
coexist there from origin commit `886eaceb4` onward (local `-next` @ `cda8a707b` adds the
det determinism merge on top; workers pull from ORIGIN, so push `-next` first — step 2's
rule). The old 7a prereq ("merge `02e44b36` into `glm-5.2-v4-mtp`") is **OBSOLETE**: do
NOT run M2 from the frozen `-mtp` branch — use the step-2 staging engine (`-next`). No
fork-side gate is involved: MTP activates ONLY through vLLM's `speculative_config` (absent
⇒ byte-identical target path — the M1 forward-hash test's contract).

```bash
cd ~/tpu-inference && git merge-base --is-ancestor 89e1d5b5a glm-5.2-v4-next \
  && git merge-base --is-ancestor 02e44b360 glm-5.2-v4-next && echo "M2 prereqs on -next OK"
# engine still up from step 2 on -next? reuse it for the baseline. Otherwise:
TPU_INFERENCE_BRANCH=glm-5.2-v4-next bash ~/glm-tpu/scripts/sync_workers.sh
GLM_FLIGHT_RECORDER=1 TPU_MIN_TOKEN_BUCKET=32 bash ~/glm-tpu/scripts/launch_glm_32chip.sh
```

**7b. The spec knob is LANDED in `bench/engine.py` (no code to write on the pod):**
`GLM_SPEC_K=k` → `speculative_config={"method": "mtp", "num_speculative_tokens": k}` in
the `LLM(...)` args. Kwarg + dict shape verified against the installed vLLM
(`~/vllm-build/vllm/engine/arg_utils.py:616` `EngineArgs.speculative_config: dict | None`
→ `create_speculative_config` → `SpeculativeConfig(**dict)`; `"mtp"` is a valid method,
glm_moe_dsa config surgery → `DeepSeekMTPModel`/`n_predict=1`, and k=5 passes the
`k % n_predict == 0` module-reuse check). The fork routes method `"mtp"` →
`Eagle3Proposer` (`tpu_runner.py:711`). Unset/0 = kwarg ABSENT, engine args byte-identical
(unit-tested: `test_bench.py::test_spec_engine_kwarg`). GLM_SPEC_K is a DRIVER-side env
(build_llm reads it in-process — no raylet baking needed) and lands in `runs.env_json`
via the GLM_* os_env sweep; the engine-built log line prints `spec=mtp:k=<k>`.

**7c. Baseline (spec OFF) then k=1 — n=8 sequential, token-identity compare** (docs/08:
start k=1 — single trace, simplest shaping; `GLM_LOG_STATS=1` in BASE is REQUIRED for the
acceptance-stats scrape in `mtp_m2_check.py --log`):

```bash
cd ~/glm-tpu/bench && set -a && . ~/glm-tpu/.env && set +a
BASE="NEW_MODEL_DESIGN=1 MODEL_IMPL_TYPE=vllm TPU_MULTIHOST_BACKEND=ray OMP_NUM_THREADS=1 \
HF_HUB_DISABLE_XET=1 TPU_DISABLE_DSA_INDEXER=1 DISABLE_WEIGHT_REQUANTIZATION=1 \
REQUANTIZE_WEIGHT_DTYPE=float8_e4m3fn TPU_MIN_TOKEN_BUCKET=32 GLM_TP=32 \
GLM_ASYNC_SCHED=0 GLM_LOG_STATS=1 GLM_FLIGHT_RECORDER=1"
env $BASE ~/vllm-env/bin/python -u run_bench.py --benchmark gsm8k --limit 8 \
  --max-len 4096 --max-new 1024 --max-seqs 8 --num-gpu-blocks 0 --gmu 0.90 \
  --max-batched-tokens 512 --batch-size 1 --note "M2 baseline non-spec n=8 sequential" \
  > ~/glm-run/m2_baseline.log 2>&1
env $BASE GLM_SPEC_K=1 ~/vllm-env/bin/python -u run_bench.py --benchmark gsm8k --limit 8 \
  --max-len 4096 --max-new 1024 --max-seqs 8 --num-gpu-blocks 0 --gmu 0.90 \
  --max-batched-tokens 512 --batch-size 1 --note "M2 spec-decode mtp k=1 n=8 sequential" \
  > ~/glm-run/m2_k1.log 2>&1
# identity gate — the two newest runs; roles are oriented from env_json GLM_SPEC_K, never
# from argument order. Exit 0 = PASS, 1 = mismatch (tie-flip protocol), 2 = not comparable.
~/vllm-env/bin/python mtp_m2_check.py --latest --log ~/glm-run/m2_k1.log
# (explicit ids instead of --latest:
#  sqlite3 results.db "SELECT run_id,note FROM runs ORDER BY run_id DESC LIMIT 4")
```

Watch in `m2_k1.log`: the engine-built line must say `spec=mtp:k=1`; the G3 filter log
(only layer-78-bearing files re-streamed — a full 755.7 GB re-stream = G3 failed open →
fix before calling M2, it fails OPEN by design). KV note: MTP adds a draft KV group —
auto-size (`--num-gpu-blocks 0`); expect fewer blocks than 128.

**7d. k=5 vs the SAME baseline** (per-run pair; `--latest` no longer applies — pass ids):

```bash
env $BASE GLM_SPEC_K=5 ~/vllm-env/bin/python -u run_bench.py --benchmark gsm8k --limit 8 \
  --max-len 4096 --max-new 1024 --max-seqs 8 --num-gpu-blocks 0 --gmu 0.90 \
  --max-batched-tokens 512 --batch-size 1 --note "M2 spec-decode mtp k=5 n=8 sequential" \
  > ~/glm-run/m2_k5.log 2>&1
~/vllm-env/bin/python mtp_m2_check.py <k5_run_id> <baseline_run_id> --log ~/glm-run/m2_k5.log
```

Same identity bar, plus record the acceptance stats the checker scrapes from the log
(vLLM's interval `SpecDecoding metrics:` lines — mean acceptance length, per-position
rates; M3's ≥~4.5 acceptance-length target starts here; exact `SpecDecodingStats`/
`get_metrics` aggregation into the DB is an M3 deliverable — at minimum capture the tok/s
A/B the checker prints from the summary notes). **Isolated mismatches**: apply the docs/08
tie-flip protocol (a logit tie at the mismatch position — top-2 gap below bf16 eps —
exonerates spec-decode; anything else is a real bug). **3/3 for the k that passes**, then
scale the passing k to the N≥32 batched gate (`--limit 32 --batch-size 0 --max-seqs 16`,
same checker) — the docs/08 M2 bar proper.

---

## §8 GLM_DSA_DCP sparse ladder (post-granularity-fix)

*(Added 2026-07-10. This replaces §5's DCP debugging as the active frontier. Prereqs: the DENSE 128K gate
(run 124, n=77) has finished and is recorded; the Stage-C adversarial review has returned its verdict —
it is the hard gate before any of this touches metal. The sparse-DCP stack (Stages A `ed2a021be` + B
`4f390a61e`/`e7c239c91` + C `6f8855c3f`) is **CPU-certified only** — Stage B reviewed SAFE-FOR-METAL-LADDER,
gate-off jaxpr byte-identical to HEAD, but ZERO sparse-DCP tokens have ever run on metal. This ladder is
where that changes, one falsifiable rung at a time.)*

**Why a ladder at all:** sparse@128K under DCP is NEW DISTRIBUTED CODE, not "32K with a bigger number".
DCP shards the context, so a per-chip indexer scores only its stripe — local top-2048 ≠ global top-2048.
The stack does: per-shard score → local top-min(k, S_local) → all-gather candidates → position-sort →
global top_k (elementwise-exact vs single-chip incl. tie order, gather-order-invariant by construction) →
owned-subset attend → `_dcp_lse_combine`. Each rung below isolates one metal surface CPU cannot certify.

**8a. Relaunch env (identical for every rung; only GLM_DCP + harness args vary).**

```bash
# 0) SYNC FIRST — the stale-worker rule (workers 1-7 ran 8802ebab all night once):
cd ~/tpu-inference && git push origin glm-5.2-v4-next
TPU_INFERENCE_BRANCH=glm-5.2-v4-next bash ~/glm-tpu/scripts/sync_workers.sh
# ABORT unless all 8 lines print tpu-inference @ THE SAME hash (currently 6f8855c3f).

# 1) Relaunch with the sparse-DCP envs raylet-baked (LIBTPU quoting: inner double quotes,
#    exactly as below — verified in the live raylet /proc environ on the dense gate):
EXTRA_ENVS='GLM_MLA_DCP=1 GLM_DSA_MODE=pallas_decode GLM_DSA_DCP=1 GLM_DCP_SCATTER_IMPL=pageloop GLM_DSA_SCORER=xla LIBTPU_INIT_ARGS="--xla_latency_hiding_scheduler_rerun=5 --xla_tpu_rwb_fusion=false"' \
  TPU_MIN_TOKEN_BUCKET=32 bash ~/glm-tpu/scripts/launch_glm_32chip.sh
# GLM_DSA_SCORER=xla and SCATTER_IMPL=pageloop are the Stage-B review's ladder settings
# (pallas scorer + other impls come later, one variable at a time). Optionally add
# GLM_EXPECT_CODE_HASH=6f8855c3f to EXTRA_ENVS — a mismatched worker then RAISES at init.

# 2) VERIFY the raylets actually carry the envs on ALL 8 hosts (EXTRA_ENVS lesson):
gcloud compute tpus tpu-vm ssh db-v4-64-od --zone us-central2-b --worker=all --command \
  'P=$(pgrep -f raylet | head -1); tr "\0" "\n" < /proc/$P/environ | \
   grep -E "GLM_MLA_DCP|GLM_DSA_MODE|GLM_DSA_DCP|GLM_DCP_SCATTER_IMPL|GLM_DSA_SCORER|LIBTPU_INIT_ARGS"'
# ABORT unless every host prints all six, identically. Also read the per-worker
# code_fingerprint log lines at TPUWorker init (git hash + GLM_* env names) — 8× identical.

# 3) Driver env: mirror the raylet envs + the standard block; ALWAYS setsid the driver:
cd ~/glm-tpu/bench && set -a && . ~/glm-tpu/.env && set +a
_S8="NEW_MODEL_DESIGN=1 MODEL_IMPL_TYPE=vllm TPU_MULTIHOST_BACKEND=ray OMP_NUM_THREADS=1 \
HF_HUB_DISABLE_XET=1 TPU_DISABLE_DSA_INDEXER=1 DISABLE_WEIGHT_REQUANTIZATION=1 \
REQUANTIZE_WEIGHT_DTYPE=float8_e4m3fn TPU_MIN_TOKEN_BUCKET=32 GLM_TP=32 GLM_ASYNC_SCHED=0 \
GLM_LOG_STATS=1 GLM_MLA_DCP=1 GLM_DSA_MODE=pallas_decode GLM_DSA_DCP=1 \
GLM_DCP_SCATTER_IMPL=pageloop GLM_DSA_SCORER=xla"
# per-run: env $_S8 GLM_DCP=<n> setsid nohup ~/vllm-env/bin/python -u ... </dev/null > log 2>&1 &
```

House rules for every rung: `GLM_DCP_ASSERT_SHARDING=1 GLM_DCP_ASSERT_CACHE_SANITY=1` on diagnostic runs
(they now RAISE on internal failure instead of failing open); after any dump run,
`~/vllm-env/bin/python -m tpu_inference.runner.dcp_dump_check <prefix>` (accepts the .npz-suffixed
runbook form) — "requested but nothing written" is an ABORT, not a shrug; **never grep bare "PASS"**
(`hlo_passes.cc` matches — read the `[longctx] ... correct=True/False` lines); results.db rows or it
didn't happen.

**8b. The ladder — rungs, exact configs, expected outcomes.**

1. **emit_lse unit @ dcp=1, single chip** (pod idle; `TPU_VISIBLE_DEVICES=0`, the §2a/§2b pattern):
   run the sparse-decode kernel with `emit_lse=True` on real MXU at small shapes.
   **Expect:** Mosaic ACCEPTS the lse out tile; attention output BITWISE vs the non-lse kernel;
   lse finite and matching the XLA oracle. **This is the #1 metal risk** (watchlist below) — if Mosaic
   rejects the `[1,H,128]` lse block, apply the reviewed fallback (widen the m/l scratch to `[H,128]`
   dsv4-style, or emit m/l separately) BEFORE climbing further. Save the log to `docs/artifacts/`.
2. **Sparse decode @ dcp=2, prompts ≤2048 tokens, vs dcp=1** (≤ index_topk AND one prefill chunk ⇒
   only the Stage-B decode path fires — the ctx>topk masked-prefill path stays out of the frame):
   same prompts at `GLM_DCP=2` and dcp=1 (gate on, both).
   **Expect:** logprobs match dcp=1 within the D0 tolerance AND the IndexShare stash (selected indices)
   is IDENTICAL — the distributed selection must reproduce the single-chip selection elementwise.
   Any divergence here is in select/attend/combine, at the cheapest possible size.
3. **Chunked prefill @ dcp=2, 4–6K prompt** (mbt 2048 ⇒ 2–3 chunks — Stage C's masked-prefill path fires)
   **+ the step-HLO honesty check**: run with `GLM_DUMP_STEP_HLO=1` and inspect what IS dumped — the
   repaired tool now says loudly when a program compiled inline and was NOT dumped (absence must be
   visible, not assumed). **Expect in the HLO:** exactly ONE candidate all-gather pair (scores+positions)
   per FULL indexer layer (IndexShare amortizes the other 3), the LSE combine as pmax + 2·psum, and
   **NO whole-cache collectives** — a `gather_kv_segment`-style full-cache all-gather is the flagged
   structural conflict and an instant ABORT (it would "work" while silently defeating CP).
   Needle must be correct (this size was pred=None under the granularity bug — it also re-proves the fix
   composes with sparse).
4. **32K selected-set dump, dcp=2 vs dcp=1** — Gate-2b ON METAL: dump the selected sets for identical
   32K prompts at dcp=2 and dcp=1. **Expect: SELECTED-SET-EXACT (elementwise, tie order included)** —
   the free regression the union argument guarantees. DIFFER here = the distributed top-k on metal
   (collective ordering / packed scores), NOT attention — fix before any passkey claim. Tooling:
   `GLM_DSA_DUMP_TOPK=<prefix>` (raylet-baked, all hosts) dumps the per-step stashed topk indices;
   `python -m tpu_inference.runner.dsa_topk_diff <prefix_dcp2> <prefix_dcp1>` compares (exit 0 MATCH /
   1 DIFFER / 2 no-verdict on partial coverage; also asserts cross-shard replication within each run).
   Rungs 4+6 are the CHEAP DISCRIMINATORS: only launch rung 7 (~10-19 h) after both are green — an
   output-only pass can hide a subtly-wrong merge that still lands plausible needles.
5. **32K sparse passkey @ dcp=2 then dcp=4 (smoke, ~6–9 needles).** **Expect 100%** — dcp=1 sparse
   already holds 72/72 at 32K; any miss is distributed-stack regression, not capacity.
6. **64K sparse smoke** (dcp=2/4, few needles across depths incl. 0.0/1.0). **Expect:** clean build
   (KV/dcp shrinks the pool — the frag lottery is not in play at dcp≥2) + all correct. First sparse
   retrieval above 32K on any hardware.
7. **THE 128K SPARSE GATE — n≥73 zero-failure** (the ≥95% Wilson bar; mirror the dense gate: depths
   {0.0,0.05,0.25,0.5,0.75,0.95,1.0} × 11 = 77 needles, `GLM_DCP=4`, pool 68 blocks, chunk 2048,
   gmu 0.90, max-len 131840, ~19 h — budget an overnight, arm a watchdog on `correct=False`/driver
   death). Depths 0.0/0.05 are THE mechanism cells (needle at max range must survive top-2048 selection
   out of 128K); 0.75 passes nearly by construction. **Smoke the 4 mechanism depths ×1 first** (~1 h) **GATE ARITHMETIC (owner-pinned): 77/77 → Wilson LB ~95.3% — clears ≥95% by a hair; a SINGLE
   miss → 76/77 → LB ~91% → the gate FAILS. On one miss: EXTEND the same run to n≈130 total (129/130
   recovers the bar) — never round, never rerun-until-green.**
   before committing the long run. Record as run rows + summary; SMOKE ≠ GATE.

**8c. Metal-risk watchlist (from the Stage B/C reviews — what CPU certification CANNOT see).**

- **scan-with-lse lowering (rungs 1 & 3):** the decode kernel lane-broadcasts `[H,1]→[H,128]` for the
  lse out block, and Stage C's masked-prefill flash scan carries (m,l) through the scan — interpret mode
  hides layout/DMA acceptance entirely. Top risk; reviewed fallback in rung 1.
- **Candidate all-gather arena (rung 3+):** `[T, dcp·k]` score+position candidates = 64 KiB/token at
  dcp=4 ⇒ **~134 MB/layer/chunk transient at chunk T=2048** (Stage-B review measured ~50 MB incl. the
  argsort arena at T=512; scales linearly). IndexShare amortizes it 1-in-4 layers, but watch
  CompileTimeHbmOom at 128K — if it OOMs, drop mbt/chunk to 1024 before touching anything else.
- **Per-shard flash transient (rung 3):** each shard's masked-prefill scan materializes its chunk×selected
  working set post-gather; sized on CPU only structurally. Watch the compile-time HBM breakdown
  (deepsea_compiler_util lines) on the first 4–6K prefill.
- **Replicated-q boundary:** q/ql/hidden/positions enter the three shard_map sites replicated `P()` —
  on metal, GSPMD may insert a reshard/copy at the boundary per step. Not a correctness risk (Guard-1
  asserts sharding) but a throughput cliff to note in the step timings before the 256K A/B.
- Scatter-in-cond-in-shard_map (Stage B's flagged composition) + the donated striped caches remain the
  historically metal-treacherous class — that is WHY the ladder validates caches bitwise at rung 2/4
  before any long run.

**Abort discipline:** any rung failing → STOP the ladder, instrument (the obs kit is repaired and
fail-loud now), localize with the smallest discriminator — exactly the method that found the granularity
bug. Do NOT tune past a failure with fragmentation/lottery luck; 3/3 for every gate claim.

---

## After the sequence

Log every step in `docs/RESEARCH_LOG.md` (numbers + honest nulls), refresh `HANDOFF.md`,
commit + push, and run `bash scripts/backup_bundle.sh` again. The next frontier after this
runbook: the §8 sparse ladder → 256K throughput A/B (dsa-sparse vs dense, dcp≥4; fp8-KV
only after its own dcp=1 needle validates — fp8 has never generated a metal token), then
benchmarks at scale (GSM8K n≥200, GPQA-198 @16K) and the M3 acceptance instrumentation
(MTP frozen until the gates close).

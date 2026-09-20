# Goal — Publish the retained ordinary engine as the usable stable release

The owner stopped optimization research, then requested that main actually run
the retained approximately 14.3 tok/s ordinary engine. A documentation-only merge
does not meet that goal. Follow the earlier supported-code extraction and curation
at c7f6d42f / 493b67de; preserve research history and protected originals.

Scope: promote D1 grouped experts with the empty-slot fix, D8 BF16 non-routed
weights, D10 DSA, D4 packed delivery, and D8/P1/P2 prefill. The measured profile is
greedy decoding with 8,192 total prompt/output slots. Preserve the legacy sampled
long-context entry explicitly; never silently substitute its sampling or capacity.
MTP, decode D5, fused reductions, global-max and fused EP remain excluded.

1. Extract only supported dependencies from pinned, trained research code.
   Keep frozen MODEL_SOURCE unchanged. Prove numerical/host interfaces on CPU.
2. Connect private request preparation, protected fleet launch, real checkpoint
   loading, fresh graph/memory admission, token delivery and authenticated cleanup.
3. Run one bounded real-weight release validation through the actual new entry.
   Check the DB610 reference prefix, all-rank agreement, timing and memory. This
   validates integration; it does not restart cancelled answer or MTP campaigns.
4. Record exact source, checks and receipts. Main README must describe the usable
   release and its measured limits, with future work at most two sentences at the
   end. Put detailed failed/superseded research in the retained history index.
5. Update file dispositions, pass release checks, resolve review findings, verify
   regional backup, and merge the implementation into private main. Report actual
   publication state. Self-review is not independent review.

Current worktree: /home/gianl/glm-tpu-optimized-release, branch
release/optimized-ordinary-20260920, from main e9ef0dda. Candidate extraction and
protected launch integration are in progress, not yet TPU-admitted or merged.
Research remains recoverable at e3290fd8 and the pre-cleanup preservation ref.
See docs/release/OPTIMIZED_PROMOTION.md for provenance and gates.

Existing pod only: db-v4-64-od, us-central2-b, eight hosts / 32 v4 chips. One
workload under both workload leases; respect both sync leases during staging.
Authenticate all eight hosts idle before and after. No automatic workload retries,
resource lifecycle operations, environment upgrades or full-model safety copies.
pytest always uses JAX_PLATFORMS=cpu. Keep private inputs, weights and raw output
outside Git. Only gs://driftbench-dsv4-uc in US-CENTRAL2; retain storage bounds.
Commit/push private release milestones without asking; no force-push, history
rewrites, Co-Authored-By lines, subagents or external reviewers.

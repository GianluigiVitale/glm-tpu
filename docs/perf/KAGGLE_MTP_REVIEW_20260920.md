# Kaggle MTP implementation comparison — 2026-09-20

This follow-up is a source review prompted by the owner, not a new TPU run or
an independent parity proof. Our completed experiment establishes the measured
behavior of our implementation, not a hardware ceiling or proof that MTP cannot
help GLM-5.2. Long-output equivalence remains unresolved.

## What was previously checked

The earlier [reference survey](REFERENCE_LOWHANGING_FRUIT_20260919.md) identifies
both model folders, Qwen's MTP speedup and its rollback patch. The native MTP
[source audit](mtp-source-audit-20260919.json) instead pins GLM-5.2's upstream
vLLM implementation. Neither document establishes a prior detailed comparison
of both Kaggle speculative loops with our completed implementation. That review
gap should not be covered by a claim that the implementations were already proved
equivalent.

## Sources inspected now

Local clone `/home/gianl/reference-repos/kaggle-tpu-lab`, immutable revision
`1aa1f083ae253470d9f355fe9c3eb12003300e91`, contains both folders:

- [Qwen README](https://github.com/ARahim3/kaggle-tpu-lab/blob/1aa1f083ae253470d9f355fe9c3eb12003300e91/qwen38-27b/README.md),
  `kernel/serve_qwen38.py` configuration/launcher and
  `patches/mtp-rollback-v0280.diff` state, batch-layout and runner changes.
- [GLM resident engine](https://github.com/ARahim3/kaggle-tpu-lab/blob/1aa1f083ae253470d9f355fe9c3eb12003300e91/glm53-flash/engine/glm53/resident.py),
  especially `prefill`, `_prog_mtp`, `_run_verify`, `spec_decode` and rollback;
  `model.py::mtp_layer/rollback_states`, `tests/test_spec_decode.py`, and
  the serving scheduler/launcher call sites.
- Our `mtp_draft.py`, `mtp_state.py`, `speculative_request.py`,
  `speculative_verify.py`, `speculative_moe.py`, `speculative_experts.py`,
  existing prefix/native tests and [completed fleet receipt](tpu-real-native-suite-20260920T031727Z.json).
- The retained pinned vLLM proposer and GLM/DeepSeek MTP norm/IndexShare code
  described in [the state contract](MTP_STATE_CONTRACT_20260919.md).

## Findings

1. **Qwen's actual paired MTP claim is 78 to 104 tok/s (+34%).** Its later
   approximately 130 tok/s uses a changed token-bucket configuration; comparing
   130 directly with the old 78 does not isolate MTP. Default proposal length
   is three drafts. Our completed native experiment used one and two drafts.
   Qwen's model is dense and predominantly gated-DeltaNet; its speedup is not a
   transferable multiplier for our routed-MoE/DSA model and 32-chip layout.
2. **Their critical correctness fix is real.** The Qwen patch retains recurrent
   state checkpoints for verification positions, selects the accepted checkpoint
   using offsets attached to physical request slots, and handles verification
   windows separately from one-token decode and prefill batches. The README
   reports 0/12 greedy matches before and 12/12 after. GLM-5.2 here has no GDN
   state to patch, but the accepted-prefix invariant applies. Our implementation
   restores rejected physical KV/index rows and accepted selection/frontier state;
   this is implemented and CPU-tested, not independently proved on long trained
   runs by that test alone.
3. **GLM's advertised 64 tok/s is not evidence of an MTP speedup.** This pinned
   serving path calls `decode_rows`; neither its launcher nor scheduler installs
   MTP or calls `spec_decode`. A separate MTP implementation and tiny-model tests
   do exist. Those tests compare full greedy token trails with random drafts
   (mostly rejection) and oracle drafts (acceptance), including snapshot restore.
   They do not establish a trained 64 tok/s speculative benchmark.
4. **Our high-level protocol matches the references in several important ways.**
   Draft inputs pair the next token's embedding with the previous position's
   normalized hidden state; verification consumes pending token plus drafts in
   one layer-major pass; acceptance takes the matching prefix and a target
   correction/bonus. Recurrent MTP uses post-final-norm hidden. Our GLM-5.2 path
   also reuses the shortlist on later IndexShare iterations, as its pinned upstream
   proposer specifies. No missing basic accept/reject step was found in this review.
5. **The draft-cache policies are not identical.** The Kaggle GLM loop retains
   draft cache writes, restores small rejected tails, and carries one selected
   target hidden row to the next round. Ours refreshes accepted rows from the
   committed draft root using target hidden states and shifted target tokens,
   caching the next first draft. That follows the inspected GLM-5.2 proposer
   contract; the difference alone does not prove a bug. It needs trained draft
   parity before either policy is described as interchangeable.
6. **Our target verifier is expensive and output equivalence is unproved.**
   The final verifier groups expert rows into M8 weight-reuse panels, but keeps
   per-row dense dots, DSA selection and several reductions to limit numerical
   changes. It is a real multi-row verifier, yet much work still scales with rows.
   First structured R3 run: about 114.13–114.19 s verification out of 128.689 s
   decode wall (~89%); draft 1.24–1.27 s and refresh 2.762–2.767 s combined are
   only ~3.1%. It accepts 2.924 tokens/round out of a maximum three, so poor
   drafting does not explain that case's small speedup. These are recorded timings,
   not a new trace or proof that one particular kernel causes the cost.

## Correctness gate and next diagnostic

A wrong draft should increase rejection, not change greedy output, provided the
verifier computes the same target function and commits the correct state.
Our longer outputs diverge (code at token index 5; prose at 6). Calling this a
floating-point boundary records the observation but does not isolate its cause.
The review found no conclusive new rollback or token-shift bug; it also does not
rule out one. Neither high acceptance nor DB610's short match proves correctness.

The next useful diagnostic is a short, teacher-forced replay at the first
mismatch: start ordinary and multi-row verification from the same committed
state, feed the same tokens, compare each row's target choice, logit margin,
selected keys and accepted-prefix cache. Force oracle drafts and rejection at
each position to separate verifier arithmetic from acceptance/rollback; locate
the first differing layer before another long speed test. Compare native draft
hidden/logit outputs against the pinned trained GLM-5.2 reference separately.
Then measure candidate changes to row batching, reductions and cache handling;
three drafts/four verifier rows require their own correctness and memory gates.

No runtime changed and no TPU workload started for this review. Keep the existing
measurements, but do not interpret experiment completion as implementation
correctness or exhaustion of the available speedups.

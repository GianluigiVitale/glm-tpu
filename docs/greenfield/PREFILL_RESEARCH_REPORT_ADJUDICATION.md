# Prefill research report — local adjudication, 2026-09-08

Source: [owner-supplied report](research_report_prefillv4.md), read in full,
354 lines / 58,811 bytes. External citation handles embedded in that export are
not independently resolvable evidence here. Its bibliography supplies source leads;
external implementation claims still need pinned-source verification before reuse.
This note does not change the numerical contract, authorize a TPU launch, or set
performance targets. Implementation baseline: `b5ff369c` plus current launch WIP.

## Decisions that change the next engineering work

1. **Keep the immediate B128 layer6 discriminator.** DB590 acquired its four graphs;
   it did not execute them. Finish bounded numerical admission, then equivalent-work
   timing and actual routing statistics. Do not run a full decoder merely to learn
   whether its layer window is correct. B128 is a control point, not a final size.
2. **Investigate B512, then B1024 conditionally.** Preserve <=32-row attention/DSA
   tiles and the existing M64 repair boundary. Larger routing windows need their
   own executable-size, runtime scratch, per-row numerical and cache proofs. Do not
   merely raise row guards and extrapolate the existing MoE phase speedup.
3. **Measure row-tile reuse, not just window size.** Current
   `kernels/pallas/prefill_grouped_fp8.py` uses grid
   `(tiles_n, active_tiles, tiles_k)` with row_tile8 at the current callsite. Its
   weight block map depends on expert/N/K; each row/group tile traverses K. Source
   does not establish one HBM load of each expert per entire B512 window. Whether
   the compiler retains/reuses weight blocks across those traversals needs actual
   lowered scheduling/trace evidence. Compare 8/16/32 row tiles with fixed grouped
   inputs only when numerical/VMEM constraints permit; these values are already
   accepted by the primitive, not yet admitted as changed full-model variants.
4. **Reuse existing causal DSA.** `kernels/prefill_dsa.py` already uses bounded
   key slabs, conditional future-only scoring/selection bypass, exact running
   top-k and expert8 candidate exchange. It still visits every allocated tile's
   metadata, and its merge/sort cost is unmeasured. Decompose score, selection and
   gather before proposing a replacement. Smaller key tiles may increase merge
   frequency: K256/512 is a hypothesis, not an automatic improvement.
5. **Capture actual router metadata compactly.** Save rows/expert, actual active
   row/group tiles (including boundary revisits), useful/padded lanes, owner totals
   and imbalance. A sum of `ceil(rows/expert/8)` need not equal this kernel's tile
   count because adjacent groups share aligned row tiles. Real weights with
   synthetic activations still do not establish natural full-prompt routing;
   clearly separate that discriminator from statistics of a genuine model prompt.

## Corrections and limits

- The 5.57GB/token B128 and 1.42GB/token B512 traffic estimates assume independent
  uniform routing and one weight load per active expert/window. Their ceilings
  are conditional scenario bounds, not universal proofs against B128 or measured
  traffic of our current kernel. Skew changes reuse and owner critical paths.
- The report's 3K–12K tok/s bands are unmeasured hypotheses. Dividing useful FLOPs
  by a chosen peak-efficiency percentage omits selection, movement, synchronization
  and real memory scheduling; it does not produce a reliable pessimistic bound.
- The final diagram puts target setting AFTER 256K. Reject that ordering: §24
  requires quantitative prefill/TTFT preregistration before candidate performance
  trials. Bounded baseline measurements can inform those targets; successful
  results cannot retroactively determine their own acceptance bar.
- Host `TraceAnnotation` surrounding one compiled call cannot subdivide its
  internal device execution by itself. Reuse actual HLO/kernel identities and
  device timelines; separately compiled phase probes may perturb fusion and are
  diagnostic, not an additive decomposition of the original whole-layer wall.
- Preserve feature4 FP32 reduction BEFORE BF16/SwiGLU. A fused upstream GMM1
  activation using different sharding cannot be imported unchanged. No new pack,
  full BF16 weight expansion, speculative live-environment upgrade or platform swap.
- The final quality requirements remain four NEW-path L7 depths plus full L8 and
  the changed path's own competitive short-context proof. One 128K smoke is an
  early discriminator, not completion of four-depth coverage.

## Immediate engineering boundary

Current numerical-only launcher/collector changes are uncommitted WIP. Independent
review found unvoted final journal/report failures that could strand peers at the
next rendezvous. The fix uses existing matched fleet phases and original-output
preservation; composed CPU tests include final-publication/peer failure and all8
worker→files→generation collector→DB. This is necessary integration work, not a
new mathematical proof campaign or a reason to repeat cleared kernel tests.

Next: complete that review/test/persistence, one protected selected-layer numerical
discriminator, then equal-work phase attribution and routing/grouping decisions.
No full-model speedup, delivered TTFT, B512 admission or 10K promise is established.

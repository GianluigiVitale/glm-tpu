# Frozen — GLM-5.2 TPU performance research

Owner stopped the optimization campaign on 2026-09-20. Do not resume experiments, diagnostics, queued work, wider MTP, or the cancelled answer comparison. This supersedes earlier execution goals and pending-work lists.

The final answer-suite run `perf_real_ordinary_suite_20260920T152417Z` was stopped by authenticated process identity on all eight hosts. Controller cleanup completed at 15:43:39 UTC; all eight hosts were idle. No answer case completed. The short profiling-off/device-acceptance control was prepared but never launched. Neither is a performance result.

Frozen research baseline: approximately 14.3 accepted wall decode tok/s; 138.85 prompt tok/s at 2,034 tokens in the qualified short request-loop trial. D1/D8/D10/D4 and D8/P1/P2 are the retained research path. This is not general model-quality certification or deployment into the frozen release engine.

MTP R2/R3 long outputs diverge and are not qualified for serving. Global-max attention improved a synthetic primitive but failed trained verifier parity; its 2K prefill was 135.13 tok/s versus 140.59 in a separate ordinary run. Public fused EP failed v4 VMEM admission: 35.39 MiB required, 16 MiB available. No qualified new end-to-end gain was established by those candidates.

Research is frozen. Detailed history and all measured comparison rows are in [the research record](docs/perf/frozen-20260920/RESULTS_AND_DECISIONS.md). Verified regional archives preserve the removed raw logs and receipts; all 11 notebooks remain as historical Markdown. Cleanup removed 1,397 exact archived log copies and 103 obsolete receipt files. Weights, original private outputs and protected DB616–621 evidence remain intact. No new TPU workload, environment change or infrastructure operation is authorized by this file. Publication/check details are in [the freeze record](docs/perf/frozen-20260920/FREEZE_OPERATIONS.md).

Pre-cleanup history: `preserve/glm52-tpu-research-freeze-20260920` at `9dedce4b`. Experimental implementation remains on the private perf branch; MODEL_SOURCE remains `edecdd94`. Main publication must state whether it contains documentation or implementation.

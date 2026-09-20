"""Opt-in performance challengers for the WS32 engine (CPU-proven, not TPU-admitted).

This package lives outside the frozen ``MODEL_SOURCE`` pin
(``glm_tpu/greenfield/{kernels,runtime,sharding}``): importing it changes no
sealed source identity, and nothing in the supported user path imports it.
Every function here mirrors a frozen WS32 body with one structural change
that the reference engines in ``ARahim3/kaggle-tpu-lab`` showed to matter on
TPU, and each is proven bitwise (or documented-boundary) equal to the frozen
body on the forced 32-device CPU mesh by ``tests/perf``.

Nothing here is admitted for a protected run: a TPU acquisition of the new
graphs (HLO, memory, real-layer timing) is still required before any of it
replaces the sealed decode or prefill programs.  See
``docs/research/glm52-tpu-20260920/history/REFERENCE_LOWHANGING_FRUIT_20260919.md``.
"""

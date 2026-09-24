"""Pallas kernel names: the ``name=`` of every production ``pallas_call``.

The name labels the kernel's custom call in the lowered program and its launches in TPU profiles.
Kernels whose shape varies append their parameters to the base name (``fp8_routed_projection``
adds ``_s{slots}_t{tables}_k{K}_n{N}_tn{tn}_tk{tk}``, ``sparse_mla`` adds
``_h{heads}_k{K}_b{block}_w{width}`` and ``_prefill_m{rows}``), so every compiled variant has
its own name.

Kernels look the names up when they build their ``pallas_call`` (at trace time) and never bind a
value at import: the equivalence harness replaces the values in place while it lowers
(``tools/equivalence/kernel_renames.toml``).
"""

from __future__ import annotations

KERNEL_NAMES: dict[str, str] = {
    # fp8_grouped_matmul/kernel.py: decode routed-expert projection over scalar-prefetched routes
    "fp8_routed_projection": "fp8_routed_projection",
    # fp8_grouped_matmul/panel_kernel.py: prefill expert-panel matmul
    "fp8_expert_panel_matmul": "fp8_expert_panel_matmul",
    # sparse_mla/kernel.py: sparse MLA over the selected KV rows
    "sparse_mla": "sparse_mla",
    # sparse_mla/partial_kernel.py: owner-local partial attention with log-sum-exp outputs (prefill)
    "sparse_mla_partial_attention": "sparse_mla_partial_attention",
}

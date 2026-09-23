"""The admitted production prefill program: resident BF16, owner-local LSE, one-pass selection.

Since the S2d fold the composition is explicit production code (``optimized/prefill*.py``):
``prefill.build_ws32_batched_prefill_program`` builds it, calling the resident BF16 projections,
the owner-local LSE attention and the one-pass DSA selector directly; no function is rebound.
"""


def build_ws32_prefill_challenger_program(mesh, config, **options):
    """Build the resident-BF16, local-LSE, one-pass-selection prefill program (B128 / B114)."""
    from .prefill import build_ws32_batched_prefill_program

    return build_ws32_batched_prefill_program(mesh, config, **options)

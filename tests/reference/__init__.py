"""Unsharded single-device pure-JAX reference forward of GLM-5.3.

The reference restates the model on one device over the dequantized frozen
fixture weights: absorbed MLA, the DSA lightning indexer with exact top-k,
IndexShare reuse, noaux_tc routing, SwiGLU experts and the greedy head. It
never calls production ``shard_map`` or Pallas code; where an oracle function
of the engine's reference modules (``glm_tpu/layers``, S3) is exact it is reused as is.

``model.py`` composes the layers in ``linear.py``, ``norm.py``,
``attention.py``, ``dsa.py`` and ``moe.py``. ``oracle_run.py`` is the 32-device
CPU child that compares the reference with the production composition (until S2f
also with the frozen FP8 oracle; ``floors.json`` keeps the production-to-oracle
distances of its final run); ``VALIDATION.md`` is the acceptance receipt.

Importing this package imports nothing.
"""

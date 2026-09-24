"""Per-shard JAX layer bodies, run inside ``shard_map`` on the expert-8 x feature-4 device mesh: norms,
rotary embeddings, linear projections, the dense MLP, FP8 decoding, sampling, attention (``attention``)
and mixture-of-experts (``moe``)."""

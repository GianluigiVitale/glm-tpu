"""Configuration: the model configuration and identity (``model``), the site file (``site``), the
mesh axis names (``parallel``) and the decoder's cache configuration (``cache``). ``model``, ``site``
and ``parallel`` use the standard library only; ``cache`` derives the layer contracts
(``glm_tpu.layers.contracts``) and therefore imports JAX."""

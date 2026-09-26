"""Configuration: the model configuration and identity (``model``), the site file (``site``), the
mesh axis names (``parallel``) and the decoder's cache configuration (``cache``), which also defines
the numerical contracts the layers share (``glm_tpu.layers.contracts`` re-exports them). Importing
any of them imports no JAX (``StageLocalKvLayout.owner`` imports ``jax.numpy`` when a layer calls it)."""

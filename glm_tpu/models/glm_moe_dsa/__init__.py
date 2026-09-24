"""GLM-5.3 (``GlmMoeDsaForCausalLM``): 78 decoder layers of multi-head latent attention with the DSA
indexer and a mixture of 256 routed FP8 experts.

``model`` builds the decode programs, ``prefill`` the prefill program, ``decoder_layer`` holds the
layer bodies, ``state`` the device state and ``weights`` the checkpoint and resident weight trees.
``hf_config/`` holds the pinned Hugging Face assets (package data; SHA-256 pins in
:mod:`glm_tpu.config`)."""

"""Device-program preparation and execution for one worker: the model runner (``tpu_runner``), the
program set (``programs``), compilation with preserved graph originals (``compilation_manager``),
the fresh KV-cache initializer (``kv_cache_manager``), memory and collective admission
(``admission``) and the HLO parser it uses (``hlo_utils``)."""

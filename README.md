# glm-tpu — GLM-5.2 (DeepSeek-sparse-attention family) porting harness for TPU v4

Harness repo for bringing **GLM-5.2** up on **TPU v4** via the vLLM `tpu-inference`
(torchax) path — the GLM analogue of `moe-tpu` (which did DeepSeek-V4-Flash).

- **Model / kernel / quant code** lives in the fork
  `GianluigiVitale/tpu-inference` (a GLM working branch), building on the DSV4-Flash
  work (the DSA lightning-indexer / top-k Pallas paged-decode kernel).
- **This repo** holds everything around it: configs, parity harnesses, eval/bench
  scripts, and docs.

GLM-5.2's DSA differs from DSV4-Flash's (`index_topk=2048`, 32 indexer heads,
interleaved-RoPE indexer, IndexShare — one `full` layer per 4-layer block — and
MTP), so the DSV4 cores are a starting point, not a copy.

## Setup
```bash
bash ~/setup.sh --folder=glm-tpu   # clone + bucket restore + venv + 5-min sync cron
```

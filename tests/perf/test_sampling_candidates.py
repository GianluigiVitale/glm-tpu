"""Candidate-set nucleus sampler: same tokens as the frozen sampler, exact fallback.

CPU semantics on the forced 32-device mesh; no TPU speed or model-quality claim.
"""

from __future__ import annotations

import os
import subprocess
import sys

import pytest

from glm_tpu.greenfield.kernels.ws32_sampling import NucleusConfig


def test_candidates_per_shard_is_validated():
    import jax.numpy as jnp

    from glm_tpu.perf.ws32_sampling_candidates import ws32_nucleus_sample_candidates_mapped

    with pytest.raises(ValueError):
        ws32_nucleus_sample_candidates_mapped(
            jnp.zeros((1, 8), jnp.bfloat16), jnp.float32(0.5), vocab_size=64,
            config=NucleusConfig(), candidates_per_shard=0,
        )
    with pytest.raises(ValueError):
        ws32_nucleus_sample_candidates_mapped(
            jnp.zeros((1, 8), jnp.bfloat16), jnp.float32(0.5), vocab_size=64,
            config=None,  # type: ignore[arg-type]
        )


def test_candidate_sampler_matches_frozen_sampler_cpu32():
    program = r'''
import json
from collections import Counter
import jax, jax.numpy as jnp, numpy as np
from jax.sharding import Mesh, PartitionSpec as P
from glm_tpu.greenfield.kernels.ws32_io import Ws32GreedySampleResult
from glm_tpu.greenfield.kernels.ws32_sampling import NucleusConfig, ws32_nucleus_sample_mapped
from glm_tpu.greenfield.sharding.hlo_contract import parse_hlo_module
from glm_tpu.perf.ws32_sampling_candidates import ws32_nucleus_sample_candidates_mapped
mesh = Mesh(np.array(jax.devices()).reshape(8, 4), ("expert", "feature"))
V = 4096
def build(fn, **kw):
    return jax.jit(jax.shard_map(lambda x, u: fn(x, u, vocab_size=V, **kw), mesh=mesh,
                                 in_specs=(P(None, "expert"), P()),
                                 out_specs=Ws32GreedySampleResult(P(), P()), check_vma=False))
rng = np.random.default_rng(3)
report = dict(draws=0, token_mismatches=0, health_mismatches=0)
for temperature, top_p in ((1., .95), (.7, .8), (1., 1.), (2., .5), (1., .01)):
    cfg = NucleusConfig(temperature, top_p)
    frozen, fast = build(ws32_nucleus_sample_mapped, config=cfg), build(
        ws32_nucleus_sample_candidates_mapped, config=cfg, candidates_per_shard=32)
    # Peaked distributions stay inside the candidates; flat ones force the exact fallback.
    for scale in (4., 1., .2):
        for _ in range(3):
            logits = jnp.asarray(rng.normal(0, scale, (1, V)), jnp.bfloat16)
            for u in np.linspace(.001, .999, 21, dtype=np.float32):
                a, b = frozen(logits, jnp.float32(u)), fast(logits, jnp.float32(u))
                report["draws"] += 1
                report["token_mismatches"] += int(a.token_id[0] != b.token_id[0])
                report["health_mismatches"] += int(a.contract_valid[0] != b.contract_valid[0])
# Ties: identical logits select ascending ids inclusive of the crossing token, as frozen.
cfg = NucleusConfig(top_p=.5)
frozen, fast = build(ws32_nucleus_sample_mapped, config=cfg), build(
    ws32_nucleus_sample_candidates_mapped, config=cfg, candidates_per_shard=4)
tied = jnp.zeros((1, V), jnp.bfloat16)
for u in (0., .249, .25, .5, .75, .999):
    a, b = frozen(tied, jnp.float32(u)), fast(tied, jnp.float32(u))
    report["draws"] += 1
    report["token_mismatches"] += int(a.token_id[0] != b.token_id[0])
# Fail-closed paths are the frozen ones.
cfg = NucleusConfig()
fast = build(ws32_nucleus_sample_candidates_mapped, config=cfg, candidates_per_shard=32)
peaked = jnp.asarray(rng.normal(0, 4., (1, V)), jnp.bfloat16)
closed = []
for bad in (float("nan"), 1., -.1):
    r = fast(peaked, jnp.float32(bad)); closed.append([int(r.token_id[0]), bool(r.contract_valid[0])])
r = fast(peaked.at[0, 7].set(jnp.inf), jnp.float32(.3)); closed.append([int(r.token_id[0]), bool(r.contract_valid[0])])
report["fail_closed"] = closed
frozen = build(ws32_nucleus_sample_mapped, config=cfg)
for label, fn in (("frozen", frozen), ("candidates", fast)):
    counts = Counter(x.opcode for x in parse_hlo_module(fn.lower(peaked, jnp.float32(.5)).compile().as_text()).instructions)
    report[label] = {k: counts.get(k, 0) for k in ("all-gather", "all-reduce", "sort", "conditional")}
print(json.dumps(report, sort_keys=True))
'''
    env = dict(os.environ, JAX_PLATFORMS="cpu",
               XLA_FLAGS=(os.environ.get("XLA_FLAGS", "") + " --xla_force_host_platform_device_count=32").strip())
    completed = subprocess.run([sys.executable, "-c", program], env=env, text=True,
                               capture_output=True, check=False, timeout=900)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    import json

    report = json.loads(completed.stdout.strip().splitlines()[-1])
    assert report["draws"] >= 900
    assert report["token_mismatches"] == 0 and report["health_mismatches"] == 0
    assert report["fail_closed"] == [[-1, False]] * 4
    # The frozen head sorts the whole vocabulary; the fast path sorts only the
    # gathered candidates and keeps the frozen sort inside its fallback branch.
    assert report["frozen"]["sort"] == 1 and report["frozen"]["conditional"] == 0
    assert report["candidates"]["conditional"] == 1

"""A partial or mixed fleet must never become a complete timing receipt."""
import json

import pytest

from tools.perf_tpu_microbench import summarize


def test_summary_requires_complete_consistent_fleet(tmp_path):
    def write(rank, **changes):
        row = dict(rank=rank, hostname=f"host-{rank}", devices=32, which="attention",
                   jax="test", finished_utc="test", source_sha256={"source.py": "abc"})
        row.update(changes)
        (tmp_path / f"microbench.rank{rank}.json").write_text(json.dumps(row))

    for rank in range(7):
        write(rank)
    with pytest.raises(ValueError, match="eight"):
        summarize(tmp_path)
    write(7, finished_utc=None)
    with pytest.raises(ValueError, match="completed"):
        summarize(tmp_path)
    write(7, source_sha256={"source.py": "different"})
    with pytest.raises(ValueError, match="fingerprints"):
        summarize(tmp_path)
    write(7)
    assert summarize(tmp_path)["ranks"] == list(range(8))


def test_synthetic_generator_respects_partial_replication_cpu32():
    import os
    import subprocess
    import sys

    code = r'''
import jax, numpy as np
from jax.sharding import Mesh, PartitionSpec as P
from tools.perf_tpu_microbench import _sharded_random
mesh=Mesh(np.asarray(jax.devices(),object).reshape(8,4),('expert','feature'))
for spec in (P('expert'), P(None,'feature'), P('expert','feature'), P()):
    a=_sharded_random(mesh,(8,128),spec,'bf16',1)
    groups={}
    for shard in a.addressable_shards:
        groups.setdefault(str(shard.index),[]).append(np.asarray(shard.data))
    for values in groups.values():
        for other in values[1:]: np.testing.assert_array_equal(values[0],other)
    assert len({values[0].tobytes() for values in groups.values()})==len(groups)
print('all replicated axes agree; partitioned shards differ')
'''
    env = dict(os.environ, JAX_PLATFORMS="cpu", XLA_FLAGS="--xla_force_host_platform_device_count=32")
    result = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr

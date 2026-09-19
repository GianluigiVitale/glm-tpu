"""Post-timing, host-local shard comparisons; never a model execution path."""
import numpy as np


def compare_addressable_state(actual, expected):
    """Aggregate every local shard, including replicas, without token payloads.

    Call on every host after both immutable results are ready. Counts include
    replicated copies; they are not counts of unique global elements. Only one
    pair of shards is copied to host at a time. Matching infinities are valid
    sentinels; changed nonfinite values are counted separately from finite error.
    """
    import jax
    import jax.numpy as jnp

    if jax.tree.structure(actual) != jax.tree.structure(expected):
        raise ValueError('state comparison structures differ')
    report = {}
    for (path, x), y in zip(jax.tree_util.tree_flatten_with_path(actual)[0], jax.tree.leaves(expected)):
        if x.shape != y.shape or x.dtype != y.dtype:
            raise ValueError('state comparison leaf geometry differs')
        xs, ys = x.addressable_shards, y.addressable_shards
        if not xs or len(xs) != len(ys):
            raise ValueError('state comparison has missing shards')
        row = dict(elements=0,bitwise_different=0,nonfinite_different=0,max_finite_abs_difference=0.0)
        for a,b in zip(xs,ys):
            if a.device != b.device or a.index != b.index:
                raise ValueError('state comparison shard placement differs')
            av,bv = np.ascontiguousarray(a.data),np.ascontiguousarray(b.data)
            different = np.any(av.view(np.uint8).reshape(-1,av.dtype.itemsize) !=
                               bv.view(np.uint8).reshape(-1,bv.dtype.itemsize),axis=1)
            row['elements'] += av.size
            row['bitwise_different'] += int(different.sum())
            if jnp.issubdtype(x.dtype,jnp.floating):
                af,bf = av.astype(np.float64).reshape(-1),bv.astype(np.float64).reshape(-1)
                finite = np.isfinite(af) & np.isfinite(bf)
                row['nonfinite_different'] += int(np.count_nonzero(different & ~finite))
                row['max_finite_abs_difference'] = max(row['max_finite_abs_difference'],
                    float(np.max(np.abs(af[finite]-bf[finite]),initial=0)))
        row['bitwise_equal'] = row['bitwise_different']==0
        report[jax.tree_util.keystr(path)] = row
    return dict(scope='all addressable shards on this host; replicated copies counted; outside model timing',
                bitwise_equal=all(r['bitwise_equal'] for r in report.values()),leaves=report)

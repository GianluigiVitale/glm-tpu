"""Leased WS32 verifier economics on synthetic weights at real GLM geometry.

The controller owns workload/sync leases, immutable deployment, idle checks and
cleanup. This worker creates no infrastructure and loads no trained checkpoint.
Rates are perfect-acceptance estimates without drafting, votes or delivery.
"""
from __future__ import annotations

import argparse
import gc
from hashlib import sha256
import json
import os
from pathlib import Path
import socket
import sys
import time

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def _local_pairs(actual, expected):
    if actual.shape != expected.shape or actual.dtype != expected.dtype:
        raise ValueError('comparison requires matching shape and dtype')
    a, b = actual.addressable_shards, expected.addressable_shards
    if len(a) != 4 or len(b) != 4:
        raise ValueError('comparison requires four local TPU owners')
    for x, y in zip(a, b, strict=True):
        if x.device != y.device or x.index != y.index:
            raise ValueError('comparison owner mismatch')
        yield x, y


def _comparison(pairs):
    import numpy as np
    count = different = 0
    maximum = squared_error = squared_reference = 0.
    finite = bitwise = True
    for x, y in pairs:
        x, y = np.ascontiguousarray(x), np.ascontiguousarray(y)
        bitwise &= bool(np.array_equal(x.view(np.uint8), y.view(np.uint8)))
        different += int(np.count_nonzero(x != y)); count += x.size
        x, y = x.astype(np.float64), y.astype(np.float64)
        ok = np.isfinite(x) & np.isfinite(y)
        finite &= bool(ok.all())
        error = x[ok] - y[ok]
        maximum = max(maximum, float(np.max(np.abs(error), initial=0)))
        squared_error += float(np.square(error).sum())
        squared_reference += float(np.square(y[ok]).sum())
    return dict(bitwise_equal=bitwise, finite=finite, differing_elements=different,
        local_replica_elements=count, max_abs=maximum,
        relative_l2=float((squared_error / max(squared_reference, 1e-300))**.5))


def local_comparison(actual, expected):
    """Compare every local device replica; never gather model arrays across hosts."""
    return _comparison((x.data,y.data) for x,y in _local_pairs(actual,expected))


def local_cache_span_comparison(actual, expected, block_tables, start, stop):
    """Compare only written WS32 cache positions, following physical page mapping.

    Cache layout is [layers, physical pages, logical page rows, width], with
    contiguous expert8 sharding on rows and feature4 replication. Local shards
    with no positions in this span contribute zero elements. This host-only
    diagnostic performs no collective and must stay outside model timing.
    """
    import numpy as np
    table = np.asarray(block_tables)
    if (len(actual.shape) != 4 or actual.shape != expected.shape
            or any(n <= 0 for n in actual.shape)
            or actual.shape[2] % 8
            or table.shape != (1,actual.shape[1]) or table.dtype != np.int32
            or sorted(table[0].tolist()) != list(range(actual.shape[1]))
            or type(start) is not int or type(stop) is not int
            or not 0 <= start < stop <= actual.shape[1]*actual.shape[2]):
        raise ValueError('cache comparison requires a valid WS32 span and page permutation')
    layers,pages,page_rows,width = actual.shape
    positions = np.arange(start,stop,dtype=np.int64)
    mask = np.zeros((pages,page_rows),np.bool_)
    mask[table[0,positions//page_rows],positions%page_rows] = True
    replica_rows = 0
    def selected_pairs():
        nonlocal replica_rows
        for x,y in _local_pairs(actual,expected):
            if (len(x.index) != 4 or any(not isinstance(s,slice) for s in x.index)
                    or any(x.index[i].indices(actual.shape[i]) != (0,actual.shape[i],1)
                           for i in (0,1,3))):
                raise ValueError('cache comparison requires expert row sharding only')
            lo,hi,step = x.index[2].indices(page_rows)
            if (step != 1 or hi-lo != page_rows//8 or lo % (page_rows//8)
                    or x.data.shape != (layers,pages,page_rows//8,width)
                    or y.data.shape != x.data.shape):
                raise ValueError('cache comparison requires complete contiguous expert shards')
            local_mask = mask[:,lo:hi]
            replica_rows += int(local_mask.sum())
            yield np.asarray(x.data)[:,local_mask,:],np.asarray(y.data)[:,local_mask,:]
    result = _comparison(selected_pairs())
    return dict(result,logical_start=start,logical_stop=stop,
        local_replica_rows=replica_rows,
        global_unique_elements=(stop-start)*layers*width,
        expected_feature_replicas=4)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--coordinator', default='192.168.0.37:8476')
    p.add_argument('--code-hash', required=True)
    p.add_argument('--source-manifest-sha256', required=True)
    p.add_argument('--iters', type=int, default=20)
    p.add_argument('--capacity', type=int, default=8192)
    p.add_argument('--rows', default='2,3,5')
    p.add_argument('--trace', action='store_true')
    p.add_argument('--canonical-mlp', action='store_true')
    p.add_argument('--batched-attention', action='store_true')
    p.add_argument('--small-expert-tiles', action='store_true',
                   help='experimental M8 expert reuse; requires --canonical-mlp')
    p.add_argument('--rowwise-dsa', action='store_true',
                   help='preserve per-token DSA fallback conditions; requires --batched-attention')
    args = p.parse_args()
    rows = tuple(int(n) for n in args.rows.split(','))
    if (not rows or len(set(rows)) != len(rows) or any(n not in (2,3,5) for n in rows)
            or args.iters < 5 or args.capacity < 128 or len(args.code_hash) != 40
            or (args.small_expert_tiles and not args.canonical_mlp)
            or (args.rowwise_dsa and not args.batched_attention)):
        raise ValueError('invalid synthetic verifier benchmark geometry/identity')
    os.umask(0o077)
    root = args.output.resolve()
    if root.is_relative_to(REPO) or not root.is_dir():
        raise ValueError('output must be an existing private run directory outside source')
    manifest_path = root / 'source_manifest.json'
    if sha256(manifest_path.read_bytes()).hexdigest() != args.source_manifest_sha256:
        raise ValueError('source manifest identity differs')
    manifest = json.loads(manifest_path.read_bytes())
    for name, digest in manifest.items():
        path = (REPO / name).resolve()
        if not path.is_relative_to(REPO) or sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError('deployed source identity differs: ' + name)

    import jax
    import jax.numpy as jnp
    import numpy as np
    from jax.experimental import multihost_utils
    from jax.sharding import NamedSharding, PartitionSpec as P
    from glm_tpu.greenfield.runtime import ws32_decoder as dec
    from glm_tpu.greenfield.types import ModelGeometry
    from glm_tpu.perf.bf16_resident import bf16_resident_weights
    from glm_tpu.perf.real_validation import build_db610_decoder, inspect_research_hlo, memory_projection
    from glm_tpu.perf.speculative_verify import build_verifier, build_prefix_committer
    from scripts.greenfield.ws32_compile_originals import compile_program
    from tools.perf_tpu_microbench import _ws32_mesh, _sharded_random, _timeit, synthetic_decoder_weights
    from tools.perf_real_validation import write_json

    rank = int(socket.gethostname().rsplit('-w-', 1)[1])
    jax.distributed.initialize(coordinator_address=args.coordinator, num_processes=8, process_id=rank)
    if jax.default_backend() != 'tpu' or jax.device_count() != 32 or len(jax.local_devices()) != 4:
        raise RuntimeError('expected existing eight-host 32-chip TPU pod')
    mesh = _ws32_mesh()
    record = dict(schema='glm_perf_speculative_verifier_rank_v1', rank=rank,
        hostname=socket.gethostname(), jax_process_index=jax.process_index(), devices=32,
        jax=jax.__version__, which='speculative_verifier', source_sha256=manifest,
        code_hash=args.code_hash, source_manifest_sha256=args.source_manifest_sha256,
        capacity=args.capacity, rows=list(rows), iters=args.iters, programs={}, phases={}, results={},
        verifier_options=dict(canonical_mlp=args.canonical_mlp,batched_attention=args.batched_attention,
                              small_expert_tiles=args.small_expert_tiles,rowwise_dsa=args.rowwise_dsa),
        synthetic_generator='partition_axes_only_v2', measured_speculative_throughput=False,
        trained_model_quality_claim=False, frozen_graph_admission_inherited=False,
        timing_scope='dispatch and completion; draft, fleet votes, delivery and compilation excluded',
        started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))
    receipt = root / f'microbench.rank{rank}.json'
    def save():write_json(receipt, record)
    def vote(valid):
        return bool(np.asarray(multihost_utils.process_allgather(np.asarray(bool(valid)))).all())
    def phase(name, action):
        result = error = None
        started = time.perf_counter()
        try:result = action()
        except Exception as exc:error = exc
        agreed = vote(error is None)
        record['phases'][name] = dict(seconds=time.perf_counter()-started,passed=error is None and agreed)
        save()
        if error is not None:raise error
        if not agreed:raise RuntimeError('peer refused phase: ' + name)
        print('VERIFIER_PHASE ' + name, flush=True)
        return result
    def require(condition, message):
        if not bool(condition):raise RuntimeError(message)
    def stats():return [dict(device_id=int(d.id), **d.memory_stats()) for d in jax.local_devices()]
    hlo_root = Path('/dev/shm/glm-perf-hlo') / root.name / f'hlo.rank{rank}'
    def prepare_hlo():
        fs = os.statvfs('/dev/shm')
        require(fs.f_bavail*fs.f_frsize > 4*1024**3, 'insufficient HLO storage')
        hlo_root.mkdir(parents=True, exist_ok=False)
        (root/f'hlo.rank{rank}').symlink_to(hlo_root, target_is_directory=True)
    phase('hlo_storage', prepare_hlo)
    def compile_checked(name, fn, values):
        compiled = phase('compile_'+name, lambda: compile_program(fn, values, name, hlo_root, record))
        item = record['programs'][name]
        def graph_consensus():
            digests = bytes.fromhex(item['stablehlo_sha256'] + item['optimized_hlo_sha256'])
            hashes = np.asarray(multihost_utils.process_allgather(np.frombuffer(digests,np.uint8)))
            require((hashes == hashes[0]).all(), 'fleet compiled graph identities differ')
        phase('graph_'+name, graph_consensus)
        item['hlo_admission'] = phase('hlo_'+name,
            lambda: inspect_research_hlo((hlo_root/f'{name}.optimized_hlo.txt').read_text()))
        item['memory_admission'] = memory_projection(stats(), item['compiled_memory'])
        phase('memory_'+name, lambda: require(item['memory_admission']['passed'], 'memory admission refused'))
        save()
        return compiled
    def trace(name, fn, values):
        if args.trace:
            target = hlo_root / 'trace' / name
            target.mkdir(parents=True)
            with jax.profiler.trace(str(target), create_perfetto_link=False):
                for _ in range(2):jax.block_until_ready(fn(*values))
            record['programs'][name]['trace_relative_to_hlo'] = str(Path('trace')/name)
            save()
    try:
        geometry = ModelGeometry.from_hf_config(json.loads((REPO/'configs/glm-5.2-fp8-config.json').read_bytes()))
        config = dec.Ws32DecoderConfig(geometry,args.capacity,host_main_rope_table=True)
        seed = [0]
        def make(shape,spec,kind):
            seed[0] += 1
            return _sharded_random(mesh,shape,spec,kind,seed[0])
        raw = phase('synthetic_weights', lambda: jax.block_until_ready(synthetic_decoder_weights(config,make,progress=print)))
        weights = phase('resident_tables', lambda: jax.block_until_ready(bf16_resident_weights(mesh,config,raw)))
        del raw; gc.collect()
        def put(value):return jax.device_put(value,NamedSharding(mesh,P()))
        state = dec.make_ws32_initial_state(mesh,config)._replace(
            position=put(np.array([64],np.int32)),context_lengths=put(np.array([65],np.int32)))
        record['prefix_scope'] = '64 synthetic zero-cache positions; no trained prompt or prefill'
        rope = put(np.asarray(dec.build_ws32_main_rope_table(config)))
        token = put(np.array([5],np.int32))
        ordinary = compile_checked('ordinary',build_db610_decoder(mesh,config).execute,(token,state,weights,rope))
        references = [state]; tokens = [token]; residuals = []
        def reference_trail():
            for _ in range(max(rows)):
                result = jax.block_until_ready(ordinary(tokens[-1],references[-1],weights,rope))
                require(np.asarray(result.state.contract_valid).all(), 'unhealthy ordinary reference')
                references.append(result.state); tokens.append(result.next_token); residuals.append(result.final_residual_local)
        phase('reference_trail',reference_trail)
        record['results']['ordinary'] = phase('time_ordinary',
            lambda: _timeit(ordinary,(token,state,weights,rope),warmup=5,iters=args.iters))
        trace('ordinary',ordinary,(token,state,weights,rope))
        for n in rows:
            name = f'verify_{n}'
            inputs = jnp.concatenate(tokens[:n])
            values = (inputs,state,weights,rope)
            verifier = compile_checked(name,build_verifier(mesh,config,
                canonical_mlp=args.canonical_mlp,batched_attention=args.batched_attention,
                small_expert_tiles=args.small_expert_tiles,rowwise_dsa=args.rowwise_dsa),values)
            proposal = phase('execute_'+name,lambda: jax.block_until_ready(verifier(*values)))
            phase('health_'+name,lambda: require(np.asarray(proposal.contract_valid).all(), 'unhealthy proposal'))
            result = dict(target_prediction_agreement=bool(np.array_equal(np.asarray(proposal.predictions),np.asarray(jnp.concatenate(tokens[1:n+1])))),
                comparisons=dict(residual=local_comparison(proposal.final_residual_local,jnp.concatenate(residuals[:n]))))
            committer = compile_checked('commit_'+str(n),build_prefix_committer(mesh,config),(state,proposal,put(np.int32(n))))
            committed = phase('commit_'+name,lambda: jax.block_until_ready(committer(state,proposal,put(np.int32(n)))))
            result['comparisons']['kv'] = local_comparison(committed.kv_cache_local,references[n].kv_cache_local)
            result['comparisons']['index'] = local_comparison(committed.index_cache_local,references[n].index_cache_local)
            result['selected_positions_equal'] = bool(np.array_equal(np.asarray(committed.selected_positions),np.asarray(references[n].selected_positions)))
            phase('finite_'+name,lambda: require(all(v['finite'] for v in result['comparisons'].values()), 'nonfinite target comparison'))
            result['verify_timing'] = phase('time_'+name,lambda: _timeit(verifier,values,warmup=5,iters=args.iters))
            result['commit_timing'] = phase('time_commit_'+str(n),
                lambda: _timeit(committer,(state,proposal,put(np.int32(n))),warmup=5,iters=args.iters))
            verify_ms = result['verify_timing']['mean_ms']
            total_ms = verify_ms + result['commit_timing']['mean_ms']
            result['perfect_acceptance_estimate_tok_s'] = 1000*n/total_ms
            result['perfect_acceptance_estimate_speedup'] = n*record['results']['ordinary']['mean_ms']/total_ms
            result['draft_cost_included'] = result['measured_speculative_throughput'] = False
            record['results'][name] = result
            trace(name,verifier,values)
            save()
            print(f'VERIFIER_RESULT rows={n} ms={total_ms:.3f} perfect_acceptance_estimate={1000*n/total_ms:.3f} tok/s',flush=True)
            del verifier, committer, proposal, committed
            gc.collect()
        record['device_memory_after'] = stats()
        record['finished_utc'] = time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())
        save()
    except Exception as exc:
        record['failure'] = dict(type=type(exc).__name__,message=str(exc))
        save()
        raise
    finally:
        jax.distributed.shutdown()


if __name__ == '__main__':
    main()

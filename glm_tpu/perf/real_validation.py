"""Research-only DB610 input authentication and fresh graph admission.

This does not inherit frozen HLO admission or claim universal numerical parity.
Private prompt/reference arrays are returned to the caller, never serialized in
public receipts. The sealed DB610 ledger authenticates the original runner.
"""
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path

import numpy as np


def db610_inputs(repo: Path, original_root: Path):
    from ..greenfield.validation.ws32_short_context import load_ws32_short_context_oracle

    seal = json.loads((repo / 'docs/artifacts/prefill-canonical-short-db610-sealed-20260909.json').read_bytes())
    if original_root.name != seal['run_tag'] or seal['results_db_run_id'] != 610:
        raise ValueError('original DB610 identity differs')
    ledger_bytes = (original_root / 'remote_objects.json').read_bytes()
    ledger_pin = next(r for r in seal['remote_readbacks'] if r['name'] == 'remote_objects.json')
    if len(ledger_bytes) != ledger_pin['size'] or sha256(ledger_bytes).hexdigest() != ledger_pin['sha256']:
        raise ValueError('DB610 original ledger hash differs')
    ledger = json.loads(ledger_bytes)
    runner_pin = next(r for r in ledger['objects'] if r['name'] == 'host_records/runner.rank0.json')
    runner_bytes = (original_root / 'runner.rank0.json').read_bytes()
    if len(runner_bytes) != runner_pin['size'] or sha256(runner_bytes).hexdigest() != runner_pin['sha256']:
        raise ValueError('DB610 original runner hash differs')
    runner = json.loads(runner_bytes)
    base = Path('/home/gianl/gcs-models/oracles/greenfield/glm52')
    oracle = load_ws32_short_context_oracle(
        base/'short_context/2k/greenfield_short_context_oracle_20260806T202544155912103Z/oracle',
        base/'short_context_dsa/2k/greenfield_short_context_dsa_oracle_recovery_20260806T231905802593249Z/oracle',
        **{'expected_'+k.replace('_oracle',''): runner[k] for k in ('token_oracle_manifest_sha256',
             'dsa_oracle_manifest_sha256','token_oracle_success_sha256','dsa_oracle_success_sha256')})
    prompt = np.asarray(oracle.prompt_token_ids, np.int32)
    expected = np.asarray(runner['observed_generated_token_ids'], np.int32)
    if prompt.shape != (2034,) or expected.shape != (29,) or runner['correctness_passed'] is not True:
        raise ValueError('DB610 prompt/reference geometry or original verdict differs')
    if not np.array_equal(expected[:len(oracle.generated_token_ids)], oracle.generated_token_ids):
        raise ValueError('DB610 reference does not match its sealed legacy prefix')
    identity = dict(db_run_id=610, run_tag=seal['run_tag'], runner_sha256=runner_pin['sha256'],
        ledger_sha256=ledger_pin['sha256'], prompt_tokens=2034, reference_tokens=29,
        prompt_sha256=sha256(prompt.tobytes()).hexdigest(),
        reference_sha256=sha256(expected.tobytes()).hexdigest(),
        token_oracle_manifest_sha256=runner['token_oracle_manifest_sha256'])
    return prompt, expected, identity


def inspect_research_hlo(text: str) -> dict:
    """Fresh WS32 graph check: physical axis groups and bounded exchanges.

    This is a deliberately scoped structural check, not the frozen graph's
    exact opcode/identity proof. CPU/TPU numerical evidence remains separate.
    Both operands and results are checked, including async collective forms
    normalized by the retained parser. Full-pod consensus is limited to4KiB.
    """
    from ..greenfield.sharding.hlo_contract import parse_hlo_module

    module = parse_hlo_module(text)
    if module.num_partitions != 32 or module.num_replicas not in (None, 1):
        raise ValueError('research model graph requires exactly32 partitions/one replica')
    widths = {'pred':1,'s8':1,'u8':1,'bf16':2,'f16':2,'s16':2,'u16':2,
              'f32':4,'s32':4,'u32':4,'f64':8,'s64':8,'u64':8}
    allowed = {
        frozenset(frozenset(range(r*4,r*4+4)) for r in range(8)),
        frozenset(frozenset(range(c,32,4)) for c in range(4)),
        frozenset((frozenset(range(32)),)),
    }
    maximum = 0
    for op in module.collectives:
        if op.opcode not in ('all-reduce','all-gather','reduce-scatter'):
            raise ValueError('unreviewed collective kind in research graph: '+op.opcode)
        groups = frozenset(frozenset(g) for g in op.replica_groups)
        if (not op.use_global_device_ids or groups not in allowed
                or sum(map(len, op.replica_groups)) != 32):
            raise ValueError('research collective does not follow physical expert8/feature4 axes')
        sizes = []
        for shape in (*op.result_shapes,*op.operand_shapes):
            if shape.dtype not in widths:
                raise ValueError('unknown collective dtype')
            sizes.append(shape.element_count * widths[shape.dtype])
        payload = max(sizes, default=0)
        maximum = max(maximum,payload)
        if not sizes or payload > 128*1024**2 or (op.maximum_group_size == 32 and payload > 4096):
            raise ValueError('oversized or unparsed collective in research graph')
    if not module.collectives:
        raise ValueError('model graph has no parsed collectives')
    return dict(passed=True, profile='research_ws32_axis_payload_v1',
        num_partitions=32, instructions=len(module.instructions),
        collectives=dict(Counter(op.opcode for op in module.collectives)),
        maximum_collective_payload_bytes=maximum, frozen_graph_admission_inherited=False)


def memory_projection(stats, memory, *, reserve_bytes=512*1024**2):
    fields = ('output_size_in_bytes','temp_size_in_bytes','generated_code_size_in_bytes','alias_size_in_bytes')
    if any(type(memory.get(k)) is not int or memory[k] < 0 for k in fields) or reserve_bytes < 0:
        raise ValueError('invalid compiler memory accounting')
    extra = max(0, memory[fields[0]]+memory[fields[1]]+memory[fields[2]]-memory[fields[3]])
    if len(stats) != 4 or len({s['device_id'] for s in stats}) != 4:
        raise ValueError('memory admission needs four distinct local chips')
    rows = []
    for s in stats:
        if not 0 <= s['bytes_in_use'] <= s['bytes_limit']:
            raise ValueError('invalid live allocator accounting')
        predicted = s['bytes_in_use'] + extra + reserve_bytes
        rows.append(dict(device_id=s['device_id'], predicted_bytes=predicted,
                         limit=s['bytes_limit'], fits=predicted < s['bytes_limit']))
    return dict(passed=all(r['fits'] for r in rows), reserve_bytes=reserve_bytes, chips=rows)

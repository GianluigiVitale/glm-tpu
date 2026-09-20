"""Fresh optimized graph and live-memory admission; no inherited HLO seal."""
from collections import Counter

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

"""All-resident sampled request memory admission, without new model math.

Reuse the original complete live-array census and conservative allocation
arithmetic. Unlike the one-way long prefill profile, count BOTH prefill shapes,
decode, observer, probe and cache-initializer code, plus every live raw/exact/overlay/WK array.
Only the active program's proven state aliases reduce its output allocation.
This is a request-boundary check, not a full census/dump on every generated token.
The protected worker still owns checkpoint/HLO validation and peak monitoring.
"""
from __future__ import annotations

from copy import deepcopy
import json
from typing import Any, Mapping, Sequence

import jax

from glm_tpu.greenfield.runtime.ws32_batched_prefill import Ws32BatchedPrefillState
from glm_tpu.greenfield.runtime.ws32_decoder import Ws32DecoderState
from glm_tpu.greenfield.validation import ws32_prefill_memory as original
from glm_tpu.greenfield.validation.ws32_prefill_admission import (
    SHORT_DEVICE_LIMIT_BYTES as DEVICE_LIMIT, SHORT_RESERVE_BYTES as RESERVE,
)
from scripts.greenfield.prefill_window_worker import validate_memory_owners

SCHEMA = 'ws32_native_resident_memory_v1'
ROLES = ('prefill_chunk', 'prefill_tail', 'decode', 'observer', 'cache_probe', 'cache_init')
OWNERS = {name: (2 if name.startswith('prefill_') else 1) for name in ROLES[:4]}


def budgets(record: Mapping[str, Any]) -> dict[str, Any]:
    """Replay actual aliases and all resident code; never trust stored fit flags."""
    if (record.get('schema_version') != SCHEMA or record.get('reserve_bytes') != RESERVE
            or set(record['compiled_memory']) != set(ROLES)
            or record.get('donated_arguments') != OWNERS
            or type(record.get('cache_present')) is not bool
            or record.get('phase') not in ('before_cache','cache_ready','prefill_done')
            or (record['phase'] == 'before_cache') == record['cache_present']):
        raise ValueError('native resident program/ownership/reserve contract differs')
    census = record['census']
    owners = record['local_slots']
    if (not isinstance(owners, list) or len(owners) != 4
            or any(set(row) != {'device_id','slot'} for row in owners)
            or len({row['device_id'] for row in owners}) != 4
            or any(type(row['device_id']) is not int or row['device_id'] < 0 for row in owners)
            or type(record['process_index']) is not int or not 0 <= record['process_index'] < 8):
        raise ValueError('native physical owner identity differs')
    slots = {row['device_id']:row['slot'] for row in owners}
    validate_memory_owners(census['devices'], local_slots=slots,
                           process_index=record['process_index'])
    analyses = record['compiled_memory']
    for name, analysis in analyses.items():
        if set(analysis) != set(original.MEMORY_FIELDS):
            raise ValueError('native compiled memory fields differ')
        for key, value in analysis.items():
            original._integer(value, f'{name}.{key}')
    result = {}
    # Final prefill installs repaired index rows into decoder state. Its two
    # index roots may now alias legitimately; no further prefill consumes them.
    # Count all resident code, but do not pretend that finished state is again
    # a valid donating prefill input. Fresh requests allocate their own state.
    active = (('cache_init',) if not record['cache_present'] else
              (ROLES[2:-1] if record['phase']=='prefill_done' else ROLES[:-1]))
    for graph in active:
        group = 'prefill_state' if graph.startswith('prefill_') else 'decode_state'
        alias = analyses[graph]['alias_size_in_bytes']
        if graph in ('cache_probe','cache_init') and alias != 0:
            raise ValueError('cache probe/initializer must not donate')
        for row in census['devices']:
            if row['memory_stats']['bytes_limit'] != DEVICE_LIMIT:
                raise ValueError('native device limit differs')
            if graph == 'cache_init':
                continue  # full output+scratch added; no previous-cache alias credit
            state_bytes = 0
            for entry in row['buffers']:
                if group in entry.get('groups', []):
                    if (entry.get('identity_mode') != 'physical_pointer'
                            or 'retained_weights' in entry['groups']):
                        raise ValueError('native state lacks ownership or aliases weights')
                    state_bytes += original._integer(entry['bytes'], 'state bytes', positive=True)
            if state_bytes <= 0 or alias > min(state_bytes,
                    analyses[graph]['output_size_in_bytes'], analyses[graph]['argument_size_in_bytes']):
                raise ValueError('native compiled aliases exceed active state/output/arguments')
        effective = deepcopy(analyses)
        effective[graph]['output_size_in_bytes'] -= alias
        effective[graph]['alias_size_in_bytes'] = 0
        result[graph] = original.budget_resident_execution(census, effective,
            active_graph=graph, resident_graphs=ROLES, required_reserve_bytes=RESERVE)
    return result


def make_record(compiled: Mapping[str, Any], state: Ws32BatchedPrefillState | None, *,
                retained_roots: Mapping[str, Any], devices: Sequence[Any],
                local_slots: Mapping[int, int], process_index: int,
                phase: str | None = None) -> dict[str, Any]:
    """Admit both phases against the SAME live cache and all actual code handles.

    Six role handles are conservatively counted six times even if
    PJRT internally shares code. Compiler identity/HLO checks are separate.
    With state=None, admit cache_init's complete compiled output and scratch
    against existing live arrays/code before allocating a fresh request cache.
    """
    if (set(compiled) != set(ROLES) or (state is not None and
            (type(state) is not Ws32BatchedPrefillState or type(state.decoder) is not Ws32DecoderState))):
        raise ValueError('native memory requires complete cache or explicit preallocation and six programs')
    if set(retained_roots) != {'raw_weights','decode_weights','wk','exact_weights','rope'}:
        raise ValueError('native resident weight views missing')
    analyses = {}
    for role in ROLES:
        fn = compiled[role]
        args, kwargs = fn.args_info
        index = OWNERS.get(role)
        count = 7 if role.startswith('prefill_') else (1 if role in ('cache_probe','cache_init') else 6)
        state_index = index if index is not None else 0
        expected_type = Ws32BatchedPrefillState if role.startswith('prefill_') else Ws32DecoderState
        if kwargs or len(args) != count or (role != 'cache_init' and type(args[state_index]) is not expected_type):
            raise ValueError('native compiled argument/state structure differs')
        for number, value in enumerate(args):
            if any(leaf.donated is not (number == index) for leaf in jax.tree.leaves(value)):
                raise ValueError('native compiled donation is not exclusively state')
        if state is not None and role != 'cache_init':
            actual_state = state if role.startswith('prefill_') else state.decoder
            expected = fn.in_avals[0][state_index]
            for actual, abstract in zip(jax.tree.leaves(actual_state), jax.tree.leaves(expected), strict=True):
                if actual.shape != abstract.shape or actual.dtype != abstract.dtype:
                    raise ValueError('native compiled state shape/dtype differs')
        analysis = fn.memory_analysis()
        analyses[role] = {key: getattr(analysis, key, None) for key in original.MEMORY_FIELDS}
    roots = {'retained_weights':retained_roots}
    if state is not None:
        roots.update(prefill_state=state, decode_state=state.decoder)
    census = original.capture_resident_buffers(roots, devices=devices)
    record = dict(schema_version=SCHEMA, compiled_memory=analyses, census=census,
        local_slots=[dict(device_id=device,slot=slot) for device,slot in sorted(local_slots.items())],
        process_index=process_index,
        donated_arguments=dict(OWNERS), reserve_bytes=RESERVE, cache_present=state is not None,
        phase=phase if phase is not None else ('before_cache' if state is None else 'cache_ready'))
    record['budgets'] = budgets(record)
    return record


def validate_record(record: Mapping[str, Any]) -> None:
    """Reject fabricated or failing budgets; physical-owner checks are replayed."""
    if set(record) != {'schema_version','compiled_memory','census','local_slots',
                      'process_index','donated_arguments','reserve_bytes','budgets','cache_present','phase'}:
        raise ValueError('native memory record fields differ')
    expected = budgets(record)
    if (json.dumps(expected, sort_keys=True, allow_nan=False) !=
            json.dumps(record['budgets'], sort_keys=True, allow_nan=False)
            or not all(value['estimate_fits'] for value in expected.values())):
        raise ValueError('native all-resident memory fit fails or record drifted')


class RequestMemoryAdmission:
    """NativeRuntime's real authorize callback, not a boolean attestation.

    Call only after the worker binds each compiled object to its inspected HLO.
    Persists three boundary records per request, never per-token full censuses.
    The writer must enforce the outer run's bounded evidence budget.
    """
    def __init__(self, *, compiled: Mapping[str, Any], devices: Sequence[Any],
                 local_slots: Mapping[int, int], process_index: int, preserve: Any):
        if set(compiled) != set(ROLES) or not callable(preserve):
            raise ValueError('native request admission needs six programs and evidence writer')
        self.compiled, self.devices = dict(compiled), tuple(devices)
        self.local_slots, self.process_index = dict(local_slots), process_index
        self.preserve = preserve
        self._next = 'before_cache'

    def __call__(self, stage: str, roots: Mapping[str, Any], state: Any) -> None:
        if stage != self._next or (stage == 'before_cache') != (state is None):
            raise ValueError('native memory phase/cache presence drifted')
        record = make_record(self.compiled, state, retained_roots=roots,
            devices=self.devices, local_slots=self.local_slots, process_index=self.process_index,
            phase=stage)
        self.preserve(stage, record)  # retain failing evidence before refusal
        validate_record(record)
        self._next = {'before_cache':'cache_ready', 'cache_ready':'prefill_done',
                      'prefill_done':'before_cache'}[stage]

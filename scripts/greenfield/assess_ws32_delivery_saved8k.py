"""Read-only §26 reassessment of the fixed historical 8K task originals.

Authenticates 16 generation-qualified cloud objects in memory. Writes no DB,
SUCCESS, model state or cloud objects; emits compact JSON to stdout. This is
not a protected performance seal or a model-card benchmark.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path

import ml_dtypes
import numpy as np

from glm_tpu.greenfield.validation import ws32_delivery_quality as delivery
from glm_tpu.greenfield.validation.ws32_short_context import (
    load_ws32_short_context_oracle, validate_ws32_cache_probe,
)

ROOT = Path(__file__).resolve().parents[2]
RECEIPT = 'docs/artifacts/prefill-canonical8k-token-refusal-20260909.json'
RECEIPT_SHA = 'f659925dec463a1473076d913cfd577a81ba0d584c9feba30acb71f5480df175'


def main() -> None:
    from google.cloud import storage

    source = (ROOT / RECEIPT).read_bytes()
    if sha256(source).hexdigest() != RECEIPT_SHA:
        raise ValueError('original failure receipt changed')
    original = json.loads(source)
    roots = Path('/home/gianl/gcs-models/oracles/greenfield/glm52')
    oracle = load_ws32_short_context_oracle(
        roots / 'short_context/8k/greenfield_short_context_oracle_8k_20260807T172307269147351Z/oracle',
        roots / 'short_context_dsa/8k/greenfield_short_context_dsa_oracle_8k_recovery_20260807T174904381704076Z/oracle',
        expected_token_manifest_sha256=delivery.TOKEN_MANIFEST_SHA256,
        expected_dsa_manifest_sha256='f8154c5f79b909efd9ebc14c8e004925482844d05ef28fcf0a4d29bb4a7b26da',
        expected_token_success_sha256='38c0aeb6c4833a0256d4e50152b645e85d24a4f00ca7b2b1731db2d892c5b3cc',
        expected_dsa_success_sha256='0b798974ae8a9f95c32d3aa2eff532213624f1e2ae7f1de809a161e18dbdf1b9',
    )
    bucket = storage.Client().get_bucket('driftbench-dsv4-uc', timeout=30)
    if bucket.location.upper() != 'US-CENTRAL2':
        raise ValueError('wrong storage region')
    objects = [obj for rank in original['rank_records'] for obj in rank['records'][:2]]
    expected = {f'results/{original["tag"]}/host_records/runner.rank{r}.{ext}'
                for r in range(8) for ext in ('json', 'npz')}
    if len(objects) != 16 or {obj['name'] for obj in objects} != expected:
        raise ValueError('original object inventory differs')

    def read(obj: dict) -> tuple[str, bytes]:
        blob = bucket.blob(obj['name'], generation=int(obj['generation']))
        blob.reload(timeout=30)
        if blob.size != obj['size'] or blob.crc32c != obj['crc32c']:
            raise ValueError('original generation metadata differs')
        data = blob.download_as_bytes(if_generation_match=int(obj['generation']), timeout=60)
        if len(data) != obj['size'] or sha256(data).hexdigest() != obj['sha256']:
            raise ValueError('original generation bytes differ')
        return Path(obj['name']).name, data

    with ThreadPoolExecutor(max_workers=4) as pool:
        payloads = dict(pool.map(read, objects))
    rows = []
    for rank in range(8):
        record = json.loads(payloads[f'runner.rank{rank}.json'])
        with np.load(BytesIO(payloads[f'runner.rank{rank}.npz']), allow_pickle=False) as archive:
            arrays = {name: archive[name] for name in archive.files}
        if record['code_hash'] != original['pin'] or record['status'] != 'ORACLE_MISMATCH':
            raise ValueError('original execution identity/status differs')
        manifest = {name: dict(dtype=value.dtype.name, shape=list(value.shape),
                               sha256=sha256(value.tobytes()).hexdigest())
                    for name, value in arrays.items()}
        if manifest != record['numerical_tensors']['arrays']:
            raise ValueError('original array manifest differs')
        if rank and record['observed_generated_token_ids'] != rows[0]['tokens']:
            raise ValueError('replicated token outputs differ')
        tokens = delivery.token_result(record['observed_generated_token_ids'], oracle,
            tokenizer_root=Path('/home/gianl/gcs-models/models/GLM-5.2-FP8'))
        dsa = [delivery.dsa_result(
            producer_layer_ids=arrays['dsa_producer_layer_ids'],
            selected_positions=arrays['dsa_selected_positions'][step],
            selected_valid_counts=arrays['dsa_selected_valid_counts'][step],
            selected_scores=arrays['dsa_selected_scores'][step],
            decode_position=8155 + step, step=step,
        ) for step in range(14)]
        cache = validate_ws32_cache_probe(
            position=arrays['cache_position'],
            kv_rows=arrays['cache_kv_bfloat16_bits'].view(ml_dtypes.bfloat16),
            index_rows=arrays['cache_index_bfloat16_bits'].view(ml_dtypes.bfloat16),
            contract_valid=arrays['cache_contract_valid'], expected_position=8182,
            num_layers=78, full_indexer_count=21, packed_cache_width=640, index_width=128,
        )
        healthy = record['state'] == dict(position=[8183], context_lengths=[8184], contract_valid=[True])
        rows.append(dict(rank=rank, tokens=record['observed_generated_token_ids'],
            original_status=record['status'], task=tokens,
            dsa_steps_passed=sum(item['passed'] for item in dsa),
            cache_probe_passed=cache['passed'] and cache == record['cache_write_probe'],
            final_state_healthy=healthy))
    passed = all(row['task']['passkey_matches_gold'] and row['dsa_steps_passed'] == 14
                 and row['cache_probe_passed'] and row['final_state_healthy'] for row in rows)
    print(json.dumps(dict(
        schema_version=1, assessed_utc=datetime.now(timezone.utc).isoformat(),
        contract=delivery.CONTRACT, task_smoke_passed=passed, source_receipt=RECEIPT,
        source_receipt_sha256=RECEIPT_SHA, original_tag=original['tag'],
        original_status_unchanged=original['status'], bucket_location=bucket.location,
        generation_qualified_objects=objects, ranks=rows,
        model_card_quality_claim=False, performance_claim=False, protected_seal=False,
        dsa_scope='SELECTED_ROW_STRUCTURE;UNSELECTED_SCORE_ROWS_NOT_RECOMPUTED',
        execution='OFFLINE_REPLAY_NO_TPU_NO_DB_OR_CLOUD_WRITES',
        source_sha256={str(path.relative_to(ROOT)): sha256(path.read_bytes()).hexdigest()
                       for path in (Path(__file__).resolve(), ROOT / 'glm_tpu/greenfield/validation/ws32_delivery_quality.py')},
    ), sort_keys=True))
    if not passed:
        raise SystemExit(1)


if __name__ == '__main__':
    main()

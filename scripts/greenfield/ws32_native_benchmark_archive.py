"""Join replayed answers, atomic DB rows and bounded regional originals.

Called only by the leased outer after authenticated idle and original replay.
EVIDENCE_ARCHIVED is a transport/provenance receipt, NEVER a quality SUCCESS.
Partial campaigns and unjudged maths remain incomplete. No model rerun, full DB
snapshot, checkpoint copy or historical row edit is used for recovery.
"""
from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
from typing import Any

from scripts.greenfield import watch_ws32_run as watch
from scripts.greenfield import ws32_native_benchmark_transport as cold
from scripts.greenfield import ws32_native_benchmark_collect as requests
from scripts.greenfield import ws32_native_benchmark_database as database
from scripts.greenfield.collect_ws32_worker_evidence import digest_file, publish_exact
from scripts.greenfield.fp8_baseline_guard import validate_fleet
from scripts.greenfield.ws32_native_benchmark_protocol import canonical
from scripts.greenfield.ws32_native_benchmark_result import read

CONTROLLER_CAP = 128 << 20
FILE_CAP = 64 << 20
ARCHIVE_CAP = 10 << 30
FILES = ('launch.json', 'census_pre.txt', 'census_post.txt', 'sync.txt', 'prepared.txt',
         'final_watch.jsonl', 'requests.json', 'protocol.json', 'request_replay.json',
         'collection.json', 'db_link.json')


def verify_ownership(root: Path, tag: str, pin: str) -> None:
    """Bind original worker PID/start/boot/argv to the authenticated SSH history."""
    from scripts.greenfield.launch_ws32_native_benchmark import observe_originals
    original = None
    rows = read(root/'final_watch.jsonl', FILE_CAP).decode().splitlines()
    idle = 0
    for line in rows:
        record = json.loads(line)
        if record.get('tag') != tag or record.get('pin') != pin:
            raise ValueError('native archived watcher identity differs')
        if record.get('status') != 'OBSERVED':
            idle = 0
            continue
        fleet = watch.parse_fleet('\n'.join(watch.PREFIX+json.dumps(r)
            for r in record['fleet']), tag, pin)
        original = observe_originals(original, fleet)
        idle = idle+1 if all(not r['processes'] and not r['holders'] for r in fleet) else 0
    if original is None or idle < 2 or not all(len(r['processes']) == 1 for r in original):
        raise ValueError('native archive lacks original owners and two idle observations')
    for phase in ('pre', 'post'):
        census = [json.loads(line[len('FP8_IDLE '):])
                  for line in read(root/f'census_{phase}.txt', FILE_CAP).decode().splitlines()
                  if line.startswith('FP8_IDLE ')]
        if {(r['host'], r['boot_id']) for r in census} != {(r['host'], r['boot_id']) for r in original}:
            raise ValueError('native root census differs from original host/boot identities')
    for rank, observed in enumerate(original):
        record = json.loads(read(root/'collected'/f'runner.rank{rank}.json', 2<<20))
        process = observed['processes'][0]
        expected = dict(pid=process['pid'], start_ticks=process['start_ticks'],
            argv_sha256=process['argv_sha256'], hostname=observed['host'], boot_id=observed['boot_id'])
        if (record['code_hash'] != pin or record['launch_process_id'] != rank
                or record['owner'] != expected):
            raise ValueError('native original runner differs from authenticated process owner')


def archive(*, root: Path, tag: str, pin: str, report: dict,
            cold_receipts: list[dict], request_receipts: list[dict], blobs: dict,
            client: Any, db_path: Path = database.PRIMARY_DB) -> dict:
    cold._identity(tag, pin, 0)
    if root.name != tag:
        raise ValueError('native archive root/tag differs')
    bucket = cold._bucket(client)
    for phase in ('pre', 'post'):
        validate_fleet(read(root/f'census_{phase}.txt', FILE_CAP).decode())
    launch = json.loads(read(root/'launch.json', FILE_CAP))
    if launch['tag'] != tag or launch['code_hash'] != pin:
        raise ValueError('native archive launch identity differs')
    final_watch = root/'final_watch.jsonl'
    if not final_watch.exists():
        requests._write_once(final_watch, read(root/'native_watch.jsonl', FILE_CAP))
    verify_ownership(root, tag, pin)
    if report['completed_requests'] and report.get('trace_physical_coverage_verified') is not True:
        raise ValueError('native archive lacks physical request trace replay')
    if not report.get('cold'):
        raise ValueError('native archive lacks original cold replay')
    payload = json.loads(read(root/'requests.json', 16<<20))
    plan = json.loads(read(root/'protocol.json', 1<<20))
    # Exact children and their manifests were read back by both collectors.
    # Reconcile their union again before linking rows or publishing a terminal.
    expected: dict[str, tuple] = {}
    if len(cold_receipts) != 8 or len(request_receipts) != 8:
        raise ValueError('native archive requires both eight-rank collections')
    for rank in range(8):
        for manifest in (cold_receipts[rank]['manifest'], request_receipts[rank]):
            if (manifest['rank'] != rank or manifest['tag'] != tag or manifest['code_hash'] != pin):
                raise ValueError('native archive manifest identity differs')
            for row in manifest['files']:
                facts = (row['generation'], row['size'], row['crc32c'])
                if row['name'] in expected and expected[row['name']] != facts:
                    raise ValueError('native shared original identity differs')
                expected[row['name']] = facts
        for prefix in (cold.prefix(tag, rank), requests.prefix(tag, rank)):
            blob = blobs[prefix+'manifest.json']
            expected[blob.name] = (str(blob.generation), int(blob.size), blob.crc32c)
    worker_names = {name for name in blobs if name.startswith((f'results/{tag}/native_cold/',
                                                             f'results/{tag}/native_requests/'))}
    if worker_names != set(expected):
        raise ValueError('native archive original union differs')
    for name, facts in expected.items():
        blob = bucket.get_blob(name)
        if blob is None or (str(blob.generation), int(blob.size), blob.crc32c) != facts:
            raise ValueError('native archived generation changed after collection')
    worker_bytes = sum(facts[1] for facts in expected.values())
    if worker_bytes+CONTROLLER_CAP+(8<<20) > ARCHIVE_CAP:
        raise ValueError('native archive exceeds registered 10GiB budget')
    link = database.record_result(database=db_path, root=root, tag=tag, pin=pin,
                                  payload=payload, plan=plan, report=report)
    link_bytes = canonical(link)+b'\n'
    if len(link_bytes) > FILE_CAP:
        raise ValueError('native logical DB export exceeds controller file cap; rows remain saved')
    requests._write_once(root/'db_link.json', link_bytes)
    # Export only THIS run's logical rows; primary DB/history stay in place.
    paths = [root/name for name in FILES]
    for path in paths:
        read(path, FILE_CAP)
    if sum(path.stat().st_size for path in paths) > CONTROLLER_CAP:
        raise ValueError('native controller archive exceeds 128MiB cap')
    controller = [publish_exact(bucket, f'results/{tag}/controller/{path.name}',
                  path, digest_file(path), compressed=False) for path in paths]
    ledger = dict(schema='ws32_native_archive_ledger_v1', tag=tag, code_hash=pin,
        worker_objects=[dict(name=name, generation=facts[0], size=facts[1], crc32c=facts[2])
                        for name, facts in sorted(expected.items())], controller=controller)
    requests._write_once(root/'archive_ledger.json', canonical(ledger)+b'\n')
    if (root/'archive_ledger.json').stat().st_size > 4<<20:
        raise ValueError('native archive ledger exceeds cap')
    ledger_receipt = publish_exact(bucket, f'results/{tag}/controller/archive_ledger.json',
        root/'archive_ledger.json', digest_file(root/'archive_ledger.json'), compressed=False)
    receipt = dict(schema='ws32_native_evidence_archived_v1', tag=tag, code_hash=pin,
        run_id=link['run_id'], db_rows_sha256=link['rows_sha256'],
        request_replay_sha256=sha256(canonical(report)).hexdigest(), ledger=ledger_receipt,
        completed_requests=report['completed_requests'], benchmarks=report['benchmarks'],
        evidence_archived=True, quality_pass_claim=False, matched_card_parity_claim=False,
        project_complete=False, cleanup='authenticated8host_idle',
        limits='AIME unjudged; incomplete sets have no full score; local JSONL delivery, not HTTP')
    requests._write_once(root/'EVIDENCE_ARCHIVED.json', canonical(receipt)+b'\n')
    publish_exact(bucket, f'results/{tag}/EVIDENCE_ARCHIVED.json', root/'EVIDENCE_ARCHIVED.json',
                  digest_file(root/'EVIDENCE_ARCHIVED.json'), compressed=False)
    return receipt

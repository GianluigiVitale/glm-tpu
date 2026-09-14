"""Atomic user-request provenance, without benchmark scoring or raw prompt copies.

The outer sealer authenticates original replay and cleanup before calling this.
Private request/answer bytes remain in the generation-bound regional archive;
only their identifiers, hashes, measured timings and limits are linked here.
Retry uses the same rows, never regenerates tokens or duplicates a run.
"""

from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path

from bench import provenance
from glm_tpu import user_request
from scripts.release import ws32_user_worker as worker
from scripts.release.ws32_user_result import SCHEMA, same, seconds
from scripts.greenfield.ws32_native_benchmark_database import (
    Transaction,
    export_rows,
    PRIMARY_DB,
)
from glm_tpu.host_paths import _plain_path

LEDGER = """CREATE TABLE IF NOT EXISTS user_request_links (
    tag TEXT PRIMARY KEY, run_id INTEGER NOT NULL REFERENCES runs(run_id),
    code_hash TEXT NOT NULL, report_sha256 TEXT NOT NULL,
    request_file_sha256 TEXT NOT NULL, rows_sha256 TEXT NOT NULL
)"""


def validate_report(report: dict, tag: str, pin: str) -> None:
    worker.identity(tag, pin, 0)
    expected = dict(
        schema=SCHEMA,
        complete=True,
        benchmark=False,
        quality_score=None,
        protected_result_sealed=False,
        cold_admission_verified=True,
        worker_ownership_verified=True,
        cleanup="authenticated8host_idle",
        execution_code_hash=pin,
        replay_code_hash=pin,
        code_hash=pin,
        delivery_boundary="rank0_local_jsonl_write_flush",
        durable_resume_verified=False,
    )
    same({key: report[key] for key in expected}, expected, "user database replay scope")
    count = report["generated_tokens"]
    if type(count) is not int or not 0 < count <= user_request.MAX_NEW:
        raise ValueError("user database token count invalid")
    same(
        report["trace_physical_coverage_verified"],
        count > 1,
        "user database trace coverage",
    )
    if not report["cold"] or count > 1 and not report["trace"]:
        raise ValueError("user database requires original cold/trace replay")
    for key in ("request_file_sha256", "request_sha256", "token_ids_sha256"):
        value = report[key]
        if (
            type(value) is not str
            or len(value) != 64
            or any(c not in "0123456789abcdef" for c in value)
        ):
            raise ValueError("user database original digest invalid")
    for key in (
        "cold_load_compile_seconds",
        "ttft_seconds",
        "delivered_request_seconds",
    ):
        seconds(report[key])


def record_result(
    *, database: Path = PRIMARY_DB, tag: str, pin: str, report: dict
) -> dict:
    validate_report(report, tag, pin)
    _plain_path(database)
    digest = sha256(user_request.canonical(report)).hexdigest()
    request_digest = report["request_file_sha256"]
    conn = provenance.connect(str(database))
    try:
        conn.execute(LEDGER)
        conn.commit()
        conn.execute("BEGIN IMMEDIATE")
        existing = conn.execute(
            "SELECT run_id,code_hash,report_sha256,request_file_sha256,rows_sha256 "
            "FROM user_request_links WHERE tag=?",
            (tag,),
        ).fetchone()
        if existing is not None:
            run_id, old_pin, old_digest, old_request, old_rows = existing
            if (old_pin, old_digest, old_request) != (pin, digest, request_digest):
                raise ValueError(
                    "user run tag already links different originals; no overwrite"
                )
            rows = export_rows(conn, run_id)
            if sha256(user_request.canonical(rows)).hexdigest() != old_rows:
                raise ValueError("user linked database rows changed")
        else:
            # No model-card score, legacy fork, raw prompt or fabricated asked time.
            tx = Transaction(conn)
            run_id = provenance.start_run(
                tx,
                model="zai-org/GLM-5.2-FP8",
                revision=None,
                env=dict(
                    native=True,
                    benchmark=False,
                    run_tag=tag,
                    code_hash=pin,
                    request_sha256=report["request_sha256"],
                    request_file_sha256=request_digest,
                    token_ids_sha256=report["token_ids_sha256"],
                    report_sha256=digest,
                    originals_prefix=f"gs://driftbench-dsv4-uc/results/{tag}/",
                    prompt_tokens=report["prompt_tokens"],
                    generated_tokens=report["generated_tokens"],
                    finish_reason=report["finish_reason"],
                    delivery_boundary=report["delivery_boundary"],
                    request_wall_instrumented=report["request_wall_instrumented"],
                    live_session_resume=report["live_session_resume"],
                    durable_resume_verified=False,
                    quality_score=None,
                ),
                note="User response original replay; NOT a benchmark or task-quality score. "
                + tag,
                harness_repo=str(worker.REPO),
                fork_repo=str(worker.REPO),
            )
            metrics = {
                "cold_load_compile_seconds": (1, report["cold_load_compile_seconds"]),
                "local_ttft_seconds": (1, report["ttft_seconds"]),
                "local_delivered_request_seconds": (
                    1,
                    report["delivered_request_seconds"],
                ),
                "ordinary_decode_p50_ms": (
                    report["ordinary_decode_samples"],
                    report["decode_p50_ms"],
                ),
                "ordinary_decode_p99_ms": (
                    report["ordinary_decode_samples"],
                    report["decode_p99_ms"],
                ),
                "ordinary_decode_wall_tokens_per_second": (
                    report["ordinary_decode_samples"],
                    report["decode_wall_tokens_per_second"],
                ),
            }
            now = datetime.now(timezone.utc).isoformat()
            for name, (n, value) in metrics.items():
                if value is not None:
                    seconds(
                        value
                    )  # finite, nonnegative scalar; metric name supplies units
                conn.execute(
                    "INSERT INTO summary(run_id,benchmark,created_utc,n,metric,value,card_value,delta,note) "
                    "VALUES (?,?,?,?,?,?,NULL,NULL,?)",
                    (
                        run_id,
                        "user_request",
                        now,
                        n,
                        name,
                        value,
                        "Not task quality; local JSONL delivery, not network. Observer sample excluded from ordinary decode.",
                    ),
                )
            rows = export_rows(conn, run_id)
            conn.execute(
                "INSERT INTO user_request_links VALUES (?,?,?,?,?,?)",
                (
                    tag,
                    run_id,
                    pin,
                    digest,
                    request_digest,
                    sha256(user_request.canonical(rows)).hexdigest(),
                ),
            )
        conn.commit()
        return dict(
            schema="glm_ws32_user_database_link_v1",
            tag=tag,
            run_id=run_id,
            code_hash=pin,
            request_file_sha256=request_digest,
            report_sha256=digest,
            rows=rows,
            rows_sha256=sha256(user_request.canonical(rows)).hexdigest(),
            benchmark=False,
            quality_score=None,
            protected_result_sealed=False,
        )
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()

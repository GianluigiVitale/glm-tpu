"""Partial acquisition diagnostics survive the real worker/parser/upload path."""

import ast
from hashlib import sha256
import json
from pathlib import Path
import shlex
import subprocess
from types import SimpleNamespace

import pytest

from glm_tpu.greenfield.benchmarking import ws32_batched_prefill
from glm_tpu.greenfield.validation.ws32_prefill import PREFILL_MODE
from scripts.greenfield import run_short_decoder_ws32 as worker
from scripts.greenfield.ws32_acquisition_journal import Ws32AcquisitionJournal


ROOT = Path(__file__).resolve().parents[3]
GRAPHS = (
    "exact_materialize",
    "exact_promote",
    "prefill_chunk",
    "prefill_tail",
    "observer",
    "decode",
    "cache_probe",
)


def journal(tmp_path):
    return Ws32AcquisitionJournal(
        tmp_path / "acquisition_journal.rank0.jsonl",
        {
            "prefill_mode": PREFILL_MODE,
            "compile_only": True,
            "code_hash": "a" * 40,
            "hostname": "test-w-0",
            "local_device_slots": [0, 1, 2, 3],
        },
    )


def rows(value):
    return [json.loads(line) for line in value.path.read_text().splitlines()]


def compiled(value, graph):
    value.begin(graph)
    value.compiled(
        graph,
        seconds=1.25,
        memory={
            "argument_size_in_bytes": 100,
            "alias_size_in_bytes": 0,
            "temp_size_in_bytes": 7,
            "output_size_in_bytes": 8,
        },
        device_memory=[{"bytes_in_use": 200, "peak_bytes_in_use": 210}],
    )


@pytest.mark.parametrize("graph", GRAPHS)
def test_actual_write_helper_preserves_bytes_memory_and_partial_status(
    tmp_path, monkeypatch, graph
):
    value = journal(tmp_path)
    compiled(value, graph)
    hlo_dir = tmp_path / "hlo"
    report = {"passed": True, "violations": []}

    def validate(*args, **kwargs):
        assert rows(value)[-1]["stage"] == "raw_written"
        assert (hlo_dir / f"{graph}.stablehlo.mlir").read_text() == "stable"
        assert (hlo_dir / f"{graph}.optimized_hlo.txt").read_text() == "optimized"
        return (
            report
            if graph.startswith("prefill")
            else SimpleNamespace(to_dict=lambda: report)
        )

    monkeypatch.setattr(
        ws32_batched_prefill, "inspect_ws32_batched_prefill_hlo", validate
    )
    monkeypatch.setattr(worker, "validate_ws32_decoder_hlo", validate)
    monkeypatch.setattr(worker, "validate_ws32_exact_dsa_materializer_hlo", validate)
    kwargs = dict(
        graph=graph,
        lowered=SimpleNamespace(compiler_ir=lambda **kw: "stable"),
        compiled=SimpleNamespace(as_text=lambda: "optimized"),
        hlo_dir=hlo_dir,
        expected_stable="0" * 64,
        expected_optimized="0" * 64,
        acquisition_journal=value,
    )
    if graph.startswith("exact"):
        result = worker._write_exact_materializer_graph(**kwargs)
    else:
        result, _, _ = worker._write_graph(
            **kwargs,
            hidden_size=6144,
            exact_dsa=True,
            strategy_nd_dense=True,
            prefill_mode=PREFILL_MODE,
            block_rows=17,
        )
    assert result == report
    record = rows(value)
    assert [r["stage"] for r in record] == [
        "identity",
        "lower_compile_started",
        "compiled",
        "raw_written",
        "inspected",
    ]
    assert record[2]["compiled_memory"]["temp_size_in_bytes"] == 7
    assert record[2]["device_memory"][0]["peak_bytes_in_use"] == 210
    assert record[3]["optimized_hlo_sha256"] == sha256(b"optimized").hexdigest()
    assert all(
        r["status"] == "HLO_ACQUISITION_PARTIAL"
        and not r["performance_claim"]
        and not r["numerical_claim"]
        for r in record
    )
    assert not (tmp_path / "runner.rank0.json").exists()
    with pytest.raises(RuntimeError):
        worker._require_acquisition_authorized({"journal": report}, exact_dsa=True)
    value.close()


def test_actual_inspection_exception_preserves_partial_and_rethrows(
    tmp_path, monkeypatch
):
    value = journal(tmp_path)
    compiled(value, "prefill_chunk")

    def refuse(*args, **kwargs):
        raise RecursionError("injected deep graph failure")

    monkeypatch.setattr(
        ws32_batched_prefill, "inspect_ws32_batched_prefill_hlo", refuse
    )
    with pytest.raises(RecursionError, match="injected"):
        worker._write_graph(
            graph="prefill_chunk",
            lowered=SimpleNamespace(compiler_ir=lambda **kw: "s"),
            compiled=SimpleNamespace(as_text=lambda: "o"),
            hlo_dir=tmp_path / "hlo",
            expected_stable="0" * 64,
            expected_optimized="0" * 64,
            hidden_size=6144,
            exact_dsa=True,
            strategy_nd_dense=True,
            prefill_mode=PREFILL_MODE,
            block_rows=17,
            acquisition_journal=value,
        )
    record = rows(value)
    assert record[-1]["stage"] == "inspection_failed"
    assert record[-1]["exception_type"] == "RecursionError"
    assert record[-2]["stage"] == "raw_written"
    assert record[-3]["stage"] == "compiled"
    value.close()
    with pytest.raises(FileExistsError):
        journal(tmp_path)


def test_missing_compile_memory_and_non_acquisition_modes_refuse(tmp_path):
    for mode, acquire in (("serial_teacher_forced_v1", True), (PREFILL_MODE, False)):
        with pytest.raises(ValueError):
            Ws32AcquisitionJournal(
                tmp_path / "forbidden", {"prefill_mode": mode, "compile_only": acquire}
            )
    value = journal(tmp_path)
    value.begin("prefill_chunk")
    with pytest.raises(ValueError, match="compile memory"):
        value.inspect("prefill_chunk", "s", "o", lambda: {})
    value.close()


def test_worker_records_each_compile_before_inspection_and_tail_before_release():
    source = Path(worker.__file__).read_text()
    main = next(
        n
        for n in ast.parse(source).body
        if isinstance(n, ast.FunctionDef) and n.name == "main"
    )
    calls = sorted(
        (n.lineno, n.func.id, n)
        for n in ast.walk(main)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Name)
        and n.func.id
        in {
            "begin_compile",
            "record_compile",
            "_write_graph",
            "_write_exact_materializer_graph",
        }
    )
    assert [name for _, name, _ in calls] == [
        name
        for writer in (
            "_write_exact_materializer_graph",
            "_write_exact_materializer_graph",
            "_write_graph",
            "_write_graph",
            "_write_graph",
            "_write_graph",
        )
        for name in ("begin_compile", "record_compile", writer)
    ]
    for _, name, call in calls:
        if name.startswith("_write"):
            assert any(k.arg == "acquisition_journal" for k in call.keywords)
    assert source.index("record_compile(graph)") < source.index("del prefill_compiled")
    assert "if batched_prefill and args.compile_only:" in source
    assert '"local_device_ids": [int(device.id)' in source


@pytest.mark.parametrize("upload_rc", [0, 1])
def test_actual_remote_upload_function_retains_journal_and_propagates_failure(
    tmp_path, upload_rc
):
    source = (ROOT / "scripts/greenfield/run_short_decoder_ws32.sh").read_text()
    assignment = next(
        line for line in source.splitlines() if line.startswith("execute_command=")
    )
    expanded = subprocess.run(
        ["bash", "-c", assignment + '\nprintf "%s" "$execute_command"'],
        check=True,
        text=True,
        capture_output=True,
    ).stdout
    function = (
        "upload(){"
        + expanded.split("upload(){", 1)[1].split('; trap "upload || true" EXIT;', 1)[0]
    )
    (tmp_path / "acquisition_journal.rank0.jsonl").write_text(
        '{"status":"HLO_ACQUISITION_PARTIAL"}\n'
    )
    (tmp_path / "trace").mkdir()
    command = f"""run={shlex.quote(str(tmp_path))}; idx=0; tag=example; remote=gs://driftbench-dsv4-uc/results/example
output="$run/absent"; tensors="$run/absent"; log="$run/absent"; ended="$run/absent"; hlo="$run/hlo"; trace="$run/trace"
gcloud() {{ printf '%s\\n' "$*" >&3; return {upload_rc}; }}
{function}
upload
"""
    result = subprocess.run(
        ["bash", "-c", "exec 3>&1\n" + command], text=True, capture_output=True
    )
    assert result.returncode == upload_rc, result.stderr
    assert "storage cp --no-clobber" in result.stdout
    assert "diagnostic_local/example/acquisition_journal.rank0.jsonl" in result.stdout
    assert "host_records/runner" not in result.stdout

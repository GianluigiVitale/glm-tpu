"""CPU-only token-equivalence comparator for real TPU runs (S8), ported from the proven private
``compare_golden.py`` (including its fix: resident-round ``output_directory`` is relative to the
run root, batch items to their job directory).

Goldens are run directories given as arguments. Requests are matched by ``request_sha256``,
which is identical when the exact prepared request object is replayed. Checks per request:
token ids from ``tokens.jsonl`` equal the golden's, the receipt ``token_sha256`` equals both, all
ranks agree, ``finish_reason``/``stop_cause`` equal, the extracted final answer equals the golden's
(compared, never printed), and slowest-host decode speed within the tolerance.

Output contains request ids, counts, verdicts and hashes only -- never prompt, answer or token
text. Private run directories stay outside Git; the optional ``--out`` report holds the same
hash-only rows.
"""

from __future__ import annotations

import argparse
from decimal import Decimal, InvalidOperation
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Iterator

NUMBER = r"[-+]?(?:\d[\d,]*(?:\.\d+)?|\.\d+)"


def token_ids(path: Path) -> list[int]:
    return [json.loads(line)["token_id"] for line in path.open()]


def token_sha(ids: list[int]) -> str:
    import numpy as np

    return hashlib.sha256(np.asarray(ids, np.int32).tobytes()).hexdigest()


def number(value: str) -> str | None:
    value = value.strip().replace(",", "").replace("$", "").strip()
    if not re.fullmatch(NUMBER, value):
        return None
    try:
        return str(Decimal(value).normalize()) if Decimal(value) else "0"
    except InvalidOperation:
        return None


def extract(text: str, eos: bool) -> str | None:
    """Same policy as the GSM8K scorer: completed final channel only."""
    if not eos or "</think>" not in text:
        return None
    final = text.split("</think>", 1)[1].split("<|user|>", 1)[0].strip()
    boxed = re.findall(r"\\boxed\{([^{}]*)\}", final)
    if boxed:
        return number(boxed[-1])
    marked = re.findall(r"####\s*(" + NUMBER + r")", final)
    if marked:
        return number(marked[-1])
    values = re.findall(NUMBER, final)
    return number(values[-1]) if values else None


def runner_rows(directory: Path) -> list[dict[str, Any]]:
    return [json.loads((directory / f"runner.rank{r}.json").read_text()) for r in range(8)]


def jobs(run: Path) -> Iterator[tuple[Path, list[dict[str, Any]]]]:
    """Yield (job_dir, rows) for the root request and every resident round."""
    if (run / "runner.rank7.json").exists():
        yield run, runner_rows(run)
    for job in sorted(run.glob("resident-[0-9][0-9][0-9][0-9]")):
        if (job / "runner.rank7.json").exists():
            yield job, runner_rows(job)


def items(run: Path) -> dict[str, dict[str, Any]]:
    """Per-request records: request_sha -> dict(tokens path, answer path, per-rank reports)."""
    out: dict[str, dict[str, Any]] = {}
    for job, rows in jobs(run):
        for i in range(len(rows[0]["requests"])):
            reports = [row["requests"][i] for row in rows]
            relative = reports[0]["output_directory"]
            # Resident rounds record output_directory relative to the run root
            # (e.g. 'resident-0001'); batch items record it relative to the job.
            item_dir = job / relative if (job / relative / "tokens.jsonl").exists() else run / relative
            out[reports[0]["request_sha256"]] = dict(
                job=job, dir=item_dir, reports=reports, tokens=item_dir / "tokens.jsonl", answer=item_dir / "answer.txt"
            )
    return out


def golden_index(runs: list[Path]) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    for run in runs:
        for sha, record in items(run).items():
            index.setdefault(sha, record)
    return index


def summarize(record: dict[str, Any]) -> dict[str, Any]:
    reports = record["reports"]
    ids = token_ids(record["tokens"])
    peaks = [m["peak_bytes_in_use"] for r in reports for m in r["peak_memory"]]
    answer = extract(record["answer"].read_text(), reports[0]["finish_reason"] == "eos")
    return dict(
        emitted=reports[0]["emitted"],
        finish_reason=reports[0]["finish_reason"],
        stop_cause=reports[0].get("stop_cause"),
        token_sha256=reports[0]["token_sha256"],
        jsonl_sha256=token_sha(ids),
        jsonl_len=len(ids),
        ranks_agree=len({(r["token_sha256"], r["emitted"]) for r in reports}) == 1,
        slowest_decode_tps=min((r["decode_tokens_per_second"] or 0) for r in reports),
        slowest_prefill_s=max(r["prefill_seconds"] for r in reports),
        peak_bytes_max=max(peaks),
        peak_bytes_min=min(peaks),
        answer_sha256=None if answer is None else hashlib.sha256(answer.encode()).hexdigest(),
        _answer=answer,
        _ids=ids,
    )


def compare(run: Path, goldens: list[Path], *, speed_tolerance: float = 0.03) -> dict[str, Any]:
    golden = golden_index(goldens)
    rows, failures = [], []
    for sha, record in items(run).items():
        new = summarize(record)
        row: dict[str, Any] = dict(
            request_sha256=sha,
            request_id=record["reports"][0]["request_id"],
            new={k: v for k, v in new.items() if not k.startswith("_")},
        )
        old_record = golden.get(sha)
        if old_record is None:
            row["golden"] = None
            failures.append([sha, "no golden for this request object"])
        else:
            old = summarize(old_record)
            pairs = zip(new["_ids"], old["_ids"], strict=False)
            first = next(
                (i for i, (a, b) in enumerate(pairs) if a != b),
                None if len(new["_ids"]) == len(old["_ids"]) else min(len(new["_ids"]), len(old["_ids"])),
            )
            ratio = new["slowest_decode_tps"] / old["slowest_decode_tps"] if old["slowest_decode_tps"] else None
            job = old_record["job"]
            row.update(
                golden_run=job.parent.name if job.name.startswith("resident-") else job.name,
                golden={k: v for k, v in old.items() if not k.startswith("_")},
                tokens_identical=new["_ids"] == old["_ids"],
                first_divergent_index=first,
                decode_speed_ratio=ratio,
                peak_equal=new["peak_bytes_max"] == old["peak_bytes_max"],
            )
            checks = dict(
                tokens=row["tokens_identical"] and new["token_sha256"] == old["token_sha256"],
                receipt_matches_jsonl=new["jsonl_sha256"] == new["token_sha256"] and new["jsonl_len"] == new["emitted"],
                ranks_agree=new["ranks_agree"],
                stop=new["finish_reason"] == old["finish_reason"] and new["stop_cause"] == old["stop_cause"],
                answer=new["_answer"] == old["_answer"],
                speed=ratio is None or ratio >= 1 - speed_tolerance,
            )
            row["checks"] = checks
            failures += [[sha, k] for k, v in checks.items() if not v]
        rows.append(row)
    summary_path = run / "summary.json"
    terminal_path = run / "controller_terminal.json"
    return dict(
        run=run.name,
        goldens=[g.name for g in goldens],
        requests=len(rows),
        failures=failures,
        summary_passed=json.loads(summary_path.read_text()).get("passed") if summary_path.exists() else None,
        controller_terminal=json.loads(terminal_path.read_text()) if terminal_path.exists() else None,
        status="pass" if rows and not failures else "fail",
        rows=rows,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="token-equivalence comparator for TPU runs (hash-only output)")
    parser.add_argument("run", type=Path)
    parser.add_argument("--golden", type=Path, action="append", required=True, help="golden run directory (repeatable)")
    parser.add_argument("--out", type=Path, help="write the hash-only JSON report here (outside Git)")
    parser.add_argument("--speed-tolerance", type=float, default=0.03)
    args = parser.parse_args(argv)
    result = compare(args.run, args.golden, speed_tolerance=args.speed_tolerance)
    if args.out:
        args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    for row in result["rows"]:
        checks = row.get("checks", {})
        verdict = "IDENTICAL" if row.get("tokens_identical") else f"DIVERGES@{row.get('first_divergent_index')}"
        failed = [k for k, v in checks.items() if not v]
        speed = f"speed_ratio={row['decode_speed_ratio']:.4f}" if row.get("decode_speed_ratio") else ""
        print(
            row["request_id"],
            verdict,
            row["new"]["emitted"],
            row["new"]["token_sha256"][:16],
            speed,
            "FAIL:" + ",".join(failed) if failed else ("ok" if checks else "NO-GOLDEN"),
        )
    print("PASS" if result["status"] == "pass" else f"FAIL ({len(result['failures'])})")
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())

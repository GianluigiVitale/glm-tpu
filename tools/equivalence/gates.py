"""Gate orchestration: record the baseline data and check the current tree against it.

Every heavy computation runs in a child process (``run_child``); this module only compares
JSON. A gate report is ``{gate, status, seconds, ...diff}`` with ``status`` in
``pass | fail | skip | refused``. Data files live in ``tests/golden/data`` and are written only by
``record`` (integrator only; DESIGN.md section 7.5.9).

``check`` statuses: ``pass``; ``fail`` (a difference, or a missing/unusable data file); ``error``
(the gate's child crashed or timed out); ``skip`` (installed package versions differ from the
recorded ones). Only ``pass`` counts as passing unless the caller explicitly allows skips.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import time
from typing import Any

from .common import DATA, digest_json, read_json, run_child, source_record, static_environment, write_json

DATA_FILES = {
    "G1": "fingerprints_fixture.json",
    "G1-protocol": "load_protocol_fixture.json",
    "G2": "fingerprints_production.json",
    "G2-protocol": "load_protocol_production.json",
    "G3": "cpu_digests.json",
    "G4": "checkpoint_identity.json",
    "G6": "import_closure.json",
    "G6-static": "import_closure.json",  # check-only subset of G6 (entry-module stages + static scan)
    "G7": "trace_closure.json",
    "G9": "wire.json",
    "G9-http": "http.json",
    "fixture": "fixture.json",
}
# Gates that build the real runtime or run fixture-scale programs on the 32-device CPU mesh
# (DESIGN 7.5.10: refused while a TPU run is live on the host). Light: G4 (its loader exercise
# places kilobytes on the mesh), G6-static (entry-module closures and the static scan, one device,
# no runtime build), G9 and the fixture record check.
HEAVY = {"G1", "G1-protocol", "G2", "G2-protocol", "G3", "G6", "G7", "G14"}
# Graph and identity goldens (and the frozen safety facts in the G1/G2 files) are recorded only
# from production paths equal to 181c013e; the characterization goldens (load protocol and
# defaults, closures, trace, wire) may be re-recorded on a changed tree, but only with a re-baseline
# marker whose reason is exactly one H number, stage or work-unit token (DESIGN 7.5.9).
FROZEN_DATA = ("G1", "G2", "G3", "G4", "fixture")
# The load protocol, production defaults and full admission reports the program child also
# records: host behaviour, option defaults and wording that planned stages change on purpose (S1 HLO
# root, S2d knob removal, S4 renames, H11 profile string), so they are a characterization record.
# What must never change -- the verify arguments, the probe and admission verdicts, the admission
# requests (verdicts.py) -- is the frozen ``safety`` record in the G1/G2 files.
PROTOCOL_OF = {"G1-protocol": "G1", "G2-protocol": "G2"}
# The characterization records (re-baselined with a marker): ``record`` keeps such a file as it is
# when the compared part of the fresh record equals it, so an unchanged file keeps its provenance and
# re-baseline marker instead of being rewritten with a new timestamp and reason.
CHARACTERIZATION = ("G1-protocol", "G2-protocol", "G6", "G7", "G9", "G9-http")
TIER_OF = {"G1": "fixture", "G2": "production"}
# The whole reason is one token: an H number of a sanctioned host change (declared in DESIGN 6.9
# before use; the pattern accepts any H1..H99 so a newly declared number needs no harness change),
# a stage or sub-stage (DESIGN 10.3: S1, S1a, S2d, S4.2b, ...) or an S5 work unit (DESIGN 11.3:
# WU-E, WU-Docs, ...). Free text, commit hashes and words that merely contain such a token are refused.
REBASELINE_REASON = re.compile(r"H[1-9][0-9]?|S[0-9](?:[a-z]|\.[0-9][a-z]?)*|WU-[A-Z][A-Za-z]*")


def valid_reason(reason: str | None) -> bool:
    return bool(reason) and REBASELINE_REASON.fullmatch(reason) is not None


# Installed packages each gate's data are bound to (a mismatch is a skip, never a silent pass).
BOUND_PACKAGES = {
    "default": ("jax", "jaxlib"),
    "G3": ("jax", "jaxlib", "numpy", "ml_dtypes"),
    "G4": ("jax", "jaxlib", "torch", "safetensors"),  # the tiny pack's source is written with torch/safetensors
}
G3_XLA_FLAGS = "--xla_force_host_platform_device_count=32 --xla_cpu_multi_thread_eigen=false"


# ----------------------------------------------------------------------------- producers
def produce(gate: str, *, cpus: str | None = None) -> Any:
    """Run the gate's child and return its raw record (not yet written). ``cpus`` pins the child
    to a CPU list with ``taskset`` (determinism probe for a smaller host)."""
    prefix = ("taskset", "-c", cpus) if cpus else ()
    if gate == "G1":
        return _programs_with_programset("fixture", ("--consistency",), timeout=3600, prefix=prefix)
    if gate == "G2":
        return _programs_with_programset("production", (), timeout=4 * 3600, prefix=prefix)
    if gate in PROTOCOL_OF:  # the same child as the frozen gate (shared within one invocation)
        tier = TIER_OF[PROTOCOL_OF[gate]]
        return _program_child(
            tier,
            ("--consistency",) if tier == "fixture" else (),
            timeout=3600 if tier == "fixture" else 4 * 3600,
            prefix=prefix,
        )
    if gate == "G3":
        return run_child(
            "tools.equivalence.golden_run", timeout=3600, extra_env={"XLA_FLAGS": G3_XLA_FLAGS}, prefix=prefix
        )
    if gate == "G4":
        return (
            run_child("tools.equivalence.identities", "--live", timeout=1800)
            if live_manifest_readable()
            else run_child("tools.equivalence.identities", timeout=1800)
        )
    if gate == "G6":
        return run_child("tools.equivalence.import_closure", timeout=3600)
    if gate == "G6-static":
        return run_child("tools.equivalence.import_closure", "--light", timeout=1800, devices=1)
    if gate == "G7":
        return run_child("tools.equivalence.trace_closure", timeout=3600, extra_env={"XLA_FLAGS": G3_XLA_FLAGS})
    if gate == "G9":
        return run_child("tools.equivalence.wire", timeout=1800, devices=1)
    if gate == "fixture":
        return run_child("tools.equivalence.fixture", timeout=1800)
    raise ValueError(f"unknown gate {gate}")


_CHILDREN: dict[tuple[Any, ...], Any] = {}


def _program_child(tier: str, extra: tuple[str, ...], *, timeout: float, prefix: tuple[str, ...]) -> Any:
    """The real runtime's program child for a tier, run once per invocation: the frozen gate
    (G1/G2) and its protocol record (G1-protocol/G2-protocol) share it."""
    key = (tier, extra, prefix)
    if key not in _CHILDREN:
        _CHILDREN[key] = run_child("tools.equivalence.programs", "--tier", tier, *extra, timeout=timeout, prefix=prefix)
    return _CHILDREN[key]


def _programs_with_programset(tier: str, extra: tuple[str, ...], *, timeout: float, prefix: tuple[str, ...]) -> Any:
    """The tier's fingerprints from the real runtime (the gate), and -- in a parallel child -- those
    of ``build_program_set`` called standalone with the runtime's arguments, compared as a
    cross-check (``programset_cross_check``; S2c, it replaced the v0 replica of ``_load``)."""
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=2) as pool:
        real = pool.submit(_program_child, tier, extra, timeout=timeout, prefix=prefix)
        standalone = (
            pool.submit(
                run_child,
                "tools.equivalence.programs",
                "--tier",
                tier,
                "--adapter",
                "programset",
                timeout=timeout,
                prefix=prefix,
            )
            if program_set_exists()
            else None
        )
        record = dict(real.result())
        if standalone is None:
            record["programset_cross_check"] = dict(status="absent", reason="no glm_tpu/runner/programs.py (pre-S2c)")
            return record
        try:
            record["programset_cross_check"] = programset_cross_check(record, standalone.result())
        except Exception as exc:  # a crashed cross-check fails the gate (_program_record_problems)
            lines = [line for line in str(exc).splitlines() if line.strip()]
            record["programset_cross_check"] = dict(status="unavailable", reason=" | ".join(lines[-3:])[-600:])
    return record


def program_set_exists() -> bool:
    from .common import REPO

    return (REPO / "glm_tpu/runner/programs.py").is_file()


def programset_cross_check(real: dict[str, Any], standalone: dict[str, Any]) -> dict[str, Any]:
    """Every program the runtime compiled (the per-table FP8 decoders excepted: they are compiled
    by jit dispatch inside the resident-weight preparation) is a spec of the standalone
    ``ProgramSet`` that fingerprints identically; each spec's declared donation is what its
    lowering donates; and in every run the runtime compiled exactly its own set's functions."""
    old, new = real["programs"], standalone["programs"]
    expected = {k for k in old if not k.startswith("fp8_table[")}

    def pair(record: dict[str, Any]) -> tuple[Any, ...]:
        return record["digest"], record["signature_digest"], record.get("jit_compiler_options")

    mismatches = sorted(expected ^ set(new))
    mismatches += sorted(k for k in expected & set(new) if pair(old[k]) != pair(new[k]))
    mismatches += sorted(
        f"{k} (donation)" for k in new if new[k].get("declared_donate_argnums") != new[k].get("donated_argnums")
    )
    sets = real.get("program_sets") or {}
    not_from_set = sorted(run for run, info in sets.items() if not info.get("identical"))
    return dict(
        status="compared",
        identical=bool(sets) and not mismatches and not not_from_set,
        programs=len(new),
        runs=len(sets),
        mismatches=mismatches[:20],
        runs_not_compiled_from_set=not_from_set,
    )


def live_manifest_readable() -> bool:
    from .identities import live_manifest_path

    path = live_manifest_path()
    return path is not None and os.access(path, os.R_OK)


def _strip(record: dict[str, Any], *keys: str) -> dict[str, Any]:
    return {k: v for k, v in record.items() if k not in keys}


def comparable(gate: str, record: dict[str, Any]) -> Any:
    """The part of a record the pass criterion compares (timings and provenance excluded)."""
    if gate in ("G1", "G2"):
        # Device programs (frozen): production's own lowering (digest, N8 signature), the compiler
        # options bound to its Lowered by jax.jit and the arguments it passed to Lowered.compile,
        # per run.
        # The frozen safety facts (verify arguments, probe and admission verdicts, admission
        # requests; verdicts.py) come from the same child and never change either.
        return dict(
            programs={
                k: dict(
                    digest=v["digest"],
                    signature_digest=v["signature_digest"],
                    jit_compiler_options=v.get("jit_compiler_options"),
                    compile=v.get("compile"),
                )
                for k, v in record["programs"].items()
            },
            safety=record.get("safety"),
        )
    if gate in PROTOCOL_OF:
        # Characterization: the load protocol the real __init__/_load/compile followed (phases,
        # admissions, loader arguments, record keys, HLO directory, consensus probe), the
        # production option and builder defaults and the full admission reports.
        return dict(runtime=record.get("runtime"), defaults=record.get("defaults"), verdicts=record.get("verdicts"))
    if gate == "G3":
        return dict(groups=record["groups"], components=record["components"])
    if gate == "G4":
        return record["record"]
    if gate in ("G6", "G6-static"):
        return dict(
            stages={
                k: dict(modules=v["modules"], jax_imported=v["jax_imported"], third_party=v.get("third_party", []))
                for k, v in record["stages"].items()
            },
            static_layering=record["static_layering"],
        )
    if gate == "G7":
        return record["functions"]
    if gate == "G9":
        return record["wire"]
    if gate == "G9-http":
        return record["http"] if "http" in record else record["cases"]
    if gate == "fixture":
        return record["fixture"]
    raise ValueError(gate)


# ----------------------------------------------------------------------------- data files
def _envelope(gate: str, payload: dict[str, Any], environment: dict[str, Any] | None, **extra: Any) -> dict[str, Any]:
    return dict(
        gate=gate,
        format=1,
        environment=environment,
        source=source_record(),
        recorded_unix=int(time.time()),
        **extra,
        **payload,
    )


def record(
    gates: list[str], *, twice: bool = True, reason: str | None = None, rename_only: bool = False
) -> list[dict[str, Any]]:
    """Record baseline data. G1 and G3 are produced twice in separate processes (G3's second run
    pinned to 4 CPUs when ``taskset`` exists) and must be identical before anything is written.

    On a tree whose production paths differ from 181c013e, graph and identity data are never
    recorded, and closure/trace/wire data only with ``reason`` (one H number, stage or work unit),
    written into the file as a ``rebaseline`` marker. ``rename_only`` (G6/G7) refuses unless the
    fresh record passes the check through ``closure_map.toml`` -- i.e. the difference is exactly
    the reviewed renames, additions and removals -- and then records it under the current names.
    A characterization file whose compared part is unchanged is not rewritten (report ``unchanged``)."""
    source = source_record()
    changed_tree = source.get("production_paths_equal_baseline") is not True
    unknown = [g for g in gates if g not in DATA_FILES or g in ("G9-http", "G6-static")]
    if unknown:
        raise SystemExit(f"cannot record {', '.join(unknown)} (G6-static is checked against the G6 record)")
    if rename_only and not set(gates) <= {"G6", "G7"}:
        raise SystemExit("--rename-only applies to G6 and G7 only")
    if changed_tree or rename_only:
        frozen = [g for g in gates if g in FROZEN_DATA]
        if frozen and changed_tree:
            raise SystemExit(
                f"{', '.join(frozen)}: graph and identity goldens are recorded only from production "
                "paths equal to 181c013e (extract that tree and point GLM_EQUIVALENCE_SOURCE_ROOT at it)"
            )
        if not valid_reason(reason):
            raise SystemExit(
                "this re-baseline needs --reason set to exactly one H number declared in DESIGN 6.9 "
                "(e.g. H11), stage token (e.g. S2d, S4.2b) or work unit (e.g. WU-E)"
            )
    marker = dict(kind="rename-only" if rename_only else "reviewed", reason=reason) if reason else None
    if marker is not None and set(gates) & set(FROZEN_DATA):
        raise SystemExit("graph and identity goldens (G1-G4, fixture) take no re-baseline marker")
    reports = []
    for gate in gates:
        started = time.perf_counter()
        first = produce(gate)
        if rename_only:
            differing, info = _closure_check(gate, read_json(DATA / DATA_FILES[gate]), first)
            if differing:
                raise SystemExit(f"{gate}: not a rename-only change: {differing[:20]}; nothing written")
            marker = dict(
                marker,
                closure_map=info["closure_map"],
                previous_digest=digest_json(comparable(gate, read_json(DATA / DATA_FILES[gate]))),
            )
        determinism = None
        if twice and gate in ("G1", "G3"):
            second = produce(gate, cpus="0-3" if shutil.which("taskset") else None)
            identical = comparable(gate, first) == comparable(gate, second)
            determinism = dict(
                runs=2,
                identical=identical,
                second_run_cpus="0-3" if shutil.which("taskset") else "all",
                first_digest=digest_json(comparable(gate, first)),
            )
            if not identical:
                raise SystemExit(f"{gate}: the two baseline recordings differ; nothing written")
            if gate == "G1":
                determinism["tier_digest"] = [first["tier_digest"], second["tier_digest"]]
        seconds = round(time.perf_counter() - started, 1)
        diffs = {
            name: _record_diff(name, first if name != "G9-http" else dict(cases=first["http"]))
            for name in (
                gate,
                *[p for p, frozen in PROTOCOL_OF.items() if frozen == gate],
                *(["G9-http"] if gate == "G9" else []),
            )
        }
        written: list[str] = []
        unchanged: list[str] = []

        def write(name: str, envelope: dict[str, Any]) -> None:
            if name in CHARACTERIZATION and diffs.get(name) is not None and diffs[name]["count"] == 0:  # noqa: B023 (called in this iteration)
                unchanged.append(DATA_FILES[name])  # noqa: B023 (called in this iteration)
                return
            write_json(DATA / DATA_FILES[name], envelope)
            written.append(DATA_FILES[name])  # noqa: B023 (called in this iteration)

        if gate in ("G1", "G2"):
            problems = _program_record_problems(gate, first)
            if problems:
                raise SystemExit(f"{gate}: {', '.join(problems)}; nothing written")
            payload = dict(
                programs=first["programs"],
                safety=first["safety"],
                tier_digest=first["tier_digest"],
                programset_cross_check=first["programset_cross_check"],
                adapter=first["adapter"],
            )
            if gate == "G1":
                payload.update(adapter_consistency=first["adapter_consistency"], determinism=determinism)
            write(gate, _envelope(gate, payload, first["environment"], tier=TIER_OF[gate]))
            # The frozen baseline comes with its characterization record (same child, same tree).
            protocol = next(name for name, frozen in PROTOCOL_OF.items() if frozen == gate)
            write(
                protocol,
                _envelope(
                    protocol,
                    dict(runtime=first["runtime"], defaults=first["defaults"], verdicts=first["verdicts"]),
                    first["environment"],
                    tier=TIER_OF[gate],
                ),
            )
        elif gate in PROTOCOL_OF:
            payload = dict(
                runtime=first["runtime"],
                defaults=first["defaults"],
                verdicts=first["verdicts"],
                **({"rebaseline": marker} if marker else {}),
            )
            write(gate, _envelope(gate, payload, first["environment"], tier=TIER_OF[PROTOCOL_OF[gate]]))
        elif gate == "G3":
            payload = dict(
                groups=first["groups"], components=first["components"], digest=first["digest"], determinism=determinism
            )
            write(gate, _envelope(gate, payload, first["environment"]))
        elif gate == "G4":
            if "live" not in first:
                raise SystemExit("G4 record needs the live manifest (read-only) to copy the 32 header SHA-256s")
            payload = dict(record=first["record"], live=first["live"], live_equal=first["live_equal"])
            write(gate, _envelope(gate, payload, static_environment()))
        elif gate == "G6":
            payload = dict(_strip(first, "source"), **({"rebaseline": marker} if marker else {}))
            write(gate, _envelope(gate, payload, static_environment("--xla_force_host_platform_device_count=32")))
        elif gate == "G7":
            payload = dict(first, **({"rebaseline": marker} if marker else {}))
            write(gate, _envelope(gate, payload, static_environment(G3_XLA_FLAGS)))
        elif gate == "G9":
            environment = static_environment("--xla_force_host_platform_device_count=1")
            extra = {"rebaseline": marker} if marker else {}
            write("G9", _envelope("G9", dict(wire=first["wire"], **extra), environment))
            write("G9-http", _envelope("G9-http", dict(cases=first["http"], **extra), environment))
        elif gate == "fixture":
            write(gate, _envelope(gate, dict(fixture=first["fixture"]), first["environment"]))
        reports.append(
            dict(
                gate=gate,
                status="recorded" if written else "unchanged",
                seconds=seconds,
                determinism=determinism,
                diff=diffs,
                written=written,
                unchanged=unchanged,
            )
        )
    return reports


def _brief(value: Any, width: int = 160) -> str:
    text = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return text if len(text) <= width else text[: width - 3] + "..."


def value_diff(old: Any, new: Any, prefix: str = "") -> list[str]:
    """``path: old -> new`` for every leaf where two JSON values differ (values abbreviated): the
    re-baseline diff ``record`` prints for the dedicated commit message."""
    if isinstance(old, dict) and isinstance(new, dict):
        out: list[str] = []
        for key in sorted(set(old) | set(new)):
            if key not in old:
                out.append(f"{prefix}{key}: <absent> -> {_brief(new[key])}")
            elif key not in new:
                out.append(f"{prefix}{key}: {_brief(old[key])} -> <absent>")
            elif old[key] != new[key]:
                out.extend(value_diff(old[key], new[key], f"{prefix}{key}."))
        return out
    where = prefix.rstrip(".") or "<root>"
    if (
        isinstance(old, list)
        and isinstance(new, list)
        and all(isinstance(x, str) for x in old + new)
        and (len(old) != len(new) or sorted(old) != sorted(new))
    ):
        # name lists (closures, executed functions, phases): what appeared and what disappeared
        added, removed = sorted(set(new) - set(old)), sorted(set(old) - set(new))
        lines = [f"{where}: + {x}" for x in added] + [f"{where}: - {x}" for x in removed]
        return lines or [f"{where}: {_brief(old)} -> {_brief(new)}"]
    if isinstance(old, list) and isinstance(new, list) and len(old) == len(new):
        out = []
        for index, (a, b) in enumerate(zip(old, new, strict=True)):
            if a != b:
                out.extend(value_diff(a, b, f"{prefix}{index}."))
        return out
    return [f"{where}: {_brief(old)} -> {_brief(new)}"]


def _record_diff(gate: str, fresh: dict[str, Any]) -> dict[str, Any] | None:
    """The diff of a gate's comparable part against the data file it is about to replace."""
    path = DATA / DATA_FILES[gate]
    if not path.is_file():
        return None
    try:
        old = comparable(gate, read_json(path))
    except (KeyError, TypeError, ValueError):
        return dict(lines=["<previous data file has another format>"], count=1)
    lines = value_diff(old, comparable(gate, fresh))
    return dict(count=len(lines), lines=lines[:300])


def _diff_keys(old: Any, new: Any, prefix: str = "") -> list[str]:
    """Paths where two JSON values differ (bounded, for the report)."""
    if isinstance(old, dict) and isinstance(new, dict):
        out: list[str] = []
        for key in sorted(set(old) | set(new)):
            if key not in old or key not in new:
                out.append(f"{prefix}{key}")
            elif old[key] != new[key]:
                out.extend(_diff_keys(old[key], new[key], f"{prefix}{key}."))
        return out
    if isinstance(old, list) and isinstance(new, list) and len(old) == len(new):
        out = []
        for index, (a, b) in enumerate(zip(old, new, strict=True)):
            if a != b:
                out.extend(_diff_keys(a, b, f"{prefix}{index}."))
        return out
    return [prefix.rstrip(".") or "<root>"]


def check(gate: str) -> dict[str, Any]:
    """Compare the tree with one gate's baseline. Never raises for a gate-level problem: a missing
    or unreadable data file is ``fail``, a crashed or timed-out child is ``error`` (so a
    multi-gate check still reports every gate), and ``skip`` is returned only for a recorded
    package-version mismatch -- which the CLI treats as a failure unless ``--allow-skip``."""
    started = time.perf_counter()
    try:
        return _check(gate, started)
    except Exception as exc:  # a crashed child or comparison must not hide the remaining gates
        lines = [line for line in str(exc).splitlines() if line.strip()]
        return dict(
            gate=gate,
            status="error",
            reason=f"{type(exc).__name__}: " + " | ".join(lines[-12:])[-3000:],
            seconds=round(time.perf_counter() - started, 1),
        )


def _check(gate: str, started: float) -> dict[str, Any]:
    from .common import version_mismatch

    files = [DATA_FILES[gate]] + ([DATA_FILES["G9-http"]] if gate == "G9" else [])
    missing = [name for name in files if not (DATA / name).is_file()]
    if missing:
        return dict(gate=gate, status="fail", reason="no baseline " + ", ".join(missing), differing=["<baseline>"])
    baseline = read_json(DATA / DATA_FILES[gate])
    if not baseline.get("environment"):
        return dict(
            gate=gate, status="fail", reason=f"{DATA_FILES[gate]} records no environment", differing=["<environment>"]
        )
    reason = version_mismatch(baseline["environment"], BOUND_PACKAGES.get(gate, BOUND_PACKAGES["default"]))
    if reason:
        return dict(gate=gate, status="skip", reason=reason)
    fresh = produce(gate)
    if gate == "G9":
        old_wire, new_wire = comparable("G9", baseline), comparable("G9", fresh)
        http_old = comparable("G9-http", read_json(DATA / DATA_FILES["G9-http"]))
        http_new = comparable("G9-http", fresh)
        differing = _diff_keys(old_wire, new_wire)[:40] + ["http." + k for k in _diff_keys(http_old, http_new)[:40]]
    elif gate == "G4":
        differing = _diff_keys(comparable(gate, baseline), comparable(gate, fresh))[:40]
        live = baseline["live"]
        new = fresh["record"]
        if (
            new["headers"] != live["headers"]
            or new["placement"]["sha256"] != live["placement_sha256"]
            or new["geometry"]["sha256"] != live["geometry_sha256"]
        ):
            differing.append("live_manifest")
    elif gate in ("G6", "G6-static", "G7"):
        differing, info = _closure_check(gate, baseline, fresh)
    elif gate in ("G1", "G2"):
        old, new = comparable(gate, baseline), comparable(gate, fresh)
        differing = sorted(
            k for k in set(old["programs"]) | set(new["programs"]) if old["programs"].get(k) != new["programs"].get(k)
        )
        differing += ["safety." + k for k in _diff_keys(old["safety"], new["safety"])][:40]
        differing += _program_record_problems(gate, fresh)
    else:
        differing = _diff_keys(comparable(gate, baseline), comparable(gate, fresh))[:40]
    report: dict[str, Any] = dict(
        gate=gate,
        status="pass" if not differing else "fail",
        differing=differing,
        seconds=round(time.perf_counter() - started, 1),
    )
    if differing and gate not in ("G1", "G2", "G6", "G6-static", "G7"):
        report["diff"] = value_diff(comparable(gate, baseline), comparable(gate, fresh))[:60]
    if differing and gate in ("G1", "G2") and any(k.startswith("safety.") for k in differing):
        report["safety_diff"] = value_diff(baseline.get("safety"), fresh.get("safety"))[:60]
    if gate in ("G6", "G6-static", "G7"):
        report["closure"] = info
    if gate in ("G1", "G2"):
        report["programset_cross_check"] = fresh.get("programset_cross_check")
        if differing:
            report["summary_diff"] = {
                k: _summary_delta(_resolved(baseline["programs"], k), _resolved(fresh["programs"], k))
                for k in differing[:10]
                if k in baseline["programs"] or k in fresh["programs"]
            }
    return report


def _closure_check(gate: str, baseline: dict[str, Any], fresh: dict[str, Any]) -> tuple[list[str], dict[str, Any]]:
    """G6/G7 through the reviewed rename table (``closure_map.toml``). G6: every stage closure,
    its third-party set and the static layering scan may only shrink; JAX may not appear in a
    stage that did not import it. G7: the executed set equals the recorded one up to reviewed
    ``[added]``/``[removed]`` entries. Current names are mapped to recorded names first."""
    from . import closure_map

    cmap = closure_map.load()
    info: dict[str, Any] = dict(closure_map=cmap.digest())
    differing: list[str] = []
    if gate in ("G6", "G6-static"):
        old, new = comparable(gate, baseline), comparable(gate, fresh)
        if gate == "G6-static":  # the light subset: the stages it ran, against the same record
            old = dict(old, stages={k: v for k, v in old["stages"].items() if k in new["stages"]})
        removed: dict[str, list[str]] = {}
        renamed = 0
        for stage in sorted(set(old["stages"]) | set(new["stages"])):
            if stage not in old["stages"] or stage not in new["stages"]:
                differing.append(f"stages.{stage}")
                continue
            a, b = old["stages"][stage], new["stages"][stage]
            result = closure_map.compare_modules(a["modules"], b["modules"], cmap)
            differing += [f"stages.{stage}.modules+{m}" for m in result["added"]]
            if b["jax_imported"] and not a["jax_imported"]:
                differing.append(f"stages.{stage}.jax_imported")
            grown = sorted(set(b["third_party"]) - set(a["third_party"]))
            differing += [f"stages.{stage}.third_party+{t}" for t in grown]
            renamed += len(result["renamed"])
            if result["removed"]:
                removed[stage] = result["removed"][:20]
        from .import_closure import lean_violations

        differing += lean_violations(new["stages"])
        recorded = {closure_map.layering_entry(e, closure_map.ClosureMap()) for e in old["static_layering"]}
        differing += [
            "static_layering+" + e
            for e in sorted({closure_map.layering_entry(e, cmap) for e in new["static_layering"]} - recorded)
        ]
        info.update(removed=removed, renamed=renamed)
    else:
        result = closure_map.compare_functions(baseline["functions"], fresh["functions"], cmap)
        differing = ["+" + f for f in result["added"][:40]] + ["-" + f for f in result["removed"][:40]]
        info.update(renamed=len(result["renamed"]), declared_removed=result["declared_removed"][:20])
    return differing[:80], info


def _program_record_problems(gate: str, record: dict[str, Any]) -> list[str]:
    """G1/G2 self-consistency that fails the gate: the fixture's concrete and abstract runs must
    agree (the licence for the abstract production tier), and -- once ``glm_tpu/runner/programs.py``
    exists (S2c) -- the ProgramSet cross-check must be identical (a crashed or missing cross-check
    fails too)."""
    problems = []
    if gate == "G1" and not (record.get("adapter_consistency") or {}).get("identical"):
        problems.append("adapter_consistency")
    cross = record.get("programset_cross_check") or {}
    if program_set_exists() and not (cross.get("status") == "compared" and cross.get("identical")):
        problems.append("programset_cross_check")
    return problems


def _resolved(programs: dict[str, Any], key: str) -> dict[str, Any] | None:
    """A program record with its diagnostic summary (records equal to an earlier one of the tier
    carry ``same_as`` instead of a summary)."""
    record = programs.get(key)
    if record is not None and "summary" not in record and record.get("same_as") in programs:
        record = dict(record, summary=programs[record["same_as"]].get("summary", {}))
    return record


def _summary_delta(old: dict[str, Any] | None, new: dict[str, Any] | None) -> Any:
    if old is None or new is None:
        return "program added" if old is None else "program removed"
    a, b = old.get("summary", {}), new.get("summary", {})
    old_ops, new_ops = a.get("ops", {}), b.get("ops", {})
    ops = {
        k: [old_ops.get(k, 0), new_ops.get(k, 0)]
        for k in set(old_ops) | set(new_ops)
        if old_ops.get(k) != new_ops.get(k)
    }
    return dict(
        signature_changed=old["signature_digest"] != new["signature_digest"],
        jit_compiler_options=[old.get("jit_compiler_options"), new.get("jit_compiler_options")],
        compile=[old.get("compile"), new.get("compile")],
        ops=ops,
        kernels_changed=a.get("kernels") != b.get("kernels"),
        collectives_changed=a.get("collectives") != b.get("collectives"),
        bytes=[a.get("bytes"), b.get("bytes")],
    )


def diff(program: str, *, against: str, tier: str) -> dict[str, Any]:
    """Unified diff of one program's normalized text between ``against`` (a git revision whose
    production paths are extracted to a temporary tree) and the working tree."""
    import difflib
    import tarfile
    import tempfile

    from .common import HARNESS_REPO, PRODUCTION_PATHS, REPO

    with tempfile.TemporaryDirectory(prefix="glm-equivalence-diff-") as scratch:
        root = Path(scratch) / "baseline"
        root.mkdir()
        present = subprocess.run(
            ["git", "ls-tree", "--name-only", against, "--", *PRODUCTION_PATHS],
            cwd=HARNESS_REPO,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.split()
        archive = subprocess.run(
            ["git", "archive", "--format=tar", against, *present], cwd=HARNESS_REPO, check=True, capture_output=True
        ).stdout
        with tarfile.open(fileobj=__import__("io").BytesIO(archive)) as tar:
            tar.extractall(root, filter="data")
        texts = []
        for source in (root, REPO):
            texts.append(
                run_child(
                    "tools.equivalence.programs",
                    "--tier",
                    tier,
                    "--only",
                    program,
                    "--text",
                    source_root=source,
                    timeout=4 * 3600,
                )
            )
    old, new = texts[0]["text"], texts[1]["text"]
    lines = list(
        difflib.unified_diff(
            old.splitlines(), new.splitlines(), f"{against}:{program}", f"worktree:{program}", n=2, lineterm=""
        )
    )
    return dict(program=program, against=against, identical=old == new, diff_lines=len(lines), diff=lines[:400])


def refuse_if_live(gates: list[str]) -> list[str]:
    from .budget import light_mode, live_tpu_run, refusal_reason

    if not live_tpu_run():
        return gates
    heavy = [g for g in gates if g in HEAVY]
    if heavy:
        raise SystemExit(f"{refusal_reason()} (heavy gates requested: {', '.join(heavy)})")
    light_mode()
    return gates


def dumps(value: Any) -> str:
    return json.dumps(value, sort_keys=True, indent=1)

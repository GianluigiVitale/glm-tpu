"""``python -m tools.equivalence <command>`` (JAX_PLATFORMS=cpu; never touches a TPU).

Commands:
  record       --gates G1,G1-protocol,G2,G2-protocol,G3,G4,G6,G7,G9,fixture [--reason R] [--rename-only]
               write tests/golden/data (integrator only; see README "Re-baselining")
  check        --gates G1,G1-protocol,G3,G4,G6,G7,G9 | --tier fixture|production [--allow-skip]
               compare the tree with the baseline; a skipped (version mismatch), failed or crashed
               gate exits 1. While a TPU run is live only light gates run (G4, G6-static, G9).
  selftest     G14 mutation self-test of the normalizer
  site-check   [--record] [--requests DIR]            G5 on rank 0 (read-only, fleet idle)
  compare-run  RUN --golden DIR [--golden DIR] [--out FILE]   TPU token equivalence (hash-only)
  diff         --program KEY [--against REV] [--tier fixture|production]   normalized text diff
  authenticity DIR [DIR ...] [--out FILE]            harness programs vs TPU StableHLO originals
  budget                                               is a TPU run live on this host?

Every report is JSON on stdout; exit status 0 only when every requested gate passes.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

GATES = ("G1", "G1-protocol", "G2", "G2-protocol", "G3", "G4", "G6", "G6-static", "G7", "G9", "fixture")
DEFAULT_CHECK = ("G1", "G1-protocol", "G3", "G4", "G6", "G7", "G9")
TIERS = {"fixture": ["G1", "G1-protocol"], "production": ["G2", "G2-protocol"]}


def _gates(text: str) -> list[str]:
    gates = [g.strip() for g in text.split(",") if g.strip()]
    unknown = [g for g in gates if g not in GATES]
    if unknown:
        raise SystemExit(f"unknown gates: {', '.join(unknown)} (known: {', '.join(GATES)})")
    return gates


def main(argv: list[str] | None = None) -> int:
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    if os.environ["JAX_PLATFORMS"] != "cpu":
        raise SystemExit("tools.equivalence requires JAX_PLATFORMS=cpu")
    parser = argparse.ArgumentParser(
        prog="python -m tools.equivalence", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    commands = parser.add_subparsers(dest="command", required=True)
    rec = commands.add_parser("record")
    rec.add_argument("--gates", required=True, type=_gates)
    rec.add_argument("--once", action="store_true", help="skip the second determinism recording (not for baselines)")
    rec.add_argument(
        "--reason",
        help="re-baseline marker: one H number, stage or work unit (e.g. H11, S4.2b, WU-E); required on a changed tree",
    )
    rec.add_argument(
        "--rename-only",
        action="store_true",
        help="G6/G7: re-record under current names only if closure_map.toml explains every difference",
    )
    chk = commands.add_parser("check")
    chk.add_argument("--gates", type=_gates, default=list(DEFAULT_CHECK))
    chk.add_argument("--tier", choices=("fixture", "production"))
    chk.add_argument(
        "--allow-skip", action="store_true", help="accept a gate skipped for a package-version mismatch (never in CI)"
    )
    commands.add_parser("selftest")
    site = commands.add_parser("site-check")
    site.add_argument("--record", action="store_true")
    site.add_argument("--requests", type=Path, help="directory of golden request inputs (private, read-only)")
    cmp_ = commands.add_parser("compare-run")
    cmp_.add_argument("run", type=Path)
    cmp_.add_argument("--golden", type=Path, action="append", required=True)
    cmp_.add_argument("--out", type=Path)
    cmp_.add_argument("--speed-tolerance", type=float, default=0.03)
    dif = commands.add_parser("diff")
    dif.add_argument("--program", required=True)
    dif.add_argument("--against", default="181c013e84ac7a2d1c2069feaa9e8aa90da51af5")
    dif.add_argument("--tier", choices=("fixture", "production"), default="fixture")
    auth = commands.add_parser("authenticity")
    auth.add_argument("directories", nargs="+", type=Path)
    auth.add_argument("--out", type=Path, help="per-kernel report (outside Git)")
    commands.add_parser("budget")
    args = parser.parse_args(argv)

    from . import gates as g

    if args.command == "record":
        requested = g.refuse_if_live(args.gates)
        print(g.dumps(g.record(requested, twice=not args.once, reason=args.reason, rename_only=args.rename_only)))
        return 0
    if args.command == "check":
        requested = TIERS[args.tier] if args.tier else args.gates
        requested = g.refuse_if_live(requested)
        started = time.perf_counter()
        reports = [g.check(gate) for gate in requested]
        print(g.dumps(dict(reports=reports, seconds=round(time.perf_counter() - started, 1))))
        accepted = ("pass", "skip") if args.allow_skip else ("pass",)
        return 0 if all(r["status"] in accepted for r in reports) else 1
    if args.command == "selftest":
        from .common import run_child

        g.refuse_if_live(["G14"])
        result = run_child("tools.equivalence.selftest", timeout=3600, devices=1)
        print(g.dumps(result))
        return 0 if result["status"] == "pass" else 1
    if args.command == "site-check":
        from .identities import site_check

        print(g.dumps(site_check(record=args.record, requests_dir=args.requests)))
        return 0
    if args.command == "compare-run":
        from .compare_run import main as compare_main

        forwarded = [str(args.run), "--speed-tolerance", str(args.speed_tolerance)]
        for golden in args.golden:
            forwarded += ["--golden", str(golden)]
        if args.out:
            forwarded += ["--out", str(args.out)]
        return compare_main(forwarded)
    if args.command == "diff":
        result = g.diff(args.program, against=args.against, tier=args.tier)
        print("\n".join(result.pop("diff")))
        print(json.dumps(result, sort_keys=True))
        return 0 if result["identical"] else 1
    if args.command == "authenticity":
        from .common import run_child

        g.refuse_if_live(["G2"])
        extra = ["--out", str(args.out.resolve())] if args.out else []
        result = run_child("tools.equivalence.authenticity", *map(str, args.directories), *extra, timeout=4 * 3600)
        print(g.dumps({k: v for k, v in result.items() if k != "environment"}))
        return 0 if result["status"] == "pass" else 1
    if args.command == "budget":
        from .budget import report

        print(json.dumps(report(), sort_keys=True))
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main())

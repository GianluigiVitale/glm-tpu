# Greenfield evidence and reusable protection map

## Accepted greenfield evidence

Physical topology/local groups:

- local: `/home/gianl/glm-run/greenfield_topology_20260805T125842425591441Z`
- remote: `gs://driftbench-dsv4-uc/results/greenfield_topology_20260805T125842425591441Z`
- `bench/results.db` run 405, one backing item, DB integrity `ok`
- code `75c8bb14cbd290930ff024937310f3aec6175090`
- contract `07ccfc470a079e66177497036c5fac2a928067c96d392d0e117d45f2e5986f99`
- topology `294e777210485f08a3b323121134296e576914eb52b42792019ceef7467dd559`
- PP8 `d5943ab8d7a074677d82f8e823c8bc983847f8df1deefdee1fbda8da98923c14`
- PP16 `6383e57c81478ac0d6de4525a4675f2a0d7cbc7aa73bd67bc662dc4e05840f21`
- 8 distinct host records, identical fleet contract hashes, remote `SUCCESS`, 8/8 pre/post census.

DB 404 is a superseded diagnostic: its physical inventory is correct, but the first lexicographic
PP8 stage sequence contained a non-neighbor boundary. It motivated the all-lane Hamiltonian-ring
invariant and must not be used as the current group manifest.

## Reusable tools, not execution dependencies

- XPlane truth: `scripts/analysis/parse_xplane.py` (nested physical HLO categories, groups, shapes,
  per-core/fleet counts) and `extract_steady_decode.py` (profiler-free wall/device agreement).
- Append-only results: `bench/provenance.py`; greenfield runs additionally put full code/plan hashes
  in `env_json` until a schema migration adds dedicated columns.
- Ownership/cleanup: strict census and exact-pin checks in `resume_health_proof.sh`,
  `dcp_live_rows_exact.sh`, and `e0_capture_arm.sh`; topology capture has a greenfield isolated form.
- Oracle-only integrity: `load_state_hash.py`, `write_probe.py`, `dsa_topk_dump.py`, and
  `dsa_topk_diff.py` in the pinned legacy checkout. Reimplement/adapt validation interfaces without
  importing legacy model execution.

## Legacy measurement oracle

Accepted corrected parent: 287.666063 ms/device token, 3.476 device tok/s, about 3.3 wall tok/s.
The dominant defect is 75 sequential physical 32-chip MoE combines costing 106.495 ms/token.
The completed compute-row smoke is legacy evidence only; it does not authorize extending TP32.

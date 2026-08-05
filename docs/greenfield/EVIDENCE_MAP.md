# Greenfield evidence and reusable protection map

## Accepted protected greenfield evidence

Every accepted run has a local directory under `/home/gianl/glm-run`, a same-tag archive under
`gs://driftbench-dsv4-uc/results/`, append-only `bench/results.db` linkage, local/remote `SUCCESS`,
fleet agreement, and eight-host clean pre/post census.

| DB | tag | evidence |
|---:|---|---|
| 405 | `greenfield_topology_20260805T125842425591441Z` | physical `2x4x4` topology and PP8/PP16 rings |
| 406 | `greenfield_collectives_20260805T133905344573798Z` | `bf16[2,6144]` control/all-reduce, g2/4/8/32 |
| 407 | `greenfield_collectives_20260805T134254891049866Z` | dominant-payload all-gather matrix |
| 408 | `greenfield_collectives_20260805T134731771265066Z` | asynchronous collective-permute HLO validation |
| 409 | `greenfield_collectives_20260805T135545030179024Z` | fused tuple all-reduce HLO validation |
| 410 | `greenfield_collectives_20260805T135644529157649Z` | dominant-payload ppermute/all-to-all/fused tuple matrix |
| 411 | `greenfield_collectives_20260805T135850389312854Z` | supported `bf16[1,6144]` six-operation matrix |
| 412 | `greenfield_collectives_20260805T140125151247631Z` | supported `bf16[1,2048]` six-operation matrix |
| 413 | `greenfield_collectives_20260805T141114991474088Z` | supported `f32[1,6144]` six-operation matrix |
| 414 | `greenfield_collectives_20260805T141333601664455Z` | `int32[1,2048]` routing-metadata five-operation matrix |

Topology code is `75c8bb14...`. Collective matrix pins are `fcd8426735...` and
`b12af9633c8b14648db8d2a2ccd9a3c577a04817`. Topology hash is `294e777...559`, PP8 group hash
`d5943ab8...c14`, and PP16 group hash `6383e57c...f21`.

Superseded or failed diagnostics are preserved but not promotion evidence. DB 404 used a first
non-neighbor PP8 ordering. Reduce-scatter diagnostics `...T134618415607642Z`,
`...T135045281384327Z`, and `...T135140118884391Z` prove XLA rewrite-to-all-reduce and contain no
accepted latency. Earlier ppermute/tuple diagnostics led to the async-HLO and dtype fixes.

## Reusable tools, not execution dependencies

- Optimized-HLO contract: `glm_tpu/greenfield/sharding/hlo_contract.py` and
  `scripts/greenfield/inspect_hlo_contract.py`.
- Dependent collective mechanism: `glm_tpu/greenfield/benchmarking/collective_chain.py` and
  `scripts/greenfield/microbench_collectives.py`.
- Protected launcher: `scripts/greenfield/run_collective_chain.sh` (exact pin, lease, census,
  fleet HLO/checksum agreement, DB, archive, cleanup).
- XPlane/wall truth: `scripts/analysis/parse_xplane.py` and `extract_steady_decode.py`.
- Append-only provenance: `bench/provenance.py`; greenfield hashes reside in `env_json` pending a
  dedicated schema field.
- Oracle-only integrity interfaces: `load_state_hash.py`, `write_probe.py`, `dsa_topk_dump.py`, and
  `dsa_topk_diff.py` in the pinned legacy checkout. Adapt interfaces without importing execution.

## Legacy measurement oracle

Accepted legacy parent: `287.666063 ms/device token`, `3.476` device tok/s, about `3.3` wall tok/s.
Its 75 sequential physical 32-chip MoE combine regions cost `106.495 ms/token`. The greenfield
collective floor shows this is not raw 12 KiB ICI latency, but no greenfield token-speed result
exists yet.

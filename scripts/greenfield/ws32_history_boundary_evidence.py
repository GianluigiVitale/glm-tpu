"""Replay retained history boundaries and original first-decode observations.

Only the first differing group and both observers retain array originals.
Other per-group comparisons and cache hashes remain worker observations, not
independent arithmetic replays. This never certifies 8K tokens or a root cause.
"""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Mapping

import ml_dtypes
import numpy as np

from scripts.greenfield import ws32_history_protocol as protocol
from scripts.greenfield import ws32_history_worker as worker
from scripts.greenfield.prefill_window_evidence import same_json
from scripts.greenfield.ws32_dense_frontier_evidence import read_npz


def layout(local_slots: Mapping[int, int], process_index: int) -> dict:
    """Original capture schema, independently derived from authenticated owners."""
    if (len(local_slots) != 4 or len(set(local_slots.values())) != 4
            or type(process_index) is not int or not 0 <= process_index < 8
            or any(type(d) is not int or not 0 <= d < 32 or type(s) is not int
                   or not 0 <= s < 32 for d, s in local_slots.items())):
        raise ValueError("history boundary owner inventory differs")
    features = sorted({s % 4 for s in local_slots.values()})
    return dict(schema_version=1, process_index=process_index, platform="tpu",
        local_slots={str(d): s for d, s in sorted(local_slots.items())},
        global_hidden_size=6144,
        feature_columns=[c for f in features for c in range(f * 1536, (f + 1) * 1536)],
        health_slots=sorted(local_slots.values()),
        owners={str(s): dict(device_id=d, slot=s,
            feature_columns=list(range((s % 4) * 1536, (s % 4 + 1) * 1536)),
            capture_columns=list(range(features.index(s % 4) * 1536,
                                       (features.index(s % 4) + 1) * 1536)))
            for d, s in sorted(local_slots.items(), key=lambda item: item[1])})


def _rows(arrays: Mapping[str, np.ndarray], *, count: int, columns: int,
          prefix: str = "", health: bool = False) -> dict:
    rows = {}
    for layer in protocol.LAYERS:
        rows[layer] = {}
        for field in (*worker.FIELDS, *(("health",) if health else ())):
            value = arrays[f"{prefix}layer{layer}_{field}"]
            if field in worker.FIELDS[:3]:
                shape, dtype = (count, columns), np.dtype(np.uint16)
            elif field == "health":
                shape, dtype = (count, 4), np.dtype(np.bool_)
            else:
                shape = (count, 8)
                dtype = np.dtype(np.int32 if field == "route_ids" else np.float32)
            if value.shape != shape or value.dtype != dtype:
                raise ValueError("history retained boundary shape/dtype differs")
            if field in worker.FIELDS[:3]:
                value = value.view(ml_dtypes.bfloat16)
            if field == "health" and not value.all():
                raise ValueError("history retained observer health failed")
            if field in worker.FLOAT_FIELDS and not np.isfinite(value).all():
                raise ValueError("history retained boundary is nonfinite")
            rows[layer][field] = value
    return rows


def _group_claims(groups: list) -> dict | None:
    candidates = [s for s in protocol.plan() if s.branch == "candidate"]
    if not isinstance(groups, list) or len(groups) != len(candidates):
        raise ValueError("history group inventory differs")
    first = None
    for group, step in zip(groups, candidates, strict=True):
        if (type(group.get("group")) is not int or group["group"] != step.group
                or type(group.get("offset")) is not int or group["offset"] != step.offset
                or set(group.get("layers", {})) != {str(l) for l in protocol.LAYERS}):
            raise ValueError("history group position/layer identity differs")
        declared_first = None
        for layer in protocol.LAYERS:
            fields = group["layers"][str(layer)]
            if set(fields) != set(worker.FIELDS):
                raise ValueError("history group field inventory differs")
            for field in worker.FIELDS:
                row = fields[field]
                n = row.get("differing_rows")
                if (type(row.get("rows")) is not int or row["rows"] != step.count
                        or type(n) is not int or not 0 <= n <= step.count):
                    raise ValueError("history group row counts differ")
                expected = {"rows", "differing_rows"}
                if n:
                    expected.add("first_position")
                    p = row.get("first_position")
                    if type(p) is not int or not step.offset <= p < step.offset + step.count:
                        raise ValueError("history group first position differs")
                    if field in worker.FLOAT_FIELDS:
                        expected.add("max_abs_difference")
                        delta = row.get("max_abs_difference")
                        if type(delta) not in (int, float) or not np.isfinite(delta) or delta < 0:
                            raise ValueError("history group difference statistic invalid")
                    if declared_first is None or p < declared_first["position"]:
                        declared_first = dict(layer=layer, field=field, position=p)
                if set(row) != expected:
                    raise ValueError("history group statistics inventory differs")
        same_json(group.get("first"), declared_first, "history declared group first difference")
        if group.get("equal") is not (declared_first is None):
            raise ValueError("history group equality claim differs")
        if declared_first is not None and first is None:
            first = dict(declared_first, group=step.group)
    return first


def replay(root: Path, record: Mapping[str, Any], local_slots: Mapping[int, int], *,
           originals: Mapping[str, Mapping[str, np.ndarray]]) -> dict:
    """Read exact NPZ originals, rederive comparisons and both reproduction claims."""
    path = root / "history_report.json"
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 2 << 20:
        raise ValueError("history report path/size differs")
    report = json.loads(path.read_bytes())
    same_json(report, record["history"], "history report original/runner binding")
    if (report.get("protocol") != protocol.PROTOCOL or report.get("complete") is not True
            or report.get("attribution_eligible") is not True
            or report.get("prompt_length") != protocol.PROMPT_LENGTH
            or report.get("steps") != len(protocol.plan())
            or any(report.get(k) is not False for k in
                   ("numerical_promotion", "performance_claim", "cause_claim"))
            or any(k in report for k in ("unhealthy", "refused_step", "refused_observer", "replica_failure"))):
        raise ValueError("history boundary report incomplete or promoted")
    geometry = layout(local_slots, record["jax_process_index"])
    same_json(report.get("boundary_layout"), geometry, "history original boundary owners")
    first = _group_claims(report["groups"])
    same_json(report.get("first_difference"), first, "history first group claim")
    labels = {f"observer_{b}" for b in protocol.BRANCHES}
    if first is not None:
        labels.add(f"first_difference_group{first['group']}")
    retained = report.get("retained", {})
    if set(retained) != labels or {p.name for p in root.glob("*.npz*")} != {f"{l}.npz" for l in labels}:
        raise ValueError("history original capsule inventory differs")
    if (any(type(v.get(k)) is not int or v[k] <= 0 for v in retained.values()
            for k in ("bytes", "raw_array_bytes"))
            or sum(v["bytes"] for v in retained.values()) > protocol.ORIGINALS_LIMIT
            or sum(v["raw_array_bytes"] for v in retained.values()) + len(retained) * (1 << 20) > protocol.ORIGINALS_LIMIT
            or type(report.get("retained_bytes")) is not int
            or report["retained_bytes"] != sum(v["raw_array_bytes"] for v in retained.values())):
        raise ValueError("history capsule aggregate bytes differ")
    observations, boundaries, fingerprints = {}, {}, {}
    columns = len(geometry["feature_columns"])

    def capture(label: str) -> dict:
        return read_npz(root / f"{label}.npz", retained[label], limit=protocol.ORIGINALS_LIMIT)

    def fingerprint(label: str, branch: str, rows: Mapping) -> None:
        for slot, owner in geometry["owners"].items():
            for layer, fields in rows.items():
                for field, value in fields.items():
                    if field in worker.FIELDS[:3]:
                        value = value[:, owner["capture_columns"]]
                    elif field == "health":
                        value = value[:, geometry["health_slots"].index(int(slot))]
                    fingerprints[f"{label}/{branch}/{slot}/{layer}/{field}"] = sha256(value.tobytes()).hexdigest()

    for branch in protocol.BRANCHES:
        label = f"observer_{branch}"
        arrays = capture(label)
        expected = set(worker.OBSERVER_FIELDS) | {f"layer{l}_{f}" for l in protocol.LAYERS
                                                    for f in (*worker.FIELDS, "health")}
        if set(arrays) != expected:
            raise ValueError("history observer array inventory differs")
        obs = {k: arrays[k] for k in worker.OBSERVER_FIELDS}
        for field, value in obs.items():
            shape = (4, 1) if field == "counts" else (4, 1, 2048)
            dtype = np.dtype(np.float32 if field == "scores" else np.int32)
            if value.shape != shape or value.dtype != dtype:
                raise ValueError("history observer DSA geometry differs")
        actual = worker.reproduce(obs, originals[branch])
        same_json(report["reproduction"][branch], actual, "history original observation reproduction")
        if not actual["reproduced"]:
            raise ValueError("history original observation did not reproduce")
        same_json(report["observer"][branch], dict(healthy=True, capture_errors=[]), "history observer health")
        rows = _rows(arrays, count=1, columns=columns, health=True)
        observations[branch], boundaries[branch] = obs, rows
        fingerprint("observer", branch, rows)
    same_json(report["observer_comparison"], worker.compare_rows(boundaries["candidate"], boundaries["control"],
        offset=protocol.WITNESS["position"]), "history observer boundary comparison")
    same_json(report["observations_equal"], worker.reproduce(observations["candidate"], observations["control"])["reproduced"],
              "history branch observation equality")
    if first is not None:
        group = report["groups"][first["group"]]
        step = next(s for s in protocol.plan() if s.branch == "candidate" and s.group == first["group"])
        arrays = capture(f"first_difference_group{step.group}")
        expected = {f"{b}_layer{l}_{f}" for b in protocol.BRANCHES for l in protocol.LAYERS for f in worker.FIELDS}
        if set(arrays) != expected:
            raise ValueError("history first boundary array inventory differs")
        rows = {b: _rows(arrays, count=step.count, columns=columns, prefix=f"{b}_") for b in protocol.BRANCHES}
        same_json(group, dict(worker.compare_rows(rows["candidate"], rows["control"], offset=step.offset),
                              group=step.group), "history retained first group comparison")
        for branch in protocol.BRANCHES:
            fingerprint(f"group{step.group}", branch, rows[branch])
    return dict(reproduced=True, first_difference=first, fingerprints=fingerprints,
                observation_sha256={b: {k: sha256(v.tobytes()).hexdigest() for k, v in obs.items()}
                                    for b, obs in observations.items()},
                retained_groups_replayed=int(first is not None),
                remaining_group_comparisons="WORKER_ASSERTIONS_NOT_RETAINED_ARRAY_REPLAY",
                cache_values_replayed=False, numerical_promotion=False, cause_claim=False)


def validate_replicas(reports: list[Mapping], records: list[Mapping]) -> dict:
    """Join retained owner bytes and cache-digest observations across all32 slots."""
    if len(reports) != 8 or len(records) != 8:
        raise ValueError("history replica replay needs eight ranks")
    rows, caches, features = {}, {}, {}
    observation = reports[0]["observation_sha256"]
    for result, record in zip(reports, records, strict=True):
        same_json(result["observation_sha256"], observation, "history fleet observation replicas")
        for key, digest in result["fingerprints"].items():
            if key in rows:
                raise ValueError("history duplicate boundary owner")
            rows[key] = digest
        history = record["history"]
        geom = history["boundary_layout"]
        feature_key = tuple(geom["feature_columns"])
        if feature_key in features:
            same_json(history["groups"], features[feature_key], "history same-feature group observations")
        features[feature_key] = history["groups"]
        own = set(map(str, geom["health_slots"]))
        for branch in protocol.BRANCHES:
            families = history["cache_digests"][branch]
            if set(families) != set(worker.CACHE_FAMILIES):
                raise ValueError("history cache family inventory differs")
            for family, layers in families.items():
                if set(layers) != {str(i) for i in range(7 if family == "kv" else 4)}:
                    raise ValueError("history cache layer inventory differs")
                for index, hashes in layers.items():
                    if set(hashes) != own:
                        raise ValueError("history cache local owners differ")
                    for slot, digest in hashes.items():
                        if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
                            raise ValueError("history cache digest invalid")
                        key = (branch, family, index, int(slot))
                        if key in caches:
                            raise ValueError("history duplicate cache owner")
                        caches[key] = digest
        expected = {f: {i: history["cache_digests"]["candidate"][f][i]
                             == history["cache_digests"]["control"][f][i]
                        for i in history["cache_digests"]["candidate"][f]} for f in worker.CACHE_FAMILIES}
        same_json(history["cache_equality"], expected, "history cache equality observations")
    stems = {tuple(k.split("/")[:2] + k.split("/")[3:]) for k in rows}
    for label, branch, layer, field in stems:
        groups = [range(f, 32, 4) for f in range(4)] if field in worker.FIELDS[:3] else [range(32)]
        for group in groups:
            values = [rows[f"{label}/{branch}/{s}/{layer}/{field}"] for s in group
                      if f"{label}/{branch}/{s}/{layer}/{field}" in rows]
            # First differing groups may differ by feature: absent capsules are
            # not invented. Observer captures MUST cover every owner.
            if label == "observer" and len(values) != len(group):
                raise ValueError("history missing observer replica")
            if len(set(values)) > 1:
                raise ValueError("history boundary replica bytes disagree")
    for branch in protocol.BRANCHES:
        for family in worker.CACHE_FAMILIES:
            for i in range(7 if family == "kv" else 4):
                for expert in range(8):
                    if len({caches[(branch, family, str(i), s)] for s in range(expert * 4, expert * 4 + 4)}) != 1:
                        raise ValueError("history cache replica digests disagree")
    return dict(retained_boundary_replicas_checked=True, cache_digest_replicas_checked=True,
                cache_values_replayed=False, cause_claim=False)

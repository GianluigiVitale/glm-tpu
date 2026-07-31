#!/usr/bin/env python
"""Reusable XPlane (xplane.pb / XSpace) parser + TPU device-op aggregator.

No tensorflow dependency: the XPlane proto schema (from
tsl/profiler/protobuf/xplane.proto) is built programmatically into a
DescriptorPool, so parsing runs on the fast upb C backend of `protobuf`.

Key facts learned from JAX TPU v4 traces (verified on e0cap_sparse):
  - Device planes are named "/device:TPU:N" (one per TensorCore).
  - The "XLA Ops" line is a NESTED timeline: control-flow container ops
    (conditional, while, call) contain their child ops as events on the same
    line.  Naive duration sums double-count; self-time requires a stack sweep.
  - There is no "Steps" line for pure-JAX decode loops; step boundaries come
    from the "XLA Modules" line (one event per jitted program execution, e.g.
    jit_step_fun_impl).
  - "Async XLA Ops" (usually only on TPU:0) carries overlapping async DMA
    events (slice / async-copy); these overlap compute and are reported
    separately, not added to the critical path.

Usage:
  parse_xplane.py inspect   <xplane.pb>              # dump planes/lines/stat schema
  parse_xplane.py aggregate <xplane.pb> [out.json]   # nested-aware per-op/per-step aggregation
"""
from __future__ import annotations

import collections
import json
import re
import sys

from google.protobuf import descriptor_pb2, descriptor_pool, message_factory

PKG = "tensorflow.profiler"
_T = descriptor_pb2.FieldDescriptorProto


# ---------------------------------------------------------------------------
# Schema construction (replaces the missing xplane_pb2)
# ---------------------------------------------------------------------------
def _field(msg, name, number, ftype, label=_T.LABEL_OPTIONAL, type_name=None,
           oneof_index=None):
    f = msg.field.add()
    f.name, f.number, f.type, f.label = name, number, ftype, label
    if type_name is not None:
        f.type_name = "." + PKG + "." + type_name
    if oneof_index is not None:
        f.oneof_index = oneof_index
    return f


def _map_field(msg, name, number, value_type_name):
    """Adds `map<int64, ValueType> name = number;` to msg."""
    entry = msg.nested_type.add()
    entry.name = "".join(p.capitalize() for p in name.split("_")) + "Entry"
    entry.options.map_entry = True
    _field(entry, "key", 1, _T.TYPE_INT64)
    _field(entry, "value", 2, _T.TYPE_MESSAGE, type_name=value_type_name)
    f = msg.field.add()
    f.name, f.number, f.type, f.label = name, number, _T.TYPE_MESSAGE, _T.LABEL_REPEATED
    f.type_name = "." + PKG + "." + msg.name + "." + entry.name


def build_xplane_classes():
    fdp = descriptor_pb2.FileDescriptorProto()
    fdp.name = "xplane_standalone.proto"
    fdp.package = PKG
    fdp.syntax = "proto3"

    m = fdp.message_type.add(); m.name = "XStat"
    _field(m, "metadata_id", 1, _T.TYPE_INT64)
    m.oneof_decl.add().name = "value"
    _field(m, "double_value", 2, _T.TYPE_DOUBLE, oneof_index=0)
    _field(m, "uint64_value", 3, _T.TYPE_UINT64, oneof_index=0)
    _field(m, "int64_value", 4, _T.TYPE_INT64, oneof_index=0)
    _field(m, "str_value", 5, _T.TYPE_STRING, oneof_index=0)
    _field(m, "bytes_value", 6, _T.TYPE_BYTES, oneof_index=0)
    _field(m, "ref_value", 7, _T.TYPE_UINT64, oneof_index=0)

    m = fdp.message_type.add(); m.name = "XEvent"
    _field(m, "metadata_id", 1, _T.TYPE_INT64)
    m.oneof_decl.add().name = "data"
    _field(m, "offset_ps", 2, _T.TYPE_INT64, oneof_index=0)
    _field(m, "num_occurrences", 5, _T.TYPE_INT64, oneof_index=0)
    _field(m, "duration_ps", 3, _T.TYPE_INT64)
    _field(m, "stats", 4, _T.TYPE_MESSAGE, _T.LABEL_REPEATED, "XStat")

    m = fdp.message_type.add(); m.name = "XEventMetadata"
    _field(m, "id", 1, _T.TYPE_INT64)
    _field(m, "name", 2, _T.TYPE_STRING)
    _field(m, "metadata", 3, _T.TYPE_BYTES)
    _field(m, "display_name", 4, _T.TYPE_STRING)
    _field(m, "stats", 5, _T.TYPE_MESSAGE, _T.LABEL_REPEATED, "XStat")
    _field(m, "child_id", 6, _T.TYPE_INT64, _T.LABEL_REPEATED)

    m = fdp.message_type.add(); m.name = "XStatMetadata"
    _field(m, "id", 1, _T.TYPE_INT64)
    _field(m, "name", 2, _T.TYPE_STRING)
    _field(m, "description", 3, _T.TYPE_STRING)

    m = fdp.message_type.add(); m.name = "XLine"
    _field(m, "id", 1, _T.TYPE_INT64)
    _field(m, "name", 2, _T.TYPE_STRING)
    _field(m, "timestamp_ns", 3, _T.TYPE_INT64)
    _field(m, "events", 4, _T.TYPE_MESSAGE, _T.LABEL_REPEATED, "XEvent")
    _field(m, "duration_ps", 9, _T.TYPE_INT64)
    _field(m, "display_id", 10, _T.TYPE_INT64)
    _field(m, "display_name", 11, _T.TYPE_STRING)

    m = fdp.message_type.add(); m.name = "XPlane"
    _field(m, "id", 1, _T.TYPE_INT64)
    _field(m, "name", 2, _T.TYPE_STRING)
    _field(m, "lines", 3, _T.TYPE_MESSAGE, _T.LABEL_REPEATED, "XLine")
    _map_field(m, "event_metadata", 4, "XEventMetadata")
    _map_field(m, "stat_metadata", 5, "XStatMetadata")
    _field(m, "stats", 6, _T.TYPE_MESSAGE, _T.LABEL_REPEATED, "XStat")

    m = fdp.message_type.add(); m.name = "XSpace"
    _field(m, "planes", 1, _T.TYPE_MESSAGE, _T.LABEL_REPEATED, "XPlane")
    _field(m, "errors", 2, _T.TYPE_STRING, _T.LABEL_REPEATED)
    _field(m, "warnings", 3, _T.TYPE_STRING, _T.LABEL_REPEATED)
    _field(m, "hostnames", 4, _T.TYPE_STRING, _T.LABEL_REPEATED)

    pool = descriptor_pool.DescriptorPool()
    pool.Add(fdp)
    return {
        n: message_factory.GetMessageClass(
            pool.FindMessageTypeByName(PKG + "." + n))
        for n in ("XSpace", "XPlane", "XLine", "XEvent", "XStat",
                  "XEventMetadata", "XStatMetadata")
    }


_CLASSES = None


def load_xspace(path):
    global _CLASSES
    if _CLASSES is None:
        _CLASSES = build_xplane_classes()
    xs = _CLASSES["XSpace"]()
    with open(path, "rb") as f:
        xs.ParseFromString(f.read())
    return xs


def stat_value(stat):
    which = stat.WhichOneof("value")
    if which is None:
        return None
    v = getattr(stat, which)
    if isinstance(v, bytes):
        try:
            v = v.decode("utf-8", "replace")
        except Exception:
            pass
    return v


def event_stats(event, stat_md):
    return {stat_md.get(s.metadata_id, str(s.metadata_id)): stat_value(s)
            for s in event.stats}


# ---------------------------------------------------------------------------
# Categorization (decode-workload oriented)
# ---------------------------------------------------------------------------
COLLECTIVE_CATS = {"all-reduce", "all-gather", "all-to-all", "reduce-scatter",
                   "collective-permute", "send", "recv", "collective-fusion"}
COMPUTE_CATS = {"loop fusion", "convolution fusion", "input fusion",
                "output fusion", "fusion", "convolution", "dot", "reduce",
                "non-fusion elementwise", "elementwise", "rng", "reduce-window",
                "cumsum"}
DATAFMT_CATS = {"data formatting", "copy", "transpose", "reshape", "broadcast",
                "slice", "pad", "iota", "bitcast", "concatenate", "select",
                "reverse", "tuple", "get-tuple-element"}
ASYNC_CATS = {"async-start", "async-done", "copy-start", "copy-done"}
CTRL_CATS = {"conditional", "while", "call"}


def categorize(op_name, hlo_category=None):
    """Coarse decode-oriented category from op name + profiler hlo_category."""
    name = op_name.lower()
    c = (hlo_category or "").lower()
    if name.startswith("dsa_sparse_decode") or "dsa_sparse" in name:
        return "pallas: dsa_sparse_decode (sparse MLA attend)"
    if name.startswith("gmm"):
        return "pallas: gmm grouped matmul (MoE experts)"
    # Collectives BEFORE gather patterns ('\bgather' would match 'all-gather').
    if c in COLLECTIVE_CATS or re.search(
            r"all-reduce|all-gather|all_gather|all-to-all|all_to_all|reduce-scatter"
            r"|reduce_scatter|collective-permute|ppermute|^psum|^pmax|^pmin", name):
        return "collectives"
    if c == "sort" or re.search(r"top[_-]?k|sort", name):
        return "sort/top-k"
    if re.match(r"(gather|scatter)_custom_fusion", name) or \
            re.search(r"(?<![\w-])gather|(?<![\w-])scatter|dynamic-slice"
                      r"|dynamic_slice|dynamic-update-slice", name):
        return "gather/scatter/dyn-slice"
    if c in CTRL_CATS:
        return "control-flow self-time (conditional/while)"
    if c in ("gather", "scatter", "dynamic-slice", "dynamic-update-slice",
             "custom fusion"):
        # custom fusion in these traces == gather/scatter custom fusions
        return "gather/scatter/dyn-slice"
    if c in ASYNC_CATS or re.search(r"-(start|done)$", name):
        return "async start/done markers"
    if c == "custom-call" or name.startswith("%custom-call") or \
            name.startswith("custom-call"):
        return "custom-call: other (fp8 weight formatting etc.)"
    if c in COMPUTE_CATS or re.search(r"fusion|dot|einsum|conv|matmul", name):
        return "compute (fusion/dot/conv)"
    if c in ("infeed", "outfeed", "host") or re.search(r"infeed|outfeed", name):
        return "infeed/outfeed/host"
    if c in DATAFMT_CATS or re.search(
            r"copy|transpose|reshape|bitcast|concatenate|broadcast|slice|pad|iota", name):
        return "data movement (copy/transpose/reshape)"
    return "other/uncategorized"


# ---------------------------------------------------------------------------
# Device-plane aggregation
# ---------------------------------------------------------------------------
def is_device_plane(plane):
    return plane.name.startswith("/device:TPU:")


def find_line(plane, name_substr):
    for line in plane.lines:
        nm = line.display_name or line.name
        if name_substr.lower() in nm.lower():
            return line
    return None


def _self_times(evs):
    """evs: list of (offset_ps, dur_ps, key) sorted by (offset, -dur).

    Returns (self_ps_list, top_level_flags) via a nesting stack sweep.
    Assumes proper nesting (XLA Ops line guarantees this)."""
    n = len(evs)
    self_ps = [0] * n
    top = [False] * n
    stack = []  # indices
    for i, (off, dur, _k) in enumerate(evs):
        end = off + dur
        while stack and evs[stack[-1]][0] + evs[stack[-1]][1] <= off:
            stack.pop()
        if stack:
            p = stack[-1]
            self_ps[p] -= dur          # remove child time from parent
        else:
            top[i] = True
        self_ps[i] += dur
        stack.append(i)
    return self_ps, top


def aggregate_device_plane(plane, step_module_re=r"jit_step_fun_impl"):
    """Nested-aware aggregation of the XLA Ops line of one TPU core plane."""
    ev_md = dict(plane.event_metadata.items())
    st_md = {k: v.name for k, v in plane.stat_metadata.items()}

    ops_line = find_line(plane, "XLA Ops")
    mod_line = find_line(plane, "XLA Modules")
    async_line = find_line(plane, "Async")

    # ---- modules / steps ----
    modules = []
    for ev in (mod_line.events if mod_line else ()):
        md = ev_md.get(ev.metadata_id)
        modules.append({
            "name": (md.display_name or md.name) if md else str(ev.metadata_id),
            "offset_ps": ev.offset_ps, "duration_ps": ev.duration_ps,
        })
    modules.sort(key=lambda m: m["offset_ps"])
    steps = [m for m in modules if re.search(step_module_re, m["name"])]
    step_bounds = [(s["offset_ps"], s["offset_ps"] + s["duration_ps"])
                   for s in steps]

    # ---- ops: nesting-aware self time ----
    raw = [(e.offset_ps, e.duration_ps, e.metadata_id)
           for e in (ops_line.events if ops_line else ())]
    raw.sort(key=lambda t: (t[0], -t[1]))
    self_ps, top = _self_times(raw)

    md_name = {}
    md_cat = {}
    for mid in {t[2] for t in raw}:
        md = ev_md.get(mid)
        nm = (md.display_name or md.name) if md else str(mid)
        hc = None
        if md is not None:
            for s in md.stats:
                if st_md.get(s.metadata_id) == "hlo_category":
                    hc = stat_value(s)
        # collapse duplicated instances: strip trailing .N id
        base = re.sub(r"\.\d+$", "", nm)
        md_name[mid] = (nm, base)
        md_cat[mid] = (hc, categorize(nm, hc))

    per_op = collections.defaultdict(lambda: [0, 0, 0])  # base -> [self, total, count]
    base_cat = {}  # base name -> hlo-informed category
    per_cat = collections.defaultdict(lambda: [0, 0])    # cat  -> [self, count]
    per_hlo_cat = collections.defaultdict(int)
    # per step x category self-time
    n_steps = len(step_bounds)
    step_cat = [collections.defaultdict(int) for _ in range(n_steps)]
    step_busy = [0] * n_steps
    outside_step_self = 0

    si = 0
    t_min = raw[0][0] if raw else 0
    t_max = 0
    busy_ps = 0
    for i, (off, dur, mid) in enumerate(raw):
        sp = self_ps[i]
        base = md_name[mid][1]
        hc, cat = md_cat[mid]
        base_cat.setdefault(base, cat)
        acc = per_op[base]
        acc[0] += sp; acc[1] += dur; acc[2] += 1
        per_cat[cat][0] += sp; per_cat[cat][1] += 1
        per_hlo_cat[hc or "?"] += sp
        if top[i]:
            busy_ps += dur
        end = off + dur
        if end > t_max:
            t_max = end
        # step attribution by midpoint
        mid_t = off + dur // 2
        while si < n_steps and mid_t >= step_bounds[si][1]:
            si += 1
        if si < n_steps and step_bounds[si][0] <= mid_t < step_bounds[si][1]:
            step_cat[si][cat] += sp
            step_busy[si] += sp
        else:
            outside_step_self += sp

    window_ps = t_max - t_min

    # ---- async DMA line (overlapping; reported separately) ----
    async_agg = collections.defaultdict(lambda: [0, 0])
    for e in (async_line.events if async_line else ()):
        md = ev_md.get(e.metadata_id)
        nm = (md.display_name or md.name) if md else str(e.metadata_id)
        async_agg[nm][0] += e.duration_ps
        async_agg[nm][1] += 1

    return {
        "plane": plane.name,
        "n_op_events": len(raw),
        "window_ps": window_ps,
        "t_min_ps": t_min, "t_max_ps": t_max,
        "busy_ps": busy_ps,
        "per_op": {k: {"self_ps": v[0], "total_ps": v[1], "count": v[2],
                       "category": base_cat[k]}
                   for k, v in per_op.items()},
        "per_category": {k: {"self_ps": v[0], "count": v[1]}
                         for k, v in per_cat.items()},
        "per_hlo_category": dict(per_hlo_cat),
        "steps": [{"offset_ps": a, "end_ps": b, "duration_ps": b - a,
                   "busy_ps": step_busy[i],
                   "per_category_ps": dict(step_cat[i])}
                  for i, (a, b) in enumerate(step_bounds)],
        "outside_step_self_ps": outside_step_self,
        "modules": modules,
        "async_ops": {k: {"total_ps": v[0], "count": v[1]}
                      for k, v in async_agg.items()},
    }


def aggregate_host(path, step_module_re=r"jit_step_fun_impl"):
    """Aggregate all TPU device planes in one xplane.pb. Returns per-core list."""
    xs = load_xspace(path)
    host = xs.hostnames[0] if xs.hostnames else path
    out = []
    for plane in xs.planes:
        if is_device_plane(plane):
            r = aggregate_device_plane(plane, step_module_re)
            r["host"] = host
            out.append(r)
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def cmd_inspect(path):
    xs = load_xspace(path)
    print(f"hostnames: {list(xs.hostnames)}")
    for p in xs.planes:
        n_ev = sum(len(l.events) for l in p.lines)
        print(f"\nPLANE {p.name!r}  id={p.id}  lines={len(p.lines)}  events={n_ev}"
              f"  ev_md={len(p.event_metadata)}  stat_md={len(p.stat_metadata)}")
        for l in p.lines:
            nm = l.display_name or l.name
            print(f"   line {nm!r:40s} events={len(l.events):8d} "
                  f"ts_ns={l.timestamp_ns}")
        if p.name.startswith("/device:"):
            names = [v.name for v in p.stat_metadata.values()]
            print(f"   stat_metadata names: {sorted(names)[:60]}")


def cmd_aggregate(path, out_json=None):
    res = aggregate_host(path)
    for core in res:
        w, b = core["window_ps"], core["busy_ps"]
        print(f"{core['plane']}: window={w / 1e9:.1f}ms busy={b / 1e9:.1f}ms "
              f"idle={100 * (1 - b / max(w, 1)):.1f}% steps={len(core['steps'])} "
              f"ops={len(core['per_op'])}")
    if out_json:
        with open(out_json, "w") as f:
            json.dump(res, f)
        print(f"wrote {out_json}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)
    cmd, path = sys.argv[1], sys.argv[2]
    if cmd == "inspect":
        cmd_inspect(path)
    elif cmd == "aggregate":
        cmd_aggregate(path, sys.argv[3] if len(sys.argv) > 3 else None)
    else:
        print(__doc__)
        sys.exit(1)

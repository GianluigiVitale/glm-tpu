"""JAX distributed initialization and the host-boundary fleet vote of one worker.

``initialize_runtime`` joins the eight-process TPU runtime, authenticates the staged topology
captures against the live devices and builds the ``("expert", "feature")`` device mesh in
physical order; ``_batched_fleet_all`` is the all-host boolean vote the runtime takes at declared
host boundaries. Both moved verbatim in S2a out of the research decoder driver script
(``archive/research-20260922``).
"""

from __future__ import annotations

import argparse
import json
import socket
from typing import Any

import numpy as np

from glm_tpu.distributed.mesh import build_physical_mesh
from glm_tpu.distributed.topology import device_record, validate_topology_fleet


def initialize_runtime(args: argparse.Namespace) -> tuple[Any, Any, Any, Any, Any]:
    import jax
    from jax.sharding import Mesh

    jax.distributed.initialize(
        coordinator_address=args.coordinator_address,
        num_processes=args.num_processes,
        process_id=args.process_id,
    )
    if (
        jax.default_backend() != "tpu"
        or jax.device_count() != 32
        or len(jax.local_devices()) != 4
        or jax.process_count() != 8
    ):
        raise RuntimeError("runner did not initialize the exact 8x4 TPU runtime")
    captures = tuple(
        json.loads((args.topology_capture_root / f"topology.rank{launch_process_id}.json").read_text(encoding="utf-8"))
        for launch_process_id in range(8)
    )
    topology, ordered_captures, fleet_sha = validate_topology_fleet(
        captures,
        expected_topology_sha256=args.topology_sha256,
        expected_fleet_sha256=args.topology_fleet_sha256,
        slice_name=args.slice_name,
    )
    launch_capture = ordered_captures[args.process_id]
    if (
        launch_capture["hostname"] != socket.gethostname()
        or launch_capture["jax_process_index"] != jax.process_index()
        or launch_capture["local_device_ids"] != [int(device.id) for device in jax.local_devices()]
    ):
        raise RuntimeError("WS32 launch/JAX/topology fleet mapping drifted")
    physical_mesh = build_physical_mesh(topology)
    if physical_mesh.mesh_hash != args.mesh_sha256:
        raise ValueError("physical mesh hash drifted")
    runtime_by_id = {int(device.id): device for device in jax.devices()}
    if set(runtime_by_id) != set(physical_mesh.flattened_device_ids):
        raise ValueError("runtime device ids differ from physical mesh")
    for captured in topology.devices:
        if (
            device_record(
                runtime_by_id[captured.device_id],
                local_device_id=captured.local_device_id,
            )
            != captured.to_dict()
        ):
            raise ValueError(f"runtime topology drifted at {captured.device_id}")
    mesh = Mesh(
        np.asarray(
            [runtime_by_id[item] for item in physical_mesh.flattened_device_ids],
            dtype=object,
        ).reshape(8, 4),
        ("expert", "feature"),
    )
    return jax, mesh, physical_mesh, topology, fleet_sha


def _batched_fleet_all(value: bool) -> bool:
    """The all-host boolean vote at the declared host boundaries: each runtime phase, and each token's
    validity and delivery.

    Every call is one all-gather of the eight hosts' votes, run on the host between device calls; never
    inside a layer or a compiled program.
    """
    from jax.experimental import multihost_utils

    values = np.asarray(multihost_utils.process_allgather(np.asarray(int(value), np.int32)))
    if values.shape != (8,) or not np.isin(values, (0, 1)).all():
        raise ValueError("batched host consensus requires eight boolean votes")
    return bool(values.all())

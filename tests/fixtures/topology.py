"""A synthetic eight-host TPU v4-64 fleet for the topology tests: fake JAX devices, a fake ``jax`` module for one host,
and the eight captures the real capture worker (``glm_tpu.distributed.topology_capture.capture``) writes on it.

The hosts are ``example-w-<rank>`` for ranks 0 to 7; JAX numbers its processes in another order than the hosts' ranks
(``PERMUTATION``: launch rank -> JAX process index), as the TPU runtime does. Device ``d`` belongs to JAX process
``d // 4`` and sits at a synthetic 2x4x4 coordinate. Neutral example values only."""

from __future__ import annotations

from argparse import Namespace
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
from unittest import mock

import numpy as np

HOSTS = [f"example-w-{rank}" for rank in range(8)]
PERMUTATION = (2, 7, 0, 5, 1, 4, 6, 3)  # synthetic


def device(device_id: int, **changes) -> SimpleNamespace:
    """One fake TPU v4 chip as ``jax.devices()`` reports it (no ``local_hardware_id``, as on TPU v4)."""
    value = dict(
        id=device_id,
        process_index=device_id // 4,
        coords=(device_id % 2, (device_id // 2) % 4, device_id // 8),
        core_on_chip=0,
        platform="tpu",
        device_kind="TPU v4",
        local_hardware_id=None,
    )
    value.update(changes)
    return SimpleNamespace(**value)


def devices() -> list[SimpleNamespace]:
    return [device(d) for d in range(32)]


def fleet_ids() -> list[list[int]]:
    """Each JAX process's ``jax.local_devices()`` ids, in process order (what ``process_allgather`` returns)."""
    return [[4 * process + k for k in range(4)] for process in range(8)]


def local_order() -> dict[int, int]:
    return {d: index for row in fleet_ids() for index, d in enumerate(row)}


class FakeJax(ModuleType):
    """The ``jax`` of host ``rank``: its JAX process is ``PERMUTATION[rank]``; the all-gathers return every host's
    value (``digest_of`` replaces one host's contract digest, to make the hosts disagree)."""

    __version__ = "0.10.1"

    def __init__(self, rank: int, *, digest_of: dict[int, bytes] | None = None, all_devices=None):
        super().__init__("jax")
        self.rank, self.process = rank, PERMUTATION[rank]
        self.calls: list[tuple] = []
        self._devices = devices() if all_devices is None else all_devices
        self.digest_of = digest_of or {}
        self.distributed = SimpleNamespace(
            initialize=self._initialize, shutdown=lambda: self.calls.append(("shutdown",))
        )
        self.multihost_utils = SimpleNamespace(
            process_allgather=self._allgather, sync_global_devices=lambda name: self.calls.append(("sync", name))
        )
        self.experimental = SimpleNamespace(multihost_utils=self.multihost_utils)

    def _initialize(self, **kwargs):
        self.calls.append(("initialize", kwargs))

    def _allgather(self, value):
        value = np.asarray(value)
        if value.dtype == np.int32:  # the local device ids
            assert value.tolist() == fleet_ids()[self.process]
            return np.asarray(fleet_ids(), dtype=np.int32)
        rows = [np.frombuffer(self.digest_of.get(p, value.tobytes()), dtype=np.uint8) for p in range(8)]
        return np.stack(rows)

    def local_devices(self):
        return [d for d in self._devices if d.process_index == self.process]

    def local_device_count(self):
        return 4

    def devices(self):
        return list(self._devices)

    def device_count(self):
        return len(self._devices)

    def process_count(self):
        return 8

    def process_index(self):
        return self.process


@contextmanager
def fake_jax(rank: int, **options) -> Iterator[FakeJax]:
    """``jax`` and ``jax.experimental.multihost_utils`` replaced by host ``rank``'s fakes for the block."""
    jax = FakeJax(rank, **options)
    experimental = ModuleType("jax.experimental")
    experimental.multihost_utils = jax.multihost_utils
    modules = {"jax": jax, "jax.experimental": experimental, "jax.experimental.multihost_utils": jax.multihost_utils}
    with mock.patch.dict(sys.modules, modules):
        yield jax


def capture_all(root: Path, site, code_hash: str) -> list[bytes]:
    """The eight captures the capture worker writes under ``root`` for ``site`` (as each host would, in launch
    order), their bytes in launch order."""
    from glm_tpu.distributed import topology_capture

    args = Namespace(output=root, code_hash=code_hash, coordinator_address=site.fleet.coordinator_address)
    for rank in range(8):
        with fake_jax(rank), mock.patch.object(topology_capture.socket, "gethostname", lambda rank=rank: HOSTS[rank]):
            topology_capture.capture(args, site, rank)
    return [(root / topology_capture.capture_file(rank)).read_bytes() for rank in range(8)]

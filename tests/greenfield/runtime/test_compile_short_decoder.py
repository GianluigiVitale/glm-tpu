from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from scripts.greenfield.compile_short_decoder import _materialize_global_array


class _Jax:
    def __init__(self, materialized: np.ndarray) -> None:
        self.materialized = materialized
        self.device_get_calls = 0

    def device_get(self, value: object) -> np.ndarray:
        self.device_get_calls += 1
        return self.materialized


class _Multihost:
    def __init__(self, gathered: np.ndarray) -> None:
        self.gathered = gathered
        self.process_allgather_calls = 0
        self.tiled_values: list[bool] = []

    def process_allgather(self, value: object, *, tiled: bool) -> np.ndarray:
        self.process_allgather_calls += 1
        self.tiled_values.append(tiled)
        return self.gathered


def test_materialize_global_array_uses_device_get_when_fully_addressable() -> None:
    expected = np.arange(12, dtype=np.int32).reshape(4, 3)
    value = SimpleNamespace(is_fully_addressable=True, shape=expected.shape)
    jax = _Jax(expected)
    multihost = _Multihost(np.empty((0,), dtype=np.int32))

    actual = _materialize_global_array(jax, multihost, value)

    np.testing.assert_array_equal(actual, expected)
    assert jax.device_get_calls == 1
    assert multihost.process_allgather_calls == 0
    assert multihost.tiled_values == []


def test_materialize_global_array_gathers_non_addressable_global_array() -> None:
    expected = np.arange(24, dtype=np.int32).reshape(8, 3)
    value = SimpleNamespace(is_fully_addressable=False, shape=expected.shape)
    jax = _Jax(np.empty((0,), dtype=np.int32))
    multihost = _Multihost(expected)

    actual = _materialize_global_array(jax, multihost, value)

    np.testing.assert_array_equal(actual, expected)
    assert jax.device_get_calls == 0
    assert multihost.process_allgather_calls == 1
    assert multihost.tiled_values == [True]


def test_materialize_global_array_fails_closed_on_gather_shape_drift() -> None:
    value = SimpleNamespace(is_fully_addressable=False, shape=(8, 3))
    jax = _Jax(np.empty((0,), dtype=np.int32))
    multihost = _Multihost(np.empty((1, 8, 3), dtype=np.int32))

    with pytest.raises(RuntimeError, match="global array gather changed shape"):
        _materialize_global_array(jax, multihost, value)

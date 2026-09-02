"""Host-evaluated FP32 rotary tables for the DSA indexer.

The accepted GLM indexer applies an accurately evaluated FP32 cos/sin table.
On the accelerator, ``cos``/``sin`` of large rotary angles (position times
inverse frequency: thousands of radians at 8K context) lose up to ~1e-2 through
range reduction; the protected V2 projection replay measured exactly that loss
while the projection was FP32-accurate and the key LayerNorm bit-exact.  These
helpers evaluate the table once on the host.  Inverse frequencies use the same
negative-power FP32 form as :func:`rotary.rotary_cos_sin`, which reproduces the
accepted frequencies bit for bit; angles are FP32 products; cos/sin are FP32
libm values (within one ulp of XLA-CPU).  The table stays FP32 because DSA keys
and queries are FP32 and must not round to BF16.  The established
:mod:`rotary` module is untouched so historical source authorities keep their
byte pins.
"""

from __future__ import annotations

from hashlib import sha256

import numpy as np


def _require_table_geometry(rotary_dim: int, theta: float) -> None:
    if (
        not isinstance(rotary_dim, int)
        or isinstance(rotary_dim, bool)
        or rotary_dim <= 0
        or rotary_dim % 2
    ):
        raise ValueError("DSA rotary table dimension must be positive and even")
    if not isinstance(theta, (int, float)) or isinstance(theta, bool) or theta <= 0:
        raise ValueError("DSA rotary table theta must be positive")


def dsa_inverse_frequencies_host(*, rotary_dim: int, theta: float) -> np.ndarray:
    """FP32 inverse frequencies, bit-identical to the accepted on-device form."""

    _require_table_geometry(rotary_dim, theta)
    dimensions = np.arange(0, rotary_dim, 2, dtype=np.float32)
    return np.ascontiguousarray(
        np.power(
            np.float32(theta),
            -dimensions / np.float32(rotary_dim),
            dtype=np.float32,
        )
    )


def build_dsa_rotary_table_host(
    context_capacity: int,
    *,
    rotary_dim: int,
    theta: float,
) -> np.ndarray:
    """Build the FP32 ``cos|sin`` table, one row per position."""

    if (
        not isinstance(context_capacity, int)
        or isinstance(context_capacity, bool)
        or context_capacity <= 0
    ):
        raise ValueError("DSA rotary table capacity must be a positive integer")
    frequencies = dsa_inverse_frequencies_host(rotary_dim=rotary_dim, theta=theta)
    positions = np.arange(context_capacity, dtype=np.float32)
    angles = np.multiply(positions[:, None], frequencies[None, :], dtype=np.float32)
    table = np.concatenate(
        (np.cos(angles, dtype=np.float32), np.sin(angles, dtype=np.float32)),
        axis=-1,
    ).astype(np.float32)
    if table.shape != (context_capacity, rotary_dim) or not np.all(np.isfinite(table)):
        raise ValueError("DSA rotary table construction drifted")
    return np.ascontiguousarray(table)


def dsa_rotary_row_host(position: int, *, rotary_dim: int, theta: float) -> np.ndarray:
    """Return the FP32 ``cos|sin`` row of :func:`build_dsa_rotary_table_host`."""

    if not isinstance(position, int) or isinstance(position, bool) or position < 0:
        raise ValueError("DSA rotary row position must be a non-negative integer")
    frequencies = dsa_inverse_frequencies_host(rotary_dim=rotary_dim, theta=theta)
    angles = np.multiply(np.float32(position), frequencies, dtype=np.float32)
    row = np.concatenate(
        (np.cos(angles, dtype=np.float32), np.sin(angles, dtype=np.float32))
    ).astype(np.float32)
    return np.ascontiguousarray(row)


def dsa_rotary_table_sha256(table: np.ndarray) -> str:
    """Hash one contiguous FP32 DSA table (or row) in storage order."""

    value = np.ascontiguousarray(table)
    if value.dtype != np.float32 or value.ndim not in (1, 2):
        raise ValueError("DSA rotary table hash requires an FP32 row or table")
    return sha256(value.tobytes()).hexdigest()

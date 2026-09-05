"""Spec §23.8 rotary-at-long-positions diagnostic (declared side program).

Measures, on the executing accelerator, the WS32 rotary forms against FP64 truth
and against the accepted legacy main-attention form, per rotary pair index and
position band, for positions ``0 .. positions-1``:

* ``device_main``   — ``rotary_cos_sin(dtype=bf16)`` + ``apply_rotary`` in BF16 on
  the accelerator (the WS32 main-attention form, ``ws32_layer.py`` main site).
* ``device_indexer`` — ``rotary_cos_sin(dtype=f32)`` + ``apply_rotary`` in FP32 on
  the accelerator (the WS32 and legacy indexer form).
* ``legacy_main``   — host FP32 ``cos/sin`` table stored as BF16
  (``build_rotary_table_host``), FP32 products, one final BF16 round — the
  DB531-evidenced legacy main form, evaluated on the host in NumPy so it does
  not depend on the accelerator.
* ``fp64``          — exact FP64 angles and trigonometry.

Every form rotates the same unit probe ``(1, 0)`` per pair, so the rotated
components are the form's effective ``(cos, sin)`` per pair (the ``(0, 1)``
probe would give ``(-sin, cos)`` and adds nothing under sign-symmetric BF16
rounding).  By design the probe cannot see the position-independent BF16
product rounding of the real attention site (the DB530 class); that is out of
scope here.  The pre-registered decision rule (§23.8): in every
(pair, band) cell the device main form must differ from the legacy main form by
no more than ``kappa`` times the legacy form's own deviation from FP64.  A cell
where the legacy form is exact (deviation 0) requires the device form to be
exact too.  The module never decides anything about the indexer: the legacy
indexer is on-device as well; its numbers are recorded for the record.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Any

import ml_dtypes
import numpy as np

ROTARY_DIM = 64
THETA = 8_000_000.0
DEFAULT_POSITIONS = 262_657  # 513 pages * 512 (L8 capacity) + 1
BANDS: tuple[tuple[str, int, int], ...] = (
    ("le_8192", 0, 8_192),
    ("le_32768", 8_192, 32_768),
    ("le_131072", 32_768, 131_072),
    ("le_262656", 131_072, 262_657),
)
KAPPA = 2.0
SCRIPT_PATH = Path(__file__).resolve()


def script_sha256() -> str:
    return sha256(SCRIPT_PATH.read_bytes()).hexdigest()


def _bf16(value: np.ndarray) -> np.ndarray:
    return np.asarray(value, dtype=np.float32).astype(ml_dtypes.bfloat16).astype(np.float32)


def fp64_cos_sin(positions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    frequencies = np.power(
        np.float64(THETA), -np.arange(0, ROTARY_DIM, 2, dtype=np.float64) / np.float64(ROTARY_DIM)
    )
    angles = positions.astype(np.float64)[:, None] * frequencies[None, :]
    return np.cos(angles), np.sin(angles)


def legacy_main_cos_sin(positions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Rotated unit probes of the legacy main form, as float32 arrays of BF16 values.

    Table row (BF16 cos|sin from FP32 math), FP32 products with the BF16 probe,
    one final BF16 round — for the ``(1, 0)`` probe the rotated pair is
    ``(cos, sin)`` after the final round, i.e. the BF16 table row itself.
    """
    from ..kernels.reference.rotary import build_rotary_table_host

    table = build_rotary_table_host(int(positions.max()) + 1, rotary_dim=ROTARY_DIM, theta=THETA)
    rows = np.asarray(table, dtype=np.float32)[positions]
    half = ROTARY_DIM // 2
    cos = rows[:, :half]
    sin = rows[:, half:]
    # FP32 products with probe (1, 0): first*cos - second*sin = cos ; second*cos + first*sin = sin.
    return _bf16(cos), _bf16(sin)


def platform_attribution() -> dict[str, Any]:
    """Name the accelerator the device forms ran on (a CPU record must never pass as TPU)."""
    import jax

    device = jax.devices()[0]
    try:
        import libtpu  # type: ignore

        libtpu_version = getattr(libtpu, "__version__", None)
    except Exception:  # pragma: no cover - libtpu absent on CPU hosts
        libtpu_version = None
    return {
        "backend": str(jax.default_backend()),
        "device_kind": str(getattr(device, "device_kind", "")),
        "jax_version": str(jax.__version__),
        "libtpu_version": libtpu_version,
        "platform": str(getattr(device, "platform", "")),
    }


def device_forms(positions: np.ndarray) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Evaluate the WS32 on-device forms with JAX on the default backend."""
    import jax
    import jax.numpy as jnp

    from ..kernels.reference.rotary import apply_rotary, rotary_cos_sin

    half = ROTARY_DIM // 2

    def probe(dtype: Any) -> jax.Array:
        # (1, 0) per pair, interleaved layout.
        unit = np.zeros((1, ROTARY_DIM), dtype=np.float32)
        unit[0, 0::2] = 1.0
        return jnp.asarray(unit, dtype=dtype)

    @jax.jit
    def main_form(position_ids: jax.Array) -> jax.Array:
        cos, sin = rotary_cos_sin(position_ids, rotary_dim=ROTARY_DIM, theta=THETA, dtype=jnp.bfloat16)
        rotated = apply_rotary(
            jnp.broadcast_to(probe(jnp.bfloat16), (position_ids.shape[0], ROTARY_DIM)),
            cos,
            sin,
            interleaved=True,
        )
        return rotated.astype(jnp.float32)

    @jax.jit
    def indexer_form(position_ids: jax.Array) -> jax.Array:
        cos, sin = rotary_cos_sin(position_ids, rotary_dim=ROTARY_DIM, theta=THETA, dtype=jnp.float32)
        return apply_rotary(
            jnp.broadcast_to(probe(jnp.float32), (position_ids.shape[0], ROTARY_DIM)),
            cos,
            sin,
            interleaved=True,
        )

    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    ids = jnp.asarray(positions, dtype=jnp.int32)
    for name, form in (("device_main", main_form), ("device_indexer", indexer_form)):
        rotated = np.asarray(jax.device_get(form(ids)), dtype=np.float32)
        out[name] = (rotated[:, 0::2], rotated[:, 1::2])  # (cos, sin) per pair
    return out


@dataclass(frozen=True, slots=True)
class RotaryDiagnosticResult:
    record: dict[str, Any]

    @property
    def passed(self) -> bool:
        return bool(self.record["verdict"] == "PASS")


def run_rotary_diagnostic(*, positions: int = DEFAULT_POSITIONS, kappa: float = KAPPA) -> RotaryDiagnosticResult:
    if not isinstance(positions, int) or positions <= 0 or positions > 1_048_576:
        raise ValueError("diagnostic positions must be a positive integer within the model window")
    ids = np.arange(positions, dtype=np.int64)
    truth = fp64_cos_sin(ids)
    legacy = legacy_main_cos_sin(ids)
    device = device_forms(ids)
    forms = {"legacy_main": legacy, **device}
    cells: list[dict[str, Any]] = []
    failing: list[str] = []
    for band, lo, hi in BANDS:
        if lo >= positions:
            continue
        hi = min(hi, positions)
        sl = slice(lo, hi)
        for pair in range(ROTARY_DIM // 2):
            cell: dict[str, Any] = {"band": band, "pair": pair, "positions": [int(lo), int(hi - 1)]}
            for name, (cos, sin) in forms.items():
                err = np.maximum(
                    np.abs(cos[sl, pair].astype(np.float64) - truth[0][sl, pair]),
                    np.abs(sin[sl, pair].astype(np.float64) - truth[1][sl, pair]),
                )
                cell[f"{name}_max_abs_error_vs_fp64"] = float(err.max())
                cell[f"{name}_p99_abs_error_vs_fp64"] = float(np.percentile(err, 99))
            dm_cos, dm_sin = device["device_main"]
            lm_cos, lm_sin = legacy
            diff = np.maximum(np.abs(dm_cos[sl, pair] - lm_cos[sl, pair]), np.abs(dm_sin[sl, pair] - lm_sin[sl, pair]))
            cell["device_main_vs_legacy_main_max_abs"] = float(diff.max())
            cell["device_main_vs_legacy_main_differing_bf16_components"] = int(
                np.count_nonzero(dm_cos[sl, pair] != lm_cos[sl, pair]) + np.count_nonzero(dm_sin[sl, pair] != lm_sin[sl, pair])
            )
            bound = kappa * cell["legacy_main_max_abs_error_vs_fp64"]
            cell["bound"] = float(bound)
            cell["pass"] = bool(cell["device_main_vs_legacy_main_max_abs"] <= bound)
            if not cell["pass"]:
                failing.append(f"{band}/pair{pair}")
            cells.append(cell)
    record = {
        "artifact_kind": "greenfield_ws32_rotary_long_position_diagnostic",
        "bands": [list(b) for b in BANDS],
        "cells": cells,
        "decision_rule": (
            "per (pair, band): device_main_vs_legacy_main_max_abs <= kappa * legacy_main_max_abs_error_vs_fp64"
        ),
        "failing_cells": failing,
        "forms": {
            "device_indexer": "rotary_cos_sin(f32) + apply_rotary f32 on device (WS32 and legacy indexer form; recorded only)",
            "device_main": "rotary_cos_sin(bf16) + apply_rotary bf16 on device (WS32 main-attention form)",
            "fp64": "float64 angles and trigonometry",
            "legacy_main": "host FP32 table stored BF16, FP32 products, final BF16 round (DB531 legacy main form)",
        },
        "kappa": float(kappa),
        "platform": platform_attribution(),
        "positions": int(positions),
        "probe": "(1, 0) per pair; position-independent BF16 product rounding of the real site is out of scope",
        "rotary_dim": ROTARY_DIM,
        "script_sha256": script_sha256(),
        "theta": THETA,
        "verdict": "PASS" if not failing else "FAIL",
    }
    record["record_sha256"] = sha256(
        json.dumps(record, allow_nan=False, separators=(",", ":"), sort_keys=True).encode("utf-8")
    ).hexdigest()
    return RotaryDiagnosticResult(record)


def verify_rotary_diagnostic_record(
    record: Any,
    *,
    expected_script_sha256: str | None = None,
    expected_backend: str | None = "tpu",
    pinned: bool = True,
) -> None:
    """Fail closed on a malformed, self-inconsistent or off-protocol diagnostic record.

    ``pinned`` enforces the pre-registered protocol of §23.8 (kappa, bands,
    theta, rotary dim, the full 262,657-position window, all 4 x 32 cells);
    ``expected_backend`` refuses a record computed on any other platform.
    """
    if type(record) is not dict or record.get("artifact_kind") != "greenfield_ws32_rotary_long_position_diagnostic":
        raise ValueError("rotary diagnostic record kind drifted")
    if pinned:
        if (
            record.get("kappa") != KAPPA
            or record.get("bands") != [list(b) for b in BANDS]
            or record.get("theta") != THETA
            or record.get("rotary_dim") != ROTARY_DIM
            or type(record.get("positions")) is not int
            or record["positions"] < DEFAULT_POSITIONS
            or type(record.get("cells")) is not list
            or len(record["cells"]) != len(BANDS) * (ROTARY_DIM // 2)
        ):
            raise ValueError("rotary diagnostic protocol differs from the pre-registered §23.8 rule")
    platform = record.get("platform")
    if type(platform) is not dict or not platform.get("backend"):
        raise ValueError("rotary diagnostic record lacks platform attribution")
    if expected_backend is not None and platform.get("backend") != expected_backend:
        raise ValueError(f"rotary diagnostic ran on {platform.get('backend')!r}, expected {expected_backend!r}")
    body = {k: v for k, v in record.items() if k != "record_sha256"}
    digest = sha256(json.dumps(body, allow_nan=False, separators=(",", ":"), sort_keys=True).encode("utf-8")).hexdigest()
    if record.get("record_sha256") != digest:
        raise ValueError("rotary diagnostic record checksum drifted")
    if expected_script_sha256 is not None and record.get("script_sha256") != expected_script_sha256:
        raise ValueError("rotary diagnostic script identity drifted")
    cells = record.get("cells")
    if type(cells) is not list or not cells or record.get("rotary_dim") != ROTARY_DIM:
        raise ValueError("rotary diagnostic cells drifted")
    failing = [f"{c['band']}/pair{c['pair']}" for c in cells if c.get("pass") is not True]
    if failing != record.get("failing_cells") or record.get("verdict") != ("PASS" if not failing else "FAIL"):
        raise ValueError("rotary diagnostic verdict does not match its cells")

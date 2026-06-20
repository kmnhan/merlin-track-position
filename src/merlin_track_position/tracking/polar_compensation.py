from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any

import numpy as np
import xarray as xr

POLAR_COMPENSATION_FORMAT = "merlin_track_position_polar_compensation"
POLAR_COMPENSATION_VERSION = 1
POLAR_COMPENSATION_AXES = ("x", "z")
POLAR_COMPENSATION_PROBE_DIM = "polar_compensation_probe"
POLAR_COMPENSATION_AXIS_DIM = "polar_compensation_axis"


def polar_rotation_matrix(delta_deg: float) -> np.ndarray:
    """Return the project polar rotation matrix in the x/z plane."""
    delta_rad = np.deg2rad(float(delta_deg))
    if not np.isfinite(delta_rad):
        raise ValueError("polar delta must be finite")
    cosine = float(np.cos(delta_rad))
    sine = float(np.sin(delta_rad))
    return np.asarray(
        [
            [cosine, sine],
            [-sine, cosine],
        ],
        dtype=np.float64,
    )


def predict_polar_compensation_xz(
    polar_deg: float | Sequence[float] | np.ndarray,
    *,
    anchor_polar_deg: float,
    anchor_xz_mm: Sequence[float] | np.ndarray,
    anchor_to_center_xz_mm: Sequence[float] | np.ndarray,
) -> np.ndarray:
    """Predict x/z compensation positions for polar angle(s)."""
    polar_values = np.asarray(polar_deg, dtype=np.float64)
    scalar_input = polar_values.ndim == 0
    polar_values = np.atleast_1d(polar_values)
    if not np.isfinite(polar_values).all():
        raise ValueError("polar_deg must contain finite values")
    anchor = _xz_vector(anchor_xz_mm, "anchor_xz_mm")
    radius = _xz_vector(anchor_to_center_xz_mm, "anchor_to_center_xz_mm")
    anchor_polar = float(anchor_polar_deg)
    if not np.isfinite(anchor_polar):
        raise ValueError("anchor_polar_deg must be finite")

    predicted = np.empty((polar_values.size, len(POLAR_COMPENSATION_AXES)))
    for index, polar in enumerate(polar_values):
        rotation = polar_rotation_matrix(float(polar) - anchor_polar)
        predicted[index, :] = anchor + (rotation - np.eye(2)) @ radius
    return predicted[0] if scalar_input else predicted


def fit_polar_compensation_model(
    polar_deg: Sequence[float] | np.ndarray,
    x_mm: Sequence[float] | np.ndarray,
    y_mm: Sequence[float] | np.ndarray,
    z_mm: Sequence[float] | np.ndarray,
    *,
    anchor_polar_deg: float,
    duplicate_tolerance_deg: float = 1.0e-9,
) -> xr.Dataset:
    """Fit the analytic polar compensation model from corrected p/x/y/z probes."""
    polar = _finite_vector(polar_deg, "polar_deg")
    x = _finite_vector(x_mm, "x_mm")
    y = _finite_vector(y_mm, "y_mm")
    z = _finite_vector(z_mm, "z_mm")
    if not (polar.shape == x.shape == y.shape == z.shape):
        raise ValueError("polar_deg, x_mm, y_mm, and z_mm must have matching shapes")
    if polar.size < 3:
        raise ValueError("polar compensation fit requires at least three points")

    tolerance = float(duplicate_tolerance_deg)
    if not np.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("duplicate_tolerance_deg must be finite and non-negative")
    _validate_unique_polar_values(polar, tolerance)

    anchor_polar = float(anchor_polar_deg)
    if not np.isfinite(anchor_polar):
        raise ValueError("anchor_polar_deg must be finite")
    anchor_matches = np.nonzero(np.isclose(polar, anchor_polar, rtol=0.0, atol=tolerance))[0]
    if anchor_matches.size != 1:
        raise ValueError("exactly one probe point must match anchor_polar_deg")
    anchor_index = int(anchor_matches[0])
    q0 = np.asarray([x[anchor_index], z[anchor_index]], dtype=np.float64)

    rows: list[np.ndarray] = []
    rhs: list[np.ndarray] = []
    for index, probe_polar in enumerate(polar):
        if index == anchor_index:
            continue
        rotation = polar_rotation_matrix(float(probe_polar) - anchor_polar)
        rows.append(rotation - np.eye(2, dtype=np.float64))
        rhs.append(np.asarray([x[index], z[index]], dtype=np.float64) - q0)
    if len(rows) < 2:
        raise ValueError("polar compensation fit requires at least two non-anchor points")

    design = np.concatenate(rows, axis=0)
    target = np.concatenate(rhs, axis=0)
    anchor_to_center, residuals, rank, singular_values = np.linalg.lstsq(
        design,
        target,
        rcond=None,
    )
    if rank < len(POLAR_COMPENSATION_AXES):
        raise ValueError("polar compensation fit is rank deficient")
    center = q0 - anchor_to_center
    predicted = predict_polar_compensation_xz(
        polar,
        anchor_polar_deg=anchor_polar,
        anchor_xz_mm=q0,
        anchor_to_center_xz_mm=anchor_to_center,
    )
    measured = np.column_stack((x, z))
    residual_mm = measured - predicted
    residual_norm_um = 1000.0 * np.linalg.norm(residual_mm, axis=1)
    residual_rms_um = float(np.sqrt(np.mean(residual_norm_um**2)))
    residual_max_um = float(np.max(residual_norm_um))

    return xr.Dataset(
        {
            "polar_compensation_probe_polar_deg": (
                (POLAR_COMPENSATION_PROBE_DIM,),
                polar,
                {"units": "deg"},
            ),
            "polar_compensation_probe_xz_mm": (
                (POLAR_COMPENSATION_PROBE_DIM, POLAR_COMPENSATION_AXIS_DIM),
                measured,
                {"units": "readback-mm"},
            ),
            "polar_compensation_probe_y_mm": (
                (POLAR_COMPENSATION_PROBE_DIM,),
                y,
                {"units": "readback-mm"},
            ),
            "polar_compensation_predicted_xz_mm": (
                (POLAR_COMPENSATION_PROBE_DIM, POLAR_COMPENSATION_AXIS_DIM),
                predicted,
                {"units": "readback-mm"},
            ),
            "polar_compensation_residual_mm": (
                (POLAR_COMPENSATION_PROBE_DIM, POLAR_COMPENSATION_AXIS_DIM),
                residual_mm,
                {"units": "readback-mm"},
            ),
            "polar_compensation_residual_um": (
                (POLAR_COMPENSATION_PROBE_DIM, POLAR_COMPENSATION_AXIS_DIM),
                1000.0 * residual_mm,
                {"units": "um"},
            ),
            "polar_compensation_residual_norm_um": (
                (POLAR_COMPENSATION_PROBE_DIM,),
                residual_norm_um,
                {"units": "um"},
            ),
        },
        coords={
            POLAR_COMPENSATION_PROBE_DIM: np.arange(polar.size, dtype=np.int64),
            POLAR_COMPENSATION_AXIS_DIM: list(POLAR_COMPENSATION_AXES),
        },
        attrs={
            "polar_compensation_format": POLAR_COMPENSATION_FORMAT,
            "polar_compensation_version": POLAR_COMPENSATION_VERSION,
            "polar_compensation_anchor_polar_deg": float(anchor_polar),
            "polar_compensation_anchor_x_mm": float(q0[0]),
            "polar_compensation_anchor_z_mm": float(q0[1]),
            "polar_compensation_anchor_to_center_x_mm": float(anchor_to_center[0]),
            "polar_compensation_anchor_to_center_z_mm": float(anchor_to_center[1]),
            "polar_compensation_center_x_mm": float(center[0]),
            "polar_compensation_center_z_mm": float(center[1]),
            "polar_compensation_residual_rms_um": residual_rms_um,
            "polar_compensation_residual_max_um": residual_max_um,
            "polar_compensation_probe_count": int(polar.size),
            "polar_compensation_fit_rank": int(rank),
            "polar_compensation_fit_singular_values": " ".join(
                f"{float(value):.17g}" for value in singular_values
            ),
            "polar_compensation_fit_lstsq_residual": " ".join(
                f"{float(value):.17g}" for value in np.asarray(residuals).reshape(-1)
            ),
        },
    )


def apply_polar_compensation_model(
    calibration: xr.Dataset,
    model: xr.Dataset,
    *,
    created_at_utc: str | None = None,
) -> xr.Dataset:
    """Return calibration with the polar compensation model embedded."""
    if created_at_utc is None:
        created_at_utc = datetime.now(UTC).isoformat()
    updated = calibration.copy()
    for name, variable in model.data_vars.items():
        updated[name] = variable
    updated = updated.assign_coords(
        {
            POLAR_COMPENSATION_PROBE_DIM: model.coords[POLAR_COMPENSATION_PROBE_DIM],
            POLAR_COMPENSATION_AXIS_DIM: model.coords[POLAR_COMPENSATION_AXIS_DIM],
        }
    )
    attrs: dict[str, Any] = dict(model.attrs)
    attrs["polar_compensation_created_at_utc"] = str(created_at_utc)
    return updated.assign_attrs(attrs)


def predict_polar_compensation_from_attrs(
    attrs: dict[str, Any] | xr.Dataset,
    polar_deg: float | Sequence[float] | np.ndarray,
) -> np.ndarray:
    """Predict x/z from model attrs stored on a calibration dataset."""
    source = attrs.attrs if isinstance(attrs, xr.Dataset) else attrs
    return predict_polar_compensation_xz(
        polar_deg,
        anchor_polar_deg=float(source["polar_compensation_anchor_polar_deg"]),
        anchor_xz_mm=(
            float(source["polar_compensation_anchor_x_mm"]),
            float(source["polar_compensation_anchor_z_mm"]),
        ),
        anchor_to_center_xz_mm=(
            float(source["polar_compensation_anchor_to_center_x_mm"]),
            float(source["polar_compensation_anchor_to_center_z_mm"]),
        ),
    )


def _finite_vector(values: Sequence[float] | np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError(f"{name} must be a 1-D vector")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    return array


def _xz_vector(values: Sequence[float] | np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.shape != (len(POLAR_COMPENSATION_AXES),):
        raise ValueError(f"{name} must contain x and z")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    return array


def _validate_unique_polar_values(polar: np.ndarray, tolerance: float) -> None:
    for left in range(polar.size):
        for right in range(left + 1, polar.size):
            if abs(float(polar[left] - polar[right])) <= tolerance:
                raise ValueError("polar compensation probe angles must be unique")

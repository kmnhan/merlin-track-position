from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any

import numpy as np
import xarray as xr

from merlin_track_position.tracking.calibration_core import _image_coords, _image_dims

POLAR_REFERENCE_FORMAT = "merlin_track_position_polar_reference"
POLAR_REFERENCE_VERSION = 1
POLAR_REFERENCE_DIM = "polar_reference"
POLAR_REFERENCE_REQUIRED_VARIABLES = (
    "polar_reference_polar_deg",
    "polar_reference_tilt_deg",
    "polar_reference_azi_deg",
    "polar_reference_cam0",
    "polar_reference_cam1",
)


def polar_reference_target_positions(
    minimum_deg: float,
    maximum_deg: float,
    *,
    step_deg: float = 1.0,
) -> np.ndarray:
    minimum = float(minimum_deg)
    maximum = float(maximum_deg)
    step = float(step_deg)
    if not np.isfinite(minimum) or not np.isfinite(maximum):
        raise ValueError("polar reference minimum and maximum must be finite")
    if maximum < minimum:
        raise ValueError(
            "polar reference maximum must be greater than or equal to minimum"
        )
    if not np.isfinite(step) or step <= 0.0:
        raise ValueError("polar reference step must be finite and positive")

    span = maximum - minimum
    intervals = max(1, int(np.ceil(span / step))) if span > 0.0 else 0
    if intervals == 0:
        return np.asarray([minimum], dtype=np.float64)
    return np.linspace(minimum, maximum, intervals + 1, dtype=np.float64)


def apply_polar_reference_stack(
    calibration: xr.Dataset,
    *,
    polar_deg: Sequence[float] | np.ndarray,
    tilt_deg: Sequence[float] | np.ndarray,
    azi_deg: Sequence[float] | np.ndarray,
    x_mm: Sequence[float] | np.ndarray | None = None,
    z_mm: Sequence[float] | np.ndarray | None = None,
    cam0: Sequence[Any] | np.ndarray,
    cam1: Sequence[Any] | np.ndarray,
    source_motor_name: str,
    minimum_deg: float,
    maximum_deg: float,
    step_deg: float = 1.0,
    created_at_utc: str | None = None,
) -> xr.Dataset:
    polar = _finite_vector(polar_deg, "polar_deg")
    tilt = _finite_vector(tilt_deg, "tilt_deg")
    azi = _finite_vector(azi_deg, "azi_deg")
    if not (polar.shape == tilt.shape == azi.shape):
        raise ValueError(
            "polar, tilt, and azi reference values must have matching shapes"
        )
    if polar.size < 1:
        raise ValueError("at least one polar reference point is required")
    x = _optional_finite_vector(x_mm, "x_mm", polar.size)
    z = _optional_finite_vector(z_mm, "z_mm", polar.size)

    cam0_stack = _image_stack(cam0, "polar_reference_cam0", polar.size)
    cam1_stack = _image_stack(cam1, "polar_reference_cam1", polar.size)
    if created_at_utc is None:
        created_at_utc = datetime.now(UTC).isoformat()

    drop_names = [
        name
        for name in (
            *POLAR_REFERENCE_REQUIRED_VARIABLES,
            "polar_reference_x_mm",
            "polar_reference_z_mm",
            POLAR_REFERENCE_DIM,
        )
        if name in calibration.variables
    ]
    updated = calibration.drop_vars(drop_names) if drop_names else calibration.copy()
    coords: dict[str, Any] = {
        POLAR_REFERENCE_DIM: np.arange(polar.size, dtype=np.int64)
    }
    coords |= _reference_image_coords("cam0", cam0_stack)
    coords |= _reference_image_coords("cam1", cam1_stack)
    updated = updated.assign_coords(coords)
    updated["polar_reference_polar_deg"] = (
        (POLAR_REFERENCE_DIM,),
        polar,
        {"units": "deg"},
    )
    updated["polar_reference_tilt_deg"] = (
        (POLAR_REFERENCE_DIM,),
        tilt,
        {"units": "deg"},
    )
    updated["polar_reference_azi_deg"] = (
        (POLAR_REFERENCE_DIM,),
        azi,
        {"units": "deg"},
    )
    updated["polar_reference_x_mm"] = (
        (POLAR_REFERENCE_DIM,),
        x,
        {"units": "mm"},
    )
    updated["polar_reference_z_mm"] = (
        (POLAR_REFERENCE_DIM,),
        z,
        {"units": "mm"},
    )
    updated["polar_reference_cam0"] = (
        _reference_image_dims("cam0", cam0_stack),
        cam0_stack,
    )
    updated["polar_reference_cam1"] = (
        _reference_image_dims("cam1", cam1_stack),
        cam1_stack,
    )
    return updated.assign_attrs(
        {
            **updated.attrs,
            "polar_reference_format": POLAR_REFERENCE_FORMAT,
            "polar_reference_version": POLAR_REFERENCE_VERSION,
            "polar_reference_source_motor_name": str(source_motor_name),
            "polar_reference_min_deg": float(minimum_deg),
            "polar_reference_max_deg": float(maximum_deg),
            "polar_reference_step_deg": float(step_deg),
            "polar_reference_count": int(polar.size),
            "polar_reference_created_at_utc": str(created_at_utc),
        }
    )


def closest_polar_reference(
    calibration: xr.Dataset,
    polar_deg: float,
) -> tuple[np.ndarray, np.ndarray, float, float, float, int] | None:
    if any(name not in calibration for name in POLAR_REFERENCE_REQUIRED_VARIABLES):
        return None
    polar_values = np.asarray(
        calibration["polar_reference_polar_deg"].values, dtype=np.float64
    )
    if polar_values.ndim != 1 or polar_values.size < 1:
        return None
    if not np.isfinite(polar_values).all():
        raise ValueError("stored polar reference values must be finite")
    current_polar = float(polar_deg)
    if not np.isfinite(current_polar):
        raise ValueError("current polar must be finite")
    index = int(np.argmin(np.abs(polar_values - current_polar)))
    tilt_values = np.asarray(
        calibration["polar_reference_tilt_deg"].values, dtype=np.float64
    )
    azi_values = np.asarray(
        calibration["polar_reference_azi_deg"].values, dtype=np.float64
    )
    if (
        tilt_values.shape != polar_values.shape
        or azi_values.shape != polar_values.shape
    ):
        raise ValueError(
            "stored polar reference orientation arrays must match polar values"
        )
    tilt = float(tilt_values[index])
    azi = float(azi_values[index])
    if not np.isfinite(tilt) or not np.isfinite(azi):
        raise ValueError("stored polar reference tilt/azi values must be finite")
    cam0 = np.asarray(calibration["polar_reference_cam0"].values)
    cam1 = np.asarray(calibration["polar_reference_cam1"].values)
    if cam0.ndim not in (3, 4) or cam1.ndim not in (3, 4):
        raise ValueError("stored polar reference images must be image stacks")
    if cam0.shape[0] != polar_values.size or cam1.shape[0] != polar_values.size:
        raise ValueError("stored polar reference image counts must match polar values")
    return (
        np.asarray(cam0[index]).copy(),
        np.asarray(cam1[index]).copy(),
        float(polar_values[index]),
        tilt,
        azi,
        index,
    )


def _finite_vector(values: Sequence[float] | np.ndarray, name: str) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64)
    if result.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional array")
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must contain only finite values")
    return result


def _optional_finite_vector(
    values: Sequence[float] | np.ndarray | None,
    name: str,
    count: int,
) -> np.ndarray:
    if values is None:
        return np.full(int(count), np.nan, dtype=np.float64)
    result = _finite_vector(values, name)
    if result.shape != (int(count),):
        raise ValueError(f"{name} must contain one value per polar reference")
    return result


def _image_stack(
    images: Sequence[Any] | np.ndarray, name: str, count: int
) -> np.ndarray:
    stack = np.asarray(images)
    if stack.ndim not in (3, 4):
        raise ValueError(f"{name} must be a stack of 2-D or 3-D images")
    if stack.shape[0] != count:
        raise ValueError(f"{name} must contain one image per polar reference")
    if stack.shape[1] == 0 or stack.shape[2] == 0:
        raise ValueError(f"{name} images must not be empty")
    if not np.issubdtype(stack.dtype, np.number) and not np.issubdtype(
        stack.dtype, np.bool_
    ):
        raise ValueError(f"{name} must contain numeric or boolean images")
    return stack.copy()


def _reference_image_dims(camera: str, stack: np.ndarray) -> tuple[str, ...]:
    dims = _image_dims(camera, stack[0])
    return (POLAR_REFERENCE_DIM, *dims)


def _reference_image_coords(camera: str, stack: np.ndarray) -> dict[str, Any]:
    return _image_coords(camera, stack[0])

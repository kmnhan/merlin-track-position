from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

import numpy as np
import xarray as xr

from merlin_track_position import constants
from merlin_track_position.instruments.cameras import (
    CameraPairPlugin,
    default_camera_pair,
    normalize_capture_count,
)
from merlin_track_position.instruments.motors import get_positions
from merlin_track_position.tracking.calibration_core import (
    CAPTURE_AGGREGATION_MEDIAN_SHIFTS,
    COMMAND_AXES,
    estimate_command_offset,
    validate_visual_calibration_dataset,
    weighted_pixel_residual,
)
from merlin_track_position.tracking.correct import (
    _capture_measurement,
    _orientation_ecc_seed_shift_kwargs,
    _position_values,
    _prefixed_polar_attrs,
    _prefixed_orientation_attrs,
    _runtime_orientation_attrs,
    _runtime_px_per_readback_mm_for_polar,
)

__all__ = ("detect_shift",)

logger = logging.getLogger("merlin_track_position.tracking.detect")


def detect_shift(
    calibration: xr.Dataset,
    camera_pair: CameraPairPlugin | None = None,
    *,
    capture_count: int = constants.DEFAULT_CORRECTION_CAPTURE_COUNT,
    capture_aggregation: str = CAPTURE_AGGREGATION_MEDIAN_SHIFTS,
    weights: Sequence[float] | np.ndarray | None = None,
    **shift_kwargs: Any,
) -> xr.Dataset:
    """Measure signed readback-space displacement without moving motors or saving."""

    validate_visual_calibration_dataset(calibration)
    capture_count = normalize_capture_count(capture_count)
    if camera_pair is None:
        camera_pair = default_camera_pair()

    reference_cam0 = np.asarray(calibration["reference_cam0"].values)
    reference_cam1 = np.asarray(calibration["reference_cam1"].values)
    current_position, current_orientation = _read_current_detection_position()
    current_readback_position_mm = current_position.copy()
    current_polar_deg = float(current_orientation["polar"])
    jacobian, polar_attrs = _runtime_px_per_readback_mm_for_polar(
        calibration,
        current_polar_deg,
    )
    orientation_attrs = _runtime_orientation_attrs(
        calibration,
        polar_attrs=polar_attrs,
        current_tilt_deg=current_orientation["tilt"],
        current_azi_deg=current_orientation["azi"],
    )
    logger.info("Detecting shift without motor correction.")
    measurement_shift_kwargs = _orientation_ecc_seed_shift_kwargs(
        shift_kwargs,
        calibration=calibration,
        jacobian=jacobian,
        polar_attrs=polar_attrs,
        orientation_attrs=orientation_attrs,
        readback_position_mm=current_readback_position_mm,
    )
    measurement = _capture_measurement(
        calibration,
        camera_pair,
        reference_cam0,
        reference_cam1,
        capture_count,
        capture_aggregation=capture_aggregation,
        **measurement_shift_kwargs,
    )
    estimated_offset_mm = estimate_command_offset(
        jacobian,
        measurement,
        weights=weights,
    )
    weighted_residual_px = weighted_pixel_residual(measurement, weights=weights)
    logger.info(
        "Detected readback offset mm=%s, weighted_residual_px=%.6g",
        estimated_offset_mm.tolist(),
        weighted_residual_px,
    )
    return (
        measurement.assign(
            {
                "estimated_readback_offset_mm": (
                    ("command_axis",),
                    estimated_offset_mm,
                    {"units": "readback-mm"},
                ),
                "detected_shift_um": (
                    ("command_axis",),
                    1000.0 * estimated_offset_mm,
                    {"units": "um"},
                ),
                "weighted_residual_px": (
                    (),
                    float(weighted_residual_px),
                    {"units": "px"},
                ),
            }
        )
        .assign_coords(command_axis=list(COMMAND_AXES))
        .assign_attrs(
            _prefixed_polar_attrs("detection", polar_attrs)
            | _prefixed_orientation_attrs("detection", orientation_attrs)
        )
    )


def _read_current_detection_position() -> tuple[np.ndarray, dict[str, float]]:
    try:
        values = _position_values(
            get_positions((*COMMAND_AXES, "p", "t", "a")),
            len(COMMAND_AXES) + 3,
            "current x/y/z/p/t/a readback",
        )
        return (
            values[: len(COMMAND_AXES)].copy(),
            {
                "polar": float(values[len(COMMAND_AXES)]),
                "tilt": float(values[len(COMMAND_AXES) + 1]),
                "azi": float(values[len(COMMAND_AXES) + 2]),
            },
        )
    except Exception as exc:
        logger.info(
            "Could not read current x/y/z/p/t/a for detection (%s); "
            "falling back to x/y/z/p only.",
            exc,
        )
    values = _position_values(
        get_positions((*COMMAND_AXES, "p")),
        len(COMMAND_AXES) + 1,
        "current x/y/z/p readback",
    )
    return (
        values[: len(COMMAND_AXES)].copy(),
        {
            "polar": float(values[-1]),
            "tilt": np.nan,
            "azi": np.nan,
        },
    )

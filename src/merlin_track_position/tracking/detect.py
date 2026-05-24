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
    _polar_ecc_seed_shift_kwargs,
    _position_values,
    _prefixed_polar_attrs,
    _runtime_px_per_cmd_mm_for_polar,
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
    """Measure signed command-space displacement without moving motors or saving."""

    validate_visual_calibration_dataset(calibration)
    capture_count = normalize_capture_count(capture_count)
    if camera_pair is None:
        camera_pair = default_camera_pair()

    reference_cam0 = np.asarray(calibration["reference_cam0"].values)
    reference_cam1 = np.asarray(calibration["reference_cam1"].values)
    current_position = _position_values(
        get_positions((*COMMAND_AXES, "p")),
        len(COMMAND_AXES) + 1,
        "current x/y/z/p readback",
    )
    current_commanded_position_mm = current_position[: len(COMMAND_AXES)].copy()
    current_polar_deg = float(current_position[-1])
    jacobian, polar_attrs = _runtime_px_per_cmd_mm_for_polar(
        calibration,
        current_polar_deg,
    )
    logger.info("Detecting shift without motor correction.")
    measurement_shift_kwargs = _polar_ecc_seed_shift_kwargs(
        shift_kwargs,
        calibration=calibration,
        jacobian=jacobian,
        polar_attrs=polar_attrs,
        commanded_position_mm=current_commanded_position_mm,
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
        "Detected command offset mm=%s, weighted_residual_px=%.6g",
        estimated_offset_mm.tolist(),
        weighted_residual_px,
    )
    return (
        measurement.assign(
            {
                "estimated_command_offset_mm": (
                    ("command_axis",),
                    estimated_offset_mm,
                    {"units": "commanded-mm"},
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
        .assign_attrs(_prefixed_polar_attrs("detection", polar_attrs))
    )

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
    compute_lqr_correction_design,
    derive_axis_scale_from_jacobian,
    estimate_command_offset,
    lqr_projected_observation_residual_from_design,
    validate_visual_calibration_dataset,
    weighted_pixel_residual,
)
from merlin_track_position.tracking.correct import (
    ANALYZER_AXES,
    BEAM_AXES,
    BEAM_CORRECTION_CRITERION,
    BEAM_OBSERVATION_AXES,
    CORRECTION_MODE_BEAM,
    _capture_measurement,
    _analyzer_offset_from_estimated_offset,
    _beam_analyzer_observation_from_offsets,
    _beam_geometry_from_calibration,
    _beam_offset_from_estimated_offset,
    _orientation_ecc_seed_shift_kwargs,
    _position_values,
    _positive_um_to_mm,
    _prefixed_polar_attrs,
    _prefixed_orientation_attrs,
    _resolve_correction_mode,
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
    correction_mode: str = constants.DEFAULT_CORRECTION_MODE,
    lqr_motor_penalty: float = constants.DEFAULT_LQR_CORRECTION_MOTOR_PENALTY,
    lqr_svd_relative_tolerance: float = (
        constants.DEFAULT_LQR_CORRECTION_SVD_RELATIVE_TOLERANCE
    ),
    beam_xz_angle_from_analyzer_deg: float = (
        constants.DEFAULT_BEAM_XZ_ANGLE_FROM_ANALYZER_DEG
    ),
    beam_transverse_tolerance_um: float = (
        constants.DEFAULT_BEAM_TRANSVERSE_TOLERANCE_UM
    ),
    beam_analyzer_transverse_tolerance_um: float = (
        constants.DEFAULT_BEAM_ANALYZER_TRANSVERSE_TOLERANCE_UM
    ),
    beam_vertical_tolerance_um: float = constants.DEFAULT_BEAM_VERTICAL_TOLERANCE_UM,
    **shift_kwargs: Any,
) -> xr.Dataset:
    """Measure signed readback-space displacement without moving motors or saving."""

    validate_visual_calibration_dataset(calibration)
    capture_count = normalize_capture_count(capture_count)
    correction_mode = _resolve_correction_mode(correction_mode)
    if camera_pair is None:
        camera_pair = default_camera_pair()

    reference_cam0 = np.asarray(calibration["reference_cam0"].values)
    reference_cam1 = np.asarray(calibration["reference_cam1"].values)
    current_orientation = _read_current_detection_orientation()
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
        orientation_attrs=orientation_attrs,
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
    data_vars: dict[str, Any] = {
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
    coords: dict[str, Any] = {"command_axis": list(COMMAND_AXES)}
    attrs: dict[str, Any] = {
        "detection_correction_mode": correction_mode,
    }

    if correction_mode == CORRECTION_MODE_BEAM:
        beam_metrics = _beam_detection_metrics(
            calibration,
            jacobian=jacobian,
            estimated_offset_mm=estimated_offset_mm,
            current_polar_deg=float(polar_attrs["current_polar_deg"]),
            polar_rotation_applied=bool(polar_attrs["polar_rotation_applied"]),
            lqr_motor_penalty=lqr_motor_penalty,
            lqr_svd_relative_tolerance=lqr_svd_relative_tolerance,
            beam_xz_angle_from_analyzer_deg=beam_xz_angle_from_analyzer_deg,
            beam_transverse_tolerance_um=beam_transverse_tolerance_um,
            beam_analyzer_transverse_tolerance_um=beam_analyzer_transverse_tolerance_um,
            beam_vertical_tolerance_um=beam_vertical_tolerance_um,
        )
        data_vars.update(beam_metrics["data_vars"])
        coords.update(beam_metrics["coords"])
        attrs.update(beam_metrics["attrs"])

    return (
        measurement.assign(data_vars)
        .assign_coords(coords)
        .assign_attrs(
            attrs
            | _prefixed_polar_attrs("detection", polar_attrs)
            | _prefixed_orientation_attrs("detection", orientation_attrs)
        )
    )


def _beam_detection_metrics(
    calibration: xr.Dataset,
    *,
    jacobian: np.ndarray,
    estimated_offset_mm: np.ndarray,
    current_polar_deg: float,
    polar_rotation_applied: bool,
    lqr_motor_penalty: float,
    lqr_svd_relative_tolerance: float,
    beam_xz_angle_from_analyzer_deg: float,
    beam_transverse_tolerance_um: float,
    beam_analyzer_transverse_tolerance_um: float,
    beam_vertical_tolerance_um: float,
) -> dict[str, dict[str, Any]]:
    axis_scale = np.asarray(
        calibration["axis_scale_readback_mm"].values,
        dtype=np.float64,
    )
    if polar_rotation_applied:
        axis_scale, *_ = derive_axis_scale_from_jacobian(
            jacobian,
            np.asarray(calibration["probe_readback_delta_mm"].values, dtype=np.float64),
        )
    beam_geometry = _beam_geometry_from_calibration(
        calibration,
        beam_xz_angle_from_analyzer_deg=beam_xz_angle_from_analyzer_deg,
        polar_deg=current_polar_deg,
    )
    beam_transverse_tolerance_mm = _positive_um_to_mm(
        beam_transverse_tolerance_um,
        "beam_transverse_tolerance_um",
    )
    beam_analyzer_transverse_tolerance_mm = _positive_um_to_mm(
        beam_analyzer_transverse_tolerance_um,
        "beam_analyzer_transverse_tolerance_um",
    )
    beam_vertical_tolerance_mm = _positive_um_to_mm(
        beam_vertical_tolerance_um,
        "beam_vertical_tolerance_um",
    )
    beam_lqr_weights = np.asarray(
        [
            1.0 / (beam_transverse_tolerance_mm**2),
            1.0 / (beam_analyzer_transverse_tolerance_mm**2),
            1.0 / (beam_vertical_tolerance_mm**2),
        ],
        dtype=np.float64,
    )
    lqr_design = compute_lqr_correction_design(
        np.asarray(beam_geometry["projection_matrix"], dtype=np.float64),
        axis_scale,
        image_scale_px=1.0,
        motor_penalty=lqr_motor_penalty,
        svd_relative_tolerance=lqr_svd_relative_tolerance,
        weights=beam_lqr_weights,
    )
    beam_offset_mm = _beam_offset_from_estimated_offset(
        estimated_offset_mm,
        beam_geometry,
    )
    analyzer_offset_mm = _analyzer_offset_from_estimated_offset(
        estimated_offset_mm,
        beam_geometry,
    )
    beam_observation_mm = _beam_analyzer_observation_from_offsets(
        beam_offset_mm,
        analyzer_offset_mm,
    )
    if (
        beam_offset_mm is None
        or analyzer_offset_mm is None
        or beam_observation_mm is None
    ):
        raise RuntimeError("beam detection observation was not initialized")
    criterion_residual = lqr_projected_observation_residual_from_design(
        lqr_design,
        beam_observation_mm,
    )
    return {
        "data_vars": {
            "beam_offset_um": (
                ("beam_axis",),
                1000.0 * beam_offset_mm,
                {"units": "um"},
            ),
            "analyzer_offset_um": (
                ("analyzer_axis",),
                1000.0 * analyzer_offset_mm,
                {"units": "um"},
            ),
            "beam_analyzer_observation_um": (
                ("beam_observation_axis",),
                1000.0 * beam_observation_mm,
                {"units": "um"},
            ),
            "detection_correction_criterion_residual": (
                (),
                float(criterion_residual),
            ),
        },
        "coords": {
            "beam_axis": list(BEAM_AXES),
            "analyzer_axis": list(ANALYZER_AXES),
            "beam_observation_axis": list(BEAM_OBSERVATION_AXES),
        },
        "attrs": {
            "detection_correction_criterion": BEAM_CORRECTION_CRITERION,
        },
    }


def _read_current_detection_orientation() -> dict[str, float]:
    try:
        values = _position_values(
            get_positions(("p", "t", "a")),
            3,
            "current p/t/a readback",
        )
        return {
            "polar": float(values[0]),
            "tilt": float(values[1]),
            "azi": float(values[2]),
        }
    except Exception as exc:
        logger.info(
            "Could not read current p/t/a for detection (%s); "
            "falling back to polar only.",
            exc,
        )
    values = _position_values(
        get_positions(("p",)),
        1,
        "current polar readback",
    )
    return {
        "polar": float(values[0]),
        "tilt": np.nan,
        "azi": np.nan,
    }

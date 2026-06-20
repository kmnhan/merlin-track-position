from __future__ import annotations

import logging
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, NamedTuple, Protocol

import h5py
import numpy as np
import xarray as xr

from merlin_track_position import constants
from merlin_track_position.instruments.cameras import (
    CameraPairPlugin,
    capture_image_stack,
    default_camera_pair,
    normalize_capture_count,
)
from merlin_track_position.instruments.motors import get_positions, move_motors_and_wait
from merlin_track_position.tracking.calibration_core import (
    CAMERAS,
    CAPTURE_AGGREGATION_MEDIAN_SHIFTS,
    COMMAND_AXES,
    PIXEL_AXES,
    compute_lqr_correction_design,
    derive_axis_scale_from_jacobian,
    estimate_command_offset,
    initialize_lqr_kalman_state,
    load_calibration_dataset,
    lqr_projected_observation_residual_from_design,
    lqr_projected_residual_from_design,
    measure_image_error,
    predict_lqr_kalman_state,
    solve_lqr_command_correction,
    solve_lqr_observation_command_correction,
    solve_lqr_state_command_correction,
    update_lqr_kalman_state,
    validate_visual_calibration_dataset,
    weighted_pixel_residual,
)
from merlin_track_position.tracking.persistence import (
    PendingEntry,
    PersistenceResult,
    discard_spool_entry,
    hdf5_image_encoding,
    iter_pending_entries,
    load_spooled_dataset,
    normalize_target_path,
    persistence_result_attrs,
    stage_dataset,
)
from merlin_track_position.tracking.roi import (
    beam_target_point_from_attrs_or_default,
    matching_reference_and_stack,
    roi_local_point_from_full_frame,
)

logger = logging.getLogger("merlin_track_position.tracking.correct")

CORRECTION_CRITERION = "lqr_projected_normalized_error"
BEAM_CORRECTION_CRITERION = "beam_analyzer_projected_normalized_error"
CORRECTION_MODE_CAMERA = "camera"
CORRECTION_MODE_BEAM = "beam"
CORRECTION_MODES = (CORRECTION_MODE_CAMERA, CORRECTION_MODE_BEAM)
BEAM_AXES = ("beam_transverse", "beam_vertical", "beam_longitudinal")
ANALYZER_AXES = (
    "analyzer_transverse",
    "analyzer_vertical",
    "analyzer_longitudinal",
)
BEAM_OBSERVATION_AXES = ("beam_transverse", "analyzer_transverse", "vertical")
ORIENTATION_READBACK_AXES = ("p", "t", "a")
_USE_DEFAULT = object()
_CORRECTION_HISTORY_FORMAT = "merlin_track_position_correction_history"
_CORRECTION_HISTORY_RESIZABLE_DIMS = ("move", "iteration")
_ORIENTATION_ECC_MAX_BASIS_CONDITION = 1.0e6
_SAMPLE_PLANE_BASIS = np.asarray(
    [
        [1.0, 0.0],
        [0.0, 1.0],
        [0.0, 0.0],
    ],
    dtype=np.float64,
)


class CorrectionFeedback(NamedTuple):
    predicted_weighted_response_px: float
    measured_weighted_response_px: float
    alpha: float
    parallel_px: float


class CorrectionMotorBackend(Protocol):
    def get_positions(self, motor_aliases: Sequence[str]) -> tuple[float, ...]:
        """Return motor readback positions in the order requested."""

    def move_motors_and_wait(
        self,
        motor_aliases: Sequence[str],
        goals: Sequence[float],
        *,
        backlash_correction: dict[str, float] | None = None,
        move_timeout_s: float = 60.0,
    ) -> tuple[float, ...]:
        """Move motors and return final positions in the order requested."""


class DirectBCSMotorBackend:
    def get_positions(self, motor_aliases: Sequence[str]) -> tuple[float, ...]:
        return get_positions(motor_aliases)

    def move_motors_and_wait(
        self,
        motor_aliases: Sequence[str],
        goals: Sequence[float],
        *,
        backlash_correction: dict[str, float] | None = None,
        move_timeout_s: float = 60.0,
    ) -> tuple[float, ...]:
        return move_motors_and_wait(
            motor_aliases,
            goals,
            backlash_correction=backlash_correction,
            move_timeout_s=move_timeout_s,
        )


def do_correction(
    calibration: xr.Dataset | str | Path,
    camera_pair: CameraPairPlugin | None = None,
    *,
    calibration_path: str | Path | None = None,
    capture_count: int = constants.DEFAULT_CORRECTION_CAPTURE_COUNT,
    capture_aggregation: str = CAPTURE_AGGREGATION_MEDIAN_SHIFTS,
    lqr_projected_tolerance: Any = _USE_DEFAULT,
    gain: Any = _USE_DEFAULT,
    max_normalized_step: Any = _USE_DEFAULT,
    min_command_norm_mm: float = constants.DEFAULT_CORRECTION_MIN_COMMAND_NORM_MM,
    max_moves: int = constants.DEFAULT_CORRECTION_MAX_MOVES,
    lqr_image_scale_px: float = constants.DEFAULT_LQR_CORRECTION_IMAGE_SCALE_PX,
    lqr_motor_penalty: float = constants.DEFAULT_LQR_CORRECTION_MOTOR_PENALTY,
    lqr_svd_relative_tolerance: float = (
        constants.DEFAULT_LQR_CORRECTION_SVD_RELATIVE_TOLERANCE
    ),
    correction_mode: str = constants.DEFAULT_CORRECTION_MODE,
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
    weights: Sequence[float] | np.ndarray | None = None,
    progress_callback: Callable[[xr.Dataset], None] | None = None,
    motor_backend: CorrectionMotorBackend | None = None,
    active_command_axes: Sequence[str] | None = None,
    **shift_kwargs: Any,
) -> xr.Dataset:
    """Run LQR closed-loop visual-servo correction in readback-mm space."""

    calibration, resolved_path = _resolve_calibration_and_path(
        calibration,
        calibration_path,
    )
    logger.info("Correction setup: calibration_path=%s", resolved_path)
    validate_visual_calibration_dataset(calibration)
    capture_count = normalize_capture_count(capture_count)
    if camera_pair is None:
        camera_pair = default_camera_pair()
    if motor_backend is None:
        motor_backend = DirectBCSMotorBackend()
    active_axis_indices = _active_command_axis_indices(active_command_axes)
    active_axis_names = tuple(COMMAND_AXES[index] for index in active_axis_indices)
    active_axis_reduced = len(active_axis_indices) != len(COMMAND_AXES)
    correction_backlash_enabled = (
        isinstance(motor_backend, DirectBCSMotorBackend)
        and constants.CORRECTION_USE_BCS_API_BACKLASH
    )
    correction_backlash = (
        dict(constants.MOTOR_BACKLASH_CORRECTION) if correction_backlash_enabled else {}
    )
    weights = _correction_weights(weights)
    if gain is _USE_DEFAULT:
        gain = constants.DEFAULT_LQR_CORRECTION_GAIN
    if max_normalized_step is _USE_DEFAULT:
        max_normalized_step = constants.DEFAULT_LQR_CORRECTION_MAX_NORMALIZED_STEP
    lqr_projected_tolerance = _resolve_lqr_projected_tolerance(lqr_projected_tolerance)
    correction_mode = _resolve_correction_mode(correction_mode)
    correction_criterion = (
        BEAM_CORRECTION_CRITERION
        if correction_mode == CORRECTION_MODE_BEAM
        else CORRECTION_CRITERION
    )
    correction_tolerance = lqr_projected_tolerance
    current_gain = float(gain)
    min_command_norm_mm = float(min_command_norm_mm)
    max_moves = int(max_moves)
    if not np.isfinite(current_gain) or current_gain <= 0.0:
        raise ValueError("gain must be finite and positive")
    if max_normalized_step is not None:
        max_normalized_step = float(max_normalized_step)
        if np.isnan(max_normalized_step) or max_normalized_step <= 0.0:
            raise ValueError("max_normalized_step must be positive or None")
    if not np.isfinite(min_command_norm_mm) or min_command_norm_mm < 0.0:
        raise ValueError("min_command_norm_mm must be finite and non-negative")
    if max_moves < 0:
        raise ValueError("max_moves must be >= 0")
    logger.info("Reading initial readback x/y/z and orientation positions.")
    readback_position_mm, current_orientation_deg = (
        _read_initial_readback_position_and_orientation_deg(motor_backend)
    )
    logger.info("Initial readback positions: %s", readback_position_mm.tolist())
    jacobian, polar_attrs = _runtime_px_per_readback_mm_for_polar(
        calibration,
        float(current_orientation_deg["polar"]),
    )
    orientation_attrs = _runtime_orientation_attrs(
        calibration,
        polar_attrs=polar_attrs,
        current_tilt_deg=current_orientation_deg["tilt"],
        current_azi_deg=current_orientation_deg["azi"],
    )
    logger.info(
        "Correction polar geometry: calibration=%g deg, current=%g deg, "
        "delta=%g deg, deadband=%g deg, rotation_applied=%s",
        polar_attrs["calibration_polar_deg"],
        polar_attrs["current_polar_deg"],
        polar_attrs["polar_delta_deg"],
        polar_attrs["polar_deadband_deg"],
        polar_attrs["polar_rotation_applied"],
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
    lqr_kalman_filter_enabled = (
        bool(constants.DEFAULT_LQR_CORRECTION_USE_KALMAN_FILTER)
        and correction_mode == CORRECTION_MODE_CAMERA
        and not active_axis_reduced
    )
    lqr_kalman_process_noise = constants.DEFAULT_LQR_CORRECTION_KALMAN_PROCESS_NOISE
    lqr_kalman_measurement_noise = (
        constants.DEFAULT_LQR_CORRECTION_KALMAN_MEASUREMENT_NOISE
    )
    lqr_kalman_measurement_covariance = (
        constants.DEFAULT_LQR_CORRECTION_KALMAN_MEASUREMENT_COVARIANCE
    )
    lqr_kalman_initial_covariance = (
        constants.DEFAULT_LQR_CORRECTION_KALMAN_INITIAL_COVARIANCE
    )
    lqr_kalman_innovation_gate = constants.DEFAULT_LQR_CORRECTION_KALMAN_INNOVATION_GATE
    reference_cam0 = np.asarray(calibration["reference_cam0"].values)
    reference_cam1 = np.asarray(calibration["reference_cam1"].values)
    axis_scale = np.asarray(
        calibration["axis_scale_readback_mm"].values,
        dtype=np.float64,
    )
    if polar_attrs["polar_rotation_applied"]:
        axis_scale, *_ = derive_axis_scale_from_jacobian(
            jacobian,
            np.asarray(calibration["probe_readback_delta_mm"].values, dtype=np.float64),
        )
    active_axis_scale = axis_scale[list(active_axis_indices)]
    beam_geometry: dict[str, np.ndarray | float] | None = None
    beam_lqr_weights: np.ndarray | None = None
    lqr_image_scale: float | np.ndarray = lqr_image_scale_px
    lqr_weights = weights
    if correction_mode == CORRECTION_MODE_BEAM:
        beam_geometry = _beam_geometry_from_calibration(
            calibration,
            beam_xz_angle_from_analyzer_deg=beam_xz_angle_from_analyzer_deg,
            polar_deg=float(polar_attrs["current_polar_deg"]),
        )
        beam_lqr_weights = np.asarray(
            [
                1.0 / (beam_transverse_tolerance_mm**2),
                1.0 / (beam_analyzer_transverse_tolerance_mm**2),
                1.0 / (beam_vertical_tolerance_mm**2),
            ],
            dtype=np.float64,
        )
        lqr_observation_model = np.asarray(
            beam_geometry["projection_matrix"],
            dtype=np.float64,
        )[:, list(active_axis_indices)]
        lqr_image_scale = 1.0
        lqr_weights = beam_lqr_weights
    else:
        lqr_observation_model = jacobian.reshape(
            len(CAMERAS) * len(PIXEL_AXES),
            len(COMMAND_AXES),
        )[:, list(active_axis_indices)]
    lqr_design = compute_lqr_correction_design(
        lqr_observation_model,
        active_axis_scale,
        image_scale_px=lqr_image_scale,
        motor_penalty=lqr_motor_penalty,
        svd_relative_tolerance=lqr_svd_relative_tolerance,
        weights=lqr_weights,
    )
    logger.info(
        "Correction parameters: mode=%s, capture_count=%d, criterion=%s, "
        "tolerance=%g, gain=%g, max_moves=%d",
        correction_mode,
        capture_count,
        correction_criterion,
        correction_tolerance,
        current_gain,
        max_moves,
    )
    initial_readback_position_mm = readback_position_mm.copy()
    correction_log_path = correction_history_path(resolved_path)
    correction_run_id = _next_correction_history_run_id(correction_log_path)
    correction_started_at_utc = datetime.now(UTC).isoformat()
    correction_move_started_at: str | None = None
    correction_move_finished_at: str | None = None

    logger.info("Capturing initial correction measurement.")
    initial_shift_kwargs = _orientation_ecc_seed_shift_kwargs(
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
        **initial_shift_kwargs,
    )
    weighted_residual = weighted_pixel_residual(measurement, weights=weights)
    estimated_offset_mm = estimate_command_offset(
        jacobian,
        measurement,
        weights=weights,
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
    criterion_residual = _correction_criterion_residual(
        correction_mode=correction_mode,
        lqr_design=lqr_design,
        measurement=measurement,
        beam_observation_mm=beam_observation_mm,
    )
    logger.info(
        "Initial correction criterion residual: %.6g (%s); weighted residual: %.6g px",
        criterion_residual,
        correction_criterion,
        weighted_residual,
    )

    iteration_shift_px = [np.asarray(measurement["shift_px"].values, dtype=np.float64)]
    iteration_weighted_residuals = [weighted_residual]
    iteration_criterion_residuals = [criterion_residual]
    iteration_beam_offset_mm: list[np.ndarray] = []
    if beam_offset_mm is not None:
        iteration_beam_offset_mm.append(beam_offset_mm)
    iteration_analyzer_offset_mm: list[np.ndarray] = []
    if analyzer_offset_mm is not None:
        iteration_analyzer_offset_mm.append(analyzer_offset_mm)
    move_requested_position_mm: list[np.ndarray] = []
    move_final_readback_position_mm: list[np.ndarray] = []
    move_gain: list[float] = []
    move_pre_weighted_residuals: list[float] = []
    move_post_weighted_residuals: list[float] = []
    move_predicted_delta_px: list[np.ndarray] = []
    move_measured_delta_px: list[np.ndarray] = []
    move_model_residual_delta_px: list[np.ndarray] = []
    move_predicted_weighted_response_px: list[float] = []
    move_measured_weighted_response_px: list[float] = []
    move_feedback_alpha: list[float] = []
    move_feedback_parallel_px: list[float] = []
    move_feedback_valid: list[bool] = []
    move_max_normalized_component: list[float] = []
    move_active_axis_mask: list[np.ndarray] = []
    iteration_lqr_kalman_state: list[np.ndarray] = []
    iteration_lqr_kalman_predicted_state: list[np.ndarray] = []
    iteration_lqr_kalman_innovation: list[np.ndarray] = []
    iteration_lqr_kalman_innovation_mahalanobis: list[float] = []
    iteration_lqr_kalman_measurement_accepted: list[bool] = []
    warnings: list[str] = [
        line.strip()
        for line in str(measurement.attrs.get("warnings", "")).splitlines()
        if line.strip()
    ]

    lqr_kalman_state: dict[str, np.ndarray | float | bool] | None = None
    if lqr_kalman_filter_enabled:
        if lqr_design is None:
            raise RuntimeError("LQR Kalman design was not initialized")
        lqr_kalman_state = initialize_lqr_kalman_state(
            measurement,
            lqr_design,
            initial_covariance=lqr_kalman_initial_covariance,
        )
        _append_lqr_kalman_diagnostics(
            lqr_kalman_state,
            iteration_lqr_kalman_state=iteration_lqr_kalman_state,
            iteration_lqr_kalman_predicted_state=(iteration_lqr_kalman_predicted_state),
            iteration_lqr_kalman_innovation=iteration_lqr_kalman_innovation,
            iteration_lqr_kalman_innovation_mahalanobis=(
                iteration_lqr_kalman_innovation_mahalanobis
            ),
            iteration_lqr_kalman_measurement_accepted=(
                iteration_lqr_kalman_measurement_accepted
            ),
        )

    move_count = 0
    converged = criterion_residual <= correction_tolerance
    stop_reason: str | None = None

    def build_result(completed: bool) -> xr.Dataset:
        return _build_correction_result(
            measurement=measurement,
            jacobian=jacobian,
            axis_scale=axis_scale,
            estimated_offset=estimated_offset_mm,
            next_correction=_reported_next_correction(
                converged=converged,
                jacobian=jacobian,
                measurement=measurement,
                axis_scale=axis_scale,
                active_axis_indices=active_axis_indices,
                correction_mode=correction_mode,
                beam_observation=beam_observation_mm,
                beam_projection_matrix=(
                    None
                    if beam_geometry is None
                    else np.asarray(
                        beam_geometry["projection_matrix"], dtype=np.float64
                    )
                ),
                beam_lqr_weights=beam_lqr_weights,
                gain=current_gain,
                max_normalized_step=max_normalized_step,
                lqr_image_scale_px=lqr_image_scale_px,
                lqr_motor_penalty=lqr_motor_penalty,
                lqr_svd_relative_tolerance=lqr_svd_relative_tolerance,
                lqr_kalman_filter_enabled=lqr_kalman_filter_enabled,
                lqr_design=lqr_design,
                lqr_kalman_state=lqr_kalman_state,
                weights=weights,
            ),
            iteration_shift_px=iteration_shift_px,
            iteration_weighted_residuals=iteration_weighted_residuals,
            iteration_criterion_residuals=iteration_criterion_residuals,
            beam_offset_mm=beam_offset_mm,
            iteration_beam_offset_mm=iteration_beam_offset_mm,
            analyzer_offset_mm=analyzer_offset_mm,
            iteration_analyzer_offset_mm=iteration_analyzer_offset_mm,
            iteration_lqr_kalman_state=iteration_lqr_kalman_state,
            iteration_lqr_kalman_predicted_state=(iteration_lqr_kalman_predicted_state),
            iteration_lqr_kalman_innovation=iteration_lqr_kalman_innovation,
            iteration_lqr_kalman_innovation_mahalanobis=(
                iteration_lqr_kalman_innovation_mahalanobis
            ),
            iteration_lqr_kalman_measurement_accepted=(
                iteration_lqr_kalman_measurement_accepted
            ),
            move_requested_position_mm=move_requested_position_mm,
            move_final_readback_position_mm=move_final_readback_position_mm,
            move_gain=move_gain,
            move_pre_weighted_residuals=move_pre_weighted_residuals,
            move_post_weighted_residuals=move_post_weighted_residuals,
            move_predicted_delta_px=move_predicted_delta_px,
            move_measured_delta_px=move_measured_delta_px,
            move_model_residual_delta_px=move_model_residual_delta_px,
            move_predicted_weighted_response_px=(move_predicted_weighted_response_px),
            move_measured_weighted_response_px=move_measured_weighted_response_px,
            move_feedback_alpha=move_feedback_alpha,
            move_feedback_parallel_px=move_feedback_parallel_px,
            move_feedback_valid=move_feedback_valid,
            move_max_normalized_component=move_max_normalized_component,
            move_active_axis_mask=move_active_axis_mask,
            calibration_path=resolved_path,
            correction_history_path=correction_log_path,
            correction_run_id=correction_run_id,
            correction_started_at_utc=correction_started_at_utc,
            correction_move_started_at=correction_move_started_at,
            correction_move_finished_at=correction_move_finished_at,
            correction_history_completed=completed,
            converged=converged,
            move_count=move_count,
            lqr_projected_tolerance=lqr_projected_tolerance,
            correction_tolerance=correction_tolerance,
            correction_criterion=correction_criterion,
            correction_mode=correction_mode,
            beam_xz_angle_from_analyzer_deg=beam_xz_angle_from_analyzer_deg,
            polar_attrs=polar_attrs,
            orientation_attrs=orientation_attrs,
            beam_polar_deg=(
                None if beam_geometry is None else float(beam_geometry["polar_deg"])
            ),
            beam_runtime_xz_angle_deg=(
                None
                if beam_geometry is None
                else float(beam_geometry["beam_xz_angle_deg"])
            ),
            analyzer_runtime_xz_angle_deg=(
                None
                if beam_geometry is None
                else float(beam_geometry["analyzer_xz_angle_deg"])
            ),
            beam_transverse_tolerance_um=beam_transverse_tolerance_um,
            beam_analyzer_transverse_tolerance_um=(
                beam_analyzer_transverse_tolerance_um
            ),
            beam_vertical_tolerance_um=beam_vertical_tolerance_um,
            gain=gain,
            current_gain=current_gain,
            max_normalized_step=max_normalized_step,
            min_command_norm_mm=min_command_norm_mm,
            max_moves=max_moves,
            lqr_image_scale_px=lqr_image_scale_px,
            lqr_motor_penalty=lqr_motor_penalty,
            lqr_svd_relative_tolerance=lqr_svd_relative_tolerance,
            lqr_kalman_filter_enabled=lqr_kalman_filter_enabled,
            lqr_kalman_process_noise=lqr_kalman_process_noise,
            lqr_kalman_measurement_noise=lqr_kalman_measurement_noise,
            lqr_kalman_measurement_covariance=lqr_kalman_measurement_covariance,
            lqr_kalman_initial_covariance=lqr_kalman_initial_covariance,
            lqr_kalman_innovation_gate=lqr_kalman_innovation_gate,
            correction_backlash_enabled=correction_backlash_enabled,
            initial_readback_position_mm=initial_readback_position_mm,
            readback_position_mm=readback_position_mm,
            warnings=warnings,
            active_axis_names=active_axis_names,
        )

    def save_progress(completed: bool) -> xr.Dataset:
        logger.info(
            "Saving correction progress: completed=%s, run_id=%d, path=%s",
            completed,
            correction_run_id,
            correction_log_path,
        )
        progress = build_result(completed)
        progress = _apply_calibration_persistence_attrs(progress, calibration)
        persistence = save_correction_history_dataset_deferred(
            progress,
            correction_log_path,
            run_id=correction_run_id,
        )
        progress = progress.assign_attrs(
            persistence_result_attrs("correction_history", persistence)
        )
        if progress_callback is not None and not completed:
            progress_callback(progress)
        logger.info("Saved correction progress: completed=%s", completed)
        return progress

    # Publish the initial measurement and planned correction before any motor move.
    save_progress(completed=False)

    while not converged and move_count < max_moves:
        logger.info(
            "Correction iteration %d starting: criterion_residual=%.6g, "
            "weighted_residual=%.6g px, gain=%g",
            move_count + 1,
            criterion_residual,
            weighted_residual,
            current_gain,
        )
        gain_used = float(current_gain)
        jacobian_before = jacobian.copy()
        estimated_offset_mm = estimate_command_offset(
            jacobian,
            measurement,
            weights=weights,
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
        if correction_mode == CORRECTION_MODE_BEAM:
            if beam_geometry is None or beam_observation_mm is None:
                raise RuntimeError("beam correction geometry was not initialized")
            raw_correction_readback_delta_mm = _expand_active_command_correction(
                solve_lqr_observation_command_correction(
                    np.asarray(
                        beam_geometry["projection_matrix"],
                        dtype=np.float64,
                    )[:, list(active_axis_indices)],
                    beam_observation_mm,
                    active_axis_scale,
                    gain=gain_used,
                    image_scale=1.0,
                    motor_penalty=lqr_motor_penalty,
                    svd_relative_tolerance=lqr_svd_relative_tolerance,
                    max_normalized_step=max_normalized_step,
                    weights=beam_lqr_weights,
                ),
                active_axis_indices,
            )
        elif lqr_kalman_filter_enabled:
            if lqr_design is None or lqr_kalman_state is None:
                raise RuntimeError("LQR Kalman state was not initialized")
            raw_correction_readback_delta_mm = solve_lqr_state_command_correction(
                lqr_design,
                np.asarray(lqr_kalman_state["state"], dtype=np.float64),
                gain=gain_used,
                max_normalized_step=max_normalized_step,
            )
        else:
            if active_axis_reduced:
                raw_correction_readback_delta_mm = _expand_active_command_correction(
                    solve_lqr_observation_command_correction(
                        jacobian.reshape(
                            len(CAMERAS) * len(PIXEL_AXES),
                            len(COMMAND_AXES),
                        )[:, list(active_axis_indices)],
                        _measurement_observation(measurement),
                        active_axis_scale,
                        gain=gain_used,
                        max_normalized_step=max_normalized_step,
                        image_scale=lqr_image_scale_px,
                        motor_penalty=lqr_motor_penalty,
                        svd_relative_tolerance=lqr_svd_relative_tolerance,
                        weights=weights,
                    ),
                    active_axis_indices,
                )
            else:
                raw_correction_readback_delta_mm = solve_lqr_command_correction(
                    jacobian,
                    measurement,
                    axis_scale,
                    gain=gain_used,
                    max_normalized_step=max_normalized_step,
                    image_scale_px=lqr_image_scale_px,
                    motor_penalty=lqr_motor_penalty,
                    svd_relative_tolerance=lqr_svd_relative_tolerance,
                    weights=weights,
                )
        correction_readback_delta_mm = _validate_readback_correction(
            raw_correction_readback_delta_mm
        )
        correction_norm_mm = float(np.linalg.norm(correction_readback_delta_mm))
        active_indices = _active_correction_indices(correction_readback_delta_mm)
        logger.info(
            "Computed correction readback delta: estimated_offset_mm=%s, raw_delta_mm=%s, "
            "delta_mm=%s, norm_mm=%.6g, active_axes=%s",
            estimated_offset_mm.tolist(),
            raw_correction_readback_delta_mm.tolist(),
            correction_readback_delta_mm.tolist(),
            correction_norm_mm,
            tuple(COMMAND_AXES[index] for index in active_indices),
        )
        if correction_norm_mm <= min_command_norm_mm or not active_indices:
            warnings.append(
                _correction_stop_warning(
                    raw_correction_readback_mm=raw_correction_readback_delta_mm,
                    min_command_norm_mm=min_command_norm_mm,
                )
            )
            stop_reason = warnings[-1]
            logger.info(
                "Stopping correction before move: %s",
                warnings[-1],
            )
            break

        pre_move_readback_mm = readback_position_mm.copy()
        normalized_component = correction_readback_delta_mm / axis_scale
        requested_position_mm = readback_position_mm + correction_readback_delta_mm
        active_axes = tuple(COMMAND_AXES[index] for index in active_indices)
        active_requested_position_mm = tuple(
            float(requested_position_mm[index]) for index in active_indices
        )
        logger.info(
            "Moving correction axes: active_axes=%s, requested_position_mm=%s",
            active_axes,
            active_requested_position_mm,
        )
        if correction_move_started_at is None:
            correction_move_started_at = _local_timestamp_iso()
        active_final_readback_mm = np.asarray(
            motor_backend.move_motors_and_wait(
                active_axes,
                active_requested_position_mm,
                backlash_correction=correction_backlash,
            ),
            dtype=np.float64,
        )
        if active_final_readback_mm.shape == (len(COMMAND_AXES),):
            final_readback_mm = active_final_readback_mm.copy()
        elif active_final_readback_mm.shape == (len(active_axes),):
            final_readback_mm = pre_move_readback_mm.copy()
            final_readback_mm[list(active_indices)] = active_final_readback_mm
        else:
            raise ValueError("move result readback must match the moved axes")
        correction_move_finished_at = _local_timestamp_iso()
        logger.info(
            "Correction motor move returned final readbacks: active_axes=%s, "
            "active_final_readback_mm=%s",
            active_axes,
            active_final_readback_mm.tolist(),
        )
        logger.info("Final readback positions: %s", final_readback_mm.tolist())
        executed_readback_delta_mm = final_readback_mm - pre_move_readback_mm
        predicted_delta_px = (
            jacobian_before.reshape(len(CAMERAS) * len(PIXEL_AXES), len(COMMAND_AXES))
            @ executed_readback_delta_mm
        ).reshape(len(CAMERAS), len(PIXEL_AXES))
        readback_position_mm = final_readback_mm

        lqr_kalman_prediction: dict[str, np.ndarray] | None = None
        if lqr_kalman_filter_enabled:
            if lqr_design is None or lqr_kalman_state is None:
                raise RuntimeError("LQR Kalman state was not initialized")
            lqr_kalman_prediction = predict_lqr_kalman_state(
                np.asarray(lqr_kalman_state["state"], dtype=np.float64),
                np.asarray(lqr_kalman_state["covariance"], dtype=np.float64),
                executed_readback_delta_mm,
                lqr_design,
                process_noise=lqr_kalman_process_noise,
            )

        logger.info("Capturing post-move correction measurement.")
        after_shift_kwargs = _orientation_ecc_seed_shift_kwargs(
            shift_kwargs,
            calibration=calibration,
            orientation_attrs=orientation_attrs,
        )
        after_measurement = _capture_measurement(
            calibration,
            camera_pair,
            reference_cam0,
            reference_cam1,
            capture_count,
            capture_aggregation=capture_aggregation,
            **after_shift_kwargs,
        )
        after_weighted_residual = weighted_pixel_residual(
            after_measurement,
            weights=weights,
        )
        after_estimated_offset_mm = estimate_command_offset(
            jacobian,
            after_measurement,
            weights=weights,
        )
        after_beam_offset_mm = _beam_offset_from_estimated_offset(
            after_estimated_offset_mm,
            beam_geometry,
        )
        after_analyzer_offset_mm = _analyzer_offset_from_estimated_offset(
            after_estimated_offset_mm,
            beam_geometry,
        )
        after_beam_observation_mm = _beam_analyzer_observation_from_offsets(
            after_beam_offset_mm,
            after_analyzer_offset_mm,
        )
        after_criterion_residual = _correction_criterion_residual(
            correction_mode=correction_mode,
            lqr_design=lqr_design,
            measurement=after_measurement,
            beam_observation_mm=after_beam_observation_mm,
        )
        decreased = bool(after_criterion_residual < criterion_residual)
        logger.info(
            "Post-move criterion residual: %.6g; weighted residual: %.6g px; "
            "decreased=%s",
            after_criterion_residual,
            after_weighted_residual,
            decreased,
        )
        if lqr_kalman_filter_enabled:
            if lqr_design is None or lqr_kalman_prediction is None:
                raise RuntimeError("LQR Kalman prediction was not initialized")
            lqr_kalman_update = update_lqr_kalman_state(
                lqr_kalman_prediction["state"],
                lqr_kalman_prediction["covariance"],
                after_measurement,
                lqr_design,
                measurement_noise=lqr_kalman_measurement_noise,
                measurement_covariance=lqr_kalman_measurement_covariance,
                innovation_gate=lqr_kalman_innovation_gate,
            )
            lqr_kalman_state = lqr_kalman_update
            _append_lqr_kalman_diagnostics(
                lqr_kalman_update,
                iteration_lqr_kalman_state=iteration_lqr_kalman_state,
                iteration_lqr_kalman_predicted_state=(
                    iteration_lqr_kalman_predicted_state
                ),
                iteration_lqr_kalman_innovation=iteration_lqr_kalman_innovation,
                iteration_lqr_kalman_innovation_mahalanobis=(
                    iteration_lqr_kalman_innovation_mahalanobis
                ),
                iteration_lqr_kalman_measurement_accepted=(
                    iteration_lqr_kalman_measurement_accepted
                ),
            )
            if not bool(lqr_kalman_update["measurement_accepted"]):
                warnings.append(
                    _lqr_kalman_gated_measurement_warning(
                        innovation_mahalanobis=float(
                            lqr_kalman_update["innovation_mahalanobis"]
                        ),
                        innovation_gate=lqr_kalman_innovation_gate,
                    )
                )
                stop_reason = (
                    "Kalman innovation gating rejected the post-move measurement"
                )
                logger.info("Stopping correction after gated Kalman measurement.")
        measured_delta_px = np.asarray(
            after_measurement["shift_px"].values, dtype=np.float64
        ) - np.asarray(measurement["shift_px"].values, dtype=np.float64)
        model_residual_delta_px = measured_delta_px - predicted_delta_px
        feedback = _correction_feedback_metrics(
            predicted_delta_px,
            measured_delta_px,
            weights=weights,
        )
        feedback_valid = _correction_feedback_is_valid(feedback)
        logger.info(
            "Post-move feedback metrics: predicted_response=%.6g px, "
            "measured_response=%.6g px, alpha=%.6g, parallel=%.6g px, "
            "valid=%s",
            feedback.predicted_weighted_response_px,
            feedback.measured_weighted_response_px,
            feedback.alpha,
            feedback.parallel_px,
            feedback_valid,
        )
        if feedback_valid and decreased:
            logger.info("Residual decreased; keeping nominal Jacobian fixed.")
        elif feedback_valid:
            logger.info(
                "Residual did not decrease; keeping gain fixed at %g.", current_gain
            )

        move_requested_position_mm.append(requested_position_mm.copy())
        move_final_readback_position_mm.append(final_readback_mm)
        move_gain.append(gain_used)
        move_pre_weighted_residuals.append(float(weighted_residual))
        move_post_weighted_residuals.append(float(after_weighted_residual))
        move_predicted_delta_px.append(predicted_delta_px)
        move_measured_delta_px.append(measured_delta_px)
        move_model_residual_delta_px.append(model_residual_delta_px)
        move_predicted_weighted_response_px.append(
            feedback.predicted_weighted_response_px
        )
        move_measured_weighted_response_px.append(
            feedback.measured_weighted_response_px
        )
        move_feedback_alpha.append(feedback.alpha)
        move_feedback_parallel_px.append(feedback.parallel_px)
        move_feedback_valid.append(feedback_valid)
        move_max_normalized_component.append(
            float(np.max(np.abs(normalized_component)))
        )
        active_axis_mask = np.zeros(len(COMMAND_AXES), dtype=bool)
        active_axis_mask[list(active_indices)] = True
        move_active_axis_mask.append(active_axis_mask)

        measurement = after_measurement
        weighted_residual = after_weighted_residual
        estimated_offset_mm = after_estimated_offset_mm
        beam_offset_mm = after_beam_offset_mm
        analyzer_offset_mm = after_analyzer_offset_mm
        beam_observation_mm = after_beam_observation_mm
        criterion_residual = after_criterion_residual
        iteration_shift_px.append(
            np.asarray(measurement["shift_px"].values, dtype=np.float64)
        )
        iteration_weighted_residuals.append(float(weighted_residual))
        iteration_criterion_residuals.append(float(criterion_residual))
        if beam_offset_mm is not None:
            iteration_beam_offset_mm.append(beam_offset_mm)
        if analyzer_offset_mm is not None:
            iteration_analyzer_offset_mm.append(analyzer_offset_mm)
        warnings.extend(
            line.strip()
            for line in str(measurement.attrs.get("warnings", "")).splitlines()
            if line.strip()
        )
        move_count += 1
        converged = criterion_residual <= correction_tolerance
        logger.info(
            "Correction iteration %d complete: criterion_residual=%.6g, "
            "weighted_residual=%.6g px, converged=%s",
            move_count,
            criterion_residual,
            weighted_residual,
            converged,
        )
        save_progress(completed=False)
        if stop_reason is not None:
            break

    if not converged:
        if stop_reason is None:
            warnings.append(
                "correction did not converge within "
                f"{max_moves} move(s); final {correction_criterion} "
                f"{criterion_residual:.4g} exceeds {correction_tolerance:.4g}"
            )
        else:
            warnings.append(
                "correction did not converge; stopped before convergence: "
                f"{stop_reason}; final {correction_criterion} "
                f"{criterion_residual:.4g} exceeds {correction_tolerance:.4g}"
            )
        logger.info(
            "Correction finished without convergence: moves=%d, "
            "criterion_residual=%.6g",
            move_count,
            criterion_residual,
        )
    else:
        logger.info(
            "Correction converged: moves=%d, criterion_residual=%.6g",
            move_count,
            criterion_residual,
        )

    return save_progress(completed=True)


def _resolve_calibration_and_path(
    calibration: xr.Dataset | str | Path,
    calibration_path: str | Path | None,
) -> tuple[xr.Dataset, Path]:
    if isinstance(calibration, xr.Dataset):
        path_value = (
            calibration_path
            if calibration_path is not None
            else calibration.attrs.get("calibration_path")
        )
        if path_value is None or not str(path_value).strip():
            raise ValueError(
                "correction requires a calibration file path for correction history"
            )
        path = Path(path_value)
        if not path.exists():
            raise ValueError(f"calibration file does not exist: {path}")
        loaded = calibration.load()
        loaded.attrs["calibration_path"] = str(path)
        return loaded, path

    path = Path(calibration if calibration_path is None else calibration_path)
    if not path.exists():
        raise ValueError(f"calibration file does not exist: {path}")
    return load_calibration_dataset(path), path


def correction_history_path(calibration_path: str | Path) -> Path:
    """Return the sibling correction-history path for a calibration file."""

    path = Path(calibration_path)
    suffix = path.suffix if path.suffix else ".h5"
    return path.with_name(f"{path.stem}_corrections{suffix}")


def correction_timestamps_from_history(path: str | Path) -> xr.Dataset:
    """Extract run-level correction timestamps from a correction-history file."""

    timestamp_attrs = (
        "correction_started_at_utc",
        "correction_move_started_at",
        "correction_move_finished_at",
    )
    run_ids: list[int] = []
    run_groups: list[str] = []
    timestamp_values: dict[str, list[np.datetime64]] = {
        attr_name: [] for attr_name in timestamp_attrs
    }
    with h5py.File(path, "r") as history_file:
        run_names = sorted(
            (
                (run_id, group_name)
                for group_name in history_file
                if (run_id := _correction_history_run_id(group_name)) is not None
            ),
            key=lambda item: item[0],
        )
        for run_id, group_name in run_names:
            group = history_file[group_name]
            run_ids.append(run_id)
            run_groups.append(group_name)
            for attr_name in timestamp_attrs:
                timestamp_values[attr_name].append(
                    _correction_timestamp_attr_as_utc64(
                        group.attrs.get(attr_name),
                        attr_name,
                    )
                )

    data_vars: dict[str, tuple[tuple[str, ...], np.ndarray]] = {}
    for attr_name, values in timestamp_values.items():
        output_name = attr_name if attr_name.endswith("_utc") else f"{attr_name}_utc"
        data_vars[output_name] = (
            ("run_id",),
            np.asarray(values, dtype="datetime64[ns]"),
        )

    return xr.Dataset(
        data_vars=data_vars,
        coords={
            "run_id": np.asarray(run_ids, dtype=np.int64),
            "run_group": ("run_id", np.asarray(run_groups, dtype=object)),
        },
    )


def correction_total_move_by_axis_from_history(
    path: str | Path,
) -> xr.DataArray:
    """Extract signed readback and image-detected x/y/z moves from history."""

    timestamps: list[np.datetime64] = []
    residuals: list[float] = []
    moves_by_run: list[np.ndarray] = []
    run_ids: list[int] = []
    run_groups: list[str] = []

    with h5py.File(path, "r") as history_file:
        run_names = sorted(
            (
                (run_id, group_name)
                for group_name in history_file
                if (run_id := _correction_history_run_id(group_name)) is not None
            ),
            key=lambda item: item[0],
        )
        for run_id, group_name in run_names:
            group = history_file[group_name]
            run_ids.append(run_id)
            run_groups.append(group_name)
            timestamps.append(
                _correction_timestamp_attr_as_utc64(
                    group.attrs.get("correction_move_finished_at"),
                    "correction_move_finished_at",
                )
            )
            residual_px = np.nan
            if (
                "iteration_weighted_residual_px" in group
                and group["iteration_weighted_residual_px"].shape[0] > 0
            ):
                residual_px = float(group["iteration_weighted_residual_px"][-1])
            elif (
                "move_post_weighted_residual_px" in group
                and group["move_post_weighted_residual_px"].shape[0] > 0
            ):
                residual_px = float(group["move_post_weighted_residual_px"][-1])
            residuals.append(residual_px)

            readback = np.full(len(COMMAND_AXES), np.nan, dtype=np.float64)
            if (
                "initial_readback_position_mm" in group
                and "final_readback_position_mm" in group
            ):
                readback = np.asarray(
                    group["final_readback_position_mm"][...],
                    dtype=np.float64,
                ) - np.asarray(
                    group["initial_readback_position_mm"][...],
                    dtype=np.float64,
                )
            elif "move_final_readback_position_mm" in group:
                moves = np.asarray(
                    group["move_final_readback_position_mm"][...],
                    dtype=np.float64,
                )
                if moves.shape[0] > 0:
                    readback = moves[-1] - moves[0]

            detected = np.full(len(COMMAND_AXES), np.nan, dtype=np.float64)
            if "px_per_readback_mm" in group:
                jacobian = np.asarray(
                    group["px_per_readback_mm"][...],
                    dtype=np.float64,
                )
                if (
                    "iteration_shift_px" in group
                    and group["iteration_shift_px"].shape[0] >= 2
                ):
                    shifts = group["iteration_shift_px"]
                    initial_shift = np.asarray(shifts[0], dtype=np.float64)
                    final_shift = np.asarray(shifts[-1], dtype=np.float64)
                    total_delta = final_shift - initial_shift
                    detected = estimate_command_offset(jacobian, total_delta)
                elif "move_measured_delta_px" in group:
                    total_delta = np.asarray(
                        group["move_measured_delta_px"][...],
                        dtype=np.float64,
                    ).sum(axis=0)
                    detected = estimate_command_offset(jacobian, total_delta)

            moves_by_run.append(np.stack((readback, detected), axis=0))

    dimension = "after_move_timestamp_utc"
    return xr.DataArray(
        np.asarray(
            moves_by_run,
            dtype=np.float64,
        ).reshape(len(run_ids), 2, len(COMMAND_AXES)),
        dims=(dimension, "move_source", "command_axis"),
        coords={
            dimension: np.asarray(timestamps, dtype="datetime64[ns]"),
            "run_id": (dimension, np.asarray(run_ids, dtype=np.int64)),
            "run_group": (dimension, np.asarray(run_groups, dtype=object)),
            "weighted_residual_px": (
                dimension,
                np.asarray(residuals, dtype=np.float64),
            ),
            "move_source": ["readback", "detected"],
            "command_axis": list(COMMAND_AXES),
        },
        name="total_move_mm",
        attrs={
            "units": "readback-mm",
            "long_name": "signed total correction move by source and axis",
        },
    )


def _correction_timestamp_attr_as_utc64(
    value: Any,
    attr_name: str,
) -> np.datetime64:
    if value is None:
        return np.datetime64("NaT", "ns")
    if isinstance(value, bytes):
        value = value.decode()
    text = str(value).strip()
    if not text:
        return np.datetime64("NaT", "ns")
    if text.endswith("Z"):
        text = f"{text[:-1]}+00:00"
    try:
        timestamp = datetime.fromisoformat(text)
    except ValueError as exc:
        raise ValueError(f"invalid {attr_name}: {value!r}") from exc
    if timestamp.tzinfo is None:
        raise ValueError(f"{attr_name} must include a UTC offset: {value!r}")
    timestamp_utc = timestamp.astimezone(UTC).replace(tzinfo=None)
    return np.datetime64(timestamp_utc, "ns")


def _apply_calibration_persistence_attrs(
    result: xr.Dataset,
    calibration: xr.Dataset,
) -> xr.Dataset:
    attrs = {
        key: value
        for key, value in calibration.attrs.items()
        if key.startswith("calibration_persistence_")
        or key == "calibration_pending_spool_path"
    }
    return result.assign_attrs(attrs) if attrs else result


def save_correction_history_dataset(
    result: xr.Dataset,
    path: str | Path,
    *,
    run_id: int,
) -> Path:
    """Persist one correction run into the correction-history HDF5 file."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    group_name = _correction_history_group_name(run_id)
    saved = _netcdf_safe_correction_result(result.load().copy(deep=True))

    with h5py.File(output_path, "a") as history_file:
        if group_name in history_file:
            if _update_correction_history_group_in_place(
                history_file,
                group_name,
                saved,
                run_id=int(run_id),
            ):
                return output_path
            del history_file[group_name]
        _update_correction_history_root_attrs(
            history_file,
            group_name,
            saved,
            run_id=int(run_id),
        )

    saved.to_netcdf(
        output_path,
        engine="h5netcdf",
        mode="a",
        group=group_name,
        encoding=hdf5_image_encoding(saved),
        unlimited_dims=_correction_history_unlimited_dims(saved),
    )
    return output_path


def save_correction_history_dataset_deferred(
    result: xr.Dataset,
    path: str | Path,
    *,
    run_id: int,
) -> PersistenceResult:
    """Save correction history, queueing the latest snapshot if HDF5 is locked."""

    output_path = normalize_target_path(path)
    try:
        save_correction_history_dataset(result, output_path, run_id=run_id)
    except Exception as exc:
        group_name = _correction_history_group_name(run_id)
        saved = _netcdf_safe_correction_result(result.load().copy(deep=True))
        entry = stage_dataset(
            saved,
            output_path,
            operation="correction_history",
            metadata={
                "run_id": int(run_id),
                "group_name": group_name,
                "calibration_path": str(saved.attrs.get("calibration_path", "")),
                "completed": bool(
                    saved.attrs.get("correction_history_completed", False)
                ),
            },
        )
        _discard_pending_correction_history_run(
            output_path,
            int(run_id),
            keep_entry=entry,
        )
        return PersistenceResult(
            target_path=output_path,
            spool_path=entry.path,
            flushed=False,
            pending=True,
            message=(
                f"queued correction history write because target is unavailable: {exc}"
            ),
        )

    _discard_pending_correction_history_run(output_path, int(run_id))
    return PersistenceResult(
        target_path=output_path,
        spool_path=None,
        flushed=True,
        pending=False,
        message="correction history write flushed to target",
    )


def flush_pending_correction_history_datasets(
    target_path: str | Path | None = None,
) -> list[PersistenceResult]:
    """Flush queued correction history snapshots whose target files are writable."""

    results: list[PersistenceResult] = []
    for stale_entry in _superseded_pending_correction_history_entries(target_path):
        discard_spool_entry(stale_entry)

    pending_entries = _latest_pending_correction_history_entries_by_run(target_path)
    for entry in pending_entries.values():
        target = Path(str(entry.metadata["target_path"]))
        run_id = int(entry.metadata["run_id"])
        dataset = load_spooled_dataset(entry)
        try:
            save_correction_history_dataset(dataset, target, run_id=run_id)
        except Exception as exc:
            results.append(
                PersistenceResult(
                    target_path=target,
                    spool_path=entry.path,
                    flushed=False,
                    pending=True,
                    message=(
                        "queued correction history write is still pending because "
                        f"target is unavailable: {exc}"
                    ),
                )
            )
            continue

        discard_spool_entry(entry)
        _discard_pending_correction_history_run(target, run_id)
        results.append(
            PersistenceResult(
                target_path=target,
                spool_path=entry.path,
                flushed=True,
                pending=False,
                message="queued correction history write flushed to target",
            )
        )
    return results


def load_latest_correction_history_dataset(
    calibration_path: str | Path,
) -> xr.Dataset | None:
    """Load the most recent correction run for a calibration, if one exists."""

    history_path = correction_history_path(calibration_path)
    target_run_id = _latest_correction_history_run_id(history_path)
    pending = _latest_pending_correction_history_entry(history_path)
    pending_run_id = int(pending.metadata["run_id"]) if pending is not None else None

    if target_run_id is None and pending_run_id is None:
        return None

    if pending is not None and (
        target_run_id is None
        or pending_run_id is None
        or pending_run_id >= target_run_id
    ):
        result = load_spooled_dataset(pending)
        result = result.assign_attrs(
            persistence_result_attrs(
                "correction_history",
                PersistenceResult(
                    target_path=Path(str(pending.metadata["target_path"])),
                    spool_path=pending.path,
                    flushed=False,
                    pending=True,
                    message="correction history write is pending flush to target",
                ),
            )
        )
    else:
        assert target_run_id is not None
        with xr.open_dataset(
            history_path,
            engine="h5netcdf",
            group=_correction_history_group_name(target_run_id),
        ) as dataset_on_disk:
            result = dataset_on_disk.load()

    result = _restore_netcdf_safe_correction_result(result)
    result.attrs.setdefault("calibration_path", str(calibration_path))
    result.attrs.setdefault("correction_history_path", str(history_path))
    return result


def _netcdf_safe_correction_result(result: xr.Dataset) -> xr.Dataset:
    saved = result.copy(deep=True)
    for name in saved.data_vars:
        if saved[name].dtype == bool:
            saved[name] = saved[name].astype(np.int8)
            saved[name].attrs["flag_values"] = np.asarray([0, 1], dtype=np.int8)
            saved[name].attrs["flag_meanings"] = "false true"
    for key, value in tuple(saved.attrs.items()):
        if isinstance(value, (bool, np.bool_)):
            saved.attrs[key] = int(value)
    return saved


def _restore_netcdf_safe_correction_result(result: xr.Dataset) -> xr.Dataset:
    restored = result.copy(deep=True)
    for name in (
        "move_active_axis_mask",
        "move_feedback_valid",
    ):
        if name in restored:
            restored[name] = restored[name].astype(bool)
    for key in (
        "correction_applied",
        "correction_converged",
        "correction_history_completed",
    ):
        if key in restored.attrs:
            restored.attrs[key] = bool(restored.attrs[key])
    return restored


def _update_correction_history_group_in_place(
    history_file: h5py.File,
    group_name: str,
    saved: xr.Dataset,
    *,
    run_id: int,
) -> bool:
    group = history_file[group_name]
    if not _correction_history_group_can_update(group, saved):
        return False

    _update_correction_history_root_attrs(
        history_file,
        group_name,
        saved,
        run_id=run_id,
    )
    _update_hdf5_attrs(group.attrs, saved.attrs)

    for dim_name in _CORRECTION_HISTORY_RESIZABLE_DIMS:
        if dim_name in saved.coords:
            _update_hdf5_dataset(group[dim_name], saved.coords[dim_name])
    for name, variable in saved.data_vars.items():
        _update_hdf5_dataset(group[name], variable)
    return True


def _correction_history_group_can_update(
    group: h5py.Group,
    saved: xr.Dataset,
) -> bool:
    if any(name not in saved.variables for name in group.keys()):
        return False

    required_names = set(saved.data_vars)
    required_names.update(
        dim_name
        for dim_name in _CORRECTION_HISTORY_RESIZABLE_DIMS
        if dim_name in saved.coords
    )
    for name in required_names:
        if name not in group:
            return False
        if not isinstance(group[name], h5py.Dataset):
            return False
        if not _hdf5_dataset_can_store_variable(group[name], saved[name]):
            return False
    return True


def _hdf5_dataset_can_store_variable(
    dataset: h5py.Dataset,
    variable: xr.DataArray,
) -> bool:
    shape = tuple(int(size) for size in variable.shape)
    if len(dataset.shape) != len(shape):
        return False

    for axis, (current_size, target_size) in enumerate(
        zip(dataset.shape, shape, strict=True)
    ):
        dim_name = variable.dims[axis]
        if dim_name in _CORRECTION_HISTORY_RESIZABLE_DIMS:
            max_size = dataset.maxshape[axis]
            if max_size is not None and target_size > max_size:
                return False
            continue
        if current_size != target_size:
            return False
    return True


def _update_hdf5_dataset(
    dataset: h5py.Dataset,
    variable: xr.DataArray,
) -> None:
    values = np.asarray(variable.values)
    shape = tuple(int(size) for size in values.shape)
    if dataset.shape != shape:
        dataset.resize(shape)
    if shape:
        dataset[...] = values
    else:
        dataset[()] = values
    _update_hdf5_attrs(dataset.attrs, variable.attrs)


def _update_hdf5_attrs(
    attrs: h5py.AttributeManager,
    values: dict[Any, Any],
) -> None:
    for key, value in values.items():
        attrs[str(key)] = value


def _update_correction_history_root_attrs(
    history_file: h5py.File,
    group_name: str,
    saved: xr.Dataset,
    *,
    run_id: int,
) -> None:
    history_file.attrs["format"] = _CORRECTION_HISTORY_FORMAT
    history_file.attrs["latest_run_group"] = group_name
    history_file.attrs["latest_run_id"] = int(run_id)
    history_file.attrs["calibration_path"] = str(
        saved.attrs.get("calibration_path", "")
    )


def _correction_history_unlimited_dims(saved: xr.Dataset) -> tuple[str, ...]:
    return tuple(
        dim_name
        for dim_name in _CORRECTION_HISTORY_RESIZABLE_DIMS
        if dim_name in saved.dims
    )


def _latest_correction_history_run_id(path: Path) -> int | None:
    if not path.exists():
        return None
    with h5py.File(path, "r") as history_file:
        latest_run_id = _latest_correction_history_run_id_from_attrs(history_file)
        if latest_run_id is not None:
            return latest_run_id

        run_ids = [
            run_id
            for run_id in (
                _correction_history_run_id(name) for name in history_file.keys()
            )
            if run_id is not None
        ]
    return max(run_ids, default=None)


def _next_correction_history_run_id(path: Path) -> int:
    run_ids: list[int] = []
    if path.exists():
        with h5py.File(path, "r") as history_file:
            latest_run_id = _latest_correction_history_run_id_from_attrs(history_file)
            if latest_run_id is not None:
                run_ids.append(latest_run_id)
            else:
                run_ids.extend(
                    run_id
                    for run_id in (
                        _correction_history_run_id(name) for name in history_file.keys()
                    )
                    if run_id is not None
                )
    run_ids.extend(
        int(entry.metadata["run_id"])
        for entry in iter_pending_entries(
            operation="correction_history",
            target_path=path,
        )
    )
    return max(run_ids, default=-1) + 1


def _latest_correction_history_run_id_from_attrs(
    history_file: h5py.File,
) -> int | None:
    latest_run_id = _attr_as_int(history_file.attrs.get("latest_run_id"))
    if latest_run_id is not None:
        group_name = _correction_history_group_name(latest_run_id)
        if group_name in history_file:
            return latest_run_id

    latest_attr = history_file.attrs.get("latest_run_group")
    if isinstance(latest_attr, bytes):
        latest_attr = latest_attr.decode()
    latest_group_run_id = (
        _correction_history_run_id(latest_attr)
        if isinstance(latest_attr, str) and latest_attr in history_file
        else None
    )
    return latest_group_run_id


def _attr_as_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bytes):
        value = value.decode()
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed >= 0 else None


def _correction_history_group_name(run_id: int) -> str:
    run_id = int(run_id)
    if run_id < 0:
        raise ValueError("run_id must be non-negative")
    return f"run_{run_id:06d}"


def _correction_history_run_id(group_name: str | None) -> int | None:
    if (
        not isinstance(group_name, str)
        or not group_name.startswith("run_")
        or not group_name.removeprefix("run_").isdigit()
    ):
        return None
    return int(group_name.removeprefix("run_"))


def _pending_correction_history_datasets(
    path: Path,
    *,
    exclude_run_id: int | None = None,
) -> list[xr.Dataset]:
    datasets: list[xr.Dataset] = []
    for entry in _latest_pending_correction_history_entries_by_run(path).values():
        run_id = int(entry.metadata["run_id"])
        if run_id == exclude_run_id:
            continue
        datasets.append(
            _restore_netcdf_safe_correction_result(load_spooled_dataset(entry))
        )
    return datasets


def _latest_pending_correction_history_entry(path: Path) -> PendingEntry | None:
    entries = _latest_pending_correction_history_entries_by_run(path)
    if not entries:
        return None
    return entries[max(entries)]


def _latest_pending_correction_history_entries_by_run(
    path: Path | str | None,
) -> dict[int, PendingEntry]:
    entries_by_run: dict[int, PendingEntry] = {}
    for entry in iter_pending_entries(
        operation="correction_history",
        target_path=path,
    ):
        run_id = int(entry.metadata["run_id"])
        entries_by_run[run_id] = entry
    return entries_by_run


def _superseded_pending_correction_history_entries(
    path: Path | str | None,
) -> list[PendingEntry]:
    all_entries: dict[int, list[PendingEntry]] = {}
    for entry in iter_pending_entries(
        operation="correction_history",
        target_path=path,
    ):
        all_entries.setdefault(int(entry.metadata["run_id"]), []).append(entry)
    superseded: list[PendingEntry] = []
    for entries in all_entries.values():
        superseded.extend(entries[:-1])
    return superseded


def _discard_pending_correction_history_run(
    path: Path | str,
    run_id: int,
    *,
    keep_entry: PendingEntry | None = None,
) -> None:
    for entry in iter_pending_entries(
        operation="correction_history",
        target_path=path,
    ):
        if keep_entry is not None and entry.path == keep_entry.path:
            continue
        if int(entry.metadata["run_id"]) == int(run_id):
            discard_spool_entry(entry)


def _correction_feedback_metrics(
    predicted_delta_px: np.ndarray,
    measured_delta_px: np.ndarray,
    *,
    weights: Sequence[float] | np.ndarray | None,
) -> CorrectionFeedback:
    predicted = _shift_to_observation_vector(predicted_delta_px)
    measured = _shift_to_observation_vector(measured_delta_px)
    weight_values = _observation_weight_values(weights)
    weighted_predicted = weight_values * predicted
    weighted_measured = weight_values * measured
    predicted_energy = float(predicted @ weighted_predicted)
    measured_energy = float(measured @ weighted_measured)
    predicted_response = float(np.sqrt(max(predicted_energy, 0.0)))
    measured_response = float(np.sqrt(max(measured_energy, 0.0)))
    projected_energy = float(predicted @ weighted_measured)
    if predicted_energy > 0.0 and predicted_response > 0.0:
        alpha = projected_energy / predicted_energy
        parallel_px = projected_energy / predicted_response
    else:
        alpha = np.nan
        parallel_px = np.nan
    return CorrectionFeedback(
        predicted_weighted_response_px=predicted_response,
        measured_weighted_response_px=measured_response,
        alpha=float(alpha),
        parallel_px=float(parallel_px),
    )


def _correction_feedback_is_valid(
    feedback: CorrectionFeedback,
) -> bool:
    return bool(
        feedback.predicted_weighted_response_px > 0.0
        and np.isfinite(feedback.alpha)
        and np.isfinite(feedback.parallel_px)
    )


def _weighted_shift_norm(
    shift_px: np.ndarray,
    *,
    weights: Sequence[float] | np.ndarray | None,
) -> float:
    values = _shift_to_observation_vector(shift_px)
    weight_values = _observation_weight_values(weights)
    return float(np.sqrt(max(float(values @ (weight_values * values)), 0.0)))


def _shift_to_observation_vector(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.shape != (len(CAMERAS), len(PIXEL_AXES)):
        raise ValueError("shift array must have shape (camera, pixel_axis)")
    if not np.isfinite(array).all():
        raise ValueError("shift array must contain only finite values")
    return array.reshape(-1)


def _observation_weight_values(
    weights: Sequence[float] | np.ndarray | None,
) -> np.ndarray:
    if weights is None:
        return np.ones(len(CAMERAS) * len(PIXEL_AXES), dtype=np.float64)
    values = np.asarray(weights, dtype=np.float64)
    if values.shape == (len(CAMERAS), len(PIXEL_AXES)):
        values = values.reshape(-1)
    if values.shape != (len(CAMERAS) * len(PIXEL_AXES),):
        raise ValueError("weights must have four observation values")
    if not np.isfinite(values).all() or np.any(values < 0.0):
        raise ValueError("weights must contain finite non-negative values")
    if not np.any(values > 0.0):
        raise ValueError("weights must include at least one positive value")
    return values


def _correction_weights(
    weights: Sequence[float] | np.ndarray | None,
) -> Sequence[float] | np.ndarray | None:
    if weights is not None:
        return weights
    return constants.CORRECTION_OBSERVATION_WEIGHTS


def _resolve_correction_mode(value: str) -> str:
    mode = str(value).strip().lower()
    if mode not in CORRECTION_MODES:
        raise ValueError(
            "correction_mode must be one of "
            + ", ".join(repr(item) for item in CORRECTION_MODES)
        )
    return mode


def _resolve_lqr_projected_tolerance(value: Any) -> float:
    if value is _USE_DEFAULT:
        value = constants.DEFAULT_LQR_CORRECTION_PROJECTED_TOLERANCE
    tolerance = float(value)
    if not np.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("lqr_projected_tolerance must be finite and non-negative")
    return tolerance


def _positive_um_to_mm(value: float, name: str) -> float:
    value_um = float(value)
    if not np.isfinite(value_um) or value_um <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return value_um / 1000.0


def _calibration_polar_deg(calibration: xr.Dataset) -> float:
    if "polar" not in calibration.attrs:
        raise ValueError("correction requires calibration attr 'polar'")
    polar_deg = float(calibration.attrs["polar"])
    if not np.isfinite(polar_deg):
        raise ValueError("correction requires finite calibration attr 'polar'")
    return polar_deg


def _read_initial_readback_position_and_polar_deg(
    motor_backend: CorrectionMotorBackend,
) -> tuple[np.ndarray, float]:
    readback_position_mm, orientation = _read_initial_readback_position_and_orientation_deg(
        motor_backend
    )
    return readback_position_mm, float(orientation["polar"])


def _read_initial_readback_position_and_orientation_deg(
    motor_backend: CorrectionMotorBackend,
) -> tuple[np.ndarray, dict[str, float]]:
    backend_has_orientation = True
    try:
        values = _position_values(
            motor_backend.get_positions((*COMMAND_AXES, "p")),
            len(COMMAND_AXES) + 1,
            "initial x/y/z/p readback",
        )
        readback_position_mm = values[: len(COMMAND_AXES)].copy()
        current_polar_deg = float(values[-1])
    except Exception as exc:
        if isinstance(motor_backend, DirectBCSMotorBackend):
            raise
        logger.info(
            "Motor backend did not provide x/y/z/p readback (%s); "
            "falling back to delegated x/y/z plus direct BCS polar read.",
            exc,
        )
        backend_has_orientation = False
        readback_position_mm = _position_values(
            motor_backend.get_positions(COMMAND_AXES),
            len(COMMAND_AXES),
            "initial x/y/z readback",
        )
        current_polar_deg = _single_position_value(
            get_positions(("p",)),
            "current polar",
        )

    current_tilt_deg = np.nan
    current_azi_deg = np.nan
    try:
        if not backend_has_orientation:
            raise ValueError("motor backend did not provide polar readback")
        orientation_values = _position_values(
            motor_backend.get_positions(("t", "a")),
            2,
            "current t/a readback",
        )
        current_tilt_deg = float(orientation_values[0])
        current_azi_deg = float(orientation_values[1])
    except Exception as exc:
        logger.info(
            "Motor backend did not provide t/a readback (%s); "
            "trying direct BCS t/a read.",
            exc,
        )
        try:
            orientation_values = _position_values(
                get_positions(("t", "a")),
                2,
                "current t/a readback",
            )
            current_tilt_deg = float(orientation_values[0])
            current_azi_deg = float(orientation_values[1])
        except Exception as direct_exc:
            logger.info(
                "Direct t/a readback failed (%s); orientation ECC seed will be skipped.",
                direct_exc,
            )
    return (
        readback_position_mm,
        {
            "polar": current_polar_deg,
            "tilt": current_tilt_deg,
            "azi": current_azi_deg,
        },
    )


def _position_values(values: Sequence[float], count: int, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.shape != (count,):
        raise ValueError(f"{name} must contain exactly {count} value(s)")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    return array


def _single_position_value(values: Sequence[float], name: str) -> float:
    return float(_position_values(values, 1, name)[0])


def _polar_deadband_deg() -> float:
    deadband = float(constants.MOTOR_READBACK_DEADBAND["p"])
    if not np.isfinite(deadband) or deadband < 0.0:
        raise ValueError("polar deadband must be finite and non-negative")
    return deadband


def _runtime_px_per_readback_mm_for_polar(
    calibration: xr.Dataset,
    current_polar_deg: float,
) -> tuple[np.ndarray, dict[str, float | bool]]:
    polar_attrs = _runtime_polar_attrs(calibration, current_polar_deg)
    jacobian = np.asarray(calibration["px_per_readback_mm"].values, dtype=np.float64)
    if polar_attrs["polar_rotation_applied"]:
        jacobian = _rotate_px_per_readback_mm_for_polar_delta(
            jacobian,
            float(polar_attrs["polar_applied_delta_deg"]),
        )
    return jacobian, polar_attrs


def _runtime_polar_attrs(
    calibration: xr.Dataset,
    current_polar_deg: float,
) -> dict[str, float | bool]:
    calibration_polar_deg = _calibration_polar_deg(calibration)
    current_polar_deg = float(current_polar_deg)
    if not np.isfinite(current_polar_deg):
        raise ValueError("current polar readback must be finite")
    polar_deadband_deg = _polar_deadband_deg()
    polar_delta_deg = current_polar_deg - calibration_polar_deg
    polar_rotation_applied = bool(abs(polar_delta_deg) > polar_deadband_deg)
    return {
        "calibration_polar_deg": float(calibration_polar_deg),
        "current_polar_deg": float(current_polar_deg),
        "polar_delta_deg": float(polar_delta_deg),
        "polar_applied_delta_deg": float(
            polar_delta_deg if polar_rotation_applied else 0.0
        ),
        "polar_deadband_deg": float(polar_deadband_deg),
        "polar_rotation_applied": polar_rotation_applied,
    }


def _prefixed_polar_attrs(
    prefix: str,
    polar_attrs: Mapping[str, float | bool],
) -> dict[str, float | bool]:
    return {f"{prefix}_{key}": value for key, value in polar_attrs.items()}


def _runtime_orientation_attrs(
    calibration: xr.Dataset,
    *,
    polar_attrs: Mapping[str, float | bool],
    current_tilt_deg: float,
    current_azi_deg: float,
) -> dict[str, float | bool | str]:
    attrs: dict[str, float | bool | str] = {
        "orientation_seed_inputs_valid": False,
        "orientation_seed_applied": False,
        "orientation_seed_warning": "",
        "orientation_seed_basis_condition": np.nan,
    }
    try:
        calibration_tilt_deg = _calibration_orientation_attr_deg(calibration, "tilt")
        calibration_azi_deg = _calibration_orientation_attr_deg(calibration, "azi")
        current_tilt_value = float(current_tilt_deg)
        current_azi_value = float(current_azi_deg)
        if not np.isfinite(current_tilt_value):
            raise ValueError("current tilt readback must be finite")
        if not np.isfinite(current_azi_value):
            raise ValueError("current azimuth readback must be finite")
    except ValueError as exc:
        attrs["orientation_seed_warning"] = (
            f"orientation ECC seed skipped: {exc}"
        )
        return attrs

    azi_deadband_deg = float(constants.DEFAULT_ORIENTATION_ECC_AZIMUTH_DEADBAND_DEG)
    if not np.isfinite(azi_deadband_deg) or azi_deadband_deg < 0.0:
        raise ValueError("azimuth deadband must be finite and non-negative")

    tilt_delta_deg = current_tilt_value - calibration_tilt_deg
    azi_delta_deg = current_azi_value - calibration_azi_deg
    azi_deadband_active = bool(abs(azi_delta_deg) < azi_deadband_deg)
    azi_applied_delta_deg = 0.0 if azi_deadband_active else azi_delta_deg
    attrs |= {
        "orientation_seed_inputs_valid": True,
        "calibration_tilt_deg": float(calibration_tilt_deg),
        "current_tilt_deg": float(current_tilt_value),
        "tilt_delta_deg": float(tilt_delta_deg),
        "tilt_applied_delta_deg": float(tilt_delta_deg),
        "calibration_azi_deg": float(calibration_azi_deg),
        "current_azi_deg": float(current_azi_value),
        "azi_delta_deg": float(azi_delta_deg),
        "azi_applied_delta_deg": float(azi_applied_delta_deg),
        "azi_deadband_deg": float(azi_deadband_deg),
        "azi_deadband_active": azi_deadband_active,
        "effective_current_azi_deg": float(calibration_azi_deg + azi_applied_delta_deg),
        "calibration_polar_deg": float(polar_attrs["calibration_polar_deg"]),
        "current_polar_deg": float(polar_attrs["current_polar_deg"]),
    }
    return attrs


def _calibration_orientation_attr_deg(calibration: xr.Dataset, name: str) -> float:
    if name not in calibration.attrs:
        raise ValueError(f"calibration attr {name!r} missing")
    value = float(calibration.attrs[name])
    if not np.isfinite(value):
        raise ValueError(f"calibration attr {name!r} must be finite")
    return value


def _prefixed_orientation_attrs(
    prefix: str,
    orientation_attrs: Mapping[str, float | bool | str],
) -> dict[str, float | bool | str]:
    return {
        f"{prefix}_{key}": value
        for key, value in orientation_attrs.items()
        if value != ""
    }


def orientation_ecc_initial_warps_for_readbacks(
    calibration: xr.Dataset,
    readbacks: Mapping[str, Any],
    reference_points: Mapping[str, Any],
) -> tuple[dict[str, np.ndarray], dict[str, float | bool | str]]:
    """Return per-camera ROI-local affine ECC seeds for current p/t/a readbacks."""
    values = _orientation_readback_values(readbacks)
    polar_attrs = _runtime_polar_attrs(calibration, values["p"])
    orientation_attrs = _runtime_orientation_attrs(
        calibration,
        polar_attrs=polar_attrs,
        current_tilt_deg=values["t"],
        current_azi_deg=values["a"],
    )
    seed_kwargs = _orientation_ecc_seed_shift_kwargs(
        {
            "use_ecc_refinement": True,
            "ecc_motion_model": "affine",
            "ecc_reference_point_px": {
                camera: np.asarray(reference_points[camera], dtype=np.float64)
                for camera in CAMERAS
            },
        },
        calibration=calibration,
        orientation_attrs=orientation_attrs,
    )
    if "ecc_initial_warp" not in seed_kwargs:
        warning = str(orientation_attrs.get("orientation_seed_warning", "")).strip()
        raise ValueError(warning or "orientation ECC seed did not produce a warp")
    warps = {
        camera: np.asarray(seed_kwargs["ecc_initial_warp"][camera], dtype=np.float64)
        for camera in CAMERAS
    }
    attrs = dict(polar_attrs)
    attrs.update(orientation_attrs)
    return warps, attrs


def _orientation_readback_values(readbacks: Mapping[str, Any]) -> dict[str, float]:
    values: dict[str, float] = {}
    for axis in ORIENTATION_READBACK_AXES:
        if axis not in readbacks:
            raise ValueError(f"current motor readbacks missing {axis!r}")
        value = float(readbacks[axis])
        if not np.isfinite(value):
            raise ValueError(f"current motor readback for {axis!r} must be finite")
        values[axis] = value
    return values


def _orientation_ecc_seed_shift_kwargs(
    shift_kwargs: Mapping[str, Any],
    *,
    orientation_attrs: Mapping[str, float | bool | str],
    calibration: xr.Dataset | None = None,
) -> dict[str, Any]:
    kwargs = dict(shift_kwargs)
    if not kwargs.get("use_ecc_refinement"):
        return kwargs

    use_orientation_seed = bool(orientation_attrs.get("orientation_seed_inputs_valid"))
    if not use_orientation_seed:
        return kwargs

    if calibration is None:
        raise ValueError("calibration is required for orientation ECC seed")
    try:
        kwargs = _shift_kwargs_with_calibration_ecc_points(calibration, kwargs)
        seed_warps, basis_condition = _orientation_ecc_initial_warps(
            calibration,
            orientation_attrs=orientation_attrs,
            reference_points=kwargs["ecc_reference_point_px"],
        )
    except ValueError as exc:
        logger.info("Orientation ECC seed skipped: %s", exc)
        if isinstance(orientation_attrs, dict):
            orientation_attrs["orientation_seed_warning"] = (
                f"orientation ECC seed skipped: {exc}"
            )
            orientation_attrs["orientation_seed_applied"] = False
        return kwargs
    kwargs["ecc_initial_warp"] = seed_warps
    if isinstance(orientation_attrs, dict):
        orientation_attrs["orientation_seed_applied"] = True
        orientation_attrs["orientation_seed_basis_condition"] = float(basis_condition)
    return kwargs


def _orientation_ecc_initial_warps(
    calibration: xr.Dataset,
    *,
    orientation_attrs: Mapping[str, float | bool | str],
    reference_points: Mapping[str, Any],
) -> tuple[dict[str, np.ndarray], float]:
    jacobian = np.asarray(calibration["px_per_readback_mm"].values, dtype=np.float64)
    if jacobian.shape != (len(CAMERAS), len(PIXEL_AXES), len(COMMAND_AXES)):
        raise ValueError("calibration Jacobian has unexpected shape")
    if not np.isfinite(jacobian).all():
        raise ValueError("calibration Jacobian must contain only finite values")

    calibration_basis = _sample_plane_basis(
        polar_deg=float(orientation_attrs["calibration_polar_deg"]),
        tilt_deg=float(orientation_attrs["calibration_tilt_deg"]),
        azi_deg=float(orientation_attrs["calibration_azi_deg"]),
    )
    current_basis = _sample_plane_basis(
        polar_deg=float(orientation_attrs["current_polar_deg"]),
        tilt_deg=float(orientation_attrs["current_tilt_deg"]),
        azi_deg=float(orientation_attrs["effective_current_azi_deg"]),
    )

    warps: dict[str, np.ndarray] = {}
    condition_values: list[float] = []
    for camera_index, camera in enumerate(CAMERAS):
        point = np.asarray(reference_points[camera], dtype=np.float64)
        if point.shape != (len(PIXEL_AXES),) or not np.isfinite(point).all():
            raise ValueError(f"ECC reference point for {camera} must be finite")
        projection_calibration = jacobian[camera_index] @ calibration_basis
        projection_current = jacobian[camera_index] @ current_basis
        condition = float(np.linalg.cond(projection_calibration))
        if (
            not np.isfinite(condition)
            or condition > _ORIENTATION_ECC_MAX_BASIS_CONDITION
        ):
            raise ValueError(
                f"projected sample basis for {camera} is ill-conditioned "
                f"({condition:.4g})"
            )
        try:
            affine = projection_current @ np.linalg.inv(projection_calibration)
        except np.linalg.LinAlgError as exc:
            raise ValueError(
                f"projected sample basis for {camera} is singular"
            ) from exc
        if not np.isfinite(affine).all():
            raise ValueError(f"orientation affine for {camera} is not finite")
        warps[camera] = _affine_warp_about_point(
            affine,
            point,
            np.zeros(len(PIXEL_AXES), dtype=np.float64),
        )
        condition_values.append(condition)
    return warps, max(condition_values)


def _sample_plane_basis(
    *,
    polar_deg: float,
    tilt_deg: float,
    azi_deg: float,
) -> np.ndarray:
    return (
        _rotation_y_deg(polar_deg)
        @ _rotation_x_deg(-tilt_deg)
        @ _rotation_z_deg(-azi_deg)
        @ _SAMPLE_PLANE_BASIS
    )


def _rotation_x_deg(angle_deg: float) -> np.ndarray:
    angle_rad = np.deg2rad(float(angle_deg))
    cosine = float(np.cos(angle_rad))
    sine = float(np.sin(angle_rad))
    return np.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.0, cosine, -sine],
            [0.0, sine, cosine],
        ],
        dtype=np.float64,
    )


def _rotation_y_deg(angle_deg: float) -> np.ndarray:
    angle_rad = np.deg2rad(float(angle_deg))
    cosine = float(np.cos(angle_rad))
    sine = float(np.sin(angle_rad))
    return np.asarray(
        [
            [cosine, 0.0, sine],
            [0.0, 1.0, 0.0],
            [-sine, 0.0, cosine],
        ],
        dtype=np.float64,
    )


def _rotation_z_deg(angle_deg: float) -> np.ndarray:
    angle_rad = np.deg2rad(float(angle_deg))
    cosine = float(np.cos(angle_rad))
    sine = float(np.sin(angle_rad))
    return np.asarray(
        [
            [cosine, -sine, 0.0],
            [sine, cosine, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def _affine_warp_about_point(
    affine: np.ndarray,
    point: np.ndarray,
    translation: np.ndarray,
) -> np.ndarray:
    affine_array = np.asarray(affine, dtype=np.float64)
    if affine_array.shape != (len(PIXEL_AXES), len(PIXEL_AXES)):
        raise ValueError("orientation affine must have shape (pixel_axis, pixel_axis)")
    point_array = np.asarray(point, dtype=np.float64)
    translation_array = np.asarray(translation, dtype=np.float64)
    offset = point_array + translation_array - affine_array @ point_array
    return np.asarray(
        [
            [affine_array[0, 0], affine_array[0, 1], offset[0]],
            [affine_array[1, 0], affine_array[1, 1], offset[1]],
        ],
        dtype=np.float64,
    )


def _rotate_px_per_readback_mm_for_polar_delta(
    px_per_readback_mm: np.ndarray,
    polar_delta_deg: float,
) -> np.ndarray:
    jacobian = np.asarray(px_per_readback_mm, dtype=np.float64)
    if jacobian.shape != (len(CAMERAS), len(PIXEL_AXES), len(COMMAND_AXES)):
        raise ValueError(
            "px_per_readback_mm must have shape (camera, pixel_axis, command_axis)"
        )
    if not np.isfinite(jacobian).all():
        raise ValueError("px_per_readback_mm must contain only finite values")
    delta_deg = float(polar_delta_deg)
    if not np.isfinite(delta_deg):
        raise ValueError("polar_delta_deg must be finite")

    delta_rad = np.deg2rad(delta_deg)
    cosine = float(np.cos(delta_rad))
    sine = float(np.sin(delta_rad))
    rotation = np.asarray(
        [
            [cosine, 0.0, sine],
            [0.0, 1.0, 0.0],
            [-sine, 0.0, cosine],
        ],
        dtype=np.float64,
    )
    return (
        jacobian.reshape(len(CAMERAS) * len(PIXEL_AXES), len(COMMAND_AXES)) @ rotation
    ).reshape(len(CAMERAS), len(PIXEL_AXES), len(COMMAND_AXES))


def _beam_geometry_from_calibration(
    calibration: xr.Dataset,
    *,
    beam_xz_angle_from_analyzer_deg: float,
    polar_deg: float | None = None,
) -> dict[str, np.ndarray | float]:
    if polar_deg is None:
        polar_deg = _calibration_polar_deg(calibration)
    else:
        polar_deg = float(polar_deg)
    beam_angle_deg = float(beam_xz_angle_from_analyzer_deg)
    if not np.isfinite(polar_deg):
        raise ValueError("beam correction requires finite polar angle")
    if not np.isfinite(beam_angle_deg):
        raise ValueError("beam_xz_angle_from_analyzer_deg must be finite")
    beam_runtime_angle_deg = beam_angle_deg - polar_deg
    beam_angle_rad = np.deg2rad(beam_runtime_angle_deg)
    beam_unit = np.asarray(
        [np.sin(beam_angle_rad), 0.0, np.cos(beam_angle_rad)],
        dtype=np.float64,
    )
    beam_transverse_unit = np.asarray(
        [np.cos(beam_angle_rad), 0.0, -np.sin(beam_angle_rad)],
        dtype=np.float64,
    )
    analyzer_runtime_angle_deg = -polar_deg
    analyzer_angle_rad = np.deg2rad(analyzer_runtime_angle_deg)
    analyzer_unit = np.asarray(
        [np.sin(analyzer_angle_rad), 0.0, np.cos(analyzer_angle_rad)],
        dtype=np.float64,
    )
    analyzer_transverse_unit = np.asarray(
        [np.cos(analyzer_angle_rad), 0.0, -np.sin(analyzer_angle_rad)],
        dtype=np.float64,
    )
    projection_matrix = np.asarray(
        [
            beam_transverse_unit,
            analyzer_transverse_unit,
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float64,
    )
    return {
        "polar_deg": polar_deg,
        "beam_xz_angle_deg": beam_runtime_angle_deg,
        "analyzer_xz_angle_deg": analyzer_runtime_angle_deg,
        "beam_unit": beam_unit,
        "beam_transverse_unit": beam_transverse_unit,
        "analyzer_unit": analyzer_unit,
        "analyzer_transverse_unit": analyzer_transverse_unit,
        "projection_matrix": projection_matrix,
    }


def _beam_offset_from_estimated_offset(
    estimated_offset_mm: np.ndarray,
    beam_geometry: dict[str, np.ndarray | float] | None,
) -> np.ndarray | None:
    if beam_geometry is None:
        return None
    offset = np.asarray(estimated_offset_mm, dtype=np.float64)
    if offset.shape != (len(COMMAND_AXES),):
        raise ValueError("estimated offset must have one value for x/y/z")
    if not np.isfinite(offset).all():
        raise ValueError("estimated offset must contain only finite values")
    transverse_unit = np.asarray(
        beam_geometry["beam_transverse_unit"],
        dtype=np.float64,
    )
    beam_unit = np.asarray(beam_geometry["beam_unit"], dtype=np.float64)
    transverse = float(transverse_unit @ offset)
    vertical = float(offset[COMMAND_AXES.index("y")])
    longitudinal = float(beam_unit @ offset)
    return np.asarray(
        [transverse, vertical, longitudinal],
        dtype=np.float64,
    )


def _analyzer_offset_from_estimated_offset(
    estimated_offset_mm: np.ndarray,
    beam_geometry: dict[str, np.ndarray | float] | None,
) -> np.ndarray | None:
    if beam_geometry is None:
        return None
    offset = np.asarray(estimated_offset_mm, dtype=np.float64)
    if offset.shape != (len(COMMAND_AXES),):
        raise ValueError("estimated offset must have one value for x/y/z")
    if not np.isfinite(offset).all():
        raise ValueError("estimated offset must contain only finite values")
    transverse_unit = np.asarray(
        beam_geometry["analyzer_transverse_unit"],
        dtype=np.float64,
    )
    analyzer_unit = np.asarray(beam_geometry["analyzer_unit"], dtype=np.float64)
    transverse = float(transverse_unit @ offset)
    vertical = float(offset[COMMAND_AXES.index("y")])
    longitudinal = float(analyzer_unit @ offset)
    return np.asarray(
        [transverse, vertical, longitudinal],
        dtype=np.float64,
    )


def _beam_analyzer_observation_from_offsets(
    beam_offset_mm: np.ndarray | None,
    analyzer_offset_mm: np.ndarray | None,
) -> np.ndarray | None:
    if beam_offset_mm is None or analyzer_offset_mm is None:
        return None
    beam_offset = np.asarray(beam_offset_mm, dtype=np.float64)
    analyzer_offset = np.asarray(analyzer_offset_mm, dtype=np.float64)
    if beam_offset.shape != (len(BEAM_AXES),):
        raise ValueError("beam offset must have one value for each beam axis")
    if analyzer_offset.shape != (len(ANALYZER_AXES),):
        raise ValueError("analyzer offset must have one value for each analyzer axis")
    if not np.isfinite(beam_offset).all() or not np.isfinite(analyzer_offset).all():
        raise ValueError("beam/analyzer offsets must contain only finite values")
    return np.asarray(
        [
            beam_offset[BEAM_AXES.index("beam_transverse")],
            analyzer_offset[ANALYZER_AXES.index("analyzer_transverse")],
            beam_offset[BEAM_AXES.index("beam_vertical")],
        ],
        dtype=np.float64,
    )


def _correction_criterion_residual(
    *,
    correction_mode: str,
    lqr_design: dict[str, Any],
    measurement: xr.Dataset,
    beam_observation_mm: np.ndarray | None,
) -> float:
    if correction_mode == CORRECTION_MODE_BEAM:
        if beam_observation_mm is None:
            raise RuntimeError("beam correction observation was not initialized")
        return lqr_projected_observation_residual_from_design(
            lqr_design,
            beam_observation_mm,
        )
    return lqr_projected_residual_from_design(
        lqr_design,
        measurement,
    )


def _lqr_kalman_gated_measurement_warning(
    *,
    innovation_mahalanobis: float,
    innovation_gate: float | None,
) -> str:
    gate_text = "disabled" if innovation_gate is None else f"{innovation_gate:.4g}"
    return (
        "LQR Kalman measurement rejected by innovation gate: "
        f"mahalanobis={innovation_mahalanobis:.4g}, gate={gate_text}; "
        "using predicted state and stopping correction"
    )


def _append_lqr_kalman_diagnostics(
    update: dict[str, np.ndarray | float | bool],
    *,
    iteration_lqr_kalman_state: list[np.ndarray],
    iteration_lqr_kalman_predicted_state: list[np.ndarray],
    iteration_lqr_kalman_innovation: list[np.ndarray],
    iteration_lqr_kalman_innovation_mahalanobis: list[float],
    iteration_lqr_kalman_measurement_accepted: list[bool],
) -> None:
    iteration_lqr_kalman_state.append(np.asarray(update["state"], dtype=np.float64))
    iteration_lqr_kalman_predicted_state.append(
        np.asarray(update["predicted_state"], dtype=np.float64)
    )
    iteration_lqr_kalman_innovation.append(
        np.asarray(update["innovation"], dtype=np.float64).reshape(
            len(CAMERAS),
            len(PIXEL_AXES),
        )
    )
    iteration_lqr_kalman_innovation_mahalanobis.append(
        float(update["innovation_mahalanobis"])
    )
    iteration_lqr_kalman_measurement_accepted.append(
        bool(update["measurement_accepted"])
    )


def _attrs_safe_numeric_config(value: Any) -> str | float:
    if value is None:
        return ""
    array = np.asarray(value, dtype=np.float64)
    if array.ndim == 0:
        return float(array)
    if not np.isfinite(array).all():
        raise ValueError("numeric config attrs must contain only finite values")
    return " ".join(f"{float(item):.17g}" for item in array.reshape(-1))


def _reported_next_correction(
    *,
    converged: bool,
    jacobian: np.ndarray,
    measurement: xr.Dataset,
    axis_scale: np.ndarray,
    active_axis_indices: Sequence[int],
    correction_mode: str,
    beam_observation: np.ndarray | None,
    beam_projection_matrix: np.ndarray | None,
    beam_lqr_weights: Sequence[float] | np.ndarray | None,
    gain: float,
    max_normalized_step: float | None,
    lqr_image_scale_px: float,
    lqr_motor_penalty: float,
    lqr_svd_relative_tolerance: float,
    lqr_kalman_filter_enabled: bool = False,
    lqr_design: dict[str, Any] | None = None,
    lqr_kalman_state: dict[str, np.ndarray | float | bool] | None = None,
    weights: Sequence[float] | np.ndarray | None = None,
) -> np.ndarray:
    if converged:
        return np.zeros(len(COMMAND_AXES), dtype=np.float64)
    if correction_mode == CORRECTION_MODE_BEAM:
        if beam_observation is None or beam_projection_matrix is None:
            raise RuntimeError("beam correction geometry was not initialized")
        raw_correction = _expand_active_command_correction(
            solve_lqr_observation_command_correction(
                np.asarray(beam_projection_matrix, dtype=np.float64)[
                    :, list(active_axis_indices)
                ],
                beam_observation,
                np.asarray(axis_scale, dtype=np.float64)[list(active_axis_indices)],
                gain=gain,
                image_scale=1.0,
                motor_penalty=lqr_motor_penalty,
                svd_relative_tolerance=lqr_svd_relative_tolerance,
                max_normalized_step=max_normalized_step,
                weights=beam_lqr_weights,
            ),
            active_axis_indices,
        )
    elif lqr_kalman_filter_enabled:
        if lqr_design is None or lqr_kalman_state is None:
            raise RuntimeError("LQR Kalman state was not initialized")
        raw_correction = solve_lqr_state_command_correction(
            lqr_design,
            np.asarray(lqr_kalman_state["state"], dtype=np.float64),
            gain=gain,
            max_normalized_step=max_normalized_step,
        )
    else:
        if len(active_axis_indices) != len(COMMAND_AXES):
            raw_correction = _expand_active_command_correction(
                solve_lqr_observation_command_correction(
                    np.asarray(jacobian, dtype=np.float64).reshape(
                        len(CAMERAS) * len(PIXEL_AXES),
                        len(COMMAND_AXES),
                    )[:, list(active_axis_indices)],
                    _measurement_observation(measurement),
                    np.asarray(axis_scale, dtype=np.float64)[list(active_axis_indices)],
                    gain=gain,
                    max_normalized_step=max_normalized_step,
                    image_scale=lqr_image_scale_px,
                    motor_penalty=lqr_motor_penalty,
                    svd_relative_tolerance=lqr_svd_relative_tolerance,
                    weights=weights,
                ),
                active_axis_indices,
            )
        else:
            raw_correction = solve_lqr_command_correction(
                jacobian,
                measurement,
                axis_scale,
                gain=gain,
                max_normalized_step=max_normalized_step,
                image_scale_px=lqr_image_scale_px,
                motor_penalty=lqr_motor_penalty,
                svd_relative_tolerance=lqr_svd_relative_tolerance,
                weights=weights,
            )
    return _validate_readback_correction(raw_correction)


def _build_correction_result(
    *,
    measurement: xr.Dataset,
    jacobian: np.ndarray,
    axis_scale: np.ndarray,
    estimated_offset: np.ndarray,
    next_correction: np.ndarray,
    iteration_shift_px: Sequence[np.ndarray],
    iteration_weighted_residuals: Sequence[float],
    iteration_criterion_residuals: Sequence[float],
    beam_offset_mm: np.ndarray | None,
    iteration_beam_offset_mm: Sequence[np.ndarray],
    analyzer_offset_mm: np.ndarray | None,
    iteration_analyzer_offset_mm: Sequence[np.ndarray],
    iteration_lqr_kalman_state: Sequence[np.ndarray],
    iteration_lqr_kalman_predicted_state: Sequence[np.ndarray],
    iteration_lqr_kalman_innovation: Sequence[np.ndarray],
    iteration_lqr_kalman_innovation_mahalanobis: Sequence[float],
    iteration_lqr_kalman_measurement_accepted: Sequence[bool],
    move_requested_position_mm: Sequence[np.ndarray],
    move_final_readback_position_mm: Sequence[np.ndarray],
    move_gain: Sequence[float],
    move_pre_weighted_residuals: Sequence[float],
    move_post_weighted_residuals: Sequence[float],
    move_predicted_delta_px: Sequence[np.ndarray],
    move_measured_delta_px: Sequence[np.ndarray],
    move_model_residual_delta_px: Sequence[np.ndarray],
    move_predicted_weighted_response_px: Sequence[float],
    move_measured_weighted_response_px: Sequence[float],
    move_feedback_alpha: Sequence[float],
    move_feedback_parallel_px: Sequence[float],
    move_feedback_valid: Sequence[bool],
    move_max_normalized_component: Sequence[float],
    move_active_axis_mask: Sequence[np.ndarray],
    calibration_path: Path,
    correction_history_path: Path,
    correction_run_id: int,
    correction_started_at_utc: str,
    correction_move_started_at: str | None,
    correction_move_finished_at: str | None,
    correction_history_completed: bool,
    converged: bool,
    move_count: int,
    lqr_projected_tolerance: float,
    correction_tolerance: float,
    correction_criterion: str,
    correction_mode: str,
    beam_xz_angle_from_analyzer_deg: float,
    polar_attrs: Mapping[str, float | bool],
    orientation_attrs: Mapping[str, float | bool | str],
    beam_polar_deg: float | None,
    beam_runtime_xz_angle_deg: float | None,
    analyzer_runtime_xz_angle_deg: float | None,
    beam_transverse_tolerance_um: float,
    beam_analyzer_transverse_tolerance_um: float,
    beam_vertical_tolerance_um: float,
    gain: float,
    current_gain: float,
    max_normalized_step: float | None,
    min_command_norm_mm: float,
    max_moves: int,
    lqr_image_scale_px: float,
    lqr_motor_penalty: float,
    lqr_svd_relative_tolerance: float,
    lqr_kalman_filter_enabled: bool,
    lqr_kalman_process_noise: Any,
    lqr_kalman_measurement_noise: float,
    lqr_kalman_measurement_covariance: Any,
    lqr_kalman_initial_covariance: float,
    lqr_kalman_innovation_gate: float | None,
    correction_backlash_enabled: bool,
    initial_readback_position_mm: np.ndarray,
    readback_position_mm: np.ndarray,
    warnings: Sequence[str],
    active_axis_names: Sequence[str],
) -> xr.Dataset:
    result = measurement.assign(
        {
            "estimated_readback_offset_mm": (
                ("command_axis",),
                estimated_offset,
                {"units": "readback-mm"},
            ),
            "correction_readback_delta_mm": (
                ("command_axis",),
                next_correction,
                {"units": "readback-mm"},
            ),
            "axis_scale_readback_mm": (
                ("command_axis",),
                axis_scale,
                {"units": "readback-mm"},
            ),
            "initial_readback_position_mm": (
                ("command_axis",),
                initial_readback_position_mm,
                {"units": "readback-mm"},
            ),
            "final_readback_position_mm": (
                ("command_axis",),
                readback_position_mm,
                {"units": "readback-mm"},
            ),
            "px_per_readback_mm": (
                ("camera", "pixel_axis", "command_axis"),
                jacobian,
                {"units": "px/readback-mm"},
            ),
            "iteration_shift_px": (
                ("iteration", "camera", "pixel_axis"),
                np.stack(iteration_shift_px, axis=0),
                {"units": "px"},
            ),
            "iteration_weighted_residual_px": (
                ("iteration",),
                np.asarray(iteration_weighted_residuals, dtype=np.float64),
                {"units": "px"},
            ),
            "iteration_correction_criterion_residual": (
                ("iteration",),
                np.asarray(iteration_criterion_residuals, dtype=np.float64),
            ),
            "move_requested_position_mm": (
                ("move", "command_axis"),
                _stack_or_empty(move_requested_position_mm),
                {"units": "requested-mm"},
            ),
            "move_final_readback_position_mm": (
                ("move", "command_axis"),
                _stack_or_empty(move_final_readback_position_mm),
                {"units": "readback-mm"},
            ),
            "move_pre_weighted_residual_px": (
                ("move",),
                np.asarray(move_pre_weighted_residuals, dtype=np.float64),
                {"units": "px"},
            ),
            "move_post_weighted_residual_px": (
                ("move",),
                np.asarray(move_post_weighted_residuals, dtype=np.float64),
                {"units": "px"},
            ),
            "move_predicted_delta_px": (
                ("move", "camera", "pixel_axis"),
                _stack_shift_or_empty(move_predicted_delta_px),
                {"units": "px"},
            ),
            "move_measured_delta_px": (
                ("move", "camera", "pixel_axis"),
                _stack_shift_or_empty(move_measured_delta_px),
                {"units": "px"},
            ),
            "move_model_residual_delta_px": (
                ("move", "camera", "pixel_axis"),
                _stack_shift_or_empty(move_model_residual_delta_px),
                {"units": "px"},
            ),
            "move_predicted_weighted_response_px": (
                ("move",),
                np.asarray(move_predicted_weighted_response_px, dtype=np.float64),
                {"units": "px"},
            ),
            "move_measured_weighted_response_px": (
                ("move",),
                np.asarray(move_measured_weighted_response_px, dtype=np.float64),
                {"units": "px"},
            ),
            "move_feedback_alpha": (
                ("move",),
                np.asarray(move_feedback_alpha, dtype=np.float64),
            ),
            "move_feedback_parallel_px": (
                ("move",),
                np.asarray(move_feedback_parallel_px, dtype=np.float64),
                {"units": "px"},
            ),
            "move_feedback_valid": (
                ("move",),
                np.asarray(move_feedback_valid, dtype=bool),
            ),
            "move_gain": (
                ("move",),
                np.asarray(move_gain, dtype=np.float64),
            ),
            "move_max_normalized_component": (
                ("move",),
                np.asarray(move_max_normalized_component, dtype=np.float64),
            ),
            "move_active_axis_mask": (
                ("move", "command_axis"),
                _stack_bool_or_empty(move_active_axis_mask),
            ),
        }
    ).assign_coords(
        command_axis=list(COMMAND_AXES),
        iteration=np.arange(len(iteration_weighted_residuals), dtype=np.int64),
        move=np.arange(move_count, dtype=np.int64),
    )
    if beam_offset_mm is not None:
        result = result.assign(
            {
                "beam_offset_mm": (
                    ("beam_axis",),
                    np.asarray(beam_offset_mm, dtype=np.float64),
                    {"units": "readback-mm"},
                ),
                "iteration_beam_offset_mm": (
                    ("iteration", "beam_axis"),
                    _stack_beam_or_empty(iteration_beam_offset_mm),
                    {"units": "readback-mm"},
                ),
            }
        ).assign_coords(beam_axis=list(BEAM_AXES))
    if analyzer_offset_mm is not None:
        result = result.assign(
            {
                "analyzer_offset_mm": (
                    ("analyzer_axis",),
                    np.asarray(analyzer_offset_mm, dtype=np.float64),
                    {"units": "readback-mm"},
                ),
                "iteration_analyzer_offset_mm": (
                    ("iteration", "analyzer_axis"),
                    _stack_analyzer_or_empty(iteration_analyzer_offset_mm),
                    {"units": "readback-mm"},
                ),
            }
        ).assign_coords(analyzer_axis=list(ANALYZER_AXES))
    if lqr_kalman_filter_enabled:
        result = result.assign(
            {
                "iteration_lqr_kalman_state": (
                    ("iteration", "lqr_state"),
                    _stack_lqr_state_or_empty(iteration_lqr_kalman_state),
                ),
                "iteration_lqr_kalman_predicted_state": (
                    ("iteration", "lqr_state"),
                    _stack_lqr_state_or_empty(iteration_lqr_kalman_predicted_state),
                ),
                "iteration_lqr_kalman_innovation": (
                    ("iteration", "camera", "pixel_axis"),
                    _stack_shift_or_empty(iteration_lqr_kalman_innovation),
                ),
                "iteration_lqr_kalman_innovation_mahalanobis": (
                    ("iteration",),
                    np.asarray(
                        iteration_lqr_kalman_innovation_mahalanobis,
                        dtype=np.float64,
                    ),
                ),
                "iteration_lqr_kalman_measurement_accepted": (
                    ("iteration",),
                    np.asarray(
                        iteration_lqr_kalman_measurement_accepted,
                        dtype=bool,
                    ),
                ),
            }
        ).assign_coords(
            lqr_state=np.arange(
                _lqr_state_count(iteration_lqr_kalman_state),
                dtype=np.int64,
            )
        )
    max_normalized_attr = (
        float(max_normalized_step) if max_normalized_step is not None else np.inf
    )
    attrs: dict[str, Any] = {
        "calibration_path": str(calibration_path),
        "correction_history_path": str(correction_history_path),
        "correction_history_run_id": int(correction_run_id),
        "correction_history_completed": bool(correction_history_completed),
        "correction_started_at_utc": correction_started_at_utc,
        "correction_converged": bool(converged),
        "correction_iterations": int(move_count),
        "correction_mode": correction_mode,
        "correction_active_command_axes": " ".join(active_axis_names),
        "correction_active_command_axis_mask": " ".join(
            "1" if axis in active_axis_names else "0" for axis in COMMAND_AXES
        ),
        "correction_criterion": correction_criterion,
        "correction_tolerance": float(correction_tolerance),
        **_prefixed_polar_attrs("correction", polar_attrs),
        **_prefixed_orientation_attrs("correction", orientation_attrs),
        "correction_gain": float(gain),
        "correction_final_gain": float(current_gain),
        "correction_max_normalized_step": max_normalized_attr,
        "correction_min_command_norm_mm": float(min_command_norm_mm),
        "correction_move_delta_deadband_um": float(
            constants.DEFAULT_CORRECTION_MOVE_DELTA_DEADBAND_UM
        ),
        "correction_backlash_enabled": bool(correction_backlash_enabled),
        "capture_count": int(measurement.attrs.get("capture_count", 1)),
        "capture_aggregation": str(
            measurement.attrs.get(
                "capture_aggregation",
                CAPTURE_AGGREGATION_MEDIAN_SHIFTS,
            )
        ),
        "max_correction_moves": int(max_moves),
        "correction_applied": move_count > 0,
        "warnings": "\n".join(tuple(dict.fromkeys(warnings))),
    }
    if correction_mode == CORRECTION_MODE_BEAM:
        attrs |= {
            "correction_beam_xz_angle_deg": float(beam_xz_angle_from_analyzer_deg),
            "correction_beam_transverse_tolerance_um": float(
                beam_transverse_tolerance_um
            ),
            "correction_beam_analyzer_transverse_tolerance_um": float(
                beam_analyzer_transverse_tolerance_um
            ),
            "correction_beam_vertical_tolerance_um": float(beam_vertical_tolerance_um),
            "correction_beam_observation_axes": " ".join(BEAM_OBSERVATION_AXES),
            "correction_lqr_kalman_disabled_reason": (
                "beam mode uses direct LQR without the camera-space Kalman observer"
            ),
        }
        if beam_polar_deg is not None:
            attrs["correction_polar_deg"] = float(beam_polar_deg)
        if beam_runtime_xz_angle_deg is not None:
            attrs["correction_beam_runtime_xz_angle_deg"] = float(
                beam_runtime_xz_angle_deg
            )
        if analyzer_runtime_xz_angle_deg is not None:
            attrs["correction_analyzer_runtime_xz_angle_deg"] = float(
                analyzer_runtime_xz_angle_deg
            )
    if (
        move_count > 0
        and correction_move_started_at is not None
        and correction_move_finished_at is not None
    ):
        attrs |= {
            "correction_move_started_at": correction_move_started_at,
            "correction_move_finished_at": correction_move_finished_at,
        }
    attrs |= {
        "correction_lqr_image_scale_px": float(lqr_image_scale_px),
        "correction_lqr_motor_penalty": float(lqr_motor_penalty),
        "correction_lqr_svd_relative_tolerance": float(lqr_svd_relative_tolerance),
        "correction_lqr_projected_tolerance": float(lqr_projected_tolerance),
        "correction_lqr_kalman_filter_enabled": bool(lqr_kalman_filter_enabled),
        "correction_lqr_kalman_process_noise": _attrs_safe_numeric_config(
            lqr_kalman_process_noise
        ),
        "correction_lqr_kalman_measurement_noise": float(lqr_kalman_measurement_noise),
        "correction_lqr_kalman_measurement_covariance": (
            _attrs_safe_numeric_config(lqr_kalman_measurement_covariance)
        ),
        "correction_lqr_kalman_initial_covariance": float(
            lqr_kalman_initial_covariance
        ),
        "correction_lqr_kalman_innovation_gate": (
            np.inf
            if lqr_kalman_innovation_gate is None
            else float(lqr_kalman_innovation_gate)
        ),
    }
    return result.assign_attrs(attrs)


def _local_timestamp_iso() -> str:
    return datetime.now().astimezone().isoformat()


def _capture_measurement(
    calibration: xr.Dataset,
    camera_pair: CameraPairPlugin,
    reference_cam0: np.ndarray,
    reference_cam1: np.ndarray,
    capture_count: int,
    capture_aggregation: str = CAPTURE_AGGREGATION_MEDIAN_SHIFTS,
    **shift_kwargs: Any,
) -> xr.Dataset:
    logger.info(
        "Capturing image stack: capture_count=%d, capture_aggregation=%s",
        capture_count,
        capture_aggregation,
    )
    current_cam0, current_cam1 = capture_image_stack(camera_pair, capture_count)
    logger.info(
        "Captured image stacks: cam0_shape=%s, cam1_shape=%s",
        current_cam0.shape,
        current_cam1.shape,
    )
    reference_cam0, current_cam0 = matching_reference_and_stack(
        calibration.attrs,
        "cam0",
        reference_cam0,
        current_cam0,
    )
    reference_cam1, current_cam1 = matching_reference_and_stack(
        calibration.attrs,
        "cam1",
        reference_cam1,
        current_cam1,
    )
    shift_kwargs = _shift_kwargs_with_calibration_ecc_points(
        calibration,
        shift_kwargs,
    )
    logger.info("Measuring image error against calibration references.")
    measurement = measure_image_error(
        reference_cam0,
        current_cam0,
        reference_cam1,
        current_cam1,
        capture_aggregation=capture_aggregation,
        **shift_kwargs,
    )
    logger.info(
        "Measured image shift_px=%s",
        np.asarray(measurement["shift_px"].values, dtype=np.float64).tolist(),
    )
    return measurement


def _shift_kwargs_with_calibration_ecc_points(
    calibration: xr.Dataset,
    shift_kwargs: Mapping[str, Any],
) -> dict[str, Any]:
    kwargs = dict(shift_kwargs)
    if not kwargs.get("use_ecc_refinement"):
        return kwargs
    if "ecc_reference_point_px" in kwargs:
        return kwargs

    attrs = calibration.attrs
    kwargs["ecc_reference_point_px"] = {
        camera: roi_local_point_from_full_frame(
            attrs,
            camera,
            beam_target_point_from_attrs_or_default(attrs, camera),
        )
        for camera in CAMERAS
    }
    return kwargs


def _stack_or_empty(rows: Sequence[np.ndarray]) -> np.ndarray:
    if not rows:
        return np.empty((0, len(COMMAND_AXES)), dtype=np.float64)
    return np.stack(rows, axis=0).astype(np.float64, copy=False)


def _stack_bool_or_empty(rows: Sequence[np.ndarray]) -> np.ndarray:
    if not rows:
        return np.empty((0, len(COMMAND_AXES)), dtype=bool)
    return np.stack(rows, axis=0).astype(bool, copy=False)


def _stack_shift_or_empty(rows: Sequence[np.ndarray]) -> np.ndarray:
    if not rows:
        return np.empty((0, len(CAMERAS), len(PIXEL_AXES)), dtype=np.float64)
    return np.stack(rows, axis=0).astype(np.float64, copy=False)


def _stack_beam_or_empty(rows: Sequence[np.ndarray]) -> np.ndarray:
    if not rows:
        return np.empty((0, len(BEAM_AXES)), dtype=np.float64)
    return np.stack(rows, axis=0).astype(np.float64, copy=False)


def _stack_analyzer_or_empty(rows: Sequence[np.ndarray]) -> np.ndarray:
    if not rows:
        return np.empty((0, len(ANALYZER_AXES)), dtype=np.float64)
    return np.stack(rows, axis=0).astype(np.float64, copy=False)


def _stack_lqr_state_or_empty(rows: Sequence[np.ndarray]) -> np.ndarray:
    if not rows:
        return np.empty((0, 0), dtype=np.float64)
    return np.stack(rows, axis=0).astype(np.float64, copy=False)


def _lqr_state_count(rows: Sequence[np.ndarray]) -> int:
    if not rows:
        return 0
    return int(np.asarray(rows[0]).size)


def _validate_readback_correction(correction_readback_mm: np.ndarray) -> np.ndarray:
    correction = np.asarray(correction_readback_mm, dtype=np.float64).copy()
    if correction.shape != (len(COMMAND_AXES),):
        raise ValueError("correction_readback_mm must have one value for x/y/z")
    if not np.isfinite(correction).all():
        raise ValueError("correction_readback_mm must contain finite values")
    deadband_um = float(constants.DEFAULT_CORRECTION_MOVE_DELTA_DEADBAND_UM)
    if not np.isfinite(deadband_um) or deadband_um < 0.0:
        raise ValueError(
            "DEFAULT_CORRECTION_MOVE_DELTA_DEADBAND_UM must be finite and non-negative"
        )
    deadband_mm = deadband_um / 1000.0
    if deadband_mm > 0.0:
        correction[np.abs(correction) < deadband_mm] = 0.0
    return correction


def _active_command_axis_indices(active_command_axes: Sequence[str] | None) -> tuple[int, ...]:
    if active_command_axes is None:
        return tuple(range(len(COMMAND_AXES)))
    axes = tuple(str(axis) for axis in active_command_axes)
    if not axes:
        raise ValueError("active_command_axes must include at least one axis")
    indices: list[int] = []
    for axis in axes:
        if axis not in COMMAND_AXES:
            raise ValueError(f"unsupported active command axis {axis!r}")
        index = COMMAND_AXES.index(axis)
        if index in indices:
            raise ValueError("active_command_axes must not contain duplicates")
        indices.append(index)
    return tuple(indices)


def _expand_active_command_correction(
    active_correction_readback_mm: Sequence[float] | np.ndarray,
    active_axis_indices: Sequence[int],
) -> np.ndarray:
    active = np.asarray(active_correction_readback_mm, dtype=np.float64)
    indices = tuple(int(index) for index in active_axis_indices)
    if active.shape != (len(indices),):
        raise ValueError("active correction must match active command axes")
    if not np.isfinite(active).all():
        raise ValueError("active correction must contain finite values")
    correction = np.zeros(len(COMMAND_AXES), dtype=np.float64)
    correction[list(indices)] = active
    return correction


def _measurement_observation(measurement: xr.Dataset) -> np.ndarray:
    values = np.asarray(measurement["shift_px"].values, dtype=np.float64)
    if values.shape != (len(CAMERAS), len(PIXEL_AXES)):
        raise ValueError("measurement shift_px has unexpected shape")
    if not np.isfinite(values).all():
        raise ValueError("measurement shift_px must contain finite values")
    return values.reshape(len(CAMERAS) * len(PIXEL_AXES))


def _correction_stop_warning(
    *,
    raw_correction_readback_mm: np.ndarray,
    min_command_norm_mm: float,
) -> str:
    raw_norm_mm = float(np.linalg.norm(raw_correction_readback_mm))
    if raw_norm_mm <= min_command_norm_mm:
        return (
            "computed correction step is below the minimum move norm "
            f"{min_command_norm_mm:.4g} mm; stopping before another move"
        )
    return (
        "computed correction step has no active axes after applying the "
        f"{constants.DEFAULT_CORRECTION_MOVE_DELTA_DEADBAND_UM:.4g} um "
        "move-delta deadband; stopping before another move"
    )


def _active_correction_indices(correction_readback_mm: np.ndarray) -> tuple[int, ...]:
    return tuple(
        index
        for index, value in enumerate(
            np.asarray(correction_readback_mm, dtype=np.float64)
        )
        if value != 0.0
    )

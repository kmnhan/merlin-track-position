from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
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
    COMMAND_AXES,
    PIXEL_AXES,
    compute_lqr_correction_design,
    estimate_command_offset,
    initialize_lqr_kalman_state,
    load_calibration_dataset,
    lqr_projected_residual_from_design,
    measure_image_error,
    predict_lqr_kalman_state,
    solve_lqr_command_correction,
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
from merlin_track_position.tracking.roi import matching_reference_and_stack

logger = logging.getLogger("merlin_track_position.tracking.correct")

CORRECTION_CRITERION = "lqr_projected_normalized_error"
_USE_DEFAULT = object()


class CorrectionFeedback(NamedTuple):
    predicted_weighted_response_px: float
    measured_weighted_response_px: float
    alpha: float
    parallel_px: float


class CorrectionMotorBackend(Protocol):
    def get_positions(self, motor_aliases: Sequence[str]) -> tuple[float, ...]:
        """Return commanded motor positions in the order requested."""

    def move_motors_and_wait(
        self,
        motor_aliases: Sequence[str],
        goals: Sequence[float],
        *,
        max_retries: int = 4,
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
        max_retries: int = 4,
        backlash_correction: dict[str, float] | None = None,
        move_timeout_s: float = 60.0,
    ) -> tuple[float, ...]:
        return move_motors_and_wait(
            motor_aliases,
            goals,
            max_retries=max_retries,
            backlash_correction=backlash_correction,
            move_timeout_s=move_timeout_s,
        )


def do_correction(
    calibration: xr.Dataset | str | Path,
    camera_pair: CameraPairPlugin | None = None,
    *,
    calibration_path: str | Path | None = None,
    max_retries: int = 4,
    capture_count: int = constants.DEFAULT_CAPTURE_COUNT,
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
    weights: Sequence[float] | np.ndarray | None = None,
    progress_callback: Callable[[xr.Dataset], None] | None = None,
    motor_backend: CorrectionMotorBackend | None = None,
    **shift_kwargs: Any,
) -> xr.Dataset:
    """Run LQR closed-loop visual-servo correction in commanded-mm space."""

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
    weights = _correction_weights(weights)
    if gain is _USE_DEFAULT:
        gain = constants.DEFAULT_LQR_CORRECTION_GAIN
    if max_normalized_step is _USE_DEFAULT:
        max_normalized_step = constants.DEFAULT_LQR_CORRECTION_MAX_NORMALIZED_STEP
    lqr_projected_tolerance = _resolve_lqr_projected_tolerance(
        lqr_projected_tolerance
    )
    correction_tolerance = lqr_projected_tolerance
    correction_criterion = CORRECTION_CRITERION
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
    lqr_kalman_filter_enabled = bool(constants.DEFAULT_LQR_CORRECTION_USE_KALMAN_FILTER)
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
    lqr_kalman_innovation_gate = (
        constants.DEFAULT_LQR_CORRECTION_KALMAN_INNOVATION_GATE
    )
    logger.info(
        "Correction parameters: capture_count=%d, criterion=%s, tolerance=%g, "
        "gain=%g, max_moves=%d",
        capture_count,
        correction_criterion,
        correction_tolerance,
        current_gain,
        max_moves,
    )

    reference_cam0 = np.asarray(calibration["reference_cam0"].values)
    reference_cam1 = np.asarray(calibration["reference_cam1"].values)
    axis_scale = np.asarray(calibration["axis_scale_cmd_mm"].values, dtype=np.float64)
    jacobian = np.asarray(
        calibration["px_per_cmd_mm"].values,
        dtype=np.float64,
    )
    lqr_design = compute_lqr_correction_design(
        jacobian.reshape(len(CAMERAS) * len(PIXEL_AXES), len(COMMAND_AXES)),
        axis_scale,
        image_scale_px=lqr_image_scale_px,
        motor_penalty=lqr_motor_penalty,
        svd_relative_tolerance=lqr_svd_relative_tolerance,
        weights=weights,
    )
    logger.info("Reading initial commanded x/y/z positions.")
    commanded_position_mm = np.asarray(
        motor_backend.get_positions(COMMAND_AXES),
        dtype=np.float64,
    )
    if commanded_position_mm.shape != (len(COMMAND_AXES),):
        raise ValueError("initial command position readback must have x/y/z values")
    logger.info("Initial commanded positions: %s", commanded_position_mm.tolist())
    initial_commanded_position_mm = commanded_position_mm.copy()
    correction_log_path = correction_history_path(resolved_path)
    correction_run_id = _next_correction_history_run_id(correction_log_path)
    correction_started_at_utc = datetime.now(UTC).isoformat()
    correction_move_started_at: str | None = None
    correction_move_finished_at: str | None = None

    logger.info("Capturing initial correction measurement.")
    measurement = _capture_measurement(
        calibration,
        camera_pair,
        reference_cam0,
        reference_cam1,
        capture_count,
        **shift_kwargs,
    )
    weighted_residual = weighted_pixel_residual(measurement, weights=weights)
    criterion_residual = lqr_projected_residual_from_design(
        lqr_design,
        measurement,
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
    move_command_delta_mm: list[np.ndarray] = []
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

    move_count = 0
    converged = criterion_residual <= correction_tolerance

    def build_result(completed: bool) -> xr.Dataset:
        return _build_correction_result(
            measurement=measurement,
            jacobian=jacobian,
            axis_scale=axis_scale,
            estimated_offset=estimate_command_offset(
                jacobian,
                measurement,
                weights=weights,
            ),
            next_correction=_reported_next_correction(
                converged=converged,
                jacobian=jacobian,
                measurement=measurement,
                axis_scale=axis_scale,
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
            move_command_delta_mm=move_command_delta_mm,
            move_requested_position_mm=move_requested_position_mm,
            move_final_readback_position_mm=move_final_readback_position_mm,
            move_gain=move_gain,
            move_pre_weighted_residuals=move_pre_weighted_residuals,
            move_post_weighted_residuals=move_post_weighted_residuals,
            move_predicted_delta_px=move_predicted_delta_px,
            move_measured_delta_px=move_measured_delta_px,
            move_model_residual_delta_px=move_model_residual_delta_px,
            move_predicted_weighted_response_px=(
                move_predicted_weighted_response_px
            ),
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
            initial_commanded_position_mm=initial_commanded_position_mm,
            commanded_position_mm=commanded_position_mm,
            warnings=warnings,
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
        if lqr_kalman_filter_enabled:
            if lqr_design is None or lqr_kalman_state is None:
                raise RuntimeError("LQR Kalman state was not initialized")
            raw_correction_cmd_mm = solve_lqr_state_command_correction(
                lqr_design,
                np.asarray(lqr_kalman_state["state"], dtype=np.float64),
                gain=gain_used,
                max_normalized_step=max_normalized_step,
            )
        else:
            raw_correction_cmd_mm = solve_lqr_command_correction(
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
        correction_cmd_mm = _validate_command_correction(raw_correction_cmd_mm)
        correction_norm_mm = float(np.linalg.norm(correction_cmd_mm))
        active_indices = _active_correction_indices(correction_cmd_mm)
        logger.info(
            "Computed correction command: estimated_offset_mm=%s, raw_delta_mm=%s, "
            "delta_mm=%s, norm_mm=%.6g, active_axes=%s",
            estimated_offset_mm.tolist(),
            raw_correction_cmd_mm.tolist(),
            correction_cmd_mm.tolist(),
            correction_norm_mm,
            tuple(COMMAND_AXES[index] for index in active_indices),
        )
        if correction_norm_mm <= min_command_norm_mm or not active_indices:
            warnings.append(
                _correction_stop_warning(
                    raw_correction_cmd_mm=raw_correction_cmd_mm,
                    correction_cmd_mm=correction_cmd_mm,
                    min_command_norm_mm=min_command_norm_mm,
                )
            )
            logger.info(
                "Stopping correction before move: %s",
                warnings[-1],
            )
            break

        lqr_kalman_prediction: dict[str, np.ndarray] | None = None
        if lqr_kalman_filter_enabled:
            if lqr_design is None or lqr_kalman_state is None:
                raise RuntimeError("LQR Kalman state was not initialized")
            lqr_kalman_prediction = predict_lqr_kalman_state(
                np.asarray(lqr_kalman_state["state"], dtype=np.float64),
                np.asarray(lqr_kalman_state["covariance"], dtype=np.float64),
                correction_cmd_mm,
                lqr_design,
                process_noise=lqr_kalman_process_noise,
            )

        predicted_delta_px = (
            jacobian_before.reshape(len(CAMERAS) * len(PIXEL_AXES), len(COMMAND_AXES))
            @ correction_cmd_mm
        ).reshape(len(CAMERAS), len(PIXEL_AXES))
        normalized_component = correction_cmd_mm / axis_scale
        requested_position_mm = commanded_position_mm + correction_cmd_mm
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
        motor_backend.move_motors_and_wait(
            active_axes,
            active_requested_position_mm,
            max_retries=max_retries,
            # Correction moves are already closed-loop and can be micron-scale.
            # Do not expand a small correction into a large backlash pre-position.
            backlash_correction={},
        )
        correction_move_finished_at = _local_timestamp_iso()
        logger.info("Correction motor move returned; reading final x/y/z positions.")
        final_readback_mm = np.asarray(
            motor_backend.get_positions(COMMAND_AXES),
            dtype=np.float64,
        )
        if final_readback_mm.shape != (len(COMMAND_AXES),):
            raise ValueError("final command position readback must have x/y/z values")
        logger.info("Final readback positions: %s", final_readback_mm.tolist())
        commanded_position_mm = requested_position_mm

        logger.info("Capturing post-move correction measurement.")
        after_measurement = _capture_measurement(
            calibration,
            camera_pair,
            reference_cam0,
            reference_cam1,
            capture_count,
            **shift_kwargs,
        )
        after_weighted_residual = weighted_pixel_residual(
            after_measurement,
            weights=weights,
        )
        after_criterion_residual = lqr_projected_residual_from_design(
            lqr_design,
            after_measurement,
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
            logger.info("Residual did not decrease; keeping gain fixed at %g.", current_gain)

        move_command_delta_mm.append(correction_cmd_mm)
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
        criterion_residual = after_criterion_residual
        iteration_shift_px.append(
            np.asarray(measurement["shift_px"].values, dtype=np.float64)
        )
        iteration_weighted_residuals.append(float(weighted_residual))
        iteration_criterion_residuals.append(float(criterion_residual))
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

    if not converged:
        warnings.append(
            "correction did not converge within "
            f"{max_moves} move(s); final {correction_criterion} "
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
            del history_file[group_name]
        history_file.attrs["format"] = "merlin_track_position_correction_history"
        history_file.attrs["latest_run_group"] = group_name
        history_file.attrs["latest_run_id"] = int(run_id)
        history_file.attrs["calibration_path"] = str(
            saved.attrs.get("calibration_path", "")
        )

    saved.to_netcdf(
        output_path,
        engine="h5netcdf",
        mode="a",
        group=group_name,
        encoding=hdf5_image_encoding(saved),
    )
    return output_path


def save_correction_history_dataset_deferred(
    result: xr.Dataset,
    path: str | Path,
    *,
    run_id: int,
) -> PersistenceResult:
    """Save correction history through a local spool before appending to HDF5."""

    output_path = normalize_target_path(path)
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
            "completed": bool(saved.attrs.get("correction_history_completed", False)),
        },
    )
    try:
        save_correction_history_dataset(result, output_path, run_id=run_id)
    except Exception as exc:
        return PersistenceResult(
            target_path=output_path,
            spool_path=entry.path,
            flushed=False,
            pending=True,
            message=(
                f"queued correction history write because target is unavailable: {exc}"
            ),
        )

    discard_spool_entry(entry)
    _discard_pending_correction_history_run(output_path, int(run_id))
    return PersistenceResult(
        target_path=output_path,
        spool_path=entry.path,
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


def _latest_correction_history_run_id(path: Path) -> int | None:
    if not path.exists():
        return None
    with h5py.File(path, "r") as history_file:
        latest_attr = history_file.attrs.get("latest_run_group")
        if isinstance(latest_attr, bytes):
            latest_attr = latest_attr.decode()
        latest_run_id = (
            _correction_history_run_id(latest_attr)
            if isinstance(latest_attr, str) and latest_attr in history_file
            else None
        )
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
) -> None:
    for entry in iter_pending_entries(
        operation="correction_history",
        target_path=path,
    ):
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


def _resolve_lqr_projected_tolerance(value: Any) -> float:
    if value is _USE_DEFAULT:
        value = constants.DEFAULT_LQR_CORRECTION_PROJECTED_TOLERANCE
    tolerance = float(value)
    if not np.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("lqr_projected_tolerance must be finite and non-negative")
    return tolerance


def _lqr_kalman_gated_measurement_warning(
    *,
    innovation_mahalanobis: float,
    innovation_gate: float | None,
) -> str:
    gate_text = "disabled" if innovation_gate is None else f"{innovation_gate:.4g}"
    return (
        "LQR Kalman measurement rejected by innovation gate: "
        f"mahalanobis={innovation_mahalanobis:.4g}, gate={gate_text}; "
        "using predicted state"
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
    iteration_lqr_kalman_state.append(
        np.asarray(update["state"], dtype=np.float64)
    )
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
    if lqr_kalman_filter_enabled:
        if lqr_design is None or lqr_kalman_state is None:
            raise RuntimeError("LQR Kalman state was not initialized")
        raw_correction = solve_lqr_state_command_correction(
            lqr_design,
            np.asarray(lqr_kalman_state["state"], dtype=np.float64),
            gain=gain,
            max_normalized_step=max_normalized_step,
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
    return _validate_command_correction(raw_correction)


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
    iteration_lqr_kalman_state: Sequence[np.ndarray],
    iteration_lqr_kalman_predicted_state: Sequence[np.ndarray],
    iteration_lqr_kalman_innovation: Sequence[np.ndarray],
    iteration_lqr_kalman_innovation_mahalanobis: Sequence[float],
    iteration_lqr_kalman_measurement_accepted: Sequence[bool],
    move_command_delta_mm: Sequence[np.ndarray],
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
    initial_commanded_position_mm: np.ndarray,
    commanded_position_mm: np.ndarray,
    warnings: Sequence[str],
) -> xr.Dataset:
    result = measurement.assign(
        {
            "estimated_command_offset_mm": (
                ("command_axis",),
                estimated_offset,
                {"units": "commanded-mm"},
            ),
            "correction_cmd_mm": (
                ("command_axis",),
                next_correction,
                {"units": "commanded-mm"},
            ),
            "axis_scale_cmd_mm": (
                ("command_axis",),
                axis_scale,
                {"units": "commanded-mm"},
            ),
            "initial_commanded_position_mm": (
                ("command_axis",),
                initial_commanded_position_mm,
                {"units": "commanded-mm"},
            ),
            "final_commanded_position_mm": (
                ("command_axis",),
                commanded_position_mm,
                {"units": "commanded-mm"},
            ),
            "px_per_cmd_mm": (
                ("camera", "pixel_axis", "command_axis"),
                jacobian,
                {"units": "px/commanded-mm"},
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
            "move_command_delta_mm": (
                ("move", "command_axis"),
                _stack_or_empty(move_command_delta_mm),
                {"units": "commanded-mm"},
            ),
            "move_requested_position_mm": (
                ("move", "command_axis"),
                _stack_or_empty(move_requested_position_mm),
                {"units": "commanded-mm"},
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
    if lqr_kalman_filter_enabled:
        result = result.assign(
            {
                "iteration_lqr_kalman_state": (
                    ("iteration", "lqr_state"),
                    _stack_lqr_state_or_empty(iteration_lqr_kalman_state),
                ),
                "iteration_lqr_kalman_predicted_state": (
                    ("iteration", "lqr_state"),
                    _stack_lqr_state_or_empty(
                        iteration_lqr_kalman_predicted_state
                    ),
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
        "correction_criterion": correction_criterion,
        "correction_tolerance": float(correction_tolerance),
        "correction_gain": float(gain),
        "correction_final_gain": float(current_gain),
        "correction_max_normalized_step": max_normalized_attr,
        "correction_min_command_norm_mm": float(min_command_norm_mm),
        "max_correction_moves": int(max_moves),
        "correction_applied": move_count > 0,
        "warnings": "\n".join(tuple(dict.fromkeys(warnings))),
    }
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
        "correction_lqr_svd_relative_tolerance": float(
            lqr_svd_relative_tolerance
        ),
        "correction_lqr_projected_tolerance": float(lqr_projected_tolerance),
        "correction_lqr_kalman_filter_enabled": bool(lqr_kalman_filter_enabled),
        "correction_lqr_kalman_process_noise": _attrs_safe_numeric_config(
            lqr_kalman_process_noise
        ),
        "correction_lqr_kalman_measurement_noise": float(
            lqr_kalman_measurement_noise
        ),
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
    **shift_kwargs: Any,
) -> xr.Dataset:
    logger.info("Capturing image stack: capture_count=%d", capture_count)
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
    logger.info("Measuring image error against calibration references.")
    measurement = measure_image_error(
        reference_cam0,
        current_cam0,
        reference_cam1,
        current_cam1,
        **shift_kwargs,
    )
    logger.info(
        "Measured image shift_px=%s",
        np.asarray(measurement["shift_px"].values, dtype=np.float64).tolist(),
    )
    return measurement


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


def _stack_lqr_state_or_empty(rows: Sequence[np.ndarray]) -> np.ndarray:
    if not rows:
        return np.empty((0, 0), dtype=np.float64)
    return np.stack(rows, axis=0).astype(np.float64, copy=False)


def _lqr_state_count(rows: Sequence[np.ndarray]) -> int:
    if not rows:
        return 0
    return int(np.asarray(rows[0]).size)


def _validate_command_correction(correction_cmd_mm: np.ndarray) -> np.ndarray:
    correction = np.asarray(correction_cmd_mm, dtype=np.float64).copy()
    if correction.shape != (len(COMMAND_AXES),):
        raise ValueError("correction_cmd_mm must have one value for x/y/z")
    if not np.isfinite(correction).all():
        raise ValueError("correction_cmd_mm must contain finite values")
    return correction


def _correction_stop_warning(
    *,
    raw_correction_cmd_mm: np.ndarray,
    correction_cmd_mm: np.ndarray,
    min_command_norm_mm: float,
) -> str:
    raw_norm_mm = float(np.linalg.norm(raw_correction_cmd_mm))
    if raw_norm_mm <= min_command_norm_mm:
        return (
            "computed correction step is below the minimum command norm "
            f"{min_command_norm_mm:.4g} mm; stopping before another move"
        )
    return "computed correction step has no active axes; stopping before another move"


def _active_correction_indices(correction_cmd_mm: np.ndarray) -> tuple[int, ...]:
    return tuple(
        index
        for index, value in enumerate(np.asarray(correction_cmd_mm, dtype=np.float64))
        if value != 0.0
    )

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, NamedTuple, Protocol

import h5py
import numpy as np
import xarray as xr

from merlin_track_position import constants
from merlin_track_position.instruments.cameras import (
    CameraPairPlugin,
    RoiGeometry,
    capture_image_stack,
    crop_image_to_roi,
    default_camera_pair,
    normalize_capture_count,
)
from merlin_track_position.instruments.motors import get_positions, move_motors_and_wait
from merlin_track_position.tracking.calibration_core import (
    CAMERAS,
    COMMAND_AXES,
    OBSERVATION_AXES,
    PIXEL_AXES,
    estimate_command_offset,
    load_calibration_dataset,
    measure_image_error,
    refine_visual_jacobian_from_observations,
    solve_damped_command_correction,
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

logger = logging.getLogger("merlin_track_position.tracking.correct")

ROI_ATTR_KEYS: dict[str, tuple[str, str, str, str]] = {
    camera: (
        f"roi_{camera}_x",
        f"roi_{camera}_y",
        f"roi_{camera}_width",
        f"roi_{camera}_height",
    )
    for camera in CAMERAS
}


class CorrectionFeedback(NamedTuple):
    predicted_weighted_response_px: float
    measured_weighted_response_px: float
    alpha: float
    parallel_px: float


@dataclass(frozen=True)
class CorrectionState:
    commanded_position_mm: np.ndarray
    measurement: xr.Dataset
    residual_px: float


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
    pixel_tolerance_px: float = constants.DEFAULT_CORRECTION_PIXEL_TOLERANCE_PX,
    target_residual_px: float | None = None,
    min_verified_improvement_px: float = 0.03,
    max_duration_s: float = 120.0,
    gain: float = constants.DEFAULT_CORRECTION_GAIN,
    min_gain: float = constants.DEFAULT_CORRECTION_MIN_GAIN,
    damping_mu: float = constants.DEFAULT_CORRECTION_DAMPING_MU,
    max_normalized_step: float | None = (
        constants.DEFAULT_CORRECTION_MAX_NORMALIZED_STEP
    ),
    min_axis_predicted_shift_px: float = (
        constants.DEFAULT_CORRECTION_MIN_AXIS_PREDICTED_SHIFT_PX
    ),
    min_total_predicted_shift_px: float = (
        constants.DEFAULT_CORRECTION_MIN_TOTAL_PREDICTED_SHIFT_PX
    ),
    min_feedback_alpha: float = constants.DEFAULT_CORRECTION_MIN_FEEDBACK_ALPHA,
    min_feedback_parallel_shift_px: float = (
        constants.DEFAULT_CORRECTION_MIN_FEEDBACK_PARALLEL_SHIFT_PX
    ),
    min_command_norm_mm: float = constants.DEFAULT_CORRECTION_MIN_COMMAND_NORM_MM,
    max_moves: int = constants.DEFAULT_CORRECTION_MAX_MOVES,
    local_probe_mode: str = "fallback",
    local_probe_target_response_px: float = 1.0,
    return_to_best_on_failure: bool = True,
    verify_return_to_best: bool = True,
    weights: Sequence[float] | np.ndarray | None = None,
    progress_callback: Callable[[xr.Dataset], None] | None = None,
    motor_backend: CorrectionMotorBackend | None = None,
    **shift_kwargs: Any,
) -> xr.Dataset:
    """Run guarded closed-loop visual-servo correction in commanded-mm space.

    The calibration must have a backing file. Accepted Jacobian refinements rewrite that
    calibration file so future corrections continue from the refined Jacobian.
    """

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

    pixel_tolerance_px = float(pixel_tolerance_px)
    if not np.isfinite(pixel_tolerance_px) or pixel_tolerance_px < 0.0:
        raise ValueError("pixel_tolerance_px must be finite and non-negative")
    if target_residual_px is None:
        target_residual_px = pixel_tolerance_px
    target_residual_px = float(target_residual_px)
    min_verified_improvement_px = float(min_verified_improvement_px)
    max_duration_s = float(max_duration_s)
    current_gain = float(gain)
    min_gain = float(min_gain)
    current_mu = float(damping_mu)
    min_total_predicted_shift_px = float(min_total_predicted_shift_px)
    min_feedback_alpha = float(min_feedback_alpha)
    min_feedback_parallel_shift_px = float(min_feedback_parallel_shift_px)
    min_command_norm_mm = float(min_command_norm_mm)
    local_probe_target_response_px = float(local_probe_target_response_px)
    max_moves = int(max_moves)
    local_probe_mode = str(local_probe_mode).strip().lower()
    return_to_best_on_failure = bool(return_to_best_on_failure)
    verify_return_to_best = bool(verify_return_to_best)
    if not np.isfinite(target_residual_px) or target_residual_px < 0.0:
        raise ValueError("target_residual_px must be finite and non-negative")
    if (
        not np.isfinite(min_verified_improvement_px)
        or min_verified_improvement_px < 0.0
    ):
        raise ValueError(
            "min_verified_improvement_px must be finite and non-negative"
        )
    if not np.isfinite(max_duration_s) or max_duration_s < 0.0:
        raise ValueError("max_duration_s must be finite and non-negative")
    if not np.isfinite(current_gain) or current_gain <= 0.0:
        raise ValueError("gain must be finite and positive")
    if not np.isfinite(min_gain) or min_gain <= 0.0:
        raise ValueError("min_gain must be finite and positive")
    if current_gain < min_gain:
        current_gain = min_gain
    if not np.isfinite(current_mu) or current_mu < 0.0:
        raise ValueError("damping_mu must be finite and non-negative")
    if (
        not np.isfinite(min_total_predicted_shift_px)
        or min_total_predicted_shift_px < 0.0
    ):
        raise ValueError(
            "min_total_predicted_shift_px must be finite and non-negative"
        )
    if not np.isfinite(min_feedback_alpha) or min_feedback_alpha < 0.0:
        raise ValueError("min_feedback_alpha must be finite and non-negative")
    if (
        not np.isfinite(min_feedback_parallel_shift_px)
        or min_feedback_parallel_shift_px < 0.0
    ):
        raise ValueError(
            "min_feedback_parallel_shift_px must be finite and non-negative"
        )
    if not np.isfinite(min_command_norm_mm) or min_command_norm_mm < 0.0:
        raise ValueError("min_command_norm_mm must be finite and non-negative")
    if (
        not np.isfinite(local_probe_target_response_px)
        or local_probe_target_response_px <= 0.0
    ):
        raise ValueError("local_probe_target_response_px must be finite and positive")
    if max_moves < 0:
        raise ValueError("max_moves must be >= 0")
    if local_probe_mode not in {"off", "fallback", "always"}:
        raise ValueError("local_probe_mode must be 'off', 'fallback', or 'always'")
    logger.info(
        "Correction parameters: capture_count=%d, target_residual_px=%g, "
        "min_verified_improvement_px=%g, max_duration_s=%g, gain=%g, "
        "min_gain=%g, damping_mu=%g, min_total_predicted_shift_px=%g, "
        "min_feedback_alpha=%g, min_feedback_parallel_shift_px=%g, "
        "max_moves=%d, local_probe_mode=%s",
        capture_count,
        target_residual_px,
        min_verified_improvement_px,
        max_duration_s,
        current_gain,
        min_gain,
        current_mu,
        min_total_predicted_shift_px,
        min_feedback_alpha,
        min_feedback_parallel_shift_px,
        max_moves,
        local_probe_mode,
    )

    reference_cam0 = np.asarray(calibration["reference_cam0"].values)
    reference_cam1 = np.asarray(calibration["reference_cam1"].values)
    axis_scale = np.asarray(calibration["axis_scale_cmd_mm"].values, dtype=np.float64)
    calibration_jacobian = np.asarray(
        calibration["visual_jacobian_px_per_cmd_mm"].values,
        dtype=np.float64,
    )
    working_jacobian = calibration_jacobian.copy()
    latest_local_jacobian: np.ndarray | None = None
    correction_started_monotonic = time.monotonic()

    logger.info("Reading initial commanded x/y/z positions.")
    initial_commanded_position_mm = np.asarray(
        motor_backend.get_positions(COMMAND_AXES),
        dtype=np.float64,
    )
    if initial_commanded_position_mm.shape != (len(COMMAND_AXES),):
        raise ValueError("initial command position readback must have x/y/z values")
    logger.info("Initial commanded positions: %s", initial_commanded_position_mm.tolist())
    correction_log_path = correction_history_path(resolved_path)
    correction_run_id = _next_correction_history_run_id(correction_log_path)
    correction_started_at_utc = datetime.now(UTC).isoformat()

    logger.info("Capturing initial correction measurement.")
    initial_measurement = _capture_measurement(
        calibration,
        camera_pair,
        reference_cam0,
        reference_cam1,
        capture_count,
        **shift_kwargs,
    )
    initial_residual = weighted_pixel_residual(initial_measurement, weights=weights)
    logger.info("Initial weighted residual: %.6g px", initial_residual)

    current_state = CorrectionState(
        commanded_position_mm=initial_commanded_position_mm.copy(),
        measurement=initial_measurement,
        residual_px=float(initial_residual),
    )
    best_state = current_state
    converged = current_state.residual_px <= target_residual_px
    stop_reason = "target_residual_reached" if converged else ""
    returned_to_best = False
    return_to_best_verified = False

    iteration_shift_px = [_measurement_shift_px(current_state.measurement)]
    iteration_weighted_residuals = [current_state.residual_px]
    move_command_delta_mm: list[np.ndarray] = []
    move_requested_position_mm: list[np.ndarray] = []
    move_final_readback_position_mm: list[np.ndarray] = []
    move_gain: list[float] = []
    move_damping_mu: list[float] = []
    move_pre_weighted_residuals: list[float] = []
    move_post_weighted_residuals: list[float] = []
    move_predicted_delta_px: list[np.ndarray] = []
    move_measured_delta_px: list[np.ndarray] = []
    move_predicted_weighted_response_px: list[float] = []
    move_measured_weighted_response_px: list[float] = []
    move_feedback_alpha: list[float] = []
    move_feedback_parallel_px: list[float] = []
    move_feedback_valid: list[bool] = []
    move_jacobian_before: list[np.ndarray] = []
    move_jacobian_after: list[np.ndarray] = []
    move_max_normalized_component: list[float] = []
    move_active_axis_mask: list[np.ndarray] = []
    move_jacobian_refined: list[bool] = []
    move_kind: list[str] = []
    move_accepted: list[bool] = []
    move_is_best_after: list[bool] = []
    move_verified_residuals: list[float] = []
    move_local_probe_jacobian: list[np.ndarray | None] = []
    warnings: list[str] = [
        line.strip()
        for line in str(current_state.measurement.attrs.get("warnings", "")).splitlines()
        if line.strip()
    ]

    def move_count() -> int:
        return len(move_command_delta_mm)

    def duration_exhausted() -> bool:
        return (time.monotonic() - correction_started_monotonic) >= max_duration_s

    def move_budget_exhausted() -> bool:
        return move_count() >= max_moves

    def should_try_local_probe() -> bool:
        return local_probe_mode in {"fallback", "always"}

    def build_result(completed: bool) -> xr.Dataset:
        return _build_correction_result(
            measurement=current_state.measurement,
            jacobian=working_jacobian,
            axis_scale=axis_scale,
            estimated_offset=estimate_command_offset(
                working_jacobian,
                current_state.measurement,
                weights=weights,
            ),
            next_correction=_reported_next_correction(
                converged=converged,
                jacobian=working_jacobian,
                measurement=current_state.measurement,
                axis_scale=axis_scale,
                gain=current_gain,
                damping_mu=current_mu,
                max_normalized_step=max_normalized_step,
                min_axis_predicted_shift_px=min_axis_predicted_shift_px,
                weights=weights,
            ),
            iteration_shift_px=iteration_shift_px,
            iteration_weighted_residuals=iteration_weighted_residuals,
            move_command_delta_mm=move_command_delta_mm,
            move_requested_position_mm=move_requested_position_mm,
            move_final_readback_position_mm=move_final_readback_position_mm,
            move_gain=move_gain,
            move_damping_mu=move_damping_mu,
            move_pre_weighted_residuals=move_pre_weighted_residuals,
            move_post_weighted_residuals=move_post_weighted_residuals,
            move_predicted_delta_px=move_predicted_delta_px,
            move_measured_delta_px=move_measured_delta_px,
            move_predicted_weighted_response_px=(
                move_predicted_weighted_response_px
            ),
            move_measured_weighted_response_px=move_measured_weighted_response_px,
            move_feedback_alpha=move_feedback_alpha,
            move_feedback_parallel_px=move_feedback_parallel_px,
            move_feedback_valid=move_feedback_valid,
            move_jacobian_before=move_jacobian_before,
            move_jacobian_after=move_jacobian_after,
            move_max_normalized_component=move_max_normalized_component,
            move_active_axis_mask=move_active_axis_mask,
            move_jacobian_refined=move_jacobian_refined,
            move_kind=move_kind,
            move_accepted=move_accepted,
            move_is_best_after=move_is_best_after,
            move_verified_residuals=move_verified_residuals,
            move_local_probe_jacobian=move_local_probe_jacobian,
            calibration_path=resolved_path,
            correction_history_path=correction_log_path,
            correction_run_id=correction_run_id,
            correction_started_at_utc=correction_started_at_utc,
            correction_history_completed=completed,
            converged=converged,
            move_count=move_count(),
            pixel_tolerance_px=pixel_tolerance_px,
            target_residual_px=target_residual_px,
            min_verified_improvement_px=min_verified_improvement_px,
            max_duration_s=max_duration_s,
            gain=gain,
            min_gain=min_gain,
            damping_mu=damping_mu,
            current_gain=current_gain,
            current_mu=current_mu,
            max_normalized_step=max_normalized_step,
            min_axis_predicted_shift_px=min_axis_predicted_shift_px,
            min_total_predicted_shift_px=min_total_predicted_shift_px,
            min_feedback_alpha=min_feedback_alpha,
            min_feedback_parallel_shift_px=min_feedback_parallel_shift_px,
            min_command_norm_mm=min_command_norm_mm,
            max_moves=max_moves,
            initial_commanded_position_mm=initial_commanded_position_mm,
            commanded_position_mm=current_state.commanded_position_mm,
            best_commanded_position_mm=best_state.commanded_position_mm,
            best_verified_residual_px=best_state.residual_px,
            final_verified_residual_px=current_state.residual_px,
            local_probe_mode=local_probe_mode,
            local_probe_target_response_px=local_probe_target_response_px,
            returned_to_best=returned_to_best,
            return_to_best_verified=return_to_best_verified,
            correction_stop_reason=stop_reason,
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

    def record_verified_move(
        *,
        kind: str,
        requested_position_mm: np.ndarray,
        jacobian_before: np.ndarray,
        predicted_delta_px: np.ndarray,
        normalized_component: np.ndarray,
        gain_used: float,
        mu_used: float,
        local_jacobian_for_log: np.ndarray | None,
        allow_over_budget: bool = False,
    ) -> CorrectionFeedback | None:
        nonlocal current_state, best_state, returned_to_best, return_to_best_verified

        if not allow_over_budget and (move_budget_exhausted() or duration_exhausted()):
            return None
        pre_state = current_state
        command_delta_mm = requested_position_mm - pre_state.commanded_position_mm
        active_indices = _active_correction_indices(command_delta_mm)
        if not active_indices:
            return None
        active_axes = tuple(COMMAND_AXES[index] for index in active_indices)
        active_requested_position_mm = tuple(
            float(requested_position_mm[index]) for index in active_indices
        )
        logger.info(
            "Moving correction axes: kind=%s, active_axes=%s, "
            "requested_position_mm=%s",
            kind,
            active_axes,
            active_requested_position_mm,
        )
        motor_backend.move_motors_and_wait(
            active_axes,
            active_requested_position_mm,
            max_retries=max_retries,
            backlash_correction={},
        )
        logger.info("Correction motor move returned; reading final x/y/z positions.")
        final_readback_mm = np.asarray(
            motor_backend.get_positions(COMMAND_AXES),
            dtype=np.float64,
        )
        if final_readback_mm.shape != (len(COMMAND_AXES),):
            raise ValueError("final command position readback must have x/y/z values")
        logger.info("Final readback positions: %s", final_readback_mm.tolist())

        logger.info("Capturing post-move correction measurement.")
        after_measurement = _capture_measurement(
            calibration,
            camera_pair,
            reference_cam0,
            reference_cam1,
            capture_count,
            **shift_kwargs,
        )
        after_residual = weighted_pixel_residual(after_measurement, weights=weights)
        measured_delta_px = (
            _measurement_shift_px(after_measurement)
            - _measurement_shift_px(pre_state.measurement)
        )
        feedback = _correction_feedback_metrics(
            predicted_delta_px,
            measured_delta_px,
            weights=weights,
        )
        feedback_valid = _correction_feedback_is_valid(
            feedback,
            min_total_predicted_shift_px=min_total_predicted_shift_px,
            min_feedback_alpha=min_feedback_alpha,
            min_feedback_parallel_shift_px=min_feedback_parallel_shift_px,
        )
        accepted = bool(after_residual < pre_state.residual_px)
        new_state = CorrectionState(
            commanded_position_mm=final_readback_mm.copy(),
            measurement=after_measurement,
            residual_px=float(after_residual),
        )
        is_best_after = bool(after_residual < best_state.residual_px)
        if is_best_after:
            best_state = new_state

        move_command_delta_mm.append(command_delta_mm.copy())
        move_requested_position_mm.append(requested_position_mm.copy())
        move_final_readback_position_mm.append(final_readback_mm.copy())
        move_gain.append(float(gain_used))
        move_damping_mu.append(float(mu_used))
        move_pre_weighted_residuals.append(float(pre_state.residual_px))
        move_post_weighted_residuals.append(float(after_residual))
        move_predicted_delta_px.append(predicted_delta_px.copy())
        move_measured_delta_px.append(measured_delta_px)
        move_predicted_weighted_response_px.append(
            feedback.predicted_weighted_response_px
        )
        move_measured_weighted_response_px.append(
            feedback.measured_weighted_response_px
        )
        move_feedback_alpha.append(feedback.alpha)
        move_feedback_parallel_px.append(feedback.parallel_px)
        move_feedback_valid.append(feedback_valid)
        move_jacobian_before.append(jacobian_before.copy())
        move_jacobian_after.append(working_jacobian.copy())
        move_max_normalized_component.append(
            float(np.max(np.abs(normalized_component)))
        )
        active_axis_mask = np.zeros(len(COMMAND_AXES), dtype=bool)
        active_axis_mask[list(active_indices)] = True
        move_active_axis_mask.append(active_axis_mask)
        move_jacobian_refined.append(False)
        move_kind.append(kind)
        move_accepted.append(accepted)
        move_is_best_after.append(is_best_after)
        move_verified_residuals.append(float(after_residual))
        move_local_probe_jacobian.append(
            None if local_jacobian_for_log is None else local_jacobian_for_log.copy()
        )

        current_state = new_state
        iteration_shift_px.append(_measurement_shift_px(current_state.measurement))
        iteration_weighted_residuals.append(float(current_state.residual_px))
        warnings.extend(
            line.strip()
            for line in str(current_state.measurement.attrs.get("warnings", "")).splitlines()
            if line.strip()
        )
        if kind == "return_best":
            returned_to_best = True
            return_to_best_verified = True
        logger.info(
            "Verified correction move: kind=%s, residual %.6g -> %.6g px, "
            "accepted=%s, best=%s, feedback_valid=%s",
            kind,
            pre_state.residual_px,
            current_state.residual_px,
            accepted,
            is_best_after,
            feedback_valid,
        )
        save_progress(completed=False)
        return feedback

    def command_return_to_state(target_state: CorrectionState) -> bool:
        requested_position = target_state.commanded_position_mm.copy()
        delta = requested_position - current_state.commanded_position_mm
        if not _active_correction_indices(delta):
            return False
        predicted_delta_px = _predict_shift_delta(working_jacobian, delta)
        normalized = _normalized_component(delta, axis_scale)
        return (
            record_verified_move(
                kind="return_best",
                requested_position_mm=requested_position,
                jacobian_before=working_jacobian,
                predicted_delta_px=predicted_delta_px,
                normalized_component=normalized,
                gain_used=np.nan,
                mu_used=np.nan,
                local_jacobian_for_log=latest_local_jacobian,
                allow_over_budget=True,
            )
            is not None
        )

    def run_local_probe() -> bool:
        nonlocal working_jacobian, latest_local_jacobian, stop_reason

        if move_budget_exhausted() or duration_exhausted():
            return False
        if (
            return_to_best_on_failure
            and best_state.residual_px + min_verified_improvement_px
            < current_state.residual_px
        ):
            command_return_to_state(best_state)
            if move_budget_exhausted() or duration_exhausted():
                return False

        local_observation = working_jacobian.reshape(
            len(OBSERVATION_AXES),
            len(COMMAND_AXES),
        ).copy()
        measured_columns = np.zeros(len(COMMAND_AXES), dtype=bool)
        base_state = current_state
        for axis_index, axis in enumerate(COMMAND_AXES):
            if move_budget_exhausted() or duration_exhausted():
                break
            probe_step_mm = _local_probe_step_mm(
                calibration_jacobian,
                axis_index,
                local_probe_target_response_px,
            )
            requested_position = base_state.commanded_position_mm.copy()
            requested_position[axis_index] += probe_step_mm
            command_delta = requested_position - current_state.commanded_position_mm
            predicted_delta_px = _predict_shift_delta(working_jacobian, command_delta)
            feedback = record_verified_move(
                kind="probe",
                requested_position_mm=requested_position,
                jacobian_before=working_jacobian,
                predicted_delta_px=predicted_delta_px,
                normalized_component=_normalized_component(command_delta, axis_scale),
                gain_used=np.nan,
                mu_used=np.nan,
                local_jacobian_for_log=latest_local_jacobian,
            )
            if feedback is None:
                break
            probe_shift_delta = (
                _measurement_shift_px(current_state.measurement)
                - _measurement_shift_px(base_state.measurement)
            )
            local_observation[:, axis_index] = (
                probe_shift_delta.reshape(-1) / probe_step_mm
            )
            measured_columns[axis_index] = True

            if move_budget_exhausted() or duration_exhausted():
                break
            if _active_correction_indices(
                base_state.commanded_position_mm - current_state.commanded_position_mm
            ):
                return_delta = (
                    base_state.commanded_position_mm - current_state.commanded_position_mm
                )
                record_verified_move(
                    kind="probe",
                    requested_position_mm=base_state.commanded_position_mm.copy(),
                    jacobian_before=working_jacobian,
                    predicted_delta_px=_predict_shift_delta(
                        working_jacobian,
                        return_delta,
                    ),
                    normalized_component=_normalized_component(
                        return_delta,
                        axis_scale,
                    ),
                    gain_used=np.nan,
                    mu_used=np.nan,
                    local_jacobian_for_log=latest_local_jacobian,
                )
                base_state = current_state

        if not np.any(measured_columns):
            stop_reason = "local_probe_unavailable"
            return False
        latest_local_jacobian = local_observation.reshape(
            len(CAMERAS),
            len(PIXEL_AXES),
            len(COMMAND_AXES),
        )
        working_jacobian = latest_local_jacobian.copy()
        logger.info(
            "Local probe updated working Jacobian from %d measured column(s).",
            int(np.count_nonzero(measured_columns)),
        )
        return True

    # Publish the initial measurement and planned correction before any motor move.
    save_progress(completed=False)

    poor_model_streak = 0
    local_probe_ran = False
    if local_probe_mode == "always" and not converged:
        local_probe_ran = run_local_probe()

    while not converged:
        if move_budget_exhausted():
            stop_reason = "max_moves_exhausted"
            break
        if duration_exhausted():
            stop_reason = "max_duration_exhausted"
            break

        logger.info(
            "Correction iteration %d starting: residual=%.6g px, gain=%g, mu=%g",
            move_count() + 1,
            current_state.residual_px,
            current_gain,
            current_mu,
        )
        gain_used = float(current_gain)
        mu_used = float(current_mu)
        jacobian_before = working_jacobian.copy()
        estimated_offset_mm = estimate_command_offset(
            working_jacobian,
            current_state.measurement,
            weights=weights,
        )
        correction_cmd_mm = solve_damped_command_correction(
            working_jacobian,
            current_state.measurement,
            axis_scale,
            gain=gain_used,
            damping_mu=mu_used,
            max_normalized_step=max_normalized_step,
            min_axis_predicted_shift_px=min_axis_predicted_shift_px,
            weights=weights,
        )
        correction_norm_mm = float(np.linalg.norm(correction_cmd_mm))
        active_indices = _active_correction_indices(correction_cmd_mm)
        logger.info(
            "Computed correction command: estimated_offset_mm=%s, "
            "delta_mm=%s, norm_mm=%.6g, active_axes=%s",
            estimated_offset_mm.tolist(),
            correction_cmd_mm.tolist(),
            correction_norm_mm,
            tuple(COMMAND_AXES[index] for index in active_indices),
        )
        if correction_norm_mm <= min_command_norm_mm or not active_indices:
            if should_try_local_probe() and not local_probe_ran:
                local_probe_ran = run_local_probe()
                if local_probe_ran:
                    continue
            stop_reason = "no_active_correction"
            warnings.append(
                _correction_stop_warning(
                    raw_correction_cmd_mm=correction_cmd_mm,
                    correction_cmd_mm=correction_cmd_mm,
                    estimated_offset_mm=estimated_offset_mm,
                    min_command_norm_mm=min_command_norm_mm,
                )
            )
            break

        predicted_delta_px = _predict_shift_delta(jacobian_before, correction_cmd_mm)
        predicted_weighted_response_px = _weighted_shift_norm(
            predicted_delta_px,
            weights=weights,
        )
        if predicted_weighted_response_px < min_total_predicted_shift_px:
            if should_try_local_probe() and not local_probe_ran:
                local_probe_ran = run_local_probe()
                if local_probe_ran:
                    continue
            stop_reason = "predicted_response_too_small"
            warnings.append(
                _below_observable_feedback_warning(
                    predicted_weighted_response_px=predicted_weighted_response_px,
                    min_total_predicted_shift_px=min_total_predicted_shift_px,
                )
            )
            break

        pre_residual = current_state.residual_px
        feedback = record_verified_move(
            kind="trial",
            requested_position_mm=(
                current_state.commanded_position_mm + correction_cmd_mm
            ),
            jacobian_before=jacobian_before,
            predicted_delta_px=predicted_delta_px,
            normalized_component=_normalized_component(correction_cmd_mm, axis_scale),
            gain_used=gain_used,
            mu_used=mu_used,
            local_jacobian_for_log=latest_local_jacobian,
        )
        if feedback is None:
            stop_reason = "move_not_available"
            break

        converged = current_state.residual_px <= target_residual_px
        if converged:
            stop_reason = "target_residual_reached"
            break

        improvement = pre_residual - current_state.residual_px
        weak_improvement = improvement < min_verified_improvement_px
        feedback_valid = _correction_feedback_is_valid(
            feedback,
            min_total_predicted_shift_px=min_total_predicted_shift_px,
            min_feedback_alpha=min_feedback_alpha,
            min_feedback_parallel_shift_px=min_feedback_parallel_shift_px,
        )
        if weak_improvement or not feedback_valid:
            poor_model_streak += 1
        else:
            poor_model_streak = 0
        if weak_improvement or not feedback_valid:
            current_gain = max(min_gain, 0.5 * current_gain)
            current_mu = 2.0 * current_mu if current_mu > 0.0 else 1e-12
            logger.info(
                "Verified improvement was weak or invalid; reducing gain to %g "
                "and increasing mu to %g.",
                current_gain,
                current_mu,
            )
        if (weak_improvement or not feedback_valid) and poor_model_streak >= 2:
            if should_try_local_probe() and not local_probe_ran:
                local_probe_ran = run_local_probe()
                if local_probe_ran:
                    poor_model_streak = 0
                    continue
            stop_reason = "no_verified_improvement"
            break

    converged = current_state.residual_px <= target_residual_px
    if (
        not converged
        and return_to_best_on_failure
        and verify_return_to_best
        and best_state.residual_px + min_verified_improvement_px
        < current_state.residual_px
    ):
        logger.info(
            "Returning to best verified correction state: current=%.6g px, best=%.6g px",
            current_state.residual_px,
            best_state.residual_px,
        )
        command_return_to_state(best_state)
        converged = current_state.residual_px <= target_residual_px
        if not converged and current_state.residual_px > (
            best_state.residual_px + min_verified_improvement_px
        ):
            warnings.append(
                "return to best verified motor coordinates did not reproduce the "
                "historical best image residual; final verified residual is "
                f"{current_state.residual_px:.4g} px"
            )
        if stop_reason == "":
            stop_reason = "returned_to_best"

    if converged:
        stop_reason = "target_residual_reached"
        logger.info(
            "Correction converged: moves=%d, final residual=%.6g px",
            move_count(),
            current_state.residual_px,
        )
    else:
        if stop_reason == "":
            stop_reason = "best_effort_not_converged"
        warnings.append(
            "correction best effort did not converge; final verified residual "
            f"{current_state.residual_px:.4g} px exceeds target "
            f"{target_residual_px:.4g} px after {move_count()} move(s)"
        )
        logger.info(
            "Correction finished without convergence: moves=%d, "
            "final residual=%.6g px, reason=%s",
            move_count(),
            current_state.residual_px,
            stop_reason,
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
                "correction requires a calibration file path so refined "
                "Jacobians can be persisted"
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
    if "jacobian_refinement_count" in calibration.attrs:
        attrs["calibration_jacobian_refinement_count"] = int(
            calibration.attrs["jacobian_refinement_count"]
        )
    return result.assign_attrs(attrs) if attrs else result


def _jacobian_refinement_observations(
    history_path: Path,
    run_command_delta_mm: Sequence[np.ndarray],
    run_measured_delta_px: Sequence[np.ndarray],
    run_jacobian_refined: Sequence[bool],
    current_command_delta_mm: np.ndarray,
    current_measured_delta_px: np.ndarray,
    *,
    exclude_run_id: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    history_delta, history_measured = _load_jacobian_refinement_history(
        history_path,
        exclude_run_id=exclude_run_id,
    )
    rows: list[np.ndarray] = [*history_delta]
    measurements: list[np.ndarray] = [*history_measured]
    rows.extend(
        np.asarray(delta, dtype=np.float64)
        for delta, updated in zip(
            run_command_delta_mm,
            run_jacobian_refined,
            strict=True,
        )
        if updated
    )
    measurements.extend(
        np.asarray(measured, dtype=np.float64)
        for measured, updated in zip(
            run_measured_delta_px,
            run_jacobian_refined,
            strict=True,
        )
        if updated
    )
    rows.append(np.asarray(current_command_delta_mm, dtype=np.float64))
    measurements.append(np.asarray(current_measured_delta_px, dtype=np.float64))
    return np.stack(rows, axis=0), np.stack(measurements, axis=0)


def _load_jacobian_refinement_history(
    path: Path,
    *,
    exclude_run_id: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    command_rows: list[np.ndarray] = []
    measured_rows: list[np.ndarray] = []
    if path.exists():
        with h5py.File(path, "r") as history_file:
            for group_name in sorted(history_file.keys()):
                run_id = _correction_history_run_id(group_name)
                if run_id is not None and run_id == exclude_run_id:
                    continue
                _extend_refinement_rows_from_group(
                    history_file[group_name],
                    command_rows,
                    measured_rows,
                )

    for dataset in _pending_correction_history_datasets(
        path,
        exclude_run_id=exclude_run_id,
    ):
        _extend_refinement_rows_from_dataset(dataset, command_rows, measured_rows)

    if not command_rows:
        return _empty_refinement_observations()
    return np.stack(command_rows, axis=0), np.stack(measured_rows, axis=0)


def _extend_refinement_rows_from_group(
    group: h5py.Group,
    command_rows: list[np.ndarray],
    measured_rows: list[np.ndarray],
) -> None:
    required = (
        "move_command_delta_mm",
        "move_measured_delta_px",
        "move_jacobian_refined",
    )
    if not all(name in group for name in required):
        return
    updated = np.asarray(group["move_jacobian_refined"], dtype=bool)
    if updated.size == 0 or not np.any(updated):
        return
    command_rows.extend(
        np.asarray(row, dtype=np.float64)
        for row in np.asarray(group["move_command_delta_mm"])[updated]
    )
    measured_rows.extend(
        np.asarray(row, dtype=np.float64)
        for row in np.asarray(group["move_measured_delta_px"])[updated]
    )


def _extend_refinement_rows_from_dataset(
    dataset: xr.Dataset,
    command_rows: list[np.ndarray],
    measured_rows: list[np.ndarray],
) -> None:
    required = (
        "move_command_delta_mm",
        "move_measured_delta_px",
        "move_jacobian_refined",
    )
    if not all(name in dataset for name in required):
        return
    updated = np.asarray(dataset["move_jacobian_refined"].values, dtype=bool)
    if updated.size == 0 or not np.any(updated):
        return
    command_rows.extend(
        np.asarray(row, dtype=np.float64)
        for row in np.asarray(dataset["move_command_delta_mm"].values)[updated]
    )
    measured_rows.extend(
        np.asarray(row, dtype=np.float64)
        for row in np.asarray(dataset["move_measured_delta_px"].values)[updated]
    )


def _empty_refinement_observations() -> tuple[np.ndarray, np.ndarray]:
    return (
        np.empty((0, len(COMMAND_AXES)), dtype=np.float64),
        np.empty((0, len(CAMERAS), len(PIXEL_AXES)), dtype=np.float64),
    )


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
        "move_jacobian_refined",
        "move_feedback_valid",
        "move_accepted",
        "move_is_best_after",
    ):
        if name in restored:
            restored[name] = restored[name].astype(bool)
    for key in (
        "correction_applied",
        "correction_converged",
        "correction_history_completed",
        "returned_to_best",
        "return_to_best_verified",
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
    *,
    min_total_predicted_shift_px: float,
    min_feedback_alpha: float,
    min_feedback_parallel_shift_px: float,
) -> bool:
    return bool(
        feedback.predicted_weighted_response_px >= min_total_predicted_shift_px
        and np.isfinite(feedback.alpha)
        and feedback.alpha >= min_feedback_alpha
        and np.isfinite(feedback.parallel_px)
        and feedback.parallel_px >= min_feedback_parallel_shift_px
    )


def _weighted_shift_norm(
    shift_px: np.ndarray,
    *,
    weights: Sequence[float] | np.ndarray | None,
) -> float:
    values = _shift_to_observation_vector(shift_px)
    weight_values = _observation_weight_values(weights)
    return float(np.sqrt(max(float(values @ (weight_values * values)), 0.0)))


def _measurement_shift_px(measurement: xr.Dataset) -> np.ndarray:
    return np.asarray(measurement["shift_px"].values, dtype=np.float64)


def _predict_shift_delta(jacobian: np.ndarray, command_delta_mm: np.ndarray) -> np.ndarray:
    return (
        np.asarray(jacobian, dtype=np.float64).reshape(
            len(CAMERAS) * len(PIXEL_AXES),
            len(COMMAND_AXES),
        )
        @ np.asarray(command_delta_mm, dtype=np.float64)
    ).reshape(len(CAMERAS), len(PIXEL_AXES))


def _normalized_component(
    command_delta_mm: np.ndarray,
    axis_scale_cmd_mm: np.ndarray,
) -> np.ndarray:
    axis_scale = np.asarray(axis_scale_cmd_mm, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        normalized = np.asarray(command_delta_mm, dtype=np.float64) / axis_scale
    normalized[~np.isfinite(normalized)] = np.nan
    return normalized


def _local_probe_step_mm(
    calibration_jacobian: np.ndarray,
    axis_index: int,
    target_response_px: float,
) -> float:
    observation = np.asarray(calibration_jacobian, dtype=np.float64).reshape(
        len(CAMERAS) * len(PIXEL_AXES),
        len(COMMAND_AXES),
    )
    sensitivity = float(np.linalg.norm(observation[:, axis_index]))
    axis = COMMAND_AXES[axis_index]
    if not np.isfinite(sensitivity) or sensitivity <= 0.0:
        nominal = constants.DEFAULT_VISUAL_CALIBRATION_STEP_MM_BY_AXIS[axis]
    else:
        nominal = float(target_response_px) / sensitivity
    minimum = 2.0 * _correction_command_deadband(axis)
    maximum = 0.25 * constants.DEFAULT_VISUAL_CALIBRATION_STEP_MM_BY_AXIS[axis]
    if maximum < minimum:
        maximum = minimum
    return float(np.clip(nominal, minimum, maximum))


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


def _below_observable_feedback_warning(
    *,
    predicted_weighted_response_px: float,
    min_total_predicted_shift_px: float,
) -> str:
    return (
        "predicted correction response is below the observable feedback "
        f"threshold: predicted={predicted_weighted_response_px:.4g} px, "
        f"threshold={min_total_predicted_shift_px:.4g} px; stopping before "
        "another move"
    )


def _invalid_feedback_warning(
    feedback: CorrectionFeedback,
    *,
    min_feedback_alpha: float,
    min_feedback_parallel_shift_px: float,
) -> str:
    return (
        "post-move image response below effective feedback threshold: "
        f"alpha={feedback.alpha:.4g} < {min_feedback_alpha:.4g} or "
        f"parallel={feedback.parallel_px:.4g} px < "
        f"{min_feedback_parallel_shift_px:.4g} px; stopping before another move"
    )


def _reported_next_correction(
    *,
    converged: bool,
    jacobian: np.ndarray,
    measurement: xr.Dataset,
    axis_scale: np.ndarray,
    gain: float,
    damping_mu: float,
    max_normalized_step: float | None,
    min_axis_predicted_shift_px: float,
    weights: Sequence[float] | np.ndarray | None,
) -> np.ndarray:
    if converged:
        return np.zeros(len(COMMAND_AXES), dtype=np.float64)
    return solve_damped_command_correction(
        jacobian,
        measurement,
        axis_scale,
        gain=gain,
        damping_mu=damping_mu,
        max_normalized_step=max_normalized_step,
        min_axis_predicted_shift_px=min_axis_predicted_shift_px,
        weights=weights,
    )


def _build_correction_result(
    *,
    measurement: xr.Dataset,
    jacobian: np.ndarray,
    axis_scale: np.ndarray,
    estimated_offset: np.ndarray,
    next_correction: np.ndarray,
    iteration_shift_px: Sequence[np.ndarray],
    iteration_weighted_residuals: Sequence[float],
    move_command_delta_mm: Sequence[np.ndarray],
    move_requested_position_mm: Sequence[np.ndarray],
    move_final_readback_position_mm: Sequence[np.ndarray],
    move_gain: Sequence[float],
    move_damping_mu: Sequence[float],
    move_pre_weighted_residuals: Sequence[float],
    move_post_weighted_residuals: Sequence[float],
    move_predicted_delta_px: Sequence[np.ndarray],
    move_measured_delta_px: Sequence[np.ndarray],
    move_predicted_weighted_response_px: Sequence[float],
    move_measured_weighted_response_px: Sequence[float],
    move_feedback_alpha: Sequence[float],
    move_feedback_parallel_px: Sequence[float],
    move_feedback_valid: Sequence[bool],
    move_jacobian_before: Sequence[np.ndarray],
    move_jacobian_after: Sequence[np.ndarray],
    move_max_normalized_component: Sequence[float],
    move_active_axis_mask: Sequence[np.ndarray],
    move_jacobian_refined: Sequence[bool],
    move_kind: Sequence[str],
    move_accepted: Sequence[bool],
    move_is_best_after: Sequence[bool],
    move_verified_residuals: Sequence[float],
    move_local_probe_jacobian: Sequence[np.ndarray | None],
    calibration_path: Path,
    correction_history_path: Path,
    correction_run_id: int,
    correction_started_at_utc: str,
    correction_history_completed: bool,
    converged: bool,
    move_count: int,
    pixel_tolerance_px: float,
    target_residual_px: float,
    min_verified_improvement_px: float,
    max_duration_s: float,
    gain: float,
    min_gain: float,
    damping_mu: float,
    current_gain: float,
    current_mu: float,
    max_normalized_step: float | None,
    min_axis_predicted_shift_px: float,
    min_total_predicted_shift_px: float,
    min_feedback_alpha: float,
    min_feedback_parallel_shift_px: float,
    min_command_norm_mm: float,
    max_moves: int,
    initial_commanded_position_mm: np.ndarray,
    commanded_position_mm: np.ndarray,
    best_commanded_position_mm: np.ndarray,
    best_verified_residual_px: float,
    final_verified_residual_px: float,
    local_probe_mode: str,
    local_probe_target_response_px: float,
    returned_to_best: bool,
    return_to_best_verified: bool,
    correction_stop_reason: str,
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
            "visual_jacobian_px_per_cmd_mm": (
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
            "move_visual_jacobian_before_px_per_cmd_mm": (
                ("move", "camera", "pixel_axis", "command_axis"),
                _stack_jacobian_or_empty(move_jacobian_before),
                {"units": "px/commanded-mm"},
            ),
            "move_visual_jacobian_after_px_per_cmd_mm": (
                ("move", "camera", "pixel_axis", "command_axis"),
                _stack_jacobian_or_empty(move_jacobian_after),
                {"units": "px/commanded-mm"},
            ),
            "move_gain": (
                ("move",),
                np.asarray(move_gain, dtype=np.float64),
            ),
            "move_damping_mu": (
                ("move",),
                np.asarray(move_damping_mu, dtype=np.float64),
            ),
            "move_max_normalized_component": (
                ("move",),
                np.asarray(move_max_normalized_component, dtype=np.float64),
            ),
            "move_active_axis_mask": (
                ("move", "command_axis"),
                _stack_bool_or_empty(move_active_axis_mask),
            ),
            "move_jacobian_refined": (
                ("move",),
                np.asarray(move_jacobian_refined, dtype=bool),
            ),
            "move_kind": (
                ("move",),
                np.asarray(move_kind, dtype="U16"),
            ),
            "move_accepted": (
                ("move",),
                np.asarray(move_accepted, dtype=bool),
            ),
            "move_is_best_after": (
                ("move",),
                np.asarray(move_is_best_after, dtype=bool),
            ),
            "move_verified_residual_px": (
                ("move",),
                np.asarray(move_verified_residuals, dtype=np.float64),
                {"units": "px"},
            ),
            "local_probe_jacobian_px_per_cmd_mm": (
                ("move", "camera", "pixel_axis", "command_axis"),
                _stack_optional_jacobian_or_empty(move_local_probe_jacobian),
                {"units": "px/commanded-mm"},
            ),
            "best_commanded_position_mm": (
                ("command_axis",),
                best_commanded_position_mm,
                {"units": "commanded-mm"},
            ),
        }
    ).assign_coords(
        command_axis=list(COMMAND_AXES),
        iteration=np.arange(len(iteration_weighted_residuals), dtype=np.int64),
        move=np.arange(move_count, dtype=np.int64),
    )
    max_normalized_attr = (
        float(max_normalized_step) if max_normalized_step is not None else np.inf
    )
    return result.assign_attrs(
        {
            "calibration_path": str(calibration_path),
            "correction_history_path": str(correction_history_path),
            "correction_history_run_id": int(correction_run_id),
            "correction_history_completed": bool(correction_history_completed),
            "correction_started_at_utc": correction_started_at_utc,
            "correction_converged": bool(converged),
            "correction_iterations": int(move_count),
            "pixel_tolerance_px": float(pixel_tolerance_px),
            "target_residual_px": float(target_residual_px),
            "min_verified_improvement_px": float(min_verified_improvement_px),
            "max_correction_duration_s": float(max_duration_s),
            "correction_gain": float(gain),
            "correction_min_gain": float(min_gain),
            "correction_damping_mu": float(damping_mu),
            "correction_final_gain": float(current_gain),
            "correction_final_damping_mu": float(current_mu),
            "correction_max_normalized_step": max_normalized_attr,
            "correction_min_axis_predicted_shift_px": float(
                min_axis_predicted_shift_px
            ),
            "correction_min_total_predicted_shift_px": float(
                min_total_predicted_shift_px
            ),
            "correction_min_feedback_alpha": float(min_feedback_alpha),
            "correction_min_feedback_parallel_shift_px": float(
                min_feedback_parallel_shift_px
            ),
            "correction_min_command_norm_mm": float(min_command_norm_mm),
            "max_correction_moves": int(max_moves),
            "correction_applied": move_count > 0,
            "best_verified_residual_px": float(best_verified_residual_px),
            "final_verified_residual_px": float(final_verified_residual_px),
            "local_probe_mode": str(local_probe_mode),
            "local_probe_target_response_px": float(local_probe_target_response_px),
            "returned_to_best": bool(returned_to_best),
            "return_to_best_verified": bool(return_to_best_verified),
            "correction_stop_reason": str(correction_stop_reason),
            "warnings": "\n".join(tuple(dict.fromkeys(warnings))),
        }
    )


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
    current_cam0 = _crop_current_stack_if_needed(
        calibration,
        "cam0",
        reference_cam0,
        current_cam0,
    )
    current_cam1 = _crop_current_stack_if_needed(
        calibration,
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


def _stack_jacobian_or_empty(rows: Sequence[np.ndarray]) -> np.ndarray:
    if not rows:
        return np.empty(
            (0, len(CAMERAS), len(PIXEL_AXES), len(COMMAND_AXES)),
            dtype=np.float64,
        )
    return np.stack(rows, axis=0).astype(np.float64, copy=False)


def _stack_optional_jacobian_or_empty(rows: Sequence[np.ndarray | None]) -> np.ndarray:
    if not rows:
        return np.empty(
            (0, len(CAMERAS), len(PIXEL_AXES), len(COMMAND_AXES)),
            dtype=np.float64,
        )
    nan_row = np.full(
        (len(CAMERAS), len(PIXEL_AXES), len(COMMAND_AXES)),
        np.nan,
        dtype=np.float64,
    )
    return np.stack(
        [
            nan_row if row is None else np.asarray(row, dtype=np.float64)
            for row in rows
        ],
        axis=0,
    )


def _zero_deadband_axis_corrections(
    correction_cmd_mm: np.ndarray,
    *,
    reference_cmd_mm: np.ndarray | None = None,
) -> np.ndarray:
    correction = np.asarray(correction_cmd_mm, dtype=np.float64).copy()
    if correction.shape != (len(COMMAND_AXES),):
        raise ValueError("correction_cmd_mm must have one value for x/y/z")
    if reference_cmd_mm is None:
        reference = correction
    else:
        reference = np.asarray(reference_cmd_mm, dtype=np.float64)
    if reference.shape != (len(COMMAND_AXES),):
        raise ValueError("reference_cmd_mm must have one value for x/y/z")
    if not np.isfinite(reference).all():
        raise ValueError("reference_cmd_mm must contain finite values")
    for index, axis in enumerate(COMMAND_AXES):
        deadband = _correction_command_deadband(axis)
        if abs(reference[index]) <= deadband:
            correction[index] = 0.0
    return correction


def _correction_command_deadband(axis: str) -> float:
    deadband = float(constants.CORRECTION_COMMAND_DEADBAND_MM_BY_AXIS.get(axis, 0.0))
    if not np.isfinite(deadband) or deadband < 0.0:
        raise ValueError("correction command deadbands must be finite and non-negative")
    return deadband


def _correction_stop_warning(
    *,
    raw_correction_cmd_mm: np.ndarray,
    correction_cmd_mm: np.ndarray,
    estimated_offset_mm: np.ndarray,
    min_command_norm_mm: float,
) -> str:
    raw_norm_mm = float(np.linalg.norm(raw_correction_cmd_mm))
    if raw_norm_mm <= min_command_norm_mm:
        return (
            "computed correction step is below the minimum command norm "
            f"{min_command_norm_mm:.4g} mm; stopping before another move"
        )
    if not _active_correction_indices(
        correction_cmd_mm
    ) and _estimated_offset_within_deadbands(estimated_offset_mm):
        deadband_text = ", ".join(
            f"{axis}={1000.0 * _correction_command_deadband(axis):.4g} um"
            for axis in COMMAND_AXES
        )
        offset_text = ", ".join(
            f"{axis}={1000.0 * value:.4g} um"
            for axis, value in zip(COMMAND_AXES, estimated_offset_mm, strict=True)
        )
        return (
            "estimated command offset is within correction deadbands "
            f"({deadband_text}); offset=({offset_text}); stopping before another move"
        )
    return (
        "computed correction step has no active axes after pruning; "
        "stopping before another move"
    )


def _estimated_offset_within_deadbands(estimated_offset_mm: np.ndarray) -> bool:
    offset = np.asarray(estimated_offset_mm, dtype=np.float64)
    if offset.shape != (len(COMMAND_AXES),):
        raise ValueError("estimated_offset_mm must have one value for x/y/z")
    return all(
        abs(offset[index]) <= _correction_command_deadband(axis)
        for index, axis in enumerate(COMMAND_AXES)
    )


def _active_correction_indices(correction_cmd_mm: np.ndarray) -> tuple[int, ...]:
    return tuple(
        index
        for index, value in enumerate(np.asarray(correction_cmd_mm, dtype=np.float64))
        if value != 0.0
    )


def _crop_current_stack_if_needed(
    calibration: xr.Dataset,
    camera: str,
    reference: np.ndarray,
    current_stack: np.ndarray,
) -> np.ndarray:
    if current_stack.shape[1:] == np.shape(reference):
        return current_stack

    roi_geometry = _roi_geometry_from_attrs(calibration, camera)
    if roi_geometry is None:
        return current_stack

    cropped_stack = np.stack(
        [crop_image_to_roi(image, roi_geometry) for image in current_stack],
        axis=0,
    )
    if cropped_stack.shape[1:] != np.shape(reference):
        raise ValueError(
            f"cropped {camera} image shape {cropped_stack.shape[1:]!r} "
            f"does not match calibration reference shape {np.shape(reference)!r}"
        )
    return cropped_stack


def _roi_geometry_from_attrs(
    calibration: xr.Dataset,
    camera: str,
) -> RoiGeometry | None:
    keys = ROI_ATTR_KEYS[camera]
    present = tuple(key in calibration.attrs for key in keys)
    if not any(present):
        return None
    if not all(present):
        missing = ", ".join(key for key, exists in zip(keys, present) if not exists)
        raise ValueError(f"incomplete ROI metadata for {camera}; missing {missing}")

    try:
        roi_geometry = tuple(float(calibration.attrs[key]) for key in keys)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"ROI metadata for {camera} must be numeric") from exc

    if not np.isfinite(roi_geometry).all():
        raise ValueError(f"ROI metadata for {camera} must be finite")
    return roi_geometry

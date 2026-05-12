from __future__ import annotations

from collections.abc import Iterable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

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
    PIXEL_AXES,
    estimate_command_offset,
    load_calibration_dataset,
    measure_image_error,
    refine_visual_jacobian_from_observations,
    solve_damped_command_correction,
    validate_visual_calibration_dataset,
    weighted_pixel_residual,
)

ROI_ATTR_KEYS: dict[str, tuple[str, str, str, str]] = {
    camera: (
        f"roi_{camera}_x",
        f"roi_{camera}_y",
        f"roi_{camera}_width",
        f"roi_{camera}_height",
    )
    for camera in CAMERAS
}


def do_correction(
    calibration: xr.Dataset | str | Path,
    camera_pair: CameraPairPlugin | None = None,
    *,
    calibration_path: str | Path | None = None,
    move_tolerance_mm: float | Iterable[float] | None = None,
    max_retries: int = 4,
    capture_count: int = constants.DEFAULT_CAPTURE_COUNT,
    pixel_tolerance_px: float = constants.DEFAULT_CORRECTION_PIXEL_TOLERANCE_PX,
    gain: float = constants.DEFAULT_CORRECTION_GAIN,
    min_gain: float = constants.DEFAULT_CORRECTION_MIN_GAIN,
    damping_mu: float = constants.DEFAULT_CORRECTION_DAMPING_MU,
    max_normalized_step: float | None = (
        constants.DEFAULT_CORRECTION_MAX_NORMALIZED_STEP
    ),
    min_axis_predicted_shift_px: float = (
        constants.DEFAULT_CORRECTION_MIN_AXIS_PREDICTED_SHIFT_PX
    ),
    min_command_norm_mm: float = constants.DEFAULT_CORRECTION_MIN_COMMAND_NORM_MM,
    max_moves: int = constants.DEFAULT_CORRECTION_MAX_MOVES,
    weights: Sequence[float] | np.ndarray | None = None,
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
    validate_visual_calibration_dataset(calibration)
    capture_count = normalize_capture_count(capture_count)
    if camera_pair is None:
        camera_pair = default_camera_pair()

    pixel_tolerance_px = float(pixel_tolerance_px)
    current_gain = float(gain)
    min_gain = float(min_gain)
    current_mu = float(damping_mu)
    min_command_norm_mm = float(min_command_norm_mm)
    max_moves = int(max_moves)
    if not np.isfinite(pixel_tolerance_px) or pixel_tolerance_px < 0.0:
        raise ValueError("pixel_tolerance_px must be finite and non-negative")
    if not np.isfinite(current_gain) or current_gain <= 0.0:
        raise ValueError("gain must be finite and positive")
    if not np.isfinite(min_gain) or min_gain <= 0.0:
        raise ValueError("min_gain must be finite and positive")
    if current_gain < min_gain:
        current_gain = min_gain
    if not np.isfinite(current_mu) or current_mu < 0.0:
        raise ValueError("damping_mu must be finite and non-negative")
    if not np.isfinite(min_command_norm_mm) or min_command_norm_mm < 0.0:
        raise ValueError("min_command_norm_mm must be finite and non-negative")
    if max_moves < 0:
        raise ValueError("max_moves must be >= 0")

    reference_cam0 = np.asarray(calibration["reference_cam0"].values)
    reference_cam1 = np.asarray(calibration["reference_cam1"].values)
    axis_scale = np.asarray(calibration["axis_scale_cmd_mm"].values, dtype=np.float64)
    jacobian = np.asarray(
        calibration["visual_jacobian_px_per_cmd_mm"].values,
        dtype=np.float64,
    )
    commanded_position_mm = np.asarray(
        get_positions(COMMAND_AXES),
        dtype=np.float64,
    )
    if commanded_position_mm.shape != (len(COMMAND_AXES),):
        raise ValueError("initial command position readback must have x/y/z values")
    initial_commanded_position_mm = commanded_position_mm.copy()
    correction_log_path = correction_history_path(resolved_path)
    correction_run_id = _next_correction_history_run_id(correction_log_path)
    correction_started_at_utc = datetime.now(UTC).isoformat()

    measurement = _capture_measurement(
        calibration,
        camera_pair,
        reference_cam0,
        reference_cam1,
        capture_count,
        **shift_kwargs,
    )
    residual = weighted_pixel_residual(measurement, weights=weights)

    iteration_shift_px = [np.asarray(measurement["shift_px"].values, dtype=np.float64)]
    iteration_weighted_residuals = [residual]
    move_command_delta_mm: list[np.ndarray] = []
    move_requested_position_mm: list[np.ndarray] = []
    move_final_readback_position_mm: list[np.ndarray] = []
    move_gain: list[float] = []
    move_damping_mu: list[float] = []
    move_pre_weighted_residuals: list[float] = []
    move_post_weighted_residuals: list[float] = []
    move_predicted_delta_px: list[np.ndarray] = []
    move_measured_delta_px: list[np.ndarray] = []
    move_jacobian_before: list[np.ndarray] = []
    move_jacobian_after: list[np.ndarray] = []
    move_max_normalized_component: list[float] = []
    move_active_axis_mask: list[np.ndarray] = []
    move_jacobian_refined: list[bool] = []
    warnings: list[str] = [
        line.strip()
        for line in str(measurement.attrs.get("warnings", "")).splitlines()
        if line.strip()
    ]

    move_count = 0
    converged = residual <= pixel_tolerance_px

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
            move_jacobian_before=move_jacobian_before,
            move_jacobian_after=move_jacobian_after,
            move_max_normalized_component=move_max_normalized_component,
            move_active_axis_mask=move_active_axis_mask,
            move_jacobian_refined=move_jacobian_refined,
            calibration_path=resolved_path,
            correction_history_path=correction_log_path,
            correction_run_id=correction_run_id,
            correction_started_at_utc=correction_started_at_utc,
            correction_history_completed=completed,
            converged=converged,
            move_count=move_count,
            pixel_tolerance_px=pixel_tolerance_px,
            gain=gain,
            min_gain=min_gain,
            damping_mu=damping_mu,
            current_gain=current_gain,
            current_mu=current_mu,
            max_normalized_step=max_normalized_step,
            min_axis_predicted_shift_px=min_axis_predicted_shift_px,
            min_command_norm_mm=min_command_norm_mm,
            max_moves=max_moves,
            initial_commanded_position_mm=initial_commanded_position_mm,
            commanded_position_mm=commanded_position_mm,
            warnings=warnings,
        )

    def save_progress(completed: bool) -> xr.Dataset:
        progress = build_result(completed)
        save_correction_history_dataset(
            progress,
            correction_log_path,
            run_id=correction_run_id,
        )
        return progress

    while not converged and move_count < max_moves:
        gain_used = float(current_gain)
        mu_used = float(current_mu)
        jacobian_before = jacobian.copy()
        correction_cmd_mm = solve_damped_command_correction(
            jacobian,
            measurement,
            axis_scale,
            gain=gain_used,
            damping_mu=mu_used,
            max_normalized_step=max_normalized_step,
            min_axis_predicted_shift_px=min_axis_predicted_shift_px,
            weights=weights,
        )
        correction_cmd_mm = _zero_deadband_axis_corrections(correction_cmd_mm)
        correction_norm_mm = float(np.linalg.norm(correction_cmd_mm))
        active_indices = _active_correction_indices(correction_cmd_mm)
        if correction_norm_mm <= min_command_norm_mm or not active_indices:
            warnings.append(
                "computed correction step is below the minimum command norm "
                f"{min_command_norm_mm:.4g} mm; stopping before another move"
            )
            break

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
        move_motors_and_wait(
            active_axes,
            active_requested_position_mm,
            tolerance=_active_move_tolerance(move_tolerance_mm, active_indices),
            max_retries=max_retries,
        )
        final_readback_mm = np.asarray(
            get_positions(COMMAND_AXES),
            dtype=np.float64,
        )
        if final_readback_mm.shape != (len(COMMAND_AXES),):
            raise ValueError("final command position readback must have x/y/z values")
        commanded_position_mm = requested_position_mm

        after_measurement = _capture_measurement(
            calibration,
            camera_pair,
            reference_cam0,
            reference_cam1,
            capture_count,
            **shift_kwargs,
        )
        after_residual = weighted_pixel_residual(after_measurement, weights=weights)
        decreased = bool(after_residual < residual)
        jacobian_refined = False
        measured_delta_px = (
            np.asarray(after_measurement["shift_px"].values, dtype=np.float64)
            - np.asarray(measurement["shift_px"].values, dtype=np.float64)
        )
        if decreased:
            try:
                refinement_delta, refinement_measured = (
                    _jacobian_refinement_observations(
                        correction_log_path,
                        move_command_delta_mm,
                        move_measured_delta_px,
                        move_jacobian_refined,
                        correction_cmd_mm,
                        measured_delta_px,
                    )
                )
                calibration = refine_visual_jacobian_from_observations(
                    calibration,
                    refinement_delta,
                    refinement_measured,
                    save_path=resolved_path,
                )
            except ValueError as exc:
                warnings.append(f"skipped Jacobian refinement: {exc}")
            else:
                jacobian = np.asarray(
                    calibration["visual_jacobian_px_per_cmd_mm"].values,
                    dtype=np.float64,
                )
                jacobian_refined = True
        else:
            current_gain = max(min_gain, 0.5 * current_gain)
            current_mu = 2.0 * current_mu if current_mu > 0.0 else 1e-12

        move_command_delta_mm.append(correction_cmd_mm)
        move_requested_position_mm.append(requested_position_mm.copy())
        move_final_readback_position_mm.append(final_readback_mm)
        move_gain.append(gain_used)
        move_damping_mu.append(mu_used)
        move_pre_weighted_residuals.append(float(residual))
        move_post_weighted_residuals.append(float(after_residual))
        move_predicted_delta_px.append(predicted_delta_px)
        move_measured_delta_px.append(measured_delta_px)
        move_jacobian_before.append(jacobian_before)
        move_jacobian_after.append(jacobian.copy())
        move_max_normalized_component.append(
            float(np.max(np.abs(normalized_component)))
        )
        active_axis_mask = np.zeros(len(COMMAND_AXES), dtype=bool)
        active_axis_mask[list(active_indices)] = True
        move_active_axis_mask.append(active_axis_mask)
        move_jacobian_refined.append(jacobian_refined)

        measurement = after_measurement
        residual = after_residual
        iteration_shift_px.append(
            np.asarray(measurement["shift_px"].values, dtype=np.float64)
        )
        iteration_weighted_residuals.append(float(residual))
        warnings.extend(
            line.strip()
            for line in str(measurement.attrs.get("warnings", "")).splitlines()
            if line.strip()
        )
        move_count += 1
        converged = residual <= pixel_tolerance_px
        save_progress(completed=False)

    if not converged:
        warnings.append(
            "correction did not converge within "
            f"{max_moves} move(s); final residual {residual:.4g} px exceeds "
            f"{pixel_tolerance_px:.4g} px"
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


def _jacobian_refinement_observations(
    history_path: Path,
    run_command_delta_mm: Sequence[np.ndarray],
    run_measured_delta_px: Sequence[np.ndarray],
    run_jacobian_refined: Sequence[bool],
    current_command_delta_mm: np.ndarray,
    current_measured_delta_px: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    history_delta, history_measured = _load_jacobian_refinement_history(history_path)
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


def _load_jacobian_refinement_history(path: Path) -> tuple[np.ndarray, np.ndarray]:
    if not path.exists():
        return _empty_refinement_observations()

    command_rows: list[np.ndarray] = []
    measured_rows: list[np.ndarray] = []
    with h5py.File(path, "r") as history_file:
        for group_name in sorted(history_file.keys()):
            group = history_file[group_name]
            required = (
                "move_command_delta_mm",
                "move_measured_delta_px",
                "move_jacobian_refined",
            )
            if not all(name in group for name in required):
                continue
            updated = np.asarray(group["move_jacobian_refined"], dtype=bool)
            if updated.size == 0 or not np.any(updated):
                continue
            command_rows.extend(
                np.asarray(row, dtype=np.float64)
                for row in np.asarray(group["move_command_delta_mm"])[updated]
            )
            measured_rows.extend(
                np.asarray(row, dtype=np.float64)
                for row in np.asarray(group["move_measured_delta_px"])[updated]
            )

    if not command_rows:
        return _empty_refinement_observations()
    return np.stack(command_rows, axis=0), np.stack(measured_rows, axis=0)


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
    )
    return output_path


def load_latest_correction_history_dataset(
    calibration_path: str | Path,
) -> xr.Dataset | None:
    """Load the most recent correction run for a calibration, if one exists."""

    history_path = correction_history_path(calibration_path)
    if not history_path.exists():
        return None

    group_name = _latest_correction_history_group_name(history_path)
    if group_name is None:
        return None

    with xr.open_dataset(
        history_path,
        engine="h5netcdf",
        group=group_name,
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
    for name in ("move_active_axis_mask", "move_jacobian_refined"):
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


def _latest_correction_history_group_name(path: Path) -> str | None:
    with h5py.File(path, "r") as history_file:
        latest_attr = history_file.attrs.get("latest_run_group")
        if isinstance(latest_attr, bytes):
            latest_attr = latest_attr.decode()
        if isinstance(latest_attr, str) and latest_attr in history_file:
            return latest_attr

        run_ids = [
            int(name.removeprefix("run_"))
            for name in history_file.keys()
            if name.startswith("run_") and name.removeprefix("run_").isdigit()
        ]
    if not run_ids:
        return None
    return _correction_history_group_name(max(run_ids))


def _next_correction_history_run_id(path: Path) -> int:
    if not path.exists():
        return 0
    with h5py.File(path, "r") as history_file:
        run_ids = [
            int(name.removeprefix("run_"))
            for name in history_file.keys()
            if name.startswith("run_") and name.removeprefix("run_").isdigit()
        ]
    return max(run_ids, default=-1) + 1


def _correction_history_group_name(run_id: int) -> str:
    run_id = int(run_id)
    if run_id < 0:
        raise ValueError("run_id must be non-negative")
    return f"run_{run_id:06d}"


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
    move_jacobian_before: Sequence[np.ndarray],
    move_jacobian_after: Sequence[np.ndarray],
    move_max_normalized_component: Sequence[float],
    move_active_axis_mask: Sequence[np.ndarray],
    move_jacobian_refined: Sequence[bool],
    calibration_path: Path,
    correction_history_path: Path,
    correction_run_id: int,
    correction_started_at_utc: str,
    correction_history_completed: bool,
    converged: bool,
    move_count: int,
    pixel_tolerance_px: float,
    gain: float,
    min_gain: float,
    damping_mu: float,
    current_gain: float,
    current_mu: float,
    max_normalized_step: float | None,
    min_axis_predicted_shift_px: float,
    min_command_norm_mm: float,
    max_moves: int,
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
            "correction_gain": float(gain),
            "correction_min_gain": float(min_gain),
            "correction_damping_mu": float(damping_mu),
            "correction_final_gain": float(current_gain),
            "correction_final_damping_mu": float(current_mu),
            "correction_max_normalized_step": max_normalized_attr,
            "correction_min_axis_predicted_shift_px": float(
                min_axis_predicted_shift_px
            ),
            "correction_min_command_norm_mm": float(min_command_norm_mm),
            "max_correction_moves": int(max_moves),
            "correction_applied": move_count > 0,
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
    current_cam0, current_cam1 = capture_image_stack(camera_pair, capture_count)
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
    return measure_image_error(
        reference_cam0,
        current_cam0,
        reference_cam1,
        current_cam1,
        **shift_kwargs,
    )


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


def _zero_deadband_axis_corrections(correction_cmd_mm: np.ndarray) -> np.ndarray:
    correction = np.asarray(correction_cmd_mm, dtype=np.float64).copy()
    for index, axis in enumerate(COMMAND_AXES):
        deadband = float(constants.MOTOR_MOVE_DEADBAND.get(axis, 0.0))
        if not np.isfinite(deadband) or deadband < 0.0:
            raise ValueError("move deadbands must be finite and non-negative")
        if abs(correction[index]) <= deadband:
            correction[index] = 0.0
    return correction


def _active_correction_indices(correction_cmd_mm: np.ndarray) -> tuple[int, ...]:
    return tuple(
        index
        for index, value in enumerate(np.asarray(correction_cmd_mm, dtype=np.float64))
        if value != 0.0
    )


def _active_move_tolerance(
    tolerance: float | Iterable[float] | None,
    active_indices: Sequence[int],
) -> float | tuple[float, ...] | None:
    if tolerance is None or np.isscalar(tolerance):
        return tolerance
    values = tuple(float(value) for value in tolerance)
    if len(values) != len(COMMAND_AXES):
        raise ValueError("move_tolerance_mm must be scalar or have x/y/z values")
    return tuple(values[index] for index in active_indices)


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

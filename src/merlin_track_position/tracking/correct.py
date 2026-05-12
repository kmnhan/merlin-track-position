from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

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
    assign_refined_visual_jacobian,
    broyden_update_jacobian,
    estimate_command_offset,
    load_calibration_dataset,
    measure_image_error,
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
    max_moves: int = constants.DEFAULT_CORRECTION_MAX_MOVES,
    broyden_update_blend: float = constants.DEFAULT_BROYDEN_UPDATE_BLEND,
    weights: Sequence[float] | np.ndarray | None = None,
    **shift_kwargs: Any,
) -> xr.Dataset:
    """Run guarded closed-loop visual-servo correction in commanded-mm space.

    The calibration must have a backing file. Accepted Broyden updates rewrite that
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
    move_jacobian_updated: list[bool] = []
    warnings: list[str] = [
        line.strip()
        for line in str(measurement.attrs.get("warnings", "")).splitlines()
        if line.strip()
    ]

    move_count = 0
    converged = residual <= pixel_tolerance_px
    while not converged and move_count < max_moves:
        gain_used = float(current_gain)
        mu_used = float(current_mu)
        correction_cmd_mm = solve_damped_command_correction(
            jacobian,
            measurement,
            axis_scale,
            gain=gain_used,
            damping_mu=mu_used,
            weights=weights,
        )
        requested_position_mm = commanded_position_mm + correction_cmd_mm
        final_readback_mm = np.asarray(
            move_motors_and_wait(
                COMMAND_AXES,
                tuple(float(value) for value in requested_position_mm),
                tolerance=move_tolerance_mm,
                max_retries=max_retries,
            ),
            dtype=np.float64,
        )
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
        jacobian_updated = False
        if decreased:
            measured_delta_px = (
                np.asarray(after_measurement["shift_px"].values, dtype=np.float64)
                - np.asarray(measurement["shift_px"].values, dtype=np.float64)
            )
            try:
                candidate_jacobian = broyden_update_jacobian(
                    jacobian,
                    correction_cmd_mm,
                    measured_delta_px,
                    blend=broyden_update_blend,
                )
                calibration = assign_refined_visual_jacobian(
                    calibration,
                    candidate_jacobian,
                    save_path=resolved_path,
                )
            except ValueError as exc:
                warnings.append(f"skipped Broyden update: {exc}")
            else:
                jacobian = candidate_jacobian
                jacobian_updated = True
        else:
            current_gain = max(min_gain, 0.5 * current_gain)
            current_mu = 2.0 * current_mu if current_mu > 0.0 else 1e-12

        move_command_delta_mm.append(correction_cmd_mm)
        move_requested_position_mm.append(requested_position_mm.copy())
        move_final_readback_position_mm.append(final_readback_mm)
        move_gain.append(gain_used)
        move_damping_mu.append(mu_used)
        move_jacobian_updated.append(jacobian_updated)

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

    if not converged:
        warnings.append(
            "correction did not converge within "
            f"{max_moves} move(s); final residual {residual:.4g} px exceeds "
            f"{pixel_tolerance_px:.4g} px"
        )

    estimated_offset = estimate_command_offset(jacobian, measurement, weights=weights)
    if converged:
        next_correction = np.zeros(len(COMMAND_AXES), dtype=np.float64)
    else:
        next_correction = solve_damped_command_correction(
            jacobian,
            measurement,
            axis_scale,
            gain=current_gain,
            damping_mu=current_mu,
            weights=weights,
        )

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
            "move_gain": (
                ("move",),
                np.asarray(move_gain, dtype=np.float64),
            ),
            "move_damping_mu": (
                ("move",),
                np.asarray(move_damping_mu, dtype=np.float64),
            ),
            "move_jacobian_updated": (
                ("move",),
                np.asarray(move_jacobian_updated, dtype=bool),
            ),
        }
    ).assign_coords(
        command_axis=list(COMMAND_AXES),
        iteration=np.arange(len(iteration_weighted_residuals), dtype=np.int64),
        move=np.arange(move_count, dtype=np.int64),
    )
    return result.assign_attrs(
        {
            "calibration_path": str(resolved_path),
            "correction_converged": bool(converged),
            "correction_iterations": int(move_count),
            "pixel_tolerance_px": float(pixel_tolerance_px),
            "correction_gain": float(gain),
            "correction_min_gain": float(min_gain),
            "correction_damping_mu": float(damping_mu),
            "correction_final_gain": float(current_gain),
            "correction_final_damping_mu": float(current_mu),
            "max_correction_moves": int(max_moves),
            "correction_applied": move_count > 0,
            "warnings": "\n".join(tuple(dict.fromkeys(warnings))),
        }
    )


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

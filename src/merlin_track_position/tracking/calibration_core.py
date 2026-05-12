"""Visual-Jacobian calibration and correction in commanded-mm space."""

from __future__ import annotations

import os
import tempfile
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import xarray as xr

from merlin_track_position import constants
from merlin_track_position.tracking.shift import estimate_shift

CAMERAS = ("cam0", "cam1")
COMMAND_AXES = ("x", "y", "z")
PIXEL_AXES = ("du_px", "dv_px")
OBSERVATION_AXES = tuple(
    f"{camera}_{pixel_axis}" for camera in CAMERAS for pixel_axis in PIXEL_AXES
)
MEASUREMENT_WARNING_SUMMARY = (
    "one or more shift measurements reported image-matching warnings"
)

ProgressCallback = Callable[[int, int], None]
CaptureShiftResult = tuple[np.ndarray, tuple[str, ...]]
IndexedCaptureShiftResult = tuple[int, int, int, np.ndarray, tuple[str, ...]]

REQUIRED_CALIBRATION_VARIABLES: tuple[str, ...] = (
    "visual_jacobian_px_per_cmd_mm",
    "axis_scale_cmd_mm",
    "reference_cam0",
    "reference_cam1",
    "probe_command_delta_mm",
    "probe_measured_delta_px",
    "pre_commanded_position_mm",
    "post_commanded_position_mm",
    "pre_readback_position_mm",
    "post_readback_position_mm",
)
REDUNDANT_CALIBRATION_VARIABLES: tuple[str, ...] = (
    "axis_scale_unclamped_cmd_mm",
    "axis_sensitivity_px_per_cmd_mm",
    "axis_scale_bounds_cmd_mm",
    "axis_scale_target_response_px",
    "probe_predicted_delta_px",
    "probe_residual_delta_px",
    "probe_before_cam0",
    "probe_after_cam0",
    "probe_before_cam1",
    "probe_after_cam1",
)
REDUNDANT_CALIBRATION_COORDS: tuple[str, ...] = ("bound",)
REDUNDANT_CALIBRATION_ATTRS: tuple[str, ...] = (
    "condition_number",
    "residual_rms_px",
    "residual_max_px",
    "calibration_path",
)


def fit_visual_jacobian_calibration(
    *,
    reference_cam0: Any,
    reference_cam1: Any,
    before_images_cam0: Sequence[Any],
    after_images_cam0: Sequence[Any],
    before_images_cam1: Sequence[Any],
    after_images_cam1: Sequence[Any],
    command_delta_mm: Sequence[Sequence[float]],
    pre_commanded_position_mm: Sequence[Sequence[float]],
    post_commanded_position_mm: Sequence[Sequence[float]],
    pre_readback_position_mm: Sequence[Sequence[float]],
    post_readback_position_mm: Sequence[Sequence[float]],
    min_shift_px: float = constants.DEFAULT_VISUAL_CALIBRATION_MIN_SHIFT_PX,
    condition_warning_threshold: float = (
        constants.DEFAULT_VISUAL_JACOBIAN_CONDITION_WARNING
    ),
    additional_context: dict[str, Any] | None = None,
    progress_callback: ProgressCallback | None = None,
    n_jobs: int | None = None,
    **shift_kwargs: Any,
) -> xr.Dataset:
    """Fit a local visual Jacobian from commanded-mm motor probes.

    The fitted model is linear through the local command point:

    ``[du0, dv0, du1, dv1] = J @ [dx_mm, dy_mm, dz_mm]``.
    """

    if "reference_index" in shift_kwargs:
        raise TypeError(
            "fit_visual_jacobian_calibration() got an unexpected keyword "
            "argument 'reference_index'"
        )
    n_jobs = _resolve_n_jobs(n_jobs)

    reference_image_cam0 = _as_reference_image("reference_cam0", reference_cam0)
    reference_image_cam1 = _as_reference_image("reference_cam1", reference_cam1)
    before_cam0, _ = _as_capture_arrays(
        "before_images_cam0",
        before_images_cam0,
    )
    after_cam0, _ = _as_capture_arrays(
        "after_images_cam0",
        after_images_cam0,
    )
    before_cam1, _ = _as_capture_arrays(
        "before_images_cam1",
        before_images_cam1,
    )
    after_cam1, _ = _as_capture_arrays(
        "after_images_cam1",
        after_images_cam1,
    )
    capture_count = _common_capture_count(
        before_cam0,
        after_cam0,
        before_cam1,
        after_cam1,
    )

    probe_count = _validate_probe_stack_lengths(
        before_cam0,
        after_cam0,
        before_cam1,
        after_cam1,
    )
    command_delta = _as_probe_command_matrix("command_delta_mm", command_delta_mm)
    pre_commanded = _as_probe_command_matrix(
        "pre_commanded_position_mm",
        pre_commanded_position_mm,
    )
    post_commanded = _as_probe_command_matrix(
        "post_commanded_position_mm",
        post_commanded_position_mm,
    )
    pre_readback = _as_probe_command_matrix(
        "pre_readback_position_mm",
        pre_readback_position_mm,
    )
    post_readback = _as_probe_command_matrix(
        "post_readback_position_mm",
        post_readback_position_mm,
    )
    for name, values in (
        ("command_delta_mm", command_delta),
        ("pre_commanded_position_mm", pre_commanded),
        ("post_commanded_position_mm", post_commanded),
        ("pre_readback_position_mm", pre_readback),
        ("post_readback_position_mm", post_readback),
    ):
        if values.shape[0] != probe_count:
            raise ValueError(
                f"{name} must have {probe_count} rows, got {values.shape[0]}"
            )
    if not np.allclose(post_commanded - pre_commanded, command_delta):
        raise ValueError(
            "post_commanded_position_mm - pre_commanded_position_mm must equal "
            "probe_command_delta_mm"
        )

    measured_delta_px, capture_shift_mad_px, measurement_warnings = (
        _estimate_probe_capture_shifts(
            ((before_cam0, after_cam0), (before_cam1, after_cam1)),
            progress_callback=progress_callback,
            n_jobs=n_jobs,
            **shift_kwargs,
        )
    )
    observation = _pixels_to_observations(measured_delta_px)
    finite_probe_rows = np.isfinite(observation).all(axis=1)
    if not np.all(finite_probe_rows):
        probe_indices = np.nonzero(~finite_probe_rows)[0]
        probe_text = ", ".join(str(int(index)) for index in probe_indices)
        raise ValueError(
            "visual calibration probe shift measurements must be finite for "
            f"probe(s): {probe_text}"
        )
    min_shift_px = float(min_shift_px)
    if not np.isfinite(min_shift_px) or min_shift_px < 0.0:
        raise ValueError("min_shift_px must be finite and non-negative")
    response_norms = np.linalg.norm(observation, axis=1)
    small_probe_indices = np.nonzero(response_norms < min_shift_px)[0]
    if small_probe_indices.size:
        probe_text = ", ".join(str(int(index)) for index in small_probe_indices)
        raise ValueError(
            "visual calibration probe image response below threshold "
            f"{min_shift_px:.4g} px for probe(s): {probe_text}"
        )

    command_rank = int(np.linalg.matrix_rank(command_delta))
    if command_rank < len(COMMAND_AXES):
        raise ValueError("probe command deltas must span x/y/z command space")

    coef = _fit_robust_least_squares(command_delta, observation)
    visual_jacobian = coef.T.reshape(len(CAMERAS), len(PIXEL_AXES), len(COMMAND_AXES))
    jacobian_observation = visual_jacobian.reshape(
        len(OBSERVATION_AXES),
        len(COMMAND_AXES),
    )
    singular_values = np.linalg.svd(jacobian_observation, compute_uv=False)
    largest_singular_value = float(singular_values[0]) if singular_values.size else 0.0
    rank_tolerance = 1e-3 * largest_singular_value
    calibration_rank = int(np.count_nonzero(singular_values > rank_tolerance))
    if calibration_rank < len(COMMAND_AXES):
        raise ValueError(
            "fitted visual Jacobian must have rank 3; "
            f"got rank {calibration_rank}"
        )

    condition_number = float(np.linalg.cond(jacobian_observation))
    condition_warning_threshold = float(condition_warning_threshold)
    if (
        not np.isfinite(condition_warning_threshold)
        or condition_warning_threshold <= 0.0
    ):
        raise ValueError("condition_warning_threshold must be finite and positive")
    if condition_number > condition_warning_threshold:
        raise ValueError(
            "visual Jacobian is poorly conditioned: condition number "
            f"{condition_number:.4g} > {condition_warning_threshold:.4g}"
        )

    axis_scale, *_ = derive_axis_scale_from_jacobian(visual_jacobian, command_delta)
    warnings = list(
        format_probe_warning_lines(measurement_warnings, command_delta)
    )
    attrs: dict[str, Any] = {
        "capture_count": capture_count,
        "min_shift_px": min_shift_px,
        "condition_warning_threshold": condition_warning_threshold,
        "warnings": "\n".join(tuple(warnings)),
        "broyden_update_count": 0,
    }
    if additional_context is not None:
        attrs |= additional_context

    warning_strings = np.asarray(
        [
            ["\n".join(camera_warnings) for camera_warnings in probe_warnings]
            for probe_warnings in _pad_warnings(measurement_warnings, probe_count)
        ],
        dtype=str,
    )

    coords: dict[str, Any] = {
        "probe": np.arange(probe_count, dtype=np.int64),
        "command_axis": list(COMMAND_AXES),
        "camera": list(CAMERAS),
        "pixel_axis": list(PIXEL_AXES),
        "y_cam0": np.arange(reference_image_cam0.shape[0], dtype=np.int64),
        "x_cam0": np.arange(reference_image_cam0.shape[1], dtype=np.int64),
        "y_cam1": np.arange(reference_image_cam1.shape[0], dtype=np.int64),
        "x_cam1": np.arange(reference_image_cam1.shape[1], dtype=np.int64),
    }
    dataset = xr.Dataset(
        data_vars={
            "visual_jacobian_px_per_cmd_mm": (
                ("camera", "pixel_axis", "command_axis"),
                visual_jacobian,
                {"units": "px/commanded-mm"},
            ),
            "axis_scale_cmd_mm": (
                ("command_axis",),
                axis_scale,
                {"units": "commanded-mm"},
            ),
            "reference_cam0": (("y_cam0", "x_cam0"), reference_image_cam0),
            "reference_cam1": (("y_cam1", "x_cam1"), reference_image_cam1),
            "probe_command_delta_mm": (
                ("probe", "command_axis"),
                command_delta,
                {"units": "commanded-mm"},
            ),
            "probe_measured_delta_px": (
                ("probe", "camera", "pixel_axis"),
                measured_delta_px,
                {"units": "px"},
            ),
            "pre_commanded_position_mm": (
                ("probe", "command_axis"),
                pre_commanded,
                {"units": "commanded-mm"},
            ),
            "post_commanded_position_mm": (
                ("probe", "command_axis"),
                post_commanded,
                {"units": "commanded-mm"},
            ),
            "pre_readback_position_mm": (
                ("probe", "command_axis"),
                pre_readback,
                {"units": "readback-mm"},
            ),
            "post_readback_position_mm": (
                ("probe", "command_axis"),
                post_readback,
                {"units": "readback-mm"},
            ),
            "probe_capture_shift_mad_px": (
                ("probe", "camera", "pixel_axis"),
                capture_shift_mad_px,
                {
                    "units": "px",
                    "description": (
                        "median absolute deviation of per-capture probe shifts"
                    ),
                },
            ),
            "probe_registration_warnings": (
                ("probe", "camera"),
                warning_strings,
            ),
        },
        coords=coords,
        attrs=attrs,
    )
    validate_visual_calibration_dataset(dataset)
    return dataset


def validate_visual_calibration_dataset(dataset: xr.Dataset) -> None:
    """Validate the required visual-Jacobian calibration schema."""

    missing = tuple(
        name for name in REQUIRED_CALIBRATION_VARIABLES if name not in dataset
    )
    if missing:
        raise ValueError(
            "missing required calibration variables: " + ", ".join(missing)
        )

    jacobian = np.asarray(dataset["visual_jacobian_px_per_cmd_mm"].values, dtype=float)
    axis_scale = np.asarray(dataset["axis_scale_cmd_mm"].values, dtype=float)
    reference_cam0 = np.asarray(dataset["reference_cam0"].values)
    reference_cam1 = np.asarray(dataset["reference_cam1"].values)
    command_delta = np.asarray(dataset["probe_command_delta_mm"].values, dtype=float)
    measured_delta = np.asarray(dataset["probe_measured_delta_px"].values, dtype=float)
    pre_commanded = np.asarray(dataset["pre_commanded_position_mm"].values, dtype=float)
    post_commanded = np.asarray(dataset["post_commanded_position_mm"].values, dtype=float)
    pre_readback = np.asarray(dataset["pre_readback_position_mm"].values, dtype=float)
    post_readback = np.asarray(dataset["post_readback_position_mm"].values, dtype=float)

    if jacobian.shape != (len(CAMERAS), len(PIXEL_AXES), len(COMMAND_AXES)):
        raise ValueError(
            "visual_jacobian_px_per_cmd_mm must have shape "
            "(camera, pixel_axis, command_axis)"
        )
    if not np.isfinite(jacobian).all():
        raise ValueError("visual_jacobian_px_per_cmd_mm must contain only finite values")
    jacobian_observation = jacobian.reshape(
        len(OBSERVATION_AXES),
        len(COMMAND_AXES),
    )
    singular_values = np.linalg.svd(jacobian_observation, compute_uv=False)
    largest_singular_value = float(singular_values[0]) if singular_values.size else 0.0
    rank_tolerance = 1e-3 * largest_singular_value
    rank = int(np.count_nonzero(singular_values > rank_tolerance))
    if rank < len(COMMAND_AXES):
        raise ValueError(f"visual Jacobian must have rank 3; got rank {rank}")
    condition_number = float(np.linalg.cond(jacobian_observation))
    if condition_number > constants.DEFAULT_VISUAL_JACOBIAN_CONDITION_WARNING:
        raise ValueError(
            "visual Jacobian is poorly conditioned: condition number "
            f"{condition_number:.4g} > "
            f"{constants.DEFAULT_VISUAL_JACOBIAN_CONDITION_WARNING:.4g}"
        )
    if axis_scale.shape != (len(COMMAND_AXES),):
        raise ValueError("axis_scale_cmd_mm must have shape (command_axis,)")
    if not np.isfinite(axis_scale).all() or np.any(axis_scale <= 0.0):
        raise ValueError("axis_scale_cmd_mm must contain finite positive values")
    scale_bounds = _axis_scale_bounds_array()
    if np.any(axis_scale < scale_bounds[:, 0]) or np.any(
        axis_scale > scale_bounds[:, 1]
    ):
        raise ValueError("axis_scale_cmd_mm must stay within configured bounds")
    if reference_cam0.ndim != 2 or reference_cam1.ndim != 2:
        raise ValueError("reference_cam0 and reference_cam1 must be 2D images")
    if reference_cam0.size == 0 or reference_cam1.size == 0:
        raise ValueError("reference images must not be empty")

    probe_count = command_delta.shape[0] if command_delta.ndim == 2 else -1
    expected_probe_shape = (probe_count, len(COMMAND_AXES))
    if command_delta.shape != expected_probe_shape:
        raise ValueError("probe_command_delta_mm must have shape (probe, command_axis)")
    if probe_count < 1:
        raise ValueError("probe_command_delta_mm must contain at least one probe")
    if measured_delta.shape != (probe_count, len(CAMERAS), len(PIXEL_AXES)):
        raise ValueError(
            "probe_measured_delta_px must have shape (probe, camera, pixel_axis)"
        )
    for name, values in (
        ("probe_command_delta_mm", command_delta),
        ("probe_measured_delta_px", measured_delta),
        ("pre_commanded_position_mm", pre_commanded),
        ("post_commanded_position_mm", post_commanded),
        ("pre_readback_position_mm", pre_readback),
        ("post_readback_position_mm", post_readback),
    ):
        if not np.isfinite(values).all():
            raise ValueError(f"{name} must contain only finite values")
    for name, values in (
        ("pre_commanded_position_mm", pre_commanded),
        ("post_commanded_position_mm", post_commanded),
        ("pre_readback_position_mm", pre_readback),
        ("post_readback_position_mm", post_readback),
    ):
        if values.shape != expected_probe_shape:
            raise ValueError(f"{name} must have shape (probe, command_axis)")
    if not np.allclose(post_commanded - pre_commanded, command_delta):
        raise ValueError(
            "post_commanded_position_mm - pre_commanded_position_mm must equal "
            "probe_command_delta_mm"
        )
    command_rank = int(np.linalg.matrix_rank(command_delta))
    if command_rank < len(COMMAND_AXES):
        raise ValueError("probe_command_delta_mm must span x/y/z command space")


def _canonical_visual_calibration_dataset(dataset: xr.Dataset) -> xr.Dataset:
    """Drop information that is exactly derivable from required variables."""

    drop_names = [
        name
        for name in (*REDUNDANT_CALIBRATION_VARIABLES, *REDUNDANT_CALIBRATION_COORDS)
        if name in dataset
    ]
    canonical = dataset.drop_vars(drop_names) if drop_names else dataset
    if any(name in canonical.attrs for name in REDUNDANT_CALIBRATION_ATTRS):
        canonical = canonical.copy()
        for name in REDUNDANT_CALIBRATION_ATTRS:
            canonical.attrs.pop(name, None)
    return canonical


def save_calibration_dataset(
    dataset: xr.Dataset,
    path: str | Path,
) -> Path:
    """Validate and atomically save a calibration dataset."""

    output_path = Path(path)
    dataset = _canonical_visual_calibration_dataset(dataset)
    validate_visual_calibration_dataset(dataset)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    saved = dataset.load().copy(deep=True)

    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{output_path.name}.",
        suffix=".tmp",
        dir=str(output_path.parent),
    )
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        tmp_path.unlink(missing_ok=True)
        saved.to_netcdf(tmp_path, engine="h5netcdf")
        tmp_path.replace(output_path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise
    return output_path


def load_calibration_dataset(path: str | Path) -> xr.Dataset:
    """Load and validate a calibration dataset from disk."""

    input_path = Path(path)
    with xr.open_dataset(input_path, engine="h5netcdf") as dataset_on_disk:
        dataset = _canonical_visual_calibration_dataset(dataset_on_disk.load())
    dataset.attrs["calibration_path"] = str(input_path)
    validate_visual_calibration_dataset(dataset)
    return dataset


def assign_refined_visual_jacobian(
    calibration: xr.Dataset,
    visual_jacobian: np.ndarray,
    *,
    save_path: str | Path | None = None,
) -> xr.Dataset:
    """Return calibration with a refined Jacobian and persist it when requested."""

    validate_visual_calibration_dataset(calibration)
    jacobian = _as_visual_jacobian(visual_jacobian)
    updated = _canonical_visual_calibration_dataset(calibration).copy(deep=True)
    updated["visual_jacobian_px_per_cmd_mm"] = (
        ("camera", "pixel_axis", "command_axis"),
        jacobian,
        {"units": "px/commanded-mm"},
    )
    updated.attrs["broyden_update_count"] = (
        int(updated.attrs.get("broyden_update_count", 0)) + 1
    )
    updated.attrs["jacobian_refined"] = "true"
    validate_visual_calibration_dataset(updated)

    if save_path is not None:
        save_calibration_dataset(updated, save_path)
    return updated


def measure_image_error(
    reference_cam0: Any,
    current_cam0: Any,
    reference_cam1: Any,
    current_cam1: Any,
    **shift_kwargs: Any,
) -> xr.Dataset:
    """Measure the current two-camera image error against the references."""

    reference_image_cam0 = _as_reference_image("reference_cam0", reference_cam0)
    reference_image_cam1 = _as_reference_image("reference_cam1", reference_cam1)
    current_captures_cam0, current_dtypes_cam0 = _as_capture_arrays(
        "current_cam0",
        [current_cam0],
    )
    current_captures_cam1, current_dtypes_cam1 = _as_capture_arrays(
        "current_cam1",
        [current_cam1],
    )
    capture_count = _common_capture_count(current_captures_cam0, current_captures_cam1)
    current_stack_cam0 = current_captures_cam0[0]
    current_stack_cam1 = current_captures_cam1[0]

    shift_cam0, mad_cam0, warnings_cam0 = _estimate_median_capture_shift(
        reference_image_cam0,
        current_stack_cam0,
        **shift_kwargs,
    )
    shift_cam1, mad_cam1, warnings_cam1 = _estimate_median_capture_shift(
        reference_image_cam1,
        current_stack_cam1,
        **shift_kwargs,
    )
    current_image_cam0 = _representative_capture_image(
        current_stack_cam0,
        current_dtypes_cam0[0],
    )
    current_image_cam1 = _representative_capture_image(
        current_stack_cam1,
        current_dtypes_cam1[0],
    )
    shift_px = np.stack([shift_cam0, shift_cam1], axis=0)
    capture_shift_mad_px = np.stack([mad_cam0, mad_cam1], axis=0)
    if not np.isfinite(shift_px).all():
        raise ValueError("shift_px must contain only finite values")
    if not np.isfinite(capture_shift_mad_px).all():
        raise ValueError("capture_shift_mad_px must contain only finite values")

    shift_warnings = _format_correction_warning_lines(
        warnings_cam0,
        warnings_cam1,
        capture_count,
    )
    return xr.Dataset(
        data_vars={
            "shift_px": (("camera", "pixel_axis"), shift_px, {"units": "px"}),
            "capture_shift_mad_px": (
                ("camera", "pixel_axis"),
                capture_shift_mad_px,
                {
                    "units": "px",
                    "description": (
                        "median absolute deviation of per-capture shift estimates"
                    ),
                },
            ),
            "current_cam0": (("y_cam0", "x_cam0"), current_image_cam0),
            "current_cam1": (("y_cam1", "x_cam1"), current_image_cam1),
        },
        coords={
            "camera": list(CAMERAS),
            "pixel_axis": list(PIXEL_AXES),
            "y_cam0": np.arange(current_image_cam0.shape[0], dtype=np.int64),
            "x_cam0": np.arange(current_image_cam0.shape[1], dtype=np.int64),
            "y_cam1": np.arange(current_image_cam1.shape[0], dtype=np.int64),
            "x_cam1": np.arange(current_image_cam1.shape[1], dtype=np.int64),
        },
        attrs={
            "capture_count": capture_count,
            "warnings": "\n".join(shift_warnings),
        },
    )


def estimate_command_offset(
    calibration_or_jacobian: xr.Dataset | np.ndarray,
    shift: xr.Dataset | xr.DataArray | Sequence[float],
    *,
    weights: Sequence[float] | np.ndarray | None = None,
) -> np.ndarray:
    """Convert observed two-camera image error to estimated command offset."""

    if isinstance(calibration_or_jacobian, xr.Dataset):
        validate_visual_calibration_dataset(calibration_or_jacobian)
        jacobian = np.asarray(
            calibration_or_jacobian["visual_jacobian_px_per_cmd_mm"].values,
            dtype=np.float64,
        )
    else:
        jacobian = _as_visual_jacobian(calibration_or_jacobian)

    observation = _shift_to_observation(_shift_values(shift))
    jacobian_observation = _jacobian_to_observation(jacobian)
    weight_matrix = _observation_weight_matrix(weights)
    lhs = jacobian_observation.T @ weight_matrix @ jacobian_observation
    rhs = jacobian_observation.T @ weight_matrix @ observation
    return np.linalg.pinv(lhs) @ rhs


def solve_damped_command_correction(
    visual_jacobian: np.ndarray,
    shift: xr.Dataset | xr.DataArray | Sequence[float],
    axis_scale_cmd_mm: Sequence[float],
    *,
    gain: float = constants.DEFAULT_CORRECTION_GAIN,
    damping_mu: float = constants.DEFAULT_CORRECTION_DAMPING_MU,
    weights: Sequence[float] | np.ndarray | None = None,
) -> np.ndarray:
    """Solve the damped normalized-command correction ``Delta a``."""

    jacobian_observation = _jacobian_to_observation(_as_visual_jacobian(visual_jacobian))
    observation = _shift_to_observation(_shift_values(shift))
    axis_scale = np.asarray(axis_scale_cmd_mm, dtype=np.float64)
    if axis_scale.shape != (len(COMMAND_AXES),):
        raise ValueError("axis_scale_cmd_mm must have one value for x/y/z")
    if not np.isfinite(axis_scale).all() or np.any(axis_scale <= 0.0):
        raise ValueError("axis_scale_cmd_mm must contain finite positive values")

    gain = float(gain)
    damping_mu = float(damping_mu)
    if not np.isfinite(gain) or gain <= 0.0:
        raise ValueError("gain must be finite and positive")
    if not np.isfinite(damping_mu) or damping_mu < 0.0:
        raise ValueError("damping_mu must be finite and non-negative")

    scale_matrix = np.diag(axis_scale)
    normalized_jacobian = jacobian_observation @ scale_matrix
    weight_matrix = _observation_weight_matrix(weights)
    lhs = (
        normalized_jacobian.T @ weight_matrix @ normalized_jacobian
        + damping_mu * np.eye(len(COMMAND_AXES))
    )
    rhs = normalized_jacobian.T @ weight_matrix @ observation
    delta_q = -gain * np.linalg.solve(lhs, rhs)
    correction_cmd_mm = scale_matrix @ delta_q
    if correction_cmd_mm.shape != (len(COMMAND_AXES),):
        raise ValueError("computed correction has unexpected shape")
    if not np.isfinite(correction_cmd_mm).all():
        raise ValueError("computed correction contains non-finite values")
    return correction_cmd_mm


def weighted_pixel_residual(
    shift: xr.Dataset | xr.DataArray | Sequence[float],
    *,
    weights: Sequence[float] | np.ndarray | None = None,
) -> float:
    """Return the formal visual-servo residual ``sqrt(p.T W p)``."""

    observation = _shift_to_observation(_shift_values(shift))
    weight_matrix = _observation_weight_matrix(weights)
    return float(np.sqrt(observation @ weight_matrix @ observation))


def broyden_update_jacobian(
    visual_jacobian: np.ndarray,
    command_delta_mm: Sequence[float],
    measured_delta_px: xr.Dataset | xr.DataArray | Sequence[float],
    *,
    blend: float = constants.DEFAULT_BROYDEN_UPDATE_BLEND,
    min_command_norm_mm: float = 1e-9,
) -> np.ndarray:
    """Apply a blended Broyden update to the visual Jacobian."""

    jacobian = _as_visual_jacobian(visual_jacobian)
    jacobian_observation = _jacobian_to_observation(jacobian)
    command_delta = np.asarray(command_delta_mm, dtype=np.float64)
    if command_delta.shape != (len(COMMAND_AXES),):
        raise ValueError("command_delta_mm must have one value for x/y/z")
    if not np.isfinite(command_delta).all():
        raise ValueError("command_delta_mm must contain only finite values")
    measured_delta = _shift_to_observation(_shift_values(measured_delta_px))
    blend = float(blend)
    min_command_norm_mm = float(min_command_norm_mm)
    if not np.isfinite(blend) or blend < 0.0 or blend > 1.0:
        raise ValueError("blend must be finite and between 0 and 1")
    if not np.isfinite(min_command_norm_mm) or min_command_norm_mm <= 0.0:
        raise ValueError("min_command_norm_mm must be finite and positive")

    denom = float(command_delta @ command_delta)
    if denom < min_command_norm_mm * min_command_norm_mm:
        raise ValueError("command_delta_mm norm is too small for Broyden update")
    residual = measured_delta - jacobian_observation @ command_delta
    updated = jacobian_observation + blend * np.outer(residual, command_delta) / denom
    return updated.reshape(len(CAMERAS), len(PIXEL_AXES), len(COMMAND_AXES))


def _estimate_probe_capture_shifts(
    capture_arrays_by_camera: tuple[
        tuple[Sequence[np.ndarray], Sequence[np.ndarray]],
        tuple[Sequence[np.ndarray], Sequence[np.ndarray]],
    ],
    *,
    progress_callback: ProgressCallback | None,
    n_jobs: int,
    **shift_kwargs: Any,
) -> tuple[
    np.ndarray,
    np.ndarray,
    list[tuple[tuple[str, ...], tuple[str, ...]]],
]:
    probe_count = len(capture_arrays_by_camera[0][0])
    capture_count = int(capture_arrays_by_camera[0][0][0].shape[0])
    camera_count = len(CAMERAS)
    total = probe_count * camera_count * capture_count
    if progress_callback is not None:
        progress_callback(0, total)

    shift_array = np.empty(
        (probe_count, camera_count, capture_count, len(PIXEL_AXES)),
        dtype=np.float64,
    )
    warnings_by_capture: list[list[list[tuple[str, ...]]]] = [
        [[() for _ in range(capture_count)] for _ in CAMERAS]
        for _ in range(probe_count)
    ]
    tasks: list[tuple[int, int, int, np.ndarray, np.ndarray]] = []
    for probe_index in range(probe_count):
        for camera_index, (before_arrays, after_arrays) in enumerate(
            capture_arrays_by_camera
        ):
            reference = _mean_capture_image(before_arrays[probe_index])
            after_stack = after_arrays[probe_index]
            for capture_index, current in enumerate(after_stack):
                tasks.append(
                    (
                        probe_index,
                        camera_index,
                        capture_index,
                        reference,
                        current,
                    )
                )

    completed = 0
    for probe_index, camera_index, capture_index, shift_px, capture_warnings in (
        _iter_indexed_capture_shift_results(
            tasks,
            capture_count,
            n_jobs=n_jobs,
            **shift_kwargs,
        )
    ):
        shift_array[probe_index, camera_index, capture_index, :] = shift_px
        warnings_by_capture[probe_index][camera_index][capture_index] = (
            capture_warnings
        )
        completed += 1
        if progress_callback is not None:
            progress_callback(completed, total)

    pixels = np.empty(
        (probe_count, camera_count, len(PIXEL_AXES)),
        dtype=np.float64,
    )
    capture_shift_mad_px = np.empty_like(pixels)
    measurement_warnings: list[tuple[tuple[str, ...], tuple[str, ...]]] = []
    for probe_index in range(probe_count):
        probe_warnings: list[tuple[str, ...]] = []
        for camera_index in range(camera_count):
            try:
                median_shift, shift_mad, camera_warnings = _summarize_capture_shifts(
                    (
                        (
                            shift_array[probe_index, camera_index, capture_index],
                            warnings_by_capture[probe_index][camera_index][
                                capture_index
                            ],
                        )
                        for capture_index in range(capture_count)
                    )
                )
            except ValueError as exc:
                median_shift = np.full(len(PIXEL_AXES), np.nan, dtype=np.float64)
                shift_mad = np.full(len(PIXEL_AXES), np.nan, dtype=np.float64)
                camera_warnings = (str(exc),)
            pixels[probe_index, camera_index, :] = median_shift
            capture_shift_mad_px[probe_index, camera_index, :] = shift_mad
            probe_warnings.append(camera_warnings)
        measurement_warnings.append((probe_warnings[0], probe_warnings[1]))

    return pixels, capture_shift_mad_px, measurement_warnings


def _resolve_n_jobs(n_jobs: int | None) -> int:
    if n_jobs is None:
        n_jobs = constants.CALIBRATION_FIT_N_JOBS
    n_jobs = int(n_jobs)
    if n_jobs < 1:
        raise ValueError("n_jobs must be >= 1")
    return n_jobs


def _iter_indexed_capture_shift_results(
    tasks: Sequence[tuple[int, int, int, np.ndarray, np.ndarray]],
    capture_count: int,
    *,
    n_jobs: int,
    **shift_kwargs: Any,
) -> Iterator[IndexedCaptureShiftResult]:
    if n_jobs == 1:
        for task in tasks:
            yield _estimate_indexed_capture_shift(
                *task,
                capture_count,
                **shift_kwargs,
            )
        return

    with ThreadPoolExecutor(max_workers=n_jobs) as executor:
        futures = [
            executor.submit(
                _estimate_indexed_capture_shift,
                *task,
                capture_count,
                **shift_kwargs,
            )
            for task in tasks
        ]
        for future in as_completed(futures):
            yield future.result()


def _estimate_indexed_capture_shift(
    sample_index: int,
    camera_index: int,
    capture_index: int,
    reference: np.ndarray,
    current: np.ndarray,
    capture_count: int,
    **shift_kwargs: Any,
) -> IndexedCaptureShiftResult:
    shift_px, capture_warnings = _estimate_capture_shift(
        reference,
        current,
        **shift_kwargs,
    )
    warnings = tuple(
        _format_capture_warning(line, capture_index, capture_count)
        for line in capture_warnings
        if line
    )
    return sample_index, camera_index, capture_index, shift_px, warnings


def _estimate_capture_shift(
    reference: np.ndarray,
    current: np.ndarray,
    **shift_kwargs: Any,
) -> CaptureShiftResult:
    try:
        shift = estimate_shift(reference, current, **shift_kwargs)
        shift_px = np.asarray(shift["shift_px"].values, dtype=np.float64)
        capture_warnings = tuple(
            line
            for line in str(shift.attrs.get("warnings", "")).splitlines()
            if line
        )
    except Exception as exc:
        shift_px = np.array([np.nan, np.nan], dtype=np.float64)
        capture_warnings = (f"shift estimation failed: {exc}",)

    if shift_px.shape != (len(PIXEL_AXES),):
        capture_warnings = (
            *capture_warnings,
            f"shift estimate has unexpected shape {shift_px.shape!r}",
        )
        shift_px = np.array([np.nan, np.nan], dtype=np.float64)
    if not np.isfinite(shift_px).all():
        capture_warnings = (
            *capture_warnings,
            "shift estimate is not finite",
        )
    return shift_px, capture_warnings


def _as_reference_image(name: str, image: Any) -> np.ndarray:
    image_array = np.asarray(image, dtype=np.float64)
    if image_array.ndim != 2:
        raise ValueError(f"{name} must be 2D, got {image_array.shape!r}")
    if image_array.size == 0:
        raise ValueError(f"{name} must not be empty")
    if not np.isfinite(image_array).all():
        raise ValueError(f"{name} must contain only finite values")
    return image_array


def _as_capture_arrays(
    name: str,
    images: Sequence[Any],
) -> tuple[list[np.ndarray], list[np.dtype]]:
    image_values = [np.asarray(image) for image in images]
    if not image_values:
        raise ValueError(f"{name} must not be empty")
    capture_arrays: list[np.ndarray] = []
    image_dtypes: list[np.dtype] = []
    first_shape: tuple[int, int] | None = None
    first_capture_count: int | None = None
    for index, image in enumerate(image_values):
        if image.ndim != 3:
            raise ValueError(f"{name}[{index}] must be 3D, got {image.shape!r}")
        capture_array = image
        if capture_array.shape[0] < 1:
            raise ValueError(f"{name}[{index}] must contain at least one capture")
        if capture_array.shape[1] == 0 or capture_array.shape[2] == 0:
            raise ValueError(f"{name}[{index}] images must not be empty")
        if first_capture_count is None:
            first_capture_count = int(capture_array.shape[0])
        elif capture_array.shape[0] != first_capture_count:
            raise ValueError(
                f"all capture stacks in {name} must have the same capture count; "
                f"image 0 has {first_capture_count}, image {index} has "
                f"{capture_array.shape[0]}"
            )
        image_shape = tuple(capture_array.shape[1:])
        if first_shape is None:
            first_shape = image_shape
        elif image_shape != first_shape:
            raise ValueError(
                f"all images in {name} must have the same shape; "
                f"image 0 has {first_shape!r}, image {index} has {image_shape!r}"
            )
        try:
            capture_float = np.asarray(capture_array, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name}[{index}] must be numeric") from exc
        if not np.isfinite(capture_float).all():
            raise ValueError(f"{name}[{index}] must contain only finite values")
        capture_arrays.append(capture_float)
        image_dtypes.append(image.dtype)
    return capture_arrays, image_dtypes


def _validate_probe_stack_lengths(*groups: Sequence[np.ndarray]) -> int:
    counts = [len(group) for group in groups]
    if not counts or counts[0] == 0:
        raise ValueError("at least one visual calibration probe is required")
    if any(count != counts[0] for count in counts):
        raise ValueError(
            "before/after camera probe stacks must have the same length; got "
            + ", ".join(str(count) for count in counts)
        )
    return int(counts[0])


def _as_probe_command_matrix(name: str, values: Sequence[Sequence[float]]) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] != len(COMMAND_AXES):
        raise ValueError(f"{name} must have shape (probe, command_axis)")
    if matrix.shape[0] == 0:
        raise ValueError(f"{name} must contain at least one probe")
    if not np.isfinite(matrix).all():
        raise ValueError(f"{name} must contain only finite values")
    return matrix


def _common_capture_count(
    *capture_array_groups: Sequence[np.ndarray],
) -> int:
    capture_counts = [
        int(capture_array.shape[0])
        for capture_arrays in capture_array_groups
        for capture_array in capture_arrays
    ]
    if not capture_counts:
        raise ValueError("at least one capture stack is required")
    capture_count = capture_counts[0]
    if any(count != capture_count for count in capture_counts):
        raise ValueError("all image capture stacks must have the same capture count")
    return capture_count


def _mean_capture_image(capture_array: np.ndarray) -> np.ndarray:
    return np.mean(np.asarray(capture_array, dtype=np.float64), axis=0)


def _representative_capture_image(
    capture_array: np.ndarray,
    dtype: np.dtype,
) -> np.ndarray:
    image = _mean_capture_image(capture_array)
    return _cast_representative_image(image, dtype)


def _cast_representative_image(image: np.ndarray, dtype: np.dtype) -> np.ndarray:
    if np.issubdtype(dtype, np.bool_):
        return np.asarray(image >= 0.5, dtype=dtype)
    if np.issubdtype(dtype, np.integer):
        dtype_info = np.iinfo(dtype)
        image = np.clip(np.rint(image), dtype_info.min, dtype_info.max)
    return np.asarray(image, dtype=dtype)


def _estimate_median_capture_shift(
    reference: np.ndarray,
    captures: np.ndarray,
    **shift_kwargs: Any,
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    capture_count = int(captures.shape[0])
    results: list[CaptureShiftResult] = []
    for capture_index, current in enumerate(captures):
        shift_px, capture_warnings = _estimate_capture_shift(
            reference,
            current,
            **shift_kwargs,
        )
        results.append(
            (
                shift_px,
                tuple(
                    _format_capture_warning(line, capture_index, capture_count)
                    for line in capture_warnings
                    if line
                ),
            )
        )
    return _summarize_capture_shifts(results)


def _summarize_capture_shifts(
    results: Iterable[CaptureShiftResult],
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    shifts: list[np.ndarray] = []
    warnings: list[str] = []
    for shift_px, capture_warnings in results:
        shifts.append(shift_px)
        warnings.extend(warning for warning in capture_warnings if warning)

    shift_array = np.vstack(shifts)
    finite_rows = np.isfinite(shift_array).all(axis=1)
    if not np.any(finite_rows):
        raise ValueError("all capture shift estimates failed or were non-finite")
    finite_shifts = shift_array[finite_rows]
    median_shift = np.median(finite_shifts, axis=0)
    shift_mad = np.median(np.abs(finite_shifts - median_shift), axis=0)
    return median_shift, shift_mad, tuple(warnings)


def _format_capture_warning(
    warning: str,
    capture_index: int,
    capture_count: int,
) -> str:
    warning_text = str(warning).strip()
    if not warning_text:
        return ""
    if capture_count == 1:
        return warning_text
    return f"capture {capture_index + 1}: {warning_text}"


def _format_correction_warning_lines(
    warnings_cam0: Sequence[str],
    warnings_cam1: Sequence[str],
    capture_count: int,
) -> tuple[str, ...]:
    if capture_count == 1:
        return tuple(line for line in (*warnings_cam0, *warnings_cam1) if line)
    return tuple(
        [
            *(f"cam0 {line}" for line in warnings_cam0 if line),
            *(f"cam1 {line}" for line in warnings_cam1 if line),
        ]
    )


def format_probe_warning_lines(
    measurement_warnings: Sequence[Sequence[Sequence[str]]] | None,
    command_delta_mm: Sequence[Sequence[float]],
) -> tuple[str, ...]:
    """Return display-ready per-probe image matching warning lines."""

    command_delta = np.asarray(command_delta_mm, dtype=np.float64)
    if command_delta.ndim != 2 or command_delta.shape[1] != len(COMMAND_AXES):
        raise ValueError("command_delta_mm must have shape (probe, command_axis)")

    warning_rows = _pad_warnings(measurement_warnings, command_delta.shape[0])
    lines: list[str] = []
    for probe_index, probe_warnings in enumerate(warning_rows):
        command_text = _format_command_warning_context(command_delta[probe_index])
        for camera, camera_warnings in zip(CAMERAS, probe_warnings, strict=True):
            for warning in camera_warnings:
                warning_text = str(warning).strip()
                if warning_text:
                    lines.append(
                        f"probe {probe_index + 1} ({command_text}), "
                        f"{camera}: {warning_text}"
                    )
    return tuple(lines)


def _format_command_warning_context(command_row: np.ndarray) -> str:
    axis_values = ", ".join(
        f"{axis}={_format_warning_number(value)}"
        for axis, value in zip(COMMAND_AXES, command_row, strict=True)
    )
    return f"{axis_values} mm"


def _format_warning_number(value: float) -> str:
    return f"{float(value):.4g}"


def _fit_robust_least_squares(
    command_delta: np.ndarray,
    observation: np.ndarray,
) -> np.ndarray:
    weights = np.ones(command_delta.shape[0], dtype=np.float64)
    coef = np.zeros((len(COMMAND_AXES), len(OBSERVATION_AXES)), dtype=np.float64)
    for _ in range(5):
        sqrt_weights = np.sqrt(weights)[:, np.newaxis]
        coef, _, _, _ = np.linalg.lstsq(
            command_delta * sqrt_weights,
            observation * sqrt_weights,
            rcond=None,
        )
        residual = observation - command_delta @ coef
        residual_norm = np.linalg.norm(residual, axis=1)
        finite = residual_norm[np.isfinite(residual_norm)]
        if finite.size == 0:
            break
        median = float(np.median(finite))
        mad = float(np.median(np.abs(finite - median)))
        scale = 1.4826 * mad
        if not np.isfinite(scale) or scale <= 1e-12:
            break
        threshold = 1.345 * scale
        weights = np.minimum(1.0, threshold / np.maximum(residual_norm, 1e-12))
    return coef


def derive_axis_scale_from_jacobian(
    visual_jacobian: np.ndarray,
    command_delta: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """Derive correction damping scales from fitted visual-Jacobian sensitivity."""

    axis_sensitivity = np.linalg.norm(_jacobian_to_observation(visual_jacobian), axis=0)
    if not np.isfinite(axis_sensitivity).all() or np.any(axis_sensitivity <= 0.0):
        raise ValueError("visual Jacobian axis sensitivity must be finite and positive")

    command_delta = np.asarray(command_delta, dtype=np.float64)
    if command_delta.ndim != 2 or command_delta.shape[1] != len(COMMAND_AXES):
        raise ValueError("command_delta must have shape (probe, command_axis)")
    probe_step = np.max(np.abs(command_delta), axis=0)
    if not np.isfinite(probe_step).all() or np.any(probe_step <= 0.0):
        raise ValueError("command deltas must include a finite non-zero step per axis")

    probe_response = axis_sensitivity * probe_step
    scale_target = float(np.median(probe_response))
    if not np.isfinite(scale_target) or scale_target <= 0.0:
        raise ValueError("cannot derive a finite positive axis-scale target")

    axis_scale_unclamped = scale_target / axis_sensitivity
    scale_bounds = _axis_scale_bounds_array()
    axis_scale = np.clip(
        axis_scale_unclamped,
        scale_bounds[:, 0],
        scale_bounds[:, 1],
    )
    return axis_scale, axis_sensitivity, axis_scale_unclamped, scale_bounds, scale_target


def _axis_scale_bounds_array() -> np.ndarray:
    bounds_config = constants.DEFAULT_AXIS_SCALE_BOUNDS_CMD_MM_BY_AXIS
    bounds: list[tuple[float, float]] = []
    for axis in COMMAND_AXES:
        if axis not in bounds_config:
            raise ValueError(f"missing axis scale bounds for axis {axis!r}")
        lower, upper = (float(value) for value in bounds_config[axis])
        if (
            not np.isfinite(lower)
            or not np.isfinite(upper)
            or lower <= 0.0
            or upper < lower
        ):
            raise ValueError(
                f"axis scale bounds for axis {axis!r} must be finite positive "
                "values with min <= max"
            )
        bounds.append((lower, upper))
    return np.asarray(bounds, dtype=np.float64)


def _pixels_to_observations(pixels: np.ndarray) -> np.ndarray:
    pixels = np.asarray(pixels, dtype=np.float64)
    if pixels.ndim == 2 and pixels.shape == (len(CAMERAS), len(PIXEL_AXES)):
        return pixels.reshape(-1)
    if pixels.ndim == 3 and pixels.shape[1:] == (len(CAMERAS), len(PIXEL_AXES)):
        return pixels.reshape(pixels.shape[0], -1)
    raise ValueError(
        "pixel shifts must have shape (camera, pixel_axis) or "
        "(sample, camera, pixel_axis)"
    )


def _shift_values(
    shift: xr.Dataset | xr.DataArray | Sequence[float],
) -> np.ndarray:
    if isinstance(shift, xr.Dataset):
        return np.asarray(shift["shift_px"].values, dtype=np.float64)
    if isinstance(shift, xr.DataArray):
        return np.asarray(shift.values, dtype=np.float64)
    return np.asarray(shift, dtype=np.float64)


def _shift_to_observation(shift_values: np.ndarray) -> np.ndarray:
    values = np.asarray(shift_values, dtype=np.float64)
    if values.shape == (len(CAMERAS), len(PIXEL_AXES)):
        return values.reshape(-1)
    if values.size == len(OBSERVATION_AXES):
        return values.reshape(-1)
    raise ValueError("shift must have shape (2, 2) or contain four observation values")


def _as_visual_jacobian(values: np.ndarray) -> np.ndarray:
    jacobian = np.asarray(values, dtype=np.float64)
    if jacobian.shape != (len(CAMERAS), len(PIXEL_AXES), len(COMMAND_AXES)):
        raise ValueError(
            "visual_jacobian_px_per_cmd_mm must have shape "
            "(camera, pixel_axis, command_axis)"
        )
    if not np.isfinite(jacobian).all():
        raise ValueError("visual_jacobian_px_per_cmd_mm must contain only finite values")
    return jacobian


def _jacobian_to_observation(jacobian: np.ndarray) -> np.ndarray:
    return _as_visual_jacobian(jacobian).reshape(
        len(OBSERVATION_AXES),
        len(COMMAND_AXES),
    )


def _observation_weight_matrix(
    weights: Sequence[float] | np.ndarray | None,
) -> np.ndarray:
    if weights is None:
        return np.eye(len(OBSERVATION_AXES), dtype=np.float64)
    values = np.asarray(weights, dtype=np.float64)
    if values.shape == (len(CAMERAS), len(PIXEL_AXES)):
        values = values.reshape(-1)
    if values.shape != (len(OBSERVATION_AXES),):
        raise ValueError("weights must have four observation values")
    if not np.isfinite(values).all() or np.any(values < 0.0):
        raise ValueError("weights must contain finite non-negative values")
    if not np.any(values > 0.0):
        raise ValueError("weights must include at least one positive value")
    return np.diag(values)


def _pad_warnings(
    values: Sequence[Sequence[Sequence[str]]] | None,
    count: int,
) -> tuple[tuple[tuple[str, ...], tuple[str, ...]], ...]:
    if values is None:
        return tuple(((), ()) for _ in range(count))
    if len(values) != count:
        raise ValueError(f"expected {count} warning lists, got {len(values)}")
    result: list[tuple[tuple[str, ...], tuple[str, ...]]] = []
    for sample_warnings in values:
        if len(sample_warnings) != len(CAMERAS):
            raise ValueError(f"expected {len(CAMERAS)} camera warning lists per sample")
        result.append(
            (
                tuple(sample_warnings[0]),
                tuple(sample_warnings[1]),
            )
        )
    return tuple(result)

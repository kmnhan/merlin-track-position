"""Two-camera motor-axis calibration from observed image shifts."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Sequence

import numpy as np
import xarray as xr

from merlin_track_position import constants
from merlin_track_position.tracking.shift import estimate_shift

CAMERAS = ("cam0", "cam1")
STAGE_AXES = ("x_um", "y_um", "z_um")
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


def fit_calibration_from_images(
    images_cam0: Sequence[Any],
    images_cam1: Sequence[Any],
    stage_um: Sequence[Sequence[float]],
    *,
    origin_stability_um: float,
    residual_warning_px: float = 1.0,
    condition_warning_threshold: float = 50.0,
    additional_context: dict[str, Any] | None = None,
    progress_callback: ProgressCallback | None = None,
    n_jobs: int | None = None,
    **shift_kwargs: Any,
) -> xr.Dataset:
    """Fit a 3D stage calibration from two camera image stacks.

    The model is linear through the origin:

    ``[du0, dv0, du1, dv1] = J @ [dx_um, dy_um, dz_um]``.
    """

    if "reference_index" in shift_kwargs:
        raise TypeError(
            "fit_calibration_from_images() got an unexpected keyword argument "
            "'reference_index'"
        )
    n_jobs = _resolve_n_jobs(n_jobs)

    capture_arrays_cam0, image_dtypes_cam0 = _as_capture_arrays(
        "images_cam0",
        images_cam0,
    )
    capture_arrays_cam1, image_dtypes_cam1 = _as_capture_arrays(
        "images_cam1",
        images_cam1,
    )
    capture_count = _common_capture_count(capture_arrays_cam0, capture_arrays_cam1)
    stage = np.asarray(stage_um, dtype=np.float64)
    origin_stability_um = float(origin_stability_um)

    if not np.isfinite(origin_stability_um) or origin_stability_um <= 0.0:
        raise ValueError("origin_stability_um must be finite and positive")
    if len(capture_arrays_cam0) < 5:
        raise ValueError("at least five calibration image pairs are required")
    if len(capture_arrays_cam0) != len(capture_arrays_cam1):
        raise ValueError(
            "images_cam0 and images_cam1 must have the same length; "
            f"got {len(capture_arrays_cam0)} and {len(capture_arrays_cam1)}"
        )
    if stage.ndim != 2 or stage.shape[1] != 3:
        raise ValueError("stage_um must have shape (n, 3)")
    if not np.isfinite(stage).all():
        raise ValueError("stage_um must contain only finite values")
    if len(capture_arrays_cam0) != stage.shape[0]:
        raise ValueError(
            f"image pairs and stage_um must have the same length; "
            f"got {len(capture_arrays_cam0)} and {stage.shape[0]}"
        )
    if not np.allclose(stage[0], 0.0, rtol=0.0, atol=1e-9):
        raise ValueError("stage_um[0] must be the origin")

    reference_cam0 = _mean_capture_image(capture_arrays_cam0[0])
    reference_cam1 = _mean_capture_image(capture_arrays_cam1[0])
    pixels, capture_shift_mad_px, measurement_warnings = (
        _estimate_calibration_capture_shifts(
            (capture_arrays_cam0, capture_arrays_cam1),
            (reference_cam0, reference_cam1),
            progress_callback=progress_callback,
            n_jobs=n_jobs,
            **shift_kwargs,
        )
    )
    observations = _pixels_to_observations(pixels)

    rank = int(np.linalg.matrix_rank(stage))
    if rank < 3:
        raise ValueError("stage positions must span three independent motor axes")

    coef, _, _, _ = np.linalg.lstsq(stage, observations, rcond=None)
    stage_to_observation = coef.T
    singular_values = np.linalg.svd(stage_to_observation, compute_uv=False)
    largest_singular_value = float(singular_values[0]) if singular_values.size else 0.0
    calibration_rank_tolerance = 1e-3 * largest_singular_value
    calibration_rank = int(
        np.count_nonzero(singular_values > calibration_rank_tolerance)
    )
    if calibration_rank < 3:
        raise ValueError(
            "fitted two-camera calibration matrix must have rank 3; "
            f"got rank {calibration_rank}"
        )
    stage_to_pixel = stage_to_observation.reshape(
        len(CAMERAS),
        len(PIXEL_AXES),
        len(STAGE_AXES),
    )

    condition_number = float(np.linalg.cond(stage_to_observation))
    predicted_observations = stage @ coef
    predicted = predicted_observations.reshape(
        stage.shape[0],
        len(CAMERAS),
        len(PIXEL_AXES),
    )
    residual_px = pixels - predicted
    pixel_to_stage = np.linalg.pinv(stage_to_observation)
    return_to_origin_motor_error_norm_um = float(np.linalg.norm(stage[-1]))
    return_to_origin_image_error_um = observations[-1] @ pixel_to_stage.T
    return_to_origin_image_error_norm_um = float(
        np.linalg.norm(return_to_origin_image_error_um)
    )

    warnings: list[str] = []
    residual_rms_px = float(
        np.sqrt(np.mean(np.sum(residual_px * residual_px, axis=(1, 2))))
    )
    if residual_rms_px > residual_warning_px:
        warnings.append(
            f"calibration residual RMS {residual_rms_px:.3g} px exceeds "
            f"{residual_warning_px:.3g} px"
        )
    if condition_number > condition_warning_threshold:
        warnings.append(
            f"calibration matrix is poorly conditioned: condition number "
            f"{condition_number:.3g} > {condition_warning_threshold:.3g}"
        )
    if return_to_origin_motor_error_norm_um > origin_stability_um:
        warnings.append(
            f"return-to-origin motor error "
            f"{return_to_origin_motor_error_norm_um:.3g} um exceeds "
            f"{origin_stability_um:.3g} um"
        )
    if return_to_origin_image_error_norm_um > origin_stability_um:
        warnings.append(
            f"return-to-origin image error "
            f"{return_to_origin_image_error_norm_um:.3g} um exceeds "
            f"{origin_stability_um:.3g} um"
        )

    measurement_warnings_tuple = _pad_warnings(
        measurement_warnings,
        stage.shape[0],
    )
    warnings.extend(format_measurement_warning_lines(measurement_warnings_tuple, stage))

    sample_count = stage.shape[0]
    coords: dict[str, Any] = {
        "sample": np.arange(sample_count, dtype=np.int64),
        "stage_axis": list(STAGE_AXES),
        "camera": list(CAMERAS),
        "pixel_axis": list(PIXEL_AXES),
    }
    data_vars: dict[str, Any] = {
        "stage_to_pixel": (
            ("camera", "pixel_axis", "stage_axis"),
            stage_to_pixel,
            {"units": "px/um"},
        ),
        "stage_um": (("sample", "stage_axis"), stage, {"units": "um"}),
        "measured_shift_px": (
            ("sample", "camera", "pixel_axis"),
            pixels,
            {"units": "px"},
        ),
        "capture_shift_mad_px": (
            ("sample", "camera", "pixel_axis"),
            capture_shift_mad_px,
            {
                "units": "px",
                "description": (
                    "median absolute deviation of per-capture shift estimates"
                ),
            },
        ),
    }
    if any(
        warning
        for sample_warnings in measurement_warnings_tuple
        for camera_warnings in sample_warnings
        for warning in camera_warnings
    ):
        data_vars["measurement_warnings"] = (
            ("sample", "camera"),
            np.asarray(
                [
                    ["\n".join(camera_warnings) for camera_warnings in sample_warnings]
                    for sample_warnings in measurement_warnings_tuple
                ],
                dtype=str,
            ),
        )

    images_cam0_stack = np.stack(
        [
            _representative_capture_image(captures, dtype)
            for captures, dtype in zip(
                capture_arrays_cam0,
                image_dtypes_cam0,
                strict=True,
            )
        ],
        axis=0,
    )
    images_cam1_stack = np.stack(
        [
            _representative_capture_image(captures, dtype)
            for captures, dtype in zip(
                capture_arrays_cam1,
                image_dtypes_cam1,
                strict=True,
            )
        ],
        axis=0,
    )
    coords["y_cam0"] = np.arange(images_cam0_stack.shape[1], dtype=np.int64)
    coords["x_cam0"] = np.arange(images_cam0_stack.shape[2], dtype=np.int64)
    coords["y_cam1"] = np.arange(images_cam1_stack.shape[1], dtype=np.int64)
    coords["x_cam1"] = np.arange(images_cam1_stack.shape[2], dtype=np.int64)
    data_vars["image_cam0"] = (
        ("sample", "y_cam0", "x_cam0"),
        images_cam0_stack,
        {"description": "camera 0 calibration grayscale image stack"},
    )
    data_vars["image_cam1"] = (
        ("sample", "y_cam1", "x_cam1"),
        images_cam1_stack,
        {"description": "camera 1 calibration grayscale image stack"},
    )

    calibration_attrs = {
        "format_version": "1",
        "capture_count": capture_count,
        "warnings": "\n".join(tuple(warnings)),
    }

    if additional_context is not None:
        calibration_attrs = calibration_attrs | additional_context

    return xr.Dataset(data_vars=data_vars, coords=coords, attrs=calibration_attrs)


def _estimate_calibration_capture_shifts(
    capture_arrays_by_camera: tuple[Sequence[np.ndarray], Sequence[np.ndarray]],
    references_by_camera: tuple[np.ndarray, np.ndarray],
    *,
    progress_callback: ProgressCallback | None,
    n_jobs: int,
    **shift_kwargs: Any,
) -> tuple[
    np.ndarray,
    np.ndarray,
    list[tuple[tuple[str, ...], tuple[str, ...]]],
]:
    sample_count = len(capture_arrays_by_camera[0])
    capture_count = int(capture_arrays_by_camera[0][0].shape[0])
    camera_count = len(CAMERAS)
    total = sample_count * camera_count * capture_count
    if progress_callback is not None:
        progress_callback(0, total)

    shift_array = np.empty(
        (sample_count, camera_count, capture_count, len(PIXEL_AXES)),
        dtype=np.float64,
    )
    warnings_by_capture: list[list[list[tuple[str, ...]]]] = [
        [[() for _ in range(capture_count)] for _ in CAMERAS]
        for _ in range(sample_count)
    ]
    tasks: list[tuple[int, int, int, np.ndarray, np.ndarray]] = []
    for sample_index in range(sample_count):
        for camera_index, capture_arrays in enumerate(capture_arrays_by_camera):
            captures = capture_arrays[sample_index]
            for capture_index, current in enumerate(captures):
                tasks.append(
                    (
                        sample_index,
                        camera_index,
                        capture_index,
                        references_by_camera[camera_index],
                        current,
                    )
                )

    completed = 0
    for sample_index, camera_index, capture_index, shift_px, capture_warnings in (
        _iter_indexed_capture_shift_results(
            tasks,
            capture_count,
            n_jobs=n_jobs,
            **shift_kwargs,
        )
    ):
        shift_array[sample_index, camera_index, capture_index, :] = shift_px
        warnings_by_capture[sample_index][camera_index][capture_index] = (
            capture_warnings
        )
        completed += 1
        if progress_callback is not None:
            progress_callback(completed, total)

    pixels = np.empty(
        (sample_count, camera_count, len(PIXEL_AXES)),
        dtype=np.float64,
    )
    capture_shift_mad_px = np.empty_like(pixels)
    measurement_warnings: list[tuple[tuple[str, ...], tuple[str, ...]]] = []
    for sample_index in range(sample_count):
        sample_warnings: list[tuple[str, ...]] = []
        for camera_index in range(camera_count):
            median_shift, shift_mad, camera_warnings = _summarize_capture_shifts(
                (
                    (
                        shift_array[sample_index, camera_index, capture_index],
                        warnings_by_capture[sample_index][camera_index][capture_index],
                    )
                    for capture_index in range(capture_count)
                )
            )
            pixels[sample_index, camera_index, :] = median_shift
            capture_shift_mad_px[sample_index, camera_index, :] = shift_mad
            sample_warnings.append(camera_warnings)
        measurement_warnings.append((sample_warnings[0], sample_warnings[1]))

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


def estimate_stage_offset(
    calibration: xr.Dataset,
    shift: xr.Dataset | xr.DataArray | Sequence[float],
) -> np.ndarray:
    """Convert observed two-camera image shifts to estimated stage offset."""

    if isinstance(shift, xr.Dataset):
        shift_values = np.asarray(shift["shift_px"].values, dtype=np.float64)
    elif isinstance(shift, xr.DataArray):
        shift_values = np.asarray(shift.values, dtype=np.float64)
    else:
        shift_values = np.asarray(shift, dtype=np.float64)
    observation = _shift_to_observation(shift_values)
    stage_to_pixel = np.asarray(calibration["stage_to_pixel"].values, dtype=np.float64)
    stage_to_observation = stage_to_pixel.reshape(
        len(OBSERVATION_AXES),
        len(STAGE_AXES),
    )
    pixel_to_stage = np.linalg.pinv(stage_to_observation)
    return pixel_to_stage @ observation


def get_correction(
    calibration: xr.Dataset,
    reference_cam0: Any,
    current_cam0: Any,
    reference_cam1: Any,
    current_cam1: Any,
    **shift_kwargs: Any,
) -> xr.Dataset:
    """Estimate two-camera image shift and return calibrated motor correction."""

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

    estimated_offset = estimate_stage_offset(calibration, shift_px)
    correction = -estimated_offset
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
            "estimated_stage_offset_um": (
                ("stage_axis",),
                estimated_offset,
                {"units": "um"},
            ),
            "correction_um": (("stage_axis",), correction, {"units": "um"}),
            "current_cam0": (("y_cam0", "x_cam0"), current_image_cam0),
            "current_cam1": (("y_cam1", "x_cam1"), current_image_cam1),
        },
        coords={
            "camera": list(CAMERAS),
            "pixel_axis": list(PIXEL_AXES),
            "stage_axis": list(STAGE_AXES),
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
    lines = [
        *(f"cam0 {line}" for line in warnings_cam0 if line),
        *(f"cam1 {line}" for line in warnings_cam1 if line),
    ]
    return tuple(lines)


def format_measurement_warning_lines(
    measurement_warnings: Sequence[Sequence[Sequence[str]]] | None,
    stage_um: Sequence[Sequence[float]],
) -> tuple[str, ...]:
    """Return display-ready per-step image matching warning lines."""
    stage = np.asarray(stage_um, dtype=np.float64)
    if stage.ndim != 2 or stage.shape[1] != len(STAGE_AXES):
        raise ValueError("stage_um must have shape (sample, 3)")

    warning_rows = _pad_warnings(measurement_warnings, stage.shape[0])
    lines: list[str] = []
    for sample_index, sample_warnings in enumerate(warning_rows):
        stage_text = _format_stage_warning_context(stage[sample_index])
        for camera, camera_warnings in zip(CAMERAS, sample_warnings, strict=True):
            for warning in camera_warnings:
                warning_text = str(warning).strip()
                if warning_text:
                    lines.append(
                        f"step {sample_index + 1} ({stage_text}), "
                        f"{camera}: {warning_text}"
                    )
    return tuple(lines)


def _format_stage_warning_context(stage_row: np.ndarray) -> str:
    axis_values = ", ".join(
        f"{axis.removesuffix('_um')}={_format_warning_number(value)}"
        for axis, value in zip(STAGE_AXES, stage_row, strict=True)
    )
    return f"{axis_values} um"


def _format_warning_number(value: float) -> str:
    return f"{float(value):.4g}"


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


def _shift_to_observation(shift_values: np.ndarray) -> np.ndarray:
    values = np.asarray(shift_values, dtype=np.float64)
    if values.shape == (len(CAMERAS), len(PIXEL_AXES)):
        return values.reshape(-1)
    if values.size == len(OBSERVATION_AXES):
        return values.reshape(-1)
    raise ValueError("shift must have shape (2, 2) or contain four observation values")


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

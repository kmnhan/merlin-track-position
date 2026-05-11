"""Two-camera motor-axis calibration from observed image shifts."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import xarray as xr

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


def fit_calibration_from_images(
    images_cam0: Sequence[Any],
    images_cam1: Sequence[Any],
    stage_um: Sequence[Sequence[float]],
    *,
    origin_stability_um: float,
    residual_warning_px: float = 1.0,
    condition_warning_threshold: float = 50.0,
    additional_context: dict[str, Any] | None = None,
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

    image_arrays_cam0 = _as_image_arrays("images_cam0", images_cam0)
    image_arrays_cam1 = _as_image_arrays("images_cam1", images_cam1)
    stage = np.asarray(stage_um, dtype=np.float64)
    origin_stability_um = float(origin_stability_um)

    if not np.isfinite(origin_stability_um) or origin_stability_um <= 0.0:
        raise ValueError("origin_stability_um must be finite and positive")
    if len(image_arrays_cam0) < 5:
        raise ValueError("at least five calibration image pairs are required")
    if len(image_arrays_cam0) != len(image_arrays_cam1):
        raise ValueError(
            "images_cam0 and images_cam1 must have the same length; "
            f"got {len(image_arrays_cam0)} and {len(image_arrays_cam1)}"
        )
    if stage.ndim != 2 or stage.shape[1] != 3:
        raise ValueError("stage_um must have shape (n, 3)")
    if not np.isfinite(stage).all():
        raise ValueError("stage_um must contain only finite values")
    if len(image_arrays_cam0) != stage.shape[0]:
        raise ValueError(
            f"image pairs and stage_um must have the same length; "
            f"got {len(image_arrays_cam0)} and {stage.shape[0]}"
        )
    if not np.allclose(stage[0], 0.0, rtol=0.0, atol=1e-9):
        raise ValueError("stage_um[0] must be the origin")

    reference_cam0 = image_arrays_cam0[0]
    reference_cam1 = image_arrays_cam1[0]
    shifts_cam0: list[np.ndarray] = []
    shifts_cam1: list[np.ndarray] = []
    measurement_warnings: list[tuple[tuple[str, ...], tuple[str, ...]]] = []
    for image_cam0, image_cam1 in zip(
        image_arrays_cam0,
        image_arrays_cam1,
        strict=True,
    ):
        shift_cam0 = estimate_shift(reference_cam0, image_cam0, **shift_kwargs)
        shift_cam1 = estimate_shift(reference_cam1, image_cam1, **shift_kwargs)
        shifts_cam0.append(
            np.asarray(shift_cam0["shift_px"].values, dtype=np.float64)
        )
        shifts_cam1.append(
            np.asarray(shift_cam1["shift_px"].values, dtype=np.float64)
        )
        measurement_warnings.append(
            (
                tuple(str(shift_cam0.attrs.get("warnings", "")).splitlines()),
                tuple(str(shift_cam1.attrs.get("warnings", "")).splitlines()),
            )
        )

    pixels = np.stack([np.vstack(shifts_cam0), np.vstack(shifts_cam1)], axis=1)
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
    return_to_origin_motor_error_norm_um = float(
        np.linalg.norm(stage[-1])
    )
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
    warnings.extend(
        format_measurement_warning_lines(measurement_warnings_tuple, stage)
    )

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

    images_cam0_stack = np.stack(image_arrays_cam0, axis=0)
    images_cam1_stack = np.stack(image_arrays_cam1, axis=0)
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
        "warnings": "\n".join(tuple(warnings)),
    }

    if additional_context is not None:
        calibration_attrs = calibration_attrs | additional_context

    return xr.Dataset(data_vars=data_vars, coords=coords, attrs=calibration_attrs)


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


def correct(
    calibration: xr.Dataset,
    reference_cam0: Any,
    current_cam0: Any,
    reference_cam1: Any,
    current_cam1: Any,
    **shift_kwargs: Any,
) -> xr.Dataset:
    """Estimate two-camera image shift and return calibrated motor correction."""

    shift_cam0 = estimate_shift(reference_cam0, current_cam0, **shift_kwargs)
    shift_cam1 = estimate_shift(reference_cam1, current_cam1, **shift_kwargs)
    shift_px = np.stack(
        [
            np.asarray(shift_cam0["shift_px"].values, dtype=np.float64),
            np.asarray(shift_cam1["shift_px"].values, dtype=np.float64),
        ],
        axis=0,
    )
    estimated_offset = estimate_stage_offset(calibration, shift_px)
    correction = -estimated_offset
    calibration_warnings = str(calibration.attrs.get("warnings", "")).splitlines()
    shift_warnings = [
        line
        for line in (
            *str(shift_cam0.attrs.get("warnings", "")).splitlines(),
            *str(shift_cam1.attrs.get("warnings", "")).splitlines(),
        )
        if line
    ]
    return xr.Dataset(
        data_vars={
            "shift_px": (("camera", "pixel_axis"), shift_px, {"units": "px"}),
            "estimated_stage_offset_um": (
                ("stage_axis",),
                estimated_offset,
                {"units": "um"},
            ),
            "correction_um": (("stage_axis",), correction, {"units": "um"}),
            "registration_error": (
                ("camera",),
                [
                    float(shift_cam0["registration_error"].values),
                    float(shift_cam1["registration_error"].values),
                ],
            ),
            "phase_difference": (
                ("camera",),
                [
                    float(shift_cam0["phase_difference"].values),
                    float(shift_cam1["phase_difference"].values),
                ],
            ),
            "texture_dynamic_range": (
                ("camera",),
                [
                    float(shift_cam0["texture_dynamic_range"].values),
                    float(shift_cam1["texture_dynamic_range"].values),
                ],
            ),
            "texture_gradient_rms": (
                ("camera",),
                [
                    float(shift_cam0["texture_gradient_rms"].values),
                    float(shift_cam1["texture_gradient_rms"].values),
                ],
            ),
            "tile_median_shift_px": (
                ("camera", "pixel_axis"),
                np.stack(
                    [
                        np.asarray(
                            shift_cam0["tile_median_shift_px"].values,
                            dtype=np.float64,
                        ),
                        np.asarray(
                            shift_cam1["tile_median_shift_px"].values,
                            dtype=np.float64,
                        ),
                    ],
                    axis=0,
                ),
                {"units": "px"},
            ),
            "tile_shift_std_px": (
                ("camera",),
                [
                    float(shift_cam0["tile_shift_std_px"].values),
                    float(shift_cam1["tile_shift_std_px"].values),
                ],
                {"units": "px"},
            ),
        },
        coords={
            "camera": list(CAMERAS),
            "pixel_axis": list(PIXEL_AXES),
            "stage_axis": list(STAGE_AXES),
        },
        attrs={
            "method": "calibrated_stereo_shift_correction",
            "warnings": "\n".join([*calibration_warnings, *shift_warnings]),
        },
    )


def _as_image_arrays(name: str, images: Sequence[Any]) -> list[np.ndarray]:
    image_arrays = [np.asarray(image, dtype=np.float64) for image in images]
    if not image_arrays:
        raise ValueError(f"{name} must not be empty")
    shape = image_arrays[0].shape
    for index, image in enumerate(image_arrays):
        if image.ndim != 2:
            raise ValueError(f"{name}[{index}] must be 2D, got {image.shape!r}")
        if image.shape != shape:
            raise ValueError(
                f"all images in {name} must have the same shape; "
                f"image 0 has {shape!r}, image {index} has {image.shape!r}"
            )
        if not np.isfinite(image).all():
            raise ValueError(f"{name}[{index}] must contain only finite values")
    return image_arrays


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
    raise ValueError(
        "shift must have shape (2, 2) or contain four observation values"
    )


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
            raise ValueError(
                f"expected {len(CAMERAS)} camera warning lists per sample"
            )
        result.append(
            (
                tuple(sample_warnings[0]),
                tuple(sample_warnings[1]),
            )
        )
    return tuple(result)

"""2D motor-axis calibration from observed image shifts and stage positions."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import xarray as xr

from merlin_track_position.tracking.shift import estimate_shift

STAGE_AXES = ("stage_a_um", "stage_b_um")
PIXEL_AXES = ("du_px", "dv_px")


def fit_calibration_from_images(
    images: Sequence[Any],
    stage_um: Sequence[Sequence[float]],
    *,
    origin_stability_um: float,
    residual_warning_px: float = 1.0,
    condition_warning_threshold: float = 50.0,
    additional_context: dict[str, Any] | None = None,
    **shift_kwargs: Any,
) -> xr.Dataset:
    """Fit calibration directly from grayscale image arrays and stage offsets.

    Parameters
    ----------
    images
        Sequence of 2D grayscale images.
    stage_um
        Sequence of stage positions in microns corresponding to each image,
        expressed as encoder displacements relative to the first image:
        (stage_a_um, stage_b_um). The first row must be the origin.
    origin_stability_um
        Warning threshold in microns for the final return-to-origin motor and
        image error checks.

    """

    image_arrays = [np.asarray(image, dtype=np.float64) for image in images]
    stage = np.asarray(stage_um, dtype=np.float64)
    origin_stability_um = float(origin_stability_um)
    if "reference_index" in shift_kwargs:
        raise TypeError(
            "fit_calibration_from_images() got an unexpected keyword argument "
            "'reference_index'"
        )
    if not np.isfinite(origin_stability_um) or origin_stability_um <= 0.0:
        raise ValueError("origin_stability_um must be finite and positive")
    if len(image_arrays) < 4:
        raise ValueError("at least four calibration images are required")
    if stage.ndim != 2 or stage.shape[1] != 2:
        raise ValueError("stage_um must have shape (n, 2)")
    if not np.isfinite(stage).all():
        raise ValueError("stage_um must contain only finite values")
    if len(image_arrays) != stage.shape[0]:
        raise ValueError(
            f"images and stage_um must have the same length; "
            f"got {len(image_arrays)} and {stage.shape[0]}"
        )
    if not np.allclose(stage[0], 0.0, rtol=0.0, atol=1e-9):
        raise ValueError("stage_um[0] must be the origin")
    shape = image_arrays[0].shape
    for index, image in enumerate(image_arrays):
        if image.shape != shape:
            raise ValueError(
                f"all images must have the same shape; image 0 has {shape!r}, "
                f"image {index} has {image.shape!r}"
            )

    reference_image = image_arrays[0]
    shifts: list[np.ndarray] = []
    measurement_warnings: list[tuple[str, ...]] = []
    for image in image_arrays:
        shift = estimate_shift(reference_image, image, **shift_kwargs)
        shifts.append(np.asarray(shift["shift_px"].values, dtype=np.float64))
        measurement_warnings.append(
            tuple(str(shift.attrs.get("warnings", "")).splitlines())
        )

    pixels = np.vstack(shifts)
    images = np.stack(image_arrays, axis=0)
    reference_stage_um = np.zeros(2, dtype=np.float64)

    rank = int(np.linalg.matrix_rank(stage))
    if rank < 2:
        raise ValueError("stage positions must span two independent motor directions")

    coef, _, _, _ = np.linalg.lstsq(stage, pixels, rcond=None)
    stage_to_pixel = coef.T

    condition_number = float(np.linalg.cond(stage_to_pixel))
    predicted = stage @ stage_to_pixel.T
    residual_px = pixels - predicted
    pixel_to_stage = np.linalg.inv(stage_to_pixel)
    residual_um = residual_px @ pixel_to_stage.T
    return_to_origin_motor_error_um = stage[-1]
    return_to_origin_motor_error_norm_um = float(
        np.linalg.norm(return_to_origin_motor_error_um)
    )
    return_to_origin_image_error_px = pixels[-1]
    return_to_origin_image_error_um = return_to_origin_image_error_px @ pixel_to_stage.T
    return_to_origin_image_error_norm_um = float(
        np.linalg.norm(return_to_origin_image_error_um)
    )

    warnings: list[str] = []
    residual_rms_px = float(np.sqrt(np.mean(np.sum(residual_px * residual_px, axis=1))))
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
        measurement_warnings, stage.shape[0]
    )
    if any(measurement_warnings_tuple):
        warnings.append(
            "one or more shift measurements reported image-matching warnings"
        )

    sample_count = stage.shape[0]
    coords: dict[str, Any] = {
        "sample": np.arange(sample_count, dtype=np.int64),
        "stage_axis": list(STAGE_AXES),
        "pixel_axis": list(PIXEL_AXES),
    }
    data_vars: dict[str, Any] = {
        "stage_to_pixel": (
            ("pixel_axis", "stage_axis"),
            stage_to_pixel,
            {"units": "px/um"},
        ),
        "pixel_to_stage": (
            ("stage_axis", "pixel_axis"),
            np.linalg.inv(stage_to_pixel),
            {"units": "um/px"},
        ),
        "reference_stage_um": (
            ("stage_axis",),
            np.asarray(reference_stage_um, dtype=np.float64),
            {"units": "um"},
        ),
        "condition_number": ((), float(condition_number)),
        "origin_stability_um": ((), origin_stability_um, {"units": "um"}),
        "return_to_origin_motor_error_um": (
            ("stage_axis",),
            return_to_origin_motor_error_um,
            {"units": "um"},
        ),
        "return_to_origin_motor_error_norm_um": (
            (),
            return_to_origin_motor_error_norm_um,
            {"units": "um"},
        ),
        "return_to_origin_image_error_px": (
            ("pixel_axis",),
            return_to_origin_image_error_px,
            {"units": "px"},
        ),
        "return_to_origin_image_error_um": (
            ("stage_axis",),
            return_to_origin_image_error_um,
            {"units": "um"},
        ),
        "return_to_origin_image_error_norm_um": (
            (),
            return_to_origin_image_error_norm_um,
            {"units": "um"},
        ),
        "stage_um": (("sample", "stage_axis"), stage, {"units": "um"}),
        "measured_shift_px": (("sample", "pixel_axis"), pixels, {"units": "px"}),
        "predicted_shift_px": (("sample", "pixel_axis"), predicted, {"units": "px"}),
        "residual_shift_px": (("sample", "pixel_axis"), residual_px, {"units": "px"}),
        "residual_stage_um": (("sample", "stage_axis"), residual_um, {"units": "um"}),
        "measurement_warnings": (
            ("sample",),
            np.asarray(
                ["\n".join(items) for items in measurement_warnings_tuple], dtype=str
            ),
        ),
    }

    repeatability = _repeatability(stage, pixels)
    if repeatability is not None:
        (
            repeatability_stage_um,
            repeatability_count,
            repeatability_mean,
            repeatability_std,
        ) = repeatability
        coords["repeatability_position"] = np.arange(
            repeatability_stage_um.shape[0], dtype=np.int64
        )
        data_vars["repeatability_stage_um"] = (
            ("repeatability_position", "stage_axis"),
            repeatability_stage_um,
            {"units": "um"},
        )
        data_vars["repeatability_count"] = (
            ("repeatability_position",),
            repeatability_count,
        )
        data_vars["repeatability_mean_shift_px"] = (
            ("repeatability_position", "pixel_axis"),
            repeatability_mean,
            {"units": "px"},
        )
        data_vars["repeatability_std_shift_px"] = (
            ("repeatability_position", "pixel_axis"),
            repeatability_std,
            {"units": "px"},
        )
        data_vars["repeatability_rms_std_px"] = (
            ("repeatability_position",),
            np.sqrt(np.mean(repeatability_std * repeatability_std, axis=1)),
            {"units": "px"},
        )

    if images is not None:
        coords["y"] = np.arange(images.shape[1], dtype=np.int64)
        coords["x"] = np.arange(images.shape[2], dtype=np.int64)
        data_vars["image"] = (
            ("sample", "y", "x"),
            images,
            {"description": "calibration grayscale image stack"},
        )

    calibration_attrs = {
        "format": "merlin-track-position calibration",
        "format_version": "1",
        "warnings": "\n".join(tuple(warnings)),
    }

    if additional_context is not None:
        calibration_attrs = calibration_attrs | additional_context

    return xr.Dataset(data_vars=data_vars, coords=coords, attrs=calibration_attrs)


def estimate_stage_offset(
    calibration: xr.Dataset, shift: xr.Dataset | Sequence[float]
) -> np.ndarray:
    """Convert an observed image shift to estimated stage-plane offset."""

    shift_px = (
        np.asarray(shift["shift_px"].values, dtype=np.float64)
        if isinstance(shift, xr.Dataset)
        else np.asarray(shift, dtype=float)
    )
    pixel_to_stage = np.asarray(calibration["pixel_to_stage"].values, dtype=np.float64)
    return pixel_to_stage @ shift_px


def correct(
    calibration: xr.Dataset,
    reference: Any,
    current: Any,
    **shift_kwargs: Any,
) -> xr.Dataset:
    """Estimate image shift and return the calibrated motor correction."""

    shift = estimate_shift(reference, current, **shift_kwargs)
    estimated_offset = estimate_stage_offset(calibration, shift)
    correction = -estimated_offset
    calibration_warnings = str(calibration.attrs.get("warnings", "")).splitlines()
    shift_warnings = str(shift.attrs.get("warnings", "")).splitlines()
    return xr.Dataset(
        data_vars={
            "shift_px": shift["shift_px"],
            "estimated_stage_offset_um": (
                ("stage_axis",),
                estimated_offset,
                {"units": "um"},
            ),
            "correction_um": (("stage_axis",), correction, {"units": "um"}),
            "registration_error": shift["registration_error"],
            "phase_difference": shift["phase_difference"],
            "texture_dynamic_range": shift["texture_dynamic_range"],
            "texture_gradient_rms": shift["texture_gradient_rms"],
            "tile_median_shift_px": shift["tile_median_shift_px"],
            "tile_shift_std_px": shift["tile_shift_std_px"],
        },
        coords={
            "pixel_axis": list(PIXEL_AXES),
            "stage_axis": list(STAGE_AXES),
        },
        attrs={
            "method": "calibrated_shift_correction",
            "warnings": "\n".join([*calibration_warnings, *shift_warnings]),
        },
    )


def _repeatability(
    stage: np.ndarray,
    pixels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    groups: dict[tuple[float, float], list[np.ndarray]] = {}
    for stage_row, pixel_row in zip(stage, pixels, strict=True):
        key = (float(stage_row[0]), float(stage_row[1]))
        groups.setdefault(key, []).append(pixel_row)

    stage_rows: list[np.ndarray] = []
    counts: list[int] = []
    means: list[np.ndarray] = []
    stds: list[np.ndarray] = []
    for key, rows in groups.items():
        if len(rows) < 2:
            continue
        values = np.vstack(rows)
        stage_rows.append(np.asarray(key, dtype=np.float64))
        counts.append(len(rows))
        means.append(np.mean(values, axis=0))
        stds.append(np.std(values, axis=0, ddof=1))

    if not stage_rows:
        return None
    return (
        np.vstack(stage_rows),
        np.asarray(counts, dtype=np.int64),
        np.vstack(means),
        np.vstack(stds),
    )


def _pad_warnings(
    values: Sequence[Sequence[str]] | None,
    count: int,
) -> tuple[tuple[str, ...], ...]:
    if values is None:
        return tuple(() for _ in range(count))
    if len(values) != count:
        raise ValueError(f"expected {count} warning lists, got {len(values)}")
    return tuple(tuple(item) for item in values)

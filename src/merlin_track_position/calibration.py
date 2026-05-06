"""2D motor-axis calibration from observed image shifts and stage positions."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import xarray as xr

from merlin_track_position.image_io import as_grayscale_array
from merlin_track_position.shift import estimate_shift

STAGE_AXES = ("stage_a_um", "stage_b_um")
PIXEL_AXES = ("du_px", "dv_px")


def fit_calibration_from_images(
    images: Sequence[Any],
    stage_um: Sequence[Sequence[float]],
    *,
    reference_index: int | None = None,
    residual_warning_px: float = 1.0,
    condition_warning_threshold: float = 50.0,
    **shift_kwargs: Any,
) -> xr.Dataset:
    """Fit calibration directly from grayscale image arrays and stage offsets."""

    image_arrays = [as_grayscale_array(image) for image in images]
    stage = np.asarray(stage_um, dtype=np.float64)
    if len(image_arrays) < 3:
        raise ValueError("at least three calibration images are required")
    if stage.ndim != 2 or stage.shape[1] != 2:
        raise ValueError("stage_um must have shape (n, 2)")
    if len(image_arrays) != stage.shape[0]:
        raise ValueError(
            f"images and stage_um must have the same length; "
            f"got {len(image_arrays)} and {stage.shape[0]}"
        )
    shape = image_arrays[0].shape
    for index, image in enumerate(image_arrays):
        if image.shape != shape:
            raise ValueError(
                f"all images must have the same shape; image 0 has {shape!r}, "
                f"image {index} has {image.shape!r}"
            )
    if reference_index is None:
        reference_index = int(np.argmin(np.linalg.norm(stage, axis=1)))
    if reference_index < 0 or reference_index >= len(image_arrays):
        raise IndexError("reference_index is out of range")

    reference_image = image_arrays[reference_index]
    relative_stage = stage - stage[reference_index]
    shifts: list[np.ndarray] = []
    measurement_warnings: list[tuple[str, ...]] = []
    for image in image_arrays:
        shift = estimate_shift(reference_image, image, **shift_kwargs)
        shifts.append(np.asarray(shift["shift_px"].values, dtype=np.float64))
        measurement_warnings.append(
            tuple(str(shift.attrs.get("warnings", "")).splitlines())
        )

    return _fit_calibration_from_measurement_arrays(
        relative_stage,
        np.vstack(shifts),
        measurement_warnings=measurement_warnings,
        reference_stage_um=stage[reference_index],
        reference_index=reference_index,
        residual_warning_px=residual_warning_px,
        condition_warning_threshold=condition_warning_threshold,
        images=np.stack(image_arrays, axis=0),
    )


def fit_calibration_from_measurements(
    stage_um: Sequence[Sequence[float]],
    pixel_shift_px: Sequence[Sequence[float]],
    *,
    measurement_warnings: Sequence[Sequence[str]] | None = None,
    reference_stage_um: Sequence[float] = (0.0, 0.0),
    reference_index: int | None = None,
    residual_warning_px: float = 1.0,
    condition_warning_threshold: float = 50.0,
) -> xr.Dataset:
    """Fit ``pixel_shift = J @ stage_um`` from measured shifts."""

    stage = np.asarray(stage_um, dtype=np.float64)
    pixels = np.asarray(pixel_shift_px, dtype=np.float64)
    return _fit_calibration_from_measurement_arrays(
        stage,
        pixels,
        measurement_warnings=measurement_warnings,
        reference_stage_um=reference_stage_um,
        reference_index=reference_index,
        residual_warning_px=residual_warning_px,
        condition_warning_threshold=condition_warning_threshold,
    )


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


def _fit_calibration_from_measurement_arrays(
    stage: np.ndarray,
    pixels: np.ndarray,
    *,
    measurement_warnings: Sequence[Sequence[str]] | None = None,
    reference_stage_um: Sequence[float] = (0.0, 0.0),
    reference_index: int | None = None,
    residual_warning_px: float = 1.0,
    condition_warning_threshold: float = 50.0,
    images: np.ndarray | None = None,
) -> xr.Dataset:
    if stage.ndim != 2 or stage.shape[1] != 2:
        raise ValueError("stage_um must have shape (n, 2)")
    if pixels.ndim != 2 or pixels.shape != stage.shape:
        raise ValueError("pixel_shift_px must have shape (n, 2)")
    if stage.shape[0] < 3:
        raise ValueError("at least three measured points are required")

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

    measurement_warnings_tuple = _pad_warnings(measurement_warnings, stage.shape[0])
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

    return xr.Dataset(
        data_vars=data_vars,
        coords=coords,
        attrs={
            "format": "merlin-track-position calibration",
            "format_version": "1",
            "model": "through_origin_linear",
            "reference_index": -1 if reference_index is None else int(reference_index),
            "warnings": "\n".join(tuple(warnings)),
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

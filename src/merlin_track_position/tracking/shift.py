"""Subpixel whole-frame shift estimation for grayscale image pairs."""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import xarray as xr
from skimage.registration import phase_cross_correlation

PIXEL_AXES = ("du_px", "dv_px")


def normalize_intensity(
    image: Any,
    *,
    clip_percentiles: tuple[float, float] | None = (1.0, 99.0),
    eps: float = 1e-12,
) -> np.ndarray:
    """Robustly normalize a grayscale image to zero mean and unit variance."""

    working = np.asarray(image, dtype=np.float64)
    if working.ndim != 2:
        raise ValueError(f"image must be 2D, got shape {working.shape!r}")
    if working.size == 0:
        raise ValueError("image must not be empty")
    if not np.isfinite(working).all():
        raise ValueError("image must contain only finite values")
    working = working.copy()

    if clip_percentiles is not None:
        low, high = np.percentile(working, clip_percentiles)
        if high - low > eps:
            working = np.clip(working, low, high)

    mean = float(np.mean(working))
    std = float(np.std(working))
    if std <= eps:
        return np.zeros_like(working)
    return (working - mean) / std


def estimate_shift(
    reference: Any,
    current: Any,
    *,
    clip_percentiles: tuple[float, float] | None = (1.0, 99.0),
    use_window: bool = False,
    upsample_factor: int = 50,
    normalization: str | None = "phase",
    check_tiles: bool = False,
    high_error_threshold: float = 0.5,
) -> xr.Dataset:
    """Estimate subpixel translation between two grayscale images.

    By default, images are percentile-clipped, normalized, and registered with
    phase correlation at 50x subpixel upsampling.
    Pass ``use_window=True`` to apply a Hanning taper before registration.
    Pass ``check_tiles=True`` to compare local tile shifts against the full-frame
    estimate.
    """

    reference_image = np.asarray(reference, dtype=np.float64)
    current_image = np.asarray(current, dtype=np.float64)
    if reference_image.ndim != 2:
        raise ValueError(
            f"reference image must be 2D, got shape {reference_image.shape!r}"
        )
    if current_image.ndim != 2:
        raise ValueError(f"current image must be 2D, got shape {current_image.shape!r}")
    if reference_image.size == 0 or current_image.size == 0:
        raise ValueError("images must not be empty")
    if not np.isfinite(reference_image).all() or not np.isfinite(current_image).all():
        raise ValueError("images must contain only finite values")
    if reference_image.shape != current_image.shape:
        raise ValueError(
            "reference and current images must have identical shapes; "
            f"got {reference_image.shape!r} and {current_image.shape!r}"
        )

    dynamic_range = float(np.max(reference_image) - np.min(reference_image))
    standard_deviation = float(np.std(reference_image))
    gy, gx = np.gradient(reference_image)
    gradient_rms = float(np.sqrt(np.mean(gx * gx + gy * gy)))

    diagnostic_warnings: list[str] = []
    if dynamic_range <= 1e-12 or standard_deviation <= 1e-12:
        diagnostic_warnings.append(
            "reference image has little or no intensity contrast"
        )
    if gradient_rms <= max(1e-12, 1e-4 * max(dynamic_range, 1.0)):
        diagnostic_warnings.append(
            "reference image has low texture; shift may be unreliable"
        )

    if dynamic_range <= 1e-12 or standard_deviation <= 1e-12:
        reference_norm = normalize_intensity(
            reference_image, clip_percentiles=clip_percentiles
        )
        current_norm = normalize_intensity(
            current_image, clip_percentiles=clip_percentiles
        )
        shift_px = np.array([np.nan, np.nan], dtype=np.float64)
        registration_error = np.inf
    else:
        reference_norm = normalize_intensity(
            reference_image, clip_percentiles=clip_percentiles
        )
        current_norm = normalize_intensity(
            current_image, clip_percentiles=clip_percentiles
        )
        shift_px, registration_error, _, skimage_warnings = _estimate_translation(
            reference_norm,
            current_norm,
            use_window=use_window,
            upsample_factor=upsample_factor,
            normalization=normalization,
        )
        diagnostic_warnings.extend(
            f"skimage registration warning: {message}" for message in skimage_warnings
        )

    if not np.isfinite(registration_error):
        shift_px = np.array([np.nan, np.nan], dtype=np.float64)
        diagnostic_warnings.append(
            "registration error is not finite; shift estimate is unreliable"
        )
    elif normalization != "phase" and registration_error > high_error_threshold:
        diagnostic_warnings.append(
            f"high registration error: {registration_error:.3g} > {high_error_threshold:.3g}"
        )

    if check_tiles and np.isfinite(shift_px).all():
        tile_warning = _tile_consistency(
            reference_norm,
            current_norm,
            shift_px,
            use_window=use_window,
            upsample_factor=upsample_factor,
            normalization=normalization,
        )
        if tile_warning is not None:
            diagnostic_warnings.append(tile_warning)

    return xr.Dataset(
        data_vars={
            "shift_px": (("pixel_axis",), shift_px, {"units": "px"}),
        },
        coords={"pixel_axis": list(PIXEL_AXES)},
        attrs={"warnings": "\n".join(diagnostic_warnings)},
    )


def _estimate_translation(
    reference_norm: np.ndarray,
    current_norm: np.ndarray,
    *,
    use_window: bool,
    upsample_factor: int,
    normalization: str | None,
) -> tuple[np.ndarray, float, float, tuple[str, ...]]:
    if reference_norm.shape != current_norm.shape:
        raise ValueError("images must have identical shapes")
    if min(reference_norm.shape) < 3:
        raise ValueError("images must be at least 3x3 pixels")
    if upsample_factor < 1:
        raise ValueError("upsample_factor must be >= 1")

    reference_work = reference_norm
    current_work = current_norm
    if use_window:
        window = np.outer(
            np.hanning(reference_norm.shape[0]),
            np.hanning(reference_norm.shape[1]),
        )
        reference_work = reference_work * window
        current_work = current_work * window

    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always", UserWarning)
        shift_yx, registration_error, phase_difference = phase_cross_correlation(
            reference_work,
            current_work,
            upsample_factor=upsample_factor,
            normalization=normalization,
        )
    return (
        np.array([-shift_yx[1], -shift_yx[0]], dtype=np.float64),
        float(registration_error),
        float(phase_difference),
        tuple(str(item.message) for item in caught_warnings),
    )


def _tile_consistency(
    reference_norm: np.ndarray,
    current_norm: np.ndarray,
    full_shift_px: np.ndarray,
    *,
    use_window: bool,
    upsample_factor: int,
    normalization: str | None,
) -> str | None:
    height, width = reference_norm.shape
    grid = 3 if min(height, width) >= 192 else 2
    tile_height = height // grid
    tile_width = width // grid
    if min(tile_height, tile_width) < 48:
        return None

    shifts: list[np.ndarray] = []
    for row in range(grid):
        y0 = row * tile_height
        y1 = height if row == grid - 1 else (row + 1) * tile_height
        for col in range(grid):
            x0 = col * tile_width
            x1 = width if col == grid - 1 else (col + 1) * tile_width
            reference_tile = reference_norm[y0:y1, x0:x1]
            current_tile = current_norm[y0:y1, x0:x1]
            if float(np.std(reference_tile)) < 1e-6:
                continue
            tile_shift, registration_error, _, _ = _estimate_translation(
                reference_tile,
                current_tile,
                use_window=use_window,
                upsample_factor=upsample_factor,
                normalization=normalization,
            )
            if np.isfinite(tile_shift).all() and registration_error <= 0.75:
                shifts.append(tile_shift)

    if len(shifts) < max(3, grid):
        return None

    shift_array = np.vstack(shifts)
    median = np.median(shift_array, axis=0)
    distances = np.linalg.norm(shift_array - median, axis=1)
    tile_std = float(np.sqrt(np.mean(distances * distances)))
    full_distance = float(np.linalg.norm(full_shift_px - median))
    warning = None
    tolerance = max(2.0, 0.25 * float(np.linalg.norm(full_shift_px)))
    if tile_std > tolerance or full_distance > tolerance:
        warning = (
            "tile shift estimates are inconsistent with the whole-frame shift; "
            "static background or low contrast within ROI may be influencing the match"
        )
    return warning

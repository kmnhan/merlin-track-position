"""Subpixel whole-frame shift estimation for grayscale or RGB image pairs."""

from __future__ import annotations

import operator
from typing import Any

import cv2
import numpy as np
import numpy.typing as npt
import xarray as xr

PIXEL_AXES = ("du_px", "dv_px")
BASLER_RGB_TO_MONO_WEIGHTS = np.asarray((0.25, 0.625, 0.125), dtype=np.float32)


def normalize_intensity(
    image: Any,
    *,
    clip_percentiles: tuple[float, float] | None = (1.0, 99.0),
    eps: float = 1e-12,
) -> np.ndarray:
    """Robustly normalize a grayscale image to zero mean and unit variance."""

    working = np.asarray(image, dtype=np.float32)
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
    return np.asarray((working - mean) / std, dtype=np.float32)


def estimate_shift(
    reference: Any,
    current: Any,
    *,
    clip_percentiles: tuple[float, float] | None = (1.0, 99.0),
    use_window: bool = False,
    phase_l2_size: int = 7,
    phase_max_iters: int = 50,
    check_tiles: bool = False,
    use_ecc_refinement: bool = False,
    ecc_motion_model: str = "homography",
    ecc_reference_point_px: npt.ArrayLike | None = None,
    ecc_initial_shift_px: npt.ArrayLike | None = None,
    ecc_fallback_to_phase_shift: bool = True,
    ecc_use_window: bool = False,
) -> xr.Dataset:
    """Estimate subpixel translation between two grayscale or RGB images.

    By default, images are percentile-clipped, normalized, converted to grayscale
    when needed, and registered with OpenCV iterative phase correlation.
    Pass ``use_window=True`` to apply a Hanning taper before registration.
    Pass ``check_tiles=True`` to compare local tile shifts against the full-frame
    estimate.
    Pass ``use_ecc_refinement=True`` to refine the phase-correlation result with
    an OpenCV ECC affine or homography registration and return the displacement
    at ``ecc_reference_point_px``. If no point is supplied, the image center is
    used. ``ecc_initial_shift_px`` overrides the phase-correlation shift used to
    initialize ECC. Pass ``ecc_fallback_to_phase_shift=False`` when the supplied
    ECC seed is trusted and a failed refinement should invalidate the estimate.
    Pass ``ecc_use_window=True`` to apply a Hanning taper to ECC inputs.
    """

    reference_image = _as_registration_image("reference image", reference)
    current_image = _as_registration_image("current image", current)
    if reference_image.shape != current_image.shape:
        raise ValueError(
            "reference and current images must have identical shapes; "
            f"got {reference_image.shape!r} and {current_image.shape!r}"
        )
    phase_l2_size = _positive_int(phase_l2_size, "phase_l2_size")
    phase_max_iters = _positive_int(phase_max_iters, "phase_max_iters")

    reference_gray = _grayscale_registration_image(reference_image)
    current_gray = _grayscale_registration_image(current_image)

    dynamic_range = float(np.max(reference_gray) - np.min(reference_gray))
    standard_deviation = float(np.std(reference_gray))
    gy, gx = np.gradient(reference_gray)
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

    reference_norm = normalize_intensity(
        reference_gray,
        clip_percentiles=clip_percentiles,
    )
    current_norm = normalize_intensity(
        current_gray,
        clip_percentiles=clip_percentiles,
    )

    if dynamic_range <= 1e-12 or standard_deviation <= 1e-12:
        shift_px = np.array([np.nan, np.nan], dtype=np.float64)
        diagnostic_warnings.append(
            "registration skipped because reference image has little or no intensity contrast"
        )
    else:
        shift_px = _estimate_translation(
            reference_norm,
            current_norm,
            use_window=use_window,
            phase_l2_size=phase_l2_size,
            phase_max_iters=phase_max_iters,
        )

    if not np.isfinite(shift_px).all():
        shift_px = np.array([np.nan, np.nan], dtype=np.float64)
        diagnostic_warnings.append(
            "registration shift is not finite; shift estimate is unreliable"
        )

    explicit_ecc_initial_shift = None
    if use_ecc_refinement and ecc_initial_shift_px is not None:
        explicit_ecc_initial_shift = np.asarray(ecc_initial_shift_px, dtype=np.float64)
        if (
            explicit_ecc_initial_shift.shape != (2,)
            or not np.isfinite(explicit_ecc_initial_shift).all()
        ):
            raise ValueError("ecc_initial_shift_px must be a finite 2-vector")
    if use_ecc_refinement and (
        explicit_ecc_initial_shift is not None or np.isfinite(shift_px).all()
    ):
        ecc_initial_shift = (
            explicit_ecc_initial_shift
            if explicit_ecc_initial_shift is not None
            else shift_px
        )
        try:
            shift_px = _estimate_ecc_shift(
                reference_image,
                current_image,
                ecc_initial_shift,
                use_window=ecc_use_window,
                motion_model=ecc_motion_model,
                reference_point_px=ecc_reference_point_px,
            )
        except Exception as exc:
            diagnostic_warnings.append(f"ECC refinement failed: {exc}")
            if not ecc_fallback_to_phase_shift:
                shift_px = np.array([np.nan, np.nan], dtype=np.float64)

    if check_tiles and np.isfinite(shift_px).all():
        tile_warning = _tile_consistency(
            reference_norm,
            current_norm,
            shift_px,
            use_window=use_window,
            phase_l2_size=phase_l2_size,
            phase_max_iters=phase_max_iters,
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
    phase_l2_size: int,
    phase_max_iters: int,
) -> np.ndarray:
    if reference_norm.shape != current_norm.shape:
        raise ValueError("images must have identical shapes")
    if min(reference_norm.shape) < 3:
        raise ValueError("images must be at least 3x3 pixels")
    l2_size = _positive_int(phase_l2_size, "phase_l2_size")
    max_iters = _positive_int(phase_max_iters, "phase_max_iters")

    reference_work, current_work = _registration_work_images(
        reference_norm,
        current_norm,
        use_window=use_window,
    )

    shift_xy = cv2.phaseCorrelateIterative(
        np.ascontiguousarray(reference_work, dtype=np.float32),
        np.ascontiguousarray(current_work, dtype=np.float32),
        l2_size,
        max_iters,
    )
    shift_px = np.asarray(shift_xy, dtype=np.float64)
    if shift_px.shape != (2,):
        raise ValueError("phaseCorrelateIterative returned an unexpected shift shape")
    return shift_px


def _as_registration_image(name: str, image: Any) -> np.ndarray:
    image_array = np.asarray(image)
    if image_array.ndim not in (2, 3):
        raise ValueError(
            f"{name} must be 2D grayscale or 3D RGB, got {image_array.shape!r}"
        )
    if image_array.ndim == 3 and image_array.shape[2] != 3:
        raise ValueError(f"{name} color images must have exactly 3 channels")
    if image_array.size == 0:
        raise ValueError(f"{name} must not be empty")
    if (
        not np.issubdtype(image_array.dtype, np.number)
        or np.issubdtype(image_array.dtype, np.complexfloating)
    ):
        raise ValueError(f"{name} must have a real numeric dtype")
    if np.issubdtype(image_array.dtype, np.floating) and not np.isfinite(
        image_array
    ).all():
        raise ValueError(f"{name} must contain only finite values")
    return image_array


def _grayscale_registration_image(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return np.asarray(image, dtype=np.float32)
    return np.asarray(
        np.tensordot(
            np.asarray(image, dtype=np.float32),
            BASLER_RGB_TO_MONO_WEIGHTS,
            axes=([-1], [0]),
        ),
        dtype=np.float32,
    )


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive integer")
    try:
        numeric = operator.index(value)
    except TypeError as exc:
        raise ValueError(f"{name} must be a positive integer") from exc
    if numeric < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(numeric)


def _estimate_ecc_shift(
    reference_image: np.ndarray,
    current_image: np.ndarray,
    initial_shift_px: np.ndarray,
    *,
    use_window: bool,
    motion_model: str,
    reference_point_px: npt.ArrayLike | None,
) -> np.ndarray:
    reference_work, current_work = _registration_work_images(
        reference_image,
        current_image,
        use_window=use_window,
    )
    initial_shift = np.asarray(initial_shift_px, dtype=np.float64)
    if initial_shift.shape != (2,) or not np.isfinite(initial_shift).all():
        raise ValueError("initial ECC shift must be a finite 2-vector")

    motion_code, warp = _initial_ecc_warp(motion_model, initial_shift)
    criteria = (
        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
        50,
        1e-5,
    )
    _correlation, refined_warp = cv2.findTransformECC(
        np.ascontiguousarray(reference_work),
        np.ascontiguousarray(current_work),
        warp,
        motion_code,
        criteria,
        None,
        5,
    )
    refined = np.asarray(refined_warp, dtype=np.float64)
    if refined.shape != warp.shape or not np.isfinite(refined).all():
        raise ValueError(f"ECC returned a non-finite {motion_model} warp")

    point = _ecc_reference_point(reference_image.shape[:2], reference_point_px)
    if motion_code == cv2.MOTION_AFFINE:
        mapped = refined @ np.asarray([point[0], point[1], 1.0], dtype=np.float64)
        shift = mapped - point
        if not np.isfinite(shift).all():
            raise ValueError("ECC returned a non-finite point displacement")
        return np.asarray(shift, dtype=np.float64)

    mapped_homogeneous = refined @ np.asarray(
        [point[0], point[1], 1.0],
        dtype=np.float64,
    )
    denominator = float(mapped_homogeneous[2])
    if abs(denominator) <= 1e-12:
        raise ValueError("ECC homography maps reference point to infinity")
    mapped = mapped_homogeneous[:2] / denominator
    shift = mapped - point
    if not np.isfinite(shift).all():
        raise ValueError("ECC returned a non-finite point displacement")
    return np.asarray(shift, dtype=np.float64)


def _initial_ecc_warp(
    motion_model: str,
    initial_shift: np.ndarray,
) -> tuple[int, np.ndarray]:
    normalized_model = str(motion_model).strip().lower()
    if normalized_model == "affine":
        return (
            cv2.MOTION_AFFINE,
            np.asarray(
                [
                    [1.0, 0.0, initial_shift[0]],
                    [0.0, 1.0, initial_shift[1]],
                ],
                dtype=np.float32,
            ),
        )
    if normalized_model == "homography":
        return (
            cv2.MOTION_HOMOGRAPHY,
            np.asarray(
                [
                    [1.0, 0.0, initial_shift[0]],
                    [0.0, 1.0, initial_shift[1]],
                    [0.0, 0.0, 1.0],
                ],
                dtype=np.float32,
            ),
        )
    raise ValueError(f"unsupported ECC motion model: {motion_model!r}")


def _ecc_reference_point(
    shape: tuple[int, int],
    reference_point_px: npt.ArrayLike | None,
) -> np.ndarray:
    if reference_point_px is None:
        height, width = shape
        return np.asarray(
            [(width - 1.0) / 2.0, (height - 1.0) / 2.0],
            dtype=np.float64,
        )

    point = np.asarray(reference_point_px, dtype=np.float64)
    if point.shape != (2,) or not np.isfinite(point).all():
        raise ValueError("ECC reference point must be a finite 2-vector")
    return point


def _registration_work_images(
    reference_image: np.ndarray,
    current_image: np.ndarray,
    *,
    use_window: bool,
) -> tuple[np.ndarray, np.ndarray]:
    reference_work = reference_image
    current_work = current_image
    if use_window:
        reference_work = np.asarray(reference_work, dtype=np.float32)
        current_work = np.asarray(current_work, dtype=np.float32)
        window = np.outer(
            np.hanning(reference_work.shape[0]).astype(np.float32),
            np.hanning(reference_work.shape[1]).astype(np.float32),
        )
        if reference_work.ndim == 3:
            window = window[..., np.newaxis]
        reference_work = reference_work * window
        current_work = current_work * window
    return reference_work, current_work


def _tile_consistency(
    reference_norm: np.ndarray,
    current_norm: np.ndarray,
    full_shift_px: np.ndarray,
    *,
    use_window: bool,
    phase_l2_size: int,
    phase_max_iters: int,
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
            tile_shift = _estimate_translation(
                reference_tile,
                current_tile,
                use_window=use_window,
                phase_l2_size=phase_l2_size,
                phase_max_iters=phase_max_iters,
            )
            if np.isfinite(tile_shift).all():
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

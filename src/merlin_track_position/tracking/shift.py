"""Subpixel whole-frame shift estimation for grayscale or RGB image pairs."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np
import numpy.typing as npt
import xarray as xr

PIXEL_AXES = ("du_px", "dv_px")
BASLER_RGB_TO_MONO_WEIGHTS = np.asarray((0.25, 0.625, 0.125), dtype=np.float32)
ECC_NATIVE_DTYPES = (
    np.dtype(np.uint8),
    np.dtype(np.uint16),
    np.dtype(np.float32),
    np.dtype(np.float64),
)


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
    check_tiles: bool = False,
    use_ecc_refinement: bool = False,
    ecc_motion_model: str = "homography",
    ecc_reference_point_px: npt.ArrayLike | None = None,
    ecc_initial_shift_px: npt.ArrayLike | None = None,
    ecc_initial_warp: npt.ArrayLike | None = None,
    ecc_fallback_to_phase_shift: bool = True,
    ecc_use_window: bool = False,
    ecc_gauss_filter_size: int = 5,
) -> xr.Dataset:
    """Estimate subpixel translation between two grayscale or RGB images.

    By default, images are percentile-clipped, normalized, converted to grayscale
    when needed, and registered with OpenCV phase correlation.
    Pass ``use_window=True`` to apply a Hanning taper before registration.
    Pass ``check_tiles=True`` to compare local tile shifts against the full-frame
    estimate.
    Pass ``use_ecc_refinement=True`` to refine the phase-correlation result with
    an OpenCV ECC affine or homography registration and return the displacement
    at ``ecc_reference_point_px``. If no point is supplied, the image center is
    used. ``ecc_initial_shift_px`` overrides the phase-correlation translation
    used to initialize ECC. When ``ecc_initial_warp`` is supplied, the reference
    image is prewarped with that model before phase correlation estimates the
    residual translation; invalid pixels exposed by the prewarp are masked out
    of both phase-correlation inputs, and ECC starts from the combined
    model+translation warp. Pass ``ecc_fallback_to_phase_shift=False`` when the
    supplied ECC seed is trusted and a failed refinement should invalidate the
    estimate.
    Pass ``ecc_use_window=True`` to apply a Hanning taper to ECC inputs.
    ``ecc_gauss_filter_size`` controls OpenCV ECC's Gaussian prefilter; it must
    be a positive odd integer.
    """

    reference_image = _as_registration_image("reference image", reference)
    current_image = _as_registration_image("current image", current)
    if reference_image.shape != current_image.shape:
        raise ValueError(
            "reference and current images must have identical shapes; "
            f"got {reference_image.shape!r} and {current_image.shape!r}"
        )
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

    explicit_ecc_initial_shift = None
    explicit_ecc_initial_warp = None
    if use_ecc_refinement and ecc_initial_shift_px is not None:
        explicit_ecc_initial_shift = np.asarray(ecc_initial_shift_px, dtype=np.float64)
        if (
            explicit_ecc_initial_shift.shape != (2,)
            or not np.isfinite(explicit_ecc_initial_shift).all()
        ):
            raise ValueError("ecc_initial_shift_px must be a finite 2-vector")
    if use_ecc_refinement and ecc_initial_warp is not None:
        explicit_ecc_initial_warp = _validate_ecc_initial_warp(ecc_initial_warp)
    if use_ecc_refinement:
        ecc_gauss_filter_size = _validate_ecc_gauss_filter_size(
            ecc_gauss_filter_size
        )

    if dynamic_range <= 1e-12 or standard_deviation <= 1e-12:
        phase_shift_px = np.array([np.nan, np.nan], dtype=np.float64)
        diagnostic_warnings.append(
            "registration skipped because reference image has little or no intensity contrast"
        )
    else:
        phase_shift_px = _estimate_translation(
            reference_norm,
            current_norm,
            use_window=use_window,
        )

    shift_px = phase_shift_px.copy()
    ecc_phase_shift_px = phase_shift_px.copy()
    if not np.isfinite(phase_shift_px).all():
        phase_shift_px = np.array([np.nan, np.nan], dtype=np.float64)
        shift_px = np.array([np.nan, np.nan], dtype=np.float64)
        ecc_phase_shift_px = np.array([np.nan, np.nan], dtype=np.float64)
        diagnostic_warnings.append(
            "registration shift is not finite; shift estimate is unreliable"
        )

    if (
        use_ecc_refinement
        and explicit_ecc_initial_shift is None
        and explicit_ecc_initial_warp is not None
        and np.isfinite(phase_shift_px).all()
    ):
        try:
            model_reference_norm, model_phase_window = _prewarp_reference_for_phase_seed(
                reference_norm,
                explicit_ecc_initial_warp,
            )
            ecc_phase_shift_px = _estimate_translation(
                model_reference_norm,
                current_norm,
                use_window=use_window,
                phase_window=model_phase_window,
            )
        except Exception as exc:
            ecc_phase_shift_px = np.array([np.nan, np.nan], dtype=np.float64)
            diagnostic_warnings.append(
                f"model-prewarped ECC phase seed failed: {exc}"
            )
        else:
            if not np.isfinite(ecc_phase_shift_px).all():
                ecc_phase_shift_px = np.array([np.nan, np.nan], dtype=np.float64)
                diagnostic_warnings.append(
                    "model-prewarped ECC phase seed is not finite; shift estimate is unreliable"
                )

    if use_ecc_refinement and (
        explicit_ecc_initial_warp is not None
        or explicit_ecc_initial_shift is not None
        or np.isfinite(shift_px).all()
    ):
        ecc_initial_shift = (
            explicit_ecc_initial_shift
            if explicit_ecc_initial_shift is not None
            else ecc_phase_shift_px
        )
        try:
            shift_px = _estimate_ecc_shift(
                reference_image,
                current_image,
                ecc_initial_shift,
                use_window=ecc_use_window,
                motion_model=ecc_motion_model,
                reference_point_px=ecc_reference_point_px,
                initial_warp=explicit_ecc_initial_warp,
                gauss_filter_size=ecc_gauss_filter_size,
            )
        except Exception as exc:
            diagnostic_warnings.append(f"ECC refinement failed: {exc}")
            if not ecc_fallback_to_phase_shift:
                shift_px = np.array([np.nan, np.nan], dtype=np.float64)

    if check_tiles and np.isfinite(phase_shift_px).all():
        tile_warning = _tile_consistency(
            reference_norm,
            current_norm,
            phase_shift_px,
            use_window=use_window,
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
    phase_window: np.ndarray | None = None,
) -> np.ndarray:
    if reference_norm.shape != current_norm.shape:
        raise ValueError("images must have identical shapes")
    if min(reference_norm.shape) < 3:
        raise ValueError("images must be at least 3x3 pixels")

    reference_work, current_work = _registration_work_images(
        reference_norm,
        current_norm,
        use_window=use_window,
        phase_window=phase_window,
    )

    reference_phase = np.ascontiguousarray(reference_work, dtype=np.float32)
    current_phase = np.ascontiguousarray(current_work, dtype=np.float32)
    forward_shift_xy, _forward_response = cv2.phaseCorrelate(
        reference_phase,
        current_phase,
    )
    reverse_shift_xy, _reverse_response = cv2.phaseCorrelate(
        current_phase,
        reference_phase,
    )
    shift_px = 0.5 * (
        np.asarray(forward_shift_xy, dtype=np.float64)
        - np.asarray(reverse_shift_xy, dtype=np.float64)
    )
    if shift_px.shape != (2,):
        raise ValueError("phaseCorrelate returned an unexpected shift shape")
    return shift_px


def _prewarp_reference_for_phase_seed(
    reference_norm: np.ndarray,
    initial_warp: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    height, width = reference_norm.shape
    source = np.asarray(reference_norm, dtype=np.float32)
    source_valid = np.ones((height, width), dtype=np.float32)
    if initial_warp.shape == (2, 3):
        warp = np.asarray(initial_warp, dtype=np.float32)
        warped_reference = cv2.warpAffine(
            source,
            warp,
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0.0,
        )
        phase_window = cv2.warpAffine(
            source_valid,
            warp,
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0.0,
        )
    elif initial_warp.shape == (3, 3):
        warp = np.asarray(initial_warp, dtype=np.float32)
        warped_reference = cv2.warpPerspective(
            source,
            warp,
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0.0,
        )
        phase_window = cv2.warpPerspective(
            source_valid,
            warp,
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0.0,
        )
    else:
        raise ValueError("ecc_initial_warp must have shape (2, 3) or (3, 3)")
    return warped_reference, np.clip(phase_window, 0.0, 1.0)


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
    if not np.issubdtype(image_array.dtype, np.number) or np.issubdtype(
        image_array.dtype, np.complexfloating
    ):
        raise ValueError(f"{name} must have a real numeric dtype")
    if (
        np.issubdtype(image_array.dtype, np.floating)
        and not np.isfinite(image_array).all()
    ):
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


def _estimate_ecc_shift(
    reference_image: np.ndarray,
    current_image: np.ndarray,
    initial_shift_px: np.ndarray,
    *,
    use_window: bool,
    motion_model: str,
    reference_point_px: npt.ArrayLike | None,
    initial_warp: np.ndarray | None,
    gauss_filter_size: int,
) -> np.ndarray:
    reference_work, current_work = _registration_work_images(
        reference_image,
        current_image,
        use_window=use_window,
    )
    initial_shift = np.asarray(initial_shift_px, dtype=np.float64)
    if initial_shift.shape != (2,) or not np.isfinite(initial_shift).all():
        raise ValueError("initial ECC shift must be a finite 2-vector")

    motion_code, warp = _initial_ecc_warp(
        motion_model,
        initial_shift,
        initial_warp=initial_warp,
    )
    criteria = (
        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
        50,
        1e-5,
    )
    reference_ecc, current_ecc = _ecc_input_images(reference_work, current_work)
    _correlation, refined_warp = cv2.findTransformECC(
        reference_ecc,
        current_ecc,
        warp,
        motion_code,
        criteria,
        None,
        int(gauss_filter_size),
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
    *,
    initial_warp: np.ndarray | None = None,
) -> tuple[int, np.ndarray]:
    normalized_model = str(motion_model).strip().lower()
    if normalized_model == "affine":
        if initial_warp is not None:
            if initial_warp.shape != (2, 3):
                raise ValueError(
                    "ecc_initial_warp must have shape (2, 3) for affine ECC"
                )
            warp = np.asarray(initial_warp, dtype=np.float64).copy()
            warp[:, 2] += initial_shift
            return cv2.MOTION_AFFINE, np.asarray(warp, dtype=np.float32)
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
        if initial_warp is not None:
            if initial_warp.shape == (2, 3):
                homography = np.eye(3, dtype=np.float64)
                homography[:2, :] = initial_warp
            elif initial_warp.shape == (3, 3):
                homography = np.asarray(initial_warp, dtype=np.float64)
            else:
                raise ValueError(
                    "ecc_initial_warp must have shape (2, 3) or (3, 3) "
                    "for homography ECC"
                )
            translation = np.eye(3, dtype=np.float64)
            translation[:2, 2] = initial_shift
            homography = translation @ homography
            return cv2.MOTION_HOMOGRAPHY, np.asarray(homography, dtype=np.float32)
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


def _validate_ecc_initial_warp(warp: npt.ArrayLike) -> np.ndarray:
    warp_array = np.asarray(warp, dtype=np.float64)
    if warp_array.shape not in ((2, 3), (3, 3)):
        raise ValueError("ecc_initial_warp must have shape (2, 3) or (3, 3)")
    if not np.isfinite(warp_array).all():
        raise ValueError("ecc_initial_warp must contain only finite values")
    return warp_array


def _validate_ecc_gauss_filter_size(value: int) -> int:
    try:
        size = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("ecc_gauss_filter_size must be a positive odd integer") from exc
    if size < 1 or size % 2 != 1:
        raise ValueError("ecc_gauss_filter_size must be a positive odd integer")
    return size


def _ecc_input_images(
    reference_image: np.ndarray,
    current_image: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    reference_array = np.asarray(reference_image)
    current_array = np.asarray(current_image)
    if (
        reference_array.dtype == current_array.dtype
        and reference_array.dtype in ECC_NATIVE_DTYPES
    ):
        return (
            np.ascontiguousarray(reference_array),
            np.ascontiguousarray(current_array),
        )
    return (
        np.ascontiguousarray(reference_array, dtype=np.float32),
        np.ascontiguousarray(current_array, dtype=np.float32),
    )


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
    phase_window: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    reference_work = reference_image
    current_work = current_image
    window: np.ndarray | None = None
    if phase_window is not None:
        window = _validate_phase_window(phase_window, reference_work.shape[:2])
    if use_window:
        reference_work = np.asarray(reference_work, dtype=np.float32)
        current_work = np.asarray(current_work, dtype=np.float32)
        hanning_window = np.outer(
            np.hanning(reference_work.shape[0]).astype(np.float32),
            np.hanning(reference_work.shape[1]).astype(np.float32),
        )
        window = hanning_window if window is None else window * hanning_window
    if window is not None:
        reference_work = np.asarray(reference_work, dtype=np.float32)
        current_work = np.asarray(current_work, dtype=np.float32)
        work_window = window
        if reference_work.ndim == 3:
            work_window = work_window[..., np.newaxis]
        reference_work = reference_work * work_window
        current_work = current_work * work_window
    return reference_work, current_work


def _validate_phase_window(window: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    phase_window = np.asarray(window, dtype=np.float32)
    if phase_window.shape != shape:
        raise ValueError("phase window must match image shape")
    if not np.isfinite(phase_window).all():
        raise ValueError("phase window must contain only finite values")
    phase_window = np.clip(phase_window, 0.0, 1.0)
    if float(np.sum(phase_window > 1.0e-3)) < 9.0:
        raise ValueError("phase window has too few valid pixels")
    return phase_window


def _tile_consistency(
    reference_norm: np.ndarray,
    current_norm: np.ndarray,
    full_shift_px: np.ndarray,
    *,
    use_window: bool,
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

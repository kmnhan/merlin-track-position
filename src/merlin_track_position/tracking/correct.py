from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np
import xarray as xr

from merlin_track_position.instruments.cameras import (
    CameraPairPlugin,
    RoiGeometry,
    capture_image_stack,
    crop_image_to_roi,
    default_camera_pair,
    normalize_capture_count,
)
from merlin_track_position.instruments.motors import get_positions, move_motors_and_wait
from merlin_track_position.tracking.calibration_core import get_correction

STAGE_MOTOR_ALIASES = ("x", "y", "z")
ROI_ATTR_KEYS: dict[str, tuple[str, str, str, str]] = {
    camera: (
        f"roi_{camera}_x",
        f"roi_{camera}_y",
        f"roi_{camera}_width",
        f"roi_{camera}_height",
    )
    for camera in ("cam0", "cam1")
}


def do_correction(
    calibration: xr.Dataset,
    camera_pair: CameraPairPlugin | None = None,
    *,
    move_tolerance_um: float | Iterable[float] | None = None,
    max_retries: int = 4,
    capture_count: int = 5,
    **shift_kwargs: Any,
) -> xr.Dataset:
    """Estimate and apply the x/y/z motor correction for current camera images.

    The first saved calibration sample is used as the reference image pair. The current
    image pair is captured with ``camera_pair``. If the calibration carries ROI
    metadata and the captured images are still full-frame, the current images are
    cropped to match the calibration references before comparison. The resulting
    ``correction_um`` vector is applied to the ``x``, ``y``, and ``z`` motors.

    Parameters
    ----------
    calibration
        Calibration dataset produced by `fit_calibration_from_images`.
    camera_pair
        Camera plugin pair used to capture cam0 and cam1 images. Raw full-frame capture
        is supported when the calibration dataset contains ROI metadata. Already-cropped
        images are left unchanged when their shape matches the saved calibration
        references.
    move_tolerance_um
        Optional motor move tolerance in microns. A scalar is applied to all three stage
        axes; an iterable supplies per-axis tolerances.
    max_retries
        Maximum number of motor move retries to pass to :func:`move_motors_and_wait`.
    capture_count
        Number of current image pairs captured before estimating the correction.
        Default is 5.
    **shift_kwargs
        Additional keyword arguments forwarded to :func:`get_correction` and ultimately
        to the image-shift estimator.

    Returns
    -------
    xarray.Dataset
        The correction dataset returned by :func:`get_correction`, with added attributes
        recording pre-move, requested, and final x/y/z motor positions in millimeters,
        plus ``correction_applied=True``.

    Raises
    ------
    ValueError
        If ROI metadata is incomplete or invalid, an ROI crop does not match the
        reference image shape, or ``correction_um`` is not a finite three-element x/y/z
        vector.
    """

    capture_count = normalize_capture_count(capture_count)
    if camera_pair is None:
        camera_pair = default_camera_pair()
    reference_cam0 = calibration["image_cam0"].isel(sample=0).values
    reference_cam1 = calibration["image_cam1"].isel(sample=0).values
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

    correction = get_correction(
        calibration,
        reference_cam0,
        current_cam0,
        reference_cam1,
        current_cam1,
        **shift_kwargs,
    )

    correction_um = np.asarray(correction["correction_um"].values, dtype=np.float64)
    if correction_um.shape != (len(STAGE_MOTOR_ALIASES),):
        raise ValueError(
            "correction_um must have one value for each x/y/z motor axis; "
            f"got shape {correction_um.shape!r}"
        )
    if not np.isfinite(correction_um).all():
        raise ValueError("correction_um must contain only finite values before moving")

    pre_move_position_mm = np.asarray(
        get_positions(STAGE_MOTOR_ALIASES),
        dtype=np.float64,
    )
    requested_position_mm = pre_move_position_mm + correction_um * 1e-3
    move_tolerance_mm = _um_to_mm(move_tolerance_um)
    final_position_mm = np.asarray(
        move_motors_and_wait(
            STAGE_MOTOR_ALIASES,
            tuple(float(value) for value in requested_position_mm),
            tolerance=move_tolerance_mm,
            max_retries=max_retries,
        ),
        dtype=np.float64,
    )

    return correction.assign_attrs(
        {
            "pre_move_x_mm": float(pre_move_position_mm[0]),
            "pre_move_y_mm": float(pre_move_position_mm[1]),
            "pre_move_z_mm": float(pre_move_position_mm[2]),
            "requested_x_mm": float(requested_position_mm[0]),
            "requested_y_mm": float(requested_position_mm[1]),
            "requested_z_mm": float(requested_position_mm[2]),
            "final_x_mm": float(final_position_mm[0]),
            "final_y_mm": float(final_position_mm[1]),
            "final_z_mm": float(final_position_mm[2]),
            "correction_applied": True,
        }
    )


def _um_to_mm(
    value: float | Iterable[float] | None,
) -> float | tuple[float, ...] | None:
    if value is None:
        return None
    if np.isscalar(value):
        return float(value) * 1e-3
    return tuple(float(item) * 1e-3 for item in value)


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

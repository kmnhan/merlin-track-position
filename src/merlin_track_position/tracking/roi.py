from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import numpy as np
import numpy.typing as npt

from merlin_track_position import constants
from merlin_track_position.instruments.cameras import RoiGeometry, crop_image_to_roi
from merlin_track_position.tracking.calibration_core import CAMERAS

ROI_ATTR_KEYS: dict[str, tuple[str, str, str, str]] = {
    camera: (
        f"roi_{camera}_x",
        f"roi_{camera}_y",
        f"roi_{camera}_width",
        f"roi_{camera}_height",
    )
    for camera in CAMERAS
}
BEAM_TARGET_ATTR_KEYS: dict[str, tuple[str, str]] = {
    camera: (f"beam_target_{camera}_u", f"beam_target_{camera}_v") for camera in CAMERAS
}
CAMERA_IMAGE_SHAPES: dict[str, tuple[int, int]] = {
    "cam0": (constants.IMAGE_HEIGHT_CAM0, constants.IMAGE_WIDTH_CAM0),
    "cam1": (constants.IMAGE_HEIGHT_CAM1, constants.IMAGE_WIDTH_CAM1),
}


def beam_target_attrs_from_points(
    points: Mapping[str, Any],
) -> dict[str, float]:
    attrs: dict[str, float] = {}
    for camera in CAMERAS:
        point = np.asarray(points[camera], dtype=np.float64)
        if point.shape != (2,) or not np.isfinite(point).all():
            raise ValueError(f"beam target for {camera} must be a finite 2-vector")
        for key, value in zip(BEAM_TARGET_ATTR_KEYS[camera], point, strict=True):
            attrs[key] = float(value)
    return attrs


def roi_geometry_from_attrs(
    attrs: Mapping[str, Any],
    camera: str,
) -> RoiGeometry | None:
    keys = ROI_ATTR_KEYS[camera]
    present = tuple(key in attrs for key in keys)
    if not any(present):
        return None
    if not all(present):
        missing = ", ".join(key for key, exists in zip(keys, present) if not exists)
        raise ValueError(f"incomplete ROI metadata for {camera}; missing {missing}")

    try:
        roi_geometry = tuple(float(attrs[key]) for key in keys)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"ROI metadata for {camera} must be numeric") from exc

    if not np.isfinite(roi_geometry).all():
        raise ValueError(f"ROI metadata for {camera} must be finite")
    return roi_geometry


def beam_target_point_from_attrs_or_default(
    attrs: Mapping[str, Any],
    camera: str,
) -> npt.NDArray[np.float64]:
    keys = BEAM_TARGET_ATTR_KEYS[camera]
    present = tuple(key in attrs for key in keys)
    if all(present):
        try:
            point = np.asarray([float(attrs[key]) for key in keys], dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"beam target metadata for {camera} must be numeric"
            ) from exc
        if not np.isfinite(point).all():
            raise ValueError(f"beam target metadata for {camera} must be finite")
        return point
    if any(present):
        missing = ", ".join(key for key, exists in zip(keys, present) if not exists)
        raise ValueError(
            f"incomplete beam target metadata for {camera}; missing {missing}"
        )
    return default_beam_target_point(attrs, camera)


def default_beam_target_point(
    attrs: Mapping[str, Any],
    camera: str,
) -> npt.NDArray[np.float64]:
    roi_geometry = roi_geometry_from_attrs(attrs, camera)
    if roi_geometry is None:
        image_height, image_width = camera_image_shape_from_attrs(attrs, camera)
        return np.asarray(
            [(image_width - 1.0) / 2.0, (image_height - 1.0) / 2.0],
            dtype=np.float64,
        )

    x0, y0, x1, y1 = roi_crop_bounds(
        roi_geometry,
        camera_image_shape_from_attrs(attrs, camera),
    )
    return np.asarray(
        [x0 + (x1 - x0 - 1.0) / 2.0, y0 + (y1 - y0 - 1.0) / 2.0],
        dtype=np.float64,
    )


def roi_local_point_from_full_frame(
    attrs: Mapping[str, Any],
    camera: str,
    point: npt.ArrayLike,
) -> npt.NDArray[np.float64]:
    full_point = np.asarray(point, dtype=np.float64)
    if full_point.shape != (2,) or not np.isfinite(full_point).all():
        raise ValueError(f"beam target for {camera} must be a finite 2-vector")

    roi_geometry = roi_geometry_from_attrs(attrs, camera)
    if roi_geometry is None:
        image_height, image_width = camera_image_shape_from_attrs(attrs, camera)
        return np.asarray(
            [
                min(max(full_point[0], 0.0), image_width - 1.0),
                min(max(full_point[1], 0.0), image_height - 1.0),
            ],
            dtype=np.float64,
        )

    x0, y0, x1, y1 = roi_crop_bounds(
        roi_geometry,
        camera_image_shape_from_attrs(attrs, camera),
    )
    clamped = np.asarray(
        [
            min(max(full_point[0], float(x0)), float(x1 - 1)),
            min(max(full_point[1], float(y0)), float(y1 - 1)),
        ],
        dtype=np.float64,
    )
    return clamped - np.asarray([float(x0), float(y0)], dtype=np.float64)


def crop_stack_to_roi(
    stack: npt.ArrayLike,
    roi_geometry: RoiGeometry,
) -> npt.NDArray:
    return np.stack(
        [crop_image_to_roi(image, roi_geometry) for image in np.asarray(stack)],
        axis=0,
    )


def roi_crop_bounds(
    roi_geometry: RoiGeometry,
    image_shape: tuple[int, int],
) -> tuple[int, int, int, int]:
    x, y, width, height = (float(value) for value in roi_geometry)
    image_height, image_width = image_shape

    width = min(max(width, 1.0), float(image_width))
    height = min(max(height, 1.0), float(image_height))
    x = min(max(x, 0.0), float(image_width) - width)
    y = min(max(y, 0.0), float(image_height) - height)

    x0 = min(max(int(math.floor(x)), 0), image_width - 1)
    y0 = min(max(int(math.floor(y)), 0), image_height - 1)
    x1 = min(max(int(math.ceil(x + width)), x0 + 1), image_width)
    y1 = min(max(int(math.ceil(y + height)), y0 + 1), image_height)
    return x0, y0, x1, y1


def matching_reference_and_stack(
    attrs: Mapping[str, Any],
    camera: str,
    reference: npt.ArrayLike,
    current_stack: npt.ArrayLike,
) -> tuple[npt.NDArray, npt.NDArray]:
    reference_array = np.asarray(reference)
    stack_array = np.asarray(current_stack)
    if stack_array.ndim < 3:
        raise ValueError(f"{camera} image stack must be at least 3D")

    roi_geometry = roi_geometry_from_attrs(attrs, camera)
    if roi_geometry is None:
        return reference_array, stack_array

    expected_roi_shape = _roi_crop_shape(
        roi_geometry,
        camera_image_shape_from_attrs(attrs, camera),
    )
    if reference_array.shape[:2] == expected_roi_shape:
        matching_reference = reference_array
    else:
        matching_reference = crop_image_to_roi(reference_array, roi_geometry)

    if stack_array.shape[1:3] == matching_reference.shape[:2]:
        return matching_reference, stack_array

    cropped_stack = crop_stack_to_roi(stack_array, roi_geometry)
    if cropped_stack.shape[1:3] != matching_reference.shape[:2]:
        raise ValueError(
            f"cropped {camera} image shape {cropped_stack.shape[1:3]!r} "
            f"does not match calibration reference shape "
            f"{matching_reference.shape[:2]!r}"
        )
    return matching_reference, cropped_stack


def _roi_crop_shape(
    roi_geometry: RoiGeometry,
    image_shape: tuple[int, int],
) -> tuple[int, int]:
    x0, y0, x1, y1 = roi_crop_bounds(roi_geometry, image_shape)
    return y1 - y0, x1 - x0


def camera_image_shape_from_attrs(
    attrs: Mapping[str, Any],
    camera: str,
) -> tuple[int, int]:
    try:
        height = int(attrs[f"camera_{camera}_height"])
        width = int(attrs[f"camera_{camera}_width"])
    except (KeyError, TypeError, ValueError):
        return CAMERA_IMAGE_SHAPES[camera]
    if height < 1 or width < 1:
        return CAMERA_IMAGE_SHAPES[camera]
    return height, width

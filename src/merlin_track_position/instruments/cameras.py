"""Helpers for acquiring one logical sample from both tracking cameras."""

from __future__ import annotations

import math
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import numpy.typing as npt

from merlin_track_position.instruments.basler import get_basler_image
from merlin_track_position.instruments.framegrab import get_framegrabber_image

RoiGeometry = tuple[float, float, float, float]
CameraCapture = Callable[[], npt.NDArray]


def capture_camera_pair(
    cam0_capture: CameraCapture | None = None,
    cam1_capture: CameraCapture | None = None,
) -> tuple[npt.NDArray, npt.NDArray]:
    """Capture the current cam0 and cam1 images."""
    if cam0_capture is None:
        cam0_capture = get_framegrabber_image
    if cam1_capture is None:
        cam1_capture = get_basler_image

    with ThreadPoolExecutor(
        max_workers=2,
        thread_name_prefix="camera-capture",
    ) as pool:
        image_cam0 = pool.submit(cam0_capture)
        image_cam1 = pool.submit(cam1_capture)
        return image_cam0.result(), image_cam1.result()


def crop_image_to_roi(
    image: npt.NDArray,
    roi_geometry: RoiGeometry,
) -> npt.NDArray:
    """Return a copy of ``image`` cropped to a full-frame ROI geometry.

    The ROI geometry is ``(x, y, width, height)`` in source image pixels. Fractional
    boundaries are expanded outward so the crop contains the requested region.
    """
    array = np.asarray(image)
    if array.ndim < 2:
        raise ValueError("image must be at least 2-dimensional")

    x, y, width, height = (float(value) for value in roi_geometry)
    if not all(math.isfinite(value) for value in (x, y, width, height)):
        raise ValueError("roi geometry values must be finite")

    image_height, image_width = array.shape[:2]
    if image_width < 1 or image_height < 1:
        raise ValueError("image must have nonzero width and height")

    width = min(max(width, 1.0), float(image_width))
    height = min(max(height, 1.0), float(image_height))
    x = min(max(x, 0.0), float(image_width) - width)
    y = min(max(y, 0.0), float(image_height) - height)

    x0 = min(max(int(math.floor(x)), 0), image_width - 1)
    y0 = min(max(int(math.floor(y)), 0), image_height - 1)
    x1 = min(max(int(math.ceil(x + width)), x0 + 1), image_width)
    y1 = min(max(int(math.ceil(y + height)), y0 + 1), image_height)

    return array[y0:y1, x0:x1, ...].copy()


def make_cropped_camera_pair_capture(
    roi_cam0: RoiGeometry,
    roi_cam1: RoiGeometry,
    base_capture: Callable[[], tuple[npt.NDArray, npt.NDArray]] = capture_camera_pair,
) -> Callable[[], tuple[npt.NDArray, npt.NDArray]]:
    """Build a camera-pair acquisition function that crops both images after capture."""

    def _capture() -> tuple[npt.NDArray, npt.NDArray]:
        image_cam0, image_cam1 = base_capture()
        return (
            crop_image_to_roi(image_cam0, roi_cam0),
            crop_image_to_roi(image_cam1, roi_cam1),
        )

    return _capture

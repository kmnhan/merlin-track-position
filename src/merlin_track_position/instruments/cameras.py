"""Helpers for acquiring one logical sample from both tracking cameras."""

from __future__ import annotations

import logging
import math
import time
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from numbers import Integral
from typing import Any

import numpy as np
import numpy.typing as npt

from merlin_track_position.instruments.basler import (
    get_basler_image,
    get_basler_image_stack,
)
from merlin_track_position.instruments.framegrab import (
    get_framegrabber_image,
    get_framegrabber_image_stack,
)

RoiGeometry = tuple[float, float, float, float]
_NO_FRAME = object()
logger = logging.getLogger("merlin_track_position.instruments.cameras")


class _ImageContentKey:
    def __init__(self, image: npt.NDArray):
        self.image = np.asarray(image).copy()

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, _ImageContentKey):
            return NotImplemented
        return self.image.shape == other.image.shape and np.array_equal(
            self.image, other.image
        )


class CameraPlugin(ABC):
    """Object interface for one image source used by calibration workflows."""

    def __init__(
        self,
        name: str,
        *,
        fresh_frame_timeout_s: float = 5.0,
        fresh_frame_poll_interval_s: float = 0.05,
    ):
        self.name = str(name)
        self.fresh_frame_timeout_s = _validate_nonnegative_float(
            "fresh_frame_timeout_s",
            fresh_frame_timeout_s,
        )
        self.fresh_frame_poll_interval_s = _validate_nonnegative_float(
            "fresh_frame_poll_interval_s",
            fresh_frame_poll_interval_s,
        )
        self._last_frame_key: Any = _NO_FRAME
        self._last_image: npt.NDArray | None = None
        self._capture_serial = 0

    def capture(self) -> npt.NDArray:
        """Return the next frame this camera plugin considers fresh."""
        deadline = time.monotonic() + self.fresh_frame_timeout_s
        while True:
            image = np.asarray(self._capture_once())
            frame_key = self._frame_key(image)
            if self._last_frame_key is _NO_FRAME or frame_key != self._last_frame_key:
                self._last_frame_key = frame_key
                self._last_image = image.copy()
                return image
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"Timed out waiting for a fresh image from {self.name}; "
                    "received repeated frames"
                )
            time.sleep(self.fresh_frame_poll_interval_s)

    def capture_stack(self, frame_count: int) -> tuple[npt.NDArray, npt.NDArray]:
        """Return processing and display stacks for consecutive captures."""
        frame_count = normalize_capture_count(frame_count)
        images = []
        display_images = []
        for _frame_index in range(frame_count):
            images.append(np.asarray(self.capture()))
            display_images.append(np.asarray(self.display_image()))
        return np.stack(images, axis=0), np.stack(display_images, axis=0)

    @abstractmethod
    def _capture_once(self) -> npt.NDArray:
        """Return one image from the underlying camera source."""

    def _frame_key(self, image: npt.NDArray) -> Any:
        """Return a key that changes when this camera has a fresh frame."""
        del image
        self._capture_serial += 1
        return self._capture_serial

    def cropped(self, roi_geometry: RoiGeometry) -> "CameraPlugin":
        """Return a camera plugin that crops this camera's captured images."""
        return CroppedCameraPlugin(self, roi_geometry)

    def display_image(self) -> npt.NDArray:
        """Return the image that should be displayed for the last capture."""
        if self._last_image is None:
            raise RuntimeError(f"{self.name} has not captured an image yet")
        return self._last_image.copy()


class FramegrabberCameraPlugin(CameraPlugin):
    """Camera plugin backed by the framegrabber ZMQ image source."""

    def __init__(
        self,
        name: str = "cam0",
        *,
        fresh_frame_timeout_s: float = 5.0,
        fresh_frame_poll_interval_s: float = 0.05,
    ):
        super().__init__(
            name,
            fresh_frame_timeout_s=fresh_frame_timeout_s,
            fresh_frame_poll_interval_s=fresh_frame_poll_interval_s,
        )

    def _capture_once(self) -> npt.NDArray:
        return get_framegrabber_image()

    def capture_stack(self, frame_count: int) -> tuple[npt.NDArray, npt.NDArray]:
        frame_count = normalize_capture_count(frame_count)
        timeout_ms = max(1, int(self.fresh_frame_timeout_s * 1000.0 * frame_count))
        stack = np.asarray(
            get_framegrabber_image_stack(frame_count, timeout_ms=timeout_ms)
        )
        self._capture_serial += frame_count
        self._last_frame_key = self._capture_serial
        self._last_image = stack[-1].copy()
        return stack, stack.copy()


class BaslerCameraPlugin(CameraPlugin):
    """Camera plugin backed by the Basler SDK image source."""

    def __init__(
        self,
        name: str = "cam1",
        *,
        fresh_frame_timeout_s: float = 5.0,
        fresh_frame_poll_interval_s: float = 0.05,
    ):
        super().__init__(
            name,
            fresh_frame_timeout_s=fresh_frame_timeout_s,
            fresh_frame_poll_interval_s=fresh_frame_poll_interval_s,
        )

    def _capture_once(self) -> npt.NDArray:
        return get_basler_image()

    def capture_stack(self, frame_count: int) -> tuple[npt.NDArray, npt.NDArray]:
        frame_count = normalize_capture_count(frame_count)
        stack = np.asarray(get_basler_image_stack(frame_count))
        self._capture_serial += frame_count
        self._last_frame_key = self._capture_serial
        self._last_image = stack[-1].copy()
        return stack, stack.copy()


class CallableCameraPlugin(CameraPlugin):
    """Adapter for camera implementations supplied by application code or tests."""

    def __init__(
        self,
        name: str,
        capture_image: Callable[[], npt.NDArray],
        *,
        fresh_frame_timeout_s: float = 5.0,
        fresh_frame_poll_interval_s: float = 0.05,
        use_image_content_key: bool = False,
    ):
        super().__init__(
            name,
            fresh_frame_timeout_s=fresh_frame_timeout_s,
            fresh_frame_poll_interval_s=fresh_frame_poll_interval_s,
        )
        self._capture_image = capture_image
        self._use_image_content_key = bool(use_image_content_key)

    def _capture_once(self) -> npt.NDArray:
        return self._capture_image()

    def _frame_key(self, image: npt.NDArray) -> Any:
        if self._use_image_content_key:
            return _ImageContentKey(image)
        return super()._frame_key(image)


class CroppedCameraPlugin(CameraPlugin):
    """Camera plugin wrapper that crops images from another camera plugin."""

    def __init__(
        self,
        camera: CameraPlugin,
        roi_geometry: RoiGeometry,
    ):
        super().__init__(
            camera.name,
            fresh_frame_timeout_s=camera.fresh_frame_timeout_s,
            fresh_frame_poll_interval_s=camera.fresh_frame_poll_interval_s,
        )
        self._camera = camera
        self._roi_geometry = roi_geometry

    def _capture_once(self) -> npt.NDArray:
        return crop_image_to_roi(self._camera.capture(), self._roi_geometry)

    def capture_stack(self, frame_count: int) -> tuple[npt.NDArray, npt.NDArray]:
        image_stack, display_stack = self._camera.capture_stack(frame_count)
        cropped_stack = np.stack(
            [crop_image_to_roi(image, self._roi_geometry) for image in image_stack],
            axis=0,
        )
        self._last_image = cropped_stack[-1].copy()
        return cropped_stack, display_stack

    def display_image(self) -> npt.NDArray:
        return self._camera.display_image()


class CameraPairPlugin:
    """The two camera plugins used together by calibration and correction."""

    def __init__(
        self,
        cam0: CameraPlugin,
        cam1: CameraPlugin,
    ):
        self.cam0 = cam0
        self.cam1 = cam1

    def as_tuple(self) -> tuple[CameraPlugin, CameraPlugin]:
        return self.cam0, self.cam1

    def capture_pair(self) -> tuple[npt.NDArray, npt.NDArray]:
        return self.cam0.capture(), self.cam1.capture()

    def cropped(
        self,
        roi_cam0: RoiGeometry,
        roi_cam1: RoiGeometry,
    ) -> "CameraPairPlugin":
        return CameraPairPlugin(
            self.cam0.cropped(roi_cam0),
            self.cam1.cropped(roi_cam1),
        )


def default_camera_pair() -> CameraPairPlugin:
    """Return the default two-camera hardware plugin pair."""
    return CameraPairPlugin(FramegrabberCameraPlugin(), BaslerCameraPlugin())


def normalize_capture_count(capture_count: int) -> int:
    """Return a validated integer number of captures for one motor position."""
    if isinstance(capture_count, bool) or not isinstance(capture_count, Integral):
        raise ValueError("capture_count must be an integer >= 1")
    value = int(capture_count)
    if value < 1:
        raise ValueError("capture_count must be an integer >= 1")
    return value


def capture_image_stack(
    cameras: CameraPairPlugin | Sequence[CameraPlugin],
    capture_count: int,
) -> tuple[npt.NDArray, ...]:
    """Capture image stacks from camera plugins."""
    image_stacks, _ = capture_image_and_display_stacks(cameras, capture_count)
    return image_stacks


def capture_image_and_display_stacks(
    cameras: CameraPairPlugin | Sequence[CameraPlugin],
    capture_count: int,
) -> tuple[tuple[npt.NDArray, ...], tuple[npt.NDArray, ...]]:
    """Capture processing-image stacks and their corresponding display stacks."""

    capture_count = normalize_capture_count(capture_count)
    camera_plugins = _camera_plugins_tuple(cameras)
    if not camera_plugins:
        raise ValueError("at least one camera plugin is required")

    def capture_camera_stack(
        camera: CameraPlugin,
    ) -> tuple[npt.NDArray, npt.NDArray]:
        logger.info(
            "Capturing camera stack: camera=%s, frame_count=%d",
            camera.name,
            capture_count,
        )
        image_stack, display_stack = camera.capture_stack(capture_count)
        image_stack = np.asarray(image_stack)
        display_stack = np.asarray(display_stack)
        logger.info(
            "Captured camera stack: camera=%s, shape=%s",
            camera.name,
            image_stack.shape,
        )
        return image_stack, display_stack

    with ThreadPoolExecutor(max_workers=len(camera_plugins)) as executor:
        captured_stacks = tuple(
            executor.map(capture_camera_stack, camera_plugins),
        )

    return (
        tuple(image_stack for image_stack, _display_stack in captured_stacks),
        tuple(display_stack for _image_stack, display_stack in captured_stacks),
    )


def _camera_plugins_tuple(
    cameras: CameraPairPlugin | Sequence[CameraPlugin],
) -> tuple[CameraPlugin, ...]:
    if isinstance(cameras, CameraPairPlugin):
        return cameras.as_tuple()
    return tuple(cameras)


def _validate_nonnegative_float(name: str, value: float) -> float:
    numeric = float(value)
    if not math.isfinite(numeric) or numeric < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return numeric


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

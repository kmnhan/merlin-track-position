"""Development-mode simulator for the sample manipulator and framegrabber."""

from __future__ import annotations

import functools
import logging
import math
import threading
import time
from collections.abc import Iterable
from dataclasses import dataclass
from importlib import resources

import numpy as np
import numpy.typing as npt
from scipy import ndimage

from merlin_track_position import constants

logger = logging.getLogger("merlin_track_position.instruments.simulated_hardware")

SYNTHETIC_CALIBRATION_FILE = "synthetic_framegrabber_calibration.npz"
STATIC_TEMPERATURE_K = (30.0, 30.0, 30.0, 30.0)
DEFAULT_POSITIONS = {
    "x": 0.0,
    "y": 0.0,
    "z": 0.0,
    "p": 0.0,
    "t": 0.0,
    "cam": 5.0,
    "TA": STATIC_TEMPERATURE_K[0],
    "TB": STATIC_TEMPERATURE_K[1],
    "TC": STATIC_TEMPERATURE_K[2],
    "TD": STATIC_TEMPERATURE_K[3],
}


@dataclass(frozen=True)
class SyntheticCalibration:
    reference_image: npt.NDArray[np.float64]
    reference_image_cam1: npt.NDArray[np.float64]
    command_um_to_pixel_3d: npt.NDArray[np.float64]


CAM0_COMMAND_UM_TO_PIXEL_3D = np.array(
    [
        [0.27, -0.14, 0.07],
        [0.09, 0.31, -0.12],
    ],
    dtype=np.float64,
)
CAM1_COMMAND_UM_TO_PIXEL_3D = np.array(
    [
        [-0.21, 0.18, 0.33],
        [0.24, 0.05, 0.16],
    ],
    dtype=np.float64,
)


def _readonly_float64(array: npt.ArrayLike) -> npt.NDArray[np.float64]:
    result = np.asarray(array, dtype=np.float64).copy()
    result.flags.writeable = False
    return result


def _default_command_um_to_pixel_3d() -> npt.NDArray[np.float64]:
    return np.stack(
        [CAM0_COMMAND_UM_TO_PIXEL_3D, CAM1_COMMAND_UM_TO_PIXEL_3D],
        axis=0,
    )


@functools.cache
def load_synthetic_calibration() -> SyntheticCalibration:
    """Load the packaged development-mode reference image and calibration."""
    data_path = (
        resources.files("merlin_track_position.instruments")
        / "data"
        / SYNTHETIC_CALIBRATION_FILE
    )
    with data_path.open("rb") as file:
        with np.load(file) as archive:
            archive_keys = set(archive.files)
            if "reference_image" in archive_keys:
                reference_image = archive["reference_image"]
            else:
                reference_image = archive["reference_cam0"]

            if "reference_cam1" in archive_keys:
                reference_image_cam1 = archive["reference_cam1"]
            else:
                reference_image_cam1 = _generated_synthetic_cam1_reference()

            if "command_um_to_pixel_3d" in archive_keys:
                command_um_to_pixel_3d = archive["command_um_to_pixel_3d"]
            elif "px_per_readback_mm" in archive_keys:
                command_um_to_pixel_3d = archive["px_per_readback_mm"] / 1000.0
            elif "px_per_cmd_mm" in archive_keys:
                command_um_to_pixel_3d = archive["px_per_cmd_mm"] / 1000.0
            else:
                command_um_to_pixel_3d = _default_command_um_to_pixel_3d()

            return SyntheticCalibration(
                reference_image=_readonly_float64(
                    _reference_source_image(
                        reference_image,
                        archive,
                        "cam0",
                        (constants.IMAGE_HEIGHT_CAM0, constants.IMAGE_WIDTH_CAM0),
                    )
                ),
                reference_image_cam1=_readonly_float64(
                    _reference_source_image(
                        reference_image_cam1,
                        archive,
                        "cam1",
                        (constants.IMAGE_HEIGHT_CAM1, constants.IMAGE_WIDTH_CAM1),
                    )
                ),
                command_um_to_pixel_3d=_readonly_float64(command_um_to_pixel_3d),
            )


@functools.cache
def load_synthetic_cam1_reference() -> npt.NDArray[np.float64]:
    """Load the packaged development-mode reference image for camera 1."""
    return load_synthetic_calibration().reference_image_cam1


def _generated_synthetic_cam1_reference() -> npt.NDArray[np.float64]:
    """Generate a fallback development-mode reference image for camera 1."""
    rng = np.random.default_rng(20260507)
    shape = (constants.IMAGE_HEIGHT_CAM1, constants.IMAGE_WIDTH_CAM1)
    y, x = np.indices(shape, dtype=np.float64)
    image = ndimage.gaussian_filter(rng.normal(size=shape), sigma=3.0, mode="wrap")
    image += 0.25 * np.sin(x / 17.0) + 0.18 * np.cos(y / 23.0)
    image += 0.08 * np.sin((x + y) / 41.0)
    image -= float(np.min(image))
    maximum = float(np.max(image))
    if maximum > 0.0:
        image /= maximum
    return _readonly_float64(image)


def _reference_source_image(
    image: npt.ArrayLike,
    archive: np.lib.npyio.NpzFile,
    camera: str,
    full_shape: tuple[int, int],
) -> npt.NDArray:
    image_array = np.asarray(image)
    if image_array.shape[:2] == full_shape:
        return image_array

    rect = _reference_roi_rect(archive, camera, full_shape)
    if rect is None:
        return image_array

    y0, y1, x0, x1 = rect
    if image_array.shape[:2] != (y1 - y0, x1 - x0):
        logger.warning(
            "Ignoring %s ROI metadata because image shape %s does not match "
            "ROI crop shape %s.",
            camera,
            image_array.shape[:2],
            (y1 - y0, x1 - x0),
        )
        return image_array

    fill_value = float(np.median(image_array))
    canvas = np.full(full_shape, fill_value, dtype=image_array.dtype)
    canvas[y0:y1, x0:x1] = image_array
    return canvas


def _reference_roi_rect(
    archive: np.lib.npyio.NpzFile,
    camera: str,
    full_shape: tuple[int, int],
) -> tuple[int, int, int, int] | None:
    keys = (
        f"roi_{camera}_x",
        f"roi_{camera}_y",
        f"roi_{camera}_width",
        f"roi_{camera}_height",
    )
    if any(key not in archive.files for key in keys):
        return None

    try:
        x, y, width, height = (
            float(np.asarray(archive[key]).reshape(-1)[0]) for key in keys
        )
    except (TypeError, ValueError, IndexError):
        return None

    if not all(math.isfinite(value) for value in (x, y, width, height)):
        return None

    image_height, image_width = full_shape
    width = min(max(width, 1.0), float(image_width))
    height = min(max(height, 1.0), float(image_height))
    x = min(max(x, 0.0), float(image_width) - width)
    y = min(max(y, 0.0), float(image_height) - height)

    x0 = min(max(int(math.floor(x)), 0), image_width - 1)
    y0 = min(max(int(math.floor(y)), 0), image_height - 1)
    x1 = min(max(int(math.ceil(x + width)), x0 + 1), image_width)
    y1 = min(max(int(math.ceil(y + height)), y0 + 1), image_height)
    return y0, y1, x0, x1


class SimulatedHardware:
    """Shared deterministic fake hardware state for local development."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._move_lock = threading.Lock()
        self._positions = dict(DEFAULT_POSITIONS)

    def reset(self) -> None:
        """Reset fake motors to their default development-mode positions."""
        with self._move_lock:
            with self._lock:
                self._positions = dict(DEFAULT_POSITIONS)

    def get_positions(self, motor_aliases: Iterable[str]) -> tuple[float, ...]:
        aliases = tuple(motor_aliases)
        with self._lock:
            return tuple(self._position_for_alias(alias) for alias in aliases)

    def get_temperatures(self) -> tuple[float, float, float, float]:
        return STATIC_TEMPERATURE_K

    def move_motors_and_wait(
        self,
        motor_aliases: Iterable[str],
        goals: Iterable[float],
        *,
        max_retries: int = 4,
    ) -> tuple[float, ...]:
        aliases = tuple(motor_aliases)
        goals = tuple(float(goal) for goal in goals)

        if len(goals) != len(aliases):
            logger.error(
                "Simulated move failed: length of goals does not match "
                "length of motor_aliases."
            )
            return self.get_positions(aliases)
        if max_retries < 0:
            logger.error("Simulated move failed: max_retries must be non-negative.")
            return self.get_positions(aliases)

        with self._move_lock:
            with self._lock:
                previous = tuple(self._position_for_alias(alias) for alias in aliases)
                delay_s = self._settling_delay(previous, goals)
                for alias, goal in zip(aliases, goals, strict=True):
                    self._positions[alias] = goal
                final_positions = tuple(
                    self._position_for_alias(alias) for alias in aliases
                )

            if delay_s > 0:
                time.sleep(delay_s)

            return final_positions

    def get_reference_image(self) -> npt.NDArray[np.float64]:
        return load_synthetic_calibration().reference_image.copy()

    def get_reference_image_cam1(self) -> npt.NDArray[np.float64]:
        return load_synthetic_cam1_reference().copy()

    def get_command_um_to_pixel(
        self,
        camera: str | None = None,
    ) -> npt.NDArray[np.float64]:
        command_um_to_pixel = load_synthetic_calibration().command_um_to_pixel_3d
        if camera == "cam0":
            return command_um_to_pixel[0].copy()
        if camera == "cam1":
            return command_um_to_pixel[1].copy()
        return command_um_to_pixel.copy()

    def get_framegrabber_image(self) -> npt.NDArray[np.float64]:
        x_mm, y_mm, z_mm = self.get_positions(("x", "y", "z"))
        command_offset_um = np.array(
            [x_mm * 1000.0, y_mm * 1000.0, z_mm * 1000.0],
            dtype=np.float64,
        )
        calibration = load_synthetic_calibration()
        du_px, dv_px = calibration.command_um_to_pixel_3d[0] @ command_offset_um
        shifted = ndimage.shift(
            calibration.reference_image,
            shift=(float(dv_px), float(du_px)),
            order=3,
            mode="nearest",
        )
        return np.asarray(shifted, dtype=np.float64)[
            : constants.IMAGE_HEIGHT_CAM0, : constants.IMAGE_WIDTH_CAM0
        ].copy()

    def get_basler_image(self) -> npt.NDArray[np.float64]:
        x_mm, y_mm, z_mm = self.get_positions(("x", "y", "z"))
        command_offset_um = np.array(
            [x_mm * 1000.0, y_mm * 1000.0, z_mm * 1000.0],
            dtype=np.float64,
        )
        calibration = load_synthetic_calibration()
        du_px, dv_px = calibration.command_um_to_pixel_3d[1] @ command_offset_um
        shifted = ndimage.shift(
            calibration.reference_image_cam1,
            shift=(float(dv_px), float(du_px)),
            order=3,
            mode="nearest",
        )
        return np.asarray(shifted, dtype=np.float64)[
            : constants.IMAGE_HEIGHT_CAM1, : constants.IMAGE_WIDTH_CAM1
        ].copy()

    def _position_for_alias(self, alias: str) -> float:
        if alias not in constants.MOTOR_NAMES:
            raise KeyError(alias)
        return float(self._positions.get(alias, 0.0))

    @staticmethod
    def _settling_delay(
        previous: tuple[float, ...],
        goals: tuple[float, ...],
    ) -> float:
        if not previous:
            return 0.0
        max_delta = max(abs(goal - position) for position, goal in zip(previous, goals))
        if max_delta == 0.0:
            return 0.0
        return min(0.5, 0.05 + 2.0 * max_delta)


simulator = SimulatedHardware()

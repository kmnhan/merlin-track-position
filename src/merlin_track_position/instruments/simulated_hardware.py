"""Development-mode simulator for the sample manipulator and framegrabber."""

from __future__ import annotations

import functools
import logging
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
    stage_to_pixel: npt.NDArray[np.float64]
    reference_stage_um: npt.NDArray[np.float64]


CAM0_STAGE_TO_PIXEL_3D = np.array(
    [
        [0.27, -0.14, 0.07],
        [0.09, 0.31, -0.12],
    ],
    dtype=np.float64,
)
CAM1_STAGE_TO_PIXEL_3D = np.array(
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
            return SyntheticCalibration(
                reference_image=_readonly_float64(archive["reference_image"]),
                stage_to_pixel=_readonly_float64(archive["stage_to_pixel"]),
                reference_stage_um=_readonly_float64(archive["reference_stage_um"]),
            )


@functools.cache
def load_synthetic_cam1_reference() -> npt.NDArray[np.float64]:
    """Generate a deterministic development-mode reference image for camera 1."""
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


def _normalize_tolerances(
    tolerance: float | Iterable[float] | None,
    count: int,
) -> tuple[float, ...] | None:
    if tolerance is None:
        return None
    if np.isscalar(tolerance):
        return (float(tolerance),) * count
    tolerances = tuple(float(t) for t in tolerance)
    if len(tolerances) != count:
        raise ValueError(
            f"expected {count} tolerance values, got {len(tolerances)}"
        )
    return tolerances


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
        tolerance: float | Iterable[float] | None = None,
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

        _normalize_tolerances(tolerance, len(aliases))

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

    def get_stage_to_pixel(self, camera: str | None = None) -> npt.NDArray[np.float64]:
        if camera == "cam0":
            return CAM0_STAGE_TO_PIXEL_3D.copy()
        if camera == "cam1":
            return CAM1_STAGE_TO_PIXEL_3D.copy()
        return np.stack(
            [CAM0_STAGE_TO_PIXEL_3D, CAM1_STAGE_TO_PIXEL_3D],
            axis=0,
        )

    def get_framegrabber_image(self) -> npt.NDArray[np.float64]:
        x_mm, y_mm, z_mm = self.get_positions(("x", "y", "z"))
        stage_offset_um = np.array(
            [x_mm * 1000.0, y_mm * 1000.0, z_mm * 1000.0],
            dtype=np.float64,
        )
        du_px, dv_px = CAM0_STAGE_TO_PIXEL_3D @ stage_offset_um
        shifted = ndimage.shift(
            load_synthetic_calibration().reference_image,
            shift=(float(dv_px), float(du_px)),
            order=3,
            mode="nearest",
        )
        return (
            np.asarray(shifted, dtype=np.float64)[
                : constants.IMAGE_HEIGHT_CAM0, : constants.IMAGE_WIDTH_CAM0
            ]
            .copy()
        )

    def get_basler_image(self) -> npt.NDArray[np.float64]:
        x_mm, y_mm, z_mm = self.get_positions(("x", "y", "z"))
        stage_offset_um = np.array(
            [x_mm * 1000.0, y_mm * 1000.0, z_mm * 1000.0],
            dtype=np.float64,
        )
        du_px, dv_px = CAM1_STAGE_TO_PIXEL_3D @ stage_offset_um
        shifted = ndimage.shift(
            load_synthetic_cam1_reference(),
            shift=(float(dv_px), float(du_px)),
            order=3,
            mode="nearest",
        )
        return (
            np.asarray(shifted, dtype=np.float64)[
                : constants.IMAGE_HEIGHT_CAM1, : constants.IMAGE_WIDTH_CAM1
            ]
            .copy()
        )

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

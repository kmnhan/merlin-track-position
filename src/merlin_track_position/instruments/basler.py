"""Placeholder access layer for the second, Basler-based camera."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from merlin_track_position import constants
from merlin_track_position.instruments.simulated_hardware import simulator


def get_basler_image(timeout_ms: int = 5000) -> npt.NDArray[np.float64]:
    """Return the latest image from camera 1.

    The real Basler acquisition framework is intentionally left as a future
    implementation. Development mode uses the deterministic simulator so the
    stereo tracking path can be exercised without camera hardware.
    """
    del timeout_ms
    if not constants.IS_DAQ_PC:
        return simulator.get_basler_image()

    # TODO: Replace with the Basler/pypylon acquisition framework for camera 1.
    raise NotImplementedError("Basler camera acquisition is not implemented yet")

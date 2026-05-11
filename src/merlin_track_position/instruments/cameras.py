"""Helpers for acquiring one logical sample from both tracking cameras."""

from __future__ import annotations

import numpy.typing as npt

from merlin_track_position.instruments.basler import get_basler_image
from merlin_track_position.instruments.framegrab import get_framegrabber_image


def capture_camera_pair() -> tuple[npt.NDArray, npt.NDArray]:
    """Capture the current cam0 and cam1 images."""
    return get_framegrabber_image(), get_basler_image()

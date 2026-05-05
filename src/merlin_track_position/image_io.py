"""Image loading and normalization helpers for grayscale tracking."""

from __future__ import annotations

from typing import Any

import numpy as np


def as_grayscale_array(image: Any) -> np.ndarray:
    """Return *image* as a finite 2D float64 grayscale array.

    The project expects grayscale camera images. This helper accepts 2D arrays
    directly and also tolerates single-channel or RGB/RGBA files loaded by
    image libraries so notebook experiments are less brittle.
    """

    array = np.asarray(image)
    if array.ndim == 2:
        gray = array
    elif array.ndim == 3 and array.shape[2] == 1:
        gray = array[..., 0]
    elif array.ndim == 3 and array.shape[2] in (3, 4):
        rgb = array[..., :3].astype(np.float64, copy=False)
        gray = 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]
    else:
        raise ValueError(f"expected a grayscale 2D image, got shape {array.shape!r}")

    gray = gray.astype(np.float64, copy=False)
    if gray.size == 0:
        raise ValueError("image is empty")
    if not np.isfinite(gray).all():
        raise ValueError("image contains non-finite values")
    return gray


def normalize_intensity(
    image: Any,
    *,
    clip_percentiles: tuple[float, float] | None = (1.0, 99.0),
    eps: float = 1e-12,
) -> np.ndarray:
    """Robustly normalize a grayscale image to zero mean and unit variance."""

    gray = as_grayscale_array(image)
    working = gray.astype(np.float64, copy=True)

    if clip_percentiles is not None:
        low, high = np.percentile(working, clip_percentiles)
        if high - low > eps:
            working = np.clip(working, low, high)

    mean = float(np.mean(working))
    std = float(np.std(working))
    if std <= eps:
        return np.zeros_like(working)
    return (working - mean) / std

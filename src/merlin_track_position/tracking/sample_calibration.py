"""Generate deterministic sample stereo calibration datasets."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import xarray as xr
from scipy import ndimage

from merlin_track_position.constants import (
    IMAGE_HEIGHT_CAM0,
    IMAGE_HEIGHT_CAM1,
    IMAGE_WIDTH_CAM0,
    IMAGE_WIDTH_CAM1,
)
from merlin_track_position.tracking.calibration_core import (
    CAMERAS,
    OBSERVATION_AXES,
    PIXEL_AXES,
    STAGE_AXES,
)

DEFAULT_SAMPLE_CALIBRATION_PATH = Path(
    "/Users/khan/Downloads/cal_30.0um_origin_new_scheme.h5"
)
SAMPLE_STAGE_TO_PIXEL = np.array(
    [
        [[0.27, -0.14, 0.07], [0.09, 0.31, -0.12]],
        [[-0.21, 0.18, 0.33], [0.24, 0.05, 0.16]],
    ],
    dtype=np.float64,
)
SAMPLE_STAGE_UM = np.array(
    [
        [0.0, 0.0, 0.0],
        [30.0, 0.0, 0.0],
        [-30.0, 0.0, 0.0],
        [0.0, 30.0, 0.0],
        [0.0, -30.0, 0.0],
        [0.0, 0.0, 30.0],
        [0.0, 0.0, -30.0],
        [0.0, 0.0, 0.0],
    ],
    dtype=np.float64,
)


def build_sample_calibration_dataset(
    *,
    image_shape_cam0: tuple[int, int] = (IMAGE_HEIGHT_CAM0, IMAGE_WIDTH_CAM0),
    image_shape_cam1: tuple[int, int] = (IMAGE_HEIGHT_CAM1, IMAGE_WIDTH_CAM1),
) -> xr.Dataset:
    """Build a deterministic two-camera calibration dataset."""
    stage_to_observation = SAMPLE_STAGE_TO_PIXEL.reshape(
        len(OBSERVATION_AXES),
        len(STAGE_AXES),
    )
    pixel_to_stage = np.linalg.pinv(stage_to_observation)
    measured = (SAMPLE_STAGE_UM @ stage_to_observation.T).reshape(
        SAMPLE_STAGE_UM.shape[0],
        len(CAMERAS),
        len(PIXEL_AXES),
    )
    residual_shift = np.zeros_like(measured)
    residual_stage = np.zeros_like(SAMPLE_STAGE_UM)

    images_cam0 = _shifted_stack(_texture(image_shape_cam0, 1100), measured[:, 0, :])
    images_cam1 = _shifted_stack(_texture(image_shape_cam1, 2200), measured[:, 1, :])

    coords = {
        "sample": np.arange(SAMPLE_STAGE_UM.shape[0], dtype=np.int64),
        "stage_axis": list(STAGE_AXES),
        "camera": list(CAMERAS),
        "pixel_axis": list(PIXEL_AXES),
        "observation_axis": list(OBSERVATION_AXES),
        "repeatability_position": np.arange(1, dtype=np.int64),
        "y_cam0": np.arange(image_shape_cam0[0], dtype=np.int64),
        "x_cam0": np.arange(image_shape_cam0[1], dtype=np.int64),
        "y_cam1": np.arange(image_shape_cam1[0], dtype=np.int64),
        "x_cam1": np.arange(image_shape_cam1[1], dtype=np.int64),
    }
    repeatability_shift = measured[[0, -1]]
    repeatability_std = np.std(repeatability_shift, axis=0, ddof=1)
    repeatability_mean = np.mean(repeatability_shift, axis=0)

    return xr.Dataset(
        data_vars={
            "stage_to_pixel": (
                ("camera", "pixel_axis", "stage_axis"),
                SAMPLE_STAGE_TO_PIXEL,
                {"units": "px/um"},
            ),
            "pixel_to_stage": (
                ("stage_axis", "observation_axis"),
                pixel_to_stage,
                {"units": "um/px"},
            ),
            "reference_stage_um": (
                ("stage_axis",),
                np.zeros(len(STAGE_AXES), dtype=np.float64),
                {"units": "um"},
            ),
            "condition_number": ((), float(np.linalg.cond(stage_to_observation))),
            "origin_stability_um": ((), 5.0, {"units": "um"}),
            "return_to_origin_motor_error_um": (
                ("stage_axis",),
                np.zeros(len(STAGE_AXES), dtype=np.float64),
                {"units": "um"},
            ),
            "return_to_origin_motor_error_norm_um": ((), 0.0, {"units": "um"}),
            "return_to_origin_image_error_px": (
                ("camera", "pixel_axis"),
                np.zeros((len(CAMERAS), len(PIXEL_AXES)), dtype=np.float64),
                {"units": "px"},
            ),
            "return_to_origin_image_error_um": (
                ("stage_axis",),
                np.zeros(len(STAGE_AXES), dtype=np.float64),
                {"units": "um"},
            ),
            "return_to_origin_image_error_norm_um": ((), 0.0, {"units": "um"}),
            "stage_um": (
                ("sample", "stage_axis"),
                SAMPLE_STAGE_UM,
                {"units": "um"},
            ),
            "measured_shift_px": (
                ("sample", "camera", "pixel_axis"),
                measured,
                {"units": "px"},
            ),
            "predicted_shift_px": (
                ("sample", "camera", "pixel_axis"),
                measured.copy(),
                {"units": "px"},
            ),
            "residual_shift_px": (
                ("sample", "camera", "pixel_axis"),
                residual_shift,
                {"units": "px"},
            ),
            "residual_stage_um": (
                ("sample", "stage_axis"),
                residual_stage,
                {"units": "um"},
            ),
            "measurement_warnings": (
                ("sample", "camera"),
                np.full((SAMPLE_STAGE_UM.shape[0], len(CAMERAS)), "", dtype=str),
            ),
            "repeatability_stage_um": (
                ("repeatability_position", "stage_axis"),
                np.zeros((1, len(STAGE_AXES)), dtype=np.float64),
                {"units": "um"},
            ),
            "repeatability_count": (
                ("repeatability_position",),
                np.array([2], dtype=np.int64),
            ),
            "repeatability_mean_shift_px": (
                ("repeatability_position", "camera", "pixel_axis"),
                repeatability_mean.reshape(1, len(CAMERAS), len(PIXEL_AXES)),
                {"units": "px"},
            ),
            "repeatability_std_shift_px": (
                ("repeatability_position", "camera", "pixel_axis"),
                repeatability_std.reshape(1, len(CAMERAS), len(PIXEL_AXES)),
                {"units": "px"},
            ),
            "repeatability_rms_std_px": (
                ("repeatability_position",),
                np.array(
                    [float(np.sqrt(np.mean(repeatability_std * repeatability_std)))],
                    dtype=np.float64,
                ),
                {"units": "px"},
            ),
            "image_cam0": (
                ("sample", "y_cam0", "x_cam0"),
                images_cam0,
                {"description": "camera 0 calibration grayscale image stack"},
            ),
            "image_cam1": (
                ("sample", "y_cam1", "x_cam1"),
                images_cam1,
                {"description": "camera 1 calibration grayscale image stack"},
            ),
        },
        coords=coords,
        attrs={
            "format": "merlin-track-position calibration",
            "format_version": "1",
            "model": "through_origin_linear_stereo",
            "warnings": "",
        },
    )


def write_sample_calibration_dataset(
    path: Path = DEFAULT_SAMPLE_CALIBRATION_PATH,
) -> Path:
    """Write the deterministic sample calibration dataset and return its path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    build_sample_calibration_dataset().to_netcdf(path, engine="h5netcdf")
    return path


def _texture(shape: tuple[int, int], seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    y, x = np.indices(shape, dtype=np.float64)
    image = ndimage.gaussian_filter(rng.normal(size=shape), sigma=3.0, mode="wrap")
    image += 0.25 * np.sin(x / 19.0) + 0.20 * np.cos(y / 29.0)
    image += 0.10 * np.sin((x + 2.0 * y) / 47.0)
    image -= float(np.min(image))
    maximum = float(np.max(image))
    if maximum > 0.0:
        image /= maximum
    return image.astype(np.float32)


def _shifted_stack(reference: np.ndarray, shifts_px: np.ndarray) -> np.ndarray:
    images = [
        ndimage.shift(
            reference,
            shift=(float(dv_px), float(du_px)),
            order=3,
            mode="nearest",
        ).astype(np.float32)
        for du_px, dv_px in shifts_px
    ]
    return np.stack(images, axis=0).astype(np.float32)

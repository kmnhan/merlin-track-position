"""Generate deterministic sample calibration datasets."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import xarray as xr
from scipy import ndimage

from merlin_track_position import constants
from merlin_track_position.constants import (
    IMAGE_HEIGHT_CAM0,
    IMAGE_HEIGHT_CAM1,
    IMAGE_WIDTH_CAM0,
    IMAGE_WIDTH_CAM1,
)
from merlin_track_position.tracking.calibration_core import (
    CAMERAS,
    COMMAND_AXES,
    OBSERVATION_AXES,
    PIXEL_AXES,
    PROBE_COMMAND_DELTA_MODE_ABSOLUTE_CENTER,
    PROBE_COMMAND_DELTA_MODE_ATTR,
    derive_axis_scale_from_jacobian,
    save_calibration_dataset,
)
from merlin_track_position.tracking.calibrate import _make_visual_probe_offsets_um

DEFAULT_SAMPLE_CALIBRATION_PATH = Path(
    "/Users/khan/Downloads/calibration.h5"
)
SAMPLE_PX_PER_CMD_MM = np.array(
    [
        [[270.0, -140.0, 70.0], [90.0, 310.0, -120.0]],
        [[-210.0, 180.0, 330.0], [240.0, 50.0, 160.0]],
    ],
    dtype=np.float64,
)


def build_sample_calibration_dataset(
    *,
    image_shape_cam0: tuple[int, int] = (IMAGE_HEIGHT_CAM0, IMAGE_WIDTH_CAM0),
    image_shape_cam1: tuple[int, int] = (IMAGE_HEIGHT_CAM1, IMAGE_WIDTH_CAM1),
) -> xr.Dataset:
    """Build a deterministic commanded-mm calibration dataset."""

    command_delta = _sample_probe_deltas()
    jacobian_observation = SAMPLE_PX_PER_CMD_MM.reshape(
        len(OBSERVATION_AXES),
        len(COMMAND_AXES),
    )
    measured = (command_delta @ jacobian_observation.T).reshape(
        command_delta.shape[0],
        len(CAMERAS),
        len(PIXEL_AXES),
    )
    axis_scale, *_ = derive_axis_scale_from_jacobian(
        SAMPLE_PX_PER_CMD_MM,
        command_delta,
    )

    reference_cam0 = _texture(image_shape_cam0, 1100)
    reference_cam1 = _texture(image_shape_cam1, 2200)

    pre_commanded, post_commanded = _command_positions(command_delta)
    readback_offset = np.array([0.002, -0.004, 0.001], dtype=np.float64)
    pre_readback = pre_commanded + readback_offset
    post_readback = post_commanded + readback_offset

    coords = {
        "probe": np.arange(command_delta.shape[0], dtype=np.int64),
        "command_axis": list(COMMAND_AXES),
        "camera": list(CAMERAS),
        "pixel_axis": list(PIXEL_AXES),
        "y_cam0": np.arange(image_shape_cam0[0], dtype=np.int64),
        "x_cam0": np.arange(image_shape_cam0[1], dtype=np.int64),
        "y_cam1": np.arange(image_shape_cam1[0], dtype=np.int64),
        "x_cam1": np.arange(image_shape_cam1[1], dtype=np.int64),
    }
    return xr.Dataset(
        data_vars={
            "px_per_cmd_mm": (
                ("camera", "pixel_axis", "command_axis"),
                SAMPLE_PX_PER_CMD_MM,
                {"units": "px/commanded-mm"},
            ),
            "axis_scale_cmd_mm": (
                ("command_axis",),
                axis_scale,
                {"units": "commanded-mm"},
            ),
            "reference_cam0": (("y_cam0", "x_cam0"), reference_cam0),
            "reference_cam1": (("y_cam1", "x_cam1"), reference_cam1),
            "probe_command_delta_mm": (
                ("probe", "command_axis"),
                command_delta,
                {"units": "commanded-mm"},
            ),
            "probe_measured_delta_px": (
                ("probe", "camera", "pixel_axis"),
                measured,
                {"units": "px"},
            ),
            "pre_commanded_position_mm": (
                ("probe", "command_axis"),
                pre_commanded,
                {"units": "commanded-mm"},
            ),
            "post_commanded_position_mm": (
                ("probe", "command_axis"),
                post_commanded,
                {"units": "commanded-mm"},
            ),
            "pre_readback_position_mm": (
                ("probe", "command_axis"),
                pre_readback,
                {"units": "readback-mm"},
            ),
            "post_readback_position_mm": (
                ("probe", "command_axis"),
                post_readback,
                {"units": "readback-mm"},
            ),
            "probe_capture_shift_mad_px": (
                ("probe", "camera", "pixel_axis"),
                np.zeros_like(measured),
                {"units": "px"},
            ),
            "probe_registration_warnings": (
                ("probe", "camera"),
                np.full((command_delta.shape[0], len(CAMERAS)), "", dtype=str),
            ),
        },
        coords=coords,
        attrs={
            "capture_count": 1,
            "initial_x_mm": 1.0,
            "initial_y_mm": 2.0,
            "initial_z_mm": 3.0,
            PROBE_COMMAND_DELTA_MODE_ATTR: PROBE_COMMAND_DELTA_MODE_ABSOLUTE_CENTER,
            "warnings": "",
        },
    )


def write_sample_calibration_dataset(
    path: Path = DEFAULT_SAMPLE_CALIBRATION_PATH,
) -> Path:
    """Write the deterministic sample calibration dataset and return its path."""

    return save_calibration_dataset(build_sample_calibration_dataset(), path)


def _sample_probe_deltas() -> np.ndarray:
    return np.asarray(
        _make_visual_probe_offsets_um(
            constants.DEFAULT_VISUAL_CALIBRATION_N,
            constants.DEFAULT_VISUAL_CALIBRATION_STEP_UM,
        )
        / 1000.0,
        dtype=np.float64,
    )


def _command_positions(
    command_delta: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    center = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    current = center.copy()
    pre_rows: list[np.ndarray] = []
    post_rows: list[np.ndarray] = []
    for offset in command_delta:
        pre_rows.append(current.copy())
        current = center + offset
        post_rows.append(current.copy())
    return np.stack(pre_rows, axis=0), np.stack(post_rows, axis=0)


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

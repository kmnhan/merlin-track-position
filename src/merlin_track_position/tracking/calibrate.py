from __future__ import annotations

import logging
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr

from merlin_track_position import constants
from merlin_track_position.instruments.cameras import (
    CameraPairPlugin,
    capture_image_and_display_stacks,
    default_camera_pair,
    normalize_capture_count,
)
from merlin_track_position.instruments.motors import get_positions, move_motors_and_wait
from merlin_track_position.tracking.calibration_core import (
    COMMAND_AXES,
    fit_visual_jacobian_calibration,
    load_calibration_dataset,
    save_calibration_dataset,
)

logger = logging.getLogger("merlin_track_position.tracking.calibrate")


def visual_calibration_probe_count(
    repeats_per_direction: int = constants.DEFAULT_VISUAL_CALIBRATION_REPEATS_PER_DIRECTION,
) -> int:
    """Return the number of commanded-mm probes in the visual calibration."""

    repeats = int(repeats_per_direction)
    if repeats < 1:
        raise ValueError("repeats_per_direction must be >= 1")
    return repeats * len(COMMAND_AXES) * 2


def run_calibration(
    camera_pair: CameraPairPlugin | None = None,
    *,
    output_path: str | Path,
    step_mm_by_axis: Mapping[str, float] | None = None,
    repeats_per_direction: int = constants.DEFAULT_VISUAL_CALIBRATION_REPEATS_PER_DIRECTION,
    min_shift_px: float = constants.DEFAULT_VISUAL_CALIBRATION_MIN_SHIFT_PX,
    capture_count: int = constants.DEFAULT_CAPTURE_COUNT,
    additional_context: Mapping[str, Any] | None = None,
    processing_callback: Callable[[int, int], None] | None = None,
    step_callback: Callable[
        [int, float, float, float, np.ndarray, np.ndarray],
        None,
    ]
    | None = None,
    **shift_kwargs: Any,
) -> xr.Dataset:
    """Run and save a visual-Jacobian calibration.

    Motor positions are treated as commanded-mm coordinates. Readback positions are
    recorded only as diagnostics; they are not used to fit the Jacobian.
    """

    capture_count = normalize_capture_count(capture_count)
    output_path = Path(output_path)
    probe_deltas = _make_visual_probe_deltas(step_mm_by_axis, repeats_per_direction)

    if camera_pair is None:
        camera_pair = default_camera_pair()

    x0, y0, z0, polar, tilt, cam = get_positions(
        ("x", "y", "z", "p", "t", "cam")
    )
    commanded_position = np.asarray([x0, y0, z0], dtype=np.float64)

    if not np.isclose(cam, 5.0):
        # Camera 5 is the sample-view video-switch position used for alignment.
        move_motors_and_wait(("cam",), (5,))
        time.sleep(4.0)

    logger.info(
        "Starting visual-Jacobian calibration with %d probes",
        len(probe_deltas),
    )
    reference_stacks, _ = capture_image_and_display_stacks(
        camera_pair,
        capture_count,
    )
    reference_cam0, reference_cam1 = (
        _representative_image(reference_stacks[0]),
        _representative_image(reference_stacks[1]),
    )

    before_images_cam0: list[np.ndarray] = []
    after_images_cam0: list[np.ndarray] = []
    before_images_cam1: list[np.ndarray] = []
    after_images_cam1: list[np.ndarray] = []
    command_delta_mm: list[np.ndarray] = []
    pre_commanded_position_mm: list[np.ndarray] = []
    post_commanded_position_mm: list[np.ndarray] = []
    pre_readback_position_mm: list[np.ndarray] = []
    post_readback_position_mm: list[np.ndarray] = []

    for probe_index, delta in enumerate(probe_deltas):
        pre_commanded = commanded_position.copy()
        pre_readback = np.asarray(get_positions(COMMAND_AXES), dtype=np.float64)
        logger.info(
            "Probe %d/%d: commanded delta mm=(%.4g, %.4g, %.4g)",
            probe_index + 1,
            len(probe_deltas),
            delta[0],
            delta[1],
            delta[2],
        )
        before_stacks, _ = capture_image_and_display_stacks(
            camera_pair,
            capture_count,
        )
        target_position = commanded_position + delta
        final_readback = np.asarray(
            move_motors_and_wait(
                COMMAND_AXES,
                tuple(float(value) for value in target_position),
            ),
            dtype=np.float64,
        )
        time.sleep(0.5)
        after_stacks, after_display_stacks = capture_image_and_display_stacks(
            camera_pair,
            capture_count,
        )

        before_images_cam0.append(before_stacks[0])
        before_images_cam1.append(before_stacks[1])
        after_images_cam0.append(after_stacks[0])
        after_images_cam1.append(after_stacks[1])
        command_delta_mm.append(delta.copy())
        pre_commanded_position_mm.append(pre_commanded)
        post_commanded_position_mm.append(target_position.copy())
        pre_readback_position_mm.append(pre_readback)
        post_readback_position_mm.append(final_readback)

        commanded_position = target_position
        if step_callback is not None:
            display_cam0, display_cam1 = after_display_stacks
            step_callback(
                probe_index,
                float(delta[0]),
                float(delta[1]),
                float(delta[2]),
                _representative_image(display_cam0),
                _representative_image(display_cam1),
            )

    context: dict[str, Any] = {
        "initial_x_mm": float(x0),
        "initial_y_mm": float(y0),
        "initial_z_mm": float(z0),
        "polar": float(polar),
        "tilt": float(tilt),
        "calibration_path": str(output_path),
    }
    if additional_context is not None:
        context |= {str(key): value for key, value in additional_context.items()}

    shift_options = {"check_tiles": True} | shift_kwargs
    calibration = fit_visual_jacobian_calibration(
        reference_cam0=reference_cam0,
        reference_cam1=reference_cam1,
        before_images_cam0=before_images_cam0,
        after_images_cam0=after_images_cam0,
        before_images_cam1=before_images_cam1,
        after_images_cam1=after_images_cam1,
        command_delta_mm=command_delta_mm,
        pre_commanded_position_mm=pre_commanded_position_mm,
        post_commanded_position_mm=post_commanded_position_mm,
        pre_readback_position_mm=pre_readback_position_mm,
        post_readback_position_mm=post_readback_position_mm,
        min_shift_px=min_shift_px,
        progress_callback=processing_callback,
        additional_context=context,
        **shift_options,
    )
    save_calibration_dataset(calibration, output_path)
    return load_calibration_dataset(output_path)


def _make_visual_probe_deltas(
    step_mm_by_axis: Mapping[str, float] | None,
    repeats_per_direction: int,
) -> list[np.ndarray]:
    repeats = int(repeats_per_direction)
    if repeats < 1:
        raise ValueError("repeats_per_direction must be >= 1")

    steps = _step_mm_by_axis_from_config(step_mm_by_axis)
    deltas: list[np.ndarray] = []
    for _ in range(repeats):
        for axis_index in range(len(COMMAND_AXES)):
            for sign in (1.0, -1.0):
                delta = np.zeros(len(COMMAND_AXES), dtype=np.float64)
                delta[axis_index] = sign * steps[axis_index]
                deltas.append(delta)
    return deltas


def _step_mm_by_axis_from_config(
    step_mm_by_axis: Mapping[str, float] | None,
) -> np.ndarray:
    config = (
        constants.DEFAULT_VISUAL_CALIBRATION_STEP_MM_BY_AXIS
        if step_mm_by_axis is None
        else step_mm_by_axis
    )
    values: list[float] = []
    for axis in COMMAND_AXES:
        if axis not in config:
            raise ValueError(f"missing visual calibration step for axis {axis!r}")
        value = float(config[axis])
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(
                f"visual calibration step for axis {axis!r} must be finite and positive"
            )
        values.append(value)
    return np.asarray(values, dtype=np.float64)


def _representative_image(images: Sequence[Any]) -> np.ndarray:
    image_array = np.asarray(images)
    representative = np.mean(image_array, axis=0)
    return _cast_representative_image(representative, image_array.dtype)


def _cast_representative_image(image: np.ndarray, dtype: np.dtype) -> np.ndarray:
    if np.issubdtype(dtype, np.bool_):
        return np.asarray(image >= 0.5, dtype=dtype)
    if np.issubdtype(dtype, np.integer):
        dtype_info = np.iinfo(dtype)
        image = np.clip(np.rint(image), dtype_info.min, dtype_info.max)
    return np.asarray(image, dtype=dtype)

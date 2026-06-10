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
    CAMERAS,
    CAPTURE_AGGREGATION_MEDIAN_SHIFTS,
    COMMAND_AXES,
    PROBE_COMMAND_DELTA_MODE_ABSOLUTE_CENTER,
    PROBE_COMMAND_DELTA_MODE_ATTR,
    _image_coords,
    _image_dims,
    fit_jacobian_calibration,
    load_calibration_dataset,
    save_calibration_dataset_deferred,
    validate_visual_calibration_dataset,
)
from merlin_track_position.tracking.persistence import persistence_result_attrs
from merlin_track_position.tracking.roi import (
    crop_stack_to_roi,
    roi_geometry_from_attrs,
)

logger = logging.getLogger("merlin_track_position.tracking.calibrate")


def visual_calibration_probe_count(
    n: int = constants.DEFAULT_VISUAL_CALIBRATION_N,
) -> int:
    """Return the number of requested probes in the visual calibration."""

    return len(
        _make_visual_probe_offsets_um(
            n,
            constants.DEFAULT_VISUAL_CALIBRATION_STEP_UM,
        )
    )


def run_calibration(
    camera_pair: CameraPairPlugin | None = None,
    *,
    output_path: str | Path,
    n: int = constants.DEFAULT_VISUAL_CALIBRATION_N,
    step_um: float = constants.DEFAULT_VISUAL_CALIBRATION_STEP_UM,
    min_shift_px: float = constants.DEFAULT_VISUAL_CALIBRATION_MIN_SHIFT_PX,
    capture_count: int = constants.DEFAULT_CALIBRATION_CAPTURE_COUNT,
    capture_aggregation: str = CAPTURE_AGGREGATION_MEDIAN_SHIFTS,
    additional_context: Mapping[str, Any] | None = None,
    processing_callback: Callable[[int, int], None] | None = None,
    step_callback: Callable[
        [int, float, float, float, np.ndarray, np.ndarray],
        None,
    ]
    | None = None,
    **shift_kwargs: Any,
) -> xr.Dataset:
    """Run and save a calibration.

    Motor readbacks are treated as the calibrated x/y/z coordinates. Requested
    positions are retained as diagnostics for command-vs-readback comparison.
    """

    capture_count = normalize_capture_count(capture_count)
    output_path = Path(output_path)
    probe_offsets = _make_visual_probe_offsets_um(n, step_um) / 1000.0

    if camera_pair is None:
        camera_pair = default_camera_pair()

    roi_geometries = {
        camera: roi_geometry_from_attrs(additional_context or {}, camera)
        for camera in CAMERAS
    }

    x0, y0, z0, polar, tilt, azi, cam = get_positions(
        ("x", "y", "z", "p", "t", "a", "cam")
    )
    initial_readback_position = np.asarray([x0, y0, z0], dtype=np.float64)
    requested_position = initial_readback_position.copy()

    if not np.isclose(cam, 5.0):
        # Camera 5 is the sample-view video-switch position used for alignment.
        move_motors_and_wait(("cam",), (5,))
        time.sleep(4.0)

    logger.info(
        "Starting calibration with %d probes",
        len(probe_offsets),
    )
    reference_stacks, _ = capture_image_and_display_stacks(
        camera_pair,
        capture_count,
    )
    full_reference_cam0, full_reference_cam1 = (
        _representative_image(reference_stacks[0]),
        _representative_image(reference_stacks[1]),
    )
    processing_reference_stacks = _crop_stacks_for_calibration(
        reference_stacks,
        roi_geometries,
    )
    reference_cam0, reference_cam1 = (
        _representative_image(processing_reference_stacks[0]),
        _representative_image(processing_reference_stacks[1]),
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

    for probe_index, offset in enumerate(probe_offsets):
        pre_commanded = requested_position.copy()
        pre_readback = np.asarray(get_positions(COMMAND_AXES), dtype=np.float64)
        logger.info(
            "Probe %d/%d: commanded offset mm=(%.4g, %.4g, %.4g)",
            probe_index + 1,
            len(probe_offsets),
            offset[0],
            offset[1],
            offset[2],
        )
        target_position = initial_readback_position + offset
        final_readback = np.asarray(
            move_motors_and_wait(
                COMMAND_AXES,
                tuple(float(value) for value in target_position),
            ),
            dtype=np.float64,
        )
        after_stacks, after_display_stacks = capture_image_and_display_stacks(
            camera_pair,
            capture_count,
        )
        after_processing_stacks = _crop_stacks_for_calibration(
            after_stacks,
            roi_geometries,
        )

        before_images_cam0.append(processing_reference_stacks[0])
        before_images_cam1.append(processing_reference_stacks[1])
        after_images_cam0.append(after_processing_stacks[0])
        after_images_cam1.append(after_processing_stacks[1])
        command_delta_mm.append(offset.copy())
        pre_commanded_position_mm.append(pre_commanded)
        post_commanded_position_mm.append(target_position.copy())
        pre_readback_position_mm.append(pre_readback)
        post_readback_position_mm.append(final_readback)

        requested_position = target_position
        if step_callback is not None:
            display_cam0, display_cam1 = after_display_stacks
            step_callback(
                probe_index,
                float(offset[0]),
                float(offset[1]),
                float(offset[2]),
                _representative_image(display_cam0),
                _representative_image(display_cam1),
            )

    context: dict[str, Any] = {
        "initial_x_mm": float(x0),
        "initial_y_mm": float(y0),
        "initial_z_mm": float(z0),
        "polar": float(polar),
        "tilt": float(tilt),
        "azi": float(azi),
        "calibration_path": str(output_path),
        PROBE_COMMAND_DELTA_MODE_ATTR: PROBE_COMMAND_DELTA_MODE_ABSOLUTE_CENTER,
    }
    if additional_context is not None:
        context |= {str(key): value for key, value in additional_context.items()}

    shift_options = {"check_tiles": True} | shift_kwargs
    calibration = fit_jacobian_calibration(
        reference_cam0=reference_cam0,
        reference_cam1=reference_cam1,
        before_images_cam0=before_images_cam0,
        after_images_cam0=after_images_cam0,
        before_images_cam1=before_images_cam1,
        after_images_cam1=after_images_cam1,
        command_delta_mm=command_delta_mm,
        initial_readback_position_mm=initial_readback_position,
        pre_commanded_position_mm=pre_commanded_position_mm,
        post_commanded_position_mm=post_commanded_position_mm,
        pre_readback_position_mm=pre_readback_position_mm,
        post_readback_position_mm=post_readback_position_mm,
        min_shift_px=min_shift_px,
        capture_aggregation=capture_aggregation,
        progress_callback=processing_callback,
        additional_context=context,
        **shift_options,
    )
    calibration = _replace_reference_images(
        calibration,
        full_reference_cam0,
        full_reference_cam1,
    )
    persistence = save_calibration_dataset_deferred(calibration, output_path)
    if persistence.flushed:
        saved = load_calibration_dataset(output_path)
    else:
        saved = calibration.load().copy(deep=True)
        saved.attrs["calibration_path"] = str(output_path)
    return saved.assign_attrs(persistence_result_attrs("calibration", persistence))


def _crop_stacks_for_calibration(
    stacks: tuple[np.ndarray, ...],
    roi_geometries: Mapping[str, tuple[float, float, float, float] | None],
) -> tuple[np.ndarray, ...]:
    cropped: list[np.ndarray] = []
    for camera, stack in zip(CAMERAS, stacks, strict=True):
        roi_geometry = roi_geometries[camera]
        if roi_geometry is None:
            cropped.append(stack)
        else:
            cropped.append(crop_stack_to_roi(stack, roi_geometry))
    return tuple(cropped)


def _replace_reference_images(
    calibration: xr.Dataset,
    reference_cam0: np.ndarray,
    reference_cam1: np.ndarray,
) -> xr.Dataset:
    updated = calibration.drop_vars(
        [
            name
            for name in (
                "reference_cam0",
                "reference_cam1",
                "y_cam0",
                "x_cam0",
                "channel_cam0",
                "y_cam1",
                "x_cam1",
                "channel_cam1",
            )
            if name in calibration.variables
        ]
    )
    updated = updated.assign_coords(
        _image_coords("cam0", reference_cam0) | _image_coords("cam1", reference_cam1)
    )
    updated["reference_cam0"] = (_image_dims("cam0", reference_cam0), reference_cam0)
    updated["reference_cam1"] = (_image_dims("cam1", reference_cam1), reference_cam1)
    validate_visual_calibration_dataset(updated)
    return updated


def _make_visual_probe_deltas(
    n: int,
    step_um: float,
) -> list[np.ndarray]:
    offsets_um = _make_visual_probe_offsets_um(n, step_um)
    origin_um = np.zeros((1, len(COMMAND_AXES)), dtype=np.float64)
    delta_um = np.diff(np.vstack((origin_um, offsets_um)), axis=0)
    return [row.astype(np.float64, copy=True) / 1000.0 for row in delta_um]


def _make_visual_probe_offsets_um(n: int, step_um: float) -> np.ndarray:
    points = int(n)
    if points < 2:
        raise ValueError("n must be >= 2")
    step = float(step_um)
    if not np.isfinite(step) or step <= 0.0:
        raise ValueError("step_um must be finite and positive")

    offsets = (np.arange(points, dtype=np.float64) - (points - 1) / 2.0) * step
    offsets = offsets[~np.isclose(offsets, 0.0)]
    positive_offsets = sorted(float(offset) for offset in offsets if offset > 0.0)
    negative_offsets = sorted(float(offset) for offset in offsets if offset < 0.0)
    corner_step = step

    rows: list[list[float]] = []
    for z_offset in (*positive_offsets, *negative_offsets):
        rows.append([0.0, 0.0, z_offset])

    positive_y_levels = sorted({*positive_offsets, corner_step})
    for y_offset in positive_y_levels:
        if _contains_offset(positive_offsets, y_offset):
            rows.append([0.0, y_offset, 0.0])
        if np.isclose(y_offset, corner_step):
            rows.extend(
                [
                    [corner_step, y_offset, corner_step],
                    [-corner_step, y_offset, corner_step],
                    [-corner_step, y_offset, -corner_step],
                    [corner_step, y_offset, -corner_step],
                ]
            )

    negative_y_levels = sorted({*negative_offsets, -corner_step})
    for y_offset in negative_y_levels:
        if np.isclose(y_offset, -corner_step):
            rows.extend(
                [
                    [corner_step, y_offset, -corner_step],
                    [corner_step, y_offset, corner_step],
                    [-corner_step, y_offset, corner_step],
                    [-corner_step, y_offset, -corner_step],
                ]
            )
        if _contains_offset(negative_offsets, y_offset):
            rows.append([0.0, y_offset, 0.0])

    for x_offset in (*positive_offsets, *negative_offsets):
        rows.append([x_offset, 0.0, 0.0])

    # Minimize negative Z moves to reduce backlash effects, then sort by increasing
    # distance from origin.
    rows.sort(key=lambda row: (0 if row[2] >= 0.0 else 1, row[2]))
    rows.append([0.0, 0.0, 0.0])
    return np.asarray(rows, dtype=np.float64)


def _contains_offset(offsets: Sequence[float], value: float) -> bool:
    return any(np.isclose(offset, value) for offset in offsets)


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

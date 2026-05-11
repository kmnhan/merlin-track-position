import logging
import time
from collections.abc import Callable

import numpy as np
import xarray as xr

from merlin_track_position.instruments.cameras import capture_camera_pair
from merlin_track_position.instruments.motors import get_positions, move_motors_and_wait
from merlin_track_position.tracking.calibration_core import fit_calibration_from_images

logger = logging.getLogger("merlin_track_position.tracking.calibrate")


def _make_calibration_path(
    n: int,
    step_um: float,
) -> np.ndarray:
    """Build a 3D calibration path as rows of [x_um, y_um, z_um]."""
    if n < 2:
        raise ValueError("n must be >= 2")

    offsets = (np.arange(n, dtype=float) - (n - 1) / 2) * step_um

    rows: list[list[float]] = []
    for axis in range(3):
        for offset in offsets:
            if np.isclose(offset, 0.0):
                continue
            row = [0.0, 0.0, 0.0]
            row[axis] = float(offset)
            rows.append(row)

    corner_step = float(step_um)
    for dx in (-corner_step, corner_step):
        for dy in (-corner_step, corner_step):
            for dz in (-corner_step, corner_step):
                rows.append([dx, dy, dz])

    rows.append([0.0, 0.0, 0.0])
    return np.asarray(rows, dtype=float)


def calibration_sample_count(n: int) -> int:
    """Return the number of image pairs captured by ``run_calibration``."""
    return int(_make_calibration_path(n, 1.0).shape[0] + 1)


def run_calibration(
    n: int,
    step_um: float,
    image_generator: Callable[[], tuple[np.ndarray, np.ndarray]] = capture_camera_pair,
    *,
    origin_stability_um: float = 1.0,
    home_tolerance_um: float = 1.0,
    step_callback: Callable[
        [int, float, float, float, np.ndarray, np.ndarray],
        None,
    ]
    | None = None,
) -> xr.Dataset:
    """Run the calibration routine.

    Parameters
    ----------
    n : int
        Number of offset levels per translation axis. The routine skips the zero
        offset, then adds the eight +/-step_um 3D corners and a final origin
        return.
    step_um : float
        Step size in microns between adjacent grid points.
    image_generator : Callable[[], tuple[np.ndarray, np.ndarray]]
        Function that returns the current cam0 and cam1 images when called.
    origin_stability_um : float
        Warning threshold in microns for the final origin-return motor and image
        closure checks.
    home_tolerance_um : float, optional
        Tolerance in microns for returning to the home position at the end of the
        routine. Default is 1 micron.
    step_callback : Callable[[int, float, float, float, np.ndarray, np.ndarray], None] | None, optional
        Optional callback function that will be called after each move to a grid point,
        with the following arguments:

        - The index of the current step (0-based)
        - The x offset in microns for this step
        - The y offset in microns for this step
        - The z offset in microns for this step
        - The cam0 image captured at this step as a 2D numpy array
        - The cam1 image captured at this step as a 2D numpy array
    """
    x0, y0, z0, polar, cam = get_positions(("x", "y", "z", "p", "cam"))

    if not np.isclose(cam, 5.0):
        # Wait for camera to change to #5, which is the position for the sample view.
        move_motors_and_wait(("cam",), (5,))
        time.sleep(4.0)  # check if we can reduce this?

    goal_path_um = _make_calibration_path(n, step_um)

    actual_grid_um = np.empty((goal_path_um.shape[0] + 1, 3), dtype=float)
    images_cam0 = []
    images_cam1 = []

    def _update_step(idx, dx, dy, dz):
        actual_grid_um[idx, :] = [dx, dy, dz]
        image_cam0, image_cam1 = image_generator()
        images_cam0.append(image_cam0)
        images_cam1.append(image_cam1)
        if step_callback is not None:
            step_callback(idx, dx, dy, dz, image_cam0, image_cam1)

    _update_step(0, 0.0, 0.0, 0.0)
    actual_grid_um[0, :] = [0.0, 0.0, 0.0]

    logger.info("Starting calibration routine with n=%d, step_um=%.2f", n, step_um)

    for i, (dx, dy, dz) in enumerate(goal_path_um * 1e-3):
        logger.info(
            "Moving to grid point %d: dx=%.4f mm, dy=%.4f mm, dz=%.4f mm",
            i + 1,
            dx * 1e3,
            dy * 1e3,
            dz * 1e3,
        )
        x_goal, y_goal, z_goal = x0 + dx, y0 + dy, z0 + dz

        if i == goal_path_um.shape[0] - 1:
            # This is the last point, which is the home position. Use tolerance.
            logger.info(
                "Finished moving through grid points, returning to home position"
            )
            x_real, y_real, z_real = move_motors_and_wait(
                ("x", "y", "z"),
                (x0, y0, z0),
                tolerance=home_tolerance_um * 1e-3,
            )
        else:
            logger.info(
                "Commanding move to (%.4f, %.4f, %.4f) mm",
                x_goal,
                y_goal,
                z_goal,
            )
            x_real, y_real, z_real = move_motors_and_wait(
                ("x", "y", "z"),
                (x_goal, y_goal, z_goal),
            )

        logger.info(
            "Actual position is (%.4f, %.4f, %.4f) mm",
            x_real,
            y_real,
            z_real,
        )
        time.sleep(0.5)  # wait for image to update

        _update_step(
            i + 1,
            (x_real - x0) * 1000,
            (y_real - y0) * 1000,
            (z_real - z0) * 1000,
        )

    return fit_calibration_from_images(
        images_cam0=images_cam0,
        images_cam1=images_cam1,
        stage_um=actual_grid_um,
        origin_stability_um=origin_stability_um,
        check_tiles=True,
        additional_context={"polar": polar},
    )

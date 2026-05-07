import time
import logging
import numpy as np
import xarray as xr

from merlin_track_position.instruments.framegrab import get_framegrabber_image
from merlin_track_position.instruments.motors import get_positions, move_motors_and_wait
from merlin_track_position.tracking.calibration import fit_calibration_from_images

logger = logging.getLogger("merlin_track_position.tracking.calibrate")


def _make_grid(
    n: int,
    step_um: float,
    *,
    serpentine: bool = True,
) -> np.ndarray:
    """Build a centered n-by-n grid as rows of [stage_a_um, stage_b_um].

    For n=3 and step=10: offsets are [-10, 0, 10]
    For n=4 and step=10: offsets are [-15, -5, 5, 15]
    For n=5 and step=10: offsets are [-20, -10, 0, 10, 20]
    """
    if n < 2:
        raise ValueError("n must be >= 2")

    offsets = (np.arange(n, dtype=float) - (n - 1) / 2) * step_um

    rows = []
    for row_index, stage_b in enumerate(offsets):
        stage_a_values = offsets[::-1] if serpentine and row_index % 2 else offsets

        for stage_a in stage_a_values:
            rows.append([stage_a, stage_b])

    grid = np.array(rows, dtype=float)

    return grid


def run_calibration(
    n: int, step_um: float, *, home_tolerance_um: float = 5.0
) -> xr.Dataset:
    """Run the calibration routine.

    Parameters
    ----------
    n : int
        Number of points along each axis of the grid (total points will be n^2).
    step_um : float
        Step size in microns between adjacent grid points.
    home_tolerance_um : float, optional
        Tolerance in microns for returning to the home position at the end of the
        routine. Default is 10 microns.
    """
    x0, y0, cam = get_positions(("x", "y", "cam"))

    if not np.isclose(cam, 5.0):
        # Wait for camera to change to #5, which is the position for the sample view.
        move_motors_and_wait(("cam",), (5,))
        time.sleep(4.0)  # check if we can reduce this?

    goal_grid_um = _make_grid(n, step_um)

    actual_grid_um = np.empty((goal_grid_um.shape[0] + 2, 2), dtype=float)
    actual_grid_um[0, :] = [0.0, 0.0]
    images = [get_framegrabber_image()]

    logger.info("Starting calibration routine with n=%d, step_um=%.2f", n, step_um)

    for i, (dx, dy) in enumerate(goal_grid_um * 1e-3):
        logger.info(
            "Moving to grid point %d: dx=%.4f mm, dy=%.4f mm", i + 1, dx * 1e3, dy * 1e3
        )
        x_goal, y_goal = x0 + dx, y0 + dy
        logger.info("Commanding move to (%.4f, %.4f) mm", x_goal, y_goal)
        x_real, y_real = move_motors_and_wait(("x", "y"), (x_goal, y_goal))
        logger.info("Actual position is (%.4f, %.4f) mm", x_real, y_real)
        time.sleep(0.5)  # wait for image to update

        actual_grid_um[i + 1, :] = [(x_real - x0) * 1000, (y_real - y0) * 1000]
        images.append(get_framegrabber_image())

    logger.info("Finished moving through grid points, returning to home position")
    move_motors_and_wait(("x", "y"), (x0, y0), tolerance=home_tolerance_um * 1e-3)
    time.sleep(0.5)  # wait for image to update

    actual_grid_um[-1, :] = [0.0, 0.0]
    images.append(get_framegrabber_image())

    return fit_calibration_from_images(
        images=images, stage_um=actual_grid_um, check_tiles=True
    )

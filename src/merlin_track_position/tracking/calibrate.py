import logging
import time
from collections.abc import Callable

import numpy as np
import xarray as xr

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
    n: int,
    step_um: float,
    image_generator: Callable[[], np.ndarray],
    *,
    home_tolerance_um: float = 5.0,
    step_callback: Callable[[int, float, float, np.ndarray], None] | None = None,
) -> xr.Dataset:
    """Run the calibration routine.

    Parameters
    ----------
    n : int
        Number of points along each axis of the grid (total points will be n^2).
    step_um : float
        Step size in microns between adjacent grid points.
    image_generator : Callable[[], np.ndarray]
        Function that returns the current image as a 2D numpy array when called. This is
        typically a wrapper around get_framegrabber_image() that may include additional
        processing if needed.
    home_tolerance_um : float, optional
        Tolerance in microns for returning to the home position at the end of the
        routine. Default is 10 microns.
    step_callback : Callable[[int, float, float, np.ndarray], None] | None, optional
        Optional callback function that will be called after each move to a grid point,
        with the following arguments:

        - The index of the current step (0-based)
        - The x offset in microns for this step
        - The y offset in microns for this step
        - The image captured at this step as a 2D numpy array
    """
    x0, y0, polar, cam = get_positions(("x", "y", "p", "cam"))

    if not np.isclose(cam, 5.0):
        # Wait for camera to change to #5, which is the position for the sample view.
        move_motors_and_wait(("cam",), (5,))
        time.sleep(4.0)  # check if we can reduce this?

    goal_grid_um = _make_grid(n, step_um)
    goal_grid_um = np.vstack([goal_grid_um, [0.0, 0.0]])

    actual_grid_um = np.empty((goal_grid_um.shape[0] + 1, 2), dtype=float)
    images = []

    def _update_step(idx, dx, dy):
        actual_grid_um[idx, :] = [dx, dy]
        image = image_generator()
        images.append(image)
        if step_callback is not None:
            step_callback(idx, dx, dy, image)

    _update_step(0, 0.0, 0.0)
    actual_grid_um[0, :] = [0.0, 0.0]

    logger.info("Starting calibration routine with n=%d, step_um=%.2f", n, step_um)

    for i, (dx, dy) in enumerate(goal_grid_um * 1e-3):
        logger.info(
            "Moving to grid point %d: dx=%.4f mm, dy=%.4f mm", i + 1, dx * 1e3, dy * 1e3
        )
        x_goal, y_goal = x0 + dx, y0 + dy

        if i == goal_grid_um.shape[0] - 1:
            # This is the last point, which is the home position. Use tolerance.
            logger.info(
                "Finished moving through grid points, returning to home position"
            )
            x_real, y_real = move_motors_and_wait(
                ("x", "y"), (x0, y0), tolerance=home_tolerance_um * 1e-3
            )
        else:
            logger.info("Commanding move to (%.4f, %.4f) mm", x_goal, y_goal)
            x_real, y_real = move_motors_and_wait(("x", "y"), (x_goal, y_goal))

        logger.info("Actual position is (%.4f, %.4f) mm", x_real, y_real)
        time.sleep(0.5)  # wait for image to update

        _update_step(i + 1, (x_real - x0) * 1000, (y_real - y0) * 1000)

    return fit_calibration_from_images(
        images=images,
        stage_um=actual_grid_um,
        check_tiles=True,
        additional_context={"polar": polar},
    )

import contextlib
import logging
import time
from collections.abc import Iterable

import numpy as np

import merlin_track_position.instruments.BCSz as BCSz
from merlin_track_position import constants
from merlin_track_position.instruments.simulated_hardware import simulator

logger = logging.getLogger("merlin_track_position.instruments.motors")


@contextlib.contextmanager
def _bcs_server_context():
    server = BCSz.BCSServer()
    server.connect(addr=constants.BCS_SERVER_HOST, port=constants.BCS_SERVER_PORT)
    try:
        yield server
    finally:
        server._zmq_socket.close()


def _get_motor_info(
    bcs_server: BCSz.BCSServer, motor_aliases: Iterable[str], keys: Iterable[str]
) -> tuple[tuple[float, ...], ...]:
    info_dict = bcs_server.get_motor(
        motors=[constants.MOTOR_NAMES[m] for m in motor_aliases]
    )
    return tuple(tuple(m[k] for m in info_dict["data"]) for k in keys)


def _get_positions(
    bcs_server: BCSz.BCSServer, motor_aliases: Iterable[str]
) -> tuple[float, ...]:
    return _get_motor_info(bcs_server, motor_aliases, ("position",))[0]


def _wait_until_move_complete(
    bcs_server: BCSz.BCSServer, motor_aliases: Iterable[str]
) -> tuple[float, ...]:
    while True:
        positions, status = _get_motor_info(
            bcs_server, motor_aliases, ("position", "status")
        )
        if all(
            BCSz.MotorStatus(s).is_set(BCSz.MotorStatus.MOVE_COMPLETE) for s in status
        ):
            time.sleep(0.2)

            # Get the final positions one more time just to be extra careful
            positions, _ = _get_motor_info(
                bcs_server, motor_aliases, ("position", "status")
            )
            return positions
        else:
            time.sleep(0.2)  # don't hit the api server constantly


def _move_motors_and_wait(
    bcs_server: BCSz.BCSServer,
    motor_aliases: Iterable[str],
    goals: Iterable[float],
    *,
    tolerance: float | Iterable[float] | None = None,
    max_retries: int = 4,
) -> tuple[float, ...]:
    logger.debug(
        "Requesting move: motor_aliases=%s, goals=%s, tolerance=%s, max_retries=%d",
        motor_aliases,
        goals,
        tolerance,
        max_retries,
    )

    motor_aliases = tuple(motor_aliases)
    goals = tuple(float(goal) for goal in goals)

    if len(goals) != len(motor_aliases):
        logger.error(
            "Move failed: length of goals does not match length of motor_aliases."
        )
        return get_positions(motor_aliases)
    if max_retries < 0:
        logger.error("Move failed: max_retries must be non-negative.")
        return get_positions(motor_aliases)

    if tolerance is None:
        tolerances = None
    elif np.isscalar(tolerance):
        tolerances = (float(tolerance),) * len(motor_aliases)
    else:
        tolerances = tuple(float(t) for t in tolerance)

    motor_names = [constants.MOTOR_NAMES[m] for m in motor_aliases]

    for _ in range(max_retries + 1):
        bcs_server.move_motor(motors=motor_names, goals=list(goals))
        # wait just a bit to let the move begin.
        time.sleep(0.2)
        final_positions = _wait_until_move_complete(bcs_server, motor_aliases)

        logger.debug(
            "Move attempt: motor_aliases=%s, goals=%s, final_positions=%s",
            motor_aliases,
            goals,
            final_positions,
        )

        if tolerances is None or all(
            abs(position - goal) <= tolerance
            for position, goal, tolerance in zip(
                final_positions, goals, tolerances, strict=True
            )
        ):
            return final_positions

    position_errors = tuple(
        position - goal for position, goal in zip(final_positions, goals, strict=True)
    )
    logger.error(
        "Move failed: final positions are outside tolerance after %d retries. "
        "motor_aliases=%s, goals=%s, final_positions=%s, "
        "position_errors=%s, tolerances=%s",
        max_retries,
        motor_aliases,
        goals,
        final_positions,
        position_errors,
        tolerances,
    )
    return final_positions


def get_positions(motor_aliases: Iterable[str]) -> tuple[float, ...]:
    """Get current positions of the specified motor aliases."""
    if not constants.IS_DAQ_PC:
        return simulator.get_positions(motor_aliases)

    with _bcs_server_context() as server:
        return _get_positions(server, motor_aliases)


def get_temperatures() -> tuple[float, float, float, float]:
    """Get current temperatures of the cryostat temp sensors."""
    if not constants.IS_DAQ_PC:
        return simulator.get_temperatures()

    return get_positions(("TA", "TB", "TC", "TD"))


def move_motors_and_wait(
    motor_aliases: Iterable[str],
    goals: Iterable[float],
    *,
    tolerance: float | Iterable[float] | None = None,
    max_retries: int = 4,
) -> tuple[float, ...]:
    """Move the specified motor aliases and wait until move is complete.

    Parameters
    ----------
    motor_aliases
        Iterable of motor aliases to move, e.g. ("x", "y").
    goals
        Iterable of goal positions corresponding to motor_aliases, e.g. (1.0, 2.0).
    tolerance
        Optional tolerance(s) for verifying that the final positions are within
        tolerance of the goals. If a single float is provided, it will be used for all
        motors. If an iterable of floats is provided, it must have the same length as
        motor_aliases and specify a tolerance for each motor.
    max_retries
        Maximum number of times to retry the move if the final positions are not within
        tolerance of the goals. Default is 4.

    Raises
    ------
    RuntimeError
        If tolerance is provided and the final positions are outside tolerance after
        max_retries.

    """
    if not constants.IS_DAQ_PC:
        return simulator.move_motors_and_wait(
            motor_aliases,
            goals,
            tolerance=tolerance,
            max_retries=max_retries,
        )

    with _bcs_server_context() as server:
        return _move_motors_and_wait(
            server,
            motor_aliases,
            goals,
            tolerance=tolerance,
            max_retries=max_retries,
        )

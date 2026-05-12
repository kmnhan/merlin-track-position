import contextlib
import logging
import time
from collections.abc import Iterable, Mapping

import numpy as np

import merlin_track_position.instruments.BCSz as BCSz
from merlin_track_position import constants
from merlin_track_position.instruments.simulated_hardware import simulator

logger = logging.getLogger("merlin_track_position.instruments.motors")
DEFAULT_MOVE_TIMEOUT_S = 60.0


@contextlib.contextmanager
def _bcs_server_context():
    server = BCSz.BCSServer()
    server.connect(addr=constants.BCS_SERVER_HOST, port=constants.BCS_SERVER_PORT)
    _configure_bcs_socket_timeouts(server)
    try:
        yield server
    finally:
        server._zmq_socket.close()


def _configure_bcs_socket_timeouts(server: BCSz.BCSServer) -> None:
    timeout_ms = int(constants.BCS_REQUEST_TIMEOUT_MS)
    if timeout_ms < 0:
        raise ValueError("BCS_REQUEST_TIMEOUT_MS must be non-negative")
    server._zmq_socket.setsockopt(BCSz.zmq.RCVTIMEO, timeout_ms)
    server._zmq_socket.setsockopt(BCSz.zmq.SNDTIMEO, timeout_ms)
    server._zmq_socket.setsockopt(BCSz.zmq.LINGER, 0)


def _get_motor_info(
    bcs_server: BCSz.BCSServer, motor_aliases: Iterable[str], keys: Iterable[str]
) -> tuple[tuple[float, ...], ...]:
    aliases = tuple(motor_aliases)
    try:
        info_dict = bcs_server.get_motor(
            motors=[constants.MOTOR_NAMES[m] for m in aliases]
        )
    except BCSz.zmq.Again as exc:
        raise TimeoutError(
            "Timed out waiting for BCS GetMotor response: "
            f"motor_aliases={aliases}, timeout_ms={constants.BCS_REQUEST_TIMEOUT_MS}"
        ) from exc
    return tuple(tuple(m[k] for m in info_dict["data"]) for k in keys)


def _get_positions(
    bcs_server: BCSz.BCSServer, motor_aliases: Iterable[str]
) -> tuple[float, ...]:
    return _get_motor_info(bcs_server, motor_aliases, ("position",))[0]


def _wait_until_move_complete(
    bcs_server: BCSz.BCSServer,
    motor_aliases: Iterable[str],
    goals: Iterable[float],
    *,
    timeout_s: float = DEFAULT_MOVE_TIMEOUT_S,
) -> tuple[float, ...]:
    motor_aliases = tuple(motor_aliases)
    goals = tuple(float(goal) for goal in goals)
    timeout_s = _validate_move_timeout(timeout_s)
    stale_readback_deadbands = _stale_readback_deadbands(motor_aliases)
    stale_readback_delay_s = _stale_readback_delay_s()
    started_at = time.monotonic()
    next_log_elapsed_s = 5.0
    logger.info(
        "Waiting for motor move completion: motor_aliases=%s, goals=%s, "
        "stale_readback_deadbands=%s, stale_readback_delay_s=%g, timeout_s=%g",
        motor_aliases,
        goals,
        stale_readback_deadbands,
        stale_readback_delay_s,
        timeout_s,
    )
    while True:
        positions, status = _get_motor_info(
            bcs_server, motor_aliases, ("position", "status")
        )
        elapsed_s = time.monotonic() - started_at
        if all(
            BCSz.MotorStatus(s).is_set(BCSz.MotorStatus.MOVE_COMPLETE) for s in status
        ):
            time.sleep(0.2)

            # Get the final positions one more time just to be extra careful
            positions, _ = _get_motor_info(
                bcs_server, motor_aliases, ("position", "status")
            )
            logger.info(
                "Motor move complete: motor_aliases=%s, positions=%s, elapsed_s=%.1f",
                motor_aliases,
                positions,
                time.monotonic() - started_at,
            )
            return positions
        if elapsed_s >= stale_readback_delay_s and _positions_within_deadband(
            positions,
            goals,
            stale_readback_deadbands,
        ):
            logger.info(
                "Motor move accepted by stale-status position readback: "
                "motor_aliases=%s, positions=%s, goals=%s, "
                "stale_readback_deadbands=%s, status=%s, elapsed_s=%.1f",
                motor_aliases,
                positions,
                goals,
                stale_readback_deadbands,
                status,
                elapsed_s,
            )
            return positions
        if elapsed_s >= timeout_s:
            position_errors = tuple(
                position - goal
                for position, goal in zip(positions, goals, strict=True)
            )
            raise TimeoutError(
                "Timed out waiting for motor move completion: "
                f"motor_aliases={motor_aliases}, goals={goals}, "
                f"positions={positions}, position_errors={position_errors}, "
                f"stale_readback_deadbands={stale_readback_deadbands}, "
                f"status={status}, elapsed_s={elapsed_s:.1f}"
            )
        if elapsed_s >= next_log_elapsed_s:
            logger.info(
                "Still waiting for motor move completion: motor_aliases=%s, "
                "positions=%s, goals=%s, stale_readback_deadbands=%s, "
                "status=%s, elapsed_s=%.1f",
                motor_aliases,
                positions,
                goals,
                stale_readback_deadbands,
                status,
                elapsed_s,
            )
            next_log_elapsed_s = elapsed_s + 5.0
        time.sleep(0.2)  # don't hit the api server constantly


def _move_motors_and_wait(
    bcs_server: BCSz.BCSServer,
    motor_aliases: Iterable[str],
    goals: Iterable[float],
    *,
    max_retries: int = 4,
    move_timeout_s: float = DEFAULT_MOVE_TIMEOUT_S,
    backlash_correction: Mapping[str, float] | None = None,
) -> tuple[float, ...]:
    motor_aliases = tuple(motor_aliases)
    goals = tuple(float(goal) for goal in goals)
    logger.debug(
        "Requesting move: motor_aliases=%s, goals=%s, max_retries=%d, "
        "backlash_correction=%s",
        motor_aliases,
        goals,
        max_retries,
        backlash_correction,
    )

    if len(goals) != len(motor_aliases):
        logger.error(
            "Move failed: length of goals does not match length of motor_aliases."
        )
        return get_positions(motor_aliases)
    if max_retries < 0:
        logger.error("Move failed: max_retries must be non-negative.")
        return get_positions(motor_aliases)
    move_timeout_s = _validate_move_timeout(move_timeout_s)

    backlash_corrections = _backlash_corrections(
        motor_aliases,
        backlash_correction,
    )
    current_positions = _get_positions(bcs_server, motor_aliases)

    pre_indices = tuple(
        index
        for index in range(len(motor_aliases))
        if backlash_corrections[index] > 0.0 and goals[index] < current_positions[index]
    )
    if pre_indices:
        pre_aliases = _items_at(motor_aliases, pre_indices)
        pre_goals = _backlash_pre_goals(
            _items_at(current_positions, pre_indices),
            _items_at(goals, pre_indices),
            _items_at(backlash_corrections, pre_indices),
        )
        logger.debug(
            "Backlash pre-position: motor_aliases=%s, current_positions=%s, "
            "pre_goals=%s, final_goals=%s",
            pre_aliases,
            _items_at(current_positions, pre_indices),
            pre_goals,
            _items_at(goals, pre_indices),
        )
        bcs_server.move_motor(
            motors=[constants.MOTOR_NAMES[alias] for alias in pre_aliases],
            goals=list(pre_goals),
        )
        time.sleep(0.2)
        _wait_until_move_complete(
            bcs_server,
            pre_aliases,
            pre_goals,
            timeout_s=move_timeout_s,
        )

    bcs_server.move_motor(
        motors=[constants.MOTOR_NAMES[alias] for alias in motor_aliases],
        goals=list(goals),
    )
    # wait just a bit to let the move begin.
    time.sleep(0.2)
    _wait_until_move_complete(
        bcs_server,
        motor_aliases,
        goals,
        timeout_s=move_timeout_s,
    )
    final_positions = _get_positions(bcs_server, motor_aliases)
    logger.debug(
        "Move attempt: motor_aliases=%s, goals=%s, final_positions=%s",
        motor_aliases,
        goals,
        final_positions,
    )
    return final_positions


def _validate_move_timeout(timeout_s: float) -> float:
    timeout = float(timeout_s)
    if not np.isfinite(timeout) or timeout < 0.0:
        raise ValueError("move_timeout_s must be finite and non-negative")
    return timeout


def _stale_readback_deadbands(motor_aliases: Iterable[str]) -> tuple[float, ...]:
    aliases = tuple(motor_aliases)
    configured_deadbands = constants.MOTOR_STALE_READBACK_DEADBAND
    if isinstance(configured_deadbands, Mapping):
        deadbands = []
        for alias in aliases:
            try:
                deadband_value = configured_deadbands[alias]
            except KeyError as exc:
                raise ValueError(
                    "MOTOR_STALE_READBACK_DEADBAND missing value for "
                    f"motor alias {alias!r}"
                ) from exc
            deadbands.append(_validate_stale_readback_deadband(deadband_value))
        return tuple(deadbands)

    deadband = _validate_stale_readback_deadband(configured_deadbands)
    return (deadband,) * len(aliases)


def _validate_stale_readback_deadband(deadband_value: float) -> float:
    deadband = float(deadband_value)
    if not np.isfinite(deadband) or deadband < 0.0:
        raise ValueError(
            "MOTOR_STALE_READBACK_DEADBAND values must be finite and non-negative"
        )
    return deadband


def _stale_readback_delay_s() -> float:
    delay_s = float(constants.MOTOR_STALE_READBACK_DELAY_S)
    if not np.isfinite(delay_s) or delay_s < 0.0:
        raise ValueError("MOTOR_STALE_READBACK_DELAY_S must be finite and non-negative")
    return delay_s


def _positions_within_deadband(
    positions: Iterable[float],
    goals: Iterable[float],
    deadbands: Iterable[float],
) -> bool:
    return all(
        abs(float(position) - float(goal)) <= float(deadband)
        for position, goal, deadband in zip(positions, goals, deadbands, strict=True)
    )


def _items_at(values: Iterable, indices: Iterable[int]) -> tuple:
    values_tuple = tuple(values)
    return tuple(values_tuple[index] for index in indices)


def _backlash_corrections(
    motor_aliases: Iterable[str],
    backlash_correction: Mapping[str, float] | None,
) -> tuple[float, ...]:
    if backlash_correction is None:
        return (0.0,) * len(tuple(motor_aliases))

    corrections = []
    for alias in motor_aliases:
        correction = float(backlash_correction.get(alias, 0.0))
        if not np.isfinite(correction) or correction < 0.0:
            raise ValueError(
                "backlash correction values must be finite and non-negative"
            )
        corrections.append(correction)
    return tuple(corrections)


def _backlash_pre_goals(
    current_positions: Iterable[float],
    goals: Iterable[float],
    backlash_corrections: Iterable[float],
) -> tuple[float, ...]:
    return tuple(
        min(float(current), float(goal)) - float(correction)
        if correction > 0.0
        else float(goal)
        for current, goal, correction in zip(
            current_positions,
            goals,
            backlash_corrections,
            strict=True,
        )
    )


def get_positions(motor_aliases: Iterable[str]) -> tuple[float, ...]:
    """Get current positions of the specified motor aliases."""
    motor_aliases = tuple(motor_aliases)
    logger.info("Reading motor positions: motor_aliases=%s", motor_aliases)
    if not constants.IS_DAQ_PC:
        positions = simulator.get_positions(motor_aliases)
        logger.info("Read simulated motor positions: positions=%s", positions)
        return positions

    with _bcs_server_context() as server:
        positions = _get_positions(server, motor_aliases)
        logger.info("Read motor positions: positions=%s", positions)
        return positions


def get_temperatures() -> tuple[float, float, float, float]:
    """Get current temperatures of the cryostat temp sensors."""
    if not constants.IS_DAQ_PC:
        return simulator.get_temperatures()

    return get_positions(("TA", "TB", "TC", "TD"))


def move_motors_and_wait(
    motor_aliases: Iterable[str],
    goals: Iterable[float],
    *,
    max_retries: int = 4,
    move_timeout_s: float = DEFAULT_MOVE_TIMEOUT_S,
    backlash_correction: Mapping[str, float] | None = None,
) -> tuple[float, ...]:
    """Move the specified motor aliases and wait until move is complete.

    Parameters
    ----------
    motor_aliases
        Iterable of motor aliases to move, e.g. ("x", "y").
    goals
        Iterable of goal positions corresponding to motor_aliases, e.g. (1.0, 2.0).
    max_retries
        Retained for API compatibility; motor completion is controlled by controller
        status, stale-status readback fallback, and ``move_timeout_s``.
    move_timeout_s
        Maximum time to wait for each move phase to either report complete or, after
        the stale-readback delay, reach its requested readback position. Default is
        60 seconds.
    backlash_correction
        Mapping from motor alias to backlash take-up distance in that motor's command
        units. For x and z this is millimeters. When omitted,
        :data:`merlin_track_position.constants.MOTOR_BACKLASH_CORRECTION` is used.
        Pass an empty mapping to disable backlash correction for a move.
    """
    motor_aliases = tuple(motor_aliases)
    goals = tuple(float(goal) for goal in goals)
    move_timeout_s = _validate_move_timeout(move_timeout_s)
    logger.info(
        "Moving motors and waiting: motor_aliases=%s, goals=%s, max_retries=%d, "
        "move_timeout_s=%g",
        motor_aliases,
        goals,
        max_retries,
        move_timeout_s,
    )
    if not constants.IS_DAQ_PC:
        positions = simulator.move_motors_and_wait(
            motor_aliases,
            goals,
            max_retries=max_retries,
        )
        logger.info("Simulated motor move returned: positions=%s", positions)
        return positions

    if backlash_correction is None:
        backlash_correction = constants.MOTOR_BACKLASH_CORRECTION

    with _bcs_server_context() as server:
        positions = _move_motors_and_wait(
            server,
            motor_aliases,
            goals,
            max_retries=max_retries,
            move_timeout_s=move_timeout_s,
            backlash_correction=backlash_correction,
        )
        logger.info("Motor move returned: positions=%s", positions)
        return positions

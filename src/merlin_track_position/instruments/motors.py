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


class _MotorMoveDidNotStartError(TimeoutError):
    pass


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
) -> tuple[tuple[object, ...], ...]:
    aliases = tuple(motor_aliases)
    bcs_motor_names = _bcs_motor_names(aliases)
    keys = tuple(keys)
    try:
        info_dict = bcs_server.get_motor(motors=list(bcs_motor_names))
    except BCSz.zmq.Again as exc:
        raise TimeoutError(
            "Timed out waiting for BCS GetMotor response: "
            f"motor_aliases={aliases}, timeout_ms={constants.BCS_REQUEST_TIMEOUT_MS}"
        ) from exc
    info_dict = _validate_bcs_common_response(
        "GetMotor",
        info_dict,
        motor_aliases=aliases,
        bcs_motor_names=bcs_motor_names,
    )
    motor_data = _validated_get_motor_data(
        info_dict,
        aliases,
        bcs_motor_names,
        keys,
    )
    return tuple(
        tuple(_validated_motor_field(motor, key) for motor in motor_data)
        for key in keys
    )


def _get_positions(
    bcs_server: BCSz.BCSServer, motor_aliases: Iterable[str]
) -> tuple[float, ...]:
    return tuple(
        float(position)
        for position in _get_motor_info(bcs_server, motor_aliases, ("position",))[0]
    )


def _wait_until_move_complete(
    bcs_server: BCSz.BCSServer,
    motor_aliases: Iterable[str],
    goals: Iterable[float],
    *,
    timeout_s: float = DEFAULT_MOVE_TIMEOUT_S,
    start_positions: Iterable[float] | None = None,
) -> tuple[float, ...]:
    motor_aliases = tuple(motor_aliases)
    goals = tuple(float(goal) for goal in goals)
    if start_positions is not None:
        start_positions = tuple(float(position) for position in start_positions)
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
        positions, status, goal_readbacks, readback_times = _get_motor_info(
            bcs_server,
            motor_aliases,
            ("position", "status", "goal", "time"),
        )
        positions = tuple(float(position) for position in positions)
        status = tuple(int(motor_status) for motor_status in status)
        goal_readbacks = tuple(float(goal) for goal in goal_readbacks)
        elapsed_s = time.monotonic() - started_at
        goal_latched = _positions_within_deadband(
            goal_readbacks,
            goals,
            stale_readback_deadbands,
        )
        position_at_goal = _positions_within_deadband(
            positions,
            goals,
            stale_readback_deadbands,
        )
        position_at_start = (
            start_positions is not None
            and _positions_within_deadband(
                positions,
                start_positions,
                stale_readback_deadbands,
            )
        )
        start_at_goal = (
            start_positions is not None
            and _positions_within_deadband(
                start_positions,
                goals,
                stale_readback_deadbands,
            )
        )
        if _motor_statuses_all_set(status, BCSz.MotorStatus.MOVE_COMPLETE):
            move_complete_is_current = goal_latched and (
                position_at_goal
                or start_positions is None
                or start_at_goal
                or not position_at_start
            )
            if move_complete_is_current:
                time.sleep(0.25)

                # Get the final positions one more time just to be extra careful
                positions = _get_positions(bcs_server, motor_aliases)
                logger.info(
                    "Motor move complete: motor_aliases=%s, positions=%s, "
                    "goal_readbacks=%s, elapsed_s=%.1f",
                    motor_aliases,
                    positions,
                    goal_readbacks,
                    time.monotonic() - started_at,
                )
                return positions
        if _motor_statuses_all_set(
            status,
            BCSz.MotorStatus.RAW_MOVE_COMPLETE,
        ) and position_at_goal:
            logger.info(
                "Motor move accepted by raw-complete position readback: "
                "motor_aliases=%s, positions=%s, goals=%s, "
                "goal_readbacks=%s, readback_times=%s, "
                "stale_readback_deadbands=%s, status=%s, elapsed_s=%.1f",
                motor_aliases,
                positions,
                goals,
                goal_readbacks,
                readback_times,
                stale_readback_deadbands,
                status,
                elapsed_s,
            )
            return positions
        if (
            start_positions is not None
            and elapsed_s >= stale_readback_delay_s
            and (
                _motor_statuses_all_set(status, BCSz.MotorStatus.RAW_MOVE_COMPLETE)
                or _motor_statuses_all_set(status, BCSz.MotorStatus.MOVE_COMPLETE)
            )
            and position_at_start
        ):
            stall_reason = (
                "latched_but_position_did_not_move"
                if goal_latched
                else "goal_did_not_latch"
            )
            raise _MotorMoveDidNotStartError(
                "Motor move did not start: "
                f"reason={stall_reason}, "
                f"motor_aliases={motor_aliases}, goals={goals}, "
                f"goal_readbacks={goal_readbacks}, start_positions={start_positions}, "
                f"positions={positions}, readback_times={readback_times}, "
                f"stale_readback_deadbands={stale_readback_deadbands}, "
                f"status={status}, elapsed_s={elapsed_s:.1f}"
            )
        if elapsed_s >= stale_readback_delay_s and position_at_goal:
            logger.info(
                "Motor move accepted by stale-status position readback: "
                "motor_aliases=%s, positions=%s, goals=%s, "
                "goal_readbacks=%s, readback_times=%s, "
                "stale_readback_deadbands=%s, status=%s, elapsed_s=%.1f",
                motor_aliases,
                positions,
                goals,
                goal_readbacks,
                readback_times,
                stale_readback_deadbands,
                status,
                elapsed_s,
            )
            return positions
        if elapsed_s >= timeout_s:
            position_errors = tuple(
                position - goal for position, goal in zip(positions, goals, strict=True)
            )
            raise TimeoutError(
                "Timed out waiting for motor move completion: "
                f"motor_aliases={motor_aliases}, goals={goals}, "
                f"goal_readbacks={goal_readbacks}, positions={positions}, "
                f"position_errors={position_errors}, readback_times={readback_times}, "
                f"stale_readback_deadbands={stale_readback_deadbands}, "
                f"status={status}, elapsed_s={elapsed_s:.1f}"
            )
        if elapsed_s >= next_log_elapsed_s:
            logger.info(
                "Still waiting for motor move completion: motor_aliases=%s, "
                "positions=%s, goals=%s, goal_readbacks=%s, readback_times=%s, "
                "goal_latched=%s, stale_readback_deadbands=%s, status=%s, "
                "elapsed_s=%.1f",
                motor_aliases,
                positions,
                goals,
                goal_readbacks,
                readback_times,
                goal_latched,
                stale_readback_deadbands,
                status,
                elapsed_s,
            )
            next_log_elapsed_s = elapsed_s + 5.0
        time.sleep(0.25)  # don't hit the api server constantly


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
        _move_motor_phase_and_wait(
            bcs_server,
            pre_aliases,
            pre_goals,
            phase="backlash_preposition",
            max_retries=max_retries,
            timeout_s=move_timeout_s,
            start_positions=_items_at(current_positions, pre_indices),
        )
        current_positions = _get_positions(bcs_server, motor_aliases)

    _move_motor_phase_and_wait(
        bcs_server,
        motor_aliases,
        goals,
        phase="final",
        max_retries=max_retries,
        timeout_s=move_timeout_s,
        start_positions=current_positions,
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


def _move_motor(
    bcs_server: BCSz.BCSServer,
    motor_aliases: Iterable[str],
    goals: Iterable[float],
) -> None:
    aliases = tuple(motor_aliases)
    goals = tuple(float(goal) for goal in goals)
    bcs_motor_names = _bcs_motor_names(aliases)
    try:
        response = bcs_server.move_motor(
            motors=list(bcs_motor_names),
            goals=list(goals),
        )
    except BCSz.zmq.Again as exc:
        raise TimeoutError(
            "Timed out waiting for BCS MoveMotor response: "
            f"motor_aliases={aliases}, bcs_motor_names={bcs_motor_names}, "
            f"goals={goals}, timeout_ms={constants.BCS_REQUEST_TIMEOUT_MS}"
        ) from exc
    response = _validate_bcs_common_response(
        "MoveMotor",
        response,
        motor_aliases=aliases,
        bcs_motor_names=bcs_motor_names,
        goals=goals,
    )
    timed_out = _response_field(response, "timed_out", "timed out")
    if _has_response_value(timed_out):
        raise RuntimeError(
            "BCS MoveMotor reported timed_out: "
            f"timed_out={timed_out!r}; "
            f"{_bcs_response_context(aliases, bcs_motor_names, goals, response)}"
        )


def _move_motor_phase_and_wait(
    bcs_server: BCSz.BCSServer,
    motor_aliases: Iterable[str],
    goals: Iterable[float],
    *,
    phase: str,
    max_retries: int,
    timeout_s: float,
    start_positions: Iterable[float] | None = None,
) -> tuple[float, ...]:
    aliases = tuple(motor_aliases)
    goals = tuple(float(goal) for goal in goals)
    if start_positions is not None:
        start_positions = tuple(float(position) for position in start_positions)

    for attempt_index in range(max_retries + 1):
        if start_positions is None or attempt_index > 0:
            start_positions = _get_positions(bcs_server, aliases)
        logger.info(
            "Starting motor move phase: phase=%s, attempt=%d/%d, "
            "motor_aliases=%s, goals=%s, start_positions=%s",
            phase,
            attempt_index + 1,
            max_retries + 1,
            aliases,
            goals,
            start_positions,
        )
        _move_motor(
            bcs_server,
            aliases,
            goals,
        )
        # wait just a bit to let the move begin.
        time.sleep(0.25)
        try:
            return _wait_until_move_complete(
                bcs_server,
                aliases,
                goals,
                timeout_s=timeout_s,
                start_positions=start_positions,
            )
        except _MotorMoveDidNotStartError as exc:
            if attempt_index >= max_retries:
                raise TimeoutError(
                    "Motor move did not start after retries: "
                    f"phase={phase}, attempts={max_retries + 1}, "
                    f"motor_aliases={aliases}, goals={goals}, "
                    f"start_positions={start_positions}, last_error={exc}"
                ) from exc
            logger.warning(
                "Motor move did not start; retrying: phase=%s, attempt=%d/%d, "
                "motor_aliases=%s, goals=%s, start_positions=%s, error=%s",
                phase,
                attempt_index + 1,
                max_retries + 1,
                aliases,
                goals,
                start_positions,
                exc,
            )

    raise AssertionError("unreachable move retry loop exit")


def _bcs_motor_names(motor_aliases: Iterable[str]) -> tuple[str, ...]:
    return tuple(constants.MOTOR_NAMES[alias] for alias in motor_aliases)


def _validate_bcs_common_response(
    command_name: str,
    response: object,
    *,
    motor_aliases: tuple[str, ...],
    bcs_motor_names: tuple[str, ...],
    goals: tuple[float, ...] | None = None,
) -> Mapping:
    if not isinstance(response, Mapping):
        raise RuntimeError(
            f"BCS {command_name} returned malformed response: "
            f"expected mapping, got {type(response).__name__}; "
            f"{_bcs_response_context(motor_aliases, bcs_motor_names, goals, None)}"
        )

    if not bool(response.get("success", True)):
        raise RuntimeError(
            f"BCS {command_name} failed: "
            f"{_bcs_error_description(response)}; "
            f"{_bcs_response_context(motor_aliases, bcs_motor_names, goals, response)}"
        )

    not_found = _response_field(response, "not_found", "not found")
    if _has_response_value(not_found):
        raise RuntimeError(
            f"BCS {command_name} reported not_found: "
            f"not_found={not_found!r}; "
            f"{_bcs_response_context(motor_aliases, bcs_motor_names, goals, response)}"
        )

    return response


def _validated_get_motor_data(
    response: Mapping,
    motor_aliases: tuple[str, ...],
    bcs_motor_names: tuple[str, ...],
    keys: tuple[str, ...],
) -> tuple[Mapping, ...]:
    if "data" not in response:
        raise RuntimeError(
            "BCS GetMotor returned malformed response: missing data; "
            f"{_bcs_response_context(motor_aliases, bcs_motor_names, None, response)}"
        )

    data = response["data"]
    if not isinstance(data, list | tuple):
        raise RuntimeError(
            "BCS GetMotor returned malformed response: data must be a sequence; "
            f"{_bcs_response_context(motor_aliases, bcs_motor_names, None, response)}"
        )
    if len(data) != len(motor_aliases):
        raise RuntimeError(
            "BCS GetMotor returned malformed response: "
            f"expected {len(motor_aliases)} data rows, got {len(data)}; "
            f"{_bcs_response_context(motor_aliases, bcs_motor_names, None, response)}"
        )

    motor_data = tuple(data)
    for index, motor in enumerate(motor_data):
        if not isinstance(motor, Mapping):
            raise RuntimeError(
                "BCS GetMotor returned malformed response: "
                f"data row {index} must be a mapping; "
                f"{_bcs_response_context(motor_aliases, bcs_motor_names, None, response)}"
            )
        for key in keys:
            if key not in motor:
                raise RuntimeError(
                    "BCS GetMotor returned malformed response: "
                    f"data row {index} missing key {key!r}; "
                    f"{_bcs_response_context(motor_aliases, bcs_motor_names, None, response)}"
                )
    return motor_data


def _validated_motor_field(motor: Mapping, key: str) -> object:
    value = motor[key]
    if key == "status":
        if isinstance(value, bool):
            raise RuntimeError("BCS GetMotor returned malformed status field")
        try:
            status = int(value)
        except (TypeError, ValueError) as exc:
            raise RuntimeError("BCS GetMotor returned malformed status field") from exc
        if status != value:
            raise RuntimeError("BCS GetMotor returned malformed status field")
        try:
            BCSz.MotorStatus(status)
        except ValueError as exc:
            raise RuntimeError("BCS GetMotor returned malformed status field") from exc
        return status

    if key in {"position", "goal"}:
        try:
            position = float(value)
        except (TypeError, ValueError) as exc:
            raise RuntimeError(
                f"BCS GetMotor returned malformed {key} field"
            ) from exc
        if not np.isfinite(position):
            raise RuntimeError(f"BCS GetMotor returned malformed {key} field")
        return position

    return value


def _response_field(response: Mapping, *keys: str) -> object:
    for key in keys:
        if key in response:
            return response[key]
    return None


def _has_response_value(value: object) -> bool:
    if value is None:
        return False
    if isinstance(value, (str, bytes, list, tuple, set, dict)):
        return len(value) > 0
    return bool(value)


def _bcs_error_description(response: Mapping) -> str:
    description = _response_field(response, "error_description", "error description")
    if description:
        return str(description)
    return "no BCS error description"


def _bcs_response_context(
    motor_aliases: tuple[str, ...],
    bcs_motor_names: tuple[str, ...],
    goals: tuple[float, ...] | None,
    response: Mapping | None,
) -> str:
    parts = [
        f"motor_aliases={motor_aliases}",
        f"bcs_motor_names={bcs_motor_names}",
    ]
    if goals is not None:
        parts.append(f"goals={goals}")
    if response is not None:
        parts.append(f"error={_bcs_error_description(response)!r}")
    return ", ".join(parts)


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


def _motor_statuses_all_set(
    statuses: Iterable[int],
    flag: BCSz.MotorStatus,
) -> bool:
    return all(BCSz.MotorStatus(status).is_set(flag) for status in statuses)


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
        Number of times to reissue a move phase when the controller reports raw
        completion but readback remains at the pre-move position outside the
        goal deadband.
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

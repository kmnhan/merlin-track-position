import contextlib
import time
from collections.abc import Iterable

import merlin_track_position.instruments.BCSz as BCSz
from merlin_track_position.constants import BCS_SERVER_HOST, BCS_SERVER_PORT

MOTOR_NAMES = {
    "x": "Sample X",
    "y": "Sample Y (Vert)",
    "z": "Sample Z",
    "cam": "Video Switch",
}  #: Mapping from shorthand motor names to actual motor names at the beamline.


@contextlib.contextmanager
def _bcs_server_context():
    server = BCSz.BCSServer()
    server.connect(addr=BCS_SERVER_HOST, port=BCS_SERVER_PORT)
    try:
        yield server
    finally:
        server._zmq_socket.close()


def _get_motor_info(
    bcs_server: BCSz.BCSServer, motor_aliases: Iterable[str], keys: Iterable[str]
) -> tuple[tuple[float, ...], ...]:
    info_dict = bcs_server.get_motor(motors=[MOTOR_NAMES[m] for m in motor_aliases])
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
            return positions
        else:
            time.sleep(0.25)  # don't hit the api server constantly


def _move_motors_and_wait(
    bcs_server: BCSz.BCSServer, motor_aliases: Iterable[str], goals: Iterable[float]
) -> tuple[float, ...]:
    bcs_server.move_motor(
        motors=[MOTOR_NAMES[m] for m in motor_aliases], goals=list(goals)
    )
    # wait just a bit to let the move begin.
    time.sleep(0.25)
    return _wait_until_move_complete(bcs_server, motor_aliases)


def get_positions(motor_aliases: Iterable[str]) -> tuple[float, ...]:
    """Get current positions of the specified motor aliases."""
    with _bcs_server_context() as server:
        return _get_positions(server, motor_aliases)


def move_motors_and_wait(
    motor_aliases: Iterable[str], goals: Iterable[float]
) -> tuple[float, ...]:
    """Move the specified motor aliases and wait until move is complete."""
    with _bcs_server_context() as server:
        return _move_motors_and_wait(server, motor_aliases, goals)

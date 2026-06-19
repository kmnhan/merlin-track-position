import contextlib
import threading
import unittest
from unittest.mock import call, patch

import merlin_track_position.instruments.BCSz as BCSz
from merlin_track_position import constants
from merlin_track_position.constants import MOTOR_NAMES
from merlin_track_position.instruments.motors import (
    _bcs_server_context,
    _clear_motor_position_cache,
    _move_motors_and_wait,
    cached_motor_positions,
    get_positions,
    get_temperatures,
    move_motors_and_wait,
    refresh_motor_positions,
    update_motor_position_cache,
)
from merlin_track_position.instruments.simulated_hardware import simulator


class FakeBCSServer:
    def __init__(
        self,
        final_positions_by_move,
        initial_positions=(),
        status=BCSz.MotorStatus.MOVE_COMPLETE.value,
        move_responses=(),
        get_motor_responses=(),
        goal_latched_by_move=(),
    ):
        self._final_positions_by_move = [
            tuple(positions) for positions in final_positions_by_move
        ]
        self._initial_positions = tuple(initial_positions)
        self._status = int(status)
        self._move_responses = list(move_responses)
        self._get_motor_responses = list(get_motor_responses)
        self._goal_latched_by_move = list(goal_latched_by_move)
        self._positions_by_motor = {}
        self._goals_by_motor = {}
        self._times_by_motor = {}
        self.move_calls = []

    def move_motor(self, *, motors, goals):
        self._ensure_positions(motors)
        self.move_calls.append((tuple(motors), tuple(goals)))
        response = self._next_response(
            self._move_responses,
            {"success": True, "not_found": [], "timed_out": []},
            motors,
        )
        if (
            not response.get("success", True)
            or response.get("not_found")
            or response.get("timed_out")
        ):
            return response

        move_index = min(
            len(self.move_calls) - 1, len(self._final_positions_by_move) - 1
        )
        latch_goal = (
            self._goal_latched_by_move[move_index]
            if move_index < len(self._goal_latched_by_move)
            else True
        )
        final_positions = self._final_positions_by_move[move_index]
        for motor, goal, position in zip(motors, goals, final_positions, strict=True):
            if latch_goal:
                self._goals_by_motor[motor] = float(goal)
            self._positions_by_motor[motor] = position
            self._times_by_motor[motor] = len(self.move_calls)
        return response

    def get_motor(self, *, motors):
        if self._get_motor_responses:
            return self._next_response(self._get_motor_responses, {}, motors)

        self._ensure_positions(motors)
        return {
            "success": True,
            "not_found": [],
            "data": [
                {
                    "position": self._positions_by_motor[motor],
                    "goal": self._goals_by_motor[motor],
                    "status": self._status,
                    "time": self._times_by_motor[motor],
                }
                for motor in motors
            ],
        }

    def _ensure_positions(self, motors):
        if self._positions_by_motor:
            for motor in motors:
                self._positions_by_motor.setdefault(motor, 0.0)
                self._goals_by_motor.setdefault(motor, self._positions_by_motor[motor])
                self._times_by_motor.setdefault(motor, 0)
            return

        if self._initial_positions:
            for motor, position in zip(motors, self._initial_positions, strict=True):
                self._positions_by_motor[motor] = position
        for motor in motors:
            self._positions_by_motor.setdefault(motor, 0.0)
            self._goals_by_motor.setdefault(motor, self._positions_by_motor[motor])
            self._times_by_motor.setdefault(motor, 0)

    def _next_response(self, responses, default, motors):
        if not responses:
            return default

        response = responses.pop(0)
        if callable(response):
            return response(motors)
        return response


class MotorConfigurationTests(unittest.TestCase):
    def test_all_configured_motor_aliases_have_stale_readback_deadbands(self):
        missing_aliases = sorted(
            set(MOTOR_NAMES) - set(constants.MOTOR_STALE_READBACK_DEADBAND)
        )

        self.assertEqual(missing_aliases, [])


def _get_motor_response(
    positions,
    status=BCSz.MotorStatus.MOVE_COMPLETE.value,
    goals=None,
    times=None,
):
    if isinstance(status, tuple):
        statuses = status
    else:
        statuses = (status,) * len(positions)
    if goals is None:
        goals = positions
    if times is None:
        times = (0,) * len(positions)
    return {
        "success": True,
        "not_found": [],
        "data": [
            {
                "position": position,
                "goal": goal,
                "status": motor_status,
                "time": readback_time,
            }
            for position, goal, motor_status, readback_time in zip(
                positions,
                goals,
                statuses,
                times,
                strict=True,
            )
        ],
    }


class MoveMotorsAndWaitTests(unittest.TestCase):
    def test_bcs_context_configures_request_timeouts(self):
        class FakeSocket:
            def __init__(self):
                self.options = {}
                self.closed = False

            def setsockopt(self, option, value):
                self.options[option] = value

            def close(self):
                self.closed = True

        class FakeServer:
            def __init__(self):
                self._zmq_socket = FakeSocket()
                created.append(self)

            def connect(self, *, addr, port):
                self.addr = addr
                self.port = port

        created = []
        with patch(
            "merlin_track_position.instruments.motors.BCSz.BCSServer", FakeServer
        ):
            with _bcs_server_context() as server:
                self.assertEqual(server.addr, constants.BCS_SERVER_HOST)
                self.assertEqual(server.port, constants.BCS_SERVER_PORT)
                self.assertEqual(
                    server._zmq_socket.options[BCSz.zmq.RCVTIMEO],
                    constants.BCS_REQUEST_TIMEOUT_MS,
                )
                self.assertEqual(
                    server._zmq_socket.options[BCSz.zmq.SNDTIMEO],
                    constants.BCS_REQUEST_TIMEOUT_MS,
                )
                self.assertEqual(server._zmq_socket.options[BCSz.zmq.LINGER], 0)

        self.assertEqual(len(created), 1)
        self.assertTrue(created[0]._zmq_socket.closed)

    def test_public_helper_forwards_retry_arguments(self):
        server = object()

        @contextlib.contextmanager
        def fake_bcs_server_context():
            yield server

        with (
            patch.object(constants, "IS_DAQ_PC", True),
            patch(
                "merlin_track_position.instruments.motors._bcs_server_context",
                fake_bcs_server_context,
            ),
            patch(
                "merlin_track_position.instruments.motors._move_motors_and_wait",
                return_value=(1.0,),
            ) as move,
        ):
            positions = move_motors_and_wait(("x",), (1.0,), max_retries=2)

        self.assertEqual(positions, (1.0,))
        move.assert_called_once_with(
            server,
            ("x",),
            (1.0,),
            max_retries=2,
            backlash_correction=constants.MOTOR_BACKLASH_CORRECTION,
            move_timeout_s=60.0,
        )

    def test_move_complete_status_is_trusted_without_readback_validation(self):
        server = FakeBCSServer([(9.5, 2.5), (10.0, 2.0)])

        with patch("merlin_track_position.instruments.motors.time.sleep"):
            positions = _move_motors_and_wait(
                server,
                ("x", "y"),
                (10.0, 2.0),
                max_retries=3,
            )

        self.assertEqual(positions, (9.5, 2.5))
        self.assertEqual(len(server.move_calls), 1)

    def test_stale_move_complete_before_goal_latch_is_not_accepted(self):
        server = FakeBCSServer(
            [(1.0,)],
            get_motor_responses=[
                _get_motor_response((0.0,), goals=(0.0,)),
                _get_motor_response((0.0,), goals=(0.0,)),
                _get_motor_response((1.0,), goals=(1.0,)),
                _get_motor_response((1.0,), goals=(1.0,)),
            ],
        )

        with (
            patch("merlin_track_position.instruments.motors.time.sleep") as sleep,
            patch(
                "merlin_track_position.instruments.motors.time.monotonic",
                side_effect=[0.0, 0.0, 0.1, 0.2],
            ),
        ):
            positions = _move_motors_and_wait(
                server,
                ("x",),
                (1.0,),
                max_retries=0,
                backlash_correction={},
            )

        self.assertEqual(positions, (1.0,))
        self.assertEqual(len(server.move_calls), 1)
        self.assertEqual(
            sleep.call_args_list,
            [call(0.25), call(0.25), call(0.25)],
        )

    def test_motor_timing_matches_bcs_example(self):
        server = FakeBCSServer(
            [(1.0,)],
            get_motor_responses=[
                _get_motor_response((0.0,)),
                _get_motor_response((0.5,), status=0),
                _get_motor_response((1.0,)),
                _get_motor_response((1.0,)),
                _get_motor_response((1.0,)),
            ],
        )

        with (
            patch("merlin_track_position.instruments.motors.time.sleep") as sleep,
            patch(
                "merlin_track_position.instruments.motors.time.monotonic",
                side_effect=[0.0, 0.1, 0.2, 0.3],
            ),
        ):
            positions = _move_motors_and_wait(
                server,
                ("x",),
                (1.0,),
                backlash_correction={},
            )

        self.assertEqual(positions, (1.0,))
        self.assertEqual(sleep.call_args_list[:2], [call(0.25), call(0.25)])
        self.assertTrue(all(args.args[0] == 0.25 for args in sleep.call_args_list))

    def test_move_motor_not_found_response_raises_with_context(self):
        server = FakeBCSServer(
            [(1.0,)],
            initial_positions=(0.0,),
            move_responses=[
                {
                    "success": True,
                    "not_found": [MOTOR_NAMES["x"]],
                    "timed_out": [],
                    "error description": "missing motor",
                }
            ],
        )

        with (
            patch("merlin_track_position.instruments.motors.time.sleep"),
            self.assertRaisesRegex(
                RuntimeError,
                r"MoveMotor reported not_found.*Sample X.*goals=\(1\.0,\).*missing motor",
            ),
        ):
            _move_motors_and_wait(
                server,
                ("x",),
                (1.0,),
                backlash_correction={},
            )

        self.assertEqual(len(server.move_calls), 1)

    def test_move_motor_timed_out_response_raises_with_context(self):
        server = FakeBCSServer(
            [(1.0,)],
            initial_positions=(0.0,),
            move_responses=[
                {
                    "success": True,
                    "not_found": [],
                    "timed_out": [MOTOR_NAMES["x"]],
                    "error_description": "queue timed out",
                }
            ],
        )

        with (
            patch("merlin_track_position.instruments.motors.time.sleep"),
            self.assertRaisesRegex(
                RuntimeError,
                r"MoveMotor reported timed_out.*Sample X.*goals=\(1\.0,\).*queue timed out",
            ),
        ):
            _move_motors_and_wait(
                server,
                ("x",),
                (1.0,),
                backlash_correction={},
            )

        self.assertEqual(len(server.move_calls), 1)

    def test_move_motor_unsuccessful_response_raises_with_error_text(self):
        server = FakeBCSServer(
            [(1.0,)],
            initial_positions=(0.0,),
            move_responses=[
                {
                    "success": False,
                    "not_found": [],
                    "timed_out": [],
                    "error description": "queue unavailable",
                }
            ],
        )

        with (
            patch("merlin_track_position.instruments.motors.time.sleep"),
            self.assertRaisesRegex(
                RuntimeError,
                r"MoveMotor failed: queue unavailable.*Sample X.*goals=\(1\.0,\)",
            ),
        ):
            _move_motors_and_wait(
                server,
                ("x",),
                (1.0,),
                backlash_correction={},
            )

        self.assertEqual(len(server.move_calls), 1)

    def test_get_motor_missing_data_response_raises(self):
        server = FakeBCSServer(
            [(1.0,)],
            get_motor_responses=[{"success": True, "not_found": []}],
        )

        with self.assertRaisesRegex(RuntimeError, "GetMotor.*missing data"):
            _move_motors_and_wait(
                server,
                ("x",),
                (1.0,),
                backlash_correction={},
            )

        self.assertEqual(len(server.move_calls), 0)

    def test_get_motor_short_data_response_raises(self):
        server = FakeBCSServer(
            [(1.0,)],
            get_motor_responses=[{"success": True, "not_found": [], "data": []}],
        )

        with self.assertRaisesRegex(RuntimeError, "expected 1 data rows, got 0"):
            _move_motors_and_wait(
                server,
                ("x",),
                (1.0,),
                backlash_correction={},
            )

        self.assertEqual(len(server.move_calls), 0)

    def test_get_motor_missing_key_response_raises(self):
        server = FakeBCSServer(
            [(1.0,)],
            get_motor_responses=[
                {
                    "success": True,
                    "not_found": [],
                    "data": [{"status": BCSz.MotorStatus.MOVE_COMPLETE.value}],
                }
            ],
        )

        with self.assertRaisesRegex(RuntimeError, r"missing key 'position'"):
            _move_motors_and_wait(
                server,
                ("x",),
                (1.0,),
                backlash_correction={},
            )

        self.assertEqual(len(server.move_calls), 0)

    def test_get_motor_not_found_response_raises(self):
        server = FakeBCSServer(
            [(1.0,)],
            get_motor_responses=[
                {
                    "success": True,
                    "not_found": [MOTOR_NAMES["x"]],
                    "data": [],
                    "error description": "unknown motor",
                }
            ],
        )

        with self.assertRaisesRegex(
            RuntimeError,
            r"GetMotor reported not_found.*Sample X.*unknown motor",
        ):
            _move_motors_and_wait(
                server,
                ("x",),
                (1.0,),
                backlash_correction={},
            )

        self.assertEqual(len(server.move_calls), 0)

    def test_get_motor_unsuccessful_response_raises_with_error_text(self):
        server = FakeBCSServer(
            [(1.0,)],
            get_motor_responses=[
                {
                    "success": False,
                    "not_found": [],
                    "data": [],
                    "error_description": "backend unavailable",
                }
            ],
        )

        with self.assertRaisesRegex(
            RuntimeError,
            r"GetMotor failed: backend unavailable.*Sample X",
        ):
            _move_motors_and_wait(
                server,
                ("x",),
                (1.0,),
                backlash_correction={},
            )

        self.assertEqual(len(server.move_calls), 0)

    def test_get_motor_malformed_status_response_raises(self):
        server = FakeBCSServer(
            [(1.0,)],
            get_motor_responses=[
                _get_motor_response((0.0,)),
                {
                    "success": True,
                    "not_found": [],
                    "data": [
                        {"position": 1.0, "goal": 1.0, "status": "bad", "time": 0}
                    ],
                },
            ],
        )

        with (
            patch("merlin_track_position.instruments.motors.time.sleep"),
            patch(
                "merlin_track_position.instruments.motors.time.monotonic",
                return_value=0.0,
            ),
            self.assertRaisesRegex(RuntimeError, "malformed status field"),
        ):
            _move_motors_and_wait(
                server,
                ("x",),
                (1.0,),
                backlash_correction={},
            )

        self.assertEqual(len(server.move_calls), 1)

    def test_stale_status_accepts_position_readback_after_delay(self):
        server = FakeBCSServer(
            [(10.0005, 1.9995)],
            status=0,
        )

        with (
            patch("merlin_track_position.instruments.motors.time.sleep"),
            patch(
                "merlin_track_position.instruments.motors.time.monotonic",
                side_effect=[0.0, constants.MOTOR_STALE_READBACK_DELAY_S],
            ),
        ):
            positions = _move_motors_and_wait(
                server,
                ("x", "y"),
                (10.0, 2.0),
                max_retries=3,
            )

        self.assertEqual(positions, (10.0005, 1.9995))
        self.assertEqual(len(server.move_calls), 1)

    def test_stale_status_accepts_axis_specific_readback_deadband(self):
        server = FakeBCSServer(
            [(10.0005, 2.004)],
            status=0,
        )

        with (
            patch.object(
                constants,
                "MOTOR_STALE_READBACK_DEADBAND",
                {"x": 0.001, "y": 0.005},
            ),
            patch("merlin_track_position.instruments.motors.time.sleep"),
            patch(
                "merlin_track_position.instruments.motors.time.monotonic",
                side_effect=[0.0, constants.MOTOR_STALE_READBACK_DELAY_S],
            ),
        ):
            positions = _move_motors_and_wait(
                server,
                ("x", "y"),
                (10.0, 2.0),
                max_retries=3,
            )

        self.assertEqual(positions, (10.0005, 2.004))
        self.assertEqual(len(server.move_calls), 1)

    def test_raw_move_complete_accepts_position_readback_without_stale_delay(self):
        server = FakeBCSServer(
            [(10.0005,)],
            status=BCSz.MotorStatus.RAW_MOVE_COMPLETE.value,
        )

        with (
            patch("merlin_track_position.instruments.motors.time.sleep"),
            patch(
                "merlin_track_position.instruments.motors.time.monotonic",
                side_effect=[0.0, 0.0],
            ),
        ):
            positions = _move_motors_and_wait(
                server,
                ("x",),
                (10.0,),
                max_retries=3,
                backlash_correction={},
            )

        self.assertEqual(positions, (10.0005,))
        self.assertEqual(len(server.move_calls), 1)

    def test_stale_status_does_not_accept_position_readback_before_delay(self):
        server = FakeBCSServer(
            [(10.0, 2.0)],
            status=0,
        )

        with (
            patch("merlin_track_position.instruments.motors.time.sleep"),
            patch(
                "merlin_track_position.instruments.motors.time.monotonic",
                side_effect=[0.0, 0.0, constants.MOTOR_STALE_READBACK_DELAY_S - 0.1],
            ),
            self.assertRaisesRegex(TimeoutError, "Timed out waiting"),
        ):
            _move_motors_and_wait(
                server,
                ("x", "y"),
                (10.0, 2.0),
                move_timeout_s=constants.MOTOR_STALE_READBACK_DELAY_S - 0.1,
            )

        self.assertEqual(len(server.move_calls), 1)

    def test_stale_status_outside_readback_deadband_times_out(self):
        outside_x_deadband = 10.0 + constants.MOTOR_STALE_READBACK_DEADBAND["x"] + 0.001
        server = FakeBCSServer(
            [(outside_x_deadband, 2.0)],
            status=0,
        )

        with (
            patch("merlin_track_position.instruments.motors.time.sleep"),
            patch(
                "merlin_track_position.instruments.motors.time.monotonic",
                side_effect=[0.0, constants.MOTOR_STALE_READBACK_DELAY_S],
            ),
            self.assertRaisesRegex(TimeoutError, "Timed out waiting"),
        ):
            _move_motors_and_wait(
                server,
                ("x", "y"),
                (10.0, 2.0),
                move_timeout_s=constants.MOTOR_STALE_READBACK_DELAY_S,
            )

        self.assertEqual(len(server.move_calls), 1)

    def test_raises_when_move_status_and_position_do_not_complete_before_timeout(self):
        server = FakeBCSServer(
            [(9.5, 2.5)],
            status=0,
        )

        with (
            patch("merlin_track_position.instruments.motors.time.sleep"),
            self.assertRaisesRegex(TimeoutError, "Timed out waiting"),
        ):
            _move_motors_and_wait(
                server,
                ("x", "y"),
                (10.0, 2.0),
                move_timeout_s=0.0,
            )

        self.assertEqual(len(server.move_calls), 1)

    def test_positive_moves_on_backlash_axes_do_not_preposition(self):
        server = FakeBCSServer(
            [
                (0.02, 2.0, 0.04),
            ],
            initial_positions=(0.005, 2.5, 0.0),
        )

        with patch("merlin_track_position.instruments.motors.time.sleep"):
            positions = _move_motors_and_wait(
                server,
                ("x", "y", "z"),
                (0.02, 2.0, 0.04),
                backlash_correction={"x": 0.030, "z": 0.030},
            )

        self.assertEqual(positions, (0.02, 2.0, 0.04))
        self.assertEqual(
            server.move_calls,
            [
                (
                    (MOTOR_NAMES["x"], MOTOR_NAMES["y"], MOTOR_NAMES["z"]),
                    (0.02, 2.0, 0.04),
                )
            ],
        )

    def test_backlash_correction_prepositions_below_target_when_above_target(self):
        server = FakeBCSServer(
            [
                (0.005,),
                (0.02, 2.0),
            ],
            initial_positions=(0.05, 2.5),
        )

        with patch("merlin_track_position.instruments.motors.time.sleep"):
            positions = _move_motors_and_wait(
                server,
                ("x", "y"),
                (0.02, 2.0),
                backlash_correction={"x": 0.015},
            )

        self.assertEqual(positions, (0.02, 2.0))
        self.assertEqual(len(server.move_calls), 2)
        self.assertEqual(server.move_calls[0][0], (MOTOR_NAMES["x"],))
        self.assertAlmostEqual(server.move_calls[0][1][0], 0.005)
        self.assertEqual(server.move_calls[1][0], (MOTOR_NAMES["x"], MOTOR_NAMES["y"]))
        self.assertEqual(server.move_calls[1][1], (0.02, 2.0))

    def test_backlash_preposition_is_retried_when_raw_complete_without_motion(self):
        server = FakeBCSServer(
            [
                (0.5,),
                (0.395,),
                (0.495,),
            ],
            initial_positions=(0.5,),
            status=BCSz.MotorStatus.RAW_MOVE_COMPLETE.value,
        )

        with (
            patch("merlin_track_position.instruments.motors.time.sleep"),
            patch(
                "merlin_track_position.instruments.motors.time.monotonic",
                side_effect=[
                    0.0,
                    constants.MOTOR_STALE_READBACK_DELAY_S,
                    constants.MOTOR_STALE_READBACK_DELAY_S,
                    constants.MOTOR_STALE_READBACK_DELAY_S,
                    constants.MOTOR_STALE_READBACK_DELAY_S,
                    constants.MOTOR_STALE_READBACK_DELAY_S,
                ],
            ),
        ):
            positions = _move_motors_and_wait(
                server,
                ("x",),
                (0.495,),
                max_retries=1,
                backlash_correction={"x": 0.1},
            )

        self.assertEqual(positions, (0.495,))
        self.assertEqual(
            server.move_calls,
            [
                ((MOTOR_NAMES["x"],), (0.395,)),
                ((MOTOR_NAMES["x"],), (0.395,)),
                ((MOTOR_NAMES["x"],), (0.495,)),
            ],
        )

    def test_raw_complete_without_goal_latch_reports_goal_did_not_latch(self):
        server = FakeBCSServer(
            [
                (0.5,),
            ],
            initial_positions=(0.5,),
            status=BCSz.MotorStatus.RAW_MOVE_COMPLETE.value,
            goal_latched_by_move=[False],
        )

        with (
            patch("merlin_track_position.instruments.motors.time.sleep"),
            patch(
                "merlin_track_position.instruments.motors.time.monotonic",
                side_effect=[0.0, constants.MOTOR_STALE_READBACK_DELAY_S],
            ),
            self.assertRaisesRegex(TimeoutError, "goal_did_not_latch"),
        ):
            _move_motors_and_wait(
                server,
                ("x",),
                (0.395,),
                max_retries=0,
                backlash_correction={},
            )

        self.assertEqual(
            server.move_calls,
            [
                ((MOTOR_NAMES["x"],), (0.395,)),
            ],
        )

    def test_requested_axes_are_sent_even_when_already_close(self):
        server = FakeBCSServer(
            [(0.0205, 2.0, 0.0409)],
            initial_positions=(0.02, 2.0, 0.04),
        )

        with patch("merlin_track_position.instruments.motors.time.sleep"):
            positions = _move_motors_and_wait(
                server,
                ("x", "y", "z"),
                (0.0205, 2.0, 0.0409),
                backlash_correction={"x": 0.030, "z": 0.030},
            )

        self.assertEqual(positions, (0.0205, 2.0, 0.0409))
        self.assertEqual(
            server.move_calls,
            [
                (
                    (MOTOR_NAMES["x"], MOTOR_NAMES["y"], MOTOR_NAMES["z"]),
                    (0.0205, 2.0, 0.0409),
                )
            ],
        )

    def test_positive_z_move_does_not_get_backlash_preposition(self):
        server = FakeBCSServer(
            [
                (0.0204, 2.0005, 0.04),
            ],
            initial_positions=(0.02, 2.0, 0.0),
        )

        with patch("merlin_track_position.instruments.motors.time.sleep"):
            positions = _move_motors_and_wait(
                server,
                ("x", "y", "z"),
                (0.0204, 2.0005, 0.04),
                backlash_correction={"x": 0.030, "z": 0.030},
            )

        self.assertEqual(positions, (0.0204, 2.0005, 0.04))
        self.assertEqual(
            server.move_calls,
            [
                (
                    (MOTOR_NAMES["x"], MOTOR_NAMES["y"], MOTOR_NAMES["z"]),
                    (0.0204, 2.0005, 0.04),
                ),
            ],
        )

    def test_negative_z_move_gets_backlash_and_final_move(self):
        server = FakeBCSServer(
            [
                (-0.03,),
                (0.0204, 2.0005, 0.04),
            ],
            initial_positions=(0.02, 2.0, 0.08),
        )

        with patch("merlin_track_position.instruments.motors.time.sleep"):
            positions = _move_motors_and_wait(
                server,
                ("x", "y", "z"),
                (0.0204, 2.0005, 0.04),
                backlash_correction={"x": 0.030, "z": 0.030},
            )

        self.assertEqual(positions, (0.0204, 2.0005, 0.04))
        self.assertEqual(len(server.move_calls), 2)
        self.assertEqual(server.move_calls[0][0], (MOTOR_NAMES["z"],))
        self.assertAlmostEqual(server.move_calls[0][1][0], 0.01)
        self.assertEqual(
            server.move_calls[1],
            (
                (MOTOR_NAMES["x"], MOTOR_NAMES["y"], MOTOR_NAMES["z"]),
                (0.0204, 2.0005, 0.04),
            ),
        )

    def test_all_requested_axes_move_without_backlash(self):
        server = FakeBCSServer(
            [(0.0205, 2.02, 0.0405)],
            initial_positions=(0.02, 2.0, 0.04),
        )

        with patch("merlin_track_position.instruments.motors.time.sleep"):
            positions = _move_motors_and_wait(
                server,
                ("x", "y", "z"),
                (0.0205, 2.02, 0.0405),
                backlash_correction={"x": 0.030, "z": 0.030},
            )

        self.assertEqual(positions, (0.0205, 2.02, 0.0405))
        self.assertEqual(
            server.move_calls,
            [
                (
                    (MOTOR_NAMES["x"], MOTOR_NAMES["y"], MOTOR_NAMES["z"]),
                    (0.0205, 2.02, 0.0405),
                )
            ],
        )

    def test_mixed_move_keeps_all_requested_axes(self):
        server = FakeBCSServer(
            [
                (0.02, 0.02, 0.0005),
            ],
            initial_positions=(0.0, 0.0, 0.0),
        )

        with patch("merlin_track_position.instruments.motors.time.sleep"):
            positions = _move_motors_and_wait(
                server,
                ("x", "y", "z"),
                (0.02, 0.02, 0.0005),
                backlash_correction={"x": 0.015, "z": 0.030},
            )

        self.assertEqual(positions, (0.02, 0.02, 0.0005))
        self.assertEqual(
            server.move_calls,
            [
                (
                    (MOTOR_NAMES["x"], MOTOR_NAMES["y"], MOTOR_NAMES["z"]),
                    (0.02, 0.02, 0.0005),
                ),
            ],
        )

    def test_rejects_invalid_backlash_correction(self):
        server = FakeBCSServer([(10.0,)], initial_positions=(0.0,))

        with self.assertRaises(ValueError):
            _move_motors_and_wait(
                server,
                ("x",),
                (10.0,),
                backlash_correction={"x": -1.0},
            )


class MotorPositionCacheTests(unittest.TestCase):
    def setUp(self):
        simulator.reset()
        _clear_motor_position_cache()

    def tearDown(self):
        _clear_motor_position_cache()

    def test_update_and_cached_motor_positions_are_explicit(self):
        with patch("merlin_track_position.instruments.motors.time.monotonic", return_value=10.0):
            updated = update_motor_position_cache(
                {"x": 1.25, "p": -4.5},
                source="test",
            )

        self.assertEqual(updated, {"x": 1.25, "p": -4.5})
        with patch("merlin_track_position.instruments.motors.time.monotonic", return_value=12.0):
            self.assertEqual(
                cached_motor_positions(("p", "x"), max_age_s=3.0),
                (-4.5, 1.25),
            )
            with self.assertRaisesRegex(RuntimeError, "stale"):
                cached_motor_positions(("x",), max_age_s=1.0)

    def test_cached_motor_positions_reject_missing_and_nonfinite_values(self):
        with self.assertRaisesRegex(RuntimeError, "missing"):
            cached_motor_positions(("x",))
        with self.assertRaisesRegex(ValueError, "finite"):
            update_motor_position_cache({"x": float("nan")})
        with self.assertRaisesRegex(ValueError, "max_age_s"):
            cached_motor_positions(("x",), max_age_s=-1.0)

    def test_live_get_positions_updates_cache(self):
        with patch.object(constants, "IS_DAQ_PC", False):
            self.assertEqual(get_positions(("x", "cam")), (0.0, 5.0))

        self.assertEqual(cached_motor_positions(("cam", "x")), (5.0, 0.0))

    def test_refresh_motor_positions_live_reads_and_updates_cache(self):
        with patch.object(constants, "IS_DAQ_PC", False):
            positions = refresh_motor_positions(("x", "y"))

        self.assertEqual(positions, (0.0, 0.0))
        self.assertEqual(cached_motor_positions(("x", "y")), (0.0, 0.0))


class DevelopmentModeMotorTests(unittest.TestCase):
    def setUp(self):
        simulator.reset()
        _clear_motor_position_cache()

    def tearDown(self):
        _clear_motor_position_cache()

    def test_get_positions_uses_simulator_without_bcs_context(self):
        with (
            patch.object(constants, "IS_DAQ_PC", False),
            patch(
                "merlin_track_position.instruments.motors._bcs_server_context",
                side_effect=AssertionError("BCS context should not be opened"),
            ),
        ):
            positions = get_positions(("x", "y", "cam"))

        self.assertEqual(positions, (0.0, 0.0, 5.0))

    def test_move_motors_and_wait_updates_simulated_positions(self):
        with (
            patch.object(constants, "IS_DAQ_PC", False),
            patch(
                "merlin_track_position.instruments.motors._bcs_server_context",
                side_effect=AssertionError("BCS context should not be opened"),
            ),
            patch(
                "merlin_track_position.instruments.simulated_hardware.time.sleep"
            ) as sleep,
        ):
            positions = move_motors_and_wait(("x", "y"), (0.03, -0.015))
            saved_positions = get_positions(("x", "y"))

        self.assertEqual(positions, (0.03, -0.015))
        self.assertEqual(saved_positions, (0.03, -0.015))
        self.assertEqual(cached_motor_positions(("x", "y")), (0.03, -0.015))
        sleep.assert_called_once()
        self.assertLessEqual(sleep.call_args.args[0], 0.5)

    def test_get_temperatures_uses_static_development_values(self):
        with patch.object(constants, "IS_DAQ_PC", False):
            self.assertEqual(get_temperatures(), (30.0, 30.0, 30.0, 30.0))

    def test_overlapping_simulated_moves_are_serialized(self):
        first_move_settling = threading.Event()
        second_move_started = threading.Event()
        second_move_finished = threading.Event()
        release_first_move = threading.Event()
        sleep_call_count = 0
        sleep_call_lock = threading.Lock()
        first_result = []
        second_result = []

        def fake_sleep(delay_s):
            nonlocal sleep_call_count
            with sleep_call_lock:
                sleep_call_count += 1
                call_index = sleep_call_count
            if call_index == 1:
                first_move_settling.set()
                if not release_first_move.wait(timeout=2.0):
                    raise TimeoutError("timed out waiting to release first move")

        def first_move():
            first_result.append(move_motors_and_wait(("x",), (0.1,))[0])

        def second_move():
            second_move_started.set()
            second_result.append(move_motors_and_wait(("x",), (0.2,))[0])
            second_move_finished.set()

        with (
            patch.object(constants, "IS_DAQ_PC", False),
            patch(
                "merlin_track_position.instruments.simulated_hardware.time.sleep",
                side_effect=fake_sleep,
            ),
        ):
            first_thread = threading.Thread(target=first_move)
            second_thread = threading.Thread(target=second_move)

            first_thread.start()
            self.assertTrue(first_move_settling.wait(timeout=2.0))
            second_thread.start()
            self.assertTrue(second_move_started.wait(timeout=2.0))
            self.assertFalse(second_move_finished.wait(timeout=0.05))

            release_first_move.set()
            first_thread.join(timeout=2.0)
            second_thread.join(timeout=2.0)

            self.assertFalse(first_thread.is_alive())
            self.assertFalse(second_thread.is_alive())
            self.assertEqual(get_positions(("x",)), (0.2,))

        self.assertEqual(first_result, [0.1])
        self.assertEqual(second_result, [0.2])


if __name__ == "__main__":
    unittest.main()

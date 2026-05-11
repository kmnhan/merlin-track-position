import contextlib
import threading
import unittest
from unittest.mock import patch

import merlin_track_position.instruments.BCSz as BCSz
from merlin_track_position import constants
from merlin_track_position.constants import MOTOR_NAMES
from merlin_track_position.instruments.motors import (
    _move_motors_and_wait,
    get_positions,
    get_temperatures,
    move_motors_and_wait,
)
from merlin_track_position.instruments.simulated_hardware import simulator


class FakeBCSServer:
    def __init__(self, final_positions_by_move, initial_positions=()):
        self._final_positions_by_move = [
            tuple(positions) for positions in final_positions_by_move
        ]
        self._initial_positions = tuple(initial_positions)
        self._positions_by_motor = {}
        self.move_calls = []

    def move_motor(self, *, motors, goals):
        self._ensure_positions(motors)
        self.move_calls.append((tuple(motors), tuple(goals)))
        move_index = min(
            len(self.move_calls) - 1, len(self._final_positions_by_move) - 1
        )
        final_positions = self._final_positions_by_move[move_index]
        for motor, position in zip(motors, final_positions, strict=True):
            self._positions_by_motor[motor] = position

    def get_motor(self, *, motors):
        self._ensure_positions(motors)
        return {
            "data": [
                {
                    "position": self._positions_by_motor[motor],
                    "status": BCSz.MotorStatus.MOVE_COMPLETE.value,
                }
                for motor in motors
            ]
        }

    def _ensure_positions(self, motors):
        if self._positions_by_motor:
            for motor in motors:
                self._positions_by_motor.setdefault(motor, 0.0)
            return

        if self._initial_positions:
            for motor, position in zip(motors, self._initial_positions, strict=True):
                self._positions_by_motor[motor] = position
        for motor in motors:
            self._positions_by_motor.setdefault(motor, 0.0)


class MoveMotorsAndWaitTests(unittest.TestCase):
    def test_public_helper_forwards_tolerance_and_retry_arguments(self):
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
            positions = move_motors_and_wait(
                ("x",), (1.0,), tolerance=0.01, max_retries=2
            )

        self.assertEqual(positions, (1.0,))
        move.assert_called_once_with(
            server,
            ("x",),
            (1.0,),
            tolerance=0.01,
            max_retries=2,
            backlash_correction=constants.MOTOR_BACKLASH_CORRECTION,
        )

    def test_returns_after_one_move_when_tolerance_is_not_requested(self):
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

    def test_retries_until_positions_are_within_scalar_tolerance(self):
        server = FakeBCSServer([(9.5, 2.5), (10.2, 2.0), (10.03,)])

        with patch("merlin_track_position.instruments.motors.time.sleep"):
            positions = _move_motors_and_wait(
                server,
                ("x", "y"),
                (10.0, 2.0),
                tolerance=0.05,
                max_retries=4,
            )

        self.assertEqual(positions, (10.03, 2.0))
        self.assertEqual(len(server.move_calls), 3)
        self.assertEqual(
            server.move_calls[0],
            ((MOTOR_NAMES["x"], MOTOR_NAMES["y"]), (10.0, 2.0)),
        )
        self.assertEqual(server.move_calls[2], ((MOTOR_NAMES["x"],), (10.0,)))

    def test_accepts_per_motor_tolerances(self):
        server = FakeBCSServer([(10.04, 2.2), (2.08,)])

        with patch("merlin_track_position.instruments.motors.time.sleep"):
            positions = _move_motors_and_wait(
                server,
                ("x", "y"),
                (10.0, 2.0),
                tolerance=(0.05, 0.1),
                max_retries=1,
            )

        self.assertEqual(positions, (10.04, 2.08))
        self.assertEqual(len(server.move_calls), 2)
        self.assertEqual(server.move_calls[1], ((MOTOR_NAMES["y"],), (2.0,)))

    def test_raises_after_retry_limit_is_exhausted(self):
        server = FakeBCSServer([(9.5, 2.5), (9.75, 2.25)])

        with patch("merlin_track_position.instruments.motors.time.sleep"):
            _move_motors_and_wait(
                server,
                ("x", "y"),
                (10.0, 2.0),
                tolerance=0.05,
                max_retries=1,
            )

        self.assertEqual(len(server.move_calls), 2)

    def test_validates_tolerance_length(self):
        server = FakeBCSServer([(10.0, 2.0)])

        with self.assertRaises(ValueError):
            _move_motors_and_wait(
                server,
                ("x", "y"),
                (10.0, 2.0),
                tolerance=(0.05,),
            )

    def test_backlash_correction_prepositions_below_current_when_already_below_target(self):
        server = FakeBCSServer(
            [
                (-0.025, -0.03),
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
        self.assertEqual(server.move_calls[0][0], (MOTOR_NAMES["x"], MOTOR_NAMES["z"]))
        self.assertAlmostEqual(server.move_calls[0][1][0], -0.025)
        self.assertAlmostEqual(server.move_calls[0][1][1], -0.03)
        self.assertEqual(
            server.move_calls[1][0],
            (MOTOR_NAMES["x"], MOTOR_NAMES["y"], MOTOR_NAMES["z"]),
        )
        self.assertEqual(server.move_calls[1][1], (0.02, 2.0, 0.04))

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

    def test_skips_axes_within_default_deadband(self):
        server = FakeBCSServer([], initial_positions=(0.02, 2.0, 0.04))

        with patch("merlin_track_position.instruments.motors.time.sleep"):
            positions = _move_motors_and_wait(
                server,
                ("x", "y", "z"),
                (0.0205, 2.0, 0.0409),
                backlash_correction={"x": 0.030, "z": 0.030},
            )

        self.assertEqual(positions, (0.02, 2.0, 0.04))
        self.assertEqual(server.move_calls, [])

    def test_only_changed_z_axis_gets_backlash_and_final_move(self):
        server = FakeBCSServer(
            [
                (-0.03,),
                (0.04,),
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

        self.assertEqual(positions, (0.02, 2.0, 0.04))
        self.assertEqual(
            server.move_calls,
            [
                ((MOTOR_NAMES["z"],), (-0.03,)),
                ((MOTOR_NAMES["z"],), (0.04,)),
            ],
        )

    def test_only_changed_y_axis_moves_without_backlash(self):
        server = FakeBCSServer([(2.01,)], initial_positions=(0.02, 2.0, 0.04))

        with patch("merlin_track_position.instruments.motors.time.sleep"):
            positions = _move_motors_and_wait(
                server,
                ("x", "y", "z"),
                (0.0205, 2.01, 0.0395),
                backlash_correction={"x": 0.030, "z": 0.030},
            )

        self.assertEqual(positions, (0.02, 2.01, 0.04))
        self.assertEqual(server.move_calls, [((MOTOR_NAMES["y"],), (2.01,))])

    def test_mixed_move_omits_unchanged_backlash_axis(self):
        server = FakeBCSServer(
            [
                (-0.015,),
                (0.02, 0.02),
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

        self.assertEqual(positions, (0.02, 0.02, 0.0))
        self.assertEqual(
            server.move_calls,
            [
                ((MOTOR_NAMES["x"],), (-0.015,)),
                ((MOTOR_NAMES["x"], MOTOR_NAMES["y"]), (0.02, 0.02)),
            ],
        )

    def test_explicit_tolerance_overrides_default_deadband(self):
        server = FakeBCSServer(
            [
                (-0.015,),
                (0.0008,),
            ],
            initial_positions=(0.0,),
        )

        with patch("merlin_track_position.instruments.motors.time.sleep"):
            positions = _move_motors_and_wait(
                server,
                ("x",),
                (0.0008,),
                tolerance=0.0005,
                backlash_correction={"x": 0.015},
            )

        self.assertEqual(positions, (0.0008,))
        self.assertEqual(
            server.move_calls,
            [
                ((MOTOR_NAMES["x"],), (-0.015,)),
                ((MOTOR_NAMES["x"],), (0.0008,)),
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


class DevelopmentModeMotorTests(unittest.TestCase):
    def setUp(self):
        simulator.reset()

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
            patch("merlin_track_position.instruments.simulated_hardware.time.sleep")
            as sleep,
        ):
            positions = move_motors_and_wait(("x", "y"), (0.03, -0.015))
            saved_positions = get_positions(("x", "y"))

        self.assertEqual(positions, (0.03, -0.015))
        self.assertEqual(saved_positions, (0.03, -0.015))
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

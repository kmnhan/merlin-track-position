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
    def __init__(self, final_positions_by_move):
        self._final_positions_by_move = [
            tuple(positions) for positions in final_positions_by_move
        ]
        self.move_calls = []
        self._current_positions = ()

    def move_motor(self, *, motors, goals):
        self.move_calls.append((tuple(motors), tuple(goals)))
        move_index = min(
            len(self.move_calls) - 1, len(self._final_positions_by_move) - 1
        )
        self._current_positions = self._final_positions_by_move[move_index]

    def get_motor(self, *, motors):
        return {
            "data": [
                {
                    "position": position,
                    "status": BCSz.MotorStatus.MOVE_COMPLETE.value,
                }
                for position in self._current_positions
            ]
        }


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
        server = FakeBCSServer([(9.5, 2.5), (10.2, 2.0), (10.03, 2.04)])

        with patch("merlin_track_position.instruments.motors.time.sleep"):
            positions = _move_motors_and_wait(
                server,
                ("x", "y"),
                (10.0, 2.0),
                tolerance=0.05,
                max_retries=4,
            )

        self.assertEqual(positions, (10.03, 2.04))
        self.assertEqual(len(server.move_calls), 3)
        self.assertEqual(
            server.move_calls[0],
            ((MOTOR_NAMES["x"], MOTOR_NAMES["y"]), (10.0, 2.0)),
        )

    def test_accepts_per_motor_tolerances(self):
        server = FakeBCSServer([(10.04, 2.2), (10.04, 2.08)])

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

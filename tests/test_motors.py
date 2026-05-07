import contextlib
import unittest
from unittest.mock import patch

import merlin_track_position.instruments.BCSz as BCSz
from merlin_track_position.constants import MOTOR_NAMES
from merlin_track_position.instruments.motors import (
    _move_motors_and_wait,
    move_motors_and_wait,
)


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


if __name__ == "__main__":
    unittest.main()

import json
import threading
import unittest

from merlin_track_position.server import MotorServer, TrackShiftMotorBackend


class TrackShiftMotorBackendTests(unittest.TestCase):
    def test_move_request_payload_and_result_update_positions(self):
        requests = []
        backend = None

        def send_move_request(payload):
            requests.append(payload)
            backend.submit_move_result(
                {
                    "command": "MOVE_RESULT",
                    "session_id": payload["session_id"],
                    "move_id": payload["move_id"],
                    "ok": True,
                    "positions_mm": {
                        "x": payload["targets_mm"].get("x", 0.1),
                        "y": 2.0,
                        "z": payload["targets_mm"].get("z", -0.3),
                    },
                    "message": "move complete",
                }
            )

        backend = TrackShiftMotorBackend(
            session_id="session-1",
            initial_positions_mm={"x": 0.1, "y": 2.0, "z": -0.3},
            send_move_request=send_move_request,
        )

        final_positions = backend.move_motors_and_wait(
            ("x", "z"),
            (0.5, -1.25),
            max_retries=2,
            move_timeout_s=12.0,
        )

        self.assertEqual(final_positions, (0.5, -1.25))
        self.assertEqual(len(requests), 1)
        self.assertEqual(requests[0]["session_id"], "session-1")
        self.assertEqual(requests[0]["move_id"], 1)
        self.assertEqual(requests[0]["axes"], ["x", "z"])
        self.assertEqual(requests[0]["targets_mm"], {"x": 0.5, "z": -1.25})
        self.assertEqual(requests[0]["timeout_ms"], 60_000)
        self.assertEqual(requests[0]["max_retries"], 2)
        self.assertEqual(backend.get_positions(("x", "y", "z")), (0.5, 2.0, -1.25))

    def test_failed_move_result_raises(self):
        backend = None

        def send_move_request(payload):
            backend.submit_move_result(
                {
                    "command": "MOVE_RESULT",
                    "session_id": payload["session_id"],
                    "move_id": payload["move_id"],
                    "ok": False,
                    "positions_mm": {"x": 0.1, "y": 2.0, "z": -0.3},
                    "message": "Sample Z move timeout",
                }
            )

        backend = TrackShiftMotorBackend(
            session_id="session-1",
            initial_positions_mm={"x": 0.1, "y": 2.0, "z": -0.3},
            send_move_request=send_move_request,
        )

        with self.assertRaisesRegex(RuntimeError, "Sample Z move timeout"):
            backend.move_motors_and_wait(("z",), (-1.25,))


class MotorServerDialogueTests(unittest.TestCase):
    def test_start_creates_backend_and_move_result_unblocks_move(self):
        server = MotorServer()
        server._running.set()
        targets = []
        server.sigMoveDetected.connect(lambda target: targets.append(target))

        try:
            response = server._handle_request(
                {
                    "command": "START",
                    "target": 1,
                    "positions_mm": {"x": 0.1, "y": 2.0, "z": -0.3},
                    "session_id": "session-1",
                }
            )
            self.assertIsNone(response)
            self.assertEqual(targets, [1])
            backend = server.current_motor_backend()
            self.assertIsNotNone(backend)

            move_result = []

            def request_move():
                move_result.append(
                    backend.move_motors_and_wait(("x",), (0.25,), move_timeout_s=1.0)
                )

            thread = threading.Thread(target=request_move)
            thread.start()
            status, payload = server._wait_for_response()
            decoded_payload = json.loads(payload)

            self.assertEqual(status, "MOVE")
            self.assertEqual(decoded_payload["session_id"], "session-1")
            self.assertEqual(decoded_payload["move_id"], 1)
            self.assertEqual(decoded_payload["axes"], ["x"])
            self.assertEqual(decoded_payload["targets_mm"], {"x": 0.25})

            response = server._handle_request(
                {
                    "command": "MOVE_RESULT",
                    "session_id": "session-1",
                    "move_id": 1,
                    "ok": True,
                    "positions_mm": {"x": 0.25, "y": 2.0, "z": -0.3},
                    "message": "done",
                }
            )
            thread.join(timeout=2.0)

            self.assertIsNone(response)
            self.assertFalse(thread.is_alive())
            self.assertEqual(move_result, [(0.25,)])
        finally:
            server.stop()


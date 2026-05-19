import json
import threading
import time
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
        self.assertEqual(
            requests[0]["targets_mm"],
            {"x": 0.5, "y": 2.0, "z": -1.25},
        )
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
    def _decoded_response(self, response):
        status, payload = response
        return status, json.loads(payload)

    def _wait_for_state(self, server, expected_state, *, timeout_s=2.0):
        deadline = time.monotonic() + timeout_s
        last_payload = None
        while time.monotonic() < deadline:
            status, payload = self._decoded_response(
                server._handle_request({"command": "STATUS"})
            )
            self.assertEqual(status, "OK")
            last_payload = payload
            if payload["state"] == expected_state:
                return payload
            time.sleep(0.01)
        self.fail(f"state {expected_state!r} not reached; last payload={last_payload!r}")

    def test_start_creates_backend_and_returns_correcting_immediately(self):
        server = MotorServer()
        server._running.set()
        targets = []
        server.sigMoveDetected.connect(lambda target: targets.append(target))

        try:
            status, payload = self._decoded_response(
                server._handle_request(
                    {
                        "command": "START",
                        "target": 1,
                        "positions_mm": {"x": 0.1, "y": 2.0, "z": -0.3},
                        "session_id": "session-1",
                    }
                )
            )
            self.assertEqual(status, "OK")
            self.assertEqual(payload["state"], "correcting")
            self.assertEqual(payload["session_id"], "session-1")
            self.assertEqual(payload["target"], 1)
            self.assertEqual(targets, [1])
            self.assertIsNotNone(server.current_motor_backend())
        finally:
            server.stop()

    def test_status_returns_pending_move_idempotently_and_move_result_unblocks_move(
        self,
    ):
        server = MotorServer()
        server._running.set()
        try:
            server._handle_request(
                {
                    "command": "START",
                    "target": 1,
                    "positions_mm": {"x": 0.1, "y": 2.0, "z": -0.3},
                    "session_id": "session-1",
                    "timeout_ms": 1000,
                }
            )
            backend = server.current_motor_backend()
            self.assertIsNotNone(backend)
            move_result = []

            def request_move():
                move_result.append(
                    backend.move_motors_and_wait(("x",), (0.25,), move_timeout_s=1.0)
                )

            thread = threading.Thread(target=request_move)
            thread.start()
            pending = self._wait_for_state(server, "move_pending")
            repeated = self._wait_for_state(server, "move_pending")

            pending_move = pending["pending_move"]
            repeated_move = repeated["pending_move"]
            self.assertEqual(pending_move, repeated_move)
            self.assertEqual(pending_move["session_id"], "session-1")
            self.assertEqual(pending_move["move_id"], 1)
            self.assertEqual(pending_move["axes"], ["x"])
            self.assertEqual(
                pending_move["targets_mm"],
                {"x": 0.25, "y": 2.0, "z": -0.3},
            )

            status, payload = self._decoded_response(
                server._handle_request(
                    {
                        "command": "MOVE_RESULT",
                        "session_id": "session-1",
                        "move_id": 1,
                        "ok": True,
                        "positions_mm": {"x": 0.25, "y": 2.0, "z": -0.3},
                        "message": "done",
                    }
                )
            )
            self.assertEqual(status, "OK")
            self.assertEqual(payload["state"], "correcting")
            self.assertEqual(payload["accepted_move_result"]["move_id"], 1)
            thread.join(timeout=2.0)

            self.assertFalse(thread.is_alive())
            self.assertEqual(move_result, [(0.25,)])

            server.set_result(True, "Correction converged after 1 move(s).")
            status, payload = self._decoded_response(
                server._handle_request({"command": "STATUS"})
            )
            self.assertEqual(status, "OK")
            self.assertEqual(payload["state"], "complete")
            self.assertEqual(payload["message"], "Correction converged after 1 move(s).")
            self.assertIsNone(server.current_motor_backend())
        finally:
            server.stop()

    def test_session_mismatch_does_not_consume_pending_move(self):
        server = MotorServer()
        server._running.set()
        try:
            server._handle_request(
                {
                    "command": "START",
                    "target": 1,
                    "positions_mm": {"x": 0.1, "y": 2.0, "z": -0.3},
                    "session_id": "session-1",
                    "timeout_ms": 1000,
                }
            )
            backend = server.current_motor_backend()
            self.assertIsNotNone(backend)
            move_errors = []

            def request_move():
                try:
                    backend.move_motors_and_wait(("z",), (-0.5,), move_timeout_s=1.0)
                except Exception as exc:
                    move_errors.append(str(exc))

            thread = threading.Thread(target=request_move)
            thread.start()
            pending = self._wait_for_state(server, "move_pending")
            move_id = pending["pending_move"]["move_id"]

            status, payload = self._decoded_response(
                server._handle_request(
                    {
                        "command": "MOVE_RESULT",
                        "session_id": "stale-session",
                        "move_id": move_id,
                        "ok": True,
                        "positions_mm": {"x": 0.1, "y": 2.0, "z": -0.5},
                    }
                )
            )

            self.assertEqual(status, "ERROR")
            self.assertIn("did not match active session", payload["message"])
            self.assertEqual(self._wait_for_state(server, "move_pending"), pending)
            self.assertTrue(thread.is_alive())

            server._handle_request(
                {
                    "command": "MOVE_RESULT",
                    "session_id": "session-1",
                    "move_id": move_id,
                    "ok": False,
                    "positions_mm": {"x": 0.1, "y": 2.0, "z": -0.3},
                    "message": "cleanup",
                }
            )
            thread.join(timeout=2.0)
            self.assertFalse(thread.is_alive())
            self.assertEqual(move_errors, ["cleanup"])
        finally:
            server.stop()

    def test_abort_marks_error_and_unblocks_pending_move(self):
        server = MotorServer()
        server._running.set()
        try:
            server._handle_request(
                {
                    "command": "START",
                    "target": 1,
                    "positions_mm": {"x": 0.1, "y": 2.0, "z": -0.3},
                    "session_id": "session-1",
                    "timeout_ms": 1000,
                }
            )
            backend = server.current_motor_backend()
            self.assertIsNotNone(backend)
            move_errors = []

            def request_move():
                try:
                    backend.move_motors_and_wait(("x",), (0.5,), move_timeout_s=1.0)
                except Exception as exc:
                    move_errors.append(str(exc))

            thread = threading.Thread(target=request_move)
            thread.start()
            self._wait_for_state(server, "move_pending")

            status, payload = self._decoded_response(
                server._handle_request(
                    {
                        "command": "ABORT",
                        "session_id": "session-1",
                        "message": "operator abort",
                    }
                )
            )
            thread.join(timeout=2.0)

            self.assertEqual(status, "OK")
            self.assertEqual(payload["state"], "error")
            self.assertEqual(payload["message"], "operator abort")
            self.assertFalse(thread.is_alive())
            self.assertEqual(move_errors, ["operator abort"])
            self.assertIsNone(server.current_motor_backend())
        finally:
            server.stop()

    def test_bcs_api_backend_start_returns_correcting_without_pending_move(self):
        server = MotorServer(use_bcs_api_backend=True)
        server._running.set()
        targets = []
        server.sigMoveDetected.connect(lambda target: targets.append(target))

        try:
            status, payload = self._decoded_response(
                server._handle_request(
                    {
                        "command": "START",
                        "target": 1,
                        "positions_mm": {"x": 0.1, "y": 2.0, "z": -0.3},
                        "session_id": "session-1",
                    }
                )
            )
            self.assertEqual(status, "OK")
            self.assertEqual(payload["state"], "correcting")
            self.assertEqual(payload["session_id"], "session-1")
            self.assertEqual(payload["target"], 1)
            self.assertNotIn("pending_move", payload)
            self.assertEqual(targets, [1])
            self.assertIsNone(server.current_motor_backend())

            status, repeated = self._decoded_response(
                server._handle_request({"command": "STATUS"})
            )
            self.assertEqual(status, "OK")
            self.assertEqual(repeated["state"], "correcting")
            self.assertNotIn("pending_move", repeated)
        finally:
            server.stop()

    def test_bcs_api_backend_final_status_persists(self):
        server = MotorServer(use_bcs_api_backend=True)
        server._running.set()
        try:
            server._handle_request(
                {
                    "command": "START",
                    "target": 1,
                    "positions_mm": {"x": 0.1, "y": 2.0, "z": -0.3},
                    "session_id": "session-1",
                }
            )

            server.set_result(True, "Correction converged after 0 move(s).")
            status, payload = self._decoded_response(
                server._handle_request({"command": "STATUS"})
            )
            self.assertEqual(status, "OK")
            self.assertEqual(payload["state"], "complete")
            self.assertEqual(
                payload["message"],
                "Correction converged after 0 move(s).",
            )
            self.assertNotIn("pending_move", payload)
            self.assertIsNone(server.current_motor_backend())

            server._handle_request(
                {
                    "command": "START",
                    "target": 2,
                    "positions_mm": {"x": 0.1, "y": 2.0, "z": -0.3},
                    "session_id": "session-2",
                }
            )
            server.set_result(False, "BCS API move failed")
            status, payload = self._decoded_response(
                server._handle_request({"command": "STATUS"})
            )
            self.assertEqual(status, "OK")
            self.assertFalse(payload["ok"])
            self.assertEqual(payload["state"], "error")
            self.assertEqual(payload["message"], "BCS API move failed")
            self.assertNotIn("pending_move", payload)
        finally:
            server.stop()

    def test_bcs_api_backend_rejects_move_result_and_second_start(self):
        server = MotorServer(use_bcs_api_backend=True)
        server._running.set()
        try:
            server._handle_request(
                {
                    "command": "START",
                    "target": 1,
                    "positions_mm": {"x": 0.1, "y": 2.0, "z": -0.3},
                    "session_id": "session-1",
                }
            )

            status, payload = self._decoded_response(
                server._handle_request(
                    {
                        "command": "MOVE_RESULT",
                        "session_id": "session-1",
                        "move_id": 1,
                        "ok": True,
                        "positions_mm": {"x": 0.1, "y": 2.0, "z": -0.3},
                    }
                )
            )
            self.assertEqual(status, "ERROR")
            self.assertEqual(payload["state"], "correcting")
            self.assertIn(
                "without an active LabVIEW correction move",
                payload["message"],
            )

            status, payload = self._decoded_response(
                server._handle_request(
                    {
                        "command": "START",
                        "target": 2,
                        "positions_mm": {"x": 0.1, "y": 2.0, "z": -0.3},
                        "session_id": "session-2",
                    }
                )
            )
            self.assertEqual(status, "ERROR")
            self.assertIn("already active", payload["message"])
        finally:
            server.stop()

    def test_bcs_api_backend_abort_marks_error(self):
        server = MotorServer(use_bcs_api_backend=True)
        server._running.set()
        try:
            server._handle_request(
                {
                    "command": "START",
                    "target": 1,
                    "positions_mm": {"x": 0.1, "y": 2.0, "z": -0.3},
                    "session_id": "session-1",
                }
            )

            status, payload = self._decoded_response(
                server._handle_request(
                    {
                        "command": "ABORT",
                        "session_id": "session-1",
                        "message": "operator abort",
                    }
                )
            )

            self.assertEqual(status, "OK")
            self.assertFalse(payload["ok"])
            self.assertEqual(payload["state"], "error")
            self.assertEqual(payload["message"], "operator abort")
            self.assertNotIn("pending_move", payload)
            self.assertIsNone(server.current_motor_backend())
        finally:
            server.stop()

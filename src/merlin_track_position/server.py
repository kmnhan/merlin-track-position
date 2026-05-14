"""Server thread for receiving motor move requests from the motor driver."""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import json
import logging
import math
import threading
import time
import typing
import uuid

import zmq
from qtpy import QtCore

from merlin_track_position.constants import MOTOR_SERVER_PORT

logger = logging.getLogger("merlin_track_position.server")
_UNSET = object()
_XYZ_AXES = ("x", "y", "z")
_FINAL_STATUSES = {"OK", "ERROR"}
_DEFAULT_XYZ_MOVE_TIMEOUT_MS = 60_000
_MOVE_RESULT_TIMEOUT_MARGIN_S = 30.0
_REQUEST_LOG_LIMIT = 500


class TrackShiftProtocolError(RuntimeError):
    """Raised when the TrackTemperatureBL403 ZMQ dialogue is malformed."""


@dataclass(frozen=True)
class _MoveResult:
    ok: bool
    positions_mm: dict[str, float]
    message: str


def _coerce_axis_positions(
    positions: Mapping[str, typing.Any],
    *,
    required_axes: Sequence[str] = (),
) -> dict[str, float]:
    if not isinstance(positions, Mapping):
        raise TrackShiftProtocolError("positions_mm must be an object")

    coerced: dict[str, float] = {}
    for axis, value in positions.items():
        axis_name = str(axis)
        if axis_name not in _XYZ_AXES:
            raise TrackShiftProtocolError(f"unsupported correction axis {axis_name!r}")
        position = float(value)
        if not math.isfinite(position):
            raise TrackShiftProtocolError(
                f"position for axis {axis_name!r} must be finite"
            )
        coerced[axis_name] = position

    missing = [axis for axis in required_axes if axis not in coerced]
    if missing:
        raise TrackShiftProtocolError(
            "positions_mm is missing required axis readbacks: "
            + ", ".join(missing)
        )
    return coerced


def _json_payload(payload: Mapping[str, typing.Any]) -> str:
    return json.dumps(payload, separators=(",", ":"))


def _decode_json_request(raw: str) -> typing.Any:
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        preview = raw[:_REQUEST_LOG_LIMIT]
        raise TrackShiftProtocolError(
            "invalid JSON request from LabVIEW: "
            f"{exc}; length={len(raw)}; preview={preview!r}"
        ) from exc


class TrackShiftMotorBackend:
    """Correction motor backend that delegates XYZ moves to LabVIEW."""

    def __init__(
        self,
        *,
        session_id: str | None = None,
        initial_positions_mm: Mapping[str, typing.Any],
        send_move_request: Callable[[dict[str, typing.Any]], None],
        default_move_timeout_ms: int = _DEFAULT_XYZ_MOVE_TIMEOUT_MS,
    ):
        self.session_id = session_id or uuid.uuid4().hex
        self._positions_mm = _coerce_axis_positions(
            initial_positions_mm,
            required_axes=_XYZ_AXES,
        )
        self._send_move_request = send_move_request
        self._default_move_timeout_ms = int(default_move_timeout_ms)
        self._condition = threading.Condition()
        self._next_move_id = 1
        self._pending_move_id: int | None = None
        self._pending_axes: tuple[str, ...] = ()
        self._pending_result: _MoveResult | object = _UNSET

    def get_positions(self, motor_aliases: Sequence[str]) -> tuple[float, ...]:
        positions: list[float] = []
        for motor_alias in motor_aliases:
            axis = str(motor_alias)
            if axis not in _XYZ_AXES:
                raise ValueError(f"unsupported Track Shift correction axis: {axis!r}")
            if axis not in self._positions_mm:
                raise RuntimeError(f"missing readback for correction axis {axis!r}")
            positions.append(self._positions_mm[axis])
        return tuple(positions)

    def move_motors_and_wait(
        self,
        motor_aliases: Sequence[str],
        goals: Sequence[float],
        *,
        max_retries: int = 4,
        backlash_correction: dict[str, float] | None = None,
        move_timeout_s: float = 60.0,
    ) -> tuple[float, ...]:
        axes = tuple(str(axis) for axis in motor_aliases)
        goals = tuple(float(goal) for goal in goals)
        if len(axes) != len(goals):
            raise ValueError("motor_aliases and goals must have the same length")
        if not axes:
            return ()

        targets_mm: dict[str, float] = {}
        for axis, goal in zip(axes, goals, strict=True):
            if axis not in _XYZ_AXES:
                raise ValueError(f"unsupported Track Shift correction axis: {axis!r}")
            if not math.isfinite(goal):
                raise ValueError(f"target for axis {axis!r} must be finite")
            targets_mm[axis] = goal

        timeout_ms = max(1, int(round(float(move_timeout_s) * 1000.0)))
        if self._default_move_timeout_ms > 0:
            timeout_ms = max(timeout_ms, self._default_move_timeout_ms)

        with self._condition:
            if self._pending_move_id is not None:
                raise RuntimeError("Track Shift correction already has a pending move")
            move_id = self._next_move_id
            self._next_move_id += 1
            self._pending_move_id = move_id
            self._pending_axes = axes
            self._pending_result = _UNSET

        payload = {
            "session_id": self.session_id,
            "move_id": move_id,
            "axes": list(axes),
            "targets_mm": targets_mm,
            "timeout_ms": timeout_ms,
            "max_retries": int(max_retries),
        }
        logger.info(
            "Requesting LabVIEW XYZ correction move: session_id=%s, move_id=%s, "
            "axes=%s, targets_mm=%s, timeout_ms=%d",
            self.session_id,
            move_id,
            axes,
            targets_mm,
            timeout_ms,
        )
        try:
            self._send_move_request(payload)
            result = self._wait_for_move_result(timeout_ms)
        except Exception:
            with self._condition:
                if self._pending_move_id == move_id:
                    self._clear_pending_move()
            raise

        if not result.ok:
            raise RuntimeError(result.message or "LabVIEW XYZ correction move failed")

        logger.info(
            "LabVIEW XYZ correction move completed: session_id=%s, move_id=%s, "
            "positions_mm=%s",
            self.session_id,
            move_id,
            result.positions_mm,
        )
        return self.get_positions(axes)

    def submit_move_result(self, request: Mapping[str, typing.Any]) -> None:
        with self._condition:
            if self._pending_move_id is None:
                raise TrackShiftProtocolError(
                    "received MOVE_RESULT without a pending MOVE request"
                )

            ok = bool(request.get("ok", False))
            message = str(request.get("message", ""))
            try:
                session_id = str(request["session_id"])
                move_id = int(request["move_id"])
            except Exception as exc:
                ok = False
                message = f"malformed MOVE_RESULT identifiers: {exc}"
            else:
                if session_id != self.session_id:
                    ok = False
                    message = (
                        f"MOVE_RESULT session_id {session_id!r} did not match "
                        f"active session {self.session_id!r}"
                    )
                elif move_id != self._pending_move_id:
                    ok = False
                    message = (
                        f"MOVE_RESULT move_id {move_id!r} did not match pending "
                        f"move_id {self._pending_move_id!r}"
                    )

            try:
                positions = _coerce_axis_positions(
                    request.get("positions_mm", {}),
                    required_axes=(),
                )
            except Exception as exc:
                positions = {}
                ok = False
                message = f"malformed MOVE_RESULT positions_mm: {exc}"

            missing = [axis for axis in self._pending_axes if axis not in positions]
            if ok and missing:
                ok = False
                message = (
                    "MOVE_RESULT missing final readbacks for moved axes: "
                    + ", ".join(missing)
                )

            if positions:
                self._positions_mm.update(positions)
            self._pending_result = _MoveResult(
                ok=ok,
                positions_mm=dict(self._positions_mm),
                message=message,
            )
            self._condition.notify_all()

    def cancel_pending(self, message: str) -> None:
        with self._condition:
            if self._pending_move_id is not None:
                self._pending_result = _MoveResult(
                    ok=False,
                    positions_mm=dict(self._positions_mm),
                    message=message,
                )
                self._condition.notify_all()

    def _wait_for_move_result(self, timeout_ms: int) -> _MoveResult:
        deadline = time.monotonic() + timeout_ms / 1000.0 + (
            _MOVE_RESULT_TIMEOUT_MARGIN_S
        )
        with self._condition:
            while self._pending_result is _UNSET:
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    raise TimeoutError(
                        "Timed out waiting for LabVIEW MOVE_RESULT for "
                        f"session_id={self.session_id}, "
                        f"move_id={self._pending_move_id}"
                    )
                self._condition.wait(remaining)

            result = self._pending_result
            self._clear_pending_move()

        if not isinstance(result, _MoveResult):
            raise RuntimeError("invalid Track Shift move result state")
        return result

    def _clear_pending_move(self) -> None:
        self._pending_move_id = None
        self._pending_axes = ()
        self._pending_result = _UNSET


class MotorServer(QtCore.QThread):
    sigMoveDetected = QtCore.Signal(int)

    def __init__(self, parent: QtCore.QObject | None = None):
        super().__init__(parent)

        self._ret_val: typing.Any = _UNSET
        self._ret_val_cv = threading.Condition()
        self._active_motor_backend: TrackShiftMotorBackend | None = None

        self._running = threading.Event()

    @QtCore.Slot(object)
    def set_result(self, success: bool, msg: str) -> None:
        if success:
            self._set_response("OK", msg)
        else:
            self._set_response(
                "ERROR",
                msg
                + " This will shut down the motor subsystem, and it will need to be restarted by clicking on the indicator box in the right panel.",
            )

    def current_motor_backend(self) -> TrackShiftMotorBackend | None:
        return self._active_motor_backend

    def run(self) -> None:
        _socket = None
        try:
            ctx = zmq.Context.instance()
            _socket = ctx.socket(zmq.REP)
            _socket.setsockopt(zmq.SNDHWM, 0)
            _socket.setsockopt(zmq.RCVHWM, 0)
            _socket.setsockopt(zmq.LINGER, 0)
            _socket.bind(f"tcp://127.0.0.1:{MOTOR_SERVER_PORT}")
            logger.info("ZMQ server bound")

            poller = zmq.Poller()
            poller.register(_socket, zmq.POLLIN)
            self._running.set()

            while self._running.is_set() and not self.isInterruptionRequested():
                events = dict(poller.poll(100))
                if _socket in events and events[_socket] & zmq.POLLIN:
                    try:
                        raw: str = _socket.recv_string(flags=zmq.NOBLOCK)
                        req = _decode_json_request(raw)
                        immediate_response = self._handle_request(req)
                        if immediate_response is None:
                            status, msg = self._wait_for_response()
                        else:
                            status, msg = immediate_response

                        _socket.send_multipart([status.encode(), msg.encode()])
                        if status in _FINAL_STATUSES:
                            self._clear_active_backend()

                    except Exception as exc:
                        logger.warning("Failed to process message: %s", exc)
                        self._clear_active_backend(str(exc))
                        try:
                            _socket.send_multipart(
                                [b"ERROR", f"Track Shift protocol error: {exc}".encode()]
                            )
                        except Exception:
                            logger.exception("Failed to send protocol error response.")

        except Exception:
            logger.exception("ZMQ server error")
        finally:
            if _socket is not None:
                try:
                    _socket.close(0)
                except Exception:
                    logger.exception("Failed to close ZMQ server socket.")
            self._running.clear()
            self._clear_active_backend("motor server stopped")
            with self._ret_val_cv:
                self._ret_val_cv.notify_all()
            logger.info("ZMQ server stopped")

    def stop(self) -> None:
        self._running.clear()
        self.requestInterruption()
        self._clear_active_backend("motor server stop requested")
        with self._ret_val_cv:
            self._ret_val_cv.notify_all()

    def _handle_request(
        self,
        req: Mapping[str, typing.Any],
    ) -> tuple[str, str] | None:
        command = str(req.get("command", "")).upper()
        if command == "MOVE_RESULT":
            if self._active_motor_backend is None:
                return ("ERROR", "received MOVE_RESULT without an active correction")
            self._active_motor_backend.submit_move_result(req)
            return None

        if command in ("", "START"):
            return self._handle_start_request(req, use_labview_backend=command == "START")

        return ("ERROR", f"unknown Track Shift command: {command!r}")

    def _handle_start_request(
        self,
        req: Mapping[str, typing.Any],
        *,
        use_labview_backend: bool,
    ) -> tuple[str, str] | None:
        if self._active_motor_backend is not None:
            return ("ERROR", "a Track Shift correction session is already active")

        target = round(req["target"])
        logger.debug(
            "Received Track Shift correction request: target=%d, labview_backend=%s",
            target,
            use_labview_backend,
        )

        if use_labview_backend:
            initial_positions = _coerce_axis_positions(
                req.get("positions_mm", {}),
                required_axes=_XYZ_AXES,
            )
            timeout_ms = int(req.get("timeout_ms", _DEFAULT_XYZ_MOVE_TIMEOUT_MS))
            self._active_motor_backend = TrackShiftMotorBackend(
                session_id=str(req.get("session_id") or uuid.uuid4().hex),
                initial_positions_mm=initial_positions,
                send_move_request=self._send_move_request,
                default_move_timeout_ms=timeout_ms,
            )
            logger.info(
                "Started Track Shift LabVIEW motor backend: session_id=%s, "
                "positions_mm=%s",
                self._active_motor_backend.session_id,
                initial_positions,
            )

        self.sigMoveDetected.emit(target)
        return None

    def _send_move_request(self, payload: dict[str, typing.Any]) -> None:
        self._set_response("MOVE", _json_payload(payload))

    def _set_response(self, status: str, msg: str) -> None:
        with self._ret_val_cv:
            if self._ret_val is not _UNSET:
                logger.warning(
                    "Overwriting pending Track Shift response: old=%s, new=%s",
                    self._ret_val,
                    (status, msg),
                )
            self._ret_val = (status, str(msg))
            self._ret_val_cv.notify_all()

    def _wait_for_response(self) -> tuple[str, str]:
        with self._ret_val_cv:
            while self._ret_val is _UNSET:
                if not self._running.is_set() or self.isInterruptionRequested():
                    raise RuntimeError("motor server stopped before response was ready")
                self._ret_val_cv.wait(0.1)
            status, msg = self._ret_val
            self._ret_val = _UNSET
        return str(status), str(msg)

    def _clear_active_backend(self, message: str = "correction session ended") -> None:
        if self._active_motor_backend is not None:
            self._active_motor_backend.cancel_pending(message)
            self._active_motor_backend = None

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

from merlin_track_position.constants import (
    MOTOR_SERVER_PORT,
    MOTOR_SERVER_USE_BCS_API_BACKEND,
)

logger = logging.getLogger("merlin_track_position.server")
_UNSET = object()
_XYZ_AXES = ("x", "y", "z")
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
        self._pending_payload: dict[str, typing.Any] | None = None
        self._pending_result: _MoveResult | object = _UNSET
        self._cancel_message: str | None = None

    def get_positions(self, motor_aliases: Sequence[str]) -> tuple[float, ...]:
        self._raise_if_cancelled()
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

        self._raise_if_cancelled()
        active_targets_mm: dict[str, float] = {}
        for axis, goal in zip(axes, goals, strict=True):
            if axis not in _XYZ_AXES:
                raise ValueError(f"unsupported Track Shift correction axis: {axis!r}")
            if not math.isfinite(goal):
                raise ValueError(f"target for axis {axis!r} must be finite")
            active_targets_mm[axis] = goal

        timeout_ms = max(1, int(round(float(move_timeout_s) * 1000.0)))
        if self._default_move_timeout_ms > 0:
            timeout_ms = max(timeout_ms, self._default_move_timeout_ms)

        with self._condition:
            if self._pending_move_id is not None:
                raise RuntimeError("Track Shift correction already has a pending move")
            self._raise_if_cancelled()
            move_id = self._next_move_id
            self._next_move_id += 1
            self._pending_move_id = move_id
            self._pending_axes = axes
            self._pending_result = _UNSET

        targets_mm = dict(self._positions_mm)
        targets_mm.update(active_targets_mm)
        payload = {
            "session_id": self.session_id,
            "move_id": move_id,
            "axes": list(axes),
            "targets_mm": targets_mm,
            "timeout_ms": timeout_ms,
            "max_retries": int(max_retries),
        }
        with self._condition:
            if self._pending_move_id == move_id:
                self._pending_payload = dict(payload)
        logger.info(
            "Requesting LabVIEW XYZ correction move: session_id=%s, move_id=%s, "
            "axes=%s, targets_mm=%s, timeout_ms=%d",
            self.session_id,
            move_id,
            axes,
            active_targets_mm,
            timeout_ms,
        )
        try:
            self._send_move_request(dict(payload))
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

    def submit_move_result(self, request: Mapping[str, typing.Any]) -> _MoveResult:
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
                raise TrackShiftProtocolError(
                    f"malformed MOVE_RESULT identifiers: {exc}"
                ) from exc
            else:
                if session_id != self.session_id:
                    raise TrackShiftProtocolError(
                        f"MOVE_RESULT session_id {session_id!r} did not match "
                        f"active session {self.session_id!r}"
                    )
                elif move_id != self._pending_move_id:
                    raise TrackShiftProtocolError(
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
            return self._pending_result

    def cancel_pending(self, message: str) -> None:
        with self._condition:
            self._cancel_message = str(message)
            if self._pending_move_id is not None:
                self._pending_result = _MoveResult(
                    ok=False,
                    positions_mm=dict(self._positions_mm),
                    message=message,
                )
                self._condition.notify_all()

    def pending_move(self) -> dict[str, typing.Any] | None:
        with self._condition:
            if (
                self._pending_move_id is None
                or self._pending_payload is None
                or self._pending_result is not _UNSET
            ):
                return None
            return dict(self._pending_payload)

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
        self._pending_payload = None
        self._pending_result = _UNSET

    def _raise_if_cancelled(self) -> None:
        if self._cancel_message is not None:
            raise RuntimeError(self._cancel_message)


class MotorServer(QtCore.QThread):
    sigMoveDetected = QtCore.Signal(int)

    def __init__(
        self,
        parent: QtCore.QObject | None = None,
        *,
        use_bcs_api_backend: bool | None = None,
    ):
        super().__init__(parent)

        self._use_bcs_api_backend = (
            MOTOR_SERVER_USE_BCS_API_BACKEND
            if use_bcs_api_backend is None
            else bool(use_bcs_api_backend)
        )
        self._state_lock = threading.RLock()
        self._active_motor_backend: TrackShiftMotorBackend | None = None
        self._correction_active = False
        self._session_id: str | None = None
        self._target: int | None = None
        self._state = "idle"
        self._message = ""

        self._running = threading.Event()

    @QtCore.Slot(object)
    def set_result(self, success: bool, msg: str) -> None:
        message = str(msg)
        with self._state_lock:
            if self._active_motor_backend is not None and not success:
                self._active_motor_backend.cancel_pending(message)
            self._active_motor_backend = None
            self._correction_active = False
            self._state = "complete" if success else "error"
            self._message = message

    def current_motor_backend(self) -> TrackShiftMotorBackend | None:
        with self._state_lock:
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
                        status, msg = self._handle_request(req)
                        _socket.send_multipart([status.encode(), msg.encode()])

                    except Exception as exc:
                        logger.warning("Failed to process message: %s", exc)
                        try:
                            _socket.send_multipart(
                                [
                                    b"ERROR",
                                    _json_payload(
                                        {
                                            "ok": False,
                                            "state": "error",
                                            "message": (
                                                "Track Shift protocol error: "
                                                f"{exc}"
                                            ),
                                        }
                                    ).encode(),
                                ]
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
            self._clear_active_session("motor server stopped")
            logger.info("ZMQ server stopped")

    def stop(self) -> None:
        self._running.clear()
        self.requestInterruption()
        self._clear_active_session("motor server stop requested")

    def _handle_request(
        self,
        req: Mapping[str, typing.Any],
    ) -> tuple[str, str]:
        if not isinstance(req, Mapping):
            return self._error_response("Track Shift request must be a JSON object")
        command = str(req.get("command", "")).upper()
        if command == "STATUS":
            return ("OK", _json_payload(self._status_payload()))

        if command == "MOVE_RESULT":
            return self._handle_move_result(req)

        if command == "ABORT":
            return self._handle_abort_request(req)

        if command == "START":
            return self._handle_start_request(req)

        return self._error_response(f"unknown Track Shift command: {command!r}")

    def _handle_start_request(
        self,
        req: Mapping[str, typing.Any],
    ) -> tuple[str, str]:
        with self._state_lock:
            if self._correction_active:
                return self._error_response(
                    "a Track Shift correction session is already active"
                )

        try:
            target = round(req["target"])
        except Exception as exc:
            return self._error_response(f"START target is required and numeric: {exc}")
        logger.debug(
            "Received Track Shift correction request: target=%d",
            target,
        )

        try:
            initial_positions = _coerce_axis_positions(
                req.get("positions_mm", {}),
                required_axes=_XYZ_AXES,
            )
            timeout_ms = int(req.get("timeout_ms", _DEFAULT_XYZ_MOVE_TIMEOUT_MS))
        except Exception as exc:
            return self._error_response(f"malformed START request: {exc}")

        session_id = str(req.get("session_id") or uuid.uuid4().hex)
        backend = None
        if not self._use_bcs_api_backend:
            backend = TrackShiftMotorBackend(
                session_id=session_id,
                initial_positions_mm=initial_positions,
                send_move_request=self._send_move_request,
                default_move_timeout_ms=timeout_ms,
            )
        with self._state_lock:
            if self._correction_active:
                return self._error_response(
                    "a Track Shift correction session is already active"
                )
            self._active_motor_backend = backend
            self._correction_active = True
            self._session_id = session_id
            self._target = int(target)
            self._state = "correcting"
            self._message = ""
            if backend is None:
                logger.info(
                    "Started Track Shift BCS API motor backend: session_id=%s, "
                    "positions_mm=%s",
                    session_id,
                    initial_positions,
                )
            else:
                logger.info(
                    "Started Track Shift LabVIEW motor backend: session_id=%s, "
                    "positions_mm=%s",
                    backend.session_id,
                    initial_positions,
                )

        self.sigMoveDetected.emit(target)
        return ("OK", _json_payload(self._status_payload()))

    def _handle_move_result(self, req: Mapping[str, typing.Any]) -> tuple[str, str]:
        with self._state_lock:
            backend = self._active_motor_backend
        if backend is None:
            return self._error_response(
                "received MOVE_RESULT without an active LabVIEW correction move"
            )
        try:
            move_result = backend.submit_move_result(req)
        except Exception as exc:
            return self._error_response(str(exc))

        with self._state_lock:
            if self._active_motor_backend is backend:
                self._state = "correcting"
                self._message = move_result.message

        payload = self._status_payload()
        payload["accepted_move_result"] = {
            "session_id": backend.session_id,
            "move_id": int(req["move_id"]),
            "ok": bool(move_result.ok),
            "message": move_result.message,
        }
        return ("OK", _json_payload(payload))

    def _handle_abort_request(self, req: Mapping[str, typing.Any]) -> tuple[str, str]:
        requested_session_id = req.get("session_id")
        with self._state_lock:
            if (
                requested_session_id
                and self._session_id is not None
                and str(requested_session_id) != self._session_id
            ):
                return self._error_response(
                    f"ABORT session_id {requested_session_id!r} did not match "
                    f"active session {self._session_id!r}"
                )
            message = str(req.get("message") or "Track Shift correction aborted")
            self._clear_active_session(message)
            self._state = "error"
            self._message = message
        return ("OK", _json_payload(self._status_payload()))

    def _send_move_request(self, payload: dict[str, typing.Any]) -> None:
        with self._state_lock:
            if (
                self._active_motor_backend is None
                or payload.get("session_id") != self._active_motor_backend.session_id
            ):
                raise RuntimeError(
                    "cannot publish Track Shift move without an active session"
                )
            self._state = "move_pending"
            self._message = "LabVIEW XYZ correction move pending"

    def _clear_active_session(self, message: str = "correction session ended") -> None:
        with self._state_lock:
            if self._active_motor_backend is not None:
                self._active_motor_backend.cancel_pending(message)
                self._active_motor_backend = None
            self._correction_active = False

    def _status_payload(self) -> dict[str, typing.Any]:
        with self._state_lock:
            backend = self._active_motor_backend
            state = self._state
            pending_move = None if backend is None else backend.pending_move()
            if backend is not None:
                state = "move_pending" if pending_move is not None else "correcting"
            elif self._correction_active:
                state = "correcting"
            elif state not in {"complete", "error"}:
                state = "idle"

            payload: dict[str, typing.Any] = {
                "ok": state != "error",
                "state": state,
                "session_id": self._session_id or "",
                "target": self._target,
                "message": self._message,
            }
            if pending_move is not None:
                payload["pending_move"] = pending_move
            return payload

    def _error_response(self, message: str) -> tuple[str, str]:
        payload = self._status_payload()
        payload["ok"] = False
        payload["message"] = str(message)
        return ("ERROR", _json_payload(payload))

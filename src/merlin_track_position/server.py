"""Server thread for receiving motor move requests from the motor driver."""

import json
import logging
import threading
import typing

import zmq
from qtpy import QtCore

from merlin_track_position.constants import MOTOR_SERVER_PORT

logger = logging.getLogger("merlin_track_position.server")
_UNSET = object()


class MotorServer(QtCore.QThread):
    sigMoveDetected = QtCore.Signal(int)

    def __init__(self, parent: QtCore.QObject | None = None):
        super().__init__(parent)

        self._ret_val: typing.Any = _UNSET

        self._running = threading.Event()
        self._mutex = QtCore.QMutex()
        self._cv = QtCore.QWaitCondition()

    @QtCore.Slot(object)
    def set_result(self, success: bool, msg: str) -> None:
        with QtCore.QMutexLocker(self._mutex):
            if success:
                self._ret_val = ("OK", msg)
            else:
                self._ret_val = (
                    "ERROR",
                    msg
                    + " This will shut down the motor subsystem, and it will need to be restarted by clicking on the indicator box in the right panel.",
                )
            self._cv.wakeAll()

    def run(self) -> None:
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
                        req = json.loads(raw)
                        target: int = round(req["target"])
                        logger.debug("Received request for target %d", target)

                        self.sigMoveDetected.emit(target)
                        with QtCore.QMutexLocker(self._mutex):
                            while self._ret_val is _UNSET:
                                self._cv.wait(self._mutex)
                            status, msg = self._ret_val
                            self._ret_val = _UNSET

                        _socket.send_multipart([status.encode(), msg.encode()])

                    except Exception as exc:
                        logger.warning("Failed to process message: %s", exc)

        except Exception:
            logger.exception("ZMQ server error")
        finally:
            try:
                _socket.close(0)
            finally:
                self._running.clear()
                logger.info("ZMQ server stopped")

    def stop(self) -> None:
        self._running.clear()
        self.requestInterruption()

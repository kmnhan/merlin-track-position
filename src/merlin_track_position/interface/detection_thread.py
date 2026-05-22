from __future__ import annotations

import logging
import threading
from collections.abc import Mapping
from typing import Any

import xarray as xr
from qtpy import QtCore

from merlin_track_position.instruments.cameras import CameraPairPlugin
from merlin_track_position.tracking.detect import detect_shift

__all__ = ("DetectShiftThread",)

logger = logging.getLogger("merlin_track_position.interface.detection_thread")


class DetectShiftThread(QtCore.QThread):
    sigDetectionReady = QtCore.Signal(object)
    sigDetectionFailed = QtCore.Signal(str)

    def __init__(
        self,
        parent: QtCore.QObject | None = None,
    ):
        super().__init__(parent)
        self._running = threading.Event()
        self._calibration: xr.Dataset | None = None
        self._camera_pair: CameraPairPlugin | None = None
        self._shift_kwargs: dict[str, Any] = {}

    def configure(
        self,
        calibration: xr.Dataset,
        camera_pair: CameraPairPlugin,
        *,
        shift_kwargs: Mapping[str, Any] | None = None,
    ) -> None:
        """Set the parameters for the next no-move shift detection."""
        if self.isRunning():
            raise RuntimeError("cannot configure shift detection while it is running")
        self._calibration = calibration
        self._camera_pair = camera_pair
        self._shift_kwargs = {} if shift_kwargs is None else dict(shift_kwargs)

    def run(self) -> None:
        self._running.set()
        try:
            if not self._running.is_set() or self.isInterruptionRequested():
                logger.info("Shift detection skipped because stop was requested.")
                return

            try:
                if self._calibration is None or self._camera_pair is None:
                    raise RuntimeError("shift detection thread has not been configured")
                result = detect_shift(
                    self._calibration,
                    self._camera_pair,
                    **self._shift_kwargs,
                )
            except Exception as exc:
                logger.exception("Shift detection failed.")
                if self._running.is_set() and not self.isInterruptionRequested():
                    self.sigDetectionFailed.emit(str(exc))
                return

            if self._running.is_set() and not self.isInterruptionRequested():
                logger.info("Shift detection finished; emitting ready signal.")
                self.sigDetectionReady.emit(result)
            else:
                logger.info(
                    "Shift detection finished after stop request; no signal emitted."
                )
        finally:
            self._running.clear()
            logger.info("Shift detection thread stopped.")

    def stop(self) -> None:
        logger.info("Shift detection stop requested.")
        self._running.clear()
        self.requestInterruption()

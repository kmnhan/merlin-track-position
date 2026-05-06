from __future__ import annotations

import threading
import time
from pathlib import Path

import xarray as xr
from qtpy import QtCore

from merlin_track_position.interface.calibration_panel import (
    _validate_calibration_dataset,
)

__all__ = ("CalibrationThread",)


class CalibrationThread(QtCore.QThread):
    sigCalibrationReady = QtCore.Signal(object)
    sigCalibrationFailed = QtCore.Signal(str)

    def __init__(
        self,
        parent: QtCore.QObject | None = None,
        *,
        sleep_seconds: float = 5.0,
    ):
        super().__init__(parent)

        self._sleep_seconds = sleep_seconds
        self._running = threading.Event()

    def run(self) -> None:
        self._running.set()
        try:
            time.sleep(self._sleep_seconds)
            if not self._running.is_set() or self.isInterruptionRequested():
                return

            try:
                with xr.open_dataset(
                    Path("/Users/khan/Downloads/cal_30.0um_origin.h5"),
                    engine="h5netcdf",
                ) as dataset_on_disk:
                    calibration = dataset_on_disk.load()
                _validate_calibration_dataset(calibration)
            except Exception as exc:
                if self._running.is_set() and not self.isInterruptionRequested():
                    self.sigCalibrationFailed.emit(str(exc))
                return

            if self._running.is_set() and not self.isInterruptionRequested():
                self.sigCalibrationReady.emit(calibration)
        finally:
            self._running.clear()

    def stop(self) -> None:
        self._running.clear()
        self.requestInterruption()

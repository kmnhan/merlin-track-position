from __future__ import annotations

import threading
import time

import xarray as xr
from qtpy import QtCore

from merlin_track_position import constants
from merlin_track_position.interface.calibration_panel import (
    _validate_calibration_dataset,
)
from merlin_track_position.tracking.sample_calibration import (
    DEFAULT_SAMPLE_CALIBRATION_PATH,
)

__all__ = ("CalibrationThread",)


class CalibrationThread(QtCore.QThread):
    sigCalibrationReady = QtCore.Signal(object)
    sigCalibrationStep = QtCore.Signal(int, float, float, float, object, object)
    sigCalibrationFailed = QtCore.Signal(str)

    def __init__(
        self,
        parent: QtCore.QObject | None = None,
    ):
        super().__init__(parent)
        self._running = threading.Event()

    def run(self) -> None:
        self._running.set()
        try:
            if not self._running.is_set() or self.isInterruptionRequested():
                return

            try:
                time.sleep(3.0)  # Simulate a long-running calibration process.
                with xr.open_dataset(
                    DEFAULT_SAMPLE_CALIBRATION_PATH,
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

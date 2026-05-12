from __future__ import annotations

import threading
from pathlib import Path

import xarray as xr
from qtpy import QtCore

from merlin_track_position.instruments.cameras import CameraPairPlugin
from merlin_track_position.tracking.correct import do_correction

__all__ = ("CorrectionThread",)


class CorrectionThread(QtCore.QThread):
    sigCorrectionReady = QtCore.Signal(object)
    sigCorrectionFailed = QtCore.Signal(str)

    def __init__(
        self,
        parent: QtCore.QObject | None = None,
    ):
        super().__init__(parent)
        self._running = threading.Event()
        self._calibration: xr.Dataset | None = None
        self._camera_pair: CameraPairPlugin | None = None
        self._calibration_path: Path | None = None

    def configure(
        self,
        calibration: xr.Dataset,
        camera_pair: CameraPairPlugin,
        calibration_path: str | Path,
    ) -> None:
        """Set the parameters for the next correction run."""
        if self.isRunning():
            raise RuntimeError("cannot configure correction while it is running")
        self._calibration = calibration
        self._camera_pair = camera_pair
        self._calibration_path = Path(calibration_path)

    def run(self) -> None:
        self._running.set()
        try:
            if not self._running.is_set() or self.isInterruptionRequested():
                return

            try:
                if (
                    self._calibration is None
                    or self._camera_pair is None
                    or self._calibration_path is None
                ):
                    raise RuntimeError("correction thread has not been configured")
                result = do_correction(
                    self._calibration,
                    self._camera_pair,
                    calibration_path=self._calibration_path,
                )
            except Exception as exc:
                if self._running.is_set() and not self.isInterruptionRequested():
                    self.sigCorrectionFailed.emit(str(exc))
                return

            if self._running.is_set() and not self.isInterruptionRequested():
                self.sigCorrectionReady.emit(result)
        finally:
            self._running.clear()

    def stop(self) -> None:
        self._running.clear()
        self.requestInterruption()

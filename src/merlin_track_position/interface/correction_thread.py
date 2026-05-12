from __future__ import annotations

import logging
import threading
from pathlib import Path

import xarray as xr
from qtpy import QtCore

from merlin_track_position.instruments.cameras import CameraPairPlugin
from merlin_track_position.tracking.correct import do_correction

__all__ = ("CorrectionThread",)

logger = logging.getLogger("merlin_track_position.interface.correction_thread")


class CorrectionThread(QtCore.QThread):
    sigCorrectionProgress = QtCore.Signal(object)
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
        logger.info(
            "Configured correction thread: calibration_path=%s",
            calibration_path,
        )

    def run(self) -> None:
        self._running.set()
        try:
            if not self._running.is_set() or self.isInterruptionRequested():
                logger.info("Correction thread run skipped because stop was requested.")
                return

            try:
                if (
                    self._calibration is None
                    or self._camera_pair is None
                    or self._calibration_path is None
                ):
                    raise RuntimeError("correction thread has not been configured")
                logger.info(
                    "Correction thread starting do_correction: calibration_path=%s",
                    self._calibration_path,
                )
                result = do_correction(
                    self._calibration,
                    self._camera_pair,
                    calibration_path=self._calibration_path,
                    progress_callback=self._emit_progress,
                )
            except Exception as exc:
                logger.exception("Correction thread failed.")
                if self._running.is_set() and not self.isInterruptionRequested():
                    self.sigCorrectionFailed.emit(str(exc))
                return

            if self._running.is_set() and not self.isInterruptionRequested():
                logger.info("Correction thread finished; emitting ready signal.")
                self.sigCorrectionReady.emit(result)
            else:
                logger.info(
                    "Correction thread finished after stop request; no signal emitted."
                )
        finally:
            self._running.clear()
            logger.info("Correction thread stopped.")

    def stop(self) -> None:
        logger.info("Correction thread stop requested.")
        self._running.clear()
        self.requestInterruption()

    def _emit_progress(self, result: xr.Dataset) -> None:
        if self._running.is_set() and not self.isInterruptionRequested():
            self.sigCorrectionProgress.emit(result)

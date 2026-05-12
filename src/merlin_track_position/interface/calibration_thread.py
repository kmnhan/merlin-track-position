from __future__ import annotations

import threading
from collections.abc import Mapping

import numpy as np
from qtpy import QtCore

from merlin_track_position.instruments.cameras import CameraPairPlugin
from merlin_track_position.interface.calibration_panel import (
    _validate_calibration_dataset,
)
from merlin_track_position.tracking.calibrate import run_calibration

__all__ = ("CalibrationThread",)


class CalibrationThread(QtCore.QThread):
    sigCalibrationReady = QtCore.Signal(object)
    sigCalibrationStep = QtCore.Signal(int, float, float, float, object, object)
    sigCalibrationProcessingStep = QtCore.Signal(int, int)
    sigCalibrationFailed = QtCore.Signal(str)

    def __init__(
        self,
        parent: QtCore.QObject | None = None,
    ):
        super().__init__(parent)
        self._running = threading.Event()
        self._n: int | None = None
        self._step_um: float | None = None
        self._camera_pair: CameraPairPlugin | None = None
        self._roi_metadata: dict[str, float] = {}

    def configure(
        self,
        n: int,
        step_um: float,
        camera_pair: CameraPairPlugin,
        roi_metadata: Mapping[str, float],
    ) -> None:
        """Set the parameters for the next calibration run."""
        if self.isRunning():
            raise RuntimeError("cannot configure calibration while it is running")
        self._n = int(n)
        self._step_um = float(step_um)
        self._camera_pair = camera_pair
        self._roi_metadata = {
            str(key): float(value) for key, value in roi_metadata.items()
        }

    def run(self) -> None:
        self._running.set()
        try:
            if not self._running.is_set() or self.isInterruptionRequested():
                return

            try:
                if (
                    self._n is None
                    or self._step_um is None
                    or self._camera_pair is None
                ):
                    raise RuntimeError("calibration thread has not been configured")
                calibration = run_calibration(
                    self._n,
                    self._step_um,
                    self._camera_pair,
                    step_callback=self._emit_step,
                    processing_callback=self._emit_processing_step,
                )
                calibration = calibration.assign_attrs(self._roi_metadata)
                _validate_calibration_dataset(calibration)
            except Exception as exc:
                if self._running.is_set() and not self.isInterruptionRequested():
                    self.sigCalibrationFailed.emit(str(exc))
                return

            if self._running.is_set() and not self.isInterruptionRequested():
                self.sigCalibrationReady.emit(calibration)
        finally:
            self._running.clear()

    def _emit_step(
        self,
        idx: int,
        dx: float,
        dy: float,
        dz: float,
        image_cam0: np.ndarray,
        image_cam1: np.ndarray,
    ) -> None:
        if self._running.is_set() and not self.isInterruptionRequested():
            self.sigCalibrationStep.emit(idx, dx, dy, dz, image_cam0, image_cam1)

    def _emit_processing_step(self, completed: int, total: int) -> None:
        if self._running.is_set() and not self.isInterruptionRequested():
            self.sigCalibrationProcessingStep.emit(int(completed), int(total))

    def stop(self) -> None:
        self._running.clear()
        self.requestInterruption()

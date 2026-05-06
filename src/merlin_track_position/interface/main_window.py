from __future__ import annotations

import math
import multiprocessing
import sys
from pathlib import Path

import numpy as np
import pyqtgraph as pg
import xarray as xr
from qtpy import QtCore, QtGui, QtWidgets

from merlin_track_position.interface.calibration_thread import CalibrationThread
from merlin_track_position.interface.calibration_panel import (
    CalibrationPanel,
    _calibration_summary,
    _validate_calibration_dataset,
)
from merlin_track_position.server import MotorServer

__all__ = (
    "MainWindow",
    "_MainWindowGUI",
    "_calibration_summary",
    "_clamp_roi_geometry",
    "_default_roi_geometry",
    "_validate_calibration_dataset",
)

IMAGE_WIDTH: int = 704
IMAGE_HEIGHT: int = 480
ROI_SETTINGS_KEYS: tuple[str, str, str, str] = (
    "roi/x",
    "roi/y",
    "roi/width",
    "roi/height",
)


def _default_roi_geometry(
    image_width: float = IMAGE_WIDTH,
    image_height: float = IMAGE_HEIGHT,
) -> tuple[float, float, float, float]:
    width = 0.25 * image_width
    height = 0.25 * image_height
    return (
        0.5 * (image_width - width),
        0.5 * (image_height - height),
        width,
        height,
    )


def _clamp_roi_geometry(
    geometry: tuple[float, float, float, float],
    image_width: float = IMAGE_WIDTH,
    image_height: float = IMAGE_HEIGHT,
) -> tuple[float, float, float, float]:
    x, y, width, height = geometry
    if not all(math.isfinite(value) for value in geometry):
        return _default_roi_geometry(image_width, image_height)

    width = min(max(width, 1.0), image_width)
    height = min(max(height, 1.0), image_height)
    x = min(max(x, 0.0), image_width - width)
    y = min(max(y, 0.0), image_height - height)
    return (x, y, width, height)


class _MainWindowGUI(QtWidgets.QMainWindow):
    def __init__(self, parent: QtCore.QObject | None = None):
        super().__init__(parent)

        self.setWindowTitle("Track Positions")

        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QtWidgets.QVBoxLayout(central_widget)
        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        main_layout.addWidget(splitter)

        self.image_plot = pg.PlotWidget()
        self.image_plot.setAspectLocked(True)
        self.image_plot.setLabel("bottom", "x", units="px")
        self.image_plot.setLabel("left", "y", units="px")
        self.image_plot.showGrid(x=True, y=True, alpha=0.2)
        self.image_plot.invertY(True)

        self.image_item = pg.ImageItem(axisOrder="row-major")
        self.image_item.setImage(
            np.zeros((int(IMAGE_HEIGHT), int(IMAGE_WIDTH)), dtype=np.int64),
        )
        self.image_item.setRect(QtCore.QRectF(0.0, 0.0, IMAGE_WIDTH, IMAGE_HEIGHT))
        self.image_plot.addItem(self.image_item)
        self.image_plot.setXRange(0.0, IMAGE_WIDTH, padding=0.0)
        self.image_plot.setYRange(0.0, IMAGE_HEIGHT, padding=0.0)

        roi_geometry = _default_roi_geometry()
        self.image_roi = pg.RectROI(
            roi_geometry[:2],
            roi_geometry[2:],
            sideScalers=True,
            maxBounds=QtCore.QRectF(0.0, 0.0, IMAGE_WIDTH, IMAGE_HEIGHT),
            pen=pg.mkPen("#008c99", width=2),
            hoverPen=pg.mkPen("#00c2d1", width=2),
        )
        self.image_roi.setZValue(10)
        self.image_plot.addItem(self.image_roi)
        splitter.addWidget(self.image_plot)

        self.calibration_panel = CalibrationPanel()
        splitter.addWidget(self.calibration_panel)
        splitter.setSizes([760, 440])

    def _build_calibration_details_dialog(
        self,
        calibration: xr.Dataset,
    ) -> QtWidgets.QDialog:
        return self.calibration_panel.build_details_dialog(calibration)


class MainWindow(_MainWindowGUI):
    def __init__(self, parent: QtCore.QObject | None = None):
        super().__init__(parent)

        self._settings = QtCore.QSettings("merlin-track-position", "Track Positions")
        self._calibration: xr.Dataset | None = None
        self._calibration_path: Path | None = None
        self._calibration_thread = CalibrationThread(self)

        default_roi_geometry = _default_roi_geometry()
        roi_values: list[float] = []
        for key, fallback in zip(ROI_SETTINGS_KEYS, default_roi_geometry, strict=True):
            value = self._settings.value(key, fallback)
            try:
                roi_values.append(float(value))
            except (TypeError, ValueError):
                roi_values.append(fallback)
        self._set_roi_geometry(_clamp_roi_geometry(tuple(roi_values)))

        self.image_roi.sigRegionChangeFinished.connect(
            self._on_roi_region_change_finished
        )
        self.calibration_panel.load_calibration_button.clicked.connect(
            self._on_load_calibration_clicked
        )
        self.calibration_panel.save_calibration_button.clicked.connect(
            self._on_save_calibration_clicked
        )
        self.calibration_panel.calibration_details_button.clicked.connect(
            self._on_calibration_details_clicked
        )
        self.calibration_panel.new_calibration_button.clicked.connect(
            self._on_new_calibration_clicked
        )
        self._calibration_thread.sigCalibrationReady.connect(
            self._on_new_calibration_ready
        )
        self._calibration_thread.sigCalibrationFailed.connect(
            self._on_new_calibration_failed
        )
        self.calibration_panel.reset()

        self._server = MotorServer(self)
        self._server.sigMoveDetected.connect(self._on_move_detected)
        self._server.start()

    @staticmethod
    def _load_calibration_from_path(path: Path) -> xr.Dataset:
        with xr.open_dataset(path, engine="h5netcdf") as dataset_on_disk:
            calibration = dataset_on_disk.load()
        _validate_calibration_dataset(calibration)
        return calibration

    @QtCore.Slot(int)
    def _on_move_detected(self, target: int) -> None:
        self._server.set_result(True, "")

    @QtCore.Slot(object)
    def _on_roi_region_change_finished(self, roi: object | None = None) -> None:
        del roi
        position = self.image_roi.pos()
        size = self.image_roi.size()
        geometry = _clamp_roi_geometry(
            (
                float(position.x()),
                float(position.y()),
                float(size.x()),
                float(size.y()),
            )
        )
        self._set_roi_geometry(geometry)
        for key, value in zip(ROI_SETTINGS_KEYS, geometry, strict=True):
            self._settings.setValue(key, float(value))
        self._settings.sync()

    @QtCore.Slot()
    def _on_load_calibration_clicked(self) -> None:
        file_name, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Load calibration",
            "",
            "Calibration files (*.h5 *.hdf5 *.nc);;All files (*)",
        )
        if not file_name:
            return

        path = Path(file_name)
        try:
            calibration = self._load_calibration_from_path(path)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(
                self,
                "Could not load calibration",
                str(exc),
            )
            return

        self._calibration = calibration
        self._calibration_path = path
        self.calibration_panel.show_loaded_calibration(calibration, path.name)

    @QtCore.Slot()
    def _on_save_calibration_clicked(self) -> None:
        if self._calibration is None:
            return

        default_path = self._calibration_path or Path.home() / "calibration.h5"
        file_name, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save calibration",
            str(default_path),
            "Calibration files (*.h5 *.hdf5 *.nc);;All files (*)",
        )
        if not file_name:
            return

        path = Path(file_name)
        try:
            self._calibration.to_netcdf(path, engine="h5netcdf")
        except Exception as exc:
            QtWidgets.QMessageBox.critical(
                self,
                "Could not save calibration",
                str(exc),
            )
            return

        self._calibration_path = path
        self.calibration_panel.show_saved_calibration(path.name)

    @QtCore.Slot()
    def _on_new_calibration_clicked(self) -> None:
        if self._calibration_thread.isRunning():
            return

        self.calibration_panel.show_calibration_in_progress()
        self._calibration_thread.start()

    @QtCore.Slot(object)
    def _on_new_calibration_ready(self, calibration: object) -> None:
        try:
            if not isinstance(calibration, xr.Dataset):
                raise TypeError("calibration thread did not return an xarray Dataset")
            _validate_calibration_dataset(calibration)
        except Exception as exc:
            self._restore_calibration_idle_state()
            QtWidgets.QMessageBox.critical(
                self,
                "Could not use calibration",
                str(exc),
            )
            return

        self._calibration = calibration
        self._calibration_path = None
        self.calibration_panel.show_loaded_calibration(calibration, "new calibration")

    @QtCore.Slot(str)
    def _on_new_calibration_failed(self, error_message: str) -> None:
        self._restore_calibration_idle_state()
        QtWidgets.QMessageBox.critical(
            self,
            "Could not create calibration",
            error_message,
        )

    @QtCore.Slot()
    def _on_calibration_details_clicked(self) -> None:
        if self._calibration is None:
            return

        self.calibration_panel.build_details_dialog(self._calibration).exec()

    def _restore_calibration_idle_state(self) -> None:
        if self._calibration is None:
            self.calibration_panel.reset()
            return

        display_name = (
            self._calibration_path.name
            if self._calibration_path is not None
            else "current calibration"
        )
        self.calibration_panel.show_loaded_calibration(self._calibration, display_name)

    def _set_roi_geometry(self, geometry: tuple[float, float, float, float]) -> None:
        x, y, width, height = _clamp_roi_geometry(geometry)
        self.image_roi.setPos((x, y), update=False, finish=False)
        self.image_roi.setSize((width, height), update=True, finish=False)

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        self._calibration_thread.stop()
        self._calibration_thread.wait()

        self._server.stop()
        self._server.wait()

        super().closeEvent(event)


if __name__ == "__main__":
    multiprocessing.freeze_support()

    qapp = QtWidgets.QApplication(sys.argv)
    qapp.setStyle("Fusion")
    win = MainWindow()
    win.show()
    win.activateWindow()
    qapp.exec()

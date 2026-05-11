from __future__ import annotations

import math
import multiprocessing
import sys
import time
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pyqtgraph as pg
import xarray as xr
from qtpy import QtCore, QtGui, QtWidgets

from merlin_track_position.constants import (
    IMAGE_HEIGHT_CAM0,
    IMAGE_HEIGHT_CAM1,
    IMAGE_WIDTH_CAM0,
    IMAGE_WIDTH_CAM1,
)
from merlin_track_position.instruments.cameras import (
    RoiGeometry,
    make_cropped_camera_pair_capture,
)
from merlin_track_position.instruments.basler import get_basler_image
from merlin_track_position.instruments.framegrab import get_framegrabber_image
from merlin_track_position.interface.calibration_panel import (
    CalibrationPanel,
    _validate_calibration_dataset,
)
from merlin_track_position.interface.calibration_thread import CalibrationThread
from merlin_track_position.server import MotorServer
from merlin_track_position.tracking.calibrate import calibration_sample_count

__all__ = ("CalibrationStartDialog", "MainWindow")


CAMERA_IMAGE_SIZES: dict[str, tuple[int, int]] = {
    "cam0": (IMAGE_WIDTH_CAM0, IMAGE_HEIGHT_CAM0),
    "cam1": (IMAGE_WIDTH_CAM1, IMAGE_HEIGHT_CAM1),
}
DEFAULT_CALIBRATION_N = 5
DEFAULT_CALIBRATION_STEP_UM = 15.0
ROI_SETTINGS_KEYS: dict[str, tuple[str, str, str, str]] = {
    camera: (
        f"roi/{camera}/x",
        f"roi/{camera}/y",
        f"roi/{camera}/width",
        f"roi/{camera}/height",
    )
    for camera in CAMERA_IMAGE_SIZES
}
ROI_METADATA_KEYS: dict[str, tuple[str, str, str, str]] = {
    camera: (
        f"roi_{camera}_x",
        f"roi_{camera}_y",
        f"roi_{camera}_width",
        f"roi_{camera}_height",
    )
    for camera in CAMERA_IMAGE_SIZES
}


def _default_roi_geometry(
    image_width: float = IMAGE_WIDTH_CAM0,
    image_height: float = IMAGE_HEIGHT_CAM0,
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
    image_width: float = IMAGE_WIDTH_CAM0,
    image_height: float = IMAGE_HEIGHT_CAM0,
) -> tuple[float, float, float, float]:
    x, y, width, height = geometry
    if not all(math.isfinite(value) for value in geometry):
        return _default_roi_geometry(image_width, image_height)

    width = min(max(width, 1.0), image_width)
    height = min(max(height, 1.0), image_height)
    x = min(max(x, 0.0), image_width - width)
    y = min(max(y, 0.0), image_height - height)
    return (x, y, width, height)


def _roi_metadata_from_geometries(
    roi_geometries: Mapping[str, RoiGeometry],
) -> dict[str, float]:
    metadata: dict[str, float] = {}
    for camera, (image_width, image_height) in CAMERA_IMAGE_SIZES.items():
        geometry = _clamp_roi_geometry(
            tuple(float(value) for value in roi_geometries[camera]),
            image_width,
            image_height,
        )
        for key, value in zip(ROI_METADATA_KEYS[camera], geometry, strict=True):
            metadata[key] = float(value)
    return metadata


def _roi_geometries_from_calibration_metadata(
    calibration: xr.Dataset,
) -> dict[str, RoiGeometry] | None:
    attrs = calibration.attrs
    geometries: dict[str, RoiGeometry] = {}
    for camera, (image_width, image_height) in CAMERA_IMAGE_SIZES.items():
        keys = ROI_METADATA_KEYS[camera]
        if any(key not in attrs for key in keys):
            return None
        try:
            values = tuple(float(attrs[key]) for key in keys)
        except (TypeError, ValueError):
            return None
        if not all(math.isfinite(value) for value in values):
            return None
        geometries[camera] = _clamp_roi_geometry(
            values,
            image_width,
            image_height,
        )
    return geometries


class CalibrationStartDialog(QtWidgets.QDialog):
    def __init__(self, parent: QtWidgets.QWidget | None = None):
        super().__init__(parent)
        self.setWindowTitle("New Calibration")

        layout = QtWidgets.QVBoxLayout(self)
        form_layout = QtWidgets.QFormLayout()

        self.n_spin = QtWidgets.QSpinBox()
        self.n_spin.setObjectName("calibration_n_spin")
        self.n_spin.setRange(2, 101)
        self.n_spin.setValue(DEFAULT_CALIBRATION_N)
        form_layout.addRow("n", self.n_spin)

        self.step_um_spin = QtWidgets.QDoubleSpinBox()
        self.step_um_spin.setObjectName("calibration_step_um_spin")
        self.step_um_spin.setRange(0.001, 1_000_000.0)
        self.step_um_spin.setDecimals(3)
        self.step_um_spin.setSingleStep(1.0)
        self.step_um_spin.setSuffix(" um")
        self.step_um_spin.setValue(DEFAULT_CALIBRATION_STEP_UM)
        form_layout.addRow("step_um", self.step_um_spin)

        layout.addLayout(form_layout)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def parameters(self) -> tuple[int, float]:
        return int(self.n_spin.value()), float(self.step_um_spin.value())


class _MainWindowGUI(QtWidgets.QMainWindow):
    def __init__(self, parent: QtCore.QObject | None = None):
        super().__init__(parent)

        self.setWindowTitle("Track Positions")

        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QtWidgets.QVBoxLayout(central_widget)
        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        main_layout.addWidget(splitter)

        self.image_graphics_layout = pg.GraphicsLayoutWidget()
        self.image_plots: dict[str, pg.PlotItem] = {}
        self.image_items: dict[str, pg.ImageItem] = {}
        self.image_rois: dict[str, pg.RectROI] = {}
        for row, (camera, (image_width, image_height)) in enumerate(
            CAMERA_IMAGE_SIZES.items()
        ):
            image_plot = self.image_graphics_layout.addPlot(row=row, col=0)
            image_plot.setTitle(camera)
            image_plot.setAspectLocked(True)
            image_plot.setLabel(
                "bottom", "x", units="px", siPrefixEnableRanges=()
            )
            image_plot.setLabel("left", "y", units="px", siPrefixEnableRanges=())
            image_plot.showGrid(x=True, y=True, alpha=0.2)
            image_plot.invertY(True)

            image_item = pg.ImageItem(axisOrder="row-major")
            sample_img = np.ones((int(image_height), int(image_width)), dtype=np.int64)
            sample_img[0, 0] = 0
            image_item.setImage(sample_img)
            image_plot.addItem(image_item)
            image_plot.vb.setRange(
                rect=QtCore.QRectF(0, 0, image_width, image_height),
                padding=0,
            )

            roi_geometry = _default_roi_geometry(image_width, image_height)
            image_roi = pg.RectROI(
                roi_geometry[:2],
                roi_geometry[2:],
                sideScalers=True,
                maxBounds=QtCore.QRectF(0.0, 0.0, image_width, image_height),
                pen=pg.mkPen("#008c99", width=2),
                hoverPen=pg.mkPen("#00c2d1", width=2),
            )
            image_roi.addScaleHandle((0.0, 0.0), (1.0, 1.0))
            image_roi.addScaleHandle((1.0, 0.0), (0.0, 1.0))
            image_roi.addScaleHandle((0.0, 1.0), (1.0, 0.0))
            image_roi.addScaleHandle((0.5, 0.0), (0.5, 1.0))
            image_roi.addScaleHandle((0.0, 0.5), (1.0, 0.5))
            image_roi.setZValue(10)
            image_plot.addItem(image_roi)

            self.image_plots[camera] = image_plot
            self.image_items[camera] = image_item
            self.image_rois[camera] = image_roi

        splitter.addWidget(self.image_graphics_layout)

        self.calibration_panel = CalibrationPanel()
        splitter.addWidget(self.calibration_panel)

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
        self._calibration_total_steps = 0
        self._calibration_started_at: float | None = None

        for camera, (image_width, image_height) in CAMERA_IMAGE_SIZES.items():
            default_roi_geometry = _default_roi_geometry(image_width, image_height)
            roi_values: list[float] = []
            for key, fallback in zip(
                ROI_SETTINGS_KEYS[camera],
                default_roi_geometry,
                strict=True,
            ):
                value = self._settings.value(key, fallback)
                try:
                    roi_values.append(float(value))
                except (TypeError, ValueError):
                    roi_values.append(fallback)
            self._set_roi_geometry(
                camera,
                _clamp_roi_geometry(tuple(roi_values), image_width, image_height),
            )
            self.image_rois[camera].sigRegionChangeFinished.connect(
                lambda _roi=None, camera=camera: self._on_roi_region_change_finished(
                    camera
                )
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
        self._calibration_thread.sigCalibrationStep.connect(
            self._on_calibration_step
        )
        self.calibration_panel.reset()

        self._server = MotorServer(self)
        self._server.sigMoveDetected.connect(self._on_move_detected)
        self._server.start()

        QtCore.QTimer.singleShot(0, self._refresh_image)

    @staticmethod
    def _load_calibration_from_path(path: Path) -> xr.Dataset:
        with xr.open_dataset(path, engine="h5netcdf") as dataset_on_disk:
            calibration = dataset_on_disk.load()
        _validate_calibration_dataset(calibration)
        return calibration

    @QtCore.Slot(int)
    def _on_move_detected(self, target: int) -> None:
        # TODO: Estimate xyz displacement from both cameras and move x/y/z.
        del target
        self._server.set_result(True, "")

    def _on_roi_region_change_finished(self, camera: str) -> None:
        image_width, image_height = CAMERA_IMAGE_SIZES[camera]
        roi = self.image_rois[camera]
        position = roi.pos()
        size = roi.size()
        geometry = _clamp_roi_geometry(
            (
                float(position.x()),
                float(position.y()),
                float(size.x()),
                float(size.y()),
            ),
            image_width,
            image_height,
        )
        self._set_roi_geometry(camera, geometry)
        self._persist_roi_geometry(camera, geometry)

    @QtCore.Slot()
    def _refresh_image(self) -> None:
        self.image_items["cam0"].setImage(get_framegrabber_image())
        self.image_items["cam1"].setImage(get_basler_image())

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
        self._apply_calibration_roi_metadata(calibration)
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

        dialog = CalibrationStartDialog(self)
        if dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return

        n, step_um = dialog.parameters()
        roi_geometries = {
            camera: self._get_roi_geometry(camera) for camera in CAMERA_IMAGE_SIZES
        }
        roi_metadata = _roi_metadata_from_geometries(roi_geometries)
        image_generator = make_cropped_camera_pair_capture(
            roi_geometries["cam0"],
            roi_geometries["cam1"],
        )
        try:
            self._calibration_total_steps = calibration_sample_count(n)
            self._calibration_thread.configure(
                n,
                step_um,
                image_generator,
                roi_metadata,
            )
        except Exception as exc:
            QtWidgets.QMessageBox.critical(
                self,
                "Could not start calibration",
                str(exc),
            )
            return

        self._calibration_started_at = time.monotonic()
        self.calibration_panel.show_calibration_in_progress(
            self._calibration_total_steps
        )
        self._calibration_thread.start()

    @QtCore.Slot(int, float, float, float, object, object)
    def _on_calibration_step(
        self,
        idx: int,
        dx: float,
        dy: float,
        dz: float,
        image_cam0: object,
        image_cam1: object,
    ) -> None:
        del image_cam0, image_cam1
        total_steps = max(self._calibration_total_steps, int(idx) + 1, 1)
        started_at = self._calibration_started_at
        elapsed_s = 0.0 if started_at is None else time.monotonic() - started_at
        completed = min(max(int(idx) + 1, 1), total_steps)
        remaining = max(total_steps - completed, 0)
        eta_s = (
            (elapsed_s / completed) * remaining
            if completed > 0 and remaining > 0
            else 0.0
        )
        self.calibration_panel.show_calibration_step(
            idx=idx,
            total_steps=total_steps,
            dx=dx,
            dy=dy,
            dz=dz,
            elapsed_s=elapsed_s,
            eta_s=eta_s,
        )

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
        self._calibration_started_at = None
        self._calibration_total_steps = 0
        self.calibration_panel.show_loaded_calibration(calibration, "new calibration")

    @QtCore.Slot(str)
    def _on_new_calibration_failed(self, error_message: str) -> None:
        self._calibration_started_at = None
        self._calibration_total_steps = 0
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
        self._calibration_started_at = None
        self._calibration_total_steps = 0
        if self._calibration is None:
            self.calibration_panel.reset()
            return

        display_name = (
            self._calibration_path.name
            if self._calibration_path is not None
            else "current calibration"
        )
        self.calibration_panel.show_loaded_calibration(self._calibration, display_name)

    def _get_roi_geometry(self, camera: str) -> RoiGeometry:
        image_width, image_height = CAMERA_IMAGE_SIZES[camera]
        image_roi = self.image_rois[camera]
        position = image_roi.pos()
        size = image_roi.size()
        return _clamp_roi_geometry(
            (
                float(position.x()),
                float(position.y()),
                float(size.x()),
                float(size.y()),
            ),
            image_width,
            image_height,
        )

    def _set_roi_geometry(
        self,
        camera: str,
        geometry: tuple[float, float, float, float],
    ) -> None:
        image_width, image_height = CAMERA_IMAGE_SIZES[camera]
        x, y, width, height = _clamp_roi_geometry(
            geometry,
            image_width,
            image_height,
        )
        image_roi = self.image_rois[camera]
        image_roi.setPos((x, y), update=False, finish=False)
        image_roi.setSize((width, height), update=True, finish=False)

    def _persist_roi_geometry(
        self,
        camera: str,
        geometry: RoiGeometry,
    ) -> None:
        for key, value in zip(ROI_SETTINGS_KEYS[camera], geometry, strict=True):
            self._settings.setValue(key, float(value))
        self._settings.sync()

    def _apply_calibration_roi_metadata(self, calibration: xr.Dataset) -> bool:
        roi_geometries = _roi_geometries_from_calibration_metadata(calibration)
        if roi_geometries is None:
            return False
        for camera, geometry in roi_geometries.items():
            self._set_roi_geometry(camera, geometry)
            self._persist_roi_geometry(camera, geometry)
        return True

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

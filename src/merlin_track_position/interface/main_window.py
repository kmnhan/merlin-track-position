from __future__ import annotations

import logging
import math
import multiprocessing
import sys
import threading
import time
from collections.abc import Callable, Mapping
from pathlib import Path

import numpy as np
import pyqtgraph as pg
import xarray as xr
from qtpy import QtCore, QtGui, QtWidgets

from merlin_track_position.constants import (
    DEFAULT_VISUAL_CALIBRATION_MIN_SHIFT_PX,
    DEFAULT_VISUAL_CALIBRATION_REPEATS_PER_DIRECTION,
    DEFAULT_VISUAL_CALIBRATION_STEP_MM_BY_AXIS,
    IMAGE_HEIGHT_CAM0,
    IMAGE_HEIGHT_CAM1,
    IMAGE_WIDTH_CAM0,
    IMAGE_WIDTH_CAM1,
)
from merlin_track_position.instruments.cameras import (
    CallableCameraPlugin,
    CameraPairPlugin,
    RoiGeometry,
)
from merlin_track_position.instruments.basler import (
    close_basler_camera,
    get_basler_image,
)
from merlin_track_position.instruments.framegrab import get_framegrabber_image
from merlin_track_position.interface.calibration_panel import CalibrationPanel
from merlin_track_position.interface.calibration_thread import CalibrationThread
from merlin_track_position.server import MotorServer
from merlin_track_position.tracking.calibrate import visual_calibration_probe_count
from merlin_track_position.tracking.calibration_core import (
    load_calibration_dataset,
    save_calibration_dataset,
    validate_visual_calibration_dataset,
)

__all__ = ("CalibrationStartDialog", "MainWindow")

logger = logging.getLogger("merlin_track_position.interface.main_window")

CAMERA_IMAGE_SIZES: dict[str, tuple[int, int]] = {
    "cam0": (IMAGE_WIDTH_CAM0, IMAGE_HEIGHT_CAM0),
    "cam1": (IMAGE_WIDTH_CAM1, IMAGE_HEIGHT_CAM1),
}
IMAGE_REFRESH_INTERVAL_MS = 400
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
        self.setWindowTitle("New Visual-Jacobian Calibration")

        layout = QtWidgets.QVBoxLayout(self)
        form_layout = QtWidgets.QFormLayout()

        step_text = ", ".join(
            f"{axis}={value:g} mm"
            for axis, value in DEFAULT_VISUAL_CALIBRATION_STEP_MM_BY_AXIS.items()
        )
        form_layout.addRow("Probe steps (cmd mm)", QtWidgets.QLabel(step_text))
        form_layout.addRow(
            "Repeats",
            QtWidgets.QLabel(str(DEFAULT_VISUAL_CALIBRATION_REPEATS_PER_DIRECTION)),
        )
        form_layout.addRow(
            "Minimum image response",
            QtWidgets.QLabel(f"{DEFAULT_VISUAL_CALIBRATION_MIN_SHIFT_PX:g} px"),
        )

        path_row = QtWidgets.QHBoxLayout()
        self.path_edit = QtWidgets.QLineEdit()
        self.path_edit.setObjectName("calibration_output_path_edit")
        self.path_edit.setText(str(Path.home() / "visual_jacobian_calibration.h5"))
        browse_button = QtWidgets.QPushButton("Browse...")
        browse_button.clicked.connect(self._browse_output_path)
        path_row.addWidget(self.path_edit, stretch=1)
        path_row.addWidget(browse_button)
        form_layout.addRow("Save to", path_row)

        layout.addLayout(form_layout)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _browse_output_path(self) -> None:
        file_name, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save visual-Jacobian calibration",
            self.path_edit.text(),
            "Calibration files (*.h5 *.hdf5 *.nc);;All files (*)",
        )
        if file_name:
            self.path_edit.setText(file_name)

    def accept(self) -> None:
        if not self.path_edit.text().strip():
            QtWidgets.QMessageBox.warning(
                self,
                "Visual-Jacobian calibration path required",
                "Choose a file path for the calibration dataset.",
            )
            return
        super().accept()

    def output_path(self) -> Path:
        return Path(self.path_edit.text()).expanduser()


class _ImageCaptureThread(QtCore.QThread):
    sigImageReady = QtCore.Signal(str, object)
    sigImageCaptureFailed = QtCore.Signal(str, str)

    def __init__(
        self,
        camera: str,
        image_capture: Callable[[], np.ndarray],
        interval_ms: int,
        parent: QtCore.QObject | None = None,
    ):
        super().__init__(parent)
        self.camera = str(camera)
        self._image_capture = image_capture
        self.interval_ms = int(interval_ms)
        self._running = threading.Event()
        self._enabled = threading.Event()
        self._wake = threading.Event()

    def set_enabled(self, enabled: bool) -> None:
        if enabled:
            self._enabled.set()
        else:
            self._enabled.clear()
        self._wake.set()

    def is_enabled(self) -> bool:
        return self._enabled.is_set()

    def run(self) -> None:
        self._running.set()
        try:
            while self._running.is_set() and not self.isInterruptionRequested():
                if not self._enabled.is_set():
                    self._wake.wait()
                    self._wake.clear()
                    continue

                self._wake.clear()
                try:
                    image = self._image_capture()
                except Exception as exc:
                    self.sigImageCaptureFailed.emit(self.camera, str(exc))
                else:
                    self.sigImageReady.emit(self.camera, image)

                self._wake.wait(self.interval_ms / 1000.0)
                self._wake.clear()
        finally:
            self._running.clear()

    def stop(self) -> None:
        self._running.clear()
        self.requestInterruption()
        self._wake.set()


class _MainWindowGUI(QtWidgets.QMainWindow):
    def __init__(self, parent: QtCore.QObject | None = None):
        super().__init__(parent)

        self.setWindowTitle("Track Positions")

        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QtWidgets.QVBoxLayout(central_widget)
        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        main_layout.addWidget(splitter)

        image_widget = QtWidgets.QWidget()
        image_layout = QtWidgets.QVBoxLayout(image_widget)
        image_layout.setContentsMargins(0, 0, 0, 0)

        self.image_auto_refresh_checkbox = QtWidgets.QCheckBox("Update images")
        self.image_auto_refresh_checkbox.setObjectName("image_auto_refresh_checkbox")
        self.image_auto_refresh_checkbox.setChecked(True)
        image_layout.addWidget(self.image_auto_refresh_checkbox)

        self.image_graphics_layout = pg.GraphicsLayoutWidget()
        image_layout.addWidget(self.image_graphics_layout)
        self.image_plots: dict[str, pg.PlotItem] = {}
        self.image_items: dict[str, pg.ImageItem] = {}
        self.image_rois: dict[str, pg.RectROI] = {}
        for row, (camera, (image_width, image_height)) in enumerate(
            CAMERA_IMAGE_SIZES.items()
        ):
            image_plot = self.image_graphics_layout.addPlot(row=row, col=0)
            image_plot.setTitle(camera)
            image_plot.setAspectLocked(True)
            image_plot.setLabel("bottom", "x", units="px", siPrefixEnableRanges=())
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

        splitter.addWidget(image_widget)

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
        self._calibration_processing_started_at: float | None = None
        self._roi_editing_enabled = True
        self._latest_images: tuple[np.ndarray, np.ndarray] | None = None
        self._latest_images_by_camera: dict[str, np.ndarray] = {}
        self._image_capture_locks = {
            "cam0": threading.Lock(),
            "cam1": threading.Lock(),
        }
        self._image_refresh_threads = {
            "cam0": _ImageCaptureThread(
                "cam0",
                self._capture_cam0_image,
                IMAGE_REFRESH_INTERVAL_MS,
                self,
            ),
            "cam1": _ImageCaptureThread(
                "cam1",
                self._capture_cam1_image,
                IMAGE_REFRESH_INTERVAL_MS,
                self,
            ),
        }
        for thread in self._image_refresh_threads.values():
            thread.sigImageReady.connect(self._on_image_capture_ready)
            thread.sigImageCaptureFailed.connect(self._on_image_capture_failed)
            thread.start()
        self._image_auto_refresh_checked_before_calibration: bool | None = None

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
        self._calibration_thread.sigCalibrationStep.connect(self._on_calibration_step)
        self._calibration_thread.sigCalibrationProcessingStep.connect(
            self._on_calibration_processing_step
        )
        self.image_auto_refresh_checkbox.toggled.connect(
            self._on_image_auto_refresh_toggled
        )
        self.calibration_panel.reset()
        self._set_roi_editing_enabled(True)

        self._server = MotorServer(self)
        self._server.sigMoveDetected.connect(self._on_move_detected)
        self._server.start()

        self._on_image_auto_refresh_toggled(
            self.image_auto_refresh_checkbox.isChecked()
        )

    @staticmethod
    def _load_calibration_from_path(path: Path) -> xr.Dataset:
        return load_calibration_dataset(path)

    @QtCore.Slot(int)
    def _on_move_detected(self, target: int) -> None:
        # TODO: Estimate xyz displacement from both cameras and move x/y/z.
        del target
        self._server.set_result(True, "")

    def _set_roi_editing_enabled(self, enabled: bool) -> None:
        enabled = bool(enabled)
        self._roi_editing_enabled = enabled
        for roi in self.image_rois.values():
            roi.translatable = enabled
            for handle in roi.getHandles():
                handle.setVisible(enabled)

    def _on_roi_region_change_finished(self, camera: str) -> None:
        if self._calibration is not None or not self._roi_editing_enabled:
            return

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

    def _capture_cam0_image(self) -> np.ndarray:
        with self._image_capture_locks["cam0"]:
            return get_framegrabber_image()

    def _capture_cam1_image(self) -> np.ndarray:
        with self._image_capture_locks["cam1"]:
            return get_basler_image()

    @QtCore.Slot(str, object)
    def _on_image_capture_ready(self, camera: str, image: object) -> None:
        if camera not in self.image_items:
            logger.warning("Image refresh returned unknown camera %s", camera)
            return

        self._latest_images_by_camera[camera] = image
        if {"cam0", "cam1"}.issubset(self._latest_images_by_camera):
            self._latest_images = (
                self._latest_images_by_camera["cam0"],
                self._latest_images_by_camera["cam1"],
            )
        self.image_items[camera].setImage(image)

    @QtCore.Slot(str, str)
    def _on_image_capture_failed(self, camera: str, error_message: str) -> None:
        logger.warning("Image refresh failed for %s: %s", camera, error_message)

    @QtCore.Slot(bool)
    def _on_image_auto_refresh_toggled(self, enabled: bool) -> None:
        self._set_image_refresh_enabled(enabled)

    def _set_image_refresh_enabled(self, enabled: bool) -> None:
        for thread in self._image_refresh_threads.values():
            thread.set_enabled(enabled)

    def _pause_image_auto_refresh_for_calibration(self) -> None:
        self._image_auto_refresh_checked_before_calibration = (
            self.image_auto_refresh_checkbox.isChecked()
        )
        self._set_image_refresh_enabled(False)
        self.image_auto_refresh_checkbox.setEnabled(False)

    def _restore_image_auto_refresh_after_calibration(self) -> None:
        restore_checked = self._image_auto_refresh_checked_before_calibration
        self._image_auto_refresh_checked_before_calibration = None
        if restore_checked is None:
            restore_checked = self.image_auto_refresh_checkbox.isChecked()

        was_blocked = self.image_auto_refresh_checkbox.blockSignals(True)
        self.image_auto_refresh_checkbox.setChecked(restore_checked)
        self.image_auto_refresh_checkbox.blockSignals(was_blocked)
        self.image_auto_refresh_checkbox.setEnabled(True)

        self._set_image_refresh_enabled(restore_checked)

    @QtCore.Slot()
    def _on_load_calibration_clicked(self) -> None:
        file_name, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Load visual-Jacobian calibration",
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

        self._apply_calibration_roi_metadata(calibration)
        self._calibration = calibration
        self._calibration_path = path
        self._set_roi_editing_enabled(False)
        self.calibration_panel.show_loaded_calibration(calibration, path.name)

    @QtCore.Slot()
    def _on_save_calibration_clicked(self) -> None:
        if self._calibration is None:
            return

        default_path = (
            self._calibration_path or Path.home() / "visual_jacobian_calibration.h5"
        )
        file_name, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save visual-Jacobian calibration",
            str(default_path),
            "Calibration files (*.h5 *.hdf5 *.nc);;All files (*)",
        )
        if not file_name:
            return

        path = Path(file_name)
        try:
            save_calibration_dataset(self._calibration, path)
            self._calibration = load_calibration_dataset(path)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(
                self,
                "Could not save calibration",
                str(exc),
            )
            return

        self._calibration_path = path
        self._set_roi_editing_enabled(False)
        self.calibration_panel.show_saved_calibration(path.name)

    @QtCore.Slot()
    def _on_new_calibration_clicked(self) -> None:
        if self._calibration_thread.isRunning():
            return
        if self._calibration is not None:
            self._clear_loaded_calibration()
            return

        dialog = CalibrationStartDialog(self)
        if dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return

        output_path = dialog.output_path()
        roi_geometries = {
            camera: self._get_roi_geometry(camera) for camera in CAMERA_IMAGE_SIZES
        }
        roi_metadata = _roi_metadata_from_geometries(roi_geometries)
        camera_pair = CameraPairPlugin(
            CallableCameraPlugin("cam0", self._capture_cam0_image),
            CallableCameraPlugin("cam1", self._capture_cam1_image),
        ).cropped(roi_geometries["cam0"], roi_geometries["cam1"])
        try:
            self._calibration_total_steps = visual_calibration_probe_count()
            self._calibration_thread.configure(
                camera_pair,
                roi_metadata,
                output_path,
            )
        except Exception as exc:
            QtWidgets.QMessageBox.critical(
                self,
                "Could not start calibration",
                str(exc),
            )
            return

        self._pause_image_auto_refresh_for_calibration()
        self._set_roi_editing_enabled(False)
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
        self._on_image_capture_ready("cam0", image_cam0)
        self._on_image_capture_ready("cam1", image_cam1)

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

    @QtCore.Slot(int, int)
    def _on_calibration_processing_step(self, completed: int, total: int) -> None:
        total = max(int(total), 1)
        completed = min(max(int(completed), 0), total)
        if completed == 0 or self._calibration_processing_started_at is None:
            self._calibration_processing_started_at = time.monotonic()

        elapsed_s = time.monotonic() - self._calibration_processing_started_at
        remaining = max(total - completed, 0)
        eta_s = (
            (elapsed_s / completed) * remaining
            if completed > 0 and remaining > 0
            else 0.0 if remaining == 0 else None
        )
        self.calibration_panel.show_calibration_processing(
            completed=completed,
            total=total,
            elapsed_s=elapsed_s,
            eta_s=eta_s,
        )

    @QtCore.Slot(object)
    def _on_new_calibration_ready(self, calibration: object) -> None:
        self._restore_image_auto_refresh_after_calibration()
        try:
            if not isinstance(calibration, xr.Dataset):
                raise TypeError("calibration thread did not return an xarray Dataset")
            validate_visual_calibration_dataset(calibration)
        except Exception as exc:
            self._restore_calibration_idle_state()
            QtWidgets.QMessageBox.critical(
                self,
                "Could not use calibration",
                str(exc),
            )
            return

        path_value = calibration.attrs.get("calibration_path")
        calibration_path = Path(str(path_value)) if path_value else None
        self._calibration_started_at = None
        self._calibration_processing_started_at = None
        self._calibration_total_steps = 0
        self._apply_calibration_roi_metadata(calibration, persist=False)
        self._calibration = calibration
        self._calibration_path = calibration_path
        self._set_roi_editing_enabled(False)
        display_name = (
            self._calibration_path.name
            if self._calibration_path is not None
            else "new calibration"
        )
        self.calibration_panel.show_loaded_calibration(calibration, display_name)

    @QtCore.Slot(str)
    def _on_new_calibration_failed(self, error_message: str) -> None:
        self._restore_image_auto_refresh_after_calibration()
        self._calibration_started_at = None
        self._calibration_processing_started_at = None
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
        self._calibration_processing_started_at = None
        self._calibration_total_steps = 0
        if self._calibration is None:
            self.calibration_panel.reset()
            self._set_roi_editing_enabled(True)
            return

        self._set_roi_editing_enabled(False)
        display_name = (
            self._calibration_path.name
            if self._calibration_path is not None
            else "current calibration"
        )
        self.calibration_panel.show_loaded_calibration(self._calibration, display_name)

    def _clear_loaded_calibration(self) -> None:
        self._calibration = None
        self._calibration_path = None
        self._restore_calibration_idle_state()

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

    def _apply_calibration_roi_metadata(
        self,
        calibration: xr.Dataset,
        *,
        persist: bool = True,
    ) -> bool:
        roi_geometries = _roi_geometries_from_calibration_metadata(calibration)
        if roi_geometries is None:
            return False
        for camera, geometry in roi_geometries.items():
            self._set_roi_geometry(camera, geometry)
            if persist:
                self._persist_roi_geometry(camera, geometry)
        return True

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        for thread in self._image_refresh_threads.values():
            thread.stop()
        for thread in self._image_refresh_threads.values():
            thread.wait()

        self._calibration_thread.stop()
        self._calibration_thread.wait()

        close_basler_camera()

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

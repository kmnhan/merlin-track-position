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
    DEFAULT_CORRECTION_MODE,
    DEFAULT_VISUAL_CALIBRATION_N,
    DEFAULT_VISUAL_CALIBRATION_MIN_SHIFT_PX,
    DEFAULT_VISUAL_CALIBRATION_STEP_UM,
    IMAGE_HEIGHT_CAM0,
    IMAGE_HEIGHT_CAM1,
    IMAGE_WIDTH_CAM0,
    IMAGE_WIDTH_CAM1,
    IS_DAQ_PC,
)
from merlin_track_position.instruments.parse_config import get_base_file_dir
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
from merlin_track_position.interface.correction_thread import CorrectionThread
from merlin_track_position.interface.detection_thread import DetectShiftThread
from merlin_track_position.interface.registration_settings import (
    normalized_registration_config,
    registration_config_from_settings,
    registration_config_to_shift_kwargs,
)
from merlin_track_position.interface.shift_monitor_window import ShiftMonitorWindow
from merlin_track_position.server import MotorServer
from merlin_track_position.tracking.calibrate import visual_calibration_probe_count
from merlin_track_position.tracking.calibration_core import (
    flush_pending_calibration_datasets,
    load_calibration_dataset,
    save_calibration_dataset_deferred,
    validate_visual_calibration_dataset,
)
from merlin_track_position.tracking.correct import (
    flush_pending_correction_history_datasets,
    load_latest_correction_history_dataset,
)
from merlin_track_position.tracking.persistence import (
    pending_entry_count,
    persistence_result_attrs,
)

__all__ = ("CalibrationStartDialog", "MainWindow")

logger = logging.getLogger("merlin_track_position.interface.main_window")
DEFAULT_CALIBRATION_FILE_NAME = "calibration.h5"


def _default_calibration_directory() -> Path:
    try:
        return get_base_file_dir().expanduser()
    except Exception:
        if not IS_DAQ_PC:
            logger.info(
                "Could not read scan base file directory; using home directory.",
                exc_info=True,
            )
            return Path.home()
        raise


def _default_calibration_path() -> Path:
    return _default_calibration_directory() / DEFAULT_CALIBRATION_FILE_NAME


class _CorrectionUnavailable(RuntimeError):
    """Expected state that prevents a correction run from starting."""


CAMERA_IMAGE_SIZES: dict[str, tuple[int, int]] = {
    "cam0": (IMAGE_WIDTH_CAM0, IMAGE_HEIGHT_CAM0),
    "cam1": (IMAGE_WIDTH_CAM1, IMAGE_HEIGHT_CAM1),
}
TRANSPOSED_IMAGE_DISPLAY_CAMERAS = frozenset({"cam1"})
IMAGE_REFRESH_INTERVAL_MS = 400
PERSISTENCE_FLUSH_INTERVAL_MS = 5000
DEFAULT_AUTO_CORRECTION_INTERVAL_SECONDS = 180.0
AUTO_CORRECTION_INTERVAL_SETTINGS_KEY = "auto_correction/interval_seconds"
CORRECTION_MODE_SETTINGS_KEY = "correction/mode"
LEGACY_AUTO_CORRECTION_INTERVAL_MS_SETTINGS_KEY = "auto_correction/interval_ms"
LEGACY_AUTO_CORRECTION_INTERVAL_MINUTES_SETTINGS_KEY = (
    "auto_correction/interval_minutes"
)
AUTO_CORRECTION_INTERVAL_MS_PER_SECOND = 1_000
AUTO_CORRECTION_INTERVAL_MS_PER_MINUTE = 60_000
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
ROI_SCALE_HANDLES: tuple[
    tuple[tuple[float, float], tuple[float, float]],
    ...,
] = (
    ((1.0, 1.0), (0.0, 0.0)),
    ((1.0, 0.5), (0.0, 0.5)),
    ((0.5, 1.0), (0.5, 0.0)),
    ((0.0, 0.0), (1.0, 1.0)),
    ((1.0, 0.0), (0.0, 1.0)),
    ((0.0, 1.0), (1.0, 0.0)),
    ((0.5, 0.0), (0.5, 1.0)),
    ((0.0, 0.5), (1.0, 0.5)),
)


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


def _display_geometry(
    camera: str,
    geometry: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    x, y, width, height = geometry
    if camera in TRANSPOSED_IMAGE_DISPLAY_CAMERAS:
        return (y, x, height, width)
    return (x, y, width, height)


def _raw_geometry_from_display(
    camera: str,
    geometry: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    return _display_geometry(camera, geometry)


def _display_image_size(
    camera: str,
    image_width: float,
    image_height: float,
) -> tuple[float, float]:
    _x, _y, display_width, display_height = _display_geometry(
        camera,
        (0.0, 0.0, image_width, image_height),
    )
    return display_width, display_height


def _raw_rect_from_display_rect(camera: str, rect: QtCore.QRectF) -> QtCore.QRectF:
    return QtCore.QRectF(
        *_raw_geometry_from_display(
            camera,
            (rect.x(), rect.y(), rect.width(), rect.height()),
        )
    )


def _full_raw_image_rect(camera: str) -> QtCore.QRectF:
    image_width, image_height = CAMERA_IMAGE_SIZES[camera]
    return QtCore.QRectF(0.0, 0.0, float(image_width), float(image_height))


def _set_image_item_raw_rect(
    camera: str,
    image_item: pg.ImageItem,
    raw_rect: QtCore.QRectF,
) -> None:
    image_width = float(image_item.width() or 1.0)
    image_height = float(image_item.height() or 1.0)
    u_scale = raw_rect.width() / image_width
    v_scale = raw_rect.height() / image_height
    if camera in TRANSPOSED_IMAGE_DISPLAY_CAMERAS:
        image_item.setTransform(
            QtGui.QTransform(
                0.0,
                u_scale,
                v_scale,
                0.0,
                raw_rect.y(),
                raw_rect.x(),
            )
        )
        return

    transform = QtGui.QTransform()
    transform.translate(raw_rect.x(), raw_rect.y())
    transform.scale(u_scale, v_scale)
    image_item.setTransform(transform)


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


def _add_roi_scale_handles(roi: pg.ROI) -> None:
    if len(roi.getHandles()) == len(ROI_SCALE_HANDLES):
        return

    _remove_roi_scale_handles(roi)
    for position, center in ROI_SCALE_HANDLES:
        roi.addScaleHandle(position, center)


def _remove_roi_scale_handles(roi: pg.ROI) -> None:
    for handle in list(roi.getHandles()):
        roi.removeHandle(handle)


class CalibrationStartDialog(QtWidgets.QDialog):
    def __init__(
        self,
        parent: QtWidgets.QWidget | None = None,
        default_output_path: Path | None = None,
    ):
        super().__init__(parent)
        self.setWindowTitle("New Calibration")

        layout = QtWidgets.QVBoxLayout(self)
        form_layout = QtWidgets.QFormLayout()

        self.n_spin = QtWidgets.QSpinBox()
        self.n_spin.setObjectName("calibration_n_spin")
        self.n_spin.setRange(2, 101)
        self.n_spin.setValue(DEFAULT_VISUAL_CALIBRATION_N)
        form_layout.addRow("N", self.n_spin)

        self.step_um_spin = QtWidgets.QDoubleSpinBox()
        self.step_um_spin.setObjectName("calibration_step_um_spin")
        self.step_um_spin.setRange(0.001, 1_000_000.0)
        self.step_um_spin.setDecimals(3)
        self.step_um_spin.setSingleStep(1.0)
        self.step_um_spin.setSuffix(" um")
        self.step_um_spin.setValue(DEFAULT_VISUAL_CALIBRATION_STEP_UM)
        form_layout.addRow("Step", self.step_um_spin)

        form_layout.addRow(
            "Minimum image response",
            QtWidgets.QLabel(f"{DEFAULT_VISUAL_CALIBRATION_MIN_SHIFT_PX:g} px"),
        )

        path_row = QtWidgets.QHBoxLayout()
        self.path_edit = QtWidgets.QLineEdit()
        self.path_edit.setObjectName("calibration_output_path_edit")
        if default_output_path is None:
            default_output_path = Path.home() / DEFAULT_CALIBRATION_FILE_NAME
        self.path_edit.setText(str(default_output_path))
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
            "Save calibration",
            self.path_edit.text(),
            "Calibration files (*.h5 *.hdf5 *.nc);;All files (*)",
        )
        if file_name:
            self.path_edit.setText(file_name)

    def accept(self) -> None:
        if not self.path_edit.text().strip():
            QtWidgets.QMessageBox.warning(
                self,
                "Calibration path required",
                "Choose a file path for the calibration dataset.",
            )
            return
        super().accept()

    def output_path(self) -> Path:
        return Path(self.path_edit.text()).expanduser()

    def parameters(self) -> tuple[int, float]:
        return int(self.n_spin.value()), float(self.step_um_spin.value())


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
        tools_menu = self.menuBar().addMenu("Tools")
        self.shift_monitor_action = QtGui.QAction("Shift Monitor", self)
        self.shift_monitor_action.setObjectName("shift_monitor_action")
        tools_menu.addAction(self.shift_monitor_action)

        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QtWidgets.QVBoxLayout(central_widget)
        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        main_layout.addWidget(splitter)

        image_widget = QtWidgets.QWidget()
        image_layout = QtWidgets.QVBoxLayout(image_widget)
        image_layout.setContentsMargins(0, 0, 0, 0)

        image_controls_layout = QtWidgets.QHBoxLayout()
        self.image_auto_refresh_checkbox = QtWidgets.QCheckBox("Update images")
        self.image_auto_refresh_checkbox.setObjectName("image_auto_refresh_checkbox")
        self.image_auto_refresh_checkbox.setChecked(True)
        image_controls_layout.addWidget(self.image_auto_refresh_checkbox)
        self.show_reference_images_button = QtWidgets.QPushButton("Reference")
        self.show_reference_images_button.setObjectName("show_reference_images_button")
        self.show_reference_images_button.setEnabled(False)
        image_controls_layout.addWidget(self.show_reference_images_button)
        image_controls_layout.addStretch(1)
        image_layout.addLayout(image_controls_layout)

        self.image_graphics_layout = pg.GraphicsLayoutWidget()
        image_layout.addWidget(self.image_graphics_layout)
        self.image_plots: dict[str, pg.PlotItem] = {}
        self.image_items: dict[str, pg.ImageItem] = {}
        self.image_rois: dict[str, pg.ROI] = {}
        self._image_raw_rects: dict[str, QtCore.QRectF] = {}
        for row, (camera, (image_width, image_height)) in enumerate(
            CAMERA_IMAGE_SIZES.items()
        ):
            image_plot = self.image_graphics_layout.addPlot(row=row, col=0)
            image_plot.setTitle(camera)
            image_plot.setAspectLocked(True)
            bottom_label, left_label = (
                ("v", "u")
                if camera in TRANSPOSED_IMAGE_DISPLAY_CAMERAS
                else ("u", "v")
            )
            image_plot.setLabel(
                "bottom",
                bottom_label,
                units="px",
                siPrefixEnableRanges=(),
            )
            image_plot.setLabel(
                "left",
                left_label,
                units="px",
                siPrefixEnableRanges=(),
            )
            image_plot.showGrid(x=True, y=True, alpha=0.2)
            image_plot.invertX(camera in TRANSPOSED_IMAGE_DISPLAY_CAMERAS)
            image_plot.invertY(True)

            image_item = pg.ImageItem(axisOrder="row-major")
            sample_img = np.ones(
                (int(image_height), int(image_width)),
                dtype=np.int64,
            )
            sample_img[0, 0] = 0
            image_item.setImage(sample_img)
            raw_rect = _full_raw_image_rect(camera)
            _set_image_item_raw_rect(camera, image_item, raw_rect)
            image_plot.addItem(image_item)
            display_width, display_height = _display_image_size(
                camera,
                image_width,
                image_height,
            )
            image_plot.vb.setRange(
                rect=QtCore.QRectF(0, 0, display_width, display_height),
                padding=0,
            )

            roi_geometry = _default_roi_geometry(image_width, image_height)
            display_roi_geometry = _display_geometry(camera, roi_geometry)
            image_roi = pg.ROI(
                display_roi_geometry[:2],
                display_roi_geometry[2:],
                maxBounds=QtCore.QRectF(
                    0.0,
                    0.0,
                    display_width,
                    display_height,
                ),
                pen=pg.mkPen("#008c99", width=2),
                hoverPen=pg.mkPen("#00c2d1", width=2),
            )
            _add_roi_scale_handles(image_roi)
            image_roi.setZValue(10)
            image_plot.addItem(image_roi)

            self.image_plots[camera] = image_plot
            self.image_items[camera] = image_item
            self.image_rois[camera] = image_roi
            self._image_raw_rects[camera] = QtCore.QRectF(raw_rect)

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
        self._registration_config = registration_config_from_settings(self._settings)
        self._shift_monitor_window: ShiftMonitorWindow | None = None
        self.calibration_panel.auto_correction_interval_spinbox.setValue(
            self._stored_auto_correction_interval_seconds()
        )
        self.calibration_panel.set_correction_mode(self._stored_correction_mode())
        self._calibration: xr.Dataset | None = None
        self._calibration_path: Path | None = None
        self._calibration_thread = CalibrationThread(self)
        self._correction_thread = CorrectionThread(self)
        self._detect_shift_thread = DetectShiftThread(self)
        self._calibration_total_steps = 0
        self._calibration_started_at: float | None = None
        self._calibration_processing_started_at: float | None = None
        self._roi_editing_enabled = True
        self._last_correction_result: xr.Dataset | None = None
        self._server_correction_pending = False
        self._server_correction_target: int | None = None
        self._latest_images: tuple[np.ndarray, np.ndarray] | None = None
        self._latest_images_by_camera: dict[str, np.ndarray] = {}
        self._reference_preview_active = False
        self._reference_preview_restore_state: dict[
            str,
            tuple[np.ndarray, QtCore.QRectF],
        ] = {}
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
        self._auto_correction_timer = QtCore.QTimer(self)
        self._auto_correction_timer.setSingleShot(False)
        self._auto_correction_timer.timeout.connect(self._on_auto_correction_timeout)

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
        self.calibration_panel.correct_sample_button.clicked.connect(
            self._on_correct_sample_clicked
        )
        self.calibration_panel.auto_correction_checkbox.toggled.connect(
            self._on_auto_correction_toggled
        )
        self.calibration_panel.auto_correction_interval_spinbox.valueChanged.connect(
            self._on_auto_correction_interval_changed
        )
        self.calibration_panel.correction_mode_combo.currentIndexChanged.connect(
            self._on_correction_mode_changed
        )
        self.calibration_panel.detect_shift_button.clicked.connect(
            self._on_detect_shift_clicked
        )
        self.calibration_panel.new_calibration_button.clicked.connect(
            self._on_new_calibration_clicked
        )
        self.shift_monitor_action.triggered.connect(self._on_shift_monitor_triggered)
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
        self._correction_thread.sigCorrectionProgress.connect(
            self._on_correction_progress
        )
        self._correction_thread.sigCorrectionReady.connect(self._on_correction_ready)
        self._correction_thread.sigCorrectionFailed.connect(self._on_correction_failed)
        self._detect_shift_thread.sigDetectionReady.connect(
            self._on_detect_shift_ready
        )
        self._detect_shift_thread.sigDetectionFailed.connect(
            self._on_detect_shift_failed
        )
        self.image_auto_refresh_checkbox.toggled.connect(
            self._on_image_auto_refresh_toggled
        )
        self.show_reference_images_button.pressed.connect(
            self._on_show_reference_images_pressed
        )
        self.show_reference_images_button.released.connect(
            self._on_show_reference_images_released
        )
        self.calibration_panel.reset()
        self._set_reference_preview_button_enabled(False)
        self._set_roi_editing_enabled(True)

        self._server = MotorServer(self)
        self._server.sigMoveDetected.connect(self._on_move_detected)
        self._server.start()

        self._persistence_flush_timer = QtCore.QTimer(self)
        self._persistence_flush_timer.setInterval(PERSISTENCE_FLUSH_INTERVAL_MS)
        self._persistence_flush_timer.timeout.connect(self._flush_pending_persistence)
        self._flush_pending_persistence()

        self._on_image_auto_refresh_toggled(
            self.image_auto_refresh_checkbox.isChecked()
        )

    @staticmethod
    def _load_calibration_from_path(path: Path) -> xr.Dataset:
        return load_calibration_dataset(path)

    def _stored_auto_correction_interval_seconds(self) -> float:
        spinbox = self.calibration_panel.auto_correction_interval_spinbox
        value = self._settings.value(
            AUTO_CORRECTION_INTERVAL_SETTINGS_KEY,
            None,
        )
        multiplier = 1.0
        if value is None:
            value = self._settings.value(
                LEGACY_AUTO_CORRECTION_INTERVAL_MS_SETTINGS_KEY,
                None,
            )
            if value is None:
                default_interval_minutes = (
                    DEFAULT_AUTO_CORRECTION_INTERVAL_SECONDS
                    * AUTO_CORRECTION_INTERVAL_MS_PER_SECOND
                    / AUTO_CORRECTION_INTERVAL_MS_PER_MINUTE
                )
                value = self._settings.value(
                    LEGACY_AUTO_CORRECTION_INTERVAL_MINUTES_SETTINGS_KEY,
                    default_interval_minutes,
                )
                multiplier = (
                    AUTO_CORRECTION_INTERVAL_MS_PER_MINUTE
                    / AUTO_CORRECTION_INTERVAL_MS_PER_SECOND
                )
            else:
                multiplier = 1.0 / AUTO_CORRECTION_INTERVAL_MS_PER_SECOND
        try:
            interval_seconds = round(float(value) * multiplier, 3)
        except (TypeError, ValueError):
            interval_seconds = DEFAULT_AUTO_CORRECTION_INTERVAL_SECONDS
        return min(max(interval_seconds, spinbox.minimum()), spinbox.maximum())

    def _stored_correction_mode(self) -> str:
        mode = str(
            self._settings.value(
                CORRECTION_MODE_SETTINGS_KEY,
                DEFAULT_CORRECTION_MODE,
            )
        ).strip().lower()
        if mode not in {"camera", "beam"}:
            return DEFAULT_CORRECTION_MODE
        return mode

    def _auto_correction_interval_ms(self) -> int:
        interval_seconds = (
            self.calibration_panel.auto_correction_interval_spinbox.value()
        )
        return max(
            int(round(interval_seconds * AUTO_CORRECTION_INTERVAL_MS_PER_SECOND)),
            1,
        )

    def _restart_auto_correction_timer(self) -> None:
        if self._calibration is None:
            self._stop_auto_correction(uncheck=True)
            return
        interval_ms = self._auto_correction_interval_ms()
        self._auto_correction_timer.setInterval(interval_ms)
        self._auto_correction_timer.start()
        logger.info("Automatic timed correction enabled every %d ms.", interval_ms)

    def _registration_shift_kwargs(self) -> dict[str, object]:
        return registration_config_to_shift_kwargs(self._registration_config)

    @QtCore.Slot()
    def _on_shift_monitor_triggered(self) -> None:
        if self._shift_monitor_window is None:
            monitor = ShiftMonitorWindow(
                self._settings,
                registration_config=self._registration_config,
                parent=self,
            )
            monitor.sigRegistrationConfigSaved.connect(
                self._on_registration_config_saved
            )
            monitor.destroyed.connect(self._on_shift_monitor_destroyed)
            monitor.set_calibration(self._calibration)
            self._shift_monitor_window = monitor

        self._shift_monitor_window.show()
        self._shift_monitor_window.raise_()
        self._shift_monitor_window.activateWindow()

    @QtCore.Slot(object)
    def _on_registration_config_saved(self, config: object) -> None:
        if isinstance(config, Mapping):
            self._registration_config = normalized_registration_config(config)
        else:
            self._registration_config = registration_config_from_settings(
                self._settings
            )

    def _on_shift_monitor_destroyed(self, _object: object | None = None) -> None:
        self._shift_monitor_window = None

    def _set_shift_monitor_calibration(self) -> None:
        if self._shift_monitor_window is not None:
            self._shift_monitor_window.set_calibration(self._calibration)

    def _stop_auto_correction(self, *, uncheck: bool) -> None:
        if hasattr(self, "_auto_correction_timer"):
            self._auto_correction_timer.stop()
        if uncheck:
            checkbox = self.calibration_panel.auto_correction_checkbox
            was_blocked = checkbox.blockSignals(True)
            checkbox.setChecked(False)
            checkbox.blockSignals(was_blocked)
        logger.info("Automatic timed correction disabled.")

    def _calibration_after_correction_result(self, result: xr.Dataset) -> xr.Dataset:
        if self._calibration_path is None:
            raise RuntimeError("correction finished without a calibration path")
        if (
            result.attrs.get("calibration_persistence_status") == "pending"
            and self._calibration is not None
            and "px_per_cmd_mm" in result
        ):
            calibration = self._calibration.load().copy(deep=True)
            for key, value in result.attrs.items():
                if (
                    key.startswith("calibration_persistence_")
                    or key == "calibration_pending_spool_path"
                ):
                    calibration.attrs[key] = value
            calibration.attrs["calibration_path"] = str(self._calibration_path)
            validate_visual_calibration_dataset(calibration)
            return calibration
        return self._load_calibration_from_path(self._calibration_path)

    def _flush_pending_persistence(self) -> None:
        results = [
            *flush_pending_calibration_datasets(),
            *flush_pending_correction_history_datasets(),
        ]
        for result in results:
            if result.pending:
                logger.info(
                    "HDF5 persistence still pending for %s: %s",
                    result.target_path,
                    result.message,
                )
            else:
                logger.info(
                    "HDF5 persistence update for %s: %s",
                    result.target_path,
                    result.message,
                )
        self._schedule_persistence_flush_if_needed()

    def _schedule_persistence_flush_if_needed(self) -> None:
        if not hasattr(self, "_persistence_flush_timer"):
            return
        if pending_entry_count() > 0:
            if not self._persistence_flush_timer.isActive():
                self._persistence_flush_timer.start()
        elif self._persistence_flush_timer.isActive():
            self._persistence_flush_timer.stop()

    @QtCore.Slot(int)
    def _on_move_detected(self, target: int) -> None:
        logger.info("Move detected by motor server: target=%d", target)
        try:
            current_motor_backend = getattr(
                self._server,
                "current_motor_backend",
                None,
            )
            motor_backend = (
                current_motor_backend() if callable(current_motor_backend) else None
            )
            self._start_correction(motor_backend=motor_backend)
        except _CorrectionUnavailable as exc:
            message = (
                f"Move target {target} detected, but automatic correction did "
                f"not start: {exc}"
            )
            logger.warning(message)
            self._server.set_result(True, message)
            self._raise_for_user_attention()
            QtWidgets.QMessageBox.warning(
                self,
                "Correction not started",
                message,
            )
        except Exception as exc:
            message = f"Could not start automatic correction for target {target}: {exc}"
            logger.exception(message)
            self._server.set_result(False, message)
            self._raise_for_user_attention()
            QtWidgets.QMessageBox.critical(
                self,
                "Could not start correction",
                message,
            )
        else:
            self._server_correction_pending = True
            self._server_correction_target = int(target)
            logger.info(
                "Automatic correction started for target=%d; server reply pending.",
                target,
            )

    def _raise_for_user_attention(self) -> None:
        if self.isMinimized():
            self.showNormal()
        self.raise_()
        self.activateWindow()

    def _correction_unavailable_message(self) -> str | None:
        if self._calibration is None:
            return "Correction requires a loaded calibration."
        if self._calibration_thread.isRunning():
            return "Correction is unavailable while calibration is running."
        if self._correction_thread.isRunning():
            return "Correction is already in progress."
        if self._detect_shift_thread.isRunning():
            return "Correction is unavailable while shift detection is running."
        if self._calibration_path is None or not self._calibration_path.exists():
            return "Correction requires a calibration file on disk."
        return None

    def _detection_unavailable_message(self) -> str | None:
        if self._calibration is None:
            return "Shift detection requires a loaded calibration."
        if self._calibration_thread.isRunning():
            return "Shift detection is unavailable while calibration is running."
        if self._correction_thread.isRunning():
            return "Shift detection is unavailable while correction is running."
        if self._detect_shift_thread.isRunning():
            return "Shift detection is already in progress."
        return None

    def _start_correction(self, *, motor_backend: object | None = None) -> None:
        unavailable_message = self._correction_unavailable_message()
        if unavailable_message is not None:
            raise _CorrectionUnavailable(unavailable_message)
        if self._calibration is None or self._calibration_path is None:
            raise RuntimeError("correction state changed before startup")

        logger.info("Starting correction: calibration_path=%s", self._calibration_path)
        camera_pair = self._camera_pair_for_current_images()
        self._correction_thread.configure(
            self._calibration,
            camera_pair,
            self._calibration_path,
            motor_backend=motor_backend,
            correction_mode=self.calibration_panel.correction_mode(),
            shift_kwargs=self._registration_shift_kwargs(),
        )

        ui_marked_busy = False
        try:
            self._pause_image_auto_refresh_for_calibration()
            ui_marked_busy = True
            self._set_roi_editing_enabled(False)
            self.calibration_panel.show_correction_in_progress()
            self._correction_thread.start()
            logger.info("Correction thread start requested.")
        except Exception:
            if ui_marked_busy:
                self._restore_image_auto_refresh_after_calibration()
                self._restore_calibration_idle_state()
            logger.exception("Failed while starting correction.")
            raise

    def _reply_to_pending_server_correction(
        self,
        success: bool,
        message: str,
    ) -> None:
        if not self._server_correction_pending:
            return
        logger.info("Replying to pending server correction: success=%s", success)
        self._server_correction_pending = False
        self._server_correction_target = None
        self._server.set_result(success, message)

    @staticmethod
    def _correction_server_result_message(result: xr.Dataset) -> str:
        converged = bool(result.attrs.get("correction_converged", False))
        moves = int(
            result.attrs.get("correction_iterations", result.sizes.get("move", 0))
        )
        status = "converged" if converged else "did not converge"
        return f"Correction {status} after {moves} move(s)."

    def _set_roi_editing_enabled(self, enabled: bool) -> None:
        enabled = bool(enabled)
        self._roi_editing_enabled = enabled
        for roi in self.image_rois.values():
            was_blocked = roi.blockSignals(True)
            try:
                roi.setSelected(False)
                roi.translatable = enabled
                roi.rotatable = enabled
                roi.resizable = enabled
                if enabled:
                    _add_roi_scale_handles(roi)
                else:
                    _remove_roi_scale_handles(roi)
                for handle in roi.getHandles():
                    handle.setEnabled(enabled)
                    handle.setVisible(enabled)
            finally:
                roi.blockSignals(was_blocked)
            roi.update()

    def _on_roi_region_change_finished(self, camera: str) -> None:
        if self._calibration is not None or not self._roi_editing_enabled:
            return

        image_width, image_height = CAMERA_IMAGE_SIZES[camera]
        roi = self.image_rois[camera]
        position = roi.pos()
        size = roi.size()
        raw_geometry = _raw_geometry_from_display(
            camera,
            (
                float(position.x()),
                float(position.y()),
                float(size.x()),
                float(size.y()),
            ),
        )
        geometry = _clamp_roi_geometry(
            raw_geometry,
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
        if not self._reference_preview_active:
            self._show_current_image(camera, image)
        if self._shift_monitor_window is not None:
            self._shift_monitor_window.submit_frame(camera, image)

    @QtCore.Slot(str, str)
    def _on_image_capture_failed(self, camera: str, error_message: str) -> None:
        logger.warning("Image refresh failed for %s: %s", camera, error_message)

    @QtCore.Slot(bool)
    def _on_image_auto_refresh_toggled(self, enabled: bool) -> None:
        self._set_image_refresh_enabled(enabled)

    @QtCore.Slot()
    def _on_show_reference_images_pressed(self) -> None:
        if self._calibration is None:
            return
        self._reference_preview_active = True
        self._reference_preview_restore_state = self._current_image_item_state()
        for camera in CAMERA_IMAGE_SIZES:
            self._show_reference_image(camera)

    @QtCore.Slot()
    def _on_show_reference_images_released(self) -> None:
        if not self._reference_preview_active:
            return
        self._reference_preview_active = False
        self._restore_latest_current_images()
        self._reference_preview_restore_state = {}

    @QtCore.Slot(bool)
    def _on_auto_correction_toggled(self, enabled: bool) -> None:
        if enabled:
            if self._calibration is None:
                logger.warning(
                    "Automatic timed correction requested without a calibration."
                )
                self._stop_auto_correction(uncheck=True)
                return
            self._restart_auto_correction_timer()
            return
        self._stop_auto_correction(uncheck=False)

    @QtCore.Slot(float)
    def _on_auto_correction_interval_changed(self, interval_seconds: float) -> None:
        interval_seconds = round(float(interval_seconds), 3)
        self._settings.setValue(
            AUTO_CORRECTION_INTERVAL_SETTINGS_KEY,
            interval_seconds,
        )
        self._settings.sync()
        if self._auto_correction_timer.isActive():
            self._restart_auto_correction_timer()

    @QtCore.Slot(int)
    def _on_correction_mode_changed(self, _index: int) -> None:
        self._settings.setValue(
            CORRECTION_MODE_SETTINGS_KEY,
            self.calibration_panel.correction_mode(),
        )
        self._settings.sync()

    @QtCore.Slot()
    def _on_auto_correction_timeout(self) -> None:
        if not self.calibration_panel.auto_correction_checkbox.isChecked():
            return
        try:
            self._start_correction()
        except _CorrectionUnavailable as exc:
            logger.info("Automatic timed correction skipped: %s", exc)
        except Exception as exc:
            logger.exception("Failed to start automatic timed correction.")
            QtWidgets.QMessageBox.critical(
                self,
                "Could not start automatic correction",
                str(exc),
            )

    def _set_image_refresh_enabled(self, enabled: bool) -> None:
        for thread in self._image_refresh_threads.values():
            thread.set_enabled(enabled)

    def _set_reference_preview_button_enabled(self, enabled: bool) -> None:
        enabled = bool(enabled)
        self.show_reference_images_button.setEnabled(enabled)
        if not enabled and self._reference_preview_active:
            self._reference_preview_active = False
            self._restore_latest_current_images()
            self._reference_preview_restore_state = {}

    def _show_current_image(self, camera: str, image: object) -> None:
        self._set_camera_image(camera, image, self._full_image_rect(camera))

    def _show_reference_image(self, camera: str) -> None:
        if self._calibration is None:
            return
        reference_name = f"reference_{camera}"
        if reference_name not in self._calibration:
            logger.warning("Calibration is missing %s.", reference_name)
            return
        image = np.asarray(self._calibration[reference_name].values)
        self._set_camera_image(
            camera,
            image,
            self._reference_image_rect(camera, image),
        )

    def _restore_latest_current_images(self) -> None:
        for camera in CAMERA_IMAGE_SIZES:
            if camera in self._latest_images_by_camera:
                image = self._latest_images_by_camera[camera]
                self._show_current_image(camera, image)
            elif camera in self._reference_preview_restore_state:
                image, rect = self._reference_preview_restore_state[camera]
                self._set_camera_image(camera, image, rect)

    def _set_camera_image(
        self,
        camera: str,
        image: object,
        rect: QtCore.QRectF,
    ) -> None:
        image_item = self.image_items[camera]
        image_item.setImage(image)
        raw_rect = QtCore.QRectF(rect)
        _set_image_item_raw_rect(camera, image_item, raw_rect)
        self._image_raw_rects[camera] = raw_rect

    def _current_image_item_state(
        self,
    ) -> dict[str, tuple[np.ndarray, QtCore.QRectF]]:
        state: dict[str, tuple[np.ndarray, QtCore.QRectF]] = {}
        for camera, image_item in self.image_items.items():
            image = image_item.image
            if image is None:
                continue
            raw_rect = self._image_raw_rects.get(camera)
            if raw_rect is None:
                display_rect = image_item.mapRectToParent(image_item.boundingRect())
                raw_rect = _raw_rect_from_display_rect(camera, display_rect)
            state[camera] = (np.asarray(image).copy(), QtCore.QRectF(raw_rect))
        return state

    @staticmethod
    def _full_image_rect(camera: str) -> QtCore.QRectF:
        return _full_raw_image_rect(camera)

    def _reference_image_rect(
        self,
        camera: str,
        image: np.ndarray,
    ) -> QtCore.QRectF:
        image_width, image_height = CAMERA_IMAGE_SIZES[camera]
        if image.shape[:2] == (image_height, image_width):
            return self._full_image_rect(camera)

        roi_geometries = (
            None
            if self._calibration is None
            else _roi_geometries_from_calibration_metadata(self._calibration)
        )
        if roi_geometries is not None and camera in roi_geometries:
            return self._roi_crop_image_rect(camera, roi_geometries[camera])

        height, width = image.shape[:2]
        return QtCore.QRectF(0.0, 0.0, float(width), float(height))

    @staticmethod
    def _roi_crop_image_rect(
        camera: str,
        geometry: RoiGeometry,
    ) -> QtCore.QRectF:
        image_width, image_height = CAMERA_IMAGE_SIZES[camera]
        x, y, width, height = _clamp_roi_geometry(
            geometry,
            image_width,
            image_height,
        )
        x0 = min(max(int(math.floor(x)), 0), image_width - 1)
        y0 = min(max(int(math.floor(y)), 0), image_height - 1)
        x1 = min(max(int(math.ceil(x + width)), x0 + 1), image_width)
        y1 = min(max(int(math.ceil(y + height)), y0 + 1), image_height)
        return QtCore.QRectF(
            float(x0),
            float(y0),
            float(x1 - x0),
            float(y1 - y0),
        )

    def _pause_image_auto_refresh_for_calibration(self) -> None:
        self._image_auto_refresh_checked_before_calibration = (
            self.image_auto_refresh_checkbox.isChecked()
        )
        self._set_image_refresh_enabled(False)
        self.image_auto_refresh_checkbox.setEnabled(False)
        self._set_reference_preview_button_enabled(False)

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
        if self._correction_thread.isRunning() or self._detect_shift_thread.isRunning():
            return

        self._flush_pending_persistence()
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

        self._apply_calibration_roi_metadata(calibration)
        self._calibration = calibration
        self._calibration_path = path
        self._set_roi_editing_enabled(False)
        self.calibration_panel.show_loaded_calibration(calibration, path.name)
        self._restore_latest_correction_result(path)
        self._set_reference_preview_button_enabled(True)
        self._set_shift_monitor_calibration()
        self._schedule_persistence_flush_if_needed()

    @QtCore.Slot()
    def _on_save_calibration_clicked(self) -> None:
        if (
            self._calibration is None
            or self._correction_thread.isRunning()
            or self._detect_shift_thread.isRunning()
        ):
            return

        try:
            default_path = self._calibration_path or _default_calibration_path()
        except Exception as exc:
            QtWidgets.QMessageBox.critical(
                self,
                "Could not choose calibration directory",
                str(exc),
            )
            return

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
            persistence = save_calibration_dataset_deferred(self._calibration, path)
            if persistence.flushed:
                self._calibration = load_calibration_dataset(path)
            else:
                self._calibration = self._calibration.load().copy(deep=True)
                self._calibration.attrs["calibration_path"] = str(path)
                self._calibration = self._calibration.assign_attrs(
                    persistence_result_attrs("calibration", persistence)
                )
        except Exception as exc:
            QtWidgets.QMessageBox.critical(
                self,
                "Could not save calibration",
                str(exc),
            )
            return

        self._calibration_path = path
        self._set_roi_editing_enabled(False)
        self._set_shift_monitor_calibration()
        if persistence.pending:
            self.calibration_panel.show_loaded_calibration(self._calibration, path.name)
            self.calibration_panel.calibration_status_label.setText(
                f"Calibration queued for save: {path.name}"
            )
            self._schedule_persistence_flush_if_needed()
        else:
            self.calibration_panel.show_saved_calibration(path.name)

    @QtCore.Slot()
    def _on_new_calibration_clicked(self) -> None:
        if (
            self._calibration_thread.isRunning()
            or self._correction_thread.isRunning()
            or self._detect_shift_thread.isRunning()
        ):
            return
        if self._calibration is not None:
            self._clear_loaded_calibration()
            return

        try:
            default_output_path = _default_calibration_path()
        except Exception as exc:
            QtWidgets.QMessageBox.critical(
                self,
                "Could not choose calibration directory",
                str(exc),
            )
            return

        dialog = CalibrationStartDialog(self, default_output_path=default_output_path)
        if dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return

        n, step_um = dialog.parameters()
        output_path = dialog.output_path()
        roi_geometries = self._current_roi_geometries()
        roi_metadata = _roi_metadata_from_geometries(roi_geometries)
        camera_pair = self._camera_pair_for_current_images()
        try:
            self._calibration_total_steps = visual_calibration_probe_count(n)
            self._calibration_thread.configure(
                camera_pair,
                roi_metadata,
                output_path,
                n=n,
                step_um=step_um,
                shift_kwargs=self._registration_shift_kwargs(),
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

    @QtCore.Slot()
    def _on_correct_sample_clicked(self) -> None:
        unavailable_message = self._correction_unavailable_message()
        if unavailable_message is not None:
            if (
                self._calibration is not None
                and not self._calibration_thread.isRunning()
                and not self._correction_thread.isRunning()
            ):
                QtWidgets.QMessageBox.critical(
                    self,
                    "Could not start correction",
                    unavailable_message,
                )
            return

        response = QtWidgets.QMessageBox.warning(
            self,
            "Start sample correction?",
            "Correction may move the x/y/z motors.",
            QtWidgets.QMessageBox.StandardButton.Ok
            | QtWidgets.QMessageBox.StandardButton.Cancel,
            QtWidgets.QMessageBox.StandardButton.Cancel,
        )
        if response != QtWidgets.QMessageBox.StandardButton.Ok:
            return

        try:
            self._start_correction()
        except _CorrectionUnavailable as exc:
            QtWidgets.QMessageBox.critical(
                self,
                "Could not start correction",
                str(exc),
            )
        except Exception as exc:
            QtWidgets.QMessageBox.critical(
                self,
                "Could not start correction",
                str(exc),
            )

    @QtCore.Slot()
    def _on_detect_shift_clicked(self) -> None:
        unavailable_message = self._detection_unavailable_message()
        if unavailable_message is not None:
            if (
                self._calibration is not None
                and not self._calibration_thread.isRunning()
                and not self._correction_thread.isRunning()
                and not self._detect_shift_thread.isRunning()
            ):
                QtWidgets.QMessageBox.critical(
                    self,
                    "Could not detect shift",
                    unavailable_message,
                )
            return

        if self._calibration is None:
            return

        camera_pair = self._camera_pair_for_current_images()
        try:
            self._detect_shift_thread.configure(
                self._calibration,
                camera_pair,
                shift_kwargs=self._registration_shift_kwargs(),
            )
        except Exception as exc:
            QtWidgets.QMessageBox.critical(
                self,
                "Could not detect shift",
                str(exc),
            )
            return

        self._pause_image_auto_refresh_for_calibration()
        self._set_roi_editing_enabled(False)
        self.calibration_panel.show_detection_in_progress()
        self._detect_shift_thread.start()

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
            else 0.0
            if remaining == 0
            else None
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
        self._last_correction_result = None
        self._set_roi_editing_enabled(False)
        display_name = (
            self._calibration_path.name
            if self._calibration_path is not None
            else "new calibration"
        )
        self.calibration_panel.show_loaded_calibration(calibration, display_name)
        self._set_reference_preview_button_enabled(True)
        self._set_shift_monitor_calibration()
        self._schedule_persistence_flush_if_needed()

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

    @QtCore.Slot(object)
    def _on_correction_ready(self, result: object) -> None:
        logger.info("Correction ready signal received.")
        try:
            self._restore_image_auto_refresh_after_calibration()
            if not isinstance(result, xr.Dataset):
                raise TypeError("correction thread did not return an xarray Dataset")
            self._last_correction_result = result
            if self._calibration_path is None:
                raise RuntimeError("correction finished without a calibration path")
            calibration = self._calibration_after_correction_result(result)
            self._apply_calibration_roi_metadata(calibration, persist=False)

            self._calibration = calibration
            self._set_roi_editing_enabled(False)
            display_name = (
                self._calibration_path.name
                if self._calibration_path is not None
                else "current calibration"
            )
            self.calibration_panel.show_loaded_calibration(calibration, display_name)
            self.calibration_panel.show_correction_result(result)
            self._set_reference_preview_button_enabled(True)
            self._set_shift_monitor_calibration()
            self._flush_pending_persistence()
        except Exception as exc:
            self._restore_calibration_idle_state()
            self._reply_to_pending_server_correction(False, str(exc))
            QtWidgets.QMessageBox.critical(
                self,
                "Could not use correction result",
                str(exc),
            )
            return

        self._reply_to_pending_server_correction(
            True,
            self._correction_server_result_message(result),
        )
        logger.info("Correction result applied to GUI.")

    @QtCore.Slot(object)
    def _on_correction_progress(self, result: object) -> None:
        logger.info("Correction progress signal received.")
        if not isinstance(result, xr.Dataset):
            logger.warning(
                "Ignoring correction progress with unexpected type: %s",
                type(result).__name__,
            )
            return
        self._last_correction_result = result
        self.calibration_panel.show_correction_progress(result)

    @QtCore.Slot(str)
    def _on_correction_failed(self, error_message: str) -> None:
        logger.error("Correction failed signal received: %s", error_message)
        self._reply_to_pending_server_correction(False, error_message)
        self._restore_image_auto_refresh_after_calibration()
        self._restore_calibration_idle_state()
        QtWidgets.QMessageBox.critical(
            self,
            "Could not correct sample",
            error_message,
        )

    @QtCore.Slot(object)
    def _on_detect_shift_ready(self, result: object) -> None:
        logger.info("Shift detection ready signal received.")
        try:
            self._restore_image_auto_refresh_after_calibration()
            if not isinstance(result, xr.Dataset):
                raise TypeError("shift detection thread did not return an xarray Dataset")
            self._set_roi_editing_enabled(False)
            self.calibration_panel.show_detection_result(result)
            self._set_reference_preview_button_enabled(True)
        except Exception as exc:
            self._restore_calibration_idle_state()
            QtWidgets.QMessageBox.critical(
                self,
                "Could not use shift detection result",
                str(exc),
            )
            return

        logger.info("Shift detection result applied to GUI.")

    @QtCore.Slot(str)
    def _on_detect_shift_failed(self, error_message: str) -> None:
        logger.error("Shift detection failed signal received: %s", error_message)
        self._restore_image_auto_refresh_after_calibration()
        self._restore_calibration_idle_state()
        QtWidgets.QMessageBox.critical(
            self,
            "Could not detect shift",
            error_message,
        )

    @QtCore.Slot()
    def _on_calibration_details_clicked(self) -> None:
        if (
            self._calibration is None
            or self._correction_thread.isRunning()
            or self._detect_shift_thread.isRunning()
        ):
            return

        self.calibration_panel.build_details_dialog(self._calibration).exec()

    def _restore_calibration_idle_state(self) -> None:
        self._calibration_started_at = None
        self._calibration_processing_started_at = None
        self._calibration_total_steps = 0
        if self._calibration is None:
            self.calibration_panel.reset()
            self._set_reference_preview_button_enabled(False)
            self._set_roi_editing_enabled(True)
            return

        self._set_roi_editing_enabled(False)
        display_name = (
            self._calibration_path.name
            if self._calibration_path is not None
            else "current calibration"
        )
        self.calibration_panel.show_loaded_calibration(self._calibration, display_name)
        self._set_reference_preview_button_enabled(True)

    def _restore_latest_correction_result(self, calibration_path: Path) -> None:
        try:
            result = load_latest_correction_history_dataset(calibration_path)
        except Exception:
            logger.exception(
                "Could not load correction history for %s",
                calibration_path,
            )
            self._last_correction_result = None
            return

        self._last_correction_result = result
        if result is not None:
            self.calibration_panel.show_correction_result(result)

    def _clear_loaded_calibration(self) -> None:
        self._stop_auto_correction(uncheck=True)
        self._calibration = None
        self._calibration_path = None
        self._last_correction_result = None
        self._set_reference_preview_button_enabled(False)
        self._restore_calibration_idle_state()
        self._set_shift_monitor_calibration()

    def _get_roi_geometry(self, camera: str) -> RoiGeometry:
        image_width, image_height = CAMERA_IMAGE_SIZES[camera]
        image_roi = self.image_rois[camera]
        position = image_roi.pos()
        size = image_roi.size()
        raw_geometry = _raw_geometry_from_display(
            camera,
            (
                float(position.x()),
                float(position.y()),
                float(size.x()),
                float(size.y()),
            ),
        )
        return _clamp_roi_geometry(
            raw_geometry,
            image_width,
            image_height,
        )

    def _current_roi_geometries(self) -> dict[str, RoiGeometry]:
        return {
            camera: self._get_roi_geometry(camera) for camera in CAMERA_IMAGE_SIZES
        }

    def _camera_pair_for_current_images(self) -> CameraPairPlugin:
        return CameraPairPlugin(
            CallableCameraPlugin("cam0", self._capture_cam0_image),
            CallableCameraPlugin("cam1", self._capture_cam1_image),
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
        display_geometry = _display_geometry(camera, (x, y, width, height))
        image_roi = self.image_rois[camera]
        image_roi.setPos(display_geometry[:2], update=False, finish=False)
        image_roi.setSize(display_geometry[2:], update=True, finish=False)

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
        if self._shift_monitor_window is not None:
            self._shift_monitor_window.close()
            self._shift_monitor_window = None

        if hasattr(self, "_auto_correction_timer"):
            self._auto_correction_timer.stop()

        if hasattr(self, "_persistence_flush_timer"):
            self._persistence_flush_timer.stop()

        for thread in self._image_refresh_threads.values():
            thread.stop()
        for thread in self._image_refresh_threads.values():
            thread.wait()

        self._calibration_thread.stop()
        self._calibration_thread.wait()

        self._correction_thread.stop()
        self._correction_thread.wait()

        self._detect_shift_thread.stop()
        self._detect_shift_thread.wait()

        close_basler_camera()

        self._server.stop()
        self._server.wait()

        super().closeEvent(event)


if __name__ == "__main__":
    multiprocessing.freeze_support()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )
    logger.info("Starting Track Positions GUI.")

    qapp = QtWidgets.QApplication(sys.argv)
    qapp.setStyle("Fusion")
    win = MainWindow()
    win.show()
    win.activateWindow()
    qapp.exec()

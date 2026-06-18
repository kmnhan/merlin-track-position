from __future__ import annotations

import logging
import math
import multiprocessing
import sys
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import cv2
import numpy as np
import pyqtgraph as pg
import xarray as xr
from qtpy import QtCore, QtGui, QtWidgets

from merlin_track_position.constants import (
    DEFAULT_CORRECTION_MODE,
    DEFAULT_VISUAL_CALIBRATION_N,
    DEFAULT_VISUAL_CALIBRATION_MIN_SHIFT_PX,
    DEFAULT_VISUAL_CALIBRATION_STEP_UM,
    IS_DAQ_PC,
)
from merlin_track_position.instruments.parse_config import get_base_file_dir
from merlin_track_position.instruments.cameras import (
    CameraPairPlugin,
    RoiGeometry,
    camera_pair_from_configs,
    crop_image_to_roi,
)
from merlin_track_position.instruments.camera_config import (
    CAMERA_SLOTS,
    SOURCE_BASLER,
    SOURCE_FRAMEGRABBER,
    SOURCE_SIMULATED,
    SOURCE_TYPES,
    CameraConfig,
    DisplayTransform,
    camera_config_mismatches,
    camera_configs_from_settings,
    camera_metadata,
    default_camera_config,
    default_camera_configs,
    save_camera_config,
)
from merlin_track_position.instruments.basler import (
    BaslerCameraCapabilities,
    close_basler_camera,
    get_basler_image,
    list_basler_devices,
    preferred_basler_pixel_format,
    read_basler_capabilities,
    validate_basler_config,
)
from merlin_track_position.instruments.framegrab import get_framegrabber_image
from merlin_track_position.instruments.motors import (
    cached_motor_positions,
    move_motors_and_wait,
    refresh_motor_positions,
)
from merlin_track_position.instruments.simulated_hardware import simulator
from merlin_track_position.interface.calibration_panel import (
    STORED_ORIENTATION_AXES,
    CalibrationPanel,
)
from merlin_track_position.interface.calibration_thread import CalibrationThread
from merlin_track_position.interface.correction_thread import CorrectionThread
from merlin_track_position.interface.detection_thread import DetectShiftThread
from merlin_track_position.interface.registration_settings import (
    normalized_registration_config,
    registration_config_from_settings,
    registration_config_to_measurement_kwargs,
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
    ORIENTATION_READBACK_AXES,
    flush_pending_correction_history_datasets,
    load_latest_correction_history_dataset,
    orientation_ecc_initial_warps_for_readbacks,
)
from merlin_track_position.tracking.persistence import (
    pending_entry_count,
    persistence_result_attrs,
)
from merlin_track_position.tracking.roi import (
    BEAM_TARGET_ATTR_KEYS,
    beam_target_attrs_from_points,
    beam_target_point_from_attrs_or_default,
    roi_crop_bounds,
    roi_local_point_from_full_frame,
)

__all__ = ("CalibrationStartDialog", "CameraSettingsDialog", "MainWindow")

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


def _load_calibration_dialog_path(current_path: Path | None) -> Path:
    if current_path is not None:
        return current_path
    try:
        return _default_calibration_path()
    except Exception:
        logger.info(
            "Could not read scan base file directory for load dialog; "
            "using home directory.",
            exc_info=True,
        )
        return Path.home() / DEFAULT_CALIBRATION_FILE_NAME


class _CorrectionUnavailable(RuntimeError):
    """Expected state that prevents a correction run from starting."""


STORED_ORIENTATION_LABELS_BY_ALIAS = {
    axis_alias: display_name for _, axis_alias, display_name in STORED_ORIENTATION_AXES
}


class _StoredAxisMoveThread(QtCore.QThread):
    sigStoredAxisMoveReady = QtCore.Signal(str, float, float)
    sigStoredAxisMoveFailed = QtCore.Signal(str, str)

    def __init__(self, parent: QtCore.QObject | None = None):
        super().__init__(parent)
        self._running = threading.Event()
        self._axis_alias: str | None = None
        self._target_value: float | None = None

    def configure(self, axis_alias: str, target_value: float) -> None:
        if self.isRunning():
            raise RuntimeError("cannot configure stored-axis move while it is running")
        if axis_alias not in STORED_ORIENTATION_LABELS_BY_ALIAS:
            raise ValueError(f"unsupported stored calibration axis {axis_alias!r}")
        if not math.isfinite(float(target_value)):
            raise ValueError("stored calibration target must be finite")
        self._axis_alias = str(axis_alias)
        self._target_value = float(target_value)

    def run(self) -> None:
        self._running.set()
        try:
            if not self._running.is_set() or self.isInterruptionRequested():
                return
            try:
                if self._axis_alias is None or self._target_value is None:
                    raise RuntimeError(
                        "stored-axis move thread has not been configured"
                    )
                final_positions = move_motors_and_wait(
                    (self._axis_alias,),
                    (self._target_value,),
                )
                final_value = float(final_positions[0])
            except Exception as exc:
                logger.exception("Stored calibration axis move failed.")
                if self._running.is_set() and not self.isInterruptionRequested():
                    axis_alias = "" if self._axis_alias is None else self._axis_alias
                    self.sigStoredAxisMoveFailed.emit(axis_alias, str(exc))
                return

            if self._running.is_set() and not self.isInterruptionRequested():
                self.sigStoredAxisMoveReady.emit(
                    self._axis_alias,
                    self._target_value,
                    final_value,
                )
        finally:
            self._running.clear()

    def stop(self) -> None:
        self._running.clear()
        self.requestInterruption()


_ACTIVE_CAMERA_CONFIGS: dict[str, CameraConfig] = default_camera_configs()
CAMERA_IMAGE_SIZES: dict[str, tuple[int, int]] = {
    slot: (_ACTIVE_CAMERA_CONFIGS[slot].width, _ACTIVE_CAMERA_CONFIGS[slot].height)
    for slot in CAMERA_SLOTS
}
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
BAYER_DISPLAY_CONVERSIONS = {
    "bayerbg": cv2.COLOR_BayerBGGR2RGB,
    "bayergb": cv2.COLOR_BayerGBRG2RGB,
    "bayergr": cv2.COLOR_BayerGRBG2RGB,
    "bayerrg": cv2.COLOR_BayerRGGB2RGB,
}
ROI_SETTINGS_KEYS: dict[str, tuple[str, str, str, str]] = {
    camera: (
        f"roi/{camera}/x",
        f"roi/{camera}/y",
        f"roi/{camera}/width",
        f"roi/{camera}/height",
    )
    for camera in CAMERA_IMAGE_SIZES
}
BEAM_TARGET_SETTINGS_KEYS: dict[str, tuple[str, str]] = {
    camera: (f"beam_target/{camera}/u", f"beam_target/{camera}/v")
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


def _set_active_camera_configs(configs: Mapping[str, CameraConfig]) -> None:
    global _ACTIVE_CAMERA_CONFIGS, CAMERA_IMAGE_SIZES
    _ACTIVE_CAMERA_CONFIGS = {slot: configs[slot] for slot in CAMERA_SLOTS}
    CAMERA_IMAGE_SIZES = {
        slot: (_ACTIVE_CAMERA_CONFIGS[slot].width, _ACTIVE_CAMERA_CONFIGS[slot].height)
        for slot in CAMERA_SLOTS
    }


def _default_roi_geometry(
    image_width: float | None = None,
    image_height: float | None = None,
) -> tuple[float, float, float, float]:
    if image_width is None or image_height is None:
        default_config = default_camera_config("cam0")
        if image_width is None:
            image_width = float(default_config.width)
        if image_height is None:
            image_height = float(default_config.height)
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
    image_width: float | None = None,
    image_height: float | None = None,
) -> tuple[float, float, float, float]:
    if image_width is None or image_height is None:
        default_config = default_camera_config("cam0")
        if image_width is None:
            image_width = float(default_config.width)
        if image_height is None:
            image_height = float(default_config.height)
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
    if _ACTIVE_CAMERA_CONFIGS[camera].display.transpose:
        return (y, x, height, width)
    return (x, y, width, height)


def _display_point(camera: str, point: tuple[float, float]) -> tuple[float, float]:
    u, v = point
    if _ACTIVE_CAMERA_CONFIGS[camera].display.transpose:
        return (v, u)
    return (u, v)


def _raw_point_from_display(
    camera: str,
    point: tuple[float, float],
) -> tuple[float, float]:
    return _display_point(camera, point)


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


def _roi_center_point(camera: str, geometry: RoiGeometry) -> tuple[float, float]:
    image_width, image_height = CAMERA_IMAGE_SIZES[camera]
    x0, y0, x1, y1 = roi_crop_bounds(geometry, (image_height, image_width))
    return (
        x0 + (x1 - x0 - 1.0) / 2.0,
        y0 + (y1 - y0 - 1.0) / 2.0,
    )


def _clamp_point_to_roi(
    camera: str,
    point: tuple[float, float],
    geometry: RoiGeometry,
) -> tuple[float, float]:
    image_width, image_height = CAMERA_IMAGE_SIZES[camera]
    x0, y0, x1, y1 = roi_crop_bounds(geometry, (image_height, image_width))
    u, v = point
    return (
        min(max(float(u), float(x0)), float(x1 - 1)),
        min(max(float(v), float(y0)), float(y1 - 1)),
    )


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
    if _ACTIVE_CAMERA_CONFIGS[camera].display.transpose:
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


def _display_image_for_camera(image: object, config: CameraConfig) -> object:
    if config.source_type != SOURCE_BASLER:
        return image
    image_array = np.asarray(image)
    if image_array.ndim != 2:
        return image
    pixel_format = config.pixel_format.strip().lower()
    conversion = next(
        (
            conversion
            for prefix, conversion in BAYER_DISPLAY_CONVERSIONS.items()
            if pixel_format.startswith(prefix)
        ),
        None,
    )
    if conversion is None or min(image_array.shape) < 2:
        return image
    return cv2.cvtColor(np.ascontiguousarray(image_array), conversion)


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


class CameraSettingsDialog(QtWidgets.QDialog):
    def __init__(
        self,
        configs: Mapping[str, CameraConfig],
        parent: QtWidgets.QWidget | None = None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Camera Settings")
        self._rows: dict[str, dict[str, QtWidgets.QWidget]] = {}
        self._forms: dict[str, QtWidgets.QFormLayout] = {}
        self._draft_configs: dict[str, dict[str, CameraConfig]] = {}
        self._selected_sources: dict[str, str] = {}
        self._basler_devices = {
            device.serial_number: device for device in list_basler_devices()
        }
        self._basler_capabilities: dict[str, BaslerCameraCapabilities] = {}

        layout = QtWidgets.QVBoxLayout(self)
        tabs = QtWidgets.QTabWidget()
        layout.addWidget(tabs)

        for slot in CAMERA_SLOTS:
            config = configs[slot]
            page = QtWidgets.QWidget()
            form = QtWidgets.QFormLayout(page)
            rows: dict[str, QtWidgets.QWidget] = {}

            source_combo = QtWidgets.QComboBox()
            for source_type in SOURCE_TYPES:
                source_combo.addItem(source_type, source_type)
            source_combo.setCurrentIndex(
                max(source_combo.findData(config.source_type), 0)
            )
            form.addRow("Source", source_combo)
            rows["source_type"] = source_combo

            self._rows[slot] = rows
            self._forms[slot] = form
            self._draft_configs[slot] = self._initial_draft_configs(slot, config)
            self._selected_sources[slot] = str(source_combo.currentData())
            tabs.addTab(page, slot)
            source_combo.currentIndexChanged.connect(
                lambda _index, slot=slot: self._on_source_type_changed(slot)
            )
            self._rebuild_source_rows(
                slot,
                self._draft_configs[slot][self._selected_sources[slot]],
            )

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    @staticmethod
    def _default_config_for_source(slot: str, source_type: str) -> CameraConfig:
        default = default_camera_config(slot)
        if source_type == SOURCE_BASLER:
            basler_default = default_camera_config("cam1")
            return replace(
                default,
                source_type=SOURCE_BASLER,
                serial_number=basler_default.serial_number,
                model_name="",
                exposure_us=basler_default.exposure_us,
                gamma=basler_default.gamma,
                pixel_format=basler_default.pixel_format,
                max_num_buffer=basler_default.max_num_buffer,
            )
        if source_type in (SOURCE_FRAMEGRABBER, SOURCE_SIMULATED):
            return replace(
                default,
                source_type=source_type,
                serial_number="",
                model_name="",
            )
        raise ValueError(f"unsupported camera source type: {source_type!r}")

    def _initial_draft_configs(
        self,
        slot: str,
        config: CameraConfig,
    ) -> dict[str, CameraConfig]:
        drafts = {
            source_type: self._default_config_for_source(slot, source_type)
            for source_type in SOURCE_TYPES
        }
        drafts[config.source_type] = config
        return drafts

    def _on_source_type_changed(self, slot: str) -> None:
        rows = self._rows[slot]
        source_combo = cast(QtWidgets.QComboBox, rows["source_type"])
        previous_source = self._selected_sources[slot]
        self._draft_configs[slot][previous_source] = self._config_from_rows(
            slot,
            previous_source,
        )
        source_type = str(source_combo.currentData())
        self._selected_sources[slot] = source_type
        self._rebuild_source_rows(slot, self._draft_configs[slot][source_type])

    def _on_basler_serial_changed(self, slot: str) -> None:
        if self._selected_sources[slot] != SOURCE_BASLER:
            return
        config = self._config_from_rows(
            slot,
            SOURCE_BASLER,
        )
        self._draft_configs[slot][SOURCE_BASLER] = replace(config, pixel_format="")
        self._rebuild_source_rows(slot, self._draft_configs[slot][SOURCE_BASLER])

    def _rebuild_source_rows(self, slot: str, config: CameraConfig) -> None:
        form = self._forms[slot]
        rows = self._rows[slot]
        source_combo = cast(QtWidgets.QComboBox, rows["source_type"])
        self._clear_source_rows(slot)
        source_type = str(source_combo.currentData())

        if source_type == SOURCE_BASLER:
            self._add_basler_rows(slot, config)
            return

        self._add_geometry_rows(form, rows, config)
        self._add_display_rows(form, rows, config)

    def _clear_source_rows(self, slot: str) -> None:
        form = self._forms[slot]
        rows = self._rows[slot]
        source_combo = cast(QtWidgets.QComboBox, rows["source_type"])
        while form.rowCount() > 1:
            form.removeRow(1)
        rows.clear()
        rows["source_type"] = source_combo

    def _add_basler_rows(self, slot: str, config: CameraConfig) -> None:
        form = self._forms[slot]
        rows = self._rows[slot]

        serial_combo = QtWidgets.QComboBox()
        serial_combo.addItem("Select connected camera", "")
        for device in self._basler_devices.values():
            label = f"{device.model_name or 'Basler'} ({device.serial_number})"
            serial_combo.addItem(label, device.serial_number)
        if config.serial_number:
            index = serial_combo.findData(config.serial_number)
            if index >= 0:
                serial_combo.setCurrentIndex(index)
        form.addRow("Serial", serial_combo)
        rows["serial_number"] = serial_combo
        serial_combo.currentIndexChanged.connect(
            lambda _index, slot=slot: self._on_basler_serial_changed(slot)
        )

        serial = str(serial_combo.currentData() or "")
        device = self._basler_devices.get(serial)
        if not serial:
            message = QtWidgets.QLabel(
                "No connected Basler cameras were found."
                if not self._basler_devices
                else "Select a connected Basler camera."
            )
            form.addRow("", message)
            rows["source_message"] = message
            return

        model_label = QtWidgets.QLabel(device.model_name if device is not None else "")
        form.addRow("Model", model_label)
        rows["model_name"] = model_label

        capabilities = self._capabilities_for_serial(serial)
        if capabilities is None:
            message = QtWidgets.QLabel(
                "Could not read live capabilities for the selected camera."
            )
            form.addRow("", message)
            rows["source_message"] = message
            return

        self._add_geometry_rows(form, rows, config, capabilities=capabilities)

        exposure_range = capabilities.exposure_us
        exposure_spin = QtWidgets.QDoubleSpinBox()
        exposure_spin.setRange(exposure_range.minimum, exposure_range.maximum)
        exposure_spin.setSingleStep(max(exposure_range.increment, 0.001))
        exposure_spin.setDecimals(3)
        exposure_spin.setValue(float(config.exposure_us))
        form.addRow("Exposure us", exposure_spin)
        rows["exposure_us"] = exposure_spin

        gamma_range = capabilities.gamma
        gamma_spin = QtWidgets.QDoubleSpinBox()
        gamma_spin.setRange(
            gamma_range.minimum if gamma_range is not None else 0.0,
            gamma_range.maximum if gamma_range is not None else 4.0,
        )
        gamma_spin.setSingleStep(
            max(gamma_range.increment, 0.001) if gamma_range is not None else 0.1
        )
        gamma_spin.setDecimals(3)
        gamma_spin.setValue(float(config.gamma))
        form.addRow("Gamma", gamma_spin)
        rows["gamma"] = gamma_spin

        pixel_format_combo = QtWidgets.QComboBox()
        for pixel_format in capabilities.pixel_formats:
            pixel_format_combo.addItem(pixel_format, pixel_format)
        selected_pixel_format = config.pixel_format
        fallback_pixel_format = default_camera_config("cam1").pixel_format
        preferred_pixel_format = preferred_basler_pixel_format(
            capabilities.pixel_formats,
            fallback=selected_pixel_format or fallback_pixel_format,
        )
        if (
            selected_pixel_format not in capabilities.pixel_formats
            or selected_pixel_format == fallback_pixel_format
        ):
            selected_pixel_format = preferred_pixel_format
        pixel_format_combo.setCurrentIndex(
            max(pixel_format_combo.findData(selected_pixel_format), 0)
        )
        form.addRow("Pixel format", pixel_format_combo)
        rows["pixel_format"] = pixel_format_combo

        buffer_spin = QtWidgets.QSpinBox()
        buffer_spin.setRange(1, 1000)
        buffer_spin.setValue(int(config.max_num_buffer))
        form.addRow("Max buffers", buffer_spin)
        rows["max_num_buffer"] = buffer_spin

        self._add_display_rows(form, rows, config)

    @staticmethod
    def _add_geometry_rows(
        form: QtWidgets.QFormLayout,
        rows: dict[str, QtWidgets.QWidget],
        config: CameraConfig,
        *,
        capabilities: BaslerCameraCapabilities | None = None,
    ) -> None:
        for name, label, minimum, maximum, step in (
            ("width", "Width", 1, 10000, 1),
            ("height", "Height", 1, 10000, 1),
            ("offset_x", "Offset X", 0, 10000, 1),
            ("offset_y", "Offset Y", 0, 10000, 1),
        ):
            if capabilities is not None:
                value_range = getattr(capabilities, name)
                minimum = int(value_range.minimum)
                maximum = int(value_range.maximum)
                step = max(1, int(value_range.increment))
            spin = QtWidgets.QSpinBox()
            spin.setRange(minimum, maximum)
            spin.setSingleStep(step)
            spin.setValue(int(getattr(config, name)))
            form.addRow(label, spin)
            rows[name] = spin

    @staticmethod
    def _add_display_rows(
        form: QtWidgets.QFormLayout,
        rows: dict[str, QtWidgets.QWidget],
        config: CameraConfig,
    ) -> None:
        for name, label, value in (
            ("display_transpose", "Display transpose", config.display.transpose),
            ("display_invert_x", "Display invert X", config.display.invert_x),
            ("display_invert_y", "Display invert Y", config.display.invert_y),
        ):
            checkbox = QtWidgets.QCheckBox()
            checkbox.setChecked(bool(value))
            form.addRow(label, checkbox)
            rows[name] = checkbox

    def _capabilities_for_serial(
        self,
        serial: str,
    ) -> BaslerCameraCapabilities | None:
        if not serial:
            return None
        if serial in self._basler_capabilities:
            return self._basler_capabilities[serial]
        try:
            capabilities = read_basler_capabilities(serial)
        except Exception:
            return None
        self._basler_capabilities[serial] = capabilities
        return capabilities

    def _config_from_rows(self, slot: str, source_type: str) -> CameraConfig:
        rows = self._rows[slot]
        default = self._draft_configs[slot][source_type]

        def spin_value(name: str, fallback: int) -> int:
            widget = rows.get(name)
            if isinstance(widget, QtWidgets.QSpinBox):
                return int(widget.value())
            return int(fallback)

        def double_spin_value(name: str, fallback: float) -> float:
            widget = rows.get(name)
            if isinstance(widget, QtWidgets.QDoubleSpinBox):
                return float(widget.value())
            return float(fallback)

        def combo_value(name: str, fallback: str) -> str:
            widget = rows.get(name)
            if isinstance(widget, QtWidgets.QComboBox):
                return str(widget.currentData() or "") or fallback
            return str(fallback)

        def checkbox_value(name: str, fallback: bool) -> bool:
            widget = rows.get(name)
            if isinstance(widget, QtWidgets.QCheckBox):
                return bool(widget.isChecked())
            return bool(fallback)

        serial_number = default.serial_number
        model_name = default.model_name
        if source_type == SOURCE_BASLER and "serial_number" in rows:
            serial_combo = cast(QtWidgets.QComboBox, rows["serial_number"])
            serial_number = str(serial_combo.currentData() or "")
            device = self._basler_devices.get(serial_number)
            model_name = device.model_name if device is not None else ""
        elif source_type != SOURCE_BASLER:
            serial_number = ""
            model_name = ""

        return CameraConfig(
            slot=slot,
            source_type=source_type,
            serial_number=serial_number,
            model_name=model_name,
            width=spin_value("width", default.width),
            height=spin_value("height", default.height),
            offset_x=spin_value("offset_x", default.offset_x),
            offset_y=spin_value("offset_y", default.offset_y),
            exposure_us=double_spin_value("exposure_us", default.exposure_us),
            gamma=double_spin_value("gamma", default.gamma),
            pixel_format=combo_value("pixel_format", default.pixel_format),
            max_num_buffer=spin_value("max_num_buffer", default.max_num_buffer),
            display=DisplayTransform(
                transpose=checkbox_value(
                    "display_transpose", default.display.transpose
                ),
                invert_x=checkbox_value("display_invert_x", default.display.invert_x),
                invert_y=checkbox_value("display_invert_y", default.display.invert_y),
            ),
        )

    def accept(self) -> None:
        try:
            configs = self.configs()
            for config in configs.values():
                if config.source_type != SOURCE_BASLER:
                    continue
                if not config.serial_number:
                    raise ValueError(
                        f"{config.slot} Basler source requires a connected camera"
                    )
                validate_basler_config(config)
        except Exception as exc:
            QtWidgets.QMessageBox.warning(
                self,
                "Invalid camera settings",
                str(exc),
            )
            return
        super().accept()

    def configs(self) -> dict[str, CameraConfig]:
        configs: dict[str, CameraConfig] = {}
        for slot in self._rows:
            source_type = self._selected_sources[slot]
            config = self._config_from_rows(slot, source_type)
            self._draft_configs[slot][source_type] = config
            configs[slot] = config
        return configs


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
        self._capture_lock = threading.Lock()

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
                image: np.ndarray | None = None
                error_message: str | None = None
                with self._capture_lock:
                    if not self._enabled.is_set():
                        continue
                    try:
                        image = self._image_capture()
                    except Exception as exc:
                        error_message = str(exc)

                if error_message is not None:
                    self.sigImageCaptureFailed.emit(self.camera, error_message)
                elif image is not None:
                    self.sigImageReady.emit(self.camera, image)

                self._wake.wait(self.interval_ms / 1000.0)
                self._wake.clear()
        finally:
            self._running.clear()

    def stop(self) -> None:
        self._running.clear()
        self.requestInterruption()
        self._wake.set()

    def wait_until_idle(self) -> None:
        with self._capture_lock:
            pass


class _BeamTargetItem(pg.TargetItem):
    sigDragStarted = QtCore.Signal(object)

    def mouseDragEvent(self, ev: Any) -> None:  # noqa: N802
        if (
            self.movable
            and ev.button() == QtCore.Qt.MouseButton.LeftButton
            and ev.isStart()
        ):
            self.sigDragStarted.emit(self)
        super().mouseDragEvent(ev)


class _MainWindowGUI(QtWidgets.QMainWindow):
    def __init__(self, parent: QtCore.QObject | None = None):
        super().__init__(parent)

        self.setWindowTitle("Track Positions")
        tools_menu = self.menuBar().addMenu("Tools")
        self.shift_monitor_action = QtGui.QAction("Shift Monitor", self)
        self.shift_monitor_action.setObjectName("shift_monitor_action")
        tools_menu.addAction(self.shift_monitor_action)
        self.camera_settings_action = QtGui.QAction("Camera Settings", self)
        self.camera_settings_action.setObjectName("camera_settings_action")
        tools_menu.addAction(self.camera_settings_action)

        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QtWidgets.QVBoxLayout(central_widget)
        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        main_layout.addWidget(splitter)

        image_widget = QtWidgets.QWidget()
        image_layout = QtWidgets.QVBoxLayout(image_widget)
        image_layout.setContentsMargins(0, 0, 0, 0)

        self.image_controls_layout = QtWidgets.QHBoxLayout()
        self.image_auto_refresh_checkbox = QtWidgets.QCheckBox("Update images")
        self.image_auto_refresh_checkbox.setObjectName("image_auto_refresh_checkbox")
        self.image_auto_refresh_checkbox.setChecked(True)
        self.image_controls_layout.addWidget(self.image_auto_refresh_checkbox)
        self.show_reference_images_button = QtWidgets.QPushButton("Reference")
        self.show_reference_images_button.setObjectName("show_reference_images_button")
        self.show_reference_images_button.setEnabled(False)
        self.image_controls_layout.addWidget(self.show_reference_images_button)
        self.initial_transform_preview_checkbox = QtWidgets.QCheckBox("Transform")
        self.initial_transform_preview_checkbox.setObjectName(
            "initial_transform_preview_checkbox"
        )
        self.initial_transform_preview_checkbox.setEnabled(False)
        self.image_controls_layout.addWidget(self.initial_transform_preview_checkbox)
        self.reset_beam_target_button = QtWidgets.QPushButton("Reset target")
        self.reset_beam_target_button.setObjectName("reset_beam_target_button")
        self.reset_beam_target_button.setEnabled(False)
        self.image_controls_layout.addWidget(self.reset_beam_target_button)
        self.image_controls_layout.addStretch(1)
        image_layout.addLayout(self.image_controls_layout)

        self.image_graphics_layout = pg.GraphicsLayoutWidget()
        image_layout.addWidget(self.image_graphics_layout)
        self.image_plots: dict[str, pg.PlotItem] = {}
        self.image_items: dict[str, pg.ImageItem] = {}
        self.image_rois: dict[str, pg.ROI] = {}
        self.image_targets: dict[str, pg.TargetItem] = {}
        self._image_raw_rects: dict[str, QtCore.QRectF] = {}
        for row, (camera, (image_width, image_height)) in enumerate(
            CAMERA_IMAGE_SIZES.items()
        ):
            image_plot = self.image_graphics_layout.addPlot(row=row, col=0)
            image_plot.setTitle(camera)
            image_plot.setAspectLocked(True)
            display_transform = _ACTIVE_CAMERA_CONFIGS[camera].display
            bottom_label, left_label = (
                ("v", "u") if display_transform.transpose else ("u", "v")
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
            image_plot.invertX(display_transform.invert_x)
            image_plot.invertY(display_transform.invert_y)

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
            image_roi.translatable = False
            _add_roi_scale_handles(image_roi)
            image_roi.setZValue(10)
            image_plot.addItem(image_roi)

            target_point = _roi_center_point(camera, roi_geometry)
            image_target = _BeamTargetItem(
                pos=_display_point(camera, target_point),
                size=10,
                symbol="crosshair",
                pen=pg.mkPen("#d55e00", width=2),
                hoverPen=pg.mkPen("#ff8c33", width=2),
                movable=True,
            )
            image_target.setVisible(False)
            image_target.setZValue(20)
            image_plot.addItem(image_target)

            self.image_plots[camera] = image_plot
            self.image_items[camera] = image_item
            self.image_rois[camera] = image_roi
            self.image_targets[camera] = image_target
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
        settings = QtCore.QSettings("merlin-track-position", "Track Positions")
        camera_configs = camera_configs_from_settings(settings)
        _set_active_camera_configs(camera_configs)
        super().__init__(parent)

        self._settings = settings
        self._camera_configs = camera_configs
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
        self._stored_axis_move_thread = _StoredAxisMoveThread(self)
        self._calibration_total_steps = 0
        self._calibration_started_at: float | None = None
        self._calibration_processing_started_at: float | None = None
        self._roi_editing_enabled = True
        self._beam_target_user_overrides: set[str] = set()
        self._last_correction_result: xr.Dataset | None = None
        self._stored_axis_move_restore_correction_result = False
        self._server_correction_pending = False
        self._server_correction_target: int | None = None
        self._latest_images: tuple[np.ndarray, np.ndarray] | None = None
        self._latest_images_by_camera: dict[str, np.ndarray] = {}
        self._reference_preview_active = False
        self._beam_target_reference_preview_active = False
        self._reference_preview_restore_state: dict[
            str,
            tuple[np.ndarray, QtCore.QRectF],
        ] = {}
        self._initial_transform_preview_warps: dict[str, np.ndarray] = {}
        self._initial_transform_preview_attrs: dict[str, object] = {}
        self._image_capture_locks = {
            "cam0": threading.Lock(),
            "cam1": threading.Lock(),
        }
        self._image_refresh_threads = {
            "cam0": _ImageCaptureThread(
                "cam0",
                lambda: self._capture_camera_image("cam0"),
                IMAGE_REFRESH_INTERVAL_MS,
                self,
            ),
            "cam1": _ImageCaptureThread(
                "cam1",
                lambda: self._capture_camera_image("cam1"),
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
            stored_target_point = self._stored_beam_target_point(camera)
            if stored_target_point is not None:
                self._set_beam_target_point(
                    camera,
                    stored_target_point,
                    user_override=True,
                )
            self.image_rois[camera].sigRegionChangeFinished.connect(
                lambda _roi=None, camera=camera: self._on_roi_region_change_finished(
                    camera
                )
            )
            self.image_targets[camera].sigPositionChanged.connect(
                lambda *args, camera=camera: self._on_beam_target_position_changed(
                    camera
                )
            )
            self.image_targets[camera].sigDragStarted.connect(
                lambda *args, camera=camera: self._on_beam_target_drag_started(camera)
            )
            self.image_targets[camera].sigPositionChangeFinished.connect(
                lambda *args, camera=camera: (
                    self._on_beam_target_position_change_finished(camera)
                )
            )
        self._update_beam_target_visibility()
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
        self.calibration_panel.sigStoredAxisMoveRequested.connect(
            self._on_stored_axis_move_requested
        )
        self.shift_monitor_action.triggered.connect(self._on_shift_monitor_triggered)
        self.camera_settings_action.triggered.connect(
            self._on_camera_settings_triggered
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
        self._correction_thread.sigCorrectionProgress.connect(
            self._on_correction_progress
        )
        self._correction_thread.sigCorrectionReady.connect(self._on_correction_ready)
        self._correction_thread.sigCorrectionFailed.connect(self._on_correction_failed)
        self._detect_shift_thread.sigDetectionReady.connect(self._on_detect_shift_ready)
        self._detect_shift_thread.sigDetectionFailed.connect(
            self._on_detect_shift_failed
        )
        self._stored_axis_move_thread.sigStoredAxisMoveReady.connect(
            self._on_stored_axis_move_ready
        )
        self._stored_axis_move_thread.sigStoredAxisMoveFailed.connect(
            self._on_stored_axis_move_failed
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
        self.initial_transform_preview_checkbox.toggled.connect(
            self._on_initial_transform_preview_toggled
        )
        self.reset_beam_target_button.clicked.connect(
            self._on_reset_beam_targets_clicked
        )
        self.calibration_panel.reset()
        self._set_reference_preview_button_enabled(False)
        self._update_reset_beam_target_button()
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
        mode = (
            str(
                self._settings.value(
                    CORRECTION_MODE_SETTINGS_KEY,
                    DEFAULT_CORRECTION_MODE,
                )
            )
            .strip()
            .lower()
        )
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

    def _registration_measurement_kwargs(self) -> dict[str, object]:
        kwargs = registration_config_to_measurement_kwargs(self._registration_config)
        if kwargs.get("use_ecc_refinement"):
            kwargs["ecc_reference_point_px"] = self._current_ecc_reference_points_px()
        return kwargs

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

    @QtCore.Slot()
    def _on_camera_settings_triggered(self) -> None:
        refresh_checked = self.image_auto_refresh_checkbox.isChecked()
        self._set_image_refresh_enabled(False)
        self.image_auto_refresh_checkbox.setEnabled(False)
        close_basler_camera()
        try:
            dialog = CameraSettingsDialog(self._camera_configs, self)
            if dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted:
                return
            configs = dialog.configs()
            for config in configs.values():
                save_camera_config(self._settings, config)
            self._settings.sync()
            close_basler_camera()
            QtWidgets.QMessageBox.information(
                self,
                "Camera settings saved",
                "Camera settings were saved. Reopen this window before using changed hardware settings.",
            )
        finally:
            self.image_auto_refresh_checkbox.setEnabled(True)
            self._set_image_refresh_enabled(refresh_checked)

    @QtCore.Slot(object)
    def _on_registration_config_saved(self, config: object) -> None:
        if isinstance(config, Mapping):
            self._registration_config = normalized_registration_config(config)
        else:
            self._registration_config = registration_config_from_settings(
                self._settings
            )
        self._update_beam_target_visibility()
        self._set_shift_monitor_beam_targets()

    def _on_shift_monitor_destroyed(self, _object: object | None = None) -> None:
        self._shift_monitor_window = None

    def _set_shift_monitor_calibration(self) -> None:
        if self._shift_monitor_window is not None:
            self._shift_monitor_window.set_calibration(self._calibration)
            self._set_shift_monitor_beam_targets()

    def _set_shift_monitor_beam_targets(self) -> None:
        if self._shift_monitor_window is not None:
            self._shift_monitor_window.set_beam_target_points(
                self._current_beam_target_points()
            )

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
            and "px_per_readback_mm" in result
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
        if self._stored_axis_move_thread.isRunning():
            return "Correction is unavailable while a stored-axis move is running."
        if self._calibration_path is None or not self._calibration_path.exists():
            return "Correction requires a calibration file on disk."
        mismatch_message = self._camera_config_mismatch_message()
        if mismatch_message is not None:
            return mismatch_message
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
        if self._stored_axis_move_thread.isRunning():
            return "Shift detection is unavailable while a stored-axis move is running."
        mismatch_message = self._camera_config_mismatch_message()
        if mismatch_message is not None:
            return mismatch_message
        return None

    def _stored_axis_move_unavailable_message(self) -> str | None:
        if self._calibration is None:
            return "Stored-axis move requires a loaded calibration."
        if self._calibration_thread.isRunning():
            return "Stored-axis move is unavailable while calibration is running."
        if self._correction_thread.isRunning():
            return "Stored-axis move is unavailable while correction is running."
        if self._detect_shift_thread.isRunning():
            return "Stored-axis move is unavailable while shift detection is running."
        if self._stored_axis_move_thread.isRunning():
            return "Stored-axis move is already in progress."
        return None

    def _camera_config_mismatch_message(self) -> str | None:
        if self._calibration is None:
            return None
        mismatches = camera_config_mismatches(
            self._calibration.attrs,
            self._camera_configs,
        )
        if not mismatches:
            return None
        return (
            "Loaded calibration camera metadata does not match current camera "
            "settings: "
            + ", ".join(mismatches)
            + ". Recalibrate after changing cameras."
        )

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
            shift_kwargs=self._registration_measurement_kwargs(),
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

    @QtCore.Slot(str, float)
    def _on_stored_axis_move_requested(
        self,
        axis_alias: str,
        target_value: float,
    ) -> None:
        unavailable_message = self._stored_axis_move_unavailable_message()
        display_name = STORED_ORIENTATION_LABELS_BY_ALIAS.get(axis_alias, axis_alias)
        if unavailable_message is not None:
            QtWidgets.QMessageBox.critical(
                self,
                "Could not move to calibrated position",
                unavailable_message,
            )
            return

        response = QtWidgets.QMessageBox.warning(
            self,
            "Move to calibrated orientation?",
            f"Move {display_name} to calibrated value {target_value:.4f}?",
            QtWidgets.QMessageBox.StandardButton.Ok
            | QtWidgets.QMessageBox.StandardButton.Cancel,
            QtWidgets.QMessageBox.StandardButton.Cancel,
        )
        if response != QtWidgets.QMessageBox.StandardButton.Ok:
            return

        try:
            self._stored_axis_move_thread.configure(axis_alias, target_value)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(
                self,
                "Could not move to calibrated position",
                str(exc),
            )
            return

        ui_marked_busy = False
        try:
            self._stored_axis_move_restore_correction_result = (
                self.calibration_panel.display_mode() == "correction"
                and self._last_correction_result is not None
            )
            self._pause_image_auto_refresh_for_calibration()
            ui_marked_busy = True
            self._set_roi_editing_enabled(False)
            self.calibration_panel.show_stored_axis_move_in_progress(
                display_name,
                target_value,
            )
            self._stored_axis_move_thread.start()
        except Exception as exc:
            if ui_marked_busy:
                self._restore_image_auto_refresh_after_calibration()
                self._restore_stored_axis_move_idle_state()
            QtWidgets.QMessageBox.critical(
                self,
                "Could not move to calibrated position",
                str(exc),
            )

    @QtCore.Slot(str, float, float)
    def _on_stored_axis_move_ready(
        self,
        axis_alias: str,
        target_value: float,
        final_value: float,
    ) -> None:
        display_name = STORED_ORIENTATION_LABELS_BY_ALIAS.get(axis_alias, axis_alias)
        self._restore_image_auto_refresh_after_calibration()
        self._restore_stored_axis_move_idle_state()
        self.calibration_panel.show_stored_axis_move_result(
            display_name,
            target_value,
            final_value,
        )
        self._refresh_initial_transform_preview_after_known_state_change()
        logger.info(
            "Stored calibration axis move finished: axis_alias=%s, target=%g, final=%g",
            axis_alias,
            target_value,
            final_value,
        )

    @QtCore.Slot(str, str)
    def _on_stored_axis_move_failed(
        self,
        axis_alias: str,
        error_message: str,
    ) -> None:
        display_name = STORED_ORIENTATION_LABELS_BY_ALIAS.get(axis_alias, axis_alias)
        self._restore_image_auto_refresh_after_calibration()
        self._restore_stored_axis_move_idle_state()
        logger.error(
            "Stored calibration axis move failed: axis_alias=%s, error=%s",
            axis_alias,
            error_message,
        )
        QtWidgets.QMessageBox.critical(
            self,
            "Could not move to calibrated position",
            f"{display_name}: {error_message}",
        )

    def _restore_stored_axis_move_idle_state(self) -> None:
        if (
            self._stored_axis_move_restore_correction_result
            and self._last_correction_result is not None
        ):
            self._calibration_started_at = None
            self._calibration_processing_started_at = None
            self._calibration_total_steps = 0
            self._set_roi_editing_enabled(False)
            self.calibration_panel.show_correction_result(self._last_correction_result)
            self._set_reference_preview_button_enabled(True)
            self._stored_axis_move_restore_correction_result = False
            return

        self._stored_axis_move_restore_correction_result = False
        self._restore_calibration_idle_state()

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
                roi.translatable = False
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
        for target in self.image_targets.values():
            if hasattr(target, "movable"):
                target.movable = True

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

    def _on_beam_target_position_changed(self, camera: str) -> None:
        if camera not in self.image_targets:
            return
        point = self._beam_target_point(camera)
        self._set_beam_target_point(camera, point, user_override=True)
        self._set_shift_monitor_beam_targets()

    def _on_beam_target_drag_started(self, camera: str) -> None:
        if camera not in self.image_targets:
            return
        self._begin_beam_target_reference_preview()

    def _on_beam_target_position_change_finished(self, camera: str) -> None:
        self._on_beam_target_position_changed(camera)
        self._persist_beam_target_point(camera)
        self._refresh_initial_transform_preview_after_known_state_change()
        self._end_beam_target_reference_preview()

    def _beam_target_point(self, camera: str) -> tuple[float, float]:
        target = self.image_targets[camera]
        position = target.pos()
        return _raw_point_from_display(
            camera,
            (float(position.x()), float(position.y())),
        )

    def _set_beam_target_point(
        self,
        camera: str,
        point: object,
        *,
        user_override: bool,
    ) -> None:
        point_array = np.asarray(point, dtype=np.float64)
        if point_array.shape != (2,) or not np.isfinite(point_array).all():
            point_array = np.asarray(
                _roi_center_point(camera, self._get_roi_geometry(camera))
            )
        clamped = _clamp_point_to_roi(
            camera,
            (float(point_array[0]), float(point_array[1])),
            self._get_roi_geometry(camera),
        )
        target = self.image_targets[camera]
        was_blocked = target.blockSignals(True)
        try:
            target.setPos(*_display_point(camera, clamped))
        finally:
            target.blockSignals(was_blocked)
        if user_override:
            self._beam_target_user_overrides.add(camera)
        else:
            self._beam_target_user_overrides.discard(camera)
        self._update_reset_beam_target_button()

    def _sync_beam_target_after_roi_change(self, camera: str) -> None:
        if not hasattr(self, "image_targets") or camera not in self.image_targets:
            return
        if not hasattr(self, "_beam_target_user_overrides"):
            return
        if camera in self._beam_target_user_overrides:
            self._set_beam_target_point(
                camera,
                self._beam_target_point(camera),
                user_override=True,
            )
            return
        self._set_beam_target_point(
            camera,
            _roi_center_point(camera, self._get_roi_geometry(camera)),
            user_override=False,
        )

    def _current_beam_target_points(self) -> dict[str, np.ndarray]:
        return {
            camera: np.asarray(self._beam_target_point(camera), dtype=np.float64)
            for camera in CAMERA_IMAGE_SIZES
        }

    def _current_beam_target_metadata(self) -> dict[str, float]:
        return beam_target_attrs_from_points(self._current_beam_target_points())

    def _current_ecc_reference_points_px(self) -> dict[str, np.ndarray]:
        attrs = self._registration_roi_metadata()
        return {
            camera: roi_local_point_from_full_frame(
                attrs,
                camera,
                self._beam_target_point(camera),
            )
            for camera in CAMERA_IMAGE_SIZES
        }

    def _update_beam_target_visibility(self) -> None:
        visible = bool(self._registration_config.get("use_ecc_refinement"))
        for target in self.image_targets.values():
            target.setVisible(visible)

    def _registration_roi_metadata(self) -> Mapping[str, object]:
        if self._calibration is None:
            roi_geometries = self._current_roi_geometries()
        else:
            roi_geometries = _roi_geometries_from_calibration_metadata(
                self._calibration
            )
            if roi_geometries is None:
                return {}
        metadata = dict(_roi_metadata_from_geometries(roi_geometries))
        metadata.update(camera_metadata(self._camera_configs))
        return metadata

    def _apply_calibration_beam_target_metadata(
        self,
        calibration: xr.Dataset,
        *,
        preserve_user_overrides: bool = False,
        persist: bool = True,
    ) -> None:
        for camera in CAMERA_IMAGE_SIZES:
            if preserve_user_overrides and camera in self._beam_target_user_overrides:
                self._set_beam_target_point(
                    camera,
                    self._beam_target_point(camera),
                    user_override=True,
                )
                continue
            self._set_beam_target_point(
                camera,
                self._calibration_beam_target_point(calibration, camera),
                user_override=False,
            )
            if persist:
                self._persist_beam_target_point(camera)
        self._set_shift_monitor_beam_targets()
        self._update_reset_beam_target_button()
        self._refresh_initial_transform_preview_after_known_state_change()

    def _calibration_beam_target_point(
        self,
        calibration: xr.Dataset,
        camera: str,
    ) -> np.ndarray:
        keys = BEAM_TARGET_ATTR_KEYS[camera]
        if any(key in calibration.attrs for key in keys):
            return beam_target_point_from_attrs_or_default(calibration.attrs, camera)
        return np.asarray(
            _roi_center_point(camera, self._get_roi_geometry(camera)),
            dtype=np.float64,
        )

    def _loaded_calibration_beam_target_point(
        self,
        camera: str,
    ) -> np.ndarray | None:
        if self._calibration is None:
            return None
        attrs = self._calibration.attrs
        if not all(key in attrs for key in BEAM_TARGET_ATTR_KEYS[camera]):
            return None
        point = self._calibration_beam_target_point(self._calibration, camera)
        return np.asarray(
            _clamp_point_to_roi(
                camera,
                (float(point[0]), float(point[1])),
                self._get_roi_geometry(camera),
            ),
            dtype=np.float64,
        )

    def _beam_target_differs_from_loaded_calibration(self) -> bool:
        if self._calibration is None:
            return False
        for camera in CAMERA_IMAGE_SIZES:
            loaded_point = self._loaded_calibration_beam_target_point(camera)
            if loaded_point is None:
                return False
            current_point = np.asarray(
                self._beam_target_point(camera), dtype=np.float64
            )
            if not np.allclose(current_point, loaded_point, rtol=0.0, atol=1e-6):
                return True
        return False

    def _update_reset_beam_target_button(self) -> None:
        if not hasattr(self, "reset_beam_target_button"):
            return
        self.reset_beam_target_button.setEnabled(
            self._beam_target_differs_from_loaded_calibration()
        )

    @QtCore.Slot()
    def _on_reset_beam_targets_clicked(self) -> None:
        if self._calibration is None:
            self._update_reset_beam_target_button()
            return
        for camera in CAMERA_IMAGE_SIZES:
            point = self._loaded_calibration_beam_target_point(camera)
            if point is None:
                self._update_reset_beam_target_button()
                return
            self._set_beam_target_point(camera, point, user_override=False)
            self._persist_beam_target_point(camera)
        self._set_shift_monitor_beam_targets()
        self._update_reset_beam_target_button()
        self._refresh_initial_transform_preview_after_known_state_change()

    def _capture_camera_image(self, camera: str) -> np.ndarray:
        with self._image_capture_locks[camera]:
            config = self._camera_configs[camera]
            if config.source_type == "framegrabber":
                return get_framegrabber_image(config=config)
            elif config.source_type == "basler":
                return get_basler_image(config)
            elif camera == "cam0":
                image = simulator.get_framegrabber_image()
            else:
                image = simulator.get_basler_image()
            return crop_image_to_roi(
                image,
                (
                    float(config.offset_x),
                    float(config.offset_y),
                    float(config.width),
                    float(config.height),
                ),
            )

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

    @QtCore.Slot(bool)
    def _on_initial_transform_preview_toggled(self, enabled: bool) -> None:
        if not enabled:
            self._initial_transform_preview_warps = {}
            self._initial_transform_preview_attrs = {}
            if not self._reference_preview_active:
                self._restore_latest_current_images()
            return

        try:
            self._refresh_initial_transform_preview_warps(live_read=True)
        except Exception as exc:
            self._disable_initial_transform_preview(str(exc))
            return
        if not self._reference_preview_active:
            self._restore_latest_current_images()

    @QtCore.Slot()
    def _on_show_reference_images_pressed(self) -> None:
        if self._calibration is None:
            return
        self._beam_target_reference_preview_active = False
        self._reference_preview_active = True
        self._reference_preview_restore_state = self._current_image_item_state()
        for camera in CAMERA_IMAGE_SIZES:
            self._show_reference_image(camera)

    @QtCore.Slot()
    def _on_show_reference_images_released(self) -> None:
        if not self._reference_preview_active:
            return
        self._beam_target_reference_preview_active = False
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

    def _wait_for_image_refresh_idle(self) -> None:
        for thread in self._image_refresh_threads.values():
            thread.wait_until_idle()

    def _set_initial_transform_preview_checked(self, checked: bool) -> None:
        was_blocked = self.initial_transform_preview_checkbox.blockSignals(True)
        try:
            self.initial_transform_preview_checkbox.setChecked(bool(checked))
        finally:
            self.initial_transform_preview_checkbox.blockSignals(was_blocked)

    def _refresh_initial_transform_preview_warps(self, *, live_read: bool) -> None:
        if self._calibration is None:
            raise RuntimeError("Transform preview requires a loaded calibration.")
        positions = (
            refresh_motor_positions(ORIENTATION_READBACK_AXES)
            if live_read
            else cached_motor_positions(ORIENTATION_READBACK_AXES)
        )
        readbacks = {
            axis: float(position)
            for axis, position in zip(
                ORIENTATION_READBACK_AXES,
                positions,
                strict=True,
            )
        }
        warps, attrs = orientation_ecc_initial_warps_for_readbacks(
            self._calibration,
            readbacks,
            self._current_ecc_reference_points_px(),
        )
        self._initial_transform_preview_warps = warps
        self._initial_transform_preview_attrs = dict(attrs)

    def _refresh_initial_transform_preview_after_known_state_change(self) -> None:
        if (
            not self.initial_transform_preview_checkbox.isChecked()
            or self._calibration is None
        ):
            return
        try:
            self._refresh_initial_transform_preview_warps(live_read=False)
        except Exception as exc:
            self._disable_initial_transform_preview(str(exc))
            return
        if not self._reference_preview_active:
            self._restore_latest_current_images()

    def _disable_initial_transform_preview(self, message: str) -> None:
        self._set_initial_transform_preview_checked(False)
        self._initial_transform_preview_warps = {}
        self._initial_transform_preview_attrs = {}
        if not self._reference_preview_active:
            self._restore_latest_current_images()
        text = f"Transform preview unavailable: {message}"
        logger.warning(text)
        self.statusBar().showMessage(text, 5000)

    def _set_reference_preview_button_enabled(self, enabled: bool) -> None:
        enabled = bool(enabled)
        transform_was_checked = self.initial_transform_preview_checkbox.isChecked()
        self.show_reference_images_button.setEnabled(enabled)
        self.initial_transform_preview_checkbox.setEnabled(
            enabled and self._calibration is not None
        )
        if self._calibration is None:
            self._set_initial_transform_preview_checked(False)
            self._initial_transform_preview_warps = {}
            self._initial_transform_preview_attrs = {}
            if transform_was_checked and not self._reference_preview_active:
                self._restore_latest_current_images()
        if not enabled and self._reference_preview_active:
            self._beam_target_reference_preview_active = False
            self._reference_preview_active = False
            self._restore_latest_current_images()
            self._reference_preview_restore_state = {}

    def _begin_beam_target_reference_preview(self) -> None:
        if self._calibration is None or self._reference_preview_active:
            return
        self._beam_target_reference_preview_active = True
        self._reference_preview_active = True
        self._reference_preview_restore_state = self._current_image_item_state()
        for camera in CAMERA_IMAGE_SIZES:
            self._show_reference_image(camera)

    def _end_beam_target_reference_preview(self) -> None:
        if not self._beam_target_reference_preview_active:
            return
        self._beam_target_reference_preview_active = False
        if not self._reference_preview_active:
            return
        self._reference_preview_active = False
        self._restore_latest_current_images()
        self._reference_preview_restore_state = {}

    def _show_current_image(self, camera: str, image: object) -> None:
        if self.initial_transform_preview_checkbox.isChecked():
            try:
                self._show_initial_transform_preview_image(camera, image)
                return
            except Exception as exc:
                logger.info(
                    "Initial transform preview disabled for %s: %s",
                    camera,
                    exc,
                )
                self._disable_initial_transform_preview(str(exc))
        self._set_camera_image(camera, image, self._full_image_rect(camera))

    def _show_initial_transform_preview_image(
        self,
        camera: str,
        image: object,
    ) -> None:
        if self._calibration is None:
            raise RuntimeError("Transform preview requires a loaded calibration.")
        if camera not in self._initial_transform_preview_warps:
            raise RuntimeError(f"Transform preview seed is missing for {camera}.")
        reference_name = f"reference_{camera}"
        if reference_name not in self._calibration:
            raise RuntimeError(f"Calibration is missing {reference_name}.")

        reference_image = np.asarray(self._calibration[reference_name].values)
        reference_display = np.asarray(
            _display_image_for_camera(reference_image, self._camera_configs[camera])
        )
        current_display = np.asarray(
            _display_image_for_camera(image, self._camera_configs[camera])
        )
        roi_geometries = _roi_geometries_from_calibration_metadata(self._calibration)
        if roi_geometries is not None and camera in roi_geometries:
            roi_geometry = roi_geometries[camera]
            x0, y0, x1, y1 = roi_crop_bounds(
                roi_geometry,
                reference_display.shape[:2],
            )
            roi_shape = (y1 - y0, x1 - x0)
            if reference_display.shape[:2] != roi_shape:
                reference_display = crop_image_to_roi(reference_display, roi_geometry)
            if current_display.shape[:2] != reference_display.shape[:2]:
                current_display = crop_image_to_roi(current_display, roi_geometry)

        height, width = reference_display.shape[:2]
        warp = np.asarray(
            self._initial_transform_preview_warps[camera],
            dtype=np.float32,
        )
        warped = cv2.warpAffine(
            np.ascontiguousarray(current_display),
            warp,
            (int(width), int(height)),
            flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        self._set_camera_image(
            camera,
            warped,
            self._reference_image_rect(camera, reference_display),
        )

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
        image_item.setImage(
            _display_image_for_camera(image, self._camera_configs[camera])
        )
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
        self._wait_for_image_refresh_idle()
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
        if (
            self._correction_thread.isRunning()
            or self._detect_shift_thread.isRunning()
            or self._stored_axis_move_thread.isRunning()
        ):
            return

        self._flush_pending_persistence()
        default_path = _load_calibration_dialog_path(self._calibration_path)
        file_name, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Load calibration",
            str(default_path),
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
        self._apply_calibration_beam_target_metadata(calibration)
        self._calibration = calibration
        self._calibration_path = path
        self._set_roi_editing_enabled(False)
        self.calibration_panel.show_loaded_calibration(calibration, path.name)
        self._restore_latest_correction_result(path)
        self._set_reference_preview_button_enabled(True)
        self._set_shift_monitor_calibration()
        self._update_reset_beam_target_button()
        self._refresh_initial_transform_preview_after_known_state_change()
        self._schedule_persistence_flush_if_needed()

    @QtCore.Slot()
    def _on_save_calibration_clicked(self) -> None:
        if (
            self._calibration is None
            or self._correction_thread.isRunning()
            or self._detect_shift_thread.isRunning()
            or self._stored_axis_move_thread.isRunning()
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
        self._update_reset_beam_target_button()
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
            or self._stored_axis_move_thread.isRunning()
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
        roi_metadata |= self._current_beam_target_metadata()
        self._persist_current_beam_target_points()
        camera_pair = self._camera_pair_for_current_images()
        try:
            self._calibration_total_steps = visual_calibration_probe_count(n)
            self._calibration_thread.configure(
                camera_pair,
                roi_metadata,
                output_path,
                n=n,
                step_um=step_um,
                shift_kwargs=self._registration_measurement_kwargs(),
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
                shift_kwargs=self._registration_measurement_kwargs(),
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
        self._apply_calibration_beam_target_metadata(
            calibration,
            preserve_user_overrides=True,
            persist=False,
        )
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
        self._update_reset_beam_target_button()
        self._refresh_initial_transform_preview_after_known_state_change()
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
            self._apply_calibration_beam_target_metadata(
                calibration,
                preserve_user_overrides=True,
                persist=False,
            )

            self._calibration = calibration
            self._set_roi_editing_enabled(False)
            self.calibration_panel.show_correction_result(result)
            self._set_reference_preview_button_enabled(True)
            self._set_shift_monitor_calibration()
            self._update_reset_beam_target_button()
            self._refresh_initial_transform_preview_after_known_state_change()
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
                raise TypeError(
                    "shift detection thread did not return an xarray Dataset"
                )
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
            or self._stored_axis_move_thread.isRunning()
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
        beam_target_points = self._current_beam_target_points()
        self._stop_auto_correction(uncheck=True)
        self._calibration = None
        self._calibration_path = None
        self._last_correction_result = None
        self._beam_target_user_overrides.clear()
        for camera, point in beam_target_points.items():
            self._set_beam_target_point(camera, point, user_override=True)
        self._persist_current_beam_target_points()
        self._set_reference_preview_button_enabled(False)
        self._restore_calibration_idle_state()
        self._set_shift_monitor_calibration()
        self._update_reset_beam_target_button()

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
        return {camera: self._get_roi_geometry(camera) for camera in CAMERA_IMAGE_SIZES}

    def _camera_pair_for_current_images(self) -> CameraPairPlugin:
        return camera_pair_from_configs(self._camera_configs)

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
        self._sync_beam_target_after_roi_change(camera)

    def _persist_roi_geometry(
        self,
        camera: str,
        geometry: RoiGeometry,
    ) -> None:
        for key, value in zip(ROI_SETTINGS_KEYS[camera], geometry, strict=True):
            self._settings.setValue(key, float(value))
        self._settings.sync()

    def _stored_beam_target_point(self, camera: str) -> tuple[float, float] | None:
        keys = BEAM_TARGET_SETTINGS_KEYS[camera]
        values = [self._settings.value(key, None) for key in keys]
        if any(value is None for value in values):
            return None
        try:
            u_value, v_value = values
            point = (float(u_value), float(v_value))
        except (TypeError, ValueError):
            return None
        if not all(math.isfinite(value) for value in point):
            return None
        return _clamp_point_to_roi(camera, point, self._get_roi_geometry(camera))

    def _persist_beam_target_point(self, camera: str) -> None:
        point = _clamp_point_to_roi(
            camera,
            self._beam_target_point(camera),
            self._get_roi_geometry(camera),
        )
        for key, value in zip(BEAM_TARGET_SETTINGS_KEYS[camera], point, strict=True):
            self._settings.setValue(key, float(value))
        self._settings.sync()

    def _persist_current_beam_target_points(self) -> None:
        for camera in CAMERA_IMAGE_SIZES:
            self._persist_beam_target_point(camera)

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

        self._stored_axis_move_thread.stop()
        self._stored_axis_move_thread.wait()

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

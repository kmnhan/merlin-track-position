from __future__ import annotations

import logging
import threading
import time
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pyqtgraph as pg
from qtpy import QtCore, QtWidgets

from merlin_track_position.interface.registration_settings import (
    ECC_MOTION_MODEL_AFFINE,
    ECC_MOTION_MODEL_HOMOGRAPHY,
    normalized_registration_config,
    registration_config_to_shift_kwargs,
    save_registration_config,
)
from merlin_track_position.tracking.calibration_core import estimate_capture_stack_shift
from merlin_track_position.tracking.roi import (
    beam_target_point_from_attrs_or_default,
    matching_reference_and_stack,
    roi_local_point_from_full_frame,
)

__all__ = ("ShiftMonitorWindow",)

logger = logging.getLogger("merlin_track_position.interface.shift_monitor_window")

CAMERAS = ("cam0", "cam1")
DEFAULT_MONITOR_SAMPLE_PERIOD_S = 2.0
SIDE_PANEL_MAX_WIDTH = 360
SHIFT_MONITOR_EXPORT_FORMAT = "merlin_track_position.shift_monitor.v1"
PLOT_CHANNELS = (
    ("cam0", "du_px", "du"),
    ("cam0", "dv_px", "dv"),
    ("cam1", "du_px", "du"),
    ("cam1", "dv_px", "dv"),
)


class _ShiftRegistrationThread(QtCore.QThread):
    sigShiftReady = QtCore.Signal(str, float, object, str)
    sigShiftFailed = QtCore.Signal(str, str)

    def __init__(self, parent: QtCore.QObject | None = None):
        super().__init__(parent)
        self._running = threading.Event()
        self._wake = threading.Event()
        self._lock = threading.Lock()
        self._pending: dict[
            str,
            tuple[np.ndarray, np.ndarray, dict[str, object], float],
        ] = {}

    def submit(
        self,
        camera: str,
        reference: Any,
        current_stack: Any,
        config: Mapping[str, Any],
        elapsed_s: float,
    ) -> None:
        reference_array = np.asarray(reference, dtype=np.float64).copy()
        current_array = np.asarray(current_stack, dtype=np.float64).copy()
        normalized_config = normalized_registration_config(config)
        if "ecc_reference_point_px" in config:
            normalized_config["ecc_reference_point_px"] = config[
                "ecc_reference_point_px"
            ]
        with self._lock:
            self._pending[str(camera)] = (
                reference_array,
                current_array,
                normalized_config,
                float(elapsed_s),
            )
        self._wake.set()

    def clear_pending(self) -> None:
        with self._lock:
            self._pending.clear()

    def run(self) -> None:
        self._running.set()
        try:
            while self._running.is_set() and not self.isInterruptionRequested():
                with self._lock:
                    pending = dict(self._pending)
                    self._pending.clear()
                if not pending:
                    self._wake.wait()
                    self._wake.clear()
                    continue

                for camera, (reference, current, config, elapsed_s) in pending.items():
                    if not self._running.is_set() or self.isInterruptionRequested():
                        break
                    try:
                        shift_px, _shift_mad_px, warnings = estimate_capture_stack_shift(
                            reference,
                            current,
                            capture_aggregation=str(config["capture_aggregation"]),
                            **registration_config_to_shift_kwargs(config),
                        )
                        shift_px = np.asarray(shift_px, dtype=np.float64)
                        warning_text = "\n".join(warnings)
                    except Exception as exc:
                        logger.exception("Shift monitor failed for %s.", camera)
                        if (
                            self._running.is_set()
                            and not self.isInterruptionRequested()
                        ):
                            self.sigShiftFailed.emit(camera, str(exc))
                        continue
                    if self._running.is_set() and not self.isInterruptionRequested():
                        self.sigShiftReady.emit(
                            camera,
                            elapsed_s,
                            shift_px,
                            warning_text,
                        )
        finally:
            self._running.clear()

    def stop(self) -> None:
        self._running.clear()
        self.requestInterruption()
        self._wake.set()


class ShiftMonitorWindow(QtWidgets.QWidget):
    sigRegistrationConfigSaved = QtCore.Signal(object)

    def __init__(
        self,
        settings: Any,
        *,
        registration_config: Mapping[str, Any] | None = None,
        parent: QtWidgets.QWidget | None = None,
    ):
        super().__init__(parent, QtCore.Qt.WindowType.Window)
        self.setWindowTitle("Shift Monitor")
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_DeleteOnClose, True)

        self._settings = settings
        self._config = normalized_registration_config(registration_config)
        self._calibration: Any | None = None
        self._calibration_references: dict[str, np.ndarray] = {}
        self._beam_target_points: dict[str, np.ndarray] = {}
        self._live_references: dict[str, np.ndarray] = {}
        self._frame_buffers: dict[str, list[np.ndarray]] = {}
        self._last_submit_at: dict[str, float] = {}
        self._history = _empty_history()
        self._started_at = time.monotonic()
        self._worker = _ShiftRegistrationThread(self)
        self._worker.sigShiftReady.connect(self._on_shift_ready)
        self._worker.sigShiftFailed.connect(self._on_shift_failed)
        self._worker.start()

        self._build_ui()
        self._set_controls_from_config(self._config)
        self._update_reference_label()
        self._update_stats()

    def _build_ui(self) -> None:
        layout = QtWidgets.QHBoxLayout(self)

        side_panel = QtWidgets.QWidget()
        side_panel.setObjectName("shift_monitor_side_panel")
        side_panel.setMaximumWidth(SIDE_PANEL_MAX_WIDTH)
        side_layout = QtWidgets.QVBoxLayout(side_panel)
        side_layout.setContentsMargins(0, 0, 0, 0)

        controls_group = QtWidgets.QGroupBox("Registration")
        controls_layout = QtWidgets.QGridLayout(controls_group)

        self.clip_checkbox = QtWidgets.QCheckBox("Clip")
        self.clip_checkbox.setObjectName("shift_monitor_clip_checkbox")
        controls_layout.addWidget(self.clip_checkbox, 0, 0)

        self.clip_low_spin = QtWidgets.QDoubleSpinBox()
        self.clip_low_spin.setObjectName("shift_monitor_clip_low_spin")
        self.clip_low_spin.setRange(0.0, 100.0)
        self.clip_low_spin.setDecimals(2)
        self.clip_low_spin.setSuffix(" %")
        controls_layout.addWidget(QtWidgets.QLabel("Low"), 1, 0)
        controls_layout.addWidget(self.clip_low_spin, 1, 1)

        self.clip_high_spin = QtWidgets.QDoubleSpinBox()
        self.clip_high_spin.setObjectName("shift_monitor_clip_high_spin")
        self.clip_high_spin.setRange(0.0, 100.0)
        self.clip_high_spin.setDecimals(2)
        self.clip_high_spin.setSuffix(" %")
        controls_layout.addWidget(QtWidgets.QLabel("High"), 2, 0)
        controls_layout.addWidget(self.clip_high_spin, 2, 1)

        self.normalization_combo = QtWidgets.QComboBox()
        self.normalization_combo.setObjectName("shift_monitor_normalization_combo")
        self.normalization_combo.addItem("Phase", "phase")
        self.normalization_combo.addItem("None", "none")
        controls_layout.addWidget(QtWidgets.QLabel("Normalization"), 3, 0)
        controls_layout.addWidget(self.normalization_combo, 3, 1)

        self.upsample_spin = QtWidgets.QSpinBox()
        self.upsample_spin.setObjectName("shift_monitor_upsample_spin")
        self.upsample_spin.setRange(1, 1000)
        controls_layout.addWidget(QtWidgets.QLabel("Upsample"), 4, 0)
        controls_layout.addWidget(self.upsample_spin, 4, 1)

        self.window_checkbox = QtWidgets.QCheckBox("Hanning window")
        self.window_checkbox.setObjectName("shift_monitor_window_checkbox")
        controls_layout.addWidget(self.window_checkbox, 5, 0, 1, 2)

        self.ecc_refinement_checkbox = QtWidgets.QCheckBox("ECC refinement")
        self.ecc_refinement_checkbox.setObjectName(
            "shift_monitor_ecc_refinement_checkbox"
        )
        controls_layout.addWidget(self.ecc_refinement_checkbox, 6, 0, 1, 2)

        self.ecc_motion_model_combo = QtWidgets.QComboBox()
        self.ecc_motion_model_combo.setObjectName(
            "shift_monitor_ecc_motion_model_combo"
        )
        self.ecc_motion_model_combo.addItem("Affine", ECC_MOTION_MODEL_AFFINE)
        self.ecc_motion_model_combo.addItem(
            "Homography",
            ECC_MOTION_MODEL_HOMOGRAPHY,
        )
        controls_layout.addWidget(QtWidgets.QLabel("ECC model"), 7, 0)
        controls_layout.addWidget(self.ecc_motion_model_combo, 7, 1)

        self.high_error_spin = QtWidgets.QDoubleSpinBox()
        self.high_error_spin.setObjectName("shift_monitor_high_error_spin")
        self.high_error_spin.setRange(0.001, 1000.0)
        self.high_error_spin.setDecimals(3)
        self.high_error_spin.setSingleStep(0.05)
        controls_layout.addWidget(QtWidgets.QLabel("Error threshold"), 8, 0)
        controls_layout.addWidget(self.high_error_spin, 8, 1)

        self.capture_count_spin = QtWidgets.QSpinBox()
        self.capture_count_spin.setObjectName("shift_monitor_capture_count_spin")
        self.capture_count_spin.setRange(1, 100)
        controls_layout.addWidget(QtWidgets.QLabel("Images"), 9, 0)
        controls_layout.addWidget(self.capture_count_spin, 9, 1)

        self.capture_aggregation_combo = QtWidgets.QComboBox()
        self.capture_aggregation_combo.setObjectName(
            "shift_monitor_capture_aggregation_combo"
        )
        self.capture_aggregation_combo.addItem("Median shifts", "median_shifts")
        self.capture_aggregation_combo.addItem("Mean image", "mean_image")
        controls_layout.addWidget(QtWidgets.QLabel("Aggregation"), 10, 0)
        controls_layout.addWidget(self.capture_aggregation_combo, 10, 1)

        self.save_button = QtWidgets.QPushButton("Save")
        self.save_button.setObjectName("shift_monitor_save_button")
        self.reset_button = QtWidgets.QPushButton("Reset")
        self.reset_button.setObjectName("shift_monitor_reset_button")
        controls_layout.addWidget(self.save_button, 13, 0)
        controls_layout.addWidget(self.reset_button, 13, 1)

        self.export_button = QtWidgets.QPushButton("Export HDF5")
        self.export_button.setObjectName("shift_monitor_export_button")
        controls_layout.addWidget(self.export_button, 14, 0, 1, 2)

        self.live_checkbox = QtWidgets.QCheckBox("Live")
        self.live_checkbox.setObjectName("shift_monitor_live_checkbox")
        self.live_checkbox.setChecked(True)
        controls_layout.addWidget(self.live_checkbox, 11, 0, 1, 2)

        self.sample_period_spin = QtWidgets.QDoubleSpinBox()
        self.sample_period_spin.setObjectName("shift_monitor_sample_period_spin")
        self.sample_period_spin.setRange(0.2, 60.0)
        self.sample_period_spin.setDecimals(2)
        self.sample_period_spin.setSingleStep(0.5)
        self.sample_period_spin.setSuffix(" s")
        self.sample_period_spin.setValue(DEFAULT_MONITOR_SAMPLE_PERIOD_S)
        controls_layout.addWidget(QtWidgets.QLabel("Monitor period"), 12, 0)
        controls_layout.addWidget(self.sample_period_spin, 12, 1)
        controls_layout.setColumnStretch(1, 1)
        side_layout.addWidget(controls_group)

        self.reference_label = QtWidgets.QLabel()
        self.reference_label.setObjectName("shift_monitor_reference_label")
        self.reference_label.setWordWrap(True)
        side_layout.addWidget(self.reference_label)
        self.warning_label = QtWidgets.QLabel()
        self.warning_label.setObjectName("shift_monitor_warning_label")
        self.warning_label.setWordWrap(True)
        side_layout.addWidget(self.warning_label)

        self.graphics_layout = pg.GraphicsLayoutWidget()
        self.graphics_layout.setObjectName("shift_monitor_graphics_layout")

        self.plots: dict[tuple[str, str], pg.PlotItem] = {}
        self.curves: dict[tuple[str, str], pg.PlotDataItem] = {}
        linked_x_plot: pg.PlotItem | None = None
        for row, camera in enumerate(CAMERAS):
            for col, (axis_key, axis_label) in enumerate(
                (("du_px", "du"), ("dv_px", "dv"))
            ):
                plot = self.graphics_layout.addPlot(row=row, col=col)
                if linked_x_plot is None:
                    linked_x_plot = plot
                else:
                    plot.setXLink(linked_x_plot)
                plot.setTitle(f"{camera} {axis_label}")
                plot.setLabel(
                    "bottom",
                    "elapsed",
                    units="s",
                    siPrefixEnableRanges=(),
                )
                plot.setLabel(
                    "left",
                    axis_label,
                    units="px",
                    siPrefixEnableRanges=(),
                )
                plot.getAxis("bottom").enableAutoSIPrefix(False)
                plot.getAxis("left").enableAutoSIPrefix(False)
                plot.showGrid(x=True, y=True, alpha=0.25)
                curve = plot.plot(pen=pg.mkPen("#0072b2", width=2))
                self.plots[(camera, axis_key)] = plot
                self.curves[(camera, axis_key)] = curve

        self.stats_table = QtWidgets.QTableWidget(len(PLOT_CHANNELS), 6)
        self.stats_table.setObjectName("shift_monitor_stats_table")
        self.stats_table.setHorizontalHeaderLabels(
            ["Channel", "Count", "Latest", "Mean", "Std", "RMS"]
        )
        self.stats_table.verticalHeader().setVisible(False)
        self.stats_table.setHorizontalScrollBarPolicy(
            QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )
        self.stats_table.horizontalHeader().setStretchLastSection(True)
        for row, (camera, axis_key, axis_label) in enumerate(PLOT_CHANNELS):
            self.stats_table.setItem(
                row,
                0,
                QtWidgets.QTableWidgetItem(f"{camera} {axis_label}"),
            )
            self.stats_table.item(row, 0).setFlags(
                self.stats_table.item(row, 0).flags()
                & ~QtCore.Qt.ItemFlag.ItemIsEditable
            )
            del axis_key
        side_layout.addWidget(self.stats_table, stretch=1)
        layout.addWidget(self.graphics_layout, stretch=1)
        layout.addWidget(side_panel)

        for widget in (
            self.clip_checkbox,
            self.clip_low_spin,
            self.clip_high_spin,
            self.normalization_combo,
            self.upsample_spin,
            self.window_checkbox,
            self.ecc_refinement_checkbox,
            self.ecc_motion_model_combo,
            self.high_error_spin,
            self.capture_count_spin,
            self.capture_aggregation_combo,
        ):
            signal = (
                widget.currentIndexChanged
                if isinstance(widget, QtWidgets.QComboBox)
                else widget.valueChanged
                if isinstance(
                    widget,
                    (QtWidgets.QSpinBox, QtWidgets.QDoubleSpinBox),
                )
                else widget.toggled
            )
            signal.connect(self._on_control_changed)

        self.save_button.clicked.connect(self._on_save_clicked)
        self.reset_button.clicked.connect(self.reset_history)
        self.export_button.clicked.connect(self._on_export_clicked)

    def _set_controls_from_config(self, config: Mapping[str, Any]) -> None:
        normalized = normalized_registration_config(config)
        widgets = (
            self.clip_checkbox,
            self.clip_low_spin,
            self.clip_high_spin,
            self.normalization_combo,
            self.upsample_spin,
            self.window_checkbox,
            self.ecc_refinement_checkbox,
            self.ecc_motion_model_combo,
            self.high_error_spin,
            self.capture_count_spin,
            self.capture_aggregation_combo,
        )
        blockers = [widget.blockSignals(True) for widget in widgets]
        try:
            self.clip_checkbox.setChecked(bool(normalized["clip_enabled"]))
            self.clip_low_spin.setValue(float(normalized["clip_low"]))
            self.clip_high_spin.setValue(float(normalized["clip_high"]))
            index = self.normalization_combo.findData(normalized["normalization"])
            self.normalization_combo.setCurrentIndex(max(index, 0))
            self.upsample_spin.setValue(int(normalized["upsample_factor"]))
            self.window_checkbox.setChecked(bool(normalized["use_window"]))
            self.ecc_refinement_checkbox.setChecked(
                bool(normalized["use_ecc_refinement"])
            )
            index = self.ecc_motion_model_combo.findData(
                normalized["ecc_motion_model"]
            )
            self.ecc_motion_model_combo.setCurrentIndex(max(index, 0))
            self.high_error_spin.setValue(float(normalized["high_error_threshold"]))
            self.capture_count_spin.setValue(int(normalized["capture_count"]))
            index = self.capture_aggregation_combo.findData(
                normalized["capture_aggregation"]
            )
            self.capture_aggregation_combo.setCurrentIndex(max(index, 0))
        finally:
            for widget, blocked in zip(widgets, blockers, strict=True):
                widget.blockSignals(blocked)

    def _current_config_from_controls(self) -> dict[str, object]:
        low = float(self.clip_low_spin.value())
        high = float(self.clip_high_spin.value())
        if low >= high:
            if low >= 100.0:
                low = 99.0
                high = 100.0
            else:
                high = min(100.0, low + 0.1)
        return normalized_registration_config(
            {
                "clip_enabled": self.clip_checkbox.isChecked(),
                "clip_low": low,
                "clip_high": high,
                "normalization": self.normalization_combo.currentData(),
                "upsample_factor": self.upsample_spin.value(),
                "use_window": self.window_checkbox.isChecked(),
                "use_ecc_refinement": self.ecc_refinement_checkbox.isChecked(),
                "ecc_motion_model": self.ecc_motion_model_combo.currentData(),
                "high_error_threshold": self.high_error_spin.value(),
                "capture_count": self.capture_count_spin.value(),
                "capture_aggregation": self.capture_aggregation_combo.currentData(),
            }
        )

    @QtCore.Slot()
    def _on_control_changed(self) -> None:
        self._config = self._current_config_from_controls()
        self._frame_buffers.clear()

    @QtCore.Slot()
    def _on_save_clicked(self) -> None:
        self._config = save_registration_config(self._settings, self._config)
        self._set_controls_from_config(self._config)
        self.warning_label.setText("Registration settings saved.")
        self.sigRegistrationConfigSaved.emit(dict(self._config))

    def set_calibration(self, calibration: Any | None) -> None:
        self._calibration = calibration
        self._calibration_references = {}
        if calibration is not None:
            for camera in CAMERAS:
                reference_name = f"reference_{camera}"
                if reference_name in calibration:
                    self._calibration_references[camera] = np.asarray(
                        calibration[reference_name].values
                    ).copy()
        self.reset_history()

    def set_beam_target_points(
        self,
        points: Mapping[str, Any],
    ) -> None:
        self._beam_target_points = {
            str(camera): np.asarray(point, dtype=np.float64).copy()
            for camera, point in points.items()
        }

    @QtCore.Slot()
    def reset_history(self) -> None:
        self._worker.clear_pending()
        self._started_at = time.monotonic()
        self._live_references.clear()
        self._frame_buffers.clear()
        self._last_submit_at.clear()
        self._history = _empty_history()
        for camera in CAMERAS:
            for axis_key in ("du_px", "dv_px"):
                self.curves[(camera, axis_key)].setData([], [])
        self.warning_label.setText("")
        self._update_reference_label()
        self._update_stats()

    def submit_frame(self, camera: str, image: Any) -> None:
        if camera not in CAMERAS:
            return
        if not self.live_checkbox.isChecked():
            return
        now = time.monotonic()
        last_submit_at = self._last_submit_at.get(camera)
        if (
            last_submit_at is not None
            and now - last_submit_at < self.sample_period_spin.value()
        ):
            return

        try:
            reference, current = self._reference_and_current_stack(camera, image)
        except Exception as exc:
            logger.exception("Could not prepare shift monitor image for %s.", camera)
            self.warning_label.setText(f"{camera}: {exc}")
            return
        if reference is None or current is None:
            return

        elapsed_s = now - self._started_at
        self._last_submit_at[camera] = now
        config = dict(self._config)
        reference_point = self._ecc_reference_point_for_camera(camera)
        if reference_point is not None:
            config["ecc_reference_point_px"] = reference_point
        self._worker.submit(camera, reference, current, config, elapsed_s)

    def _ecc_reference_point_for_camera(self, camera: str) -> np.ndarray | None:
        if not bool(self._config.get("use_ecc_refinement")):
            return None

        point = self._beam_target_points.get(camera)
        if point is None and self._calibration is not None:
            point = beam_target_point_from_attrs_or_default(
                self._calibration.attrs,
                camera,
            )
        if point is None:
            return None
        if self._calibration is None:
            return np.asarray(point, dtype=np.float64)
        return roi_local_point_from_full_frame(
            self._calibration.attrs,
            camera,
            point,
        )

    def _reference_and_current_stack(
        self,
        camera: str,
        image: Any,
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        current = np.asarray(image)
        capture_count = int(self._config["capture_count"])
        buffer = self._frame_buffers.setdefault(camera, [])
        buffer.append(current.copy())
        if len(buffer) < capture_count:
            return None, None
        stack = np.stack(buffer[-capture_count:], axis=0)
        buffer.clear()
        if self._calibration is None:
            if camera not in self._live_references:
                self._live_references[camera] = np.mean(
                    np.asarray(stack, dtype=np.float64),
                    axis=0,
                )
                self._update_reference_label()
                return None, None
            return self._live_references[camera], stack

        if camera not in self._calibration_references:
            raise RuntimeError(f"calibration is missing reference_{camera}")
        reference, stack = matching_reference_and_stack(
            self._calibration.attrs,
            camera,
            self._calibration_references[camera],
            stack,
        )
        return reference, stack

    @QtCore.Slot(str, float, object, str)
    def _on_shift_ready(
        self,
        camera: str,
        elapsed_s: float,
        shift_px: object,
        warnings: str,
    ) -> None:
        if camera not in self._history:
            return
        shift_array = np.asarray(shift_px, dtype=np.float64)
        if shift_array.shape != (2,):
            self.warning_label.setText(f"{camera}: unexpected shift shape")
            return

        history = self._history[camera]
        history["t"].append(float(elapsed_s))
        history["du_px"].append(float(shift_array[0]))
        history["dv_px"].append(float(shift_array[1]))
        history["warnings"].append(str(warnings))
        for axis_key in ("du_px", "dv_px"):
            self.curves[(camera, axis_key)].setData(
                history["t"],
                history[axis_key],
            )
        if warnings:
            self.warning_label.setText(f"{camera}: {warnings}")
        elif self.warning_label.text().startswith(f"{camera}:"):
            self.warning_label.setText("")
        self._update_stats()

    @QtCore.Slot(str, str)
    def _on_shift_failed(self, camera: str, error_message: str) -> None:
        self.warning_label.setText(f"{camera}: {error_message}")

    @QtCore.Slot()
    def _on_export_clicked(self) -> None:
        default_name = (
            "shift_monitor_"
            f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.h5"
        )
        file_name, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Export shift monitor",
            str(Path.home() / default_name),
            "HDF5 files (*.h5 *.hdf5);;All files (*)",
        )
        if not file_name:
            return

        path = Path(file_name).expanduser()
        if path.suffix == "":
            path = path.with_suffix(".h5")
        try:
            self.export_history_to_hdf5(path)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(
                self,
                "Could not export shift monitor",
                str(exc),
            )
            return
        self.warning_label.setText(f"Exported {path.name}.")

    def export_history_to_hdf5(self, path: str | Path) -> None:
        path = Path(path).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        string_dtype = h5py.string_dtype(encoding="utf-8")
        config = normalized_registration_config(self._config)
        with h5py.File(path, "w") as file:
            file.attrs["format"] = SHIFT_MONITOR_EXPORT_FORMAT
            file.attrs["created_at_utc"] = datetime.now(UTC).isoformat()
            file.attrs["reference_source"] = (
                "calibration" if self._calibration is not None else "first_live_frame"
            )
            file.attrs["sample_period_s"] = float(self.sample_period_spin.value())
            file.attrs["live_enabled"] = bool(self.live_checkbox.isChecked())
            for key, value in config.items():
                file.attrs[f"registration_{key}"] = value

            pixel_axis = np.asarray(["du_px", "dv_px"], dtype=object)
            for camera in CAMERAS:
                history = self._history[camera]
                time_s = np.asarray(history["t"], dtype=np.float64)
                du_px = np.asarray(history["du_px"], dtype=np.float64)
                dv_px = np.asarray(history["dv_px"], dtype=np.float64)
                shift_px = (
                    np.column_stack((du_px, dv_px))
                    if time_s.size
                    else np.empty((0, 2), dtype=np.float64)
                )
                group = file.create_group(camera)
                group.attrs["reference_ready"] = (
                    camera in self._calibration_references
                    or camera in self._live_references
                )
                group.create_dataset("time_s", data=time_s)
                group.create_dataset("du_px", data=du_px)
                group.create_dataset("dv_px", data=dv_px)
                group.create_dataset("shift_px", data=shift_px)
                group.create_dataset(
                    "pixel_axis",
                    data=pixel_axis,
                    dtype=string_dtype,
                )
                group.create_dataset(
                    "warnings",
                    data=np.asarray(history["warnings"], dtype=object),
                    dtype=string_dtype,
                )

    def _update_reference_label(self) -> None:
        if self._calibration is not None:
            ready = ", ".join(sorted(self._calibration_references)) or "none"
            self.reference_label.setText(f"Reference: calibration ({ready})")
            return
        ready = ", ".join(sorted(self._live_references)) or "waiting"
        self.reference_label.setText(f"Reference: first live frame ({ready})")

    def _update_stats(self) -> None:
        for row, (camera, axis_key, _axis_label) in enumerate(PLOT_CHANNELS):
            values = np.asarray(self._history[camera][axis_key], dtype=np.float64)
            finite = values[np.isfinite(values)]
            count_item = QtWidgets.QTableWidgetItem(str(values.size))
            latest_item = QtWidgets.QTableWidgetItem(
                _format_stat(values[-1]) if values.size else "n/a"
            )
            mean_item = QtWidgets.QTableWidgetItem(
                _format_stat(float(np.mean(finite))) if finite.size else "n/a"
            )
            std_item = QtWidgets.QTableWidgetItem(
                _format_stat(float(np.std(finite))) if finite.size else "n/a"
            )
            rms_item = QtWidgets.QTableWidgetItem(
                _format_stat(float(np.sqrt(np.mean(finite * finite))))
                if finite.size
                else "n/a"
            )
            for col, item in enumerate(
                (count_item, latest_item, mean_item, std_item, rms_item),
                start=1,
            ):
                item.setFlags(item.flags() & ~QtCore.Qt.ItemFlag.ItemIsEditable)
                self.stats_table.setItem(row, col, item)

    def closeEvent(self, event: Any) -> None:
        self._worker.stop()
        self._worker.wait()
        super().closeEvent(event)


def _format_stat(value: float) -> str:
    if not np.isfinite(value):
        return "nan"
    return f"{value:.4g}"


def _empty_history() -> dict[str, dict[str, list]]:
    return {
        camera: {"t": [], "du_px": [], "dv_px": [], "warnings": []}
        for camera in CAMERAS
    }

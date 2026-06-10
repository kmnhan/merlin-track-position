from __future__ import annotations

import math
from collections.abc import Mapping

import numpy as np
import pyqtgraph as pg
import xarray as xr
from qtpy import QtCore, QtWidgets

from merlin_track_position import constants
from merlin_track_position.tracking.calibration_core import (
    CAMERAS,
    COMMAND_AXES,
    MEASUREMENT_WARNING_SUMMARY,
    OBSERVATION_AXES,
    PIXEL_AXES,
    derive_axis_scale_from_jacobian,
    format_probe_warning_lines,
    validate_visual_calibration_dataset,
)

RESIDUAL_PROJECTIONS: tuple[tuple[str, str, int, int], ...] = (
    ("x", "y", 0, 1),
    ("x", "z", 0, 2),
    ("y", "z", 1, 2),
)
CAMERA_CORRECTION_STEP_HEADERS: tuple[str, ...] = (
    "move",
    "x_um",
    "y_um",
    "z_um",
    "pre_residual_px",
    "post_residual_px",
)
BEAM_CORRECTION_STEP_HEADERS: tuple[str, ...] = (
    "move",
    "x_um",
    "y_um",
    "z_um",
    "pre_criterion",
    "post_criterion",
)


def _tooltip_html(
    computed: tuple[str, ...],
    interpretation: tuple[str, ...],
) -> str:
    paragraphs = "".join(f"<p>{item}</p>" for item in (*computed, *interpretation))
    return (
        f"<qt><div style='white-space: normal; width: 360px;'>{paragraphs}</div></qt>"
    )


METRIC_ROWS: tuple[tuple[str, str, str], ...] = (
    (
        "probe_count",
        "Probes",
        _tooltip_html(
            ("Number of readback-mm before/after visual probes.",),
            ("The default calibration uses repeated +axis/-axis probes.",),
        ),
    ),
    (
        "condition_number",
        "Condition number",
        _tooltip_html(
            (
                "<tt>np.linalg.cond(px_per_readback_mm)</tt> after reshaping "
                "to the 4x3 observation matrix.",
            ),
            (
                "Lower is better.",
                "High values mean the cameras cannot cleanly separate x/y/z commands.",
            ),
        ),
    ),
    (
        "axis_scale_readback_mm",
        "Axis scale mm",
        _tooltip_html(
            ("Saved x/y/z readback scales used by normalized LQR correction.",),
            (
                "These are readback-mm normalization scales, not measured physical travel.",
            ),
        ),
    ),
    (
        "axis_sensitivity_px_per_readback_mm",
        "Axis response px/mm",
        _tooltip_html(
            ("Euclidean image response of each command-axis column.",),
            ("This is the per-axis readback-to-image response.",),
        ),
    ),
    (
        "axis_scale_target_response_px",
        "Scale target px",
        _tooltip_html(
            ("Target image response used when deriving correction axis scales.",),
            ("Shown for diagnosing scale clamping during calibration.",),
        ),
    ),
    (
        "residual_rms_px",
        "Residual RMS px",
        _tooltip_html(
            ("Residual is measured probe image delta minus prediction.",),
            (
                "Lower means the local linear readback-to-image model is more consistent.",
            ),
        ),
    ),
    (
        "residual_max_px",
        "Residual max px",
        _tooltip_html(
            ("Largest two-camera pixel residual length among all probes.",),
            ("Use this to find one bad registration or mechanical outlier.",),
        ),
    ),
    (
        "residual_rms_readback_mm",
        "Residual RMS mm",
        _tooltip_html(
            (
                "Pixel residuals are converted through "
                "<tt>pinv(px_per_readback_mm)</tt> into readback-mm coordinates.",
            ),
            ("This is a readback-space fit-error diagnostic, not physical microns.",),
        ),
    ),
    (
        "residual_max_readback_mm",
        "Residual max mm",
        _tooltip_html(
            ("Largest readback-space residual length among all probes.",),
            ("Large values point to poor local repeatability or a bad probe.",),
        ),
    ),
    (
        "readback_command_rms_mm",
        "Readback disagreement RMS mm",
        _tooltip_html(
            ("RMS of readback motion minus commanded trajectory motion.",),
            ("Large values mean requested and encoder-readback motion disagree.",),
        ),
    ),
    (
        "readback_command_max_mm",
        "Readback disagreement max mm",
        _tooltip_html(
            ("Worst readback-vs-command disagreement among all visual probes.",),
            ("Useful for spotting BCS or mechanical reproducibility issues.",),
        ),
    ),
)
REPEATABILITY_ROWS: tuple[tuple[str, str, str], ...] = (
    (
        "mean_rms_std_px",
        "Mean RMS std px",
        _tooltip_html(
            (
                "Repeated probes with identical readback-mm offsets are grouped.",
                "The four camera/pixel components are RMS-combined after sample std.",
            ),
            (
                "Lower means repeated command probes produce more consistent image motion.",
            ),
        ),
    ),
    (
        "max_rms_std_px",
        "Max RMS std px",
        _tooltip_html(
            ("Largest repeated-probe RMS standard deviation.",),
            ("Highlights the least repeatable readback move direction.",),
        ),
    ),
)


def _format_number(value: object) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "n/a"
    return f"{number:.4g}"


def _format_metric_value(value: object) -> str:
    try:
        values = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        return _format_number(value)
    if values.ndim == 0 or values.size == 1:
        return _format_number(values.reshape(-1)[0])
    return "(" + ", ".join(_format_number(item) for item in values.reshape(-1)) + ")"


def _format_axis_triplet_um(values_mm: object) -> str:
    try:
        values = np.asarray(values_mm, dtype=float).reshape(-1)
    except (TypeError, ValueError):
        return "n/a"
    if values.size != len(COMMAND_AXES):
        return "n/a"
    return ", ".join(
        f"{axis}={_format_number(1000.0 * value)} um"
        for axis, value in zip(COMMAND_AXES, values, strict=True)
    )


def _correction_move_delta_mm(result: xr.Dataset) -> np.ndarray:
    if (
        "move_final_readback_position_mm" not in result
        or "initial_readback_position_mm" not in result
    ):
        return np.empty((0, len(COMMAND_AXES)), dtype=float)

    final_readback = np.asarray(
        result["move_final_readback_position_mm"].values,
        dtype=float,
    )
    initial_readback = np.asarray(
        result["initial_readback_position_mm"].values,
        dtype=float,
    )
    if (
        final_readback.ndim != 2
        or final_readback.shape[1] != len(COMMAND_AXES)
        or initial_readback.shape != (len(COMMAND_AXES),)
    ):
        return np.empty((0, len(COMMAND_AXES)), dtype=float)
    if final_readback.shape[0] == 0:
        return np.empty((0, len(COMMAND_AXES)), dtype=float)

    previous = np.vstack((initial_readback[np.newaxis, :], final_readback[:-1]))
    return final_readback - previous


def _correction_move_residuals(
    result: xr.Dataset,
    name: str,
    count: int,
) -> np.ndarray:
    if name not in result:
        return np.full(count, math.nan, dtype=float)
    values = np.asarray(result[name].values, dtype=float).reshape(-1)
    if values.size < count:
        return np.pad(
            values,
            (0, count - values.size),
            mode="constant",
            constant_values=math.nan,
        )
    return values[:count]


def _correction_step_headers(mode: str | None) -> tuple[str, ...]:
    if mode == "beam":
        return BEAM_CORRECTION_STEP_HEADERS
    return CAMERA_CORRECTION_STEP_HEADERS


def _correction_step_residuals(
    result: xr.Dataset,
    move_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    if result.attrs.get("correction_mode") == "beam":
        criterion = _correction_move_residuals(
            result,
            "iteration_correction_criterion_residual",
            move_count + 1,
        )
        return criterion[:move_count], criterion[1 : move_count + 1]

    pre_residual = _correction_move_residuals(
        result,
        "move_pre_weighted_residual_px",
        move_count,
    )
    post_residual = _correction_move_residuals(
        result,
        "move_post_weighted_residual_px",
        move_count,
    )
    return pre_residual, post_residual


def _correction_status_residual(result: xr.Dataset) -> tuple[float, str, str]:
    if result.attrs.get("correction_mode") == "beam":
        name = "iteration_correction_criterion_residual"
        suffix = ""
    else:
        name = "iteration_weighted_residual_px"
        suffix = " px"
    label = "residual"
    residual = math.nan
    if name in result:
        values = np.asarray(result[name].values, dtype=float)
        if values.size:
            residual = float(values.reshape(-1)[-1])
    return residual, label, suffix


def _format_duration(seconds: float | None) -> str:
    if seconds is None or not math.isfinite(seconds) or seconds < 0.0:
        return "n/a"
    rounded = int(round(seconds))
    minutes, second = divmod(rounded, 60)
    hour, minute = divmod(minutes, 60)
    if hour:
        return f"{hour:d}:{minute:02d}:{second:02d}"
    return f"{minute:d}:{second:02d}"


def _calibration_arrays(dataset: xr.Dataset) -> dict[str, np.ndarray]:
    validate_visual_calibration_dataset(dataset)
    command_delta = np.asarray(dataset["probe_command_delta_mm"].values, dtype=float)
    readback_delta = np.asarray(dataset["probe_readback_delta_mm"].values, dtype=float)
    measured_shift = np.asarray(dataset["probe_measured_delta_px"].values, dtype=float)
    px_per_readback_mm = np.asarray(
        dataset["px_per_readback_mm"].values,
        dtype=float,
    )
    jacobian_observation = px_per_readback_mm.reshape(
        len(OBSERVATION_AXES),
        len(COMMAND_AXES),
    )
    pixel_to_readback = np.linalg.pinv(jacobian_observation)
    predicted_shift = (readback_delta @ jacobian_observation.T).reshape(
        readback_delta.shape[0],
        len(CAMERAS),
        len(PIXEL_AXES),
    )
    residual_shift = measured_shift - predicted_shift
    residual_readback = (
        residual_shift.reshape(
            readback_delta.shape[0],
            len(OBSERVATION_AXES),
        )
        @ pixel_to_readback.T
    )
    commanded_motion = np.asarray(
        dataset["post_commanded_position_mm"].values, dtype=float
    ) - np.asarray(dataset["pre_commanded_position_mm"].values, dtype=float)
    readback_motion = np.asarray(
        dataset["post_readback_position_mm"].values, dtype=float
    ) - np.asarray(dataset["pre_readback_position_mm"].values, dtype=float)
    readback_disagreement = readback_motion - commanded_motion
    return {
        "command_delta": command_delta,
        "readback_delta": readback_delta,
        "measured_shift": measured_shift,
        "px_per_readback_mm": px_per_readback_mm,
        "jacobian_observation": jacobian_observation,
        "pixel_to_readback": pixel_to_readback,
        "predicted_shift": predicted_shift,
        "residual_shift": residual_shift,
        "residual_readback": residual_readback,
        "readback_disagreement": readback_disagreement,
    }


def _repeatability_summary(
    command_delta: np.ndarray,
    measured_shift: np.ndarray,
) -> dict[str, float] | None:
    groups: dict[tuple[float, float, float], list[np.ndarray]] = {}
    for command_row, shift_row in zip(command_delta, measured_shift, strict=True):
        key = (float(command_row[0]), float(command_row[1]), float(command_row[2]))
        groups.setdefault(key, []).append(shift_row)

    rms_std_px: list[float] = []
    for rows in groups.values():
        if len(rows) < 2:
            continue
        values = np.stack(rows, axis=0)
        std = np.std(values, axis=0, ddof=1)
        rms_std_px.append(float(np.sqrt(np.mean(std * std))))

    finite_rms = np.asarray(rms_std_px, dtype=float)
    finite_rms = finite_rms[np.isfinite(finite_rms)]
    if not finite_rms.size:
        return None
    return {
        "mean_rms_std_px": float(np.mean(finite_rms)),
        "max_rms_std_px": float(np.max(finite_rms)),
    }


def _calibration_summary(dataset: xr.Dataset) -> dict[str, object]:
    validate_visual_calibration_dataset(dataset)
    arrays = _calibration_arrays(dataset)
    command_delta = arrays["command_delta"]
    measured_shift = arrays["measured_shift"]
    residual_shift = arrays["residual_shift"]
    residual_readback = arrays["residual_readback"]
    readback_disagreement = arrays["readback_disagreement"]

    residual_shift_norms = np.sqrt(np.sum(residual_shift * residual_shift, axis=(1, 2)))
    finite_shift_norms = residual_shift_norms[np.isfinite(residual_shift_norms)]
    if finite_shift_norms.size:
        residual_rms_px = float(
            np.sqrt(np.mean(finite_shift_norms * finite_shift_norms))
        )
        residual_max_px = float(np.max(finite_shift_norms))
    else:
        residual_rms_px = math.nan
        residual_max_px = math.nan

    residual_readback_norms = np.linalg.norm(residual_readback, axis=1)
    finite_command_norms = residual_readback_norms[np.isfinite(residual_readback_norms)]
    if finite_command_norms.size:
        residual_rms_readback_mm = float(
            np.sqrt(np.mean(finite_command_norms * finite_command_norms))
        )
        residual_max_readback_mm = float(np.max(finite_command_norms))
    else:
        residual_rms_readback_mm = math.nan
        residual_max_readback_mm = math.nan

    readback_norms = np.linalg.norm(readback_disagreement, axis=1)
    finite_readback_norms = readback_norms[np.isfinite(readback_norms)]
    if finite_readback_norms.size:
        readback_command_rms_mm = float(
            np.sqrt(np.mean(finite_readback_norms * finite_readback_norms))
        )
        readback_command_max_mm = float(np.max(finite_readback_norms))
    else:
        readback_command_rms_mm = math.nan
        readback_command_max_mm = math.nan

    condition_number = float(np.linalg.cond(arrays["jacobian_observation"]))
    warning_lines = [
        line.strip()
        for line in str(dataset.attrs.get("warnings", "")).splitlines()
        if line.strip()
    ]
    measurement_warning_lines = _measurement_warning_lines(dataset)
    if measurement_warning_lines:
        warning_lines = [
            line for line in warning_lines if line != MEASUREMENT_WARNING_SUMMARY
        ]
        warning_lines.extend(measurement_warning_lines)
    warnings = tuple(dict.fromkeys(warning_lines))

    repeatability = _repeatability_summary(command_delta, measured_shift)
    axis_scale = np.asarray(dataset["axis_scale_readback_mm"].values, dtype=float)
    (
        _derived_axis_scale,
        axis_sensitivity,
        axis_scale_unclamped,
        axis_scale_bounds,
        axis_scale_target_response_px,
    ) = derive_axis_scale_from_jacobian(
        arrays["px_per_readback_mm"],
        arrays["readback_delta"],
    )
    return {
        "probe_count": int(command_delta.shape[0]),
        "condition_number": condition_number,
        "axis_scale_readback_mm": axis_scale,
        "axis_sensitivity_px_per_readback_mm": axis_sensitivity,
        "axis_scale_unclamped_readback_mm": axis_scale_unclamped,
        "axis_scale_bounds_readback_mm": axis_scale_bounds,
        "axis_scale_target_response_px": axis_scale_target_response_px,
        "residual_rms_px": residual_rms_px,
        "residual_max_px": residual_max_px,
        "residual_rms_readback_mm": residual_rms_readback_mm,
        "residual_max_readback_mm": residual_max_readback_mm,
        "readback_command_rms_mm": readback_command_rms_mm,
        "readback_command_max_mm": readback_command_max_mm,
        "px_per_readback_mm": arrays["px_per_readback_mm"],
        "pixel_to_readback": arrays["pixel_to_readback"],
        "warnings": warnings,
        "repeatability": repeatability,
    }


def _measurement_warning_lines(dataset: xr.Dataset) -> tuple[str, ...]:
    if "probe_registration_warnings" not in dataset:
        return ()

    command_delta = np.asarray(dataset["probe_command_delta_mm"].values, dtype=float)
    warning_values = np.asarray(
        dataset["probe_registration_warnings"].values, dtype=str
    )
    if warning_values.shape != (command_delta.shape[0], len(CAMERAS)):
        return ()

    measurement_warnings = tuple(
        tuple(
            tuple(
                line.strip()
                for line in str(warning_values[probe_index, camera_index]).splitlines()
                if line.strip()
            )
            for camera_index in range(len(CAMERAS))
        )
        for probe_index in range(command_delta.shape[0])
    )
    return format_probe_warning_lines(measurement_warnings, command_delta)


def _persistence_warning_lines(attrs: Mapping[str, object]) -> tuple[str, ...]:
    lines: list[str] = []
    labels = {
        "calibration": "Calibration file",
        "correction_history": "Correction history file",
    }
    for prefix, label in labels.items():
        if attrs.get(f"{prefix}_persistence_status") != "pending":
            continue
        message = str(
            attrs.get(
                f"{prefix}_persistence_message",
                "write is queued until the target file becomes writable",
            )
        )
        lines.append(f"{label} write pending: {message}")
    return tuple(lines)


def _calibration_warning_text(
    summary: Mapping[str, object],
    attrs: Mapping[str, object],
) -> str:
    warnings = tuple(summary["warnings"])
    warnings += _persistence_warning_lines(attrs)
    return "\n".join(warnings) if warnings else "No calibration warnings."


def _correction_warning_text(result: xr.Dataset) -> str:
    warning_lines = [
        line.strip()
        for line in str(result.attrs.get("warnings", "")).splitlines()
        if line.strip()
    ]
    warning_lines.extend(_persistence_warning_lines(result.attrs))
    return "\n".join(warning_lines) if warning_lines else "No correction warnings."


def _add_residual_projection_plots(
    graphics_layout: pg.GraphicsLayoutWidget,
) -> dict[str, pg.PlotItem]:
    plots: dict[str, pg.PlotItem] = {}
    for column, (x_label, y_label, _, _) in enumerate(RESIDUAL_PROJECTIONS):
        residual_plot = graphics_layout.addPlot(row=0, col=column)
        residual_plot.setTitle(f"{x_label}-{y_label}")
        residual_plot.setLabel(
            "bottom", x_label, units="readback mm", siPrefixEnableRanges=()
        )
        residual_plot.setLabel(
            "left", y_label, units="readback mm", siPrefixEnableRanges=()
        )
        residual_plot.showGrid(x=True, y=True, alpha=0.25)
        residual_plot.setAspectLocked(True)
        plots[f"{x_label}{y_label}"] = residual_plot
    return plots


def _plot_residuals_on(
    calibration: xr.Dataset,
    residual_plots: Mapping[str, pg.PlotItem],
) -> None:
    for residual_plot in residual_plots.values():
        residual_plot.clear()
    arrays = _calibration_arrays(calibration)
    readback_delta = arrays["readback_delta"]
    residual = arrays["residual_readback"]

    for x_label, y_label, x_index, y_index in RESIDUAL_PROJECTIONS:
        residual_plot = residual_plots[f"{x_label}{y_label}"]
        residual_plot.plot(
            readback_delta[:, x_index],
            readback_delta[:, y_index],
            pen=None,
            symbol="o",
            symbolBrush=pg.mkBrush("#1f77b4"),
            symbolPen=pg.mkPen("#1f77b4"),
        )

        x_values: list[float] = []
        y_values: list[float] = []
        residual_x_values: list[float] = []
        residual_y_values: list[float] = []
        for readback_row, residual_row in zip(readback_delta, residual, strict=True):
            x0 = readback_row[x_index]
            y0 = readback_row[y_index]
            dx = residual_row[x_index]
            dy = residual_row[y_index]
            if not np.isfinite((x0, y0, dx, dy)).all():
                continue
            x1 = x0 + dx
            y1 = y0 + dy
            x_values.extend([float(x0), float(x1), math.nan])
            y_values.extend([float(y0), float(y1), math.nan])
            residual_x_values.append(float(x1))
            residual_y_values.append(float(y1))

        if x_values:
            residual_plot.plot(
                x_values,
                y_values,
                pen=pg.mkPen("#d62728", width=2),
            )
            residual_plot.plot(
                residual_x_values,
                residual_y_values,
                pen=None,
                symbol="o",
                symbolBrush=pg.mkBrush("#d62728"),
                symbolPen=pg.mkPen("#d62728"),
            )


class CalibrationPanel(QtWidgets.QWidget):
    def __init__(self, parent: QtWidgets.QWidget | None = None):
        super().__init__(parent)

        calibration_layout = QtWidgets.QVBoxLayout(self)
        calibration_layout.setContentsMargins(0, 0, 0, 0)

        calibration_button_layout = QtWidgets.QHBoxLayout()
        self.load_calibration_button = QtWidgets.QPushButton("Load calibration")
        self.save_calibration_button = QtWidgets.QPushButton("Save copy")
        self.calibration_details_button = QtWidgets.QPushButton("Details...")
        self.correct_sample_button = QtWidgets.QPushButton("Correct sample")
        self.correct_sample_button.setEnabled(False)
        self.auto_correction_checkbox = QtWidgets.QCheckBox("Auto correct every:")
        self.auto_correction_checkbox.setObjectName("auto_correction_checkbox")
        self.auto_correction_checkbox.setEnabled(False)
        self.auto_correction_interval_spinbox = QtWidgets.QDoubleSpinBox()
        self.auto_correction_interval_spinbox.setObjectName(
            "auto_correction_interval_spinbox"
        )
        self.auto_correction_interval_spinbox.setDecimals(3)
        self.auto_correction_interval_spinbox.setRange(0.001, 86_400.0)
        self.auto_correction_interval_spinbox.setSingleStep(0.001)
        self.auto_correction_interval_spinbox.setAccelerated(True)
        self.auto_correction_interval_spinbox.setValue(180.0)
        self.auto_correction_interval_spinbox.setSuffix(" s")
        self.auto_correction_interval_spinbox.setEnabled(False)
        self.correction_mode_combo = QtWidgets.QComboBox()
        self.correction_mode_combo.setObjectName("correction_mode_combo")
        self.correction_mode_combo.addItem("Camera", "camera")
        self.correction_mode_combo.addItem("Beam", "beam")
        self.correction_mode_combo.setEnabled(False)
        self.detect_shift_button = QtWidgets.QPushButton("Detect shift")
        self.detect_shift_button.setEnabled(False)
        self.new_calibration_button = QtWidgets.QPushButton("New calibration")
        self.new_calibration_button.setEnabled(False)
        calibration_button_layout.addWidget(self.load_calibration_button)
        calibration_button_layout.addWidget(self.save_calibration_button)
        calibration_button_layout.addWidget(self.calibration_details_button)
        calibration_button_layout.addWidget(self.new_calibration_button)
        calibration_layout.addLayout(calibration_button_layout)

        self.calibration_status_label = QtWidgets.QLabel()
        self.calibration_status_label.setWordWrap(True)
        calibration_layout.addWidget(self.calibration_status_label)

        self.calibration_progress_bar = QtWidgets.QProgressBar()
        self.calibration_progress_bar.setObjectName("calibration_progress_bar")
        self.calibration_progress_bar.setTextVisible(True)
        calibration_layout.addWidget(self.calibration_progress_bar)

        self.calibration_review_widget = QtWidgets.QWidget()
        calibration_review_layout = QtWidgets.QVBoxLayout(
            self.calibration_review_widget
        )
        calibration_review_layout.setContentsMargins(0, 0, 0, 0)

        content_layout = QtWidgets.QHBoxLayout()
        left_column = QtWidgets.QVBoxLayout()
        right_column = QtWidgets.QVBoxLayout()
        content_layout.addLayout(left_column, stretch=1)
        content_layout.addLayout(right_column, stretch=1)
        calibration_review_layout.addLayout(content_layout)

        self.warnings_group = QtWidgets.QGroupBox("Warnings")
        warnings_layout = QtWidgets.QVBoxLayout(self.warnings_group)
        self.calibration_warnings_text = QtWidgets.QPlainTextEdit()
        self.calibration_warnings_text.setReadOnly(True)
        self.calibration_warnings_text.setMaximumHeight(90)
        warnings_layout.addWidget(self.calibration_warnings_text)

        self.metrics_group = QtWidgets.QGroupBox("Metrics")
        metrics_layout = QtWidgets.QFormLayout(self.metrics_group)
        self.metric_labels: dict[str, QtWidgets.QLabel] = {}
        for key, label, tooltip in METRIC_ROWS:
            name_label = QtWidgets.QLabel(label)
            name_label.setToolTip(tooltip)
            value_label = QtWidgets.QLabel()
            value_label.setToolTip(tooltip)
            value_label.setTextInteractionFlags(
                QtCore.Qt.TextInteractionFlag.TextSelectableByMouse
            )
            self.metric_labels[key] = value_label
            metrics_layout.addRow(name_label, value_label)
        left_column.addWidget(self.metrics_group)

        self.repeatability_group = QtWidgets.QGroupBox("Repeatability")
        repeatability_layout = QtWidgets.QFormLayout(self.repeatability_group)
        self.repeatability_labels: dict[str, QtWidgets.QLabel] = {}
        for key, label, tooltip in REPEATABILITY_ROWS:
            name_label = QtWidgets.QLabel(label)
            name_label.setToolTip(tooltip)
            value_label = QtWidgets.QLabel()
            value_label.setToolTip(tooltip)
            value_label.setTextInteractionFlags(
                QtCore.Qt.TextInteractionFlag.TextSelectableByMouse
            )
            self.repeatability_labels[key] = value_label
            repeatability_layout.addRow(name_label, value_label)
        left_column.addWidget(self.repeatability_group)
        right_column.addWidget(self.warnings_group)

        self.residual_graphics_layout = pg.GraphicsLayoutWidget()
        self.residual_graphics_layout.setObjectName("residual_projections_layout")
        self.residual_graphics_layout.setMinimumHeight(240)
        self.residual_plots = _add_residual_projection_plots(
            self.residual_graphics_layout
        )
        calibration_review_layout.addWidget(self.residual_graphics_layout, stretch=1)
        calibration_layout.addWidget(self.calibration_review_widget, stretch=1)

        self.correction_steps_group = QtWidgets.QGroupBox("Correction Steps")
        correction_steps_layout = QtWidgets.QVBoxLayout(self.correction_steps_group)
        self.correction_steps_summary_label = QtWidgets.QLabel()
        self.correction_steps_summary_label.setWordWrap(True)
        self.correction_steps_summary_label.setTextInteractionFlags(
            QtCore.Qt.TextInteractionFlag.TextSelectableByMouse
        )
        correction_steps_layout.addWidget(self.correction_steps_summary_label)

        self.correction_warnings_text = QtWidgets.QPlainTextEdit()
        self.correction_warnings_text.setObjectName("correction_warnings_text")
        self.correction_warnings_text.setReadOnly(True)
        self.correction_warnings_text.setMaximumHeight(90)
        correction_steps_layout.addWidget(self.correction_warnings_text)

        self.correction_steps_table = QtWidgets.QTableWidget(
            0,
            len(CAMERA_CORRECTION_STEP_HEADERS),
        )
        self.correction_steps_table.setObjectName("correction_steps_table")
        self.correction_steps_table.setHorizontalHeaderLabels(
            CAMERA_CORRECTION_STEP_HEADERS
        )
        self.correction_steps_table.setEditTriggers(
            QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers
        )
        self.correction_steps_table.setSelectionBehavior(
            QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows
        )
        self.correction_steps_table.horizontalHeader().setSectionResizeMode(
            QtWidgets.QHeaderView.ResizeMode.ResizeToContents
        )
        self.correction_steps_table.verticalHeader().setVisible(False)
        correction_steps_layout.addWidget(self.correction_steps_table, stretch=1)
        calibration_layout.addWidget(self.correction_steps_group, stretch=1)

        correction_button_layout = QtWidgets.QHBoxLayout()
        correction_button_layout.addWidget(self.detect_shift_button)
        correction_button_layout.addWidget(self.correct_sample_button)
        correction_button_layout.addWidget(self.correction_mode_combo)
        correction_button_layout.addWidget(self.auto_correction_checkbox)
        correction_button_layout.addWidget(self.auto_correction_interval_spinbox)
        correction_button_layout.addStretch(1)
        calibration_layout.addLayout(correction_button_layout)

        self._display_mode = "empty"
        self.reset()

    def correction_mode(self) -> str:
        mode = self.correction_mode_combo.currentData()
        if mode in {"camera", "beam"}:
            return str(mode)
        return constants.DEFAULT_CORRECTION_MODE

    def set_correction_mode(self, mode: str) -> None:
        mode = str(mode).strip().lower()
        index = self.correction_mode_combo.findData(mode)
        if index < 0:
            index = self.correction_mode_combo.findData(
                constants.DEFAULT_CORRECTION_MODE
            )
        self.correction_mode_combo.setCurrentIndex(max(index, 0))

    def reset(self) -> None:
        self._set_display_mode("empty")
        self.load_calibration_button.setEnabled(True)
        self.save_calibration_button.setEnabled(False)
        self.calibration_details_button.setEnabled(False)
        self.correct_sample_button.setEnabled(False)
        self.correction_mode_combo.setEnabled(False)
        self.auto_correction_checkbox.setChecked(False)
        self.auto_correction_checkbox.setEnabled(False)
        self.auto_correction_interval_spinbox.setEnabled(False)
        self.detect_shift_button.setEnabled(False)
        self.new_calibration_button.setEnabled(True)
        self.new_calibration_button.setText("New calibration")
        self.calibration_status_label.setText("No calibration loaded.")
        self.calibration_progress_bar.setVisible(False)
        self.calibration_progress_bar.setRange(0, 1)
        self.calibration_progress_bar.setValue(0)
        self.calibration_warnings_text.setPlainText("No calibration loaded.")
        self.correction_warnings_text.setPlainText("No correction result loaded.")
        for label in self.metric_labels.values():
            label.setText("n/a")
        for label in self.repeatability_labels.values():
            label.setText("n/a")
        self.repeatability_group.setVisible(False)
        self._clear_correction_steps()
        for residual_plot in self.residual_plots.values():
            residual_plot.clear()

    def show_calibration_in_progress(self, total_steps: int | None = None) -> None:
        self._set_display_mode("empty")
        self.load_calibration_button.setEnabled(False)
        self.save_calibration_button.setEnabled(False)
        self.calibration_details_button.setEnabled(False)
        self.correct_sample_button.setEnabled(False)
        self.correction_mode_combo.setEnabled(False)
        self.auto_correction_checkbox.setChecked(False)
        self.auto_correction_checkbox.setEnabled(False)
        self.auto_correction_interval_spinbox.setEnabled(False)
        self.detect_shift_button.setEnabled(False)
        self.new_calibration_button.setEnabled(False)
        self.new_calibration_button.setText("New calibration")
        self.calibration_status_label.setText("New calibration in progress...")
        self.calibration_progress_bar.setVisible(True)
        if total_steps is None or total_steps < 1:
            self.calibration_progress_bar.setRange(0, 0)
            return
        self.calibration_progress_bar.setRange(0, int(total_steps))
        self.calibration_progress_bar.setValue(0)
        self.calibration_progress_bar.setFormat(f"0 / {int(total_steps)} probes")

    def show_calibration_step(
        self,
        *,
        idx: int,
        total_steps: int,
        dx: float,
        dy: float,
        dz: float,
        elapsed_s: float,
        eta_s: float | None,
    ) -> None:
        completed = min(max(int(idx) + 1, 0), max(int(total_steps), 1))
        total_steps = max(int(total_steps), completed, 1)
        self.calibration_progress_bar.setVisible(True)
        self.calibration_progress_bar.setRange(0, total_steps)
        self.calibration_progress_bar.setValue(completed)
        self.calibration_progress_bar.setFormat(f"{completed} / {total_steps} probes")
        self.calibration_status_label.setText(
            "New calibration in progress. "
            f"Requested offset ({_format_number(dx)}, {_format_number(dy)}, "
            f"{_format_number(dz)}) mm. "
            f"Elapsed {_format_duration(elapsed_s)}, ETA {_format_duration(eta_s)}."
        )

    def show_calibration_processing(
        self,
        *,
        completed: int,
        total: int,
        elapsed_s: float,
        eta_s: float | None,
    ) -> None:
        total = max(int(total), 1)
        completed = min(max(int(completed), 0), total)
        self.calibration_progress_bar.setVisible(True)
        self.calibration_progress_bar.setRange(0, total)
        self.calibration_progress_bar.setValue(completed)
        self.calibration_progress_bar.setFormat(f"{completed} / {total} registrations")
        self.calibration_status_label.setText(
            "Processing probes. "
            f"Elapsed {_format_duration(elapsed_s)}, ETA {_format_duration(eta_s)}."
        )

    def _set_loaded_idle_controls_enabled(self, enabled: bool) -> None:
        self.load_calibration_button.setEnabled(enabled)
        self.save_calibration_button.setEnabled(enabled)
        self.calibration_details_button.setEnabled(enabled)
        self.correct_sample_button.setEnabled(enabled)
        self.correction_mode_combo.setEnabled(enabled)
        self.auto_correction_checkbox.setEnabled(enabled)
        self.auto_correction_interval_spinbox.setEnabled(enabled)
        self.detect_shift_button.setEnabled(enabled)
        self.new_calibration_button.setEnabled(enabled)
        self.new_calibration_button.setText("Clear calibration")

    def _set_display_mode(self, mode: str) -> None:
        self._display_mode = mode
        correction_mode = mode == "correction"
        self.calibration_review_widget.setVisible(not correction_mode)
        self.metrics_group.setVisible(not correction_mode)
        if not correction_mode:
            self.warnings_group.setVisible(True)
            self.residual_graphics_layout.setVisible(True)
        self.correction_steps_group.setVisible(correction_mode)
        if correction_mode:
            self._clear_calibration_diagnostics()

    def _clear_calibration_diagnostics(self) -> None:
        self.calibration_warnings_text.clear()
        for label in self.metric_labels.values():
            label.setText("n/a")
        for label in self.repeatability_labels.values():
            label.setText("n/a")
        self.repeatability_group.setVisible(False)
        self.warnings_group.setVisible(False)
        self.residual_graphics_layout.setVisible(False)
        for residual_plot in self.residual_plots.values():
            residual_plot.clear()

    def show_loaded_calibration(
        self,
        calibration: xr.Dataset,
        display_name: str,
    ) -> None:
        summary = _calibration_summary(calibration)
        self._set_display_mode("calibration")
        self._set_loaded_idle_controls_enabled(True)
        self.calibration_progress_bar.setVisible(False)
        self.calibration_status_label.setText(
            f"Loaded calibration: {display_name} ({summary['probe_count']} probes)"
        )

        self.calibration_warnings_text.setPlainText(
            _calibration_warning_text(summary, calibration.attrs)
        )

        for key, _, _ in METRIC_ROWS:
            self.metric_labels[key].setText(_format_metric_value(summary[key]))

        repeatability = summary["repeatability"]
        self.repeatability_group.setVisible(repeatability is not None)
        for key, label in self.repeatability_labels.items():
            if repeatability is None:
                label.setText("n/a")
            else:
                label.setText(_format_number(repeatability[key]))

        self._clear_correction_steps()
        self._plot_residuals(calibration)

    def _clear_correction_steps(self) -> None:
        self.correction_steps_summary_label.setText("No correction result loaded.")
        self.correction_warnings_text.setPlainText("No correction result loaded.")
        self.correction_steps_table.setRowCount(0)
        self.correction_steps_group.setVisible(False)

    def _show_pending_correction_steps(self) -> None:
        headers = _correction_step_headers(self.correction_mode())
        self.correction_steps_table.setColumnCount(len(headers))
        self.correction_steps_table.setHorizontalHeaderLabels(headers)
        self.correction_steps_summary_label.setText(
            "Capturing initial correction measurement before first move."
        )
        self.correction_warnings_text.setPlainText("No correction warnings.")
        self.correction_steps_table.setRowCount(0)
        self.correction_steps_group.setVisible(True)

    def _show_correction_steps(
        self,
        result: xr.Dataset,
        *,
        in_progress: bool = False,
    ) -> None:
        move_delta = _correction_move_delta_mm(result)
        headers = _correction_step_headers(result.attrs.get("correction_mode"))
        pre_residual, post_residual = _correction_step_residuals(
            result,
            move_delta.shape[0],
        )

        self.correction_steps_table.setColumnCount(len(headers))
        self.correction_steps_table.setHorizontalHeaderLabels(headers)
        self.correction_steps_table.setRowCount(move_delta.shape[0])
        for row, delta_mm in enumerate(move_delta):
            values = (
                str(row + 1),
                _format_number(1000.0 * delta_mm[0]),
                _format_number(1000.0 * delta_mm[1]),
                _format_number(1000.0 * delta_mm[2]),
                _format_number(pre_residual[row]),
                _format_number(post_residual[row]),
            )
            for column, value in enumerate(values):
                item = QtWidgets.QTableWidgetItem(value)
                if column > 0:
                    item.setTextAlignment(
                        QtCore.Qt.AlignmentFlag.AlignRight
                        | QtCore.Qt.AlignmentFlag.AlignVCenter
                    )
                self.correction_steps_table.setItem(row, column, item)

        summary_lines: list[str] = []
        if move_delta.size:
            summary_lines.append(
                f"Applied total: {_format_axis_triplet_um(np.sum(move_delta, axis=0))}."
            )
        else:
            if in_progress:
                summary_lines.append("No correction moves have been applied yet.")
            else:
                summary_lines.append("No correction moves were applied.")

        if "estimated_readback_offset_mm" in result:
            summary_lines.append(
                "Estimated readback offset: "
                f"{_format_axis_triplet_um(result['estimated_readback_offset_mm'].values)}."
            )
        if "correction_readback_delta_mm" in result:
            summary_lines.append(
                "Next correction: "
                f"{_format_axis_triplet_um(result['correction_readback_delta_mm'].values)}."
            )

        self.correction_steps_summary_label.setText("\n".join(summary_lines))
        self.correction_steps_group.setVisible(True)

    def show_saved_calibration(self, display_name: str) -> None:
        self.calibration_status_label.setText(f"Saved calibration: {display_name}")

    def show_correction_in_progress(self) -> None:
        self._set_display_mode("correction")
        self.load_calibration_button.setEnabled(False)
        self.save_calibration_button.setEnabled(False)
        self.calibration_details_button.setEnabled(False)
        self.correct_sample_button.setEnabled(False)
        self.correction_mode_combo.setEnabled(False)
        self.auto_correction_checkbox.setEnabled(True)
        self.auto_correction_interval_spinbox.setEnabled(True)
        self.detect_shift_button.setEnabled(False)
        self.new_calibration_button.setEnabled(False)
        self.new_calibration_button.setText("Clear calibration")
        self.calibration_progress_bar.setVisible(True)
        self.calibration_progress_bar.setRange(0, 0)
        self._show_pending_correction_steps()
        self.calibration_status_label.setText("Correction in progress...")

    def show_correction_progress(self, result: xr.Dataset) -> None:
        self._set_display_mode("correction")
        moves = int(
            result.attrs.get("correction_iterations", result.sizes.get("move", 0))
        )
        residual, residual_label, residual_suffix = _correction_status_residual(result)

        if moves == 0:
            self.calibration_status_label.setText(
                "Correction in progress before first move; current "
                f"{residual_label} {_format_number(residual)}{residual_suffix}."
            )
        else:
            self.calibration_status_label.setText(
                "Correction in progress after "
                f"{moves} move(s); current {residual_label} "
                f"{_format_number(residual)}{residual_suffix}."
            )
        self.correction_warnings_text.setPlainText(_correction_warning_text(result))
        self._show_correction_steps(result, in_progress=True)

    def show_correction_result(self, result: xr.Dataset) -> None:
        self._set_display_mode("correction")
        self._set_loaded_idle_controls_enabled(True)
        self.calibration_progress_bar.setVisible(False)
        converged = bool(result.attrs.get("correction_converged", False))
        moves = int(
            result.attrs.get("correction_iterations", result.sizes.get("move", 0))
        )
        residual, residual_label, residual_suffix = _correction_status_residual(result)

        status = "converged" if converged else "did not converge"
        self.calibration_status_label.setText(
            "Correction "
            f"{status} after {moves} move(s); final {residual_label} "
            f"{_format_number(residual)}{residual_suffix}."
        )
        self.correction_warnings_text.setPlainText(_correction_warning_text(result))
        self._show_correction_steps(result)

    def show_detection_in_progress(self) -> None:
        self._set_display_mode("calibration")
        self.load_calibration_button.setEnabled(False)
        self.save_calibration_button.setEnabled(False)
        self.calibration_details_button.setEnabled(False)
        self.correct_sample_button.setEnabled(False)
        self.correction_mode_combo.setEnabled(False)
        self.auto_correction_checkbox.setEnabled(True)
        self.auto_correction_interval_spinbox.setEnabled(True)
        self.detect_shift_button.setEnabled(False)
        self.new_calibration_button.setEnabled(False)
        self.new_calibration_button.setText("Clear calibration")
        self.calibration_progress_bar.setVisible(True)
        self.calibration_progress_bar.setRange(0, 0)
        self.calibration_status_label.setText("Detecting shift...")

    def show_detection_result(self, result: xr.Dataset) -> None:
        self._set_display_mode("calibration")
        self.load_calibration_button.setEnabled(True)
        self.save_calibration_button.setEnabled(True)
        self.calibration_details_button.setEnabled(True)
        self.correct_sample_button.setEnabled(True)
        self.correction_mode_combo.setEnabled(True)
        self.auto_correction_checkbox.setEnabled(True)
        self.auto_correction_interval_spinbox.setEnabled(True)
        self.detect_shift_button.setEnabled(True)
        self.new_calibration_button.setEnabled(True)
        self.new_calibration_button.setText("Clear calibration")
        self.calibration_progress_bar.setVisible(False)

        residual = math.nan
        if "weighted_residual_px" in result:
            residual = float(result["weighted_residual_px"].values)

        if "estimated_readback_offset_mm" in result:
            offset_text = _format_axis_triplet_um(
                result["estimated_readback_offset_mm"].values
            )
        elif "detected_shift_um" in result:
            values_um = np.asarray(result["detected_shift_um"].values, dtype=float)
            offset_text = _format_axis_triplet_um(values_um / 1000.0)
        else:
            offset_text = "x=n/a um, y=n/a um, z=n/a um"

        self.calibration_status_label.setText(
            f"Detected shift: {offset_text}. "
            f"Weighted residual {_format_number(residual)} px."
        )

        warning_lines = [
            line.strip()
            for line in str(result.attrs.get("warnings", "")).splitlines()
            if line.strip()
        ]
        self.calibration_warnings_text.setPlainText(
            "\n".join(warning_lines) if warning_lines else "No detection warnings."
        )

    def build_details_dialog(
        self,
        calibration: xr.Dataset,
    ) -> QtWidgets.QDialog:
        summary = _calibration_summary(calibration)
        arrays = _calibration_arrays(calibration)
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle("Calibration Details")
        layout = QtWidgets.QVBoxLayout(dialog)

        tabs = QtWidgets.QTabWidget()
        tabs.setObjectName("calibration_details_tabs")

        summary_tab = QtWidgets.QWidget()
        summary_layout = QtWidgets.QVBoxLayout(summary_tab)
        summary_table = QtWidgets.QTableWidget(len(METRIC_ROWS), 2)
        summary_table.setObjectName("calibration_details_summary_table")
        summary_table.setHorizontalHeaderLabels(("metric", "value"))
        summary_table.setEditTriggers(
            QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers
        )
        summary_table.setSelectionMode(
            QtWidgets.QAbstractItemView.SelectionMode.NoSelection
        )
        summary_table.horizontalHeader().setSectionResizeMode(
            QtWidgets.QHeaderView.ResizeMode.ResizeToContents
        )
        for row, (key, label, tooltip) in enumerate(METRIC_ROWS):
            label_item = QtWidgets.QTableWidgetItem(label)
            value_item = QtWidgets.QTableWidgetItem(_format_metric_value(summary[key]))
            label_item.setToolTip(tooltip)
            value_item.setToolTip(tooltip)
            value_item.setTextAlignment(
                QtCore.Qt.AlignmentFlag.AlignRight
                | QtCore.Qt.AlignmentFlag.AlignVCenter
            )
            summary_table.setItem(row, 0, label_item)
            summary_table.setItem(row, 1, value_item)
        summary_layout.addWidget(QtWidgets.QLabel("Metrics"))
        summary_layout.addWidget(summary_table)

        repeatability_table = QtWidgets.QTableWidget(len(REPEATABILITY_ROWS), 2)
        repeatability_table.setObjectName("calibration_details_repeatability_table")
        repeatability_table.setHorizontalHeaderLabels(("metric", "value"))
        repeatability_table.setEditTriggers(
            QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers
        )
        repeatability_table.setSelectionMode(
            QtWidgets.QAbstractItemView.SelectionMode.NoSelection
        )
        repeatability_table.horizontalHeader().setSectionResizeMode(
            QtWidgets.QHeaderView.ResizeMode.ResizeToContents
        )
        repeatability = summary["repeatability"]
        for row, (key, label, tooltip) in enumerate(REPEATABILITY_ROWS):
            label_item = QtWidgets.QTableWidgetItem(label)
            if repeatability is None:
                value = "n/a"
            else:
                value = _format_number(repeatability[key])
            value_item = QtWidgets.QTableWidgetItem(value)
            label_item.setToolTip(tooltip)
            value_item.setToolTip(tooltip)
            value_item.setTextAlignment(
                QtCore.Qt.AlignmentFlag.AlignRight
                | QtCore.Qt.AlignmentFlag.AlignVCenter
            )
            repeatability_table.setItem(row, 0, label_item)
            repeatability_table.setItem(row, 1, value_item)
        summary_layout.addWidget(QtWidgets.QLabel("Repeatability"))
        summary_layout.addWidget(repeatability_table)

        warnings_text = QtWidgets.QPlainTextEdit()
        warnings_text.setObjectName("calibration_details_warnings_text")
        warnings_text.setReadOnly(True)
        warnings_text.setPlainText(
            _calibration_warning_text(summary, calibration.attrs)
        )
        summary_layout.addWidget(QtWidgets.QLabel("Warnings"))
        summary_layout.addWidget(warnings_text)
        tabs.addTab(summary_tab, "Summary")

        residuals_tab = QtWidgets.QWidget()
        residuals_layout = QtWidgets.QVBoxLayout(residuals_tab)
        residuals_graphics = pg.GraphicsLayoutWidget()
        residuals_graphics.setObjectName("calibration_details_residuals_layout")
        residuals_graphics.setMinimumHeight(360)
        _plot_residuals_on(
            calibration,
            _add_residual_projection_plots(residuals_graphics),
        )
        residuals_layout.addWidget(residuals_graphics)
        tabs.addTab(residuals_tab, "Residuals")

        matrices_tab = QtWidgets.QWidget()
        matrices_layout = QtWidgets.QVBoxLayout(matrices_tab)
        for title, row_labels, column_labels, values in (
            (
                "px_per_readback_mm",
                OBSERVATION_AXES,
                COMMAND_AXES,
                np.asarray(summary["px_per_readback_mm"], dtype=float).reshape(
                    len(OBSERVATION_AXES),
                    len(COMMAND_AXES),
                ),
            ),
            (
                "pixel_to_readback_mm",
                COMMAND_AXES,
                OBSERVATION_AXES,
                np.asarray(summary["pixel_to_readback"], dtype=float),
            ),
        ):
            table = QtWidgets.QTableWidget(len(row_labels), len(column_labels))
            table.setHorizontalHeaderLabels(column_labels)
            table.setVerticalHeaderLabels(row_labels)
            table.setEditTriggers(
                QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers
            )
            table.setSelectionMode(
                QtWidgets.QAbstractItemView.SelectionMode.NoSelection
            )
            table.horizontalHeader().setSectionResizeMode(
                QtWidgets.QHeaderView.ResizeMode.ResizeToContents
            )
            table.verticalHeader().setSectionResizeMode(
                QtWidgets.QHeaderView.ResizeMode.ResizeToContents
            )
            table.setMinimumHeight(88)
            table.setSizeAdjustPolicy(
                QtWidgets.QAbstractScrollArea.SizeAdjustPolicy.AdjustToContents
            )
            for row in range(table.rowCount()):
                for column in range(table.columnCount()):
                    item = QtWidgets.QTableWidgetItem(
                        _format_number(values[row, column])
                    )
                    item.setTextAlignment(
                        QtCore.Qt.AlignmentFlag.AlignRight
                        | QtCore.Qt.AlignmentFlag.AlignVCenter
                    )
                    table.setItem(row, column, item)

            matrices_layout.addWidget(QtWidgets.QLabel(title))
            matrices_layout.addWidget(table)
        matrices_layout.addStretch(1)
        tabs.addTab(matrices_tab, "Matrices")

        axes_tab = QtWidgets.QWidget()
        axes_layout = QtWidgets.QVBoxLayout(axes_tab)
        axis_headers = (
            "axis",
            "axis_scale_readback_mm",
            "response_px_per_readback_mm",
            "unclamped_scale_readback_mm",
            "scale_min_readback_mm",
            "scale_max_readback_mm",
        )
        axis_scale = np.asarray(summary["axis_scale_readback_mm"], dtype=float)
        axis_sensitivity = np.asarray(
            summary["axis_sensitivity_px_per_readback_mm"],
            dtype=float,
        )
        axis_scale_unclamped = np.asarray(
            summary["axis_scale_unclamped_readback_mm"],
            dtype=float,
        )
        axis_scale_bounds = np.asarray(
            summary["axis_scale_bounds_readback_mm"],
            dtype=float,
        )
        axis_table = QtWidgets.QTableWidget(len(COMMAND_AXES), len(axis_headers))
        axis_table.setObjectName("calibration_axes_table")
        axis_table.setHorizontalHeaderLabels(axis_headers)
        axis_table.setEditTriggers(
            QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers
        )
        axis_table.setSelectionMode(
            QtWidgets.QAbstractItemView.SelectionMode.NoSelection
        )
        axis_table.horizontalHeader().setSectionResizeMode(
            QtWidgets.QHeaderView.ResizeMode.ResizeToContents
        )
        for row, axis in enumerate(COMMAND_AXES):
            values = (
                axis,
                _format_number(axis_scale[row]),
                _format_number(axis_sensitivity[row]),
                _format_number(axis_scale_unclamped[row]),
                _format_number(axis_scale_bounds[row, 0]),
                _format_number(axis_scale_bounds[row, 1]),
            )
            for column, value in enumerate(values):
                item = QtWidgets.QTableWidgetItem(value)
                if column > 0:
                    item.setTextAlignment(
                        QtCore.Qt.AlignmentFlag.AlignRight
                        | QtCore.Qt.AlignmentFlag.AlignVCenter
                    )
                axis_table.setItem(row, column, item)
        axes_layout.addWidget(axis_table)
        axes_layout.addStretch(1)
        tabs.addTab(axes_tab, "Axes")

        samples_tab = QtWidgets.QWidget()
        samples_layout = QtWidgets.QVBoxLayout(samples_tab)

        headers = (
            "probe",
            "x_offset_cmd_mm",
            "y_offset_cmd_mm",
            "z_offset_cmd_mm",
            "x_readback_delta_mm",
            "y_readback_delta_mm",
            "z_readback_delta_mm",
            "measured_cam0_du_px",
            "measured_cam0_dv_px",
            "measured_cam1_du_px",
            "measured_cam1_dv_px",
            "predicted_cam0_du_px",
            "predicted_cam0_dv_px",
            "predicted_cam1_du_px",
            "predicted_cam1_dv_px",
            "residual_cam0_du_px",
            "residual_cam0_dv_px",
            "residual_cam1_du_px",
            "residual_cam1_dv_px",
            "residual_x_readback_mm",
            "residual_y_readback_mm",
            "residual_z_readback_mm",
            "readback_x_disagree_mm",
            "readback_y_disagree_mm",
            "readback_z_disagree_mm",
            "registration_warnings",
        )
        command_delta = arrays["command_delta"]
        readback_delta = arrays["readback_delta"]
        measured = arrays["measured_shift"]
        predicted = arrays["predicted_shift"]
        residual_px = arrays["residual_shift"]
        residual_readback = arrays["residual_readback"]
        readback_disagreement = arrays["readback_disagreement"]
        warnings = (
            np.asarray(calibration["probe_registration_warnings"].values, dtype=str)
            if "probe_registration_warnings" in calibration
            else np.full((command_delta.shape[0], len(CAMERAS)), "", dtype=str)
        )

        table = QtWidgets.QTableWidget(command_delta.shape[0], len(headers))
        table.setObjectName("calibration_samples_table")
        table.setHorizontalHeaderLabels(headers)
        table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        table.setSelectionBehavior(
            QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows
        )
        table.horizontalHeader().setSectionResizeMode(
            QtWidgets.QHeaderView.ResizeMode.ResizeToContents
        )
        table.horizontalHeader().setMinimumSectionSize(72)
        for row in range(command_delta.shape[0]):
            warning_row = warnings[row]
            warning_text = "; ".join(
                f"{camera}: {text}"
                for camera, text in zip(CAMERAS, warning_row, strict=True)
                if str(text)
            )
            values = (
                str(row),
                _format_number(command_delta[row, 0]),
                _format_number(command_delta[row, 1]),
                _format_number(command_delta[row, 2]),
                _format_number(readback_delta[row, 0]),
                _format_number(readback_delta[row, 1]),
                _format_number(readback_delta[row, 2]),
                _format_number(measured[row, 0, 0]),
                _format_number(measured[row, 0, 1]),
                _format_number(measured[row, 1, 0]),
                _format_number(measured[row, 1, 1]),
                _format_number(predicted[row, 0, 0]),
                _format_number(predicted[row, 0, 1]),
                _format_number(predicted[row, 1, 0]),
                _format_number(predicted[row, 1, 1]),
                _format_number(residual_px[row, 0, 0]),
                _format_number(residual_px[row, 0, 1]),
                _format_number(residual_px[row, 1, 0]),
                _format_number(residual_px[row, 1, 1]),
                _format_number(residual_readback[row, 0]),
                _format_number(residual_readback[row, 1]),
                _format_number(residual_readback[row, 2]),
                _format_number(readback_disagreement[row, 0]),
                _format_number(readback_disagreement[row, 1]),
                _format_number(readback_disagreement[row, 2]),
                warning_text,
            )
            for column, value in enumerate(values):
                item = QtWidgets.QTableWidgetItem(value)
                if column != len(values) - 1:
                    item.setTextAlignment(
                        QtCore.Qt.AlignmentFlag.AlignRight
                        | QtCore.Qt.AlignmentFlag.AlignVCenter
                    )
                table.setItem(row, column, item)

        samples_layout.addWidget(table)
        tabs.addTab(samples_tab, "Probes")

        layout.addWidget(tabs)

        close_button = QtWidgets.QPushButton("Close")
        close_button.clicked.connect(dialog.accept)
        layout.addWidget(close_button, alignment=QtCore.Qt.AlignmentFlag.AlignRight)

        return dialog

    def _plot_residuals(self, calibration: xr.Dataset) -> None:
        _plot_residuals_on(calibration, self.residual_plots)

from __future__ import annotations

import math

import numpy as np
import pyqtgraph as pg
import xarray as xr
from qtpy import QtCore, QtWidgets

from merlin_track_position.tracking.calibration_core import (
    CAMERAS,
    MEASUREMENT_WARNING_SUMMARY,
    OBSERVATION_AXES,
    PIXEL_AXES,
    STAGE_AXES,
    format_measurement_warning_lines,
)

RESIDUAL_PROJECTIONS: tuple[tuple[str, str, int, int], ...] = (
    ("x", "y", 0, 1),
    ("x", "z", 0, 2),
    ("y", "z", 1, 2),
)

REQUIRED_CALIBRATION_VARIABLES: tuple[str, ...] = (
    "image_cam0",
    "image_cam1",
    "stage_um",
    "measured_shift_px",
    "stage_to_pixel",
)


def _tooltip_html(
    computed: tuple[str, ...],
    interpretation: tuple[str, ...],
) -> str:
    paragraphs = "".join(f"<p>{item}</p>" for item in (*computed, *interpretation))
    return (
        "<qt><div style='white-space: normal; width: 360px;'>"
        f"{paragraphs}"
        "</div></qt>"
    )


METRIC_ROWS: tuple[tuple[str, str, str], ...] = (
    (
        "sample_count",
        "Samples",
        _tooltip_html(
            (
                "Number of calibration image pairs: <tt>stage_um.shape[0]</tt>.",
                "Includes the first origin sample and final return-to-origin sample.",
            ),
            (
                "More samples can constrain the fit better.",
                "Use residuals and condition number as the real quality checks.",
            ),
        ),
    ),
    (
        "condition_number",
        "Condition number",
        _tooltip_html(
            (
                "<tt>np.linalg.cond(stage_to_observation)</tt>.",
                "<tt>stage_to_observation</tt> is <tt>stage_to_pixel</tt> reshaped "
                "into the 4x3 mapping from x/y/z microns to two-camera pixel shifts.",
            ),
            (
                "Lower is better.",
                "High values mean the camera observations do not separate the three "
                "stage axes cleanly.",
                "Small pixel errors can then become large stage errors.",
            ),
        ),
    ),
    (
        "residual_rms_px",
        "Residual RMS px",
        _tooltip_html(
            (
                "Predicted shifts come from <tt>stage_um</tt> and "
                "<tt>stage_to_pixel</tt>.",
                "Residual is <tt>measured_shift_px - predicted_shift_px</tt>.",
                "<tt>sqrt(mean(sum(residual^2)))</tt> across both cameras and pixel axes.",
            ),
            (
                "Lower means measured image shifts match the linear model better.",
                "High values indicate noisy images or bad motor positions.",
            ),
        ),
    ),
    (
        "residual_max_px",
        "Residual max px",
        _tooltip_html(
            (
                "Largest per-sample pixel residual length.",
                "Uses the same two-camera residual vector as Residual RMS px.",
            ),
            (
                "Shows the worst sample in pixel units.",
                "A large max with modest RMS usually points to one bad image match or motor position.",
            ),
        ),
    ),
    (
        "residual_rms_um",
        "Residual RMS um",
        _tooltip_html(
            (
                "Pixel residuals are converted to stage space with "
                "<tt>pinv(stage_to_pixel)</tt>.",
                "Then reduced as RMS Euclidean length in x/y/z microns.",
            ),
            (
                "Approximate stage-space size of the calibration fit error.",
                "Lower is better and is usually easier to reason about than pixel residuals.",
            ),
        ),
    ),
    (
        "residual_max_um",
        "Residual max um",
        _tooltip_html(
            (
                "Largest stage-space residual length after converting pixel fit error to x/y/z microns.",
            ),
            (
                "Shows the worst calibration sample in stage units.",
                "Use it to spot outliers hidden by the RMS average.",
            ),
        ),
    ),
    (
        "return_to_origin_motor_error_um",
        "Return motor error um",
        _tooltip_html(
            (
                "Final row of <tt>stage_um</tt>.",
                "This is the measured motor offset vector (x, y, z) after return-to-origin.",
            ),
            (
                "Ideally near <tt>(0, 0, 0)</tt>.",
                "Component signs show the direction of remaining motor offset.",
            ),
        ),
    ),
    (
        "return_to_origin_image_error_um",
        "Return image error um",
        _tooltip_html(
            (
                "Final row of <tt>measured_shift_px</tt>.",
                "Converted to x/y/z stage displacement with <tt>pinv(stage_to_pixel)</tt>.",
            ),
            (
                "Estimates how far the final image looks from the first image.",
                "Ideally near <tt>(0, 0, 0)</tt>.",
                "Disagreement with Return motor error means encoders and image content disagree.",
            ),
        ),
    ),
)
REPEATABILITY_ROWS: tuple[tuple[str, str, str], ...] = (
    (
        "mean_rms_std_px",
        "Mean RMS std px",
        _tooltip_html(
            (
                "Samples are grouped by identical <tt>stage_um</tt> positions.",
                "For groups with at least two captures, compute sample std for each camera/pixel component.",
                "RMS-combine the four component stds, then average across repeated positions.",
            ),
            (
                "Lower means repeated captures at the same motor position agree better.",
                "With only the origin repeated, this equals Max RMS std px.",
            ),
        ),
    ),
    (
        "max_rms_std_px",
        "Max RMS std px",
        _tooltip_html(
            (
                "Same per-position RMS standard deviation as Mean RMS std px.",
                "Reports the largest value among repeated <tt>stage_um</tt> positions.",
            ),
            (
                "Highlights the worst repeated-position image stability.",
                "With only the origin repeated, this equals Mean RMS std px.",
            ),
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
    stage = np.asarray(dataset["stage_um"].values, dtype=float)
    measured_shift = np.asarray(dataset["measured_shift_px"].values, dtype=float)
    stage_to_pixel = np.asarray(dataset["stage_to_pixel"].values, dtype=float)
    stage_to_observation = stage_to_pixel.reshape(
        len(OBSERVATION_AXES),
        len(STAGE_AXES),
    )
    pixel_to_stage = np.linalg.pinv(stage_to_observation)
    predicted_shift = (stage @ stage_to_observation.T).reshape(
        stage.shape[0],
        len(CAMERAS),
        len(PIXEL_AXES),
    )
    residual_shift = measured_shift - predicted_shift
    residual_stage = residual_shift.reshape(
        stage.shape[0],
        len(OBSERVATION_AXES),
    ) @ pixel_to_stage.T
    return {
        "stage": stage,
        "measured_shift": measured_shift,
        "stage_to_pixel": stage_to_pixel,
        "stage_to_observation": stage_to_observation,
        "pixel_to_stage": pixel_to_stage,
        "predicted_shift": predicted_shift,
        "residual_shift": residual_shift,
        "residual_stage": residual_stage,
    }


def _repeatability_summary(
    stage: np.ndarray,
    measured_shift: np.ndarray,
) -> dict[str, float] | None:
    groups: dict[tuple[float, float, float], list[np.ndarray]] = {}
    for stage_row, shift_row in zip(stage, measured_shift, strict=True):
        key = (float(stage_row[0]), float(stage_row[1]), float(stage_row[2]))
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


def _validate_calibration_dataset(dataset: xr.Dataset) -> None:
    missing = tuple(
        name for name in REQUIRED_CALIBRATION_VARIABLES if name not in dataset
    )
    if missing:
        raise ValueError(
            "missing required calibration variables: " + ", ".join(missing)
        )

    stage = np.asarray(dataset["stage_um"].values, dtype=float)
    measured_shift = np.asarray(dataset["measured_shift_px"].values, dtype=float)
    stage_to_pixel = np.asarray(dataset["stage_to_pixel"].values, dtype=float)
    image_cam0 = np.asarray(dataset["image_cam0"].values)
    image_cam1 = np.asarray(dataset["image_cam1"].values)

    if stage.ndim != 2 or stage.shape[1] != len(STAGE_AXES) or stage.shape[0] == 0:
        raise ValueError("stage_um must have shape (sample, 3)")
    if not np.allclose(stage[0], 0.0, rtol=0.0, atol=1e-9):
        raise ValueError("stage_um[0] must be the origin")
    if not np.isfinite(stage).all():
        raise ValueError("stage_um must contain only finite values")
    expected_shift_shape = (stage.shape[0], len(CAMERAS), len(PIXEL_AXES))
    if measured_shift.shape != expected_shift_shape:
        raise ValueError("measured_shift_px must have shape (sample, camera, pixel_axis)")
    if not np.isfinite(measured_shift).all():
        raise ValueError("measured_shift_px must contain only finite values")
    if image_cam0.ndim != 3 or image_cam0.shape[0] != stage.shape[0]:
        raise ValueError("image_cam0 must have shape (sample, y_cam0, x_cam0)")
    if image_cam1.ndim != 3 or image_cam1.shape[0] != stage.shape[0]:
        raise ValueError("image_cam1 must have shape (sample, y_cam1, x_cam1)")
    if stage_to_pixel.shape != (len(CAMERAS), len(PIXEL_AXES), len(STAGE_AXES)):
        raise ValueError("stage_to_pixel must have shape (camera, pixel_axis, stage_axis)")
    if not np.isfinite(stage_to_pixel).all():
        raise ValueError("stage_to_pixel must contain only finite values")


def _calibration_summary(dataset: xr.Dataset) -> dict[str, object]:
    _validate_calibration_dataset(dataset)

    arrays = _calibration_arrays(dataset)
    stage = arrays["stage"]
    measured_shift = arrays["measured_shift"]
    residual_shift = arrays["residual_shift"]
    residual_stage = arrays["residual_stage"]
    pixel_to_stage = arrays["pixel_to_stage"]
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

    residual_stage_norms = np.linalg.norm(residual_stage, axis=1)
    finite_stage_norms = residual_stage_norms[np.isfinite(residual_stage_norms)]
    if finite_stage_norms.size:
        residual_rms_um = float(
            np.sqrt(np.mean(finite_stage_norms * finite_stage_norms))
        )
        residual_max_um = float(np.max(finite_stage_norms))
    else:
        residual_rms_um = math.nan
        residual_max_um = math.nan

    condition_number = float(np.linalg.cond(arrays["stage_to_observation"]))
    warning_lines = [
        line.strip()
        for line in str(dataset.attrs.get("warnings", "")).splitlines()
        if line.strip()
    ]
    measurement_warning_lines = _measurement_warning_lines(dataset)
    if measurement_warning_lines:
        warning_lines = [
            line
            for line in warning_lines
            if line != MEASUREMENT_WARNING_SUMMARY
        ]
        warning_lines.extend(measurement_warning_lines)
    warnings = tuple(dict.fromkeys(warning_lines))

    repeatability = _repeatability_summary(stage, measured_shift)
    if repeatability is None and "repeatability_rms_std_px" in dataset:
        rms_std = np.asarray(
            dataset["repeatability_rms_std_px"].values,
            dtype=float,
        ).ravel()
        finite_rms = rms_std[np.isfinite(rms_std)]
        if finite_rms.size:
            repeatability = {
                "mean_rms_std_px": float(np.nanmean(finite_rms)),
                "max_rms_std_px": float(np.nanmax(finite_rms)),
            }

    return_to_origin_image_error_um = measured_shift[-1].reshape(-1) @ pixel_to_stage.T
    return {
        "sample_count": int(stage.shape[0]),
        "condition_number": float(condition_number),
        "residual_rms_px": residual_rms_px,
        "residual_max_px": residual_max_px,
        "residual_rms_um": residual_rms_um,
        "residual_max_um": residual_max_um,
        "return_to_origin_motor_error_um": tuple(
            float(value) for value in np.asarray(stage[-1], dtype=float).reshape(-1)
        ),
        "return_to_origin_image_error_um": tuple(
            float(value)
            for value in np.asarray(return_to_origin_image_error_um, dtype=float).reshape(
                -1
            )
        ),
        "stage_to_pixel": arrays["stage_to_pixel"],
        "pixel_to_stage": pixel_to_stage,
        "warnings": warnings,
        "repeatability": repeatability,
    }


def _measurement_warning_lines(dataset: xr.Dataset) -> tuple[str, ...]:
    if "measurement_warnings" not in dataset:
        return ()

    stage = np.asarray(dataset["stage_um"].values, dtype=float)
    warning_values = np.asarray(dataset["measurement_warnings"].values, dtype=str)
    if warning_values.shape != (stage.shape[0], len(CAMERAS)):
        return ()

    measurement_warnings = tuple(
        tuple(
            tuple(
                line.strip()
                for line in str(warning_values[sample_index, camera_index]).splitlines()
                if line.strip()
            )
            for camera_index in range(len(CAMERAS))
        )
        for sample_index in range(stage.shape[0])
    )
    return format_measurement_warning_lines(measurement_warnings, stage)


class CalibrationPanel(QtWidgets.QWidget):
    def __init__(self, parent: QtWidgets.QWidget | None = None):
        super().__init__(parent)

        calibration_layout = QtWidgets.QVBoxLayout(self)
        calibration_layout.setContentsMargins(0, 0, 0, 0)

        calibration_button_layout = QtWidgets.QHBoxLayout()
        self.load_calibration_button = QtWidgets.QPushButton("Load calibration")
        self.save_calibration_button = QtWidgets.QPushButton("Save calibration")
        self.calibration_details_button = QtWidgets.QPushButton("Details...")
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

        content_layout = QtWidgets.QHBoxLayout()
        left_column = QtWidgets.QVBoxLayout()
        right_column = QtWidgets.QVBoxLayout()
        content_layout.addLayout(left_column, stretch=1)
        content_layout.addLayout(right_column, stretch=1)
        calibration_layout.addLayout(content_layout)

        warnings_group = QtWidgets.QGroupBox("Calibration Warnings")
        warnings_layout = QtWidgets.QVBoxLayout(warnings_group)
        self.calibration_warnings_text = QtWidgets.QPlainTextEdit()
        self.calibration_warnings_text.setReadOnly(True)
        self.calibration_warnings_text.setMaximumHeight(90)
        warnings_layout.addWidget(self.calibration_warnings_text)

        metrics_group = QtWidgets.QGroupBox("Calibration Metrics")
        metrics_layout = QtWidgets.QFormLayout(metrics_group)
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
        left_column.addWidget(metrics_group)

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
        right_column.addWidget(warnings_group)
        right_column.addWidget(self.repeatability_group)

        self.residual_graphics_layout = pg.GraphicsLayoutWidget()
        self.residual_graphics_layout.setObjectName("residual_projections_layout")
        self.residual_graphics_layout.setMinimumHeight(240)
        self.residual_plots: dict[str, pg.PlotItem] = {}
        for column, (x_label, y_label, _, _) in enumerate(RESIDUAL_PROJECTIONS):
            residual_plot = self.residual_graphics_layout.addPlot(row=0, col=column)
            residual_plot.setTitle(f"{x_label}-{y_label}")
            residual_plot.setLabel(
                "bottom", x_label, units="um", siPrefixEnableRanges=()
            )
            residual_plot.setLabel(
                "left", y_label, units="um", siPrefixEnableRanges=()
            )
            residual_plot.showGrid(x=True, y=True, alpha=0.25)
            residual_plot.setAspectLocked(True)
            self.residual_plots[f"{x_label}{y_label}"] = residual_plot
        calibration_layout.addWidget(self.residual_graphics_layout, stretch=1)

        self.reset()

    def reset(self) -> None:
        self.load_calibration_button.setEnabled(True)
        self.save_calibration_button.setEnabled(False)
        self.calibration_details_button.setEnabled(False)
        self.new_calibration_button.setEnabled(True)
        self.calibration_status_label.setText("No calibration loaded.")
        self.calibration_progress_bar.setVisible(False)
        self.calibration_progress_bar.setRange(0, 1)
        self.calibration_progress_bar.setValue(0)
        self.calibration_warnings_text.setPlainText("No calibration loaded.")
        for label in self.metric_labels.values():
            label.setText("n/a")
        for label in self.repeatability_labels.values():
            label.setText("n/a")
        self.repeatability_group.setVisible(False)
        for residual_plot in self.residual_plots.values():
            residual_plot.clear()

    def show_calibration_in_progress(self, total_steps: int | None = None) -> None:
        self.load_calibration_button.setEnabled(False)
        self.save_calibration_button.setEnabled(False)
        self.calibration_details_button.setEnabled(False)
        self.new_calibration_button.setEnabled(False)
        self.calibration_status_label.setText("New calibration in progress...")
        self.calibration_progress_bar.setVisible(True)
        if total_steps is None or total_steps < 1:
            self.calibration_progress_bar.setRange(0, 0)
            return
        self.calibration_progress_bar.setRange(0, int(total_steps))
        self.calibration_progress_bar.setValue(0)
        self.calibration_progress_bar.setFormat(f"0 / {int(total_steps)} samples")

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
        self.calibration_progress_bar.setFormat(
            f"{completed} / {total_steps} samples"
        )
        self.calibration_status_label.setText(
            "New calibration in progress. "
            f"Stage ({_format_number(dx)}, {_format_number(dy)}, "
            f"{_format_number(dz)}) um. "
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
        self.calibration_progress_bar.setFormat(
            f"{completed} / {total} registrations"
        )
        self.calibration_status_label.setText(
            "Processing calibration scans. "
            f"Elapsed {_format_duration(elapsed_s)}, ETA {_format_duration(eta_s)}."
        )

    def show_loaded_calibration(
        self,
        calibration: xr.Dataset,
        display_name: str,
    ) -> None:
        summary = _calibration_summary(calibration)
        self.load_calibration_button.setEnabled(True)
        self.save_calibration_button.setEnabled(True)
        self.calibration_details_button.setEnabled(True)
        self.new_calibration_button.setEnabled(True)
        self.calibration_progress_bar.setVisible(False)
        self.calibration_status_label.setText(
            f"Loaded calibration: {display_name} ({summary['sample_count']} samples)"
        )

        warnings = summary["warnings"]
        warnings_text = "\n".join(warnings) if warnings else "No calibration warnings."
        self.calibration_warnings_text.setPlainText(warnings_text)

        for key, _, _ in METRIC_ROWS:
            self.metric_labels[key].setText(_format_metric_value(summary[key]))

        repeatability = summary["repeatability"]
        self.repeatability_group.setVisible(repeatability is not None)
        for key, label in self.repeatability_labels.items():
            if repeatability is None:
                label.setText("n/a")
            else:
                label.setText(_format_number(repeatability[key]))

        self._plot_residuals(calibration)

    def show_saved_calibration(self, display_name: str) -> None:
        self.calibration_status_label.setText(f"Saved calibration: {display_name}")

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

        matrices_tab = QtWidgets.QWidget()
        matrices_layout = QtWidgets.QVBoxLayout(matrices_tab)
        for title, row_labels, column_labels, values in (
            (
                "stage_to_pixel",
                OBSERVATION_AXES,
                STAGE_AXES,
                np.asarray(summary["stage_to_pixel"], dtype=float).reshape(
                    len(OBSERVATION_AXES),
                    len(STAGE_AXES),
                ),
            ),
            (
                "pixel_to_stage",
                STAGE_AXES,
                OBSERVATION_AXES,
                np.asarray(summary["pixel_to_stage"], dtype=float),
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
            table.horizontalHeader().setMinimumSectionSize(88)
            table.verticalHeader().setSectionResizeMode(
                QtWidgets.QHeaderView.ResizeMode.ResizeToContents
            )
            table.verticalHeader().setMinimumWidth(64)
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

        samples_tab = QtWidgets.QWidget()
        samples_layout = QtWidgets.QVBoxLayout(samples_tab)

        headers = (
            "sample",
            "x_um",
            "y_um",
            "z_um",
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
            "residual_x_um",
            "residual_y_um",
            "residual_z_um",
            "measurement_warnings",
        )
        stage = arrays["stage"]
        measured = arrays["measured_shift"]
        predicted = arrays["predicted_shift"]
        residual_px = arrays["residual_shift"]
        residual_um = arrays["residual_stage"]
        warnings = (
            np.asarray(calibration["measurement_warnings"].values, dtype=str)
            if "measurement_warnings" in calibration
            else np.full((stage.shape[0], len(CAMERAS)), "", dtype=str)
        )

        table = QtWidgets.QTableWidget(stage.shape[0], len(headers))
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
        for row in range(stage.shape[0]):
            warning_row = warnings[row]
            if np.ndim(warning_row) == 0:
                warning_text = str(warning_row)
            else:
                warning_text = "; ".join(
                    f"{camera}: {text}"
                    for camera, text in zip(CAMERAS, warning_row, strict=True)
                    if str(text)
                )
            values = (
                str(row),
                _format_number(stage[row, 0]),
                _format_number(stage[row, 1]),
                _format_number(stage[row, 2]),
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
                _format_number(residual_um[row, 0]),
                _format_number(residual_um[row, 1]),
                _format_number(residual_um[row, 2]),
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
        tabs.addTab(samples_tab, "Samples")

        images_tab = QtWidgets.QWidget()
        images_layout = QtWidgets.QVBoxLayout(images_tab)
        if "image_cam0" not in calibration or "image_cam1" not in calibration:
            images_layout.addWidget(
                QtWidgets.QLabel("No calibration images in dataset.")
            )
            images_layout.addStretch(1)
        else:
            images_by_camera = {
                "cam0": np.asarray(calibration["image_cam0"].values),
                "cam1": np.asarray(calibration["image_cam1"].values),
            }
            stage = np.asarray(calibration["stage_um"].values, dtype=float)
            if any(
                images.ndim != 3 or images.shape[0] != stage.shape[0]
                for images in images_by_camera.values()
            ):
                images_layout.addWidget(
                    QtWidgets.QLabel("No usable calibration image stacks.")
                )
                images_layout.addStretch(1)
            else:
                controls_layout = QtWidgets.QHBoxLayout()
                camera_selector = QtWidgets.QComboBox()
                camera_selector.setObjectName("calibration_image_camera_selector")
                camera_selector.addItems(CAMERAS)
                sample_selector = QtWidgets.QSpinBox()
                sample_selector.setObjectName("calibration_image_sample_selector")
                sample_selector.setRange(0, stage.shape[0] - 1)
                stage_label = QtWidgets.QLabel()
                stage_label.setTextInteractionFlags(
                    QtCore.Qt.TextInteractionFlag.TextSelectableByMouse
                )
                controls_layout.addWidget(QtWidgets.QLabel("Camera"))
                controls_layout.addWidget(camera_selector)
                controls_layout.addWidget(QtWidgets.QLabel("Sample"))
                controls_layout.addWidget(sample_selector)
                controls_layout.addWidget(stage_label, stretch=1)
                images_layout.addLayout(controls_layout)

                image_plot = pg.PlotWidget()
                image_plot.setObjectName("calibration_image_plot")
                image_plot.setAspectLocked(True)
                image_plot.invertY(True)
                image_plot.setLabel(
                    "bottom", "x", units="px", siPrefixEnableRanges=()
                )
                image_plot.setLabel(
                    "left", "y", units="px", siPrefixEnableRanges=()
                )
                image_item = pg.ImageItem(axisOrder="row-major")
                image_plot.addItem(image_item)
                images_layout.addWidget(image_plot, stretch=1)

                def update_image() -> None:
                    camera = str(camera_selector.currentText())
                    index = int(sample_selector.value())
                    images = images_by_camera[camera]
                    image_item.setImage(images[index], autoLevels=True)
                    stage_label.setText(
                        f"stage: ({_format_number(stage[index, 0])}, "
                        f"{_format_number(stage[index, 1])}, "
                        f"{_format_number(stage[index, 2])}) um"
                    )

                camera_selector.currentTextChanged.connect(lambda _: update_image())
                sample_selector.valueChanged.connect(lambda _: update_image())
                update_image()
        tabs.addTab(images_tab, "Images")
        layout.addWidget(tabs)

        close_button = QtWidgets.QPushButton("Close")
        close_button.clicked.connect(dialog.accept)
        layout.addWidget(close_button, alignment=QtCore.Qt.AlignmentFlag.AlignRight)

        return dialog

    def _plot_residuals(self, calibration: xr.Dataset) -> None:
        for residual_plot in self.residual_plots.values():
            residual_plot.clear()
        arrays = _calibration_arrays(calibration)
        stage = arrays["stage"]
        residual = arrays["residual_stage"]

        for x_label, y_label, x_index, y_index in RESIDUAL_PROJECTIONS:
            residual_plot = self.residual_plots[f"{x_label}{y_label}"]
            residual_plot.plot(
                stage[:, x_index],
                stage[:, y_index],
                pen=None,
                symbol="o",
                symbolBrush=pg.mkBrush("#1f77b4"),
                symbolPen=pg.mkPen("#1f77b4"),
            )

            x_values: list[float] = []
            y_values: list[float] = []
            residual_x_values: list[float] = []
            residual_y_values: list[float] = []
            for stage_row, residual_row in zip(stage, residual, strict=True):
                x0 = stage_row[x_index]
                y0 = stage_row[y_index]
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

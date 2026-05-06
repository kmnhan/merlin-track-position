from __future__ import annotations

import math

import numpy as np
import pyqtgraph as pg
import xarray as xr
from qtpy import QtCore, QtWidgets

REQUIRED_CALIBRATION_VARIABLES: tuple[str, ...] = (
    "stage_um",
    "residual_stage_um",
    "residual_shift_px",
    "stage_to_pixel",
    "pixel_to_stage",
    "condition_number",
)
METRIC_ROWS: tuple[tuple[str, str, str], ...] = (
    ("sample_count", "Samples", "Number of calibration positions used in the fit."),
    (
        "condition_number",
        "Condition number",
        "Numerical sensitivity of the fitted calibration matrix. Lower is better; high values mean the two stage directions are hard to separate.",
    ),
    (
        "residual_rms_px",
        "Residual RMS px",
        "Root-mean-square length of the pixel residuals after fitting. Lower means measured shifts match the model more closely.",
    ),
    (
        "residual_max_px",
        "Residual max px",
        "Largest pixel residual length across all calibration samples.",
    ),
    (
        "residual_rms_um",
        "Residual RMS um",
        "Root-mean-square residual after converting pixel fit error back to stage units.",
    ),
    (
        "residual_max_um",
        "Residual max um",
        "Largest converted stage-space residual across all calibration samples.",
    ),
)
REPEATABILITY_ROWS: tuple[tuple[str, str], ...] = (
    ("position_count", "Positions"),
    ("capture_count", "Captures"),
    ("mean_rms_std_px", "Mean RMS std px"),
    ("max_rms_std_px", "Max RMS std px"),
)


def _format_number(value: object) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "n/a"
    return f"{number:.4g}"


def _validate_calibration_dataset(dataset: xr.Dataset) -> None:
    missing = tuple(
        name for name in REQUIRED_CALIBRATION_VARIABLES if name not in dataset
    )
    if missing:
        raise ValueError(
            "missing required calibration variables: " + ", ".join(missing)
        )

    stage = np.asarray(dataset["stage_um"].values, dtype=float)
    residual_stage = np.asarray(dataset["residual_stage_um"].values, dtype=float)
    residual_shift = np.asarray(dataset["residual_shift_px"].values, dtype=float)
    stage_to_pixel = np.asarray(dataset["stage_to_pixel"].values, dtype=float)
    pixel_to_stage = np.asarray(dataset["pixel_to_stage"].values, dtype=float)
    condition_number = np.asarray(dataset["condition_number"].values, dtype=float)

    if stage.ndim != 2 or stage.shape[1] != 2 or stage.shape[0] == 0:
        raise ValueError("stage_um must have shape (sample, 2)")
    if residual_stage.shape != stage.shape:
        raise ValueError("residual_stage_um must have the same shape as stage_um")
    if residual_shift.shape != stage.shape:
        raise ValueError("residual_shift_px must have the same shape as stage_um")
    if stage_to_pixel.shape != (2, 2):
        raise ValueError("stage_to_pixel must have shape (2, 2)")
    if pixel_to_stage.shape != (2, 2):
        raise ValueError("pixel_to_stage must have shape (2, 2)")
    if condition_number.size != 1:
        raise ValueError("condition_number must be scalar")


def _calibration_summary(dataset: xr.Dataset) -> dict[str, object]:
    _validate_calibration_dataset(dataset)

    residual_shift = np.asarray(dataset["residual_shift_px"].values, dtype=float)
    residual_stage = np.asarray(dataset["residual_stage_um"].values, dtype=float)
    residual_shift_norms = np.linalg.norm(residual_shift, axis=1)
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

    condition_number = np.asarray(
        dataset["condition_number"].values, dtype=float
    ).reshape(-1)[0]
    warnings = tuple(
        line.strip()
        for line in str(dataset.attrs.get("warnings", "")).splitlines()
        if line.strip()
    )

    repeatability: dict[str, float | int] | None = None
    if all(
        name in dataset for name in ("repeatability_count", "repeatability_rms_std_px")
    ):
        counts = np.asarray(dataset["repeatability_count"].values, dtype=float).ravel()
        rms_std = np.asarray(
            dataset["repeatability_rms_std_px"].values, dtype=float
        ).ravel()
        if counts.size and counts.shape == rms_std.shape:
            finite_rms = rms_std[np.isfinite(rms_std)]
            if finite_rms.size:
                repeatability = {
                    "position_count": int(counts.size),
                    "capture_count": int(np.nansum(counts)),
                    "mean_rms_std_px": float(np.nanmean(finite_rms)),
                    "max_rms_std_px": float(np.nanmax(finite_rms)),
                }

    return {
        "sample_count": int(np.asarray(dataset["stage_um"].values).shape[0]),
        "condition_number": float(condition_number),
        "residual_rms_px": residual_rms_px,
        "residual_max_px": residual_max_px,
        "residual_rms_um": residual_rms_um,
        "residual_max_um": residual_max_um,
        "stage_to_pixel": np.asarray(dataset["stage_to_pixel"].values, dtype=float),
        "pixel_to_stage": np.asarray(dataset["pixel_to_stage"].values, dtype=float),
        "warnings": warnings,
        "repeatability": repeatability,
    }


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
        for key, label in REPEATABILITY_ROWS:
            value_label = QtWidgets.QLabel()
            value_label.setTextInteractionFlags(
                QtCore.Qt.TextInteractionFlag.TextSelectableByMouse
            )
            self.repeatability_labels[key] = value_label
            repeatability_layout.addRow(label, value_label)
        left_column.addWidget(self.repeatability_group)
        right_column.addWidget(warnings_group)

        self.residual_plot = pg.PlotWidget()
        self.residual_plot.setMinimumHeight(240)
        self.residual_plot.setLabel("bottom", "stage_a", units="um")
        self.residual_plot.setLabel("left", "stage_b", units="um")
        self.residual_plot.showGrid(x=True, y=True, alpha=0.25)
        self.residual_plot.setAspectLocked(True)
        calibration_layout.addWidget(self.residual_plot, stretch=1)

        self.reset()

    def reset(self) -> None:
        self.load_calibration_button.setEnabled(True)
        self.save_calibration_button.setEnabled(False)
        self.calibration_details_button.setEnabled(False)
        self.new_calibration_button.setEnabled(True)
        self.calibration_status_label.setText("No calibration loaded.")
        self.calibration_warnings_text.setPlainText("No calibration loaded.")
        for label in self.metric_labels.values():
            label.setText("n/a")
        for label in self.repeatability_labels.values():
            label.setText("n/a")
        self.repeatability_group.setVisible(False)
        self.residual_plot.clear()

    def show_calibration_in_progress(self) -> None:
        self.load_calibration_button.setEnabled(False)
        self.save_calibration_button.setEnabled(False)
        self.calibration_details_button.setEnabled(False)
        self.new_calibration_button.setEnabled(False)
        self.calibration_status_label.setText("New calibration in progress...")

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
        self.calibration_status_label.setText(
            f"Loaded calibration: {display_name} ({summary['sample_count']} samples)"
        )

        warnings = summary["warnings"]
        warnings_text = "\n".join(warnings) if warnings else "No calibration warnings."
        self.calibration_warnings_text.setPlainText(warnings_text)

        for key, _, _ in METRIC_ROWS:
            self.metric_labels[key].setText(_format_number(summary[key]))

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
                ("du_px", "dv_px"),
                ("stage_a_um", "stage_b_um"),
                np.asarray(summary["stage_to_pixel"], dtype=float),
            ),
            (
                "pixel_to_stage",
                ("stage_a_um", "stage_b_um"),
                ("du_px", "dv_px"),
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
            "stage_a_um",
            "stage_b_um",
            "measured_du_px",
            "measured_dv_px",
            "predicted_du_px",
            "predicted_dv_px",
            "residual_du_px",
            "residual_dv_px",
            "residual_stage_a_um",
            "residual_stage_b_um",
            "measurement_warnings",
        )
        stage = np.asarray(calibration["stage_um"].values, dtype=float)
        measured = np.asarray(calibration["measured_shift_px"].values, dtype=float)
        predicted = np.asarray(calibration["predicted_shift_px"].values, dtype=float)
        residual_px = np.asarray(calibration["residual_shift_px"].values, dtype=float)
        residual_um = np.asarray(calibration["residual_stage_um"].values, dtype=float)
        warnings = (
            np.asarray(calibration["measurement_warnings"].values, dtype=str)
            if "measurement_warnings" in calibration
            else np.full(stage.shape[0], "", dtype=str)
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
            values = (
                str(row),
                _format_number(stage[row, 0]),
                _format_number(stage[row, 1]),
                _format_number(measured[row, 0]),
                _format_number(measured[row, 1]),
                _format_number(predicted[row, 0]),
                _format_number(predicted[row, 1]),
                _format_number(residual_px[row, 0]),
                _format_number(residual_px[row, 1]),
                _format_number(residual_um[row, 0]),
                _format_number(residual_um[row, 1]),
                str(warnings[row]),
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
        if "image" not in calibration:
            images_layout.addWidget(
                QtWidgets.QLabel("No calibration images in dataset.")
            )
            images_layout.addStretch(1)
        else:
            images = np.asarray(calibration["image"].values)
            stage = np.asarray(calibration["stage_um"].values, dtype=float)
            if images.ndim != 3 or images.shape[0] != stage.shape[0]:
                images_layout.addWidget(
                    QtWidgets.QLabel("No usable calibration image stack.")
                )
                images_layout.addStretch(1)
            else:
                controls_layout = QtWidgets.QHBoxLayout()
                sample_selector = QtWidgets.QSpinBox()
                sample_selector.setObjectName("calibration_image_sample_selector")
                sample_selector.setRange(0, images.shape[0] - 1)
                stage_label = QtWidgets.QLabel()
                stage_label.setTextInteractionFlags(
                    QtCore.Qt.TextInteractionFlag.TextSelectableByMouse
                )
                controls_layout.addWidget(QtWidgets.QLabel("Sample"))
                controls_layout.addWidget(sample_selector)
                controls_layout.addWidget(stage_label, stretch=1)
                images_layout.addLayout(controls_layout)

                image_plot = pg.PlotWidget()
                image_plot.setObjectName("calibration_image_plot")
                image_plot.setAspectLocked(True)
                image_plot.invertY(True)
                image_plot.setLabel("bottom", "x", units="px")
                image_plot.setLabel("left", "y", units="px")
                image_item = pg.ImageItem(axisOrder="row-major")
                image_plot.addItem(image_item)
                images_layout.addWidget(image_plot, stretch=1)

                def update_image(index: int) -> None:
                    image_item.setImage(images[index], autoLevels=True)
                    stage_label.setText(
                        f"stage: ({_format_number(stage[index, 0])}, "
                        f"{_format_number(stage[index, 1])}) um"
                    )

                sample_selector.valueChanged.connect(update_image)
                update_image(0)
        tabs.addTab(images_tab, "Images")
        layout.addWidget(tabs)

        close_button = QtWidgets.QPushButton("Close")
        close_button.clicked.connect(dialog.accept)
        layout.addWidget(close_button, alignment=QtCore.Qt.AlignmentFlag.AlignRight)

        return dialog

    def _plot_residuals(self, calibration: xr.Dataset) -> None:
        self.residual_plot.clear()
        stage = np.asarray(calibration["stage_um"].values, dtype=float)
        residual = np.asarray(calibration["residual_stage_um"].values, dtype=float)
        self.residual_plot.plot(
            stage[:, 0],
            stage[:, 1],
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
            x0, y0 = stage_row
            dx, dy = residual_row
            if not np.isfinite((x0, y0, dx, dy)).all():
                continue
            x1 = x0 + dx
            y1 = y0 + dy
            x_values.extend([float(x0), float(x1), math.nan])
            y_values.extend([float(y0), float(y1), math.nan])
            residual_x_values.append(float(x1))
            residual_y_values.append(float(y1))

        if x_values:
            self.residual_plot.plot(
                x_values,
                y_values,
                pen=pg.mkPen("#d62728", width=2),
            )
            self.residual_plot.plot(
                residual_x_values,
                residual_y_values,
                pen=None,
                symbol="o",
                symbolBrush=pg.mkBrush("#d62728"),
                symbolPen=pg.mkPen("#d62728"),
            )

import math
import os
import unittest

import numpy as np
import pyqtgraph as pg
import xarray as xr

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from merlin_track_position.constants import (
    IMAGE_HEIGHT_CAM0,
    IMAGE_HEIGHT_CAM1,
    IMAGE_WIDTH_CAM0,
    IMAGE_WIDTH_CAM1,
)
from merlin_track_position.interface.calibration_panel import (
    CalibrationPanel,
    _calibration_summary,
)
from merlin_track_position.interface.main_window import (
    _MainWindowGUI,
    _clamp_roi_geometry,
    _default_roi_geometry,
    _validate_calibration_dataset,
)
from merlin_track_position.tracking.calibration_core import (
    CAMERAS,
    OBSERVATION_AXES,
    PIXEL_AXES,
    STAGE_AXES,
)
from qtpy import QtWidgets


def _synthetic_calibration(
    stage: np.ndarray,
    stage_to_pixel: np.ndarray,
    images_cam0: np.ndarray | None = None,
    images_cam1: np.ndarray | None = None,
) -> xr.Dataset:
    stage = np.asarray(stage, dtype=float)
    stage_to_pixel = np.asarray(stage_to_pixel, dtype=float)
    stage_to_observation = stage_to_pixel.reshape(len(OBSERVATION_AXES), len(STAGE_AXES))
    measured = (stage @ stage_to_observation.T).reshape(
        stage.shape[0],
        len(CAMERAS),
        len(PIXEL_AXES),
    )
    predicted = measured.copy()
    residual_px = measured - predicted
    pixel_to_stage = np.linalg.pinv(stage_to_observation)
    residual_um = residual_px.reshape(stage.shape[0], len(OBSERVATION_AXES)) @ pixel_to_stage.T
    return_to_origin_motor_error_um = stage[-1]
    return_to_origin_image_error_px = measured[-1]
    return_to_origin_image_error_um = return_to_origin_image_error_px.reshape(-1) @ pixel_to_stage.T

    coords = {
        "sample": np.arange(stage.shape[0], dtype=np.int64),
        "stage_axis": list(STAGE_AXES),
        "camera": list(CAMERAS),
        "pixel_axis": list(PIXEL_AXES),
        "observation_axis": list(OBSERVATION_AXES),
    }
    data_vars = {
        "stage_to_pixel": (
            ("camera", "pixel_axis", "stage_axis"),
            stage_to_pixel,
            {"units": "px/um"},
        ),
        "pixel_to_stage": (
            ("stage_axis", "observation_axis"),
            pixel_to_stage,
            {"units": "um/px"},
        ),
        "reference_stage_um": (
            ("stage_axis",),
            np.zeros(len(STAGE_AXES), dtype=float),
            {"units": "um"},
        ),
        "condition_number": ((), float(np.linalg.cond(stage_to_observation))),
        "origin_stability_um": ((), 5.0, {"units": "um"}),
        "return_to_origin_motor_error_um": (
            ("stage_axis",),
            return_to_origin_motor_error_um,
            {"units": "um"},
        ),
        "return_to_origin_motor_error_norm_um": (
            (),
            float(np.linalg.norm(return_to_origin_motor_error_um)),
            {"units": "um"},
        ),
        "return_to_origin_image_error_px": (
            ("camera", "pixel_axis"),
            return_to_origin_image_error_px,
            {"units": "px"},
        ),
        "return_to_origin_image_error_um": (
            ("stage_axis",),
            return_to_origin_image_error_um,
            {"units": "um"},
        ),
        "return_to_origin_image_error_norm_um": (
            (),
            float(np.linalg.norm(return_to_origin_image_error_um)),
            {"units": "um"},
        ),
        "stage_um": (("sample", "stage_axis"), stage, {"units": "um"}),
        "measured_shift_px": (
            ("sample", "camera", "pixel_axis"),
            measured,
            {"units": "px"},
        ),
        "predicted_shift_px": (
            ("sample", "camera", "pixel_axis"),
            predicted,
            {"units": "px"},
        ),
        "residual_shift_px": (
            ("sample", "camera", "pixel_axis"),
            residual_px,
            {"units": "px"},
        ),
        "residual_stage_um": (("sample", "stage_axis"), residual_um, {"units": "um"}),
        "measurement_warnings": (
            ("sample", "camera"),
            np.full((stage.shape[0], len(CAMERAS)), "", dtype=str),
        ),
    }

    repeatability = _synthetic_repeatability(stage, measured)
    if repeatability is not None:
        repeatability_stage, repeatability_count, _, repeatability_std = repeatability
        coords["repeatability_position"] = np.arange(
            repeatability_stage.shape[0],
            dtype=np.int64,
        )
        data_vars["repeatability_stage_um"] = (
            ("repeatability_position", "stage_axis"),
            repeatability_stage,
            {"units": "um"},
        )
        data_vars["repeatability_count"] = (
            ("repeatability_position",),
            repeatability_count,
        )
        data_vars["repeatability_rms_std_px"] = (
            ("repeatability_position",),
            np.sqrt(np.mean(repeatability_std * repeatability_std, axis=(1, 2))),
            {"units": "px"},
        )

    if images_cam0 is None:
        images_cam0 = np.zeros((stage.shape[0], 4, 5), dtype=float)
    if images_cam1 is None:
        images_cam1 = np.zeros((stage.shape[0], 6, 7), dtype=float)
    images_cam0 = np.asarray(images_cam0, dtype=float)
    images_cam1 = np.asarray(images_cam1, dtype=float)
    coords["y_cam0"] = np.arange(images_cam0.shape[1], dtype=np.int64)
    coords["x_cam0"] = np.arange(images_cam0.shape[2], dtype=np.int64)
    coords["y_cam1"] = np.arange(images_cam1.shape[1], dtype=np.int64)
    coords["x_cam1"] = np.arange(images_cam1.shape[2], dtype=np.int64)
    data_vars["image_cam0"] = (
        ("sample", "y_cam0", "x_cam0"),
        images_cam0,
        {"description": "camera 0 calibration grayscale image stack"},
    )
    data_vars["image_cam1"] = (
        ("sample", "y_cam1", "x_cam1"),
        images_cam1,
        {"description": "camera 1 calibration grayscale image stack"},
    )

    return xr.Dataset(
        data_vars=data_vars,
        coords=coords,
        attrs={
            "format": "merlin-track-position calibration",
            "format_version": "1",
            "model": "through_origin_linear_stereo",
            "warnings": "",
        },
    )


def _synthetic_repeatability(
    stage: np.ndarray,
    pixels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    groups: dict[tuple[float, float, float], list[np.ndarray]] = {}
    for stage_row, pixel_row in zip(stage, pixels, strict=True):
        groups.setdefault(
            (float(stage_row[0]), float(stage_row[1]), float(stage_row[2])),
            [],
        ).append(pixel_row)

    stage_rows: list[np.ndarray] = []
    counts: list[int] = []
    means: list[np.ndarray] = []
    stds: list[np.ndarray] = []
    for key, rows in groups.items():
        if len(rows) < 2:
            continue
        values = np.stack(rows, axis=0)
        stage_rows.append(np.asarray(key, dtype=float))
        counts.append(len(rows))
        means.append(np.mean(values, axis=0))
        stds.append(np.std(values, axis=0, ddof=1))

    if not stage_rows:
        return None
    return (
        np.vstack(stage_rows),
        np.asarray(counts, dtype=np.int64),
        np.stack(means, axis=0),
        np.stack(stds, axis=0),
    )


class GUIHelperTests(unittest.TestCase):
    def test_default_roi_geometry_is_centered_quarter_cam0_image(self):
        self.assertEqual(_default_roi_geometry(), (264.0, 180.0, 176.0, 120.0))

    def test_clamp_roi_geometry_keeps_roi_inside_cam0_image(self):
        self.assertEqual(
            _clamp_roi_geometry((-10.0, 500.0, 900.0, -4.0)),
            (0.0, 479.0, 704.0, 1.0),
        )
        self.assertEqual(
            _clamp_roi_geometry((650.0, 470.0, 100.0, 40.0)),
            (604.0, 440.0, 100.0, 40.0),
        )

    def test_clamp_roi_geometry_uses_default_for_non_finite_values(self):
        self.assertEqual(
            _clamp_roi_geometry((math.nan, 0.0, 100.0, 100.0)),
            _default_roi_geometry(),
        )

    def test_calibration_summary_reports_core_metrics_and_repeatability(self):
        stage_to_pixel = np.array(
            [
                [[0.5, -0.1, 0.2], [0.2, 0.4, -0.1]],
                [[-0.3, 0.2, 0.4], [0.1, -0.2, 0.3]],
            ]
        )
        stage = np.array(
            [
                [0.0, 0.0, 0.0],
                [20.0, 0.0, 0.0],
                [0.0, 20.0, 0.0],
                [0.0, 0.0, 20.0],
                [20.0, 0.0, 0.0],
                [0.0, 20.0, 0.0],
            ]
        )
        calibration = _synthetic_calibration(stage, stage_to_pixel)

        summary = _calibration_summary(calibration)

        self.assertEqual(summary["sample_count"], 6)
        self.assertAlmostEqual(summary["residual_rms_px"], 0.0)
        self.assertAlmostEqual(summary["residual_max_px"], 0.0)
        self.assertAlmostEqual(summary["residual_rms_um"], 0.0)
        np.testing.assert_allclose(summary["stage_to_pixel"], stage_to_pixel)
        self.assertEqual(summary["warnings"], ())
        self.assertEqual(summary["repeatability"]["position_count"], 2)
        self.assertEqual(summary["repeatability"]["capture_count"], 4)
        self.assertAlmostEqual(summary["repeatability"]["max_rms_std_px"], 0.0)

    def test_validate_calibration_dataset_rejects_missing_required_field(self):
        stage_to_pixel = np.array(
            [
                [[0.5, 0.0, 0.2], [0.0, 0.4, -0.1]],
                [[-0.2, 0.1, 0.3], [0.1, -0.1, 0.2]],
            ]
        )
        stage = np.array(
            [
                [0.0, 0.0, 0.0],
                [20.0, 0.0, 0.0],
                [0.0, 20.0, 0.0],
                [0.0, 0.0, 20.0],
            ]
        )
        calibration = _synthetic_calibration(stage, stage_to_pixel)
        broken = calibration.drop_vars("residual_stage_um")

        with self.assertRaisesRegex(ValueError, "residual_stage_um"):
            _validate_calibration_dataset(broken)


class MainWindowGUISmokeTests(unittest.TestCase):
    def test_main_window_gui_constructs_offscreen(self):
        app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        window = _MainWindowGUI()
        try:
            self.assertIsInstance(window.image_graphics_layout, pg.GraphicsLayoutWidget)
            self.assertEqual(
                window.image_items["cam0"].image.shape,
                (IMAGE_HEIGHT_CAM0, IMAGE_WIDTH_CAM0),
            )
            self.assertEqual(
                window.image_items["cam1"].image.shape,
                (IMAGE_HEIGHT_CAM1, IMAGE_WIDTH_CAM1),
            )
            self.assertIsInstance(window.calibration_panel, CalibrationPanel)
            self.assertFalse(
                window.calibration_panel.save_calibration_button.isEnabled()
            )
            self.assertFalse(
                window.calibration_panel.calibration_details_button.isEnabled()
            )
            self.assertTrue(window.calibration_panel.new_calibration_button.isEnabled())
        finally:
            window.close()
            app.processEvents()


class CalibrationPanelSmokeTests(unittest.TestCase):
    def test_reset_enables_new_calibration_without_loaded_calibration(self):
        app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        panel = CalibrationPanel()
        try:
            panel.reset()

            self.assertTrue(panel.load_calibration_button.isEnabled())
            self.assertFalse(panel.save_calibration_button.isEnabled())
            self.assertFalse(panel.calibration_details_button.isEnabled())
            self.assertTrue(panel.new_calibration_button.isEnabled())
        finally:
            panel.close()
            app.processEvents()

    def test_show_calibration_in_progress_disables_calibration_controls(self):
        app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        panel = CalibrationPanel()
        try:
            panel.show_calibration_in_progress()

            self.assertFalse(panel.load_calibration_button.isEnabled())
            self.assertFalse(panel.save_calibration_button.isEnabled())
            self.assertFalse(panel.calibration_details_button.isEnabled())
            self.assertFalse(panel.new_calibration_button.isEnabled())
            self.assertIn("in progress", panel.calibration_status_label.text())
        finally:
            panel.close()
            app.processEvents()

    def test_show_loaded_calibration_updates_display_state(self):
        app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        panel = CalibrationPanel()
        try:
            stage_to_pixel = np.array(
                [
                    [[0.5, -0.1, 0.2], [0.2, 0.4, -0.1]],
                    [[-0.3, 0.2, 0.4], [0.1, -0.2, 0.3]],
                ]
            )
            stage = np.array(
                [
                    [0.0, 0.0, 0.0],
                    [20.0, 0.0, 0.0],
                    [0.0, 20.0, 0.0],
                    [0.0, 0.0, 20.0],
                ]
            )
            calibration = _synthetic_calibration(stage, stage_to_pixel)

            panel.show_loaded_calibration(calibration, "calibration.h5")

            self.assertTrue(panel.save_calibration_button.isEnabled())
            self.assertTrue(panel.calibration_details_button.isEnabled())
            self.assertTrue(panel.new_calibration_button.isEnabled())
            self.assertIn("calibration.h5", panel.calibration_status_label.text())
            self.assertEqual(panel.metric_labels["sample_count"].text(), "4")
            self.assertEqual(
                panel.calibration_warnings_text.toPlainText(),
                "No calibration warnings.",
            )
            self.assertEqual(set(panel.residual_plots), {"xy", "xz", "yz"})
            self.assertIsInstance(
                panel.residual_graphics_layout,
                pg.GraphicsLayoutWidget,
            )
            for residual_plot in panel.residual_plots.values():
                self.assertIsInstance(residual_plot, pg.PlotItem)
        finally:
            panel.close()
            app.processEvents()

    def test_calibration_details_dialog_includes_tabs_samples_and_images(self):
        app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        panel = CalibrationPanel()
        try:
            stage_to_pixel = np.array(
                [
                    [[0.5, -0.1, 0.2], [0.2, 0.4, -0.1]],
                    [[-0.3, 0.2, 0.4], [0.1, -0.2, 0.3]],
                ]
            )
            stage = np.array(
                [
                    [0.0, 0.0, 0.0],
                    [20.0, 0.0, 0.0],
                    [0.0, 20.0, 0.0],
                    [0.0, 0.0, 20.0],
                ]
            )
            images_cam0 = np.arange(stage.shape[0] * 4 * 5, dtype=float).reshape(
                stage.shape[0],
                4,
                5,
            )
            images_cam1 = np.arange(stage.shape[0] * 6 * 7, dtype=float).reshape(
                stage.shape[0],
                6,
                7,
            )
            calibration = _synthetic_calibration(
                stage,
                stage_to_pixel,
                images_cam0,
                images_cam1,
            )

            dialog = panel.build_details_dialog(calibration)
            try:
                tabs = dialog.findChild(
                    QtWidgets.QTabWidget,
                    "calibration_details_tabs",
                )
                self.assertIsNotNone(tabs)
                self.assertEqual(tabs.count(), 3)
                self.assertEqual(
                    [tabs.tabText(index) for index in range(tabs.count())],
                    ["Matrices", "Samples", "Images"],
                )

                sample_table = dialog.findChild(
                    QtWidgets.QTableWidget,
                    "calibration_samples_table",
                )
                self.assertIsNotNone(sample_table)
                self.assertEqual(sample_table.rowCount(), stage.shape[0])
                self.assertEqual(sample_table.columnCount(), 20)

                camera_selector = dialog.findChild(
                    QtWidgets.QComboBox,
                    "calibration_image_camera_selector",
                )
                self.assertIsNotNone(camera_selector)
                self.assertEqual(camera_selector.count(), len(CAMERAS))

                selector = dialog.findChild(
                    QtWidgets.QSpinBox,
                    "calibration_image_sample_selector",
                )
                self.assertIsNotNone(selector)
                self.assertEqual(selector.maximum(), stage.shape[0] - 1)
            finally:
                dialog.close()
        finally:
            panel.close()
            app.processEvents()


if __name__ == "__main__":
    unittest.main()

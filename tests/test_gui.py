import math
import os
import unittest

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from merlin_track_position.interface.main_window import (
    _MainWindowGUI,
    _calibration_summary,
    _clamp_roi_geometry,
    _default_roi_geometry,
    _validate_calibration_dataset,
)
from merlin_track_position.interface.calibration_panel import CalibrationPanel
from merlin_track_position.tracking.calibration import fit_calibration_from_measurements
from qtpy import QtWidgets


class GUIHelperTests(unittest.TestCase):
    def test_default_roi_geometry_is_centered_quarter_image(self):
        self.assertEqual(_default_roi_geometry(), (264.0, 180.0, 176.0, 120.0))

    def test_clamp_roi_geometry_keeps_roi_inside_image(self):
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
        stage_to_pixel = np.array([[0.5, -0.1], [0.2, 0.4]])
        stage = np.array(
            [
                [0.0, 0.0],
                [20.0, 0.0],
                [0.0, 20.0],
                [20.0, 0.0],
                [0.0, 20.0],
                [20.0, 20.0],
            ]
        )
        pixels = stage @ stage_to_pixel.T
        calibration = fit_calibration_from_measurements(stage, pixels)

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
        stage_to_pixel = np.array([[0.5, 0.0], [0.0, 0.4]])
        stage = np.array([[0.0, 0.0], [20.0, 0.0], [0.0, 20.0]])
        calibration = fit_calibration_from_measurements(stage, stage @ stage_to_pixel.T)
        broken = calibration.drop_vars("residual_stage_um")

        with self.assertRaisesRegex(ValueError, "residual_stage_um"):
            _validate_calibration_dataset(broken)


class MainWindowGUISmokeTests(unittest.TestCase):
    def test_main_window_gui_constructs_offscreen(self):
        app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        window = _MainWindowGUI()
        try:
            self.assertEqual(window.image_item.image.shape, (480, 704))
            self.assertIsInstance(window.calibration_panel, CalibrationPanel)
            self.assertFalse(
                window.calibration_panel.save_calibration_button.isEnabled()
            )
            self.assertFalse(
                window.calibration_panel.calibration_details_button.isEnabled()
            )
            self.assertTrue(
                window.calibration_panel.new_calibration_button.isEnabled()
            )
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
            stage_to_pixel = np.array([[0.5, -0.1], [0.2, 0.4]])
            stage = np.array(
                [
                    [0.0, 0.0],
                    [20.0, 0.0],
                    [0.0, 20.0],
                    [20.0, 0.0],
                    [0.0, 20.0],
                    [20.0, 20.0],
                ]
            )
            calibration = fit_calibration_from_measurements(
                stage, stage @ stage_to_pixel.T
            )

            panel.show_loaded_calibration(calibration, "calibration.h5")

            self.assertTrue(panel.save_calibration_button.isEnabled())
            self.assertTrue(panel.calibration_details_button.isEnabled())
            self.assertTrue(panel.new_calibration_button.isEnabled())
            self.assertIn("calibration.h5", panel.calibration_status_label.text())
            self.assertEqual(panel.metric_labels["sample_count"].text(), "6")
            self.assertEqual(
                panel.calibration_warnings_text.toPlainText(),
                "No calibration warnings.",
            )
        finally:
            panel.close()
            app.processEvents()

    def test_calibration_details_dialog_includes_tabs_samples_and_images(self):
        app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        panel = CalibrationPanel()
        try:
            stage_to_pixel = np.array([[0.5, -0.1], [0.2, 0.4]])
            stage = np.array(
                [
                    [0.0, 0.0],
                    [20.0, 0.0],
                    [0.0, 20.0],
                    [20.0, 20.0],
                ]
            )
            calibration = fit_calibration_from_measurements(
                stage, stage @ stage_to_pixel.T
            )
            images = np.arange(stage.shape[0] * 4 * 5, dtype=float).reshape(
                stage.shape[0], 4, 5
            )
            calibration = calibration.assign_coords(
                y=np.arange(images.shape[1]), x=np.arange(images.shape[2])
            ).assign(image=(("sample", "y", "x"), images))

            dialog = panel.build_details_dialog(calibration)
            try:
                tabs = dialog.findChild(QtWidgets.QTabWidget, "calibration_details_tabs")
                self.assertIsNotNone(tabs)
                self.assertEqual(tabs.count(), 3)
                self.assertEqual(
                    [tabs.tabText(index) for index in range(tabs.count())],
                    ["Matrices", "Samples", "Images"],
                )

                sample_table = dialog.findChild(
                    QtWidgets.QTableWidget, "calibration_samples_table"
                )
                self.assertIsNotNone(sample_table)
                self.assertEqual(sample_table.rowCount(), stage.shape[0])
                self.assertEqual(sample_table.columnCount(), 12)

                selector = dialog.findChild(
                    QtWidgets.QSpinBox, "calibration_image_sample_selector"
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

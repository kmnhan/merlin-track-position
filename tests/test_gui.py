import os
import unittest

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from merlin_track_position.interface.calibration_panel import (  # noqa: E402
    CalibrationPanel,
    _calibration_summary,
)
from merlin_track_position.interface.main_window import (  # noqa: E402
    CalibrationStartDialog,
    _clamp_roi_geometry,
    _default_roi_geometry,
    _roi_geometries_from_calibration_metadata,
    _roi_metadata_from_geometries,
)
from merlin_track_position.tracking.calibration_core import (  # noqa: E402
    COMMAND_AXES,
    derive_axis_scale_from_jacobian,
)
from merlin_track_position.tracking.sample_calibration import (  # noqa: E402
    build_sample_calibration_dataset,
)
from qtpy import QtWidgets  # noqa: E402

_APP = None


def get_qapp():
    global _APP
    _APP = QtWidgets.QApplication.instance() or _APP or QtWidgets.QApplication([])
    return _APP


class GUIHelperTests(unittest.TestCase):
    def test_clamp_roi_geometry_keeps_roi_inside_cam0_image(self):
        self.assertEqual(
            _clamp_roi_geometry((-10.0, 500.0, 900.0, -4.0)),
            (0.0, 479.0, 704.0, 1.0),
        )
        self.assertEqual(
            _clamp_roi_geometry((650.0, 470.0, 100.0, 40.0)),
            (604.0, 440.0, 100.0, 40.0),
        )

    def test_default_roi_geometry_is_centered_quarter_image(self):
        self.assertEqual(_default_roi_geometry(100.0, 80.0), (37.5, 30.0, 25.0, 20.0))

    def test_roi_metadata_round_trips_from_calibration(self):
        metadata = _roi_metadata_from_geometries(
            {
                "cam0": (1.0, 2.0, 30.0, 40.0),
                "cam1": (5.0, 6.0, 70.0, 80.0),
            }
        )
        calibration = build_sample_calibration_dataset(
            image_shape_cam0=(4, 5),
            image_shape_cam1=(6, 7),
        ).assign_attrs(metadata)

        geometries = _roi_geometries_from_calibration_metadata(calibration)

        self.assertEqual(geometries["cam0"], (1.0, 2.0, 30.0, 40.0))
        self.assertEqual(geometries["cam1"], (5.0, 6.0, 70.0, 80.0))


class CalibrationPanelTests(unittest.TestCase):
    def test_summary_reports_visual_jacobian_metrics(self):
        calibration = build_sample_calibration_dataset(
            image_shape_cam0=(4, 5),
            image_shape_cam1=(6, 7),
        )

        summary = _calibration_summary(calibration)

        self.assertEqual(summary["probe_count"], calibration.sizes["probe"])
        self.assertLess(summary["condition_number"], 100.0)
        np.testing.assert_allclose(
            summary["axis_scale_cmd_mm"],
            calibration["axis_scale_cmd_mm"].values,
        )
        (
            _derived_axis_scale,
            expected_axis_sensitivity,
            _axis_scale_unclamped,
            _axis_scale_bounds,
            _target_response,
        ) = derive_axis_scale_from_jacobian(
            calibration["visual_jacobian_px_per_cmd_mm"].values,
            calibration["probe_command_delta_mm"].values,
        )
        np.testing.assert_allclose(
            summary["axis_sensitivity_px_per_cmd_mm"],
            expected_axis_sensitivity,
        )
        self.assertEqual(summary["residual_rms_px"], 0.0)
        self.assertEqual(summary["residual_max_cmd_mm"], 0.0)
        np.testing.assert_allclose(
            summary["visual_jacobian"],
            calibration["visual_jacobian_px_per_cmd_mm"].values,
        )

    def test_panel_loads_dataset_and_builds_details_dialog(self):
        get_qapp()
        calibration = build_sample_calibration_dataset(
            image_shape_cam0=(4, 5),
            image_shape_cam1=(6, 7),
        )
        panel = CalibrationPanel()

        panel.show_loaded_calibration(calibration, "test.h5")
        dialog = panel.build_details_dialog(calibration)
        tabs = dialog.findChild(QtWidgets.QTabWidget, "calibration_details_tabs")
        table = dialog.findChild(QtWidgets.QTableWidget, "calibration_samples_table")
        axes_table = dialog.findChild(QtWidgets.QTableWidget, "calibration_axes_table")

        self.assertIn("test.h5", panel.calibration_status_label.text())
        self.assertIn("visual-Jacobian", panel.calibration_status_label.text())
        self.assertNotEqual(panel.metric_labels["axis_scale_cmd_mm"].text(), "n/a")
        self.assertEqual(tabs.tabText(0), "Matrices")
        self.assertEqual(tabs.tabText(1), "Axes")
        self.assertEqual(tabs.tabText(2), "Probes")
        self.assertEqual(tabs.count(), 3)
        self.assertEqual(axes_table.rowCount(), len(COMMAND_AXES))
        self.assertEqual(table.rowCount(), calibration.sizes["probe"])
        self.assertIn(
            "dx_cmd_mm",
            [table.horizontalHeaderItem(i).text() for i in range(table.columnCount())],
        )


class CalibrationStartDialogTests(unittest.TestCase):
    def test_dialog_exposes_save_path(self):
        get_qapp()
        dialog = CalibrationStartDialog()

        self.assertIn("Visual-Jacobian", dialog.windowTitle())
        self.assertIsNotNone(
            dialog.findChild(QtWidgets.QLineEdit, "calibration_output_path_edit")
        )
        self.assertTrue(str(dialog.output_path()).endswith(".h5"))


if __name__ == "__main__":
    unittest.main()

import os
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import numpy as np
import xarray as xr

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from merlin_track_position.interface.calibration_panel import (  # noqa: E402
    CalibrationPanel,
    _calibration_summary,
)
from merlin_track_position.interface.main_window import (  # noqa: E402
    CalibrationStartDialog,
    MainWindow,
    _clamp_roi_geometry,
    _default_roi_geometry,
    _roi_geometries_from_calibration_metadata,
    _roi_metadata_from_geometries,
)
from merlin_track_position.tracking.calibration_core import (  # noqa: E402
    COMMAND_AXES,
    derive_axis_scale_from_jacobian,
    save_calibration_dataset,
)
from merlin_track_position.tracking.sample_calibration import (  # noqa: E402
    build_sample_calibration_dataset,
)
from qtpy import QtCore, QtWidgets  # noqa: E402

_APP = None


def get_qapp():
    global _APP
    _APP = QtWidgets.QApplication.instance() or _APP or QtWidgets.QApplication([])
    return _APP


class FakeSettings:
    def __init__(self):
        self.values: dict[str, float] = {}
        self.set_calls: list[tuple[str, float]] = []

    def value(self, key, fallback=None):
        return self.values.get(str(key), fallback)

    def setValue(self, key, value):
        self.values[str(key)] = float(value)
        self.set_calls.append((str(key), float(value)))

    def sync(self):
        pass


class FakeImageCaptureThread(QtCore.QObject):
    sigImageReady = QtCore.Signal(str, object)
    sigImageCaptureFailed = QtCore.Signal(str, str)

    def __init__(self, camera, image_capture, interval_ms, parent=None):
        super().__init__(parent)
        self.camera = camera
        self.image_capture = image_capture
        self.interval_ms = interval_ms
        self.enabled = False

    def start(self):
        pass

    def set_enabled(self, enabled):
        self.enabled = bool(enabled)

    def stop(self):
        pass

    def wait(self):
        pass


class FakeMotorServer(QtCore.QObject):
    sigMoveDetected = QtCore.Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)

    def start(self):
        pass

    def stop(self):
        pass

    def wait(self):
        pass

    def set_result(self, success, message):
        pass


class FakeCorrectionThread(QtCore.QObject):
    sigCorrectionReady = QtCore.Signal(object)
    sigCorrectionFailed = QtCore.Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.calibration = None
        self.camera_pair = None
        self.calibration_path = None
        self.started = False
        self.running = False

    def configure(self, calibration, camera_pair, calibration_path):
        self.calibration = calibration
        self.camera_pair = camera_pair
        self.calibration_path = Path(calibration_path)

    def start(self):
        self.started = True
        self.running = True

    def isRunning(self):
        return self.running

    def stop(self):
        self.running = False

    def wait(self):
        pass


@contextmanager
def patched_main_window_runtime(settings=None):
    settings = settings or FakeSettings()
    with (
        patch(
            "merlin_track_position.interface.main_window.QtCore.QSettings",
            return_value=settings,
        ),
        patch(
            "merlin_track_position.interface.main_window._ImageCaptureThread",
            FakeImageCaptureThread,
        ),
        patch(
            "merlin_track_position.interface.main_window.MotorServer",
            FakeMotorServer,
        ),
        patch(
            "merlin_track_position.interface.main_window.CorrectionThread",
            FakeCorrectionThread,
        ),
        patch("merlin_track_position.interface.main_window.close_basler_camera"),
    ):
        yield settings


def roi_handles_visible(window):
    return all(
        handles and all(handle.isVisible() for handle in handles)
        for handles in (roi.getHandles() for roi in window.image_rois.values())
    )


def roi_handle_count(window):
    return sum(len(roi.getHandles()) for roi in window.image_rois.values())


def roi_child_item_count(window):
    return sum(len(roi.childItems()) for roi in window.image_rois.values())


def roi_editing_enabled(window):
    return all(roi.translatable for roi in window.image_rois.values())


def correction_result(
    *,
    converged: bool = True,
    moves: int = 2,
    residual: float = 0.25,
    warnings: str = "",
) -> xr.Dataset:
    return xr.Dataset(
        data_vars={
            "iteration_weighted_residual_px": (
                ("iteration",),
                np.asarray([1.0, residual], dtype=float),
            )
        },
        coords={"iteration": np.arange(2)},
        attrs={
            "correction_converged": converged,
            "correction_iterations": moves,
            "warnings": warnings,
        },
    )


def write_sample_calibration(path: Path) -> xr.Dataset:
    calibration = build_sample_calibration_dataset(
        image_shape_cam0=(4, 5),
        image_shape_cam1=(6, 7),
    )
    save_calibration_dataset(calibration, path)
    return calibration.assign_attrs({"calibration_path": str(path)})


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
        self.assertEqual(panel.new_calibration_button.text(), "Clear calibration")
        self.assertTrue(panel.correct_sample_button.isEnabled())
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

        panel.reset()
        self.assertEqual(panel.new_calibration_button.text(), "New calibration")
        self.assertFalse(panel.correct_sample_button.isEnabled())

        panel.show_loaded_calibration(calibration, "test.h5")
        panel.show_correction_in_progress()
        self.assertIn("Correction in progress", panel.calibration_status_label.text())
        self.assertFalse(panel.load_calibration_button.isEnabled())
        self.assertFalse(panel.save_calibration_button.isEnabled())
        self.assertFalse(panel.calibration_details_button.isEnabled())
        self.assertFalse(panel.correct_sample_button.isEnabled())
        self.assertFalse(panel.new_calibration_button.isEnabled())


class MainWindowCalibrationStateTests(unittest.TestCase):
    def test_fresh_window_has_editable_roi_and_new_calibration_button(self):
        get_qapp()
        with patched_main_window_runtime():
            window = MainWindow()
            try:
                self.assertEqual(
                    window.calibration_panel.new_calibration_button.text(),
                    "New calibration",
                )
                self.assertFalse(window.calibration_panel.correct_sample_button.isEnabled())
                self.assertTrue(roi_handles_visible(window))
                self.assertTrue(roi_editing_enabled(window))
            finally:
                window.close()

    def test_loaded_calibration_locks_roi_and_button_clears_without_dialog(self):
        get_qapp()
        calibration = build_sample_calibration_dataset(
            image_shape_cam0=(4, 5),
            image_shape_cam1=(6, 7),
        ).assign_attrs({"calibration_path": "/tmp/calibration.h5"})

        with patched_main_window_runtime():
            window = MainWindow()
            try:
                window._on_new_calibration_ready(calibration)

                self.assertEqual(
                    window.calibration_panel.new_calibration_button.text(),
                    "Clear calibration",
                )
                self.assertTrue(window.calibration_panel.correct_sample_button.isEnabled())
                self.assertFalse(roi_handles_visible(window))
                self.assertFalse(roi_editing_enabled(window))

                with patch.object(
                    CalibrationStartDialog,
                    "exec",
                    side_effect=AssertionError("dialog should not open"),
                ):
                    window._on_new_calibration_clicked()

                self.assertIsNone(window._calibration)
                self.assertIsNone(window._calibration_path)
                self.assertEqual(
                    window.calibration_panel.new_calibration_button.text(),
                    "New calibration",
                )
                self.assertFalse(window.calibration_panel.correct_sample_button.isEnabled())
                self.assertTrue(roi_handles_visible(window))
                self.assertTrue(roi_editing_enabled(window))
            finally:
                window.close()

    def test_loaded_calibration_applies_roi_metadata_before_locking(self):
        get_qapp()
        metadata = _roi_metadata_from_geometries(
            {
                "cam0": (10.0, 12.0, 30.0, 32.0),
                "cam1": (14.0, 16.0, 34.0, 36.0),
            }
        )
        calibration = build_sample_calibration_dataset(
            image_shape_cam0=(4, 5),
            image_shape_cam1=(6, 7),
        ).assign_attrs({"calibration_path": "/tmp/calibration.h5"} | metadata)

        with patched_main_window_runtime() as settings:
            window = MainWindow()
            try:
                window._on_new_calibration_ready(calibration)

                self.assertEqual(
                    window._get_roi_geometry("cam0"),
                    (10.0, 12.0, 30.0, 32.0),
                )
                self.assertFalse(roi_handles_visible(window))
                self.assertEqual(settings.set_calls, [])
            finally:
                window.close()

    def test_roi_changes_are_ignored_while_calibration_is_loaded(self):
        get_qapp()
        calibration = build_sample_calibration_dataset(
            image_shape_cam0=(4, 5),
            image_shape_cam1=(6, 7),
        ).assign_attrs({"calibration_path": "/tmp/calibration.h5"})

        with patched_main_window_runtime() as settings:
            window = MainWindow()
            try:
                window._on_new_calibration_ready(calibration)
                window._set_roi_geometry("cam0", (20.0, 21.0, 40.0, 41.0))
                window._on_roi_region_change_finished("cam0")

                self.assertEqual(settings.set_calls, [])
            finally:
                window.close()

    def test_locked_roi_handles_stay_hidden_if_roi_is_selected_again(self):
        get_qapp()
        with patched_main_window_runtime():
            window = MainWindow()
            try:
                self.assertEqual(roi_handle_count(window), 16)
                self.assertEqual(roi_child_item_count(window), 16)

                window._set_roi_editing_enabled(False)
                for roi in window.image_rois.values():
                    roi.setSelected(True)

                self.assertEqual(roi_handle_count(window), 0)
                self.assertEqual(roi_child_item_count(window), 0)
                self.assertFalse(roi_handles_visible(window))
                self.assertFalse(roi_editing_enabled(window))

                window._set_roi_editing_enabled(True)

                self.assertEqual(roi_handle_count(window), 16)
                self.assertEqual(roi_child_item_count(window), 16)
                self.assertTrue(roi_handles_visible(window))
                self.assertTrue(roi_editing_enabled(window))
            finally:
                window.close()

    def test_correction_cancel_does_not_start_thread(self):
        get_qapp()
        with tempfile.TemporaryDirectory() as tmpdir:
            calibration = write_sample_calibration(Path(tmpdir) / "calibration.h5")
            with patched_main_window_runtime():
                window = MainWindow()
                try:
                    window._on_new_calibration_ready(calibration)
                    with patch(
                        "merlin_track_position.interface.main_window.QtWidgets.QMessageBox.warning",
                        return_value=QtWidgets.QMessageBox.StandardButton.Cancel,
                    ):
                        window._on_correct_sample_clicked()

                    self.assertFalse(window._correction_thread.started)
                finally:
                    window.close()

    def test_correction_confirmation_starts_thread_and_disables_actions(self):
        get_qapp()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "calibration.h5"
            calibration = write_sample_calibration(path)
            with patched_main_window_runtime():
                window = MainWindow()
                try:
                    window._on_new_calibration_ready(calibration)
                    with patch(
                        "merlin_track_position.interface.main_window.QtWidgets.QMessageBox.warning",
                        return_value=QtWidgets.QMessageBox.StandardButton.Ok,
                    ):
                        window._on_correct_sample_clicked()

                    thread = window._correction_thread
                    self.assertTrue(thread.started)
                    self.assertIs(thread.calibration, window._calibration)
                    self.assertEqual(thread.calibration_path, path)
                    self.assertIsNotNone(thread.camera_pair)
                    self.assertIn(
                        "Correction in progress",
                        window.calibration_panel.calibration_status_label.text(),
                    )
                    self.assertFalse(
                        window.calibration_panel.correct_sample_button.isEnabled()
                    )
                    self.assertFalse(
                        window.calibration_panel.new_calibration_button.isEnabled()
                    )
                finally:
                    window.close()

    def test_correction_success_stores_result_reloads_calibration_and_reports_status(self):
        get_qapp()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "calibration.h5"
            calibration = write_sample_calibration(path)
            result = correction_result(converged=True, moves=2, residual=0.125)
            with patched_main_window_runtime():
                window = MainWindow()
                try:
                    window._on_new_calibration_ready(calibration)
                    window.calibration_panel.show_correction_in_progress()

                    window._on_correction_ready(result)

                    self.assertIs(window._last_correction_result, result)
                    self.assertEqual(window._calibration_path, path)
                    self.assertTrue(
                        window.calibration_panel.correct_sample_button.isEnabled()
                    )
                    self.assertIn(
                        "Correction converged after 2 move(s)",
                        window.calibration_panel.calibration_status_label.text(),
                    )
                    self.assertEqual(
                        window.calibration_panel.calibration_warnings_text.toPlainText(),
                        "No correction warnings.",
                    )
                finally:
                    window.close()

    def test_correction_failure_restores_loaded_state_and_shows_error(self):
        get_qapp()
        with tempfile.TemporaryDirectory() as tmpdir:
            calibration = write_sample_calibration(Path(tmpdir) / "calibration.h5")
            with patched_main_window_runtime():
                window = MainWindow()
                try:
                    window._on_new_calibration_ready(calibration)
                    window.calibration_panel.show_correction_in_progress()
                    with patch(
                        "merlin_track_position.interface.main_window.QtWidgets.QMessageBox.critical"
                    ) as critical:
                        window._on_correction_failed("boom")

                    self.assertTrue(
                        window.calibration_panel.correct_sample_button.isEnabled()
                    )
                    self.assertEqual(
                        window.calibration_panel.new_calibration_button.text(),
                        "Clear calibration",
                    )
                    critical.assert_called_once()
                finally:
                    window.close()


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

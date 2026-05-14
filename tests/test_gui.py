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
import merlin_track_position.interface.main_window as main_window  # noqa: E402
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
from merlin_track_position.instruments import parse_config  # noqa: E402
from merlin_track_position.tracking.correct import (  # noqa: E402
    correction_history_path,
    save_correction_history_dataset,
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
        self.result_calls = []
        self.motor_backend = None

    def start(self):
        pass

    def stop(self):
        pass

    def wait(self):
        pass

    def set_result(self, success, message):
        self.result_calls.append((bool(success), str(message)))

    def current_motor_backend(self):
        return self.motor_backend


class FakeCorrectionThread(QtCore.QObject):
    sigCorrectionProgress = QtCore.Signal(object)
    sigCorrectionReady = QtCore.Signal(object)
    sigCorrectionFailed = QtCore.Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.calibration = None
        self.camera_pair = None
        self.calibration_path = None
        self.motor_backend = None
        self.started = False
        self.running = False

    def configure(
        self,
        calibration,
        camera_pair,
        calibration_path,
        motor_backend=None,
    ):
        self.calibration = calibration
        self.camera_pair = camera_pair
        self.calibration_path = Path(calibration_path)
        self.motor_backend = motor_backend

    def start(self):
        self.started = True
        self.running = True

    def isRunning(self):
        return self.running

    def stop(self):
        self.running = False

    def wait(self):
        pass


class FakeDetectShiftThread(QtCore.QObject):
    sigDetectionReady = QtCore.Signal(object)
    sigDetectionFailed = QtCore.Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.calibration = None
        self.camera_pair = None
        self.started = False
        self.running = False

    def configure(
        self,
        calibration,
        camera_pair,
    ):
        self.calibration = calibration
        self.camera_pair = camera_pair

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
        patch(
            "merlin_track_position.interface.main_window.DetectShiftThread",
            FakeDetectShiftThread,
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


def full_camera_image(camera, value):
    image_width, image_height = main_window.CAMERA_IMAGE_SIZES[camera]
    return np.full((image_height, image_width), value, dtype=np.float32)


def image_parent_rect(window, camera):
    image_item = window.image_items[camera]
    return image_item.mapRectToParent(image_item.boundingRect())


def assert_rect_close(testcase, rect, expected):
    actual = (rect.x(), rect.y(), rect.width(), rect.height())
    for actual_value, expected_value in zip(actual, expected, strict=True):
        testcase.assertAlmostEqual(actual_value, expected_value)


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


def correction_result_with_moves() -> xr.Dataset:
    return xr.Dataset(
        data_vars={
            "iteration_weighted_residual_px": (
                ("iteration",),
                np.asarray([1.0, 0.7, 0.25], dtype=float),
            ),
            "move_command_delta_mm": (
                ("move", "command_axis"),
                np.asarray(
                    [
                        [0.0015, -0.002, 0.0],
                        [-0.0005, 0.0, 0.00325],
                    ],
                    dtype=float,
                ),
            ),
            "move_pre_weighted_residual_px": (
                ("move",),
                np.asarray([1.0, 0.7], dtype=float),
            ),
            "move_post_weighted_residual_px": (
                ("move",),
                np.asarray([0.7, 0.25], dtype=float),
            ),
            "estimated_command_offset_mm": (
                ("command_axis",),
                np.asarray([0.004, -0.005, 0.006], dtype=float),
            ),
            "correction_cmd_mm": (
                ("command_axis",),
                np.asarray([0.00025, 0.0, -0.00075], dtype=float),
            ),
        },
        coords={
            "iteration": np.arange(3),
            "move": np.arange(2),
            "command_axis": list(COMMAND_AXES),
        },
        attrs={
            "correction_converged": True,
            "correction_iterations": 2,
            "warnings": "",
        },
    )


def correction_result_before_first_move() -> xr.Dataset:
    return xr.Dataset(
        data_vars={
            "iteration_weighted_residual_px": (
                ("iteration",),
                np.asarray([1.0], dtype=float),
            ),
            "estimated_command_offset_mm": (
                ("command_axis",),
                np.asarray([0.004, -0.005, 0.006], dtype=float),
            ),
            "correction_cmd_mm": (
                ("command_axis",),
                np.asarray([0.0015, -0.002, 0.0], dtype=float),
            ),
        },
        coords={
            "iteration": np.arange(1),
            "command_axis": list(COMMAND_AXES),
        },
        attrs={
            "correction_converged": False,
            "correction_iterations": 0,
            "warnings": "",
        },
    )


def detection_result(
    *,
    offsets_mm=(0.004, -0.005, 0.006),
    residual: float = 1.25,
    warnings: str = "",
) -> xr.Dataset:
    offsets = np.asarray(offsets_mm, dtype=float)
    return xr.Dataset(
        data_vars={
            "estimated_command_offset_mm": (
                ("command_axis",),
                offsets,
            ),
            "detected_shift_um": (
                ("command_axis",),
                1000.0 * offsets,
            ),
            "weighted_residual_px": ((), float(residual)),
        },
        coords={"command_axis": list(COMMAND_AXES)},
        attrs={"warnings": warnings},
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
        self.assertTrue(panel.correct_sample_button.isEnabled())
        self.assertTrue(panel.detect_shift_button.isEnabled())
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
        self.assertFalse(panel.correct_sample_button.isEnabled())
        self.assertFalse(panel.detect_shift_button.isEnabled())

        panel.show_loaded_calibration(calibration, "test.h5")
        panel.show_correction_in_progress()
        self.assertIn("Correction in progress", panel.calibration_status_label.text())
        self.assertFalse(panel.load_calibration_button.isEnabled())
        self.assertFalse(panel.save_calibration_button.isEnabled())
        self.assertFalse(panel.calibration_details_button.isEnabled())
        self.assertFalse(panel.correct_sample_button.isEnabled())
        self.assertFalse(panel.detect_shift_button.isEnabled())
        self.assertFalse(panel.new_calibration_button.isEnabled())
        self.assertFalse(panel.correction_steps_group.isHidden())
        self.assertEqual(panel.correction_steps_table.rowCount(), 0)
        self.assertIn(
            "initial correction measurement",
            panel.correction_steps_summary_label.text(),
        )

    def test_repeatability_section_is_below_metrics(self):
        get_qapp()
        panel = CalibrationPanel()

        content_layout = panel.layout().itemAt(3).layout()
        left_column = content_layout.itemAt(0).layout()
        right_column = content_layout.itemAt(1).layout()

        self.assertIs(left_column.itemAt(0).widget(), panel.metrics_group)
        self.assertIs(left_column.itemAt(1).widget(), panel.repeatability_group)
        self.assertIs(right_column.itemAt(0).widget(), panel.warnings_group)
        self.assertIs(right_column.itemAt(1).widget(), panel.correction_steps_group)

    def test_correction_progress_before_first_move_shows_plan(self):
        get_qapp()
        panel = CalibrationPanel()
        result = correction_result_before_first_move()

        panel.show_correction_in_progress()
        panel.show_correction_progress(result)

        table = panel.correction_steps_table
        self.assertFalse(panel.correction_steps_group.isHidden())
        self.assertEqual(table.rowCount(), 0)
        self.assertIn(
            "before first move",
            panel.calibration_status_label.text(),
        )
        summary = panel.correction_steps_summary_label.text()
        self.assertIn("No correction moves have been applied yet.", summary)
        self.assertIn(
            "Estimated command offset: x=4 um, y=-5 um, z=6 um.",
            summary,
        )
        self.assertIn("Next correction: x=1.5 um, y=-2 um, z=0 um.", summary)

    def test_correction_result_displays_move_steps_in_microns(self):
        get_qapp()
        panel = CalibrationPanel()
        result = correction_result_with_moves()

        panel.show_correction_result(result)

        table = panel.correction_steps_table
        self.assertFalse(panel.correction_steps_group.isHidden())
        self.assertEqual(table.rowCount(), 2)
        self.assertEqual(table.item(0, 1).text(), "1.5")
        self.assertEqual(table.item(0, 2).text(), "-2")
        self.assertEqual(table.item(0, 3).text(), "0")
        self.assertEqual(table.item(1, 1).text(), "-0.5")
        self.assertEqual(table.item(1, 3).text(), "3.25")
        self.assertIn(
            "Applied total: x=1 um, y=-2 um, z=3.25 um.",
            panel.correction_steps_summary_label.text(),
        )

    def test_detection_result_displays_signed_shift_in_microns(self):
        get_qapp()
        panel = CalibrationPanel()
        result = detection_result(warnings="registration warning")

        panel.show_detection_result(result)

        self.assertIn(
            "Detected shift: x=4 um, y=-5 um, z=6 um.",
            panel.calibration_status_label.text(),
        )
        self.assertIn(
            "Weighted residual 1.25 px.",
            panel.calibration_status_label.text(),
        )
        self.assertTrue(panel.correct_sample_button.isEnabled())
        self.assertTrue(panel.detect_shift_button.isEnabled())
        self.assertEqual(
            panel.calibration_warnings_text.toPlainText(),
            "registration warning",
        )


class MainWindowCalibrationStateTests(unittest.TestCase):
    def test_fresh_window_has_editable_roi_and_new_calibration_button(self):
        get_qapp()
        with patched_main_window_runtime():
            window = MainWindow()
            try:
                self.assertFalse(
                    window.calibration_panel.correct_sample_button.isEnabled()
                )
                self.assertFalse(
                    window.calibration_panel.auto_correction_checkbox.isEnabled()
                )
                self.assertFalse(
                    window.calibration_panel.auto_correction_checkbox.isChecked()
                )
                self.assertFalse(
                    window.calibration_panel.auto_correction_interval_spinbox.isEnabled()
                )
                self.assertFalse(
                    window.calibration_panel.detect_shift_button.isEnabled()
                )
                self.assertFalse(window.show_reference_images_button.isEnabled())
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

                self.assertTrue(
                    window.calibration_panel.correct_sample_button.isEnabled()
                )
                self.assertTrue(
                    window.calibration_panel.auto_correction_checkbox.isEnabled()
                )
                self.assertTrue(
                    window.calibration_panel.auto_correction_interval_spinbox.isEnabled()
                )
                self.assertTrue(
                    window.calibration_panel.detect_shift_button.isEnabled()
                )
                self.assertTrue(window.show_reference_images_button.isEnabled())
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
                self.assertFalse(
                    window.calibration_panel.correct_sample_button.isEnabled()
                )
                self.assertFalse(
                    window.calibration_panel.auto_correction_checkbox.isEnabled()
                )
                self.assertFalse(
                    window.calibration_panel.auto_correction_checkbox.isChecked()
                )
                self.assertFalse(
                    window.calibration_panel.auto_correction_interval_spinbox.isEnabled()
                )
                self.assertFalse(
                    window.calibration_panel.detect_shift_button.isEnabled()
                )
                self.assertFalse(window.show_reference_images_button.isEnabled())
                self.assertTrue(roi_handles_visible(window))
                self.assertTrue(roi_editing_enabled(window))
            finally:
                window.close()

    def test_reference_button_shows_calibration_images_until_released(self):
        get_qapp()
        metadata = _roi_metadata_from_geometries(
            {
                "cam0": (10.2, 12.4, 5.0, 4.0),
                "cam1": (14.7, 16.1, 7.0, 6.0),
            }
        )
        calibration = build_sample_calibration_dataset(
            image_shape_cam0=(5, 6),
            image_shape_cam1=(7, 8),
        ).assign_attrs({"calibration_path": "/tmp/calibration.h5"} | metadata)
        current_cam0 = full_camera_image("cam0", 3.0)
        current_cam1 = full_camera_image("cam1", 4.0)
        refreshed_cam0 = full_camera_image("cam0", 5.0)

        with patched_main_window_runtime():
            window = MainWindow()
            try:
                window._on_image_capture_ready("cam0", current_cam0)
                window._on_image_capture_ready("cam1", current_cam1)
                window._on_new_calibration_ready(calibration)

                window.show_reference_images_button.pressed.emit()

                np.testing.assert_array_equal(
                    window.image_items["cam0"].image,
                    calibration["reference_cam0"].values,
                )
                np.testing.assert_array_equal(
                    window.image_items["cam1"].image,
                    calibration["reference_cam1"].values,
                )
                assert_rect_close(
                    self,
                    image_parent_rect(window, "cam0"),
                    (10.0, 12.0, 6.0, 5.0),
                )
                assert_rect_close(
                    self,
                    image_parent_rect(window, "cam1"),
                    (14.0, 16.0, 8.0, 7.0),
                )

                window._on_image_capture_ready("cam0", refreshed_cam0)
                np.testing.assert_array_equal(
                    window.image_items["cam0"].image,
                    calibration["reference_cam0"].values,
                )

                window.show_reference_images_button.released.emit()

                np.testing.assert_array_equal(
                    window.image_items["cam0"].image,
                    refreshed_cam0,
                )
                np.testing.assert_array_equal(
                    window.image_items["cam1"].image,
                    current_cam1,
                )
                image_width, image_height = main_window.CAMERA_IMAGE_SIZES["cam0"]
                assert_rect_close(
                    self,
                    image_parent_rect(window, "cam0"),
                    (0.0, 0.0, float(image_width), float(image_height)),
                )
            finally:
                window.close()

    def test_auto_correction_toggle_starts_and_stops_timer(self):
        get_qapp()
        with tempfile.TemporaryDirectory() as tmpdir:
            calibration = write_sample_calibration(Path(tmpdir) / "calibration.h5")
            with patched_main_window_runtime():
                window = MainWindow()
                try:
                    window._on_new_calibration_ready(calibration)
                    window.calibration_panel.auto_correction_interval_spinbox.setValue(
                        2.125
                    )

                    window.calibration_panel.auto_correction_checkbox.setChecked(True)

                    self.assertTrue(window._auto_correction_timer.isActive())
                    self.assertEqual(window._auto_correction_timer.interval(), 2125)

                    window.calibration_panel.auto_correction_checkbox.setChecked(False)

                    self.assertFalse(window._auto_correction_timer.isActive())
                finally:
                    window.close()

    def test_auto_correction_interval_persists_and_updates_active_timer(self):
        get_qapp()
        settings = FakeSettings()
        settings.values["auto_correction/interval_seconds"] = 3.125
        with tempfile.TemporaryDirectory() as tmpdir:
            calibration = write_sample_calibration(Path(tmpdir) / "calibration.h5")
            with patched_main_window_runtime(settings):
                window = MainWindow()
                try:
                    self.assertAlmostEqual(
                        window.calibration_panel.auto_correction_interval_spinbox.value(),
                        3.125,
                    )

                    window._on_new_calibration_ready(calibration)
                    window.calibration_panel.auto_correction_checkbox.setChecked(True)
                    window.calibration_panel.auto_correction_interval_spinbox.setValue(
                        4.25
                    )

                    self.assertAlmostEqual(
                        settings.values["auto_correction/interval_seconds"],
                        4.25,
                    )
                    self.assertEqual(window._auto_correction_timer.interval(), 4250)
                finally:
                    window.close()

    def test_auto_correction_interval_reads_legacy_ms_setting(self):
        get_qapp()
        settings = FakeSettings()
        settings.values["auto_correction/interval_ms"] = 2500.0
        with patched_main_window_runtime(settings):
            window = MainWindow()
            try:
                self.assertAlmostEqual(
                    window.calibration_panel.auto_correction_interval_spinbox.value(),
                    2.5,
                )
            finally:
                window.close()

    def test_auto_correction_interval_reads_legacy_minutes_setting(self):
        get_qapp()
        settings = FakeSettings()
        settings.values["auto_correction/interval_minutes"] = 3.0
        with patched_main_window_runtime(settings):
            window = MainWindow()
            try:
                self.assertAlmostEqual(
                    window.calibration_panel.auto_correction_interval_spinbox.value(),
                    180.0,
                )
            finally:
                window.close()

    def test_auto_correction_timeout_starts_without_confirmation(self):
        get_qapp()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "calibration.h5"
            calibration = write_sample_calibration(path)
            with patched_main_window_runtime():
                window = MainWindow()
                try:
                    window._on_new_calibration_ready(calibration)
                    window.calibration_panel.auto_correction_checkbox.setChecked(True)

                    with patch(
                        "merlin_track_position.interface.main_window.QtWidgets.QMessageBox.warning",
                        side_effect=AssertionError("confirmation should be bypassed"),
                    ):
                        window._on_auto_correction_timeout()

                    thread = window._correction_thread
                    self.assertTrue(thread.started)
                    self.assertIs(thread.calibration, window._calibration)
                    self.assertEqual(thread.calibration_path, path)
                    self.assertTrue(
                        window.calibration_panel.auto_correction_checkbox.isEnabled()
                    )
                    self.assertTrue(
                        window.calibration_panel.auto_correction_checkbox.isChecked()
                    )
                finally:
                    window.close()

    def test_auto_correction_timeout_skips_when_busy(self):
        get_qapp()
        with tempfile.TemporaryDirectory() as tmpdir:
            calibration = write_sample_calibration(Path(tmpdir) / "calibration.h5")
            with patched_main_window_runtime():
                window = MainWindow()
                try:
                    window._on_new_calibration_ready(calibration)
                    window.calibration_panel.auto_correction_checkbox.setChecked(True)
                    window._correction_thread.running = True

                    window._on_auto_correction_timeout()

                    self.assertFalse(window._correction_thread.started)
                    self.assertTrue(
                        window.calibration_panel.auto_correction_checkbox.isChecked()
                    )
                    self.assertTrue(window._auto_correction_timer.isActive())
                finally:
                    window.close()

    def test_auto_correction_failure_leaves_toggle_enabled(self):
        get_qapp()
        with tempfile.TemporaryDirectory() as tmpdir:
            calibration = write_sample_calibration(Path(tmpdir) / "calibration.h5")
            with patched_main_window_runtime():
                window = MainWindow()
                try:
                    window._on_new_calibration_ready(calibration)
                    window.calibration_panel.auto_correction_checkbox.setChecked(True)
                    window._correction_thread.running = False
                    with patch(
                        "merlin_track_position.interface.main_window.QtWidgets.QMessageBox.critical"
                    ):
                        window._on_correction_failed("boom")

                    self.assertTrue(
                        window.calibration_panel.auto_correction_checkbox.isEnabled()
                    )
                    self.assertTrue(
                        window.calibration_panel.auto_correction_checkbox.isChecked()
                    )
                    self.assertTrue(window._auto_correction_timer.isActive())
                finally:
                    window.close()

    def test_clearing_calibration_stops_auto_correction(self):
        get_qapp()
        with tempfile.TemporaryDirectory() as tmpdir:
            calibration = write_sample_calibration(Path(tmpdir) / "calibration.h5")
            with patched_main_window_runtime():
                window = MainWindow()
                try:
                    window._on_new_calibration_ready(calibration)
                    window.calibration_panel.auto_correction_checkbox.setChecked(True)

                    window._on_new_calibration_clicked()

                    self.assertFalse(window._auto_correction_timer.isActive())
                    self.assertFalse(
                        window.calibration_panel.auto_correction_checkbox.isChecked()
                    )
                    self.assertFalse(
                        window.calibration_panel.auto_correction_checkbox.isEnabled()
                    )
                    self.assertFalse(
                        window.calibration_panel.auto_correction_interval_spinbox.isEnabled()
                    )
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

    def test_detect_shift_starts_thread_and_disables_actions(self):
        get_qapp()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "calibration.h5"
            calibration = write_sample_calibration(path)
            with patched_main_window_runtime():
                window = MainWindow()
                try:
                    window._on_new_calibration_ready(calibration)

                    window._on_detect_shift_clicked()

                    thread = window._detect_shift_thread
                    self.assertTrue(thread.started)
                    self.assertIs(thread.calibration, window._calibration)
                    self.assertIsNotNone(thread.camera_pair)
                    self.assertIsNone(window._last_correction_result)
                    self.assertFalse(window._server_correction_pending)
                    self.assertIn(
                        "Detecting shift",
                        window.calibration_panel.calibration_status_label.text(),
                    )
                    self.assertFalse(
                        window.calibration_panel.correct_sample_button.isEnabled()
                    )
                    self.assertFalse(
                        window.calibration_panel.detect_shift_button.isEnabled()
                    )
                    self.assertFalse(
                        window.calibration_panel.new_calibration_button.isEnabled()
                    )
                finally:
                    window.close()

    def test_detect_shift_result_preserves_correction_and_calibration_state(self):
        get_qapp()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "calibration.h5"
            calibration = write_sample_calibration(path)
            previous_correction = correction_result(converged=False)
            result = detection_result()
            with patched_main_window_runtime():
                window = MainWindow()
                try:
                    window._on_new_calibration_ready(calibration)
                    original_calibration = window._calibration
                    window._last_correction_result = previous_correction
                    window.calibration_panel.show_detection_in_progress()

                    window._on_detect_shift_ready(result)

                    self.assertIs(window._calibration, original_calibration)
                    self.assertIs(window._last_correction_result, previous_correction)
                    self.assertEqual(window._server.result_calls, [])
                    self.assertFalse(correction_history_path(path).exists())
                    self.assertIn(
                        "Detected shift: x=4 um, y=-5 um, z=6 um.",
                        window.calibration_panel.calibration_status_label.text(),
                    )
                    self.assertTrue(
                        window.calibration_panel.correct_sample_button.isEnabled()
                    )
                    self.assertTrue(
                        window.calibration_panel.detect_shift_button.isEnabled()
                    )
                finally:
                    window.close()

    def test_move_detected_starts_correction_without_immediate_server_reply(self):
        get_qapp()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "calibration.h5"
            calibration = write_sample_calibration(path)
            motor_backend = object()
            with patched_main_window_runtime():
                window = MainWindow()
                try:
                    window._server.motor_backend = motor_backend
                    window._on_new_calibration_ready(calibration)

                    window._on_move_detected(7)

                    thread = window._correction_thread
                    self.assertTrue(thread.started)
                    self.assertIs(thread.calibration, window._calibration)
                    self.assertEqual(thread.calibration_path, path)
                    self.assertIs(thread.motor_backend, motor_backend)
                    self.assertIsNotNone(thread.camera_pair)
                    self.assertEqual(window._server.result_calls, [])
                    self.assertTrue(window._server_correction_pending)
                    self.assertEqual(window._server_correction_target, 7)
                    self.assertFalse(
                        window.calibration_panel.correct_sample_button.isEnabled()
                    )
                    self.assertFalse(
                        window.calibration_panel.new_calibration_button.isEnabled()
                    )
                finally:
                    window.close()

    def test_server_triggered_correction_success_replies_ok_after_result(self):
        get_qapp()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "calibration.h5"
            calibration = write_sample_calibration(path)
            result = correction_result(converged=False, moves=3, residual=0.5)
            with patched_main_window_runtime():
                window = MainWindow()
                try:
                    window._on_new_calibration_ready(calibration)
                    window._on_move_detected(8)

                    window._correction_thread.running = False
                    window._on_correction_ready(result)

                    self.assertFalse(window._server_correction_pending)
                    self.assertIsNone(window._server_correction_target)
                    self.assertEqual(len(window._server.result_calls), 1)
                    success, message = window._server.result_calls[0]
                    self.assertTrue(success)
                    self.assertIn("did not converge after 3 move(s)", message)
                finally:
                    window.close()

    def test_server_triggered_motor_timeout_replies_error(self):
        get_qapp()
        with tempfile.TemporaryDirectory() as tmpdir:
            calibration = write_sample_calibration(Path(tmpdir) / "calibration.h5")
            with patched_main_window_runtime():
                window = MainWindow()
                try:
                    window._on_new_calibration_ready(calibration)
                    window._on_move_detected(9)

                    window._correction_thread.running = False
                    error_message = "Timed out waiting for motor move completion"
                    with patch(
                        "merlin_track_position.interface.main_window.QtWidgets.QMessageBox.critical"
                    ) as critical:
                        window._on_correction_failed(error_message)

                    self.assertFalse(window._server_correction_pending)
                    self.assertIsNone(window._server_correction_target)
                    self.assertEqual(
                        window._server.result_calls,
                        [(False, error_message)],
                    )
                    critical.assert_called_once()
                finally:
                    window.close()

    def test_move_detected_unavailable_returns_ok_and_warns_user(self):
        get_qapp()
        with patched_main_window_runtime():
            window = MainWindow()
            try:
                with (
                    patch.object(window, "_raise_for_user_attention") as attention,
                    patch(
                        "merlin_track_position.interface.main_window.QtWidgets.QMessageBox.warning"
                    ) as warning,
                ):
                    window._on_move_detected(10)

                self.assertFalse(window._correction_thread.started)
                self.assertFalse(window._server_correction_pending)
                self.assertEqual(len(window._server.result_calls), 1)
                success, message = window._server.result_calls[0]
                self.assertTrue(success)
                self.assertIn("requires a loaded calibration", message)
                attention.assert_called_once()
                warning.assert_called_once()
            finally:
                window.close()

    def test_correction_success_stores_result_reloads_calibration_and_reports_status(
        self,
    ):
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

    def test_correction_progress_updates_steps_while_in_progress(self):
        get_qapp()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "calibration.h5"
            calibration = write_sample_calibration(path)
            result = correction_result_with_moves()
            with patched_main_window_runtime():
                window = MainWindow()
                try:
                    window._on_new_calibration_ready(calibration)
                    window.calibration_panel.show_correction_in_progress()

                    window._on_correction_progress(result)

                    self.assertIs(window._last_correction_result, result)
                    self.assertFalse(
                        window.calibration_panel.correct_sample_button.isEnabled()
                    )
                    self.assertIn(
                        "Correction in progress after 2 move(s)",
                        window.calibration_panel.calibration_status_label.text(),
                    )
                    self.assertEqual(
                        window.calibration_panel.correction_steps_table.rowCount(),
                        2,
                    )
                    self.assertIn(
                        "Applied total: x=1 um, y=-2 um, z=3.25 um.",
                        window.calibration_panel.correction_steps_summary_label.text(),
                    )
                finally:
                    window.close()

    def test_loading_calibration_restores_latest_correction_result(self):
        get_qapp()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "calibration.h5"
            write_sample_calibration(path)
            history_path = correction_history_path(path)
            expected = correction_result(
                converged=False,
                moves=3,
                residual=0.75,
                warnings="residual increased",
            ).assign_attrs(
                {
                    "calibration_path": str(path),
                    "correction_history_path": str(history_path),
                    "correction_history_completed": True,
                    "correction_applied": True,
                }
            )
            save_correction_history_dataset(expected, history_path, run_id=0)

            with patched_main_window_runtime():
                window = MainWindow()
                try:
                    with patch(
                        "merlin_track_position.interface.main_window."
                        "QtWidgets.QFileDialog.getOpenFileName",
                        return_value=(str(path), ""),
                    ):
                        window._on_load_calibration_clicked()

                    self.assertIsNotNone(window._last_correction_result)
                    assert window._last_correction_result is not None
                    self.assertEqual(
                        window._last_correction_result.attrs["correction_history_path"],
                        str(history_path),
                    )
                    self.assertIn(
                        "Correction did not converge after 3 move(s)",
                        window.calibration_panel.calibration_status_label.text(),
                    )
                    self.assertEqual(
                        window.calibration_panel.calibration_warnings_text.toPlainText(),
                        "residual increased",
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

    def test_correction_result_shows_pending_persistence_warning(self):
        get_qapp()
        panel = CalibrationPanel()
        try:
            result = correction_result(warnings="").assign_attrs(
                {
                    "correction_history_persistence_status": "pending",
                    "correction_history_persistence_message": "locked",
                }
            )

            panel.show_correction_result(result)

            self.assertIn(
                "Correction history file write pending: locked",
                panel.calibration_warnings_text.toPlainText(),
            )
        finally:
            panel.close()


class CalibrationStartDialogTests(unittest.TestCase):
    def test_dialog_exposes_save_path(self):
        get_qapp()
        dialog = CalibrationStartDialog()

        self.assertIsNotNone(
            dialog.findChild(QtWidgets.QLineEdit, "calibration_output_path_edit")
        )
        self.assertTrue(str(dialog.output_path()).endswith(".h5"))

    def test_dialog_uses_supplied_default_output_path(self):
        get_qapp()
        path = Path("/tmp/scan/calibration.h5")
        dialog = CalibrationStartDialog(default_output_path=path)

        self.assertEqual(dialog.output_path(), path)

    def test_default_calibration_path_uses_scan_base_directory(self):
        with patch.object(
            main_window,
            "get_base_file_dir",
            return_value=Path("/tmp/current_scan"),
        ):
            self.assertEqual(
                main_window._default_calibration_path(),
                Path("/tmp/current_scan/calibration.h5"),
            )

    def test_default_calibration_path_falls_back_off_daq_pc(self):
        with (
            patch.object(
                main_window,
                "get_base_file_dir",
                side_effect=FileNotFoundError("missing setup"),
            ),
            patch.object(main_window, "IS_DAQ_PC", False),
        ):
            self.assertEqual(
                main_window._default_calibration_path(),
                Path.home() / "calibration.h5",
            )

    def test_default_calibration_path_does_not_fall_back_on_daq_pc(self):
        with (
            patch.object(
                main_window,
                "get_base_file_dir",
                side_effect=FileNotFoundError("missing setup"),
            ),
            patch.object(main_window, "IS_DAQ_PC", True),
        ):
            with self.assertRaises(FileNotFoundError):
                main_window._default_calibration_path()


class ParseConfigTests(unittest.TestCase):
    def test_get_base_file_dir_returns_configured_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            setup_path = Path(tmpdir) / "Instrument Scan Setup.txt"
            setup_path.write_text(
                "<Path>"
                "<Name>Data file base directory</Name>"
                "<Val>/tmp/current_scan</Val>"
                "</Path>",
                encoding="iso-8859-1",
            )

            with patch.object(parse_config, "INSTR_SCAN_SETUP_PATH", setup_path):
                self.assertEqual(
                    parse_config.get_base_file_dir(),
                    Path("/tmp/current_scan"),
                )


if __name__ == "__main__":
    unittest.main()

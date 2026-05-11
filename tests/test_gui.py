import math
import os
import threading
import time
import unittest
from unittest.mock import Mock, patch

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
from merlin_track_position.instruments.cameras import crop_image_to_roi
from merlin_track_position.interface.calibration_panel import (
    CalibrationPanel,
    _calibration_summary,
)
from merlin_track_position.interface.main_window import (
    CalibrationStartDialog,
    IMAGE_REFRESH_INTERVAL_MS,
    MainWindow,
    _MainWindowGUI,
    _clamp_roi_geometry,
    _default_roi_geometry,
    _roi_geometries_from_calibration_metadata,
    _roi_metadata_from_geometries,
    _validate_calibration_dataset,
)
from merlin_track_position.tracking.calibration_core import (
    CAMERAS,
    MEASUREMENT_WARNING_SUMMARY,
    OBSERVATION_AXES,
    PIXEL_AXES,
    STAGE_AXES,
)
from qtpy import QtCore, QtWidgets


class FakeSettings:
    def __init__(self, *args, **kwargs):
        del args, kwargs
        self.values = {}

    def value(self, key, fallback=None):
        return self.values.get(key, fallback)

    def setValue(self, key, value):
        self.values[key] = value

    def sync(self):
        pass


class FakeMotorServer(QtCore.QObject):
    sigMoveDetected = QtCore.Signal(int)

    def start(self):
        pass

    def stop(self):
        pass

    def wait(self):
        pass

    def set_result(self, success, msg):
        del success, msg


class FakeCalibrationThread(QtCore.QObject):
    sigCalibrationReady = QtCore.Signal(object)
    sigCalibrationStep = QtCore.Signal(int, float, float, float, object, object)
    sigCalibrationFailed = QtCore.Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.configured = None
        self.running = False
        self.started = False

    def configure(self, n, step_um, image_generator, roi_metadata):
        self.configured = (n, step_um, image_generator, roi_metadata)

    def isRunning(self):
        return self.running

    def start(self):
        self.started = True
        self.running = True

    def stop(self):
        self.running = False

    def wait(self):
        pass


class FakeAcceptedCalibrationStartDialog:
    def __init__(self, parent=None):
        del parent

    def exec(self):
        return QtWidgets.QDialog.DialogCode.Accepted

    def parameters(self):
        return (2, 1.0)


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
    coords = {
        "sample": np.arange(stage.shape[0], dtype=np.int64),
        "stage_axis": list(STAGE_AXES),
        "camera": list(CAMERAS),
        "pixel_axis": list(PIXEL_AXES),
    }
    data_vars = {
        "stage_to_pixel": (
            ("camera", "pixel_axis", "stage_axis"),
            stage_to_pixel,
            {"units": "px/um"},
        ),
        "stage_um": (("sample", "stage_axis"), stage, {"units": "um"}),
        "measured_shift_px": (
            ("sample", "camera", "pixel_axis"),
            measured,
            {"units": "px"},
        ),
    }

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
            "format_version": "1",
            "warnings": "",
        },
    )


def _wait_for_capture_count(app, get_cam0, get_cam1, expected_count):
    deadline = time.monotonic() + 2.0
    while (
        get_cam0.call_count < expected_count
        or get_cam1.call_count < expected_count
    ):
        if time.monotonic() > deadline:
            raise AssertionError(
                f"timed out waiting for {expected_count} image captures"
            )
        app.processEvents()
        QtCore.QThread.msleep(10)
    app.processEvents()
    app.processEvents()


def _image_refresh_threads_enabled(window):
    return {
        camera: thread.is_enabled()
        for camera, thread in window._image_refresh_threads.items()
    }


def _wait_for_image_item(app, image_item, expected_image):
    deadline = time.monotonic() + 2.0
    while not np.array_equal(image_item.image, expected_image):
        if time.monotonic() > deadline:
            raise AssertionError("timed out waiting for image item update")
        app.processEvents()
        QtCore.QThread.msleep(10)
    app.processEvents()


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

    def test_roi_metadata_round_trips_complete_geometries(self):
        roi_geometries = {
            "cam0": (10.0, 20.0, 30.0, 40.0),
            "cam1": (100.0, 110.0, 120.0, 130.0),
        }
        calibration = _synthetic_calibration(
            np.array(
                [
                    [0.0, 0.0, 0.0],
                    [20.0, 0.0, 0.0],
                    [0.0, 20.0, 0.0],
                    [0.0, 0.0, 20.0],
                ]
            ),
            np.array(
                [
                    [[0.5, -0.1, 0.2], [0.2, 0.4, -0.1]],
                    [[-0.3, 0.2, 0.4], [0.1, -0.2, 0.3]],
                ]
            ),
        )
        calibration = calibration.assign_attrs(
            _roi_metadata_from_geometries(roi_geometries)
        )

        self.assertEqual(
            _roi_geometries_from_calibration_metadata(calibration),
            roi_geometries,
        )

    def test_roi_metadata_helper_ignores_legacy_calibration_without_rois(self):
        calibration = _synthetic_calibration(
            np.array(
                [
                    [0.0, 0.0, 0.0],
                    [20.0, 0.0, 0.0],
                    [0.0, 20.0, 0.0],
                    [0.0, 0.0, 20.0],
                ]
            ),
            np.array(
                [
                    [[0.5, -0.1, 0.2], [0.2, 0.4, -0.1]],
                    [[-0.3, 0.2, 0.4], [0.1, -0.2, 0.3]],
                ]
            ),
        )

        self.assertIsNone(_roi_geometries_from_calibration_metadata(calibration))

    def test_calibration_summary_reports_core_metrics_and_repeatability_std(self):
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
        np.testing.assert_allclose(
            summary["return_to_origin_image_error_um"],
            [0.0, 20.0, 0.0],
            atol=1e-12,
        )
        self.assertEqual(summary["warnings"], ())
        self.assertNotIn("position_count", summary["repeatability"])
        self.assertNotIn("capture_count", summary["repeatability"])
        self.assertAlmostEqual(summary["repeatability"]["max_rms_std_px"], 0.0)

    def test_calibration_summary_reads_legacy_repeatability_without_count_metrics(self):
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
        calibration = calibration.drop_vars(
            ["repeatability_mean_rms_std_px", "repeatability_max_rms_std_px"],
            errors="ignore",
        )
        calibration = calibration.assign_coords(
            repeatability_position=np.arange(2, dtype=np.int64)
        )
        calibration["repeatability_count"] = (
            ("repeatability_position",),
            np.array([2, 3], dtype=np.int64),
        )
        calibration["repeatability_rms_std_px"] = (
            ("repeatability_position",),
            np.array([0.25, 0.75], dtype=float),
            {"units": "px"},
        )

        summary = _calibration_summary(calibration)

        self.assertNotIn("position_count", summary["repeatability"])
        self.assertNotIn("capture_count", summary["repeatability"])
        self.assertAlmostEqual(summary["repeatability"]["mean_rms_std_px"], 0.5)
        self.assertAlmostEqual(summary["repeatability"]["max_rms_std_px"], 0.75)

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
        broken = calibration.drop_vars("measured_shift_px")

        with self.assertRaisesRegex(ValueError, "measured_shift_px"):
            _validate_calibration_dataset(broken)


class MainWindowGUISmokeTests(unittest.TestCase):
    def test_main_window_gui_constructs_offscreen(self):
        app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        window = _MainWindowGUI()
        try:
            self.assertIsInstance(window.image_graphics_layout, pg.GraphicsLayoutWidget)
            self.assertEqual(
                window.image_auto_refresh_checkbox.objectName(),
                "image_auto_refresh_checkbox",
            )
            self.assertTrue(window.image_auto_refresh_checkbox.isChecked())
            self.assertIn(
                f"{IMAGE_REFRESH_INTERVAL_MS} ms",
                window.image_auto_refresh_checkbox.text(),
            )
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

    def test_main_window_auto_refresh_checkbox_controls_thread_and_cache(self):
        app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        images_cam0 = [
            np.full((IMAGE_HEIGHT_CAM0, IMAGE_WIDTH_CAM0), 1.0),
            np.full((IMAGE_HEIGHT_CAM0, IMAGE_WIDTH_CAM0), 2.0),
        ]
        images_cam1 = [
            np.full((IMAGE_HEIGHT_CAM1, IMAGE_WIDTH_CAM1), 3.0),
            np.full((IMAGE_HEIGHT_CAM1, IMAGE_WIDTH_CAM1), 4.0),
        ]
        get_cam0 = Mock(side_effect=images_cam0)
        get_cam1 = Mock(side_effect=images_cam1)

        with (
            patch(
                "merlin_track_position.interface.main_window.MotorServer",
                FakeMotorServer,
            ),
            patch(
                "merlin_track_position.interface.main_window.QtCore.QSettings",
                FakeSettings,
            ),
            patch(
                "merlin_track_position.interface.main_window.get_framegrabber_image",
                get_cam0,
            ),
            patch(
                "merlin_track_position.interface.main_window.get_basler_image",
                get_cam1,
            ),
        ):
            window = MainWindow()
            try:
                self.assertEqual(set(window._image_refresh_threads), {"cam0", "cam1"})
                for thread in window._image_refresh_threads.values():
                    self.assertEqual(thread.interval_ms, IMAGE_REFRESH_INTERVAL_MS)
                self.assertEqual(
                    _image_refresh_threads_enabled(window),
                    {"cam0": True, "cam1": True},
                )
                _wait_for_capture_count(app, get_cam0, get_cam1, 1)

                self.assertIsInstance(window._latest_images, tuple)
                self.assertEqual(len(window._latest_images), 2)
                np.testing.assert_array_equal(
                    window._latest_images[0],
                    images_cam0[0],
                )
                np.testing.assert_array_equal(
                    window._latest_images[1],
                    images_cam1[0],
                )

                window.image_auto_refresh_checkbox.setChecked(False)
                self.assertEqual(
                    _image_refresh_threads_enabled(window),
                    {"cam0": False, "cam1": False},
                )

                window.image_auto_refresh_checkbox.setChecked(True)
                self.assertEqual(
                    _image_refresh_threads_enabled(window),
                    {"cam0": True, "cam1": True},
                )
                _wait_for_capture_count(app, get_cam0, get_cam1, 2)
                window.image_auto_refresh_checkbox.setChecked(False)

                np.testing.assert_array_equal(
                    window._latest_images[0],
                    images_cam0[1],
                )
                np.testing.assert_array_equal(
                    window._latest_images[1],
                    images_cam1[1],
                )
                self.assertEqual(get_cam0.call_count, 2)
                self.assertEqual(get_cam1.call_count, 2)
            finally:
                window.close()
                app.processEvents()

    def test_camera_refresh_threads_update_independently(self):
        app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        image_cam0 = np.full((IMAGE_HEIGHT_CAM0, IMAGE_WIDTH_CAM0), 7.0)
        image_cam1 = np.full((IMAGE_HEIGHT_CAM1, IMAGE_WIDTH_CAM1), 8.0)
        cam1_started = threading.Event()
        release_cam1 = threading.Event()
        get_cam0 = Mock(return_value=image_cam0)

        def get_cam1():
            cam1_started.set()
            self.assertTrue(release_cam1.wait(timeout=2.0))
            return image_cam1

        with (
            patch(
                "merlin_track_position.interface.main_window.MotorServer",
                FakeMotorServer,
            ),
            patch(
                "merlin_track_position.interface.main_window.QtCore.QSettings",
                FakeSettings,
            ),
            patch(
                "merlin_track_position.interface.main_window.get_framegrabber_image",
                get_cam0,
            ),
            patch(
                "merlin_track_position.interface.main_window.get_basler_image",
                get_cam1,
            ),
        ):
            window = MainWindow()
            try:
                self.assertTrue(cam1_started.wait(timeout=1.0))
                _wait_for_image_item(app, window.image_items["cam0"], image_cam0)
                self.assertIsNone(window._latest_images)
                self.assertGreaterEqual(get_cam0.call_count, 1)

                release_cam1.set()
                _wait_for_image_item(app, window.image_items["cam1"], image_cam1)
                self.assertIsInstance(window._latest_images, tuple)
            finally:
                release_cam1.set()
                window.close()
                app.processEvents()

    def test_calibration_uses_shared_capture_then_restores_checked_refresh_state(self):
        app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        initial_cam0 = np.zeros((IMAGE_HEIGHT_CAM0, IMAGE_WIDTH_CAM0), dtype=float)
        initial_cam1 = np.zeros((IMAGE_HEIGHT_CAM1, IMAGE_WIDTH_CAM1), dtype=float)
        fresh_cam0 = np.arange(
            IMAGE_HEIGHT_CAM0 * IMAGE_WIDTH_CAM0,
            dtype=float,
        ).reshape(IMAGE_HEIGHT_CAM0, IMAGE_WIDTH_CAM0)
        fresh_cam1 = np.arange(
            IMAGE_HEIGHT_CAM1 * IMAGE_WIDTH_CAM1,
            dtype=float,
        ).reshape(IMAGE_HEIGHT_CAM1, IMAGE_WIDTH_CAM1)
        resumed_cam0 = np.full((IMAGE_HEIGHT_CAM0, IMAGE_WIDTH_CAM0), -1.0)
        resumed_cam1 = np.full((IMAGE_HEIGHT_CAM1, IMAGE_WIDTH_CAM1), -2.0)
        get_cam0 = Mock(side_effect=[initial_cam0, fresh_cam0, resumed_cam0])
        get_cam1 = Mock(side_effect=[initial_cam1, fresh_cam1, resumed_cam1])
        roi_cam0 = (2.0, 3.0, 5.0, 4.0)
        roi_cam1 = (4.0, 5.0, 6.0, 7.0)

        with (
            patch(
                "merlin_track_position.interface.main_window.MotorServer",
                FakeMotorServer,
            ),
            patch(
                "merlin_track_position.interface.main_window.CalibrationThread",
                FakeCalibrationThread,
            ),
            patch(
                "merlin_track_position.interface.main_window.CalibrationStartDialog",
                FakeAcceptedCalibrationStartDialog,
            ),
            patch(
                "merlin_track_position.interface.main_window.QtCore.QSettings",
                FakeSettings,
            ),
            patch(
                "merlin_track_position.interface.main_window.QtWidgets.QMessageBox.critical",
                Mock(),
            ),
            patch(
                "merlin_track_position.interface.main_window.get_framegrabber_image",
                get_cam0,
            ),
            patch(
                "merlin_track_position.interface.main_window.get_basler_image",
                get_cam1,
            ),
        ):
            window = MainWindow()
            try:
                _wait_for_capture_count(app, get_cam0, get_cam1, 1)
                window._set_roi_geometry("cam0", roi_cam0)
                window._set_roi_geometry("cam1", roi_cam1)

                window._on_new_calibration_clicked()

                thread = window._calibration_thread
                self.assertTrue(thread.started)
                self.assertEqual(
                    _image_refresh_threads_enabled(window),
                    {"cam0": False, "cam1": False},
                )
                self.assertFalse(window.image_auto_refresh_checkbox.isEnabled())
                self.assertTrue(window.image_auto_refresh_checkbox.isChecked())

                n, step_um, image_generator, roi_metadata = thread.configured
                self.assertEqual(n, 2)
                self.assertEqual(step_um, 1.0)
                self.assertEqual(roi_metadata["roi_cam0_x"], roi_cam0[0])
                self.assertEqual(roi_metadata["roi_cam1_x"], roi_cam1[0])

                cropped_cam0, cropped_cam1 = image_generator()
                np.testing.assert_array_equal(window._latest_images[0], fresh_cam0)
                np.testing.assert_array_equal(window._latest_images[1], fresh_cam1)
                np.testing.assert_array_equal(
                    cropped_cam0,
                    crop_image_to_roi(fresh_cam0, roi_cam0),
                )
                np.testing.assert_array_equal(
                    cropped_cam1,
                    crop_image_to_roi(fresh_cam1, roi_cam1),
                )

                thread.running = False
                window._on_new_calibration_failed("boom")

                self.assertTrue(window.image_auto_refresh_checkbox.isEnabled())
                self.assertTrue(window.image_auto_refresh_checkbox.isChecked())
                self.assertEqual(
                    _image_refresh_threads_enabled(window),
                    {"cam0": True, "cam1": True},
                )
                _wait_for_capture_count(app, get_cam0, get_cam1, 3)
                window.image_auto_refresh_checkbox.setChecked(False)
                np.testing.assert_array_equal(window._latest_images[0], resumed_cam0)
                np.testing.assert_array_equal(window._latest_images[1], resumed_cam1)
                self.assertEqual(get_cam0.call_count, 3)
                self.assertEqual(get_cam1.call_count, 3)
            finally:
                window.close()
                app.processEvents()

    def test_calibration_ready_restores_unchecked_refresh_state(self):
        app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        get_cam0 = Mock(
            return_value=np.zeros((IMAGE_HEIGHT_CAM0, IMAGE_WIDTH_CAM0), dtype=float)
        )
        get_cam1 = Mock(
            return_value=np.zeros((IMAGE_HEIGHT_CAM1, IMAGE_WIDTH_CAM1), dtype=float)
        )
        calibration = _synthetic_calibration(
            np.array(
                [
                    [0.0, 0.0, 0.0],
                    [20.0, 0.0, 0.0],
                    [0.0, 20.0, 0.0],
                    [0.0, 0.0, 20.0],
                ]
            ),
            np.array(
                [
                    [[0.5, -0.1, 0.2], [0.2, 0.4, -0.1]],
                    [[-0.3, 0.2, 0.4], [0.1, -0.2, 0.3]],
                ]
            ),
        )

        with (
            patch(
                "merlin_track_position.interface.main_window.MotorServer",
                FakeMotorServer,
            ),
            patch(
                "merlin_track_position.interface.main_window.QtCore.QSettings",
                FakeSettings,
            ),
            patch(
                "merlin_track_position.interface.main_window.get_framegrabber_image",
                get_cam0,
            ),
            patch(
                "merlin_track_position.interface.main_window.get_basler_image",
                get_cam1,
            ),
        ):
            window = MainWindow()
            try:
                _wait_for_capture_count(app, get_cam0, get_cam1, 1)
                window.image_auto_refresh_checkbox.setChecked(False)
                self.assertEqual(
                    _image_refresh_threads_enabled(window),
                    {"cam0": False, "cam1": False},
                )

                window._pause_image_auto_refresh_for_calibration()
                self.assertFalse(window.image_auto_refresh_checkbox.isEnabled())

                window._on_new_calibration_ready(calibration)

                self.assertTrue(window.image_auto_refresh_checkbox.isEnabled())
                self.assertFalse(window.image_auto_refresh_checkbox.isChecked())
                self.assertEqual(
                    _image_refresh_threads_enabled(window),
                    {"cam0": False, "cam1": False},
                )
                self.assertEqual(get_cam0.call_count, 1)
                self.assertEqual(get_cam1.call_count, 1)
            finally:
                window.close()
                app.processEvents()

    def test_calibration_start_dialog_defaults_to_requested_values(self):
        app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        dialog = CalibrationStartDialog()
        try:
            self.assertEqual(dialog.n_spin.value(), 5)
            self.assertAlmostEqual(dialog.step_um_spin.value(), 15.0)
            self.assertEqual(dialog.parameters(), (5, 15.0))
        finally:
            dialog.close()
            app.processEvents()

    def test_main_window_applies_roi_metadata_from_loaded_calibration(self):
        app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        roi_geometries = {
            "cam0": (10.0, 20.0, 30.0, 40.0),
            "cam1": (100.0, 110.0, 120.0, 130.0),
        }
        calibration = _synthetic_calibration(
            np.array(
                [
                    [0.0, 0.0, 0.0],
                    [20.0, 0.0, 0.0],
                    [0.0, 20.0, 0.0],
                    [0.0, 0.0, 20.0],
                ]
            ),
            np.array(
                [
                    [[0.5, -0.1, 0.2], [0.2, 0.4, -0.1]],
                    [[-0.3, 0.2, 0.4], [0.1, -0.2, 0.3]],
                ]
            ),
        ).assign_attrs(_roi_metadata_from_geometries(roi_geometries))

        with (
            patch(
                "merlin_track_position.interface.main_window.MotorServer",
                FakeMotorServer,
            ),
            patch(
                "merlin_track_position.interface.main_window.QtCore.QSettings",
                FakeSettings,
            ),
        ):
            window = MainWindow()
        try:
            self.assertTrue(window._apply_calibration_roi_metadata(calibration))
            self.assertEqual(window._get_roi_geometry("cam0"), roi_geometries["cam0"])
            self.assertEqual(window._get_roi_geometry("cam1"), roi_geometries["cam1"])
        finally:
            window.close()
            app.processEvents()

    def test_main_window_leaves_rois_for_legacy_calibration_without_metadata(self):
        app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        calibration = _synthetic_calibration(
            np.array(
                [
                    [0.0, 0.0, 0.0],
                    [20.0, 0.0, 0.0],
                    [0.0, 20.0, 0.0],
                    [0.0, 0.0, 20.0],
                ]
            ),
            np.array(
                [
                    [[0.5, -0.1, 0.2], [0.2, 0.4, -0.1]],
                    [[-0.3, 0.2, 0.4], [0.1, -0.2, 0.3]],
                ]
            ),
        )

        with (
            patch(
                "merlin_track_position.interface.main_window.MotorServer",
                FakeMotorServer,
            ),
            patch(
                "merlin_track_position.interface.main_window.QtCore.QSettings",
                FakeSettings,
            ),
        ):
            window = MainWindow()
        try:
            before = {
                camera: window._get_roi_geometry(camera)
                for camera in ("cam0", "cam1")
            }
            self.assertFalse(window._apply_calibration_roi_metadata(calibration))
            self.assertEqual(
                {
                    camera: window._get_roi_geometry(camera)
                    for camera in ("cam0", "cam1")
                },
                before,
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
            self.assertFalse(panel.calibration_progress_bar.isHidden())
        finally:
            panel.close()
            app.processEvents()

    def test_show_calibration_step_updates_progress_and_eta(self):
        app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        panel = CalibrationPanel()
        try:
            panel.show_calibration_in_progress(total_steps=4)
            panel.show_calibration_step(
                idx=1,
                total_steps=4,
                dx=1.0,
                dy=2.0,
                dz=3.0,
                elapsed_s=10.0,
                eta_s=10.0,
            )

            self.assertEqual(panel.calibration_progress_bar.value(), 2)
            self.assertEqual(panel.calibration_progress_bar.maximum(), 4)
            self.assertNotIn("2 / 4", panel.calibration_status_label.text())
            self.assertNotIn("remaining", panel.calibration_status_label.text())
            self.assertIn("ETA 0:10", panel.calibration_status_label.text())
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

    def test_show_loaded_calibration_expands_measurement_warning_details(self):
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
            measurement_warnings = np.full(
                (stage.shape[0], len(CAMERAS)),
                "",
                dtype=object,
            )
            measurement_warnings[2, 1] = (
                "low texture\nregistration error is not finite"
            )
            calibration = calibration.assign(
                measurement_warnings=(
                    ("sample", "camera"),
                    measurement_warnings,
                )
            ).assign_attrs(warnings=MEASUREMENT_WARNING_SUMMARY)

            panel.show_loaded_calibration(calibration, "calibration.h5")

            warnings_text = panel.calibration_warnings_text.toPlainText()
            self.assertNotIn(MEASUREMENT_WARNING_SUMMARY, warnings_text)
            self.assertIn(
                "step 3 (x=0, y=20, z=0 um), cam1: low texture",
                warnings_text,
            )
            self.assertIn(
                "step 3 (x=0, y=20, z=0 um), "
                "cam1: registration error is not finite",
                warnings_text,
            )
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

import os
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import h5py
import numpy as np
import xarray as xr

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from merlin_track_position.interface.calibration_panel import (  # noqa: E402
    CalibrationPanel,
    _calibration_arrays,
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
from merlin_track_position.interface.registration_settings import (  # noqa: E402
    REGISTRATION_CAPTURE_AGGREGATION_SETTINGS_KEY,
    REGISTRATION_CAPTURE_COUNT_SETTINGS_KEY,
    REGISTRATION_CLIP_HIGH_SETTINGS_KEY,
    REGISTRATION_CLIP_LOW_SETTINGS_KEY,
    REGISTRATION_NORMALIZATION_SETTINGS_KEY,
    REGISTRATION_UPSAMPLE_FACTOR_SETTINGS_KEY,
    REGISTRATION_USE_ECC_REFINEMENT_SETTINGS_KEY,
    registration_config_to_measurement_kwargs,
    registration_config_to_shift_kwargs,
)
from merlin_track_position.interface.shift_monitor_window import (  # noqa: E402
    SIDE_PANEL_MAX_WIDTH,
    SHIFT_MONITOR_EXPORT_FORMAT,
    ShiftMonitorWindow,
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
        self.values: dict[str, object] = {}
        self.set_calls: list[tuple[str, object]] = []

    def value(self, key, fallback=None):
        return self.values.get(str(key), fallback)

    def setValue(self, key, value):
        self.values[str(key)] = value
        self.set_calls.append((str(key), value))

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
        self.correction_mode = None
        self.shift_kwargs = None
        self.started = False
        self.running = False

    def configure(
        self,
        calibration,
        camera_pair,
        calibration_path,
        motor_backend=None,
        correction_mode="camera",
        shift_kwargs=None,
    ):
        self.calibration = calibration
        self.camera_pair = camera_pair
        self.calibration_path = Path(calibration_path)
        self.motor_backend = motor_backend
        self.correction_mode = correction_mode
        self.shift_kwargs = {} if shift_kwargs is None else dict(shift_kwargs)

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
        self.shift_kwargs = None
        self.started = False
        self.running = False

    def configure(
        self,
        calibration,
        camera_pair,
        *,
        shift_kwargs=None,
    ):
        self.calibration = calibration
        self.camera_pair = camera_pair
        self.shift_kwargs = {} if shift_kwargs is None else dict(shift_kwargs)

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


def roi_display_geometry(window, camera):
    roi = window.image_rois[camera]
    position = roi.pos()
    size = roi.size()
    return (position.x(), position.y(), size.x(), size.y())


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


class FakeShiftRegistrationThread(QtCore.QObject):
    sigShiftReady = QtCore.Signal(str, float, object, str)
    sigShiftFailed = QtCore.Signal(str, str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.submissions = []
        self.started = False
        self.stopped = False

    def start(self):
        self.started = True

    def submit(self, camera, reference, current, config, elapsed_s):
        self.submissions.append(
            (
                camera,
                np.asarray(reference).copy(),
                np.asarray(current).copy(),
                dict(config),
                float(elapsed_s),
            )
        )

    def clear_pending(self):
        pass

    def stop(self):
        self.stopped = True

    def wait(self):
        pass


@contextmanager
def patched_shift_monitor_worker():
    with patch(
        "merlin_track_position.interface.shift_monitor_window._ShiftRegistrationThread",
        FakeShiftRegistrationThread,
    ):
        yield


class ShiftMonitorWindowTests(unittest.TestCase):
    def test_monitor_is_top_level_window_even_with_parent(self):
        get_qapp()
        parent = QtWidgets.QWidget()
        with patched_shift_monitor_worker():
            window = ShiftMonitorWindow(FakeSettings(), parent=parent)
            try:
                self.assertTrue(window.isWindow())
            finally:
                window.close()
                parent.close()

    def test_monitor_keeps_controls_in_narrow_right_column(self):
        get_qapp()
        with patched_shift_monitor_worker():
            window = ShiftMonitorWindow(FakeSettings())
            try:
                layout = window.layout()
                side_panel = window.findChild(
                    QtWidgets.QWidget,
                    "shift_monitor_side_panel",
                )

                self.assertIs(layout.itemAt(0).widget(), window.graphics_layout)
                self.assertIs(layout.itemAt(1).widget(), side_panel)
                self.assertEqual(side_panel.maximumWidth(), SIDE_PANEL_MAX_WIDTH)
            finally:
                window.close()

    def test_monitor_plot_axes_do_not_use_si_prefixes(self):
        get_qapp()
        with patched_shift_monitor_worker():
            window = ShiftMonitorWindow(FakeSettings())
            try:
                for plot in window.plots.values():
                    self.assertFalse(plot.getAxis("bottom").autoSIPrefix)
                    self.assertFalse(plot.getAxis("left").autoSIPrefix)
            finally:
                window.close()

    def test_monitor_plot_x_axes_are_linked(self):
        get_qapp()
        with patched_shift_monitor_worker():
            window = ShiftMonitorWindow(FakeSettings())
            try:
                first_plot = window.plots[("cam0", "du_px")]
                for key, plot in window.plots.items():
                    if key == ("cam0", "du_px"):
                        continue
                    self.assertIs(plot.vb.linkedView(0), first_plot.vb)
            finally:
                window.close()

    def test_first_live_frame_becomes_reference_per_camera(self):
        get_qapp()
        settings = FakeSettings()
        first = np.arange(4 * 5, dtype=float).reshape(4, 5)
        second = first + 10.0

        with patched_shift_monitor_worker():
            window = ShiftMonitorWindow(settings)
            try:
                window.capture_count_spin.setValue(1)
                window.submit_frame("cam0", first)
                self.assertEqual(window._worker.submissions, [])
                self.assertIn("cam0", window.reference_label.text())

                window.submit_frame("cam0", second)

                self.assertEqual(len(window._worker.submissions), 1)
                camera, reference, current, _config, _elapsed_s = (
                    window._worker.submissions[0]
                )
                self.assertEqual(camera, "cam0")
                np.testing.assert_array_equal(reference, first)
                np.testing.assert_array_equal(current, second[np.newaxis, :, :])

                window.submit_frame("cam1", np.ones((3, 4)))

                self.assertEqual(len(window._worker.submissions), 1)
            finally:
                window.close()

    def test_monitor_buffers_capture_count_frames_per_submission(self):
        get_qapp()
        settings = FakeSettings()
        frames = [
            np.full((2, 2), value, dtype=float)
            for value in (1.0, 2.0, 3.0, 4.0)
        ]

        with patched_shift_monitor_worker():
            window = ShiftMonitorWindow(settings)
            try:
                window.capture_count_spin.setValue(2)
                window.submit_frame("cam0", frames[0])
                window.submit_frame("cam0", frames[1])
                self.assertEqual(window._worker.submissions, [])
                np.testing.assert_array_equal(
                    window._live_references["cam0"],
                    np.full((2, 2), 1.5),
                )

                window.submit_frame("cam0", frames[2])
                self.assertEqual(window._worker.submissions, [])
                window.submit_frame("cam0", frames[3])

                self.assertEqual(len(window._worker.submissions), 1)
                camera, reference, current, config, _elapsed_s = (
                    window._worker.submissions[0]
                )
                self.assertEqual(camera, "cam0")
                np.testing.assert_array_equal(reference, np.full((2, 2), 1.5))
                np.testing.assert_array_equal(
                    current,
                    np.stack(frames[2:4], axis=0),
                )
                self.assertEqual(config["capture_count"], 2)
            finally:
                window.close()

    def test_monitor_throttles_registration_submissions_by_camera(self):
        get_qapp()
        first = np.arange(4 * 5, dtype=float).reshape(4, 5)
        second = first + 1.0
        third = first + 2.0
        fourth = first + 3.0

        with (
            patched_shift_monitor_worker(),
            patch(
                "merlin_track_position.interface.shift_monitor_window.time.monotonic",
                side_effect=[100.0, 100.0, 101.0, 101.5, 103.2],
            ),
        ):
            window = ShiftMonitorWindow(FakeSettings())
            try:
                window.capture_count_spin.setValue(1)
                window.submit_frame("cam0", first)
                window.submit_frame("cam0", second)
                window.submit_frame("cam0", third)
                window.submit_frame("cam0", fourth)

                self.assertEqual(len(window._worker.submissions), 2)
                np.testing.assert_array_equal(
                    window._worker.submissions[0][2],
                    second[np.newaxis, :, :],
                )
                np.testing.assert_array_equal(
                    window._worker.submissions[1][2],
                    fourth[np.newaxis, :, :],
                )
            finally:
                window.close()

    def test_monitor_live_checkbox_pauses_submissions(self):
        get_qapp()
        first = np.arange(4 * 5, dtype=float).reshape(4, 5)

        with patched_shift_monitor_worker():
            window = ShiftMonitorWindow(FakeSettings())
            try:
                window.live_checkbox.setChecked(False)
                window.submit_frame("cam0", first)

                self.assertEqual(window._worker.submissions, [])
                self.assertEqual(window._live_references, {})
            finally:
                window.close()

    def test_calibration_reference_is_roi_matched_to_live_frame(self):
        get_qapp()
        settings = FakeSettings()
        metadata = _roi_metadata_from_geometries(
            {
                "cam0": (2.0, 3.0, 4.0, 5.0),
                "cam1": (0.0, 0.0, 6.0, 7.0),
            }
        )
        width0, height0 = main_window.CAMERA_IMAGE_SIZES["cam0"]
        width1, height1 = main_window.CAMERA_IMAGE_SIZES["cam1"]
        reference_cam0 = np.arange(height0 * width0, dtype=float).reshape(
            height0,
            width0,
        )
        reference_cam1 = np.arange(height1 * width1, dtype=float).reshape(
            height1,
            width1,
        )
        calibration = xr.Dataset(
            data_vars={
                "reference_cam0": (("cam0_v", "cam0_u"), reference_cam0),
                "reference_cam1": (("cam1_v", "cam1_u"), reference_cam1),
            },
            attrs=metadata,
        )
        current_cam0 = reference_cam0 + 1000.0

        with patched_shift_monitor_worker():
            window = ShiftMonitorWindow(settings)
            try:
                window.capture_count_spin.setValue(1)
                window.set_calibration(calibration)
                window.submit_frame("cam0", current_cam0)

                self.assertEqual(len(window._worker.submissions), 1)
                camera, reference, current, _config, _elapsed_s = (
                    window._worker.submissions[0]
                )
                self.assertEqual(camera, "cam0")
                np.testing.assert_array_equal(reference, reference_cam0[3:8, 2:6])
                np.testing.assert_array_equal(
                    current,
                    current_cam0[np.newaxis, 3:8, 2:6],
                )
            finally:
                window.close()

    def test_shift_ready_appends_plot_history_and_updates_stats(self):
        get_qapp()
        with patched_shift_monitor_worker():
            window = ShiftMonitorWindow(FakeSettings())
            try:
                window._on_shift_ready("cam0", 1.25, np.asarray([0.25, -0.5]), "")

                self.assertEqual(window._history["cam0"]["t"], [1.25])
                self.assertEqual(window._history["cam0"]["du_px"], [0.25])
                self.assertEqual(window._history["cam0"]["dv_px"], [-0.5])
                self.assertEqual(window.stats_table.item(0, 1).text(), "1")
                self.assertEqual(window.stats_table.item(0, 2).text(), "0.25")
                self.assertEqual(window.stats_table.item(1, 2).text(), "-0.5")
            finally:
                window.close()

    def test_save_persists_registration_controls(self):
        get_qapp()
        settings = FakeSettings()
        saved_configs = []
        with patched_shift_monitor_worker():
            window = ShiftMonitorWindow(settings)
            try:
                window.sigRegistrationConfigSaved.connect(saved_configs.append)
                window.clip_low_spin.setValue(10.0)
                window.clip_high_spin.setValue(90.0)
                window.normalization_combo.setCurrentIndex(
                    window.normalization_combo.findData("none")
                )
                window.upsample_spin.setValue(51)
                window.ecc_refinement_checkbox.setChecked(True)
                window.capture_count_spin.setValue(7)
                window.capture_aggregation_combo.setCurrentIndex(
                    window.capture_aggregation_combo.findData("mean_image")
                )

                window.save_button.click()

                self.assertEqual(
                    settings.values[REGISTRATION_CLIP_LOW_SETTINGS_KEY],
                    10.0,
                )
                self.assertEqual(
                    settings.values[REGISTRATION_CLIP_HIGH_SETTINGS_KEY],
                    90.0,
                )
                self.assertEqual(
                    settings.values[REGISTRATION_NORMALIZATION_SETTINGS_KEY],
                    "none",
                )
                self.assertEqual(
                    settings.values[REGISTRATION_UPSAMPLE_FACTOR_SETTINGS_KEY],
                    51,
                )
                self.assertEqual(
                    settings.values[REGISTRATION_USE_ECC_REFINEMENT_SETTINGS_KEY],
                    True,
                )
                self.assertEqual(
                    settings.values[REGISTRATION_CAPTURE_COUNT_SETTINGS_KEY],
                    7,
                )
                self.assertEqual(
                    settings.values[REGISTRATION_CAPTURE_AGGREGATION_SETTINGS_KEY],
                    "mean_image",
                )
                self.assertEqual(
                    registration_config_to_shift_kwargs(saved_configs[-1])[
                        "clip_percentiles"
                    ],
                    (10.0, 90.0),
                )
                self.assertEqual(
                    registration_config_to_shift_kwargs(saved_configs[-1])[
                        "use_ecc_refinement"
                    ],
                    True,
                )
                self.assertEqual(
                    registration_config_to_measurement_kwargs(saved_configs[-1])[
                        "capture_count"
                    ],
                    7,
                )
            finally:
                window.close()

    def test_export_writes_hdf5_plot_history(self):
        get_qapp()
        with tempfile.TemporaryDirectory() as tmpdir, patched_shift_monitor_worker():
            path = Path(tmpdir) / "shift_monitor.h5"
            window = ShiftMonitorWindow(FakeSettings())
            try:
                window.sample_period_spin.setValue(3.5)
                window.upsample_spin.setValue(37)
                window.ecc_refinement_checkbox.setChecked(True)
                window.capture_count_spin.setValue(5)
                window.capture_aggregation_combo.setCurrentIndex(
                    window.capture_aggregation_combo.findData("mean_image")
                )
                window._on_shift_ready(
                    "cam0",
                    1.25,
                    np.asarray([0.25, -0.5]),
                    "warn",
                )
                window._on_shift_ready("cam1", 2.5, np.asarray([1.0, -1.5]), "")

                window.export_history_to_hdf5(path)

                with h5py.File(path, "r") as exported:
                    self.assertEqual(
                        exported.attrs["format"],
                        SHIFT_MONITOR_EXPORT_FORMAT,
                    )
                    self.assertEqual(
                        exported.attrs["reference_source"],
                        "first_live_frame",
                    )
                    self.assertAlmostEqual(exported.attrs["sample_period_s"], 3.5)
                    self.assertEqual(
                        exported.attrs["registration_upsample_factor"],
                        37,
                    )
                    self.assertEqual(
                        exported.attrs["registration_use_ecc_refinement"],
                        True,
                    )
                    self.assertEqual(exported.attrs["registration_capture_count"], 5)
                    self.assertEqual(
                        exported.attrs["registration_capture_aggregation"],
                        "mean_image",
                    )
                    np.testing.assert_allclose(
                        exported["cam0/time_s"][:],
                        [1.25],
                    )
                    np.testing.assert_allclose(
                        exported["cam0/shift_px"][:],
                        [[0.25, -0.5]],
                    )
                    self.assertEqual(
                        exported["cam0/pixel_axis"].asstr()[:].tolist(),
                        ["du_px", "dv_px"],
                    )
                    self.assertEqual(
                        exported["cam0/warnings"].asstr()[:].tolist(),
                        ["warn"],
                    )
                    np.testing.assert_allclose(
                        exported["cam1/dv_px"][:],
                        [-1.5],
                    )
            finally:
                window.close()

    def test_export_button_uses_save_dialog(self):
        get_qapp()
        with tempfile.TemporaryDirectory() as tmpdir, patched_shift_monitor_worker():
            path = Path(tmpdir) / "shift_monitor"
            window = ShiftMonitorWindow(FakeSettings())
            try:
                with patch(
                    "merlin_track_position.interface.shift_monitor_window.QtWidgets.QFileDialog.getSaveFileName",
                    return_value=(str(path), "HDF5 files (*.h5 *.hdf5)"),
                ):
                    window.export_button.click()

                self.assertTrue(path.with_suffix(".h5").exists())
            finally:
                window.close()


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
    def test_summary_reports_px_per_cmd_mm_metrics(self):
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
            calibration["px_per_cmd_mm"].values,
            calibration["probe_command_delta_mm"].values,
        )
        np.testing.assert_allclose(
            summary["axis_sensitivity_px_per_cmd_mm"],
            expected_axis_sensitivity,
        )
        self.assertEqual(summary["residual_rms_px"], 0.0)
        self.assertEqual(summary["residual_max_cmd_mm"], 0.0)
        self.assertEqual(summary["readback_command_rms_mm"], 0.0)
        np.testing.assert_allclose(
            summary["px_per_cmd_mm"],
            calibration["px_per_cmd_mm"].values,
        )

    def test_readback_disagreement_uses_actual_trajectory_motion(self):
        calibration = build_sample_calibration_dataset(
            image_shape_cam0=(4, 5),
            image_shape_cam1=(6, 7),
        )

        arrays = _calibration_arrays(calibration)

        np.testing.assert_allclose(arrays["readback_disagreement"], 0.0, atol=1e-15)

    def test_calibration_progress_reports_command_offset(self):
        get_qapp()
        panel = CalibrationPanel()

        panel.show_calibration_in_progress(total_steps=3)
        panel.show_calibration_step(
            idx=0,
            total_steps=3,
            dx=0.01,
            dy=0.0,
            dz=-0.01,
            elapsed_s=1.0,
            eta_s=2.0,
        )

        self.assertIn("Command offset", panel.calibration_status_label.text())
        self.assertNotIn("Command delta", panel.calibration_status_label.text())

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
            "x_offset_cmd_mm",
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
                self.assertEqual(
                    window.image_plots["cam0"].getAxis("bottom").labelText,
                    "u",
                )
                self.assertEqual(
                    window.image_plots["cam0"].getAxis("left").labelText,
                    "v",
                )
                self.assertEqual(
                    window.image_plots["cam1"].getAxis("bottom").labelText,
                    "v",
                )
                self.assertEqual(
                    window.image_plots["cam1"].getAxis("left").labelText,
                    "u",
                )
                self.assertFalse(window.image_plots["cam0"].vb.state["xInverted"])
                self.assertTrue(window.image_plots["cam0"].vb.state["yInverted"])
                self.assertTrue(window.image_plots["cam1"].vb.state["xInverted"])
                self.assertTrue(window.image_plots["cam1"].vb.state["yInverted"])

                cam1_width, cam1_height = main_window.CAMERA_IMAGE_SIZES["cam1"]
                assert_rect_close(
                    self,
                    image_parent_rect(window, "cam1"),
                    (0.0, 0.0, float(cam1_height), float(cam1_width)),
                )
            finally:
                window.close()

    def test_shift_monitor_menu_opens_singleton_and_receives_frames(self):
        get_qapp()
        created = []

        class FakeShiftMonitorWindow(QtCore.QObject):
            sigRegistrationConfigSaved = QtCore.Signal(object)

            def __init__(self, settings, *, registration_config=None, parent=None):
                super().__init__(parent)
                self.settings = settings
                self.registration_config = dict(registration_config)
                self.calibration_calls = []
                self.frames = []
                self.show_count = 0
                created.append(self)

            def set_calibration(self, calibration):
                self.calibration_calls.append(calibration)

            def submit_frame(self, camera, image):
                self.frames.append((camera, image))

            def show(self):
                self.show_count += 1

            def raise_(self):
                pass

            def activateWindow(self):
                pass

            def close(self):
                pass

        with (
            patched_main_window_runtime(),
            patch(
                "merlin_track_position.interface.main_window.ShiftMonitorWindow",
                FakeShiftMonitorWindow,
            ),
        ):
            window = MainWindow()
            try:
                self.assertEqual(window.shift_monitor_action.text(), "Shift Monitor")

                window.shift_monitor_action.trigger()
                window.shift_monitor_action.trigger()

                self.assertEqual(len(created), 1)
                self.assertEqual(created[0].show_count, 2)
                self.assertEqual(created[0].calibration_calls, [None])

                image = full_camera_image("cam0", 2.0)
                window._on_image_capture_ready("cam0", image)

                self.assertEqual(created[0].frames, [("cam0", image)])
            finally:
                window.close()

    def test_new_calibration_passes_selected_path_parameters_and_roi(self):
        get_qapp()

        class FakeCalibrationThread:
            def __init__(self):
                self.configured = None
                self.started = False

            def isRunning(self):
                return False

            def configure(
                self,
                camera_pair,
                roi_metadata,
                output_path,
                *,
                n,
                step_um,
                shift_kwargs=None,
            ):
                self.configured = (
                    camera_pair,
                    dict(roi_metadata),
                    Path(output_path),
                    int(n),
                    float(step_um),
                    {} if shift_kwargs is None else dict(shift_kwargs),
                )

            def start(self):
                self.started = True

            def stop(self):
                pass

            def wait(self):
                pass

        output_path = Path("/tmp/scan/calibration.h5")
        with patched_main_window_runtime():
            window = MainWindow()
            fake_thread = FakeCalibrationThread()
            window._calibration_thread = fake_thread
            try:
                window._set_roi_geometry("cam0", (1.0, 2.0, 30.0, 40.0))
                window._set_roi_geometry("cam1", (3.0, 4.0, 50.0, 60.0))
                with (
                    patch.object(
                        main_window,
                        "_default_calibration_path",
                        return_value=output_path,
                    ),
                    patch.object(
                        CalibrationStartDialog,
                        "exec",
                        return_value=QtWidgets.QDialog.DialogCode.Accepted,
                    ),
                    patch.object(
                        CalibrationStartDialog,
                        "parameters",
                        return_value=(3, 10.0),
                    ),
                    patch.object(
                        CalibrationStartDialog,
                        "output_path",
                        return_value=output_path,
                    ),
                ):
                    window._on_new_calibration_clicked()

                self.assertTrue(fake_thread.started)
                self.assertEqual(window._calibration_total_steps, 15)
                (
                    camera_pair,
                    roi_metadata,
                    path,
                    n,
                    step_um,
                    shift_kwargs,
                ) = fake_thread.configured
                self.assertEqual(path, output_path)
                self.assertEqual(n, 3)
                self.assertEqual(step_um, 10.0)
                self.assertEqual(
                    shift_kwargs,
                    registration_config_to_measurement_kwargs(window._registration_config),
                )
                self.assertEqual(roi_metadata["roi_cam0_x"], 1.0)
                self.assertEqual(roi_metadata["roi_cam1_y"], 4.0)
                self.assertIsNotNone(camera_pair)
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
                np.testing.assert_array_equal(
                    window.image_items["cam1"].image,
                    current_cam1,
                )
                image_width, image_height = main_window.CAMERA_IMAGE_SIZES["cam1"]
                assert_rect_close(
                    self,
                    image_parent_rect(window, "cam1"),
                    (0.0, 0.0, float(image_height), float(image_width)),
                )
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
                    (16.0, 14.0, 7.0, 8.0),
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
                image_width, image_height = main_window.CAMERA_IMAGE_SIZES["cam1"]
                assert_rect_close(
                    self,
                    image_parent_rect(window, "cam1"),
                    (0.0, 0.0, float(image_height), float(image_width)),
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

    def test_correction_mode_persists(self):
        get_qapp()
        settings = FakeSettings()
        settings.values["correction/mode"] = "beam"
        with patched_main_window_runtime(settings):
            window = MainWindow()
            try:
                self.assertEqual(window.calibration_panel.correction_mode(), "beam")

                window.calibration_panel.set_correction_mode("camera")
                window._on_correction_mode_changed(
                    window.calibration_panel.correction_mode_combo.currentIndex()
                )

                self.assertEqual(settings.values["correction/mode"], "camera")
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
                    window.calibration_panel.set_correction_mode("beam")
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
                    self.assertEqual(thread.correction_mode, "beam")
                    self.assertEqual(
                        thread.shift_kwargs,
                        registration_config_to_measurement_kwargs(window._registration_config),
                    )
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
                self.assertEqual(
                    window._get_roi_geometry("cam1"),
                    (14.0, 16.0, 34.0, 36.0),
                )
                self.assertEqual(
                    roi_display_geometry(window, "cam1"),
                    (16.0, 14.0, 36.0, 34.0),
                )
                self.assertFalse(roi_handles_visible(window))
                self.assertEqual(settings.set_calls, [])
            finally:
                window.close()

    def test_cam1_roi_change_persists_raw_coordinates(self):
        get_qapp()
        with patched_main_window_runtime() as settings:
            window = MainWindow()
            try:
                was_blocked = window.image_rois["cam1"].blockSignals(True)
                try:
                    window.image_rois["cam1"].setPos((22.0, 11.0), update=False)
                    window.image_rois["cam1"].setSize((44.0, 33.0), update=True)
                finally:
                    window.image_rois["cam1"].blockSignals(was_blocked)
                window._on_roi_region_change_finished("cam1")

                self.assertEqual(
                    window._get_roi_geometry("cam1"),
                    (11.0, 22.0, 33.0, 44.0),
                )
                self.assertEqual(
                    roi_display_geometry(window, "cam1"),
                    (22.0, 11.0, 44.0, 33.0),
                )
                self.assertEqual(
                    settings.set_calls,
                    [
                        ("roi/cam1/x", 11.0),
                        ("roi/cam1/y", 22.0),
                        ("roi/cam1/width", 33.0),
                        ("roi/cam1/height", 44.0),
                    ],
                )
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
                        window.calibration_panel.set_correction_mode("beam")
                        window._on_correct_sample_clicked()

                    thread = window._correction_thread
                    self.assertTrue(thread.started)
                    self.assertIs(thread.calibration, window._calibration)
                    self.assertEqual(thread.calibration_path, path)
                    self.assertEqual(thread.correction_mode, "beam")
                    self.assertEqual(
                        thread.shift_kwargs,
                        registration_config_to_measurement_kwargs(window._registration_config),
                    )
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

    def test_saved_registration_config_updates_correction_kwargs(self):
        get_qapp()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "calibration.h5"
            calibration = write_sample_calibration(path)
            with patched_main_window_runtime():
                window = MainWindow()
                try:
                    window._on_registration_config_saved(
                        {
                            "clip_enabled": True,
                            "clip_low": 20.0,
                            "clip_high": 80.0,
                            "normalization": "none",
                            "upsample_factor": 51,
                            "use_window": True,
                            "use_ecc_refinement": True,
                            "high_error_threshold": 0.25,
                            "capture_count": 7,
                            "capture_aggregation": "mean_image",
                        }
                    )
                    window._on_new_calibration_ready(calibration)

                    window._start_correction()

                    self.assertEqual(
                        window._correction_thread.shift_kwargs,
                        {
                            "clip_percentiles": (20.0, 80.0),
                            "use_window": True,
                            "upsample_factor": 51,
                            "normalization": None,
                            "high_error_threshold": 0.25,
                            "use_ecc_refinement": True,
                            "capture_count": 7,
                            "capture_aggregation": "mean_image",
                        },
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
                    self.assertEqual(
                        thread.shift_kwargs,
                        registration_config_to_measurement_kwargs(window._registration_config),
                    )
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
                    window.calibration_panel.set_correction_mode("beam")

                    window._on_move_detected(7)

                    thread = window._correction_thread
                    self.assertTrue(thread.started)
                    self.assertIs(thread.calibration, window._calibration)
                    self.assertEqual(thread.calibration_path, path)
                    self.assertIs(thread.motor_backend, motor_backend)
                    self.assertEqual(thread.correction_mode, "beam")
                    self.assertEqual(
                        thread.shift_kwargs,
                        registration_config_to_measurement_kwargs(window._registration_config),
                    )
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
        self.assertIsNotNone(
            dialog.findChild(QtWidgets.QSpinBox, "calibration_n_spin")
        )
        self.assertIsNotNone(
            dialog.findChild(QtWidgets.QDoubleSpinBox, "calibration_step_um_spin")
        )
        self.assertEqual(
            dialog.parameters(),
            (5, main_window.DEFAULT_VISUAL_CALIBRATION_STEP_UM),
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

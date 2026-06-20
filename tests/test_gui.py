import logging
import logging.handlers
import os
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import cv2
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
    CameraSettingsDialog,
    CalibrationStartDialog,
    MainWindow,
    _clamp_roi_geometry,
    _default_roi_geometry,
    _roi_geometries_from_calibration_metadata,
    _roi_metadata_from_geometries,
)
from merlin_track_position.instruments.camera_config import (  # noqa: E402
    SOURCE_BASLER,
    SOURCE_FRAMEGRABBER,
    SOURCE_SIMULATED,
    CameraConfig,
    DisplayTransform,
)
from merlin_track_position.instruments.basler import (  # noqa: E402
    BaslerCameraCapabilities,
    BaslerDevice,
    BaslerValueRange,
)
from merlin_track_position.interface.registration_settings import (  # noqa: E402
    REGISTRATION_CAPTURE_AGGREGATION_SETTINGS_KEY,
    REGISTRATION_CAPTURE_COUNT_SETTINGS_KEY,
    REGISTRATION_CLIP_HIGH_SETTINGS_KEY,
    REGISTRATION_CLIP_LOW_SETTINGS_KEY,
    REGISTRATION_ECC_GAUSS_FILTER_SIZE_CAM0_SETTINGS_KEY,
    REGISTRATION_ECC_GAUSS_FILTER_SIZE_CAM1_SETTINGS_KEY,
    REGISTRATION_ECC_USE_WINDOW_SETTINGS_KEY,
    REGISTRATION_ECC_MOTION_MODEL_SETTINGS_KEY,
    REGISTRATION_USE_ECC_REFINEMENT_SETTINGS_KEY,
    registration_config_to_camera_shift_kwargs,
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
    load_calibration_dataset,
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


class ApplicationLoggingTests(unittest.TestCase):
    def setUp(self):
        self.root_logger = logging.getLogger()
        self.original_level = self.root_logger.level
        self._remove_managed_file_handlers()

    def tearDown(self):
        self._remove_managed_file_handlers()
        self.root_logger.setLevel(self.original_level)

    def _managed_file_handlers(self):
        return [
            handler
            for handler in self.root_logger.handlers
            if getattr(handler, main_window._FILE_LOG_HANDLER_MARKER, False)
        ]

    def _remove_managed_file_handlers(self):
        for handler in self._managed_file_handlers():
            self.root_logger.removeHandler(handler)
            handler.close()

    def test_application_file_logger_is_bounded_and_warning_only(self):
        self.root_logger.setLevel(logging.DEBUG)
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "track-position.log"

            configured_path = main_window.configure_application_logging(
                log_path,
                max_bytes=256,
                backup_count=2,
            )
            configured_again = main_window.configure_application_logging(
                log_path,
                max_bytes=256,
                backup_count=2,
            )

            self.assertEqual(configured_path, log_path)
            self.assertEqual(configured_again, log_path)
            handlers = self._managed_file_handlers()
            self.assertEqual(len(handlers), 1)
            handler = handlers[0]
            self.assertIsInstance(handler, logging.handlers.RotatingFileHandler)
            self.assertEqual(Path(handler.baseFilename), log_path)
            self.assertEqual(handler.level, logging.WARNING)
            self.assertEqual(handler.maxBytes, 256)
            self.assertEqual(handler.backupCount, 2)

            test_logger = logging.getLogger("merlin_track_position.tests.logging")
            test_logger.info("quiet progress message")
            test_logger.warning("visible failure warning")
            handler.flush()

            contents = log_path.read_text(encoding="utf-8")
            self.assertNotIn("quiet progress message", contents)
            self.assertIn("visible failure warning", contents)


class FakeImageCaptureThread(QtCore.QObject):
    sigImageReady = QtCore.Signal(str, object)
    sigImageCaptureFailed = QtCore.Signal(str, str)

    def __init__(self, camera, image_capture, interval_ms, parent=None):
        super().__init__(parent)
        self.camera = camera
        self.image_capture = image_capture
        self.interval_ms = interval_ms
        self.enabled = False
        self.wait_until_idle_calls = 0

    def start(self):
        pass

    def set_enabled(self, enabled):
        self.enabled = bool(enabled)

    def stop(self):
        pass

    def wait_until_idle(self):
        self.wait_until_idle_calls += 1

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
        self.active_command_axes = None
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
        active_command_axes=None,
    ):
        self.calibration = calibration
        self.camera_pair = camera_pair
        self.calibration_path = Path(calibration_path)
        self.motor_backend = motor_backend
        self.correction_mode = correction_mode
        self.shift_kwargs = {} if shift_kwargs is None else dict(shift_kwargs)
        self.active_command_axes = active_command_axes

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


class FakeStoredAxisMoveThread(QtCore.QObject):
    sigStoredAxisMoveReady = QtCore.Signal(str, float, float)
    sigStoredAxisMoveFailed = QtCore.Signal(str, str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.axis_alias = None
        self.target_value = None
        self.started = False
        self.running = False

    def configure(self, axis_alias, target_value):
        self.axis_alias = axis_alias
        self.target_value = float(target_value)

    def start(self):
        self.started = True
        self.running = True

    def isRunning(self):
        return self.running

    def stop(self):
        self.running = False

    def wait(self):
        pass


class FakePolarCompensationXzMoveThread(QtCore.QObject):
    sigPolarCompensationXzMoveReady = QtCore.Signal(float, float, float, float)
    sigPolarCompensationXzMoveFailed = QtCore.Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.target_x = None
        self.target_z = None
        self.started = False
        self.running = False

    def configure(self, target_x, target_z):
        self.target_x = float(target_x)
        self.target_z = float(target_z)

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
        patch(
            "merlin_track_position.interface.main_window._StoredAxisMoveThread",
            FakeStoredAxisMoveThread,
        ),
        patch(
            "merlin_track_position.interface.main_window._PolarCompensationXzMoveThread",
            FakePolarCompensationXzMoveThread,
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
    return all(
        roi.resizable and all(handle.isEnabled() for handle in roi.getHandles())
        for roi in window.image_rois.values()
    )


def roi_body_translation_enabled(window):
    return any(roi.translatable for roi in window.image_rois.values())


def beam_target_dragging_enabled(window):
    return all(target.movable for target in window.image_targets.values())


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


def beam_target_point(window, camera):
    return np.asarray(window._beam_target_point(camera), dtype=float)


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
            "initial_readback_position_mm": (
                ("command_axis",),
                np.zeros(len(COMMAND_AXES), dtype=float),
            ),
            "move_final_readback_position_mm": (
                ("move", "command_axis"),
                np.asarray(
                    [
                        [0.0015, -0.002, 0.0],
                        [0.001, -0.002, 0.00325],
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
            "estimated_readback_offset_mm": (
                ("command_axis",),
                np.asarray([0.004, -0.005, 0.006], dtype=float),
            ),
            "correction_readback_delta_mm": (
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
            "estimated_readback_offset_mm": (
                ("command_axis",),
                np.asarray([0.004, -0.005, 0.006], dtype=float),
            ),
            "correction_readback_delta_mm": (
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
            "estimated_readback_offset_mm": (
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

    def test_old_translation_registration_settings_do_not_seed_current_controls(self):
        get_qapp()
        settings = FakeSettings()
        settings.values["registration/normalization"] = "none"
        settings.values["registration/upsample_factor"] = 123
        settings.values["registration/high_error_threshold"] = 0.25
        settings.values["registration/phase_l2_size"] = 123
        settings.values["registration/phase_max_iters"] = 456

        with patched_shift_monitor_worker():
            window = ShiftMonitorWindow(settings)
            try:
                self.assertFalse(hasattr(window, "phase_l2_size_spin"))
                self.assertFalse(hasattr(window, "phase_max_iters_spin"))
                self.assertFalse(window.ecc_window_checkbox.isChecked())
            finally:
                window.close()

    def test_monitor_defaults_to_per_camera_ecc_gauss_filter_sizes(self):
        get_qapp()
        with patched_shift_monitor_worker():
            window = ShiftMonitorWindow(FakeSettings())
            try:
                self.assertEqual(window.ecc_gauss_filter_size_cam0_spin.value(), 1)
                self.assertEqual(window.ecc_gauss_filter_size_cam1_spin.value(), 5)

                kwargs = registration_config_to_shift_kwargs(window._config)
                self.assertEqual(
                    kwargs["ecc_gauss_filter_size"],
                    {"cam0": 1, "cam1": 5},
                )
                self.assertEqual(
                    registration_config_to_camera_shift_kwargs(
                        window._config,
                        "cam0",
                    )["ecc_gauss_filter_size"],
                    1,
                )
                self.assertEqual(
                    registration_config_to_camera_shift_kwargs(
                        window._config,
                        "cam1",
                    )["ecc_gauss_filter_size"],
                    5,
                )
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
        frames = [np.full((2, 2), value, dtype=float) for value in (1.0, 2.0, 3.0, 4.0)]

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
                window.ecc_refinement_checkbox.setChecked(True)
                window.ecc_window_checkbox.setChecked(True)
                window.ecc_motion_model_combo.setCurrentIndex(
                    window.ecc_motion_model_combo.findData("affine")
                )
                window.ecc_gauss_filter_size_cam0_spin.setValue(3)
                window.ecc_gauss_filter_size_cam1_spin.setValue(9)
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
                    settings.values[REGISTRATION_USE_ECC_REFINEMENT_SETTINGS_KEY],
                    True,
                )
                self.assertEqual(
                    settings.values[REGISTRATION_ECC_USE_WINDOW_SETTINGS_KEY],
                    True,
                )
                self.assertEqual(
                    settings.values[REGISTRATION_ECC_MOTION_MODEL_SETTINGS_KEY],
                    "affine",
                )
                self.assertEqual(
                    settings.values[
                        REGISTRATION_ECC_GAUSS_FILTER_SIZE_CAM0_SETTINGS_KEY
                    ],
                    3,
                )
                self.assertEqual(
                    settings.values[
                        REGISTRATION_ECC_GAUSS_FILTER_SIZE_CAM1_SETTINGS_KEY
                    ],
                    9,
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
                    registration_config_to_shift_kwargs(saved_configs[-1])[
                        "ecc_use_window"
                    ],
                    True,
                )
                self.assertEqual(
                    registration_config_to_shift_kwargs(saved_configs[-1])[
                        "ecc_motion_model"
                    ],
                    "affine",
                )
                self.assertEqual(
                    registration_config_to_shift_kwargs(saved_configs[-1])[
                        "ecc_gauss_filter_size"
                    ],
                    {"cam0": 3, "cam1": 9},
                )
                self.assertEqual(
                    registration_config_to_camera_shift_kwargs(
                        saved_configs[-1],
                        "cam0",
                    )["ecc_gauss_filter_size"],
                    3,
                )
                self.assertEqual(
                    registration_config_to_camera_shift_kwargs(
                        saved_configs[-1],
                        "cam1",
                    )["ecc_gauss_filter_size"],
                    9,
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
                window.ecc_refinement_checkbox.setChecked(True)
                window.ecc_window_checkbox.setChecked(True)
                window.ecc_motion_model_combo.setCurrentIndex(
                    window.ecc_motion_model_combo.findData("affine")
                )
                window.ecc_gauss_filter_size_cam0_spin.setValue(3)
                window.ecc_gauss_filter_size_cam1_spin.setValue(9)
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
                    self.assertNotIn("registration_phase_l2_size", exported.attrs)
                    self.assertNotIn("registration_phase_max_iters", exported.attrs)
                    self.assertEqual(
                        exported.attrs["registration_use_ecc_refinement"],
                        True,
                    )
                    self.assertEqual(
                        exported.attrs["registration_ecc_use_window"],
                        True,
                    )
                    self.assertEqual(
                        exported.attrs["registration_ecc_motion_model"],
                        "affine",
                    )
                    self.assertEqual(
                        exported.attrs["registration_ecc_gauss_filter_size_cam0"],
                        3,
                    )
                    self.assertEqual(
                        exported.attrs["registration_ecc_gauss_filter_size_cam1"],
                        9,
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
    def test_summary_reports_px_per_readback_mm_metrics(self):
        calibration = build_sample_calibration_dataset(
            image_shape_cam0=(4, 5),
            image_shape_cam1=(6, 7),
        )

        summary = _calibration_summary(calibration)

        self.assertEqual(summary["probe_count"], calibration.sizes["probe"])
        self.assertLess(summary["condition_number"], 100.0)
        np.testing.assert_allclose(
            summary["axis_scale_readback_mm"],
            calibration["axis_scale_readback_mm"].values,
        )
        (
            _derived_axis_scale,
            expected_axis_sensitivity,
            _axis_scale_unclamped,
            _axis_scale_bounds,
            _target_response,
        ) = derive_axis_scale_from_jacobian(
            calibration["px_per_readback_mm"].values,
            calibration["probe_readback_delta_mm"].values,
        )
        np.testing.assert_allclose(
            summary["axis_sensitivity_px_per_readback_mm"],
            expected_axis_sensitivity,
        )
        self.assertEqual(summary["residual_rms_px"], 0.0)
        self.assertEqual(summary["residual_max_readback_mm"], 0.0)
        self.assertEqual(summary["readback_command_rms_mm"], 0.0)
        np.testing.assert_allclose(
            summary["px_per_readback_mm"],
            calibration["px_per_readback_mm"].values,
        )

    def test_readback_disagreement_uses_actual_trajectory_motion(self):
        calibration = build_sample_calibration_dataset(
            image_shape_cam0=(4, 5),
            image_shape_cam1=(6, 7),
        )

        arrays = _calibration_arrays(calibration)

        np.testing.assert_allclose(arrays["readback_disagreement"], 0.0, atol=1e-15)

    def test_calibration_progress_reports_requested_offset(self):
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

        self.assertIn("Requested offset", panel.calibration_status_label.text())
        self.assertNotIn("Command delta", panel.calibration_status_label.text())

    def test_panel_loads_dataset_and_builds_details_dialog(self):
        get_qapp()
        calibration = build_sample_calibration_dataset(
            image_shape_cam0=(4, 5),
            image_shape_cam1=(6, 7),
        ).assign_attrs({"warnings": "calibration warning"})
        panel = CalibrationPanel()

        panel.show_loaded_calibration(calibration, "test.h5")
        dialog = panel.build_details_dialog(calibration)
        tabs = dialog.findChild(QtWidgets.QTabWidget, "calibration_details_tabs")
        summary_table = dialog.findChild(
            QtWidgets.QTableWidget,
            "calibration_details_summary_table",
        )
        warnings_text = dialog.findChild(
            QtWidgets.QPlainTextEdit,
            "calibration_details_warnings_text",
        )
        residuals_layout = dialog.findChild(
            QtWidgets.QWidget,
            "calibration_details_residuals_layout",
        )
        table = dialog.findChild(QtWidgets.QTableWidget, "calibration_samples_table")
        axes_table = dialog.findChild(QtWidgets.QTableWidget, "calibration_axes_table")

        self.assertEqual(
            panel.calibration_file_label.text(), "Calibration file: test.h5"
        )
        self.assertIn("test.h5", panel.calibration_status_label.text())
        self.assertTrue(panel.correct_sample_button.isEnabled())
        self.assertTrue(panel.detect_shift_button.isEnabled())
        self.assertTrue(panel.calculate_polar_compensate_button.isEnabled())
        self.assertFalse(panel.calibration_review_widget.isHidden())
        self.assertTrue(panel.correction_steps_group.isHidden())
        self.assertNotEqual(panel.metric_labels["axis_scale_readback_mm"].text(), "n/a")
        self.assertEqual(tabs.tabText(0), "Summary")
        self.assertEqual(tabs.tabText(1), "Residuals")
        self.assertEqual(tabs.tabText(2), "Matrices")
        self.assertEqual(tabs.tabText(3), "Axes")
        self.assertEqual(tabs.tabText(4), "Probes")
        self.assertEqual(tabs.count(), 5)
        self.assertEqual(summary_table.rowCount(), 11)
        self.assertIn("calibration warning", warnings_text.toPlainText())
        self.assertIsNotNone(residuals_layout)
        self.assertEqual(axes_table.rowCount(), len(COMMAND_AXES))
        self.assertEqual(table.rowCount(), calibration.sizes["probe"])
        self.assertIn(
            "x_offset_cmd_mm",
            [table.horizontalHeaderItem(i).text() for i in range(table.columnCount())],
        )

        panel.reset()
        self.assertEqual(panel.calibration_file_label.text(), "Calibration file: none")
        self.assertFalse(panel.correct_sample_button.isEnabled())
        self.assertFalse(panel.detect_shift_button.isEnabled())
        self.assertFalse(panel.calculate_polar_compensate_button.isEnabled())

        panel.show_loaded_calibration(calibration, "test.h5")
        panel.show_correction_in_progress()
        self.assertEqual(
            panel.calibration_file_label.text(), "Calibration file: test.h5"
        )
        self.assertIn("Correction in progress", panel.calibration_status_label.text())
        self.assertFalse(panel.load_calibration_button.isEnabled())
        self.assertFalse(panel.save_calibration_button.isEnabled())
        self.assertFalse(panel.calibration_details_button.isEnabled())
        self.assertFalse(panel.calculate_polar_compensate_button.isEnabled())
        self.assertFalse(panel.correct_sample_button.isEnabled())
        self.assertFalse(panel.detect_shift_button.isEnabled())
        self.assertFalse(panel.new_calibration_button.isEnabled())
        self.assertTrue(panel.calibration_review_widget.isHidden())
        self.assertTrue(panel.metrics_group.isHidden())
        self.assertTrue(panel.warnings_group.isHidden())
        self.assertTrue(panel.residual_graphics_layout.isHidden())
        self.assertFalse(panel.correction_steps_group.isHidden())
        self.assertGreater(panel.correction_steps_table.maximumHeight(), 180)
        self.assertEqual(panel.correction_steps_table.rowCount(), 0)
        self.assertIn(
            "initial correction measurement",
            panel.correction_steps_summary_label.text(),
        )

    def test_stored_orientation_row_shows_calibration_attrs(self):
        get_qapp()
        calibration = build_sample_calibration_dataset().assign_attrs(
            polar=12.34567,
            tilt=-3.2,
            azi=47.5,
        )
        panel = CalibrationPanel()
        try:
            panel.show_loaded_calibration(calibration, "test.h5")

            layout = panel.layout()
            self.assertIs(layout.itemAt(1).widget(), panel.stored_orientation_widget)
            self.assertIs(layout.itemAt(2).widget(), panel.calibration_file_label)
            self.assertIs(layout.itemAt(3).widget(), panel.calibration_status_label)
            self.assertFalse(panel.stored_orientation_widget.isHidden())
            self.assertEqual(
                panel.stored_orientation_prefix_label.text(), "Calibrated at"
            )
            self.assertEqual(
                panel.stored_orientation_value_labels["polar"].text(),
                "12.3457",
            )
            self.assertEqual(
                panel.stored_orientation_value_labels["tilt"].text(),
                "-3.2000",
            )
            self.assertEqual(
                panel.stored_orientation_value_labels["azi"].text(),
                "47.5000",
            )
            self.assertEqual(
                panel.stored_orientation_axis_widgets["polar"].layout().spacing(),
                6,
            )
            self.assertEqual(
                panel.stored_orientation_axis_widgets["polar"]
                .parentWidget()
                .layout()
                .spacing(),
                24,
            )

            requested = []
            panel.sigStoredAxisMoveRequested.connect(
                lambda axis, value: requested.append((axis, value))
            )
            panel.stored_orientation_go_buttons["polar"].click()

            self.assertEqual(len(requested), 1)
            self.assertEqual(requested[0][0], "p")
            self.assertAlmostEqual(requested[0][1], 12.34567)
        finally:
            panel.close()

    def test_stored_orientation_row_skips_missing_axes(self):
        get_qapp()
        calibration = build_sample_calibration_dataset().assign_attrs(
            polar=1.25,
            tilt=-2.5,
        )
        calibration.attrs.pop("azi", None)
        panel = CalibrationPanel()
        try:
            panel.show_loaded_calibration(calibration, "test.h5")

            self.assertFalse(panel.stored_orientation_widget.isHidden())
            self.assertFalse(panel.stored_orientation_axis_widgets["polar"].isHidden())
            self.assertFalse(panel.stored_orientation_axis_widgets["tilt"].isHidden())
            self.assertTrue(panel.stored_orientation_axis_widgets["azi"].isHidden())
        finally:
            panel.close()

    def test_stored_orientation_row_hides_without_finite_axes(self):
        get_qapp()
        calibration = build_sample_calibration_dataset()
        for key in ("polar", "tilt", "azi"):
            calibration.attrs.pop(key, None)
        panel = CalibrationPanel()
        try:
            panel.show_loaded_calibration(calibration, "test.h5")

            self.assertTrue(panel.stored_orientation_widget.isHidden())
        finally:
            panel.close()

    def test_calibration_review_layout_groups_diagnostics(self):
        get_qapp()
        panel = CalibrationPanel()

        review_widget = panel.layout().itemAt(5).widget()
        content_layout = review_widget.layout().itemAt(0).layout()
        left_column = content_layout.itemAt(0).layout()
        right_column = content_layout.itemAt(1).layout()

        self.assertIs(left_column.itemAt(0).widget(), panel.metrics_group)
        self.assertIs(left_column.itemAt(1).widget(), panel.repeatability_group)
        self.assertIs(right_column.itemAt(0).widget(), panel.warnings_group)
        self.assertIs(panel.layout().itemAt(6).widget(), panel.correction_steps_group)

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
            "Estimated readback offset: x=4 um, y=-5 um, z=6 um.",
            summary,
        )
        self.assertIn("Next correction: x=1.5 um, y=-2 um, z=0 um.", summary)

    def test_correction_result_displays_move_steps_in_microns(self):
        get_qapp()
        panel = CalibrationPanel()
        calibration = build_sample_calibration_dataset(
            image_shape_cam0=(4, 5),
            image_shape_cam1=(6, 7),
        )
        result = correction_result_with_moves()

        panel.show_loaded_calibration(calibration, "test.h5")
        panel.show_correction_result(result)

        table = panel.correction_steps_table
        self.assertEqual(
            panel.calibration_file_label.text(), "Calibration file: test.h5"
        )
        self.assertTrue(panel.calibration_review_widget.isHidden())
        self.assertTrue(panel.metrics_group.isHidden())
        self.assertTrue(panel.warnings_group.isHidden())
        self.assertTrue(panel.residual_graphics_layout.isHidden())
        self.assertEqual(panel.calibration_warnings_text.toPlainText(), "")
        self.assertEqual(
            panel.correction_warnings_text.toPlainText(),
            "No correction warnings.",
        )
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

    def test_calibration_file_label_persists_across_operation_statuses(self):
        get_qapp()
        panel = CalibrationPanel()
        calibration = build_sample_calibration_dataset(
            image_shape_cam0=(4, 5),
            image_shape_cam1=(6, 7),
        )
        correction = correction_result_with_moves()
        detection = detection_result()
        try:
            panel.show_loaded_calibration(calibration, "test.h5")

            panel.show_correction_in_progress()
            self.assertEqual(
                panel.calibration_file_label.text(),
                "Calibration file: test.h5",
            )

            panel.show_correction_progress(correction)
            self.assertEqual(
                panel.calibration_file_label.text(),
                "Calibration file: test.h5",
            )

            panel.show_correction_result(correction)
            self.assertEqual(
                panel.calibration_file_label.text(),
                "Calibration file: test.h5",
            )

            panel.show_detection_result(detection)
            self.assertEqual(
                panel.calibration_file_label.text(),
                "Calibration file: test.h5",
            )

            panel.show_stored_axis_move_in_progress("Polar", 12.34567)
            self.assertEqual(
                panel.calibration_file_label.text(),
                "Calibration file: test.h5",
            )

            panel.show_stored_axis_move_result("Polar", 12.34567, 12.34)
            self.assertEqual(
                panel.calibration_file_label.text(),
                "Calibration file: test.h5",
            )
        finally:
            panel.close()

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
    def test_polar_compensation_probe_angles_are_sorted_and_unique(self):
        self.assertEqual(
            main_window._polar_compensation_probe_angles(0.0),
            (-20.0, -12.5, -5.0, 0.0),
        )
        self.assertEqual(
            main_window._polar_compensation_probe_angles(-15.0),
            (-20.0, -15.0, -12.5, -5.0),
        )
        self.assertEqual(
            main_window._polar_compensation_probe_angles(-5.0),
            (-20.0, -12.5, -10.0, -5.0),
        )

    def test_polar_compensation_finishes_in_place_when_anchor_is_last_probe(self):
        get_qapp()
        with patched_main_window_runtime():
            window = MainWindow()
            try:
                window._polar_compensation_active = True
                window._polar_compensation_angles = (-20.0, -12.5, -5.0, 0.0)
                window._polar_compensation_index = len(
                    window._polar_compensation_angles
                )
                window._polar_compensation_current_polar = 0.0
                window._polar_compensation_anchor_polar = 0.0
                with patch.object(window, "_finish_polar_compensation") as finish:
                    window._start_polar_compensation_polar_move()

                    finish.assert_called_once()
                    self.assertFalse(window._stored_axis_move_thread.started)
                    self.assertFalse(window._polar_compensation_xz_move_thread.started)
            finally:
                window.close()

    def test_polar_compensation_returns_to_anchor_and_xz_when_anchor_is_not_last(self):
        get_qapp()
        with patched_main_window_runtime():
            window = MainWindow()
            try:
                window._polar_compensation_active = True
                window._polar_compensation_angles = (
                    -20.0,
                    -12.5,
                    -10.0,
                    -5.0,
                )
                window._polar_compensation_index = len(
                    window._polar_compensation_angles
                )
                window._polar_compensation_current_polar = -5.0
                window._polar_compensation_anchor_polar = -12.5
                window._polar_compensation_points = [
                    main_window._PolarCompensationPoint(
                        polar_deg=-20.0,
                        x_mm=0.0,
                        y_mm=0.0,
                        z_mm=0.0,
                    ),
                    main_window._PolarCompensationPoint(
                        polar_deg=-12.5,
                        x_mm=1.25,
                        y_mm=0.5,
                        z_mm=3.5,
                    ),
                    main_window._PolarCompensationPoint(
                        polar_deg=-10.0,
                        x_mm=2.0,
                        y_mm=0.0,
                        z_mm=4.0,
                    ),
                    main_window._PolarCompensationPoint(
                        polar_deg=-5.0,
                        x_mm=3.0,
                        y_mm=0.0,
                        z_mm=5.0,
                    ),
                ]
                with (
                    patch.object(window, "_finish_polar_compensation") as finish,
                    patch(
                        "merlin_track_position.interface.main_window.QtWidgets.QMessageBox.information"
                    ) as information,
                ):
                    window._start_polar_compensation_polar_move()

                    thread = window._stored_axis_move_thread
                    self.assertTrue(thread.started)
                    self.assertEqual(thread.axis_alias, "p")
                    self.assertEqual(thread.target_value, -12.5)
                    self.assertTrue(window._polar_compensation_returning_to_anchor)
                    information.assert_not_called()

                    thread.running = False
                    thread.sigStoredAxisMoveReady.emit("p", -12.5, -12.5)

                    xz_thread = window._polar_compensation_xz_move_thread
                    self.assertTrue(xz_thread.started)
                    self.assertEqual(xz_thread.target_x, 1.25)
                    self.assertEqual(xz_thread.target_z, 3.5)
                    finish.assert_not_called()

                    xz_thread.running = False
                    xz_thread.sigPolarCompensationXzMoveReady.emit(
                        1.25,
                        3.5,
                        1.25,
                        3.5,
                    )

                    finish.assert_called_once()
                    self.assertFalse(window._polar_compensation_returning_to_anchor)
                    self.assertFalse(window._polar_compensation_returning_xz)
            finally:
                window.close()

    def test_camera_display_transform_comes_from_settings(self):
        get_qapp()
        settings = FakeSettings()
        settings.values.update(
            {
                "camera/cam0/display_transpose": True,
                "camera/cam0/display_invert_x": True,
                "camera/cam0/display_invert_y": False,
            }
        )
        with patched_main_window_runtime(settings):
            window = MainWindow()
            try:
                self.assertEqual(
                    main_window._display_geometry("cam0", (1.0, 2.0, 3.0, 4.0)),
                    (2.0, 1.0, 4.0, 3.0),
                )
                view_state = window.image_plots["cam0"].vb.state
                self.assertTrue(view_state["xInverted"])
                self.assertFalse(view_state["yInverted"])
            finally:
                window.close()

    def test_bayer_basler_image_is_demosaiced_for_display(self):
        get_qapp()
        settings = FakeSettings()
        settings.values.update(
            {
                "camera/cam1/source_type": SOURCE_BASLER,
                "camera/cam1/width": 4,
                "camera/cam1/height": 4,
                "camera/cam1/pixel_format": "BayerRG12",
            }
        )
        raw = np.empty((4, 4), dtype=np.uint16)
        raw[0::2, 0::2] = 4000
        raw[0::2, 1::2] = 1000
        raw[1::2, 0::2] = 1000
        raw[1::2, 1::2] = 100
        with patched_main_window_runtime(settings):
            window = MainWindow()
            try:
                window._on_image_capture_ready("cam1", raw)

                np.testing.assert_array_equal(
                    window.image_items["cam1"].image,
                    cv2.cvtColor(raw, cv2.COLOR_BayerRGGB2RGB),
                )
                self.assertGreater(
                    float(window.image_items["cam1"].image[..., 0].mean()),
                    float(window.image_items["cam1"].image[..., 2].mean()),
                )
                np.testing.assert_array_equal(
                    window._latest_images_by_camera["cam1"],
                    raw,
                )
            finally:
                window.close()

    def test_camera_settings_dialog_returns_updated_configs(self):
        get_qapp()
        device_old = BaslerDevice("old", "old-model", "old-full-name")
        device_new = BaslerDevice("new-serial", "new-model", "new-full-name")
        capabilities = {
            "old": BaslerCameraCapabilities(
                device_old,
                width=BaslerValueRange(1, 10, 1),
                height=BaslerValueRange(1, 10, 1),
                offset_x=BaslerValueRange(0, 10, 1),
                offset_y=BaslerValueRange(0, 10, 1),
                exposure_us=BaslerValueRange(1.0, 1000000.0, 1.0),
                pixel_formats=("Mono12",),
            ),
            "new-serial": BaslerCameraCapabilities(
                device_new,
                width=BaslerValueRange(2, 12, 2),
                height=BaslerValueRange(1, 10, 1),
                offset_x=BaslerValueRange(0, 10, 1),
                offset_y=BaslerValueRange(0, 10, 1),
                exposure_us=BaslerValueRange(1.0, 1000000.0, 1.0),
                pixel_formats=("Mono12", "BayerRG12"),
            ),
        }
        configs = {
            "cam0": CameraConfig(
                slot="cam0",
                source_type=SOURCE_FRAMEGRABBER,
                width=4,
                height=3,
            ),
            "cam1": CameraConfig(
                slot="cam1",
                source_type=SOURCE_BASLER,
                serial_number="old",
                width=6,
                height=5,
                display=DisplayTransform(transpose=True, invert_x=True, invert_y=True),
            ),
        }
        with (
            patch(
                "merlin_track_position.interface.main_window.list_basler_devices",
                return_value=(device_old, device_new),
            ),
            patch(
                "merlin_track_position.interface.main_window.read_basler_capabilities",
                side_effect=lambda serial: capabilities[serial],
            ),
        ):
            dialog = CameraSettingsDialog(configs)
            try:
                rows = dialog._rows["cam1"]
                rows["serial_number"].setCurrentIndex(
                    rows["serial_number"].findData("new-serial")
                )
                self.assertEqual(rows["pixel_format"].currentData(), "BayerRG12")
                rows["width"].setValue(10)
                rows["pixel_format"].setCurrentIndex(
                    rows["pixel_format"].findData("BayerRG12")
                )
                rows["gamma"].setValue(1.2)
                rows["display_transpose"].setChecked(False)

                updated = dialog.configs()["cam1"]
            finally:
                dialog.close()

        self.assertEqual(updated.serial_number, "new-serial")
        self.assertEqual(updated.model_name, "new-model")
        self.assertEqual(updated.width, 10)
        self.assertEqual(updated.pixel_format, "BayerRG12")
        self.assertEqual(updated.gamma, 1.2)
        self.assertFalse(updated.display.transpose)

    def test_camera_settings_dialog_framegrabber_geometry_is_editable_after_basler(
        self,
    ):
        get_qapp()
        device = BaslerDevice("old", "old-model", "old-full-name")
        capabilities = BaslerCameraCapabilities(
            device,
            width=BaslerValueRange(1, 12, 1),
            height=BaslerValueRange(1, 12, 1),
            offset_x=BaslerValueRange(0, 12, 1),
            offset_y=BaslerValueRange(0, 12, 1),
            exposure_us=BaslerValueRange(1.0, 1000000.0, 1.0),
            pixel_formats=("Mono12",),
        )
        configs = {
            "cam0": CameraConfig(
                slot="cam0",
                source_type=SOURCE_BASLER,
                serial_number="old",
                width=6,
                height=5,
            ),
            "cam1": CameraConfig(
                slot="cam1",
                source_type=SOURCE_FRAMEGRABBER,
                width=6,
                height=5,
            ),
        }
        with (
            patch(
                "merlin_track_position.interface.main_window.list_basler_devices",
                return_value=(device,),
            ),
            patch(
                "merlin_track_position.interface.main_window.read_basler_capabilities",
                return_value=capabilities,
            ),
        ):
            dialog = CameraSettingsDialog(configs)
            try:
                rows = dialog._rows["cam0"]
                rows["source_type"].setCurrentIndex(
                    rows["source_type"].findData(SOURCE_FRAMEGRABBER)
                )
                rows["width"].setValue(9000)
                rows["height"].setValue(8000)
                rows["offset_x"].setValue(123)
                rows["offset_y"].setValue(456)

                updated = dialog.configs()["cam0"]
                self.assertTrue(rows["offset_x"].isEnabled())
                self.assertNotIn("serial_number", rows)
                self.assertNotIn("pixel_format", rows)
            finally:
                dialog.close()

        self.assertEqual(updated.source_type, SOURCE_FRAMEGRABBER)
        self.assertEqual(updated.width, 9000)
        self.assertEqual(updated.height, 8000)
        self.assertEqual(updated.offset_x, 123)
        self.assertEqual(updated.offset_y, 456)

    def test_camera_settings_dialog_preserves_source_drafts_when_switching(self):
        get_qapp()
        device = BaslerDevice("basler-1", "model-1", "full-name-1")
        capabilities = BaslerCameraCapabilities(
            device,
            width=BaslerValueRange(2, 12, 2),
            height=BaslerValueRange(2, 12, 2),
            offset_x=BaslerValueRange(0, 12, 2),
            offset_y=BaslerValueRange(0, 12, 2),
            exposure_us=BaslerValueRange(1.0, 1000000.0, 1.0),
            pixel_formats=("Mono12", "BayerRG12"),
        )
        configs = {
            "cam0": CameraConfig(
                slot="cam0",
                source_type=SOURCE_FRAMEGRABBER,
                width=4,
                height=3,
            ),
            "cam1": CameraConfig(
                slot="cam1",
                source_type=SOURCE_SIMULATED,
                width=6,
                height=5,
            ),
        }
        with (
            patch(
                "merlin_track_position.interface.main_window.list_basler_devices",
                return_value=(device,),
            ),
            patch(
                "merlin_track_position.interface.main_window.read_basler_capabilities",
                return_value=capabilities,
            ),
        ):
            dialog = CameraSettingsDialog(configs)
            try:
                rows = dialog._rows["cam0"]
                rows["width"].setValue(9000)
                rows["display_transpose"].setChecked(True)

                rows["source_type"].setCurrentIndex(
                    rows["source_type"].findData(SOURCE_BASLER)
                )
                rows["serial_number"].setCurrentIndex(
                    rows["serial_number"].findData("basler-1")
                )
                rows["width"].setValue(10)
                rows["pixel_format"].setCurrentIndex(
                    rows["pixel_format"].findData("BayerRG12")
                )

                rows["source_type"].setCurrentIndex(
                    rows["source_type"].findData(SOURCE_FRAMEGRABBER)
                )
                framegrabber_config = dialog.configs()["cam0"]
                self.assertEqual(rows["width"].value(), 9000)
                self.assertTrue(rows["display_transpose"].isChecked())
                self.assertNotIn("pixel_format", rows)

                rows["source_type"].setCurrentIndex(
                    rows["source_type"].findData(SOURCE_BASLER)
                )
                basler_config = dialog.configs()["cam0"]
            finally:
                dialog.close()

        self.assertEqual(framegrabber_config.source_type, SOURCE_FRAMEGRABBER)
        self.assertEqual(framegrabber_config.width, 9000)
        self.assertTrue(framegrabber_config.display.transpose)
        self.assertEqual(basler_config.source_type, SOURCE_BASLER)
        self.assertEqual(basler_config.serial_number, "basler-1")
        self.assertEqual(basler_config.width, 10)
        self.assertEqual(basler_config.pixel_format, "BayerRG12")

    def test_camera_settings_dialog_blocks_basler_without_connected_device(self):
        get_qapp()
        configs = {
            "cam0": CameraConfig(
                slot="cam0",
                source_type=SOURCE_FRAMEGRABBER,
                width=4,
                height=3,
            ),
            "cam1": CameraConfig(
                slot="cam1",
                source_type=SOURCE_BASLER,
                serial_number="missing",
                width=6,
                height=5,
            ),
        }
        with (
            patch(
                "merlin_track_position.interface.main_window.list_basler_devices",
                return_value=(),
            ),
            patch.object(QtWidgets.QMessageBox, "warning") as warning,
        ):
            dialog = CameraSettingsDialog(configs)
            try:
                rows = dialog._rows["cam1"]
                self.assertIn("serial_number", rows)
                self.assertIn("source_message", rows)
                self.assertNotIn("width", rows)
                self.assertNotIn("pixel_format", rows)
                dialog.accept()
            finally:
                dialog.close()

        self.assertTrue(warning.called)
        self.assertIn("requires a connected camera", warning.call_args.args[2])

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
                self.assertFalse(window.initial_transform_preview_checkbox.isEnabled())
                self.assertEqual(
                    window.initial_transform_preview_checkbox.text(),
                    "Transform",
                )
                self.assertFalse(window.reset_beam_target_button.isEnabled())
                self.assertIs(
                    window.image_controls_layout.itemAt(1).widget(),
                    window.show_reference_images_button,
                )
                self.assertIs(
                    window.image_controls_layout.itemAt(2).widget(),
                    window.initial_transform_preview_checkbox,
                )
                self.assertIs(
                    window.image_controls_layout.itemAt(3).widget(),
                    window.reset_beam_target_button,
                )
                self.assertTrue(roi_handles_visible(window))
                self.assertTrue(roi_editing_enabled(window))
                self.assertFalse(roi_body_translation_enabled(window))
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

    def test_beam_targets_initialize_at_roi_centers(self):
        get_qapp()
        with patched_main_window_runtime():
            window = MainWindow()
            try:
                for camera in main_window.CAMERA_IMAGE_SIZES:
                    np.testing.assert_allclose(
                        beam_target_point(window, camera),
                        main_window._roi_center_point(
                            camera,
                            window._get_roi_geometry(camera),
                        ),
                    )
            finally:
                window.close()

    def test_beam_targets_initialize_from_settings(self):
        get_qapp()
        settings = FakeSettings()
        stored_points = {}
        for camera, image_size in main_window.CAMERA_IMAGE_SIZES.items():
            point = np.asarray(
                main_window._roi_center_point(
                    camera,
                    main_window._default_roi_geometry(*image_size),
                ),
                dtype=float,
            ) + np.asarray([3.0, -2.0], dtype=float)
            stored_points[camera] = point
            for key, value in zip(
                main_window.BEAM_TARGET_SETTINGS_KEYS[camera],
                point,
                strict=True,
            ):
                settings.values[key] = float(value)

        with patched_main_window_runtime(settings):
            window = MainWindow()
            try:
                for camera, point in stored_points.items():
                    np.testing.assert_allclose(beam_target_point(window, camera), point)
                    self.assertIn(camera, window._beam_target_user_overrides)
            finally:
                window.close()

    def test_beam_target_drag_finished_persists_raw_coordinates(self):
        get_qapp()
        with patched_main_window_runtime() as settings:
            window = MainWindow()
            try:
                camera = "cam1"
                dragged_point = np.asarray(
                    main_window._roi_center_point(
                        camera,
                        window._get_roi_geometry(camera),
                    ),
                    dtype=float,
                ) + np.asarray([4.0, -3.0], dtype=float)

                window.image_targets[camera].setPos(
                    *main_window._display_point(camera, dragged_point)
                )
                settings.set_calls.clear()
                window._on_beam_target_position_change_finished(camera)

                self.assertEqual(
                    settings.set_calls,
                    [
                        (
                            main_window.BEAM_TARGET_SETTINGS_KEYS[camera][0],
                            float(dragged_point[0]),
                        ),
                        (
                            main_window.BEAM_TARGET_SETTINGS_KEYS[camera][1],
                            float(dragged_point[1]),
                        ),
                    ],
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
                self.beam_target_calls = []
                self.frames = []
                self.show_count = 0
                created.append(self)

            def set_calibration(self, calibration):
                self.calibration_calls.append(calibration)

            def set_beam_target_points(self, points):
                self.beam_target_calls.append(dict(points))

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
                    registration_config_to_measurement_kwargs(
                        window._registration_config
                    ),
                )
                self.assertEqual(roi_metadata["roi_cam0_x"], 1.0)
                self.assertEqual(roi_metadata["roi_cam1_y"], 4.0)
                np.testing.assert_allclose(
                    [
                        roi_metadata["beam_target_cam0_u"],
                        roi_metadata["beam_target_cam0_v"],
                    ],
                    beam_target_point(window, "cam0"),
                )
                np.testing.assert_allclose(
                    [
                        roi_metadata["beam_target_cam1_u"],
                        roi_metadata["beam_target_cam1_v"],
                    ],
                    beam_target_point(window, "cam1"),
                )
                self.assertIsNotNone(camera_pair)
            finally:
                window.close()

    def test_loaded_calibration_initializes_beam_targets_from_metadata(self):
        get_qapp()
        calibration = build_sample_calibration_dataset(
            image_shape_cam0=(4, 5),
            image_shape_cam1=(6, 7),
        ).assign_attrs({"calibration_path": "/tmp/calibration.h5"})

        with patched_main_window_runtime():
            window = MainWindow()
            try:
                cam0_point = main_window._roi_center_point(
                    "cam0",
                    window._get_roi_geometry("cam0"),
                )
                cam1_point = main_window._roi_center_point(
                    "cam1",
                    window._get_roi_geometry("cam1"),
                )
                calibration = calibration.assign_attrs(
                    {
                        "beam_target_cam0_u": cam0_point[0] + 3.0,
                        "beam_target_cam0_v": cam0_point[1] - 2.0,
                        "beam_target_cam1_u": cam1_point[0] - 4.0,
                        "beam_target_cam1_v": cam1_point[1] + 5.0,
                    }
                )

                window._on_new_calibration_ready(calibration)

                self.assertFalse(window.reset_beam_target_button.isEnabled())
                np.testing.assert_allclose(
                    beam_target_point(window, "cam0"),
                    [cam0_point[0] + 3.0, cam0_point[1] - 2.0],
                )
                np.testing.assert_allclose(
                    beam_target_point(window, "cam1"),
                    [cam1_point[0] - 4.0, cam1_point[1] + 5.0],
                )

                dragged_cam0 = (cam0_point[0] + 8.0, cam0_point[1] + 6.0)
                window.image_targets["cam0"].setPos(
                    *main_window._display_point("cam0", dragged_cam0)
                )
                window._on_beam_target_position_changed("cam0")

                self.assertTrue(window.reset_beam_target_button.isEnabled())

                window.reset_beam_target_button.click()

                self.assertFalse(window.reset_beam_target_button.isEnabled())
                self.assertNotIn("cam0", window._beam_target_user_overrides)
                np.testing.assert_allclose(
                    beam_target_point(window, "cam0"),
                    [cam0_point[0] + 3.0, cam0_point[1] - 2.0],
                )
            finally:
                window.close()

    def test_dragged_beam_target_does_not_overwrite_saved_calibration_metadata(self):
        get_qapp()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "calibration.h5"
            cam0_target = main_window._roi_center_point(
                "cam0",
                main_window._default_roi_geometry(
                    *main_window.CAMERA_IMAGE_SIZES["cam0"]
                ),
            )
            cam1_target = main_window._roi_center_point(
                "cam1",
                main_window._default_roi_geometry(
                    *main_window.CAMERA_IMAGE_SIZES["cam1"]
                ),
            )
            calibration = build_sample_calibration_dataset(
                image_shape_cam0=(4, 5),
                image_shape_cam1=(6, 7),
            ).assign_attrs(
                {
                    "calibration_path": str(path),
                    "beam_target_cam0_u": cam0_target[0],
                    "beam_target_cam0_v": cam0_target[1],
                    "beam_target_cam1_u": cam1_target[0],
                    "beam_target_cam1_v": cam1_target[1],
                }
            )
            save_calibration_dataset(calibration, path)

            with patched_main_window_runtime():
                window = MainWindow()
                try:
                    window._on_new_calibration_ready(calibration)
                    dragged_cam0 = (cam0_target[0] + 10.0, cam0_target[1] - 10.0)
                    window.image_targets["cam0"].setPos(
                        *main_window._display_point("cam0", dragged_cam0)
                    )
                    window._on_beam_target_position_changed("cam0")

                    with patch(
                        "merlin_track_position.interface.main_window."
                        "QtWidgets.QFileDialog.getSaveFileName",
                        return_value=(str(path), ""),
                    ):
                        window._on_save_calibration_clicked()

                    saved = load_calibration_dataset(path)
                    self.assertEqual(saved.attrs["beam_target_cam0_u"], cam0_target[0])
                    self.assertEqual(saved.attrs["beam_target_cam0_v"], cam0_target[1])
                    self.assertEqual(saved.attrs["beam_target_cam1_u"], cam1_target[0])
                    self.assertEqual(saved.attrs["beam_target_cam1_v"], cam1_target[1])
                    np.testing.assert_allclose(
                        beam_target_point(window, "cam0"),
                        dragged_cam0,
                    )
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
                self.assertTrue(beam_target_dragging_enabled(window))

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
                self.assertTrue(beam_target_dragging_enabled(window))
            finally:
                window.close()

    def test_clearing_calibration_preserves_current_beam_targets(self):
        get_qapp()
        with patched_main_window_runtime() as settings:
            window = MainWindow()
            try:
                target_points = {
                    camera: np.asarray(
                        main_window._roi_center_point(
                            camera,
                            window._get_roi_geometry(camera),
                        ),
                        dtype=float,
                    )
                    + np.asarray([6.0, -5.0], dtype=float)
                    for camera in main_window.CAMERA_IMAGE_SIZES
                }
                target_attrs = {
                    key: float(value)
                    for camera, point in target_points.items()
                    for key, value in zip(
                        main_window.BEAM_TARGET_ATTR_KEYS[camera],
                        point,
                        strict=True,
                    )
                }
                calibration = build_sample_calibration_dataset(
                    image_shape_cam0=(4, 5),
                    image_shape_cam1=(6, 7),
                ).assign_attrs(
                    {"calibration_path": "/tmp/calibration.h5"} | target_attrs
                )

                window._on_new_calibration_ready(calibration)
                settings.set_calls.clear()

                with patch.object(
                    CalibrationStartDialog,
                    "exec",
                    side_effect=AssertionError("dialog should not open"),
                ):
                    window._on_new_calibration_clicked()

                self.assertIsNone(window._calibration)
                for camera, point in target_points.items():
                    np.testing.assert_allclose(beam_target_point(window, camera), point)
                    self.assertIn(camera, window._beam_target_user_overrides)
                self.assertEqual(
                    settings.set_calls,
                    [
                        ("beam_target/cam0/u", float(target_points["cam0"][0])),
                        ("beam_target/cam0/v", float(target_points["cam0"][1])),
                        ("beam_target/cam1/u", float(target_points["cam1"][0])),
                        ("beam_target/cam1/v", float(target_points["cam1"][1])),
                    ],
                )
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

    def test_transform_preview_warps_roi_only_and_reference_release_restores_it(self):
        get_qapp()
        roi = (2.0, 3.0, 4.0, 5.0)
        metadata = _roi_metadata_from_geometries(
            {
                "cam0": roi,
                "cam1": (0.0, 0.0, 6.0, 7.0),
            }
        )
        image_width, image_height = main_window.CAMERA_IMAGE_SIZES["cam0"]
        cam1_width, cam1_height = main_window.CAMERA_IMAGE_SIZES["cam1"]
        calibration = build_sample_calibration_dataset(
            image_shape_cam0=(image_height, image_width),
            image_shape_cam1=(cam1_height, cam1_width),
        ).assign_attrs({"calibration_path": "/tmp/calibration.h5"} | metadata)
        current_cam0 = np.arange(image_height * image_width, dtype=np.int32).reshape(
            image_height,
            image_width,
        )
        refreshed_cam0 = current_cam0 + 1000.0
        warp = np.asarray([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]], dtype=np.float32)

        def fake_preview_warps(_calibration, readbacks, reference_points):
            self.assertEqual(tuple(readbacks), ("p", "t", "a"))
            self.assertIn("cam0", reference_points)
            return {
                "cam0": warp,
                "cam1": np.asarray(
                    [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                    dtype=np.float32,
                ),
            }, {"orientation_seed_applied": True}

        with (
            patched_main_window_runtime(),
            patch(
                "merlin_track_position.interface.main_window.refresh_motor_positions",
                return_value=(0.0, 0.0, 0.0),
            ) as refresh,
            patch(
                "merlin_track_position.interface.main_window.orientation_ecc_initial_warps_for_readbacks",
                side_effect=fake_preview_warps,
            ),
        ):
            window = MainWindow()
            try:
                window._on_image_capture_ready("cam0", current_cam0)
                window._on_new_calibration_ready(calibration)

                window.initial_transform_preview_checkbox.setChecked(True)

                refresh.assert_called_once_with(main_window.ORIENTATION_READBACK_AXES)
                reference_crop = main_window.crop_image_to_roi(
                    calibration["reference_cam0"].values,
                    roi,
                )
                self.assertEqual(reference_crop.shape, (5, 4))
                expected_crop = main_window.crop_image_to_roi(current_cam0, roi).astype(
                    np.float32
                )
                expected = cv2.warpAffine(
                    expected_crop,
                    warp,
                    (reference_crop.shape[1], reference_crop.shape[0]),
                    flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=0,
                )
                np.testing.assert_allclose(window.image_items["cam0"].image, expected)
                self.assertEqual(window.image_items["cam0"].image.shape, (5, 4))
                assert_rect_close(self, image_parent_rect(window, "cam0"), roi)

                window.show_reference_images_button.pressed.emit()
                np.testing.assert_array_equal(
                    window.image_items["cam0"].image,
                    calibration["reference_cam0"].values,
                )

                window._on_image_capture_ready("cam0", refreshed_cam0)
                np.testing.assert_array_equal(
                    window.image_items["cam0"].image,
                    calibration["reference_cam0"].values,
                )
                window.show_reference_images_button.released.emit()

                expected_refreshed = cv2.warpAffine(
                    main_window.crop_image_to_roi(refreshed_cam0, roi).astype(
                        np.float32
                    ),
                    warp,
                    (reference_crop.shape[1], reference_crop.shape[0]),
                    flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=0,
                )
                np.testing.assert_allclose(
                    window.image_items["cam0"].image,
                    expected_refreshed,
                )
                self.assertEqual(refresh.call_count, 1)
            finally:
                window.close()

    def test_transform_preview_failure_restores_normal_live_display(self):
        get_qapp()
        calibration = build_sample_calibration_dataset(
            image_shape_cam0=(5, 4),
            image_shape_cam1=(7, 6),
        ).assign_attrs({"calibration_path": "/tmp/calibration.h5"})
        current_cam0 = full_camera_image("cam0", 3.0)

        with (
            patched_main_window_runtime(),
            patch(
                "merlin_track_position.interface.main_window.refresh_motor_positions",
                return_value=(0.0, 0.0, 0.0),
            ),
            patch(
                "merlin_track_position.interface.main_window.orientation_ecc_initial_warps_for_readbacks",
                side_effect=ValueError("missing orientation"),
            ),
        ):
            window = MainWindow()
            try:
                window._on_image_capture_ready("cam0", current_cam0)
                window._on_new_calibration_ready(calibration)

                window.initial_transform_preview_checkbox.setChecked(True)

                self.assertFalse(window.initial_transform_preview_checkbox.isChecked())
                np.testing.assert_array_equal(
                    window.image_items["cam0"].image,
                    current_cam0,
                )
                self.assertIn("missing orientation", window.statusBar().currentMessage())
                image_width, image_height = main_window.CAMERA_IMAGE_SIZES["cam0"]
                assert_rect_close(
                    self,
                    image_parent_rect(window, "cam0"),
                    (0.0, 0.0, float(image_width), float(image_height)),
                )
            finally:
                window.close()

    def test_transform_preview_known_move_refreshes_from_motor_cache(self):
        get_qapp()
        calibration = build_sample_calibration_dataset(
            image_shape_cam0=(5, 4),
            image_shape_cam1=(7, 6),
        ).assign_attrs({"calibration_path": "/tmp/calibration.h5"})
        current_cam0 = full_camera_image("cam0", 3.0)
        helper_readbacks = []

        def fake_preview_warps(_calibration, readbacks, _reference_points):
            helper_readbacks.append(dict(readbacks))
            return {
                camera: np.asarray(
                    [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                    dtype=np.float32,
                )
                for camera in main_window.CAMERA_IMAGE_SIZES
            }, {"orientation_seed_applied": True}

        with (
            patched_main_window_runtime(),
            patch(
                "merlin_track_position.interface.main_window.refresh_motor_positions",
                return_value=(0.0, 0.0, 0.0),
            ) as refresh,
            patch(
                "merlin_track_position.interface.main_window.cached_motor_positions",
                return_value=(5.0, 0.0, 0.0),
            ) as cached,
            patch(
                "merlin_track_position.interface.main_window.orientation_ecc_initial_warps_for_readbacks",
                side_effect=fake_preview_warps,
            ),
        ):
            window = MainWindow()
            try:
                window._on_image_capture_ready("cam0", current_cam0)
                window._on_new_calibration_ready(calibration)
                window.initial_transform_preview_checkbox.setChecked(True)

                window._on_stored_axis_move_ready("p", 5.0, 5.0)

                refresh.assert_called_once_with(main_window.ORIENTATION_READBACK_AXES)
                cached.assert_called_with(main_window.ORIENTATION_READBACK_AXES)
                self.assertEqual(helper_readbacks[0]["p"], 0.0)
                self.assertEqual(helper_readbacks[-1]["p"], 5.0)
            finally:
                window.close()

    def test_beam_target_drag_previews_reference_image_until_release(self):
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

                window._on_beam_target_drag_started("cam0")

                self.assertTrue(window._reference_preview_active)
                self.assertTrue(window._beam_target_reference_preview_active)
                np.testing.assert_array_equal(
                    window.image_items["cam0"].image,
                    calibration["reference_cam0"].values,
                )
                np.testing.assert_array_equal(
                    window.image_items["cam1"].image,
                    calibration["reference_cam1"].values,
                )

                window._on_image_capture_ready("cam0", refreshed_cam0)
                np.testing.assert_array_equal(
                    window.image_items["cam0"].image,
                    calibration["reference_cam0"].values,
                )

                initial_point = beam_target_point(window, "cam0")
                dragged_cam0 = (initial_point[0] + 1.0, initial_point[1] - 1.0)
                window.image_targets["cam0"].setPos(
                    *main_window._display_point("cam0", dragged_cam0)
                )
                window._on_beam_target_position_change_finished("cam0")

                self.assertFalse(window._reference_preview_active)
                self.assertFalse(window._beam_target_reference_preview_active)
                np.testing.assert_array_equal(
                    window.image_items["cam0"].image,
                    refreshed_cam0,
                )
                np.testing.assert_array_equal(
                    window.image_items["cam1"].image,
                    current_cam1,
                )
                np.testing.assert_allclose(
                    beam_target_point(window, "cam0"),
                    dragged_cam0,
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
                        registration_config_to_measurement_kwargs(
                            window._registration_config
                        ),
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
                self.assertTrue(beam_target_dragging_enabled(window))

                window._set_roi_editing_enabled(True)

                self.assertEqual(roi_handle_count(window), 16)
                self.assertEqual(roi_child_item_count(window), 16)
                self.assertTrue(roi_handles_visible(window))
                self.assertTrue(roi_editing_enabled(window))
                self.assertFalse(roi_body_translation_enabled(window))
                self.assertTrue(beam_target_dragging_enabled(window))
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
                        registration_config_to_measurement_kwargs(
                            window._registration_config
                        ),
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
                    for refresh_thread in window._image_refresh_threads.values():
                        self.assertFalse(refresh_thread.enabled)
                        self.assertEqual(refresh_thread.wait_until_idle_calls, 1)
                finally:
                    window.close()

    def test_stored_axis_move_cancel_does_not_start_thread(self):
        get_qapp()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "calibration.h5"
            calibration = write_sample_calibration(path).assign_attrs(polar=12.34567)
            with patched_main_window_runtime():
                window = MainWindow()
                try:
                    window._on_new_calibration_ready(calibration)
                    with patch(
                        "merlin_track_position.interface.main_window.QtWidgets.QMessageBox.warning",
                        return_value=QtWidgets.QMessageBox.StandardButton.Cancel,
                    ):
                        window.calibration_panel.stored_orientation_go_buttons[
                            "polar"
                        ].click()

                    self.assertFalse(window._stored_axis_move_thread.started)
                finally:
                    window.close()

    def test_stored_axis_move_confirmation_starts_thread_and_disables_actions(self):
        get_qapp()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "calibration.h5"
            calibration = write_sample_calibration(path).assign_attrs(tilt=-3.2)
            with patched_main_window_runtime():
                window = MainWindow()
                try:
                    window._on_new_calibration_ready(calibration)
                    with patch(
                        "merlin_track_position.interface.main_window.QtWidgets.QMessageBox.warning",
                        return_value=QtWidgets.QMessageBox.StandardButton.Ok,
                    ):
                        window.calibration_panel.stored_orientation_go_buttons[
                            "tilt"
                        ].click()

                    thread = window._stored_axis_move_thread
                    self.assertTrue(thread.started)
                    self.assertEqual(thread.axis_alias, "t")
                    self.assertAlmostEqual(thread.target_value, -3.2)
                    self.assertFalse(
                        window.calibration_panel.correct_sample_button.isEnabled()
                    )
                    self.assertFalse(
                        window.calibration_panel.stored_orientation_go_buttons[
                            "tilt"
                        ].isEnabled()
                    )
                    for refresh_thread in window._image_refresh_threads.values():
                        self.assertFalse(refresh_thread.enabled)
                        self.assertEqual(refresh_thread.wait_until_idle_calls, 1)
                    self.assertIn(
                        "Moving Tilt",
                        window.calibration_panel.calibration_status_label.text(),
                    )
                finally:
                    window.close()

    def test_stored_axis_move_success_restores_loaded_state(self):
        get_qapp()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "calibration.h5"
            calibration = write_sample_calibration(path).assign_attrs(polar=12.34567)
            with patched_main_window_runtime():
                window = MainWindow()
                try:
                    window._on_new_calibration_ready(calibration)
                    with patch(
                        "merlin_track_position.interface.main_window.QtWidgets.QMessageBox.warning",
                        return_value=QtWidgets.QMessageBox.StandardButton.Ok,
                    ):
                        window.calibration_panel.stored_orientation_go_buttons[
                            "polar"
                        ].click()

                    window._stored_axis_move_thread.running = False
                    window._stored_axis_move_thread.sigStoredAxisMoveReady.emit(
                        "p",
                        12.34567,
                        12.34,
                    )

                    self.assertTrue(
                        window.calibration_panel.correct_sample_button.isEnabled()
                    )
                    self.assertTrue(
                        window.calibration_panel.stored_orientation_go_buttons[
                            "polar"
                        ].isEnabled()
                    )
                    self.assertFalse(
                        window.calibration_panel.stored_orientation_widget.isHidden()
                    )
                    status = window.calibration_panel.calibration_status_label.text()
                    self.assertIn("Moved Polar", status)
                    self.assertIn("12.3457", status)
                    self.assertIn("12.3400", status)
                finally:
                    window.close()

    def test_stored_axis_move_preserves_correction_result_layout(self):
        get_qapp()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "calibration.h5"
            calibration = write_sample_calibration(path).assign_attrs(polar=12.34567)
            result = correction_result_with_moves()
            with patched_main_window_runtime():
                window = MainWindow()
                try:
                    window._on_new_calibration_ready(calibration)
                    window._last_correction_result = result
                    window.calibration_panel.show_correction_result(result)
                    self.assertTrue(
                        window.calibration_panel.calibration_review_widget.isHidden()
                    )
                    self.assertFalse(
                        window.calibration_panel.correction_steps_group.isHidden()
                    )

                    with patch(
                        "merlin_track_position.interface.main_window.QtWidgets.QMessageBox.warning",
                        return_value=QtWidgets.QMessageBox.StandardButton.Ok,
                    ):
                        window.calibration_panel.stored_orientation_go_buttons[
                            "polar"
                        ].click()

                    self.assertTrue(
                        window.calibration_panel.calibration_review_widget.isHidden()
                    )
                    self.assertTrue(window.calibration_panel.metrics_group.isHidden())
                    self.assertTrue(window.calibration_panel.warnings_group.isHidden())
                    self.assertTrue(
                        window.calibration_panel.residual_graphics_layout.isHidden()
                    )
                    self.assertFalse(
                        window.calibration_panel.correction_steps_group.isHidden()
                    )
                    self.assertIn(
                        "Moving Polar",
                        window.calibration_panel.calibration_status_label.text(),
                    )

                    window._stored_axis_move_thread.running = False
                    window._stored_axis_move_thread.sigStoredAxisMoveReady.emit(
                        "p",
                        12.34567,
                        12.34,
                    )

                    self.assertTrue(
                        window.calibration_panel.calibration_review_widget.isHidden()
                    )
                    self.assertFalse(
                        window.calibration_panel.correction_steps_group.isHidden()
                    )
                    self.assertIn(
                        "Moved Polar",
                        window.calibration_panel.calibration_status_label.text(),
                    )
                finally:
                    window.close()

    def test_stored_axis_move_failure_restores_loaded_state_and_shows_error(self):
        get_qapp()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "calibration.h5"
            calibration = write_sample_calibration(path).assign_attrs(azi=47.5)
            with patched_main_window_runtime():
                window = MainWindow()
                try:
                    window._on_new_calibration_ready(calibration)
                    with patch(
                        "merlin_track_position.interface.main_window.QtWidgets.QMessageBox.warning",
                        return_value=QtWidgets.QMessageBox.StandardButton.Ok,
                    ):
                        window.calibration_panel.stored_orientation_go_buttons[
                            "azi"
                        ].click()

                    window._stored_axis_move_thread.running = False
                    with patch(
                        "merlin_track_position.interface.main_window.QtWidgets.QMessageBox.critical"
                    ) as critical:
                        window._stored_axis_move_thread.sigStoredAxisMoveFailed.emit(
                            "a",
                            "boom",
                        )

                    self.assertTrue(
                        window.calibration_panel.correct_sample_button.isEnabled()
                    )
                    self.assertFalse(
                        window.calibration_panel.stored_orientation_widget.isHidden()
                    )
                    critical.assert_called_once()
                    self.assertIn("Azimuth", critical.call_args.args[2])
                    self.assertIn("boom", critical.call_args.args[2])
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
                            "use_window": True,
                            "use_ecc_refinement": True,
                            "ecc_use_window": True,
                            "ecc_gauss_filter_size": {"cam0": 3, "cam1": 9},
                            "capture_count": 7,
                            "capture_aggregation": "mean_image",
                        }
                    )
                    window._on_new_calibration_ready(calibration)

                    window._start_correction()

                    shift_kwargs = dict(window._correction_thread.shift_kwargs)
                    reference_points = shift_kwargs.pop("ecc_reference_point_px")
                    self.assertEqual(
                        shift_kwargs,
                        {
                            "clip_percentiles": (20.0, 80.0),
                            "use_window": True,
                            "use_ecc_refinement": True,
                            "ecc_use_window": True,
                            "ecc_motion_model": "homography",
                            "ecc_gauss_filter_size": {"cam0": 3, "cam1": 9},
                            "capture_count": 7,
                            "capture_aggregation": "mean_image",
                        },
                    )
                    for camera in main_window.CAMERA_IMAGE_SIZES:
                        np.testing.assert_allclose(
                            reference_points[camera],
                            beam_target_point(window, camera),
                        )
                finally:
                    window.close()

    def test_ecc_beam_target_drag_overrides_metadata_for_detect_and_correction(self):
        get_qapp()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "calibration.h5"
            roi_metadata = _roi_metadata_from_geometries(
                {
                    "cam0": (10.0, 20.0, 100.0, 80.0),
                    "cam1": (30.0, 40.0, 120.0, 90.0),
                }
            )
            calibration = write_sample_calibration(path).assign_attrs(
                roi_metadata
                | {
                    "beam_target_cam0_u": 35.0,
                    "beam_target_cam0_v": 45.0,
                    "beam_target_cam1_u": 70.0,
                    "beam_target_cam1_v": 85.0,
                }
            )
            with patched_main_window_runtime():
                window = MainWindow()
                try:
                    config = dict(window._registration_config)
                    config["use_ecc_refinement"] = True
                    window._on_registration_config_saved(config)
                    window._on_new_calibration_ready(calibration)

                    np.testing.assert_allclose(
                        beam_target_point(window, "cam0"),
                        [35.0, 45.0],
                    )
                    dragged_cam0 = (50.0, 60.0)
                    window.image_targets["cam0"].setPos(
                        *main_window._display_point("cam0", dragged_cam0)
                    )
                    window._on_beam_target_position_changed("cam0")

                    updated_calibration = calibration.assign_attrs(
                        {
                            "beam_target_cam0_u": 90.0,
                            "beam_target_cam0_v": 95.0,
                            "beam_target_cam1_u": 75.0,
                            "beam_target_cam1_v": 90.0,
                        }
                    )
                    window._apply_calibration_beam_target_metadata(
                        updated_calibration,
                        preserve_user_overrides=True,
                    )

                    np.testing.assert_allclose(
                        beam_target_point(window, "cam0"),
                        dragged_cam0,
                    )
                    np.testing.assert_allclose(
                        beam_target_point(window, "cam1"),
                        [75.0, 90.0],
                    )
                    expected_points = {
                        "cam0": np.asarray([40.0, 40.0], dtype=float),
                        "cam1": np.asarray([45.0, 50.0], dtype=float),
                    }
                    registration_kwargs = window._registration_measurement_kwargs()
                    for camera, point in expected_points.items():
                        np.testing.assert_allclose(
                            registration_kwargs["ecc_reference_point_px"][camera],
                            point,
                        )

                    window._on_detect_shift_clicked()
                    for camera, point in expected_points.items():
                        np.testing.assert_allclose(
                            window._detect_shift_thread.shift_kwargs[
                                "ecc_reference_point_px"
                            ][camera],
                            point,
                        )

                    window._detect_shift_thread.running = False
                    window._start_correction()
                    for camera, point in expected_points.items():
                        np.testing.assert_allclose(
                            window._correction_thread.shift_kwargs[
                                "ecc_reference_point_px"
                            ][camera],
                            point,
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
                        registration_config_to_measurement_kwargs(
                            window._registration_config
                        ),
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
                        registration_config_to_measurement_kwargs(
                            window._registration_config
                        ),
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
                        window.calibration_panel.correction_warnings_text.toPlainText(),
                        "No correction warnings.",
                    )
                    self.assertTrue(
                        window.calibration_panel.calibration_review_widget.isHidden()
                    )
                    self.assertTrue(window.calibration_panel.metrics_group.isHidden())
                    self.assertFalse(
                        window.calibration_panel.correction_steps_group.isHidden()
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

    def test_load_calibration_dialog_starts_at_scan_default_path(self):
        get_qapp()
        default_path = Path("/tmp/current_scan/calibration.h5")
        with patched_main_window_runtime():
            window = MainWindow()
            try:
                with (
                    patch.object(
                        main_window,
                        "_default_calibration_path",
                        return_value=default_path,
                    ),
                    patch(
                        "merlin_track_position.interface.main_window."
                        "QtWidgets.QFileDialog.getOpenFileName",
                        return_value=("", ""),
                    ) as get_open_file_name,
                ):
                    window._on_load_calibration_clicked()

                get_open_file_name.assert_called_once_with(
                    window,
                    "Load calibration",
                    str(default_path),
                    "Calibration files (*.h5 *.hdf5 *.nc);;All files (*)",
                )
            finally:
                window.close()

    def test_load_calibration_dialog_starts_at_current_calibration_path(self):
        get_qapp()
        current_path = Path("/tmp/current_scan/current_calibration.h5")
        with patched_main_window_runtime():
            window = MainWindow()
            try:
                window._calibration_path = current_path
                with patch(
                    "merlin_track_position.interface.main_window."
                    "QtWidgets.QFileDialog.getOpenFileName",
                    return_value=("", ""),
                ) as get_open_file_name:
                    window._on_load_calibration_clicked()

                get_open_file_name.assert_called_once_with(
                    window,
                    "Load calibration",
                    str(current_path),
                    "Calibration files (*.h5 *.hdf5 *.nc);;All files (*)",
                )
            finally:
                window.close()

    def test_load_calibration_dialog_falls_back_to_home_when_scan_path_fails(self):
        get_qapp()
        fallback_path = Path.home() / main_window.DEFAULT_CALIBRATION_FILE_NAME
        with patched_main_window_runtime():
            window = MainWindow()
            try:
                with (
                    patch.object(
                        main_window,
                        "_default_calibration_path",
                        side_effect=FileNotFoundError("missing setup"),
                    ),
                    patch(
                        "merlin_track_position.interface.main_window."
                        "QtWidgets.QFileDialog.getOpenFileName",
                        return_value=("", ""),
                    ) as get_open_file_name,
                ):
                    window._on_load_calibration_clicked()

                get_open_file_name.assert_called_once_with(
                    window,
                    "Load calibration",
                    str(fallback_path),
                    "Calibration files (*.h5 *.hdf5 *.nc);;All files (*)",
                )
            finally:
                window.close()

    def test_loading_calibration_without_history_shows_calibration_review(self):
        get_qapp()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "calibration.h5"
            write_sample_calibration(path)

            with patched_main_window_runtime():
                window = MainWindow()
                try:
                    with patch(
                        "merlin_track_position.interface.main_window."
                        "QtWidgets.QFileDialog.getOpenFileName",
                        return_value=(str(path), ""),
                    ):
                        window._on_load_calibration_clicked()

                    self.assertIsNone(window._last_correction_result)
                    self.assertIn(
                        "Loaded calibration",
                        window.calibration_panel.calibration_status_label.text(),
                    )
                    self.assertEqual(
                        window.calibration_panel.calibration_file_label.text(),
                        "Calibration file: calibration.h5",
                    )
                    self.assertFalse(
                        window.calibration_panel.calibration_review_widget.isHidden()
                    )
                    self.assertFalse(window.calibration_panel.metrics_group.isHidden())
                    self.assertFalse(window.calibration_panel.warnings_group.isHidden())
                    self.assertTrue(
                        window.calibration_panel.correction_steps_group.isHidden()
                    )
                    self.assertNotEqual(
                        window.calibration_panel.metric_labels[
                            "axis_scale_readback_mm"
                        ].text(),
                        "n/a",
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
                        window.calibration_panel.calibration_file_label.text(),
                        "Calibration file: calibration.h5",
                    )
                    self.assertEqual(
                        window.calibration_panel.correction_warnings_text.toPlainText(),
                        "residual increased",
                    )
                    self.assertTrue(
                        window.calibration_panel.calibration_review_widget.isHidden()
                    )
                    self.assertTrue(window.calibration_panel.metrics_group.isHidden())
                    self.assertTrue(window.calibration_panel.warnings_group.isHidden())
                    self.assertFalse(
                        window.calibration_panel.correction_steps_group.isHidden()
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
                panel.correction_warnings_text.toPlainText(),
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
        self.assertIsNotNone(dialog.findChild(QtWidgets.QSpinBox, "calibration_n_spin"))
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

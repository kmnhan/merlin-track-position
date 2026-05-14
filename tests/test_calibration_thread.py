import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import xarray as xr
from qtpy import QtWidgets

from merlin_track_position.instruments.cameras import (
    CallableCameraPlugin,
    CameraPairPlugin,
)
from merlin_track_position.interface.calibration_thread import CalibrationThread
from merlin_track_position.interface.correction_thread import CorrectionThread
from merlin_track_position.interface.detection_thread import DetectShiftThread
from merlin_track_position.tracking.sample_calibration import (
    build_sample_calibration_dataset,
)

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

_APP = None


def get_qapp():
    global _APP
    _APP = QtWidgets.QApplication.instance() or _APP or QtWidgets.QApplication([])
    return _APP


class CalibrationThreadTests(unittest.TestCase):
    def test_run_calls_run_calibration_and_emits_step_and_ready(self):
        get_qapp()
        image_cam0 = np.arange(4 * 5, dtype=float).reshape(4, 5)
        image_cam1 = np.arange(6 * 7, dtype=float).reshape(6, 7)
        calls = []

        camera_pair = CameraPairPlugin(
            CallableCameraPlugin("cam0", lambda: image_cam0),
            CallableCameraPlugin("cam1", lambda: image_cam1),
        )

        def fake_run_calibration(
            pair,
            *,
            output_path,
            additional_context,
            step_callback,
            processing_callback,
        ):
            calls.append((pair, Path(output_path), dict(additional_context)))
            captured_cam0, captured_cam1 = pair.capture_pair()
            step_callback(0, 1.0, 2.0, 3.0, captured_cam0, captured_cam1)
            processing_callback(0, 2)
            processing_callback(1, 2)
            processing_callback(2, 2)
            return build_sample_calibration_dataset(
                image_shape_cam0=(4, 5),
                image_shape_cam1=(6, 7),
            ).assign_attrs(additional_context | {"calibration_path": str(output_path)})

        thread = CalibrationThread()
        steps = []
        processing_steps = []
        ready = []
        failed = []
        thread.sigCalibrationStep.connect(
            lambda idx, dx, dy, dz, cam0, cam1: steps.append(
                (idx, dx, dy, dz, cam0, cam1)
            )
        )
        thread.sigCalibrationProcessingStep.connect(
            lambda completed, total: processing_steps.append((completed, total))
        )
        thread.sigCalibrationReady.connect(
            lambda calibration: ready.append(calibration)
        )
        thread.sigCalibrationFailed.connect(lambda message: failed.append(message))

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "calibration.h5"
            roi_metadata = {
                "roi_cam0_x": 1.0,
                "roi_cam0_y": 2.0,
                "roi_cam0_width": 3.0,
                "roi_cam0_height": 4.0,
            }
            thread.configure(
                camera_pair,
                roi_metadata,
                output_path,
            )
            with patch(
                "merlin_track_position.interface.calibration_thread.run_calibration",
                side_effect=fake_run_calibration,
            ):
                thread.run()

        self.assertEqual(calls, [(camera_pair, output_path, roi_metadata)])
        self.assertEqual(len(steps), 1)
        self.assertEqual(steps[0][:4], (0, 1.0, 2.0, 3.0))
        np.testing.assert_array_equal(steps[0][4], image_cam0)
        np.testing.assert_array_equal(steps[0][5], image_cam1)
        self.assertEqual(processing_steps, [(0, 2), (1, 2), (2, 2)])
        self.assertEqual(len(ready), 1)
        self.assertEqual(ready[0].attrs["roi_cam0_width"], 3.0)
        self.assertEqual(failed, [])

    def test_run_emits_failure_when_run_calibration_raises(self):
        get_qapp()

        def fake_run_calibration(
            camera_pair,
            *,
            output_path,
            additional_context,
            step_callback,
            processing_callback,
        ):
            del (
                camera_pair,
                output_path,
                additional_context,
                step_callback,
                processing_callback,
            )
            raise RuntimeError("boom")

        thread = CalibrationThread()
        ready = []
        failed = []
        thread.sigCalibrationReady.connect(
            lambda calibration: ready.append(calibration)
        )
        thread.sigCalibrationFailed.connect(lambda message: failed.append(message))
        with tempfile.TemporaryDirectory() as tmpdir:
            thread.configure(
                CameraPairPlugin(
                    CallableCameraPlugin("cam0", lambda: np.zeros((2, 2))),
                    CallableCameraPlugin("cam1", lambda: np.zeros((2, 2))),
                ),
                {},
                Path(tmpdir) / "calibration.h5",
            )

            with patch(
                "merlin_track_position.interface.calibration_thread.run_calibration",
                side_effect=fake_run_calibration,
            ):
                thread.run()

        self.assertEqual(ready, [])
        self.assertEqual(failed, ["boom"])


class CorrectionThreadTests(unittest.TestCase):
    def test_run_calls_do_correction_and_emits_ready(self):
        get_qapp()
        image_cam0 = np.arange(4 * 5, dtype=float).reshape(4, 5)
        image_cam1 = np.arange(6 * 7, dtype=float).reshape(6, 7)
        camera_pair = CameraPairPlugin(
            CallableCameraPlugin("cam0", lambda: image_cam0),
            CallableCameraPlugin("cam1", lambda: image_cam1),
        )
        calibration = build_sample_calibration_dataset(
            image_shape_cam0=(4, 5),
            image_shape_cam1=(6, 7),
        )
        progress = xr.Dataset(attrs={"correction_iterations": 1})
        result = xr.Dataset(attrs={"correction_converged": True})
        calls = []

        def fake_do_correction(
            passed_calibration,
            passed_camera_pair,
            *,
            calibration_path,
            progress_callback,
            motor_backend,
        ):
            calls.append(
                (
                    passed_calibration,
                    passed_camera_pair,
                    Path(calibration_path),
                    motor_backend,
                )
            )
            progress_callback(progress)
            return result

        thread = CorrectionThread()
        progress_results = []
        ready = []
        failed = []
        thread.sigCorrectionProgress.connect(
            lambda value: progress_results.append(value)
        )
        thread.sigCorrectionReady.connect(lambda value: ready.append(value))
        thread.sigCorrectionFailed.connect(lambda message: failed.append(message))

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "calibration.h5"
            thread.configure(calibration, camera_pair, path)
            with patch(
                "merlin_track_position.interface.correction_thread.do_correction",
                side_effect=fake_do_correction,
            ):
                thread.run()

        self.assertEqual(calls, [(calibration, camera_pair, path, None)])
        self.assertEqual(progress_results, [progress])
        self.assertEqual(ready, [result])
        self.assertEqual(failed, [])

    def test_run_emits_failure_when_do_correction_raises(self):
        get_qapp()
        thread = CorrectionThread()
        ready = []
        failed = []
        thread.sigCorrectionReady.connect(lambda value: ready.append(value))
        thread.sigCorrectionFailed.connect(lambda message: failed.append(message))

        with tempfile.TemporaryDirectory() as tmpdir:
            thread.configure(
                build_sample_calibration_dataset(
                    image_shape_cam0=(4, 5),
                    image_shape_cam1=(6, 7),
                ),
                CameraPairPlugin(
                    CallableCameraPlugin("cam0", lambda: np.zeros((2, 2))),
                    CallableCameraPlugin("cam1", lambda: np.zeros((2, 2))),
                ),
                Path(tmpdir) / "calibration.h5",
            )
            with patch(
                "merlin_track_position.interface.correction_thread.do_correction",
                side_effect=RuntimeError("boom"),
            ):
                thread.run()

        self.assertEqual(ready, [])
        self.assertEqual(failed, ["boom"])


class DetectShiftThreadTests(unittest.TestCase):
    def test_run_calls_detect_shift_and_emits_ready(self):
        get_qapp()
        image_cam0 = np.arange(4 * 5, dtype=float).reshape(4, 5)
        image_cam1 = np.arange(6 * 7, dtype=float).reshape(6, 7)
        camera_pair = CameraPairPlugin(
            CallableCameraPlugin("cam0", lambda: image_cam0),
            CallableCameraPlugin("cam1", lambda: image_cam1),
        )
        calibration = build_sample_calibration_dataset(
            image_shape_cam0=(4, 5),
            image_shape_cam1=(6, 7),
        )
        result = xr.Dataset(attrs={"warnings": ""})
        calls = []

        def fake_detect_shift(passed_calibration, passed_camera_pair):
            calls.append((passed_calibration, passed_camera_pair))
            return result

        thread = DetectShiftThread()
        ready = []
        failed = []
        thread.sigDetectionReady.connect(lambda value: ready.append(value))
        thread.sigDetectionFailed.connect(lambda message: failed.append(message))

        thread.configure(calibration, camera_pair)
        with patch(
            "merlin_track_position.interface.detection_thread.detect_shift",
            side_effect=fake_detect_shift,
        ):
            thread.run()

        self.assertEqual(calls, [(calibration, camera_pair)])
        self.assertEqual(ready, [result])
        self.assertEqual(failed, [])

    def test_run_emits_failure_when_detect_shift_raises(self):
        get_qapp()
        thread = DetectShiftThread()
        ready = []
        failed = []
        thread.sigDetectionReady.connect(lambda value: ready.append(value))
        thread.sigDetectionFailed.connect(lambda message: failed.append(message))
        thread.configure(
            build_sample_calibration_dataset(
                image_shape_cam0=(4, 5),
                image_shape_cam1=(6, 7),
            ),
            CameraPairPlugin(
                CallableCameraPlugin("cam0", lambda: np.zeros((2, 2))),
                CallableCameraPlugin("cam1", lambda: np.zeros((2, 2))),
            ),
        )

        with patch(
            "merlin_track_position.interface.detection_thread.detect_shift",
            side_effect=RuntimeError("boom"),
        ):
            thread.run()

        self.assertEqual(ready, [])
        self.assertEqual(failed, ["boom"])


if __name__ == "__main__":
    unittest.main()

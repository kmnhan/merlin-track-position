import os
import unittest
from unittest.mock import patch

import numpy as np
from qtpy import QtWidgets

from merlin_track_position.interface.calibration_thread import CalibrationThread
from merlin_track_position.tracking.sample_calibration import build_sample_calibration_dataset

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


class CalibrationThreadTests(unittest.TestCase):
    def test_run_calls_run_calibration_and_emits_step_and_ready(self):
        app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        del app
        image_cam0 = np.arange(4 * 5, dtype=float).reshape(4, 5)
        image_cam1 = np.arange(6 * 7, dtype=float).reshape(6, 7)
        calls = []

        def image_generator():
            return image_cam0, image_cam1

        def fake_run_calibration(n, step_um, generator, *, step_callback):
            calls.append((n, step_um, generator))
            captured_cam0, captured_cam1 = generator()
            step_callback(0, 1.0, 2.0, 3.0, captured_cam0, captured_cam1)
            return build_sample_calibration_dataset(
                image_shape_cam0=(4, 5),
                image_shape_cam1=(6, 7),
            )

        thread = CalibrationThread()
        steps = []
        ready = []
        failed = []
        thread.sigCalibrationStep.connect(
            lambda idx, dx, dy, dz, cam0, cam1: steps.append(
                (idx, dx, dy, dz, cam0, cam1)
            )
        )
        thread.sigCalibrationReady.connect(lambda calibration: ready.append(calibration))
        thread.sigCalibrationFailed.connect(lambda message: failed.append(message))

        thread.configure(
            5,
            15.0,
            image_generator,
            {
                "roi_cam0_x": 1.0,
                "roi_cam0_y": 2.0,
                "roi_cam0_width": 3.0,
                "roi_cam0_height": 4.0,
            },
        )
        with patch(
            "merlin_track_position.interface.calibration_thread.run_calibration",
            side_effect=fake_run_calibration,
        ):
            thread.run()

        self.assertEqual(calls, [(5, 15.0, image_generator)])
        self.assertEqual(len(steps), 1)
        self.assertEqual(steps[0][:4], (0, 1.0, 2.0, 3.0))
        np.testing.assert_array_equal(steps[0][4], image_cam0)
        np.testing.assert_array_equal(steps[0][5], image_cam1)
        self.assertEqual(len(ready), 1)
        self.assertEqual(ready[0].attrs["roi_cam0_width"], 3.0)
        self.assertEqual(failed, [])

    def test_run_emits_failure_when_run_calibration_raises(self):
        app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        del app

        def fake_run_calibration(n, step_um, generator, *, step_callback):
            del n, step_um, generator, step_callback
            raise RuntimeError("boom")

        thread = CalibrationThread()
        ready = []
        failed = []
        thread.sigCalibrationReady.connect(lambda calibration: ready.append(calibration))
        thread.sigCalibrationFailed.connect(lambda message: failed.append(message))
        thread.configure(
            5,
            15.0,
            lambda: (np.zeros((2, 2)), np.zeros((2, 2))),
            {},
        )

        with patch(
            "merlin_track_position.interface.calibration_thread.run_calibration",
            side_effect=fake_run_calibration,
        ):
            thread.run()

        self.assertEqual(ready, [])
        self.assertEqual(failed, ["boom"])


if __name__ == "__main__":
    unittest.main()

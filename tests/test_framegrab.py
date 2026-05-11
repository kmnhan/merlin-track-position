import threading
import unittest
from unittest.mock import patch

import numpy as np

from merlin_track_position import constants
from merlin_track_position.instruments import basler
from merlin_track_position.instruments.basler import get_basler_image
from merlin_track_position.instruments.cameras import (
    capture_camera_pair,
    crop_image_to_roi,
    make_cropped_camera_pair_capture,
)
from merlin_track_position.instruments.framegrab import get_framegrabber_image
from merlin_track_position.instruments.motors import move_motors_and_wait
from merlin_track_position.instruments.simulated_hardware import simulator
from merlin_track_position.tracking.shift import estimate_shift


class FakeNode:
    def __init__(
        self,
        value=None,
        *,
        minimum=0,
        maximum=10_000,
        increment=1,
        writable=True,
    ):
        self.Value = value
        self.Min = minimum
        self.Max = maximum
        self.Inc = increment
        self.writable = writable


class FakeCommand:
    writable = True

    def __init__(self):
        self.executed = False

    def Execute(self):
        self.executed = True


class FakeBaslerCamera:
    def __init__(self):
        self.UserSetSelector = FakeNode()
        self.UserSetLoad = FakeCommand()
        self.GainAuto = FakeNode("Continuous")
        self.GammaSelector = FakeNode()
        self.ExposureAuto = FakeNode("Continuous")
        self.ExposureTimeAbs = FakeNode()
        self.GammaEnable = FakeNode(False)
        self.OffsetX = FakeNode(100, minimum=0)
        self.OffsetY = FakeNode(50, minimum=0)
        self.Width = FakeNode(maximum=constants.IMAGE_WIDTH_CAM1)
        self.Height = FakeNode(maximum=constants.IMAGE_HEIGHT_CAM1)
        self.PixelFormat = FakeNode()


class DevelopmentModeFramegrabTests(unittest.TestCase):
    def setUp(self):
        simulator.reset()

    def test_zero_position_frame_matches_packaged_reference_without_zmq(self):
        with (
            patch.object(constants, "IS_DAQ_PC", False),
            patch(
                "merlin_track_position.instruments.framegrab.zmq.Context.instance",
                side_effect=AssertionError("ZMQ should not be opened"),
            ),
        ):
            image = get_framegrabber_image()

        np.testing.assert_allclose(
            image,
            simulator.get_reference_image()[
                : constants.IMAGE_HEIGHT_CAM0, : constants.IMAGE_WIDTH_CAM0
            ],
        )
        self.assertEqual(image.dtype, np.float64)

    def test_development_mode_basler_placeholder_uses_simulator(self):
        with patch.object(constants, "IS_DAQ_PC", False):
            image = get_basler_image()

        self.assertEqual(
            image.shape,
            (constants.IMAGE_HEIGHT_CAM1, constants.IMAGE_WIDTH_CAM1),
        )
        self.assertEqual(image.dtype, np.float64)

    def test_basler_configuration_sets_manual_exposure_and_expected_aoi(self):
        camera = FakeBaslerCamera()

        with patch(
            "merlin_track_position.instruments.basler.genicam.IsWritable",
            lambda node: node.writable,
        ):
            basler._configure_camera(camera)

        self.assertEqual(camera.UserSetSelector.Value, "Default")
        self.assertTrue(camera.UserSetLoad.executed)
        self.assertEqual(camera.GainAuto.Value, "Off")
        self.assertEqual(camera.ExposureAuto.Value, "Off")
        self.assertEqual(camera.ExposureTimeAbs.Value, constants.BASLER_EXPOSURE)
        self.assertEqual(camera.OffsetX.Value, camera.OffsetX.Min)
        self.assertEqual(camera.OffsetY.Value, camera.OffsetY.Min)
        self.assertEqual(camera.Width.Value, constants.IMAGE_WIDTH_CAM1)
        self.assertEqual(camera.Height.Value, constants.IMAGE_HEIGHT_CAM1)
        self.assertEqual(camera.PixelFormat.Value, "Mono10")

    def test_daq_mode_basler_image_accepts_expected_shape(self):
        raw = np.arange(6, dtype=np.uint16).reshape(2, 3)

        with (
            patch.object(constants, "IS_DAQ_PC", True),
            patch.object(constants, "IMAGE_HEIGHT_CAM1", 2),
            patch.object(constants, "IMAGE_WIDTH_CAM1", 3),
            patch.object(basler, "_grab_single_image", return_value=raw),
        ):
            image = get_basler_image()

        np.testing.assert_array_equal(image, raw.astype(np.float64))
        self.assertEqual(image.shape, (2, 3))
        self.assertEqual(image.dtype, np.float64)

    def test_daq_mode_basler_image_rejects_mismatched_frame_shape(self):
        raw = np.zeros((3, 4), dtype=np.uint16)

        with (
            patch.object(constants, "IS_DAQ_PC", True),
            patch.object(constants, "IMAGE_HEIGHT_CAM1", 2),
            patch.object(constants, "IMAGE_WIDTH_CAM1", 3),
            patch.object(basler, "_grab_single_image", return_value=raw),
        ):
            with self.assertRaisesRegex(RuntimeError, "does not match constants"):
                get_basler_image()

    def test_capture_camera_pair_returns_both_images(self):
        with patch.object(constants, "IS_DAQ_PC", False):
            image_cam0, image_cam1 = capture_camera_pair()

        self.assertEqual(
            image_cam0.shape,
            (constants.IMAGE_HEIGHT_CAM0, constants.IMAGE_WIDTH_CAM0),
        )
        self.assertEqual(
            image_cam1.shape,
            (constants.IMAGE_HEIGHT_CAM1, constants.IMAGE_WIDTH_CAM1),
        )

    def test_capture_camera_pair_runs_camera_acquisitions_concurrently(self):
        image_cam0 = np.array([[1.0]])
        image_cam1 = np.array([[2.0]])
        cam0_started = threading.Event()
        cam1_started = threading.Event()

        def capture_cam0():
            cam0_started.set()
            self.assertTrue(cam1_started.wait(timeout=1.0))
            return image_cam0

        def capture_cam1():
            self.assertTrue(cam0_started.wait(timeout=1.0))
            cam1_started.set()
            return image_cam1

        captured_cam0, captured_cam1 = capture_camera_pair(
            capture_cam0,
            capture_cam1,
        )

        np.testing.assert_array_equal(captured_cam0, image_cam0)
        np.testing.assert_array_equal(captured_cam1, image_cam1)

    def test_crop_image_to_roi_uses_integer_boundaries(self):
        image = np.arange(5 * 6).reshape(5, 6)

        cropped = crop_image_to_roi(image, (1.0, 2.0, 3.0, 2.0))

        np.testing.assert_array_equal(cropped, image[2:4, 1:4])
        self.assertFalse(np.shares_memory(cropped, image))

    def test_crop_image_to_roi_expands_fractional_boundaries(self):
        image = np.arange(5 * 6).reshape(5, 6)

        cropped = crop_image_to_roi(image, (1.2, 1.8, 2.1, 2.2))

        np.testing.assert_array_equal(cropped, image[1:4, 1:4])

    def test_crop_image_to_roi_clamps_to_image_bounds(self):
        image = np.arange(5 * 6).reshape(5, 6)

        cropped = crop_image_to_roi(image, (-3.0, 1.0, 4.0, 2.0))

        np.testing.assert_array_equal(cropped, image[1:3, 0:4])

    def test_crop_image_to_roi_enforces_one_pixel_minimum(self):
        image = np.arange(5 * 6).reshape(5, 6)

        cropped = crop_image_to_roi(image, (3.0, 4.0, -5.0, 0.0))

        np.testing.assert_array_equal(cropped, image[4:5, 3:4])

    def test_make_cropped_camera_pair_capture_wraps_base_capture(self):
        image_cam0 = np.arange(4 * 5).reshape(4, 5)
        image_cam1 = np.arange(6 * 7).reshape(6, 7)

        def base_capture():
            return image_cam0, image_cam1

        capture = make_cropped_camera_pair_capture(
            (1.0, 1.0, 3.0, 2.0),
            (2.0, 3.0, 2.0, 2.0),
            base_capture=base_capture,
        )

        cropped_cam0, cropped_cam1 = capture()

        np.testing.assert_array_equal(cropped_cam0, image_cam0[1:3, 1:4])
        np.testing.assert_array_equal(cropped_cam1, image_cam1[3:5, 2:4])

    def test_simulated_frame_shift_tracks_motor_positions(self):
        stage_um = np.array([60.0, -30.0, 20.0], dtype=np.float64)

        with (
            patch.object(constants, "IS_DAQ_PC", False),
            patch("merlin_track_position.instruments.simulated_hardware.time.sleep"),
        ):
            reference = get_framegrabber_image()
            move_motors_and_wait(("x", "y", "z"), stage_um * 1e-3)
            shifted = get_framegrabber_image()

        expected_shift_px = simulator.get_stage_to_pixel("cam0") @ stage_um
        measured = estimate_shift(
            reference,
            shifted,
            check_tiles=False,
            clip_percentiles=None,
        )

        self.assertGreater(float(np.mean(np.abs(shifted - reference))), 0.0)
        np.testing.assert_allclose(
            measured["shift_px"].values,
            expected_shift_px,
            atol=0.2,
        )


if __name__ == "__main__":
    unittest.main()

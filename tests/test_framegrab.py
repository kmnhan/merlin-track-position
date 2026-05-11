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
    def __init__(self, arrays=None, *, grab_succeeded=True):
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
        self.MaxNumBuffer = None
        self.arrays = list(arrays or [])
        self.grab_succeeded = grab_succeeded
        self.open_count = 0
        self.close_count = 0
        self.start_grabbing_count = 0
        self.stop_grabbing_count = 0
        self.retrieve_result_count = 0
        self.retrieve_result_args = []
        self.is_open = False
        self.is_grabbing = False

    def Open(self):
        self.open_count += 1
        self.is_open = True

    def Close(self):
        self.close_count += 1
        self.is_open = False

    def StartGrabbing(self, strategy):
        self.start_grabbing_count += 1
        self.start_grabbing_strategy = strategy
        self.is_grabbing = True

    def StopGrabbing(self):
        self.stop_grabbing_count += 1
        self.is_grabbing = False

    def IsGrabbing(self):
        return self.is_grabbing

    def RetrieveResult(self, timeout_ms, timeout_handling):
        self.retrieve_result_count += 1
        self.retrieve_result_args.append((timeout_ms, timeout_handling))
        if self.arrays:
            array = self.arrays.pop(0)
        else:
            array = np.zeros(
                (constants.IMAGE_HEIGHT_CAM1, constants.IMAGE_WIDTH_CAM1),
                dtype=np.uint16,
            )
        return FakeGrabResult(array, succeeded=self.grab_succeeded)


class FakeGrabResult:
    def __init__(self, array, *, succeeded=True):
        self.array = np.asarray(array)
        self.succeeded = succeeded
        self.released = False

    def GrabSucceeded(self):
        return self.succeeded

    def GetArray(self, raw=False):
        del raw
        return self.array

    def Release(self):
        self.released = True

    def GetErrorCode(self):
        return 7

    def GetErrorDescription(self):
        return "fake failure"


class DevelopmentModeFramegrabTests(unittest.TestCase):
    def setUp(self):
        basler.close_basler_camera()
        simulator.reset()

    def tearDown(self):
        basler.close_basler_camera()

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

    def test_daq_mode_basler_session_reuses_open_camera_for_consecutive_images(self):
        raw0 = np.arange(6, dtype=np.uint16).reshape(2, 3)
        raw1 = raw0 + 10
        camera = FakeBaslerCamera([raw0, raw1])

        with (
            patch.object(constants, "IS_DAQ_PC", True),
            patch.object(constants, "IMAGE_HEIGHT_CAM1", 2),
            patch.object(constants, "IMAGE_WIDTH_CAM1", 3),
            patch(
                "merlin_track_position.instruments.basler.genicam.IsWritable",
                lambda node: node.writable,
            ),
            patch.object(basler, "_get_camera_by_serial_number", return_value=camera),
        ):
            image0 = get_basler_image()
            image1 = get_basler_image()

        np.testing.assert_array_equal(image0, raw0.astype(np.float64))
        np.testing.assert_array_equal(image1, raw1.astype(np.float64))
        self.assertEqual(camera.open_count, 1)
        self.assertEqual(camera.start_grabbing_count, 1)
        self.assertEqual(camera.retrieve_result_count, 2)
        self.assertEqual(camera.MaxNumBuffer, 5)
        self.assertEqual(camera.close_count, 0)

    def test_close_basler_camera_stops_closes_and_allows_reopen(self):
        raw0 = np.arange(6, dtype=np.uint16).reshape(2, 3)
        raw1 = raw0 + 20
        camera0 = FakeBaslerCamera([raw0])
        camera1 = FakeBaslerCamera([raw1])

        with (
            patch.object(constants, "IS_DAQ_PC", True),
            patch.object(constants, "IMAGE_HEIGHT_CAM1", 2),
            patch.object(constants, "IMAGE_WIDTH_CAM1", 3),
            patch(
                "merlin_track_position.instruments.basler.genicam.IsWritable",
                lambda node: node.writable,
            ),
            patch.object(
                basler,
                "_get_camera_by_serial_number",
                side_effect=[camera0, camera1],
            ),
        ):
            image0 = get_basler_image()
            basler.close_basler_camera()
            image1 = get_basler_image()

        np.testing.assert_array_equal(image0, raw0.astype(np.float64))
        np.testing.assert_array_equal(image1, raw1.astype(np.float64))
        self.assertEqual(camera0.stop_grabbing_count, 1)
        self.assertEqual(camera0.close_count, 1)
        self.assertEqual(camera1.open_count, 1)
        self.assertEqual(camera1.start_grabbing_count, 1)

    def test_failed_basler_grab_resets_session_for_next_capture(self):
        raw = np.arange(6, dtype=np.uint16).reshape(2, 3)
        failing_camera = FakeBaslerCamera([raw], grab_succeeded=False)
        recovered_camera = FakeBaslerCamera([raw])

        with (
            patch.object(constants, "IS_DAQ_PC", True),
            patch.object(constants, "IMAGE_HEIGHT_CAM1", 2),
            patch.object(constants, "IMAGE_WIDTH_CAM1", 3),
            patch(
                "merlin_track_position.instruments.basler.genicam.IsWritable",
                lambda node: node.writable,
            ),
            patch.object(
                basler,
                "_get_camera_by_serial_number",
                side_effect=[failing_camera, recovered_camera],
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "Image grab failed"):
                get_basler_image()
            image = get_basler_image()

        np.testing.assert_array_equal(image, raw.astype(np.float64))
        self.assertEqual(failing_camera.stop_grabbing_count, 1)
        self.assertEqual(failing_camera.close_count, 1)
        self.assertEqual(recovered_camera.open_count, 1)
        self.assertEqual(recovered_camera.retrieve_result_count, 1)

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

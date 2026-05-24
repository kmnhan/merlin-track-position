import json
import threading
import unittest
from importlib import resources
from unittest.mock import patch

import numpy as np

from merlin_track_position import constants
from merlin_track_position.instruments import basler, framegrab as framegrab_module
from merlin_track_position.instruments import simulated_hardware
from merlin_track_position.instruments.basler import (
    get_basler_image,
    get_basler_image_stack,
)
from merlin_track_position.instruments.camera_config import (
    BASLER_OUTPUT_RGB8,
    SOURCE_BASLER,
    SOURCE_FRAMEGRABBER,
    CameraConfig,
    DisplayTransform,
    camera_config_mismatches,
    camera_config_from_settings,
    camera_metadata,
)
from merlin_track_position.instruments.cameras import (
    BaslerCameraPlugin,
    CallableCameraPlugin,
    CameraPairPlugin,
    FramegrabberCameraPlugin,
    camera_pair_from_configs,
    capture_image_and_display_stacks,
    capture_image_stack,
    crop_image_to_roi,
    default_camera_pair,
)
from merlin_track_position.instruments.framegrab import (
    get_framegrabber_image,
    get_framegrabber_image_stack,
)
from merlin_track_position.instruments.motors import move_motors_and_wait
from merlin_track_position.instruments.simulated_hardware import simulator
from merlin_track_position.tracking.shift import estimate_shift


def framegrabber_message(raw, dt_ms):
    topic = "framegrabber/main "
    metadata = json.dumps(
        {"dtype": str(raw.dtype), "shape": list(raw.shape), "dt": str(int(dt_ms))}
    ).encode("utf-8")
    return topic.encode("utf-8") + metadata + b"\n" + raw.tobytes()


class FakeFramegrabberSocket:
    def __init__(self, messages):
        self.messages = list(messages)
        self.recv_count = 0
        self.receive_timeouts = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        del exc_type, exc, traceback

    def setsockopt(self, option, value):
        if option == framegrab_module.zmq.RCVTIMEO:
            self.receive_timeouts.append(value)

    def setsockopt_string(self, *args):
        del args

    def connect(self, address):
        del address

    def recv(self):
        self.recv_count += 1
        if self.messages:
            return self.messages.pop(0)
        raise framegrab_module.zmq.Again()


class FakeFramegrabberContext:
    def __init__(self, socket):
        self._socket = socket

    def socket(self, socket_type):
        del socket_type
        return self._socket


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


class FakeConvertedImage:
    def __init__(self, array):
        self.array = np.asarray(array)

    def GetArray(self):
        return self.array


class FakeImageFormatConverter:
    def __init__(self, converted):
        self.converted = np.asarray(converted)
        self.OutputPixelFormat = None
        self.OutputBitAlignment = None
        self.convert_calls = []

    def Convert(self, grab_result):
        self.convert_calls.append(grab_result)
        return FakeConvertedImage(self.converted)


class FakeSettings:
    def __init__(self, values):
        self.values = dict(values)

    def value(self, key, fallback=None):
        return self.values.get(str(key), fallback)


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

    def test_zero_position_basler_frame_matches_packaged_reference(self):
        with patch.object(constants, "IS_DAQ_PC", False):
            image = get_basler_image()

        data_path = (
            resources.files("merlin_track_position.instruments")
            / "data"
            / simulated_hardware.SYNTHETIC_CALIBRATION_FILE
        )
        with data_path.open("rb") as file:
            with np.load(file, allow_pickle=False) as archive:
                reference_cam1 = archive["reference_cam1"]
                roi = tuple(
                    float(np.asarray(archive[f"roi_cam1_{key}"]).reshape(-1)[0])
                    for key in ("x", "y", "width", "height")
                )

        self.assertEqual(
            image.shape,
            (constants.IMAGE_HEIGHT_CAM1, constants.IMAGE_WIDTH_CAM1),
        )
        np.testing.assert_allclose(
            crop_image_to_roi(image, roi),
            reference_cam1,
        )
        self.assertEqual(image.dtype, np.float64)

    def test_daq_mode_framegrabber_image_preserves_source_dtype(self):
        raw = np.arange(4 * 5, dtype=np.uint16).reshape(4, 5)
        message = framegrabber_message(
            raw,
            framegrab_module.LABVIEW_UNIX_EPOCH_OFFSET_MS + 1,
        )
        socket = FakeFramegrabberSocket([message])

        with (
            patch.object(constants, "IS_DAQ_PC", True),
            patch.object(constants, "IMAGE_HEIGHT_CAM0", 3),
            patch.object(constants, "IMAGE_WIDTH_CAM0", 4),
            patch(
                "merlin_track_position.instruments.framegrab.time.time_ns",
                return_value=0,
            ),
            patch(
                "merlin_track_position.instruments.framegrab.zmq.Context.instance",
                return_value=FakeFramegrabberContext(socket),
            ),
        ):
            image = get_framegrabber_image()

        np.testing.assert_array_equal(image, raw[:3, :4])
        self.assertEqual(image.dtype, np.uint16)
        self.assertEqual(socket.recv_count, 1)

    def test_daq_mode_framegrabber_discards_frames_not_later_than_request(self):
        stale = np.zeros((3, 4), dtype=np.uint16)
        fresh = stale + 7
        request_start_ms = framegrab_module.LABVIEW_UNIX_EPOCH_OFFSET_MS + 1000
        socket = FakeFramegrabberSocket(
            [
                framegrabber_message(stale, request_start_ms - 1),
                framegrabber_message(stale + 1, request_start_ms),
                framegrabber_message(fresh, request_start_ms + 1),
            ]
        )

        with (
            patch.object(constants, "IS_DAQ_PC", True),
            patch.object(constants, "IMAGE_HEIGHT_CAM0", 3),
            patch.object(constants, "IMAGE_WIDTH_CAM0", 4),
            patch(
                "merlin_track_position.instruments.framegrab.time.time_ns",
                return_value=1_000_000_000,
            ),
            patch(
                "merlin_track_position.instruments.framegrab.zmq.Context.instance",
                return_value=FakeFramegrabberContext(socket),
            ),
        ):
            image = get_framegrabber_image()

        np.testing.assert_array_equal(image, fresh)
        self.assertEqual(socket.recv_count, 3)

    def test_daq_mode_framegrabber_stack_uses_one_request_threshold(self):
        stale = np.zeros((3, 4), dtype=np.uint16)
        fresh0 = stale + 7
        duplicate_timestamp = stale + 8
        fresh1 = stale + 9
        request_start_ms = framegrab_module.LABVIEW_UNIX_EPOCH_OFFSET_MS + 1000
        socket = FakeFramegrabberSocket(
            [
                framegrabber_message(stale, request_start_ms),
                framegrabber_message(fresh0, request_start_ms + 1),
                framegrabber_message(duplicate_timestamp, request_start_ms + 1),
                framegrabber_message(fresh1, request_start_ms + 2),
            ]
        )

        with (
            patch.object(constants, "IS_DAQ_PC", True),
            patch.object(constants, "IMAGE_HEIGHT_CAM0", 3),
            patch.object(constants, "IMAGE_WIDTH_CAM0", 4),
            patch(
                "merlin_track_position.instruments.framegrab.time.time_ns",
                return_value=1_000_000_000,
            ) as time_ns,
            patch(
                "merlin_track_position.instruments.framegrab.zmq.Context.instance",
                return_value=FakeFramegrabberContext(socket),
            ),
        ):
            stack = get_framegrabber_image_stack(2)

        time_ns.assert_called_once()
        self.assertEqual(socket.recv_count, 4)
        np.testing.assert_array_equal(stack, np.stack([fresh0, fresh1]))

    def test_daq_mode_framegrabber_times_out_when_only_stale_frames_arrive(self):
        raw = np.zeros((3, 4), dtype=np.uint16)
        request_start_ms = framegrab_module.LABVIEW_UNIX_EPOCH_OFFSET_MS + 1000
        socket = FakeFramegrabberSocket(
            [
                framegrabber_message(raw, request_start_ms - 1),
                framegrabber_message(raw + 1, request_start_ms),
            ]
        )

        with (
            patch.object(constants, "IS_DAQ_PC", True),
            patch.object(constants, "IMAGE_HEIGHT_CAM0", 3),
            patch.object(constants, "IMAGE_WIDTH_CAM0", 4),
            patch(
                "merlin_track_position.instruments.framegrab.time.time_ns",
                return_value=1_000_000_000,
            ),
            patch(
                "merlin_track_position.instruments.framegrab.zmq.Context.instance",
                return_value=FakeFramegrabberContext(socket),
            ),
        ):
            with self.assertRaisesRegex(TimeoutError, "No fresh frame received"):
                get_framegrabber_image()

        self.assertEqual(socket.recv_count, 3)

    def test_basler_configuration_sets_manual_exposure_and_expected_aoi(self):
        config = CameraConfig(
            slot="cam1",
            source_type=SOURCE_BASLER,
            serial_number="abc",
            width=23,
            height=17,
            offset_x=3,
            offset_y=5,
            exposure_us=125000.0,
            pixel_format="BayerRG12",
            max_num_buffer=7,
        )
        camera = FakeBaslerCamera()

        with patch(
            "merlin_track_position.instruments.basler.genicam.IsWritable",
            lambda node: node.writable,
        ):
            basler._configure_camera(camera, config)

        self.assertEqual(camera.UserSetSelector.Value, "Default")
        self.assertTrue(camera.UserSetLoad.executed)
        self.assertEqual(camera.GainAuto.Value, "Off")
        self.assertEqual(camera.ExposureAuto.Value, "Off")
        self.assertEqual(camera.ExposureTimeAbs.Value, config.exposure_us)
        self.assertEqual(camera.OffsetX.Value, config.offset_x)
        self.assertEqual(camera.OffsetY.Value, config.offset_y)
        self.assertEqual(camera.Width.Value, config.width)
        self.assertEqual(camera.Height.Value, config.height)
        self.assertEqual(camera.PixelFormat.Value, config.pixel_format)

    def test_camera_config_from_settings_normalizes_slot_values(self):
        settings = FakeSettings(
            {
                "camera/cam0/source_type": SOURCE_BASLER,
                "camera/cam0/serial_number": "1234",
                "camera/cam0/width": "2592",
                "camera/cam0/height": "1944",
                "camera/cam0/output_mode": BASLER_OUTPUT_RGB8,
                "camera/cam0/display_transpose": "true",
                "camera/cam0/display_invert_x": "1",
                "camera/cam0/display_invert_y": "false",
            }
        )

        config = camera_config_from_settings(settings, "cam0")

        self.assertEqual(config.source_type, SOURCE_BASLER)
        self.assertEqual(config.serial_number, "1234")
        self.assertEqual(config.width, 2592)
        self.assertEqual(config.height, 1944)
        self.assertEqual(config.output_mode, BASLER_OUTPUT_RGB8)
        self.assertEqual(
            config.display,
            DisplayTransform(transpose=True, invert_x=True, invert_y=False),
        )

    def test_camera_pair_from_configs_maps_algorithm_slots_to_sources(self):
        configs = {
            "cam0": CameraConfig(
                slot="cam0",
                source_type=SOURCE_BASLER,
                serial_number="left",
                width=5,
                height=4,
            ),
            "cam1": CameraConfig(
                slot="cam1",
                source_type=SOURCE_FRAMEGRABBER,
                width=7,
                height=6,
            ),
        }

        pair = camera_pair_from_configs(configs)

        self.assertIsInstance(pair.cam0, BaslerCameraPlugin)
        self.assertIsInstance(pair.cam1, FramegrabberCameraPlugin)
        self.assertEqual(pair.cam0.config.serial_number, "left")
        self.assertEqual(pair.cam1.config.width, 7)

    def test_camera_config_mismatch_checks_only_present_metadata(self):
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
                serial_number="serial-a",
                width=6,
                height=5,
                exposure_us=125000.0,
            ),
        }
        metadata = camera_metadata(configs)
        self.assertEqual(metadata["camera_cam1_exposure_us"], 125000.0)
        metadata["camera_cam1_serial_number"] = "serial-b"

        self.assertEqual(
            camera_config_mismatches(metadata, configs),
            ["cam1 serial_number"],
        )
        metadata["camera_cam1_serial_number"] = "serial-a"
        metadata["camera_cam1_exposure_us"] = 100000.0
        self.assertEqual(
            camera_config_mismatches(metadata, configs),
            ["cam1 exposure_us"],
        )
        self.assertEqual(camera_config_mismatches({}, configs), [])

    def test_distinct_basler_configs_keep_distinct_sessions(self):
        raw0 = np.arange(6, dtype=np.uint16).reshape(2, 3)
        raw1 = raw0 + 10
        camera0 = FakeBaslerCamera([raw0])
        camera1 = FakeBaslerCamera([raw1])
        config0 = CameraConfig(
            slot="cam0",
            source_type=SOURCE_BASLER,
            serial_number="a",
            width=3,
            height=2,
        )
        config1 = CameraConfig(
            slot="cam1",
            source_type=SOURCE_BASLER,
            serial_number="b",
            width=3,
            height=2,
        )

        def camera_for_serial(serial):
            return {"a": camera0, "b": camera1}[serial]

        with (
            patch.object(constants, "IS_DAQ_PC", True),
            patch(
                "merlin_track_position.instruments.basler.genicam.IsWritable",
                lambda node: node.writable,
            ),
            patch.object(
                basler,
                "_get_camera_by_serial_number",
                side_effect=camera_for_serial,
            ),
        ):
            image0 = get_basler_image(config0)
            image1 = get_basler_image(config1)

        np.testing.assert_array_equal(image0, raw0)
        np.testing.assert_array_equal(image1, raw1)
        self.assertEqual(camera0.open_count, 1)
        self.assertEqual(camera1.open_count, 1)

    def test_basler_rgb_output_uses_pylon_converter(self):
        raw = np.arange(6, dtype=np.uint16).reshape(2, 3)
        converted = np.zeros((2, 3, 3), dtype=np.uint8)
        converted[..., 0] = 10
        camera = FakeBaslerCamera([raw])
        converter = FakeImageFormatConverter(converted)
        config = CameraConfig(
            slot="cam1",
            source_type=SOURCE_BASLER,
            serial_number="rgb",
            width=3,
            height=2,
            output_mode=BASLER_OUTPUT_RGB8,
            pixel_format="BayerRG12",
        )

        with (
            patch.object(constants, "IS_DAQ_PC", True),
            patch(
                "merlin_track_position.instruments.basler.genicam.IsWritable",
                lambda node: node.writable,
            ),
            patch.object(basler, "_get_camera_by_serial_number", return_value=camera),
            patch.object(
                basler.pylon,
                "ImageFormatConverter",
                return_value=converter,
            ),
        ):
            image = get_basler_image(config)

        np.testing.assert_array_equal(image, converted)
        self.assertEqual(len(converter.convert_calls), 1)

    def test_daq_mode_basler_image_accepts_expected_shape(self):
        raw = np.arange(6, dtype=np.uint16).reshape(2, 3)

        with (
            patch.object(constants, "IS_DAQ_PC", True),
            patch.object(constants, "IMAGE_HEIGHT_CAM1", 2),
            patch.object(constants, "IMAGE_WIDTH_CAM1", 3),
            patch.object(basler, "_session_for_config") as session_for_config,
        ):
            session_for_config.return_value.get_image.return_value = raw
            image = get_basler_image()

        np.testing.assert_array_equal(image, raw)
        self.assertEqual(image.shape, (2, 3))
        self.assertEqual(image.dtype, np.uint16)

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

        np.testing.assert_array_equal(image0, raw0)
        np.testing.assert_array_equal(image1, raw1)
        self.assertEqual(image0.dtype, np.uint16)
        self.assertEqual(image1.dtype, np.uint16)
        self.assertEqual(camera.open_count, 1)
        self.assertEqual(camera.start_grabbing_count, 1)
        self.assertEqual(camera.retrieve_result_count, 2)
        self.assertEqual(camera.close_count, 0)

    def test_daq_mode_basler_image_stack_reuses_open_camera(self):
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
            stack = get_basler_image_stack(2)

        np.testing.assert_array_equal(stack, np.stack([raw0, raw1]))
        self.assertEqual(camera.open_count, 1)
        self.assertEqual(camera.start_grabbing_count, 1)
        self.assertEqual(camera.retrieve_result_count, 2)
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

        np.testing.assert_array_equal(image0, raw0)
        np.testing.assert_array_equal(image1, raw1)
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

        np.testing.assert_array_equal(image, raw)
        self.assertEqual(image.dtype, np.uint16)
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
            patch.object(basler, "_session_for_config") as session_for_config,
        ):
            session_for_config.return_value.get_image.return_value = raw
            with self.assertRaisesRegex(RuntimeError, "does not match configured"):
                get_basler_image()

    def test_capture_image_stack_discards_repeated_fresh_camera_frames(self):
        stale_cam0 = np.zeros((2, 3), dtype=np.uint16)
        fresh_cam0 = stale_cam0 + 1
        cam1 = np.ones((3, 4), dtype=np.uint16)
        cam0_captures = [stale_cam0, stale_cam0, fresh_cam0]
        cam1_captures = [cam1, cam1 + 2]

        def capture_cam0():
            return cam0_captures.pop(0)

        def capture_cam1():
            return cam1_captures.pop(0)

        camera_pair = CameraPairPlugin(
            CallableCameraPlugin(
                "cam0",
                capture_cam0,
                fresh_frame_timeout_s=1.0,
                use_image_content_key=True,
            ),
            CallableCameraPlugin("cam1", capture_cam1, fresh_frame_timeout_s=1.0),
        )
        with patch("merlin_track_position.instruments.cameras.time.sleep") as sleep:
            stack_cam0, stack_cam1 = capture_image_stack(camera_pair, 2)

        self.assertEqual(len(cam0_captures), 0)
        self.assertEqual(len(cam1_captures), 0)
        sleep.assert_called_once()
        np.testing.assert_array_equal(stack_cam0, np.stack([stale_cam0, fresh_cam0]))
        np.testing.assert_array_equal(stack_cam1, np.stack([cam1, cam1 + 2]))

    def test_capture_image_stack_runs_camera_stacks_in_parallel(self):
        image_cam0 = np.zeros((2, 3), dtype=np.uint16)
        image_cam1 = np.ones((3, 4), dtype=np.uint16)
        barrier = threading.Barrier(2, timeout=1.0)

        def capture_cam0():
            barrier.wait()
            return image_cam0

        def capture_cam1():
            barrier.wait()
            return image_cam1

        camera_pair = CameraPairPlugin(
            CallableCameraPlugin("cam0", capture_cam0),
            CallableCameraPlugin("cam1", capture_cam1),
        )

        stack_cam0, stack_cam1 = capture_image_stack(camera_pair, 1)

        np.testing.assert_array_equal(stack_cam0, image_cam0[np.newaxis, :, :])
        np.testing.assert_array_equal(stack_cam1, image_cam1[np.newaxis, :, :])

    def test_capture_image_stack_rejects_empty_camera_list(self):
        with self.assertRaisesRegex(ValueError, "at least one camera plugin"):
            capture_image_stack((), 1)

    def test_capture_image_stack_times_out_on_repeated_fresh_camera_frame(self):
        image_cam0 = np.zeros((2, 3), dtype=np.uint16)
        image_cam1 = np.ones((3, 4), dtype=np.uint16)
        capture_count = {"cam0": 0, "cam1": 0}

        def capture_cam0():
            capture_count["cam0"] += 1
            return image_cam0

        def capture_cam1():
            capture_count["cam1"] += 1
            return image_cam1

        camera_pair = CameraPairPlugin(
            CallableCameraPlugin(
                "cam0",
                capture_cam0,
                fresh_frame_timeout_s=0.0,
                use_image_content_key=True,
            ),
            CallableCameraPlugin("cam1", capture_cam1, fresh_frame_timeout_s=0.0),
        )
        with self.assertRaisesRegex(TimeoutError, "cam0"):
            capture_image_stack(camera_pair, 2)
        self.assertEqual(capture_count["cam0"], 2)
        self.assertLessEqual(capture_count["cam1"], 2)

    def test_hardware_camera_plugins_capture_stacks_once_per_camera(self):
        stack_cam0 = np.stack(
            [
                np.zeros((2, 3), dtype=np.uint16),
                np.zeros((2, 3), dtype=np.uint16),
            ]
        )
        stack_cam1 = np.stack(
            [
                np.ones((3, 4), dtype=np.uint16),
                np.ones((3, 4), dtype=np.uint16) + 1,
            ]
        )
        camera_pair = CameraPairPlugin(
            FramegrabberCameraPlugin(),
            BaslerCameraPlugin(),
        )
        with (
            patch(
                "merlin_track_position.instruments.cameras.get_framegrabber_image_stack",
                return_value=stack_cam0,
            ) as get_cam0_stack,
            patch(
                "merlin_track_position.instruments.cameras.get_basler_image_stack",
                return_value=stack_cam1,
            ) as get_cam1_stack,
            patch("merlin_track_position.instruments.cameras.time.sleep") as sleep,
        ):
            captured_cam0, captured_cam1 = capture_image_stack(camera_pair, 2)

        get_cam0_stack.assert_called_once_with(2, timeout_ms=10000)
        self.assertEqual(get_cam1_stack.call_args.args[0], 2)
        sleep.assert_not_called()
        np.testing.assert_array_equal(captured_cam0, stack_cam0)
        np.testing.assert_array_equal(captured_cam1, stack_cam1)

    def test_capture_image_stack_works_with_cropped_camera_callables(self):
        stale_cam0 = np.arange(4 * 5).reshape(4, 5)
        fresh_cam0 = stale_cam0 + 100
        image_cam1_a = np.arange(6 * 7).reshape(6, 7)
        image_cam1_b = image_cam1_a + 100
        cam0_captures = [stale_cam0, stale_cam0, fresh_cam0]
        cam1_captures = [image_cam1_a, image_cam1_b]

        def capture_cam0():
            return cam0_captures.pop(0)

        def capture_cam1():
            return cam1_captures.pop(0)

        camera_pair = CameraPairPlugin(
            CallableCameraPlugin(
                "cam0",
                capture_cam0,
                fresh_frame_timeout_s=1.0,
                use_image_content_key=True,
            ),
            CallableCameraPlugin("cam1", capture_cam1, fresh_frame_timeout_s=1.0),
        ).cropped((1.0, 1.0, 3.0, 2.0), (2.0, 3.0, 2.0, 2.0))

        with patch("merlin_track_position.instruments.cameras.time.sleep"):
            stack_cam0, stack_cam1 = capture_image_stack(camera_pair, 2)

        self.assertEqual(len(cam0_captures), 0)
        self.assertEqual(len(cam1_captures), 0)
        np.testing.assert_array_equal(
            stack_cam0,
            np.stack([stale_cam0[1:3, 1:4], fresh_cam0[1:3, 1:4]]),
        )
        np.testing.assert_array_equal(
            stack_cam1,
            np.stack([image_cam1_a[3:5, 2:4], image_cam1_b[3:5, 2:4]]),
        )

    def test_cropped_camera_pair_keeps_full_display_images(self):
        full_cam0 = np.arange(4 * 5).reshape(4, 5)
        full_cam1 = np.arange(6 * 7).reshape(6, 7)
        camera_pair = CameraPairPlugin(
            CallableCameraPlugin("cam0", lambda: full_cam0),
            CallableCameraPlugin("cam1", lambda: full_cam1),
        ).cropped((1.0, 1.0, 3.0, 2.0), (2.0, 3.0, 2.0, 2.0))

        image_stacks, display_stacks = capture_image_and_display_stacks(
            camera_pair,
            1,
        )

        np.testing.assert_array_equal(image_stacks[0][0], full_cam0[1:3, 1:4])
        np.testing.assert_array_equal(image_stacks[1][0], full_cam1[3:5, 2:4])
        np.testing.assert_array_equal(display_stacks[0][0], full_cam0)
        np.testing.assert_array_equal(display_stacks[1][0], full_cam1)

    def test_default_camera_pair_allows_identical_development_images(self):
        with patch.object(constants, "IS_DAQ_PC", False):
            stack_cam0, stack_cam1 = capture_image_stack(default_camera_pair(), 2)

        self.assertEqual(stack_cam0.shape[0], 2)
        self.assertEqual(stack_cam1.shape[0], 2)
        np.testing.assert_array_equal(stack_cam0[0], stack_cam0[1])
        np.testing.assert_array_equal(stack_cam1[0], stack_cam1[1])

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

    def test_cropped_camera_pair_plugin_wraps_base_cameras(self):
        image_cam0 = np.arange(4 * 5).reshape(4, 5)
        image_cam1 = np.arange(6 * 7).reshape(6, 7)

        camera_pair = CameraPairPlugin(
            CallableCameraPlugin("cam0", lambda: image_cam0),
            CallableCameraPlugin("cam1", lambda: image_cam1),
        ).cropped((1.0, 1.0, 3.0, 2.0), (2.0, 3.0, 2.0, 2.0))

        cropped_cam0, cropped_cam1 = camera_pair.capture_pair()

        np.testing.assert_array_equal(cropped_cam0, image_cam0[1:3, 1:4])
        np.testing.assert_array_equal(cropped_cam1, image_cam1[3:5, 2:4])

    def test_simulated_frame_shift_tracks_motor_positions(self):
        command_offset_um = np.array([60.0, -30.0, 20.0], dtype=np.float64)

        with (
            patch.object(constants, "IS_DAQ_PC", False),
            patch("merlin_track_position.instruments.simulated_hardware.time.sleep"),
        ):
            reference = get_framegrabber_image()
            move_motors_and_wait(("x", "y", "z"), command_offset_um * 1e-3)
            shifted = get_framegrabber_image()

        expected_shift_px = (
            simulator.get_command_um_to_pixel("cam0") @ command_offset_um
        )
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

    def test_simulated_basler_shift_tracks_motor_positions(self):
        command_offset_um = np.array([60.0, -30.0, 20.0], dtype=np.float64)

        with (
            patch.object(constants, "IS_DAQ_PC", False),
            patch("merlin_track_position.instruments.simulated_hardware.time.sleep"),
        ):
            reference = get_basler_image()
            move_motors_and_wait(("x", "y", "z"), command_offset_um * 1e-3)
            shifted = get_basler_image()

        expected_shift_px = (
            simulator.get_command_um_to_pixel("cam1") @ command_offset_um
        )
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

"""Placeholder access layer for the second, Basler-based camera."""

from __future__ import annotations

import atexit
import logging
import threading

import numpy as np
import numpy.typing as npt
from pypylon import genicam, pylon

from merlin_track_position import constants
from merlin_track_position.instruments.camera_config import (
    BASLER_OUTPUT_BGR8,
    BASLER_OUTPUT_NATIVE,
    BASLER_OUTPUT_RGB8,
    CameraConfig,
    default_camera_config,
)
from merlin_track_position.instruments.simulated_hardware import simulator

logger = logging.getLogger("merlin_track_position.instruments.basler")


def _close_camera(camera: pylon.InstantCamera) -> None:
    try:
        if hasattr(camera, "IsGrabbing") and camera.IsGrabbing():
            camera.StopGrabbing()
    except Exception:
        logger.exception("Failed to stop Basler camera grabbing")

    try:
        camera.Close()
    except Exception:
        logger.exception("Failed to close Basler camera")


def _configure_camera(
    camera: pylon.InstantCamera,
    config: CameraConfig | None = None,
) -> None:
    if config is None:
        config = default_camera_config("cam1")
    try:
        if not genicam.IsWritable(camera.UserSetSelector):
            raise genicam.RuntimeException("UserSetSelector is not writable")
        camera.UserSetSelector.Value = "Default"

        if not genicam.IsWritable(camera.UserSetLoad):
            raise genicam.RuntimeException("UserSetLoad is not executable")
        camera.UserSetLoad.Execute()

        # # Flip image.
        # if genicam.IsWritable(camera.ReverseX):
        #     camera.ReverseX.Value = True
        # if genicam.IsWritable(camera.ReverseY):
        #     camera.ReverseY.Value = False

        if genicam.IsWritable(camera.GainAuto):
            logger.debug("Disabling automatic gain control.")
            camera.GainAuto.Value = "Off"

        if genicam.IsWritable(camera.GammaSelector):
            logger.debug("Setting gamma selector to sRGB.")
            camera.GammaSelector.Value = "sRGB"

        if genicam.IsWritable(camera.ExposureAuto):
            logger.debug("Disabling automatic exposure control.")
            camera.ExposureAuto.Value = "Off"
        logger.debug("Setting exposure time.")
        if not genicam.IsWritable(camera.ExposureTimeAbs):
            raise genicam.RuntimeException("ExposureTimeAbs is not writable")
        camera.ExposureTimeAbs.Value = float(config.exposure_us)

        if genicam.IsWritable(camera.GammaEnable):
            logger.debug("Enabling gamma correction.")
            # camera.GammaEnable.Value = False
            camera.GammaEnable.Value = True

        for node, value, name in (
            (camera.OffsetX, int(config.offset_x), "OffsetX"),
            (camera.OffsetY, int(config.offset_y), "OffsetY"),
            (camera.Width, int(config.width), "Width"),
            (camera.Height, int(config.height), "Height"),
        ):
            if not genicam.IsWritable(node):
                raise genicam.RuntimeException(f"{name} is not writable")
            node.Value = value

        # Set the pixel data format.
        if not genicam.IsWritable(camera.PixelFormat):
            raise genicam.RuntimeException("PixelFormat is not writable")
        camera.PixelFormat.Value = str(config.pixel_format)

    except genicam.GenericException as e:
        raise genicam.RuntimeException(f"Could not apply configuration: {e}") from e


def _get_camera_by_serial_number(serial_number: str) -> pylon.InstantCamera:
    tlf: pylon.TlFactory = pylon.TlFactory.GetInstance()
    for device in tlf.EnumerateDevices():
        if device.GetSerialNumber().strip() == serial_number.strip():
            device_info = pylon.CDeviceInfo()
            device_info.SetFullName(device.GetFullName())
            return pylon.InstantCamera(
                tlf.CreateDevice(device_info),
            )
    raise ValueError(f"No camera found with serial number {serial_number}")


class _BaslerCameraSession:
    def __init__(self, config: CameraConfig) -> None:
        self._config = config
        self._lock = threading.RLock()
        self._camera: pylon.InstantCamera | None = None
        self._converter: pylon.ImageFormatConverter | None = None
        self._latest_image: npt.NDArray | None = None

    def get_image(self, timeout_ms: int = 5000) -> npt.NDArray:
        return self.get_images(1, timeout_ms=timeout_ms)[0]

    def get_images(
        self,
        frame_count: int,
        timeout_ms: int = 5000,
    ) -> npt.NDArray:
        frame_count = _validate_frame_count(frame_count)
        with self._lock:
            camera = self._ensure_camera()
            try:
                images = [
                    self._retrieve_image(camera, timeout_ms, self._converter)
                    for _frame_index in range(frame_count)
                ]
            except Exception:
                self.close()
                raise
            stack = np.stack(images, axis=0)
            self._latest_image = stack[-1].copy()
            return stack

    def close(self) -> None:
        with self._lock:
            camera = self._camera
            self._camera = None
            self._converter = None
            self._latest_image = None
            if camera is not None:
                _close_camera(camera)

    def _ensure_camera(self) -> pylon.InstantCamera:
        if self._camera is not None:
            return self._camera

        camera = _get_camera_by_serial_number(self._config.serial_number)
        camera.Open()
        try:
            _configure_camera(camera, self._config)
            camera.MaxNumBuffer = int(self._config.max_num_buffer)
            self._converter = _image_converter(self._config)
            camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
        except Exception:
            _close_camera(camera)
            self._converter = None
            raise

        self._camera = camera
        return camera

    @staticmethod
    def _retrieve_image(
        camera: pylon.InstantCamera,
        timeout_ms: int,
        converter: pylon.ImageFormatConverter | None,
    ) -> npt.NDArray:
        grab_result = camera.RetrieveResult(
            int(timeout_ms),
            pylon.TimeoutHandling_ThrowException,
        )
        try:
            if not grab_result.GrabSucceeded():
                raise RuntimeError(
                    "Image grab failed: "
                    f"{grab_result.GetErrorCode()} {grab_result.GetErrorDescription()}"
                )
            if converter is not None:
                converted = converter.Convert(grab_result)
                return converted.GetArray().copy()
            return grab_result.GetArray(raw=False).copy()
        finally:
            grab_result.Release()


_SESSIONS: dict[tuple[object, ...], _BaslerCameraSession] = {}


def _image_converter(config: CameraConfig) -> pylon.ImageFormatConverter | None:
    if config.output_mode == BASLER_OUTPUT_NATIVE:
        return None
    converter = pylon.ImageFormatConverter()
    if config.output_mode == BASLER_OUTPUT_RGB8:
        converter.OutputPixelFormat = pylon.PixelType_RGB8packed
    elif config.output_mode == BASLER_OUTPUT_BGR8:
        converter.OutputPixelFormat = pylon.PixelType_BGR8packed
    else:
        raise ValueError(f"unsupported Basler output mode: {config.output_mode!r}")
    converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned
    return converter


def _session_for_config(config: CameraConfig) -> _BaslerCameraSession:
    key = config.session_key
    session = _SESSIONS.get(key)
    if session is None:
        session = _BaslerCameraSession(config)
        _SESSIONS[key] = session
    return session


def close_basler_camera(config: CameraConfig | None = None) -> None:
    """Close cached Basler camera sessions."""
    if config is None:
        sessions = list(_SESSIONS.values())
        _SESSIONS.clear()
        for session in sessions:
            session.close()
        return

    session = _SESSIONS.pop(config.session_key, None)
    if session is not None:
        session.close()


def get_basler_image(config: CameraConfig | None = None) -> npt.NDArray:
    """Return the latest image from a Basler camera."""
    if config is None:
        config = default_camera_config("cam1")
    if not constants.IS_DAQ_PC:
        image = _simulated_image(config)
    else:
        image = _session_for_config(config).get_image()
    _validate_image_shape(image, config)
    return np.asarray(image).copy()


def get_basler_image_stack(
    frame_count: int,
    config: CameraConfig | None = None,
) -> npt.NDArray:
    """Return consecutive images from a Basler camera."""
    frame_count = _validate_frame_count(frame_count)
    if config is None:
        config = default_camera_config("cam1")
    if not constants.IS_DAQ_PC:
        images = np.stack(
            [_simulated_image(config) for _ in range(frame_count)],
            axis=0,
        )
    else:
        images = _session_for_config(config).get_images(frame_count)
    images = np.asarray(images)
    if images.ndim < 3:
        raise RuntimeError(
            f"Basler image stack must be at least 3D, got {images.shape!r}"
        )
    _validate_image_shape(images[0], config)
    return images.copy()


def _simulated_image(config: CameraConfig) -> npt.NDArray:
    if config.slot == "cam0":
        image = simulator.get_framegrabber_image()
    else:
        image = simulator.get_basler_image()
    return np.asarray(image)[
        : int(config.height),
        : int(config.width),
        ...,
    ].copy()


def _validate_image_shape(image: npt.NDArray, config: CameraConfig) -> None:
    expected_shape = (int(config.height), int(config.width))
    image_shape = tuple(np.asarray(image).shape)
    if image_shape[:2] != expected_shape:
        raise RuntimeError(
            f"Basler image shape {image_shape} does not match configured "
            f"height/width {expected_shape}; update the camera configuration "
            "or AOI settings."
        )
    if config.output_mode != BASLER_OUTPUT_NATIVE and image_shape[2:] != (3,):
        raise RuntimeError(
            f"Basler converted color image must have shape "
            f"({config.height}, {config.width}, 3), got {image_shape!r}"
        )


def _validate_frame_count(frame_count: int) -> int:
    value = int(frame_count)
    if value < 1:
        raise ValueError("frame_count must be >= 1")
    return value


atexit.register(close_basler_camera)

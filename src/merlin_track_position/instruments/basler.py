"""Placeholder access layer for the second, Basler-based camera."""

from __future__ import annotations

import atexit
import logging
import threading

import numpy as np
import numpy.typing as npt
from pypylon import genicam, pylon

from merlin_track_position import constants
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


def _configure_camera(camera: pylon.InstantCamera) -> None:
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
        camera.ExposureTimeAbs.Value = constants.BASLER_EXPOSURE

        if genicam.IsWritable(camera.GammaEnable):
            logger.debug("Disabling gamma correction.")
            camera.GammaEnable.Value = False
            # camera.GammaEnable.Value = True

        for node, value, name in (
            (camera.OffsetX, int(camera.OffsetX.Min), "OffsetX"),
            (camera.OffsetY, int(camera.OffsetY.Min), "OffsetY"),
            (camera.Width, constants.IMAGE_WIDTH_CAM1, "Width"),
            (camera.Height, constants.IMAGE_HEIGHT_CAM1, "Height"),
        ):
            if not genicam.IsWritable(node):
                raise genicam.RuntimeException(f"{name} is not writable")
            node.Value = value

        # Set the pixel data format.
        if not genicam.IsWritable(camera.PixelFormat):
            raise genicam.RuntimeException("PixelFormat is not writable")
        camera.PixelFormat.Value = "Mono12"

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
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._camera: pylon.InstantCamera | None = None
        self._latest_image: npt.NDArray | None = None

    def get_image(self, timeout_ms: int = 5000) -> npt.NDArray:
        with self._lock:
            camera = self._ensure_camera()
            try:
                image = self._retrieve_image(camera, timeout_ms)
            except Exception:
                self.close()
                raise
            self._latest_image = image
            return image

    def close(self) -> None:
        with self._lock:
            camera = self._camera
            self._camera = None
            self._latest_image = None
            if camera is not None:
                _close_camera(camera)

    def _ensure_camera(self) -> pylon.InstantCamera:
        if self._camera is not None:
            return self._camera

        camera = _get_camera_by_serial_number(constants.BASLER_CAMERA_SERIAL)
        camera.Open()
        try:
            _configure_camera(camera)
            camera.MaxNumBuffer = 5
            camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
        except Exception:
            _close_camera(camera)
            raise

        self._camera = camera
        return camera

    @staticmethod
    def _retrieve_image(
        camera: pylon.InstantCamera,
        timeout_ms: int,
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
            return grab_result.GetArray(raw=False).copy()
        finally:
            grab_result.Release()


_SESSION = _BaslerCameraSession()


def close_basler_camera() -> None:
    """Close the cached Basler camera session if it is open."""
    _SESSION.close()


def _grab_single_image(timeout_ms: int = 5000):
    return _SESSION.get_image(timeout_ms)


def get_basler_image() -> npt.NDArray:
    """Return the latest image from camera 1."""
    if not constants.IS_DAQ_PC:
        return simulator.get_basler_image()
    image = _grab_single_image()
    expected_shape = (constants.IMAGE_HEIGHT_CAM1, constants.IMAGE_WIDTH_CAM1)
    if image.shape != expected_shape:
        raise RuntimeError(
            f"Basler image shape {image.shape} does not match constants "
            f"{expected_shape}; update IMAGE_HEIGHT_CAM1/IMAGE_WIDTH_CAM1 "
            "or the camera AOI configuration."
        )
    return image.astype(np.float64)


atexit.register(close_basler_camera)

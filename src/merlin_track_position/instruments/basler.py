"""Placeholder access layer for the second, Basler-based camera."""

from __future__ import annotations

import logging

import numpy as np
import numpy.typing as npt
from pypylon import genicam, pylon
from qtpy import QtCore

from merlin_track_position import constants
from merlin_track_position.instruments.simulated_hardware import simulator

logger = logging.getLogger("merlin_track_position.instruments.basler")


class CameraConfiguration(pylon.ConfigurationEventHandler, QtCore.QObject):
    def OnOpened(self, camera):
        try:
            # # Maximize the Image AOI.
            # if genicam.IsWritable(camera.OffsetX):
            #     camera.OffsetX.Value = camera.OffsetX.Min
            # if genicam.IsWritable(camera.OffsetY):
            #     camera.OffsetY.Value = camera.OffsetY.Min
            # camera.Width.Value = camera.Width.Max
            # camera.Height.Value = camera.Height.Max

            camera.UserSetSelector.Value = "Default"
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

            # if genicam.IsWritable(camera.ExposureAuto):
            #     camera.ExposureAuto.Value = "Off"
            if genicam.IsWritable(camera.ExposureTime):
                logger.debug("Setting exposure time.")
                camera.ExposureTime.Value = constants.BASLER_EXPOSURE_US

            if genicam.IsWritable(camera.GammaEnable):
                logger.debug("Enabling gamma correction.")
                camera.GammaEnable.Value = True

            # Set the pixel data format.
            camera.PixelFormat.Value = "Mono10"

        except genicam.GenericException as e:
            raise genicam.RuntimeException(
                "Could not apply configuration."
                "GenICam::GenericException caught in OnOpened method"
            ) from e


def _get_camera_by_serial_number(serial_number: str) -> pylon.InstantCamera:
    tlf: pylon.TlFactory = pylon.TlFactory.GetInstance()
    for device in tlf.EnumerateDevices():
        if device.GetSerialNumber().strip() == serial_number.strip():
            return pylon.InstantCamera(
                tlf.CreateDevice(device),
            )
    raise ValueError(f"No camera found with serial number {serial_number}")


def _grab_single_image():
    camera = _get_camera_by_serial_number(constants.BASLER_CAMERA_ID)
    camera.RegisterConfiguration(
        CameraConfiguration(), pylon.RegistrationMode_Append, pylon.Cleanup_Delete
    )
    camera.Open()

    try:
        camera.StartGrabbingMax(1, pylon.GrabStrategy_LatestImageOnly)
        grab_result = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
        try:
            if not grab_result.GrabSucceeded():
                raise RuntimeError(
                    f"Image grab failed: {grab_result.ErrorCode} {grab_result.ErrorDescription}"
                )
            return grab_result.GetArray(raw=False)
        finally:
            grab_result.Release()
    finally:
        camera.Close()


def get_basler_image() -> npt.NDArray:
    """Return the latest image from camera 1."""
    if not constants.IS_DAQ_PC:
        return simulator.get_basler_image()
    return _grab_single_image().astype(np.float64)

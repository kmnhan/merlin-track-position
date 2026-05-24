"""Placeholder access layer for the second, Basler-based camera."""

from __future__ import annotations

import atexit
from dataclasses import dataclass
import logging
import math
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


@dataclass(frozen=True)
class BaslerDevice:
    serial_number: str
    model_name: str
    full_name: str


@dataclass(frozen=True)
class BaslerValueRange:
    minimum: float
    maximum: float
    increment: float


@dataclass(frozen=True)
class BaslerCameraCapabilities:
    device: BaslerDevice
    width: BaslerValueRange
    height: BaslerValueRange
    offset_x: BaslerValueRange
    offset_y: BaslerValueRange
    exposure_us: BaslerValueRange
    pixel_formats: tuple[str, ...]


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
        _load_default_user_set(camera)

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
        _set_exposure_us(camera, float(config.exposure_us))

        if genicam.IsWritable(camera.GammaEnable):
            logger.debug("Enabling gamma correction.")
            # camera.GammaEnable.Value = False
            camera.GammaEnable.Value = True

        for name in ("CenterX", "CenterY"):
            try:
                node = getattr(camera, name)
            except (AttributeError, genicam.GenericException):
                continue
            if node is not None and genicam.IsWritable(node):
                node.Value = False

        for node in (camera.OffsetX, camera.OffsetY):
            if genicam.IsWritable(node):
                node.Value = int(node.Min)

        for node, value, name in (
            (camera.Width, int(config.width), "Width"),
            (camera.Height, int(config.height), "Height"),
        ):
            if not genicam.IsWritable(node):
                raise genicam.RuntimeException(f"{name} is not writable")
            node.Value = value

        for node, value, name in (
            (camera.OffsetX, int(config.offset_x), "OffsetX"),
            (camera.OffsetY, int(config.offset_y), "OffsetY"),
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


def _load_default_user_set(camera: pylon.InstantCamera) -> None:
    if not genicam.IsWritable(camera.UserSetSelector):
        raise genicam.RuntimeException("UserSetSelector is not writable")
    camera.UserSetSelector.Value = "Default"

    if not genicam.IsWritable(camera.UserSetLoad):
        raise genicam.RuntimeException("UserSetLoad is not executable")
    camera.UserSetLoad.Execute()


def _exposure_node(camera: pylon.InstantCamera) -> object:
    for name in ("ExposureTimeAbs", "ExposureTime"):
        try:
            node = getattr(camera, name)
        except (AttributeError, genicam.GenericException):
            continue
        if node is not None:
            return node
    raise genicam.RuntimeException("ExposureTimeAbs/ExposureTime is not available")


def _set_exposure_us(camera: pylon.InstantCamera, exposure_us: float) -> None:
    logger.debug("Setting exposure time.")
    node = _exposure_node(camera)
    if not genicam.IsWritable(node):
        raise genicam.RuntimeException("ExposureTimeAbs/ExposureTime is not writable")
    node.Value = exposure_us


def list_basler_devices() -> tuple[BaslerDevice, ...]:
    """Return currently connected Basler devices."""
    devices = pylon.TlFactory.GetInstance().EnumerateDevices()
    return tuple(_device_metadata(device) for device in devices)


def read_basler_capabilities(serial_number: str) -> BaslerCameraCapabilities:
    """Read live writable configuration constraints from a connected Basler camera."""
    device = _device_by_serial_number(serial_number)
    camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateDevice(device))
    camera.Open()
    try:
        _load_default_user_set(camera)
        return _camera_capabilities(camera, _device_metadata(device))
    finally:
        _close_camera(camera)


def validate_basler_config(config: CameraConfig) -> None:
    """Raise ValueError when a Basler config is invalid for its connected camera."""
    device = _device_by_serial_number(config.serial_number)
    camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateDevice(device))
    camera.Open()
    try:
        _load_default_user_set(camera)
        capabilities = _camera_capabilities(camera, _device_metadata(device))
        errors = _basler_config_errors(config, capabilities)
        if errors:
            raise ValueError("; ".join(errors))
        _configure_camera(camera, config)
    finally:
        _close_camera(camera)


def _basler_config_errors(
    config: CameraConfig,
    capabilities: BaslerCameraCapabilities,
) -> list[str]:
    errors = []
    if config.model_name and config.model_name != capabilities.device.model_name:
        errors.append(
            "model_name must match selected camera "
            f"{capabilities.device.model_name!r}"
        )
    for name, value, value_range in (
        ("width", config.width, capabilities.width),
        ("height", config.height, capabilities.height),
    ):
        if not _value_in_range(float(value), value_range):
            errors.append(
                f"{name}={value} is outside valid range "
                f"{_range_text(value_range, integer=True)}"
            )
    for name, value, value_range in (
        ("offset_x", config.offset_x, capabilities.offset_x),
        ("offset_y", config.offset_y, capabilities.offset_y),
    ):
        if (
            float(value) < value_range.minimum
            or not _value_matches_increment(float(value), value_range)
        ):
            errors.append(
                f"{name}={value} is outside valid range "
                f"{_range_text(value_range, integer=True)}"
            )
    if not _value_in_range(float(config.exposure_us), capabilities.exposure_us):
        errors.append(
            f"exposure_us={config.exposure_us:g} is outside valid range "
            f"{_range_text(capabilities.exposure_us, integer=False)}"
        )
    if (
        capabilities.pixel_formats
        and config.pixel_format not in capabilities.pixel_formats
    ):
        errors.append(
            "pixel_format must be one of " + ", ".join(capabilities.pixel_formats)
        )
    return errors


def _value_in_range(value: float, value_range: BaslerValueRange) -> bool:
    if value < value_range.minimum or value > value_range.maximum:
        return False
    return _value_matches_increment(value, value_range)


def _value_matches_increment(value: float, value_range: BaslerValueRange) -> bool:
    increment = value_range.increment
    if increment <= 0:
        return True
    steps = (value - value_range.minimum) / increment
    return math.isclose(steps, round(steps), rel_tol=1e-9, abs_tol=1e-9)


def _range_text(value_range: BaslerValueRange, *, integer: bool) -> str:
    if integer:
        return (
            f"{int(value_range.minimum)}..{int(value_range.maximum)} "
            f"step {int(value_range.increment)}"
        )
    if value_range.increment <= 0:
        return f"{value_range.minimum:g}..{value_range.maximum:g}"
    return (
        f"{value_range.minimum:g}..{value_range.maximum:g} "
        f"step {value_range.increment:g}"
    )


def _camera_capabilities(
    camera: pylon.InstantCamera,
    device: BaslerDevice,
) -> BaslerCameraCapabilities:
    return BaslerCameraCapabilities(
        device=device,
        width=_value_range(camera.Width, default_increment=1.0),
        height=_value_range(camera.Height, default_increment=1.0),
        offset_x=_value_range(camera.OffsetX, default_increment=1.0),
        offset_y=_value_range(camera.OffsetY, default_increment=1.0),
        exposure_us=_value_range(_exposure_node(camera), default_increment=0.0),
        pixel_formats=_enum_entries(camera.PixelFormat),
    )


def _value_range(node: object, *, default_increment: float) -> BaslerValueRange:
    increment = default_increment
    if hasattr(node, "HasInc"):
        try:
            if node.HasInc():
                increment = float(node.Inc)
        except genicam.GenericException:
            increment = default_increment
    elif hasattr(node, "Inc"):
        increment = float(node.Inc)
    return BaslerValueRange(
        minimum=float(getattr(node, "Min")),
        maximum=float(getattr(node, "Max")),
        increment=max(increment, 0.0),
    )


def _enum_entries(node: object) -> tuple[str, ...]:
    if hasattr(node, "GetEntries"):
        entries = []
        for entry in node.GetEntries():
            if genicam.IsAvailable(entry) and genicam.IsReadable(entry):
                entries.append(str(entry.GetSymbolic()))
        if entries:
            return tuple(entries)
    if hasattr(node, "Symbolics"):
        return tuple(str(value) for value in getattr(node, "Symbolics"))
    if hasattr(node, "GetSymbolics"):
        return tuple(str(value) for value in node.GetSymbolics())
    current = str(getattr(node, "Value", ""))
    return (current,) if current else ()


def _device_metadata(device: object) -> BaslerDevice:
    model_name = (
        str(device.GetModelName()).strip()
        if not hasattr(device, "IsModelNameAvailable") or device.IsModelNameAvailable()
        else ""
    )
    full_name = (
        str(device.GetFullName()).strip()
        if not hasattr(device, "IsFullNameAvailable") or device.IsFullNameAvailable()
        else ""
    )
    return BaslerDevice(
        serial_number=str(device.GetSerialNumber()).strip(),
        model_name=model_name,
        full_name=full_name,
    )


def _device_by_serial_number(serial_number: str) -> object:
    for device in pylon.TlFactory.GetInstance().EnumerateDevices():
        if str(device.GetSerialNumber()).strip() == serial_number.strip():
            return device
    raise ValueError(f"No camera found with serial number {serial_number}")


def _get_camera_by_serial_number(serial_number: str) -> pylon.InstantCamera:
    device = _device_by_serial_number(serial_number)
    return pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateDevice(device))


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
            camera.MaxNumBuffer.Value = int(self._config.max_num_buffer)
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

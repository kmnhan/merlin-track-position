"""Configuration for mapping algorithm camera slots to hardware sources."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping

from merlin_track_position import constants

CAMERA_SLOTS = ("cam0", "cam1")
SOURCE_FRAMEGRABBER = "framegrabber"
SOURCE_BASLER = "basler"
SOURCE_SIMULATED = "simulated"
SOURCE_TYPES = (SOURCE_FRAMEGRABBER, SOURCE_BASLER, SOURCE_SIMULATED)

BASLER_OUTPUT_NATIVE = "native"
BASLER_OUTPUT_RGB8 = "rgb8"
BASLER_OUTPUT_BGR8 = "bgr8"
BASLER_OUTPUT_MODES = (BASLER_OUTPUT_NATIVE, BASLER_OUTPUT_RGB8, BASLER_OUTPUT_BGR8)


@dataclass(frozen=True)
class DisplayTransform:
    transpose: bool = False
    invert_x: bool = False
    invert_y: bool = True


@dataclass(frozen=True)
class CameraConfig:
    slot: str
    source_type: str
    serial_number: str = ""
    width: int = 1
    height: int = 1
    offset_x: int = 0
    offset_y: int = 0
    exposure_us: float = 0.0
    pixel_format: str = "Mono12"
    output_mode: str = BASLER_OUTPUT_NATIVE
    max_num_buffer: int = 10
    display: DisplayTransform = DisplayTransform()

    @property
    def session_key(self) -> tuple[object, ...]:
        return (
            self.source_type,
            self.serial_number,
            self.width,
            self.height,
            self.offset_x,
            self.offset_y,
            self.exposure_us,
            self.pixel_format,
            self.output_mode,
            self.max_num_buffer,
        )


def default_camera_config(slot: str) -> CameraConfig:
    slot = _slot_value(slot)
    if slot == "cam0":
        return CameraConfig(
            slot=slot,
            source_type=SOURCE_FRAMEGRABBER,
            width=int(constants.IMAGE_WIDTH_CAM0),
            height=int(constants.IMAGE_HEIGHT_CAM0),
            display=DisplayTransform(transpose=False, invert_x=False, invert_y=True),
        )
    return CameraConfig(
        slot=slot,
        source_type=SOURCE_BASLER,
        serial_number=str(constants.BASLER_CAMERA_SERIAL),
        width=int(constants.IMAGE_WIDTH_CAM1),
        height=int(constants.IMAGE_HEIGHT_CAM1),
        exposure_us=float(constants.BASLER_EXPOSURE),
        pixel_format="Mono12",
        output_mode=BASLER_OUTPUT_NATIVE,
        max_num_buffer=10,
        display=DisplayTransform(transpose=True, invert_x=True, invert_y=True),
    )


def default_camera_configs() -> dict[str, CameraConfig]:
    return {slot: default_camera_config(slot) for slot in CAMERA_SLOTS}


def camera_config_from_settings(settings: Any, slot: str) -> CameraConfig:
    default = default_camera_config(slot)
    prefix = f"camera/{default.slot}"
    source_type = _choice(
        settings.value(f"{prefix}/source_type", default.source_type),
        SOURCE_TYPES,
        default.source_type,
    )
    output_mode = _choice(
        settings.value(f"{prefix}/output_mode", default.output_mode),
        BASLER_OUTPUT_MODES,
        default.output_mode,
    )
    return CameraConfig(
        slot=default.slot,
        source_type=source_type,
        serial_number=str(
            settings.value(f"{prefix}/serial_number", default.serial_number)
        ),
        width=_positive_int(
            settings.value(f"{prefix}/width", default.width), default.width
        ),
        height=_positive_int(
            settings.value(f"{prefix}/height", default.height), default.height
        ),
        offset_x=_nonnegative_int(
            settings.value(f"{prefix}/offset_x", default.offset_x),
            default.offset_x,
        ),
        offset_y=_nonnegative_int(
            settings.value(f"{prefix}/offset_y", default.offset_y),
            default.offset_y,
        ),
        exposure_us=_nonnegative_float(
            settings.value(f"{prefix}/exposure_us", default.exposure_us),
            default.exposure_us,
        ),
        pixel_format=str(
            settings.value(f"{prefix}/pixel_format", default.pixel_format)
        ),
        output_mode=output_mode,
        max_num_buffer=_positive_int(
            settings.value(f"{prefix}/max_num_buffer", default.max_num_buffer),
            default.max_num_buffer,
        ),
        display=DisplayTransform(
            transpose=_bool_value(
                settings.value(
                    f"{prefix}/display_transpose", default.display.transpose
                ),
                default.display.transpose,
            ),
            invert_x=_bool_value(
                settings.value(f"{prefix}/display_invert_x", default.display.invert_x),
                default.display.invert_x,
            ),
            invert_y=_bool_value(
                settings.value(f"{prefix}/display_invert_y", default.display.invert_y),
                default.display.invert_y,
            ),
        ),
    )


def camera_configs_from_settings(settings: Any) -> dict[str, CameraConfig]:
    return {slot: camera_config_from_settings(settings, slot) for slot in CAMERA_SLOTS}


def save_camera_config(settings: Any, config: CameraConfig) -> None:
    prefix = f"camera/{_slot_value(config.slot)}"
    settings.setValue(f"{prefix}/source_type", config.source_type)
    settings.setValue(f"{prefix}/serial_number", config.serial_number)
    settings.setValue(f"{prefix}/width", int(config.width))
    settings.setValue(f"{prefix}/height", int(config.height))
    settings.setValue(f"{prefix}/offset_x", int(config.offset_x))
    settings.setValue(f"{prefix}/offset_y", int(config.offset_y))
    settings.setValue(f"{prefix}/exposure_us", float(config.exposure_us))
    settings.setValue(f"{prefix}/pixel_format", config.pixel_format)
    settings.setValue(f"{prefix}/output_mode", config.output_mode)
    settings.setValue(f"{prefix}/max_num_buffer", int(config.max_num_buffer))
    settings.setValue(f"{prefix}/display_transpose", bool(config.display.transpose))
    settings.setValue(f"{prefix}/display_invert_x", bool(config.display.invert_x))
    settings.setValue(f"{prefix}/display_invert_y", bool(config.display.invert_y))


def camera_metadata(configs: Mapping[str, CameraConfig]) -> dict[str, object]:
    metadata: dict[str, object] = {}
    for slot in CAMERA_SLOTS:
        config = configs[slot]
        prefix = f"camera_{slot}"
        metadata[f"{prefix}_source_type"] = config.source_type
        metadata[f"{prefix}_serial_number"] = config.serial_number
        metadata[f"{prefix}_width"] = int(config.width)
        metadata[f"{prefix}_height"] = int(config.height)
        metadata[f"{prefix}_offset_x"] = int(config.offset_x)
        metadata[f"{prefix}_offset_y"] = int(config.offset_y)
        metadata[f"{prefix}_exposure_us"] = float(config.exposure_us)
        metadata[f"{prefix}_pixel_format"] = config.pixel_format
        metadata[f"{prefix}_output_mode"] = config.output_mode
    return metadata


def camera_config_mismatches(
    attrs: Mapping[str, Any],
    configs: Mapping[str, CameraConfig],
) -> list[str]:
    mismatches: list[str] = []
    for slot in CAMERA_SLOTS:
        config = configs[slot]
        prefix = f"camera_{slot}"
        comparisons = (
            ("source_type", config.source_type, str),
            ("serial_number", config.serial_number, str),
            ("width", config.width, int),
            ("height", config.height, int),
            ("offset_x", config.offset_x, int),
            ("offset_y", config.offset_y, int),
            ("exposure_us", config.exposure_us, float),
            ("pixel_format", config.pixel_format, str),
            ("output_mode", config.output_mode, str),
        )
        for name, expected, converter in comparisons:
            key = f"{prefix}_{name}"
            if key not in attrs:
                continue
            try:
                actual = converter(attrs[key])
            except (TypeError, ValueError):
                mismatches.append(f"{slot} {name}")
                continue
            if converter is float:
                if not math.isclose(
                    float(actual),
                    float(expected),
                    rel_tol=1e-9,
                    abs_tol=1e-9,
                ):
                    mismatches.append(f"{slot} {name}")
                continue
            if actual != expected:
                mismatches.append(f"{slot} {name}")
    return mismatches


def _slot_value(value: str) -> str:
    slot = str(value)
    if slot not in CAMERA_SLOTS:
        raise ValueError(f"camera slot must be one of {CAMERA_SLOTS!r}")
    return slot


def _choice(value: object, choices: tuple[str, ...], fallback: str) -> str:
    text = str(value).strip().lower()
    if text in choices:
        return text
    return fallback


def _bool_value(value: object, fallback: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off"}:
            return False
    try:
        return bool(int(value))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return bool(fallback)


def _positive_int(value: object, fallback: int) -> int:
    try:
        numeric = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return int(fallback)
    return numeric if numeric > 0 else int(fallback)


def _nonnegative_int(value: object, fallback: int) -> int:
    try:
        numeric = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return int(fallback)
    return numeric if numeric >= 0 else int(fallback)


def _nonnegative_float(value: object, fallback: float) -> float:
    try:
        numeric = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return float(fallback)
    return numeric if numeric >= 0.0 else float(fallback)

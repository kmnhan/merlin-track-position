from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from merlin_track_position import constants

__all__ = (
    "CAPTURE_AGGREGATION_MEAN_IMAGE",
    "CAPTURE_AGGREGATION_MEDIAN_SHIFTS",
    "DEFAULT_REGISTRATION_CONFIG",
    "ECC_MOTION_MODEL_AFFINE",
    "ECC_MOTION_MODEL_HOMOGRAPHY",
    "REGISTRATION_CAPTURE_AGGREGATION_SETTINGS_KEY",
    "REGISTRATION_CAPTURE_COUNT_SETTINGS_KEY",
    "REGISTRATION_CLIP_ENABLED_SETTINGS_KEY",
    "REGISTRATION_CLIP_LOW_SETTINGS_KEY",
    "REGISTRATION_CLIP_HIGH_SETTINGS_KEY",
    "REGISTRATION_USE_WINDOW_SETTINGS_KEY",
    "REGISTRATION_USE_ECC_REFINEMENT_SETTINGS_KEY",
    "REGISTRATION_ECC_USE_WINDOW_SETTINGS_KEY",
    "REGISTRATION_ECC_MOTION_MODEL_SETTINGS_KEY",
    "REGISTRATION_ECC_GAUSS_FILTER_SIZE_CAM0_SETTINGS_KEY",
    "REGISTRATION_ECC_GAUSS_FILTER_SIZE_CAM1_SETTINGS_KEY",
    "normalized_registration_config",
    "registration_config_from_settings",
    "registration_config_to_camera_shift_kwargs",
    "registration_config_to_measurement_kwargs",
    "registration_config_to_shift_kwargs",
    "save_registration_config",
)


CAPTURE_AGGREGATION_MEDIAN_SHIFTS = "median_shifts"
CAPTURE_AGGREGATION_MEAN_IMAGE = "mean_image"
CAPTURE_AGGREGATIONS = (
    CAPTURE_AGGREGATION_MEDIAN_SHIFTS,
    CAPTURE_AGGREGATION_MEAN_IMAGE,
)
ECC_MOTION_MODEL_AFFINE = "affine"
ECC_MOTION_MODEL_HOMOGRAPHY = "homography"
ECC_MOTION_MODELS = (ECC_MOTION_MODEL_AFFINE, ECC_MOTION_MODEL_HOMOGRAPHY)
CAMERAS = ("cam0", "cam1")
REGISTRATION_CAPTURE_COUNT_MIN = 1
REGISTRATION_CAPTURE_COUNT_MAX = 100
REGISTRATION_ECC_GAUSS_FILTER_SIZE_MIN = 1
REGISTRATION_ECC_GAUSS_FILTER_SIZE_MAX = 31
DEFAULT_ECC_GAUSS_FILTER_SIZE_BY_CAMERA: dict[str, int] = {
    "cam0": 1,
    "cam1": 5,
}

REGISTRATION_CLIP_ENABLED_SETTINGS_KEY = "registration/clip_enabled"
REGISTRATION_CLIP_LOW_SETTINGS_KEY = "registration/clip_low"
REGISTRATION_CLIP_HIGH_SETTINGS_KEY = "registration/clip_high"
REGISTRATION_USE_WINDOW_SETTINGS_KEY = "registration/use_window"
REGISTRATION_USE_ECC_REFINEMENT_SETTINGS_KEY = "registration/use_ecc_refinement"
REGISTRATION_ECC_USE_WINDOW_SETTINGS_KEY = "registration/ecc_use_window"
REGISTRATION_ECC_MOTION_MODEL_SETTINGS_KEY = "registration/ecc_motion_model"
REGISTRATION_ECC_GAUSS_FILTER_SIZE_CAM0_SETTINGS_KEY = (
    "registration/ecc_gauss_filter_size_cam0"
)
REGISTRATION_ECC_GAUSS_FILTER_SIZE_CAM1_SETTINGS_KEY = (
    "registration/ecc_gauss_filter_size_cam1"
)
REGISTRATION_CAPTURE_COUNT_SETTINGS_KEY = "registration/capture_count"
REGISTRATION_CAPTURE_AGGREGATION_SETTINGS_KEY = "registration/capture_aggregation"

DEFAULT_REGISTRATION_CONFIG: dict[str, object] = {
    "clip_enabled": True,
    "clip_low": 1.0,
    "clip_high": 99.0,
    "use_window": False,
    "use_ecc_refinement": False,
    "ecc_use_window": False,
    "ecc_motion_model": ECC_MOTION_MODEL_HOMOGRAPHY,
    "ecc_gauss_filter_size": DEFAULT_ECC_GAUSS_FILTER_SIZE_BY_CAMERA.copy(),
    "capture_count": constants.DEFAULT_CORRECTION_CAPTURE_COUNT,
    "capture_aggregation": CAPTURE_AGGREGATION_MEDIAN_SHIFTS,
}


def normalized_registration_config(
    config: Mapping[str, Any] | None = None,
) -> dict[str, object]:
    values = dict(DEFAULT_REGISTRATION_CONFIG)
    if config is not None:
        values.update(config)

    clip_enabled = _as_bool(
        values["clip_enabled"],
        bool(DEFAULT_REGISTRATION_CONFIG["clip_enabled"]),
    )
    clip_low = _as_float(values["clip_low"], DEFAULT_REGISTRATION_CONFIG["clip_low"])
    clip_high = _as_float(
        values["clip_high"],
        DEFAULT_REGISTRATION_CONFIG["clip_high"],
    )
    if not (0.0 <= clip_low < clip_high <= 100.0):
        clip_low = float(DEFAULT_REGISTRATION_CONFIG["clip_low"])
        clip_high = float(DEFAULT_REGISTRATION_CONFIG["clip_high"])

    capture_count = _as_int(
        values["capture_count"],
        DEFAULT_REGISTRATION_CONFIG["capture_count"],
    )
    if capture_count < REGISTRATION_CAPTURE_COUNT_MIN:
        capture_count = int(DEFAULT_REGISTRATION_CONFIG["capture_count"])
    capture_count = min(capture_count, REGISTRATION_CAPTURE_COUNT_MAX)

    return {
        "clip_enabled": clip_enabled,
        "clip_low": float(clip_low),
        "clip_high": float(clip_high),
        "use_window": _as_bool(
            values["use_window"],
            bool(DEFAULT_REGISTRATION_CONFIG["use_window"]),
        ),
        "use_ecc_refinement": _as_bool(
            values["use_ecc_refinement"],
            bool(DEFAULT_REGISTRATION_CONFIG["use_ecc_refinement"]),
        ),
        "ecc_use_window": _as_bool(
            values["ecc_use_window"],
            bool(DEFAULT_REGISTRATION_CONFIG["ecc_use_window"]),
        ),
        "ecc_motion_model": _ecc_motion_model_value(values["ecc_motion_model"]),
        "ecc_gauss_filter_size": _ecc_gauss_filter_size_by_camera(
            values["ecc_gauss_filter_size"]
        ),
        "capture_count": int(capture_count),
        "capture_aggregation": _capture_aggregation_value(
            values["capture_aggregation"]
        ),
    }


def registration_config_from_settings(settings: Any) -> dict[str, object]:
    return normalized_registration_config(
        {
            "clip_enabled": settings.value(
                REGISTRATION_CLIP_ENABLED_SETTINGS_KEY,
                DEFAULT_REGISTRATION_CONFIG["clip_enabled"],
            ),
            "clip_low": settings.value(
                REGISTRATION_CLIP_LOW_SETTINGS_KEY,
                DEFAULT_REGISTRATION_CONFIG["clip_low"],
            ),
            "clip_high": settings.value(
                REGISTRATION_CLIP_HIGH_SETTINGS_KEY,
                DEFAULT_REGISTRATION_CONFIG["clip_high"],
            ),
            "use_window": settings.value(
                REGISTRATION_USE_WINDOW_SETTINGS_KEY,
                DEFAULT_REGISTRATION_CONFIG["use_window"],
            ),
            "use_ecc_refinement": settings.value(
                REGISTRATION_USE_ECC_REFINEMENT_SETTINGS_KEY,
                DEFAULT_REGISTRATION_CONFIG["use_ecc_refinement"],
            ),
            "ecc_use_window": settings.value(
                REGISTRATION_ECC_USE_WINDOW_SETTINGS_KEY,
                DEFAULT_REGISTRATION_CONFIG["ecc_use_window"],
            ),
            "ecc_motion_model": settings.value(
                REGISTRATION_ECC_MOTION_MODEL_SETTINGS_KEY,
                DEFAULT_REGISTRATION_CONFIG["ecc_motion_model"],
            ),
            "ecc_gauss_filter_size": {
                "cam0": settings.value(
                    REGISTRATION_ECC_GAUSS_FILTER_SIZE_CAM0_SETTINGS_KEY,
                    DEFAULT_ECC_GAUSS_FILTER_SIZE_BY_CAMERA["cam0"],
                ),
                "cam1": settings.value(
                    REGISTRATION_ECC_GAUSS_FILTER_SIZE_CAM1_SETTINGS_KEY,
                    DEFAULT_ECC_GAUSS_FILTER_SIZE_BY_CAMERA["cam1"],
                ),
            },
            "capture_count": settings.value(
                REGISTRATION_CAPTURE_COUNT_SETTINGS_KEY,
                DEFAULT_REGISTRATION_CONFIG["capture_count"],
            ),
            "capture_aggregation": settings.value(
                REGISTRATION_CAPTURE_AGGREGATION_SETTINGS_KEY,
                DEFAULT_REGISTRATION_CONFIG["capture_aggregation"],
            ),
        }
    )


def save_registration_config(
    settings: Any,
    config: Mapping[str, Any],
) -> dict[str, object]:
    normalized = normalized_registration_config(config)
    settings.setValue(
        REGISTRATION_CLIP_ENABLED_SETTINGS_KEY,
        normalized["clip_enabled"],
    )
    settings.setValue(REGISTRATION_CLIP_LOW_SETTINGS_KEY, normalized["clip_low"])
    settings.setValue(REGISTRATION_CLIP_HIGH_SETTINGS_KEY, normalized["clip_high"])
    settings.setValue(REGISTRATION_USE_WINDOW_SETTINGS_KEY, normalized["use_window"])
    settings.setValue(
        REGISTRATION_USE_ECC_REFINEMENT_SETTINGS_KEY,
        normalized["use_ecc_refinement"],
    )
    settings.setValue(
        REGISTRATION_ECC_USE_WINDOW_SETTINGS_KEY,
        normalized["ecc_use_window"],
    )
    settings.setValue(
        REGISTRATION_ECC_MOTION_MODEL_SETTINGS_KEY,
        normalized["ecc_motion_model"],
    )
    gauss_filter_size = normalized["ecc_gauss_filter_size"]
    assert isinstance(gauss_filter_size, dict)
    settings.setValue(
        REGISTRATION_ECC_GAUSS_FILTER_SIZE_CAM0_SETTINGS_KEY,
        gauss_filter_size["cam0"],
    )
    settings.setValue(
        REGISTRATION_ECC_GAUSS_FILTER_SIZE_CAM1_SETTINGS_KEY,
        gauss_filter_size["cam1"],
    )
    settings.setValue(
        REGISTRATION_CAPTURE_COUNT_SETTINGS_KEY,
        normalized["capture_count"],
    )
    settings.setValue(
        REGISTRATION_CAPTURE_AGGREGATION_SETTINGS_KEY,
        normalized["capture_aggregation"],
    )
    settings.sync()
    return normalized


def registration_config_to_measurement_kwargs(
    config: Mapping[str, Any],
) -> dict[str, object]:
    normalized = normalized_registration_config(config)
    return {
        "capture_count": normalized["capture_count"],
        "capture_aggregation": normalized["capture_aggregation"],
        **registration_config_to_shift_kwargs(normalized),
    }


def registration_config_to_shift_kwargs(
    config: Mapping[str, Any],
) -> dict[str, object]:
    normalized = normalized_registration_config(config)
    clip_percentiles = (
        (normalized["clip_low"], normalized["clip_high"])
        if normalized["clip_enabled"]
        else None
    )
    shift_kwargs = {
        "clip_percentiles": clip_percentiles,
        "use_window": normalized["use_window"],
        "use_ecc_refinement": normalized["use_ecc_refinement"],
        "ecc_use_window": normalized["ecc_use_window"],
        "ecc_motion_model": normalized["ecc_motion_model"],
        "ecc_gauss_filter_size": normalized["ecc_gauss_filter_size"],
    }
    if config is not None and "ecc_reference_point_px" in config:
        shift_kwargs["ecc_reference_point_px"] = config["ecc_reference_point_px"]
    return shift_kwargs


def registration_config_to_camera_shift_kwargs(
    config: Mapping[str, Any],
    camera: str,
) -> dict[str, object]:
    camera_name = str(camera)
    shift_kwargs = registration_config_to_shift_kwargs(config)
    for name in (
        "ecc_reference_point_px",
        "ecc_initial_shift_px",
        "ecc_initial_warp",
        "ecc_gauss_filter_size",
    ):
        value = shift_kwargs.get(name)
        if not isinstance(value, Mapping):
            continue
        camera_value = value.get(camera_name)
        if camera_value is None:
            shift_kwargs.pop(name)
        else:
            shift_kwargs[name] = camera_value
    return shift_kwargs


def _as_bool(value: Any, fallback: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off"}:
            return False
    try:
        return bool(int(value))
    except (TypeError, ValueError):
        return bool(fallback)


def _as_float(value: Any, fallback: object) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return float(fallback)
    if not math.isfinite(numeric):
        return float(fallback)
    return numeric


def _as_int(value: Any, fallback: object) -> int:
    try:
        numeric = int(value)
    except (TypeError, ValueError):
        return int(fallback)
    return numeric


def _as_odd_int(value: Any, fallback: object) -> int:
    numeric = _as_int(value, fallback)
    if (
        numeric < REGISTRATION_ECC_GAUSS_FILTER_SIZE_MIN
        or numeric > REGISTRATION_ECC_GAUSS_FILTER_SIZE_MAX
        or numeric % 2 != 1
    ):
        return int(fallback)
    return numeric


def _ecc_gauss_filter_size_by_camera(value: Any) -> dict[str, int]:
    defaults = DEFAULT_ECC_GAUSS_FILTER_SIZE_BY_CAMERA
    if isinstance(value, Mapping):
        return {
            camera: _as_odd_int(value.get(camera), defaults[camera])
            for camera in CAMERAS
        }
    fallback = _as_odd_int(value, defaults["cam1"])
    return {camera: fallback for camera in CAMERAS}


def _capture_aggregation_value(value: Any) -> str:
    lowered = str(value).strip().lower()
    if lowered in CAPTURE_AGGREGATIONS:
        return lowered
    return CAPTURE_AGGREGATION_MEDIAN_SHIFTS


def _ecc_motion_model_value(value: Any) -> str:
    lowered = str(value).strip().lower()
    if lowered in ECC_MOTION_MODELS:
        return lowered
    return ECC_MOTION_MODEL_HOMOGRAPHY

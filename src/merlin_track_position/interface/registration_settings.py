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
    "REGISTRATION_PHASE_L2_SIZE_SETTINGS_KEY",
    "REGISTRATION_PHASE_MAX_ITERS_SETTINGS_KEY",
    "REGISTRATION_USE_WINDOW_SETTINGS_KEY",
    "REGISTRATION_USE_ECC_REFINEMENT_SETTINGS_KEY",
    "REGISTRATION_ECC_MOTION_MODEL_SETTINGS_KEY",
    "normalized_registration_config",
    "registration_config_from_settings",
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
REGISTRATION_CAPTURE_COUNT_MIN = 1
REGISTRATION_CAPTURE_COUNT_MAX = 100

REGISTRATION_CLIP_ENABLED_SETTINGS_KEY = "registration/clip_enabled"
REGISTRATION_CLIP_LOW_SETTINGS_KEY = "registration/clip_low"
REGISTRATION_CLIP_HIGH_SETTINGS_KEY = "registration/clip_high"
REGISTRATION_PHASE_L2_SIZE_SETTINGS_KEY = "registration/phase_l2_size"
REGISTRATION_PHASE_MAX_ITERS_SETTINGS_KEY = "registration/phase_max_iters"
REGISTRATION_USE_WINDOW_SETTINGS_KEY = "registration/use_window"
REGISTRATION_USE_ECC_REFINEMENT_SETTINGS_KEY = "registration/use_ecc_refinement"
REGISTRATION_ECC_MOTION_MODEL_SETTINGS_KEY = "registration/ecc_motion_model"
REGISTRATION_CAPTURE_COUNT_SETTINGS_KEY = "registration/capture_count"
REGISTRATION_CAPTURE_AGGREGATION_SETTINGS_KEY = "registration/capture_aggregation"

DEFAULT_REGISTRATION_CONFIG: dict[str, object] = {
    "clip_enabled": True,
    "clip_low": 1.0,
    "clip_high": 99.0,
    "phase_l2_size": 7,
    "phase_max_iters": 50,
    "use_window": False,
    "use_ecc_refinement": False,
    "ecc_motion_model": ECC_MOTION_MODEL_HOMOGRAPHY,
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

    phase_l2_size = _as_int(
        values["phase_l2_size"],
        DEFAULT_REGISTRATION_CONFIG["phase_l2_size"],
    )
    if phase_l2_size < 1:
        phase_l2_size = int(DEFAULT_REGISTRATION_CONFIG["phase_l2_size"])

    phase_max_iters = _as_int(
        values["phase_max_iters"],
        DEFAULT_REGISTRATION_CONFIG["phase_max_iters"],
    )
    if phase_max_iters < 1:
        phase_max_iters = int(DEFAULT_REGISTRATION_CONFIG["phase_max_iters"])

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
        "phase_l2_size": int(phase_l2_size),
        "phase_max_iters": int(phase_max_iters),
        "use_window": _as_bool(
            values["use_window"],
            bool(DEFAULT_REGISTRATION_CONFIG["use_window"]),
        ),
        "use_ecc_refinement": _as_bool(
            values["use_ecc_refinement"],
            bool(DEFAULT_REGISTRATION_CONFIG["use_ecc_refinement"]),
        ),
        "ecc_motion_model": _ecc_motion_model_value(values["ecc_motion_model"]),
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
            "phase_l2_size": settings.value(
                REGISTRATION_PHASE_L2_SIZE_SETTINGS_KEY,
                DEFAULT_REGISTRATION_CONFIG["phase_l2_size"],
            ),
            "phase_max_iters": settings.value(
                REGISTRATION_PHASE_MAX_ITERS_SETTINGS_KEY,
                DEFAULT_REGISTRATION_CONFIG["phase_max_iters"],
            ),
            "use_window": settings.value(
                REGISTRATION_USE_WINDOW_SETTINGS_KEY,
                DEFAULT_REGISTRATION_CONFIG["use_window"],
            ),
            "use_ecc_refinement": settings.value(
                REGISTRATION_USE_ECC_REFINEMENT_SETTINGS_KEY,
                DEFAULT_REGISTRATION_CONFIG["use_ecc_refinement"],
            ),
            "ecc_motion_model": settings.value(
                REGISTRATION_ECC_MOTION_MODEL_SETTINGS_KEY,
                DEFAULT_REGISTRATION_CONFIG["ecc_motion_model"],
            ),
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
    settings.setValue(
        REGISTRATION_PHASE_L2_SIZE_SETTINGS_KEY,
        normalized["phase_l2_size"],
    )
    settings.setValue(
        REGISTRATION_PHASE_MAX_ITERS_SETTINGS_KEY,
        normalized["phase_max_iters"],
    )
    settings.setValue(REGISTRATION_USE_WINDOW_SETTINGS_KEY, normalized["use_window"])
    settings.setValue(
        REGISTRATION_USE_ECC_REFINEMENT_SETTINGS_KEY,
        normalized["use_ecc_refinement"],
    )
    settings.setValue(
        REGISTRATION_ECC_MOTION_MODEL_SETTINGS_KEY,
        normalized["ecc_motion_model"],
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
        "phase_l2_size": normalized["phase_l2_size"],
        "phase_max_iters": normalized["phase_max_iters"],
        "use_ecc_refinement": normalized["use_ecc_refinement"],
        "ecc_motion_model": normalized["ecc_motion_model"],
    }
    if config is not None and "ecc_reference_point_px" in config:
        shift_kwargs["ecc_reference_point_px"] = config["ecc_reference_point_px"]
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

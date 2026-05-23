from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from merlin_track_position import constants

__all__ = (
    "CAPTURE_AGGREGATION_MEAN_IMAGE",
    "CAPTURE_AGGREGATION_MEDIAN_SHIFTS",
    "DEFAULT_REGISTRATION_CONFIG",
    "REGISTRATION_CAPTURE_AGGREGATION_SETTINGS_KEY",
    "REGISTRATION_CAPTURE_COUNT_SETTINGS_KEY",
    "REGISTRATION_CLIP_ENABLED_SETTINGS_KEY",
    "REGISTRATION_CLIP_LOW_SETTINGS_KEY",
    "REGISTRATION_CLIP_HIGH_SETTINGS_KEY",
    "REGISTRATION_NORMALIZATION_SETTINGS_KEY",
    "REGISTRATION_UPSAMPLE_FACTOR_SETTINGS_KEY",
    "REGISTRATION_USE_WINDOW_SETTINGS_KEY",
    "REGISTRATION_USE_ECC_REFINEMENT_SETTINGS_KEY",
    "REGISTRATION_HIGH_ERROR_THRESHOLD_SETTINGS_KEY",
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
REGISTRATION_CAPTURE_COUNT_MIN = 1
REGISTRATION_CAPTURE_COUNT_MAX = 100

REGISTRATION_CLIP_ENABLED_SETTINGS_KEY = "registration/clip_enabled"
REGISTRATION_CLIP_LOW_SETTINGS_KEY = "registration/clip_low"
REGISTRATION_CLIP_HIGH_SETTINGS_KEY = "registration/clip_high"
REGISTRATION_NORMALIZATION_SETTINGS_KEY = "registration/normalization"
REGISTRATION_UPSAMPLE_FACTOR_SETTINGS_KEY = "registration/upsample_factor"
REGISTRATION_USE_WINDOW_SETTINGS_KEY = "registration/use_window"
REGISTRATION_USE_ECC_REFINEMENT_SETTINGS_KEY = "registration/use_ecc_refinement"
REGISTRATION_HIGH_ERROR_THRESHOLD_SETTINGS_KEY = (
    "registration/high_error_threshold"
)
REGISTRATION_CAPTURE_COUNT_SETTINGS_KEY = "registration/capture_count"
REGISTRATION_CAPTURE_AGGREGATION_SETTINGS_KEY = "registration/capture_aggregation"

DEFAULT_REGISTRATION_CONFIG: dict[str, object] = {
    "clip_enabled": True,
    "clip_low": 1.0,
    "clip_high": 99.0,
    "normalization": "phase",
    "upsample_factor": 50,
    "use_window": False,
    "use_ecc_refinement": False,
    "high_error_threshold": 0.5,
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

    upsample_factor = _as_int(
        values["upsample_factor"],
        DEFAULT_REGISTRATION_CONFIG["upsample_factor"],
    )
    if upsample_factor < 1:
        upsample_factor = int(DEFAULT_REGISTRATION_CONFIG["upsample_factor"])

    high_error_threshold = _as_float(
        values["high_error_threshold"],
        DEFAULT_REGISTRATION_CONFIG["high_error_threshold"],
    )
    if high_error_threshold <= 0.0:
        high_error_threshold = float(
            DEFAULT_REGISTRATION_CONFIG["high_error_threshold"]
        )

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
        "normalization": _normalization_value(values["normalization"]),
        "upsample_factor": int(upsample_factor),
        "use_window": _as_bool(
            values["use_window"],
            bool(DEFAULT_REGISTRATION_CONFIG["use_window"]),
        ),
        "use_ecc_refinement": _as_bool(
            values["use_ecc_refinement"],
            bool(DEFAULT_REGISTRATION_CONFIG["use_ecc_refinement"]),
        ),
        "high_error_threshold": float(high_error_threshold),
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
            "normalization": settings.value(
                REGISTRATION_NORMALIZATION_SETTINGS_KEY,
                DEFAULT_REGISTRATION_CONFIG["normalization"],
            ),
            "upsample_factor": settings.value(
                REGISTRATION_UPSAMPLE_FACTOR_SETTINGS_KEY,
                DEFAULT_REGISTRATION_CONFIG["upsample_factor"],
            ),
            "use_window": settings.value(
                REGISTRATION_USE_WINDOW_SETTINGS_KEY,
                DEFAULT_REGISTRATION_CONFIG["use_window"],
            ),
            "use_ecc_refinement": settings.value(
                REGISTRATION_USE_ECC_REFINEMENT_SETTINGS_KEY,
                DEFAULT_REGISTRATION_CONFIG["use_ecc_refinement"],
            ),
            "high_error_threshold": settings.value(
                REGISTRATION_HIGH_ERROR_THRESHOLD_SETTINGS_KEY,
                DEFAULT_REGISTRATION_CONFIG["high_error_threshold"],
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
        REGISTRATION_NORMALIZATION_SETTINGS_KEY,
        normalized["normalization"],
    )
    settings.setValue(
        REGISTRATION_UPSAMPLE_FACTOR_SETTINGS_KEY,
        normalized["upsample_factor"],
    )
    settings.setValue(REGISTRATION_USE_WINDOW_SETTINGS_KEY, normalized["use_window"])
    settings.setValue(
        REGISTRATION_USE_ECC_REFINEMENT_SETTINGS_KEY,
        normalized["use_ecc_refinement"],
    )
    settings.setValue(
        REGISTRATION_HIGH_ERROR_THRESHOLD_SETTINGS_KEY,
        normalized["high_error_threshold"],
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
    normalization = (
        "phase" if normalized["normalization"] == "phase" else None
    )
    shift_kwargs = {
        "clip_percentiles": clip_percentiles,
        "use_window": normalized["use_window"],
        "upsample_factor": normalized["upsample_factor"],
        "normalization": normalization,
        "high_error_threshold": normalized["high_error_threshold"],
        "use_ecc_refinement": normalized["use_ecc_refinement"],
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


def _normalization_value(value: Any) -> str:
    if value is None:
        return "none"
    lowered = str(value).strip().lower()
    if lowered == "none":
        return "none"
    return "phase"


def _capture_aggregation_value(value: Any) -> str:
    lowered = str(value).strip().lower()
    if lowered in CAPTURE_AGGREGATIONS:
        return lowered
    return CAPTURE_AGGREGATION_MEDIAN_SHIFTS

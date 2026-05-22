from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

__all__ = (
    "DEFAULT_REGISTRATION_CONFIG",
    "REGISTRATION_CLIP_ENABLED_SETTINGS_KEY",
    "REGISTRATION_CLIP_LOW_SETTINGS_KEY",
    "REGISTRATION_CLIP_HIGH_SETTINGS_KEY",
    "REGISTRATION_NORMALIZATION_SETTINGS_KEY",
    "REGISTRATION_UPSAMPLE_FACTOR_SETTINGS_KEY",
    "REGISTRATION_USE_WINDOW_SETTINGS_KEY",
    "REGISTRATION_HIGH_ERROR_THRESHOLD_SETTINGS_KEY",
    "normalized_registration_config",
    "registration_config_from_settings",
    "registration_config_to_shift_kwargs",
    "save_registration_config",
)


REGISTRATION_CLIP_ENABLED_SETTINGS_KEY = "registration/clip_enabled"
REGISTRATION_CLIP_LOW_SETTINGS_KEY = "registration/clip_low"
REGISTRATION_CLIP_HIGH_SETTINGS_KEY = "registration/clip_high"
REGISTRATION_NORMALIZATION_SETTINGS_KEY = "registration/normalization"
REGISTRATION_UPSAMPLE_FACTOR_SETTINGS_KEY = "registration/upsample_factor"
REGISTRATION_USE_WINDOW_SETTINGS_KEY = "registration/use_window"
REGISTRATION_HIGH_ERROR_THRESHOLD_SETTINGS_KEY = (
    "registration/high_error_threshold"
)

DEFAULT_REGISTRATION_CONFIG: dict[str, object] = {
    "clip_enabled": True,
    "clip_low": 1.0,
    "clip_high": 99.0,
    "normalization": "phase",
    "upsample_factor": 50,
    "use_window": False,
    "high_error_threshold": 0.5,
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
        "high_error_threshold": float(high_error_threshold),
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
            "high_error_threshold": settings.value(
                REGISTRATION_HIGH_ERROR_THRESHOLD_SETTINGS_KEY,
                DEFAULT_REGISTRATION_CONFIG["high_error_threshold"],
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
        REGISTRATION_HIGH_ERROR_THRESHOLD_SETTINGS_KEY,
        normalized["high_error_threshold"],
    )
    settings.sync()
    return normalized


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
    return {
        "clip_percentiles": clip_percentiles,
        "use_window": normalized["use_window"],
        "upsample_factor": normalized["upsample_factor"],
        "normalization": normalization,
        "high_error_threshold": normalized["high_error_threshold"],
    }


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

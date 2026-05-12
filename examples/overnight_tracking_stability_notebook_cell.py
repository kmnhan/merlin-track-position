"""Paste this cell into a notebook to record overnight tracking shifts."""

# %%
from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import xarray as xr

from merlin_track_position.instruments.cameras import default_camera_pair
from merlin_track_position.tracking.shift import PIXEL_AXES, estimate_shift


# Edit these values.
OUTPUT_PATH = Path("overnight_tracking_shifts.h5")
INTERVAL_SECONDS = 10 * 60
N_SAMPLES = 72
SHIFT_KWARGS = {
    # "upsample_factor": 50,
    # "check_tiles": True,
}


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _build_dataset() -> xr.Dataset:
    return xr.Dataset(
        data_vars={
            "shift_px": (
                ("sample", "camera", "pixel_axis"),
                shift_px,
                {"units": "px"},
            ),
        },
        coords={
            "sample": np.arange(N_SAMPLES, dtype=np.int64),
            "camera": ["cam0", "cam1"],
            "pixel_axis": list(PIXEL_AXES),
            "timestamp_utc": ("sample", timestamp_utc),
            "elapsed_s": ("sample", elapsed_s),
        },
        attrs={
            "interval_seconds": INTERVAL_SECONDS,
            "n_samples": N_SAMPLES,
            "created_at_utc": created_at_utc,
            "updated_at_utc": _utc_timestamp(),
            "completed_samples": int(
                np.count_nonzero(np.isfinite(shift_px).all(axis=(1, 2)))
            ),
        },
    )


def _save_dataset() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = OUTPUT_PATH.with_name(f"{OUTPUT_PATH.name}.tmp")
    _build_dataset().to_netcdf(tmp_path, engine="h5netcdf")
    tmp_path.replace(OUTPUT_PATH)


camera_pair = default_camera_pair()
reference_cam0, reference_cam1 = camera_pair.capture_pair()

created_at_utc = _utc_timestamp()
start_monotonic = time.monotonic()

shift_px = np.full((N_SAMPLES, 2, len(PIXEL_AXES)), np.nan, dtype=np.float64)
timestamp_utc = np.full(N_SAMPLES, "", dtype=object)
elapsed_s = np.full(N_SAMPLES, np.nan)

print(
    f"Recording {N_SAMPLES} shift samples every {INTERVAL_SECONDS} s -> "
    f"{OUTPUT_PATH}"
)

for sample_index in range(N_SAMPLES):
    target_monotonic = start_monotonic + sample_index * INTERVAL_SECONDS
    sleep_s = target_monotonic - time.monotonic()
    if sleep_s > 0:
        time.sleep(sleep_s)

    timestamp_utc[sample_index] = _utc_timestamp()
    elapsed_s[sample_index] = time.monotonic() - start_monotonic

    try:
        if sample_index == 0:
            current_cam0, current_cam1 = reference_cam0, reference_cam1
        else:
            current_cam0, current_cam1 = camera_pair.capture_pair()

        shift_px[sample_index, 0, :] = estimate_shift(
            reference_cam0,
            current_cam0,
            **SHIFT_KWARGS,
        )["shift_px"].values
        shift_px[sample_index, 1, :] = estimate_shift(
            reference_cam1,
            current_cam1,
            **SHIFT_KWARGS,
        )["shift_px"].values
    except KeyboardInterrupt:
        _save_dataset()
        raise
    except Exception as exc:
        print(f"[{sample_index + 1}/{N_SAMPLES}] error: {type(exc).__name__}: {exc}")
    finally:
        _save_dataset()

    print(
        f"[{sample_index + 1}/{N_SAMPLES}] "
        f"elapsed={elapsed_s[sample_index]:.1f}s "
        f"shift_px={shift_px[sample_index].tolist()}"
    )

results = _build_dataset()
results

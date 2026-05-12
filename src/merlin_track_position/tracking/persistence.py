from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, NamedTuple

import xarray as xr

SPOOL_ENV_VAR = "MERLIN_TRACK_POSITION_SPOOL_DIR"
SPOOL_SCHEMA_VERSION = 1
HDF5_IMAGE_COMPRESSION = "gzip"
HDF5_IMAGE_COMPRESSION_LEVEL = 4


class PersistenceResult(NamedTuple):
    target_path: Path
    spool_path: Path | None
    flushed: bool
    pending: bool
    message: str


class PendingEntry(NamedTuple):
    path: Path
    metadata: dict[str, Any]


def persistence_result_attrs(
    prefix: str,
    result: PersistenceResult,
) -> dict[str, str]:
    status = "pending" if result.pending else "flushed" if result.flushed else "skipped"
    attrs = {
        f"{prefix}_persistence_status": status,
        f"{prefix}_persistence_message": result.message,
    }
    if result.pending and result.spool_path is not None:
        attrs[f"{prefix}_pending_spool_path"] = str(result.spool_path)
    return attrs


def hdf5_spool_dir() -> Path:
    override = os.environ.get(SPOOL_ENV_VAR)
    if override:
        return Path(override).expanduser()

    if os.name == "nt":
        root = Path(
            os.environ.get("LOCALAPPDATA")
            or os.environ.get("APPDATA")
            or Path.home() / "AppData" / "Local"
        )
    elif sys.platform == "darwin":
        root = Path.home() / "Library" / "Application Support"
    else:
        root = Path(os.environ.get("XDG_STATE_HOME", Path.home() / ".local" / "state"))
    return root / "merlin-track-position" / "hdf5-spool"


def normalize_target_path(path: str | Path) -> Path:
    return Path(path).expanduser().absolute()


def target_fingerprint(path: str | Path) -> dict[str, Any]:
    target = normalize_target_path(path)
    try:
        stat = target.stat()
    except FileNotFoundError:
        return {"exists": False}
    return {
        "exists": True,
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def fingerprint_matches(
    path: str | Path,
    expected: dict[str, Any] | None,
) -> bool:
    if expected is None:
        return True
    return target_fingerprint(path) == expected


def hdf5_image_encoding(dataset: xr.Dataset) -> dict[str, dict[str, Any]]:
    """Return HDF5 compression settings for image-like data variables."""

    encoding: dict[str, dict[str, Any]] = {}
    for name, variable in dataset.data_vars.items():
        dims = set(variable.dims)
        if (
            {"y_cam0", "x_cam0"}.issubset(dims)
            or {"y_cam1", "x_cam1"}.issubset(dims)
        ):
            encoding[name] = {
                "compression": HDF5_IMAGE_COMPRESSION,
                "compression_opts": HDF5_IMAGE_COMPRESSION_LEVEL,
                "shuffle": True,
            }
    return encoding


def stage_dataset(
    dataset: xr.Dataset,
    target_path: str | Path,
    *,
    operation: str,
    metadata: dict[str, Any] | None = None,
) -> PendingEntry:
    target = normalize_target_path(target_path)
    pending_root = hdf5_spool_dir() / "pending"
    pending_root.mkdir(parents=True, exist_ok=True)

    entry_id = f"{datetime.now(UTC).strftime('%Y%m%dT%H%M%S%fZ')}-{uuid.uuid4().hex}"
    tmp_dir = Path(
        tempfile.mkdtemp(
            prefix=f".{entry_id}.",
            suffix=".tmp",
            dir=str(pending_root),
        )
    )
    final_dir = pending_root / entry_id
    entry_metadata: dict[str, Any] = {
        "schema_version": SPOOL_SCHEMA_VERSION,
        "operation": str(operation),
        "target_path": str(target),
        "created_at_utc": datetime.now(UTC).isoformat(),
    }
    if metadata is not None:
        entry_metadata.update(metadata)

    try:
        loaded = dataset.load()
        loaded.to_netcdf(
            tmp_dir / "data.h5",
            engine="h5netcdf",
            encoding=hdf5_image_encoding(loaded),
        )
        _write_metadata(tmp_dir, entry_metadata)
        tmp_dir.rename(final_dir)
    except Exception:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise

    return PendingEntry(final_dir, entry_metadata)


def iter_pending_entries(
    *,
    operation: str | None = None,
    target_path: str | Path | None = None,
) -> list[PendingEntry]:
    pending_root = hdf5_spool_dir() / "pending"
    if not pending_root.exists():
        return []

    normalized_target = (
        str(normalize_target_path(target_path)) if target_path is not None else None
    )
    entries: list[PendingEntry] = []
    for entry_path in sorted(path for path in pending_root.iterdir() if path.is_dir()):
        try:
            metadata = _read_metadata(entry_path)
        except Exception:
            continue
        if operation is not None and metadata.get("operation") != operation:
            continue
        if (
            normalized_target is not None
            and metadata.get("target_path") != normalized_target
        ):
            continue
        entries.append(PendingEntry(entry_path, metadata))
    return entries


def load_spooled_dataset(entry: PendingEntry | Path) -> xr.Dataset:
    entry_path = entry.path if isinstance(entry, PendingEntry) else Path(entry)
    with xr.open_dataset(entry_path / "data.h5", engine="h5netcdf") as on_disk:
        return on_disk.load()


def discard_spool_entry(entry: PendingEntry | Path) -> None:
    entry_path = entry.path if isinstance(entry, PendingEntry) else Path(entry)
    shutil.rmtree(entry_path, ignore_errors=True)


def mark_spool_entry_stale(entry: PendingEntry, reason: str) -> Path:
    metadata = dict(entry.metadata)
    metadata["stale_reason"] = str(reason)
    metadata["stale_at_utc"] = datetime.now(UTC).isoformat()
    _write_metadata(entry.path, metadata)

    stale_root = hdf5_spool_dir() / "stale"
    stale_root.mkdir(parents=True, exist_ok=True)
    stale_path = stale_root / entry.path.name
    if stale_path.exists():
        stale_path = stale_root / f"{entry.path.name}-{uuid.uuid4().hex}"
    entry.path.rename(stale_path)
    return stale_path


def pending_entry_count() -> int:
    return len(iter_pending_entries())


def _write_metadata(entry_path: Path, metadata: dict[str, Any]) -> None:
    metadata_path = entry_path / "metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _read_metadata(entry_path: Path) -> dict[str, Any]:
    metadata = json.loads((entry_path / "metadata.json").read_text(encoding="utf-8"))
    if int(metadata.get("schema_version", -1)) != SPOOL_SCHEMA_VERSION:
        raise ValueError("unsupported HDF5 spool entry schema")
    return metadata

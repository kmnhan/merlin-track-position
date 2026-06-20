"""Access images from the FrameGrabber LabVIEW panel."""

import json
import time

import numpy as np
import numpy.typing as npt
import zmq

from merlin_track_position import constants
from merlin_track_position.instruments.camera_config import (
    CameraConfig,
    default_camera_config,
)
from merlin_track_position.instruments.simulated_hardware import simulator

LABVIEW_UNIX_EPOCH_OFFSET_MS = 2_082_844_800_000


def get_framegrabber_image(
    timeout_ms: int = 5000,
    *,
    config: CameraConfig | None = None,
) -> npt.NDArray:
    """Request one fresh image from the FrameGrabber LabVIEW panel."""

    return get_framegrabber_image_stack(1, timeout_ms=timeout_ms, config=config)[0]


def get_framegrabber_image_stack(
    frame_count: int,
    timeout_ms: int = 5000,
    *,
    config: CameraConfig | None = None,
) -> npt.NDArray:
    """Request consecutive fresh images from the FrameGrabber LabVIEW panel.

    Parameters
    ----------
    frame_count
        Number of images to capture after the shared request-start timestamp.
    timeout_ms
        Maximum time set to the zmq RCVTIMEO option, to wait for a response before
        raising TimeoutError.
    config
        Logical camera configuration used to crop raw framegrabber frames.
    """
    stack, _timestamps_ns = get_framegrabber_image_stack_with_timestamps(
        frame_count,
        timeout_ms=timeout_ms,
        config=config,
    )
    return stack


def get_framegrabber_image_stack_with_timestamps(
    frame_count: int,
    timeout_ms: int = 5000,
    *,
    config: CameraConfig | None = None,
) -> tuple[npt.NDArray, npt.NDArray[np.int64]]:
    """Request consecutive fresh images and their Unix-epoch timestamps."""
    frame_count = _validate_frame_count(frame_count)
    if config is None:
        config = default_camera_config("cam0")
    if not constants.IS_DAQ_PC:
        frames: list[npt.NDArray] = []
        timestamps: list[int] = []
        for _ in range(frame_count):
            frames.append(
                _crop_frame_to_config(simulator.get_framegrabber_image(), config)
            )
            timestamps.append(time.time_ns())
        return np.stack(frames, axis=0), np.asarray(timestamps, dtype=np.int64)

    request_start_labview_ms = (
        time.time_ns() // 1_000_000 + LABVIEW_UNIX_EPOCH_OFFSET_MS
    )
    deadline = time.monotonic() + timeout_ms / 1000.0

    # This topic prefix is used by the LabVIEW panel (FrameGrabbber FSM UI2.vi) to
    # filter messages, so it must match the one used there.
    topic = "framegrabber/main "
    prefix = topic.encode("utf-8")

    with zmq.Context.instance().socket(zmq.SUB) as sock:
        sock.setsockopt(zmq.LINGER, 0)
        sock.setsockopt_string(zmq.SUBSCRIBE, topic)
        sock.connect(f"tcp://127.0.0.1:{constants.FRAMEGRAB_SERVER_PORT}")
        frames: list[npt.NDArray] = []
        timestamps: list[int] = []
        last_accepted_dt = request_start_labview_ms
        while len(frames) < frame_count:
            remaining_s = deadline - time.monotonic()
            if remaining_s <= 0.0:
                raise TimeoutError(f"No fresh frame received within {timeout_ms} ms")
            sock.setsockopt(zmq.RCVTIMEO, max(1, int(remaining_s * 1000)))
            try:
                msg = sock.recv()
            except zmq.Again as exc:
                raise TimeoutError(
                    f"No fresh frame received within {timeout_ms} ms"
                ) from exc

            payload = msg[len(prefix) :]
            meta_b, data = payload.split(b"\n", 1)
            meta = json.loads(meta_b.decode("utf-8"))
            frame_dt = int(meta["dt"])
            if frame_dt <= last_accepted_dt:
                continue

            raw = np.frombuffer(data, dtype=np.dtype(meta["dtype"])).reshape(
                tuple(meta["shape"])
            )
            frames.append(_crop_frame_to_config(raw, config))
            timestamps.append(_labview_ms_to_unix_ns(frame_dt))
            last_accepted_dt = frame_dt

    return np.stack(frames, axis=0), np.asarray(timestamps, dtype=np.int64)


def _labview_ms_to_unix_ns(labview_ms: int) -> int:
    return int(labview_ms - LABVIEW_UNIX_EPOCH_OFFSET_MS) * 1_000_000


def _validate_frame_count(frame_count: int) -> int:
    value = int(frame_count)
    if value < 1:
        raise ValueError("frame_count must be >= 1")
    return value


def _crop_frame_to_config(
    image: npt.ArrayLike,
    config: CameraConfig,
) -> npt.NDArray:
    array = np.asarray(image)
    if array.ndim < 2:
        raise RuntimeError(
            f"Framegrabber image must be at least 2D, got shape {array.shape!r}"
        )
    x0 = int(config.offset_x)
    y0 = int(config.offset_y)
    width = int(config.width)
    height = int(config.height)
    if x0 < 0 or y0 < 0 or width < 1 or height < 1:
        raise RuntimeError(
            "Framegrabber configured crop must have non-negative offsets and "
            f"positive size, got offset=({x0}, {y0}) size=({width}, {height})"
        )
    x1 = x0 + width
    y1 = y0 + height
    if array.shape[0] < y1 or array.shape[1] < x1:
        raise RuntimeError(
            f"Framegrabber image shape {array.shape[:2]!r} cannot satisfy "
            f"configured crop x={x0}..{x1}, y={y0}..{y1}"
        )
    return array[y0:y1, x0:x1, ...].copy()

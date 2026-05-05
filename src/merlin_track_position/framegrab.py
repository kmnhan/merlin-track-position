"""Access images from the FrameGrabber LabVIEW panel."""

import json

import numpy as np
import zmq

from merlin_track_position.constants import FRAMEGRAB_SERVER_PORT
import numpy.typing as npt


def get_framegrabber_image(timeout_ms: int = 5000) -> npt.NDArray[np.float64]:
    """Request the latest image from the FrameGrabber LabVIEW panel.

    Parameters
    ----------
    timeout_ms
        Maximum time set to the zmq RCVTIMEO option, to wait for a response before
        raising TimeoutError.
    """

    # This topic prefix is used by the LabVIEW panel (FrameGrabbber FSM UI2.vi) to
    # filter messages, so it must match the one used there.
    topic = "framegrabber/main "
    prefix = topic.encode("utf-8")

    with zmq.Context.instance().socket(zmq.SUB) as sock:
        sock.setsockopt(zmq.LINGER, 0)
        sock.setsockopt(zmq.RCVTIMEO, timeout_ms)
        sock.setsockopt_string(zmq.SUBSCRIBE, topic)
        sock.connect(f"tcp://127.0.0.1:{FRAMEGRAB_SERVER_PORT}")
        try:
            msg = sock.recv()
        except zmq.Again as exc:
            raise TimeoutError(f"No frame received within {timeout_ms} ms") from exc

    payload = msg[len(prefix) :]
    meta_b, data = payload.split(b"\n", 1)
    meta = json.loads(meta_b.decode("utf-8"))

    return (
        np.frombuffer(data, dtype=np.dtype(meta["dtype"]))
        .reshape(tuple(meta["shape"]))[:-1, :-17]  # crop empty border
        .copy()
    )

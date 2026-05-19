from __future__ import annotations

import argparse
import json
import math
import sys
import time
import uuid
from dataclasses import dataclass
from typing import Any

import zmq


DEFAULT_BIND = "tcp://127.0.0.1:6554"
DEFAULT_TIMEOUT_MS = 60_000
DEFAULT_BCS_API_CORRECTING_STATUSES = 1
XYZ_AXES = ("x", "y", "z")


@dataclass(frozen=True)
class Move:
    axes: tuple[str, ...]
    targets_mm: dict[str, float]
    timeout_ms: int


def _parse_move(spec: str, *, timeout_ms: int) -> Move:
    targets: dict[str, float] = {}
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise argparse.ArgumentTypeError(
                f"move item {item!r} must look like axis=value"
            )
        axis, value = item.split("=", 1)
        axis = axis.strip().lower()
        if axis not in XYZ_AXES:
            raise argparse.ArgumentTypeError(
                f"unsupported axis {axis!r}; expected one of {XYZ_AXES}"
            )
        try:
            target = float(value)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                f"target for axis {axis!r} must be numeric"
            ) from exc
        if not math.isfinite(target):
            raise argparse.ArgumentTypeError(
                f"target for axis {axis!r} must be finite"
            )
        targets[axis] = target
    if not targets:
        raise argparse.ArgumentTypeError("move must contain at least one axis=value")
    return Move(axes=tuple(targets), targets_mm=targets, timeout_ms=timeout_ms)


def _decode_frame(frame: bytes) -> str:
    try:
        return frame.decode("utf-8")
    except UnicodeDecodeError:
        return frame.decode("utf-8", errors="replace")


def _preview(text: str, limit: int = 800) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"... <truncated {len(text) - limit} chars>"


def _json_dumps(payload: Any) -> str:
    return json.dumps(payload, separators=(",", ":"))


def _pretty(payload: Any) -> str:
    return json.dumps(payload, indent=2, sort_keys=True)


def _send(socket: zmq.Socket, status: str, payload: dict[str, Any]) -> None:
    payload_text = _json_dumps(payload)
    print(f"\n--> reply status={status!r}")
    print(f"--> reply payload={_preview(payload_text)!r}")
    socket.send_multipart([status.encode("utf-8"), payload_text.encode("utf-8")])


def _recv_json(socket: zmq.Socket) -> tuple[list[bytes], str | None, Any | None]:
    frames = socket.recv_multipart()
    print("\n" + "=" * 80)
    print(f"<-- received {len(frames)} frame(s)")
    for index, frame in enumerate(frames):
        text = _decode_frame(frame)
        print(f"<-- frame[{index}] bytes={len(frame)} text={_preview(text)!r}")

    if not frames:
        return frames, None, None

    raw = _decode_frame(frames[0])
    try:
        request = json.loads(raw)
    except json.JSONDecodeError as exc:
        print(f"!! JSON decode failed: {exc}")
        print(f"!! raw length={len(raw)} preview={_preview(raw)!r}")
        return frames, raw, None

    print("<-- parsed JSON:")
    print(_pretty(request))
    return frames, raw, request


def _move_payload(
    *,
    session_id: str,
    move_id: int,
    move: Move,
    max_retries: int,
    current_positions_mm: dict[str, float],
) -> dict[str, Any]:
    targets_mm = {axis: float(current_positions_mm.get(axis, 0.0)) for axis in XYZ_AXES}
    targets_mm.update(move.targets_mm)
    return {
        "session_id": session_id,
        "move_id": move_id,
        "axes": list(move.axes),
        "targets_mm": targets_mm,
        "timeout_ms": move.timeout_ms,
        "max_retries": max_retries,
    }


def _status_payload(
    *,
    ok: bool,
    state: str,
    session_id: str,
    target: Any,
    message: str,
    pending_move: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = {
        "ok": ok,
        "state": state,
        "session_id": session_id,
        "target": target,
        "message": message,
    }
    if pending_move is not None:
        payload["pending_move"] = pending_move
    return payload


def _validate_move_result(
    request: dict[str, Any],
    *,
    expected_session_id: str,
    expected_move_id: int,
) -> list[str]:
    problems: list[str] = []
    if str(request.get("command", "")).upper() != "MOVE_RESULT":
        problems.append("command is not MOVE_RESULT")
    if str(request.get("session_id", "")) != expected_session_id:
        problems.append(
            f"session_id {request.get('session_id')!r} != {expected_session_id!r}"
        )
    try:
        move_id = int(request.get("move_id"))
    except Exception:
        problems.append(f"move_id {request.get('move_id')!r} is not an integer")
    else:
        if move_id != expected_move_id:
            problems.append(f"move_id {move_id!r} != {expected_move_id!r}")

    positions = request.get("positions_mm")
    if not isinstance(positions, dict):
        problems.append("positions_mm is not an object")
    else:
        for axis in XYZ_AXES:
            if axis not in positions:
                problems.append(f"positions_mm missing {axis!r}")
                continue
            try:
                value = float(positions[axis])
            except Exception:
                problems.append(f"positions_mm[{axis!r}] is not numeric")
            else:
                if not math.isfinite(value):
                    problems.append(f"positions_mm[{axis!r}] is not finite")
    return problems


def run_server(args: argparse.Namespace) -> int:
    moves = [_parse_move(spec, timeout_ms=args.timeout_ms) for spec in args.move]
    bcs_api_correcting_statuses = max(0, int(args.bcs_api_correcting_statuses))
    ctx = zmq.Context.instance()
    socket = ctx.socket(zmq.REP)
    socket.setsockopt(zmq.LINGER, 0)
    socket.bind(args.bind)

    print(f"Track Shift polling debug server listening on {args.bind}")
    if args.use_bcs_api_backend:
        print("Backend mode: BCS API (no pending_move responses).")
        print(
            "BCS API correction completes after "
            f"{bcs_api_correcting_statuses} correcting STATUS response(s)."
        )
        if moves:
            print("Ignoring --move entries because BCS API mode owns motor moves.")
    elif moves:
        print("Scripted STATUS moves:")
        for index, move in enumerate(moves, start=1):
            print(
                f"  {index}: axes={move.axes}, targets_mm={move.targets_mm}, "
                f"timeout_ms={move.timeout_ms}"
            )
    else:
        print("No --move supplied; first STATUS after START will report complete.")
    print("Press Ctrl-C to stop.\n")

    session_id = args.session_id or uuid.uuid4().hex
    target: Any = None
    state = "idle"
    message = ""
    next_move_index = 0
    pending_move: dict[str, Any] | None = None
    bcs_api_correcting_status_count = 0
    current_positions_mm = {axis: 0.0 for axis in XYZ_AXES}

    try:
        while True:
            _frames, raw, request = _recv_json(socket)
            if request is None:
                _send(
                    socket,
                    "ERROR",
                    _status_payload(
                        ok=False,
                        state="error",
                        session_id=session_id,
                        target=target,
                        message=(
                            "invalid JSON request from LabVIEW; "
                            f"raw length={len(raw or '')}; "
                            f"preview={_preview(raw or '')!r}"
                        ),
                    ),
                )
                continue

            command = str(request.get("command", "")).upper()
            if command == "START":
                if state in {"correcting", "move_pending"}:
                    _send(
                        socket,
                        "ERROR",
                        _status_payload(
                            ok=False,
                            state=state,
                            session_id=session_id,
                            target=target,
                            message=(
                                "a Track Shift correction session is already active"
                            ),
                            pending_move=(
                                None if args.use_bcs_api_backend else pending_move
                            ),
                        ),
                    )
                    continue

                session_id = str(request.get("session_id") or session_id)
                target = request.get("target")
                positions = request.get("positions_mm")
                if isinstance(positions, dict):
                    for axis in XYZ_AXES:
                        try:
                            current_positions_mm[axis] = float(positions.get(axis, 0.0))
                        except Exception:
                            current_positions_mm[axis] = 0.0
                next_move_index = 0
                pending_move = None
                bcs_api_correcting_status_count = 0
                state = "correcting"
                if args.use_bcs_api_backend:
                    message = "debug server: BCS API correction started"
                else:
                    message = "debug server: correction started"
                print(f"START target={target!r}")
                print(f"START positions_mm={request.get('positions_mm')!r}")
                _send(
                    socket,
                    "OK",
                    _status_payload(
                        ok=True,
                        state=state,
                        session_id=session_id,
                        target=target,
                        message=message,
                    ),
                )
                continue

            if command == "STATUS":
                if args.use_bcs_api_backend and state == "correcting":
                    if (
                        bcs_api_correcting_status_count
                        < bcs_api_correcting_statuses
                    ):
                        bcs_api_correcting_status_count += 1
                        message = (
                            "debug server: BCS API correction running "
                            f"({bcs_api_correcting_status_count}/"
                            f"{bcs_api_correcting_statuses})"
                        )
                    else:
                        state = "complete"
                        message = "debug server: BCS API correction complete"
                elif state == "correcting" and pending_move is None:
                    if next_move_index < len(moves):
                        move_id = next_move_index + 1
                        pending_move = _move_payload(
                            session_id=session_id,
                            move_id=move_id,
                            move=moves[next_move_index],
                            max_retries=args.max_retries,
                            current_positions_mm=current_positions_mm,
                        )
                        next_move_index += 1
                        state = "move_pending"
                        message = "debug server: LabVIEW XYZ move pending"
                    else:
                        state = "complete"
                        message = "debug server: scripted moves complete"

                _send(
                    socket,
                    "OK",
                    _status_payload(
                        ok=state != "error",
                        state=state,
                        session_id=session_id,
                        target=target,
                        message=message,
                        pending_move=(
                            None if args.use_bcs_api_backend else pending_move
                        ),
                    ),
                )
                continue

            if command == "MOVE_RESULT":
                if args.use_bcs_api_backend:
                    _send(
                        socket,
                        "ERROR",
                        _status_payload(
                            ok=False,
                            state=state,
                            session_id=session_id,
                            target=target,
                            message=(
                                "MOVE_RESULT received while using BCS API backend; "
                                "no pending LabVIEW move was published"
                            ),
                        ),
                    )
                    continue

                if pending_move is None:
                    _send(
                        socket,
                        "ERROR",
                        _status_payload(
                            ok=False,
                            state=state,
                            session_id=session_id,
                            target=target,
                            message="MOVE_RESULT received with no pending move",
                        ),
                    )
                    continue

                problems = _validate_move_result(
                    request,
                    expected_session_id=session_id,
                    expected_move_id=int(pending_move["move_id"]),
                )
                if problems:
                    print("!! MOVE_RESULT validation problems:")
                    for problem in problems:
                        print(f"   - {problem}")
                    _send(
                        socket,
                        "ERROR",
                        _status_payload(
                            ok=False,
                            state=state,
                            session_id=session_id,
                            target=target,
                            message="; ".join(problems),
                            pending_move=pending_move,
                        ),
                    )
                    continue

                ok = bool(request.get("ok", False))
                print(f"MOVE_RESULT ok={ok!r} message={request.get('message')!r}")
                positions = request.get("positions_mm")
                if isinstance(positions, dict):
                    for axis in XYZ_AXES:
                        current_positions_mm[axis] = float(positions[axis])

                if ok:
                    pending_move = None
                    state = "correcting"
                    message = "debug server: MOVE_RESULT accepted"
                else:
                    pending_move = None
                    state = "error"
                    message = (
                        "debug server: LabVIEW reported failed MOVE_RESULT: "
                        + str(request.get("message", ""))
                    )

                _send(
                    socket,
                    "OK",
                    _status_payload(
                        ok=state != "error",
                        state=state,
                        session_id=session_id,
                        target=target,
                        message=message,
                    ),
                )
                continue

            if command == "ABORT":
                requested_session_id = request.get("session_id")
                if (
                    requested_session_id
                    and session_id
                    and str(requested_session_id) != session_id
                ):
                    _send(
                        socket,
                        "ERROR",
                        _status_payload(
                            ok=False,
                            state=state,
                            session_id=session_id,
                            target=target,
                            message=(
                                f"ABORT session_id {requested_session_id!r} "
                                f"did not match active session {session_id!r}"
                            ),
                            pending_move=(
                                None if args.use_bcs_api_backend else pending_move
                            ),
                        ),
                    )
                    continue

                pending_move = None
                state = "error"
                message = str(request.get("message") or "debug server: aborted")
                _send(
                    socket,
                    "OK",
                    _status_payload(
                        ok=False,
                        state=state,
                        session_id=session_id,
                        target=target,
                        message=message,
                    ),
                )
                continue

            _send(
                socket,
                "ERROR",
                _status_payload(
                    ok=False,
                    state=state,
                    session_id=session_id,
                    target=target,
                    message=f"unknown command {command!r}",
                    pending_move=None if args.use_bcs_api_backend else pending_move,
                ),
            )
    except KeyboardInterrupt:
        print("\nStopping debug server.")
        return 0
    finally:
        socket.close(0)
        time.sleep(0.05)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Standalone Track Shift polling debug server for LabVIEW driver testing."
        )
    )
    parser.add_argument("--bind", default=DEFAULT_BIND, help=f"default: {DEFAULT_BIND}")
    parser.add_argument(
        "--move",
        action="append",
        default=[],
        metavar="AXIS=MM[,AXIS=MM...]",
        help=(
            "script one pending move returned by STATUS, e.g. "
            "--move x=0.5,z=-1.4. Repeat for multiple iterative moves."
        ),
    )
    parser.add_argument(
        "--timeout-ms",
        type=int,
        default=DEFAULT_TIMEOUT_MS,
        help=f"timeout_ms included in pending moves; default {DEFAULT_TIMEOUT_MS}",
    )
    parser.add_argument("--max-retries", type=int, default=4)
    parser.add_argument("--session-id", default="")
    parser.add_argument(
        "--use-bcs-api-backend",
        action="store_true",
        help=(
            "emulate MOTOR_SERVER_USE_BCS_API_BACKEND=True: STATUS never includes "
            "pending_move and MOVE_RESULT is rejected"
        ),
    )
    parser.add_argument(
        "--bcs-api-correcting-statuses",
        type=int,
        default=DEFAULT_BCS_API_CORRECTING_STATUSES,
        metavar="N",
        help=(
            "in --use-bcs-api-backend mode, return N correcting STATUS replies "
            "after START before reporting complete; default "
            f"{DEFAULT_BCS_API_CORRECTING_STATUSES}"
        ),
    )
    return run_server(parser.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

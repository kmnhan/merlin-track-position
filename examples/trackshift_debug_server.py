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


def _send(socket: zmq.Socket, status: str, payload: str | dict[str, Any]) -> None:
    if isinstance(payload, dict):
        payload_text = _json_dumps(payload)
    else:
        payload_text = payload
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


def _validate_move_result(
    request: dict[str, Any],
    *,
    expected_session_id: str,
    expected_move_id: int,
) -> None:
    problems: list[str] = []
    if request.get("command") != "MOVE_RESULT":
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

    if problems:
        print("!! MOVE_RESULT validation problems:")
        for problem in problems:
            print(f"   - {problem}")
    else:
        print("MOVE_RESULT validation: OK")


def run_server(args: argparse.Namespace) -> int:
    moves = [_parse_move(spec, timeout_ms=args.timeout_ms) for spec in args.move]
    ctx = zmq.Context.instance()
    socket = ctx.socket(zmq.REP)
    socket.setsockopt(zmq.LINGER, 0)
    socket.bind(args.bind)

    print(f"Track Shift debug server listening on {args.bind}")
    if moves:
        print("Scripted moves:")
        for index, move in enumerate(moves, start=1):
            print(
                f"  {index}: axes={move.axes}, targets_mm={move.targets_mm}, "
                f"timeout_ms={move.timeout_ms}"
            )
    else:
        print("No --move supplied; START will receive OK without XYZ motion.")
    print("Press Ctrl-C to stop.\n")

    session_id = args.session_id or uuid.uuid4().hex
    next_move_index = 0
    awaiting_move_id: int | None = None
    current_positions_mm = {axis: 0.0 for axis in XYZ_AXES}

    try:
        while True:
            _frames, raw, request = _recv_json(socket)
            if request is None:
                _send(
                    socket,
                    "ERROR",
                    "invalid JSON request from LabVIEW; "
                    f"raw length={len(raw or '')}; preview={_preview(raw or '')!r}",
                )
                continue

            command = str(request.get("command", "")).upper()
            if command in ("", "START"):
                session_id = str(request.get("session_id") or session_id)
                positions = request.get("positions_mm")
                if isinstance(positions, dict):
                    for axis in XYZ_AXES:
                        try:
                            current_positions_mm[axis] = float(positions.get(axis, 0.0))
                        except Exception:
                            current_positions_mm[axis] = 0.0
                next_move_index = 0
                awaiting_move_id = None
                print(f"START target={request.get('target')!r}")
                print(f"START positions_mm={request.get('positions_mm')!r}")
                if not moves:
                    _send(socket, "OK", "debug server: no scripted moves configured")
                    continue

                move_id = 1
                awaiting_move_id = move_id
                payload = _move_payload(
                    session_id=session_id,
                    move_id=move_id,
                    move=moves[next_move_index],
                    max_retries=args.max_retries,
                    current_positions_mm=current_positions_mm,
                )
                next_move_index += 1
                _send(socket, "MOVE", payload)
                continue

            if command == "MOVE_RESULT":
                if awaiting_move_id is None:
                    _send(socket, "ERROR", "MOVE_RESULT received with no pending MOVE")
                    continue

                _validate_move_result(
                    request,
                    expected_session_id=session_id,
                    expected_move_id=awaiting_move_id,
                )
                ok = bool(request.get("ok", False))
                print(f"MOVE_RESULT ok={ok!r} message={request.get('message')!r}")
                positions = request.get("positions_mm")
                if isinstance(positions, dict):
                    for axis in XYZ_AXES:
                        try:
                            current_positions_mm[axis] = float(
                                positions.get(axis, current_positions_mm[axis])
                            )
                        except Exception:
                            pass

                if not ok:
                    _send(
                        socket,
                        "ERROR",
                        "debug server: LabVIEW reported failed MOVE_RESULT: "
                        + str(request.get("message", "")),
                    )
                    awaiting_move_id = None
                    continue

                if next_move_index >= len(moves):
                    _send(socket, "OK", "debug server: scripted moves complete")
                    awaiting_move_id = None
                    continue

                move_id = awaiting_move_id + 1
                awaiting_move_id = move_id
                payload = _move_payload(
                    session_id=session_id,
                    move_id=move_id,
                    move=moves[next_move_index],
                    max_retries=args.max_retries,
                    current_positions_mm=current_positions_mm,
                )
                next_move_index += 1
                _send(socket, "MOVE", payload)
                continue

            _send(socket, "ERROR", f"unknown command {command!r}")
    except KeyboardInterrupt:
        print("\nStopping debug server.")
        return 0
    finally:
        socket.close(0)
        time.sleep(0.05)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Standalone Track Shift ZMQ debug server for LabVIEW driver testing."
        )
    )
    parser.add_argument("--bind", default=DEFAULT_BIND, help=f"default: {DEFAULT_BIND}")
    parser.add_argument(
        "--move",
        action="append",
        default=[],
        metavar="AXIS=MM[,AXIS=MM...]",
        help=(
            "script one MOVE reply, e.g. --move x=0.5,z=-1.4. "
            "Repeat for multiple iterative moves. If omitted, START replies OK."
        ),
    )
    parser.add_argument(
        "--timeout-ms",
        type=int,
        default=DEFAULT_TIMEOUT_MS,
        help=f"timeout_ms included in MOVE payloads; default {DEFAULT_TIMEOUT_MS}",
    )
    parser.add_argument("--max-retries", type=int, default=4)
    parser.add_argument("--session-id", default="")
    return run_server(parser.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())

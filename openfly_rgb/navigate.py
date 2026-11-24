"""Run OpenFly on first-person RGB from the room65 simulator."""
import argparse
import base64
from contextlib import ExitStack, contextmanager
from datetime import datetime, timezone
from io import BytesIO
import json
import math
import os
from pathlib import Path
import subprocess

from PIL import Image


PROJECT = Path(__file__).resolve().parents[1]
ROOM = PROJECT / "scenes" / "room65_selected_anchored"
CACHE = Path(os.environ.get("OPENFLY_CACHE_DIR", Path.home() / ".cache" / "openfly-room65")).expanduser()
DEFAULT_INSTRUCTION = (
    "Fly along the side of the conference table toward the door. Go around the end "
    "of the table, then turn to face the wall-mounted air conditioner and hover."
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instruction", help="Navigation instruction; defaults to the room65 conference-table route.")
    parser.add_argument("--spawn", type=float, nargs=3, metavar=("X", "Y", "Z"), help="Initial position in simulator meters.")
    parser.add_argument("--yaw", type=float, help="Initial heading in degrees.")
    parser.add_argument("--goal", type=float, nargs=3, metavar=("X", "Y", "Z"), help="Evaluation-only target.")
    parser.add_argument("--goal-yaw", type=float, help="Optional evaluation-only final heading in degrees.")
    parser.add_argument("--max-decisions", type=int, default=40)
    parser.add_argument("--resolution", type=float, default=1., choices=(.5, .75, 1.),
                        help="Scale of the calibrated 640x480 RGB camera.")
    parser.add_argument("--output", type=Path, help="New run directory; defaults to outputs/openfly_rgb_<timestamp>.")
    parser.add_argument("--model-python", type=Path, default=CACHE / "venv/bin/python")
    parser.add_argument("--openfly-root", type=Path, default=CACHE / "OpenFly-Platform")
    parser.add_argument("--checkpoint", type=Path, default=CACHE / "openfly-agent-7b")
    parser.add_argument("--device", default="cuda:0", help="Model device (cuda:0, cpu, auto); the renderer uses CUDA.")
    parser.add_argument("--precision", default="int4", choices=("int4", "bf16", "fp16", "fp32"))
    parser.add_argument("--unnorm-key", default="vlnv1", help="Published action statistics key.")
    parser.add_argument("--room-head", type=Path, help="Use a fitted indoor RGB waypoint readout (.npz).")
    args = parser.parse_args()
    if args.max_decisions < 1:
        parser.error("--max-decisions must be positive")
    if args.instruction is not None and not args.instruction.strip():
        parser.error("--instruction cannot be empty")
    if args.goal_yaw is not None and args.goal is None and args.instruction is not None:
        parser.error("--goal-yaw requires --goal for a custom instruction")
    numbers = (args.spawn or []) + (args.goal or []) + [v for v in (args.yaw, args.goal_yaw) if v is not None]
    if not all(map(math.isfinite, numbers)):
        parser.error("positions and headings must be finite")
    return args


def worker_reply(process):
    line = process.stdout.readline()
    if not line:
        raise RuntimeError(f"OpenFly worker closed its output (exit={process.poll()}); see model.log")
    message = json.loads(line)
    if "error" in message:
        raise RuntimeError(f"OpenFly worker: {message['error']}; see model.log")
    return message


@contextmanager
def open_worker(args, device, log, *, extract_features=False):
    cmd = [str(args.model_python.expanduser()), "-B", str(Path(__file__).with_name("model_worker.py")),
           "--openfly-root", str(args.openfly_root.expanduser().resolve()),
           "--checkpoint", str(args.checkpoint.expanduser().resolve()), "--local-files-only",
           "--device", device, "--precision", args.precision, "--unnorm-key", args.unnorm_key]
    if extract_features:
        cmd.append("--extract-features")
    elif getattr(args, "room_head", None):
        cmd.extend(["--room-head", str(args.room_head.expanduser().resolve())])
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=log,
                            text=True, encoding="utf-8", bufsize=1)
    try:
        if not worker_reply(proc).get("ready"):
            raise RuntimeError("Unexpected OpenFly worker handshake; see model.log")
        yield proc
    finally:
        try:
            proc.stdin.close()
        except BrokenPipeError:
            pass
        proc.stdout.close()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()


def query_worker(proc, instruction, rgb):
    buffer = BytesIO()
    Image.fromarray(rgb).save(buffer, format="PNG")
    image = base64.b64encode(buffer.getvalue()).decode("ascii")
    proc.stdin.write(json.dumps({"instruction": instruction, "image": image}) + "\n")
    proc.stdin.flush()
    return worker_reply(proc)


def decode_prediction(reply):
    raw = reply["raw_action"]
    action = reply["action_id"]
    if len(raw) != 8 or not all(map(math.isfinite, raw)):
        raise ValueError("OpenFly returned a malformed action vector")
    if action is not None and (type(action) is not int or not 0 <= action <= 9):
        raise ValueError(f"Invalid OpenFly action ID: {action!r}")
    return action


def evaluate_stop(state, outcome, goal, goal_yaw):
    if goal is None:
        return {"success": None}
    distance = math.dist(state["position"], goal)
    yaw_error = None if goal_yaw is None else abs((state["euler_deg"][2] - goal_yaw + 180.) % 360. - 180.)
    success = (outcome == "policy_stop" and distance <= .30 and state["speed"] <= .10
               and (yaw_error is None or yaw_error <= 20.))
    return {"goal_m": goal, "goal_yaw_deg": goal_yaw, "distance_m": distance,
            "yaw_error_deg": yaw_error, "final_speed_mps": state["speed"], "success": success}


def evaluate_table_route(frames_path):
    import numpy as np
    from scipy.spatial import cKDTree

    def sample_positions(path):
        with path.open() as rows:
            points = np.array([json.loads(row)["state"]["position"] for row in rows])
        lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
        keep = np.r_[True, lengths > 1e-8]
        points = points[keep]
        if len(points) == 1:
            return points
        arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
        distances = np.linspace(0., arc[-1], int(np.ceil(arc[-1] / .05)) + 1)
        return np.column_stack([np.interp(distances, arc, column) for column in points.T])

    ref = sample_positions(ROOM / "flight/frames.jsonl")
    actual = sample_positions(frames_path)
    radius = .35
    corners = np.asarray(json.loads((ROOM / "route.json").read_text())["knots"])[[3, 4], 1:4]
    next_corner = 0
    for point in actual:
        if np.linalg.norm(point - corners[next_corner]) <= radius:
            next_corner += 1
            if next_corner == len(corners):
                break
    return {"radius_m": radius, "reference_coverage": float((cKDTree(actual).query(ref)[0] <= radius).mean()),
            "path_adherence": float((cKDTree(ref).query(actual)[0] <= radius).mean()),
            "ordered_table_corners": int(next_corner), "required_table_corners": len(corners)}


def free_gpu_memory(torch):
    cuda_free, _ = torch.cuda.mem_get_info()
    # Under WSL, CUDA can report pageable memory beyond the card's free VRAM.
    uuid = str(getattr(torch.cuda.get_device_properties(0), "uuid", ""))
    if uuid:
        try:
            result = subprocess.run(["nvidia-smi", "-i", "GPU-" + uuid.removeprefix("GPU-"),
                                     "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                                    check=True, capture_output=True, text=True, timeout=4)
            return min(cuda_free, int(result.stdout.strip()) * 1024**2)
        except (OSError, subprocess.SubprocessError, ValueError):
            pass
    return cuda_free


def resolve_device(args):
    device = args.device
    if device == "auto":
        import torch
        # Include model activations and headroom for the concurrent room renderer.
        required_gib = {"int4": 7, "bf16": 18, "fp16": 18, "fp32": 33}[args.precision]
        free_bytes = free_gpu_memory(torch)
        device = "cuda:0" if free_bytes >= required_gib * 1024**3 else "cpu"
        print(f"Model device: {device} ({free_bytes / 1024**3:.1f} GiB GPU memory free)", flush=True)
    return device


def run(args):
    from openfly_rgb.flight import ACTION_NAMES, RoomFlight
    from record_sequence import FpvVideoWriter, verify_video

    device = resolve_device(args)
    instruction = args.instruction or DEFAULT_INSTRUCTION
    goal, goal_yaw = args.goal, args.goal_yaw
    if args.instruction is None and goal is None:
        end = json.loads((ROOM / "route.json").read_text())["knots"][-1]
        goal = end[1:4]
        if goal_yaw is None:
            goal_yaw = end[4]
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    output = (args.output or PROJECT / "outputs" / f"openfly_rgb_{timestamp}").expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    print(f"Recording to {output}", flush=True)
    summary = {"policy": "OpenFly-Agent", "checkpoint": str(args.checkpoint.expanduser().resolve()),
               "precision": args.precision, "device": device, "unnorm_key": args.unnorm_key,
               "scene": "room65_selected_anchored/models/rgb.ply", "instruction": instruction,
               "room_head": str(args.room_head.expanduser().resolve()) if args.room_head else None,
               "resolution": args.resolution, "max_decisions": args.max_decisions, "decisions": 0}
    try:
        with ExitStack() as stack:
            log = stack.enter_context((output / "model.log").open("w"))
            proc = stack.enter_context(open_worker(args, device, log))
            flight = stack.enter_context(RoomFlight(spawn=args.spawn, yaw_deg=args.yaw, resolution=args.resolution,
                                                   max_steps=max(1200, 40 * args.max_decisions + 1)
                                                   if args.room_head else None))
            frames = stack.enter_context((output / "frames.jsonl").open("w", buffering=1))
            decisions = stack.enter_context((output / "decisions.jsonl").open("w", buffering=1))
            h, w = flight.rgb.shape[:2]
            video = FpvVideoWriter(output / "rgb_fpv.mp4", w, h)
            stack.callback(video.close)

            def record(receipt, rgb):
                keys = ("frame_id", "sim_time", "state", "raw_action", "action_format", "applied_control",
                        "substeps_executed", "collision", "terminated", "truncated", "reason")
                row = {key: receipt[key] for key in keys}
                frames.write(json.dumps(row, allow_nan=False) + "\n")
                video.write(rgb)

            record(flight.receipt, flight.rgb)
            outcome = flight.outcome
            for step in range(args.max_decisions):
                if outcome:
                    break
                observed_frame = flight.receipt["frame_id"]
                prediction = query_worker(proc, instruction, flight.rgb)
                if args.room_head:
                    waypoint = prediction["waypoint"]
                    stop = prediction["stop"]
                    if len(waypoint) != 4 or not all(map(math.isfinite, waypoint)) or type(stop) is not bool:
                        raise ValueError("Malformed indoor waypoint prediction")
                    action_name = "stop" if stop else "waypoint"
                else:
                    action = decode_prediction(prediction)
                    action_name = "invalid" if action is None else ACTION_NAMES[action]
                summary["decisions"] = step + 1
                decisions.write(json.dumps({"decision": step + 1, "observed_frame": observed_frame,
                                            "sim_time": flight.receipt["sim_time"],
                                            "action": action_name, **prediction}, allow_nan=False) + "\n")
                if args.room_head:
                    outcome = flight.execute_waypoint(waypoint, stop=stop, on_tick=record)
                else:
                    outcome = "invalid_model_action" if action is None else flight.execute(action, on_tick=record)
                print(f"Decision {step + 1}: {action_name}, frame={flight.receipt['frame_id']}, "
                      f"outcome={outcome or 'flying'}", flush=True)
            outcome = outcome or "decision_limit"
            summary.update(outcome=outcome, frames=video.frames, path_m=flight.path_m,
                           sim_time_s=flight.receipt["sim_time"],
                           evaluation=evaluate_stop(flight.receipt["state"], outcome, goal, goal_yaw))
        summary["video_check"] = verify_video(output / "rgb_fpv.mp4", summary["frames"])
        if instruction == DEFAULT_INSTRUCTION:
            summary["route_check"] = evaluate_table_route(output / "frames.jsonl")
    except (OSError, ValueError, RuntimeError, KeyError, KeyboardInterrupt) as exc:
        summary.update(outcome="run_error", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        (output / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    run(parse_args())

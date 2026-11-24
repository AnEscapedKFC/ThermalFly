"""Fit an OpenFly continuous waypoint readout on the recorded conference-room route."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import tempfile

import numpy as np
from scipy.spatial.transform import Rotation

from openfly_rgb.navigate import CACHE, DEFAULT_INSTRUCTION, ROOM

# forward metres, left metres, yaw degrees, level body attitude.
# Vary heading and translation independently before including combined errors.
VIEWS = (
    (0., 0., 0., False),
    (0., .14, 0., True), (0., -.14, 0., True),
    (0., 0., 12., True), (0., 0., -12., True),
    (.12, 0., 0., True), (-.12, 0., 0., True),
    (0., 0., 0., True),
    (.10, .12, 22., True), (-.10, -.12, -22., True),
    (.08, -.14, -8., False), (-.08, .14, 8., False),
)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True, help="New .npz room readout")
    p.add_argument("--resolution", type=float, choices=(.5, .75, 1.), default=.5)
    p.add_argument("--augment", type=int, choices=range(len(VIEWS)), default=len(VIEWS) - 1,
                   help="Number of additional RGB sequences with perturbed pose or level attitude")
    p.add_argument("--model-python", type=Path, default=CACHE / "venv/bin/python")
    p.add_argument("--openfly-root", type=Path, default=CACHE / "OpenFly-Platform")
    p.add_argument("--checkpoint", type=Path, default=CACHE / "openfly-agent-7b")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--precision", choices=("int4", "bf16", "fp16", "fp32"), default="int4")
    p.add_argument("--unnorm-key", default="vlnv1")
    args = p.parse_args()
    if args.output.suffix != ".npz":
        p.error("--output must end in .npz")
    return args


def demo_samples():
    """Sample the original expert states at one-second decisions and label terminal hover."""
    route = json.loads((ROOM / "route.json").read_text())
    knots = np.asarray(route["knots"])
    moving = np.flatnonzero(np.any(np.diff(knots[:, 1:], axis=0) != 0, axis=1))
    if not moving.size or moving[-1] + 1 == len(knots) - 1:
        raise ValueError("The demonstration requires a motion segment followed by terminal hover")
    start = knots[moving[0], 0]
    terminal = knots[moving[-1] + 1, 0]
    with (ROOM / "flight/frames.jsonl").open() as f:
        rows = [json.loads(line) for line in f]
    times = np.array([r["sim_time"] for r in rows])
    if np.any(np.diff(times) <= 0) or any(r["collision"] for r in rows):
        raise ValueError("The room demonstration requires ordered, collision-free frames")
    # Include two terminal observations with different causal frame histories.
    samples = []
    for t in np.arange(start, min(terminal + 1., times[-1]) + 1e-5, 1.):
        now = rows[np.argmin(np.abs(times - t))]
        later = rows[np.argmin(np.abs(times - min(t + 1., times[-1])))]
        samples.append((now["state"], later["state"], bool(t >= terminal - 1e-5)))
    return samples


def sample_pose_target(state, future, stop, side_m, yaw_deg, forward_m=0., level=False):
    """Render the displaced current pose; supervise progress and correction toward the route."""
    heading = np.deg2rad(state["euler_deg"][2])
    left = np.array([-np.sin(heading), np.cos(heading), 0.])
    forward = np.array([np.cos(heading), np.sin(heading), 0.])
    offset = side_m * left + forward_m * forward
    position = np.asarray(state["position"]) + offset
    current_yaw = state["euler_deg"][2] + yaw_deg
    if level:
        orientation = Rotation.from_euler("z", current_yaw, degrees=True)
    else:
        original = Rotation.from_quat(np.roll(state["quaternion_wxyz"], -1))
        orientation = Rotation.from_euler("z", yaw_deg, degrees=True) * original
    quat = np.roll(orientation.as_quat(), 1)
    # fpv_camera consumes body xyz and wxyz at the physical state offsets.
    body = np.r_[position, state["velocity"], quat]
    difference = np.asarray(future["position"]) - position
    turn = (future["euler_deg"][2] - current_yaw + 180.) % 360. - 180.
    if stop and np.linalg.norm(difference) <= .18 and abs(turn) <= 15.:
        return body, [0., 0., 0., 0., 1.]
    body_from_world = Rotation.from_euler("z", -current_yaw, degrees=True)
    # Relative to the nominal expert displacement, subtract the observed
    # position and heading offsets. Bound recovery using the same action units
    # and physical limits as the online head.
    movement = body_from_world.apply(difference)
    planar = np.linalg.norm(movement[:2])
    if planar > .70:
        movement[:2] *= .70 / planar
    movement[2] = np.clip(movement[2], -.30, .30)
    turn = np.clip(turn, -35., 35.)
    return body, [*movement, turn, 0.]


def run(args):
    from cameras import fpv_camera
    from openfly_rgb.flight import RigScene, WAYPOINT_HOLD_SECONDS, WAYPOINT_MOVE_SECONDS
    from openfly_rgb.model_worker import fit_room_head
    from openfly_rgb.navigate import open_worker, query_worker, resolve_device, worker_reply

    output = args.output.expanduser().resolve()
    log = output.with_suffix(".model.log")
    manifest = output.with_suffix(".fit.json")
    if output.exists() or manifest.exists():
        raise FileExistsError(f"Choose a new output basename: {output}")
    if log.exists():
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
        log = log.with_name(f"{output.stem}.{stamp}.model.log")
    output.parent.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args)
    samples = demo_samples()
    variants = VIEWS[:args.augment + 1]
    width = round(640 * args.resolution)
    rig = RigScene(ROOM / "models/rgb.ply", width, round(.75 * width), "z", 1., "cuda", render_thermal=False)
    features, targets = [], []
    with log.open("x") as stream, open_worker(args, device, stream, extract_features=True) as proc:
        for forward, side, yaw, level in variants:
            proc.stdin.write('{"reset":true}\n')
            proc.stdin.flush()
            if worker_reply(proc).get("reset") is not True:
                raise RuntimeError("Could not reset OpenFly history between demonstrations")
            for state, future, stop in samples:
                body, label = sample_pose_target(state, future, stop, side, yaw, forward, level)
                rgb = rig.render(fpv_camera(body))[0]
                frame = np.rint(255 * rgb).astype(np.uint8)
                features.append(query_worker(proc, DEFAULT_INSTRUCTION, frame)["feature"])
                targets.append(label)
                if len(features) % 5 == 0:
                    print(f"Extracted {len(features)}/{len(samples) * len(variants)} samples", flush=True)
    # Publish the fitted archive atomically, after all model calls have succeeded.
    with tempfile.TemporaryDirectory(dir=output.parent) as folder:
        temporary = Path(folder) / output.name
        fit_room_head(np.asarray(features, np.float32), np.asarray(targets, np.float32), temporary)
        os.link(temporary, output)
    metadata = {
        "training_route": "room65_selected_anchored/flight/frames.jsonl", "scene": "models/rgb.ply",
        "instruction": DEFAULT_INSTRUCTION, "expert_label_horizon_s": 1., "samples": len(features),
        "controller_tracking_s": WAYPOINT_MOVE_SECONDS, "controller_settle_s": WAYPOINT_HOLD_SECONDS,
        "stops": sum(int(y[-1]) for y in targets), "augmentation_forward_m_left_m_yaw_deg_level": variants,
        "model_log": log.name,
        "resolution": args.resolution, "checkpoint": str(args.checkpoint.expanduser().resolve()),
        "precision": args.precision, "device": device,
    }
    manifest.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n")
    print(f"Wrote {output} from {len(features)} labeled current-scene RGB observations", flush=True)


if __name__ == "__main__":
    run(parse_args())

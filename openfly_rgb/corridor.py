"""Fit a route-specific RGB readout and fly it with the corridor RGB/thermal rig."""
import argparse
from contextlib import ExitStack
import csv
from functools import partial
import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np

from openfly_rgb.fit_room import sample_pose_target
from openfly_rgb.navigate import (CACHE, PROJECT, decode_prediction, evaluate_stop, open_worker,
                                  query_worker, resolve_device, worker_reply)

SCENE = PROJECT / "scenes/corridor_native"
DEFAULT_TASK = SCENE / "navigation_task.json"


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("fit", "fly"))
    parser.add_argument("--task", type=Path, default=DEFAULT_TASK)
    parser.add_argument("--output", type=Path, required=True, help="New directory for this run")
    parser.add_argument("--room-head", type=Path, help="Fitted .npz; omit to run the published discrete head")
    parser.add_argument("--fit-profile", choices=("route-only", "paired"), default="paired",
                        help="Use the initial route-only fit or paired navigation/hold supervision")
    parser.add_argument("--instruction", help="Override the instruction during flight")
    parser.add_argument("--goal", type=float, nargs=4, metavar=("X", "Y", "Z", "YAW"),
                        help="Evaluation target for an overridden instruction")
    parser.add_argument("--compare-instruction", help="After flight, query another instruction on the identical first RGB frame with a fresh model history")
    parser.add_argument("--max-decisions", type=int, default=40)
    parser.add_argument("--resolution", type=float, default=.5, choices=(.5, .75, 1.))
    parser.add_argument("--rgb-exposure-ev", type=float, default=0.,
                        help="RGB camera exposure offset for flight, in stops (0 = original, negative = darker)")
    parser.add_argument("--save-policy-inputs", action="store_true",
                        help="Save the lossless RGB image supplied at each decision and the initial thermal view")
    parser.add_argument("--checkpoint", type=Path, default=CACHE / "openfly-agent-7b")
    parser.add_argument("--openfly-root", type=Path, default=CACHE / "OpenFly-Platform")
    parser.add_argument("--model-python", type=Path, default=CACHE / "venv/bin/python")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--precision", default="int4", choices=("int4", "fp16", "bf16", "fp32"))
    parser.add_argument("--unnorm-key", default="vlnv1")
    args = parser.parse_args()
    if args.max_decisions < 1 or (args.mode == "fit" and args.room_head is not None):
        parser.error("Choose a positive decision budget; --room-head applies only to fly.")
    if args.mode != "fly" and (args.instruction is not None or args.goal is not None):
        parser.error("--instruction and --goal are fly options.")
    if args.instruction is not None and not args.instruction.strip():
        parser.error("--instruction must contain text.")
    if args.goal is not None and not np.isfinite(args.goal).all():
        parser.error("--goal must contain finite coordinates and heading.")
    if not np.isfinite(args.rgb_exposure_ev) or args.rgb_exposure_ev > 0:
        parser.error("--rgb-exposure-ev must be finite and <= 0.")
    if args.mode != "fly" and (args.rgb_exposure_ev != 0. or args.save_policy_inputs):
        parser.error("--rgb-exposure-ev and --save-policy-inputs apply to flights.")
    return args


def json_line(target, item):
    target.write(json.dumps(item, ensure_ascii=False, allow_nan=False) + "\n")


def read_reference(task):
    with (PROJECT / task["demonstration"]).open() as stream:
        rows = [json.loads(line) for line in stream]
    positions = np.asarray([r["state"]["position"] for r in rows])
    begin, end = (int(np.argmin(np.linalg.norm(positions-np.asarray(task[key][:3]), axis=1)))
                  for key in ("spawn_xyz_yaw", "goal_xyz_yaw"))
    if begin >= end or max(np.linalg.norm(positions[i]-np.asarray(task[k][:3])) for i, k in
                           ((begin, "spawn_xyz_yaw"), (end, "goal_xyz_yaw"))) > .16:
        raise ValueError("Demonstration must pass the annotated start and stopping landmark in order")
    rows = rows[begin:end+1]
    first, last = rows[0]["frame_id"], rows[-1]["frame_id"]
    if [r["frame_id"] for r in rows] != list(range(first, last+1)):
        raise ValueError("Incomplete expert trajectory segment")
    if any(r.get("reason") == "collision" or r.get("collision") is True or r.get("collision_event")
           for r in rows):
        raise ValueError("Collision in expert trajectory segment")
    return [r["state"] for r in rows], [first, last]


def fit(args, task, device):
    from cameras import fpv_camera
    from openfly_rgb.model_worker import fit_room_head
    from scenes.corridor_native.room import RigScene
    from scipy.spatial.transform import Rotation

    states, frame_range = read_reference(task)
    positions = np.array([s["position"] for s in states])
    arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(positions, axis=0), axis=1))]
    if arc[-1] < 1.:
        raise ValueError("Expert segment is too short")
    # Denser labels close to the stopping landmark, including a short stationary history.
    offsets = np.unique(np.clip(np.r_[np.arange(0, arc[-1], .28), arc[-1] - .32,
                                      arc[-1] - .14, arc[-1]], 0, arc[-1]))
    paired = args.fit_profile == "paired"
    if paired:
        offsets = np.unique(np.r_[offsets, arc[-1] - np.array([.48, .38, .26, .18, .10, .05])])
    offsets = np.r_[offsets, arc[-1], arc[-1]]
    # Augment camera pose independently of the future goal; repeat the same instruction.
    variants = [(front, side, yaw, 0., 0.) for front, side, yaw in
                ((0., 0., 0.), (0., .18, 0.), (0., -.18, 0.), (0., 0., 8.),
                 (0., 0., -8.), (.12, .06, 0.), (-.10, -.06, 0.), (.23, 0., 0.))]
    if paired:
        variants.extend(((0., 0., 0., -.07, 0.), (0., 0., 0., .07, 0.),
                         (0., .07, 0., -.055, -2.5), (0., -.07, 0., -.055, 2.5),
                         (0., .13, 0., -.10, 1.5), (0., -.13, 0., -.10, -1.5)))
    rig = RigScene(SCENE / "models/rgb.ply", round(640*args.resolution),
                   round(480*args.resolution), "z", 1., "cuda", render_thermal=False)
    features, labels = [], []
    hold_samples = 0
    with (args.output / "model.log").open("x") as log, open_worker(args, device, log, extract_features=True) as proc:
        for forward, side, yaw, height, pitch in variants:
            json_line(proc.stdin, {"reset": True})
            proc.stdin.flush()
            if not worker_reply(proc).get("reset"):
                raise RuntimeError("Could not reset model history")
            hover_views = []
            for i, d in enumerate(offsets):
                state = dict(states[np.argmin(abs(arc-d))])
                state["position"] = np.asarray(state["position"]) + [0., 0., height]
                future = states[np.argmin(abs(arc-min(d+.50, arc[-1])))]
                body, label = sample_pose_target(state, future, not paired and d >= arc[-1]-.145, side, yaw,
                                                 forward_m=forward, level=True)
                if paired:
                    yaw_degrees = state["euler_deg"][2] + yaw
                    body[6:10] = np.roll(Rotation.from_euler("ZY", [yaw_degrees, pitch],
                                                            degrees=True).as_quat(), 1)
                    if np.linalg.norm(body[:3]-positions[-1]) <= .10 and abs(yaw) <= 12.:
                        label = [0., 0., 0., 0., 1.]
                rgb = np.rint(255*rig.render(fpv_camera(body))[0]).astype(np.uint8)
                features.append(query_worker(proc, task["instruction"], rgb)["feature"])
                labels.append(label)
                if paired and i % 5 == 0:
                    hover_views.append(rgb)
            if hover_views:
                json_line(proc.stdin, {"reset": True})
                proc.stdin.flush()
                if not worker_reply(proc).get("reset"):
                    raise RuntimeError("Could not reset before stationary demonstrations")
                for rgb in hover_views:
                    features.append(query_worker(proc, task["hold_instruction"], rgb)["feature"])
                    labels.append([0., 0., 0., 0., 1.])
                hold_samples += len(hover_views)
            print(f"Encoded {len(features)} RGB/instruction samples", flush=True)
    with tempfile.TemporaryDirectory(dir=args.output) as tmp:
        head = Path(tmp) / "head.npz"
        fit_room_head(np.asarray(features, np.float32), np.asarray(labels, np.float32), head)
        os.link(head, args.output / "head.npz")
    return {"samples": len(labels), "stops": sum(int(x[-1]) for x in labels), "reference_length_m": arc[-1],
            "demonstration_frame_range": frame_range, "fit_profile": args.fit_profile,
            "navigation_samples": len(labels)-hold_samples, "hold_samples": hold_samples,
            "label_lookahead_m": .50, "reference_offsets_m": offsets.tolist(),
            "view_augment_forward_left_yaw_height_pitch": variants, "precision": args.precision,
            "checkpoint": str(args.checkpoint), **task}


def fly(args, task, device):
    from PIL import Image
    from openfly_rgb.flight import ACTION_NAMES, INDOOR_ACTIONS, RoomFlight, waypoint_trajectory
    from record_sequence import CSV_FIELDS, FpvVideoWriter, trajectory_row, verify_video
    from scenes.corridor_native.room import KnownSpace, RigScene

    start, end = task["spawn_xyz_yaw"], task["goal_xyz_yaw"]
    result = {"policy": "OpenFly-Agent 7B", "device": device, "precision": args.precision,
              "checkpoint": str(args.checkpoint), "room_head": str(args.room_head) if args.room_head else None,
              "instruction": task["instruction"], "spawn_xyz_yaw": start, "goal_xyz_yaw": end,
              "task_file": str(args.task), "max_decisions": args.max_decisions,
              "model_inputs": ["instruction", "current RGB and two prior decision RGB frames"],
              "video": "RGB left, thermal right (recording only)", "decisions": 0}
    with ExitStack() as stack:
        log = stack.enter_context((args.output/"model.log").open("x"))
        proc = stack.enter_context(open_worker(args, device, log))
        flight = stack.enter_context(RoomFlight(spawn=start[:3], yaw_deg=start[3], resolution=args.resolution,
                                                max_steps=max(1200, args.max_decisions*40+1), scene_root=SCENE,
                                                scene_factory=partial(RigScene, rgb_exposure_ev=args.rgb_exposure_ev),
                                                known_space=KnownSpace(), render_thermal=True))
        result.update(scene=flight.sim.scene.metadata(), collision_geometry=flight.sim.obstacles.metadata())
        poses = stack.enter_context((args.output/"frames.jsonl").open("x", buffering=1))
        decisions = stack.enter_context((args.output/"decisions.jsonl").open("x", buffering=1))
        trajectory = csv.writer(stack.enter_context((args.output/"trajectory.csv").open("x", newline="")))
        waypoints = csv.writer(stack.enter_context((args.output/"waypoints.csv").open("x", newline="")))
        trajectory.writerow(CSV_FIELDS)
        waypoints.writerow(("decision", "observed_frame", "sim_time", "action", "x", "y", "z", "yaw_deg"))
        h, w = flight.rgb.shape[:2]
        ir_h, ir_w = flight.sim.scene.ir.shape[:2]
        if ir_h != h:
            raise ValueError("RGB/thermal video requires matched output heights")
        video = FpvVideoWriter(args.output/"rgb_thermal_fpv.mp4", w+ir_w, h)
        result.update(rgb_resolution=[w, h], thermal_resolution=[ir_w, ir_h])
        stack.callback(video.close)
        decision = 0
        min_clearance = float("inf")

        def record(receipt, rgb):
            nonlocal min_clearance
            rig = flight.sim.scene
            fields = ("frame_id", "sim_time", "state", "raw_action", "action_format", "applied_control",
                      "substeps_executed", "min_sampled_safety_clearance", "collision", "terminated",
                      "truncated", "reason")
            json_line(poses, {**{key: receipt[key] for key in fields}, "rgb_w2c_native": rig.W.tolist(),
                              "thermal_w2c_native": rig.Wir.tolist()})
            trajectory.writerow(trajectory_row(receipt, video.frames, decision))
            ir = np.rint(255*np.clip(rig.ir, 0, 1)).astype(np.uint8)
            video.write(np.concatenate((rgb, ir), axis=1))
            clearance = receipt["min_sampled_safety_clearance"]
            if clearance is not None:
                min_clearance = min(min_clearance, clearance)

        record(flight.receipt, flight.rgb)
        initial_rgb = flight.rgb.copy()
        if args.save_policy_inputs:
            (args.output/"policy_rgb").mkdir()
            thermal = np.rint(255*np.clip(flight.sim.scene.ir, 0, 1)).astype(np.uint8)
            Image.fromarray(thermal).save(args.output/"initial_thermal.png")
        first_prediction = None
        outcome = flight.outcome
        for decision in range(1, args.max_decisions+1):
            if outcome:
                break
            receipt = flight.receipt
            observation = flight.rgb
            input_record = {}
            if args.save_policy_inputs:
                name = f"policy_rgb/frame_{receipt['frame_id']:06d}.png"
                Image.fromarray(observation).save(args.output/name)
                input_record = {"policy_rgb_file": name,
                                "policy_rgb_sha256": hashlib.sha256(observation.tobytes()).hexdigest()}
            prediction = query_worker(proc, task["instruction"], observation)
            if decision == 1:
                first_prediction = prediction
            if args.room_head:
                motion = prediction["waypoint"]
                action = "stop" if prediction["stop"] else "waypoint"
            else:
                action_id = decode_prediction(prediction)
                action = "invalid" if action_id is None else ACTION_NAMES[action_id]
                motion = None if action_id is None else INDOOR_ACTIONS[action_id]
            result["decisions"] = decision
            target = None
            if motion is not None:
                delta = [0., 0., 0., 0.] if action == "stop" else motion
                target = waypoint_trajectory(receipt["state"], delta)[-1, 1:].tolist()
            json_line(decisions, {"decision": decision, "observed_frame": receipt["frame_id"],
                                  "sim_time": receipt["sim_time"], "action": action, "world_target": target,
                                  **input_record, **prediction})
            waypoints.writerow([decision, receipt["frame_id"], receipt["sim_time"], action, *(target or [""]*4)])
            if args.room_head:
                outcome = flight.execute_waypoint(motion, stop=prediction["stop"], on_tick=record)
            else:
                outcome = "invalid_model_action" if motion is None else flight.execute(action_id, on_tick=record)
            print(f"Decision {decision}: {action} {flight.receipt['state']['position']} {outcome or ''}", flush=True)

        outcome = outcome or "decision_limit"
        result.update(outcome=outcome, frames=video.frames, path_m=flight.path_m,
                      sim_time_s=flight.receipt["sim_time"], final_state=flight.receipt["state"],
                      collision=flight.receipt["collision"].get("event"),
                      min_sampled_safety_clearance_m=min_clearance if np.isfinite(min_clearance) else None,
                      evaluation=evaluate_stop(flight.receipt["state"], outcome,
                                               end[:3] if end else None, end[3] if end else None))
        if args.compare_instruction and first_prediction is not None:
            json_line(proc.stdin, {"reset": True})
            proc.stdin.flush()
            if not worker_reply(proc).get("reset"):
                raise RuntimeError("Could not reset model history before instruction comparison")
            alternative = query_worker(proc, args.compare_instruction, initial_rgb)
            result["first_observation_comparison"] = {
                "frame_id": 0, "same_weights_and_rgb": True, "both_histories_initial": True,
                "flight_instruction": task["instruction"], "flight_prediction": first_prediction,
                "comparison_instruction": args.compare_instruction, "comparison_prediction": alternative,
                "comparison_executed": False}
    result["video_check"] = verify_video(args.output/"rgb_thermal_fpv.mp4", result["frames"])
    return result


def main():
    args = arguments()
    task = json.loads(args.task.read_text())
    if args.instruction is not None:
        task.update(instruction=args.instruction, goal_xyz_yaw=args.goal)
    elif args.goal is not None:
        task.update(goal_xyz_yaw=args.goal)
    args.output.mkdir(parents=True, exist_ok=False)
    try:
        device = resolve_device(args)
        result = fit(args, task, device) if args.mode == "fit" else fly(args, task, device)
    except Exception as exc:
        (args.output/"error.json").write_text(json.dumps({"error": f"{type(exc).__name__}: {exc}"})+"\n")
        raise
    (args.output/"summary.json").write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False)+"\n")
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

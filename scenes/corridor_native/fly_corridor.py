"""Record a waypoint-controlled RGB/thermal flight in the calibrated corridor."""
import argparse
import csv
import hashlib
import json
from contextlib import ExitStack
from pathlib import Path
import sys

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation

SCENE = Path(__file__).resolve().parent
PROJECT = SCENE.parents[1]
sys.path.insert(0, str(PROJECT))
from dynamics import CONTROL_DT
from engine import PolicyFlightSimulator
from openfly_rgb.flight import controller
from record_sequence import CSV_FIELDS, FpvVideoWriter, check_output_directory, trajectory_row, verify_video
from scenes.corridor_native.room import KnownSpace, RigScene


def save_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + "\n")


def fly(route_path, output, resolution, preview_dir=None):
    route = json.loads(route_path.read_text())
    expect_contact = route["kind"] == "collision_test"
    knots = np.asarray(route["knots"], dtype=float)
    if (knots.ndim != 2 or knots.shape[1] != 5 or len(knots) < 2 or not np.isfinite(knots).all()
            or knots[0, 0] != 0 or np.any(np.diff(knots[:, 0]) <= 0)):
        raise ValueError("Expected increasing time, X/Y/Z metres and unwrapped yaw in degrees")
    check_output_directory(output)
    output.mkdir(parents=True, exist_ok=True)
    if preview_dir:
        preview_dir.mkdir(parents=True, exist_ok=True)
    save_json(output / "route.json", route)
    with (output / "reference_waypoints.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["sim_time_s", "x_m", "y_m", "z_m", "yaw_deg"])
        writer.writerows(knots)
    ticks = int(np.ceil(knots[-1, 0] / CONTROL_DT))
    geometry_hashes = {name: hashlib.sha256((SCENE / "collision" / name).read_bytes()).hexdigest()
                       for name in ("free_volume.npz", "rgb_surface_m.ply")}
    known = KnownSpace()
    with PolicyFlightSimulator(SCENE / "models/rgb.ply", point_cloud=SCENE / "collision/rgb_surface_m.ply",
                               scene_factory=RigScene, config="nominal", seed=65, resolution=resolution,
                               spawn=knots[0, 1:4], yaw=knots[0, 4]) as sim, ExitStack() as stack:
        sim.limits["max_steps"] = max(sim.limits["max_steps"], ticks + 1)
        states = stack.enter_context((output / "frames.jsonl").open("w"))
        pose_csv = csv.writer(stack.enter_context((output / "trajectory.csv").open("w", newline="")))
        pose_csv.writerow(CSV_FIELDS)
        wp = csv.writer(stack.enter_context((output / "waypoints.csv").open("w", newline="")))
        wp.writerow(["waypoint_id", "sim_time_s", "frame_id", "x_m", "y_m", "z_m", "qw", "qx", "qy", "qz",
                     "vx_mps", "vy_mps", "vz_mps", "yaw_deg"])
        rgb_height, rgb_width = sim.last_rgb.shape[:2]
        ir_height, ir_width = sim.scene.ir.shape[:2]
        if ir_height != rgb_height:
            raise ValueError("The calibrated RGB and thermal panels must have the same display height")
        name = "rgb_thermal_fpv.mp4"
        video = FpvVideoWriter(output / name, rgb_width + ir_width, rgb_height, fps=1 / CONTROL_DT)
        stack.callback(video.close)
        receipt = sim.last_receipt
        first = np.asarray(receipt["state"]["position"])
        last = first
        distance, min_clearance, max_speed = 0., float("inf"), 0.
        n_unknown, substeps, n_waypoints, segment = 0, 0, 0, 0
        pre_step_speed = 0.
        for index in range(ticks + 1):
            if index:
                pre_step_speed = receipt["state"]["speed"]
                action, segment = controller(sim, receipt, knots, (index - .5) * CONTROL_DT)
                receipt = sim.step(action, format="body_rates_thrust")
            if receipt["frame_id"] != index:
                raise RuntimeError(f"Missing frame at {index}")
            state = receipt["state"]
            pos = np.asarray(state["position"])
            distance += np.linalg.norm(pos - last)
            last = pos
            current_quat = np.asarray(state["quaternion_wxyz"])
            rotation = Rotation.from_quat(current_quat[[1, 2, 3, 0]]).as_matrix()
            body_in_known_space = known.contains_box(pos, rotation, sim.obstacles.safety_half_size)
            n_unknown += int(not body_in_known_space)
            min_clearance = min(min_clearance, receipt["min_sampled_safety_clearance"])
            max_speed = max(max_speed, state["speed"])
            substeps += receipt["substeps_executed"]
            pose_csv.writerow(trajectory_row(receipt, index, segment))
            event = receipt["collision"]["event"]
            record = dict(frame_id=index, sim_time=receipt["sim_time"], dt_actual=receipt["dt_actual"],
                          state=state, segment=segment, raw_action=receipt["raw_action"],
                          applied_control=receipt["applied_control"], action_format=receipt["action_format"],
                          safety_clearance_m=receipt["collision"]["safety_clearance"],
                          body_clearance_m=receipt["collision"]["clearance"],
                          min_sampled_safety_clearance_m=receipt["min_sampled_safety_clearance"],
                          observed_body=bool(body_in_known_space), collision=bool(event), collision_event=event,
                          terminated=receipt["terminated"], truncated=receipt["truncated"], reason=receipt["reason"],
                          substeps=receipt["substeps_executed"], rgb_world_to_camera_native=sim.scene.W.tolist(),
                          thermal_world_to_camera_native=sim.scene.Wir.tolist())
            states.write(json.dumps(record, allow_nan=False) + "\n")
            video.write(np.concatenate([sim.last_rgb, sim.scene.ir], axis=1))
            finished = receipt["terminated"] or receipt["truncated"] or index == ticks
            if index % 20 == 0 or finished:
                wp.writerow([n_waypoints, receipt["sim_time"], index, *pos, *current_quat, *state["velocity"],
                             state["euler_deg"][2]])
                n_waypoints += 1
            if preview_dir and (index % 40 == 0 or finished):
                Image.fromarray(np.rint(sim.last_rgb * 255).astype("uint8")).save(preview_dir / f"rgb_{index:05}.jpg")
            if index % 100 == 0:
                print(f"Frame {index}/{ticks}, position {pos.round(2)}, path {distance:.2f} m", flush=True)
                states.flush()
            if finished:
                break
        end_error = np.linalg.norm(pos - knots[-1, 1:4])
        passed = (receipt["reason"] == "collision" and event is not None and index > 0 and state["speed"] == 0
                  if expect_contact else
                  receipt["reason"] is None and index == ticks and end_error <= .12 and state["speed"] <= .08
                  and not n_unknown)
        summary = dict(kind=route["kind"], passed=bool(passed), instruction=route["instruction"], frames=index + 1,
                       sim_seconds=receipt["sim_time"], video_fps=1 / CONTROL_DT, video_layout="left RGB / right thermal",
                       rgb_resolution=[rgb_width, rgb_height], thermal_resolution=[ir_width, ir_height],
                       frame_convention=receipt["frames"], substeps_executed=substeps, path_m=distance,
                       net_displacement_m=float(np.linalg.norm(pos - first)), longitudinal_displacement_m=pos[0] - first[0],
                       start_position_m=first.tolist(), end_position_m=pos.tolist(), endpoint_error_m=end_error,
                       max_speed_mps=max_speed, final_speed_mps=state["speed"], min_sampled_safety_clearance_m=min_clearance,
                       outside_observed_body_frames=n_unknown, waypoints_recorded=n_waypoints,
                       reference_knots=len(knots), termination_reason=receipt["reason"] or "route_complete",
                       collision_events=int(event is not None), first_contact=event, collision_geometry=sim.obstacles.metadata(),
                       first_contact_frame=index if event is not None else None,
                       preceding_frame_speed_mps=pre_step_speed if event is not None else None,
                       collision_asset_sha256=geometry_hashes,
                       renderer=sim.scene.metadata(), controller="quintic reference, body-rate/thrust state feedback")
    summary["video_check"] = verify_video(output / name, summary["frames"])
    save_json(output / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    if not passed:
        raise RuntimeError(f"Flight did not satisfy {route['kind']} checks; see {output / 'summary.json'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("route", type=Path, help="Corridor-local JSON with kind, instruction and world-frame knots")
    parser.add_argument("--output", type=Path, required=True, help="New or empty run directory")
    parser.add_argument("--resolution", type=float, default=.5, help="Scale of the calibrated 640 pixel RGB camera")
    parser.add_argument("--preview-dir", type=Path, help="Optional RGB snapshots for inspecting language landmarks")
    args = parser.parse_args()
    fly(args.route, args.output, args.resolution, args.preview_dir)


if __name__ == "__main__":
    main()

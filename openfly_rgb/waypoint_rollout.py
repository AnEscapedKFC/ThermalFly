"""Roll out the shared RGB waypoint head or a privileged reference through one executor."""
import argparse
from contextlib import ExitStack
import json
import math
from pathlib import Path

import numpy as np
from scipy.interpolate import CubicHermiteSpline
from scipy.spatial.transform import Rotation

from dynamics import CONTROL_DT
from openfly_rgb.flight import KnownSpace, RigScene, RoomFlight, waypoint_trajectory
from openfly_rgb.navigate import CACHE, PROJECT, open_worker, query_worker, worker_reply
from record_sequence import FpvVideoWriter, verify_video
from scenes.corridor_native.room import KnownSpace as CorridorSpace, RigScene as CorridorRig

DEFAULT_DATASET = PROJECT / "outputs/expert_rgbt_20261123"
ROOM_SCENE = "room65_selected_anchored"
CORRIDOR_SCENE = "corridor_native"
HOVER_SECONDS = 2.5


def read_json(path):
    return json.loads(path.read_text())


def put_line(file, record):
    file.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")


def pitch_rig(base, initial_pitch):
    """Apply the current actuator angle to the same camera mount as the collector."""
    class ActuatedRig(base):
        pitch = initial_pitch

        def render(self, camera):
            mount = np.asarray(camera).copy()
            mount[:3, :3] = mount[:3, :3] @ Rotation.from_euler("x", self.pitch, degrees=True).as_matrix()
            return super().render(mount)
    return ActuatedRig


def limit_size(vector, maximum):
    norm = np.linalg.norm(vector)
    return vector * (maximum / norm) if norm > maximum else vector


def bounded_waypoint(values):
    waypoint = np.asarray(values, dtype=float).copy()
    if waypoint.shape != (4,) or not np.isfinite(waypoint).all():
        raise ValueError("Expected four finite FLU waypoint components")
    waypoint[:2] = limit_size(waypoint[:2], .70)
    waypoint[2:] = np.clip(waypoint[2:], [-.30, -35.], [.30, 35.])
    return waypoint


def continuous_segment(state, chunk, threshold):
    """Interpolate one segment from measured velocity to a neighbouring-segment tangent."""
    first, following = chunk[:2]
    wp = bounded_waypoint(first["delta_flu_m_yaw_deg"])
    ticks = int(np.clip(round(first["duration_s"] / CONTROL_DT), 1, 40))
    duration = ticks * CONTROL_DT
    knots = waypoint_trajectory(state, wp, duration)
    displacement = knots[1, 1:4] - knots[0, 1:4]
    end_velocity = np.zeros(4)
    if following["stop_probability"] < threshold:
        next_wp = bounded_waypoint(following["delta_flu_m_yaw_deg"])
        next_state = {"position": knots[1, 1:4], "euler_deg": [0., 0., knots[1, 4]]}
        next_duration = float(np.clip(following["duration_s"], CONTROL_DT, 2.))
        next_knots = waypoint_trajectory(next_state, next_wp, next_duration)
        outgoing = next_knots[1, 1:4] - knots[1, 1:4]
        incoming_length, outgoing_length = np.linalg.norm(displacement), np.linalg.norm(outgoing)
        if incoming_length > 1e-4 and outgoing_length > 1e-4:
            tangent = displacement / incoming_length + outgoing / outgoing_length
            size = np.linalg.norm(tangent)
            if size > 1e-4:
                end_velocity[:3] = tangent / size * max(0., first["target_reference_speed_mps"])
        rates = np.array([wp[3] / duration, next_wp[3] / next_duration])
        if np.prod(rates) > 0.:
            end_velocity[3] = 2 / np.sum(1. / rates)

    roll, pitch = np.deg2rad(state["euler_deg"][:2])
    _, q, r = state["angular_velocity"]
    heading_rate = np.rad2deg((q * np.sin(roll) + r * np.cos(roll)) / max(.1, np.cos(pitch)))
    start_velocity = np.r_[state["velocity"], heading_rate]
    spline = CubicHermiteSpline(knots[:, 0], knots[:, 1:], [start_velocity, end_velocity])

    def sample(_knots, t):
        pose, velocity, acceleration = spline(t), spline(t, 1), spline(t, 2)
        # Very short end segments can contain centimetres of accumulated tracking error.
        # Bound feed-forward demand while position feedback still tracks the submitted target.
        acceleration[:3] = limit_size(acceleration[:3], 3.)
        velocity[3] = np.clip(velocity[3], -50., 50.)
        return pose, velocity, acceleration, 0

    return knots, sample


class ContinuousFlight(RoomFlight):
    def execute_chunk(self, prediction, movable_camera, on_tick):
        chunk, threshold = prediction["chunk"], prediction["stop_threshold"]
        if chunk[0]["stop_probability"] >= threshold:
            hold = waypoint_trajectory(self.receipt["state"], [0., 0., 0., 0.], HOVER_SECONDS)
            return self._track(hold, True, on_tick)
        knots, sample = continuous_segment(self.receipt["state"], chunk, threshold)
        initial_pitch = float(self.sim.scene.pitch)
        target_pitch = float(np.clip(chunk[0]["target_rig_pitch_deg"], -82., 70.)) if movable_camera else initial_pitch

        def update_pitch(t):
            fraction = min(1., t / knots[-1, 0])
            self.sim.scene.pitch = initial_pitch + (target_pitch - initial_pitch) * fraction

        return self._track(knots, False, on_tick, sample=sample, before_tick=update_pitch)


class ReferenceOracle:
    """Re-express the next reference goal relative to the current simulated pose."""
    def __init__(self, episode_dir):
        from scripts.collect_expert_episodes import Reference, local_delta
        self.ref = Reference(read_json(episode_dir / "route.json"))
        self.decisions = [json.loads(row) for row in (episode_dir / "decisions.jsonl").read_text().splitlines()]
        self.local_delta = local_delta

    def predict(self, decision_id, state):
        source = self.decisions[decision_id]["teacher"]["chunk"]
        chunk = []
        for j, slot in enumerate(source):
            if slot["kind"] != "move":
                chunk.append({"stop_probability": 1.})
                continue
            delta = slot["delta_flu_m_yaw_deg"]
            if j == 0:
                pose = np.r_[state["position"], state["euler_deg"][2]]
                delta = self.local_delta(pose, self.ref.poses[slot["target_frame_id"]]).tolist()
            chunk.append({"stop_probability": 0., "delta_flu_m_yaw_deg": delta,
                          "duration_s": slot["duration_s"], "target_reference_speed_mps": slot["target_reference_speed_mps"],
                          "target_rig_pitch_deg": slot.get("target_rig_pitch_deg", 0.)})
        return {"chunk": chunk, "stop_threshold": .5}


def terminal_score(task, state, outcome):
    distance = math.dist(task["goal_w_m"], state["position"])
    yaw_error = abs((task["goal_yaw_deg"] - state["euler_deg"][2] + 180.) % 360. - 180.)
    reached = distance <= task["position_tol_m"] and yaw_error <= task["yaw_tol_deg"] and state["speed"] <= task["speed_tol_mps"]
    return {"position_error_m": distance, "yaw_error_deg": yaw_error, "final_speed_mps": state["speed"],
            "goal_within_tolerance": reached, "terminal_success": reached and outcome == "policy_stop"}


def run_episode(args, episode_dir, out, policy, worker):
    from openfly_rgb.waypoint_data import instruction_for, state_vector

    episode = read_json(episode_dir / "episode.json")
    spawn_dir = args.dataset / "episodes" / args.spawn_from if args.spawn_from else episode_dir
    spawn_episode = read_json(spawn_dir / "episode.json")
    scene_id = episode["scene_version"]
    if spawn_episode["scene_version"] != scene_id or scene_id not in (ROOM_SCENE, CORRIDOR_SCENE):
        raise ValueError("Task and initial conditions must belong to the same supported scene")
    with (spawn_dir / "frames.jsonl").open() as rows:
        initial = json.loads(next(rows))
    yaw = Rotation.from_quat(np.array(initial["q_w_from_b_wxyz"])[[1, 2, 3, 0]]).as_euler("xyz", degrees=True)[2]
    movable = scene_id == ROOM_SCENE
    rig, known = (RigScene, KnownSpace()) if movable else (CorridorRig, CorridorSpace())
    factory = pitch_rig(rig, initial["rig_pitch_deg"])
    instruction = instruction_for(episode)
    oracle = ReferenceOracle(episode_dir) if args.oracle else None
    if worker is not None:
        worker.stdin.write('{"reset":true}\n')
        worker.stdin.flush()
        if worker_reply(worker) != {"reset": True}:
            raise RuntimeError("OpenFly history reset failed")

    out.mkdir(parents=True, exist_ok=True)
    decisions_executed = 0
    min_clearance = math.inf
    with ExitStack() as stack:
        flight = stack.enter_context(ContinuousFlight(
            initial["position_w_m"], yaw, seed=spawn_episode["seed"], resolution=.5,
            max_steps=math.ceil(args.max_seconds / CONTROL_DT), scene_root=PROJECT / "scenes" / scene_id,
            known_space=known, scene_factory=factory, spawn_velocity_w_mps=initial["velocity_w_mps"]))
        decisions = stack.enter_context((out / "decisions.jsonl").open("x"))
        frames = stack.enter_context((out / "frames.jsonl").open("x"))
        scene = flight.sim.scene
        video = FpvVideoWriter(out / "rgb_fpv.mp4", scene.width, scene.height, fps=1 / CONTROL_DT)
        stack.callback(video.close)

        def record(receipt, rgb):
            nonlocal min_clearance
            state = receipt["state"]
            clearance = receipt["min_sampled_safety_clearance"]
            if clearance is not None:
                min_clearance = min(min_clearance, clearance)
            put_line(frames, {"frame_id": receipt["frame_id"], "t_s": receipt["sim_time"],
                     "position_w_m": state["position"], "q_w_from_b_wxyz": state["quaternion_wxyz"],
                     "velocity_w_mps": state["velocity"], "omega_b_radps": state["angular_velocity"],
                     "rig_pitch_deg": scene.pitch, "rgb_w2c_native": scene.W.tolist(),
                     "collision": receipt["collision"]["in_collision"], "min_safety_clearance_m": clearance,
                     "outcome": flight.outcome})
            video.write(rgb)

        record(flight.receipt, flight.rgb)
        for decision_id in range(args.max_decisions):
            if flight.outcome is not None:
                break
            observed = flight.receipt
            state = observed["state"]
            if oracle is not None:
                prediction = oracle.predict(decision_id, state)
            else:
                z = query_worker(worker, instruction, flight.rgb)["feature"]
                body = state_vector(state["velocity"], state["quaternion_wxyz"], state["angular_velocity"], scene.pitch)
                prediction = policy.predict(z, body)
            start = observed["frame_id"]
            flight.execute_chunk(prediction, movable, record)
            decisions_executed += 1
            put_line(decisions, {"decision_id": decision_id, "observed_frame_id": start,
                     "end_frame_id": flight.receipt["frame_id"], "start_t_s": observed["sim_time"],
                     "end_t_s": flight.receipt["sim_time"], **prediction, "outcome": flight.outcome})
            decisions.flush()
        final = flight.receipt
        outcome = flight.outcome or "max_decisions"
        score = terminal_score(episode["task_spec"], final["state"], outcome)
        summary = {"episode_id": episode["episode_id"], "spawn_from": spawn_episode["episode_id"],
                   "scene_version": scene_id, "instruction": instruction, "policy": "reference_oracle" if oracle else "shared_rgb_head",
                   "head": str(args.head) if args.head else None, "termination_reason": outcome, "terminal": score,
                   "feature_device": None if oracle else args.device, "feature_precision": None if oracle else args.precision,
                   "frames": final["frame_id"] + 1, "decisions": decisions_executed, "duration_s": final["sim_time"],
                   "path_m": flight.path_m, "min_safety_clearance_m": min_clearance if math.isfinite(min_clearance) else None,
                   "initial_velocity_w_mps": initial["velocity_w_mps"], "final_position_w_m": final["state"]["position"],
                   "max_decisions": args.max_decisions, "max_seconds": args.max_seconds,
                   "commanded_hover_s": HOVER_SECONDS}
    summary["video"] = verify_video(out / "rgb_fpv.mp4", summary["frames"])
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, allow_nan=False, indent=2) + "\n")
    print(f'{episode["episode_id"]} from {spawn_episode["episode_id"]}: {outcome}, endpoint error {score["position_error_m"]:.3f} m, {decisions_executed} decisions', flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    p.add_argument("--episode", nargs="+", required=True, help="One or more task episode IDs")
    p.add_argument("--spawn-from", help="Optional alternate same-scene episode providing only the initial state")
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--oracle", action="store_true")
    mode.add_argument("--head", type=Path, help="Shared RGB PyTorch checkpoint")
    p.add_argument("--output", type=Path, required=True, help="New directory")
    p.add_argument("--max-decisions", type=int, default=90)
    p.add_argument("--max-seconds", type=float, default=120.)
    p.add_argument("--model-python", type=Path, default=CACHE / "venv/bin/python")
    p.add_argument("--checkpoint", type=Path, default=CACHE / "openfly-agent-7b")
    p.add_argument("--openfly-root", type=Path, default=CACHE / "OpenFly-Platform")
    p.add_argument("--device", default="cuda:0", choices=["cuda:0", "cpu"])
    p.add_argument("--precision", default="int4", choices=["int4", "bf16", "fp16", "fp32"])
    args = p.parse_args()
    if args.max_decisions < 1 or not math.isfinite(args.max_seconds) or args.max_seconds < CONTROL_DT:
        p.error("provide a positive decision count and at least one control tick")
    if args.oracle and args.spawn_from:
        p.error("--spawn-from is for learned policies; oracle uses its recorded initial condition")
    if len(set(args.episode)) < len(args.episode):
        p.error("episode IDs must be unique within a run")
    args.unnorm_key = "vlnv1"
    args.output.mkdir(parents=True, exist_ok=False)
    with ExitStack() as stack:
        worker = policy = None
        if args.head:
            from openfly_rgb.waypoint_head import load_policy
            from openfly_rgb.waypoint_data import extractor_id
            policy = load_policy(args.head)
            config = policy.feature_config
            fingerprint_args = argparse.Namespace(**vars(args))
            fingerprint_args.device = config.get("device", args.device)
            if config.get("precision") != args.precision or config.get("extractor_sha256") != extractor_id(fingerprint_args):
                raise ValueError("OpenFly source, weights, worker or precision differs from the training cache")
            log = stack.enter_context((args.output / "model.log").open("w"))
            worker = stack.enter_context(open_worker(args, args.device, log, extract_features=True))
        for name in args.episode:
            out = args.output if len(args.episode) == 1 else args.output / name
            run_episode(args, args.dataset / "episodes" / name, out, policy, worker)


if __name__ == "__main__":
    main()

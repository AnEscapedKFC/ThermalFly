"""Compare same-instruction flights at the first actual crossing of a shared physical gate."""
import argparse
from itertools import combinations
import json
from pathlib import Path

import numpy as np

from collect_expert_episodes import Reference
from verify_expert_episodes import check_decisions, read_json, read_lines, require, yaw_deg


def state(frame):
    velocity = frame["velocity_w_mps"]
    return {
        "frame_id": frame["frame_id"], "t_s": frame["t_s"],
        "position_w_m": frame["position_w_m"], "yaw_deg": yaw_deg(frame["q_w_from_b_wxyz"]),
        "velocity_w_mps": velocity, "speed_mps": float(np.linalg.norm(velocity)),
        "omega_b_radps": frame["omega_b_radps"],
        "rgb_path": frame["rgb_path"], "thermal_path": frame["thermal_path"],
    }


def entry_frame(positions, gate, landmarks):
    if gate["kind"] == "landmark_range":
        center = np.asarray(landmarks[gate["landmark"]]["center_w_m"])
        require(gate["radius_m"] > 0, "Landmark radius must be positive")
        distance = np.linalg.norm(positions - center, axis=1) - gate["radius_m"]
    elif gate["kind"] == "world_x":
        require(gate["direction"] in (-1, 1), "Direction must be +1 or -1")
        distance = (gate["x_m"] - positions[:, 0]) * gate["direction"]
    else:
        raise ValueError(f"Unknown gate {gate}")
    require(distance[0] > 0, "Start pose must be on the approaching side of the gate")
    crossed = np.flatnonzero((distance[:-1] > 0) & (distance[1:] <= 0))
    require(crossed.size > 0, f"Gate {gate} was never crossed")
    i = int(crossed[0] + 1)
    return i, distance[i-1:i+1].tolist()


def verify_targets(route, frames, decisions):
    reference = Reference(route)
    max_position = 0.
    max_angle = 0.
    for decision in decisions:
        frame = frames[decision["observed_frame_id"]]
        xyz = np.asarray(frame["position_w_m"])
        yaw = yaw_deg(frame["q_w_from_b_wxyz"])
        begin_time = frame["t_s"]
        for slot in decision["teacher"]["chunk"]:
            if slot["kind"] != "move":
                break
            forward, left, up, turn = slot["delta_flu_m_yaw_deg"]
            c, s = np.cos(np.deg2rad(yaw)), np.sin(np.deg2rad(yaw))
            xyz = xyz + [c*forward - s*left, s*forward + c*left, up]
            yaw += turn
            idx = slot["target_frame_id"]
            require(0 <= idx < len(frames), f"Unknown target frame {idx}")
            target_time = frames[idx]["t_s"]
            require(abs(target_time - begin_time - slot["duration_s"]) < 1.e-6,
                    f"Reference timing mismatch at frame {idx}")
            target = reference.poses[idx]
            max_position = max(max_position, float(np.linalg.norm(target[:3] - xyz)))
            max_angle = max(max_angle, abs((yaw - target[3] + 180.) % 360. - 180.))
            begin_time = target_time
    require(max_position < 1.e-6 and max_angle < 1.e-5,
            f"World-frame target mismatch: {max_position} m / {max_angle} deg")
    return {"reconstruction_error_m": max_position, "yaw_reconstruction_error_deg": max_angle}


def inspect_episode(folder, meta, gate, landmarks):
    route = read_json(folder / "route.json")
    frames = read_lines(folder / "frames.jsonl")
    decisions = read_lines(folder / "decisions.jsonl")
    require(bool(frames) and all(f["frame_id"] == n for n, f in enumerate(frames)), "Missing frame")
    require(all(not f["collision"] and f["in_known_space"] for f in frames), "Unsafe frame")
    require(meta["task_success"] and meta["termination_reason"] == "expert_stop", "Unfinished expert flight")
    check_decisions(decisions, {f["frame_id"]: f for f in frames}, meta)
    checks = verify_targets(route, frames, decisions)
    positions = np.array([f["position_w_m"] for f in frames])
    velocities = np.array([f["velocity_w_mps"] for f in frames])
    require(np.allclose(positions[0], route["knots"][0][1:4], atol=1.e-5, rtol=0), "Reset position mismatch")
    planned_velocity = route.get("initial_velocity_w_mps", [0., 0., 0.])
    require(np.allclose(velocities[0], planned_velocity, atol=1.e-5, rtol=0), "Reset velocity mismatch")
    if gate["kind"] == "world_x":
        lengths = np.r_[0., np.linalg.norm(np.diff(positions, axis=0), axis=1).cumsum()]
        core = (lengths >= .1 * lengths[-1]) & (lengths <= .9 * lengths[-1])
        require(core.any(), "Empty corridor cruise interval")
        speeds = np.linalg.norm(velocities[core], axis=1)
        forward = velocities[core, 0] * gate["direction"]
        require(float(forward.min()) > .08, "Corridor motion paused or reversed in the central 80% of path")
        checks.update(core_min_speed_mps=float(speeds.min()), core_median_speed_mps=float(np.median(speeds)),
                      core_max_speed_mps=float(speeds.max()), core_min_forward_speed_mps=float(forward.min()))
    idx, distances = entry_frame(positions, gate, landmarks)
    entry = state(frames[idx])
    entry["gate_signed_distances_m"] = distances
    return {"start": state(frames[0]), "entry": entry, "reference_verification": checks}


def difference(a, b):
    def norm_difference(key):
        return float(np.linalg.norm(np.subtract(a[key], b[key])))
    return {"position_distance_m": norm_difference("position_w_m"),
            "velocity_vector_distance_mps": norm_difference("velocity_w_mps"),
            "speed_difference_mps": abs(a["speed_mps"] - b["speed_mps"]),
            "yaw_difference_deg": abs((a["yaw_deg"] - b["yaw_deg"] + 180.) % 360. - 180.)}


def compare_group(group, roots, splits):
    source_id, variants, gate = (group[k] for k in ("source_episode_id", "variant_episode_ids", "entry_gate"))
    ids = [source_id, *variants]
    require(bool(variants) and len(ids) == len(set(ids)), f"Duplicate or empty variants for {source_id}")
    folders = {}
    for eid in ids:
        choices = [r / "episodes" / eid for r in roots if (r / "episodes" / eid / "episode.json").is_file()]
        require(len(choices) == 1, f"Expected one location for {eid}, got {choices}")
        folders[eid] = choices[0]
    source = read_json(folders[source_id] / "episode.json")
    scene = source["scene_version"]
    landmarks = (read_json(roots[0] / "scenes" / scene / "landmarks.json")
                 if gate["kind"] == "landmark_range" else None)
    match_fields = ("instruction", "instruction_zh", "instruction_lang", "route_family_id", "scene_version", "task_id")
    members = {}
    for eid, folder in folders.items():
        meta = read_json(folder / "episode.json")
        require(meta["episode_id"] == eid and all(meta[k] == source[k] for k in match_fields),
                f"{eid}: source task, bilingual text, family or scene differs")
        require(eid in splits and splits[eid] == splits.get(source_id), f"{eid}: split differs from source")
        if eid != source_id:
            require(meta["source_episode_id"] == source_id, f"{eid}: missing source link")
        members[eid] = inspect_episode(folder, meta, gate, landmarks)
        if eid != source_id:
            require(members[eid]["start"]["speed_mps"] >= .02, f"{eid}: actual rolling start missing")
    pairs = []
    for a, b in combinations(ids, 2):
        first = difference(members[a]["start"], members[b]["start"])
        entry = difference(members[a]["entry"], members[b]["entry"])
        require(first["position_distance_m"] >= .25 and first["velocity_vector_distance_mps"] >= .02,
                f"{a}, {b}: starts too similar {first}")
        require(entry["velocity_vector_distance_mps"] >= .08 or entry["yaw_difference_deg"] >= 8.,
                f"{a}, {b}: entry velocities and headings too similar {entry}")
        pairs.append({"episode_ids": [a, b], "start": first, "entry": entry})
    return {**group, **{k: source[k] for k in match_fields}, "split": splits[source_id],
            "members": members, "pairwise": pairs}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path, help="Root containing the original episodes and scene calibration")
    parser.add_argument("spec", type=Path, help="JSON with groups, source and variant IDs and physical entry gates")
    parser.add_argument("--stage", action="append", type=Path, default=[], help="Extra root with unpublished variants")
    parser.add_argument("--splits", type=Path, help="Final planned split file when auditing across stages")
    parser.add_argument("--output", type=Path, help="Write the verification report to this file (default: stdout)")
    args = parser.parse_args()
    groups = read_json(args.spec)["groups"]
    require(bool(groups), "Empty comparison specification")
    split_map = {}
    for split, ids in read_json(args.splits or args.dataset / "splits.json")["splits"].items():
        for eid in ids:
            require(eid not in split_map, f"Duplicate split assignment for {eid}")
            split_map[eid] = split
    results = [compare_group(group, [args.dataset, *args.stage], split_map) for group in groups]
    variant_ids = [eid for result in results for eid in result["variant_episode_ids"]]
    require(len(variant_ids) == len(set(variant_ids)), "A variant is assigned to multiple comparison groups")
    report = {"passed": True, "variant_count": len(variant_ids),
              "policy": {"start_position_min_m": .25, "start_velocity_vector_min_mps": .02,
                         "variant_start_speed_min_mps": .02, "entry_velocity_vector_min_mps_or_yaw_min_deg": [.08, 8.],
                         "corridor_middle_path_min_forward_mps_exclusive": .08,
                         "entry_event_sample": "first actual frame after crossing the group's shared gate"},
              "groups": results}
    contents = json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    if args.output:
        args.output.write_text(contents, encoding="utf-8")
    else:
        print(contents, end="")


if __name__ == "__main__":
    main()

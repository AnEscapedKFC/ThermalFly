"""Score completed RGB rollouts against the task's geometric evidence, offline."""
import argparse
import json
from pathlib import Path

import numpy as np

from scripts.audit_room_expert_semantics import projection


def read_frames(path):
    with path.open() as source:
        return [json.loads(line) for line in source if line.strip()]


def viewing_geometry(frames, positions, center, calibration):
    uv, _, visible = projection(frames, center, calibration, "rgb")
    wh = np.asarray(calibration["wh"])
    central = visible & np.all((uv >= 0.1 * wh) & (uv <= 0.9 * wh), axis=1)
    return np.linalg.norm(positions - center, axis=1), visible, central


def longest_view_sweep(positions, center, mask):
    """Largest uninterrupted sweep about an object while meeting the viewing criteria."""
    indices = np.flatnonzero(mask)
    if not len(indices):
        return 0.0
    sweeps = []
    for segment in np.split(indices, np.flatnonzero(np.diff(indices) != 1) + 1):
        delta = positions[segment, :2] - center[:2]
        angle = np.unwrap(np.arctan2(delta[:, 1], delta[:, 0]))
        sweeps.append(float(np.rad2deg(np.ptp(angle))))
    return max(sweeps)


def stable_hover_seconds(times, positions, speed, stop_index):
    if stop_index is None:
        return 0.
    times = times[stop_index:]
    distance = np.linalg.norm(positions[stop_index:] - positions[stop_index], axis=1)
    outside = np.flatnonzero((speed[stop_index:] > .06) | (distance > .08))
    start = int(outside[-1] + 1) if outside.size else 0
    return float(times[-1] - times[start]) if start < len(times) else 0.


def score(dataset, episode_id, frames, stop_index=None):
    if not frames:
        raise ValueError("Empty rollout")
    episode_dir = dataset / "episodes" / episode_id
    meta = json.loads((episode_dir / "episode.json").read_text())
    route = json.loads((episode_dir / "route.json").read_text())
    pos = np.array([f["position_w_m"] for f in frames])
    times = np.array([f["t_s"] for f in frames])
    speed = np.linalg.norm([f["velocity_w_mps"] for f in frames], axis=1)
    if np.any(np.diff(times) <= 0):
        raise ValueError("Expected increasing simulator time")
    result = dict(episode_id=episode_id, frames=len(frames), duration_s=float(times[-1] - times[0]),
                  path_m=float(np.linalg.norm(np.diff(pos, axis=0), axis=1).sum()),
                  peak_speed_mps=float(speed.max()), final_speed_mps=float(speed[-1]),
                  height_range_m=[float(pos[:, 2].min()), float(pos[:, 2].max())],
                  time_under_0_05_mps_s=float(np.diff(times) @ (speed[1:] < 0.05)),
                  final_continuous_hover_s=stable_hover_seconds(times, pos, speed, stop_index))
    scene = dataset / "scenes" / meta["scene_version"]
    targets = route.get("inspection_landmarks", [])
    if "orbit_center_w_m" in route:
        result["table_sweep_over_flight_deg"] = longest_view_sweep(
            pos, np.array(route["orbit_center_w_m"]), np.ones(len(pos), dtype=bool))
    if not targets:
        return result
    calibration = json.loads((scene / "calibration.json").read_text())
    landmarks = json.loads((scene / "landmarks.json").read_text())
    extra = scene / "object_landmarks.json"
    if extra.exists():
        landmarks.update(json.loads(extra.read_text()))
    expert = read_frames(episode_dir / "frames.jsonl")
    expert_pos = np.array([f["position_w_m"] for f in expert])
    expert_times = np.array([f["t_s"] for f in expert])
    result["inspections"] = []
    next_frame = 0
    for target in targets:
        center = np.array(landmarks[target]["center_w_m"])
        distance, visible, central = viewing_geometry(frames, pos, center, calibration)
        ref_distance, _, ref_central = viewing_geometry(expert, expert_pos, center, calibration)
        # Only the scorer reads the reference. Use its own approach stage to set
        # an object-specific viewing distance, independent of the rollout clock.
        stages = route["subgoals"]
        j = next(i for i, s in enumerate(stages) if s.get("landmark") == target
                 and s["predicate"] == "approach_and_observe")
        end = stages[j + 1]["at_s"] if j + 1 < len(stages) else expert_times[-1]
        ref_mask = ref_central & (expert_times >= stages[j]["at_s"]) & (expert_times <= end)
        limit = float(ref_distance[ref_mask].min() + 0.20) if ref_mask.any() else None
        hits = np.flatnonzero(central & (distance <= limit)) if limit is not None else []
        ordered = next((int(i) for i in hits if i >= next_frame), None)
        next_frame = len(frames) if ordered is None else ordered + 1
        closest = float(distance[central].min()) if central.any() else None
        item = dict(target=target, central_distance_limit_m=limit, closest_central_distance_m=closest,
                    ordered_inspection_t_s=None if ordered is None else float(times[ordered]),
                    in_view_fraction=float(visible.mean()), final_in_view=bool(visible[-1]))
        observation = route.get("object_observation", {})
        if observation.get("landmark") == target:
            ref_arc = (expert_times >= observation["arc_start_s"]) & (expert_times <= observation["arc_end_s"])
            low, high = ref_distance[ref_arc].min() - 0.2, ref_distance[ref_arc].max() + 0.2
            item["sweep_in_view_at_observation_range_deg"] = longest_view_sweep(
                pos, center, visible & (distance >= low) & (distance <= high))
        if route.get("search_evidence", {}).get("target") == target:
            first_view = np.flatnonzero(visible)
            item["initially_out_of_view"] = not bool(visible[0])
            item["first_in_view_s"] = float(times[first_view[0]]) if first_view.size else None
        result["inspections"].append(item)
    result["all_inspections_in_order"] = all(i["ordered_inspection_t_s"] is not None
                                              for i in result["inspections"])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("outputs/expert_rgbt_20261123"))
    parser.add_argument("runs", nargs="+", type=Path, help="Completed waypoint rollout directories")
    args = parser.parse_args()
    for run in args.runs:
        summary = json.loads((run / "summary.json").read_text())
        frames = read_frames(run / "frames.jsonl")
        decisions = read_frames(run / "decisions.jsonl")
        stop_frame = next((d["observed_frame_id"] for d in decisions
                           if d["chunk"][0]["stop_probability"] >= d["stop_threshold"]), None)
        stop_index = next((i for i, f in enumerate(frames) if f["frame_id"] == stop_frame), None)
        report = score(args.dataset, summary["episode_id"], frames, stop_index)
        text = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)
        (run / "geometry.json").write_text(text + "\n")
        print(run, text)


if __name__ == "__main__":
    main()

"""Cache frozen OpenFly RGB features and expert waypoint targets by episode."""

import argparse
from collections import Counter
from contextlib import ExitStack
import fcntl
import hashlib
import json
from pathlib import Path
import tempfile
from zipfile import BadZipFile

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation

from openfly_rgb.navigate import CACHE, PROJECT, open_worker, query_worker, worker_reply

SCHEMA = "thermalfly_rgb_waypoint_cache_v1"
DATASET = PROJECT / "outputs/expert_rgbt_20261123"
OUTPUT = PROJECT / "outputs/thermalfly_rgb_v1/cache"
FIXED_HEIGHT_TEXT = "Fly one complete inward-looking circuit around the conference table at 1.53 metres and hover at the start."
VARIABLE_HEIGHT_TEXT = ("Fly one complete inward-looking circuit around the conference table. Pass the high point "
                        "during the first half and the low point during the second half, then return to the starting "
                        "height and hover at the start.")
TARGET_FIELDS = ["forward_m", "left_m", "up_m", "yaw_deg", "stop", "duration_s", "speed_mps", "rig_pitch_deg"]


def instruction_for(episode):
    text = episode["instruction"]
    if episode["episode_id"] == "room_orbit_vary" and text == FIXED_HEIGHT_TEXT:
        return VARIABLE_HEIGHT_TEXT
    return text


def state_vector(velocity_w_mps, q_w_from_b_wxyz, omega_b_radps, rig_pitch_deg):
    """Body velocity, body angular rate, body down direction and current rig angle in degrees."""
    world_from_body = Rotation.from_quat(np.roll(q_w_from_b_wxyz, -1))
    v_body = world_from_body.apply(velocity_w_mps, inverse=True)
    down_body = world_from_body.apply([0., 0., -1.], inverse=True)
    state = np.asarray([*v_body, *omega_b_radps, *down_body, rig_pitch_deg], dtype=np.float32)
    if state.shape != (10,) or not np.isfinite(state).all():
        raise ValueError("Invalid 10-D body state")
    return state


def targets_for(chunk):
    if len(chunk) != 3:
        raise ValueError("Expected three expert slots")
    targets = np.zeros((3, 8), dtype=np.float32)
    kinds = np.array([("move", "stop", "pad").index(slot["kind"]) for slot in chunk], dtype=np.int8)
    pitch_valid = np.zeros(3, dtype=bool)
    if list(kinds) not in ([0, 0, 0], [0, 0, 1], [0, 1, 2], [1, 2, 2]):
        raise ValueError(f"Invalid move/stop/pad order: {kinds}")
    for j, slot in enumerate(chunk):
        if slot["kind"] == "stop":
            targets[j, 4] = 1.
        if slot["kind"] != "move":
            continue
        targets[j, :4] = slot["delta_flu_m_yaw_deg"]
        targets[j, 5:7] = [slot["duration_s"], slot["target_reference_speed_mps"]]
        if targets[j, 5] <= 0 or targets[j, 6] < 0:
            raise ValueError("Invalid expert duration or speed")
        if "target_rig_pitch_deg" in slot:
            targets[j, 7] = slot["target_rig_pitch_deg"]
            pitch_valid[j] = True
    if not np.isfinite(targets).all():
        raise ValueError("Non-finite expert target")
    return targets, kinds, pitch_valid


def digest_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024**2), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def extractor_id(args):
    checkpoint = args.checkpoint.expanduser().resolve()
    upstream = args.openfly_root.expanduser().resolve() / "train"
    if not checkpoint.is_dir() or not upstream.is_dir():
        raise FileNotFoundError("Local OpenFly checkpoint and source are required")
    model_files = sorted(p for p in checkpoint.iterdir() if p.suffix in {".json", ".model", ".safetensors", ".bin"})
    if not any(p.suffix in {".safetensors", ".bin"} for p in model_files):
        raise FileNotFoundError("No checkpoint weight shards")
    components = {
        "model": {p.name: digest_file(p) for p in model_files},
        "upstream": {p.relative_to(upstream).as_posix(): digest_file(p) for p in sorted(upstream.rglob("*.py"))},
        "worker": digest_file(Path(__file__).with_name("model_worker.py")),
        "format": SCHEMA, "device": args.device, "precision": args.precision, "unnorm_key": args.unnorm_key,
    }
    return canonical_digest(components)


def episode_inputs(folder, instruction, extractor):
    decisions = [json.loads(line) for line in (folder / "decisions.jsonl").read_text().splitlines()]
    frames = {}
    needed = {d["observed_frame_id"] for d in decisions}
    with (folder / "frames.jsonl").open() as stream:
        for line in stream:
            frame = json.loads(line)
            if frame["frame_id"] in needed:
                if frame["frame_id"] in frames:
                    raise ValueError("Duplicate decision frame")
                frames[frame["frame_id"]] = frame
    observed = [d["observed_frame_id"] for d in decisions]
    kinds, targets, states, masks, images, image_digests = [], [], [], [], [], []
    for j, d in enumerate(decisions):
        if d["decision_id"] != j or d["history_frame_ids"] != observed[max(0, j-2):j]:
            raise ValueError(f"Invalid decision/history in {folder.name} at {j}")
        f = frames[observed[j]]
        rgb = folder / f["rgb_path"]
        images.append(rgb)
        image_digests.append(digest_file(rgb))
        states.append(state_vector(f["velocity_w_mps"], f["q_w_from_b_wxyz"], f["omega_b_radps"], f["rig_pitch_deg"]))
        target, kind, mask = targets_for(d["teacher"]["chunk"])
        targets.append(target)
        kinds.append(kind)
        masks.append(mask)
    if not decisions or sorted(set(observed)) != observed or kinds[-1][0] != 1 or any(k[0] != 0 for k in kinds[:-1]):
        raise ValueError(f"Unordered decisions or missing terminal STOP in {folder.name}")
    arrays = dict(state=np.asarray(states), target=np.asarray(targets), kind=np.asarray(kinds),
                  pitch_valid=np.asarray(masks), decision_id=np.arange(len(observed)), observed_frame_id=np.array(observed))
    signature = canonical_digest([extractor, instruction, image_digests,
                                  {k: hashlib.sha256(v.tobytes()).hexdigest() for k, v in arrays.items()}])
    return images, arrays, signature


def reviews_for(dataset, override, ids):
    path = override or dataset / "expert_episodes_human_review.json"
    if not path.exists() and override is None:
        return {}
    review = json.loads(path.read_text())
    if review["dataset"] != dataset.name or set(review["episodes"]) - ids:
        raise ValueError(f"Review export does not belong to {dataset.name}: {path}")
    statuses = {k: entry["status"] for k, entry in review["episodes"].items()}
    if set(statuses.values()) - {"accepted", "rejected", "pending"}:
        raise ValueError(f"Unknown human review status in {path}")
    return statuses


def cached(path, signature, arrays):
    if not path.exists():
        return False
    try:
        with path.open("rb") as stream, np.load(stream, allow_pickle=False) as data:
            features = data["features"]
            return (str(data["signature"]) == signature and features.shape == (len(arrays["state"]), 5632)
                    and features.dtype == np.float32 and np.isfinite(features).all()
                    and all(np.array_equal(data[k], a) for k, a in arrays.items()))
    except (OSError, ValueError, KeyError, EOFError, BadZipFile):
        return False


def write_atomic(path, write):
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as file:
        temp_path = Path(file.name)
    try:
        with temp_path.open("wb") as stream:
            write(stream)
        temp_path.replace(path)
    finally:
        temp_path.unlink(missing_ok=True)


def build_cache(args, worker=None):
    dataset, output = args.dataset.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    split_path = dataset / "splits.json"
    splits = json.loads(split_path.read_text())["splits"]
    ids = [episode for name in ("train", "val", "test") for episode in splits[name]]
    if set(splits) != {"train", "val", "test"} or len(ids) != len(set(ids)) or any(Path(i).name != i for i in ids):
        raise ValueError("Expected disjoint episode IDs and train, val, test splits")
    reviews = reviews_for(dataset, args.review, set(ids))
    with ExitStack() as stack:
        lock = stack.enter_context((output / ".lock").open("a"))
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        print("Fingerprinting checkpoint, worker and source...", flush=True)
        extractor = extractor_id(args)
        entries, rejected, families, sources = [], [], {}, {}
        for split, episodes in splits.items():
            for episode_id in episodes:
                folder = dataset / "episodes" / episode_id
                meta = json.loads((folder / "episode.json").read_text())
                if meta["episode_id"] != episode_id or meta["schema_version"] != "rgbt_episode_v1":
                    raise ValueError(f"Unsupported or inconsistent episode metadata: {folder}")
                status = reviews.get(episode_id, meta["human_review"])
                if status == "rejected":
                    rejected.append(episode_id)
                    continue
                if not meta["task_success"]:
                    raise ValueError(f"Failed episode in expert split: {episode_id}")
                family = meta["route_family_id"]
                if families.setdefault(family, split) != split:
                    raise ValueError(f"Route family crosses dataset splits: {family}")
                source = meta.get("source_episode_id") or episode_id
                if sources.setdefault(source, split) != split:
                    raise ValueError(f"Demonstration and variants cross dataset splits: {source}")
                instruction = instruction_for(meta)
                images, arrays, signature = episode_inputs(folder, instruction, extractor)
                path = output / (episode_id + ".npz")
                if not cached(path, signature, arrays):
                    # A manifest describes a complete snapshot. Resume only publishes it
                    # again once all changed input episodes have been replaced.
                    (output / "manifest.json").unlink(missing_ok=True)
                    if worker is None:
                        log = stack.enter_context((output / "model.log").open("a"))
                        worker = stack.enter_context(open_worker(args, args.device, log, extract_features=True))
                    worker.stdin.write('{"reset":true}\n')
                    worker.stdin.flush()
                    if worker_reply(worker).get("reset") is not True:
                        raise RuntimeError("OpenFly history reset failed")
                    features = []
                    for image_path in images:
                        with Image.open(image_path) as image:
                            reply = query_worker(worker, instruction, np.asarray(image.convert("RGB")))
                        feature = np.asarray(reply["feature"], dtype=np.float32)
                        if feature.shape != (5632,) or not np.isfinite(feature).all():
                            raise ValueError(f"Malformed feature for {image_path}")
                        features.append(feature)
                    write_atomic(path, lambda out: np.savez(out, **arrays, features=np.stack(features), signature=signature))
                entries.append(dict(episode_id=episode_id, split=split, route_family_id=family, decisions=len(images),
                                    signature=signature, instruction=instruction, source_instruction=meta["instruction"],
                                    human_review=status))
                print(f"{len(entries)}/{len(ids)} {split} {episode_id}: {len(images)} decisions", flush=True)
        manifest = dict(schema=SCHEMA, dataset=dataset.name, splits_sha256=digest_file(split_path),
                        extractor_sha256=extractor, device=args.device, precision=args.precision, target_fields=TARGET_FIELDS,
                        episodes=entries, rejected=rejected)
        write_atomic(output / "manifest.json", lambda out: out.write((json.dumps(manifest, indent=2) + "\n").encode()))
        counts = Counter()
        for entry in entries:
            counts[entry["split"]] += entry["decisions"]
        print(f"Complete: {len(entries)} episodes, decisions {dict(counts)}; {output}", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", type=Path, default=DATASET)
    p.add_argument("--output", type=Path, default=OUTPUT)
    p.add_argument("--review", type=Path, help="Human-review export; rejected episodes are excluded")
    p.add_argument("--model-python", type=Path, default=CACHE / "venv/bin/python")
    p.add_argument("--openfly-root", type=Path, default=CACHE / "OpenFly-Platform")
    p.add_argument("--checkpoint", type=Path, default=CACHE / "openfly-agent-7b")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--precision", choices=("int4", "bf16", "fp16", "fp32"), default="int4")
    p.add_argument("--unnorm-key", default="vlnv1")
    build_cache(p.parse_args())


if __name__ == "__main__":
    main()

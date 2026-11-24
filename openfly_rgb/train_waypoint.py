"""Train and evaluate a small shared indoor readout on frozen RGB OpenFly features."""
import argparse
from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from statistics import mean, median

import numpy as np
import torch

from .waypoint_head import FEATURE_WIDTH, FIELDS, FORMAT, SLOTS, STATE_WIDTH
from .waypoint_head import WaypointPolicy, load_policy, physical_slots, waypoint_loss
from .waypoint_data import SCHEMA, TARGET_FIELDS


@dataclass
class Episode:
    name: str
    family: str
    features: torch.Tensor
    state: torch.Tensor
    target: torch.Tensor
    kind: torch.Tensor
    pitch: torch.Tensor


def read_episode(cache, row):
    name = row["episode_id"]
    with np.load(cache / (name + ".npz"), allow_pickle=False) as archive:
        if str(archive["signature"]) != row["signature"] or len(archive["decision_id"]) != row["decisions"]:
            raise ValueError(f"Cache does not match the manifest for {name}")
        arrays = [torch.from_numpy(archive[key]) for key in ("features", "state", "target", "kind", "pitch_valid")]
    features, state, target = (a.float() for a in arrays[:3])
    kind, pitch = arrays[3].long(), arrays[4].bool()
    n = len(features)
    if (not n or n != row["decisions"] or features.shape != (n, FEATURE_WIDTH) or state.shape != (n, STATE_WIDTH)
            or target.shape != (n, SLOTS, FIELDS) or kind.shape != (n, SLOTS) or pitch.shape != kind.shape):
        raise ValueError(f"Invalid cache dimensions for {name}")
    move = kind == 0
    if (not ((0 <= kind) & (kind <= 2)).all() or not torch.isfinite(features).all()
            or not torch.isfinite(state).all() or not torch.isfinite(target[..., :4][move]).all()
            or not torch.isfinite(target[..., 5:7][move]).all() or not torch.isfinite(target[..., 7][move & pitch]).all()):
        raise ValueError(f"Invalid cache values for {name}")
    return Episode(name, row["route_family_id"], features, state, target, kind, pitch)


def nested_order(rows, seed=20261124):
    """Round-robin across shuffled training families before drawing another variant."""
    families = defaultdict(list)
    rng = np.random.default_rng(seed)
    for row in sorted(rows, key=lambda x: x["episode_id"]):
        families[row["route_family_id"]].append(row["episode_id"])
    keys = sorted(families)
    rng.shuffle(keys)
    for episodes in families.values():
        rng.shuffle(episodes)
    return [families[k][i] for i in range(max(map(len, families.values()))) for k in keys if i < len(families[k])]


def encoded_episodes(model, episodes):
    return [(model.encode(e.features, e.state), e.target, e.kind, e.pitch) for e in episodes]


def concat(rows):
    return tuple(torch.cat(items) for items in zip(*rows))


def sampling_weights(episodes):
    counts = Counter(e.family for e in episodes)
    return torch.cat([torch.full((len(e.kind),), 1. / (counts[e.family] * len(e.kind))) for e in episodes])


def train_one(architecture, transform, train, val, weights, args, seed):
    torch.manual_seed(seed)
    model = WaypointPolicy(architecture)
    # Transfer preprocessing only; each seed obtains independent head initialization.
    buffers = dict(transform.named_buffers())
    for name, dest in model.named_buffers():
        dest.copy_(buffers[name])
    opt = torch.optim.AdamW(model.head.parameters(), lr=args.lr, weight_decay=1e-3)
    x, target, kind, pitch = concat(train)
    best_loss = math.inf
    best_state = None
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        indices = torch.multinomial(weights, len(x), replacement=True)
        for idx in indices.split(args.batch):
            loss = waypoint_loss(model(x[idx]), target[idx], kind[idx], pitch[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        model.eval()
        with torch.inference_mode():
            validation = mean(waypoint_loss(model(vx), vy, vk, vp).item() for vx, vy, vk, vp in val)
        if not math.isfinite(validation):
            raise FloatingPointError(f"Nonfinite validation loss at epoch {epoch}")
        history.append(validation)
        if validation < best_loss - 1e-5:
            best_loss = validation
            best_state = deepcopy(model.head.state_dict())
            best_epoch = epoch
        elif epoch - best_epoch >= args.patience:
            break
    model.head.load_state_dict(best_state)
    return model.eval(), {"best_epoch": best_epoch, "epochs_completed": epoch, "val_loss": best_loss,
                          "val_loss_by_epoch": history}


def stop_counts(scores, labels, threshold):
    called = scores >= threshold
    return int((called & labels).sum()), int((called & ~labels).sum()), int((~called & labels).sum())


def calibrate_stop(scores, labels):
    # First-slot F1; ties prefer fewer premature STOPs and a cutoff closest to 0.5.
    best = None
    threshold = .5
    for cut in sorted({.5, 1., *scores.tolist()}):
        tp, fp, fn = stop_counts(scores, labels, cut)
        key = (2 * tp / max(2 * tp + fp + fn, 1), -fp, -abs(cut - .5))
        if best is None or key > best:
            best, threshold = key, cut
    return threshold


def chain_positions(delta):
    """Compose successive yaw-local displacements into the first observation's frame."""
    heading = delta[..., 3].deg2rad()
    yaw = torch.cat((torch.zeros_like(heading[:, :1]), heading[:, :-1].cumsum(dim=1)), dim=1)
    forward, left = delta[..., 0], delta[..., 1]
    return torch.stack((yaw.cos() * forward - yaw.sin() * left,
                        yaw.sin() * forward + yaw.cos() * left, delta[..., 2]), dim=-1).cumsum(dim=1)


def episode_metrics(pred, e, threshold):
    truth = e.target
    move = e.kind[:, 0] == 0
    stop = e.kind[:, 0] == 1
    valid = move | stop
    tp, fp, fn = stop_counts(pred[valid, 0, 4], stop[valid], threshold)
    prefix = (e.kind == 0).int().cumprod(dim=1).bool()
    positions = (chain_positions(pred) - chain_positions(truth)).norm(dim=-1)
    def avg(values, subset):
        return values[subset].mean().item() if subset.any() else None
    dif = (pred[:, 0] - truth[:, 0]).abs()
    return {"decisions": len(move), "move": int(move.sum()), "stop": int(stop.sum()),
            "stop_tp": tp, "early_stop_fp": fp, "missed_stop_fn": fn,
            "stop_precision": tp / max(1, tp + fp), "stop_recall": tp / max(1, tp + fn),
            "position_mae_m": avg(positions[:, 0], move), "yaw_mae_deg": avg(dif[:, 3], move),
            "duration_mae_s": avg(dif[:, 5], move), "speed_mae_mps": avg(dif[:, 6], move),
            "pitch_mae_deg": avg(dif[:, 7], move & e.pitch[:, 0]),
            "two_segment_endpoint_mae_m": avg(positions[:, 1], prefix[:, 1]),
            "three_segment_endpoint_mae_m": avg(positions[:, 2], prefix[:, 2])}


def evaluate(model, episodes, encoded):
    with torch.inference_mode():
        rows = {e.name: episode_metrics(physical_slots(model(x)), e, float(model.stop_threshold))
                for e, (x, *_) in zip(episodes, encoded)}
    count_keys = ("decisions", "move", "stop", "stop_tp", "early_stop_fp", "missed_stop_fn")
    def macro(rows):
        metrics = {}
        for key in rows[0]:
            if key not in count_keys:
                values = [r[key] for r in rows if r[key] is not None]
                metrics[key] = mean(values) if values else None
        return metrics
    families = defaultdict(list)
    for e in episodes:
        families[e.family].append(rows[e.name])
    family_metrics = {f: macro(rr) for f, rr in families.items()}
    return {"counts": {k: sum(r[k] for r in rows.values()) for k in count_keys},
            "episode_macro": macro(list(rows.values())), "family_macro": macro(list(family_metrics.values())),
            "per_family": family_metrics, "per_episode": rows}


def smoke_check(episode):
    """Overfit one complete training episode with dropout disabled and report the fit."""
    torch.manual_seed(5)
    model = WaypointPolicy("mlp").eval()
    model.fit_inputs(episode.features, episode.state)
    x, y, k, p = encoded_episodes(model, [episode])[0]
    opt = torch.optim.AdamW(model.head.parameters(), lr=.006)
    first = waypoint_loss(model(x), y, k, p).item()
    for _ in range(300):
        loss = waypoint_loss(model(x), y, k, p)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    final = waypoint_loss(model(x), y, k, p).item()
    if not math.isfinite(final) or final >= first:
        raise RuntimeError(f"One-episode fit did not reduce the loss ({first} -> {final})")
    return {"episode": episode.name, "decisions": len(k), "initial_loss": first, "final_loss": final,
            "metrics": evaluate(model, [episode], [(x, y, k, p)])["per_episode"][episode.name]}


def write_json(path, obj):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def save_model(path, policy, train_ids, seed, feature_config=None):
    tmp = path.with_suffix(".tmp")
    torch.save({"format": FORMAT, "architecture": policy.architecture, "state_dict": policy.state_dict(),
                "training_episodes": train_ids, "seed": seed, "feature_config": feature_config or {}}, tmp)
    tmp.replace(path)


def run(args):
    torch.set_num_threads(args.threads)
    manifest_bytes = (args.cache / "manifest.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    if manifest.get("schema") != SCHEMA or manifest.get("target_fields") != TARGET_FIELDS:
        raise ValueError("Unsupported waypoint cache schema or output field order")
    feature_config = {key: manifest[key] for key in ("schema", "extractor_sha256", "precision", "device", "target_fields")
                      if key in manifest}
    rows = manifest["episodes"]
    if len({r["episode_id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate episodes in cache manifest")
    splits = {part: [r for r in rows if r["split"] == part] for part in ("train", "val", "test")}
    if not all(splits.values()) or sum(map(len, splits.values())) != len(rows):
        raise ValueError("Expected a nonempty train, val and test partition")
    order = nested_order(splits["train"])
    if args.sizes is None:
        args.sizes = sorted({min(n, len(order)) for n in (8, 16)} | {len(order)})
    if len(order) < max(args.sizes):
        raise ValueError(f"Requested {max(args.sizes)} training episodes but only {len(order)} are available")
    args.output.mkdir(parents=True, exist_ok=True)
    # Reserve the output before writing checkpoints, including across simultaneous launches.
    try:
        config = (args.output / "config.json").open("x")
    except FileExistsError as exc:
        raise FileExistsError(f"Choose a fresh output directory: {args.output}") from exc
    with config:
        json.dump({k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}, config, indent=2)
        config.write("\n")
    all_train = {r["episode_id"]: read_episode(args.cache, r) for r in splits["train"]}
    val = [read_episode(args.cache, r) for r in splits["val"]]
    small = min((e for e in all_train.values() if e.pitch.any()), key=lambda e: (len(e.kind), e.name), default=all_train[order[0]])
    report = {"format": FORMAT, "cache_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
              "nested_order": order, "smoke": smoke_check(small), "runs": [], "groups": []}
    for size in args.sizes:
        ids = order[:size]
        train = [all_train[name] for name in ids]
        transform = WaypointPolicy()
        transform.fit_inputs(torch.cat([e.features for e in train]), torch.cat([e.state for e in train]))
        tr, va = encoded_episodes(transform, train), encoded_episodes(transform, val)
        weights = sampling_weights(train)
        for arch in ("linear", "mlp"):
            records = []
            for seed in args.seeds:
                model, fit = train_one(arch, transform, tr, va, weights, args, seed)
                with torch.inference_mode():
                    scores = torch.cat([model(vx)[:, 0, 4].sigmoid()[e.kind[:, 0] != 2] for e, (vx, *_) in zip(val, va)])
                labels = torch.cat([e.kind[:, 0][e.kind[:, 0] != 2] == 1 for e in val])
                model.stop_threshold.fill_(calibrate_stop(scores, labels))
                tag = f"{size}-{arch}-s{seed}"
                save_model(args.output / f"{tag}.pt", model, ids, seed, feature_config)
                rec = {"tag": tag, "size": size, "architecture": arch, "seed": seed, **fit,
                       "stop_threshold": float(model.stop_threshold), "validation": evaluate(model, val, va)}
                report["runs"].append(rec)
                records.append(rec)
                print(f"{tag} epoch={fit['best_epoch']} validation={fit['val_loss']:.4f}", flush=True)
            losses = [r["val_loss"] for r in records]
            middle = min(records, key=lambda r: abs(r["val_loss"] - median(losses)))
            report["groups"].append({"size": size, "architecture": arch, "mean_val_loss": mean(losses),
                                     "std_val_loss": float(np.std(losses)), "representative": middle["tag"]})
            write_json(args.output / "progress.json", report)
    # Choose architecture and sample size by the seed-averaged validation loss. Deploy the median seed.
    winner = min(report["groups"], key=lambda g: g["mean_val_loss"])["representative"]
    best = next(r for r in report["runs"] if r["tag"] == winner)
    selected = load_policy(args.output / f"{winner}.pt")
    save_model(args.output / "policy.pt", selected, order[:best["size"]], best["seed"], feature_config)
    test = [read_episode(args.cache, r) for r in splits["test"]]
    report["selected"] = winner
    report["selection"] = "Lowest mean episode-macro validation loss across seeds; median-loss seed within group."
    report["test"] = evaluate(selected, test, encoded_episodes(selected, test))
    write_json(args.output / "report.json", report)
    print(f"Selected {winner}. Saved {args.output / 'policy.pt'} and report.json.", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=Path("outputs/thermalfly_rgb_v1/cache"))
    parser.add_argument("--output", type=Path, default=Path("outputs/thermalfly_rgb_v1/train"))
    parser.add_argument("--sizes", nargs="+", type=int, help="Nested training cohort sizes (default: 8, 16, all, capped by availability)")
    parser.add_argument("--seeds", nargs="+", type=int, default=[11, 23, 47])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--lr", type=float, default=.001)
    args = parser.parse_args()
    sizes = args.sizes or []
    if min([*sizes, args.epochs, args.patience, args.batch, args.threads]) < 1 or args.lr <= 0:
        parser.error("Training counts and learning rate must be positive")
    if len(set(sizes)) != len(sizes) or len(set(args.seeds)) != len(args.seeds):
        parser.error("Sizes and seeds must be unique")
    run(args)


if __name__ == "__main__":
    main()

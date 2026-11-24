"""Frozen OpenFly readout shared by indoor RGB navigation tasks."""
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


FEATURE_WIDTH = 5632
LANGUAGE_WIDTH = 4096
STATE_WIDTH = 10
COMPONENTS = 32
SLOTS = 3
FIELDS = 8
FORMAT = "thermalfly.rgb_waypoint.v1"
# Output order: forward, left, up, yaw, STOP logit, seconds, m/s, rig degrees.
SCALE = (.70, .70, .30, 35., 1., 2., 1., 60.)


def fit_pca(features):
    """Build a whitened projection using only the supplied training observations."""
    x = features.double()
    mean = x.mean(dim=0)
    x = x - mean
    values, vectors = torch.linalg.eigh(x @ x.T)
    keep = values.argsort(descending=True)[:COMPONENTS]
    keep = keep[values[keep] > max(1e-12, float(values.max()) * 1e-8)]
    matrix = torch.zeros(x.shape[1], COMPONENTS, dtype=x.dtype)
    if keep.numel():
        matrix[:, :keep.numel()] = (x.T @ vectors[:, keep]) * ((len(x) - 1) ** .5 / values[keep])
    return mean.float(), matrix.float()


class WaypointPolicy(nn.Module):
    def __init__(self, architecture="mlp"):
        super().__init__()
        if architecture not in {"linear", "mlp"}:
            raise ValueError(f"Unknown head architecture: {architecture}")
        self.architecture = architecture
        width = COMPONENTS * 2 + STATE_WIDTH
        self.head = (nn.Linear(width, SLOTS * FIELDS) if architecture == "linear" else
                     nn.Sequential(nn.Linear(width, 32), nn.GELU(), nn.Dropout(.1), nn.Linear(32, SLOTS * FIELDS)))
        for name, size in (("language", LANGUAGE_WIDTH), ("spatial", FEATURE_WIDTH - LANGUAGE_WIDTH)):
            self.register_buffer(name + "_mean", torch.zeros(size))
            self.register_buffer(name + "_projection", torch.zeros(size, COMPONENTS))
        self.register_buffer("state_mean", torch.zeros(STATE_WIDTH))
        self.register_buffer("state_scale", torch.ones(STATE_WIDTH))
        self.register_buffer("stop_threshold", torch.tensor(.5))

    @torch.no_grad()
    def fit_inputs(self, features, state):
        if features.shape != (len(state), FEATURE_WIDTH) or state.shape[1:] != (STATE_WIDTH,):
            raise ValueError("Invalid dimensions for fitting the readout")
        self.language_mean, self.language_projection = fit_pca(features[:, :LANGUAGE_WIDTH])
        self.spatial_mean, self.spatial_projection = fit_pca(features[:, LANGUAGE_WIDTH:])
        self.state_mean = state.mean(dim=0)
        # A near-constant sensor channel must not amplify small rollout noise.
        floor = state.new_tensor([.03] * 6 + [.01] * 3 + [1.])
        self.state_scale = torch.maximum(state.std(dim=0, unbiased=False), floor)

    def encode(self, features, state):
        return torch.cat(((features[:, :LANGUAGE_WIDTH] - self.language_mean) @ self.language_projection,
                          (features[:, LANGUAGE_WIDTH:] - self.spatial_mean) @ self.spatial_projection,
                          (state - self.state_mean) / self.state_scale), dim=-1)

    def forward(self, encoded):
        return self.head(encoded).reshape(-1, SLOTS, FIELDS)

    @torch.inference_mode()
    def predict(self, feature, state):
        feature = torch.as_tensor(feature, dtype=torch.float32)
        state = torch.as_tensor(state, dtype=torch.float32)
        if feature.shape != (FEATURE_WIDTH,) or state.shape != (STATE_WIDTH,):
            raise ValueError("Expected one 5632-dimensional feature and one 10-dimensional state")
        encoded = self.encode(feature[None], state[None])
        slots = physical_slots(self(encoded))[0]
        if not torch.isfinite(slots).all():
            raise ValueError("Nonfinite waypoint prediction")
        chunk = [{"delta_flu_m_yaw_deg": row[:4], "stop_probability": row[4], "duration_s": row[5],
                  "target_reference_speed_mps": row[6], "target_rig_pitch_deg": row[7]} for row in slots.tolist()]
        return {"chunk": chunk, "stop_threshold": float(self.stop_threshold)}


def physical_slots(output):
    """Convert scaled model outputs into executable local commands."""
    slots = output * output.new_tensor(SCALE)
    xy = slots[..., :2]
    slots[..., :2] = xy * (.70 / xy.norm(dim=-1, keepdim=True).clamp_min(.70))
    slots[..., 2].clamp_(-.30, .30)
    slots[..., 3].clamp_(-35., 35.)
    slots[..., 4] = output[..., 4].sigmoid()
    slots[..., 5] = (slots[..., 5] * 20.).round().clamp(1., 40.) / 20.
    slots[..., 6].clamp_(0., 2.)
    slots[..., 7].clamp_(-90., 90.)
    return slots


def waypoint_loss(output, target, kind, pitch_valid):
    """Each masked task is normalized separately, preserving STOP and padding semantics."""
    move = kind == 0
    usable = kind != 2
    scaled = target / target.new_tensor(SCALE)
    # Replacing masked values also permits NaN sentinels on absent regression targets.
    def regression(prediction, reference, mask):
        mask = mask.expand_as(prediction)
        safe = torch.where(mask, reference, prediction.detach())
        loss = F.smooth_l1_loss(prediction, safe, beta=.2, reduction="none")
        return loss.sum() / mask.sum().clamp_min(1)

    motion = regression(output[..., :4], scaled[..., :4], move[..., None])
    stop = F.binary_cross_entropy_with_logits(output[..., 4], (kind == 1).float(),
                                               pos_weight=output.new_tensor(3.), reduction="none")
    timing = regression(output[..., 5], scaled[..., 5], move)
    speed = regression(output[..., 6], scaled[..., 6], move)
    pitch = regression(output[..., 7], scaled[..., 7], move & pitch_valid)
    return motion + (stop * usable).sum() / usable.sum().clamp_min(1) + .25 * (timing + speed + pitch)


def load_policy(path):
    checkpoint = torch.load(Path(path), map_location="cpu", weights_only=True)
    if checkpoint.get("format") != FORMAT:
        raise ValueError("Expected a ThermalFly RGB waypoint v1 checkpoint")
    policy = WaypointPolicy(checkpoint["architecture"])
    policy.load_state_dict(checkpoint["state_dict"])
    policy.feature_config = checkpoint.get("feature_config", {})
    return policy.eval()

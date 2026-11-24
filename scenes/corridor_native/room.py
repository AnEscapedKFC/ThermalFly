"""Calibrated first-person RGB/thermal views of the corridor base_30000 reconstruction."""
from pathlib import Path
import json
import numpy as np
import torch

from scene import GaussianScene
from scenes.room65_selected_anchored.room import NativeModel, RigScene as CalibratedRig, intrinsic, tensor

ROOT = Path(__file__).resolve().parent


def read(name):
    return json.loads((ROOT / name).read_text())


class CorridorModel(NativeModel):
    def __init__(self, kind, width, height, device="cuda"):
        GaussianScene.__init__(self, ROOT / "models" / f"{kind}.ply", width, height, device=device)


class RigScene(CalibratedRig):
    def __init__(self, scene_path, width, height, up_axis, scale, device, *, render_thermal=True,
                 rgb_exposure_ev=0.):
        if Path(scene_path).resolve() != (ROOT / "models/rgb.ply").resolve() or up_axis != "z" or scale != 1.:
            raise ValueError("Use the corridor RGB model with metric Z-up collision coordinates")
        if not np.isfinite(rgb_exposure_ev) or rgb_exposure_ev > 0:
            raise ValueError("Corridor RGB attenuation requires a finite exposure offset <= 0 EV")
        self.rgb_exposure_ev = float(rgb_exposure_ev)
        self.width, self.height = width, round(width * 3 / 4)
        self.device = torch.device(device)
        self.rgb = CorridorModel("rgb", width, self.height, device)
        self.thermal = CorridorModel("thermal", width, self.height, device) if render_thermal else None
        cfg = read("inputs/frame.json")
        self.R, self.origin = np.array(cfg["rotation"]), np.array(cfg["origin"])
        self.s, self.bounds = cfg["meters_per_source_unit"], np.array(cfg["bounds_m"])
        rows = read("inputs/manifest.json")["views"]
        modalities = ("rgb", "thermal") if render_thermal else ("rgb",)
        self.ks = {m: intrinsic(next(r for r in rows if r["modality"] == m), width, self.height)
                   for m in modalities}
        if render_thermal:
            self.rig = np.array(read("inputs/rig.json")["rgb_to_thermal"])
        self.K = tensor(self.ks["rgb"])[None]

    def render(self, camera):
        rgb, depth, alpha = super().render(camera)
        if self.rgb_exposure_ev:
            # Treat the fitted display RGB as sRGB. Apply exposure in linear light
            # before the simulator quantizes the camera image for the policy.
            linear = np.where(rgb <= .04045, rgb / 12.92, ((rgb + .055) / 1.055) ** 2.4)
            linear *= 2. ** self.rgb_exposure_ev
            rgb = np.where(linear <= .0031308, 12.92 * linear, 1.055 * linear ** (1 / 2.4) - .055)
        return rgb, depth, alpha

    def metadata(self):
        return {"name": "corridor_full_native_base_30000",
                "rig": "trained RGB to thermal" if self.thermal is not None else "calibrated RGB",
                "rgb_exposure_ev": self.rgb_exposure_ev, "rgb_linear_gain": 2. ** self.rgb_exposure_ev,
                "meters_per_source_unit": self.s, "bounds_m": self.bounds.tolist()}


class KnownSpace:
    """Every grid cell possibly touched by the safety box must have free-space observations."""
    def __init__(self):
        with np.load(ROOT / "collision/free_volume.npz") as grid:
            self.free, self.lo, self.step = grid["free"], grid["origin"], float(grid["step"])

    def contains_box(self, p, R, half, extra=0.):
        p, R, half = np.asarray(p), np.asarray(R), np.asarray(half) + extra
        bound = abs(R) @ half
        low = np.floor((p - bound - self.lo) / self.step).astype(int)
        high = np.floor((p + bound - self.lo) / self.step).astype(int)
        if np.any(low < 0) or np.any(high >= self.free.shape):
            return False
        indices = np.indices(high - low + 1).reshape(3, -1).T + low
        local = (self.lo + (indices + .5) * self.step - p) @ R
        touches = (abs(local) <= half + self.step / 2 * abs(R).sum(0) + 1e-7).all(1)
        return bool(self.free[tuple(indices[touches].T)].all())

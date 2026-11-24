"""Run indoor OpenFly actions against a calibrated camera rig and quadrotor dynamics."""

import json
import sys
from functools import partial
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from dynamics import CONTROL_DT
from engine import PolicyFlightSimulator

# The scene bundle's scripts import their neighbours as top-level modules.
ROOM = Path(__file__).resolve().parents[1] / "scenes" / "room65_selected_anchored"
sys.path.insert(0, str(ROOM))
try:
    from fly import KnownSpace, commands, controller
    from room import RigScene
finally:
    sys.path.remove(str(ROOM))


# Keys are the upstream decoded action IDs. Distances use simulator metres;
# positive sideways motion is left, positive yaw turns left.
ACTION_NAMES = ("stop", "forward_short", "turn_left", "turn_right", "up", "down",
                "strafe_left", "strafe_right", "forward_medium", "forward_long")
WAYPOINT_MOVE_SECONDS = 1.5
WAYPOINT_HOLD_SECONDS = .5
#                       forward   left     up     yaw (degrees)
INDOOR_ACTIONS = {
    0: (0.00,  0.00,  0.00,   0.0),  # stop and hold
    1: (0.15,  0.00,  0.00,   0.0),
    2: (0.00,  0.00,  0.00,  15.0),
    3: (0.00,  0.00,  0.00, -15.0),
    4: (0.00,  0.00,  0.10,   0.0),
    5: (0.00,  0.00, -0.10,   0.0),
    6: (0.00,  0.15,  0.00,   0.0),
    7: (0.00, -0.15,  0.00,   0.0),
    8: (0.30,  0.00,  0.00,   0.0),
    9: (0.45,  0.00,  0.00,   0.0),
}


def action_trajectory(state, action_id):
    """Create [time, world xyz, yaw degrees] knots from the actual initial state."""
    if isinstance(action_id, bool) or not isinstance(action_id, (int, np.integer)) or action_id not in INDOOR_ACTIONS:
        raise ValueError(f"Expected a decoded OpenFly integer action in [0, 9], got {action_id!r}")
    waypoint = INDOOR_ACTIONS[action_id]
    # A quintic segment has peak speed 1.875 * displacement / duration.
    duration = max(1.5, 1.875 * np.linalg.norm(waypoint[:3]) / 0.45, 1.875 * abs(waypoint[3]) / 25)
    return waypoint_trajectory(state, waypoint, duration)


def waypoint_trajectory(state, waypoint, duration=1.0):
    """Convert relative horizontal-body-frame displacement to simulator world knots."""
    waypoint = np.asarray(waypoint, dtype=float)
    if waypoint.shape != (4,) or not np.isfinite(waypoint).all():
        raise ValueError("An indoor waypoint requires four finite components")
    forward, left, up, turn = waypoint
    if np.hypot(forward, left) > .700001 or abs(up) > .300001 or abs(turn) > 35.000001:
        raise ValueError("Indoor waypoint exceeds the room head's displacement or turn limits")
    yaw_deg = state["euler_deg"][2]
    yaw = np.deg2rad(yaw_deg)
    delta = np.array([forward * np.cos(yaw) - left * np.sin(yaw),
                      forward * np.sin(yaw) + left * np.cos(yaw), up])
    start = np.asarray(state["position"])
    return np.array([[0, *start, yaw_deg], [duration, *(start + delta), yaw_deg + turn]])


class RoomFlight:
    """Closed-loop flight with collision receipts and observed-free-volume checks."""

    def __init__(self, spawn=None, yaw_deg=None, seed=65, resolution=0.5, max_steps=None,
                 *, scene_root=ROOM, scene_factory=RigScene, known_space=None, render_thermal=False,
                 spawn_velocity_w_mps=(0., 0., 0.)):
        if max_steps is not None and (type(max_steps) is not int or max_steps < 1):
            raise ValueError("max_steps must be a positive integer")
        if spawn is None or yaw_deg is None:
            first = json.loads((scene_root / "route.json").read_text())["knots"][0]
            spawn = first[1:4] if spawn is None else spawn
            yaw_deg = first[4] if yaw_deg is None else yaw_deg
        self.known = KnownSpace() if known_space is None else known_space
        self.path_m = 0.0
        self.sim = PolicyFlightSimulator(
            scene_root / "models/rgb.ply", point_cloud=scene_root / "collision/rgb_surface_m.ply",
            config="nominal", scene_factory=partial(scene_factory, render_thermal=render_thermal),
            seed=seed, resolution=resolution,
            spawn=spawn, yaw=yaw_deg, spawn_velocity_w_mps=spawn_velocity_w_mps,
        )
        try:
            if max_steps is not None:
                self.sim.limits["max_steps"] = max_steps
            self.outcome = self._terminal_reason()
        except BaseException:
            self.sim.close()
            raise

    @property
    def rgb(self):
        return np.rint(self.sim.last_rgb * 255).astype(np.uint8)

    @property
    def receipt(self):
        return self.sim.last_receipt

    def _terminal_reason(self):
        r = self.receipt
        if r["terminated"] or r["truncated"]:
            return r["reason"]
        state = r["state"]
        orientation = Rotation.from_quat(np.asarray(state["quaternion_wxyz"])[[1, 2, 3, 0]])
        if not self.known.contains_box(np.asarray(state["position"]), orientation.as_matrix(),
                                       self.sim.obstacles.safety_half_size):
            return "unknown_space"
        return None

    def execute(self, action_id, on_tick=None):
        """Execute one decoded action; report actual stop, simulator failure or unknown space."""
        return self._track(action_trajectory(self.receipt["state"], action_id), action_id == 0, on_tick)

    def execute_waypoint(self, waypoint, *, stop=False, on_tick=None):
        """Track the predicted displacement and settle before the next camera observation."""
        if stop:
            return self.execute(0, on_tick)
        knots = waypoint_trajectory(self.receipt["state"], waypoint, WAYPOINT_MOVE_SECONDS)
        hold = knots[-1].copy()
        hold[0] += WAYPOINT_HOLD_SECONDS
        return self._track(np.vstack([knots, hold]), False, on_tick)

    def _track(self, knots, stop, on_tick, *, sample=commands, before_tick=None):
        if self.outcome is not None:
            raise RuntimeError(f"Flight has already ended: {self.outcome}")
        ticks = int(np.ceil(knots[-1, 0] / CONTROL_DT - 1e-9))
        for index in range(ticks):
            before = np.asarray(self.receipt["state"]["position"])
            mid_time = (index + 0.5) * CONTROL_DT
            command, _ = controller(self.sim, self.receipt, knots, mid_time, sample=sample)
            if before_tick is not None:
                before_tick((index + 1) * CONTROL_DT)
            self.sim.step(command, format="body_rates_thrust")
            state = self.receipt["state"]
            self.path_m += float(np.linalg.norm(state["position"] - before))
            self.outcome = self._terminal_reason()
            if on_tick is not None:
                on_tick(self.receipt, self.rgb)
            if self.outcome is not None:
                return self.outcome
        if stop:
            speed = self.receipt["state"]["speed"]
            drift = np.linalg.norm(self.receipt["state"]["position"] - knots[0, 1:4])
            self.outcome = "policy_stop" if speed <= 0.06 and drift <= 0.08 else "hover_unsettled"
        return self.outcome

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.sim.close()

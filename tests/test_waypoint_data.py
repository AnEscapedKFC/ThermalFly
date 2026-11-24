import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from openfly_rgb.waypoint_data import (
    cached, FIXED_HEIGHT_TEXT, instruction_for, reviews_for, state_vector, targets_for, VARIABLE_HEIGHT_TEXT,
)


class WaypointDataTests(unittest.TestCase):
    def test_body_state_is_body_relative_and_preserves_rig_degrees(self):
        q = np.roll(Rotation.from_euler("z", 90, degrees=True).as_quat(), 1)
        actual = state_vector([0, 2, 0], q, [.2, 0, 0], -40.)
        np.testing.assert_allclose(actual, [2, 0, 0, .2, 0, 0, 0, 0, -1, -40], atol=1e-6)

    def test_turn_move_stop_and_pad_have_distinct_targets(self):
        move = dict(kind="move", delta_flu_m_yaw_deg=[0, 0, 0, 12], duration_s=.4,
                    target_reference_speed_mps=0)
        slots = [move, {"kind": "stop"}, {"kind": "pad"}]
        target, kind, pitch_valid = targets_for(slots)
        np.testing.assert_array_equal(kind, [0, 1, 2])
        np.testing.assert_array_equal(target[:, 4], [0, 1, 0])
        self.assertEqual(target[0, 3], 12)
        self.assertFalse(pitch_valid.any())
        move["target_rig_pitch_deg"] = 0.
        np.testing.assert_array_equal(targets_for(slots)[2], [True, False, False])

    def test_review_overlay_and_instruction_correction_are_scoped(self):
        meta = {"episode_id": "room_orbit_vary", "instruction": FIXED_HEIGHT_TEXT}
        self.assertEqual(instruction_for(meta), VARIABLE_HEIGHT_TEXT)
        self.assertEqual(meta["instruction"], FIXED_HEIGHT_TEXT)
        self.assertEqual(instruction_for(dict(meta, episode_id="other")), FIXED_HEIGHT_TEXT)
        with tempfile.TemporaryDirectory() as tmp:
            dataset = Path(tmp)
            path = dataset / "expert_episodes_human_review.json"
            path.write_text(json.dumps(dict(dataset=dataset.name, episodes={"a": {"status": "rejected"}})))
            self.assertEqual(reviews_for(dataset, None, {"a"}), {"a": "rejected"})
            with self.assertRaises(ValueError):
                reviews_for(dataset, path, {"another_dataset_episode"})

    def test_partial_or_stale_episode_is_not_reused(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "episode.npz"
            arrays = {"state": np.zeros((1, 10), np.float32)}
            np.savez(path, **arrays, signature="current", features=np.zeros((1, 5632), np.float32))
            self.assertTrue(cached(path, "current", arrays))
            self.assertFalse(cached(path, "changed", arrays))
            path.write_bytes(path.read_bytes()[:100])
            self.assertFalse(cached(path, "current", arrays))

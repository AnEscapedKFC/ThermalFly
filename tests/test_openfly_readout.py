import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from openfly_rgb.model_worker import OpenFlyPolicy, RoomHead, fit_room_head


class TestOpenFlyReadout(unittest.TestCase):
    def test_fit_load_stop_and_flight_envelope(self):
        features = np.eye(5)
        targets = np.array([[0, 0, 0, 0, 1], [2, -2, .8, 75, 0],
                            [-2, 2, -.8, -75, 0], [.3, 0, 0, 0, 0], [0, .2, 0, 10, 0]])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "readout.npz"
            fit_room_head(features, targets, path)
            head = RoomHead(path)
            self.assertEqual(head.version, 2)
            for feature, target in zip(features, targets):
                answer = head.predict(feature)
                self.assertEqual(answer["stop"], bool(target[-1]))
                self.assertGreaterEqual(answer["stop_score"], 0)
                self.assertLessEqual(answer["stop_score"], 1)
            for feature in [*features, np.full(5, -1)]:
                command = head.predict(feature)["waypoint"]
                self.assertTrue(np.isfinite(command).all())
                self.assertLessEqual(np.linalg.norm(command[:2]), .700001)
                self.assertLessEqual(abs(command[2]), .300001)
                self.assertLessEqual(abs(command[3]), 35)
            np.testing.assert_allclose(head.predict(features[0])["waypoint"],
                                       RoomHead(path).predict(features[0])["waypoint"])
            with self.assertRaisesRegex(ValueError, "dimensions"):
                head.predict([0.1, 0.9])

    def test_spatial_grid_and_v1_compatibility(self):
        policy = OpenFlyPolicy.__new__(OpenFlyPolicy)
        policy.room_head = None
        policy.visual_projection = None
        # Two historical pooled tokens precede 256 spatially ordered current
        # tokens. Keep the prompt state fixed to isolate the current RGB branch.
        grid = torch.zeros(16, 16, 4)
        grid[:, :8, 0] = 1
        grid[:, 8:, 1] = 1
        history = torch.zeros(1, 2, 4)
        final_text = torch.tensor([[[0., 0., 1., 0.], [0., 0., 0., 1.]]])

        def feature(patches):
            projected = torch.cat((history, patches.reshape(1, 256, 4)), dim=1)
            policy.model = lambda **kw: SimpleNamespace(hidden_states=(final_text,), projector_features=projected)
            return policy.fused_feature({})

        initial = feature(grid)
        mirrored = feature(grid.flip(1))
        np.testing.assert_allclose(initial[:4], [0, 0, 0, .5])
        np.testing.assert_allclose(initial[:4], mirrored[:4])
        self.assertGreater(np.linalg.norm(initial[4:] - mirrored[4:]), 0.5)
        self.assertAlmostEqual(float(np.linalg.norm(initial)), 1., places=5)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "version1.npz"
            np.savez(path, version=1, center=np.zeros(4), support=np.eye(4),
                     weights=np.zeros((4, 5)), width_squared=1)
            policy.room_head = RoomHead(path)
            np.testing.assert_allclose(feature(grid), [0., 0., 0., 1.])
            self.assertFalse(policy.room_head.predict(feature(grid))["stop"])


if __name__ == "__main__":
    unittest.main()

import tempfile
import unittest
from pathlib import Path

import torch

from openfly_rgb.train_waypoint import calibrate_stop, chain_positions, nested_order, save_model
from openfly_rgb.waypoint_head import FEATURE_WIDTH, WaypointPolicy, fit_pca, load_policy, physical_slots, waypoint_loss


class TestWaypointHead(unittest.TestCase):
    def test_masked_padding_stop_and_optional_pitch(self):
        kind = torch.tensor([[0, 1, 2]])
        pitch = torch.zeros_like(kind, dtype=torch.bool)
        output = torch.full((1, 3, 8), .2, requires_grad=True)
        target = torch.full_like(output, float("nan"))
        target[0, 0, :4] = 0.
        target[0, 0, 5:7] = 0.
        loss = waypoint_loss(output, target, kind, pitch)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(output.grad).all())
        self.assertTrue((output.grad[0, 2] == 0.).all())
        self.assertTrue((output.grad[0, :, 7] == 0.).all())
        self.assertEqual(output.grad[0, 1].count_nonzero(), 1)
        self.assertLess(output.grad[0, 1, 4], 0.)
        self.assertGreater(output.grad[0, 0, 4], 0.)

    def test_pca_fit_rank_deficiency_and_reload(self):
        torch.manual_seed(5)
        rows = torch.randn(4, FEATURE_WIDTH)
        states = torch.ones(4, 10)
        model = WaypointPolicy().eval()
        model.fit_inputs(rows, states)
        self.assertTrue(torch.isfinite(model.encode(rows, states)).all())
        self.assertEqual(fit_pca(torch.ones(4, 10))[1].count_nonzero(), 0)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "test.pt"
            save_model(path, model, ["train_episode"], 5)
            self.assertEqual(load_policy(path).predict(rows[0], states[0]), model.predict(rows[0], states[0]))

    def test_physical_units_and_yaw_chain(self):
        raw = torch.ones(1, 3, 8) * 100
        command = physical_slots(raw)
        self.assertAlmostEqual(float(command[0, 0, :2].norm()), .70, places=5)
        self.assertEqual(command[0, 0, 3], 35.)
        self.assertEqual(command[0, 0, 5], 2.)
        first_two = torch.tensor([[[1., 0., 0., 90.], [1., 0., 0., 0.]]])
        torch.testing.assert_close(chain_positions(first_two), torch.tensor([[[1., 0., 0.], [1., 1., 0.]]]))

    def test_family_coverage_and_stop_threshold(self):
        rows = [{"route_family_id": f, "episode_id": f"{f}_{i}"} for f in ("A", "B", "C") for i in range(3)]
        self.assertEqual(nested_order(rows), nested_order(rows[::-1]))
        self.assertEqual({name[0] for name in nested_order(rows)[:3]}, {"A", "B", "C"})
        scores = torch.tensor([.2, .4, .35, .7])
        labels = torch.tensor([False, True, False, True])
        self.assertTrue(torch.equal(scores >= calibrate_stop(scores, labels), labels))


if __name__ == "__main__":
    unittest.main()

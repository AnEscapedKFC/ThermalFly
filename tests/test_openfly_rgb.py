import unittest

import numpy as np

from openfly_rgb.flight import action_trajectory, waypoint_trajectory
from openfly_rgb.model_worker import ACTION_VECTORS, decode_action
from openfly_rgb.navigate import evaluate_stop


class IndoorOpenFlyTests(unittest.TestCase):
    def test_only_an_explicit_stop_decodes_to_zero(self):
        for action_id, vector in enumerate(ACTION_VECTORS):
            self.assertEqual(decode_action(vector), action_id)
        self.assertIsNone(decode_action([0.01] * 8))
        with self.assertRaises(ValueError):
            decode_action([float("nan")] * 8)

    def test_turn_and_strafe_use_the_room_heading(self):
        state = {"position": [1., 2., 1.5], "euler_deg": [0., 0., 90.]}
        np.testing.assert_allclose(action_trajectory(state, 9)[-1, 1:4], [1., 2.45, 1.5])
        np.testing.assert_allclose(action_trajectory(state, 6)[-1, 1:4], [.85, 2., 1.5])
        self.assertEqual(action_trajectory(state, 2)[-1, 4], 105.)

    def test_combined_waypoint_and_physical_limits(self):
        state = {"position": [1., 2., 1.5], "euler_deg": [0., 0., 90.]}
        endpoint = waypoint_trajectory(state, [.1, -.3, .04, 24.])[-1, 1:]
        np.testing.assert_allclose(endpoint, [1.3, 2.1, 1.54, 114.])
        with self.assertRaises(ValueError):
            waypoint_trajectory(state, [0., 2., 0., 0.])

    def test_success_requires_a_settled_policy_stop_and_handles_yaw_wrap(self):
        state = {"position": [0., 0., 1.], "speed": .02, "euler_deg": [0., 0., 179.]}
        def success(outcome, goal_yaw):
            return evaluate_stop(state, outcome, [0., 0., 1.], goal_yaw)["success"]
        self.assertTrue(success("policy_stop", -179.))
        self.assertFalse(success("decision_limit", -179.))
        self.assertFalse(success("policy_stop", 0.))
        state["speed"] = .3
        self.assertFalse(success("policy_stop", -179.))
        self.assertIsNone(evaluate_stop(state, "policy_stop", None, None)["success"])

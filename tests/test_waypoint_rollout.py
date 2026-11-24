import unittest

import numpy as np

from openfly_rgb.waypoint_rollout import continuous_segment, pitch_rig, terminal_score


class ContinuousExecutionTests(unittest.TestCase):
    def setUp(self):
        self.state = dict(position=[1., 2., 1.], velocity=[0., .4, 0.],
                          euler_deg=[0., 0., 90.], angular_velocity=[0., 0., 0.])
        self.forward = dict(delta_flu_m_yaw_deg=[.4, 0., 0., 0.], duration_s=1.,
                            stop_probability=.01, target_reference_speed_mps=.4)

    def test_neighbour_tangent_and_entry_velocity(self):
        knots, sample = continuous_segment(self.state, [self.forward, self.forward], .5)
        np.testing.assert_allclose(knots[-1, 1:], [1., 2.4, 1., 90.])
        for t in (0., .5, 1.):
            np.testing.assert_allclose(sample(knots, t)[1][:3], self.state["velocity"], atol=1e-9)

    def test_turn_uses_changed_heading_and_stop_has_zero_exit_speed(self):
        turning = dict(self.forward, delta_flu_m_yaw_deg=[.4, 0., 0., 30.])
        knots, sample = continuous_segment(self.state, [turning, self.forward], .5)
        v = sample(knots, 1.)[1][:3]
        angle = np.rad2deg(np.arctan2(v[1], v[0]))
        self.assertAlmostEqual(angle, 105.)
        wp = dict(turning, duration_s=.17)
        knots, sample = continuous_segment(self.state, [wp, {"stop_probability": .9}], .5)
        self.assertAlmostEqual(knots[-1, 0], .15)
        np.testing.assert_allclose(sample(knots, knots[-1, 0])[1], np.zeros(4), atol=1e-9)

    def test_gimbal_composes_local_rotation_before_render(self):
        class Rig:
            def render(self, mount):
                return mount
        rig = pitch_rig(Rig, -30.)()
        m = np.eye(4)
        out = rig.render(m)
        self.assertAlmostEqual(out[2, 1], -.5)
        self.assertAlmostEqual(m[2, 1], 0.)

    def test_endpoint_needs_policy_stop(self):
        task = dict(goal_w_m=[0., 0., 0.], goal_yaw_deg=181., yaw_tol_deg=5.,
                    position_tol_m=.1, speed_tol_mps=.1)
        state = dict(position=[0., 0., 0.], euler_deg=[0., 0., -179.], speed=0.)
        self.assertTrue(terminal_score(task, state, "policy_stop")["terminal_success"])
        self.assertFalse(terminal_score(task, state, "max_decisions")["terminal_success"])

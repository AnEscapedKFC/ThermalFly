import unittest

import numpy as np

from openfly_rgb.score_waypoint import longest_view_sweep, stable_hover_seconds


class TestViewSweep(unittest.TestCase):
    def test_hidden_target_breaks_arc_and_yaw_wrap_is_continuous(self):
        angles = np.deg2rad([170, 179, -170, -160, -150, -140])
        points = np.column_stack((np.cos(angles), np.sin(angles), np.ones(len(angles))))
        center = np.zeros(3)
        self.assertAlmostEqual(longest_view_sweep(points, center, [True] * 6), 50.)
        self.assertAlmostEqual(longest_view_sweep(points, center, [True, True, False, True, True, True]), 20.)
        self.assertEqual(longest_view_sweep(points, center, [False] * 6), 0.)

    def test_hover_counts_only_settled_tail_after_stop(self):
        times = np.array([0, 1, 2, 3, 4])
        positions = np.zeros((5, 3))
        speed = np.array([0., 0., .1, .01, 0.])
        self.assertEqual(stable_hover_seconds(times, positions, speed, 1), 1.)
        self.assertEqual(stable_hover_seconds(times, positions, speed, None), 0.)

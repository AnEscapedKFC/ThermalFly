import copy
import importlib.util
import json
from pathlib import Path
import sys
import unittest

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from action_adapter import ActionAdapter, make_dynamics_parameters
from dynamics import HOVER, QuadrotorDynamics, initial_state, quaternion_rotation


def profile(name='nominal'):
    return json.loads((ROOT/'configs'/f'{name}.json').read_text())


class ActionAdapterTests(unittest.TestCase):
    def plain(self):
        cfg = profile();cfg['lpf']['enabled'] = False;cfg['delay']['enabled'] = False
        return ActionAdapter(cfg, HOVER)

    def test_action_order_scaling_saturation_and_nominal_hover(self):
        adapter = self.plain()
        np.testing.assert_allclose(adapter.adapt([3., -2., .4, 8.]), [.5, -.5, .2, .5])
        self.assertTrue(adapter.last_info['input_clipped'])
        np.testing.assert_allclose(adapter.adapt([0., 0., 0., -8.]), [0., 0., 0., 0.])
        raw_hover = HOVER/.25-1.
        np.testing.assert_allclose(adapter.adapt([0., 0., 0., raw_hover]), [0., 0., 0., HOVER])
        self.assertNotAlmostEqual(float(adapter.adapt([0, 0, 0, 0])[3]), HOVER)

    def test_logits_are_tanh_transformed_once(self):
        values = np.array([.3, -2., 1.2, .9])
        logits = self.plain().adapt(values, 'gradnav_logits')
        normalized = self.plain().adapt(np.tanh(values), 'gradnav_normalized')
        np.testing.assert_allclose(logits, normalized)
        self.assertNotAlmostEqual(logits[0], .5*np.tanh(np.tanh(values[0])))

    def test_physical_controls_bypass_mapping_filter_and_delay(self):
        adapter = ActionAdapter(profile(), HOVER)
        direct = [.12, -.08, .2, .8]
        np.testing.assert_allclose(adapter.adapt(direct, 'body_rates_thrust'), direct)
        self.assertFalse(adapter.last_info['delay_applied'])
        self.assertFalse(adapter.last_info['lpf_applied'])
        self.assertEqual(adapter.adapt([1., 2., 3., 2.], 'body_rates_thrust')[3], 1.)

    def test_legacy_delay_reads_two_steps_back_and_reset_clears_memory(self):
        cfg = profile();cfg['lpf']['enabled'] = False
        adapter = ActionAdapter(cfg, HOVER)
        first = adapter.adapt([1., 0., 0., 1.])
        second = adapter.adapt([0., 0., 0., -1.])
        third = adapter.adapt([0., 0., 0., -1.])
        self.assertAlmostEqual(first[0], .4)
        self.assertAlmostEqual(second[0], 0.)
        self.assertAlmostEqual(third[0], .08)
        self.assertAlmostEqual(second[3], .3*HOVER)
        self.assertAlmostEqual(third[3], .3*first[3])
        adapter.reset()
        np.testing.assert_allclose(adapter.adapt([1., 0., 0., 1.]), first)
        self.assertEqual(adapter.action_index, 1)

    def test_previous_step_delay_and_lpf_initialization(self):
        cfg = profile();cfg['delay']['history'] = 'previous_step'
        adapter = ActionAdapter(cfg, HOVER)
        first = adapter.adapt([1., 0., 0., 0.])
        second = adapter.adapt([0., 0., 0., 0.])
        self.assertAlmostEqual(first[0], .4)
        self.assertAlmostEqual(second[0], .8*.25+.2*.4)
        np.testing.assert_allclose(adapter.previous_control, first)
        adapter.reset()
        np.testing.assert_allclose(adapter.adapt([0., 0., 0., 0.])[:3], [0., 0., 0.])

    def test_invalid_actions_do_not_mutate_history(self):
        adapter = ActionAdapter(profile(), HOVER)
        adapter.adapt([.2, .1, 0., .5]);before = copy.deepcopy(vars(adapter))
        for values, format in [([0, 0, 0], 'gradnav_normalized'),
                               ([0, np.nan, 0, 0], 'gradnav_normalized'),
                               ([0, 0, 0, 0], 'unknown')]:
            with self.assertRaises(ValueError):adapter.adapt(values, format)
            self.assertEqual(adapter.action_index, before['action_index'])
            np.testing.assert_array_equal(adapter.current_control, before['current_control'])
            np.testing.assert_array_equal(adapter.previous_control, before['previous_control'])
            np.testing.assert_array_equal(adapter.filtered_action, before['filtered_action'])


class DynamicsProfileTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(2)

    def test_training_randomization_ranges_and_reproducibility(self):
        cfg = profile('training')
        first = make_dynamics_parameters(cfg, 15)
        self.assertEqual(first, make_dynamics_parameters(cfg, 15))
        self.assertNotEqual(first, make_dynamics_parameters(cfg, 16))
        self.assertTrue(1. <= first['mass'] <= 1.2)
        self.assertTrue(24. <= first['max_thrust'] <= 28.)
        nominal = np.array([.01, .012, .025])
        self.assertTrue(np.all(np.abs(np.array(first['inertia'])/nominal-1) <= .2))
        self.assertEqual(first['rotor_noise_std'], .01)
        self.assertEqual(first['br_noise_std'], .01)

    def test_seeded_control_noise_is_local_and_reproducible(self):
        parameters = make_dynamics_parameters(profile('training'), 7)
        first = QuadrotorDynamics('cpu', parameters, seed=21)
        second = QuadrotorDynamics('cpu', parameters, seed=21)
        different = QuadrotorDynamics('cpu', parameters, seed=22)
        state = initial_state('cpu')
        u = state.new_tensor([.1, -.1, .03, parameters['mass']*9.81/parameters['max_thrust']])
        torch.manual_seed(44);expected = torch.rand(4)
        torch.manual_seed(44)
        a = first.step(state, u);b = second.step(state, u);c = different.step(state, u)
        torch.testing.assert_close(a, b, rtol=0, atol=0)
        self.assertFalse(torch.equal(a, c))
        torch.testing.assert_close(torch.rand(4), expected, rtol=0, atol=0)

    def test_nominal_hover_and_acceleration_outputs(self):
        parameters = make_dynamics_parameters(profile(), 4)
        sim = QuadrotorDynamics('cpu', parameters)
        state = initial_state('cpu');control = state.new_tensor([0, 0, 0, HOVER])
        for _ in range(20):state = sim.step(state, control)
        torch.testing.assert_close(state[:3], state.new_tensor([0, 0, 1.25]), rtol=0, atol=1e-12)
        torch.testing.assert_close(sim.last_linear_acceleration, torch.zeros(3, dtype=torch.float64), rtol=0, atol=1e-12)
        torch.testing.assert_close(sim.last_angular_acceleration, torch.zeros(3, dtype=torch.float64), rtol=0, atol=1e-12)

    def test_custom_parameters_affect_dynamics_and_gradient_still_flows(self):
        sim = QuadrotorDynamics('cpu', {'mass':2., 'max_thrust':30.})
        state = initial_state('cpu')
        control = state.new_tensor([0., 0., 0., 2*9.81/30.], requires_grad=True)
        result = sim.step(state, control)
        derivative = torch.autograd.grad(result[5], control)[0]
        self.assertAlmostEqual(float(derivative[3]), 30/2*.05, places=10)
        self.assertAlmostEqual(float(result[2]), 1.25)

    def test_callback_can_stop_without_changing_collision_algorithm(self):
        sim = QuadrotorDynamics('cpu');state = initial_state('cpu')
        state[3] = 1.
        calls = []
        def stop(before, candidate, dt):
            calls.append(dt);override = before.clone();override[3:6] = 0;override[10:13] = 0
            return override
        result = sim.step(state, state.new_tensor([0, 0, 0, HOVER]), stop)
        self.assertEqual(calls, [.005])
        torch.testing.assert_close(result[:3], state[:3])
        self.assertEqual(float(sim.last_linear_acceleration.norm()), 0.)


if __name__ == '__main__':unittest.main()

import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from observations import ObservationBuilder


class VisualEncoder(torch.nn.Module):
    def forward(self, image):
        return image.mean(dim=(1, 2, 3), keepdim=False)[:, None].expand(-1, 16)


class ContextEncoder(torch.nn.Module):
    def forward(self, history):
        return history.sum(dim=1, keepdim=True).expand(-1, 24)


class ObservationChecks(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.config = {'observation': {'history': 5, 'visual_dim': 16, 'latent_dim': 24, 'noise_level': 0}}
        self.state = np.array([1, 2, 3, .1, .2, .3, np.sqrt(.5), 0, 0, np.sqrt(.5), .4, .5, .6])
        self.current = np.array([.1, .2, .3, .415])
        self.previous = np.array([-.1, -.2, -.3, .4])
        self.rgb = np.full((30, 50, 3), .25, dtype=np.float32)

    def build(self, builder, n=0, **features):
        return builder.build(self.state, self.current, self.previous, self.rgb, 10 + n, n, **features)

    def test_missing_weights_never_manufacture_policy_observation(self):
        builder = ObservationBuilder(self.config)
        first = self.build(builder)
        self.assertFalse(first['policy_ready']); self.assertIsNone(first['policy_observation'])
        self.assertEqual(len(first['base_observation']), 17)
        second = self.build(builder, 1)
        self.assertFalse(second['history_for_next']['ready'])
        self.assertIsNone(second['history_for_next']['flat'])
        self.assertEqual(second['history_for_next']['frames'][-1], {'frame_id': 10, 'step_id': 0})

    def test_exact_57_layout_and_quaternion_conversion(self):
        result = self.build(ObservationBuilder(self.config), visual_features=np.arange(16), latent_features=np.arange(24) + 100)
        expected = np.r_[3, .3, [0, 0, np.sqrt(.5), np.sqrt(.5)], self.current, self.previous, [.1, .2, .3], np.arange(16), np.arange(24) + 100]
        self.assertTrue(result['policy_ready']); self.assertEqual(len(result['policy_observation']), 57)
        np.testing.assert_allclose(result['policy_observation'], expected)
        self.assertFalse(result['normalization_applied'])

    def test_tagged_features_must_match_frame_and_step(self):
        feature = {'values': np.ones(16), 'frame_id': 10, 'step_id': 0}
        result = self.build(ObservationBuilder(self.config), visual_features=feature, latent_features=np.ones(24))
        self.assertTrue(result['policy_ready'])
        feature['frame_id'] = 9
        with self.assertRaisesRegex(ValueError, 'does not match'):
            self.build(ObservationBuilder(self.config), visual_features=feature)

    def test_history_excludes_current_frame_and_ablate_latent(self):
        builder = ObservationBuilder(self.config)
        results = [self.build(builder, i, visual_features=np.ones(16) * i, latent_features=np.ones(24) * 99) for i in range(3)]
        self.assertTrue(all(v == 0 for v in results[1]['context_history']['flat']))
        self.assertIsNone(results[1]['context_history']['frames'][-1])
        self.assertEqual(results[1]['history_for_next']['frames'][-1]['step_id'], 0)
        self.assertEqual(results[2]['context_history']['frames'][-1]['step_id'], 0)
        matrix = np.asarray(results[2]['history_for_next']['flat']).reshape(5, 57)
        np.testing.assert_allclose(matrix[-1, 17:33], 1)
        np.testing.assert_allclose(matrix[:, 33:57], 0)

    def test_cached_frame_does_not_mutate_history_or_noise(self):
        config = {'observation': {**self.config['observation'], 'noise_level': .1}}
        builder = ObservationBuilder(config, seed=8)
        first = self.build(builder, visual_features=np.ones(16), latent_features=np.ones(24))
        self.assertEqual(first, self.build(builder, visual_features=np.ones(16), latent_features=np.ones(24)))
        with self.assertRaisesRegex(ValueError, 'different observation data'):
            self.build(builder, visual_features=np.zeros(16), latent_features=np.ones(24))
        builder.reset()
        self.assertEqual(first, self.build(builder, visual_features=np.ones(16), latent_features=np.ones(24)))

    def test_sensor_noise_is_seeded_and_only_matches_source_fields(self):
        config = {'observation': {**self.config['observation'], 'noise_level': .1}}
        values = [self.build(ObservationBuilder(config, seed=5), visual_features=np.zeros(16), latent_features=np.zeros(24)) for _ in range(2)]
        self.assertEqual(values[0], values[1])
        noisy = np.asarray(values[0]['base_observation']); clean = np.asarray(values[0]['base_observation_clean'])
        self.assertTrue(np.any(noisy[:6] != clean[:6])); self.assertTrue(np.all(np.abs(noisy[:6] - clean[:6]) <= .05))
        np.testing.assert_allclose(noisy[6:], clean[6:])

    def test_rms_normalization_and_required_normalizer(self):
        rms = {'mean': np.ones(57).tolist(), 'var': (np.ones(57) * 4).tolist()}
        result = self.build(ObservationBuilder(self.config, assets={'obs_rms': rms}), visual_features=np.ones(16), latent_features=np.ones(24))
        self.assertTrue(result['normalization_applied'])
        np.testing.assert_allclose(result['policy_observation'], (np.asarray(result['raw_policy_observation']) - 1) / np.sqrt(4 + 1e-5))
        config = {'observation': {**self.config['observation'], 'require_obs_rms': True}}
        result = self.build(ObservationBuilder(config), visual_features=np.ones(16), latent_features=np.ones(24))
        self.assertFalse(result['policy_ready'])
        with self.assertRaises(ValueError):
            ObservationBuilder(self.config, assets={'obs_rms': {'mean': [0] * 57, 'var': [-1] * 57}})

    def test_real_torchscript_assets_and_relative_paths(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            torch.jit.trace(VisualEncoder(), torch.zeros(1, 3, 224, 224)).save(str(root / 'visual.pt'))
            torch.jit.trace(ContextEncoder(), torch.zeros(1, 285)).save(str(root / 'context.pt'))
            (root / 'assets.json').write_text(json.dumps({'visual_encoder': 'visual.pt', 'context_encoder': 'context.pt'}))
            builder = ObservationBuilder(self.config, assets=root / 'assets.json')
            first = self.build(builder)
            self.assertTrue(first['policy_ready'])
            np.testing.assert_allclose(first['visual_features'], .25)
            self.assertEqual(first['feature_sources']['context'], 'source_reset_zero_with_configured_encoder')
            second = self.build(builder, 1)
            np.testing.assert_allclose(second['latent_features'], 0)
            third = self.build(builder, 2)
            self.assertTrue(any(v != 0 for v in third['latent_features']))
            self.assertEqual(third['feature_sources']['context'], 'torchscript_previous_history')

    def test_skipped_steps_do_not_claim_complete_history(self):
        builder = ObservationBuilder(self.config)
        self.build(builder, visual_features=np.ones(16), latent_features=np.ones(24))
        skipped = self.build(builder, 3, visual_features=np.ones(16), latent_features=np.ones(24))
        self.assertFalse(skipped['context_history']['ready'])
        with self.assertRaisesRegex(ValueError, 'must increase'):
            self.build(builder, 2)


if __name__ == '__main__':
    unittest.main()

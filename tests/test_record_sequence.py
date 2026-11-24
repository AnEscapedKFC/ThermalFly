import csv
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from record_sequence import load_sequence, resolve_action, record_sequence, FpvVideoWriter, verify_video


class SequenceValidation(unittest.TestCase):
    def test_hover_is_normalized_using_this_episode_parameters(self):
        parameters = {'mass': 1.1, 'max_thrust': 26.}
        action = resolve_action([.1, -.2, .3, 'hover'], 'gradnav_normalized', parameters)
        self.assertAlmostEqual((action[3]+1)*.25*26., 1.1*9.81)
        direct = resolve_action([0, 0, 0, 'hover'], 'body_rates_thrust', parameters)
        self.assertAlmostEqual(direct[3]*26., 1.1*9.81)
        logits = resolve_action([0, 0, 0, 'hover'], 'gradnav_logits', parameters)
        self.assertAlmostEqual((np.tanh(logits[3])+1)*.25*26., 1.1*9.81)
        self.assertEqual(action[:3], [.1, -.2, .3])

    def test_invalid_sequence_cannot_silently_change_execution_time(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/'sequence.json'
            cases = [
                {'segments': []},
                {'dt': .1, 'segments': [{'steps': 1, 'action': [0, 0, 0, 0]}]},
                {'segments': [{'steps': 1.5, 'action': [0, 0, 0, 0]}]},
                {'segments': [{'steps': True, 'action': [0, 0, 0, 0]}]},
                {'segments': [{'steps': 1, 'action': [0, 0, 0, float('nan')]}]},
                {'segments': [{'steps': 1, 'action': [0, 0, 0, 2]}]},
                {'segments': [{'steps': 1, 'action': ['hover', 0, 0, 0]}]},
            ]
            for case in cases:
                path.write_text(json.dumps(case))
                with self.assertRaises(ValueError):
                    load_sequence(path)

    def test_cli_rejects_nonempty_output_before_importing_the_engine(self):
        with tempfile.TemporaryDirectory() as temp:
            (Path(temp)/'existing.txt').write_text('keep this file')
            script = Path(__file__).resolve().parents[1]/'record_sequence.py'
            result = subprocess.run([sys.executable, str(script), '--output-dir', temp,
                                     '--scene', '/definitely/not/a/scene.ply'],
                                    capture_output=True, text=True, timeout=10)
            self.assertEqual(result.returncode, 1)
            self.assertIn('Output directory must be empty', result.stderr)
            self.assertEqual((Path(temp)/'existing.txt').read_text(), 'keep this file')


class FakeSimulator:
    def __init__(self, collision_step=None):
        self.parameters = {'mass': 1.1, 'max_thrust': 26.}
        self.last_rgb = np.zeros((12, 16, 3), dtype=float)
        self.last_receipt = self.receipt(0, 0., 0.)
        self.collision_step = collision_step
        self.calls = []

    @staticmethod
    def receipt(step, sim_time, dt, collision=False):
        return {'step_id': step, 'frame_id': step, 'sim_time': sim_time, 'dt_actual': dt,
                'state': {'position': [.1*step, 0., 1.25], 'velocity': [0., 0., 0.],
                          'quaternion_wxyz': [1., 0., 0., 0.], 'angular_velocity': [0., 0., 0.]},
                'applied_control': [0., 0., 0., .415],
                'collision': {'event': {'reason': 'obstacle'} if collision else None,
                              'clearance': .05 if collision else .5, 'safety_clearance': .00001 if collision else .45},
                'terminated': collision, 'truncated': False, 'reason': 'collision' if collision else None}

    def step(self, action, format, repeat):
        self.calls.append((action, format, repeat))
        step = len(self.calls)
        hit = step == self.collision_step
        dt = .025 if hit else .05
        self.last_receipt = self.receipt(step, self.last_receipt['sim_time']+dt, dt, hit)
        self.last_rgb = np.full((12, 16, 3), step/10)
        return self.last_receipt


@unittest.skipUnless(importlib.util.find_spec('imageio_ffmpeg'), 'optional video encoder not installed')
class VideoRecording(unittest.TestCase):
    def test_snapshots_survive_input_cleanup_and_keep_substep_clearance(self):
        simulator = FakeSimulator()
        simulator.config = {'name': 'snapshot', 'dynamics': simulator.parameters.copy()}
        simulator.last_receipt['collision'].update(enabled=True, source='gsplat_depth', points=42)
        original_step = simulator.step
        def transient_near_obstacle(*args, **kwargs):
            receipt = original_step(*args, **kwargs)
            receipt['min_sampled_safety_clearance'] = .125
            return receipt
        simulator.step = transient_near_obstacle
        sequence = {'segments': [{'name': 'hover', 'steps': 1, 'action': [0, 0, 0, 'hover'],
                                 'format': 'gradnav_normalized'}]}
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp)/'temporary_cloud.ply'
            source.write_bytes(b'point cloud input snapshot')
            source.with_suffix('.json').write_text('{"method":"gsplat_depth"}\n')
            simulator.point_cloud_file = source
            output = Path(temp)/'recording'
            summary = record_sequence(simulator, sequence, output)
            source.unlink()
            source.with_suffix('.json').unlink()
            self.assertEqual((output/'obstacles.ply').read_bytes(), b'point cloud input snapshot')
            self.assertEqual(json.loads((output/'obstacles.json').read_text()), {'method': 'gsplat_depth'})
            self.assertEqual(json.loads((output/'profile.json').read_text()), simulator.config)
            self.assertEqual(json.loads((output/'sequence.json').read_text()), sequence)
            self.assertEqual(summary['run']['point_cloud_source'], 'gsplat_depth')
            self.assertEqual(summary['run']['point_cloud_points'], 42)
            self.assertEqual(summary['point_cloud']['file'], 'obstacles.ply')
            self.assertEqual(summary['minimum_sampled_safety_clearance_m'], .125)

    def test_collision_stops_sequence_and_preserves_frame_state_alignment(self):
        sequence = {'segments': [{'name': 'go', 'steps': 20, 'action': [0, .1, 0, 'hover'],
                                 'format': 'gradnav_normalized'}]}
        simulator = FakeSimulator(collision_step=2)
        with tempfile.TemporaryDirectory() as temp:
            summary = record_sequence(simulator, sequence, temp)
            self.assertEqual(summary['executed_steps'], 2)
            self.assertEqual(summary['frames'], 3)
            self.assertEqual(summary['status'], 'collision')
            self.assertAlmostEqual(summary['simulation_duration_seconds'], .075)
            self.assertTrue(summary['verification']['passed'])
            self.assertTrue(summary['collision'])
            receipts = [json.loads(line) for line in (Path(temp)/'states.jsonl').read_text().splitlines()]
            self.assertEqual([row['video_frame'] for row in receipts], [0, 1, 2])
            self.assertEqual([row['step_id'] for row in receipts], [0, 1, 2])
            self.assertEqual(receipts[-1]['frame_id'], summary['final_frame_id'])
            self.assertEqual(receipts[-1]['state'], summary['final_state'])
            with (Path(temp)/'trajectory.csv').open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 3)
            self.assertEqual(rows[-1]['collision'], 'True')
            self.assertTrue(all(call[2] == 1 for call in simulator.calls))
            with self.assertRaises(ValueError):
                record_sequence(FakeSimulator(), sequence, temp)

    def test_encoder_decodes_exact_frame_count_and_rejects_wrong_shape(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/'test.mp4'
            writer = FpvVideoWriter(path, 16, 12)
            try:
                for color in [0., .5, 1.]:
                    writer.write(np.full((12, 16, 3), color))
                with self.assertRaises(ValueError):
                    writer.write(np.zeros((12, 15, 3)))
            finally:
                writer.close()
            self.assertEqual(verify_video(path, 3)['decoded_frames'], 3)
            with self.assertRaises(RuntimeError):
                verify_video(path, 4)
            with self.assertRaises(ValueError):
                FpvVideoWriter(Path(temp)/'odd.mp4', 15, 12)

    def test_requested_step_limit_at_segment_boundary_is_reported(self):
        segment = {'name': 'hover', 'steps': 1, 'action': [0, 0, 0, 'hover'],
                   'format': 'gradnav_normalized'}
        sequence = {'segments': [segment, segment]}
        with tempfile.TemporaryDirectory() as temp:
            summary = record_sequence(FakeSimulator(), sequence, temp, limit_steps=1)
            self.assertEqual(summary['status'], 'step_limit')
            self.assertEqual(summary['executed_steps'], 1)
            self.assertEqual(summary['frames'], 2)


if __name__ == '__main__':
    unittest.main()

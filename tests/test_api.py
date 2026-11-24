"""CPU end-to-end action/receipt/frame tests for the serialized service."""
import io
from pathlib import Path
import sys
import threading
import time
import unittest

import numpy as np
from PIL import Image
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import SimulatorService, create_app
from engine import PolicyFlightSimulator
from helpers import FakeScene


class ApiChecks(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.services = []

    def tearDown(self):
        for service in self.services:
            service.close()

    def create(self, factory=None, wait=True):
        if factory is None:
            factory = lambda: PolicyFlightSimulator('unused.ply', resolution=.1, no_collision=True,
                                                     scene_factory=FakeScene, device='cpu')
        service = SimulatorService(factory)
        self.services.append(service)
        client = create_app(service).test_client()
        if wait:
            deadline = time.monotonic() + 5
            while not service.snapshot().get('ready'):
                state = service.snapshot()
                self.assertNotIn('error', state, state)
                self.assertLess(time.monotonic(), deadline, 'CPU scene startup timed out')
                time.sleep(.005)
        return service, client

    @staticmethod
    def hover_payload(state, request_id='step-1', **extra):
        parameters = state['parameters']
        hover = parameters['mass'] * 9.81 / parameters['max_thrust']
        return {'request_id': request_id, 'action': [0, 0, 0, 4 * hover - 1],
                'format': 'gradnav_normalized', **extra}

    def test_loading_ready_and_no_autonomous_ticking(self):
        gate = threading.Event()
        def factory():
            gate.wait(timeout=5)
            return PolicyFlightSimulator('unused.ply', resolution=.1, no_collision=True,
                                         scene_factory=FakeScene, device='cpu')
        service, client = self.create(factory, wait=False)
        try:
            loading = client.get('/api/state').json
            self.assertFalse(loading['ready']); self.assertTrue(loading['loading'])
            self.assertEqual(client.post('/api/step', json={'request_id': 'early', 'action': [0, 0, 0, 0]}).status_code, 503)
        finally:
            gate.set()
        deadline = time.monotonic() + 5
        while not service.snapshot().get('ready'):
            self.assertLess(time.monotonic(), deadline); time.sleep(.005)
        before = client.get('/api/state').json
        time.sleep(.08)
        for _ in range(4):
            self.assertEqual(client.get('/api/frame/fpv').status_code, 200)
        after = client.get('/api/state').json
        self.assertEqual(before, after)
        self.assertEqual(after['sim_time'], 0)
        self.assertEqual(after['step_id'], 0)

    def test_one_step_returns_matching_state_observation_and_image(self):
        _, client = self.create()
        initial = client.get('/api/state').json
        result = client.post('/api/step', json=self.hover_payload(initial, expected_frame_id=initial['frame_id']))
        self.assertEqual(result.status_code, 200, result.json)
        receipt = result.json
        self.assertAlmostEqual(receipt['dt_actual'], .05)
        self.assertAlmostEqual(receipt['sim_time'], .05)
        self.assertEqual(receipt['step_id'], 1)
        self.assertEqual(receipt['substeps_executed'], 10)
        self.assertEqual(receipt['frame_id'], initial['frame_id'] + 1)
        self.assertEqual(receipt['observation']['step_id'], receipt['step_id'])
        self.assertEqual(receipt['observation']['frame_id'], receipt['frame_id'])
        self.assertEqual(receipt['image']['frame_id'], receipt['frame_id'])
        self.assertFalse(receipt['observation']['policy_ready'])
        self.assertIsNone(receipt['observation']['policy_observation'])
        np.testing.assert_allclose(receipt['state']['position'], initial['state']['position'], atol=1e-10)
        frame = client.get(receipt['image']['url'])
        self.assertEqual(frame.status_code, 200)
        self.assertEqual(int(frame.headers['X-Step-ID']), receipt['step_id'])
        self.assertEqual(int(frame.headers['X-Frame-ID']), receipt['frame_id'])
        self.assertAlmostEqual(float(frame.headers['X-Sim-Time']), receipt['sim_time'])

    def test_retried_request_is_idempotent_and_changed_payload_conflicts(self):
        _, client = self.create()
        initial = client.get('/api/state').json
        payload = self.hover_payload(initial, expected_frame_id=initial['frame_id'])
        first = client.post('/api/step', json=payload)
        second = client.post('/api/step', json=payload)
        self.assertEqual(first.status_code, 200, first.json)
        self.assertEqual(first.json, second.json)
        self.assertEqual(client.get('/api/state').json['step_id'], 1)
        changed = {**payload, 'action': [0, 0, .1, payload['action'][3]]}
        self.assertEqual(client.post('/api/step', json=changed).status_code, 409)
        self.assertEqual(client.get('/api/state').json['step_id'], 1)

    def test_stale_frame_rejected_before_state_advances(self):
        _, client = self.create()
        initial = client.get('/api/state').json
        self.assertEqual(client.post('/api/step', json=self.hover_payload(initial)).status_code, 200)
        stale = self.hover_payload(initial, request_id='stale', expected_frame_id=initial['frame_id'])
        self.assertEqual(client.post('/api/step', json=stale).status_code, 409)
        self.assertEqual(client.get('/api/state').json['step_id'], 1)
        fresh = client.get('/api/state').json
        stale['expected_frame_id'] = fresh['frame_id']
        self.assertEqual(client.post('/api/step', json=stale).status_code, 200)
        self.assertEqual(client.get('/api/state').json['step_id'], 2)

    def test_numbered_jpeg_and_float_frames_are_immutable(self):
        _, client = self.create()
        initial = client.get('/api/state').json
        jpeg_response = client.get(initial['image']['url'])
        float_response = client.get(initial['image']['float_rgb_url'])
        first_rgb = np.load(io.BytesIO(float_response.data), allow_pickle=False)
        self.assertEqual(first_rgb.dtype, np.float32)
        self.assertEqual(first_rgb.shape, (36, 64, 3))
        self.assertEqual(Image.open(io.BytesIO(jpeg_response.data)).size, (64, 36))
        hover = initial['parameters']['mass'] * 9.81 / initial['parameters']['max_thrust']
        moved = client.post('/api/step', json={'request_id': 'move', 'action': [0, .3, 0, hover],
                                                'format': 'body_rates_thrust', 'repeat': 4})
        self.assertEqual(moved.status_code, 200, moved.json)
        self.assertEqual(client.get(initial['image']['url']).data, jpeg_response.data)
        self.assertEqual(client.get(initial['image']['float_rgb_url']).data, float_response.data)
        current_rgb = np.load(io.BytesIO(client.get(moved.json['image']['float_rgb_url']).data), allow_pickle=False)
        self.assertGreater(np.max(np.abs(current_rgb - first_rgb)), 0.)
        self.assertEqual(client.get('/api/frame/fpv?frame_id=9999').status_code, 404)
        self.assertEqual(client.get('/api/frame/fpv?frame_id=bad').status_code, 400)
        self.assertEqual(client.get('/api/frame/fpv?format=raw').status_code, 400)

    def test_reset_has_new_episode_and_frame_but_zero_physical_time(self):
        _, client = self.create()
        initial = client.get('/api/state').json
        client.post('/api/step', json=self.hover_payload(initial))
        before = client.get('/api/state').json
        payload = {'request_id': 'reset-1', 'seed': 123, 'spawn': [0, 1, 2], 'yaw': 90,
                   'expected_frame_id': before['frame_id']}
        result = client.post('/api/reset', json=payload)
        self.assertEqual(result.status_code, 200, result.json)
        reset = result.json
        self.assertNotEqual(reset['episode_id'], initial['episode_id'])
        self.assertGreater(reset['frame_id'], before['frame_id'])
        self.assertEqual(reset['step_id'], 0); self.assertEqual(reset['sim_time'], 0.)
        self.assertEqual(reset['seed'], 123)
        np.testing.assert_allclose(reset['state']['position'], [0, 1, 2])
        np.testing.assert_allclose(reset['state']['velocity'], [0, 0, 0])
        self.assertEqual(client.post('/api/reset', json=payload).json, reset)

    def test_malformed_requests_are_rejected_without_advancing(self):
        _, client = self.create()
        invalid = [None, [], {}, {'request_id': 'missing-action'},
                   {'request_id': 'shape', 'action': [0, 0]},
                   {'request_id': 'none', 'action': None},
                   {'request_id': 'nan', 'action': [0, 0, 0, float('nan')]},
                   {'request_id': 'repeat', 'action': [0, 0, 0, 0], 'repeat': True},
                   {'request_id': 'format', 'action': [0, 0, 0, 0], 'format': 'keyboard'},
                   {'request_id': 'unknown', 'action': [0, 0, 0, 0], 'unknown': 3},
                   {'request_id': 'expected', 'action': [0, 0, 0, 0], 'expected_frame_id': True}]
        for payload in invalid:
            response = client.post('/api/step', json=payload)
            self.assertEqual(response.status_code, 400, (payload, response.json))
        after = client.get('/api/state').json
        self.assertTrue(after['ready']); self.assertEqual(after['step_id'], 0)

    def test_bad_reset_does_not_partially_change_next_reset_defaults(self):
        _, client = self.create()
        before = client.get('/api/state').json
        bad = {'request_id': 'bad-reset', 'seed': 77, 'spawn': [9, 8, 7], 'yaw': 'invalid'}
        self.assertEqual(client.post('/api/reset', json=bad).status_code, 400)
        after = client.get('/api/state').json
        self.assertEqual(before, after)
        reset = client.post('/api/reset', json={'request_id': 'valid-reset'})
        self.assertEqual(reset.status_code, 200, reset.json)
        self.assertEqual(reset.json['seed'], before['seed'])
        self.assertEqual(reset.json['state']['position'], before['state']['position'])

    def test_post_integration_encoder_failure_blocks_ambiguous_retry(self):
        instances = []
        def factory():
            sim = PolicyFlightSimulator('unused.ply', resolution=.1, no_collision=True,
                                         scene_factory=FakeScene, device='cpu')
            def invalid_encoder(*args, **kwargs):
                raise ValueError('context_encoder TorchScript must return tensor [1,24]')
            sim.observer.build = invalid_encoder
            instances.append(sim)
            return sim
        service, client = self.create(factory)
        initial = client.get('/api/state').json
        payload = self.hover_payload(initial, request_id='uncertain-action')
        result = client.post('/api/step', json=payload)
        self.assertIn(result.status_code, (400, 500), result.json)
        deadline = time.monotonic() + 5
        while not service.stop.is_set():
            self.assertLess(time.monotonic(), deadline); time.sleep(.005)
        self.assertFalse(client.get('/api/state').json['ready'])
        self.assertEqual(client.get('/api/state').json['uncertain_request_id'], 'uncertain-action')
        self.assertEqual(instances[0].step_id, 1)
        self.assertEqual(client.post('/api/step', json=payload).status_code, 503)
        self.assertEqual(instances[0].step_id, 1)


if __name__ == '__main__':
    unittest.main()

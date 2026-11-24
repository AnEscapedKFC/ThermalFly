import copy
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(Path(__file__).resolve().parent))
from engine import EpisodeFinished,PolicyFlightSimulator
from pointcloud import open3d_module
from helpers import FakeScene


class EngineChecks(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.temporary=tempfile.TemporaryDirectory();self.directory=Path(self.temporary.name)
        self.engines=[]

    def tearDown(self):
        for engine in self.engines:engine.close()
        self.temporary.cleanup()

    def engine(self,**options):
        kwargs=dict(scene_path=self.directory/'scene.ply',scene_factory=FakeScene,
                    device='cpu',no_collision=True,resolution=.1,spawn=(0,0,1.25))
        kwargs.update(options)
        engine=PolicyFlightSimulator(**kwargs);self.engines.append(engine)
        return engine

    def test_moving_spawn_is_present_before_first_action(self):
        velocity=[.32,-.07,.01]
        engine=self.engine(spawn_velocity_w_mps=velocity)
        start=engine.last_receipt
        self.assertEqual(start['frame_id'],0)
        self.assertEqual(start['dt_actual'],0.)
        np.testing.assert_allclose(start['state']['velocity'],velocity)
        moved=engine.step([0,0,0,engine.adapter.hover_thrust], format='body_rates_thrust')
        self.assertGreater(moved['state']['position'][0],start['state']['position'][0])
        np.testing.assert_allclose(engine.reset()['state']['velocity'],velocity)

    def cloud(self,points):
        o3d=open3d_module();cloud=o3d.geometry.PointCloud()
        cloud.points=o3d.utility.Vector3dVector(np.asarray(points,dtype=float))
        path=self.directory/(str(len(list(self.directory.glob('*.ply'))))+'.ply')
        self.assertTrue(o3d.io.write_point_cloud(str(path),cloud))
        return path

    def hover(self,engine):
        return [0,0,0,engine.parameters['mass']*9.81/engine.parameters['max_thrust']]

    def test_one_action_is_exactly_50ms_and_receipt_matches_rendered_pose(self):
        engine=self.engine();initial=copy.deepcopy(engine.last_receipt)
        receipt=engine.step([0,.2,0,self.hover(engine)[3]],'body_rates_thrust')
        self.assertEqual(receipt['step_id'],1);self.assertEqual(receipt['frame_id'],1)
        self.assertAlmostEqual(receipt['sim_time'],.05,places=12)
        self.assertAlmostEqual(receipt['dt_actual'],.05,places=12)
        self.assertEqual(receipt['substeps_executed'],10)
        self.assertEqual(receipt['image']['step_id'],receipt['step_id'])
        self.assertEqual(receipt['image']['frame_id'],receipt['frame_id'])
        self.assertEqual(receipt['observation']['step_id'],receipt['step_id'])
        self.assertEqual(receipt['observation']['frame_id'],receipt['frame_id'])
        np.testing.assert_allclose(engine.scene.cameras[-1],receipt['image']['camera_to_world'])
        self.assertEqual(initial['step_id'],0)
        self.assertNotEqual(receipt['state']['position'],initial['state']['position'])

    def test_no_wall_clock_progress_between_actions_and_reads_are_frozen(self):
        engine=self.engine();engine.step(self.hover(engine),'body_rates_thrust')
        before=engine.state.clone();receipt=copy.deepcopy(engine.last_receipt)
        renders=engine.scene.render_calls
        time.sleep(.02)
        for _ in range(3):self.assertEqual(engine.last_receipt,receipt)
        torch.testing.assert_close(engine.state,before,rtol=0,atol=0)
        self.assertEqual(engine.scene.render_calls,renders)
        self.assertEqual(engine.step_id,1)

    def test_physical_hover_bypasses_keyboard_and_preserves_velocity_state(self):
        engine=self.engine();hover=self.hover(engine)
        receipt=engine.step(hover,'body_rates_thrust',repeat=20)
        np.testing.assert_allclose(receipt['state']['position'],[0,0,1.25],atol=1e-12)
        np.testing.assert_allclose(receipt['applied_control'],hover)
        self.assertFalse(receipt['action_processing']['lpf_applied'])
        self.assertFalse(receipt['action_processing']['delay_applied'])
        engine.state[3]=.4
        receipt=engine.step(hover,'body_rates_thrust')
        self.assertGreater(receipt['state']['velocity'][0],.39)
        self.assertGreater(receipt['state']['position'][0],0.)

    def test_repeat_equals_individual_ticks_including_adapter_and_observation_history(self):
        repeated=self.engine();sequential=self.engine()
        raw=[.1,-.1,.05,self.hover(repeated)[3]/.25-1]
        receipt=repeated.step(raw,repeat=5)
        for _ in range(5):single=sequential.step(raw)
        torch.testing.assert_close(repeated.state,sequential.state,rtol=0,atol=0)
        self.assertEqual(receipt['applied_control'],single['applied_control'])
        self.assertEqual(receipt['previous_control'],single['previous_control'])
        self.assertEqual(receipt['observation']['base_observation'],single['observation']['base_observation'])
        self.assertEqual(receipt['observation']['history_for_next'],single['observation']['history_for_next'])
        self.assertAlmostEqual(receipt['dt_actual'],.25,places=12)
        self.assertEqual(receipt['ticks_executed'],5)
        self.assertEqual(receipt['substeps_executed'],50)
        self.assertEqual(receipt['frame_id'],5)
        self.assertEqual(repeated.scene.render_calls,6)

    def test_real_open3d_thin_wall_stops_high_speed_motion_and_latches(self):
        path=self.cloud([[.3,0,1.25]])
        engine=self.engine(point_cloud=path,no_collision=False,drone_size=(.2,.2,.2),collision_margin=0.,
                           config={'limits':{'speed':1000.}})
        engine.state[3]=100.
        receipt=engine.step(self.hover(engine),'body_rates_thrust')
        self.assertTrue(receipt['terminated']);self.assertEqual(receipt['reason'],'collision')
        self.assertEqual(receipt['collision']['shape'],'box')
        self.assertEqual(receipt['collision']['event']['reason'],'obstacle')
        self.assertFalse(receipt['collision']['in_collision'])
        self.assertLess(receipt['state']['position'][0],.2)
        self.assertGreater(receipt['state']['position'][0],.199)
        self.assertLess(receipt['dt_actual'],.005)
        np.testing.assert_allclose(receipt['state']['velocity'],0.)
        np.testing.assert_allclose(receipt['state']['angular_velocity'],0.)
        with self.assertRaises(EpisodeFinished):engine.step(self.hover(engine),'body_rates_thrust')
        reset=engine.reset()
        self.assertFalse(reset['terminated']);self.assertIsNone(reset['collision']['event'])
        self.assertEqual(engine.step(self.hover(engine),'body_rates_thrust')['step_id'],1)

    def test_repeat_reports_minimum_safety_clearance_across_whole_action(self):
        path=self.cloud([[-.4,0,1.25]])
        engine=self.engine(point_cloud=path,no_collision=False,
                           drone_size=(.2,.2,.2),collision_margin=0.)
        engine.state[3]=.4
        result=engine.step(self.hover(engine),'body_rates_thrust',repeat=4)
        self.assertAlmostEqual(result['min_sampled_safety_clearance'],.3,places=10)
        self.assertGreater(result['collision']['safety_clearance'],.37)

    def test_spawn_overlap_prevents_motion_and_reposition_reset_recovers(self):
        path=self.cloud([[0,0,1.25]])
        engine=self.engine(point_cloud=path,no_collision=False)
        self.assertTrue(engine.terminated);self.assertEqual(engine.reason,'spawn_overlap')
        with self.assertRaises(EpisodeFinished):engine.step(self.hover(engine),'body_rates_thrust')
        receipt=engine.reset(spawn=(-1,0,1.25))
        self.assertFalse(receipt['terminated']);self.assertEqual(receipt['step_id'],0)
        self.assertFalse(engine.step(self.hover(engine),'body_rates_thrust')['terminated'])

    def test_explicit_no_collision_keeps_motion_unconstrained(self):
        path=self.cloud([[0,0,1.25]])
        engine=self.engine(point_cloud=path,no_collision=True)
        self.assertIsNone(engine.obstacles)
        receipt=engine.step(self.hover(engine),'body_rates_thrust')
        self.assertFalse(receipt['collision']['enabled']);self.assertFalse(receipt['terminated'])

    def test_training_seed_reproduces_dynamics_randomization_and_observation_noise(self):
        first=self.engine(config='training',seed=4)
        second=self.engine(config='training',seed=4)
        self.assertEqual(first.parameters,second.parameters)
        self.assertEqual(first.last_receipt['observation'],second.last_receipt['observation'])
        raw=[0,0,0,self.hover(first)[3]/.25-1]
        a=first.step(raw,repeat=3);b=second.step(raw,repeat=3)
        self.assertEqual(a['state'],b['state']);self.assertEqual(a['observation'],b['observation'])
        first.reset(seed=13)
        fresh=self.engine(config='training',seed=13)
        self.assertEqual(first.parameters,fresh.parameters)
        # IDs remain globally monotonic across reset; the actual noisy vector
        # must still reproduce the fresh episode with the selected seed.
        np.testing.assert_array_equal(first.last_receipt['observation']['base_observation'],
                                      fresh.last_receipt['observation']['base_observation'])
        raw=[0,0,0,self.hover(first)[3]/.25-1]
        a=first.step(raw);b=fresh.step(raw)
        self.assertEqual(a['state'],b['state'])
        np.testing.assert_array_equal(a['observation']['base_observation'],b['observation']['base_observation'])

    def test_invalid_requests_do_not_advance_or_mutate_adapter(self):
        engine=self.engine();before=copy.deepcopy(engine.last_receipt);state=engine.state.clone()
        invalid=[([0,0,0],'gradnav_normalized',1),([0,0,np.nan,0],'gradnav_normalized',1),
                 ([0,0,0,0],'invalid',1),([0,0,0,0],'gradnav_normalized',0),
                 ([0,0,0,0],'gradnav_normalized',1.5),([0,0,0,0],'gradnav_normalized',True)]
        for action,format,repeat in invalid:
            with self.assertRaises(ValueError):engine.step(action,format,repeat)
            self.assertEqual(engine.last_receipt,before)
            self.assertEqual(engine.adapter.action_index,0)
            torch.testing.assert_close(engine.state,state,rtol=0,atol=0)

    def test_speed_and_tilt_limits_terminate_on_substeps(self):
        speed=self.engine(config={'limits':{'speed':.1}})
        speed.state[3]=.2
        result=speed.step(self.hover(speed),'body_rates_thrust')
        self.assertTrue(result['terminated']);self.assertEqual(result['reason'],'speed_limit')
        self.assertAlmostEqual(result['dt_actual'],.005)
        tilt=self.engine(config={'limits':{'tilt_deg':5.}})
        tilt.state[6:10]=tilt.state.new_tensor([np.cos(np.deg2rad(10)),np.sin(np.deg2rad(10)),0,0])
        result=tilt.step(self.hover(tilt),'body_rates_thrust')
        self.assertTrue(result['terminated']);self.assertEqual(result['reason'],'tilt_limit')
        self.assertAlmostEqual(result['dt_actual'],.005)

    def test_nonfinite_candidate_returns_last_valid_state_and_terminal_reason(self):
        engine=self.engine();before=engine.state.clone()
        def invalid_step(state,control,callback):
            candidate=state.clone();candidate[0]=float('nan')
            return callback(state,candidate,.005)
        with patch.object(engine.sim,'step',side_effect=invalid_step):
            result=engine.step(self.hover(engine),'body_rates_thrust')
        self.assertTrue(result['terminated']);self.assertEqual(result['reason'],'nonfinite_state')
        self.assertAlmostEqual(result['dt_actual'],0.)
        torch.testing.assert_close(engine.state,before,rtol=0,atol=0)

    def test_max_steps_truncates_and_reset_frame_ids_remain_monotonic(self):
        engine=self.engine(config={'limits':{'max_steps':2}})
        result=engine.step(self.hover(engine),'body_rates_thrust',repeat=5)
        self.assertTrue(result['truncated']);self.assertEqual(result['reason'],'max_steps')
        self.assertEqual(result['step_id'],2);self.assertEqual(result['ticks_executed'],2)
        old_frame=result['frame_id'];reset=engine.reset()
        self.assertGreater(reset['frame_id'],old_frame)
        self.assertEqual(reset['step_id'],0)
        self.assertFalse(reset['truncated'])


if __name__=='__main__':unittest.main()

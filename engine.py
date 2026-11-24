"""One action -> one frozen, matching state/observation/frame receipt.

The source simulator's pointcloud.py is copied unchanged. Its OBB sweep and
contact stop are reused here with a policy input instead of keyboard control.
"""
from copy import deepcopy
from pathlib import Path
import tempfile
import uuid

import numpy as np
from scipy.spatial.transform import Rotation
import torch

from action_adapter import ActionAdapter,make_dynamics_parameters
from cameras import fpv_camera
from config import load_config
from dynamics import CONTROL_DT,QuadrotorDynamics,initial_state,quaternion_rotation
from export_pointcloud import export_ply_depth
from observations import ObservationBuilder
from pointcloud import ObstacleCloud,DEFAULT_DRONE_SIZE,interpolate_quaternion
from scene import GaussianScene


class EpisodeFinished(RuntimeError):
    pass


class PolicyFlightSimulator:
    """Single-owner engine. HTTP requests are serialized by app.py's worker.

    Inference waits never advance physical time. There is no keyboard
    controller, automatic position hold, or implicit reset after termination.
    """
    def __init__(self,scene_path,point_cloud=None,config=None,seed=0,
                 spawn=(-6.,0.,1.25),yaw=0.,resolution=.75,
                 drone_size=DEFAULT_DRONE_SIZE,collision_margin=.05,
                 no_collision=False,device='cuda',up_axis='z',scale=1.,
                 cloud_views=72,cloud_voxel=.04,point_cloud_transform=None,
                 observation_assets=None,scene_factory=GaussianScene,*,spawn_velocity_w_mps=(0.,0.,0.)):
        if not np.isfinite(resolution) or not .1<=resolution<=1:
            raise ValueError('resolution must be in [.1, 1]')
        self.config=load_config(config)
        self.seed=int(seed);self.spawn_position=self._position(spawn);self.spawn_yaw=self._yaw(yaw)
        self.spawn_velocity_w_mps=self._position(spawn_velocity_w_mps)
        self.limits=deepcopy(self.config['limits'])
        for key in ('speed','body_rate','tilt_deg','max_steps'):
            if not np.isfinite(self.limits[key]) or self.limits[key]<=0:
                raise ValueError(f'Invalid limit: {key}')
        if self.limits['tilt_deg']>180:raise ValueError('tilt_deg must not exceed 180')
        for key in ('position_min','position_max'):
            if key in self.limits:self.limits[key]=self._position(self.limits[key])
        if ('position_min' in self.limits and 'position_max' in self.limits and
                np.any(self.limits['position_min']>=self.limits['position_max'])):
            raise ValueError('position_min must be below position_max')
        self.temp=tempfile.TemporaryDirectory(prefix='gs-gradnav-')
        self.obstacles=None;self.point_cloud_file=None
        self.frame_id=-1;self.last_receipt=None;self.closed=False
        try:
            self.scene=scene_factory(Path(scene_path).expanduser().resolve(),round(640*resolution),
                                     round(360*resolution),up_axis,scale,device=device)
            if not no_collision:
                if point_cloud is None:
                    self.point_cloud_file=Path(self.temp.name)/'obstacles.ply'
                    export_ply_depth(self.scene,self.point_cloud_file,views=cloud_views,
                                     voxel=cloud_voxel,seed=self.spawn_position)
                    source='gsplat_depth';point_cloud_transform=None
                else:
                    self.point_cloud_file=Path(point_cloud).expanduser().resolve();source='external'
                self.obstacles=ObstacleCloud(self.point_cloud_file,up_axis,scale,drone_size,
                                             collision_margin,point_cloud_transform)
                self.obstacles.source=source
                if (np.any(self.obstacles.points.max(axis=0)<self.scene.bounds[0]) or
                        np.any(self.obstacles.points.min(axis=0)>self.scene.bounds[1])):
                    raise ValueError('Point cloud and scene do not overlap; check coordinates and scale')
            self.observer=ObservationBuilder(self.config,seed=self.seed,device=str(self.scene.device),assets=observation_assets)
            self.reset()
        except BaseException:
            self.temp.cleanup();raise

    @staticmethod
    def _position(value):
        p=np.asarray(value,dtype=float)
        if p.shape!=(3,) or not np.isfinite(p).all():raise ValueError('Expected three finite coordinates')
        return p.copy()

    @staticmethod
    def _yaw(value):
        value=float(value)
        if not np.isfinite(value):raise ValueError('Yaw must be finite degrees')
        return value

    @torch.no_grad()
    def reset(self,seed=None,spawn=None,yaw=None):
        if self.closed:raise RuntimeError('Simulator is closed')
        new_seed=self.seed if seed is None else int(seed)
        new_spawn=self.spawn_position if spawn is None else self._position(spawn)
        new_yaw=self.spawn_yaw if yaw is None else self._yaw(yaw)
        parameters=make_dynamics_parameters(self.config,new_seed)
        self.seed=new_seed;self.spawn_position=new_spawn.copy();self.spawn_yaw=new_yaw
        self.parameters=parameters
        self.sim=QuadrotorDynamics(str(self.scene.device),self.parameters,self.seed)
        self.state=initial_state(self.scene.device,self.spawn_position,self.spawn_yaw,self.spawn_velocity_w_mps)
        hover=self.parameters['mass']*9.81/self.parameters['max_thrust']
        self.adapter=ActionAdapter(self.config,hover)
        self.observer.reset(seed=self.seed)
        self.episode_id=uuid.uuid4().hex;self.step_id=0;self.elapsed=0.
        self.terminated=False;self.truncated=False;self.reason=None;self.collision_event=None
        self._last_raw_action=None;self._last_action_format=None
        collision=self.collision_state()
        if collision.get('in_collision'):
            self._collision_stop('spawn_overlap',collision['nearest_point'])
        else:
            reason=self._health(self.state)
            if reason:self.terminated=True;self.reason=reason
        self._publish(0.,0,collision.get('safety_clearance'))
        return deepcopy(self.last_receipt)

    def collision_state(self):
        if self.obstacles is None:return {'enabled':False,'event':None}
        state=self.state.detach().cpu().numpy()
        return {'enabled':True,**self.obstacles.metadata(),
                **self.obstacles.distance(state[:3],state[6:10]),'event':self.collision_event}

    def _health(self,state):
        if not torch.isfinite(state).all():return 'nonfinite_state'
        if abs(float(state[6:10].norm())-1)>1e-5:return 'invalid_quaternion'
        if float(state[3:6].norm())>self.limits['speed']:return 'speed_limit'
        if float(state[10:13].norm())>self.limits['body_rate']:return 'body_rate_limit'
        cosine=float(quaternion_rotation(state[6:10])[2,2])
        if np.degrees(np.arccos(np.clip(cosine,-1,1)))>self.limits['tilt_deg']:return 'tilt_limit'
        p=state[:3].detach().cpu().numpy()
        if 'position_min' in self.limits and np.any(p<self.limits['position_min']):return 'out_of_bounds'
        if 'position_max' in self.limits and np.any(p>self.limits['position_max']):return 'out_of_bounds'
        return None

    def _collision_stop(self,reason,point,time=None,position=None):
        self.collision_event={'reason':reason,'point':point,
                              'position':self.state[:3].cpu().tolist() if position is None else position,
                              'time':self.elapsed if time is None else time}
        self.terminated=True;self.reason='collision' if reason=='obstacle' else reason

    @torch.no_grad()
    def step(self,action,format='gradnav_normalized',repeat=1):
        if self.closed:raise RuntimeError('Simulator is closed')
        if isinstance(repeat,bool) or not isinstance(repeat,(int,np.integer)) or not 1<=repeat<=200:
            raise ValueError('repeat must be an integer in [1, 200]')
        raw=np.asarray(action,dtype=float)
        if raw.shape!=(4,) or not np.isfinite(raw).all():raise ValueError('action must contain four finite numbers')
        if format not in ActionAdapter.FORMATS:raise ValueError('Unknown action format')
        if self.terminated or self.truncated:raise EpisodeFinished('Episode finished; call reset before another action')
        total_dt=0.;ticks=0;total_substeps=0;overall_minimum=None
        for _ in range(int(repeat)):
            control=self.adapter.adapt(raw,format)
            self._last_raw_action=raw.tolist();self._last_action_format=format
            advanced=0.;substeps=0
            initial_collision=self.collision_state()
            minimum=initial_collision.get('safety_clearance')

            def guard(before,candidate,dt):
                nonlocal advanced,substeps,minimum
                reason=self._health(candidate)
                # Keep a finite snapshot on numerical failure; callers receive
                # a terminal reason rather than NaN in a JSON response.
                if reason=='nonfinite_state' or reason=='invalid_quaternion':
                    self.terminated=True;self.reason=reason
                    return before.clone()
                if self.obstacles is not None:
                    start=before[:3].cpu().numpy();end=candidate[:3].cpu().numpy()
                    q0=before[6:10].cpu().numpy();q1=candidate[6:10].cpu().numpy()
                    hit=self.obstacles.sweep(start,end,q0,q1)
                    if hit is not None:
                        # Same first-contact backoff and stop response as the
                        # interactive simulator, including rotational motion.
                        angle=2*np.arccos(np.clip(abs(np.dot(q0,q1)),0.,1.))
                        length=float(np.linalg.norm(end-start))+self.obstacles.search_radius*angle
                        fraction=max(0.,hit['fraction']-1e-5/max(length,1e-12))
                        stopped=before.clone()
                        stopped[:3]=before[:3]+(candidate[:3]-before[:3])*fraction
                        stopped[6:10]=stopped.new_tensor(interpolate_quaternion(q0,q1,fraction))
                        stopped[3:6]=0;stopped[10:13]=0
                        advanced+=dt*fraction;substeps+=1
                        clearance=self.obstacles.distance(stopped[:3].cpu().numpy(),stopped[6:10].cpu().numpy())['safety_clearance']
                        minimum=min(minimum,clearance)
                        self._collision_stop('obstacle',hit['point'],self.elapsed+advanced,stopped[:3].cpu().tolist())
                        return stopped
                    clearance=self.obstacles.distance(end,q1)['safety_clearance']
                    minimum=min(minimum,clearance)
                advanced+=dt;substeps+=1
                if reason:
                    self.terminated=True;self.reason=reason
                    return candidate
                return None

            if initial_collision.get('in_collision'):
                self._collision_stop('spawn_overlap',initial_collision['nearest_point'])
            else:
                self.state=self.sim.step(self.state,self.state.new_tensor(control),guard)
            self.elapsed+=advanced;self.step_id+=1;ticks+=1;total_dt+=advanced;total_substeps+=substeps
            if minimum is not None:
                overall_minimum=minimum if overall_minimum is None else min(overall_minimum,minimum)
            if not self.terminated and self.step_id>=self.limits['max_steps']:
                self.truncated=True;self.reason='max_steps'
            self._publish(advanced,substeps,minimum)
            if self.terminated or self.truncated:break
        self.last_receipt['dt_actual']=float(total_dt)
        self.last_receipt['ticks_executed']=ticks
        self.last_receipt['substeps_executed']=total_substeps
        self.last_receipt['min_sampled_safety_clearance']=overall_minimum
        self.last_receipt['requested_repeat']=int(repeat)
        self.last_receipt['completed']=bool(ticks==repeat and np.isclose(total_dt,CONTROL_DT*repeat,rtol=0,atol=1e-9))
        return deepcopy(self.last_receipt)

    def _publish(self,dt,substeps,minimum):
        self.frame_id+=1
        state=self.state.detach().cpu().numpy().copy()
        camera=fpv_camera(state)
        self.last_rgb,self.last_depth,_=self.scene.render(camera)
        if not np.isfinite(self.last_rgb).all():raise RuntimeError('Renderer returned a nonfinite frame')
        observation=self.observer.build(state,self.adapter.current_control,self.adapter.previous_control,
                                        self.last_rgb,self.frame_id,self.step_id)
        q=state[6:10]
        self.last_receipt={
            'episode_id':self.episode_id,'step_id':self.step_id,'frame_id':self.frame_id,
            'sim_time':float(self.elapsed),'dt_actual':float(dt),'control_dt':CONTROL_DT,
            'substeps_executed':substeps,'ticks_executed':0 if self.step_id==0 else 1,
            'completed':not (self.terminated or self.truncated),
            'terminated':self.terminated,'truncated':self.truncated,'reason':self.reason,
            'raw_action':self._last_raw_action,'action_format':self._last_action_format,
            'applied_control':self.adapter.current_control.tolist(),
            'previous_control':self.adapter.previous_control.tolist(),
            'action_processing':deepcopy(self.adapter.last_info),
            'state':{'position':state[:3].tolist(),'velocity':state[3:6].tolist(),
                     'quaternion_wxyz':q.tolist(),'angular_velocity':state[10:13].tolist(),
                     'linear_acceleration':self.sim.last_linear_acceleration.cpu().tolist(),
                     'angular_acceleration':self.sim.last_angular_acceleration.cpu().tolist(),
                     'euler_deg':Rotation.from_quat(q[[1,2,3,0]]).as_euler('xyz',degrees=True).tolist(),
                     'speed':float(np.linalg.norm(state[3:6]))},
            'frames':{'world':'Z-up meters','body':'X forward, Y left, Z up',
                      'velocity':'world m/s','angular_velocity':'body rad/s','quaternion':'wxyz body-to-world'},
            'collision':self.collision_state(),'min_sampled_safety_clearance':minimum,
            'observation':observation,'parameters':deepcopy(self.parameters),'profile':self.config['name'],
            'seed':self.seed,'scene':self.scene.metadata(),
            'image':{'width':self.scene.width,'height':self.scene.height,'frame_id':self.frame_id,
                     'step_id':self.step_id,'camera_to_world':camera.numpy().tolist(),
                     'intrinsics':self.scene.K[0].cpu().tolist()},
        }

    def close(self):
        if not self.closed:
            self.temp.cleanup();self.closed=True

    def __enter__(self):return self
    def __exit__(self,*_):self.close()

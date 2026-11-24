"""Fly a reference with the existing simulator and publish an RGB-T expert episode."""
import argparse
import hashlib
import json
import os
import sys
import tempfile
from contextlib import ExitStack
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.interpolate import PchipInterpolator
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cameras import fpv_camera
from dynamics import CONTROL_DT
from engine import PolicyFlightSimulator
from openfly_rgb.flight import KnownSpace as RoomSpace, RigScene as RoomRig, controller
from fly import commands
from record_sequence import FpvVideoWriter
from scenes.corridor_native.room import KnownSpace as CorridorSpace, RigScene as CorridorRig

SCENES = {'room': 'room65_selected_anchored', 'corridor': 'corridor_native'}
DATASET = ROOT / 'outputs/expert_rgbt_20261123'


def dump(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def digest(path):
    with path.open('rb') as stream:
        h = hashlib.sha256()
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def image8(arr):
    return np.rint(255 * np.clip(arr, 0., 1.)).astype(np.uint8)


def local_delta(begin, end):
    diff = end[:3] - begin[:3]
    yaw = np.deg2rad(begin[3])
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([c*diff[0] + s*diff[1], -s*diff[0] + c*diff[1], diff[2],
                     (end[3]-begin[3]+180) % 360 - 180])


class Reference:
    def __init__(self, route):
        self.knots = np.asarray(route['knots'], dtype=float)
        k = self.knots
        if (k.ndim != 2 or k.shape[1] != 5 or len(k) < 3 or not np.isfinite(k).all()
                or k[0, 0] != 0 or np.any(np.diff(k[:, 0]) <= 0)
                or not np.allclose(k[-1, 1:], k[-2, 1:], atol=1e-7)
                or k[-1, 0] - k[-2, 0] < .5):
            raise ValueError('Provide ascending time/XYZ/yaw knots with terminal hover.')
        self.ticks = round(k[-1, 0] / CONTROL_DT)
        if abs(self.ticks*CONTROL_DT-k[-1,0])>1e-5:
            raise ValueError('Final time must be on the controller clock.')
        self.stop_tick = int(np.ceil(k[-2, 0] / CONTROL_DT - 1e-8))
        self.mode = route.get('interpolation', 'quintic')
        if self.mode not in ('pchip','quintic'):
            raise ValueError(self.mode)
        if self.mode == 'pchip':
            self.poly = PchipInterpolator(k[:,0], k[:,1:], axis=0)
            self.velocity, self.acceleration = self.poly.derivative(), self.poly.derivative(2)
        self.clock = np.arange(self.ticks + 1)*CONTROL_DT
        self.poses = np.array([self.sample(k,t)[0] for t in self.clock])

    def sample(self, k, t):
        if self.mode == 'quintic':
            return commands(k,t)
        j = np.clip(np.searchsorted(k[:,0],t,side='right')-1,0,len(k)-2)
        return self.poly(t),self.velocity(t),self.acceleration(t),int(j)

    def segment_ends(self, stop, subgoals):
        # Cut on path travel, rotation, phase changes or elapsed time. No hold is
        # inserted between these training decisions in the continuous flight.
        marks = {int(np.clip(round(g['at_s']/CONTROL_DT),0,stop)) for g in subgoals}
        bounds = [0]
        travel = vertical = turn = 0.
        for i in range(1,stop+1):
            d = self.poses[i] - self.poses[i-1]
            travel += np.linalg.norm(d[:2])
            vertical += abs(d[2])
            turn += abs(d[3])
            if travel >= .50 or vertical >= .17 or turn >= 22. or i-bounds[-1] >= 40 or i in marks or i==stop:
                bounds.append(i)
                travel = vertical = turn = 0.
        return bounds


def rig_factory(base, route):
    target_keys = np.array(route.get('camera_targets', []))
    pitch_keys = np.array(route.get('camera_pitch_keys', []))
    if not target_keys.size and not pitch_keys.size:
        return base
    if target_keys.size and pitch_keys.size:
        raise ValueError('Choose one pitch control mode.')
    class DirectedRig(base):
        tick = 0
        def render(self, camera):
            mount = np.asarray(camera).copy()
            t = self.tick*CONTROL_DT
            self.tick += 1
            if pitch_keys.size:
                pitch = np.interp(t, pitch_keys[:,0],pitch_keys[:,1])
            else:
                focus = np.array([np.interp(t,target_keys[:,0],target_keys[:,j]) for j in (1,2,3)])
                difference = focus-mount[:3,3]
                elevation = np.rad2deg(np.arctan2(difference[2],np.linalg.norm(difference[:2])))
                base_elevation = np.rad2deg(np.arcsin(np.clip(-mount[2,2],-1,1)))
                pitch = np.clip(elevation-base_elevation,-82.,70.)
            self.pitch = float(pitch)
            mount[:3,:3] = mount[:3,:3] @ Rotation.from_euler('x',pitch,degrees=True).as_matrix()
            return super().render(mount)
    return DirectedRig


def publish_calibration(dataset, scene_id, scene_dir, rig):
    dest = dataset / 'scenes' / scene_id
    dest.mkdir(parents=True,exist_ok=True)
    out = dest / 'calibration.json'
    frame = json.loads((scene_dir/'inputs/frame.json').read_text())
    body_to_gl = fpv_camera(np.array([0.,0.,0.,0.,0.,0.,1.,0.,0.,0.])).numpy()
    paths = ['models/rgb.ply','models/thermal.ply','collision/rgb_surface_m.ply','collision/free_volume.npz',
             'inputs/manifest.json','inputs/frame.json','inputs/rig.json']
    cal = dict(scene_version=scene_id, wh=[rig.width,rig.height], rgb_K=rig.ks['rgb'].tolist(),
               thermal_K=rig.ks['thermal'].tolist(), distortion='pinhole render; no image distortion',
               convention='native w2c matrices are OpenCV (+x right, +y down, +z forward); rigid body FLU; simulator Z-up',
               body_from_rgb_opengl_at_zero_pitch_m=body_to_gl.tolist(), optical_axes='OpenGL +x right, +y up, -z forward',
               thermal_from_rgb_native=rig.rig.tolist(), rig_translation_unit='native reconstruction unit',
               native_to_sim='p_sim_m = meters_per_source_unit * rotation @ (p_native - origin)',
               source_to_sim=frame, per_frame_extrinsics='rgb_w2c_native and thermal_w2c_native include any commanded gimbal pitch',
               assets={p:{'path':str((scene_dir/p).resolve()),'sha256':digest(scene_dir/p)} for p in paths})
    with tempfile.TemporaryDirectory(dir=dest) as tmp:
        temp=Path(tmp)/'calibration.json'
        dump(temp,cal)
        try:
            os.link(temp,out)
        except FileExistsError:
            if json.loads(out.read_text())!=cal:
                raise RuntimeError('Existing calibration differs; use a new scene_version or dataset.')


def make_decisions(out, route, ref, states, subgoals, stop):
    bounds = ref.segment_ends(stop, subgoals)
    def motion(begin,end):
        d = local_delta(begin,end)
        if np.linalg.norm(d[:2]) > .700001 or abs(d[2]) > .300001 or abs(d[3]) > 35.00001:
            raise ValueError(f'Untrackable expert target in {route["id"]}: {d}')
        return {'kind':'move','delta_flu_m_yaw_deg':d.tolist()}
    with (out/'decisions.jsonl').open('x') as stream:
        for j,idx in enumerate(bounds):
            chunk = []
            for a in range(3):
                h = j+a
                if h < len(bounds)-1:
                    start = states[idx] if a==0 else ref.poses[bounds[h]]
                    target_id = bounds[h+1]
                    target_time = ref.clock[target_id]
                    slot = motion(start,ref.poses[target_id])
                    slot.update(target_frame_id=target_id, duration_s=target_time-ref.clock[bounds[h]],
                                target_reference_speed_mps=float(np.linalg.norm(ref.sample(ref.knots,target_time)[1][:3])))
                    if 'camera_pitch_keys' in route:
                        keys = np.asarray(route['camera_pitch_keys'])
                        slot['target_rig_pitch_deg'] = float(np.interp(target_time,keys[:,0],keys[:,1]))
                    chunk.append(slot)
                else:
                    chunk.append({'kind':'stop' if h==len(bounds)-1 else 'pad','delta_flu_m_yaw_deg':None})
            end = bounds[j+1] if j+1 < len(bounds) else ref.ticks
            active = [g['id'] for g in subgoals if g['at_s']<=ref.clock[idx]+1.e-8][-1]
            item = dict(decision_id=j,observed_frame_id=idx,t_s=ref.clock[idx],history_frame_ids=bounds[max(0,j-2):j],
                teacher=dict(source='known_reference_from_actual_pose',subgoal_id=active,chunk=chunk),
                executed=dict(source='expert',kind='stop' if idx==stop else 'reference_segment',
                              reference_file='route.json',segment_id=int(ref.sample(ref.knots,ref.clock[idx])[3]),
                              ref_start_s=ref.clock[idx],ref_end_s=ref.clock[end],end_frame_id=end))
            stream.write(json.dumps(item,allow_nan=False)+'\n')
    return len(bounds)


def capture(route, out, dataset, resolution):
    ref = Reference(route)
    key = route['scene']
    scene_id=SCENES[key]
    scene_dir=ROOT/'scenes'/scene_id
    rig, known = (RoomRig,RoomSpace()) if key=='room' else (CorridorRig,CorridorSpace())
    seed=route.get('seed',65)
    subgoals = route.get('subgoals') or [dict(id='follow_reference',at_s=0.,predicate='follow_reference')]
    if subgoals[0]['at_s']!=0 or any(b['at_s']<a['at_s'] for a,b in zip(subgoals,subgoals[1:])):
        raise ValueError('Subgoal times must be ordered starting at zero.')
    for name in ['rgb','thermal']:
        (out/name).mkdir()
    dump(out/'route.json',route)
    states=[]; speeds=[]; clearance=np.inf; stop=None
    with PolicyFlightSimulator(scene_dir/'models/rgb.ply',point_cloud=scene_dir/'collision/rgb_surface_m.ply',
             config={'limits':{'max_steps':ref.ticks+5}}, scene_factory=rig_factory(rig,route),
             seed=seed, resolution=resolution, spawn=ref.knots[0,1:4],yaw=ref.knots[0,4],
             spawn_velocity_w_mps=route.get('initial_velocity_w_mps',(0.,0.,0.))) as sim, ExitStack() as stack:
        publish_calibration(dataset,scene_id,scene_dir,sim.scene)
        log=stack.enter_context((out/'frames.jsonl').open('x'))
        width,height=sim.scene.width,sim.scene.height
        vid=FpvVideoWriter(out/'rgb_thermal_fpv.mp4',width*2,height,fps=1/CONTROL_DT)
        stack.callback(vid.close)
        r=sim.last_receipt
        for i in range(ref.ticks+1):
            if i:
                command,_ = controller(sim,r,ref.knots,(i-.5)*CONTROL_DT,sample=ref.sample)
                r=sim.step(command,format='body_rates_thrust')
            st=r['state']; p=np.asarray(st['position']); q=np.asarray(st['quaternion_wxyz'])
            R=Rotation.from_quat(q[[1,2,3,0]]).as_matrix()
            if r['frame_id']!=i or abs(r['sim_time']-ref.clock[i])>1e-6 or r['terminated'] or r['truncated'] or r['collision']['in_collision']:
                raise RuntimeError(f'{route["id"]}: unsafe or terminated at {i} {p} {r["reason"]}')
            if not known.contains_box(p,R,sim.obstacles.safety_half_size):
                raise RuntimeError(f'{route["id"]}: out of known air at {i} {p}')
            states.append(np.r_[p,st['euler_deg'][2]])
            speeds.append(st['speed'])
            gap=r['min_sampled_safety_clearance']
            if gap is not None:
                clearance=min(clearance,gap)
            rg=image8(sim.last_rgb); ir=image8(sim.scene.ir)
            rgb_path=f'rgb/{i:06d}.png'; ir_path=f'thermal/{i:06d}.png'
            Image.fromarray(rg).save(out/rgb_path,compress_level=1)
            Image.fromarray(ir).save(out/ir_path,compress_level=1)
            vid.write(np.concatenate([rg,ir],axis=1))
            frame=dict(frame_id=i,t_s=r['sim_time'],dt_actual_s=r['dt_actual'],position_w_m=p.tolist(),q_w_from_b_wxyz=q.tolist(),
                       velocity_w_mps=st['velocity'],omega_b_radps=st['angular_velocity'],rgb_path=rgb_path,thermal_path=ir_path,
                       applied_body_rates_thrust=r['applied_control'] if i else None,commanded_body_rates_thrust=r['raw_action'] if i else None,
                       collision=False,min_safety_clearance_m=gap if i else None,in_known_space=True,
                       rgb_w2c_native=sim.scene.W.tolist(),thermal_w2c_native=sim.scene.Wir.tolist(),rig_pitch_deg=getattr(sim.scene,'pitch',0.))
            log.write(json.dumps(frame,allow_nan=False)+'\n')
            error=local_delta(states[-1],ref.poses[-1])
            ready=np.linalg.norm(error[:3])<=.12 and abs(error[3])<=8 and st['speed']<=.1
            if stop is None and i>=ref.stop_tick and ready: stop=i
            if i%250==0:
                print(f'{route["id"]}: {i}/{ref.ticks}',flush=True)
        if stop is None or stop>=ref.ticks or not ready: raise RuntimeError('Terminal hold did not settle.')
    n=make_decisions(out,route,ref,states,subgoals,stop)
    poses=np.array(states)
    task=dict(ordered_subgoals=subgoals,goal_w_m=ref.poses[-1,:3].tolist(),goal_yaw_deg=ref.poses[-1,3],
              position_tol_m=.12,yaw_tol_deg=8.,speed_tol_mps=.10)
    task.update(route.get('task_spec',{}))
    stats=dict(frames=len(states),decisions=n,path_m=float(np.linalg.norm(np.diff(poses[:,:3],axis=0),axis=1).sum()),
               duration_s=ref.clock[-1],z_min_m=poses[:,2].min(),z_max_m=poses[:,2].max(),peak_speed_mps=max(speeds),
               min_safety_clearance_m=clearance,final_speed_mps=speeds[-1],final_error_m=float(np.linalg.norm(error[:3])),
               final_yaw_error_deg=abs(error[3]),collision_count=0,outside_known_frames=0)
    episode=dict(schema_version='rgbt_episode_v1',episode_id=route['id'],scene_version=scene_id,task_id=route.get('task_id',route['id']),
              route_family_id=route['family'],source_episode_id=route.get('source_episode_id'),seed=seed,
              instruction=route['instruction'],instruction_lang='en',
              instruction_zh=route['instruction_zh'],task_spec=task,behavior_policy='closed_loop_reference_expert',
              teacher_version='continuous_reference_chunk_v1',controller_version='fly.controller.nominal_v1',
              observation_config=dict(rgb_ev=0,thermal_encoding='pseudocolor_rgb8',rgb_wh=[width,height],thermal_wh=[width,height]),
              waypoint_spec=dict(horizon_steps=3,frame='per_step_yaw_flu',units='m_deg',stop_at='step_start',
                                 sampling='reference segment boundaries, <= 2 s with arc/height/yaw/phase subdivision; no inter-decision dwell',
                                 control_dt_s=CONTROL_DT,stop_frame_id=stop,terminal_hold_s=(ref.ticks-stop)*CONTROL_DT),
              initial_frame_id=0,final_frame_id=ref.ticks,termination_reason='expert_stop',task_success=True,
              search_evidence=route.get('search_evidence'),flight_metrics=stats,human_review='pending',
              video='rgb_thermal_fpv.mp4',video_layout='left RGB; right synchronized thermal',video_fps=1/CONTROL_DT)
    dump(out/'episode.json',episode)
    print('COMPLETED',route['id'],stats,flush=True)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('routes',type=Path,nargs='+')
    ap.add_argument('--dataset',type=Path,default=DATASET)
    ap.add_argument('--resolution',type=float,default=.5)
    args=ap.parse_args()
    episodes=args.dataset/'episodes'
    episodes.mkdir(parents=True,exist_ok=True)
    for path in args.routes:
        route=json.loads(path.read_text())
        dest=episodes/route['id']
        if dest.exists():
            raise FileExistsError(dest)
        with tempfile.TemporaryDirectory(dir=episodes,prefix='.capture-') as tmp:
            staged=Path(tmp)/route['id']; staged.mkdir()
            capture(route,staged,args.dataset,args.resolution)
            staged.rename(dest)


if __name__=='__main__':
    main()

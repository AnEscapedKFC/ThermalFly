"""Record an observed-air circuit and three independent first-contact flights."""
import argparse
import csv
import json
import subprocess
import tempfile
from contextlib import ExitStack
from functools import partial
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation

from fly import (ROOT, KnownSpace, ObstacleCloud, PolicyFlightSimulator, RigScene,
                 FpvVideoWriter, commands, controller, ffmpeg_executable, verify_video)
from room import read

SURFACE = ROOT / 'collision/rgb_surface_m.ply'
CHARGER = np.array(read('checks/scene.json')['landmarks_m']['charger'])
TIMESTEP = .05


def dump(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def itineraries():
    # First finish a complete circuit, then look back across the desk from closer range.
    loop = [(1.98,.04,1.54), (1.88,1.18,1.54), (1.20,1.93,1.60), (.15,2.10,1.62),
            (-.98,1.98,1.59), (-1.73,1.55,1.54), (-1.80,.88,1.51), (-1.21,.32,1.48),
            (-1.21,-.29,1.45), (-1.15,-.78,1.44), (-.50,-.85,1.45), (.27,-.81,1.47),
            (1.09,-.77,1.51), (1.88,-.58,1.53), (1.98,.04,1.54)]
    positions = np.array(loop + [(1.14,.45,1.47), (.52,.52,1.38), (-.11,.62,1.33)])
    angles = np.rad2deg(np.unwrap(np.arctan2(CHARGER[1]-positions[:,1], CHARGER[0]-positions[:,0])))
    knots = [[0.,*positions[0],angles[0]], [1.2,*positions[0],angles[0]]]
    for i in range(1, len(positions)):
        ds = np.linalg.norm(positions[i]-positions[i-1])
        duration = max(1.875*ds/.34, 1.875*abs(angles[i]-angles[i-1])/22., 2.)
        t = round((knots[-1][0]+duration)*20)/20
        knots.append([t,*positions[i],angles[i]])
    knots.append([knots[-1][0]+1.5,*knots[-1][1:]])
    return {
        '01_room_survey': dict(label='房间环绕与桌面近观', knots=knots, circuit_end=knots[len(loop)][0], mode='track_table'),
        '02_table_contact': dict(label='桌面接触', mode='table_approach', target=[.25,-.25,.70], tolerance=.65,
            knots=[[0,1.42,-.39,1.49,180],[1.2,1.42,-.39,1.49,180],[7.8,.4,-.35,1.32,180],
                   [9,.4,-.35,1.32,180],[15.8,.4,-.35,.54,180]]),
        '03_wall_contact': dict(label='房间正 X 侧墙面接触', mode='wall_approach', target=[3.615,-.229,1.30], tolerance=.58,
            knots=[[0,1.54,-.40,1.39,14],[1.2,1.54,-.40,1.39,14],[6.8,2.32,-.23,1.35,0],
                   [16.5,3.88,-.23,1.35,0]]),
        '04_floor_contact': dict(label='通道地面接触', mode='floor_approach', target=[-.08,2.14,.07], tolerance=.48,
            knots=[[0,-.62,1.48,1.38,12],[1.2,-.62,1.48,1.38,12],[5.4,-.25,1.8,1.12,0],
                   [13.7,-.25,1.8,-.12,0]])}


class SurveyRig(RigScene):
    """Pitch both sensors about the common RGB mount; preserve the calibrated IR transform."""
    def __init__(self,*args,mode,camera_targets=None,**kwargs):
        self.mode = mode
        self.camera_targets = None if camera_targets is None else np.asarray(camera_targets)
        self.tick = 0
        super().__init__(*args,**kwargs)

    def render(self,camera):
        mount = np.asarray(camera)
        forward = -mount[:3,2]
        base_elevation = np.rad2deg(np.arcsin(np.clip(forward[2],-1,1)))
        if self.mode=='track_table' or self.camera_targets is not None:
            target = CHARGER
            if self.camera_targets is not None:
                keys = self.camera_targets
                t = self.tick*TIMESTEP
                target = np.array([np.interp(t,keys[:,0],keys[:,k]) for k in (1,2,3)])
            delta = target-mount[:3,3]
            desired = np.rad2deg(np.arctan2(delta[2],np.linalg.norm(delta[:2])))
        else:
            desired = {'table_approach':-43., 'wall_approach':-7., 'floor_approach':-59.}[self.mode]
        self.pitch = float(np.clip(desired-base_elevation,-82.,0.))
        self.optical_gl = mount.copy()
        self.tick += 1
        self.optical_gl[:3,:3] = mount[:3,:3] @ Rotation.from_euler('x',self.pitch,degrees=True).as_matrix()
        return super().render(self.optical_gl)


def check_plan(spec,cloud,known):
    knots = np.array(spec['knots'])
    sweep = 'target' not in spec
    for t in np.arange(0, knots[-1,0]+.025 if sweep else .025, .05):
        goal = commands(knots,t)[0]
        q = Rotation.from_euler('z',goal[3],degrees=True)
        if not known.contains_box(goal[:3],q.as_matrix(),cloud.safety_half_size,.085):
            raise ValueError(f"{spec['label']} unknown planned volume {t:.2f} {goal[:3]}")
        if cloud.distance(goal[:3],q.as_quat()[[3,0,1,2]])['safety_clearance']<.10:
            raise ValueError(f"{spec['label']} insufficient planned clearance {t:.2f}")


def native_optical_to_metric(scene,W):
    c = np.linalg.inv(W)
    c[:3,:3] = scene.R @ c[:3,:3]
    c[:3,3] = scene.s * scene.R @ (c[:3,3]-scene.origin)
    return c


def capture(out,spec,cloud,known):
    knots = np.array(spec['knots'])
    survey = 'target' not in spec
    circuit = 'circuit_end' in spec
    total = round(knots[-1,0]/TIMESTEP)
    dump(out/'plan.json', spec)
    with PolicyFlightSimulator(ROOT/'models/rgb.ply',point_cloud=SURFACE,seed=65,
            config={'limits':{'max_steps':total+10}},spawn=knots[0,1:4],yaw=knots[0,4],resolution=1.,
            scene_factory=partial(SurveyRig,mode=spec['mode'],camera_targets=spec.get('camera_targets'))) as sim, ExitStack() as stack:
        writers = [FpvVideoWriter(out/name,640,480) for name in ('rgb_fpv.mp4','thermal_fpv.mp4')]
        for w in writers: stack.callback(w.close)
        log = stack.enter_context((out/'frames.jsonl').open('w'))
        waypoints = csv.writer(stack.enter_context((out/'waypoints.csv').open('w',newline='')))
        waypoints.writerow(['waypoint_id','sim_time_s','frame_id','x_m','y_m','z_m','qw','qx','qy','qz','vx_mps','vy_mps','vz_mps','rig_pitch_deg'])
        r = sim.last_receipt
        calibration = dict(image_size=[640,480], fps=20, nominal_control_dt_s=.05, physics_dt_s=.005,
                    rgb_K=sim.scene.ks['rgb'].tolist(),thermal_K=sim.scene.ks['thermal'].tolist(),
                    ir_from_rgb_native=sim.scene.rig.tolist(), source_to_sim=read('inputs/frame.json'),
                    simulator_config=sim.config,seed=sim.seed)
        dump(out/'calibration.json',calibration)
        path = 0.; peak_speed = 0.; min_clear = np.inf; previous = None
        waypoint_count = 0; in_rgb = 0; in_ir = 0; substeps = 0
        orbit_angles = []; elevations = []; rig_error = 0.; closure = None
        for i in range(total+1):
            if i:
                action,_ = controller(sim,r,knots,(i-.5)*TIMESTEP)
                r = sim.step(action,format='body_rates_thrust')
            t = r['sim_time']; s = r['state']; p = np.array(s['position'])
            event = r['collision']['event']; q = np.array(s['quaternion_wxyz'])
            rotation = Rotation.from_quat(q[[1,2,3,0]]).as_matrix()
            if r['frame_id']!=i or r['truncated'] or (r['terminated'] and (survey or not event or not i or r['reason']!='collision' or event['reason']!='obstacle')):
                raise RuntimeError((i,r['reason'],event))
            if not event and abs(t-i*TIMESTEP)>1e-7: raise RuntimeError(('clock',t,i))
            if survey and (s['speed']>.40 or np.linalg.norm(s['angular_velocity'])>1. or
                        not known.contains_box(p,rotation,cloud.safety_half_size,.06)):
                raise RuntimeError(('survey flight envelope',i,p))
            if previous is not None: path += np.linalg.norm(p-previous)
            previous = p; peak_speed = max(peak_speed,s['speed'])
            min_clear = min(min_clear,r['min_sampled_safety_clearance'])
            substeps += r['substeps_executed']
            centers = {}
            for m,W in [('rgb',sim.scene.W),('thermal',sim.scene.Wir)]:
                C = native_optical_to_metric(sim.scene,W)
                centers[m] = C
                xyz = C[:3,:3].T @ (CHARGER-C[:3,3])
                uvz = sim.scene.ks[m] @ xyz
                visible = xyz[2]>.001 and 0<=uvz[0]/xyz[2]<640 and 0<=uvz[1]/xyz[2]<480
                if m=='rgb': in_rgb += int(visible)
                else: in_ir += int(visible)
            e = float(np.rad2deg(np.arcsin(np.clip(centers['rgb'][2,2],-1.,1.))))
            elevations.append(e)
            rig_error = max(rig_error,np.max(np.abs(sim.scene.Wir @ np.linalg.inv(sim.scene.W)-sim.scene.rig)))
            if not np.allclose(centers['rgb'],sim.scene.optical_gl@np.diag([1,-1,-1,1]),atol=1e-6):
                raise RuntimeError('Rendered optical pose mismatch')
            if circuit and t<=spec['circuit_end']+1e-7:
                orbit_angles.append(np.arctan2(p[1]-CHARGER[1],p[0]-CHARGER[0]))
                closure = float(np.linalg.norm(p-knots[0,1:4]))
            record = {key:r[key] for key in ('episode_id','step_id','frame_id','sim_time','dt_actual',
                         'substeps_executed','action_format','raw_action','applied_control','state','collision',
                         'min_sampled_safety_clearance','terminated','truncated','reason')}
            record.update(rig_pitch_deg=sim.scene.pitch,rgb_optical_elevation_deg=e,
                          rgb_c2w_sim_m=centers['rgb'].tolist(),thermal_c2w_sim_m=centers['thermal'].tolist(),
                          rgb_w2c_native=sim.scene.W.tolist(),thermal_w2c_native=sim.scene.Wir.tolist())
            log.write(json.dumps(record,separators=(',',':'),allow_nan=False)+'\n')
            if i%6==0 and abs(t-i*TIMESTEP)<1e-7:
                waypoints.writerow([waypoint_count,f'{i*TIMESTEP:.2f}',i,*p,*q,*s['velocity'],sim.scene.pitch])
                waypoint_count += 1
            for w,img in zip(writers,[sim.last_rgb,sim.scene.ir]): w.write(img)
            if i%240==0: print(out.name,i,total,round(t,2),flush=True)
            if event:
                error = float(np.linalg.norm(np.array(event['point'])-spec['target']))
                if error>spec['tolerance']: raise RuntimeError(('unintended point',event,error))
                dump(out/'contact.json',dict(frame_id=i,sim_time_s=t,**event))
                Image.fromarray(np.rint(np.hstack((sim.last_rgb,sim.scene.ir))*255).astype(np.uint8)).save(out/'contact.jpg',quality=94)
                break
        if not survey and not event: raise RuntimeError('No obstacle contact')
        if survey and np.linalg.norm(p-knots[-1,1:4])>.08: raise RuntimeError('Survey missed endpoint')
        if circuit and (closure>.08 or abs(np.rad2deg(np.unwrap(orbit_angles)[-1]-np.unwrap(orbit_angles)[0]))<350):
            raise RuntimeError('Survey circuit incomplete')
        if rig_error>1e-7: raise RuntimeError(('rig error',rig_error))
        frames = i+1
        report = dict(label=spec['label'],episode_id=r['episode_id'],duration_s=t,frames=frames,
                      physics_substeps=substeps,waypoint_count=waypoint_count,traveled_m=path,
                      peak_speed_mps=peak_speed,min_sampled_safety_clearance_m=min_clear,
                      contact=event,optical_elevation_deg=[min(elevations),max(elevations)],max_rig_matrix_error=rig_error)
        if circuit:
            report.update(circuit_end_s=spec['circuit_end'],circuit_closure_m=closure,
                          orbit_azimuth_deg=float(np.rad2deg(np.ptp(np.unwrap(orbit_angles)))))
        if survey:
            report.update(charger_in_rgb_frustum_frames=in_rgb,charger_in_ir_frustum_frames=in_ir,
                          known_body_volume_checks=frames,extra_volume_margin_m=.06)
    pair = out/'rgb_thermal_fpv.mp4'
    subprocess.run([ffmpeg_executable(),'-v','error','-n','-i',str(out/'rgb_fpv.mp4'),'-i',str(out/'thermal_fpv.mp4'),
                    '-filter_complex','hstack=inputs=2','-an','-c:v','libx264','-threads','4','-crf','18',
                    '-pix_fmt','yuv420p','-movflags','+faststart',str(pair)],check=True)
    report['video_decode'] = {f.name:verify_video(f,frames) for f in sorted(out.glob('*.mp4'))}
    dump(out/'summary.json',report)
    print('completed',spec['label'],frames,t,flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--plan-only',action='store_true')
    args = parser.parse_args()
    plans=itineraries(); known=KnownSpace(); cloud=ObstacleCloud(SURFACE)
    for name,spec in plans.items():
        check_plan(spec,cloud,known); print('preflight',name,'nominal seconds',spec['knots'][-1][0],flush=True)
    if args.plan_only: return
    args.output.mkdir(parents=True,exist_ok=True)
    for name,spec in plans.items():
        dest=args.output/name
        if dest.exists(): raise FileExistsError(dest)
        with tempfile.TemporaryDirectory(prefix='.'+name+'-',dir=args.output) as t:
            staging=Path(t)/name; staging.mkdir()
            capture(staging,spec,cloud,known)
            staging.rename(dest)


if __name__=='__main__':
    main()


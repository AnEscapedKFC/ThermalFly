"""Replay captured commands through the room simulator. Only frame zero supplies a body pose."""
import argparse
import csv
import hashlib
import json
import math
import tempfile
from contextlib import ExitStack
from functools import partial
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from record_survey_contacts import (ROOT, SURFACE, RigScene, PolicyFlightSimulator, FpvVideoWriter,
    verify_video, dump, native_optical_to_metric, TIMESTEP)


class PlaybackRig(RigScene):
    def __init__(self,*args,first_pitch=0.,**kwargs):
        self.pitch=first_pitch
        super().__init__(*args,**kwargs)

    def render(self,camera):
        optical=np.asarray(camera).copy()
        optical[:3,:3]=optical[:3,:3] @ Rotation.from_euler('x',self.pitch,degrees=True).as_matrix()
        return super().render(optical)


def vector(v,n):
    a=np.asarray(v,dtype=float)
    if a.shape!=(n,) or not np.isfinite(a).all(): raise ValueError('向量维数或数值无效')
    return a


def parse_tape(lines):
    records=[]
    position=yaw=None
    for i,line in enumerate(lines):
        if not line.strip(): continue
        row=json.loads(line)
        n=len(records)
        if not isinstance(row,dict) or row.get('frame_id',row.get('frame'))!=n: raise ValueError(f'第 {i+1} 行的帧号应为 {n}')
        pitch=float(row.get('rig_pitch_deg',0))
        if not math.isfinite(pitch) or abs(pitch)>180: raise ValueError('支架俯仰角无效')
        action=row.get('raw_action')
        fmt=row.get('action_format') or 'body_rates_thrust'
        if n==0:
            st=row['state']; position=vector(st['position'],3)
            q=vector(st['quaternion_wxyz'],4)
            if abs(np.linalg.norm(q)-1)>1e-5 or np.linalg.norm(q[1:3])>1e-5 or action is not None or any(
                    np.linalg.norm(vector(st[k],3))>1e-6 for k in ('velocity','angular_velocity')):
                raise ValueError('首帧须为水平、静止的模拟器初始状态')
            yaw=math.degrees(2*math.atan2(q[3],q[0]))
        else:
            if fmt not in ('body_rates_thrust','gradnav_normalized'): raise ValueError(f'不支持动作格式 {fmt}')
            if action is None: raise ValueError(f'第 {n} 帧缺少 raw_action')
            action=vector(action,4).tolist()
        records.append((action,fmt,pitch))
        if n>=30000: raise ValueError('动作文件超过 30,000 个控制步')
    if len(records)<2: raise ValueError('文件需要初始帧及至少一个动作')
    return position,yaw,records


def make_sim(tape,resolution=1.):
    p,yaw,rows=tape
    return PolicyFlightSimulator(ROOT/'models/rgb.ply',point_cloud=SURFACE,seed=65,resolution=resolution,
        config={'limits':{'max_steps':len(rows)+8}},spawn=p,yaw=yaw,
        scene_factory=partial(PlaybackRig,first_pitch=rows[0][2]))


def execute_frame(sim,row):
    action,fmt,sim.scene.pitch=row
    return sim.step(action,format=fmt)


def verify_replay(source,out):
    lines=(source/'frames.jsonl').read_bytes()
    tape=parse_tape(lines.splitlines())
    reference=[json.loads(line) for line in lines.splitlines()]
    with (source/'waypoints.csv').open() as f:
        waypoints={int(row['frame_id']):row for row in csv.DictReader(f)}
    maxima={k:0. for k in ['position_m','velocity_mps','orientation_rad','action_applied','time_s','rgb_camera','thermal_camera','waypoint_m']}
    out.mkdir(parents=True,exist_ok=False)
    with make_sim(tape) as sim, ExitStack() as stack:
        video=FpvVideoWriter(out/'rgb_thermal_fpv.mp4',1280,480);stack.callback(video.close)
        log=stack.enter_context((out/'frames.jsonl').open('w'))
        way=csv.writer(stack.enter_context((out/'waypoints.csv').open('w',newline='')))
        way.writerow(['waypoint_id','sim_time_s','frame_id','x_m','y_m','z_m','qw','qx','qy','qz','vx_mps','vy_mps','vz_mps','rig_pitch_deg'])
        for i,(row,ref) in enumerate(zip(tape[2],reference)):
            r=execute_frame(sim,row) if i else sim.last_receipt
            st=r['state'];old=ref['state'];q=vector(st['quaternion_wxyz'],4);qr=vector(old['quaternion_wxyz'],4)
            p=np.array(st['position'])
            errors=dict(position_m=np.linalg.norm(p-old['position']),velocity_mps=np.linalg.norm(np.subtract(st['velocity'],old['velocity'])),
                        orientation_rad=4*np.arcsin(min(1.,min(np.linalg.norm(q-qr),np.linalg.norm(q+qr))/2)),
                        action_applied=max(abs(np.subtract(r['applied_control'],ref['applied_control']))),
                        time_s=abs(r['sim_time']-ref['sim_time']),rgb_camera=np.max(abs(sim.scene.W-ref['rgb_w2c_native'])),
                        thermal_camera=np.max(abs(sim.scene.Wir-ref['thermal_w2c_native'])))
            if i in waypoints:
                w=waypoints[i];errors['waypoint_m']=np.linalg.norm(p-[float(w[k]) for k in ['x_m','y_m','z_m']])
                way.writerow([w['waypoint_id'],f"{r['sim_time']:.2f}",i,*p,*q,*st['velocity'],sim.scene.pitch])
            for k,v in errors.items(): maxima[k]=max(maxima[k],v)
            if r['frame_id']!=i or r['substeps_executed']!=ref['substeps_executed'] or r['terminated'] or r['truncated']:
                raise RuntimeError(('flight mismatch',i,r['reason']))
            actual={k:r[k] for k in ('episode_id','frame_id','sim_time','state','action_format','raw_action','applied_control','substeps_executed','terminated','truncated')}
            actual.update(rig_pitch_deg=sim.scene.pitch,rgb_w2c_native=sim.scene.W.tolist(),thermal_w2c_native=sim.scene.Wir.tolist())
            log.write(json.dumps(actual,separators=(',',':'))+'\n')
            video.write(np.concatenate((sim.last_rgb,sim.scene.ir),axis=1))
            if i%400==0:print('replay',i,flush=True)
    n=len(reference);decode=verify_video(out/'rgb_thermal_fpv.mp4',n)
    passed=all(x<=1e-6 for x in maxima.values())
    dump(out/'verification.json',dict(passed=passed,reference=str(source),source_sha256=hashlib.sha256(lines).hexdigest(),
        frames=n,commands=n-1,duration_s=r['sim_time'],waypoints=len(waypoints),collision_events=0,maximum_errors=maxima,video=decode,
        reset='frame zero pose, nominal dynamics seed 65',camera='recorded rig pitch, calibrated two-sensor mount'))
    if not passed:raise RuntimeError(maxima)
    print(maxima,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('source',type=Path);parser.add_argument('output',type=Path)
    a=parser.parse_args()
    a.output.parent.mkdir(parents=True,exist_ok=True)
    if a.output.exists():raise FileExistsError(a.output)
    with tempfile.TemporaryDirectory(dir=a.output.parent) as scratch:
        dest=Path(scratch)/'replay';verify_replay(a.source.resolve(),dest);dest.rename(a.output)


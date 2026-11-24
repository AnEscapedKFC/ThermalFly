"""Replay the archived contact tapes against the installed simulator scene at the recorded resolution."""
import csv
import hashlib
import json
import math
from contextlib import ExitStack
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw
from scipy.spatial.transform import Rotation

OUT = Path(__file__).resolve().parent
SCENE = OUT.parents[1] / 'scenes/room65_selected_anchored'
sys.path.insert(0, str(SCENE))
from action_replay import make_sim, parse_tape, execute_frame
from record_survey_contacts import FpvVideoWriter, verify_video, native_optical_to_metric

def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2)+'\n')

def compare_task(source):
    destination = OUT / source.name
    destination.mkdir()
    tape_bytes = (source/'frames.jsonl').read_bytes()
    refs = [json.loads(line) for line in tape_bytes.splitlines()]
    tape = parse_tape(tape_bytes.splitlines())
    original = json.loads((source/'contact.json').read_text())
    with (source/'waypoints.csv').open() as stream:
        reader = csv.DictReader(stream)
        columns = reader.fieldnames
        old_waypoints = {int(row['frame_id']):row for row in reader}
    maxima = dict(position_m=0., velocity_mps=0., orientation_rad=0., angular_velocity_rad_s=0.,
                  rgb_W2C=0., thermal_W2C=0., simulation_time_s=0., applied_control=0., waypoint_m=0.,
                  safety_clearance_m=0.)
    steps_match = True
    observed = None
    events_match = True
    wp_count = 0
    with make_sim(tape) as sim, ExitStack() as stack:
        assert sim.scene.rgb.count == 778926 and sim.point_cloud_file.samefile(SCENE/'collision/rgb_surface_m.ply')
        w,h = sim.scene.width,sim.scene.height
        video = FpvVideoWriter(destination/'rgb_thermal_fpv.mp4',2*w,h)
        stack.callback(video.close)
        log = stack.enter_context((destination/'frames.jsonl').open('x'))
        wp = csv.writer(stack.enter_context((destination/'waypoints.csv').open('x',newline='')))
        wp.writerow(columns)
        for i,(cmd,reference) in enumerate(zip(tape[2],refs)):
            r = execute_frame(sim,cmd) if i else sim.last_receipt
            st,old = r['state'],reference['state']
            p,q=st['position'],st['quaternion_wxyz']
            rel=Rotation.from_quat(np.roll(old['quaternion_wxyz'],-1)).inv()*Rotation.from_quat(np.roll(q,-1))
            err=dict(position_m=math.dist(p,old['position']), velocity_mps=math.dist(st['velocity'],old['velocity']),
                     orientation_rad=rel.magnitude(), angular_velocity_rad_s=math.dist(st['angular_velocity'],old['angular_velocity']),
                     rgb_W2C=np.abs(sim.scene.W-reference['rgb_w2c_native']).max(),
                     thermal_W2C=np.abs(sim.scene.Wir-reference['thermal_w2c_native']).max(),
                     simulation_time_s=abs(r['sim_time']-reference['sim_time']),
                     applied_control=math.dist(r['applied_control'],reference['applied_control']),
                     safety_clearance_m=abs(r['collision']['safety_clearance']-reference['collision']['safety_clearance']))
            if i in old_waypoints:
                old_wp=old_waypoints[i]
                wp.writerow([wp_count,f"{r['sim_time']:.2f}",i,*p,*q,*st['velocity'],sim.scene.pitch])
                err['waypoint_m']=math.dist(p,[float(old_wp[k]) for k in ('x_m','y_m','z_m')])
                wp_count+=1
            for key,v in err.items(): maxima[key]=max(maxima[key],float(v))
            steps_match &= r['frame_id']==reference['frame_id'] and r['substeps_executed']==reference['substeps_executed']
            events_match &= (r['terminated'],r['truncated'],r['reason'])==(reference['terminated'],reference['truncated'],reference['reason'])
            data={k:r[k] for k in ['episode_id','step_id','frame_id','sim_time','dt_actual','substeps_executed','action_format','raw_action','applied_control','state','collision','min_sampled_safety_clearance','terminated','truncated','reason']}
            data.update(rig_pitch_deg=sim.scene.pitch,rgb_w2c_native=sim.scene.W.tolist(),thermal_w2c_native=sim.scene.Wir.tolist(),
                        rgb_c2w_sim_m=native_optical_to_metric(sim.scene,sim.scene.W).tolist(),
                        thermal_c2w_sim_m=native_optical_to_metric(sim.scene,sim.scene.Wir).tolist())
            log.write(json.dumps(data,separators=(',',':'),allow_nan=False)+'\n')
            pair=np.concatenate((sim.last_rgb,sim.scene.ir),axis=1)
            video.write(pair)
            if r['terminated'] or r['truncated']:
                if r['collision']['event']:
                    observed=dict(frame_id=i,sim_time_s=r['sim_time'],**r['collision']['event'])
                    Image.fromarray(np.uint8(np.rint(pair*255))).save(destination/'contact.jpg',quality=94)
                break
        save(destination/'contact.json',observed)
        original_jpg=np.asarray(Image.open(source/'contact.jpg').convert('RGB'),dtype='float32')
        render_jpg=np.asarray(Image.open(destination/'contact.jpg').convert('RGB'),dtype='float32') if observed else None
        mae=None if render_jpg is None else dict(rgb=float(abs(original_jpg[:,:w-8]-render_jpg[:,:w-8]).mean()),
                   thermal=float(abs(original_jpg[:,w+8:]-render_jpg[:,w+8:]).mean()))
    delta=None if observed is None else dict(frame=observed['frame_id']-original['frame_id'],time_s=observed['time']-original['time'],
              point_m=math.dist(observed['point'],original['point']),body_center_m=math.dist(observed['position'],original['position']))
    decoded=verify_video(destination/'rgb_thermal_fpv.mp4',i+1)
    passed=bool(observed and original['reason']==observed['reason'] and steps_match and events_match and i==len(refs)-1 and
                wp_count==len(old_waypoints) and delta['frame']==0 and max(abs(v) for v in delta.values())<=1e-6 and max(maxima.values())<=1e-6)
    report=dict(task=source.name,original_action_file=str(source/'frames.jsonl'),action_sha256=hashlib.sha256(tape_bytes).hexdigest(),
                original=original,replay=observed,delta=delta,per_frame_maxima=maxima,substeps_match=steps_match,termination_matches=events_match,
                decoded=decoded,frames=i+1,waypoints=wp_count,contact_jpeg_mean_abs_8bit_excluding_seam=mae,passed=passed)
    save(destination/'comparison.json',report)
    return report

contacts=sorted(path.parent for path in (SCENE/'runs').glob('*/*/contact.json'))
if len(contacts)!=5:
    raise ValueError(f'Expected the five archived contact tasks, found {len(contacts)}')
reports=[]
for source in contacts:
    report=compare_task(source)
    print(report['task'],report['passed'],report['delta'],flush=True)
    reports.append(report)
save(OUT/'comparison.json',dict(scene=str(SCENE),models=json.loads((SCENE/'inputs/source.json').read_text()),
     control_hz=20,reference_seed=65,passed=all(report['passed'] for report in reports),tasks=reports))
if not all(r['passed'] for r in reports):
    raise RuntimeError('Differences found in collision replay; inspect comparison.json')

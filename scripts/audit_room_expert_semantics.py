"""Build geometric evidence and synchronized review panels from recorded expert RGB-T flights."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial.transform import Rotation


def load(path):
    return json.loads(path.read_text())


def projection(frames, center, calibration, modality):
    geom = calibration['source_to_sim']
    scale = geom['meters_per_source_unit']
    native = np.asarray(geom['origin']) + np.asarray(geom['rotation']).T @ center / scale
    mat = np.array([f[modality + '_w2c_native'] for f in frames])
    optical = (mat[:, :3, :3] @ native) + mat[:, :3, 3]
    uvz = optical @ np.asarray(calibration[modality + '_K']).T
    uv = np.divide(uvz[:, :2], uvz[:, 2, None], out=np.zeros_like(uvz[:, :2]),
                   where=np.abs(uvz[:, 2, None]) > 1e-10)
    w, h = calibration['wh']
    visible = (optical[:, 2] > 0) & np.all((uv >= 0) & (uv < [w, h]), axis=1)
    return uv, optical[:, 2] * scale, visible


def object_observation_metrics(spec, meta, frames, times, positions, yaw, center, projections):
    """Measure motion around the object and the final continuous settled interval."""
    a, b = [int(np.argmin(np.abs(times - spec[key]))) for key in ('arc_start_s','arc_end_s')]
    stop = meta['waypoint_spec']['stop_frame_id']
    if not 0 < a < b <= stop < len(frames)-1:
        raise ValueError(f'Invalid observation or terminal frames: {a}, {b}, {stop}')
    relative = positions - center
    distance = np.linalg.norm(relative, axis=1)
    azimuth = np.unwrap(np.arctan2(relative[a:b+1,1],relative[a:b+1,0]))
    sweep = float(np.rad2deg(azimuth[-1]-azimuth[0]))
    closure = float(np.linalg.norm(positions[b]-positions[a]))
    kind = spec['kind']
    if kind not in ('arc','full_orbit'):
        raise ValueError(f'Unknown observation motion: {kind}')
    task = meta['task_spec']
    speeds = np.linalg.norm([f['velocity_w_mps'] for f in frames],axis=1)
    err = np.linalg.norm(positions - task['goal_w_m'],axis=1)
    yaw_error = np.abs((np.rad2deg(yaw)-task['goal_yaw_deg']+180) % 360 - 180)
    settled = ((speeds <= task['speed_tol_mps']) & (err <= task['position_tol_m'])
               & (yaw_error <= task['yaw_tol_deg']))
    bad = np.flatnonzero(~settled[stop:])
    stable = stop + int(bad[-1]) + 1 if len(bad) else stop
    if stable >= len(frames):
        raise ValueError('Final frame outside hover tolerances')
    visible = {m: float(p[2][a:b+1].mean()) for m,p in projections.items()}
    hover_visible = {m: float(p[2][stable:].mean()) for m,p in projections.items()}
    hold_s = float(times[-1]-times[stable])
    approach_delta = float(distance[0]-distance[a])
    geometry = (350 <= abs(sweep) <= 370 and closure <= .15) if kind == 'full_orbit' else (25 <= abs(sweep) < 350)
    return dict(landmark=spec['landmark'],kind=kind,arc_start_frame=a,arc_end_frame=b,
                arc_start_s=float(times[a]),arc_end_s=float(times[b]),start_range_m=float(distance[0]),
                arc_entry_range_m=float(distance[a]),approach_range_reduction_m=approach_delta,
                minimum_range_m=float(distance.min()),signed_sweep_deg=sweep,closure_m=closure,
                arc_center_in_image_fraction=visible,hover_center_in_image_fraction=hover_visible,
                stop_frame_id=stop,stable_hover_start_frame=stable,stable_hover_start_s=float(times[stable]),
                stable_hover_s=hold_s,hover_peak_speed_mps=float(speeds[stable:].max()),
                hover_max_displacement_m=float(np.linalg.norm(positions[stable:]-positions[stable],axis=1).max()),
                passed=bool(geometry and approach_delta>=.15 and min(visible.values())>=.90
                            and min(hover_visible.values())>=.90 and hold_s>=2.-1e-6))


def geometric_audit(ep, calibration, landmarks):
    frames = [json.loads(line) for line in (ep/'frames.jsonl').read_text().splitlines()]
    route = load(ep/'route.json')
    T = np.array([r['t_s'] for r in frames])
    pos = np.array([r['position_w_m'] for r in frames])
    yaw = np.unwrap(Rotation.from_quat(np.array([r['q_w_from_b_wxyz'] for r in frames])[:, [1,2,3,0]]).as_euler('xyz')[:,2])
    w,h = calibration['wh']
    ids = {g['landmark'] for g in route['subgoals']}
    projections = {k: {mod: projection(frames, np.asarray(landmarks[k]['center_w_m']), calibration, mod)
                       for mod in ('rgb','thermal')} for k in ids}
    def nearest(t):
        return int(np.argmin(np.abs(T-t)))
    def sample(i, tag, obj):
        data = dict(role=tag,landmark=obj,frame_id=frames[i]['frame_id'],t_s=round(float(T[i]),3),
                    body_height_m=round(float(pos[i,2]),3))
        for mod, (uv,z,inside) in projections[obj].items():
            data[mod] = dict(center_in_image=bool(inside[i]),optical_depth_m=round(float(z[i]),3),
                             uv_px=np.round(uv[i],1).tolist() if z[i]>0 else None)
        return data
    evidence=[]
    report=dict(episode_id=ep.name)
    if 'orbit_closed_at_s' in route:
        a,b=nearest(route.get('orbit_start_s',0)),nearest(route['orbit_closed_at_s'])
        pc=np.array(route['orbit_center_w_m'])
        phase = np.unwrap(np.arctan2(pos[a:b+1,1]-pc[1],pos[a:b+1,0]-pc[0]))
        angle=float(np.rad2deg(phase[-1]-phase[0]))
        closure=float(np.linalg.norm(pos[b]-pos[a]))
        report['orbit']=dict(start_frame=a,end_frame=b,signed_azimuth_change_deg=round(angle,3),
                             closure_error_m=round(closure,4),height_range_m=np.round([pos[a:b+1,2].min(),pos[a:b+1,2].max()],3).tolist(),
                             complete=abs(angle)>=350 and closure<=.15)
        if ep.name=='room_orbit_vary':
            orbit_height=pos[a:b+1,2]
            report['orbit']['higher_then_lower']=bool(orbit_height.argmax()<orbit_height.argmin()
                                                      and orbit_height.max()-orbit_height[0]>.12
                                                      and orbit_height[0]-orbit_height.min()>.12)
        for k in range(5):
            idx=int(a+np.argmin(np.abs(phase-(phase[0]+k*(phase[-1]-phase[0])/4.))))
            evidence.append(sample(idx,'Orbit '+str(k*90)+' deg','conference_table'))
    goals=route['subgoals']
    if route.get('search_evidence'):
        search = route['search_evidence']
        target = search['target']
        a=nearest(next(g['at_s'] for g in goals if g['id']==search['search_stage']))
        b=nearest(next(g['at_s'] for g in goals if g['id']==search['approach_stage']))
        uv,z,inside = projections[target]['rgb']
        hits = np.flatnonzero(inside[:b+1])
        candidates = np.flatnonzero(inside[a:b+1])+a
        score=np.linalg.norm((uv[candidates]-[w/2,h/2])/[w,h],axis=1)
        best = int(candidates[np.argmin(score)]) if len(candidates) else None
        report['search'] = dict(landmark=target,start_frame=0,scan_start_frame=a,approach_start_frame=b,
                                center_initially_outside=not bool(inside[0]),initial_optical_depth_m=round(float(z[0]),3),
                                climb_before_scan_m=round(float(pos[a,2]-pos[0,2]),3),
                                climb_to_reacquisition_m=round(float(pos[best,2]-pos[0,2]),3) if best is not None else None,
                                scan_yaw_range_deg=round(float(np.rad2deg(np.ptp(yaw[a:b+1]))),1),
                                first_projected_frame=int(hits[0]) if len(hits) else None,
                                first_projected_t_s=round(float(T[hits[0]]),2) if len(hits) else None,
                                search_confirm_frame=best,
                                projection_sequence_pass=not bool(inside[0]) and best is not None)
        evidence.insert(0,sample(0,'Start / search target',target))
        if best is not None:
            evidence.insert(1,sample(best,'After scan',target))
    report['inspection']=[]
    for j,g in enumerate(goals):
        if 'observe' not in g['predicate'] or g['landmark'] not in route.get('inspection_landmarks',[]):
            continue
        obj=g['landmark']; a=nearest(g['at_s']); b=nearest(goals[j+1]['at_s']) if j+1<len(goals) else len(frames)-1
        dist=np.linalg.norm(pos-np.asarray(landmarks[obj]['center_w_m']), axis=1)
        uv,z,inside=projections[obj]['rgb']
        core=inside & np.all((uv > [.1*w,.1*h]) & (uv < [.9*w,.9*h]),axis=1)
        ok=np.flatnonzero(core[a:b+1])+a
        best=int(ok[np.argmin(dist[ok])]) if len(ok) else None
        item=dict(landmark=obj,subgoal_id=g['id'],start_frame=a,end_frame=b,
                  range_at_episode_start_m=round(float(dist[0]),3),range_at_stage_start_m=round(float(dist[a]),3),
                  closest_central_frame=best,range_at_inspection_m=round(float(dist[best]),3) if best is not None else None,
                  range_at_stage_end_m=round(float(dist[b]),3),end_of_stage_view=sample(b,'End of stage',obj),
                  frames_center_in_rgb=int(inside[a:b+1].sum()),frames_observed=b-a+1)
        report['inspection'].append(item)
        if not evidence:
            evidence.append(sample(a,'Start / approach',obj))
        if best is not None:
            evidence.append(sample(best,'Inspection',obj))
    if 'object_observation' in route:
        spec = route['object_observation']
        obj = spec['landmark']
        motion = object_observation_metrics(spec, load(ep/'episode.json'), frames, T, pos, yaw,
                                            np.asarray(landmarks[obj]['center_w_m']), projections[obj])
        report['object_observation'] = motion
        evidence = [sample(0, 'Approach / start', obj)]
        for phase, idx in enumerate(np.linspace(motion['arc_start_frame'],motion['arc_end_frame'],5).round().astype(int)):
            evidence.append(sample(idx, f'Observation {phase*25}%', obj))
        evidence.append(sample(motion['stable_hover_start_frame'], 'Stable hover', obj))
        evidence.append(sample(len(frames)-1, 'Final view', obj))
    report['evidence_frames']=sorted(evidence,key=lambda item: item['frame_id'])
    return report


def panel(root, episode, evidence):
    ff='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
    ft=ImageFont.truetype(ff,16)
    cols=min(5,len(evidence))
    bands=(len(evidence)+cols-1)//cols
    # RGB on first image row and thermal on the second row; all coordinates refer to
    # actual single-view pixels. The open ring leaves the anchor content visible.
    sheet=Image.new('RGB',(cols*320,536*bands),(246,248,251))
    draw=ImageDraw.Draw(sheet)
    for i,e in enumerate(evidence):
        x=i%cols*320
        y=i//cols*536
        draw.text((x+6,y+4),e['role']+' | '+e['landmark'],font=ft,fill=(17,30,46))
        draw.text((x+6,y+25),f"#{e['frame_id']}   {e['t_s']:.2f} s   z {e['body_height_m']:.2f} m",font=ft,fill=(51,60,69))
        for row,mod in enumerate(('rgb','thermal')):
            orig=Image.open(episode/mod/(f"{e['frame_id']:06}.png")).convert('RGB')
            pen=ImageDraw.Draw(orig)
            if e[mod]['center_in_image']:
                u,v=e[mod]['uv_px']
                for radius,color in [(10,'black'),(8,'#27FF9B')]:
                    pen.ellipse((u-radius,v-radius,u+radius,v+radius),outline=color,width=2)
            sheet.paste(orig,(x,y+51+row*240))
    sheet.save(root/'evidence'/(episode.name+'.jpg'),quality=88)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('dataset',type=Path);a=p.parse_args();root=a.dataset.resolve()
    cal=load(root/'scenes/room65_selected_anchored/calibration.json')
    landmark_file=root/'scenes/room65_selected_anchored/landmarks.json'
    if not landmark_file.is_file():
        landmark_file=root/'work_room/landmarks.json'
    landmarks=load(landmark_file)
    supplementary = root/'scenes/room65_selected_anchored/object_landmarks.json'
    if supplementary.is_file():
        additional = load(supplementary)
        if landmarks.keys() & additional.keys():
            raise ValueError('Duplicate object IDs in room landmark tables')
        landmarks.update(additional)
    notes_path=root/'evidence/assistant_frame_notes.json'
    notes=load(notes_path) if notes_path.is_file() else {}
    result=dict(method='RGB and thermal projections of documented landmark centres using actual per-frame calibrated extrinsics. A centre in the image is a geometric test, not a segmentation or occlusion test. Inspect unmarked originals and videos for recognition. Orbit and range use simulated metric poses (assumed 2.80 m ceiling).',
                panels='simultaneous RGB top, thermal bottom; open ring marks projected landmark centre only, on diagnostic images exclusively',
                human_review='pending',episodes=[])
    (root/'evidence').mkdir(exist_ok=True)
    for ep in sorted((root/'episodes').glob('room_*')):
        if not (ep/'episode.json').is_file():continue
        report=geometric_audit(ep,cal,landmarks)
        panel(root,ep,report['evidence_frames'])
        report['panel']=f'evidence/{ep.name}.jpg'
        if ep.name in notes: report['assistant_keyframe_observations']=notes[ep.name]
        (root/'evidence'/(ep.name+'.json')).write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
        result['episodes'].append(report)
    (root/'room_evidence.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
    print([(r['episode_id'],r.get('search',{}).get('projection_sequence_pass'),r.get('orbit',{}).get('complete')) for r in result['episodes']])
if __name__=='__main__': main()

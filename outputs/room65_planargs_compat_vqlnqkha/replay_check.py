"""Replay eight unmodified room65 command tapes against a specified collision map."""
import json
import sys
from pathlib import Path
import numpy as np

audit=Path(__file__).resolve().parent
base=audit.parents[1]/'scenes/room65_selected_anchored'
sys.path.insert(0,str(base))
import action_replay as ar
import room
import fly
from scipy.spatial.transform import Rotation

variant=sys.argv[1]
map_root=base if variant in ('original_map','visual_only_verified') else audit
room.ROOT=ar.ROOT=audit/'visual_only' if variant in ('original_map','visual_only_verified') else audit
ar.SURFACE=map_root/'collision/rgb_surface_m.ply'
fly.ROOT=map_root
space=fly.KnownSpace()
results=[]
for path in sorted((base/'runs').glob('*/*/frames.jsonl')):
    if '05_original_actions_replay' in path.parts:
        continue
    raw=path.read_bytes().splitlines()
    tape=ar.parse_tape(raw)
    state_delta=pose_delta=time_delta=0.
    free_failures=free_total=0
    is_survey='contact' not in path.parent.name
    with ar.make_sim(tape,resolution=.125) as sim:
        for i,(line,command) in enumerate(zip(raw,tape[2])):
            r=ar.execute_frame(sim,command) if i else sim.last_receipt
            ref=json.loads(line)
            st=r['state']
            for k in ('position','velocity','quaternion_wxyz','angular_velocity'):
                state_delta=max(state_delta,float(np.max(np.abs(np.subtract(st[k],ref['state'][k])))))
            pose_delta=max(pose_delta,float(np.max(abs(sim.scene.W-ref['rgb_w2c_native']))),
                           float(np.max(abs(sim.scene.Wir-ref['thermal_w2c_native']))))
            time_delta=max(time_delta,abs(r['sim_time']-ref['sim_time']))
            if is_survey:
                q=st['quaternion_wxyz'];rot=Rotation.from_quat([q[1],q[2],q[3],q[0]])
                free_total+=1
                free_failures+=int(not space.contains_box(np.array(st['position']),rot.as_matrix(),sim.obstacles.safety_half_size,.06))
            if r['terminated'] or r['truncated']:
                break
        last=json.loads(raw[-1])
        event=r['collision']['event']
        original=last['collision']['event']
        row=dict(route=path.parent.name,actual_frame=i,reference_frame=last['frame_id'],commands=len(raw)-1,
                 source_event=original,observed_event=event,terminal_reason=r['reason'],max_body_state_component_delta=state_delta,
                 max_W2C_component_delta=pose_delta,max_time_delta_s=time_delta,free_failures=free_failures,free_checks=free_total,
                 event_time_delta_s=event['time']-original['time'] if original and event else None,
                 event_point_delta_m=float(np.linalg.norm(np.subtract(event['point'],original['point']))) if original and event else None)
        results.append(row)
        print(path.parent.name,i,r['reason'],state_delta,row['event_time_delta_s'],free_failures,flush=True)
        (audit/'checks'/(variant+'_actions.json')).write_text(json.dumps(results,indent=2)+'\n')

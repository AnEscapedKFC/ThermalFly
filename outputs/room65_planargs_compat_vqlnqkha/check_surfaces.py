"""Compare the three obstacle maps and inspect archived first-contact coordinates."""
import json, sys
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree

root=Path(__file__).resolve().parent
base=root.parents[1]/'scenes/room65_selected_anchored';sys.path.insert(0,str(base))
import fly
from pointcloud import ObstacleCloud

roots={'sim_24k':base,'anchored_60k':root/'anchored60','planargs_30k':root}
checks={};xyz={}
for name,p in roots.items():
    obs=ObstacleCloud(p/'collision/rgb_surface_m.ply');tree=cKDTree(obs.points);xyz[name]=obs.points
    fly.ROOT=p
    try:negative=fly.negative_check(obs,fly.KnownSpace())
    except RuntimeError as e:negative={'error':str(e)}
    runs={}
    for log in (base/'runs').glob('*/*/contact.json'):
        ref=json.loads(log.read_text());end=json.loads((log.parent/'frames.jsonl').read_bytes().splitlines()[-1])['state']
        distance,_=tree.query(ref['point']);body=obs.distance(end['position'],end['quaternion_wxyz'])
        runs[log.parent.name]=dict(nearest_to_old_contact_m=float(distance),safety_clearance_at_original_rest_pose_m=body['safety_clearance'])
    checks[name]=dict(negative_control=negative,old_contacts=runs,build=json.loads((p/'collision/build.json').read_text()))
comp={}
for name,cloud in xyz.items():
    dst={}
    for other,target in xyz.items():
        if name==other:continue
        d=cKDTree(target).query(cloud)[0]
        dst[other]=dict(p50_m=float(np.median(d)),p90_m=float(np.percentile(d,90)),fraction_within_8cm=float(np.mean(d<=.08)))
    comp[name]=dst
(root/'checks/collision_comparison.json').write_text(json.dumps(dict(maps=checks,directed_nearest_distances=comp),indent=2)+'\n')
print(json.dumps(dict(planargs=checks['planargs_30k'],distances=comp),indent=2))

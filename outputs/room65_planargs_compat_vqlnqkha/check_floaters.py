"""Repeat the simulator's negative control on every eligible PlanarGS center, keeping the original sample intact."""
import json,sys
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from plyfile import PlyData

root=Path(__file__).resolve().parent;base=root.parents[1]/'scenes/room65_selected_anchored';sys.path.insert(0,str(base))
import fly
from pointcloud import ObstacleCloud

frame=json.loads((root/'inputs/frame.json').read_text());fly.ROOT=root
space=fly.KnownSpace();obs=ObstacleCloud(root/'collision/rgb_surface_m.ply');tree=cKDTree(obs.points)
v=PlyData.read(root/'models/rgb.ply')['vertex'].data
xyz=(np.column_stack([v[k] for k in 'xyz'])-frame['origin'])@np.array(frame['rotation']).T*frame['meters_per_source_unit']
ijk=np.floor((xyz-space.lo)/space.step).astype(int);ids=np.flatnonzero(((ijk>=0)&(ijk<space.free.shape)).all(axis=1))
ids=ids[space.free[tuple(ijk[ids].T)]];distance=tree.query(xyz[ids])[0];mask=distance>.2;void_total=int(mask.sum())
mask&=(xyz[ids,2]>.9)&(xyz[ids,2]<2.1);ids=ids[mask];distance=distance[mask];I=np.eye(3);near=np.array([.16,0,0]);matches=[];checked=0
for j in np.argsort(-distance):
    p=xyz[ids[j]];checked+=1
    if all(space.contains_box(q,I,obs.safety_half_size) for q in (p-near,p,p+near)) and obs.sweep(p-near,p+near) is None and obs.distance(p)['safety_clearance']>.1:
        matches.append(dict(ply_row=int(ids[j]),xyz_m=p.tolist(),nearest_obstacle_m=float(distance[j]),known_free=True,sweep_hit=False))
        if len(matches)>=3:break
out=dict(all_raw_centers=len(v),in_free_farther_than_20cm=void_total,at_test_altitude=len(ids),tested=checked,matches=matches)
(root/'checks/planargs_floater_full_search.json').write_text(json.dumps(out,indent=2)+'\n');print(out)

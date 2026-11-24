"""Compare room65 PLY schemas, SfM cameras and collision geometry in a common frame."""
from pathlib import Path
import json
import hashlib
import numpy as np
from plyfile import PlyData
from scipy.spatial import cKDTree

root=Path(__file__).resolve().parent
base=root.parents[1]/'scenes/room65_selected_anchored'
planar=Path('/home/zr_fan/PlanarGS/room65/official')
old_root=Path('/home/zr_fan/ThermalGaussian_Surface/output/room_vid65_selected_anchored_20261119_104431_53417')
old=old_root/'point_cloud_color/iteration_60000/point_cloud.ply'

def read(path):
    return json.loads(path.read_text())

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()

def cols(data,fields):
    return np.stack([data[n] for n in fields],axis=1)

assets={}
vertices={}
for name, path in [('simulator_24k',base/'models/rgb.ply'),('anchored_60k',old),('planargs_30k',root/'models/rgb.ply')]:
    ply=PlyData.read(path);v=ply['vertex'].data; vertices[name]=v
    scale=np.sort(cols(v,['scale_0','scale_1','scale_2']),axis=1).astype(float)
    assets[name]=dict(path=str(path.resolve()),sha256=sha(path),binary=not ply.text,endian=ply.byte_order,count=len(v),
                      property_count=len(v.dtype.names),fields=list(v.dtype.names),field_data_types=sorted(set(v.dtype[n].str for n in v.dtype.names)),
                      median_shortest_to_longest_axis_ratio=float(np.median(np.exp(scale[:,0]-scale[:,-1]))))

a,b=list(vertices.values())[:2]
geo_fields=['x','y','z','scale_0','scale_1','scale_2',*[f'rot_{i}' for i in range(4)]]
component_diff={k:float(np.abs(a[k]-b[k]).max()) for k in geo_fields+['opacity']}
manifest=read(base/'inputs/manifest.json')['views'];rgb={v['source_name']:v for v in manifest if v['modality']=='rgb'}
cams=read(planar/'output/cameras.json');errs=[];seen=set()
for c in cams:
    name=''.join(c['img_name']);row=rgb[name];seen.add(name);C=np.linalg.inv(row['base_w2c']);k=next(iter(row['targets'].values()));
    intr_a=[c['width'],c['height'],c['fx'],c['fy'],c['width']/2,c['height']/2]
    intr_b=[k[x] for x in ('width','height','fx','fy','cx','cy')]
    errs.append([np.max(abs(C[:3,3]-c['position'])),np.max(abs(C[:3,:3]-c['rotation'])),np.max(abs(np.subtract(intr_a,intr_b)))])
assert seen==set(rgb) and len(seen)==len(cams)
clouds=[cols(PlyData.read(path/'collision/rgb_surface_m.ply')['vertex'].data,'xyz') for path in (base,root)]
surface={}
for name,source,target in [('old_to_new',*clouds),('new_to_old',*reversed(clouds))]:
    dist=cKDTree(target).query(source)[0]
    surface[name]=dict(median_m=float(np.median(dist)),p90_m=float(np.percentile(dist,90)),fraction_within_8cm=float(np.mean(dist<=.08)))
f0,f1=[read(path/'inputs/frame.json') for path in (base,root)]
metrics=dict(assets=assets,anchor24_to_60_rowwise_max_abs_difference=component_diff,
    source_manifest_sha256=[sha(base/'inputs/manifest.json'),read(planar/'data/room65.json')['sha256'],read(old_root/'plus_config.json')['manifest_sha256']],
    simulator_rig_vs_60k_max_abs=float(np.max(abs(np.subtract(read(base/'inputs/rig.json')['rgb_to_thermal'],read(planar.parent/'reference/baseline.json')['model']['rig'])))),
    rgb_cameras=len(rgb),camera_center_max_abs_native=float(np.max(errs,axis=0)[0]),camera_rotation_max_abs=float(np.max(errs,axis=0)[1]),camera_intrinsics_max_abs_px=float(np.max(errs,axis=0)[2]),
    exact_same_simulator_frame={k:f0[k]==f1[k] for k in ['origin','rotation','meters_per_source_unit','bounds_m']},
    meters_per_source_unit=f0['meters_per_source_unit'],collision_nn=surface)
(root/'checks/asset_comparison.json').write_text(json.dumps(metrics,indent=2)+'\n')
print(json.dumps({k:v for k,v in metrics.items() if k!='assets'},indent=2));print([(name,a['count'],a['property_count'],a['median_shortest_to_longest_axis_ratio']) for name,a in assets.items()])

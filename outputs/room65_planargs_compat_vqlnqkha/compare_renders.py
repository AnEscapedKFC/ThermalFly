"""Render both candidate PLYs in the simulator at the 18 native held-out camera poses."""
from pathlib import Path
import json,sys,gc
import numpy as np
import torch
from PIL import Image

root=Path(__file__).resolve().parent
base=root.parents[1]/'scenes/room65_selected_anchored';sys.path.insert(0,str(base))
import room
from scene import GaussianScene

planar=Path('/home/zr_fan/PlanarGS/room65/official');saved=planar/'output/evaluation_30000'
assets=json.loads((root/'checks/asset_comparison.json').read_text())
views=[r for r in json.loads((base/'inputs/manifest.json').read_text())['views'] if r['modality']=='rgb' and r['split']=='test']

def arr(path):
    with Image.open(path) as im:return np.array(im.convert('RGB'),dtype=np.float32)/255.

def psnr(d):return float(-10*np.log10(max(1e-12,np.mean(d*d))))

all_models={}
for name,reference in [('planargs_30k',saved/'rgb'),('anchored_60k',planar.parent/'reference')]:
    model=room.NativeModel.__new__(room.NativeModel)
    GaussianScene.__init__(model,Path(assets['assets'][name]['path']),1600,1200)
    by_view=[]
    for row in views:
        key=row['source_name'];target=next(iter(row['targets'].values()));w,h=target['width'],target['height']
        rgb,depth,alpha=model.sample(np.array(row['base_w2c']),room.intrinsic(row,w,h),w,h)
        gt=arr(planar/'data/images'/key);native=arr(reference/key)
        with Image.open(planar/'data/valid'/key) as mask:keep=np.asarray(mask.convert('L'))>=253
        delta=(native-rgb)[keep]; assert np.isfinite(rgb).all() and np.isfinite(depth).all()
        result=dict(frame=key,sim_vs_capture_psnr_db=psnr((rgb-gt)[keep]),sim_vs_native_png_psnr_db=psnr(delta),sim_vs_native_png_mae_8bit=float(255*np.mean(abs(delta))))
        if name=='planargs_30k':
            with np.load(saved/'geometry'/(Path(key).stem+'.npz')) as geo:
                z=geo['depth'].squeeze();ok=keep&(alpha>.95)&(geo['alpha'].squeeze()>.95)&np.isfinite(z)&(z>0)&(depth>0)
                delta_z=assets['meters_per_source_unit']*abs(z[ok]-depth[ok]);result.update(high_alpha_fraction=float(ok[keep].mean()),depth_difference_median_m=float(np.median(delta_z)),depth_difference_p90_m=float(np.percentile(delta_z,90)))
        by_view.append(result)
    all_models[name]=dict(views=by_view,means={k:float(np.mean([r[k] for r in by_view])) for k in by_view[0] if k!='frame'})
    print(name,all_models[name]['means'],flush=True)
    (root/'checks/render_comparison.json').write_text(json.dumps(all_models,indent=2)+'\n');del model;gc.collect();torch.cuda.empty_cache()

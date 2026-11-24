"""Room-local calibrated rasterization; all inputs reside in the scene bundle."""
from pathlib import Path
import json
import sys
import numpy as np
import torch

ROOT=Path(__file__).resolve().parent
SIM=ROOT.parents[1]
sys.path.insert(0,str(SIM))
from scene import GaussianScene

def read(path):
    return json.loads((ROOT/path).read_text())
def write(path,obj):
    (ROOT/path).write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')
def intrinsic(row,w,h):
    k=next(iter(row['targets'].values()))
    return np.array([[k['fx']*w/k['width'],0,k['cx']*w/k['width']],
                     [0,k['fy']*h/k['height'],k['cy']*h/k['height']],[0,0,1]])
def array(t):
    return t.detach().cpu().numpy()
def tensor(x):
    return torch.as_tensor(np.ascontiguousarray(x),dtype=torch.float32,device='cuda')

class NativeModel(GaussianScene):
    def __init__(self,kind,w,h):
        super().__init__(ROOT/'models'/f'{kind}.ply',w,h)
    @torch.inference_mode()
    def sample(self,W,K,w,h,moments=False):
        adjusted=W.copy()
        adjusted[:3,3]+=W[:3,:3]@self.origin
        view=tensor(adjusted)
        colors=self.colors
        if moments:
            z=self.means@view[2,:3]+view[2,3]
            colors=torch.stack((z,z.square(),torch.ones_like(z)),1)
        pix,alpha,_=self.rasterization(means=self.means,quats=self.quats,scales=self.scales,
                 opacities=self.opacities,colors=colors,viewmats=view[None],Ks=tensor(K)[None],
                 width=w,height=h,packed=False,sh_degree=None if moments else self.sh_degree,
                 near_plane=.01,far_plane=100.,rasterize_mode='classic',render_mode='RGB' if moments else 'RGB+ED')
        a=alpha[0,:,:,0]
        if moments:
            z=pix[0,:,:,0]/a.clamp_min(1e-8)
            std=(pix[0,:,:,1]/a.clamp_min(1e-8)-z.square()).clamp_min(0).sqrt()
            return array(z),array(std),array(a)
        return array(pix[0,:,:,:3].clamp(0,1)),array(pix[0,:,:,3]),array(a)

class RigScene:
    def __init__(self,scene_path,width,height,up_axis,scale,device,*,render_thermal=True):
        self.width,self.height=width,round(width*3/4)
        self.device=torch.device(device)
        self.rgb=NativeModel('rgb',width,self.height)
        self.thermal=NativeModel('thermal',width,self.height) if render_thermal else None
        if Path(scene_path).resolve()!=self.rgb.path or up_axis!='z' or scale!=1.:
            raise ValueError('Use the scene bundle RGB PLY and simulator-frame obstacle coordinates')
        cfg=read('inputs/frame.json')
        self.R=np.array(cfg['rotation']);self.origin=np.array(cfg['origin'])
        self.s=cfg['meters_per_source_unit'];self.bounds=np.array(cfg['bounds_m'])
        rows=read('inputs/manifest.json')['views']
        modalities=('rgb','thermal') if render_thermal else ('rgb',)
        self.ks={m:intrinsic(next(v for v in rows if v['sequence']==62 and v['modality']==m),width,self.height) for m in modalities}
        if render_thermal:
            self.rig=np.array(read('inputs/rig.json')['rgb_to_thermal'])
        self.K=tensor(self.ks['rgb'])[None]
    def metadata(self):
        rig='calibrated RGB to thermal' if self.thermal is not None else 'calibrated RGB'
        return dict(name='room65_selected_anchored',rig=rig,meters_per_source_unit=self.s)
    def render(self,camera):
        camera=np.asarray(camera)
        native=np.eye(4)
        native[:3,:3]=self.R.T@camera[:3,:3]@np.diag([1.,-1.,-1.])
        native[:3,3]=self.R.T@camera[:3,3]/self.s+self.origin
        self.W=np.linalg.inv(native)
        rgb,z,a=self.rgb.sample(self.W,self.ks['rgb'],self.width,self.height)
        if self.thermal is not None:
            self.Wir=self.rig@self.W
            self.ir=self.thermal.sample(self.Wir,self.ks['thermal'],self.width,self.height)[0]
        return rgb,z*self.s,a

"""Standard uncompressed Gaussian PLY loading and GPU inference, no trainer."""

from pathlib import Path
import math

import numpy as np
from plyfile import PlyData
from scipy.spatial.transform import Rotation
import torch

UP_AXES = ('z','y','x','-z','-y','-x')


def up_rotation(axis):
    """Rotate the chosen source up axis into simulator +Z."""
    rotations = {'z':(0,0,0),'y':(np.pi/2,0,0),'x':(0,-np.pi/2,0),
                 '-z':(np.pi,0,0),'-y':(-np.pi/2,0,0),'-x':(0,np.pi/2,0)}
    if axis not in rotations:
        raise ValueError('up_axis must be one of '+', '.join(UP_AXES))
    return Rotation.from_rotvec(rotations[axis]).as_matrix()


def read_gaussians(path):
    """Read INRIA/Nerfstudio 3DGS PLY, including ASCII and binary encodings.

    PLY stores logarithmic scales, opacity logits, wxyz quaternions and SH
    channels in channel-major order. Ordinary point clouds are rejected.
    """
    path = Path(path).expanduser().resolve()
    if not path.is_file() or path.suffix.lower() != '.ply':
        raise ValueError(f'Expected an existing .ply file: {path}')
    try:
        ply = PlyData.read(str(path))
        vertices = ply['vertex'].data
    except Exception as exc:
        raise ValueError(f'Cannot read PLY vertex data: {exc}') from exc
    fields = set(vertices.dtype.names or ())
    required = ['x','y','z','opacity',*[f'scale_{i}' for i in range(3)],*[f'rot_{i}' for i in range(4)]]
    missing = [name for name in required if name not in fields]
    if missing:
        raise ValueError('This is not a supported standard 3DGS PLY. Missing: '+', '.join(missing)+
                         '. Plain point clouds and compressed PLY require conversion first.')
    n = len(vertices)
    if not n:
        raise ValueError('The PLY contains no Gaussians')

    def columns(names,dtype=np.float32):
        try:
            values = np.column_stack([vertices[name] for name in names]).astype(dtype)
        except (ValueError,TypeError) as exc:
            raise ValueError('Gaussian attributes must be scalar numeric values') from exc
        if not np.isfinite(values).all():
            raise ValueError('PLY contains non-finite Gaussian attributes')
        return values

    means = columns(['x','y','z'],np.float64)
    scales = columns([f'scale_{i}' for i in range(3)])
    quats = columns([f'rot_{i}' for i in range(4)])
    if np.any(np.linalg.norm(quats,axis=1)<1e-8):
        raise ValueError('PLY contains zero-length Gaussian quaternions')
    opacity = columns(['opacity'])[:,0]
    if all(f'f_dc_{i}' in fields for i in range(3)):
        dc = columns([f'f_dc_{i}' for i in range(3)])
    elif all(name in fields for name in ['red','green','blue']):
        rgb = columns(['red','green','blue'])
        if all(np.issubdtype(vertices[name].dtype,np.integer) for name in ['red','green','blue']):
            rgb /= 255.
        if np.any((rgb<0)|(rgb>1)):
            raise ValueError('RGB colors must be uint8 [0,255] or floats in [0,1]')
        dc = (rgb-.5)/.28209479177387814
    else:
        raise ValueError('PLY needs f_dc_0..2 or red/green/blue Gaussian colors')
    rest_names = [name for name in fields if name.startswith('f_rest_')]
    if len(rest_names)%3:
        raise ValueError('Invalid SH coefficient count')
    count = len(rest_names)//3+1
    degree = math.isqrt(count)-1
    if (degree+1)**2 != count or not 0<=degree<=4:
        raise ValueError('Supported SH degrees are 0 through 4')
    expected = [f'f_rest_{i}' for i in range(len(rest_names))]
    if set(expected)!=set(rest_names):
        raise ValueError('SH properties must have contiguous f_rest indices')
    rest = columns(expected).reshape(n,3,-1).transpose(0,2,1) if expected else np.empty((n,0,3),dtype=np.float32)
    colors = np.concatenate((dc[:,None],rest),axis=1)
    return {'means':means,'log_scales':scales,'quats':quats,'opacity_logits':opacity,
            'colors':colors,'sh_degree':degree}


class GaussianScene:
    def __init__(self,path,width=480,height=270,up_axis='z',scale=1.,device='cuda'):
        if not np.isfinite(scale) or scale<=0:
            raise ValueError('scale must be a positive number (meters per PLY unit)')
        if not torch.cuda.is_available():
            raise RuntimeError('3DGS rendering requires a CUDA GPU')
        from gsplat import rasterization
        self.rasterization = rasterization
        self.path = Path(path).expanduser().resolve()
        self.name = self.path.name
        self.device = torch.device(device)
        self.width,self.height = width,height
        self.scale,self.up_axis = scale,up_axis
        self.axis_rotation = up_rotation(up_axis)
        data = read_gaussians(self.path)
        self.sh_degree = data['sh_degree']
        self.count = len(data['means'])
        # Keep Gaussians and SH directions in source-axis coordinates. Only
        # transform the camera, so arbitrary up axes do not require SH rotation.
        self.origin = np.quantile(data['means'],[.02,.98],axis=0).mean(axis=0)*scale
        world_means = (data['means'].astype(np.float64)*scale) @ self.axis_rotation.T
        lo,hi = np.quantile(world_means,[.02,.98],axis=0)
        self.bounds = np.stack((lo,hi))
        self.default_spawn = (lo+hi)/2
        self.default_spawn[2] = lo[2]+min(1.25,(hi[2]-lo[2])/2)
        tensor = lambda a: torch.as_tensor(np.ascontiguousarray(a),device=self.device,dtype=torch.float32)
        self.means = tensor(data['means'].astype(np.float64)*scale-self.origin)
        if not torch.isfinite(self.means).all():
            raise ValueError('Scene coordinates exceed rendering precision; reduce the scale')
        self.scales = tensor(data['log_scales']).exp()*scale
        if not torch.isfinite(self.scales).all() or not (self.scales>0).all():
            raise ValueError('PLY has invalid activated Gaussian scales')
        self.quats = tensor(data['quats'])
        self.opacities = tensor(data['opacity_logits']).sigmoid()
        self.colors = tensor(data['colors'])
        # Preserve the previous camera calibration, scaling with output size.
        sx,sy = width/640,height/360
        self.K = tensor([[462.956*sx,0,323.076*sx],[0,463.002*sy,181.184*sy],[0,0,1]])[None]

    def metadata(self):
        return {'name':self.name,'gaussians':self.count,'sh_degree':self.sh_degree,
                'up_axis':self.up_axis,'scale':self.scale,'bounds':self.bounds.tolist(),
                'default_spawn':self.default_spawn.tolist()}

    def camera_in_render_frame(self,camera):
        camera = np.asarray(camera,dtype=np.float64)
        result = np.eye(4)
        result[:3,:3] = self.axis_rotation.T @ camera[:3,:3]
        result[:3,3] = self.axis_rotation.T @ camera[:3,3]-self.origin
        return result

    @torch.no_grad()
    def render(self,camera):
        c2w = torch.tensor(self.camera_in_render_frame(camera),device=self.device,dtype=torch.float32)
        c2w[:3,1:3] *= -1
        view = torch.linalg.inv(c2w)[None]
        rendered,alpha,_ = self.rasterization(
            means=self.means,quats=self.quats,scales=self.scales,
            opacities=self.opacities,colors=self.colors,viewmats=view,Ks=self.K,
            width=self.width,height=self.height,packed=False,sh_degree=self.sh_degree,
            near_plane=.01,far_plane=1e10,rasterize_mode='classic',render_mode='RGB+ED')
        background = rendered.new_tensor([.1490,.1647,.2157])
        rgb = (rendered[0,...,:3]+(1-alpha[0])*background).clamp(0,1)
        depth = rendered[0,...,3]
        depth = torch.where(alpha[0,...,0]>0,depth,depth.max())
        return rgb.cpu().numpy(),depth.cpu().numpy(),alpha[0,...,0].cpu().numpy()

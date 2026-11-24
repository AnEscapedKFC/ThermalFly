"""Export ordinary XYZ/RGB surface points from 3DGS rendered depth.

Splatfacto has no native `ns-export pointcloud` implementation. The optional
Nerfstudio adapter below uses its inference pipeline and calibrated cameras;
the standalone adapter works directly with the simulator's Gaussian PLY.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from pointcloud import open3d_module
from scene import GaussianScene,UP_AXES,up_rotation


def backproject_depth(depth,rgb,alpha,K,camera,stride=4,min_alpha=.95,max_depth=30.):
    """Camera optical-axis Z depth + OpenGL c2w -> surface samples, in meters."""
    depth=np.asarray(depth);alpha=np.asarray(alpha)
    if depth.ndim==3 and depth.shape[-1]==1:depth=depth[...,0]
    if alpha.ndim==3 and alpha.shape[-1]==1:alpha=alpha[...,0]
    rgb=np.asarray(rgb);K=np.asarray(K);camera=np.asarray(camera)
    if (depth.ndim!=2 or alpha.shape!=depth.shape or rgb.shape!=(*depth.shape,3) or
            K.shape!=(3,3) or camera.shape!=(4,4) or stride<1 or
            not np.isfinite(K).all() or K[0,0]<=0 or K[1,1]<=0):
        raise ValueError('Invalid depth, alpha, RGB, intrinsics or camera pose')
    yy,xx=np.mgrid[0:depth.shape[0]:stride,0:depth.shape[1]:stride]
    d=depth[::stride,::stride];a=alpha[::stride,::stride]
    valid=np.isfinite(d)&(d>.05)&(d<=max_depth)&np.isfinite(a)&(a>=min_alpha)
    # gsplat/Nerfstudio use pixel centers for the camera projection.
    optical=np.column_stack(((xx[valid]+.5-K[0,2])*d[valid]/K[0,0],
                             -(yy[valid]+.5-K[1,2])*d[valid]/K[1,1],-d[valid]))
    points=optical@camera[:3,:3].T+camera[:3,3]
    colors=rgb[::stride,::stride][valid]
    finite=np.isfinite(points).all(axis=1)&np.isfinite(colors).all(axis=1)
    return points[finite],np.clip(colors[finite],0,1)


def virtual_cameras(scene,views=72,seed=None):
    """Sample a bounded 3x3 grid at two heights, yawing around each viewpoint.

    PLY has no capture poses or free-space labels. This finite-view coverage is
    an approximation, not a guarantee of a watertight collision surface.
    """
    lo,hi=scene.bounds
    xs=np.linspace(lo[0]+.2*(hi[0]-lo[0]),hi[0]-.2*(hi[0]-lo[0]),3)
    ys=np.linspace(lo[1]+.2*(hi[1]-lo[1]),hi[1]-.2*(hi[1]-lo[1]),3)
    heights=lo[2]+(hi[2]-lo[2])*np.array([.3,.65])
    positions=[scene.default_spawn]
    if seed is not None:positions.insert(0,np.asarray(seed,dtype=float))
    positions.extend(np.array([x,y,z]) for z in heights for x in xs for y in ys)
    # Each selected position sees eight overlapping horizontal headings.
    count=max(2,min(len(positions),int(np.ceil(views/8))))
    positions=[positions[i] for i in np.linspace(0,len(positions)-1,count,dtype=int)]
    for i in range(views):
        p=positions[i%len(positions)];yaw=((i//len(positions))%8)*np.pi/4
        forward=np.array([np.cos(yaw),np.sin(yaw),-.12]);forward/=np.linalg.norm(forward)
        right=np.cross(forward,[0.,0,1]);right/=np.linalg.norm(right)
        up=np.cross(right,forward)
        camera=np.eye(4);camera[:3,:3]=np.column_stack((right,up,-forward));camera[:3,3]=p
        yield camera


def consistent_surface_mask(points,frames,min_views=2,tolerance=.08,min_alpha=.95):
    """Require agreement from distinct camera centers, veto excess free space.

    Rotating one camera in place is not an independent geometric observation.
    Rendered expected depth can float between layers; support and free-space
    tests reduce those artifacts but cannot establish ground-truth occupancy.
    """
    points=np.asarray(points,dtype=float)
    if min_views<1 or tolerance<=0:raise ValueError('Invalid depth-consistency settings')
    groups={}
    for frame in frames:
        key=tuple(np.round(frame[3][:3,3],5))
        groups.setdefault(key,[]).append(frame)
    if len(groups)<min_views:
        raise ValueError('Not enough distinct camera positions for depth consistency; use more views or explicitly --min-views 1')
    support=np.zeros(len(points),dtype=np.int32);free=np.zeros(len(points),dtype=np.int32)
    for group in groups.values():
        supported=np.zeros(len(points),dtype=bool);free_space=np.zeros(len(points),dtype=bool)
        for depth,alpha,K,camera in group:
            depth=np.asarray(depth).reshape(alpha.shape)
            local=(points-camera[:3,3])@camera[:3,:3]
            z=-local[:,2];valid_z=z>.05
            safe_z=np.where(valid_z,z,1.)
            u=np.floor(K[0,0]*local[:,0]/safe_z+K[0,2]).astype(np.int64)
            v=np.floor(-K[1,1]*local[:,1]/safe_z+K[1,2]).astype(np.int64)
            ids=np.flatnonzero(valid_z&(u>=0)&(u<depth.shape[1])&(v>=0)&(v<depth.shape[0]))
            measured=depth[v[ids],u[ids]];confidence=alpha[v[ids],u[ids]]
            opaque=np.isfinite(measured)&(measured>.05)&np.isfinite(confidence)&(confidence>=min_alpha)
            ids=ids[opaque];measured=measured[opaque]
            supported[ids[np.abs(z[ids]-measured)<=tolerance]]=True
            free_space[ids[z[ids]<measured-tolerance]]=True
        support+=supported;free+=free_space&~supported
    return (support>=min_views)&(free<=support)


def write_surface_cloud(points,colors,output,axis_rotation,scale,voxel=.04,metadata=None,
                        frames=None,min_views=2,tolerance=.08,min_alpha=.95):
    """Downsample in metric simulator coordinates; save in source PLY frame."""
    o3d=open3d_module()
    points=np.asarray(points,dtype=float);colors=np.asarray(colors,dtype=float)
    if points.ndim!=2 or points.shape[1]!=3 or colors.shape!=points.shape or not len(points):
        raise ValueError('No valid surface samples; supply better camera poses or adjust export thresholds')
    cloud=o3d.geometry.PointCloud()
    cloud.points=o3d.utility.Vector3dVector(points)
    cloud.colors=o3d.utility.Vector3dVector(colors)
    cloud=cloud.voxel_down_sample(voxel)
    before=len(cloud.points)
    if frames is not None:
        mask=consistent_surface_mask(np.asarray(cloud.points),frames,min_views,tolerance,min_alpha)
        cloud=cloud.select_by_index(np.flatnonzero(mask).tolist())
    count=len(cloud.points)
    if not count:raise ValueError('No surface samples passed multi-view consistency; adjust camera coverage or export settings')
    # Source raw frame makes imported ordinary clouds align with --scene.
    cloud.points=o3d.utility.Vector3dVector(np.asarray(cloud.points)@axis_rotation/scale)
    output=Path(output).expanduser().resolve();output.parent.mkdir(parents=True,exist_ok=True)
    if not o3d.io.write_point_cloud(str(output),cloud,write_ascii=False):
        raise RuntimeError(f'Failed to save point cloud: {output}')
    report={**(metadata or {}),'points':count,'points_before_consistency':before,
            'min_consistent_camera_positions':min_views if frames is not None else None,
            'consistency_tolerance_m':tolerance if frames is not None else None,'frame':'source_model','scale':scale,
            'voxel_m':voxel,'file':str(output),'geometry':'sampled rendered-depth surface, not Gaussian centers'}
    output.with_suffix('.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
    return report


@torch.no_grad()
def export_ply_depth(scene,output,views=72,stride=4,min_alpha=.95,max_depth=30.,voxel=.04,seed=None,progress=None,
                     min_views=2,tolerance=.08):
    if views<4 or stride<1 or not 0<min_alpha<=1 or max_depth<=.05 or voxel<=0:
        raise ValueError('Invalid point-cloud export settings')
    samples=[];colors=[];frames=[]
    for i,camera in enumerate(virtual_cameras(scene,views,seed)):
        rgb,depth,alpha=scene.render(camera)
        points,color=backproject_depth(depth,rgb,alpha,scene.K[0].cpu().numpy(),camera,
                                      stride,min_alpha,max_depth)
        samples.append(points);colors.append(color)
        frames.append((depth,alpha,scene.K[0].cpu().numpy(),camera))
        if progress:progress(i+1,views)
    return write_surface_cloud(np.concatenate(samples),np.concatenate(colors),output,
                               scene.axis_rotation,scene.scale,voxel,
                               {'method':'gsplat_depth','views':views,'pixel_stride':stride,
                                'min_alpha':min_alpha,'max_depth_m':max_depth,'up_axis':scene.up_axis,
                                'source_scene':scene.name,'image_size':[scene.width,scene.height],
                                'seed_position':None if seed is None else np.asarray(seed).tolist()},
                               frames,min_views,tolerance,min_alpha)


@torch.no_grad()
def export_nerfstudio_depth(config,output,views=72,stride=4,min_alpha=.95,max_depth=30.,voxel=.04,up_axis='z',scale=1.,resolution=.5,
                            min_views=2,tolerance=.08):
    """Optional adapter for a locally loadable Nerfstudio Splatfacto pipeline."""
    try:
        from nerfstudio.utils.eval_utils import eval_setup
        from nerfstudio.models.splatfacto import SplatfactoModel
        from nerfstudio.cameras.cameras import CameraType
    except ImportError as exc:
        raise RuntimeError('This export route requires a compatible Nerfstudio installation and its local config/checkpoint; use --scene for a standalone PLY') from exc
    _,pipeline,_,_=eval_setup(Path(config),test_mode='inference')
    if not isinstance(pipeline.model,SplatfactoModel):
        raise ValueError('This adapter expects Splatfacto Z-depth; use native ns-export pointcloud for supported NeRF models')
    dataset=getattr(pipeline.datamanager,'train_dataset',None)
    if dataset is None:dataset=getattr(pipeline.datamanager,'eval_dataset',None)
    if dataset is None or not len(dataset.cameras):raise ValueError('No calibrated Nerfstudio cameras available')
    cameras=dataset.cameras;indices=np.linspace(0,len(cameras)-1,min(views,len(cameras)),dtype=int)
    A=up_rotation(up_axis);samples=[];colors=[];frames=[]
    for index in indices:
        camera=cameras[int(index):int(index)+1].to(pipeline.device)
        if int(camera.camera_type.flatten()[0])!=CameraType.PERSPECTIVE.value:
            raise ValueError('Depth export currently supports perspective cameras only')
        camera.rescale_output_resolution(resolution)
        outputs=pipeline.model.get_outputs_for_camera(camera)
        if 'depth' not in outputs or 'accumulation' not in outputs:
            raise ValueError('Nerfstudio Splatfacto must return depth and accumulation')
        c2w=np.eye(4);c2w[:3]=camera.camera_to_worlds[0].detach().cpu().numpy()
        world_camera=np.eye(4)
        world_camera[:3,:3]=A@c2w[:3,:3]
        world_camera[:3,3]=A@c2w[:3,3]*scale
        depth=outputs['depth'].detach().cpu().numpy().squeeze(-1)*scale
        alpha=outputs['accumulation'].detach().cpu().numpy().squeeze(-1)
        K=camera.get_intrinsics_matrices()[0].detach().cpu().numpy()
        points,color=backproject_depth(depth,outputs['rgb'].detach().cpu().numpy(),
                                      alpha,K,world_camera,stride,min_alpha,max_depth)
        samples.append(points);colors.append(color)
        frames.append((depth,alpha,K,world_camera))
    return write_surface_cloud(np.concatenate(samples),np.concatenate(colors),output,A,scale,voxel,
                               {'method':'nerfstudio_splatfacto_depth','config':str(Path(config).resolve()),
                                'views':len(indices),'pixel_stride':stride,'min_alpha':min_alpha,
                                'max_depth_m':max_depth,'up_axis':up_axis},frames,min_views,tolerance,min_alpha)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    source=parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--scene',type=Path)
    source.add_argument('--nerfstudio-config',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--views',type=int,default=72)
    parser.add_argument('--stride',type=int,default=4)
    parser.add_argument('--alpha',type=float,default=.95)
    parser.add_argument('--max-depth',type=float,default=30.)
    parser.add_argument('--voxel',type=float,default=.04)
    parser.add_argument('--min-views',type=int,default=2,help='Distinct camera positions required to support each point')
    parser.add_argument('--tolerance',type=float,default=.08,help='Metric multi-view depth tolerance')
    parser.add_argument('--up-axis',choices=UP_AXES,default='z')
    parser.add_argument('--scale',type=float,default=1.)
    parser.add_argument('--resolution',type=float,default=.5)
    args=parser.parse_args()
    if (not 4<=args.views<=512 or args.stride<1 or not 0<args.alpha<=1 or
            not np.isfinite([args.max_depth,args.voxel,args.scale,args.resolution,args.tolerance]).all() or
            args.max_depth<=.05 or args.voxel<=0 or args.scale<=0 or args.min_views<1 or args.tolerance<=0 or not .1<=args.resolution<=1):
        parser.error('Invalid point-cloud export arguments')
    if args.output.suffix.lower()!='.ply':parser.error('--output must be an ordinary .ply file')
    torch.set_num_threads(2)
    options=dict(views=args.views,stride=args.stride,min_alpha=args.alpha,max_depth=args.max_depth,voxel=args.voxel,
                 min_views=args.min_views,tolerance=args.tolerance)
    try:
        if args.scene:
            if args.scene.expanduser().resolve()==args.output.expanduser().resolve():parser.error('Do not overwrite the source Gaussian PLY')
            scene=GaussianScene(args.scene,round(640*args.resolution),round(360*args.resolution),args.up_axis,args.scale)
            report=export_ply_depth(scene,args.output,**options)
        else:
            report=export_nerfstudio_depth(args.nerfstudio_config,args.output,up_axis=args.up_axis,
                                          scale=args.scale,resolution=args.resolution,**options)
    except (ValueError,RuntimeError,OSError) as exc:
        parser.exit(1,f'Point-cloud export failed: {exc}\n')
    print(json.dumps(report,indent=2,ensure_ascii=False))


if __name__=='__main__':main()

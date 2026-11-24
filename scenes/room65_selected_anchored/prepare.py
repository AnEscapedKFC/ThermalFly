"""Fuse RGB-rendered surfaces with translated RGB depth priors; certify observed free space independently."""
from collections import Counter
import numpy as np
from scipy.ndimage import minimum_filter,maximum_filter
from scipy.spatial import cKDTree
from plyfile import PlyData,PlyElement
from room import ROOT,read,write,NativeModel,intrinsic,tensor,array
import torch

rows=[r for r in read('inputs/manifest.json')['views'] if r['modality']=='rgb' and r['split']=='train']
planes=read('inputs/planes.json')
up=np.array(planes['ceiling']['normal']); ceiling=-planes['ceiling']['offset']
poses=np.array([np.linalg.inv(r['base_w2c']) for r in rows]);centres=poses[:,:3,3]
if np.median(centres@up)>ceiling:up=-up;ceiling=-ceiling
hist=Counter()
for i,r in enumerate(rows[::4]):
    with np.load(ROOT/'inputs'/r['dense_prior']['path']) as p:
        y,x=np.where(((p['support']>=3)&(p['confidence']>=.6))[::5,::5]);y*=5;x*=5
        k=intrinsic(r,320,240); C=np.linalg.inv(r['base_w2c'])
        ray=np.stack([(x+.5-k[0,2])/k[0,0],(y+.5-k[1,2])/k[1,1],np.ones(len(x))],1)
        z=(ray*p['depth'][y,x,None]@C[:3,:3].T+C[:3,3])@up
        hist.update(np.rint(z[(z<np.percentile(centres@up,1)-1.2)&(z>ceiling-8)]/.05).astype(int))
floor=hist.most_common(1)[0][0]*.05
s=2.8/(ceiling-floor)
normal=np.array(planes['door_wall']['normal']);xaxis=normal-up*(up@normal);xaxis/=np.linalg.norm(xaxis)
R=np.stack((xaxis,np.cross(up,xaxis),up))
origin=np.median(centres,axis=0);origin+=(floor-origin@up)*up
def metric(x):return (x-origin)@R.T*s
def native(x):return (x/s)@R+origin
def save_cloud(path,xyz,color):
    v=np.empty(len(xyz),dtype=[(k,'f4') for k in 'xyz']+[(k,'u1') for k in ('red','green','blue')])
    for j,k in enumerate(v.dtype.names[:3]):v[k]=xyz[:,j]
    for j,k in enumerate(v.dtype.names[3:]):v[k]=color[:,j]
    PlyData([PlyElement.describe(v,'vertex')]).write(ROOT/path)
model=NativeModel('rgb',320,240)
raw=metric(array(model.means)+model.origin)
lo=np.quantile(raw,.002,axis=0)-.40;hi=np.quantile(raw,.998,axis=0)+.40
lo[2]=-.15;hi[2]=2.95
write('inputs/frame.json',dict(rotation=R.tolist(),origin=origin.tolist(),floor_src=float(floor),ceiling_src=float(ceiling),assumed_ceiling_height_m=2.8,meters_per_source_unit=s,bounds_m=[lo.tolist(),hi.tolist()],scale_status='assumed, no surveyed metric reference'))
surf=[];color=[];maps=[]
planes_by_view=np.load(ROOT/'inputs/planes.npz');plane_samples=0
for j,r in enumerate(rows):
    W=np.array(r['base_w2c']);K=intrinsic(r,320,240)
    z,std,a=model.sample(W,K,320,240,moments=True)
    with np.load(ROOT/'inputs'/r['dense_prior']['path']) as p:
        d=p['depth'];valid=(p['support']>=3)&(p['confidence']>=.52)&(d*s>.15)&(d*s<12)
    # Reject silhouettes and mixed layers. Free evidence uses a slightly eroded foreground depth.
    lower=minimum_filter(np.where(valid,d,0),size=3)
    upper=maximum_filter(np.where(valid,d,0),size=3)
    observed=valid&(lower>0)&((upper-lower)*s<.12)
    agreement=observed&(a>.955)&(std*s<.085)&(abs(z-d)*s<(.045+.009*d*s))
    solid=np.where(agreement,z,0).astype('f4')
    if r['view_id'] in planes_by_view:
        known=planes_by_view[r['view_id']]
        verified=known>0
        plane_samples+=int(verified.sum())
        solid[verified]=known[verified]
        lower[verified]=np.where(observed[verified],np.minimum(lower[verified],known[verified]),known[verified])
        observed|=verified
    maps.append((W,K,solid,np.where(observed,lower,0).astype('f4')))
    y,x=np.where(solid[::3,::3]>0);y*=3;x*=3
    if len(x):
        C=poses[j];rays=np.stack([(x+.5-K[0,2])/K[0,0],(y+.5-K[1,2])/K[1,1],np.ones(len(x))],1)
        pts=metric((rays*solid[y,x,None])@C[:3,:3].T+C[:3,3]);use=((pts>=lo)&(pts<=hi)).all(1)
        rgb=model.sample(W,K,320,240)[0]
        surf.append(pts[use].astype('f4'));color.append((rgb[y[use],x[use]]*255).round().astype('u1'))
    if j%100==0:print('projected RGB',j,flush=True)
surf=np.concatenate(surf);color=np.concatenate(color)
vox=.04
keys=np.floor((surf-lo)/vox).astype('i4')
_,idx,inv=np.unique(keys,axis=0,return_index=True,return_inverse=True)
count=np.bincount(inv)
surf=np.stack([np.bincount(inv,weights=surf[:,i])/count for i in range(3)],axis=1)
color=np.stack([np.bincount(inv,weights=color[:,i])/count for i in range(3)],axis=1).astype('u1')
# The flight volume is certified at a separate resolution. No-return/unknown cells never count as free.
step=.10;dims=np.ceil((hi-lo)/step).astype(int)
grid=lo+(np.indices(dims).reshape(3,-1).T+.5)*step
points=np.concatenate([surf,grid])
X=tensor(native(points))
N=len(points);device=X.device
hit=torch.full((N,3),len(rows),dtype=torch.int32,device=device)
free=torch.full_like(hit,len(rows))
ci=metric(centres);apart=np.linalg.norm(ci[:,None]-ci[None,:],axis=-1)>=.22
ind=torch.as_tensor(np.pad(apart,((0,0),(0,1)),constant_values=True),device=device)
def vote(bank,mask,i):
    ids=torch.where(mask&(bank[:,-1]==len(rows)))[0]
    old=bank[ids].long()
    accepted=ind[i,old].all(1)
    ids=ids[accepted];slot=(old[accepted]!=len(rows)).sum(1)
    bank[ids,slot]=i
for i,(W,K,Z,D) in enumerate(maps):
    c=X@tensor(W[:3,:3]).T+tensor(W[:3,3]);z=c[:,2];den=z.clamp_min(.001)
    px=(c[:,0]/den*K[0,0]+K[0,2]).floor().long();py=(c[:,1]/den*K[1,1]+K[1,2]).floor().long()
    inside=(z*s>.14)&(px>=1)&(px<319)&(py>=1)&(py<239);px=px.clamp(0,319);py=py.clamp(0,239)
    obs=tensor(Z)[py,px];far=tensor(D)[py,px]
    vote(hit,inside&(obs>0)&((obs-z).abs()*s<.045),i)
    vote(free,inside&((far-z)*s>.15),i)
    if i%150==0:print('depth votes',i,flush=True)
hit=array((hit!=len(rows)).sum(1));free=array((free!=len(rows)).sum(1));L=len(surf)
keep=(hit[:L]>=3)&(free[:L]<3)
# Remove isolated small groups. Preserve planar and thin connected support.
tree=cKDTree(surf[keep]);neighbors=tree.query_ball_point(surf[keep],.10,return_length=True)
ids=np.where(keep)[0][neighbors>=5];xyz=surf[ids];save_cloud('collision/rgb_surface_m.ply',xyz,color[ids])
fg=(free[L:]>=3).reshape(dims);np.savez_compressed(ROOT/'collision/free_volume.npz',origin=lo,step=step,free=fg)
raw_key=np.floor((raw-lo)/step).astype(int);in_map=((raw_key>=0)&(raw_key<dims)).all(1);raw_ids=np.where(in_map)[0];raw_ids=raw_ids[fg[tuple(raw_key[raw_ids].T)]]
gaps,_=cKDTree(xyz).query(raw[raw_ids]);void=raw_ids[gaps>.20]
np.savez_compressed(ROOT/'collision/probes.npz',raw_float_examples=raw[void[::max(1,len(void)//5000)]])
write('collision/build.json',dict(rgb_gaussians=len(raw),training_rgb_views=len(rows),voxel_m=vox,min_separated_origins=3,min_origin_separation_m=.22,certified_plane_pixel_observations=plane_samples,surface_candidates=L,accepted_points=len(xyz),supported_but_free_veto=int(((hit[:L]>=3)&(free[:L]>=3)).sum()),free_cells=int(fg.sum()),all_cells=int(fg.size),raw_centres_in_certified_free_over_20cm_from_surface=len(void),source='RGB 3DGS rendered depth moments + RGB SfM-aligned predicted depth, no range sensor',floor_peak=floor,ceiling=ceiling,scale=s))
print('finished',len(xyz),fg.sum(),len(void),flush=True)

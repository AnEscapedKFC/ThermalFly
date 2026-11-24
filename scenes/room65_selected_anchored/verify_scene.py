"""Exercise paired intrinsics/extrinsics against three held-out captures and probe physical surfaces."""
from room import *
from pointcloud import ObstacleCloud
from PIL import Image,ImageDraw
from scipy.spatial import cKDTree

def main():
    scene=RigScene(ROOT/'models/rgb.ply',640,480,'z',1,'cuda')
    rows=read('inputs/manifest.json')['views']
    obs=ObstacleCloud(ROOT/'collision/rgb_surface_m.ply')
    tree=cKDTree(obs.points)
    ref_checks=[]; examples=[]; landmarks={}
    for seq,name,box in [(62,'cup',(171,129,201,182)),(137,'ac',(73,22,169,72)),(288,'charger',(172,50,208,96))]:
        W=np.array(next(v for v in rows if v['sequence']==seq and v['modality']=='rgb')['base_w2c'])
        C=np.linalg.inv(W);sim=np.eye(4)
        sim[:3,3]=scene.s*scene.R@(C[:3,3]-scene.origin)
        sim[:3,:3]=scene.R@C[:3,:3]@np.diag([1.,-1.,-1.])
        rgb,_,_=scene.render(sim)
        assert np.allclose(scene.W,W) and np.allclose(scene.Wir,scene.rig@W)
        receipt={'sequence':seq}
        panels=[]
        for m,im in [('rgb',rgb),('thermal',scene.ir)]:
            v=next(v for v in rows if v['sequence']==seq and v['modality']==m)
            path=next(iter(v['targets'].values()))['path']
            gt=np.asarray(Image.open(ROOT/'inputs'/path).convert('RGB').resize((640,480),Image.Resampling.BOX))/255.
            receipt[m+'_psnr']=float(-10*np.log10(np.mean((im-gt)**2)))
            panels.extend([gt,im])
            if im.std()<.013 or receipt[m+'_psnr']<17:raise ValueError(f'Check calibrated rendering: {receipt}')
        ref_checks.append(receipt);examples.append(panels)
        z=scene.rgb.sample(scene.Wir,scene.ks['thermal'],640,480)[1]
        u,l,U,L=box;u,U=round(u*640/300),round(U*640/300);l,L=round(l*480/225),round(L*480/225)
        d=np.median(z[l:L,u:U]);K=scene.ks['thermal'];p=np.array([((u+U)/2-K[0,2])/K[0,0],((l+L)/2-K[1,2])/K[1,1],1])*d
        C=np.linalg.inv(scene.Wir);landmarks[name]=(scene.s*scene.R@(C[:3,:3]@p+C[:3,3]-scene.origin)).tolist()
    sheet=Image.new('RGB',(4*240,3*202),'white');dr=ImageDraw.Draw(sheet)
    for j,panels in enumerate(examples):
        for i,pic in enumerate(panels):
            key=['RGB photo','RGB simulator','IR photo','IR simulator'][i]
            dr.text((i*240+3,j*202+2),f"{ref_checks[j]['sequence']}  {key}",fill='black')
            sheet.paste(Image.fromarray((255*np.clip(pic,0,1)).astype('u1')).resize((240,180)),(i*240,j*202+20))
    (ROOT/'checks').mkdir(exist_ok=True);sheet.save(ROOT/'checks/heldout_rig.jpg')
    contacts={}
    for label,seq,x,y in [('wall',159,190,126),('ceiling',344,145,42),('table',314,203,190)]:
        v=next(r for r in rows if r['modality']=='rgb' and r['sequence']==seq);C=np.linalg.inv(v['base_w2c']);k=intrinsic(v,320,240)
        p=np.load(ROOT/'inputs'/v['dense_prior']['path']);d=p['depth'][y-3:y+4,x-3:x+4];depth=np.median(d[d>0])
        direction=C[:3,:3]@np.array([(x+.5-k[0,2])/k[0,0],(y+.5-k[1,2])/k[1,1],1.])
        target=(C[:3,3]+direction*depth-scene.origin)@scene.R.T*scene.s
        metric_ray=scene.R@direction;metric_ray/=np.linalg.norm(metric_ray)
        distance,idx=tree.query(target)
        if distance>.28:raise ValueError((label,'surface not near RGB observation',distance))
        if label=='table':metric_ray=np.array([0.,0.,-1.])
        if label=='ceiling':metric_ray=np.array([0.,0.,1.])
        start=target-.85*metric_ray;end=target+.22*metric_ray
        hit=obs.sweep(start,end);near=float(obs.distance(start)['safety_clearance'])
        assert hit and near>0 and np.linalg.norm(np.array(hit['point'])-target)<.65,(label,near,hit)
        contacts[label]=dict(rgb_train_sequence=seq,target_m=target.tolist(),nearest_surface_m=float(distance),start_clearance_m=near,sweep_hit=hit)
    write('checks/scene.json',dict(heldout=ref_checks,positive_contacts=contacts,landmarks_m=landmarks,observation_frame='RGB optical, IR W2C = calibrated T @ RGB W2C in native SfM units',rgb_to_ir_baseline_m=float(np.linalg.norm(scene.rig[:3,3])*scene.s)))
    print(ref_checks,landmarks,contacts,flush=True)
if __name__=='__main__':main()

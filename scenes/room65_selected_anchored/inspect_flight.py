"""Check time/rig poses and produce a navigational reference sheet and plan view."""
import argparse,csv,json
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image,ImageDraw
from scipy.spatial.transform import Rotation
from plyfile import PlyData
from room import *
from cameras import fpv_camera

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--output',type=Path,default=ROOT/'flight');out=ap.parse_args().output
    rows=[json.loads(line) for line in (out/'frames.jsonl').open()];profile=json.loads((out/'summary.json').read_text())
    frame=read('inputs/frame.json');R=np.array(frame['rotation']);s=frame['meters_per_source_unit'];rig=np.array(read('inputs/rig.json')['rgb_to_thermal'])
    errors=[];rgb_errors=[];poses=[];raw=[]
    for i,r in enumerate(rows):
        assert r['frame']==i and abs(r['sim_time']-i/20)<1e-7 and not r['collision'] and r['substeps']==(10 if i else 0)
        W=np.array(r['rgb_world_to_camera_native']);Wir=np.array(r['thermal_world_to_camera_native'])
        errors.append(float(np.max(abs(Wir@np.linalg.inv(W)-rig))))
        st=r['state'];poses.append(st['position'])
        c=fpv_camera(np.r_[st['position'],st['velocity'],st['quaternion_wxyz'],st['angular_velocity']]).numpy()
        cf=np.linalg.inv(W)
        rgb_errors.append(max(np.max(abs(cf[:3,:3]-R.T@c[:3,:3]@np.diag([1,-1,-1]))),
                              np.max(abs(R@(cf[:3,3]-frame['origin'])*s-c[:3,3]))))
    assert len(rows)==profile['frames'] and max(errors)<1e-5 and max(rgb_errors)<1e-5
    tlist=[0,6,11,15,20,26];cap=cv2.VideoCapture(str(out/'rgb_thermal_fpv.mp4'))
    assert cap.isOpened() and int(cap.get(cv2.CAP_PROP_FRAME_COUNT))==len(rows) and abs(cap.get(cv2.CAP_PROP_FPS)-20)<1e-6
    canvas=Image.new('RGB',(2*400*2,3*325),'#eeefeb');dr=ImageDraw.Draw(canvas)
    for j,t in enumerate(tlist):
        cap.set(cv2.CAP_PROP_POS_FRAMES,t*20);ok,img=cap.read();assert ok and img.shape==(480,1280,3)
        a,b=np.split(img,2,axis=1);assert min(a.std(),b.std())>2
        x=(j%2)*800;y=(j//2)*325
        for k,p in enumerate((a,b)):
            dr.text((x+400*k+3,y+4),f't = {t} s  |  '+['RGB','Thermal'][k],fill='black')
            canvas.paste(Image.fromarray(cv2.cvtColor(p,cv2.COLOR_BGR2RGB)).resize((400,300)),(x+400*k,y+20))
    cap.release();canvas.save(out/'fpv_montage.jpg',quality=87)
    p=PlyData.read(ROOT/'collision/rgb_surface_m.ply')['vertex'].data
    xyz=np.column_stack([p[k] for k in 'xyz']);poses=np.array(poses);use=(xyz[:,2]>.3)&(xyz[:,2]<2.2)
    fig,ax=plt.subplots(figsize=(7,6),layout='constrained');ax.scatter(*xyz[use,:2].T,s=1,c='0.68',label='RGB surfaces (0.3–2.2 m)')
    ax.plot(*poses[:,:2].T,color='#167787',lw=3,label='Flown trajectory')
    for t in tlist:
        x,y=poses[t*20,:2];yaw=np.deg2rad(rows[t*20]['state']['euler_deg'][2]);ax.arrow(x,y,.26*np.cos(yaw),.26*np.sin(yaw),width=.012,color='#b54625',length_includes_head=True)
    ax.scatter(*poses[0,:2],s=65,marker='o',c='#167787',edgecolors='white',label='Start')
    ax.scatter(*poses[-1,:2],s=90,marker='*',c='#b54625',edgecolors='white',label='Finish')
    for name,point in read('checks/scene.json')['landmarks_m'].items():
        ax.plot(*point[:2],marker='+',color='black');ax.annotate(name,point[:2],xytext=(5,4),textcoords='offset points',fontsize=9)
    ax.set(xlabel='Simulator x (nominal m)',ylabel='Simulator y (nominal m)',aspect='equal');ax.legend(loc='lower left',fontsize=8);fig.savefig(out/'flight_map.png',dpi=150);plt.close(fig)
    # Is the AC target inside BOTH actual cameras at the final hover?
    target=np.array(read('checks/scene.json')['landmarks_m']['ac'])
    native=R.T@target/s+frame['origin'];proj={}
    for key,field in [('rgb','rgb_world_to_camera_native'),('thermal','thermal_world_to_camera_native')]:
        W=np.array(rows[-1][field]);q=W[:3,:3]@native+W[:3,3];pixel=np.array(profile['camera_intrinsics'][key])@q;u,v=pixel[:2]/pixel[2];proj[key]=[u,v]
        assert q[2]>0 and 0<=u<640 and 0<=v<480
    audit=dict(frames=len(rows),fps=20,max_rig_matrix_residual=max(errors),max_rgb_body_mount_residual=max(rgb_errors),final_ac_landmark_pixel=proj)
    (out/'alignment_check.json').write_text(json.dumps(audit,indent=2)+'\n')
    print(audit)
if __name__=='__main__':main()

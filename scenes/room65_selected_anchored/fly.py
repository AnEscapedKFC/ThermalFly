"""Fly and record the real simulator with closed-loop body-rate/thrust commands."""
import argparse
import csv
from contextlib import ExitStack
import subprocess
from scipy.spatial.transform import Rotation
from scipy.spatial import cKDTree
from room import *
from engine import PolicyFlightSimulator
from record_sequence import FpvVideoWriter,verify_video,ffmpeg_executable,CSV_FIELDS,trajectory_row
from pointcloud import ObstacleCloud

class KnownSpace:
    def __init__(self):
        g=np.load(ROOT/'collision/free_volume.npz')
        self.free=g['free'];self.step=float(g['step']);self.lo=g['origin']
    def contains_box(self,p,R,half,extra=0.):
        # Enumerate every voxel which might intersect the expanded oriented box.
        half=half+extra
        bound=abs(R)@half
        ij=np.floor((p-bound-self.lo)/self.step).astype(int)
        end=np.floor((p+bound-self.lo)/self.step).astype(int)
        if (ij<0).any() or (end>=self.free.shape).any():return False
        cells=(np.indices(end-ij+1).reshape(3,-1).T+ij)
        centres=self.lo+(cells+.5)*self.step
        touches=(abs((centres-p)@R)<=half+self.step/2*abs(R).sum(0)+1e-7).all(1)
        return bool(self.free[tuple(cells[touches].T)].all())

def commands(k,t):
    j=min(max(0,np.searchsorted(k[:,0],t,side='right')-1),len(k)-2)
    a,b=k[j:j+2];dt=b[0]-a[0];u=np.clip((t-a[0])/dt,0,1)
    s=u**3*(10-15*u+6*u*u);v=30*u*u*(1-u)**2/dt;acc=60*u*(1-u)*(1-2*u)/dt**2
    return a[1:]+(b[1:]-a[1:])*s,(b[1:]-a[1:])*v,(b[1:]-a[1:])*acc,int(j)

def controller(sim,receipt,k,t,*,sample=commands):
    goal,vel,acc,j=sample(k,t)
    st=receipt['state'];R=Rotation.from_quat(np.array(st['quaternion_wxyz'])[[1,2,3,0]]).as_matrix()
    feed=acc[:3]+2.8*(goal[:3]-st['position'])+3.*(vel[:3]-st['velocity'])
    force=feed+[0.,0.,9.81];up=force/np.linalg.norm(force)
    yaw=np.deg2rad(goal[3]);left=np.cross(up,[np.cos(yaw),np.sin(yaw),0]);left/=np.linalg.norm(left)
    wanted=np.column_stack((np.cross(left,up),left,up))
    rate=5.*Rotation.from_matrix(R.T@wanted).as_rotvec()+R.T@np.array([0.,0.,np.deg2rad(vel[3])])
    throttle=force@R[:,2]*sim.parameters['mass']/sim.parameters['max_thrust']
    return np.r_[np.clip(rate,-.9,.9),np.clip(throttle,0.,1.)],j

def negative_check(obs,known):
    pool=np.load(ROOT/'collision/probes.npz')['raw_float_examples']
    tree=cKDTree(obs.points);gaps=tree.query(pool)[0]
    for j in np.argsort(-gaps):
        p=pool[j]
        if not .9<p[2]<2.1:continue
        a,b=p-[.16,0,0],p+[.16,0,0];I=np.eye(3)
        if not all(known.contains_box(c,I,obs.safety_half_size) for c in [a,p,b]):continue
        if obs.sweep(a,b) is None and obs.distance(p)['safety_clearance']>.1:
            return dict(raw_gaussian_center_m=p.tolist(),nearest_obstacle_m=float(gaps[j]),sweep_start_m=a.tolist(),sweep_end_m=b.tolist(),sweep_hit=False,known_free=True)
    raise RuntimeError('No sufficiently supported negative control')

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,default=ROOT/'flight');args=parser.parse_args()
    route=read('route.json');k=np.array(route['knots']);out=args.output.resolve()
    if out.exists() and any(out.iterdir()):raise FileExistsError(out)
    cloud=ROOT/'collision/rgb_surface_m.ply';obs=ObstacleCloud(cloud);known=KnownSpace()
    neg=negative_check(obs,known)
    for t in np.arange(0,k[-1,0]+.025,.05):
        goal,_,_,j=commands(k,t);rot=Rotation.from_euler('z',goal[3],degrees=True).as_matrix()
        if not known.contains_box(goal[:3],rot,obs.safety_half_size,.04):raise ValueError(f'Unobserved planned body volume at {t}: {goal}')
    out.mkdir(parents=True,exist_ok=True)
    total=int(round(k[-1,0]*20))
    with PolicyFlightSimulator(ROOT/'models/rgb.ply',point_cloud=cloud,config='nominal',seed=65,spawn=k[0,1:4],yaw=k[0,-1],resolution=1,scene_factory=RigScene) as sim,ExitStack() as stack:
        vids=[FpvVideoWriter(out/name,640,480) for name in ['rgb_fpv.mp4','thermal_fpv.mp4']]
        for writer in vids:stack.callback(writer.close)
        trajectory=csv.writer(stack.enter_context((out/'trajectory.csv').open('w',newline='')));trajectory.writerow(CSV_FIELDS)
        states=stack.enter_context((out/'frames.jsonl').open('w'))
        r=sim.last_receipt;previous=None;length=0.;minclear=np.inf;minbody=np.inf;max_speed=0.;max_err=0.;j=0
        for i in range(total+1):
            if i:
                cmd,j=controller(sim,r,k,(i-.5)/20);r=sim.step(cmd,format='body_rates_thrust')
            if r['terminated'] or r['truncated'] or r['frame_id']!=i:raise RuntimeError((i,r['reason']))
            state=r['state'];p=np.array(state['position']);quat=state['quaternion_wxyz'];R=Rotation.from_quat(np.array(quat)[[1,2,3,0]]).as_matrix()
            # Certify an envelope covering the whole next/previous control tick,
            # including rotation. The engine checks obstacle sweeps every 5 ms.
            if state['speed']>.8 or np.linalg.norm(state['angular_velocity'])>1.:raise RuntimeError('Audit envelope exceeded')
            if not known.contains_box(p,R,obs.safety_half_size,.075):raise RuntimeError(f'Unknown body margin at frame {i}: {p}')
            if previous is not None:length+=np.linalg.norm(p-previous)
            previous=p;max_speed=max(max_speed,state['speed']);goal=commands(k,i/20)[0]
            max_err=max(max_err,np.linalg.norm(p-goal[:3]));minclear=min(minclear,r['min_sampled_safety_clearance']);minbody=min(minbody,r['collision']['clearance'])
            vids[0].write(sim.last_rgb);vids[1].write(sim.scene.ir)
            trajectory.writerow(trajectory_row(r,i,j))
            record=dict(frame=i,sim_time=r['sim_time'],segment=j,state=state,raw_action=r['raw_action'],applied_control=r['applied_control'],min_safety_clearance_m=r['min_sampled_safety_clearance'],collision=r['collision']['in_collision'],substeps=r['substeps_executed'],rgb_world_to_camera_native=sim.scene.W.tolist(),thermal_world_to_camera_native=sim.scene.Wir.tolist())
            states.write(json.dumps(record,allow_nan=False)+'\n')
            if i%100==0:print(f'{i}/{total} state={p.round(2)} clearance={minclear:.3f}',flush=True)
        final_speed=state['speed'];end_error=float(np.linalg.norm(p-k[-1,1:4]))
        assert end_error<.08 and final_speed<.06
        cameras={m:K.tolist() for m,K in sim.scene.ks.items()}
    pair=out/'rgb_thermal_fpv.mp4'
    subprocess.run([ffmpeg_executable(),'-v','error','-n','-i',str(out/'rgb_fpv.mp4'),'-i',str(out/'thermal_fpv.mp4'),'-filter_complex','hstack=inputs=2','-an','-c:v','libx264','-threads','4','-crf','17','-pix_fmt','yuv420p','-movflags','+faststart',str(pair)],check=True)
    checks={file.name:verify_video(file,total+1) for file in sorted(out.glob('*.mp4'))}
    report=dict(instruction=route['instruction'],frames=total+1,sim_seconds=r['sim_time'],fps=20,integrator_steps=total*10,path_m=length,min_step_safety_clearance_m=minclear,min_frame_body_clearance_m=minbody,peak_speed_mps=max_speed,max_waypoint_tracking_error_m=max_err,final_position_error_m=end_error,final_speed_mps=final_speed,collision_events=0,unknown_body_envelope_frames=0,negative_control=neg,video_decodes=checks,camera_intrinsics=cameras,policy='state-feedback body-rate/thrust controller with scripted waypoints; no learned VLN policy')
    (out/'summary.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print('finished',out,flush=True)
if __name__=='__main__':main()

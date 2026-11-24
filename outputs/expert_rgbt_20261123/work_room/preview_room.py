"""Check reference-flight views before launching the batch."""
from pathlib import Path
import sys,json
from PIL import Image,ImageDraw
import numpy as np
from scipy.spatial.transform import Rotation
project=Path(__file__).resolve().parents[3]; root=project/'scenes/room65_selected_anchored';sys.path.insert(0,str(root))
from fly import RigScene, commands
from cameras import fpv_camera
dataset=Path(__file__).resolve().parents[1]; land=json.loads((dataset/'work_room/landmarks.json').read_text()); scene=RigScene(root/'models/rgb.ply',320,240,'z',1.,'cuda',render_thermal=False)
def assess(scene,W,name):
 target=np.array(land[name]['center_w_m']); p=scene.R.T@target/scene.s+scene.origin; q=(W@np.r_[p,1])[:3];uv=scene.ks['rgb']@q; return [*(uv[:2]/uv[2]),q[2]*scene.s]
for route in sorted((dataset/'routes').glob('room_*.json')):
 spec=json.loads(route.read_text());k=np.array(spec['knots']); pkeys=np.array(spec['camera_pitch_keys'])
 if 'orbit_center_w_m' in spec:times=[0,k[-1,0]/4,k[-1,0]/2,3*k[-1,0]/4,k[-1,0]]
 else:times=np.round(np.linspace(0,k[-1,0],min(15,max(7,len(k)))),1)
 n=len(times);sheet=Image.new('RGB',(4*320,264*((n+3)//4)),'white');dr=ImageDraw.Draw(sheet);projs=[]
 for i,t in enumerate(times):
  goal=commands(k,t)[0];q=Rotation.from_euler('z',goal[3],degrees=True).as_quat()[[3,0,1,2]]
  mount=np.array(fpv_camera(np.r_[goal[:3],np.zeros(3),q]));angle=np.interp(t,pkeys[:,0],pkeys[:,1]);mount[:3,:3]=mount[:3,:3]@Rotation.from_euler('x',angle,degrees=True).as_matrix()
  rgb=scene.render(mount)[0];im=Image.fromarray((rgb*255).round().astype('uint8'));x,y=(i%4)*320,(i//4)*264;sheet.paste(im,(x,y));dr.text((x+4,y+241),f'{spec["id"]}  {t:.1f} s',fill='black')
  dct={n:list(map(float,assess(scene,scene.W,n))) for n in ('laptop','ac','sofa')};projs.append([float(t),dct])
  for name,col in [('laptop','#ffb332'),('ac','#00eeef'),('sofa','#53fb31')]:
   u,v,z=dct[name]
   if 0<u<319 and 0<v<239 and z>0:
    dr.ellipse((x+u-3,y+v-3,x+u+3,y+v+3),outline=col,width=1)
 sheet.save(dataset/'work_room'/(spec['id']+'.jpg'),quality=92);(dataset/'work_room'/(spec['id']+'.projections.json')).write_text(json.dumps(projs))
 print(spec['id'],flush=True)

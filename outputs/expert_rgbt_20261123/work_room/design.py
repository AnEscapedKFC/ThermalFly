"""Construct and check meeting-room expert references in the calibrated simulator frame."""
from pathlib import Path
import json, math, sys
import numpy as np
from scipy.spatial.transform import Rotation
project=Path(__file__).resolve().parents[3]
scene=project/'scenes/room65_selected_anchored'
sys.path.insert(0,str(scene))
from fly import KnownSpace, commands
from pointcloud import ObstacleCloud
from record_survey_contacts import itineraries
dset=Path(__file__).resolve().parents[1]
routes=dset/'routes'
work=dset/'work_room'
data=json.loads((scene/'checks/scene.json').read_text())['landmarks_m']
sofa=[-.66,2.76,.72]
ac=data['ac']
laptop=[-.1477,-.0836,.7953]
table=[-.16,-.28,.68]
landmarks={
 'conference_table':{'center_w_m':table, 'description':'Meeting table; centre of survey region from validated table surface probe.','evidence':'scenes/room65_selected_anchored/checks/scene.json; positive_contacts.table'},
 'laptop':{'center_w_m':laptop,'description':'Open dark laptop on the conference table; screen center.','evidence':'Triangulated from native room RGB source frames at 320x240: 314 (193,129), 288 (138,76), 159 (197,198); camera-ray residuals 0.005, 0.011, 0.010 m. Frame 314 dense depth independently gives (-0.151,-0.071,0.791) m.'},
 'sofa':{'center_w_m':sofa, 'description':'Black sofa against wall, seat/backrest area.','evidence':'Prior verified sofa-view survey, SOFA coordinate; sofa identifiable in RGB source frames 137 and 314.'},
 'ac':{'center_w_m':ac,'description':'Wall-mounted air conditioner above the black sofa.','evidence':'scenes/room65_selected_anchored/checks/scene.json; landmark from calibrated capture sequence 137.'}}
(work/'landmarks.json').write_text(json.dumps(landmarks,ensure_ascii=False,indent=2)+'\n')
known=KnownSpace();cloud=ObstacleCloud(scene/'collision/rgb_surface_m.ply')
def ang(p,obj):return float(np.rad2deg(np.arctan2(obj[1]-p[1],obj[0]-p[0])))
def yaw_near(old,new):return old+(new-old+180)%360-180
def point(pos,yaw,pitch,stage):return [*pos,yaw,pitch,stage]
def objpoint(pos,obj,pitch,stage):return point(pos,ang(pos,obj),pitch,stage)
def elev(pos,obj):
 delta=np.array(obj)-pos
 return float(np.rad2deg(np.arctan2(delta[2],np.linalg.norm(delta[:2]))))
def at(pos,obj,stage):return objpoint(pos,obj,elev(pos,obj)-8,stage)
plans={}
def plan(id,title,en,cn,base,w,stages,**extra):
 # Keep one continuous unwrapped yaw branch and a one-second final settle.
 knots=[];pitches=[];sg=[];prevstage=None
 for pt in w:
  p=list(pt[:3]); yaw,pitch,stage=pt[3:]
  if not knots:
   t=0.
  else:
   a=knots[-1];yaw=yaw_near(a[4],yaw);delta=np.array(p)-a[1:4]
   dt=max(1.05,1.875*np.linalg.norm(delta)/.54,1.875*abs(yaw-a[4])/31,1.875*abs(pitch-pitches[-1][1])/39)
   t=round((a[0]+dt)*20)/20
  # stage is the segment arriving at this target. Transitions start at the departing knot.
  start=knots[-1][0] if knots else 0.
  if stage!=prevstage:
   lm,pred=stages[stage];sg.append({'id':stage,'at_s':start,'landmark':lm,'predicate':pred});prevstage=stage
  knots.append([t,*p,yaw]);pitches.append([t,pitch])
 knots.append([knots[-1][0]+1.5,*knots[-1][1:]]);pitches.append([knots[-1][0],pitches[-1][1]])
 spec={'id':id,'scene':'room','family':base,'instruction':en,'instruction_zh':cn,'title':title,'knots':knots,'camera_pitch_keys':pitches,'subgoals':sg,**extra}
 plans[id]=spec;return spec
loop=np.array(itineraries()['01_room_survey']['knots'])[1:16,1:3]
def orbit(i,level,reverse=False,variable=False):
 route=loop[::-1] if reverse else loop
 positions=[]
 for j,xy in enumerate(route):
  z=level if not variable else 1.53+.19*np.sin(2*np.pi*j/(len(route)-1))
  p=[*xy,round(z,3)]
  positions.append(at(p,laptop,'orbit'))
 mode='changing altitude' if variable else ('upper' if level>1.52 else 'lower')
 circtitle={'low':'低高度绕桌巡航','high':'高高度反向绕桌巡航','vary':'变高度绕桌巡航'}[i]
 en=f'Fly one complete inward-looking circuit around the conference table at {level:.2f} metres and hover at the start.' if not variable else 'Complete one circuit around the conference table, smoothly climb on the first half and descend on the second half, then return to the initial height and hover.'
 if reverse: en=en.replace('circuit','clockwise circuit')
 cn={'low':'以较低的固定高度绕会议桌巡航一周，保持面向桌面，回到起点悬停。','high':'以较高的固定高度沿相反方向绕会议桌巡航一周，面向桌面并回到起点悬停。','vary':'绕会议桌巡航一周，前半程经过高位、后半程经过低位，最后恢复起始高度并悬停。'}[i]
 s=plan('room_orbit_'+i,circtitle,en,cn,'room_table_circuit',positions,{'orbit':('conference_table','complete_360_degree_orbit')})
 s['orbit_center_w_m']=table;s['orbit_closed_at_s']=s['knots'][-2][0]
orbit('low',1.34);orbit('high',1.72,True);orbit('vary',1.53,variable=True)
def inspect(id,tgt,poses,en,cn):
 obj=landmarks[tgt]['center_w_m']; positions=[at(p,obj,'inspect') for p in poses]
 return plan(id,cn,en,cn,id,positions,{'inspect':(tgt,'approach_and_observe')},inspection_landmarks=[tgt])
inspect('room_inspect_laptop','laptop',[[1.57,.80,1.51],[.92,.62,1.45],[.45,.43,1.37],[.10,.46,1.32]],'Approach the open laptop on the conference table. Keep it in view, look down at its screen and hover.','接近会议桌上打开的笔记本，俯视观察屏幕并悬停。')
inspect('room_inspect_sofa','sofa',[[.85,.60,1.49],[.40,.82,1.49],[-.02,1.10,1.48],[-.48,1.46,1.39],[-.87,1.80,1.30]],'Approach the black sofa from the table side, lower the viewing height, then hover while looking at the sofa.','从会议桌一侧飞近黑色沙发，降低观察高度并面向沙发悬停。')
inspect('room_inspect_ac','ac',[[.73,-.82,1.46],[.63,-.32,1.54],[.51,.26,1.61],[.38,.81,1.66],[.29,1.41,1.66]],'Approach the wall-mounted air conditioner above the sofa. Climb slightly, then hover with it centered.','向沙发上方的挂壁空调靠近，略微爬升并朝向空调悬停。')
# Grounded long programs with an initially-away viewing direction, a vertical ascent, and a deliberate multi-heading scan.
start=[1.57,-.56,1.32];high=[1.57,-.56,1.65]
w=[point(start,-83,-8,'climb'),point(high,-83,-8,'climb'),point(high,-125,-12,'scan'),point(high,-170,-19,'scan'),point(high,-213,-29,'scan')]
for p in [[1.20,-.31,1.59],[.87,.10,1.48],[.51,.40,1.40],[.10,.46,1.33]]:w.append(at(p,laptop,'observe_laptop'))
for p in [[-.22,.81,1.43],[-.45,1.25,1.42],[-.67,1.66,1.35]]:w.append(at(p,sofa,'observe_sofa'))
s=plan('room_search_laptop_sofa','升高旋转搜索笔记本，再观察沙发','Rise and turn slowly to search for the open laptop. Approach and inspect its screen, then turn toward the black sofa and move closer before hovering.','先升高并旋转寻找会议桌上的笔记本，靠近观察屏幕，再转向黑色沙发抵近悬停。','room_search_laptop_sofa',w,{'climb':('laptop','ascend_to_search'),'scan':('laptop','scan_for_target'),'observe_laptop':('laptop','approach_and_observe'),'observe_sofa':('sofa','approach_and_observe')},inspection_landmarks=['laptop','sofa'],search_evidence={'target':'laptop','search_stage':'scan','approach_stage':'observe_laptop'})
start=[-.82,.68,1.29];high=[-.82,.68,1.62]
w=[point(start,-92,-23,'climb'),point(high,-92,-21,'climb'),point(high,-48,-9,'scan'),point(high,-6,-1,'scan'),point(high,35,6,'scan'),point(high,68,8,'scan')]
for p in [[-.51,.82,1.64],[-.08,1.04,1.65],[.20,1.39,1.67]]:w.append(at(p,ac,'observe_ac'))
for p in [[.64,1.10,1.63],[1.24,.83,1.55],[1.45,.37,1.50],[1.26,-.34,1.46],[.87,-.57,1.44],[.43,-.60,1.37]]:w.append(at(p,laptop,'observe_laptop'))
s=plan('room_search_ac_laptop','升高旋转搜索挂壁空调，再巡视桌端和笔记本','Climb and scan the walls for the wall-mounted air conditioner. Approach it, then go around the table end and look down at the open laptop before hovering.','升高并旋转搜索墙上的空调，接近观察，再绕过桌端，俯视笔记本悬停。','room_search_ac_laptop',w,{'climb':('ac','ascend_to_search'),'scan':('ac','scan_for_target'),'observe_ac':('ac','approach_and_observe'),'observe_laptop':('laptop','approach_and_observe')},inspection_landmarks=['ac','laptop'],search_evidence={'target':'ac','search_stage':'scan','approach_stage':'observe_ac'})
for name,spec in plans.items():
 a=np.array(spec['knots']);ms=100;unknown=[]
 for t in np.arange(0,a[-1,0]+.001,.10):
  target,_,_,_ = commands(a,t);rot=Rotation.from_euler('z',target[3],degrees=True)
  clear=cloud.distance(target[:3],rot.as_quat()[[3,0,1,2]])['safety_clearance'];ms=min(ms,clear)
  if not known.contains_box(target[:3],rot.as_matrix(),cloud.safety_half_size,extra=.015):unknown.append([round(t,2),target.round(2).tolist()])
 print(name,round(a[-1,0],2),round(ms,3),len(unknown),'first/last',unknown[:1],unknown[-1:],flush=True)
 (routes/(name+'.json')).write_text(json.dumps(spec,ensure_ascii=False,indent=2)+'\n')

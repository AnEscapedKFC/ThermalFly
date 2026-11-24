from pathlib import Path
import hashlib, json
from PIL import Image,ImageDraw,ImageFont

out=Path(__file__).resolve().parent;scene=out.parents[1]/'scenes/room65_selected_anchored'
before=json.loads((out/'before_sha256.json').read_text())
def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for chunk in iter(lambda:f.read(1024*1024),b''): h.update(chunk)
 return h.hexdigest()
changes=[p for p,h in before.items() if not (scene/p).is_file() or digest(scene/p)!=h]
new=[str(p.relative_to(scene)) for part in ('runs','flight','collision') for p in (scene/part).rglob('*') if p.is_file() and str(p.relative_to(scene)) not in before]
result=dict(files_checked=len(before),changed=changes,added_in_archived_directories=new,passed=not (changes or new))
(out/'integrity.json').write_text(json.dumps(result,indent=2)+'\n')
assert result['passed'],result
r=json.loads((out/'comparison.json').read_text());assert r['passed']
tasks={row['task']:row for row in r['tasks']}
labels={'02_table_contact':'桌面','03_wall_contact':'墙面','04_floor_contact':'地板','03_sofa_left_contact':'沙发左侧','04_sofa_right_contact':'沙发右侧'}
font=ImageFont.truetype('/mnt/c/Windows/Fonts/msyh.ttc',20)
cw,ch=320,240
sheet=Image.new('RGB',(3*cw,46+len(tasks)*(ch+39)),(247,246,242));draw=ImageDraw.Draw(sheet)
for i,t in enumerate(('原锚定 RGB 24k','当前 PlanarGS RGB 30k','当前红外（原模型）')):
 draw.text((i*cw+8,10),t,fill='#202c35',font=font)
for i,(task,label) in enumerate(labels.items()):
 row=tasks[task];old=Path(row['original_action_file']).parent;new=out/task
 y=46+i*(ch+39);draw.text((8,y+4),f"{label}    第 {row['replay']['frame_id']} 帧    {row['replay']['time']:.3f} s",fill='#202c35',font=font)
 with Image.open(old/'contact.jpg') as im,Image.open(new/'contact.jpg') as im2:
  w,h=im.size;panels=[im.crop((0,0,w//2,h)),im2.crop((0,0,w//2,h)),im2.crop((w//2,0,w,h))]
  for j,panel in enumerate(panels):sheet.paste(panel.resize((cw,ch),Image.Resampling.LANCZOS),(cw*j,y+39))
sheet.save(out/'contact_comparison.jpg',quality=94)
print(result)

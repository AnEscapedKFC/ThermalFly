import base64, hashlib, io, json, urllib.request
from pathlib import Path

import numpy as np
from PIL import Image

out=Path(__file__).resolve().parent
def fetch(port,path,body=None,raw=False):
    req=urllib.request.Request(f'http://127.0.0.1:{port}'+path,data=json.dumps(body).encode() if body is not None else None,
                              headers={'Content-Type':'application/json'} if body is not None else {})
    with urllib.request.urlopen(req,timeout=90) as resp:
        assert resp.status==200
        data=resp.read()
        return data if raw else json.loads(data)
def image(x):
    return np.asarray(Image.open(io.BytesIO(base64.b64decode(x))).convert('RGB'))

home=fetch(7890,'/',raw=True).decode()
assert 'PlanarGS RGB' in home
catalog=fetch(7890,'/api/catalog')
assert len(catalog)>=9
wp=fetch(7890,'/api/waypoints/04_sofa_right_contact',raw=True)
f=fetch(7890,'/api/frame')
assert f['status']=='ready'
rgb,ir=image(f['rgb']),image(f['thermal'])
assert rgb.shape==ir.shape==(480,640,3) and min(rgb.std(),ir.std())>4
Image.fromarray(np.hstack([rgb,ir])).save(out/'drone_view.jpg',quality=93)
home=fetch(7891,'/',raw=True).decode();assert 'PlanarGS RGB' in home
def render(**params):return fetch(7891,'/api/step',params)
ref='rgb_c00_0000288'
solo=render(reset='rgb_c5_p2_0006504660',view_mode='rig')
assert min(image(solo['thermal']).std(),image(solo['rgb']).std())>4
start=render(reset=ref,view_mode='rig');moved=render(keys=['d'],move=.04,view_mode='rig')
shared=render(reset=ref,view_mode='shared');end=render(reset=ref,view_mode='rig')
rgb,ir=image(end['rgb']),image(end['thermal'])
assert rgb.std()>4 and ir.std()>4 and tuple(end['sizes']['rgb'])==tuple(rgb.shape[1::-1]) and tuple(end['sizes']['thermal'])==tuple(ir.shape[1::-1])
assert np.linalg.norm(np.subtract(start['position'],moved['position']))>.039 and np.allclose(start['position'],end['position'])
assert start['rgb']!=moved['rgb'] and start['thermal']!=moved['thermal'] and start['rgb']==end['rgb'] and start['thermal']==end['thermal']
assert end['thermal']!=shared['thermal'] and end['rgb']==shared['rgb']
Image.fromarray(np.hstack([rgb,ir])).save(out/'paired_view.jpg',quality=93)
(out/'viewers.json').write_text(json.dumps(dict(drone=dict(url='http://127.0.0.1:7890/',catalog=len(catalog),frame=f['index'],size=[f['width'],f['height']],status=f['status'],
sample_waypoint_sha256=hashlib.sha256(wp).hexdigest()),dual=dict(url='http://127.0.0.1:7891/',sizes=end['sizes'],reference=ref,rig_shared_switch=True,move_reset=True,rgb_only_anchor_tested=True)),indent=2)+'\n')
print((out/'viewers.json').read_text())

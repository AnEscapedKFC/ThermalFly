"""Check first-contact points against two RGB sofa observations and index new flights."""
import csv
import json
from pathlib import Path
import argparse
import cv2
import numpy as np
from scipy.spatial import cKDTree
from PIL import Image,ImageDraw
from room import ROOT,intrinsic,read
from sofa_flights import SOFA,routes
from record_survey_contacts import SURFACE,ObstacleCloud,dump
from verify_survey_contacts import audit

# Image polygons in the displayed 320 x 240 calibration views, traced on the couch.
REGIONS={137:[(46,160),(54,156),(144,161),(150,166),(148,179),(157,181),(152,198),(92,200),(91,193),(32,189),(24,184),(29,178),(44,175)],
         314:[(65,46),(72,40),(170,42),(176,47),(175,62),(188,67),(188,81),(150,83),(144,79),(61,78),(51,75),(52,68),(64,64)]}

def main(batch):
    manifest=read('inputs/manifest.json')['views'];cfg=read('inputs/frame.json')
    matrix=np.array(cfg['rotation']);origin=np.array(cfg['origin']);scale=cfg['meters_per_source_unit']
    tree=cKDTree(ObstacleCloud(SURFACE).points)
    checks={};samples=[];hits=[];source_images={}
    for idx in REGIONS:
        p=next((ROOT/'inputs/rgb').glob(f'*{idx:07d}.png'))
        source_images[idx]=Image.open(p).convert('RGB').resize((320,240))
        ImageDraw.Draw(source_images[idx]).line(REGIONS[idx]+[REGIONS[idx][0]],fill='#68f2c9',width=1)
    for name,spec in routes().items():
        folder=batch/name;s,frames=audit(folder,tree)
        info=dict(duration_s=s['duration_s'],waypoints=s['waypoint_count'],frames=len(frames),min_clearance_m=s['min_sampled_safety_clearance_m'])
        if s['contact']:
            surface=np.array(s['contact']['point']); hits.append(surface);evidence={}
            native=np.r_[matrix.T@surface/scale+origin,1.]
            for idx,region in REGIONS.items():
                row=next(v for v in manifest if v['sequence']==idx and v['modality']=='rgb')
                xyz=(np.array(row['base_w2c'])@native)[:3];uv=intrinsic(row,320,240)@xyz
                pixel=tuple(map(float,uv[:2]/uv[2]))
                margin=cv2.pointPolygonTest(np.array(region,'float32'),pixel,True)
                evidence[idx]=dict(pixel=pixel,inside_px=margin)
                if margin<0:raise ValueError((name,idx,pixel,margin))
                pen=ImageDraw.Draw(source_images[idx]);x,y=pixel;col='yellow' if len(hits)==1 else 'cyan'
                pen.ellipse((x-3,y-3,x+3,y+3),outline=col,width=2)
                pen.text((x+4,y-8),str(len(hits)),fill=col)
            info.update(sofa_observations=evidence,contact=s['contact'],local_support_points=s['surface_points_within_15cm_at_contact'])
        else:
            seen={}
            cal=json.loads((folder/'calibration.json').read_text())
            for m in ('rgb','thermal'):
                count=0
                for r in frames:
                    C=np.array(r[m+'_c2w_sim_m']);q=C[:3,:3].T@(SOFA-C[:3,3])
                    pix=np.array(cal[m+'_K'])@q
                    if q[2]>0 and 0<=pix[0]/pix[2]<640 and 0<=pix[1]/pix[2]<480:count+=1
                seen[m]=count
            info['sofa_in_frustum_frames']=seen
            if min(seen.values())<len(frames)*.22:raise ValueError((name,seen))
        video=cv2.VideoCapture(str(folder/'rgb_thermal_fpv.mp4'));thumbs=[]
        for idx in [0,len(frames)//2,len(frames)-1]:
            video.set(cv2.CAP_PROP_POS_FRAMES,idx);ok,img=video.read()
            if not ok:raise ValueError('Cannot decode requested sample')
            thumbs.append(Image.fromarray(cv2.cvtColor(cv2.resize(img,(384,144),interpolation=cv2.INTER_AREA),cv2.COLOR_BGR2RGB)))
        video.release()
        panel=Image.new('RGB',(1152,162),(22,34,40));pen=ImageDraw.Draw(panel)
        for j,thumb in enumerate(thumbs):panel.paste(thumb,(384*j,18))
        pen.text((8,2),name,fill='white');samples.append(panel);panel.save(folder/'preview.jpg',quality=91);checks[name]=info
    checks['contact_separation_m']=float(np.linalg.norm(hits[0]-hits[1]))
    if checks['contact_separation_m']<.55:raise ValueError('Contact points too close')
    proof=Image.new('RGB',(640,240));proof.paste(source_images[137]);proof.paste(source_images[314],(320,0))
    proof.save(batch/'sofa_contacts_in_photos.jpg',quality=93)
    sheet=Image.new('RGB',(1152,162*len(samples)))
    for i,im in enumerate(samples):sheet.paste(im,(0,162*i))
    sheet.save(batch/'flight_previews.jpg',quality=89)
    dump(batch/'sofa_verification.json',checks); print(checks)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('batch',type=Path);main(p.parse_args().batch)


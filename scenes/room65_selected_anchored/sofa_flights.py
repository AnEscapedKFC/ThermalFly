"""Prepare and record the sofa flight set in the existing calibrated scene."""
import argparse
import tempfile
from pathlib import Path
import numpy as np
from record_survey_contacts import ROOT,SURFACE,CHARGER,KnownSpace,ObstacleCloud,check_plan,capture

SOFA=np.array([-.66,2.76,.72])

def survey(label, positions, focuses):
    positions=np.asarray(positions); focuses=np.asarray(focuses)
    delta=focuses-positions
    yaw=np.rad2deg(np.unwrap(np.arctan2(delta[:,1],delta[:,0])))
    times=[0.]
    for j in range(1,len(yaw)):
        duration=max(2.,1.875*np.linalg.norm(positions[j]-positions[j-1])/.315,1.875*abs(yaw[j]-yaw[j-1])/20.)
        times.append(round((times[-1]+duration)*20)/20)
    knots=np.column_stack((times,positions,yaw)).tolist()
    return dict(label=label,mode='subject',knots=knots,camera_targets=np.column_stack((times,focuses)).tolist())

def routes():
    lateral=[(-1.76,1.08,1.52),(-1.22,.64,1.52),(-.40,.55,1.56),(.48,.73,1.51),(1.10,1.10,1.46),(1.37,1.62,1.41),(1.37,1.62,1.41)]
    bridge=[(.95,.45,1.46),(.41,.68,1.41),(.17,.84,1.48),(.03,1.07,1.5),(-.51,1.50,1.38),(-1.15,1.83,1.31),(-1.15,1.83,1.31)]
    return {
        '01_sofa_arc':survey('沙发多角度横移',lateral,[SOFA]*len(lateral)),
        '02_table_to_sofa':survey('从桌面转向沙发',bridge,[CHARGER,CHARGER,[-.65,.71,.77],SOFA,SOFA,SOFA,SOFA]),
        '03_sofa_left_contact':dict(label='沙发左侧正向接触',mode='subject',target=[-.98,2.81,.81],tolerance=.41,
            camera_targets=[[0,-1.,2.82,.80]],knots=[[0,-1.23,.99,1.43,85],[1.5,-1.23,.99,1.43,85],[8,-1.23,1.69,.91,90],[16,-1.23,2.86,.91,90]]),
        '04_sofa_right_contact':dict(label='沙发右侧下降接触',mode='subject',target=[-.08,2.79,.80],tolerance=.46,
            camera_targets=[[0,-.08,2.75,.66]],knots=[[0,.15,1.39,1.51,95],[1.4,.15,1.39,1.51,95],[8.8,.12,2.48,1.42,90],[16,.12,2.48,.46,90]])}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('out',type=Path);p.add_argument('--plan-only',action='store_true');args=p.parse_args()
    known=KnownSpace();cloud=ObstacleCloud(SURFACE);plans=routes()
    for name,spec in plans.items():
        check_plan(spec,cloud,known);print(name,spec['knots'][-1][0],flush=True)
    if not args.plan_only:
        args.out.mkdir(parents=True,exist_ok=True)
        for name,spec in plans.items():
            dest=args.out/name
            if dest.exists():raise FileExistsError(dest)
            with tempfile.TemporaryDirectory(dir=args.out,prefix='.'+name) as tmp:
                path=Path(tmp)/name;path.mkdir();capture(path,spec,cloud,known);path.rename(dest)


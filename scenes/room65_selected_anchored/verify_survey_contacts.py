"""Audit synchronized records and build the batch index from completed flights."""
import argparse
import csv
import json
from itertools import combinations
from pathlib import Path

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw
from scipy.spatial import cKDTree

from fly import KnownSpace, negative_check
from record_survey_contacts import CHARGER, SURFACE, ObstacleCloud, dump, itineraries


def audit(folder,cloud_tree):
    records=[json.loads(l) for l in (folder/'frames.jsonl').read_text().splitlines()]
    with (folder/'waypoints.csv').open() as f:
        waypoints=list(csv.DictReader(f))
    summary=json.loads((folder/'summary.json').read_text())
    calibration=json.loads((folder/'calibration.json').read_text())
    rig=np.array(calibration['ir_from_rgb_native'])
    assert len(records)==summary['frames']==min(x['decoded_frames'] for x in summary['video_decode'].values())
    times=np.array([r['sim_time'] for r in records])
    assert times[0]==0 and np.all(np.diff(times)>0)
    expected=np.arange(int(np.floor((times[-1]+1e-7)/.3))+1)
    assert len(expected)==len(waypoints)==summary['waypoint_count']
    for j,(i,w) in enumerate(zip(expected,waypoints)):
        r=records[i*6]
        assert int(w['frame_id'])==i*6 and int(w['waypoint_id'])==j
        assert abs(float(w['sim_time_s'])-i*.3)<1e-8 and abs(r['sim_time']-i*.3)<1e-7
        assert np.allclose([float(w[n]) for n in ['x_m','y_m','z_m','qw','qx','qy','qz']],
                            r['state']['position']+r['state']['quaternion_wxyz'],atol=1e-10,rtol=0)
    for i,r in enumerate(records):
        assert r['frame_id']==r['step_id']==i and r['episode_id']==summary['episode_id']
        assert r['collision']['enabled'] and not r['truncated']
        if i==0:
            assert r['raw_action'] is None
        else:
            assert len(r['raw_action'])==len(r['applied_control'])==4
            assert abs(times[i]-times[i-1]-r['dt_actual'])<1e-7
        assert np.allclose(rig@np.array(r['rgb_w2c_native']),r['thermal_w2c_native'],atol=1e-8)
    contacts=[r for r in records if r['collision']['event']]
    assert len(contacts)==(0 if summary['contact'] is None else 1)
    if contacts:
        terminal=contacts[0]; e=terminal['collision']['event']
        assert terminal is records[-1] and terminal['terminated'] and terminal['reason']=='collision'
        assert abs(e['time']-times[-1])<1e-7 and terminal['state']['speed']==0
        assert cloud_tree.query(e['point'])[0]<1e-5
        assert np.linalg.norm(np.array(terminal['state']['position'])-e['position'])<1e-7
        summary['surface_points_within_15cm_at_contact']=len(cloud_tree.query_ball_point(e['point'],.15))
    else:
        assert all(not r['terminated'] for r in records)
    return summary,records


def preview(folder,records,report):
    capture=cv2.VideoCapture(str(folder/'rgb_thermal_fpv.mp4'))
    if report['contact']:
        samples=[0,len(records)//2,len(records)-1]
    else:
        az=[np.arctan2(r['state']['position'][1]-CHARGER[1],r['state']['position'][0]-CHARGER[0]) for r in records if r['sim_time']<=report['circuit_end_s']+1e-7]
        turned=np.rad2deg(np.unwrap(az)-az[0])
        samples=[int(np.argmin(abs(turned-a))) for a in range(0,360,45)]+[len(records)-1]
    width,height=448,168; sheet=Image.new('RGB',(width*3,(height+23)*((len(samples)+2)//3)),(26,28,31))
    pen=ImageDraw.Draw(sheet)
    for j,i in enumerate(samples):
        capture.set(cv2.CAP_PROP_POS_FRAMES,i)
        ok,img=capture.read()
        if not ok: raise ValueError(f'Cannot retrieve frame {i}')
        x=(j%3)*width;y=(j//3)*(height+23)
        arr=cv2.cvtColor(cv2.resize(img,(width,height),interpolation=cv2.INTER_AREA),cv2.COLOR_BGR2RGB)
        sheet.paste(Image.fromarray(arr),(x,y))
        label=f"t = {records[i]['sim_time']:.2f} s"
        if report['contact'] and j==len(samples)-1: label+='   |   FIRST CONTACT'
        pen.text((x+7,y+height+4),label,fill='white')
    capture.release()
    sheet.save(folder/'preview.jpg',quality=91)


def map_figure(batch,flights,cloud):
    fig,ax=plt.subplots(figsize=(9,6.5),layout='constrained')
    points=cloud.points[(cloud.points[:,2]>.13)&(cloud.points[:,2]<2.4)]
    ax.scatter(*points[:,:2].T,s=1,color='#aab4bd',alpha=.16,rasterized=True)
    colors=['#005e79','#cd5a0a','#7c379b','#3e832b']
    labels=['Room survey','Table contact','Wall contact','Floor contact']
    for (name,(summary,rec)),color,label in zip(flights.items(),colors,labels):
        p=np.array([r['state']['position'] for r in rec])
        ax.plot(*p[:,:2].T,label=label,color=color,lw=2)
        ax.scatter(*p[0,:2],facecolor='white',edgecolor=color,s=32,zorder=5)
        if summary['contact']:
            xyz=summary['contact']['point'];ax.scatter(*xyz[:2],marker='X',s=90,color=color,edgecolor='white',zorder=6)
        else:
            for r in rec[::125]:
                C=np.array(r['rgb_c2w_sim_m'])
                ax.arrow(*C[:2,3],*(.23*C[:2,2]),width=.012,color=color,alpha=.75,length_includes_head=True)
    ax.scatter(*CHARGER[:2],marker='*',s=145,color='#bd162f',edgecolor='white',label='Desk charger',zorder=5)
    ax.set(xlabel='Simulator X (m)',ylabel='Simulator Y (m)',aspect='equal',
           title='Recorded body paths and first contact points')
    ax.legend(loc='upper right',ncols=1,fontsize=9);ax.grid(alpha=.17)
    fig.savefig(batch/'flight_paths.png',dpi=160);plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('batch',type=Path);args=p.parse_args()
    cloud=ObstacleCloud(SURFACE);tree=cKDTree(cloud.points)
    flights={}
    for name in itineraries():
        folder=args.batch/name
        report,records=audit(folder,tree)
        preview(folder,records,report);flights[name]=(report,records)
    if len({s['episode_id'] for s,_ in flights.values()})!=len(flights):raise ValueError('Duplicated episode')
    pairs={}
    hits={n:s['contact']['point'] for n,(s,_) in flights.items() if s['contact']}
    for a,b in combinations(hits,2): pairs[a+' / '+b]=float(np.linalg.norm(np.subtract(hits[a],hits[b])))
    assert min(pairs.values())>1.
    index=dict(flights={n:s for n,(s,_) in flights.items()},contact_pair_distances_m=pairs,
               min_separation_m=min(pairs.values()),surplus_gaussian_negative_control=negative_check(cloud,KnownSpace()))
    dump(args.batch/'verification.json',index);map_figure(args.batch,flights,cloud)
    lines=['# 会议室双模态巡航与碰撞记录','',
           '四次飞行分别从各自的初始状态启动。左画面为 RGB，右画面为红外伪彩。视频均为 20 fps，每路 640 × 480。控制频率为 20 Hz，动力学与机身扫掠检测为 200 Hz。',
           '', '| 飞行 | 实际时长 | 视频帧 | 0.3 秒航点 | 视频与记录 |','|---|---:|---:|---:|---|']
    for n,(s,_) in flights.items():
        lines.append(f"| {s['label']} | {s['duration_s']:.3f} s | {s['frames']} | {s['waypoint_count']} | [双模态视频]({n}/rgb_thermal_fpv.mp4) · [缩略图]({n}/preview.jpg) · [航点]({n}/waypoints.csv) · [逐帧数据]({n}/frames.jsonl) |")
    lines+=['','[实际轨迹和碰撞位置](flight_paths.png) · [完整核验数据](verification.json)','',
        '## 数据读取','',
        '- 每个目录含 RGB、红外和同步并排视频。视频从帧 0 计数，三种视频的同一帧共享 frames.jsonl 中的同一帧号。碰撞步可能提前终止，最后一帧精确的仿真时刻取 sim_time。首帧为初始化观测，raw_action 为空。后续帧的 raw_action 是产生这一帧的三轴机体角速度指令和归一化推力，applied_control 是经延时与滤波后应用的指令。',
        '- waypoints.csv 从 0.00 秒开始每 0.30 秒取一次实际状态，包含从 0 起算的航点序号、仿真时刻、视频帧号、机体位置、机体到世界的四元数（w,x,y,z）、世界速度与共用支架俯仰角。碰撞终点的精确时刻和坐标另存于 contact.json。plan.json 保存驾驶参考轨迹。',
        '- 世界坐标 Z 向上，机体 X 向前、Y 向左、Z 向上。每帧 rgb_c2w_sim_m 与 thermal_c2w_sim_m 为光学相机到仿真世界的 4 × 4 位姿，长度单位为场景米，光学轴采用右、下、前。rgb_w2c_native 和 thermal_w2c_native 为原始重建坐标下的世界到光学相机矩阵。两路内参和刚性标定在每段 calibration.json 中。整套支架共用俯仰转动，巡航时俯仰跟随桌面充电器，碰撞段以固定目标俯角观察各自表面。',
        '- 扫掠采用多视角融合的 RGB 表面点云、0.60 × 0.60 × 0.12 m 机体和各面 0.05 m 的安全余量。首次接触即停止。contact.json 的 point 为表面接触点，position 为停止时机体中心，两者均为仿真世界坐标。尺度按 2.80 m 层高换算，缺少现场实测标尺。',
        '- verification.json 包含全片解码计数、航点与逐帧对齐核验后的汇总、各接触点间距，以及一条穿过已排除悬浮高斯的无碰撞扫掠对照。charger_in_*_frustum_frames 仅计投影进入画幅的帧数，遮挡或重建模糊会影响可辨性。','']
    (args.batch/'README.md').write_text('\n'.join(lines))
    print({n:(s['duration_s'],s['frames'],s['waypoint_count'],s['contact']) for n,(s,_) in flights.items()})
    print('min_separation_m',min(pairs.values()))


if __name__=='__main__':
    main()


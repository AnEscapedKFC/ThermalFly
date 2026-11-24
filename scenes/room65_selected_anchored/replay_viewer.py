"""Local action-driven RGB/IR viewer, with a single owner of the simulator."""
import argparse
import base64
import io
import json
import logging
import queue
import threading
import time
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, Response, jsonify, request, send_from_directory
from plyfile import PlyData

from action_replay import ROOT,parse_tape,make_sim,execute_frame,native_optical_to_metric
from record_survey_contacts import SURFACE

WEB=ROOT/'replay_web'
ORIGINAL=ROOT/'runs/20261119_survey_contacts'
SOFAS=ROOT/'runs/20261119_sofa_views'
CATALOG={p.name:(p,label) for base,names in [
    (ORIGINAL,{'01_room_survey':'桌面环绕 · 原始巡航','02_table_contact':'桌面接触','03_wall_contact':'墙面接触','04_floor_contact':'地面接触'}),
    (SOFAS,{'01_sofa_arc':'沙发多角度横移','02_table_to_sofa':'桌面转向沙发','03_sofa_left_contact':'沙发左侧接触','04_sofa_right_contact':'沙发右侧接触',
            '05_original_actions_replay':'桌面环绕 · 复刻'})]
    for name,label in names.items() for p in [base/name] if (p/'frames.jsonl').is_file()}


def jpeg(arr):
    u8=np.rint(arr*255).astype('uint8')
    ok,data=cv2.imencode('.jpg',cv2.cvtColor(u8,cv2.COLOR_RGB2BGR),[cv2.IMWRITE_JPEG_QUALITY,84])
    if not ok:raise RuntimeError('JPEG encoding failed')
    return base64.b64encode(data).decode('ascii')


def read_sample(name):
    with (CATALOG[name][0]/'frames.jsonl').open('rb') as f:return parse_tape(f)


class Player:
    def __init__(self):
        self.commands=queue.Queue(maxsize=8)
        self.lock=threading.Lock()
        self.latest={'serial':0,'status':'loading'}
        self.thread=threading.Thread(target=self.run,daemon=True)
        self.thread.start()

    def publish(self,new_frame=True):
        r=self.sim.last_receipt
        data={'status':self.status,'index':r['step_id'],'total':len(self.tape[2])-1,'time':r['sim_time'],
              'name':self.name,'episode':r['episode_id'],'p':r['state']['position'],'yaw':r['state']['euler_deg'][2],
              'speed':r['state']['speed'],'clearance':r['collision']['safety_clearance'],'contact':r['collision']['event']}
        if new_frame:
            scene=self.sim.scene
            z=cv2.resize(self.sim.last_depth,(80,60),interpolation=cv2.INTER_NEAREST)
            data.update(rgb=jpeg(self.sim.last_rgb),thermal=jpeg(scene.ir),rgb_pose=native_optical_to_metric(scene,scene.W).tolist(),
                        K=scene.ks['rgb'].tolist(),width=scene.width,height=scene.height,
                        depth=base64.b64encode(np.rint(np.clip(z,0,30)*100).astype('<u2').tobytes()).decode('ascii'))
        with self.lock:
            self.latest={**self.latest,**data,'serial':self.latest['serial']+1,'error':None}

    def run(self):
        try:
            self.tape=read_sample('01_room_survey')
            self.sim=make_sim(self.tape)
            self.name=CATALOG['01_room_survey'][1]; self.status='ready'; self.publish()
            due=time.monotonic()
            while True:
                try:
                    cmd=self.commands.get(timeout=max(0.,due-time.monotonic()) if self.status=='playing' else None)
                except queue.Empty:
                    # frame_id spans resets while step_id starts at zero for each loaded tape.
                    r=execute_frame(self.sim,self.tape[2][self.sim.step_id+1])
                    if r['terminated'] or r['truncated']:self.status='collision' if r['reason']=='collision' else 'stopped'
                    elif r['step_id']+1==len(self.tape[2]):self.status='finished'
                    self.publish();due=max(due+.05,time.monotonic());continue
                action,payload,expected,reply=cmd
                try:
                    if expected and self.sim.episode_id!=expected:raise ValueError('回放已更换，请刷新后再操作')
                    if action in ('start','restart'):
                        tape,name=payload if action=='start' else (self.tape,self.name)
                        p,yaw,rows=tape
                        self.sim.scene.pitch=rows[0][2]
                        self.sim.limits['max_steps']=len(rows)+8
                        self.sim.reset(seed=65,spawn=p,yaw=yaw)
                        self.tape=tape;self.name=name
                        self.status='stopped' if self.sim.terminated else 'playing'
                        self.publish()
                    elif action=='pause' and self.status=='playing':
                        self.status='paused';self.publish(False)
                    elif action=='play' and self.status=='paused':
                        self.status='playing';self.publish(False)
                    else:raise ValueError('此状态下无法执行该操作')
                    due=time.monotonic()+.05;reply.put((True,self.sim.episode_id))
                except (ValueError,KeyError) as err:reply.put((False,str(err)))
        except Exception as err:
            import traceback
            traceback.print_exc()
            with self.lock:self.latest={**self.latest,'status':'error','serial':self.latest['serial']+1,'error':str(err)}

    def call(self,cmd,payload=None,expected=None):
        if not self.thread.is_alive():raise RuntimeError('模拟器不可用')
        reply=queue.Queue(1);self.commands.put_nowait((cmd,payload,expected,reply))
        ok,value=reply.get(timeout=40)
        if not ok:raise ValueError(value)
        return value


def create_app(player):
    app=Flask(__name__,static_folder=None);app.config['MAX_CONTENT_LENGTH']=64*1024*1024
    @app.get('/')
    def index():return send_from_directory(WEB,'index.html')
    @app.get('/assets/<path:name>')
    def asset(name):return send_from_directory(WEB,name)
    @app.get('/api/catalog')
    def catalog():return jsonify([{'id':n,'name':l} for n,(p,l) in CATALOG.items()])
    @app.get('/api/waypoints/<name>')
    def wp(name):
        if name not in CATALOG:return jsonify(error='未知记录'),404
        return send_from_directory(CATALOG[name][0],'waypoints.csv',mimetype='text/csv')
    v=PlyData.read(SURFACE)['vertex'].data
    cloud=np.empty(len(v),dtype=[('xyz','<f4',(3,)),('rgb','u1',(3,)),('pad','u1')])
    cloud['xyz']=np.stack([v[k] for k in ('x','y','z')],1);cloud['rgb']=np.stack([v[k] for k in ('red','green','blue')],1);cloud['pad']=0
    binary=cloud.tobytes()
    @app.get('/api/surface')
    def surface():return Response(binary,mimetype='application/octet-stream',headers={'Cache-Control':'public, max-age=86400'})
    @app.get('/api/frame')
    def frame():
        with player.lock: snap=player.latest
        if str(snap['serial'])==request.args.get('after'):return Response(status=204)
        return jsonify(snap)
    @app.post('/api/start')
    def start():
        name=request.args.get('sample')
        if name is not None:
            if name not in CATALOG:raise ValueError('未知示例')
            tape=read_sample(name);label=CATALOG[name][1]
        else:
            tape=parse_tape(io.BytesIO(request.get_data(cache=False)))
            label=request.args.get('name','动作记录')[:150]
        return jsonify(episode=player.call('start',(tape,label)))
    @app.post('/api/control/<action>')
    def control(action):
        if action not in ('restart','pause','play'):raise ValueError('未知回放操作')
        return jsonify(episode=player.call(action,expected=request.args.get('episode')))
    for exc in (ValueError,TypeError,KeyError,UnicodeError,OverflowError):
        app.register_error_handler(exc,lambda e:(jsonify(error=str(e)),400))
    for exc in (RuntimeError,queue.Empty,queue.Full):
        app.register_error_handler(exc,lambda e:(jsonify(error=str(e) or '模拟器忙，请稍后重试'),503))
    return app


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--port',type=int,default=7890);args=parser.parse_args()
    logging.getLogger('werkzeug').setLevel(logging.WARNING)
    create_app(Player()).run(host='127.0.0.1',port=args.port,threaded=True,use_reloader=False)


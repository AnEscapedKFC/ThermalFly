"""Serialized GRaD-Nav stepping API and an FPV monitor. No autonomous ticking."""
from collections import OrderedDict
from concurrent.futures import Future,TimeoutError as FutureTimeout
import argparse
from copy import deepcopy
import io
import json
from pathlib import Path
import queue
import signal
import threading

from flask import Flask,Response,jsonify,request,send_from_directory
import numpy as np
import torch

from cameras import jpeg
from engine import PolicyFlightSimulator,EpisodeFinished


class StateConflict(RuntimeError):pass
class NotReady(RuntimeError):pass


class SimulatorService:
    """Own CUDA objects in one worker and publish immutable numbered snapshots."""
    def __init__(self,factory):
        self.factory=factory;self.lock=threading.Lock();self.tasks=queue.Queue(maxsize=16)
        self.frames=OrderedDict();self.results=OrderedDict()
        self.status={'ready':False,'loading':True};self.stop=threading.Event()
        self.thread=threading.Thread(target=self._run,daemon=True,name='gradnav-simulator')
        self.thread.start()

    def _publish(self,sim,receipt):
        rgb=np.asarray(sim.last_rgb,dtype=np.float32).copy()
        rgb.setflags(write=False)
        frame={'rgb':rgb,'jpeg':jpeg(rgb),'step_id':receipt['step_id'],'sim_time':receipt['sim_time']}
        receipt=deepcopy(receipt)
        receipt['ready']=True;receipt['loading']=False
        receipt['image']['url']=f"/api/frame/fpv?frame_id={receipt['frame_id']}"
        receipt['image']['float_rgb_url']=receipt['image']['url']+'&format=npy'
        with self.lock:
            self.frames[receipt['frame_id']]=frame
            while len(self.frames)>32:self.frames.popitem(last=False)
            self.status=receipt
        return deepcopy(receipt)

    def _run(self):
        sim=None
        try:
            sim=self.factory();self._publish(sim,sim.last_receipt)
            while not self.stop.is_set():
                try:item=self.tasks.get(timeout=.1)
                except queue.Empty:continue
                if item is None:break
                kind,payload,request_id,future=item
                if not future.set_running_or_notify_cancel():continue
                before=(sim.episode_id,sim.step_id,sim.frame_id)
                try:
                    fingerprint=json.dumps([kind,payload],sort_keys=True,allow_nan=False)
                    if request_id in self.results:
                        previous,result=self.results[request_id]
                        if previous!=fingerprint:raise StateConflict('request_id was already used with different input')
                        future.set_result(deepcopy(result));continue
                    expected=payload.pop('expected_frame_id',None)
                    if expected is not None and expected!=sim.last_receipt['frame_id']:
                        raise StateConflict('expected_frame_id is stale; read /api/state')
                    receipt=getattr(sim,kind)(**payload)
                    result=self._publish(sim,receipt);result['request_id']=request_id
                    with self.lock:self.status['request_id']=request_id
                    self.results[request_id]=(fingerprint,deepcopy(result))
                    while len(self.results)>128:self.results.popitem(last=False)
                    future.set_result(result)
                except (ValueError,EpisodeFinished,StateConflict) as exc:
                    if (sim.episode_id,sim.step_id,sim.frame_id)!=before:
                        with self.lock:self.status={**self.status,'ready':False,'error':str(exc),
                                                    'uncertain_request_id':request_id}
                        future.set_exception(RuntimeError(f'Execution failed after state changed: {exc}'))
                        break
                    future.set_exception(exc)
                except BaseException as exc:
                    # A renderer/encoder failure after integration must never
                    # permit an ambiguous second execution on retry.
                    with self.lock:self.status={**self.status,'ready':False,'error':str(exc)}
                    future.set_exception(exc);break
        except BaseException as exc:
            with self.lock:self.status={'ready':False,'loading':False,'error':str(exc)}
        finally:
            if sim is not None:sim.close()
            self.stop.set()
            while True:
                try:item=self.tasks.get_nowait()
                except queue.Empty:break
                if item is not None and not item[-1].done():item[-1].set_exception(NotReady('Simulator stopped'))

    def call(self,kind,payload,request_id,timeout=180):
        with self.lock:
            if not self.status.get('ready'):raise NotReady(self.status.get('error','Scene is still loading'))
        if self.stop.is_set():raise NotReady('Simulator stopped')
        future=Future()
        try:self.tasks.put_nowait((kind,deepcopy(payload),request_id,future))
        except queue.Full:raise NotReady('Action queue is full')
        try:return future.result(timeout=timeout)
        except FutureTimeout:
            future.cancel()
            raise NotReady('Request pending; retry using the same request_id')

    def snapshot(self):
        with self.lock:return deepcopy(self.status)

    def close(self):
        self.stop.set()
        self.thread.join(timeout=15)


def create_app(service):
    app=Flask(__name__,static_folder=None)
    app.config['MAX_CONTENT_LENGTH']=1024*1024
    web=Path(__file__).resolve().parent/'web'

    @app.get('/')
    def index():return send_from_directory(web,'index.html')

    @app.get('/api/state')
    def state():return jsonify(service.snapshot())

    @app.get('/api/frame/fpv')
    def frame():
        try:
            with service.lock:
                index=int(request.args.get('frame_id',service.status.get('frame_id',-1)))
                image=service.frames.get(index)
            if image is None:return jsonify(error='Frame unavailable or expired (last 32 snapshots retained)'),404
            format=request.args.get('format','jpeg')
            if format=='npy':
                buffer=io.BytesIO();np.save(buffer,image['rgb'],allow_pickle=False)
                result=Response(buffer.getvalue(),mimetype='application/x-npy')
            elif format=='jpeg':result=Response(image['jpeg'],mimetype='image/jpeg')
            else:return jsonify(error='format must be jpeg or npy'),400
            result.headers.update({'X-Frame-ID':str(index),'X-Step-ID':str(image['step_id']),
                                   'X-Sim-Time':str(image['sim_time']),'Cache-Control':'no-store'})
            return result
        except (ValueError,TypeError):return jsonify(error='Invalid frame_id'),400

    def execute(kind):
        data=request.get_json(silent=True)
        if not isinstance(data,dict):return jsonify(error='Expected a JSON object'),400
        request_id=data.pop('request_id',None)
        if not isinstance(request_id,str) or not 1<=len(request_id)<=128:
            return jsonify(error='Provide a unique string request_id (1-128 characters); reuse it for retries'),400
        allowed={'step':{'action','format','repeat','expected_frame_id'},
                 'reset':{'seed','spawn','yaw','expected_frame_id'}}[kind]
        if set(data)-allowed:return jsonify(error='Unknown request fields'),400
        if kind=='step' and 'action' not in data:return jsonify(error='action is required'),400
        if 'expected_frame_id' in data and (isinstance(data['expected_frame_id'],bool) or not isinstance(data['expected_frame_id'],int)):
            return jsonify(error='expected_frame_id must be an integer'),400
        try:
            # Reject JSON NaN/infinity before queueing or computing a request key.
            json.dumps(data,allow_nan=False)
            return jsonify(service.call(kind,data,request_id))
        except (ValueError,TypeError) as exc:return jsonify(error=str(exc)),400
        except (EpisodeFinished,StateConflict) as exc:return jsonify(error=str(exc)),409
        except NotReady as exc:return jsonify(error=str(exc),request_id=request_id),503
        except Exception as exc:return jsonify(error=str(exc),request_id=request_id),500

    @app.post('/api/step')
    def step():return execute('step')

    @app.post('/api/reset')
    def reset():return execute('reset')
    return app


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene',type=Path,default=Path.home()/'3DGS_env/no_train_outputs/export_gate_mid/gate_mid.ply')
    parser.add_argument('--point-cloud',type=Path)
    parser.add_argument('--profile',default='nominal',help='nominal, training, or a JSON/original YAML path')
    parser.add_argument('--observation-assets',type=Path,help='Real TorchScript encoders and JSON normalizer manifest')
    parser.add_argument('--seed',type=int,default=0)
    parser.add_argument('--spawn',nargs=3,type=float,default=[-6,0,1.25]);parser.add_argument('--yaw',type=float,default=0.)
    parser.add_argument('--resolution',type=float,default=.75)
    parser.add_argument('--up-axis',default='z');parser.add_argument('--scale',type=float,default=1.)
    parser.add_argument('--drone-size',nargs=3,type=float,default=[.6,.6,.12]);parser.add_argument('--collision-margin',type=float,default=.05)
    parser.add_argument('--no-collision',action='store_true')
    parser.add_argument('--host',default='127.0.0.1');parser.add_argument('--port',type=int,default=8766)
    args=parser.parse_args();torch.set_num_threads(2)
    service=SimulatorService(lambda:PolicyFlightSimulator(args.scene,args.point_cloud,args.profile,args.seed,
        args.spawn,args.yaw,args.resolution,args.drone_size,args.collision_margin,args.no_collision,
        up_axis=args.up_axis,scale=args.scale,observation_assets=args.observation_assets))
    def stop(*_):raise SystemExit(0)
    signal.signal(signal.SIGTERM,stop)
    print(f'GRaD-Nav step simulator: http://{args.host}:{args.port}',flush=True)
    try:create_app(service).run(host=args.host,port=args.port,threaded=True,use_reloader=False)
    finally:service.close()


if __name__=='__main__':main()

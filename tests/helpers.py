"""Small CPU scene fixtures; collision tests still use real Open3D geometry."""
from pathlib import Path

import numpy as np
import torch


class FakeScene:
    def __init__(self,path,width=64,height=36,up_axis='z',scale=1.,device='cpu'):
        self.path=Path(path);self.name=self.path.name
        self.width=width;self.height=height;self.up_axis=up_axis;self.scale=scale
        self.device=torch.device(device)
        self.bounds=np.array([[-100.,-100.,-100.],[100.,100.,100.]])
        self.count=3;self.sh_degree=0
        self.K=torch.tensor([[[width,0,width/2],[0,height,height/2],[0,0,1.]]],dtype=torch.float32,device=self.device)
        self.render_calls=0;self.cameras=[]

    def metadata(self):
        return {'name':self.name,'gaussians':self.count,'sh_degree':self.sh_degree,
                'bounds':self.bounds.tolist(),'up_axis':self.up_axis,'scale':self.scale}

    def render(self,camera):
        matrix=np.asarray(camera,dtype=float)
        self.render_calls+=1;self.cameras.append(matrix.copy())
        color=.5+.2*np.tanh(matrix[:3,3])
        image=np.broadcast_to(color,(self.height,self.width,3)).astype(np.float32).copy()
        depth=np.full((self.height,self.width),5.,dtype=np.float32)
        alpha=np.ones_like(depth)
        return image,depth,alpha

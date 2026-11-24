"""Differentiable rigid-body quadrotor with the existing keyboard controller.

Derived from GRaD-Nav's quadrotor dynamics. See LICENSE-third-party.md.
The quaternion rotation uses the consistent wxyz convention from the interactive
version. The original angular-acceleration stop-gradient is retained.
"""

from copy import deepcopy

import numpy as np
from scipy.spatial.transform import Rotation
import torch

CONTROL_DT = 0.05
MASS = 1.1
MAX_THRUST = 26.0
HOVER = MASS * 9.81 / MAX_THRUST
DEFAULT_DYNAMICS_PARAMETERS = {
    'mass': MASS, 'max_thrust': MAX_THRUST,
    'inertia': [.01, .012, .025], 'kp': [1., 1.2, 2.5], 'kd': [.001, .001, .002],
    'drag_coeff': .5, 'cross_area': .1, 'air_density': 1.225,
    'rotor_noise_std': 0., 'br_noise_std': 0.,
}


def validate_dynamics_parameters(parameters=None):
    result = deepcopy(DEFAULT_DYNAMICS_PARAMETERS)
    if parameters is not None:
        unknown = set(parameters)-set(result)
        if unknown:raise ValueError('Unknown dynamics parameters: '+', '.join(sorted(unknown)))
        result.update(deepcopy(parameters))
    for name in ('mass', 'max_thrust', 'drag_coeff', 'cross_area', 'air_density',
                 'rotor_noise_std', 'br_noise_std'):
        value = float(result[name])
        if not np.isfinite(value) or value < 0 or (name in ('mass','max_thrust') and value == 0):
            raise ValueError(f'Invalid dynamics parameter {name}')
        result[name] = value
    for name in ('inertia', 'kp', 'kd'):
        value = np.asarray(result[name], dtype=float)
        if value.shape != (3,) or not np.isfinite(value).all() or np.any(value < 0) or (name == 'inertia' and np.any(value == 0)):
            raise ValueError(f'Invalid dynamics parameter {name}')
        result[name] = value.tolist()
    return result


def quaternion_multiply(a, b):
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack((aw*bw-ax*bx-ay*by-az*bz,
                        aw*bx+ax*bw+ay*bz-az*by,
                        aw*by-ax*bz+ay*bw+az*bx,
                        aw*bz+ax*by-ay*bx+az*bw), dim=-1)


def quaternion_rotation(q):
    q = q / q.norm(dim=-1, keepdim=True)
    w, x, y, z = q.unbind(-1)
    return torch.stack((
        1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y),
        2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x),
        2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y),
    ), dim=-1).reshape(*q.shape[:-1], 3, 3)


def initial_state(device, position=(0., 0., 1.25), yaw=0., velocity_w_mps=(0., 0., 0.)):
    """State: position, velocity, quaternion wxyz, angular velocity; yaw in degrees."""
    angle = np.deg2rad(yaw)/2
    return torch.tensor([*position, *velocity_w_mps, np.cos(angle),0,0,np.sin(angle), 0,0,0],
                        dtype=torch.float64, device=device)


class QuadrotorDynamics:
    def __init__(self, device=None, parameters=None, seed=0):
        self.device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
        self.parameters = validate_dynamics_parameters(parameters)
        self.inertia = torch.tensor(self.parameters['inertia'], device=self.device,dtype=torch.float64)
        self.kp = torch.tensor(self.parameters['kp'], device=self.device,dtype=torch.float64)
        self.kd = torch.tensor(self.parameters['kd'], device=self.device,dtype=torch.float64)
        self.gravity = torch.tensor([0.,0.,-9.81], device=self.device,dtype=torch.float64)
        self.previous_omega = torch.zeros(3, device=self.device,dtype=torch.float64)
        self.last_linear_acceleration = torch.zeros_like(self.previous_omega)
        self.last_angular_acceleration = torch.zeros_like(self.previous_omega)
        self.generator = torch.Generator(device=self.device)
        self.generator.manual_seed(int(seed))

    def step(self, state, control,substep_callback=None):
        """Integrate 10 substeps; optional callback may stop at a safe state.

        With no callback the differentiable dynamics path is unchanged.
        Collision callbacks are discrete interactive constraints, not gradients.
        """
        p, v, q, omega = state[:3], state[3:6], state[6:10], state[10:13]
        dt = .005
        for _ in range(10):
            before=torch.cat((p,v,q,omega)) if substep_callback else None
            actual_thrust = control[3]
            desired_omega = control[:3]
            if self.parameters['rotor_noise_std']:
                actual_thrust = actual_thrust + torch.randn((),device=self.device,dtype=control.dtype,
                                                           generator=self.generator)*self.parameters['rotor_noise_std']
            if self.parameters['br_noise_std']:
                desired_omega = desired_omega + torch.randn((3,),device=self.device,dtype=control.dtype,
                                                           generator=self.generator)*self.parameters['br_noise_std']
            estimated_alpha = ((omega-self.previous_omega)/dt).detach()
            torque = self.kp*(desired_omega-omega) - self.kd*estimated_alpha
            angular_acc = (torque-torch.linalg.cross(omega,self.inertia*omega))/self.inertia
            self.previous_omega = omega
            omega = omega + angular_acc*dt
            omega_quat = torch.cat((omega.new_zeros(1),omega))
            q = q + .5*quaternion_multiply(q,omega_quat)*dt
            q = q/q.norm()
            force = quaternion_rotation(q)[:,2] * (actual_thrust*self.parameters['max_thrust'])
            drag = -.5*self.parameters['drag_coeff']*self.parameters['cross_area']*self.parameters['air_density']*v*v.norm()
            acceleration = (force+drag)/self.parameters['mass'] + self.gravity
            self.last_linear_acceleration = acceleration
            self.last_angular_acceleration = angular_acc
            v = v + acceleration*dt
            p = p + v*dt
            if substep_callback:
                override=substep_callback(before,torch.cat((p,v,q,omega)),dt)
                if override is not None:
                    self.previous_omega=override[10:13].clone()
                    # Contact is the inherited kinematic stop, not an impulse
                    # model; do not report free-flight acceleration as contact.
                    self.last_linear_acceleration = torch.zeros_like(override[3:6])
                    self.last_angular_acceleration = torch.zeros_like(override[10:13])
                    return override
        return torch.cat((p,v,q,omega))


class FlightController:
    """Same assisted position hold and direct body-rate keyboard mappings."""
    def __init__(self, state):
        self.mode = 'assisted'
        self.hold(state)

    def hold(self, state):
        state = state.detach().cpu().numpy()
        self.target = state[:3].copy()
        R = Rotation.from_quat(state[[7,8,9,6]]).as_matrix()
        self.yaw = np.arctan2(R[1,0],R[0,0])

    def command(self, state, axes):
        if self.mode == 'rates':
            return np.array([-axes[1]*.3, axes[0]*.3, axes[3]*.5,
                             np.clip(HOVER+axes[2]*.1,0,1)])
        state = state.detach().cpu().numpy()
        self.yaw += axes[3]*.65*CONTROL_DT
        c,s = np.cos(self.yaw),np.sin(self.yaw)
        velocity = np.array([c*axes[0]-s*axes[1],s*axes[0]+c*axes[1],axes[2]])*1.2
        self.target += velocity*CONTROL_DT
        # No room-specific ceiling or origin assumption.
        self.target[:2] = np.clip(self.target[:2],state[:2]-1.5,state[:2]+1.5)
        acceleration = np.clip(2.2*(self.target-state[:3])+2.6*(velocity-state[3:6]),-2.5,2.5)
        force = acceleration+[0.,0,9.81]
        b3 = force/np.linalg.norm(force)
        b2 = np.cross(b3,[c,s,0.]); b2 /= np.linalg.norm(b2)
        desired_R = np.column_stack((np.cross(b2,b3),b2,b3))
        desired_q = Rotation.from_matrix(desired_R).as_quat()[[3,0,1,2]]
        q = state[6:10]
        scalar = q[0]*desired_q[0]+np.dot(q[1:],desired_q[1:])
        vector = q[0]*desired_q[1:]-desired_q[0]*q[1:]-np.cross(q[1:],desired_q[1:])
        rates = np.clip(10*vector*(1 if scalar>=0 else -1),-1.5,1.5)
        return np.r_[rates,np.clip(MASS*np.linalg.norm(force)/MAX_THRUST,.1,.8)]

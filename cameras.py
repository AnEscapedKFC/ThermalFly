"""Rigidly mounted flight cameras and a projected quadrotor display model."""
import io
import math
import numpy as np
from PIL import Image, ImageDraw
from scipy.spatial.transform import Rotation
import torch


def rotation(state):
    return Rotation.from_quat(state[[7, 8, 9, 6]]).as_matrix()


def fpv_camera(state):
    """Body +x forward, +y left, +z up; camera OpenGL convention.

    Keep the released camera mount at the neutral pose, then move it rigidly
    with the physical body orientation instead of mixing world/body flips.
    """
    R = rotation(state)
    mount = np.array([[.990, 0, .140], [0, 1, 0], [-.140, 0, .990]])
    # Normalize the rounded calibration rotation to SO(3).
    mount = Rotation.from_matrix(mount).as_matrix()
    axes = np.array([[0, 0, -1], [1, 0, 0], [0, -1, 0]])
    flip = np.diag([1., -1, -1])
    camera = np.eye(4)
    camera[:3, :3] = R @ flip @ mount @ axes
    camera[:3, 3] = state[:3] + R @ np.array([.152, .031, .012])
    return torch.from_numpy(camera)


def chase_camera(state):
    p = state[:3]
    yaw = math.atan2(rotation(state)[1, 0], rotation(state)[0, 0])
    c, s = math.cos(yaw), math.sin(yaw)
    offset = np.array([[-c, -s, 0], [-s, c, 0], [0, 0, 1]]) @ np.array([.9, -.65, .45])
    eye = p + offset
    forward = p - eye
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, [0., 0, 1])
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    camera = np.eye(4)
    camera[:3, :3] = np.column_stack((right, up, -forward))
    camera[:3, 3] = eye
    return torch.from_numpy(camera)


def draw_drone(rgb, depth, camera, state, K, elapsed):
    """Project a quadrotor wireframe/model into the real 3DGS chase image.

    Scene depth hides model segments behind scene surfaces. It is a display
    overlay only; it does not add physical collision or alter the 3DGS weights.
    """
    image = Image.fromarray((rgb * 255).round().astype(np.uint8)).convert("RGB")
    draw = ImageDraw.Draw(image)
    world_position = state[:3]
    R = rotation(state)
    world_to_camera = np.linalg.inv(camera.numpy())
    h, w = depth.shape

    def project(local):
        world = world_position + R @ np.asarray(local)
        point = world_to_camera @ np.r_[world, 1.]
        z = -point[2]
        if z <= .01:
            return None
        u = K[0, 0] * point[0] / z + K[0, 2]
        v = -K[1, 1] * point[1] / z + K[1, 2]
        if not (0 <= u < w and 0 <= v < h):
            return None
        if 0 < depth[int(v), int(u)] < z - .12:
            return None
        return (float(u), float(v))

    def line(a, b, color, width=2):
        # Test short pieces separately so scene geometry can partially occlude.
        points = np.linspace(a, b, 7)
        for p, q in zip(points[:-1], points[1:]):
            pp, qq = project(p), project(q)
            if pp is not None and qq is not None:
                draw.line([pp, qq], fill=color, width=width)

    for x in (-.19, .19):
        for y in (-.19, .19):
            color = "#ffb158" if x > 0 else "#69e0d3"
            line([0, 0, 0], [x, y, 0], color, 3)
            angles = np.linspace(0, 2*np.pi, 25)
            ring = np.column_stack((x+.105*np.cos(angles), y+.105*np.sin(angles), np.full_like(angles, .025)))
            for a, b in zip(ring[:-1], ring[1:]):
                line(a, b, color, 1)
            spin = elapsed * 80
            delta = np.array([.09*math.cos(spin), .09*math.sin(spin), 0])
            line(np.array([x, y, .03])-delta, np.array([x, y, .03])+delta, "#f3f5ed", 2)
    body = [project(p) for p in ([.10,.065,.035], [.10,-.065,.035], [-.1,-.065,.035], [-.1,.065,.035])]
    if all(p is not None for p in body):
        draw.polygon(body, fill="#222d33", outline="#eff9f1", width=2)
    line([.06, 0, .04], [.18, 0, .04], "#ffb158", 3)
    return image


def jpeg(image):
    if isinstance(image, np.ndarray):
        image = Image.fromarray((image * 255).round().astype(np.uint8))
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=88)
    return buffer.getvalue()

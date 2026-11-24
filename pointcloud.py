"""Open3D surface points and continuous, attitude-aware box collision checks."""

import json
from pathlib import Path

import numpy as np
from plyfile import PlyData
from scipy.spatial.transform import Rotation

from scene import up_rotation

DEFAULT_DRONE_SIZE = (.60, .60, .12)  # Full body-axis X/Y/Z dimensions, meters.
SWEEP_TOLERANCE = 1e-6  # Maximum rotational envelope inflation, meters.


def body_rotation(quaternion=None):
    """Convert a body-to-world wxyz quaternion to a scipy rotation."""
    if quaternion is None:return Rotation.identity()
    q=np.asarray(quaternion,dtype=float)
    if q.shape!=(4,) or not np.isfinite(q).all() or np.linalg.norm(q)<1e-12:
        raise ValueError('Expected a finite nonzero wxyz quaternion')
    return Rotation.from_quat(q[[1,2,3,0]])


def interpolate_quaternion(start,end,fraction):
    first=body_rotation(start)
    vector=(first.inv()*body_rotation(end)).as_rotvec()
    return (first*Rotation.from_rotvec(vector*fraction)).as_quat()[[3,0,1,2]]


def box_signed_distance(local_points,half_size):
    """Euclidean distance outside a box; negative distance to its faces inside."""
    delta=np.abs(local_points)-half_size
    return np.linalg.norm(np.maximum(delta,0.),axis=-1)+np.minimum(delta.max(axis=-1),0.)


def slab_hit(start,end,half_size):
    """First segment/AABB hit among arrays of point-relative segments."""
    delta=end-start
    parallel=np.abs(delta)<1e-15
    outside=np.any(parallel & (np.abs(start)>half_size),axis=1)
    inverse=np.zeros_like(delta)
    np.divide(1.,delta,out=inverse,where=~parallel)
    t0=(-half_size-start)*inverse;t1=(half_size-start)*inverse
    near=np.where(parallel,-np.inf,np.minimum(t0,t1)).max(axis=1)
    far=np.where(parallel,np.inf,np.maximum(t0,t1)).min(axis=1)
    entry=np.maximum(near,0.)
    valid=(~outside)&(entry<=np.minimum(far,1.))
    indices=np.flatnonzero(valid)
    if not len(indices):return None
    index=int(indices[np.argmin(entry[indices])])
    return float(entry[index]),index


def open3d_module():
    try:
        import open3d as o3d
    except ImportError as exc:
        raise RuntimeError('Point-cloud collision requires Open3D: pip install open3d-cpu==0.19.0') from exc
    return o3d


def alignment_matrix(value=None):
    if value is None:return np.eye(4)
    if isinstance(value,(str,Path)):
        value=json.loads(Path(value).read_text())
        if isinstance(value,dict):value=value.get('transform')
    matrix=np.asarray(value,dtype=float)
    if (matrix.shape!=(4,4) or not np.isfinite(matrix).all() or
            not np.allclose(matrix[3],[0,0,0,1]) or abs(np.linalg.det(matrix[:3,:3]))<1e-12):
        raise ValueError('Point-cloud alignment must be a finite invertible affine 4x4 matrix')
    return matrix


class ObstacleCloud:
    def __init__(self,path,up_axis='z',scale=1.,size=DEFAULT_DRONE_SIZE,margin=.05,transform=None):
        size=np.asarray(size,dtype=float)
        if (size.shape!=(3,) or not np.isfinite(size).all() or np.any(size<=0) or
                not np.isfinite([scale,margin]).all() or scale<=0 or margin<0):
            raise ValueError('Require scale>0, three positive drone dimensions and margin>=0')
        path=Path(path).expanduser().resolve()
        if not path.is_file():raise ValueError(f'Point cloud does not exist: {path}')
        if path.suffix.lower()=='.ply':
            properties=set(PlyData.read(str(path))['vertex'].data.dtype.names or ())
            if any(name.startswith(('scale_','f_dc_','rot_')) for name in properties):
                raise ValueError('Expected an ordinary surface point cloud, not a Gaussian PLY; export rendered depth first')
        o3d=open3d_module()
        cloud=o3d.io.read_point_cloud(str(path))
        points=np.asarray(cloud.points)
        if not len(points) or not np.isfinite(points).all():
            raise ValueError('Obstacle point cloud is empty or contains non-finite coordinates')
        matrix=alignment_matrix(transform)
        points=points@matrix[:3,:3].T+matrix[:3,3]
        points=(points*scale)@up_rotation(up_axis).T
        if not np.isfinite(points).all():raise ValueError('Invalid transformed point-cloud coordinates')
        self.cloud=o3d.geometry.PointCloud()
        self.cloud.points=o3d.utility.Vector3dVector(points)
        self.tree=o3d.geometry.KDTreeFlann(self.cloud)
        self.points=np.asarray(self.cloud.points)
        self.path=path
        self.size=size.copy();self.half_size=size/2;self.margin=float(margin)
        self.safety_half_size=self.half_size+self.margin
        self.search_radius=float(np.linalg.norm(self.safety_half_size))
        self.source='external'

    def distance(self,position,quaternion=None):
        position=np.asarray(position,dtype=float)
        if position.shape!=(3,) or not np.isfinite(position).all():raise ValueError('Invalid distance query')
        count,indices,squared=self.tree.search_knn_vector_3d(position,1)
        if count!=1:raise RuntimeError('Point-cloud nearest-neighbor query failed')
        distance=float(np.sqrt(squared[0]));R=body_rotation(quaternion).as_matrix()
        seed=(self.points[indices[0]]-position)@R
        # A point outside these enclosing radii cannot beat either seed distance.
        # The center-nearest point alone need not be the box-nearest point.
        radius=max(np.linalg.norm(self.half_size)+max(float(box_signed_distance(seed,self.half_size)),0.),
                   self.search_radius+max(float(box_signed_distance(seed,self.safety_half_size)),0.))
        _,candidates,_=self.tree.search_radius_vector_3d(position,radius+1e-9)
        points=self.points[np.asarray(candidates,dtype=int)]
        local=(points-position)@R
        distances=box_signed_distance(local,self.half_size)
        safety=box_signed_distance(local,self.safety_half_size)
        nearest=int(np.argmin(distances))
        return {'center_distance':distance,'clearance':float(distances[nearest]),
                'safety_clearance':float(safety.min()),'nearest_point':points[nearest].tolist(),
                'in_collision':bool(safety.min()<=0.),
                'corners':self.corners(position,R,self.half_size),
                'safety_corners':self.corners(position,R,self.safety_half_size)}

    @staticmethod
    def corners(position,R,half_size):
        signs=np.array([[x,y,z] for x in (-1,1) for y in (-1,1) for z in (-1,1)])
        return ((signs*half_size)@R.T+position).tolist()

    def sweep(self,start,end,start_quaternion=None,end_quaternion=None):
        """Sweep an oriented box along translation and shortest-arc rotation.

        A KDTree enclosing sphere only selects candidates. Slab tests decide
        actual box hits. For rotation, midpoint boxes are expanded by a proven
        bound on corner displacement and recursively refined to 1 micrometer;
        this also catches pure rotation with no center translation.
        """
        start=np.asarray(start,dtype=float);end=np.asarray(end,dtype=float)
        if start.shape!=(3,) or end.shape!=(3,) or not np.isfinite([start,end]).all():
            raise ValueError('Invalid swept collision query')
        delta=end-start;length=float(np.linalg.norm(delta))
        first=body_rotation(start_quaternion)
        last=body_rotation(end_quaternion) if end_quaternion is not None else first
        vector=(first.inv()*last).as_rotvec();angle=float(np.linalg.norm(vector))
        count,indices,_=self.tree.search_radius_vector_3d((start+end)/2,self.search_radius+length/2+1e-9)
        if not count:return None
        points=self.points[np.asarray(indices)]

        def interval(lo,hi,subset):
            mid=(lo+hi)/2
            R=(first*Rotation.from_rotvec(vector*mid)).as_matrix()
            inflation=2*self.search_radius*np.sin(angle*(hi-lo)/4)
            half=self.safety_half_size+inflation
            p0=(points[subset]-(start+lo*delta))@R
            p1=(points[subset]-(start+hi*delta))@R
            # Keep every possible point, not just the first coarse hit: another
            # point can become the first real hit after the envelope is refined.
            lower=np.minimum(p0,p1);upper=np.maximum(p0,p1)
            possible=np.all((lower<=half)&(upper>=-half),axis=1)
            subset=subset[possible];p0=p0[possible];p1=p1[possible]
            if not len(subset):return None
            hit=slab_hit(p0,p1,half)
            if hit is None:return None
            if inflation<=SWEEP_TOLERANCE:
                fraction,index=hit
                return {'fraction':float(lo+(hi-lo)*fraction),'point':points[subset[index]].tolist()}
            return interval(lo,mid,subset) or interval(mid,hi,subset)

        return interval(0.,1.,np.arange(len(points)))

    def metadata(self):
        return {'points':len(self.points),'shape':'box','size':self.size.tolist(),'margin':self.margin,
                'effective_size':(2*self.safety_half_size).tolist(),'source':self.source}

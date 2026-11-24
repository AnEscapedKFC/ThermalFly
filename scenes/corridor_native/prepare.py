"""Build simulator coordinates and corroborated collision/free-space cells from the corridor data.

From the project root run python -m scenes.corridor_native.prepare in thermal_gaussian_5070.
"""
from pathlib import Path
import json
import numpy as np
from plyfile import PlyData, PlyElement
from scipy.ndimage import maximum_filter, minimum_filter
from scipy.optimize import least_squares
from scipy.spatial import cKDTree
import torch

from scenes.corridor_native.room import ROOT, CorridorModel, intrinsic, read, tensor

DATA = (ROOT / "inputs/manifest.json").resolve().parent
W, H = 320, 240


def write(path, obj):
    (ROOT / path).write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")


def backproject(depth, rows, cols, K, C):
    rays = np.stack(((cols + .5 - K[0, 2]) / K[0, 0], (rows + .5 - K[1, 2]) / K[1, 1],
                     np.ones(len(cols))), axis=1)
    return (rays * depth[rows, cols, None]) @ C[:3, :3].T + C[:3, 3]


def fit_frame(rows):
    # The initial SfM gauge has the first RGB camera pointing along +Z. Samples on
    # both sides of the aisle estimate two parallel walls. Between them the two
    # strongest horizontal depth surfaces are the ceiling and the floor.
    clouds = []
    for r in rows:
        with np.load(DATA / r["dense_prior"]["path"]) as prior:
            depth = prior["depth"]
            good = (prior["support"] >= 3) & (prior["confidence"] >= .55) & (depth > 0) & (depth < 11)
            y, x = np.nonzero(good[::5, ::5])
            clouds.append(backproject(depth, y * 5, x * 5, intrinsic(r, W, H), np.linalg.inv(r["base_w2c"])))
    p = np.concatenate(clouds)
    level_y = p[:, 1] + .040 * p[:, 2]
    level_x = p[:, 0] + .015 * p[:, 2]
    middle = (abs(level_x) < .19) & (abs(p[:, 2]) < 6)
    floor = middle & (abs(level_y - .26) < .055)
    ceiling = middle & (abs(level_y + .265) < .055)
    lateral = (abs(level_y) < .19) & (abs(p[:, 2]) < 6)
    right = lateral & (abs(level_x - .26) < .055)
    left = lateral & (abs(level_x + .28) < .055)

    def paired_plane_normal(ax, idx, a, b, initial):
        point = p[a | b]
        sign = a[a | b]
        def residual(par):
            return point[:, ax] + point[:, idx] @ par[:2] - np.where(sign, par[2], par[3])
        result = least_squares(residual, initial, loss="soft_l1", f_scale=.008, max_nfev=30)
        par = result.x
        normal = np.zeros(3)
        normal[ax], normal[idx] = 1., par[:2]
        norm = np.linalg.norm(normal)
        return normal / norm, par[2:] / norm, float(np.median(abs(result.fun)) / norm)

    down, (floor_projection, ceiling_projection), level_error = paired_plane_normal(
        1, [0, 2], floor, ceiling, [0., .04, .26, -.265])
    right_normal, wall_offsets, wall_error = paired_plane_normal(
        0, [1, 2], right, left, [0., .015, .26, -.28])
    up = -down
    forward = np.cross(down, right_normal)
    if forward[2] < 0:
        forward = -forward
    forward /= np.linalg.norm(forward)
    rotation = np.stack((forward, np.cross(up, forward), up))
    scale = 2.80 / (floor_projection - ceiling_projection)
    origin = floor_projection * down
    bounds = [[-35., -2.5, -.24], [38., 2.5, 3.04]]
    write("inputs/frame.json", dict(rotation=rotation.tolist(), origin=origin.tolist(),
          meters_per_source_unit=scale, bounds_m=bounds, assumed_ceiling_height_m=2.80,
          scale_status="Assumed floor-to-ceiling separation; no surveyed metric reference supplied",
          floor_ceiling_source_offsets=[floor_projection, ceiling_projection],
          wall_source_offsets=wall_offsets.tolist(), wall_source_normal=right_normal.tolist(),
          plane_fit_median_abs_error_src=[level_error, wall_error],
          frame_plane_sample_counts=[int(a.sum()) for a in [floor, ceiling, right, left]],
          source="Supported SfM-aligned training depth; corridor-local floor, ceiling and parallel side-wall fits"))
    return origin, rotation, scale, np.array(bounds)


def store_cloud(points, colors):
    vertex = np.empty(len(points), dtype=[(k, "f4") for k in "xyz"] +
                                         [(k, "u1") for k in ("red", "green", "blue")])
    for key, col in zip(vertex.dtype.names, [*points.T, *colors.T]):
        vertex[key] = col
    PlyData([PlyElement.describe(vertex, "vertex")]).write(ROOT / "collision/rgb_surface_m.ply")


def main():
    rows = [r for r in read("inputs/manifest.json")["views"]
            if r["modality"] == "rgb" and r["split"] == "train" and "dense_prior" in r]
    origin, R, scale, (lo, hi) = fit_frame(rows)
    frame = read("inputs/frame.json")
    normal = np.array(frame["wall_source_normal"])
    planes = [(normal, t) for t in frame["wall_source_offsets"]]
    planes.extend((-R[2], t) for t in frame["floor_ceiling_source_offsets"])
    yy, xx = np.indices((H, W))
    print("metres per SfM unit", scale, flush=True)
    model = CorridorModel("rgb", W, H)
    clouds, colors, views, centers = [], [], [], []
    for j, r in enumerate(rows):
        w2c = np.array(r["base_w2c"])
        C, K = np.linalg.inv(w2c), intrinsic(r, W, H)
        centers.append((C[:3, 3] - origin) @ R.T * scale)
        z, std, alpha = model.sample(w2c, K, W, H, moments=True)
        with np.load(DATA / r["dense_prior"]["path"]) as prior:
            d = prior["depth"]
            good = (prior["support"] >= 3) & (prior["confidence"] >= .52) & (d * scale > .15) & (d * scale < 35.)
            corroborates = (prior["support"] >= 2) & (prior["confidence"] >= .25) & (d*scale > .15)
        low = minimum_filter(np.where(good, d, 0), size=3)
        high = maximum_filter(np.where(good, d, 0), size=3)
        supported = good & (low > 0) & ((high - low) * scale < (.13 + .02 * d * scale))
        agree = (supported & (alpha > .95) & (std * scale < .14 + .012 * d * scale)
                 & (abs(z - d) * scale < .075 + .012 * d * scale))
        # At grazing angles depth variance becomes large even on a flat wall. Also
        # test the four fitted planes against individual observed pixels. No
        # plane is filled beyond locally witnessed footprints in multiple cameras.
        ray = np.stack(((xx+.5-K[0, 2])/K[0, 0], (yy+.5-K[1, 2])/K[1, 1], np.ones_like(xx)), axis=-1)
        directions = ray @ C[:3, :3].T
        plane_depth = np.full_like(d, np.inf)
        for normal, offset in planes:
            dot = directions @ normal
            projected = np.full_like(d, np.inf)
            np.divide(offset - normal @ C[:3, 3], dot, out=projected, where=abs(dot)>.075)
            valid_plane = (projected*scale>.25) & (projected*scale<22.)
            plane_depth = np.minimum(plane_depth, np.where(valid_plane, projected, np.inf))
        # Limit range error in the camera direction as well as the normal error.
        supported_plane = supported & np.isfinite(plane_depth) & (abs(d-plane_depth)*scale < (.08+.025*d*scale))
        supported_plane &= (alpha>.94) & (abs(z-plane_depth)*scale < (.22+.08*d*scale))
        # Geometric surface points are only proposed where diverse observations agree.
        surface = np.where(agree, z, 0).astype("f4")
        surface[supported_plane] = plane_depth[supported_plane]
        # An observed cell must lie in front of a supported depth return. Missing
        # and silhouette pixels do not vote. When RGB GS is closer, shorten the ray.
        near = np.minimum(low, np.where((alpha > .95) & (z > 0), z, np.inf))
        free_depth = np.where(supported, near, 0).astype("f4")
        # On less certain but still multiview-supported prior rays, require RGB-GS
        # depth minus three deviations AND a conservative prior depth bound. Use
        # the nearer value and erode 3x3; unsupported rays never count as free.
        gs_near = np.where((alpha>.99) & corroborates, np.maximum(0, np.minimum(z-3*std, .93*d)), 0)
        gs_near = minimum_filter(gs_near, size=3)
        gs_near = np.minimum(gs_near, 15./scale)
        free_depth = np.where(supported, free_depth, gs_near).astype("f4")
        views.append((w2c, K, surface, free_depth))
        y, x = np.nonzero((surface>0)[::2, ::2]); y *= 2; x *= 2
        if len(y):
            world = (backproject(surface, y, x, K, C) - origin) @ R.T * scale
            keep = ((world > lo) & (world < hi)).all(1)
            clouds.append(world[keep].astype("f4"))
            rgb = model.sample(w2c, K, W, H)[0]
            colors.append(np.uint8((255 * rgb[y[keep], x[keep]]).round()))
        if j % 80 == 0:
            print("rendering supported surfaces", j, flush=True)
    del model
    cloud, color = np.concatenate(clouds), np.concatenate(colors)
    _, inv = np.unique(np.floor((cloud-lo)/.04).astype("i4"), axis=0, return_inverse=True)
    counts = np.bincount(inv)
    def average(data):
        return np.stack([np.bincount(inv, weights=c) / counts for c in data.T], axis=1)
    surfaces, mean_colors = average(cloud), average(color).round().astype("u1")
    step = .10
    dims = np.ceil((hi - lo) / step).astype(int)
    cells = lo + (np.indices(dims).reshape(3, -1).T + .5) * step
    pts = tensor((np.concatenate((surfaces, cells)) / scale) @ R + origin)
    n = len(views)
    device = pts.device
    hit = torch.full((len(pts), 3), n, dtype=torch.int32, device=device)
    free = hit.clone()
    separation = np.linalg.norm(np.array(centers)[:, None] - centers, axis=-1)
    apart = torch.as_tensor(np.pad(separation >= .25, ((0, 0), (0, 1)), constant_values=True), device=device)
    def vote(bank, mask, i):
        ids = torch.where(mask & (bank[:, -1] == n))[0]
        prev = bank[ids].long()
        accept = apart[i, prev].all(1)
        bank[ids[accept], (prev[accept] != n).sum(1)] = i
    for i, (w2c, K, solid, far) in enumerate(views):
        cam = pts @ tensor(w2c[:3, :3]).T + tensor(w2c[:3, 3])
        z = cam[:, 2]
        den = z.clamp_min(.001)
        x = (cam[:, 0] / den * K[0, 0] + K[0, 2]).floor().long()
        y = (cam[:, 1] / den * K[1, 1] + K[1, 2]).floor().long()
        inside = (z * scale > .14) & (x >= 1) & (x < W-1) & (y >= 1) & (y < H-1)
        x, y = x.clamp(0, W-1), y.clamp(0, H-1)
        obs = tensor(solid)[y, x]
        vote(hit, inside & (obs > 0) & ((obs-z).abs() * scale < .055), i)
        vote(free, inside & ((tensor(far)[y, x]-z) * scale > .16), i)
        if i % 100 == 0:
            print("cross-view verification", i, flush=True)
    nh = (hit != n).sum(1).cpu().numpy()
    nf = (free != n).sum(1).cpu().numpy()
    size = len(surfaces)
    valid = np.where((nh[:size] >= 3) & (nf[:size] < 3))[0]
    local = cKDTree(surfaces[valid]).query_ball_point(surfaces[valid], .11, return_length=True)
    valid = valid[local >= 5]
    store_cloud(surfaces[valid], mean_colors[valid])
    observed = (nf[size:] >= 3).reshape(dims)
    np.savez_compressed(ROOT / "collision/free_volume.npz", free=observed, origin=lo, step=step)
    write("collision/build.json", dict(rgb_model=str((ROOT / "models/rgb.ply").resolve()),
          geometry="Filtered RGB 3DGS depth moments + independent SfM-aligned monocular predictions",
          planar_patches="Four source-fitted planes; only locally supported pixels, rechecked from three separated origins with free-space veto.",
          free_rays="All rays carry translated-view prior depth support. Weaker priors additionally require alpha>0.99 and the smaller of 0.93*prior_depth and rendered_depth-3*std.",
          training_rgb_views=len(rows), sparse_voxel_m=.04, free_voxel_m=step, distinct_views=3,
          required_view_baseline_m=.25, free_ray_margin_m=.16, candidate_voxels=size,
          fused_surface_points=len(valid), free_cells=int(observed.sum()), total_cells=int(observed.size),
          unknown_policy="Outside observed free space is unknown. Depth evidence is approximate, not range sensing."))
    print("wrote", len(valid), "surface points,", observed.sum(), "free cells", flush=True)


if __name__ == "__main__":
    main()

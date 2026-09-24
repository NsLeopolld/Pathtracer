"""
Triangle meshes for both renderers: OBJ loading, placement and BVH building.

    m = load_obj("models/suzanne.obj").placed(height=1.2, at=(0, 0, 0))

make_scenes.py writes the result to a .mesh file for pathtracer.c; the GPU
renderer uploads the same arrays straight from Python.
"""

import struct

import numpy as np


class Mesh:
    """Triangles as p[i] = (v0, v1, v2) and matching per-vertex normals n[i]."""

    def __init__(self, p, n):
        self.p = np.asarray(p, np.float64).reshape(-1, 3, 3)
        self.n = np.asarray(n, np.float64).reshape(-1, 3, 3)

    def __len__(self):
        return len(self.p)

    def bounds(self):
        return self.p.reshape(-1, 3).min(0), self.p.reshape(-1, 3).max(0)

    def transformed(self, scale=1.0, rotate=(0.0, 0.0, 0.0), translate=(0.0, 0.0, 0.0)):
        """Scale, then rotate (degrees around x, then y, then z), then move."""
        ax, ay, az = np.radians(rotate)
        rx = np.array([[1, 0, 0], [0, np.cos(ax), -np.sin(ax)], [0, np.sin(ax), np.cos(ax)]])
        # same sense as box rot_y in the renderers: +x turns toward -z
        ry = np.array([[np.cos(ay), 0, np.sin(ay)], [0, 1, 0], [-np.sin(ay), 0, np.cos(ay)]])
        rz = np.array([[np.cos(az), -np.sin(az), 0], [np.sin(az), np.cos(az), 0], [0, 0, 1]])
        r = rz @ ry @ rx
        s = np.broadcast_to(np.asarray(scale, np.float64), (3,))
        p = (self.p * s) @ r.T + np.asarray(translate, np.float64)
        # normals transform with the inverse transpose; for scale S and rotation R
        # that's R S^-1, renormalised
        n = (self.n / s) @ r.T
        n /= np.linalg.norm(n, axis=-1, keepdims=True)
        return Mesh(p, n)

    def placed(self, height, at=(0.0, 0.0, 0.0), rotate=(0.0, 0.0, 0.0)):
        """Rotate, scale uniformly to `height`, and stand it on `at` (bottom centre)."""
        m = self.transformed(rotate=rotate)
        lo, hi = m.bounds()
        s = height / (hi[1] - lo[1])
        m = m.transformed(scale=s)
        lo, hi = m.bounds()
        foot = np.array([(lo[0] + hi[0]) / 2, lo[1], (lo[2] + hi[2]) / 2])
        return m.transformed(translate=np.asarray(at, np.float64) - foot)


def icosphere(subdiv=2, smooth=True):
    """Unit sphere from a subdivided icosahedron: 20 * 4^subdiv triangles.
    smooth=False gives flat (faceted) normals."""
    t = (1 + 5 ** 0.5) / 2
    v = [(-1, t, 0), (1, t, 0), (-1, -t, 0), (1, -t, 0), (0, -1, t), (0, 1, t),
         (0, -1, -t), (0, 1, -t), (t, 0, -1), (t, 0, 1), (-t, 0, -1), (-t, 0, 1)]
    f = [(0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11), (1, 5, 9), (5, 11, 4),
         (11, 10, 2), (10, 7, 6), (7, 1, 8), (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8),
         (3, 8, 9), (4, 9, 5), (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1)]
    p = np.array(v, np.float64)[np.array(f)]
    p /= np.linalg.norm(p, axis=-1, keepdims=True)
    for _ in range(subdiv):
        a, b, c = p[:, 0], p[:, 1], p[:, 2]
        ab, bc, ca = [(x + y) / np.linalg.norm(x + y, axis=-1, keepdims=True)
                      for x, y in ((a, b), (b, c), (c, a))]
        p = np.concatenate([np.stack(t, 1) for t in ((a, ab, ca), (ab, b, bc), (ca, bc, c), (ab, bc, ca))])
    if smooth:
        n = p.copy()
    else:
        fn = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
        n = np.repeat((fn / np.linalg.norm(fn, axis=-1, keepdims=True))[:, None], 3, 1)
    return Mesh(p, n)


def load_obj(path, group=None):
    """Read an OBJ file into a Mesh.

    Handles v/vn and the f formats v, v/vt, v//vn and v/vt/vn, negative
    (relative) indices and polygons of any size (fan-triangulated).
    Texture coordinates and materials are ignored. With group=NAME only
    faces after `usemtl NAME` are kept. Faces without normals get flat ones.
    """
    verts, norms, tris, tri_n = [], [], [], []
    current = None
    with open(path) as fh:
        for line in fh:
            parts = line.split()
            if not parts:
                continue
            tag = parts[0]
            if tag == "v":
                verts.append([float(x) for x in parts[1:4]])
            elif tag == "vn":
                norms.append([float(x) for x in parts[1:4]])
            elif tag == "usemtl":
                current = parts[1] if len(parts) > 1 else None
            elif tag == "f":
                if group is not None and current != group:
                    continue
                vi, ni = [], []
                for ref in parts[1:]:
                    fields = ref.split("/")
                    v = int(fields[0])
                    vi.append(v - 1 if v > 0 else len(verts) + v)
                    if len(fields) > 2 and fields[2]:
                        n = int(fields[2])
                        ni.append(n - 1 if n > 0 else len(norms) + n)
                    else:
                        ni.append(None)
                for k in range(1, len(vi) - 1):
                    tris.append((vi[0], vi[k], vi[k + 1]))
                    tri_n.append((ni[0], ni[k], ni[k + 1]))
    if not tris:
        raise ValueError(f"{path}: no faces" + (f" in group {group!r}" if group else ""))

    V = np.array(verts, np.float64)
    N = np.array(norms, np.float64) if norms else np.zeros((0, 3))
    p = V[np.array(tris)]
    face_n = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
    face_n /= np.maximum(np.linalg.norm(face_n, axis=-1, keepdims=True), 1e-30)
    n = np.repeat(face_n[:, None, :], 3, axis=1)
    has_n = np.array([all(x is not None for x in t) for t in tri_n])
    if has_n.any():
        idx = np.array([t for t, h in zip(tri_n, has_n) if h])
        vn = N[idx]
        vn /= np.maximum(np.linalg.norm(vn, axis=-1, keepdims=True), 1e-30)
        n[has_n] = vn
    return Mesh(p, n)


# ------------------------------------------------------------------ BVH
# Node layout, shared with the C and CUDA traversal code (32 bytes):
#   float bmin[3]; int a; float bmax[3]; int count;
# count > 0: leaf holding triangles a .. a+count-1
# count == 0: inner node, left child is the next node, right child is a
NODE_DTYPE = np.dtype([("bmin", "<f4", 3), ("a", "<i4"), ("bmax", "<f4", 3), ("count", "<i4")])


def build_bvh(p, leaf_size=4, bins=12):
    """Binned-SAH BVH over triangles p (N, 3, 3). Returns (order, nodes):
    triangles must be reordered as p[order] to match the leaves."""
    lo, hi = p.min(1), p.max(1)
    cen = (lo + hi) / 2
    order = np.arange(len(p))
    nodes = []

    def area(b0, b1):
        e = np.maximum(b1 - b0, 0.0)
        return e[..., 0] * e[..., 1] + e[..., 1] * e[..., 2] + e[..., 2] * e[..., 0]

    def build(start, end):
        idx = order[start:end]
        bmin, bmax = lo[idx].min(0), hi[idx].max(0)
        me = len(nodes)
        nodes.append(None)
        n = end - start
        if n <= leaf_size:
            nodes[me] = (bmin, start, bmax, n)
            return me

        c = cen[idx]
        cmin, cmax = c.min(0), c.max(0)
        best_cost, best_axis, best_split = np.inf, -1, -1
        for axis in range(3):
            ext = cmax[axis] - cmin[axis]
            if ext <= 1e-12:
                continue
            b = np.minimum(((c[:, axis] - cmin[axis]) / ext * bins).astype(np.int64), bins - 1)
            cnt = np.bincount(b, minlength=bins)
            blo = np.full((bins, 3), np.inf)
            bhi = np.full((bins, 3), -np.inf)
            np.minimum.at(blo, b, lo[idx])
            np.maximum.at(bhi, b, hi[idx])
            # bounds and counts left/right of each of the bins-1 split planes
            llo = np.minimum.accumulate(blo, 0)[:-1]
            lhi = np.maximum.accumulate(bhi, 0)[:-1]
            rlo = np.minimum.accumulate(blo[::-1], 0)[::-1][1:]
            rhi = np.maximum.accumulate(bhi[::-1], 0)[::-1][1:]
            lc = np.cumsum(cnt)[:-1]
            rc = n - lc
            cost = np.where((lc > 0) & (rc > 0),
                            area(llo, lhi) * lc + area(rlo, rhi) * rc, np.inf)
            k = int(np.argmin(cost))
            if cost[k] < best_cost:
                best_cost, best_axis, best_split = cost[k], axis, k

        leaf_cost = n * area(bmin, bmax)
        if best_axis < 0 or (best_cost >= leaf_cost and n <= 16):
            if n <= 16:
                nodes[me] = (bmin, start, bmax, n)
                return me
            # all centroids in one spot: split the list in half
            mid = start + n // 2
        else:
            axis = best_axis
            ext = cmax[axis] - cmin[axis]
            b = np.minimum(((c[:, axis] - cmin[axis]) / ext * bins).astype(np.int64), bins - 1)
            left = b <= best_split
            order[start:end] = np.concatenate([idx[left], idx[~left]])
            mid = start + int(left.sum())

        build(start, mid)
        right = build(mid, end)
        nodes[me] = (bmin, right, bmax, 0)
        return me

    import sys
    sys.setrecursionlimit(max(sys.getrecursionlimit(), 10000))
    build(0, len(p))
    out = np.zeros(len(nodes), NODE_DTYPE)
    for i, (b0, a, b1, cnt) in enumerate(nodes):
        out[i] = (b0, a, b1, cnt)
    return order, out


def pack(meshes):
    """Merge [(Mesh, material index), ...] into one BVH.

    Returns dict with, per triangle in BVH order:
      v0, e1, e2  (N, 3) float32   vertex 0 and the two edges from it
      n           (N, 3, 3) float32 vertex normals
      mat         (N,) int32        the material index given for its mesh
    and nodes (NODE_DTYPE)."""
    p = np.concatenate([m.p for m, _ in meshes])
    n = np.concatenate([m.n for m, _ in meshes])
    mat = np.concatenate([np.full(len(m), k, np.int32) for m, k in meshes])
    order, nodes = build_bvh(p)
    p, n, mat = p[order], n[order], mat[order]
    return dict(v0=p[:, 0].astype(np.float32),
                e1=(p[:, 1] - p[:, 0]).astype(np.float32),
                e2=(p[:, 2] - p[:, 0]).astype(np.float32),
                n=n.astype(np.float32), mat=mat, nodes=nodes)


def pack_cached(meshes, cache_dir):
    """pack(), but reuse the result from cache_dir when the triangles and
    materials are byte-for-byte the same as last time."""
    import hashlib, os
    h = hashlib.sha1()
    for m, k in meshes:
        h.update(m.p.astype(np.float32).tobytes())
        h.update(m.n.astype(np.float32).tobytes())
        h.update(str(k).encode())
    path = os.path.join(cache_dir, h.hexdigest() + ".npz")
    if os.path.exists(path):
        with np.load(path) as z:
            return {k: z[k] for k in z.files}
    packed = pack(meshes)
    os.makedirs(cache_dir, exist_ok=True)
    np.savez(path, **packed)
    return packed


MAGIC = b"PTMESH1\0"


def write_mesh_file(path, packed):
    """Binary layout read by pathtracer.c (all little endian):
    magic[8], uint32 ntri, uint32 nnode, then per triangle v0,e1,e2 (9 floats),
    then per triangle vertex normals (9 floats), then per triangle int32
    material, then nnode 32-byte nodes."""
    tri = np.concatenate([packed["v0"], packed["e1"], packed["e2"]], axis=1)
    with open(path, "wb") as fh:
        fh.write(MAGIC)
        fh.write(struct.pack("<II", len(tri), len(packed["nodes"])))
        fh.write(tri.astype("<f4").tobytes())
        fh.write(packed["n"].reshape(-1, 9).astype("<f4").tobytes())
        fh.write(packed["mat"].astype("<i4").tobytes())
        fh.write(packed["nodes"].tobytes())

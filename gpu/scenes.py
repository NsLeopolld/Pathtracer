#!/usr/bin/env python3
"""Scene definitions for the GPU spectral path tracer."""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from make_scenes import build_scene

DIFFUSE, CONDUCTOR, DIELECTRIC, PLASTIC, EMISSIVE, THINFILM = range(6)

MAT_DTYPE = np.dtype([
    ("cr", "<f4"), ("cg", "<f4"), ("cb", "<f4"),
    ("c2r", "<f4"), ("c2g", "<f4"), ("c2b", "<f4"),
    ("rough", "<f4"), ("ior", "<f4"), ("abbe", "<f4"),
    ("film", "<f4"), ("cscale", "<f4"),
    ("type", "<i4"), ("checker", "<i4"),
])
assert MAT_DTYPE.itemsize == 52, MAT_DTYPE.itemsize


class Scene:
    def __init__(self, name):
        self.name = name
        self.mats, self.spheres, self.planes, self.boxes = [], [], [], []
        self.cam = dict(frm=(0, 1, 5), at=(0, 1, 0), vfov=35.0,
                        aperture=0.0, focus=None)
        self.sky = ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
        self.exposure = 1.0
        self.bloom = 0.0
        self.look = "none"

    # -- authoring -------------------------------------------------------
    def mat(self, type, colour=(0.8, 0.8, 0.8), rough=0.0, ior=1.5, abbe=0.0,
            film=0.0, checker=None, cscale=1.0):
        m = np.zeros(1, MAT_DTYPE)[0]
        m["cr"], m["cg"], m["cb"] = colour
        if checker is not None:
            m["c2r"], m["c2g"], m["c2b"] = checker
            m["checker"] = 1
        m["rough"], m["ior"], m["abbe"] = rough, ior, abbe
        m["film"], m["cscale"], m["type"] = film, cscale, type
        self.mats.append(m)
        return len(self.mats) - 1

    def sphere(self, centre, radius, mat):
        self.spheres.append((*centre, radius, mat))

    def plane(self, normal, offset, mat):
        """All points p with normal . p = offset. normal needn't be unit length."""
        n = np.array(normal, float)
        length = np.linalg.norm(n)
        self.planes.append((*(n / length), offset / length, mat))

    def box(self, centre, size, mat, rot_y=0.0):
        """Box from its centre and full size, turned rot_y degrees around y."""
        a = np.radians(rot_y)
        half = np.asarray(size, float) / 2
        self.boxes.append((*centre, np.cos(a), *half, np.sin(a), mat))

    # -- packing ---------------------------------------------------------
    def pack(self):
        sph = np.array([s[:4] for s in self.spheres], np.float32).reshape(-1, 4)
        smat = np.array([s[4] for s in self.spheres], np.int32)
        pln = (np.array([p[:4] for p in self.planes], np.float32).reshape(-1, 4)
               if self.planes else np.zeros((0, 4), np.float32))
        pmat = np.array([p[4] for p in self.planes], np.int32)
        # two float4 per box: (centre, cos rot), (half size, sin rot)
        box = np.array([b[:8] for b in self.boxes], np.float32).reshape(-1, 4)
        bmat = np.array([b[8] for b in self.boxes], np.int32)
        mats = np.array(self.mats, MAT_DTYPE)

        lights, power = [], []
        for i, s in enumerate(self.spheres):
            m = mats[s[4]]
            if m["type"] == EMISSIVE:
                lum = 0.2126*m["cr"] + 0.7152*m["cg"] + 0.0722*m["cb"]
                lights.append(i)
                power.append(float(lum) * s[3] ** 2)
        return dict(sph=sph, smat=smat, pln=pln, pmat=pmat, box=box, bmat=bmat, mats=mats,
                    lights=np.array(lights, np.int32),
                    power=np.array(power, np.float32))

    def camera(self, aspect):
        frm = np.array(self.cam["frm"], np.float64)
        at = np.array(self.cam["at"], np.float64)
        focus = self.cam["focus"] or float(np.linalg.norm(frm - at))
        half_h = np.tan(np.radians(self.cam["vfov"]) / 2)
        half_w = aspect * half_h
        w = (frm - at) / np.linalg.norm(frm - at)
        u = np.cross((0, 1, 0), w); u /= np.linalg.norm(u)
        v = np.cross(w, u)
        horiz = 2 * half_w * focus * u
        vert = 2 * half_h * focus * v
        ll = frm - horiz / 2 - vert / 2 - focus * w
        out = np.zeros(20, np.float32)
        out[0:3], out[3:6], out[6:9], out[9:12] = frm, horiz, vert, ll
        out[12:15], out[15:18] = u, v
        out[18] = self.cam["aperture"] / 2
        return out


# ====================================================================== #

def hero():
    """Same layout as scene.h, with the GPU materials."""
    c, r, a, m, p = build_scene()

    s = Scene("hero")
    s.cam = dict(frm=(8.6, 2.15, 6.2), at=(0.0, 0.92, -0.2), vfov=27.0,
                 aperture=0.10, focus=None)
    s.sky = ((0.52, 0.42, 0.38), (0.10, 0.16, 0.30))
    s.bloom, s.exposure, s.look = 0.02, 1.0, "punchy"

    # real plane instead of the big sphere, glossy
    floor = s.mat(PLASTIC, (0.62, 0.60, 0.58), rough=0.30, ior=1.5,
                  checker=(0.14, 0.15, 0.17), cscale=1.15)
    s.plane((0, 1, 0), 0.0, floor)

    for i in range(1, len(r)):                      # 0 is the old ground sphere
        col = tuple(float(x) for x in a[i])
        if m[i] == 0:
            mat = s.mat(DIFFUSE, col)
        elif m[i] == 1:
            # big metal ball gets gold F0, small ones keep their colour
            f0 = (1.00, 0.766, 0.336) if r[i] > 0.9 else col
            mat = s.mat(CONDUCTOR, f0, rough=max(1e-4, float(p[i]) * 0.9))
        elif m[i] == 2:
            mat = s.mat(DIELECTRIC, (1, 1, 1), ior=float(p[i]), abbe=36.0)
        else:
            mat = s.mat(EMISSIVE, col)
        s.sphere(tuple(float(x) for x in c[i]), float(r[i]), mat)
    return s


def cornell():
    s = Scene("cornell")
    s.cam = dict(frm=(0.0, 1.0, 3.4), at=(0.0, 1.0, 0.0), vfov=38.0,
                 aperture=0.0, focus=None)
    s.sky = ((0, 0, 0), (0, 0, 0))
    s.bloom, s.exposure, s.look = 0.015, 1.0, "punchy"

    white = s.mat(DIFFUSE, (0.73, 0.71, 0.68))
    red = s.mat(DIFFUSE, (0.63, 0.065, 0.05))
    green = s.mat(DIFFUSE, (0.14, 0.45, 0.091))
    black = s.mat(DIFFUSE, (0, 0, 0))
    s.plane((0, 1, 0), 0.0, white)        # floor
    s.plane((0, -1, 0), -2.0, white)      # ceiling
    s.plane((0, 0, 1), -1.0, white)       # back
    s.plane((1, 0, 0), -1.0, red)         # left
    s.plane((-1, 0, 0), -1.0, green)      # right
    s.plane((0, 0, -1), -3.6, black)      # behind the camera

    mirror = s.mat(CONDUCTOR, (0.95, 0.95, 0.95), rough=0.0)
    glass = s.mat(DIELECTRIC, (1, 1, 1), ior=1.6, abbe=28.0)   # dispersive
    lamp = s.mat(EMISSIVE, (15, 12, 8))
    s.sphere((-0.45, 0.42, -0.40), 0.42, mirror)
    s.sphere((0.47, 0.42, 0.22), 0.42, glass)
    s.sphere((0.0, 1.95, 0.0), 0.20, lamp)
    return s


def neon():
    s = Scene("neon")
    s.cam = dict(frm=(0.0, 1.1, 7.5), at=(0.0, 0.9, 0.0), vfov=28.0,
                 aperture=0.35, focus=None)
    s.sky = ((0.012, 0.010, 0.025), (0.001, 0.001, 0.004))
    s.bloom, s.exposure, s.look = 0.06, 1.0, "punchy"

    floor = s.mat(PLASTIC, (0.035, 0.035, 0.042), rough=0.11, ior=1.5)
    s.plane((0, 1, 0), 0.0, floor)

    bubble = s.mat(THINFILM, (1, 1, 1), ior=1.33, film=420.0)
    s.sphere((0.0, 1.0, 0.0), 1.0, bubble)

    chrome = s.mat(CONDUCTOR, (0.94, 0.94, 0.97), rough=0.015)
    s.sphere((-2.3, 0.8, -1.2), 0.8, chrome)
    s.sphere((2.1, 0.7, -0.6), 0.7, s.mat(DIFFUSE, (0.80, 0.80, 0.80)))

    palette = [(6.0, 0.35, 2.4), (0.25, 2.8, 6.0), (6.0, 2.2, 0.15),
               (0.4, 6.0, 1.4), (2.6, 0.5, 6.0)]
    for c, r in [((-1.25, 0.25, 1.1), 0.25), ((1.35, 0.2, 1.35), 0.20),
                 ((0.9, 2.6, -1.8), 0.22)]:
        col = [(7.0, 0.4, 2.8), (0.3, 3.4, 7.0), (7.0, 2.6, 0.2)][
            [(-1.25, 0.25, 1.1), (1.35, 0.2, 1.35), (0.9, 2.6, -1.8)].index(c)]
        s.sphere(c, r, s.mat(EMISSIVE, col))

    rng = np.random.default_rng(7)
    def orb(x, y, z, r, boost=1.0):
        col = palette[int(rng.integers(len(palette)))]
        s.sphere((x, y, z), r, s.mat(EMISSIVE, tuple(v*boost for v in col)))
    for _ in range(26):
        z = float(rng.uniform(-14, -4)); half = (7.5 - z) * 0.44
        orb(float(rng.uniform(-half, half)), float(rng.uniform(0.15, 3.2)), z,
            float(rng.uniform(0.07, 0.16)), boost=1.6)
    for _ in range(8):
        r = float(rng.uniform(0.07, 0.12))
        x = float(rng.choice([-1, 1]) * rng.uniform(1.0, 4.5))
        orb(x, r, float(rng.uniform(-3.5, 2.5)), r)
    for x, y, z in [(-1.0, 0.55, 5.3), (1.15, 1.45, 5.0), (0.45, 0.25, 5.6)]:
        orb(x, y, z, 0.06, boost=1.3)
    return s


# -------------------------------------------------------- test scenes --
def furnace(kind="diffuse", shape="sphere"):
    """White furnace test scene: albedo-1 object in a uniform sky of 1.0.

    Should render as exactly 1.0 everywhere. If not, something is losing
    or adding energy (BSDF, spectral conversion, or the estimator).
    """
    s = Scene(f"furnace-{kind}" + ("" if shape == "sphere" else f"-{shape}"))
    s.cam = dict(frm=(0, 0, 4), at=(0, 0, 0), vfov=30.0, aperture=0.0, focus=None)
    s.sky = ((1, 1, 1), (1, 1, 1))
    mats = dict(
        diffuse=lambda: s.mat(DIFFUSE, (1, 1, 1)),
        conductor=lambda: s.mat(CONDUCTOR, (1, 1, 1), rough=0.3),
        glass=lambda: s.mat(DIELECTRIC, (1, 1, 1), ior=1.5),
        dispersive=lambda: s.mat(DIELECTRIC, (1, 1, 1), ior=1.5, abbe=30.0),
        film=lambda: s.mat(THINFILM, (1, 1, 1), ior=1.33, film=420.0),
        plastic=lambda: s.mat(PLASTIC, (1, 1, 1), rough=0.2),
    )
    if shape == "box":
        s.box((0, 0, 0), (1.4, 1.4, 1.4), mats[kind](), rot_y=30.0)
    else:
        s.sphere((0, 0, 0), 1.0, mats[kind]())
    return s


def mistest():
    """One light over a floor, for checking NEE / BSDF / MIS give the same mean."""
    s = Scene("mistest")
    s.cam = dict(frm=(0, 2.0, 5.0), at=(0, 0.4, 0), vfov=35.0, aperture=0.0, focus=None)
    s.sky = ((0, 0, 0), (0, 0, 0))
    s.plane((0, 1, 0), 0.0, s.mat(DIFFUSE, (0.6, 0.6, 0.6)))
    s.sphere((0.0, 1.6, 0.0), 0.35, s.mat(EMISSIVE, (12, 12, 12)))
    s.sphere((-1.1, 0.5, 0.3), 0.5, s.mat(CONDUCTOR, (0.9, 0.9, 0.9), rough=0.25))
    s.sphere((1.1, 0.5, 0.3), 0.5, s.mat(PLASTIC, (0.4, 0.5, 0.7), rough=0.15))
    return s


def mistest_area():
    """mistest plus a glowing box and a glowing plane. Those aren't NEE'd,
    so this checks that non-sphere emitters are weighted right."""
    s = mistest()
    s.name = "mistest-area"
    panel = s.mat(EMISSIVE, (3, 3, 3))
    s.box((1.8, 1.2, -1.2), (0.5, 0.5, 0.5), panel, rot_y=20.0)
    s.plane((0, 0, 1), -3.0, s.mat(EMISSIVE, (0.2, 0.25, 0.3)))    # glowing back wall
    return s


def cornell_boxes():
    """Classic Cornell box with two rotated boxes, same layout as scene_boxes.h."""
    s = cornell()
    s.name = "cornell-boxes"
    s.spheres = [sp for sp in s.spheres if s.mats[sp[4]]["type"] == EMISSIVE]   # keep the lamp
    white = s.mat(DIFFUSE, (0.73, 0.71, 0.68))
    s.box((-0.33, 0.6, -0.35), (0.58, 1.2, 0.58), white, rot_y=18.0)
    s.box((0.34, 0.3, 0.3), (0.58, 0.6, 0.58), white, rot_y=-17.0)
    return s


REGISTRY = dict(hero=hero, cornell=cornell, neon=neon, mistest=mistest,
                **{"mistest-area": mistest_area, "cornell-boxes": cornell_boxes},
                **{f"furnace-{k}": (lambda k=k: furnace(k)) for k in
                   ("diffuse", "conductor", "glass", "dispersive", "film", "plastic")},
                **{f"furnace-{k}-box": (lambda k=k: furnace(k, "box")) for k in
                   ("diffuse", "glass", "film")})

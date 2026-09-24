#!/usr/bin/env python3
"""
NumPy path tracer with its own PNG writer (zlib only for deflate + crc32).

Cosine-weighted diffuse, Schlick dielectrics, fuzzy metal, thin lens DOF,
NEE on sphere lights, Russian roulette. Vectorized over rays, so the whole
frame advances one bounce per iteration.

    python3 pathtracer.py [--width 640] [--spp 64] [--out render.png]
"""

from __future__ import annotations

import argparse
import os
import struct
import sys
import time
import zlib
from concurrent.futures import ProcessPoolExecutor

import numpy as np

# --------------------------------------------------------------------------
# PNG output
# --------------------------------------------------------------------------

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def _chunk(kind: bytes, payload: bytes) -> bytes:
    """length + type + data + crc32(type + data)"""
    crc = zlib.crc32(kind + payload) & 0xFFFFFFFF
    return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", crc)


def _filter_scanlines(rgb: np.ndarray) -> bytes:
    """Per-scanline adaptive filtering (None/Sub/Up/Average/Paeth).

    Picks the filter with the smallest sum of abs signed bytes per row,
    the usual libpng heuristic. Filters read unfiltered neighbours, so all
    five candidates for a row are computed at once.
    """
    height, width, _ = rgb.shape
    stride = width * 3  # bytes per scanline
    rows = rgb.reshape(height, stride).astype(np.int16)
    prior = np.zeros(stride, dtype=np.int16)
    out = bytearray()

    for y in range(height):
        cur = rows[y]
        left = np.concatenate((np.zeros(3, np.int16), cur[:-3]))       # a
        up = prior                                                      # b
        upleft = np.concatenate((np.zeros(3, np.int16), prior[:-3]))    # c

        # paeth: whichever of a, b, c is closest to a + b - c
        p = left + up - upleft
        pa, pb, pc = np.abs(p - left), np.abs(p - up), np.abs(p - upleft)
        paeth = np.where(
            (pa <= pb) & (pa <= pc), left, np.where(pb <= pc, up, upleft)
        )

        candidates = (
            cur,                                # 0 None
            cur - left,                         # 1 Sub
            cur - up,                           # 2 Up
            cur - ((left + up) >> 1),           # 3 Average
            cur - paeth,                        # 4 Paeth
        )

        best, best_cost = 0, None
        for idx, cand in enumerate(candidates):
            band = cand.astype(np.uint8).astype(np.int16)
            # treat bytes as signed, so 255 counts as -1
            cost = int(np.minimum(band, 256 - band).sum())
            if best_cost is None or cost < best_cost:
                best, best_cost = idx, cost

        out.append(best)
        out += candidates[best].astype(np.uint8).tobytes()
        prior = cur

    return bytes(out)


def write_png(path: str, rgb: np.ndarray) -> int:
    """Write an (H, W, 3) uint8 array as an 8-bit RGB PNG. Returns file size."""
    height, width, _ = rgb.shape
    ihdr = struct.pack(
        ">IIBBBBB",
        width, height,
        8,   # bit depth
        2,   # colour type 2 = truecolour RGB
        0,   # compression method (deflate, only option)
        0,   # filter method 0
        0,   # no interlacing
    )

    compressor = zlib.compressobj(level=9, strategy=zlib.Z_FILTERED)
    idat = compressor.compress(_filter_scanlines(rgb)) + compressor.flush()

    blob = (
        PNG_SIGNATURE
        + _chunk(b"IHDR", ihdr)
        + _chunk(b"gAMA", struct.pack(">I", 45455))   # gamma 1/2.2
        + _chunk(b"IDAT", idat)
        + _chunk(b"IEND", b"")
    )
    with open(path, "wb") as fh:
        fh.write(blob)
    return len(blob)


# --------------------------------------------------------------------------
# vector helpers, all on (N, 3) arrays
# --------------------------------------------------------------------------

LAMBERTIAN, METAL, DIELECTRIC, EMISSIVE = 0, 1, 2, 3


def unit(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def dot(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.einsum("ij,ij->i", a, b)


def random_unit(n: int, rng: np.random.Generator) -> np.ndarray:
    """Uniform on the unit sphere (normalized gaussian)."""
    v = rng.standard_normal((n, 3))
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


# --------------------------------------------------------------------------
# scene
# --------------------------------------------------------------------------


def build_scene():
    centers, radii, albedos, mats, params = [], [], [], [], []

    def add(center, radius, albedo, mat, param=0.0):
        centers.append(center)
        radii.append(radius)
        albedos.append(albedo)
        mats.append(mat)
        params.append(param)

    # ground, big enough to pass for a plane. checker is applied in trace()
    add((0.0, -1000.0, 0.0), 1000.0, (0.0, 0.0, 0.0), LAMBERTIAN, 1.0)

    # main row
    add((0.0, 1.0, 0.0), 1.0, (1.0, 1.0, 1.0), DIELECTRIC, 1.52)   # glass
    add((-2.35, 1.0, -0.35), 1.0, (0.72, 0.28, 0.22), LAMBERTIAN)   # clay
    add((2.35, 1.0, -0.35), 1.0, (0.93, 0.87, 0.74), METAL, 0.04)   # gold

    # negative radius = inward normals, makes the glass ball hollow
    add((0.0, 1.0, 0.0), -0.82, (1.0, 1.0, 1.0), DIELECTRIC, 1.52)

    # warm key light (off screen) + cool rim light
    add((-6.0, 7.5, 4.0), 2.2, (10.0, 7.4, 4.6), EMISSIVE)
    add((7.5, 3.0, -6.0), 1.6, (1.6, 3.0, 5.2), EMISSIVE)

    # small random spheres, fixed seed so the scene is reproducible
    rng = np.random.default_rng(20260922)
    for _ in range(38):
        r = float(rng.uniform(0.14, 0.26))
        ang = float(rng.uniform(0, 2 * np.pi))
        dist = float(rng.uniform(2.0, 8.5))
        cx, cz = np.cos(ang) * dist, np.sin(ang) * dist - 1.0
        if abs(cx) < 3.6 and abs(cz) < 1.4:
            continue  # don't overlap the main row
        roll = rng.random()
        if roll < 0.55:
            tint = rng.uniform(0.15, 0.85, 3)
            add((cx, r, cz), r, tuple(tint * tint), LAMBERTIAN)
        elif roll < 0.85:
            tint = rng.uniform(0.55, 1.0, 3)
            add((cx, r, cz), r, tuple(tint), METAL, float(rng.uniform(0.0, 0.25)))
        else:
            add((cx, r, cz), r, (1.0, 1.0, 1.0), DIELECTRIC, 1.5)

    return (
        np.array(centers, np.float64),
        np.array(radii, np.float64),
        np.array(albedos, np.float64),
        np.array(mats, np.int32),
        np.array(params, np.float64),
    )


def sky(d: np.ndarray) -> np.ndarray:
    """Vertical gradient, horizon -> zenith. Only light besides the emitters."""
    t = (0.5 * (d[:, 1] + 1.0))[:, None]
    horizon = np.array([0.52, 0.42, 0.38])
    zenith = np.array([0.10, 0.16, 0.30])
    return (1.0 - t) * horizon + t * zenith


# --------------------------------------------------------------------------
# intersection / shading
# --------------------------------------------------------------------------


def intersect(origins, dirs, centers, radii):
    """Closest hit per ray. Loop is over spheres, rays are vectorized."""
    n = origins.shape[0]
    best_t = np.full(n, np.inf)
    best_id = np.full(n, -1, np.int32)

    for i in range(centers.shape[0]):
        oc = origins - centers[i]
        # dirs are normalized so a == 1
        half_b = dot(dirs, oc)
        c = dot(oc, oc) - radii[i] * radii[i]
        disc = half_b * half_b - c
        live = disc > 0.0
        if not live.any():
            continue
        root = np.sqrt(np.where(live, disc, 0.0))
        t = -half_b - root
        far = -half_b + root
        t = np.where(t > 1e-4, t, far)
        hit = live & (t > 1e-4) & (t < best_t)
        best_t = np.where(hit, t, best_t)
        best_id = np.where(hit, i, best_id)

    return best_t, best_id


def checker(points: np.ndarray) -> np.ndarray:
    """Floor checker from sign(sin x * sin z)."""
    s = np.sin(points[:, 0] * 1.15) * np.sin(points[:, 2] * 1.15)
    light = np.array([0.62, 0.60, 0.58])
    dark = np.array([0.14, 0.15, 0.17])
    return np.where((s > 0)[:, None], light, dark)


def onb(w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Two vectors perpendicular to w (w assumed normalized)."""
    a = np.where(
        np.abs(w[:, 0:1]) > 0.9,
        np.array([[0.0, 1.0, 0.0]]),
        np.array([[1.0, 0.0, 0.0]]),
    )
    v = unit(np.cross(w, a))
    return np.cross(w, v), v


def sample_lights(point, facing, albedo, throughput, scene, rng, lights):
    """NEE for diffuse hits: pick a light uniformly, sample the cone it
    subtends. pdf = 1 / (2*pi*(1 - cos_max)) in solid angle.
    """
    centers, radii, albedos, _, _ = scene
    m = point.shape[0]
    out = np.zeros((m, 3))
    if m == 0 or lights.size == 0:
        return out

    chosen = lights[rng.integers(0, lights.size, m)]
    to_light = centers[chosen] - point
    dist2 = np.einsum("ij,ij->i", to_light, to_light)
    dist = np.sqrt(dist2)
    radius = np.abs(radii[chosen])

    # cone half-angle as seen from the shading point
    cos_max = np.sqrt(np.maximum(0.0, 1.0 - (radius * radius) / np.maximum(dist2, 1e-12)))
    cos_t = 1.0 - rng.random(m) * (1.0 - cos_max)
    sin_t = np.sqrt(np.maximum(0.0, 1.0 - cos_t * cos_t))
    phi = 2.0 * np.pi * rng.random(m)

    w = to_light / dist[:, None]
    u, v = onb(w)
    ldir = (
        (np.cos(phi) * sin_t)[:, None] * u
        + (np.sin(phi) * sin_t)[:, None] * v
        + cos_t[:, None] * w
    )

    cos_surf = np.einsum("ij,ij->i", ldir, facing)
    valid = (dist > radius * 1.0001) & (cos_surf > 1e-6)
    if not valid.any():
        return out

    # visible if the first thing the shadow ray hits is the light itself
    origin = point[valid] + 1e-4 * facing[valid]
    _, hit_id = intersect(origin, ldir[valid], centers, radii)
    visible = hit_id == chosen[valid]
    if not visible.any():
        return out

    keep = np.flatnonzero(valid)[visible]
    pdf = 1.0 / (2.0 * np.pi * np.maximum(1.0 - cos_max[keep], 1e-9))
    # (albedo/pi) * cos / pdf, * N for the uniform light pick
    weight = (cos_surf[keep] / (np.pi * pdf) * lights.size)[:, None]
    out[keep] = throughput[keep] * albedo[keep] * albedos[chosen[keep]] * weight
    return out


def trace(origins, dirs, scene, rng, max_depth=24):
    """Bounce all rays together, dropping dead paths each iteration."""
    centers, radii, albedos, mats, params = scene
    lights = np.flatnonzero(mats == EMISSIVE)
    if os.environ.get("NO_NEE"):
        lights = lights[:0]
    n = origins.shape[0]

    radiance = np.zeros((n, 3))
    throughput = np.ones((n, 3))
    alive = np.arange(n)  # live ray -> pixel index
    # only count emitter hits after a specular bounce; diffuse bounces
    # already got that light through NEE, so it would be counted twice
    specular = np.ones(n, bool)

    for depth in range(max_depth):
        if alive.size == 0:
            break

        t, sid = intersect(origins, dirs, centers, radii)
        missed = sid < 0
        if missed.any():
            idx = alive[missed]
            radiance[idx] += throughput[missed] * sky(dirs[missed])

        keep = ~missed
        if not keep.any():
            break

        origins, dirs = origins[keep], dirs[keep]
        throughput, alive, specular = throughput[keep], alive[keep], specular[keep]
        t, sid = t[keep], sid[keep]

        point = origins + t[:, None] * dirs
        # signed radius, so the inner shell gets inward normals
        normal = (point - centers[sid]) / radii[sid][:, None]
        front = dot(dirs, normal) < 0.0
        facing = np.where(front[:, None], normal, -normal)

        mat = mats[sid]
        albedo = albedos[sid]
        param = params[sid]

        # sphere 0 is the floor
        floor = sid == 0
        if floor.any():
            albedo = albedo.copy()
            albedo[floor] = checker(point[floor])

        # emissive
        is_emissive = mat == EMISSIVE
        counts = is_emissive & specular
        if counts.any():
            radiance[alive[counts]] += throughput[counts] * albedo[counts]

        # diffuse: NEE, then cosine-weighted bounce
        new_dir = np.zeros_like(dirs)
        is_diffuse = mat == LAMBERTIAN
        if is_diffuse.any():
            radiance[alive[is_diffuse]] += sample_lights(
                point[is_diffuse], facing[is_diffuse], albedo[is_diffuse],
                throughput[is_diffuse], scene, rng, lights,
            )
            scattered = facing[is_diffuse] + random_unit(int(is_diffuse.sum()), rng)
            degenerate = np.linalg.norm(scattered, axis=-1) < 1e-8
            scattered[degenerate] = facing[is_diffuse][degenerate]
            new_dir[is_diffuse] = unit(scattered)
            throughput[is_diffuse] *= albedo[is_diffuse]

        # metal: reflect + fuzz
        is_metal = mat == METAL
        if is_metal.any():
            d, nrm = dirs[is_metal], facing[is_metal]
            reflected = d - 2.0 * dot(d, nrm)[:, None] * nrm
            reflected = unit(reflected) + param[is_metal][:, None] * random_unit(
                int(is_metal.sum()), rng
            )
            new_dir[is_metal] = unit(reflected)
            throughput[is_metal] *= albedo[is_metal]

        # glass: reflect or refract, chosen by schlick
        is_glass = mat == DIELECTRIC
        if is_glass.any():
            d, nrm = dirs[is_glass], facing[is_glass]
            ior = param[is_glass]
            ratio = np.where(front[is_glass], 1.0 / ior, ior)

            cos_theta = np.minimum(-dot(d, nrm), 1.0)
            sin_theta = np.sqrt(np.maximum(0.0, 1.0 - cos_theta * cos_theta))

            r0 = ((1.0 - ratio) / (1.0 + ratio)) ** 2
            schlick = r0 + (1.0 - r0) * (1.0 - cos_theta) ** 5
            must_reflect = (ratio * sin_theta > 1.0) | (
                rng.random(int(is_glass.sum())) < schlick
            )

            mirror = d - 2.0 * dot(d, nrm)[:, None] * nrm
            perp = ratio[:, None] * (d + cos_theta[:, None] * nrm)
            par = -np.sqrt(
                np.abs(1.0 - np.minimum(dot(perp, perp), 1.0))
            )[:, None] * nrm
            new_dir[is_glass] = np.where(must_reflect[:, None], mirror, perp + par)

        # emitters end the path
        survives = ~is_emissive
        origins = point[survives] + 1e-4 * new_dir[survives]
        dirs = unit(new_dir[survives])
        throughput, alive = throughput[survives], alive[survives]
        specular = np.ones(alive.size, bool) if lights.size == 0 else ~is_diffuse[survives]

        # russian roulette
        if depth >= 4 and alive.size:
            p = np.clip(throughput.max(axis=1), 0.05, 1.0)
            live = rng.random(alive.size) < p
            throughput = throughput[live] / p[live][:, None]
            origins, dirs, alive = origins[live], dirs[live], alive[live]
            specular = specular[live]

    return radiance


# --------------------------------------------------------------------------
# camera / main
# --------------------------------------------------------------------------


class Camera:
    def __init__(self, look_from, look_at, vup, vfov, aspect, aperture, focus):
        theta = np.radians(vfov)
        half_h = np.tan(theta / 2.0)
        half_w = aspect * half_h

        self.w = unit(np.array([look_from - look_at]))[0]
        self.u = unit(np.array([np.cross(vup, self.w)]))[0]
        self.v = np.cross(self.w, self.u)

        self.origin = np.asarray(look_from, np.float64)
        self.horizontal = 2.0 * half_w * focus * self.u
        self.vertical = 2.0 * half_h * focus * self.v
        self.lower_left = (
            self.origin - self.horizontal / 2 - self.vertical / 2 - focus * self.w
        )
        self.lens_radius = aperture / 2.0

    def rays(self, s, t, rng):
        n = s.shape[0]
        # point on the lens disc (DOF)
        ang = rng.random(n) * 2.0 * np.pi
        rad = self.lens_radius * np.sqrt(rng.random(n))
        offset = (rad * np.cos(ang))[:, None] * self.u + (
            rad * np.sin(ang)
        )[:, None] * self.v

        origin = self.origin + offset
        target = (
            self.lower_left
            + s[:, None] * self.horizontal
            + t[:, None] * self.vertical
        )
        return origin, unit(target - origin)


def render_tile(args):
    """Worker: render `samples` full-frame passes and return the sum."""
    width, height, samples, seed, max_depth = args
    rng = np.random.default_rng(seed)
    scene = build_scene()

    look_from = np.array([8.6, 2.15, 6.2])
    look_at = np.array([0.0, 0.92, -0.2])
    cam = Camera(
        look_from, look_at, np.array([0.0, 1.0, 0.0]),
        vfov=27.0, aspect=width / height,
        aperture=0.10, focus=float(np.linalg.norm(look_from - look_at)),
    )

    px, py = np.meshgrid(np.arange(width), np.arange(height))
    px = px.ravel().astype(np.float64)
    py = py.ravel().astype(np.float64)

    total = np.zeros((width * height, 3))
    for _ in range(samples):
        # jitter within the pixel for AA
        s = (px + rng.random(px.size)) / width
        t = 1.0 - (py + rng.random(py.size)) / height
        origins, dirs = cam.rays(s, t, rng)
        total += trace(origins, dirs, scene, rng, max_depth)
    return total


def tonemap(hdr: np.ndarray) -> np.ndarray:
    """ACES fit (Narkowicz) + gamma 2.2."""
    a, b, c, d, e = 2.51, 0.03, 2.43, 0.59, 0.14
    x = np.maximum(hdr, 0.0)
    mapped = np.clip((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0)
    return (np.power(mapped, 1.0 / 2.2) * 255.0 + 0.5).astype(np.uint8)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--width", type=int, default=720)
    ap.add_argument("--spp", type=int, default=96)
    ap.add_argument("--depth", type=int, default=24)
    ap.add_argument("--out", default="render.png")
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    opts = ap.parse_args()

    width = opts.width
    height = int(round(width * 9 / 16))
    workers = max(1, opts.workers)

    # split spp across processes, each with its own seed; results are summed
    per = [opts.spp // workers] * workers
    for i in range(opts.spp % workers):
        per[i] += 1
    per = [p for p in per if p]

    print(
        f"  {width}x{height}  {opts.spp} spp  depth {opts.depth}  "
        f"{len(per)} workers",
        file=sys.stderr,
    )
    start = time.time()

    jobs = [(width, height, p, 0xC0FFEE + i, opts.depth) for i, p in enumerate(per)]
    if len(jobs) == 1:
        buffers = [render_tile(jobs[0])]
    else:
        with ProcessPoolExecutor(max_workers=len(jobs)) as pool:
            buffers = list(pool.map(render_tile, jobs))

    hdr = np.sum(buffers, axis=0) / opts.spp
    elapsed = time.time() - start

    rgb = tonemap(hdr).reshape(height, width, 3)
    size = write_png(opts.out, rgb)

    rays = width * height * opts.spp
    print(
        f"  {elapsed:6.1f}s   {rays / elapsed / 1e6:.2f}M primary rays/s\n"
        f"  {opts.out}  {size / 1024:.1f} KiB  "
        f"({size / (width * height * 3) * 100:.1f}% of raw RGB)",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

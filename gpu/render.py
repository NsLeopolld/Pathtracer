#!/usr/bin/env python3
"""
Host side of the GPU spectral path tracer: scene upload, adaptive sampling
loop, post (glare -> exposure -> AgX -> dither -> PNG).
"""

import argparse, os, sys, time
import numpy as np
import cupy as cp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import scenes as SC
from png import write_png


# ----------------------------------------------------------------- module
def build_module(ld_groups=8):
    src = open(os.path.join(os.path.dirname(__file__), "kernel.cu")).read()
    opts = ["--use_fast_math", f"-DLD_GROUPS={ld_groups}u"]
    return cp.RawModule(code=src, options=tuple(opts), backend="nvrtc")


_keep = []     # device arrays the kernel reaches through c_tri etc.; must stay alive


def _pointer(mod, name, arr):
    """Store a device array's address in a __constant__ pointer."""
    _keep.append(arr)
    cp.ndarray((1,), cp.uint64, memptr=mod.get_global(name))[...] = arr.data.ptr


def _upload_tris(mod, t):
    """Triangles and BVH from meshes.pack(), in the float4 layout kernel.cu reads."""
    n = len(t["mat"])
    tri = np.zeros((n, 3, 4), np.float32)
    tri[:, 0, :3], tri[:, 1, :3], tri[:, 2, :3] = t["v0"], t["e1"], t["e2"]
    nrm = np.zeros((n, 3, 4), np.float32)
    nrm[:, :, :3] = t["n"]
    _pointer(mod, "c_tri", cp.asarray(tri))
    _pointer(mod, "c_tnrm", cp.asarray(nrm))
    _pointer(mod, "c_tmat", cp.asarray(t["mat"].astype(np.int32)))
    # the node struct is already two float4s with the ints stored as bits
    _pointer(mod, "c_bvh", cp.asarray(np.frombuffer(t["nodes"].tobytes(), np.float32)))
    return n, len(t["nodes"])


def _const(mod, name, arr):
    if arr.size == 0:
        return
    ptr = mod.get_global(name)
    cp.ndarray(arr.shape, dtype=arr.dtype, memptr=ptr)[...] = cp.asarray(arr)


# must match MAXS / MAXP / MAXB / MAXLI in kernel.cu
LIMITS = dict(sph=64, pln=8, bmat=16, lights=64)


def upload(mod, scene, width, height):
    p = scene.pack()
    for key, cap in LIMITS.items():
        if len(p[key]) > cap:
            what = dict(sph="spheres", pln="planes", bmat="boxes", lights="lights")[key]
            raise ValueError(f"scene {scene.name!r} has {len(p[key])} {what}, the kernel "
                             f"supports {cap} (see MAX* in kernel.cu)")
    _const(mod, "c_sph", p["sph"])
    _const(mod, "c_smat", p["smat"])
    _const(mod, "c_pln", p["pln"])
    _const(mod, "c_pmat", p["pmat"])
    _const(mod, "c_box", p["box"])
    _const(mod, "c_bmat", p["bmat"])
    _const(mod, "c_light", p["lights"])
    _const(mod, "c_lpow", p["power"])
    ntri, nnode = _upload_tris(mod, p["tris"]) if p["tris"] is not None else (0, 0)
    _const(mod, "c_cnt", np.array([len(p["sph"]), len(p["pln"]), len(p["lights"]),
                                   len(p["bmat"]), ntri, nnode, 0, 0], np.int32))
    _const(mod, "c_cam", scene.camera(width / height))
    sky = np.zeros(8, np.float32)
    sky[0:3], sky[3:6] = scene.sky[0], scene.sky[1]
    _const(mod, "c_sky", sky)
    _const(mod, "c_sobol", np.load(os.path.join(os.path.dirname(__file__),
                                                "sobol_directions.npy")))
    _const(mod, "c_ealb", np.load(os.path.join(os.path.dirname(__file__),
                                               "ggx_albedo.npy")).ravel())
    t = np.load(os.path.join(os.path.dirname(__file__), "spectral_tables.npz"))
    tab = np.concatenate([t["basis"][0], t["basis"][1], t["basis"][2],
                          t["illum"], t["resp"][0], t["resp"][1],
                          t["resp"][2]]).astype(np.float32)
    return cp.asarray(np.frombuffer(p["mats"].tobytes(), np.uint8)), cp.asarray(tab)


# ------------------------------------------------------------------ render
def render(scene, width, height, max_spp=4096, min_spp=64, chunk=16,
           threshold=0.010, max_time=300.0, depth=24, mis_mode=0,
           clamp=0.0, adaptive=True, ld_groups=0, quiet=False):
    mod = build_module(ld_groups)
    kern = mod.get_function("render")
    mats, tab = upload(mod, scene, width, height)

    npix = width * height
    accum = cp.zeros(npix * 4, cp.float64)
    counts = cp.zeros(npix, cp.int32)
    active = cp.arange(npix, dtype=cp.int32)

    block, t0, samples, it = 128, time.perf_counter(), 0, 0
    # keep launches around a fixed amount of work: as fewer pixels stay
    # active, give each one more samples per launch, otherwise launch/sync
    # overhead dominates
    QUANTUM = 24_000_000
    while active.size:
        n = int(active.size)
        this = chunk if not adaptive else int(min(512, max(chunk, QUANTUM // max(n, 1))))
        # active pixels have all been sampled every launch, so they hold the max count
        this = max(1, min(this, max_spp - int(counts.max())))
        grid = (n + block - 1) // block
        kern((grid,), (block,),
             (active, np.int32(n), accum, counts, mats, tab,
              np.int32(width), np.int32(height), np.int32(this),
              np.int32(depth), np.uint32(0x9e37 + it), np.int32(mis_mode),
              np.float32(clamp)))
        samples += n * this
        it += 1
        cp.cuda.Device().synchronize()
        elapsed = time.perf_counter() - t0
        done = int(counts.max())

        if done >= max_spp or elapsed > max_time:
            break
        if adaptive and done >= min_spp:
            nn = counts.astype(cp.float64)
            mean = accum.reshape(-1, 4)
            inv = 1.0 / cp.maximum(nn, 1)
            lum = (0.2126*mean[:, 0] + 0.7152*mean[:, 1] + 0.0722*mean[:, 2]) * inv
            var = cp.maximum(mean[:, 3]*inv - lum*lum, 0.0) * (nn/cp.maximum(nn-1, 1))
            rel = cp.sqrt(var*inv) / (lum + 0.02)
            active = cp.flatnonzero(rel > threshold).astype(cp.int32)
        if not quiet and it % 8 == 0:
            print(f"    {elapsed:6.1f}s  spp={done:5d}  active={active.size:8d}"
                  f"  {samples/elapsed/1e6:6.2f} Msamp/s", flush=True)

    cp.cuda.Device().synchronize()
    elapsed = time.perf_counter() - t0
    n = cp.maximum(counts, 1).astype(cp.float64)
    hdr = (accum.reshape(-1, 4)[:, :3] / n[:, None]).reshape(height, width, 3)
    stats = dict(time=elapsed, samples=samples, rate=samples/elapsed,
                 spp_max=int(counts.max()), spp_min=int(counts.min()),
                 spp_mean=float(counts.mean()))
    return hdr.astype(cp.float32), stats


# -------------------------------------------------------------------- post
AGX = np.array([[0.842479062253094, 0.0784335999999992, 0.0792237451477643],
                [0.0423282422610123, 0.878468636469772, 0.0791661274605434],
                [0.0423756549057051, 0.0784336, 0.879142973793104]], np.float32)
AGX_INV = np.array([[1.19687900512017, -0.0980208811401368, -0.0990297440797205],
                    [-0.0528968517574562, 1.15190312990417, -0.0989611768448433],
                    [-0.0529716355144438, -0.0980434501171241, 1.15107367264116]], np.float32)


def agx(img, look="none"):
    """AgX tonemap (Sobotka, polynomial fit by Wrensch).

    Using this over ACES because the saturated neon lights go to white
    instead of shifting hue when they clip."""
    x = img @ cp.asarray(AGX.T)
    x = cp.clip(cp.log2(cp.maximum(x, 1e-10)), -12.47393, 4.026069)
    x = (x + 12.47393) / (4.026069 + 12.47393)
    x2 = x*x; x4 = x2*x2
    x = (15.5*x4*x2 - 40.14*x4*x + 31.96*x4 - 6.868*x2*x
         + 0.4298*x2 + 0.1191*x - 0.00232)
    if look == "punchy":
        lum = (x*cp.asarray(np.array([0.2126, 0.7152, 0.0722], np.float32))).sum(-1, keepdims=True)
        x = cp.clip(lum + 1.28*(x - lum), 0, 1) ** 1.02
    x = x @ cp.asarray(AGX_INV.T)
    return cp.clip(x, 0.0, 1.0)


def aces(img):
    a, b, c, d, e = 2.51, 0.03, 2.43, 0.59, 0.14
    x = cp.maximum(img, 0)
    return cp.clip((x*(a*x + b)) / (x*(c*x + d) + e), 0, 1) ** (1/2.2)


def glare(img, strength):
    """Glare: convolve with a sum of 3 gaussians (FFT), blend by strength.

    Otherwise bright emitters just clip to flat white."""
    if strength <= 0:
        return img
    H, W, _ = img.shape
    yy = (cp.arange(H) - H//2)[:, None].astype(cp.float32)
    xx = (cp.arange(W) - W//2)[None, :].astype(cp.float32)
    r2 = yy*yy + xx*xx
    base = max(H, W)
    k = cp.zeros((H, W), cp.float32)
    for sig, wt in ((base*0.004, 0.55), (base*0.016, 0.30), (base*0.065, 0.15)):
        k += wt * cp.exp(-r2 / (2*sig*sig)) / (2*np.pi*sig*sig)
    k /= k.sum()
    K = cp.fft.rfft2(cp.fft.ifftshift(k))
    out = cp.empty_like(img)
    for c in range(3):
        out[:, :, c] = cp.fft.irfft2(cp.fft.rfft2(img[:, :, c]) * K, s=(H, W))
    return (1.0 - strength)*img + strength*cp.maximum(out, 0)


def to_png(hdr, path, exposure=1.0, tonemap="agx", look="none", bloom=0.0):
    img = glare(hdr * exposure, bloom)
    ldr = agx(img, look) if tonemap == "agx" else aces(img)
    # triangular dither, fixes banding in dark gradients
    rng = cp.random.RandomState(1234)
    tri = (rng.random_sample(ldr.shape, dtype=cp.float32)
           - rng.random_sample(ldr.shape, dtype=cp.float32)) * 0.5
    out = cp.clip(ldr*255.0 + tri + 0.5, 0, 255).astype(cp.uint8)
    return write_png(path, cp.asnumpy(out))


def save_pfm(hdr, path):
    a = cp.asnumpy(hdr)[::-1]
    with open(path, "wb") as f:
        f.write(b"PF\n%d %d\n-1.0\n" % (a.shape[1], a.shape[0]))
        f.write(a.astype("<f4").tobytes())


# --------------------------------------------------------------------- cli
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="cornell", choices=sorted(SC.REGISTRY))
    ap.add_argument("--width", type=int, default=960)
    ap.add_argument("--height", type=int, default=0)
    ap.add_argument("--spp", type=int, default=4096)
    ap.add_argument("--min-spp", type=int, default=64)
    ap.add_argument("--chunk", type=int, default=16)
    ap.add_argument("--threshold", type=float, default=0.010)
    ap.add_argument("--max-time", type=float, default=300.0)
    ap.add_argument("--depth", type=int, default=24)
    ap.add_argument("--mis-mode", type=int, default=0, choices=(0, 1, 2))
    ap.add_argument("--clamp", type=float, default=0.0)
    ap.add_argument("--no-adaptive", action="store_true")
    ap.add_argument("--no-bloom", action="store_true")
    ap.add_argument("--tonemap", default="agx", choices=("agx", "aces"))
    ap.add_argument("--ld-groups", type=int, default=0)  # tried sobol, wasn't worth the cost here
    ap.add_argument("--save-hdr", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    scene = SC.REGISTRY[a.scene]()
    h = a.height or int(round(a.width * 9 / 16))
    out = a.out or f"render_gpu_{a.scene}.png"
    print(f"  {a.scene}: {a.width}x{h}  max {a.spp} spp  depth {a.depth}"
          f"  {'adaptive' if not a.no_adaptive else 'uniform'}")

    hdr, st = render(scene, a.width, h, max_spp=a.spp, min_spp=a.min_spp,
                     chunk=a.chunk, threshold=a.threshold, max_time=a.max_time,
                     depth=a.depth, mis_mode=a.mis_mode, clamp=a.clamp,
                     adaptive=not a.no_adaptive, ld_groups=a.ld_groups)
    size = to_png(hdr, out, scene.exposure, a.tonemap, scene.look,
                  0.0 if a.no_bloom else scene.bloom)
    if a.save_hdr:
        save_pfm(hdr, out.replace(".png", ".pfm"))
    print(f"  {st['time']:6.1f}s   {st['rate']/1e6:.2f} Msamples/s"
          f"   spp {st['spp_min']}-{st['spp_max']} (mean {st['spp_mean']:.0f})")
    print(f"  {out}  {size/1024:.1f} KiB")


if __name__ == "__main__":
    main()

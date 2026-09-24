#!/usr/bin/env python3
"""Sobol vs PRNG and adaptive vs uniform, RMSE against an 8192 spp reference."""
import sys, os, time
import numpy as np, cupy as cp
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import render as R, scenes as SC

W, H = 640, 360
scene = SC.hero

def tonemapped(hdr):
    return cp.asnumpy(R.agx(cp.asarray(hdr), "none")).astype(np.float64)

def rmse(a, b):
    return float(np.sqrt(((a - b) ** 2).mean()) * 255.0)

print("building reference (hero, 8192 spp uniform, Sobol)...")
t0 = time.perf_counter()
ref_hdr, st = R.render(scene(), W, H, max_spp=8192, min_spp=8192, chunk=64,
                       adaptive=False, depth=24, quiet=True)
ref = tonemapped(ref_hdr)
print(f"  {time.perf_counter()-t0:.1f}s at {st['rate']/1e6:.1f} Msamples/s\n")

print(f"{'config':34} {'time':>7} {'mean spp':>9} {'RMSE/255':>9}")
rows = []
for label, kw in [
    ("random PRNG, uniform, 256 spp",   dict(ld_groups=0, adaptive=False, max_spp=256, min_spp=256)),
    ("Sobol+Owen, uniform, 256 spp",    dict(ld_groups=8, adaptive=False, max_spp=256, min_spp=256)),
    ("random PRNG, uniform, 1024 spp",  dict(ld_groups=0, adaptive=False, max_spp=1024, min_spp=1024)),
    ("Sobol+Owen, uniform, 1024 spp",   dict(ld_groups=8, adaptive=False, max_spp=1024, min_spp=1024)),
    ("Sobol+Owen, adaptive, <=4096",    dict(ld_groups=8, adaptive=True, max_spp=4096, min_spp=128,
                                             threshold=0.010)),
]:
    hdr, st = R.render(scene(), W, H, chunk=32, depth=24, quiet=True, **kw)
    e = rmse(tonemapped(hdr), ref)
    rows.append((label, st["time"], st["spp_mean"], e))
    print(f"{label:34} {st['time']:6.2f}s {st['spp_mean']:9.0f} {e:9.3f}")

print()
r = dict((x[0], x) for x in rows)
a = r["random PRNG, uniform, 256 spp"]; b = r["Sobol+Owen, uniform, 256 spp"]
print(f"  Sobol vs random @256 spp : {a[3]/b[3]:.2f}x lower RMSE "
      f"(equivalent to {(a[3]/b[3])**2:.2f}x more samples)")
a = r["random PRNG, uniform, 1024 spp"]; b = r["Sobol+Owen, uniform, 1024 spp"]
print(f"  Sobol vs random @1024 spp: {a[3]/b[3]:.2f}x lower RMSE "
      f"(equivalent to {(a[3]/b[3])**2:.2f}x more samples)")
u = r["Sobol+Owen, uniform, 1024 spp"]; ad = r["Sobol+Owen, adaptive, <=4096"]
print(f"  adaptive: RMSE {ad[3]:.3f} in {ad[1]:.1f}s vs uniform {u[3]:.3f} in {u[1]:.1f}s")

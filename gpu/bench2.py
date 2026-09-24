#!/usr/bin/env python3
"""Same as bench.py but at equal render time instead of equal spp."""
import sys, os, time
import numpy as np, cupy as cp
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import render as R, scenes as SC

W, H = 640, 360
def tm(h): return cp.asnumpy(R.agx(cp.asarray(h), "none")).astype(np.float64)
def rmse(a, b): return float(np.sqrt(((a-b)**2).mean())*255.0)

ref_hdr, _ = R.render(SC.hero(), W, H, max_spp=8192, min_spp=8192, chunk=64,
                      adaptive=False, depth=24, quiet=True)
ref = tm(ref_hdr)

for budget in (1.0, 4.0):
    print(f"\nequal time budget = {budget:.0f}s   (lower RMSE is better)")
    print(f"  {'config':32} {'time':>6} {'mean spp':>9} {'RMSE/255':>9}")
    for label, kw in [
        ("PRNG, uniform",        dict(ld_groups=0, adaptive=False)),
        ("Sobol 4 groups, uniform", dict(ld_groups=4, adaptive=False)),
        ("Sobol 8 groups, uniform", dict(ld_groups=8, adaptive=False)),
        ("PRNG, adaptive",       dict(ld_groups=0, adaptive=True, min_spp=128, threshold=0.008)),
        ("Sobol 4 groups, adaptive", dict(ld_groups=4, adaptive=True, min_spp=128, threshold=0.008)),
    ]:
        hdr, st = R.render(SC.hero(), W, H, max_spp=1 << 20, chunk=32, depth=24,
                           max_time=budget, quiet=True, **kw)
        print(f"  {label:32} {st['time']:5.2f}s {st['spp_mean']:9.0f} {rmse(tm(hdr), ref):9.3f}")

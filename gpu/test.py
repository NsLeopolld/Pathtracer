#!/usr/bin/env python3
"""Correctness tests: furnace, MIS consistency, colour round-trip."""
import sys, os
import numpy as np, cupy as cp
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import render as R, scenes as SC

FAIL = []

def check(name, got, want, tol, note=""):
    ok = abs(got - want) <= tol
    print(f"  [{'PASS' if ok else 'FAIL'}] {name:34} {got:9.5f}  (want {want} +-{tol}) {note}")
    if not ok: FAIL.append(name)

print("furnace test (albedo 1, sky 1, expect 1.0)")
for kind, tol, note in [("diffuse", 0.004, ""), ("glass", 0.004, ""),
                        ("dispersive", 0.006, "hero-wavelength reweighting"),
                        ("film", 0.004, "thin-film R+T=1"),
                        ("conductor", 0.06, "GGX single-scatter loss expected"),
                        ("plastic", 0.06, "uncoupled coat loss expected")]:
    hdr, st = R.render(SC.REGISTRY[f"furnace-{kind}"](), 192, 192, max_spp=512,
                       min_spp=512, chunk=32, adaptive=False, depth=32, quiet=True)
    v = cp.asnumpy(hdr).mean()
    check(f"furnace {kind}", float(v), 1.0, tol, note)

print("\nMIS vs NEE-only vs BSDF-only")
means = {}
for mode, label in [(0, "MIS"), (1, "BSDF-sampling only"), (2, "NEE only")]:
    hdr, st = R.render(SC.mistest(), 256, 144, max_spp=3072, min_spp=3072,
                       chunk=64, adaptive=False, depth=16, mis_mode=mode, quiet=True)
    a = cp.asnumpy(hdr)
    means[label] = a.mean()
    print(f"    {label:22} mean={a.mean():.5f}  noise={np.abs(np.diff(a,axis=1)).mean():.5f}")
ref = means["MIS"]
for k, v in means.items():
    if k != "MIS":
        check(f"{k} vs MIS", v/ref, 1.0, 0.02)

print("\nemitter colour round-trip")
s = SC.Scene("rt")
s.cam = dict(frm=(0,0,3), at=(0,0,0), vfov=30.0, aperture=0.0, focus=None)
s.sky = ((0,0,0),(0,0,0))
target = (0.82, 0.31, 0.11)
s.sphere((0,0,0), 1.2, s.mat(SC.EMISSIVE, target))
hdr, _ = R.render(s, 128, 128, max_spp=64, min_spp=64, chunk=16, adaptive=False, quiet=True)
a = cp.asnumpy(hdr)
centre = a[56:72, 56:72].reshape(-1,3).mean(0)
for i, ch in enumerate("rgb"):
    check(f"emitter {ch}", float(centre[i]), target[i], 0.005)

print("\n" + ("ALL TESTS PASSED" if not FAIL else f"FAILURES: {FAIL}"))
sys.exit(1 if FAIL else 0)

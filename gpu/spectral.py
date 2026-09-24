#!/usr/bin/env python3
"""
Spectral tables for the GPU path tracer.

Scenes are authored in RGB, so colours need converting to spectra. Uses
the Mallett & Yuksel (2019) basis: three smooth spectra B_r, B_g, B_b with

    B_r(l) + B_g(l) + B_b(l) = 1     for every wavelength
    0 <= B_k(l) <= 1                 for every wavelength
    <B_k, response_j> = delta_kj     exact RGB round-trip

The first two keep reflectances <= 1 at every wavelength, the third means
RGB -> spectrum -> RGB gives back the same colour.

Solved as constrained least squares minimising roughness. CMFs are the
Wyman-Sloan-Shirley (2013) fits, illuminant is a 6504 K blackbody (close
enough to D65, no table needed).
"""

import numpy as np
from scipy.optimize import minimize

LAMBDA_MIN, LAMBDA_MAX, STEP = 380.0, 730.0, 5.0
LAMBDAS = np.arange(LAMBDA_MIN, LAMBDA_MAX + STEP * 0.5, STEP)
NBINS = len(LAMBDAS)

# sRGB / Rec.709 primaries, D65 white.
XYZ_TO_RGB = np.array([
    [ 3.2404542, -1.5371385, -0.4985314],
    [-0.9692660,  1.8760108,  0.0415560],
    [ 0.0556434, -0.2040259,  1.0572252],
])


def _lobe(lam, mu, s1, s2):
    """Asymmetric Gaussian: different falloff either side of the peak."""
    s = np.where(lam < mu, s1, s2)
    return np.exp(-0.5 * ((lam - mu) / s) ** 2)


def cie_xyz(lam):
    """Wyman, Sloan & Shirley 2013, 'Simple Analytic Approximations to the
    CIE XYZ Colour Matching Functions'. Multi-lobe fits, ~1% accurate."""
    x = (1.056 * _lobe(lam, 599.8, 37.9, 31.0)
         + 0.362 * _lobe(lam, 442.0, 16.0, 26.7)
         - 0.065 * _lobe(lam, 501.1, 20.4, 26.2))
    y = (0.821 * _lobe(lam, 568.8, 46.9, 40.5)
         + 0.286 * _lobe(lam, 530.9, 16.3, 31.1))
    z = (1.217 * _lobe(lam, 437.0, 11.8, 36.0)
         + 0.681 * _lobe(lam, 459.0, 26.0, 13.8))
    return np.stack([x, y, z])


def planckian(lam_nm, temp_k=6504.0):
    """Planck's law. 6504 K stands in for D65 and needs no table."""
    lam = lam_nm * 1e-9
    c1, c2 = 3.7417718e-16, 1.4387769e-2
    v = c1 / (lam ** 5 * (np.exp(c2 / (lam * temp_k)) - 1.0))
    return v / v.max()


def build():
    lam = LAMBDAS
    illum = planckian(lam)
    resp = XYZ_TO_RGB @ cie_xyz(lam)          # sRGB camera response, 3 x N

    # normalize so the illuminant maps to (1,1,1), otherwise sum-to-one
    # and the round-trip constraint conflict
    norm = (resp * illum).sum(1) * STEP
    resp = resp / norm[:, None]
    C = resp * illum * STEP                    # 3 x N: <spectrum, C_j> -> RGB_j

    # --- constrained least squares -------------------------------------
    # Minimise roughness (squared first differences) subject to the three
    # conditions in the docstring.
    def unpack(v):
        return v.reshape(3, NBINS)

    def objective(v):
        B = unpack(v)
        d = np.diff(B, axis=1)
        return float((d * d).sum())

    def objective_grad(v):
        B = unpack(v)
        g = np.zeros_like(B)
        d = np.diff(B, axis=1)
        g[:, :-1] -= 2 * d
        g[:, 1:] += 2 * d
        return g.ravel()

    cons = [
        # sum to one at every wavelength
        {"type": "eq",
         "fun": lambda v: unpack(v).sum(0) - 1.0,
         "jac": lambda v: np.tile(np.eye(NBINS), (1, 3))},
        # basis k -> unit RGB in channel k only
        {"type": "eq",
         "fun": lambda v: (unpack(v) @ C.T - np.eye(3)).ravel(),
         "jac": lambda v: np.kron(np.eye(3), C).reshape(9, 3 * NBINS)},
    ]

    x0 = np.full(3 * NBINS, 1.0 / 3.0)
    res = minimize(objective, x0, jac=objective_grad, constraints=cons,
                   bounds=[(0.0, 1.0)] * (3 * NBINS),
                   method="SLSQP", options={"maxiter": 800, "ftol": 1e-12})
    B = unpack(res.x)

    # --- checks ---------------------------------------------------------
    rt = B @ C.T
    print(f"  solver: {res.message} ({res.nit} iters)")
    print(f"  round-trip error   max |B_k.C_j - I| = {np.abs(rt - np.eye(3)).max():.2e}")
    print(f"  sum-to-one error   max |sum_k B_k - 1| = {np.abs(B.sum(0) - 1).max():.2e}")
    print(f"  bounds             min {B.min():+.4f}  max {B.max():+.4f}")

    rng = np.random.default_rng(0)
    test = rng.random((4096, 3))
    back = (test @ B) @ C.T
    print(f"  random RGB round-trip  max err = {np.abs(back - test).max():.2e}")
    return dict(lambdas=lam, basis=B, illum=illum, resp=resp)


if __name__ == "__main__":
    print("building spectral basis...")
    t = build()
    np.savez("gpu/spectral_tables.npz", **t)
    print(f"  wrote gpu/spectral_tables.npz  ({NBINS} bins, "
          f"{LAMBDA_MIN:.0f}-{LAMBDA_MAX:.0f} nm)")

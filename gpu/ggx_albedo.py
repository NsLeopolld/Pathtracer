#!/usr/bin/env python3
"""
Directional albedo of single-scattering GGX, E(mu, alpha), by Monte Carlo.

Single-scatter GGX loses the energy from multiple bounces, so rough metals
come out dark (11% at alpha=0.3 in the furnace test). E is what single
scattering keeps, with F=1. Used for Turquin (2019) compensation:
    1 + F0 * (1 - E(mu_o)) / E(mu_o)

With VNDF sampling the weight is G2/G1, so E is just its mean.
"""
import numpy as np, cupy as cp

NMU, NA, NSAMP = 32, 32, 1 << 18


def smith_lambda(wz, a):
    z2 = cp.clip(wz*wz, 1e-9, 1.0)
    t2 = (1.0 - z2) / z2
    return 0.5 * (cp.sqrt(1.0 + a*a*t2) - 1.0)


def build():
    E = np.zeros((NA, NMU), np.float32)
    rng = cp.random.RandomState(7)
    for j in range(NA):
        alpha = max(((j + 0.5) / NA) ** 2, 1e-4)
        for i in range(NMU):
            mu = (i + 0.5) / NMU
            wo = cp.array([np.sqrt(max(0.0, 1 - mu*mu)), 0.0, mu], cp.float32)
            u1 = rng.random_sample(NSAMP, dtype=cp.float32)
            u2 = rng.random_sample(NSAMP, dtype=cp.float32)

            # VNDF sampling (Heitz 2018), vectorised.
            Vh = cp.stack([alpha*wo[0], alpha*wo[1], wo[2]])
            Vh = Vh / cp.linalg.norm(Vh)
            lensq = Vh[0]**2 + Vh[1]**2
            if float(lensq) > 0:
                T1 = cp.stack([-Vh[1], Vh[0], cp.zeros(())]) / cp.sqrt(lensq)
            else:
                T1 = cp.array([1.0, 0.0, 0.0], cp.float32)
            T2 = cp.cross(Vh, T1)
            r = cp.sqrt(u1); phi = 2*np.pi*u2
            t1 = r*cp.cos(phi); t2 = r*cp.sin(phi)
            s = 0.5*(1.0 + Vh[2])
            t2 = (1.0 - s)*cp.sqrt(cp.maximum(0.0, 1.0 - t1*t1)) + s*t2
            nz = cp.sqrt(cp.maximum(0.0, 1.0 - t1*t1 - t2*t2))
            Nh = (T1[:, None]*t1[None, :] + T2[:, None]*t2[None, :]
                  + Vh[:, None]*nz[None, :])
            m = cp.stack([alpha*Nh[0], alpha*Nh[1], cp.maximum(Nh[2], 1e-6)])
            m = m / cp.linalg.norm(m, axis=0, keepdims=True)

            wom = (wo[:, None]*m).sum(0)
            wi = 2.0*wom[None, :]*m - wo[:, None]
            valid = wi[2] > 0
            lo = smith_lambda(wo[2], alpha)
            li = smith_lambda(wi[2], alpha)
            weight = cp.where(valid, (1.0 + lo) / (1.0 + lo + li), 0.0)
            E[j, i] = float(weight.mean())
    return E


if __name__ == "__main__":
    print("integrating GGX directional albedo...")
    E = build()
    # sanity checks: smooth should be ~1
    print(f"  E(alpha->0)              = {E[0].mean():.4f}   (want ~1.000)")
    print(f"  E(alpha=1.0, mu=1.0)     = {E[-1, -1]:.4f}")
    print(f"  E(alpha=1.0, mu->0)      = {E[-1, 0]:.4f}   (grazing keeps the most)")
    print(f"  range                    = [{E.min():.4f}, {E.max():.4f}]")
    assert E.min() > 0.2 and E.max() <= 1.001, "table out of range"
    assert E[0].mean() > 0.995, "smooth GGX must not lose energy"
    np.save("gpu/ggx_albedo.npy", E)
    print(f"  wrote gpu/ggx_albedo.npy  {E.shape}")

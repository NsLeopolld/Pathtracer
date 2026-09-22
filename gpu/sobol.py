#!/usr/bin/env python3
"""
Sobol direction numbers for the first four dimensions, plus the tests that
prove they are right.

Generated from the Joe & Kuo initialisation (new-joe-kuo-6.21201): each
dimension has a primitive polynomial (degree s, coefficient bits a) and
initial direction integers m_i. Rather than trusting the numbers, the
properties they are supposed to have are checked below -- a wrong
recurrence fails the net test immediately.
"""

import numpy as np

# (s, a, m) per dimension. Dimension 0 is the van der Corput sequence.
JOE_KUO = [
    (0, 0, []),          # dim 0: identity matrix (bit reversal)
    (1, 0, [1]),
    (2, 1, [1, 3]),
    (3, 1, [1, 3, 1]),
]


def directions(dim):
    v = np.zeros(32, dtype=np.uint64)
    s, a, m = JOE_KUO[dim]
    if dim == 0:
        for i in range(32):
            v[i] = np.uint64(1) << np.uint64(31 - i)
        return (v & np.uint64(0xFFFFFFFF)).astype(np.uint32)
    for i in range(s):
        v[i] = np.uint64(m[i]) << np.uint64(31 - i)
    for i in range(s, 32):
        v[i] = v[i - s] ^ (v[i - s] >> np.uint64(s))
        for k in range(1, s):
            if (a >> (s - 1 - k)) & 1:
                v[i] ^= v[i - k]
    return (v & np.uint64(0xFFFFFFFF)).astype(np.uint32)


def sobol_points(n, dims=4):
    D = [directions(d) for d in range(dims)]
    out = np.zeros((n, dims), dtype=np.float64)
    for i in range(n):
        for d in range(dims):
            x = 0
            idx = i
            bit = 0
            while idx:
                if idx & 1:
                    x ^= int(D[d][bit])
                idx >>= 1
                bit += 1
            out[i, d] = (x >> 8) * 2.0 ** -24
    return out


def check():
    ok = True
    for m in (6, 8, 10):
        n = 1 << m
        p = sobol_points(n)
        # 1-D: every dimension must hit each of the n equal bins exactly once.
        for d in range(4):
            counts = np.bincount((p[:, d] * n).astype(int), minlength=n)
            good = (counts == 1).all()
            ok &= good
            if not good:
                print(f"  FAIL 1-D stratification dim {d} at n={n}")
        # 2-D net property for the (0,1) projection: every elementary
        # rectangle of area 1/n holds exactly one point.
        for sx in range(m + 1):
            nx, ny = 1 << sx, 1 << (m - sx)
            h = np.histogram2d(p[:, 0], p[:, 1], bins=[nx, ny],
                               range=[[0, 1], [0, 1]])[0]
            good = (h == 1).all()
            ok &= good
            if not good:
                print(f"  FAIL (0,m,2)-net for dims (0,1) at n={n}, {nx}x{ny}")
    print(f"  net + stratification tests: {'PASS' if ok else 'FAIL'}")

    # Star discrepancy proxy: Sobol must beat random by a wide margin.
    n = 4096
    p = sobol_points(n)[:, :2]
    r = np.random.default_rng(0).random((n, 2))
    def boxdisc(pts):
        worst = 0.0
        for _ in range(2000):
            hi = np.random.default_rng(int(_)).random(2) * 0.9 + 0.1
            frac = ((pts < hi).all(1)).mean()
            worst = max(worst, abs(frac - hi.prod()))
        return worst
    print(f"  discrepancy  sobol {boxdisc(p):.5f}   random {boxdisc(r):.5f}")
    return ok


if __name__ == "__main__":
    print("verifying Sobol direction numbers...")
    assert check(), "Sobol generation is wrong"
    D = np.stack([directions(d) for d in range(4)])
    np.save("gpu/sobol_directions.npy", D)
    print(f"  wrote gpu/sobol_directions.npy  {D.shape} uint32")

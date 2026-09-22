# Path tracer

A physically based renderer, written three times: once in NumPy, once in C,
once in CUDA. Each version is a rewrite of the one before it, and the
benchmark between them is fair because they trace byte-identical geometry.

All three write their own PNG files — chunk framing, CRC-32 and adaptive
scanline filtering, assembled from the bytes up. No image library is used
anywhere, in any of them.

![Neon night, rendered with the CUDA version](render_gpu_neon.png)

The soap bubble's iridescence above is not a texture or a gradient. It is
thin-film interference, computed from the film's thickness and the
wavelength of the light hitting it. The RGB renderers in this repo cannot
express it at all; the spectral one gets it for free.

---

## Quick start

### C (start here — fastest to run, no setup)

```sh
gcc -O3 -march=native -ffast-math -funroll-loops -fopenmp -Wall -Wextra \
    pathtracer.c -lz -lm -o pathtracer
./pathtracer --width 960 --spp 144 --depth 16 --out render.png
```

Other scenes are separate builds, selected at compile time:

```sh
python3 make_scenes.py                      # writes scene.h, scene_cornell.h, scene_neon.h
gcc -O3 -march=native -ffast-math -fopenmp \
    -DSCENE_FILE='"scene_cornell.h"' pathtracer.c -lz -lm -o pathtracer_cornell
```

Flags: `--width --height --spp --depth --out`. Set `OMP_NUM_THREADS` to control
threading, and `NO_NEE=1` to disable light sampling (used for the A/B in
"What didn't work").

### NumPy (the reference implementation)

```sh
python3 pathtracer.py --width 480 --spp 32 --out render.png
```

Slow — see the table below — but it is the definition the C version was
checked against.

### CUDA (spectral; needs an NVIDIA GPU)

```sh
.venv/bin/python gpu/render.py --scene neon  --width 1920 --spp 4096 --max-time 180
.venv/bin/python gpu/render.py --scene hero  --width 1920 --spp 4096 --max-time 180
.venv/bin/python gpu/render.py --scene cornell --width 1200 --height 1200 --spp 8192
```

See [Setup](#setup-for-the-cuda-version) for the environment. Scenes:
`hero`, `cornell`, `neon`, plus the test scenes `mistest` and `furnace-*`.

---

## Results

Same scene and same geometry throughout. Measured on an i5-10300H
(4 cores / 8 threads, AVX2) and a GTX 1650 Mobile.

| Renderer | Throughput | Relative | 960×540 @ 144 spp |
|---|---:|---:|---:|
| NumPy, 8 processes | 0.29 Msamples/s | 1× | 257.9 s |
| C, AVX2, 8 threads | 7.74 Msamples/s | **26.8×** | 9.6 s |
| CUDA, GTX 1650 | 112–139 Msamples/s | **~390×** | — |

Per core the C version is roughly **43×** the NumPy one. That gap is not
only the language: NumPy forces a worse algorithm shape. Per-ray Python is
hopeless, so the whole frame has to advance one bounce at a time, streaming
every live ray through memory at every bounce. C carries one path to
completion in registers and touches nothing outside L1.

CUDA is ~14.5× the C version *while doing considerably more work per
sample* — four wavelengths per path, MIS, microfacet metals.

### Thread scaling (C)

| Threads | Time | Speedup | Efficiency |
|---:|---:|---:|---:|
| 1 | 10.59 s | 1.00× | 100% |
| 2 | 5.39 s | 1.96× | 98% |
| 4 | 2.78 s | 3.81× | 95% |
| 8 | 2.17 s | 4.88× | 61% |

95% efficiency on the 4 physical cores. The drop at 8 is hyperthreading:
the second thread on a core shares execution units this workload already
saturates.

---

## How they work

**NumPy** (`pathtracer.py`) — vectorised across rays. Cone-sampled
next-event estimation, Schlick dielectrics, glossy metals, thin-lens depth
of field, Russian roulette, ACES tonemap.

**C** (`pathtracer.c`) — same algorithm, one path at a time. The sphere
intersection loop is written branchless so gcc vectorises it into AVX2
(`vfmadd132ps`, `vblendvps`); it tests 8 spheres per instruction. Lights are
picked proportional to power / distance². Geometry comes from `scene.h`,
generated out of the NumPy scene so the two renderers agree.

**CUDA** (`gpu/`) — a spectral path tracer. Four wavelengths per path
(hero wavelength sampling), which is what makes dispersion and thin-film
interference possible at all. Also: MIS between light and BSDF sampling
with the power heuristic, GGX metals with Turquin energy compensation,
coated-diffuse surfaces, adaptive sampling, AgX tonemapping, lens glare and
dithered output. Compiled at runtime by NVRTC.

RGB scene colours are turned into spectra with a basis solved by
constrained optimisation (Mallett & Yuksel 2019): three smooth curves that
sum to 1 at every wavelength, stay within [0,1], and round-trip RGB
exactly. Summing to one is what makes reflectance upsampling
energy-conserving — a surface can never reflect more than it receives.

---

## Correctness

The GPU renderer is tested, not assumed:

```sh
.venv/bin/python gpu/test.py
```

**White furnace test.** Put an albedo-1 object in a sky of radiance 1. It
must vanish — every pixel exactly 1.0. Anything else means energy is being
lost or invented.

| Material | Result |
|---|---:|
| Diffuse | 0.9999 |
| Glass | 0.9999 |
| Dispersive glass | 0.9999 |
| Thin film | 0.9999 |
| Rough conductor | 0.9999 |
| Coated diffuse | 0.9790 |

Dispersive glass passing is the meaningful one: it confirms the
hero-wavelength reweighting is unbiased. Rough conductor started at 0.888 —
the standard single-scattering microfacet loss — and the energy
compensation in `gpu/ggx_albedo.py` fixed it.

**Estimator agreement.** MIS, NEE-only and BSDF-only are three different
ways to compute the same integral. They agree to **0.02%**, with MIS the
least noisy (1.23× under BSDF-only).

**Colour round-trip.** Emitter RGB survives RGB → spectrum → RGB to within
0.0005.

The supporting tables verify themselves too: `gpu/sobol.py` checks the
(0,m,2)-net property rather than trusting the direction numbers, and
`gpu/spectral.py` reports round-trip error (2e-16) and basis bounds.

---

## What didn't work

Both of these are measured, and both are kept in the repo because the
result is scene-dependent rather than universal.

**Owen-scrambled Sobol sampling** is 1.1× better per sample but roughly 50%
slower, so at equal *time* plain PRNG wins. Error here is dominated by
high-dimensional caustic paths, where low-discrepancy sampling gives
nothing. Default off; `--ld-groups 8` turns it on.

**Adaptive sampling** was initially 55% *slower* than uniform, because as
pixels converge the launches shrink and become latency-bound. Sizing each
launch to a constant work quantum brought it to roughly break-even (~2%
ahead). On by default; `--no-adaptive` disables it.

Run them yourself: `gpu/bench.py` (equal sample count) and `gpu/bench2.py`
(equal time — the one that decides anything).

---

## Setup for the CUDA version

Needs an NVIDIA GPU and driver. Nothing is installed system-wide; the whole
toolchain comes from pip, and `rm -rf .venv` reverses it.

```sh
python3 -m venv --without-pip .venv        # Debian: ensurepip is often absent
python3 -m pip --python .venv/bin/python install \
    cupy-cuda12x numpy scipy pillow \
    "nvidia-cuda-nvrtc-cu12==12.4.*" "nvidia-cuda-runtime-cu12==12.4.*" \
    "nvidia-curand-cu12==10.3.5.*" "nvidia-cufft-cu12==11.2.*" \
    "nvidia-cublas-cu12==12.4.*"
```

**Pin NVRTC to your driver's CUDA version** (`nvidia-smi` reports it, top
right). A newer NVRTC emits PTX an older driver cannot load.

The lookup tables are committed, but regenerate with:

```sh
.venv/bin/python gpu/spectral.py     # RGB -> spectrum basis
.venv/bin/python gpu/sobol.py        # Sobol direction numbers
.venv/bin/python gpu/ggx_albedo.py   # GGX directional albedo
```

`.venv` is ~1.6 GB, almost all of it CUDA libraries.

---

## Layout

```
pathtracer.py        NumPy renderer; also defines the shared scene
pathtracer.c         C renderer
make_scenes.py       writes scene.h, scene_cornell.h, scene_neon.h
gpu/kernel.cu        spectral path tracer (CUDA)
gpu/render.py        host: scene upload, adaptive loop, post, PNG
gpu/scenes.py        scene definitions, including the test scenes
gpu/test.py          furnace / MIS / colour tests
gpu/bench.py         equal-sample-count benchmark
gpu/bench2.py        equal-time benchmark
gpu/spectral.py      solves the RGB -> spectrum basis
gpu/sobol.py         Sobol direction numbers, with net-property tests
gpu/ggx_albedo.py    GGX directional albedo, for energy compensation
render*.png          output
```

Generated headers and lookup tables are committed so a fresh clone builds
and runs; every one of them regenerates byte-for-byte identical.

---

## Known limitations

- **Spheres and planes only.** No triangles, so no meshes. With scenes this
  small a linear SIMD scan beats a BVH, which is why there isn't one.
- **Caustic noise.** Light paths through glass cannot be reached by shadow
  rays, so caustics converge slowly. The honest fix is samples; the
  dishonest one is `--clamp`, which is biased and off by default.
- **Coated diffuse loses 2.1%** of its energy — the coat coupling is
  approximate. It errs toward losing light rather than creating it.
- **No denoiser.** Every image here is converged, not filtered.
- **Spectral upsampling is not unique.** Infinitely many spectra share an
  RGB value; this basis picks the smoothest energy-conserving one, which is
  a defensible choice rather than a recovery of ground truth.

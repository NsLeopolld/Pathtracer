# Path tracer

A path tracer I wrote three times: NumPy first, then C, then CUDA. The NumPy
and C versions trace the same geometry (scene.h is generated from the Python
scene), so the benchmarks between them compare like for like.

None of them use an image library. PNG writing (chunks, CRC, scanline
filtering) is done by hand, with zlib only for deflate.

![Neon night, rendered with the CUDA version](render_gpu_neon.png)

The colours on the bubble come from actual thin-film interference, computed
per wavelength from the film thickness. That's the main reason the CUDA
version is spectral; the RGB versions can't do it.

---

## Quick start

### C (fastest to get running, no setup)

```sh
gcc -O3 -march=native -ffast-math -funroll-loops -fopenmp -Wall -Wextra \
    pathtracer.c -lz -lm -o pathtracer
./pathtracer --width 960 --spp 144 --depth 16 --out render.png
```

Other scenes are picked at compile time:

```sh
python3 make_scenes.py                      # writes scene.h, scene_cornell.h, scene_neon.h
gcc -O3 -march=native -ffast-math -fopenmp \
    -DSCENE_FILE='"scene_cornell.h"' pathtracer.c -lz -lm -o pathtracer_cornell
```

Flags: `--width --height --spp --depth --out`. Thread count comes from
`OMP_NUM_THREADS`.

### NumPy

```sh
python3 pathtracer.py --width 480 --spp 32 --out render.png
```

Slow (see below), but it's the reference I checked the C version against.
`NO_NEE=1` turns off light sampling in this version.

### CUDA (spectral, needs an NVIDIA GPU)

```sh
.venv/bin/python gpu/render.py --scene neon  --width 1920 --spp 4096 --max-time 180
.venv/bin/python gpu/render.py --scene hero  --width 1920 --spp 4096 --max-time 180
.venv/bin/python gpu/render.py --scene cornell --width 1200 --height 1200 --spp 8192
```

Environment setup is [below](#setup-for-the-cuda-version). Scenes: `hero`,
`cornell`, `neon`, and the test scenes `mistest` and `furnace-*`.

---

## Results

Same scene for all three. i5-10300H (4C/8T, AVX2) and a GTX 1650 Mobile.

| Renderer | Throughput | Relative | 960×540 @ 144 spp |
|---|---:|---:|---:|
| NumPy, 8 processes | 0.29 Msamples/s | 1× | 257.9 s |
| C, AVX2, 8 threads | 7.74 Msamples/s | 26.8× | 9.6 s |
| CUDA, GTX 1650 | 112–139 Msamples/s | ~390× | — |

Per core, C is about 43× faster than NumPy. Part of that is the language,
but mostly it's the structure: to get any speed out of NumPy you have to
push the whole frame through one bounce at a time, which means streaming
every live ray through memory on every bounce. The C version just traces
one path at a time and stays in cache.

CUDA is ~14.5× the C version, and it does more work per sample (4
wavelengths per path, MIS, microfacet metals), so the numbers aren't
directly comparable.

### Thread scaling (C)

| Threads | Time | Speedup | Efficiency |
|---:|---:|---:|---:|
| 1 | 10.59 s | 1.00× | 100% |
| 2 | 5.39 s | 1.96× | 98% |
| 4 | 2.78 s | 3.81× | 95% |
| 8 | 2.17 s | 4.88× | 61% |

Scales well up to the 4 physical cores. Going to 8 threads only helps a
bit since hyperthreads share the same execution units.

---

## How they work

**NumPy** (`pathtracer.py`): vectorised over rays. NEE with cone sampling
toward sphere lights, Schlick dielectrics, fuzzy metals, thin lens DOF,
Russian roulette, ACES tonemap.

**C** (`pathtracer.c`): same algorithm, one path at a time. The sphere
intersection loop is branchless so gcc vectorises it with AVX2
(`vfmadd132ps`/`vblendvps`, 8 spheres per iteration). Lights are
picked proportional to power / distance². Geometry comes from `scene.h`,
which is generated from the NumPy scene.

**CUDA** (`gpu/`): spectral. 4 wavelengths per path (hero wavelength
sampling), which is what dispersion and thin film need. Also has MIS
between light and BSDF sampling (power heuristic), GGX metals with Turquin
energy compensation, coated diffuse, adaptive sampling, AgX tonemapping,
glare and dithering. The kernel is compiled at runtime with NVRTC.

RGB colours are converted to spectra with a basis from Mallett & Yuksel
2019, solved with constrained optimisation: three smooth curves that sum to
1 at every wavelength, stay in [0,1], and round-trip RGB exactly. The
sum-to-one part is what keeps reflectances from going above 1 at any
wavelength.

---

## Correctness

```sh
.venv/bin/python gpu/test.py
```

**White furnace.** An albedo-1 object in a uniform sky of radiance 1 should
disappear, i.e. every pixel reads 1.0. If it doesn't, energy is being lost
or added somewhere.

| Material | Result |
|---|---:|
| Diffuse | 0.9999 |
| Glass | 0.9999 |
| Dispersive glass | 0.9999 |
| Thin film | 0.9999 |
| Rough conductor | 0.9999 |
| Coated diffuse | 0.9790 |

Dispersive glass passing means the hero wavelength reweighting is right.
Rough conductor was at 0.888 before energy compensation
(`gpu/ggx_albedo.py`), which is the usual single-scattering GGX loss.

**Estimator agreement.** MIS, NEE only and BSDF only should all converge to
the same image. They agree within 0.02%, and MIS has the least noise (1.23×
lower than BSDF only).

**Colour round-trip.** Emitter RGB → spectrum → RGB is off by at most 0.0005.

The table generators check themselves as well: `gpu/sobol.py` tests the
(0,m,2)-net property, and `gpu/spectral.py` prints the round-trip error
(2e-16) and basis bounds.

---

## What didn't work

Both of these are still in the repo since whether they help depends on the
scene.

**Owen-scrambled Sobol.** About 1.1× better per sample but ~50% slower, so
at equal render time plain PRNG wins. Most of the error in these scenes is
from caustic paths, where low-discrepancy sampling doesn't help much. Off
by default, `--ld-groups 8` enables it.

**Adaptive sampling.** First version was 55% slower than uniform, because
once most pixels converge the launches get tiny and latency dominates.
Sizing each launch to a fixed amount of work got it to roughly break-even
(~2% ahead). On by default, `--no-adaptive` disables it.

Benchmarks: `gpu/bench.py` (equal spp) and `gpu/bench2.py` (equal time,
which is the fairer comparison).

---

## Setup for the CUDA version

Needs an NVIDIA GPU and driver. Everything else comes from pip into a venv,
nothing system-wide. `rm -rf .venv` to undo.

```sh
python3 -m venv --without-pip .venv        # Debian often lacks ensurepip
python3 -m pip --python .venv/bin/python install \
    cupy-cuda12x numpy scipy pillow \
    "nvidia-cuda-nvrtc-cu12==12.4.*" "nvidia-cuda-runtime-cu12==12.4.*" \
    "nvidia-curand-cu12==10.3.5.*" "nvidia-cufft-cu12==11.2.*" \
    "nvidia-cublas-cu12==12.4.*"
```

Pin NVRTC to the CUDA version your driver supports (top right of
`nvidia-smi`). A newer NVRTC produces PTX that an older driver can't load.

The lookup tables are committed. To regenerate:

```sh
.venv/bin/python gpu/spectral.py     # RGB -> spectrum basis
.venv/bin/python gpu/sobol.py        # Sobol direction numbers
.venv/bin/python gpu/ggx_albedo.py   # GGX directional albedo
```

The venv ends up around 1.6 GB, nearly all CUDA libraries.

---

## Layout

```
pathtracer.py        NumPy renderer, also defines the default scene
pathtracer.c         C renderer
make_scenes.py       writes scene.h, scene_cornell.h, scene_neon.h
gpu/kernel.cu        spectral path tracer (CUDA)
gpu/render.py        host side: scene upload, adaptive loop, post, PNG
gpu/scenes.py        scene definitions, including test scenes
gpu/test.py          furnace / MIS / colour tests
gpu/bench.py         equal-spp benchmark
gpu/bench2.py        equal-time benchmark
gpu/spectral.py      solves the RGB -> spectrum basis
gpu/sobol.py         Sobol direction numbers + net-property tests
gpu/ggx_albedo.py    GGX directional albedo for energy compensation
render*.png          output
```

Generated headers and lookup tables are committed so a fresh clone builds
without running the generators first.

---

## Known limitations

- **Spheres and planes only.** No triangles or meshes. For scenes this
  small a linear SIMD loop is faster than a BVH, so there isn't one.
- **Caustics are noisy.** Shadow rays can't reach lights through glass, so
  caustics converge slowly. `--clamp` cuts the fireflies but adds bias, so
  it's off by default.
- **Coated diffuse loses 2.1%** of its energy since the coat/base coupling
  is approximate. At least it loses energy rather than creating it.
- **No denoiser.** All the renders here are just run to convergence.
- **Spectral upsampling isn't unique.** Lots of spectra map to the same
  RGB. This basis picks the smoothest energy-conserving one, which is a
  reasonable choice but not "the" spectrum.

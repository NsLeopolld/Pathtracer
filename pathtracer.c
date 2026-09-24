/*
 * CPU path tracer. Spheres, planes and boxes (rotatable around y). NEE
 * toward sphere lights, Schlick glass, fuzzy metal, thin lens DOF, Russian
 * roulette, ACES, own PNG writer.
 *
 * One path at a time per thread (OpenMP). Sphere data is SoA and the
 * intersection loop is written so gcc vectorizes it (AVX2 with
 * -march=native). Scene comes from scene.h, see make_scenes.py.
 *
 *   cc -O3 -march=native -ffast-math -fopenmp pathtracer.c -lz -lm -o pathtracer
 */

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <zlib.h>
#ifdef _OPENMP
#include <omp.h>
#endif

/* scene headers (written by make_scenes.py) fill in arrays of these */
typedef enum { LAMBERTIAN=0, METAL=1, DIELECTRIC=2, EMISSIVE=3 } Mat;
typedef struct { float cx, cy, cz, r; float ar, ag, ab; int mat; float param; } Sphere;
typedef struct { float nx, ny, nz, d; float ar, ag, ab; int mat; float param; } Plane; /* n.p = d */
typedef struct { float cx, cy, cz, sx, sy, sz, rot_y;   /* centre, full size, degrees */
                 float ar, ag, ab; int mat; float param; } Box;

#ifndef SCENE_FILE
#define SCENE_FILE "scene.h"
#endif
#include SCENE_FILE

#ifndef N_PLANES
#define N_PLANES 0
#endif
#ifndef N_BOXES
#define N_BOXES 0
#endif

/* Every object gets one id for material lookups: planes first, then
 * spheres, then boxes. With no planes a sphere's id is its index. */
#define N_OBJ     (N_PLANES + N_SPHERES + N_BOXES)
#define SPH_ID(i) (N_PLANES + (i))
#define BOX_ID(i) (N_PLANES + N_SPHERES + (i))
#define ARR(n)    ((n) > 0 ? (n) : 1)      /* avoid zero-length arrays */

/* defaults, scene headers can override any of these */
#ifndef CAM_FROM
#define CAM_FROM      8.6f, 2.15f, 6.2f
#endif
#ifndef CAM_AT
#define CAM_AT        0.0f, 0.92f, -0.2f
#endif
#ifndef CAM_VFOV
#define CAM_VFOV      27.0f
#endif
#ifndef CAM_APERTURE
#define CAM_APERTURE  0.10f
#endif
#ifndef SKY_HORIZON
#define SKY_HORIZON   0.52f, 0.42f, 0.38f
#endif
#ifndef SKY_ZENITH
#define SKY_ZENITH    0.10f, 0.16f, 0.30f
#endif
#ifndef FLOOR_CHECKER
#define FLOOR_CHECKER 1          /* checker on object 0 (first plane, else first sphere) */
#endif
#ifndef EXPOSURE
#define EXPOSURE      1.0f
#endif

#define PI 3.14159265358979323846f

/* Offset along the normal for new rays. Was 1e-4, but with the r=1000
 * wall spheres |oc|^2 - r^2 has ~0.06 of float error and grazing rays
 * re-hit their own surface (ring artifacts). 1e-3 fixes it. */
#define RAY_EPS 1e-3f

/* ---- vec3 ---- */

typedef struct { float x, y, z; } Vec3;

static inline Vec3 v3(float x, float y, float z)      { Vec3 v={x,y,z}; return v; }
static inline Vec3 vadd(Vec3 a, Vec3 b)  { return v3(a.x+b.x, a.y+b.y, a.z+b.z); }
static inline Vec3 vsub(Vec3 a, Vec3 b)  { return v3(a.x-b.x, a.y-b.y, a.z-b.z); }
static inline Vec3 vmul(Vec3 a, Vec3 b)  { return v3(a.x*b.x, a.y*b.y, a.z*b.z); }
static inline Vec3 vscl(Vec3 a, float s) { return v3(a.x*s, a.y*s, a.z*s); }
static inline Vec3 vneg(Vec3 a)          { return v3(-a.x, -a.y, -a.z); }
static inline float vdot(Vec3 a, Vec3 b) { return a.x*b.x + a.y*b.y + a.z*b.z; }
static inline float vlen2(Vec3 a)        { return vdot(a,a); }
static inline Vec3 vnorm(Vec3 a)         { return vscl(a, 1.0f/sqrtf(vdot(a,a))); }
static inline Vec3 vcross(Vec3 a, Vec3 b) {
    return v3(a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x);
}
static inline float vmaxc(Vec3 a) {
    float m = a.x > a.y ? a.x : a.y; return m > a.z ? m : a.z;
}

/* ---- rng: PCG32, seeded per pixel ---- */

typedef struct { uint64_t state, inc; } Rng;

static inline uint32_t pcg32(Rng *r) {
    uint64_t old = r->state;
    r->state = old * 6364136223846793005ULL + r->inc;
    uint32_t xs  = (uint32_t)(((old >> 18u) ^ old) >> 27u);
    uint32_t rot = (uint32_t)(old >> 59u);
    return (xs >> rot) | (xs << ((-rot) & 31u));
}

static inline void rng_seed(Rng *r, uint64_t seq, uint64_t seed) {
    r->state = 0u; r->inc = (seq << 1u) | 1u;
    pcg32(r); r->state += seed; pcg32(r);
}

/* top 24 bits -> [0,1) */
static inline float randf(Rng *r) { return (float)(pcg32(r) >> 8) * 0x1.0p-24f; }

static inline Vec3 random_unit(Rng *r) {
    /* rejection sampling, ~52% accept but cheaper than sin/cos */
    for (;;) {
        Vec3 p = v3(randf(r)*2.0f-1.0f, randf(r)*2.0f-1.0f, randf(r)*2.0f-1.0f);
        float l2 = vlen2(p);
        if (l2 > 1e-8f && l2 <= 1.0f) return vscl(p, 1.0f/sqrtf(l2));
    }
}

/* ---- scene (SoA for the intersect loop) ---- */

/* intersect() works on LANES spheres at a time, so the geometry arrays are
 * padded up to a multiple of that with spheres nothing can hit. */
#define LANES 8
#define N_PAD (((N_SPHERES) + LANES - 1) / LANES * LANES)

/* sphere geometry, indexed by sphere */
static float g_cx[ARR(N_PAD)] __attribute__((aligned(32)));
static float g_cy[ARR(N_PAD)] __attribute__((aligned(32)));
static float g_cz[ARR(N_PAD)] __attribute__((aligned(32)));
static float g_r2[ARR(N_PAD)] __attribute__((aligned(32)));
static float g_r[ARR(N_SPHERES)], g_inv_r[ARR(N_SPHERES)];
static int   g_lights[ARR(N_SPHERES)], g_nlights = 0;   /* emissive spheres */
static float g_power[ARR(N_SPHERES)];   /* luminance * r^2, rough light power */

/* plane and box geometry */
#if N_PLANES > 0
static Vec3  g_pn[N_PLANES];            /* unit normal */
static float g_pd[N_PLANES];
#endif
#if N_BOXES > 0
static Vec3  g_bc[N_BOXES], g_bh[N_BOXES];             /* centre, half size */
static float g_bcos[N_BOXES], g_bsin[N_BOXES];
#endif

/* materials, indexed by object id */
static float g_ar[ARR(N_OBJ)], g_ag[ARR(N_OBJ)], g_ab[ARR(N_OBJ)];
static int   g_mat[ARR(N_OBJ)];
static float g_param[ARR(N_OBJ)];

static void set_mat(int id, float r, float g, float b, int mat, float param) {
    g_ar[id] = r; g_ag[id] = g; g_ab[id] = b;
    g_mat[id] = mat; g_param[id] = param;
}

static void scene_init(void) {
#if N_SPHERES > 0
    for (int i = 0; i < N_SPHERES; i++) {
        const Sphere *s = &SCENE[i];
        g_cx[i] = s->cx; g_cy[i] = s->cy; g_cz[i] = s->cz;
        g_r[i]  = s->r;  g_r2[i] = s->r * s->r; g_inv_r[i] = 1.0f / s->r;
        set_mat(SPH_ID(i), s->ar, s->ag, s->ab, s->mat, s->param);
        g_power[i] = (0.2126f*s->ar + 0.7152f*s->ag + 0.0722f*s->ab) * s->r * s->r;
        if (s->mat == EMISSIVE) g_lights[g_nlights++] = i;
    }
#endif
    /* r^2 = -1e30 makes disc hugely negative, so padding never hits */
    for (int i = N_SPHERES; i < N_PAD; i++) {
        g_cx[i] = g_cy[i] = g_cz[i] = 0.0f;
        g_r2[i] = -1e30f;
    }
#if N_PLANES > 0
    for (int i = 0; i < N_PLANES; i++) {
        const Plane *pl = &PLANES[i];
        float len = sqrtf(pl->nx*pl->nx + pl->ny*pl->ny + pl->nz*pl->nz);
        g_pn[i] = v3(pl->nx/len, pl->ny/len, pl->nz/len);
        g_pd[i] = pl->d / len;
        set_mat(i, pl->ar, pl->ag, pl->ab, pl->mat, pl->param);
    }
#endif
#if N_BOXES > 0
    for (int i = 0; i < N_BOXES; i++) {
        const Box *b = &BOXES[i];
        g_bc[i] = v3(b->cx, b->cy, b->cz);
        g_bh[i] = v3(0.5f*b->sx, 0.5f*b->sy, 0.5f*b->sz);
        g_bcos[i] = cosf(b->rot_y * PI / 180.0f);
        g_bsin[i] = sinf(b->rot_y * PI / 180.0f);
        set_mat(BOX_ID(i), b->ar, b->ag, b->ab, b->mat, b->param);
    }
#endif
}

#if N_BOXES > 0
/* ---- boxes: slab test in the box's own (unrotated) frame ---- */

/* world -> box frame is a rotation by -rot_y around y */
static inline Vec3 box_to_local(int b, Vec3 v) {
    float c = g_bcos[b], s = g_bsin[b];
    return v3(c*v.x - s*v.z, v.y, s*v.x + c*v.z);
}

static inline float box_hit(int b, Vec3 o, Vec3 d, float tmin) {
    Vec3 ol = box_to_local(b, vsub(o, g_bc[b]));
    Vec3 dl = box_to_local(b, d);
    Vec3 h  = g_bh[b];
    float lo[3] = { -h.x - ol.x, -h.y - ol.y, -h.z - ol.z };
    float hi[3] = {  h.x - ol.x,  h.y - ol.y,  h.z - ol.z };
    float dd[3] = { dl.x, dl.y, dl.z };
    float tn = -1e30f, tf = 1e30f;
    for (int a = 0; a < 3; a++) {
        /* keep 1/d finite: -ffast-math assumes no infinities */
        float inv = 1.0f / (fabsf(dd[a]) > 1e-12f ? dd[a] : copysignf(1e-12f, dd[a]));
        float t1 = lo[a] * inv, t2 = hi[a] * inv;
        tn = fmaxf(tn, fminf(t1, t2));
        tf = fminf(tf, fmaxf(t1, t2));
    }
    if (tn > tf || tf <= tmin) return 1e30f;
    return tn > tmin ? tn : tf;          /* tf when the ray starts inside */
}

static inline Vec3 box_normal(int b, Vec3 p) {
    Vec3 q = box_to_local(b, vsub(p, g_bc[b]));
    Vec3 h = g_bh[b];
    /* the face we're on is the axis where |q| is closest to the half size */
    float ax = fabsf(q.x) / h.x, ay = fabsf(q.y) / h.y, az = fabsf(q.z) / h.z;
    Vec3 n = (ax >= ay && ax >= az) ? v3(copysignf(1.0f, q.x), 0.0f, 0.0f)
           : (ay >= az)             ? v3(0.0f, copysignf(1.0f, q.y), 0.0f)
           :                          v3(0.0f, 0.0f, copysignf(1.0f, q.z));
    float c = g_bcos[b], s = g_bsin[b];  /* back to world: rotate by +rot_y */
    return v3(c*n.x + s*n.z, n.y, -s*n.x + c*n.z);
}
#endif

/* Closest hit, returns an object id. gcc won't vectorize a min-with-index reduction, so keep a
 * separate best hit per lane and merge them at the end. The inner loop
 * has to stay branchless or it stops vectorizing. */
static inline int intersect(Vec3 o, Vec3 d, float tmin, float *out_t) {
    float best[LANES];
    int   id[LANES];
    for (int j = 0; j < LANES; j++) { best[j] = 1e30f; id[j] = -1; }

    for (int i = 0; i < N_PAD; i += LANES) {
        for (int j = 0; j < LANES; j++) {
            int k = i + j;
            float ox = o.x - g_cx[k], oy = o.y - g_cy[k], oz = o.z - g_cz[k];
            float hb = d.x*ox + d.y*oy + d.z*oz;      /* |d| == 1 so a == 1 */
            float c  = ox*ox + oy*oy + oz*oz - g_r2[k];
            float disc = hb*hb - c;
            float s  = sqrtf(disc > 0.0f ? disc : 0.0f);
            float t0 = -hb - s, t1 = -hb + s;
            float t  = (t0 > tmin) ? t0 : t1;
            int ok = (disc > 0.0f) & (t > tmin) & (t < best[j]);
            best[j] = ok ? t : best[j];
            id[j]   = ok ? k : id[j];
        }
    }

    /* ties go to the lower index, same as a plain sequential scan */
    float bt = best[0];
    int   bi = id[0];
    for (int j = 1; j < LANES; j++) {
        if (id[j] >= 0 && (best[j] < bt || (best[j] == bt && (bi < 0 || id[j] < bi)))) {
            bt = best[j];
            bi = id[j];
        }
    }
    int hit = bi >= 0 ? SPH_ID(bi) : -1;

    /* planes and boxes are few, plain loops are fine */
#if N_PLANES > 0
    for (int i = 0; i < N_PLANES; i++) {
        float den = vdot(g_pn[i], d);
        if (fabsf(den) < 1e-9f) continue;
        float t = (g_pd[i] - vdot(g_pn[i], o)) / den;
        if (t > tmin && t < bt) { bt = t; hit = i; }
    }
#endif
#if N_BOXES > 0
    for (int i = 0; i < N_BOXES; i++) {
        float t = box_hit(i, o, d, tmin);
        if (t < bt) { bt = t; hit = BOX_ID(i); }
    }
#endif
    *out_t = bt;
    return hit;
}

static inline Vec3 normal_at(int id, Vec3 p) {
#if N_PLANES > 0
    if (id < N_PLANES) return g_pn[id];
#endif
#if N_BOXES > 0
    if (id >= BOX_ID(0)) return box_normal(id - BOX_ID(0), p);
#endif
    /* sphere; signed radius -> inward normals for a hollow shell */
    int i = id - N_PLANES;
    return vscl(vsub(p, v3(g_cx[i], g_cy[i], g_cz[i])), g_inv_r[i]);
}

static inline Vec3 sky(Vec3 d) {
    static const float hz[3] = { SKY_HORIZON }, zn[3] = { SKY_ZENITH };
    float t = 0.5f * (d.y + 1.0f);
    return v3((1.0f-t)*hz[0] + t*zn[0],
              (1.0f-t)*hz[1] + t*zn[1],
              (1.0f-t)*hz[2] + t*zn[2]);
}

static inline Vec3 checker(Vec3 p) {
    float s = sinf(p.x * 1.15f) * sinf(p.z * 1.15f);
    return s > 0.0f ? v3(0.62f, 0.60f, 0.58f) : v3(0.14f, 0.15f, 0.17f);
}

/* ---- NEE ----
 * Pick a light with prob ~ power / dist^2 from p, then sample the cone it
 * subtends: pdf = 1 / (2*pi*(1 - cos_max)). With uniform picking a dim
 * light far away got as many shadow rays as a bright one close by. */

static Vec3 sample_lights(Vec3 p, Vec3 n, Vec3 albedo, Rng *rng) {
    const Vec3 zero = v3(0.0f, 0.0f, 0.0f);
    int nl = g_nlights;
    if (nl == 0) return zero;

    float cdf[ARR(N_SPHERES)] = {0}, total = 0.0f;
    for (int k = 0; k < nl; k++) {
        int j = g_lights[k];
        float dx = g_cx[j]-p.x, dy = g_cy[j]-p.y, dz = g_cz[j]-p.z;
        total += g_power[j] / fmaxf(dx*dx + dy*dy + dz*dz, g_r2[j]);
        cdf[k] = total;
    }
    float pick = randf(rng) * total;
    int k = 0;
    float lo = 0.0f;                    /* cdf value before light k */
    while (k < nl - 1 && cdf[k] < pick) lo = cdf[k++];
    float pick_pdf = (cdf[k] - lo) / total;
    int li = g_lights[k];
    Vec3 to = vsub(v3(g_cx[li], g_cy[li], g_cz[li]), p);
    float dist2 = vlen2(to), dist = sqrtf(dist2);
    float rad = fabsf(g_r[li]);
    if (dist <= rad * 1.0001f) return zero;

    float cos_max = sqrtf(fmaxf(0.0f, 1.0f - (rad*rad)/dist2));
    float ct  = 1.0f - randf(rng) * (1.0f - cos_max);
    float st  = sqrtf(fmaxf(0.0f, 1.0f - ct*ct));
    float phi = 2.0f * PI * randf(rng);

    Vec3 w = vscl(to, 1.0f/dist);
    Vec3 a = fabsf(w.x) > 0.9f ? v3(0,1,0) : v3(1,0,0);
    Vec3 v = vnorm(vcross(w, a));
    Vec3 u = vcross(w, v);
    Vec3 ldir = vadd(vadd(vscl(u, cosf(phi)*st), vscl(v, sinf(phi)*st)), vscl(w, ct));

    float cs = vdot(ldir, n);
    if (cs <= 1e-6f) return zero;

    /* visible iff the first hit is the light we picked */
    float t;
    Vec3 org = vadd(p, vscl(n, RAY_EPS));
    if (intersect(org, ldir, 1e-4f, &t) != SPH_ID(li)) return zero;

    float pdf = 1.0f / (2.0f * PI * fmaxf(1.0f - cos_max, 1e-9f));
    float wgt = cs / (PI * pdf * pick_pdf);            /* brdf = alb/pi */
    int lid = SPH_ID(li);
    return vscl(vmul(albedo, v3(g_ar[lid], g_ag[lid], g_ab[lid])), wgt);
}

/* ---- path tracing ---- */

static Vec3 trace(Vec3 o, Vec3 d, Rng *rng, int max_depth) {
    Vec3 L = v3(0,0,0), T = v3(1,1,1);
    int specular = 1;   /* count emitter hits? (not after diffuse, NEE has it) */

    for (int depth = 0; depth < max_depth; depth++) {
        float t;
        int id = intersect(o, d, 1e-4f, &t);
        if (id < 0) { L = vadd(L, vmul(T, sky(d))); break; }

        Vec3 p = vadd(o, vscl(d, t));
        Vec3 nrm = normal_at(id, p);
        int front = vdot(d, nrm) < 0.0f;
        Vec3 fn = front ? nrm : vneg(nrm);

        int mat = g_mat[id];
        Vec3 alb = (FLOOR_CHECKER && id == 0) ? checker(p) : v3(g_ar[id], g_ag[id], g_ab[id]);

        if (mat == EMISSIVE) {
            /* only emissive spheres are NEE'd, so other emitters always count */
            int is_sphere = id >= N_PLANES && id < BOX_ID(0);
            if (specular || !is_sphere) L = vadd(L, vmul(T, alb));
            break;
        }

        Vec3 nd;
        if (mat == LAMBERTIAN) {
            L = vadd(L, vmul(T, sample_lights(p, fn, alb, rng)));
            Vec3 sc = vadd(fn, random_unit(rng));
            nd = vlen2(sc) < 1e-16f ? fn : vnorm(sc);
            T = vmul(T, alb);
            specular = 0;
        } else if (mat == METAL) {
            Vec3 refl = vsub(d, vscl(fn, 2.0f * vdot(d, fn)));
            nd = vnorm(vadd(vnorm(refl), vscl(random_unit(rng), g_param[id])));
            T = vmul(T, alb);
            specular = 1;
        } else { /* DIELECTRIC */
            float ior = g_param[id];
            float ratio = front ? 1.0f/ior : ior;
            float ct = fminf(-vdot(d, fn), 1.0f);
            float st = sqrtf(fmaxf(0.0f, 1.0f - ct*ct));
            float r0 = (1.0f - ratio) / (1.0f + ratio); r0 *= r0;
            float sch = r0 + (1.0f - r0) * powf(1.0f - ct, 5.0f);
            if (ratio * st > 1.0f || randf(rng) < sch) {
                nd = vnorm(vsub(d, vscl(fn, 2.0f * vdot(d, fn))));
            } else {
                Vec3 perp = vscl(vadd(d, vscl(fn, ct)), ratio);
                Vec3 par  = vscl(fn, -sqrtf(fabsf(1.0f - fminf(vlen2(perp), 1.0f))));
                nd = vnorm(vadd(perp, par));
            }
            specular = 1;
        }

        /* offset to whichever side the new ray is going */
        o = vadd(p, vscl(fn, vdot(nd, fn) > 0.0f ? RAY_EPS : -RAY_EPS));
        d = nd;

        if (depth >= 4) {                       /* Russian roulette */
            float q = fmaxf(0.05f, fminf(vmaxc(T), 1.0f));
            if (randf(rng) >= q) break;
            T = vscl(T, 1.0f/q);
        }
    }
    return L;
}

/* ---- camera ---- */

static Vec3 cam_origin, cam_u, cam_v, cam_horiz, cam_vert, cam_ll;
static float cam_lens;

static void camera_init(Vec3 from, Vec3 at, Vec3 vup, float vfov,
                        float aspect, float aperture, float focus) {
    float half_h = tanf(vfov * PI / 180.0f / 2.0f);
    float half_w = aspect * half_h;
    Vec3 w = vnorm(vsub(from, at));
    cam_u = vnorm(vcross(vup, w));
    cam_v = vcross(w, cam_u);
    cam_origin = from;
    cam_horiz  = vscl(cam_u, 2.0f * half_w * focus);
    cam_vert   = vscl(cam_v, 2.0f * half_h * focus);
    cam_ll = vsub(vsub(vsub(from, vscl(cam_horiz, 0.5f)),
                       vscl(cam_vert, 0.5f)), vscl(w, focus));
    cam_lens = aperture * 0.5f;
}

static inline void camera_ray(float s, float t, Rng *rng, Vec3 *o, Vec3 *d) {
    float ang = randf(rng) * 2.0f * PI;
    float rad = cam_lens * sqrtf(randf(rng));       /* uniform on disc */
    Vec3 off = vadd(vscl(cam_u, rad*cosf(ang)), vscl(cam_v, rad*sinf(ang)));
    *o = vadd(cam_origin, off);
    Vec3 target = vadd(vadd(cam_ll, vscl(cam_horiz, s)), vscl(cam_vert, t));
    *d = vnorm(vsub(target, *o));
}

/* ---- PNG writer ---- */

static void put_be32(unsigned char *p, uint32_t v) {
    p[0]=(unsigned char)(v>>24); p[1]=(unsigned char)(v>>16);
    p[2]=(unsigned char)(v>>8);  p[3]=(unsigned char)v;
}

static void write_chunk(FILE *f, const char *kind, const unsigned char *data, size_t n) {
    unsigned char hdr[4];
    put_be32(hdr, (uint32_t)n);
    fwrite(hdr, 1, 4, f);
    fwrite(kind, 1, 4, f);
    if (n) fwrite(data, 1, n, f);
    uLong crc = crc32(0L, (const Bytef*)kind, 4);
    if (n) crc = crc32(crc, (const Bytef*)data, (uInt)n);
    put_be32(hdr, (uint32_t)crc);
    fwrite(hdr, 1, 4, f);
}

/* sum of abs values, bytes treated as signed. lowest cost wins */
static long filter_cost(const unsigned char *row, size_t n) {
    long c = 0;
    for (size_t i = 0; i < n; i++) { int b = row[i]; c += b < 128 ? b : 256 - b; }
    return c;
}

static size_t write_png(const char *path, const unsigned char *rgb, int w, int h) {
    size_t stride = (size_t)w * 3;
    unsigned char *raw  = malloc((stride + 1) * (size_t)h);
    unsigned char *cand = malloc(stride * 5);
    const unsigned char *prev = NULL;
    size_t raw_len = 0;

    for (int y = 0; y < h; y++) {
        const unsigned char *cur = rgb + (size_t)y * stride;
        for (size_t i = 0; i < stride; i++) {
            int a = i >= 3 ? cur[i-3] : 0;             /* left  */
            int b = prev ? prev[i] : 0;                /* up    */
            int c = (prev && i >= 3) ? prev[i-3] : 0;  /* upper-left */
            int pp = a + b - c;
            int pa = abs(pp-a), pb = abs(pp-b), pc = abs(pp-c);
            int paeth = (pa <= pb && pa <= pc) ? a : (pb <= pc ? b : c);
            cand[0*stride+i] = (unsigned char)(cur[i]);
            cand[1*stride+i] = (unsigned char)(cur[i] - a);
            cand[2*stride+i] = (unsigned char)(cur[i] - b);
            cand[3*stride+i] = (unsigned char)(cur[i] - ((a+b) >> 1));
            cand[4*stride+i] = (unsigned char)(cur[i] - paeth);
        }
        int best = 0; long bcost = -1;
        for (int k = 0; k < 5; k++) {
            long cost = filter_cost(cand + (size_t)k*stride, stride);
            if (bcost < 0 || cost < bcost) { bcost = cost; best = k; }
        }
        raw[raw_len++] = (unsigned char)best;
        memcpy(raw + raw_len, cand + (size_t)best*stride, stride);
        raw_len += stride;
        prev = cur;
    }

    z_stream zs; memset(&zs, 0, sizeof zs);
    deflateInit2(&zs, 9, Z_DEFLATED, 15, 9, Z_FILTERED);
    size_t cap = deflateBound(&zs, raw_len);
    unsigned char *idat = malloc(cap);
    zs.next_in = raw; zs.avail_in = (uInt)raw_len;
    zs.next_out = idat; zs.avail_out = (uInt)cap;
    deflate(&zs, Z_FINISH);
    size_t idat_len = cap - zs.avail_out;
    deflateEnd(&zs);

    FILE *f = fopen(path, "wb");
    if (!f) { perror("fopen"); exit(1); }
    fwrite("\x89PNG\r\n\x1a\n", 1, 8, f);

    unsigned char ihdr[13];
    put_be32(ihdr, (uint32_t)w); put_be32(ihdr+4, (uint32_t)h);
    ihdr[8]=8; ihdr[9]=2; ihdr[10]=0; ihdr[11]=0; ihdr[12]=0;
    write_chunk(f, "IHDR", ihdr, 13);

    unsigned char gama[4]; put_be32(gama, 45455);     /* gamma 1/2.2 */
    write_chunk(f, "gAMA", gama, 4);
    write_chunk(f, "IDAT", idat, idat_len);
    write_chunk(f, "IEND", NULL, 0);

    size_t total = (size_t)ftell(f);
    fclose(f);
    free(raw); free(cand); free(idat);
    return total;
}

/* ---- main ---- */

static inline float aces(float x) {
    const float a=2.51f, b=0.03f, c=2.43f, d=0.59f, e=0.14f;
    if (x < 0.0f) x = 0.0f;
    float m = (x*(a*x+b)) / (x*(c*x+d)+e);
    return m < 0.0f ? 0.0f : (m > 1.0f ? 1.0f : m);
}

static double now_sec(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

int main(int argc, char **argv) {
    int width = 960, height = 0, spp = 144, max_depth = 16;
    const char *out = "render_c.png";

    for (int i = 1; i < argc - 1; i++) {
        if      (!strcmp(argv[i], "--width"))  width     = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--height")) height    = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--spp"))   spp       = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--depth")) max_depth = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--out"))   out       = argv[++i];
    }
    if (height <= 0) height = (int)lrintf(width * 9.0f / 16.0f);

    scene_init();
    Vec3 from = v3(CAM_FROM), at = v3(CAM_AT);
    camera_init(from, at, v3(0,1,0), CAM_VFOV, (float)width/(float)height,
                CAM_APERTURE, sqrtf(vlen2(vsub(from, at))));

    int threads = 1;
#ifdef _OPENMP
    threads = omp_get_max_threads();
#endif
    fprintf(stderr, "  %dx%d  %d spp  depth %d  %d threads  %d objects\n",
            width, height, spp, max_depth, threads, N_OBJ);

    float *acc = malloc(sizeof(float) * 3 * (size_t)width * (size_t)height);
    double t0 = now_sec();

    /* dynamic: rows with lots of glass are much slower than sky rows */
#pragma omp parallel for schedule(dynamic, 4)
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            Rng rng;
            /* per-pixel seed so output doesn't depend on thread count */
            rng_seed(&rng, ((uint64_t)y << 20) ^ (uint64_t)x, 0x853C49E6748FEA9BULL);
            Vec3 sum = v3(0,0,0);
            for (int s = 0; s < spp; s++) {
                float u = ((float)x + randf(&rng)) / (float)width;
                float v = 1.0f - ((float)y + randf(&rng)) / (float)height;
                Vec3 o, d;
                camera_ray(u, v, &rng, &o, &d);
                sum = vadd(sum, trace(o, d, &rng, max_depth));
            }
            size_t k = 3 * ((size_t)y * (size_t)width + (size_t)x);
            acc[k+0] = sum.x / spp; acc[k+1] = sum.y / spp; acc[k+2] = sum.z / spp;
        }
    }
    double elapsed = now_sec() - t0;

    unsigned char *rgb = malloc(3 * (size_t)width * (size_t)height);
    for (size_t i = 0; i < 3 * (size_t)width * (size_t)height; i++)
        rgb[i] = (unsigned char)(powf(aces(acc[i] * EXPOSURE), 1.0f/2.2f) * 255.0f + 0.5f);

    size_t bytes = write_png(out, rgb, width, height);
    double rays = (double)width * height * spp;
    fprintf(stderr,
            "  %6.2fs   %.2fM primary rays/s\n  %s  %.1f KiB  (%.1f%% of raw RGB)\n",
            elapsed, rays/elapsed/1e6, out, bytes/1024.0,
            100.0 * bytes / (3.0 * width * height));
    free(acc); free(rgb);
    return 0;
}

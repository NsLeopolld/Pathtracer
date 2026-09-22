// =====================================================================
//  Spectral path tracer -- CUDA kernel
//
//  One thread owns one pixel and carries whole paths to completion.
//  Colour is tracked as 4 wavelengths per path (hero wavelength sampling,
//  Wilkie et al. 2014), which is what makes dispersion and thin-film
//  interference possible at all: both are wavelength-dependent, so an RGB
//  renderer physically cannot express them.
//
//  Estimator per path: NEE + BSDF sampling combined with MIS (power
//  heuristic). Lights are picked proportional to power/distance^2 from the
//  shading point. Delta surfaces skip NEE and take MIS weight 1.
// =====================================================================

#define NL        4
#define LMIN      380.0f
#define LMAX      730.0f
#define LSPAN     (LMAX - LMIN)
#define NBINS     71
#define PI        3.14159265358979f
#define INV_PI    0.318309886183791f

#define MAXS      64
#define MAXP      8
#define MAXLI     64

#define M_DIFFUSE    0
#define M_CONDUCTOR  1
#define M_DIELECTRIC 2
#define M_PLASTIC    3
#define M_EMISSIVE   4
#define M_THINFILM   5

#define RAY_EPS   1e-3f

struct Material {
    float cr, cg, cb;        // base colour / F0 / emission radiance
    float c2r, c2g, c2b;     // second checker colour
    float rough;             // GGX alpha
    float ior;
    float abbe;              // Abbe number; 0 = no dispersion
    float film;              // thin-film base thickness, nm
    float cscale;            // checker frequency
    int   type;
    int   checker;
};

extern "C" {
__constant__ float4       c_sph[MAXS];     // centre.xyz, signed radius
__constant__ int          c_smat[MAXS];
__constant__ float4       c_pln[MAXP];     // normal.xyz, plane offset
__constant__ int          c_pmat[MAXP];
__constant__ int          c_light[MAXLI];  // indices of emissive spheres
__constant__ float        c_lpow[MAXLI];   // luminance * r^2
__constant__ int          c_cnt[4];        // nsph, npln, nlight
__constant__ float        c_cam[20];
__constant__ float        c_sky[8];
__constant__ unsigned int c_sobol[4][32];
__constant__ float        c_ealb[32*32];  // GGX directional albedo E(mu, alpha)
}

// ---------------------------------------------------------------- vec3
struct V3 { float x, y, z; };
__device__ __forceinline__ V3 v3(float x, float y, float z) { V3 r; r.x=x; r.y=y; r.z=z; return r; }
__device__ __forceinline__ V3 add(V3 a, V3 b) { return v3(a.x+b.x, a.y+b.y, a.z+b.z); }
__device__ __forceinline__ V3 sub(V3 a, V3 b) { return v3(a.x-b.x, a.y-b.y, a.z-b.z); }
__device__ __forceinline__ V3 mul(V3 a, float s) { return v3(a.x*s, a.y*s, a.z*s); }
__device__ __forceinline__ V3 neg(V3 a) { return v3(-a.x, -a.y, -a.z); }
__device__ __forceinline__ float dot3(V3 a, V3 b) { return a.x*b.x + a.y*b.y + a.z*b.z; }
__device__ __forceinline__ V3 cross3(V3 a, V3 b) {
    return v3(a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x);
}
__device__ __forceinline__ V3 norm3(V3 a) { return mul(a, rsqrtf(dot3(a,a))); }

// Duff et al. 2017, branchless orthonormal basis.
__device__ __forceinline__ void onb(V3 n, V3 &t, V3 &b) {
    float sg = copysignf(1.0f, n.z);
    float a  = -1.0f / (sg + n.z);
    float bq = n.x * n.y * a;
    t = v3(1.0f + sg*n.x*n.x*a, sg*bq, -sg*n.x);
    b = v3(bq, sg + n.y*n.y*a, -n.y);
}
__device__ __forceinline__ V3 to_local(V3 w, V3 t, V3 b, V3 n) { return v3(dot3(w,t), dot3(w,b), dot3(w,n)); }
__device__ __forceinline__ V3 to_world(V3 w, V3 t, V3 b, V3 n) {
    return v3(t.x*w.x + b.x*w.y + n.x*w.z,
              t.y*w.x + b.y*w.y + n.y*w.z,
              t.z*w.x + b.z*w.y + n.z*w.z);
}

// ------------------------------------------------------------ spectrum
struct Spec { float v[NL]; };
__device__ __forceinline__ Spec sp_set(float x) { Spec s;
#pragma unroll
    for (int i=0;i<NL;i++) s.v[i]=x; return s; }
__device__ __forceinline__ Spec sp_mul(Spec a, Spec b) { Spec s;
#pragma unroll
    for (int i=0;i<NL;i++) s.v[i]=a.v[i]*b.v[i]; return s; }
__device__ __forceinline__ Spec sp_scale(Spec a, float k) { Spec s;
#pragma unroll
    for (int i=0;i<NL;i++) s.v[i]=a.v[i]*k; return s; }
__device__ __forceinline__ float sp_max(Spec a) {
    float m = a.v[0];
#pragma unroll
    for (int i=1;i<NL;i++) m = fmaxf(m, a.v[i]); return m; }
__device__ __forceinline__ float sp_avg(Spec a) {
    float m = 0.0f;
#pragma unroll
    for (int i=0;i<NL;i++) m += a.v[i]; return m * (1.0f/NL); }

struct SpecCtx {
    float lam[NL];
    float Br[NL], Bg[NL], Bb[NL];
    float il[NL];
    float Rr[NL], Rg[NL], Rb[NL];
};

__device__ __forceinline__ float tab(const float* __restrict__ t, int row, float lam) {
    float x = (lam - LMIN) * (1.0f/5.0f);
    int i = (int)x;
    i = max(0, min(NBINS-2, i));
    float f = x - (float)i;
    const float* r = t + row*NBINS;
    return fmaf(r[i+1]-r[i], f, r[i]);
}

__device__ __forceinline__ Spec rgb2spec(const SpecCtx &sc, float r, float g, float b) {
    Spec s;
#pragma unroll
    for (int i=0;i<NL;i++) s.v[i] = r*sc.Br[i] + g*sc.Bg[i] + b*sc.Bb[i];
    return s;
}
__device__ __forceinline__ Spec rgb2emit(const SpecCtx &sc, float r, float g, float b) {
    Spec s;
#pragma unroll
    for (int i=0;i<NL;i++) s.v[i] = (r*sc.Br[i] + g*sc.Bg[i] + b*sc.Bb[i]) * sc.il[i];
    return s;
}

// ------------------------------------------------------------- sampler
__device__ __forceinline__ unsigned int hashu(unsigned int x) {
    x ^= x >> 16; x *= 0x7feb352du; x ^= x >> 15; x *= 0x846ca68bu; x ^= x >> 16; return x;
}
__device__ __forceinline__ unsigned int hash2(unsigned int a, unsigned int b) {
    return hashu(a ^ (b * 0x9e3779b9u));
}
// Burley 2020, hash-based Owen scrambling.
__device__ __forceinline__ unsigned int lk_perm(unsigned int x, unsigned int seed) {
    x += seed;
    x ^= x * 0x6c50b47cu;
    x ^= x * 0xb82f1e52u;
    x ^= x * 0xc7afe638u;
    x ^= x * 0x8d22f6e6u;
    return x;
}
__device__ __forceinline__ unsigned int owen(unsigned int x, unsigned int seed) {
    x = __brev(x); x = lk_perm(x, seed); return __brev(x);
}
__device__ __forceinline__ unsigned int sobol_dim(unsigned int index, int dim) {
    unsigned int X = 0;
#pragma unroll
    for (int bit = 0; bit < 32; bit++) {
        unsigned int m = (index >> bit) & 1u;
        X ^= m * c_sobol[dim][bit];
    }
    return X;
}
struct Sampler { unsigned int pix, index, state; };
__device__ __forceinline__ float rnd(Sampler &s) {
    s.state = s.state * 747796405u + 2891336453u;
    unsigned int w = ((s.state >> ((s.state >> 28) + 4u)) ^ s.state) * 277803737u;
    w = (w >> 22) ^ w;
    return (w >> 8) * 0x1.0p-24f;
}
#ifndef LD_GROUPS
#define LD_GROUPS 8u   /* dimension groups drawn from Sobol; 0 = plain PRNG */
#endif
__device__ __forceinline__ float4 sample4(Sampler &s, unsigned int group) {
    if (group < LD_GROUPS) {
        unsigned int seed = hash2(s.pix, group);
        unsigned int idx  = owen(s.index, seed);
        float4 r;
        r.x = (owen(sobol_dim(idx,0), hash2(seed, 1u)) >> 8) * 0x1.0p-24f;
        r.y = (owen(sobol_dim(idx,1), hash2(seed, 2u)) >> 8) * 0x1.0p-24f;
        r.z = (owen(sobol_dim(idx,2), hash2(seed, 3u)) >> 8) * 0x1.0p-24f;
        r.w = (owen(sobol_dim(idx,3), hash2(seed, 4u)) >> 8) * 0x1.0p-24f;
        return r;
    }
    return make_float4(rnd(s), rnd(s), rnd(s), rnd(s));
}

// ---------------------------------------------------------- intersection
struct Hit { float t; int prim; int plane; };

__device__ bool intersect(V3 o, V3 d, float tmin, Hit &h) {
    float best = 1e30f; int id = -1, pl = 0;
    int ns = c_cnt[0];
    for (int i = 0; i < ns; i++) {
        float4 s = c_sph[i];
        float ox = o.x - s.x, oy = o.y - s.y, oz = o.z - s.z;
        float hb = d.x*ox + d.y*oy + d.z*oz;
        float c  = ox*ox + oy*oy + oz*oz - s.w*s.w;
        float disc = hb*hb - c;
        float sq = sqrtf(fmaxf(disc, 0.0f));
        float t0 = -hb - sq, t1 = -hb + sq;
        float t  = (t0 > tmin) ? t0 : t1;
        bool ok = (disc > 0.0f) && (t > tmin) && (t < best);
        best = ok ? t : best;
        id   = ok ? i : id;
    }
    int np = c_cnt[1];
    for (int i = 0; i < np; i++) {
        float4 p = c_pln[i];
        float den = d.x*p.x + d.y*p.y + d.z*p.z;
        if (fabsf(den) < 1e-7f) continue;
        float t = (p.w - (o.x*p.x + o.y*p.y + o.z*p.z)) / den;
        bool ok = (t > tmin) && (t < best);
        best = ok ? t : best;
        id   = ok ? i : id;
        pl   = ok ? 1 : pl;
    }
    h.t = best; h.prim = id; h.plane = pl;
    return id >= 0;
}

__device__ __forceinline__ int mat_of(const Hit &h) {
    return h.plane ? c_pmat[h.prim] : c_smat[h.prim];
}

// ------------------------------------------------------------ materials
__device__ __forceinline__ float fresnel_dielectric(float cosi, float eta) {
    // eta = n_transmitted / n_incident. Full unpolarised Fresnel, not Schlick.
    float s2t = (1.0f - cosi*cosi) / (eta*eta);
    if (s2t >= 1.0f) return 1.0f;                       // total internal reflection
    float cost = sqrtf(1.0f - s2t);
    float rs = (cosi - eta*cost) / (cosi + eta*cost);
    float rp = (eta*cosi - cost) / (eta*cosi + cost);
    return 0.5f * (rs*rs + rp*rp);
}

__device__ __forceinline__ float cauchy_ior(float ior_d, float abbe, float lam) {
    if (abbe <= 0.0f) return ior_d;
    // Cauchy two-term fit pinned to n_d and the Abbe number.
    const float lF = 0.4861f, lC = 0.6563f, lD = 0.5876f;   // micrometres
    float B = (ior_d - 1.0f) / (abbe * (1.0f/(lF*lF) - 1.0f/(lC*lC)));
    float A = ior_d - B/(lD*lD);
    float l = lam * 1e-3f;
    return A + B/(l*l);
}

// Airy reflectance of a thin film suspended in air (soap bubble).
__device__ float thinfilm_R(float cosi, float nf, float d_nm, float lam) {
    float s2t = (1.0f - cosi*cosi) / (nf*nf);
    if (s2t >= 1.0f) return 1.0f;
    float cost = sqrtf(1.0f - s2t);
    float rs = (cosi - nf*cost) / (cosi + nf*cost);
    float rp = (nf*cosi - cost) / (nf*cosi + cost);
    float delta = 4.0f*PI*nf*d_nm*cost / lam;
    float cd = __cosf(delta);
    // Two interfaces, r23 = -r12, summed as an Airy series.
    float rs2 = rs*rs, rp2 = rp*rp;
    float Rs = (2.0f*rs2*(1.0f-cd)) / fmaxf(1.0f + rs2*rs2 - 2.0f*rs2*cd, 1e-9f);
    float Rp = (2.0f*rp2*(1.0f-cd)) / fmaxf(1.0f + rp2*rp2 - 2.0f*rp2*cd, 1e-9f);
    return 0.5f * (Rs + Rp);
}

__device__ __forceinline__ float film_thickness(V3 p, int sid, float base) {
    float4 s = c_sph[sid];
    float inv = 1.0f / fabsf(s.w);
    float lx = (p.x - s.x)*inv, ly = (p.y - s.y)*inv, lz = (p.z - s.z)*inv;
    float h = 0.5f*(ly + 1.0f);                       // 0 at bottom, 1 at top
    float t = base * (1.45f - 0.95f*h);               // drains thinner at the top
    t *= 1.0f + 0.17f*__sinf(6.5f*lx + 3.0f*__sinf(5.0f*lz) + 2.0f*ly);
    return fmaxf(t, 20.0f);
}

__device__ __forceinline__ float ggx_D(float mz, float a) {
    float a2 = a*a;
    float t = mz*mz*(a2 - 1.0f) + 1.0f;
    return a2 / fmaxf(PI*t*t, 1e-12f);
}
__device__ __forceinline__ float smith_lambda(float wz, float a) {
    float z2 = wz*wz;
    if (z2 >= 0.999999f) return 0.0f;
    float t2 = (1.0f - z2) / z2;
    return 0.5f * (sqrtf(1.0f + a*a*t2) - 1.0f);
}
__device__ V3 ggx_sample_vndf(V3 wo, float a, float u1, float u2) {
    V3 Vh = norm3(v3(a*wo.x, a*wo.y, wo.z));
    float lensq = Vh.x*Vh.x + Vh.y*Vh.y;
    V3 T1 = lensq > 0.0f ? mul(v3(-Vh.y, Vh.x, 0.0f), rsqrtf(lensq)) : v3(1,0,0);
    V3 T2 = cross3(Vh, T1);
    float r = sqrtf(u1), phi = 2.0f*PI*u2;
    float t1 = r*__cosf(phi), t2 = r*__sinf(phi);
    float s = 0.5f*(1.0f + Vh.z);
    t2 = (1.0f - s)*sqrtf(fmaxf(0.0f, 1.0f - t1*t1)) + s*t2;
    float nz = sqrtf(fmaxf(0.0f, 1.0f - t1*t1 - t2*t2));
    V3 Nh = add(add(mul(T1,t1), mul(T2,t2)), mul(Vh,nz));
    return norm3(v3(a*Nh.x, a*Nh.y, fmaxf(1e-6f, Nh.z)));
}
#define NE 32
// Bilinear lookup into E(mu, alpha). The table is uniform in sqrt(alpha),
// which puts resolution where the curve bends.
__device__ __forceinline__ float ggx_E(float mu, float alpha) {
    float t = sqrtf(fmaxf(alpha, 0.0f));
    float fx = fminf(fmaxf(mu*NE - 0.5f, 0.0f), (float)(NE-1));
    float fy = fminf(fmaxf(t *NE - 0.5f, 0.0f), (float)(NE-1));
    int ix = (int)fx, iy = (int)fy;
    int jx = min(ix+1, NE-1), jy = min(iy+1, NE-1);
    float dx = fx - ix, dy = fy - iy;
    float a = c_ealb[iy*NE+ix]*(1.0f-dx) + c_ealb[iy*NE+jx]*dx;
    float b = c_ealb[jy*NE+ix]*(1.0f-dx) + c_ealb[jy*NE+jx]*dx;
    return fmaxf(1e-3f, a*(1.0f-dy) + b*dy);
}
// Turquin 2019: scale the single-scattering lobe to put back the energy
// that multiple microsurface bounces would have carried. For F0=1 this is
// exactly 1/E, so a white rough metal becomes perfectly energy preserving.
__device__ __forceinline__ Spec ms_comp(Spec f0, float mu, float alpha) {
    float k = (1.0f - ggx_E(mu, alpha)) / ggx_E(mu, alpha);
    Spec r;
#pragma unroll
    for (int i=0;i<NL;i++) r.v[i] = 1.0f + f0.v[i]*k;
    return r;
}
__device__ __forceinline__ float ms_comp1(float f0, float mu, float alpha) {
    float E = ggx_E(mu, alpha);
    return 1.0f + f0*(1.0f - E)/E;
}

__device__ __forceinline__ Spec fresnel_schlick_spec(Spec f0, float c) {
    float w = __powf(1.0f - c, 5.0f);
    Spec r;
#pragma unroll
    for (int i=0;i<NL;i++) r.v[i] = fmaf(1.0f - f0.v[i], w, f0.v[i]);
    return r;
}

// f * cos(theta_i), plus the pdf, for the non-delta lobes. Used by NEE.
__device__ Spec bsdf_eval(int type, Spec alb, float alpha, float ior,
                          V3 wo, V3 wi, float &pdf) {
    pdf = 0.0f;
    if (wi.z <= 0.0f || wo.z <= 0.0f) return sp_set(0.0f);

    if (type == M_DIFFUSE) {
        pdf = wi.z * INV_PI;
        return sp_scale(alb, wi.z * INV_PI);
    }
    if (type == M_CONDUCTOR) {
        V3 m = norm3(add(wo, wi));
        float D = ggx_D(m.z, alpha);
        float lo = smith_lambda(wo.z, alpha), li = smith_lambda(wi.z, alpha);
        float G2 = 1.0f / (1.0f + lo + li);
        float G1 = 1.0f / (1.0f + lo);
        Spec F = fresnel_schlick_spec(alb, fmaxf(dot3(wi, m), 0.0f));
        pdf = G1 * D / (4.0f * wo.z);
        return sp_mul(sp_scale(F, D * G2 / (4.0f * wo.z)),
                      ms_comp(alb, wo.z, alpha));          // f * cos_i
    }
    if (type == M_PLASTIC) {
        float f0 = (1.0f - ior)/(1.0f + ior); f0 *= f0;
        V3 m = norm3(add(wo, wi));
        float D = ggx_D(m.z, alpha);
        float lo = smith_lambda(wo.z, alpha), li = smith_lambda(wi.z, alpha);
        float G2 = 1.0f / (1.0f + lo + li);
        float G1 = 1.0f / (1.0f + lo);
        float Fm = f0 + (1.0f - f0)*__powf(1.0f - fmaxf(dot3(wi,m),0.0f), 5.0f);
        float spec = Fm * D * G2 / (4.0f * wo.z) * ms_comp1(f0, wo.z, alpha);
        float Fo = f0 + (1.0f - f0)*__powf(1.0f - wo.z, 5.0f);
        float Fi = f0 + (1.0f - f0)*__powf(1.0f - wi.z, 5.0f);
        // Divide by (1 - Fdr) to account for light the coat reflects back
        // down and the base gets a second go at. Fdr is the cosine-weighted
        // average of Schlick, which integrates to f0 + (1-f0)/21 exactly.
        float Fdr = f0 + (1.0f - f0)*(1.0f/21.0f);
        Spec diff = sp_scale(alb, (1.0f-Fo)*(1.0f-Fi)*INV_PI*wi.z/(1.0f - Fdr));
        float ps = fminf(0.95f, fmaxf(0.05f, Fo/(Fo + (1.0f-Fo)*fmaxf(sp_avg(alb),0.02f))));
        pdf = ps * (G1*D/(4.0f*wo.z)) + (1.0f-ps) * wi.z * INV_PI;
        Spec r;
#pragma unroll
        for (int i=0;i<NL;i++) r.v[i] = diff.v[i] + spec;
        return r;
    }
    return sp_set(0.0f);
}

struct BSample { V3 wi; Spec w; float pdf; int delta; };

__device__ bool bsdf_sample(int type, Spec alb, float alpha, float ior,
                            V3 wo, float u1, float u2, float u3, BSample &bs) {
    bs.delta = 0;
    if (wo.z <= 0.0f) return false;

    if (type == M_DIFFUSE) {
        float r = sqrtf(u1), phi = 2.0f*PI*u2;
        bs.wi = v3(r*__cosf(phi), r*__sinf(phi), sqrtf(fmaxf(0.0f, 1.0f-u1)));
        bs.pdf = bs.wi.z * INV_PI;
        bs.w = alb;                                   // f*cos/pdf collapses
        return bs.wi.z > 0.0f;
    }
    if (type == M_CONDUCTOR) {
        if (alpha < 1e-4f) {                          // perfect mirror
            bs.wi = v3(-wo.x, -wo.y, wo.z);
            bs.w = fresnel_schlick_spec(alb, wo.z);
            bs.pdf = 1.0f; bs.delta = 1;
            return true;
        }
        V3 m = ggx_sample_vndf(wo, alpha, u1, u2);
        float wom = dot3(wo, m);
        bs.wi = sub(mul(m, 2.0f*wom), wo);
        if (bs.wi.z <= 0.0f) return false;
        float lo = smith_lambda(wo.z, alpha), li = smith_lambda(bs.wi.z, alpha);
        float G2 = 1.0f/(1.0f + lo + li), G1 = 1.0f/(1.0f + lo);
        Spec F = fresnel_schlick_spec(alb, fmaxf(wom, 0.0f));
        bs.w   = sp_mul(sp_scale(F, G2/G1), ms_comp(alb, wo.z, alpha));
        bs.pdf = G1 * ggx_D(m.z, alpha) / (4.0f*wo.z);
        return true;
    }
    if (type == M_PLASTIC) {
        float f0 = (1.0f - ior)/(1.0f + ior); f0 *= f0;
        float Fo = f0 + (1.0f - f0)*__powf(1.0f - wo.z, 5.0f);
        float ps = fminf(0.95f, fmaxf(0.05f, Fo/(Fo + (1.0f-Fo)*fmaxf(sp_avg(alb),0.02f))));
        if (u3 < ps) {
            V3 m = ggx_sample_vndf(wo, alpha, u1, u2);
            bs.wi = sub(mul(m, 2.0f*dot3(wo,m)), wo);
        } else {
            float r = sqrtf(u1), phi = 2.0f*PI*u2;
            bs.wi = v3(r*__cosf(phi), r*__sinf(phi), sqrtf(fmaxf(0.0f,1.0f-u1)));
        }
        if (bs.wi.z <= 0.0f) return false;
        float pdf; Spec fc = bsdf_eval(M_PLASTIC, alb, alpha, ior, wo, bs.wi, pdf);
        if (pdf <= 0.0f) return false;
        bs.w = sp_scale(fc, 1.0f/pdf);
        bs.pdf = pdf;
        return true;
    }
    return false;
}

// ------------------------------------------------------------- lighting
__device__ float light_total(V3 p, float *w, int n) {
    float total = 0.0f;
    for (int k = 0; k < n; k++) {
        int j = c_light[k];
        float4 s = c_sph[j];
        float dx = s.x-p.x, dy = s.y-p.y, dz = s.z-p.z;
        float d2 = dx*dx + dy*dy + dz*dz;
        float wk = c_lpow[k] / fmaxf(d2, s.w*s.w);
        total += wk;
        w[k] = total;
    }
    return total;
}

__device__ __forceinline__ float cone_pdf_for(V3 p, int sid) {
    float4 s = c_sph[sid];
    float dx = s.x-p.x, dy = s.y-p.y, dz = s.z-p.z;
    float d2 = dx*dx + dy*dy + dz*dz;
    float r2 = s.w*s.w;
    if (d2 <= r2*1.0002f) return 0.0f;
    float cmax = sqrtf(fmaxf(0.0f, 1.0f - r2/d2));
    return 1.0f / (2.0f*PI*fmaxf(1.0f - cmax, 1e-9f));
}

// pdf (solid angle) that NEE would have used to generate direction -> light `sid` from p
__device__ float nee_pdf(V3 p, int sid, int nlight) {
    float w[MAXLI];
    float total = light_total(p, w, nlight);
    if (total <= 0.0f) return 0.0f;
    float pick = 0.0f;
    for (int k = 0; k < nlight; k++) {
        if (c_light[k] == sid) { pick = (w[k] - (k ? w[k-1] : 0.0f)) / total; break; }
    }
    if (pick <= 0.0f) return 0.0f;
    return pick * cone_pdf_for(p, sid);
}

__device__ __forceinline__ float power_heuristic(float a, float b) {
    float a2 = a*a, b2 = b*b;
    return a2 / fmaxf(a2 + b2, 1e-20f);
}

// Shadow ray. Thin films are not opaque: the ray passes through them and
// picks up their transmittance, which keeps soft shadows under a bubble
// from having to rely on chance BSDF hits.
__device__ bool shadow(V3 o, V3 d, int target, const SpecCtx &sc,
                       const Material* __restrict__ mats, Spec &trans) {
    trans = sp_set(1.0f);
    V3 p = o;
    for (int i = 0; i < 6; i++) {
        Hit h;
        if (!intersect(p, d, RAY_EPS, h)) return false;
        if (!h.plane && h.prim == target) return true;
        const Material m = mats[mat_of(h)];
        if (m.type != M_THINFILM) return false;
        V3 hp = add(p, mul(d, h.t));
        V3 n  = mul(sub(hp, v3(c_sph[h.prim].x, c_sph[h.prim].y, c_sph[h.prim].z)),
                    1.0f/c_sph[h.prim].w);
        float ci = fabsf(dot3(d, norm3(n)));
        float th = film_thickness(hp, h.prim, m.film);
#pragma unroll
        for (int k = 0; k < NL; k++)
            trans.v[k] *= (1.0f - thinfilm_R(ci, m.ior, th, sc.lam[k]));
        if (sp_max(trans) < 1e-4f) return false;
        p = add(hp, mul(d, RAY_EPS));
    }
    return false;
}

// ----------------------------------------------------------------- sky
__device__ __forceinline__ Spec sky_spec(const SpecCtx &sc, V3 d) {
    float t = 0.5f*(d.y + 1.0f);
    float r = (1.0f-t)*c_sky[0] + t*c_sky[3];
    float g = (1.0f-t)*c_sky[1] + t*c_sky[4];
    float b = (1.0f-t)*c_sky[2] + t*c_sky[5];
    return rgb2emit(sc, r, g, b);
}

// =================================================================== main
extern "C" __global__ void render(
        const int* __restrict__ active, int n_active,
        double* __restrict__ accum, int* __restrict__ counts,
        const Material* __restrict__ mats, const float* __restrict__ tb,
        int W, int H, int spp, int max_depth,
        unsigned int seed_base, int mis_mode, float clamp_val)
{
    int tid = blockIdx.x*blockDim.x + threadIdx.x;
    if (tid >= n_active) return;
    int pid = active[tid];
    int px = pid % W, py = pid / W;

    Sampler smp;
    smp.pix   = hash2((unsigned int)pid, seed_base);
    smp.state = hashu(smp.pix ^ 0x9e3779b9u) | 1u;
    int base  = counts[pid];

    int nlight = c_cnt[2];
    double sr = 0.0, sg = 0.0, sb = 0.0, sl2 = 0.0;

    for (int s = 0; s < spp; s++) {
        smp.index = (unsigned int)(base + s);

        // --- wavelengths: one hero, three stratified companions --------
        float4 gw = sample4(smp, 1u);
        SpecCtx sc;
#pragma unroll
        for (int i = 0; i < NL; i++) {
            float u = gw.x + (float)i * (1.0f/NL);
            u -= floorf(u);
            sc.lam[i] = LMIN + u*LSPAN;
            sc.Br[i] = tab(tb,0,sc.lam[i]); sc.Bg[i] = tab(tb,1,sc.lam[i]);
            sc.Bb[i] = tab(tb,2,sc.lam[i]); sc.il[i] = tab(tb,3,sc.lam[i]);
            sc.Rr[i] = tab(tb,4,sc.lam[i]); sc.Rg[i] = tab(tb,5,sc.lam[i]);
            sc.Rb[i] = tab(tb,6,sc.lam[i]);
        }

        // --- camera ray -------------------------------------------------
        float4 g0 = sample4(smp, 0u);
        float u = ((float)px + g0.x) / (float)W;
        float v = 1.0f - ((float)py + g0.y) / (float)H;
        float lr = c_cam[18];
        float ang = 2.0f*PI*g0.z, rad = lr*sqrtf(g0.w);
        V3 cu = v3(c_cam[12], c_cam[13], c_cam[14]);
        V3 cv = v3(c_cam[15], c_cam[16], c_cam[17]);
        V3 off = add(mul(cu, rad*__cosf(ang)), mul(cv, rad*__sinf(ang)));
        V3 o = add(v3(c_cam[0], c_cam[1], c_cam[2]), off);
        V3 tgt = add(add(v3(c_cam[9], c_cam[10], c_cam[11]),
                         mul(v3(c_cam[3], c_cam[4], c_cam[5]), u)),
                     mul(v3(c_cam[6], c_cam[7], c_cam[8]), v));
        V3 d = norm3(sub(tgt, o));

        Spec L = sp_set(0.0f), T = sp_set(1.0f);
        int   prev_delta = 1;
        float prev_pdf   = 1.0f;
        V3    prev_p     = o;
        int   secondary_alive = 1;

        for (int depth = 0; depth < max_depth; depth++) {
            Hit h;
            if (!intersect(o, d, RAY_EPS, h)) {
                Spec sk = sky_spec(sc, d);
#pragma unroll
                for (int i=0;i<NL;i++) L.v[i] += T.v[i]*sk.v[i];
                break;
            }
            V3 p = add(o, mul(d, h.t));
            V3 ng;
            if (h.plane) {
                ng = v3(c_pln[h.prim].x, c_pln[h.prim].y, c_pln[h.prim].z);
            } else {
                float4 sp4 = c_sph[h.prim];
                ng = mul(sub(p, v3(sp4.x, sp4.y, sp4.z)), 1.0f/sp4.w);
            }
            ng = norm3(ng);
            int front = dot3(d, ng) < 0.0f;
            V3 fn = front ? ng : neg(ng);

            const Material m = mats[mat_of(h)];

            // ---- emitter ------------------------------------------------
            if (m.type == M_EMISSIVE) {
                if (mis_mode != 2 || prev_delta) {
                    float w = 1.0f;
                    if (!prev_delta && mis_mode == 0) {
                        float lp = nee_pdf(prev_p, h.prim, nlight);
                        w = power_heuristic(prev_pdf, lp);
                    }
                    Spec Le = rgb2emit(sc, m.cr, m.cg, m.cb);
#pragma unroll
                    for (int i=0;i<NL;i++) L.v[i] += T.v[i]*Le.v[i]*w;
                }
                break;
            }

            // ---- surface colour ----------------------------------------
            float ar = m.cr, ag = m.cg, ab = m.cb;
            if (m.checker) {
                V3 tt, bb; onb(ng, tt, bb);
                float cu2 = dot3(p, tt)*m.cscale, cv2 = dot3(p, bb)*m.cscale;
                if (__sinf(cu2)*__sinf(cv2) <= 0.0f) { ar = m.c2r; ag = m.c2g; ab = m.c2b; }
            }
            Spec alb = rgb2spec(sc, ar, ag, ab);

            V3 nd;                                    // next direction
            // ---- delta surfaces: dielectric and thin film ---------------
            if (m.type == M_DIELECTRIC) {
                if (m.abbe > 0.0f && secondary_alive) {
                    // Refraction bends each wavelength differently, so the
                    // companions can no longer share this path. Drop them and
                    // reweight the hero (pbrt's TerminateSecondary).
                    T.v[0] *= (float)NL;
#pragma unroll
                    for (int i=1;i<NL;i++) T.v[i] = 0.0f;
                    secondary_alive = 0;
                }
                float n_lam = cauchy_ior(m.ior, m.abbe, sc.lam[0]);
                float eta = front ? n_lam : 1.0f/n_lam;
                float ci = fminf(dot3(neg(d), fn), 1.0f);
                float F = fresnel_dielectric(ci, eta);
                float uu = rnd(smp);
                if (uu < F) {
                    nd = add(d, mul(fn, 2.0f*ci));
                } else {
                    float s2t = (1.0f - ci*ci)/(eta*eta);
                    float ct = sqrtf(fmaxf(0.0f, 1.0f - s2t));
                    nd = add(mul(add(d, mul(fn, ci)), 1.0f/eta), mul(fn, -ct));
                }
                nd = norm3(nd);
                prev_delta = 1; prev_pdf = 1.0f; prev_p = p;
                o = add(p, mul(fn, dot3(nd, fn) > 0.0f ? RAY_EPS : -RAY_EPS));
                d = nd;
                continue;
            }
            if (m.type == M_THINFILM) {
                float ci = fminf(dot3(neg(d), fn), 1.0f);
                float th = film_thickness(p, h.prim, m.film);
                Spec R;
#pragma unroll
                for (int i=0;i<NL;i++) R.v[i] = thinfilm_R(ci, m.ior, th, sc.lam[i]);
                // Choose using the mean over live wavelengths, never the hero
                // alone: at a hero null the companions would never reflect.
                float Pr = 0.0f; int live = 0;
#pragma unroll
                for (int i=0;i<NL;i++) if (T.v[i] != 0.0f) { Pr += R.v[i]; live++; }
                Pr = live ? Pr/(float)live : 0.0f;
                Pr = fminf(0.98f, fmaxf(0.02f, Pr));
                if (rnd(smp) < Pr) {
#pragma unroll
                    for (int i=0;i<NL;i++) T.v[i] *= R.v[i]/Pr;
                    nd = add(d, mul(fn, 2.0f*ci));
                    nd = norm3(nd);
                } else {
#pragma unroll
                    for (int i=0;i<NL;i++) T.v[i] *= (1.0f - R.v[i])/(1.0f - Pr);
                    nd = d;                            // a film does not refract
                }
                // Deliberately leave prev_delta/prev_pdf/prev_p alone: a film
                // does not bend the ray, so for MIS the path still looks as
                // though it came straight from the previous real vertex.
                o = add(p, mul(fn, dot3(nd, fn) > 0.0f ? RAY_EPS : -RAY_EPS));
                d = nd;
                continue;
            }

            // ---- non-delta: NEE + BSDF sampling with MIS -----------------
            V3 tt, bb; onb(fn, tt, bb);
            V3 wo = to_local(neg(d), tt, bb, fn);
            if (wo.z <= 0.0f) break;
            float alpha = fmaxf(m.rough, (m.type == M_PLASTIC) ? 0.02f : 0.0f);

            float4 gB = sample4(smp, (unsigned int)(3 + 2*depth));
            if (mis_mode != 1 && nlight > 0) {
                float w[MAXLI];
                float total = light_total(p, w, nlight);
                if (total > 0.0f) {
                    float pick = gB.x * total;
                    int k = 0;
                    while (k < nlight-1 && w[k] < pick) k++;
                    float ppick = (w[k] - (k ? w[k-1] : 0.0f)) / total;
                    int li = c_light[k];
                    float4 ls = c_sph[li];
                    V3 tol = sub(v3(ls.x, ls.y, ls.z), p);
                    float d2 = dot3(tol, tol), dist = sqrtf(d2);
                    float lr2 = ls.w*ls.w;
                    if (d2 > lr2*1.0002f) {
                        float cmax = sqrtf(fmaxf(0.0f, 1.0f - lr2/d2));
                        float ct = 1.0f - gB.y*(1.0f - cmax);
                        float st = sqrtf(fmaxf(0.0f, 1.0f - ct*ct));
                        float ph = 2.0f*PI*gB.z;
                        V3 wl = mul(tol, 1.0f/dist), lt, lb2;
                        onb(wl, lt, lb2);
                        V3 ldir = add(add(mul(lt, __cosf(ph)*st), mul(lb2, __sinf(ph)*st)),
                                      mul(wl, ct));
                        V3 wi = to_local(ldir, tt, bb, fn);
                        if (wi.z > 0.0f) {
                            float bpdf;
                            Spec fc = bsdf_eval(m.type, alb, alpha, m.ior, wo, wi, bpdf);
                            float lpdf = ppick / (2.0f*PI*fmaxf(1.0f - cmax, 1e-9f));
                            if (lpdf > 0.0f && sp_max(fc) > 0.0f) {
                                Spec tr;
                                V3 so = add(p, mul(fn, RAY_EPS));
                                if (shadow(so, ldir, li, sc, mats, tr)) {
                                    const Material lm = mats[c_smat[li]];
                                    Spec Le = rgb2emit(sc, lm.cr, lm.cg, lm.cb);
                                    float wmis = (mis_mode == 0)
                                               ? power_heuristic(lpdf, bpdf) : 1.0f;
#pragma unroll
                                    for (int i=0;i<NL;i++)
                                        L.v[i] += T.v[i]*fc.v[i]*tr.v[i]*Le.v[i]*wmis/lpdf;
                                }
                            }
                        }
                    }
                }
            }

            float4 gA = sample4(smp, (unsigned int)(2 + 2*depth));
            BSample bs;
            if (!bsdf_sample(m.type, alb, alpha, m.ior, wo, gA.x, gA.y, gA.z, bs)) break;
            T = sp_mul(T, bs.w);
            prev_delta = bs.delta; prev_pdf = bs.pdf; prev_p = p;
            nd = to_world(bs.wi, tt, bb, fn);
            o = add(p, mul(fn, RAY_EPS));
            d = nd;

            // ---- Russian roulette ---------------------------------------
            if (depth >= 4) {
                float q = fminf(0.98f, fmaxf(0.04f, sp_max(T)));
                if (gA.w >= q) break;
                T = sp_scale(T, 1.0f/q);
            }
            if (sp_max(T) <= 0.0f) break;
        }

        // --- spectral estimate -> linear sRGB ---------------------------
        float cr = 0.0f, cg = 0.0f, cb = 0.0f;
        const float k = LSPAN / (float)NL;
#pragma unroll
        for (int i = 0; i < NL; i++) {
            cr += L.v[i]*sc.Rr[i]; cg += L.v[i]*sc.Rg[i]; cb += L.v[i]*sc.Rb[i];
        }
        cr *= k; cg *= k; cb *= k;
        if (!isfinite(cr)) cr = 0.0f;
        if (!isfinite(cg)) cg = 0.0f;
        if (!isfinite(cb)) cb = 0.0f;
        if (clamp_val > 0.0f) {
            float lum = 0.2126f*cr + 0.7152f*cg + 0.0722f*cb;
            if (lum > clamp_val) { float f = clamp_val/lum; cr*=f; cg*=f; cb*=f; }
        }
        sr += cr; sg += cg; sb += cb;
        float lum = 0.2126f*cr + 0.7152f*cg + 0.0722f*cb;
        sl2 += (double)lum*(double)lum;
    }

    accum[pid*4+0] += sr; accum[pid*4+1] += sg;
    accum[pid*4+2] += sb; accum[pid*4+3] += sl2;
    counts[pid] = base + spp;
}

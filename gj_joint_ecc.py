import sys
import glob
import numpy as np
import h5py
import batman
import emcee
from scipy.optimize import least_squares
import joint_fit as J

cfg = J.PLANETS["GJ 3929 b"]
H5 = "eureka/gj_e{e:02d}/S3/*/ap4_bg12_32/S3_gj_e{e:02d}_*SpecData.h5"
TRIM = 800
ECLIPSES = [int(a) for a in sys.argv[1:]] or [1, 2, 3]
SECOSW_INIT = -0.0955
N_STEPS = 9000
N_BURN = 3500
N_THIN = 10
NW_FACTOR = 3
OUT_NPZ = "gj_joint_ecc_chain.npz"

P = cfg["period"]
dur0 = P / np.pi * np.arcsin(1.0 / cfg["a_rs"])
bp = batman.TransitParams()
bp.t0 = 0.0; bp.per = P; bp.rp = cfg["rp"]; bp.a = cfg["a_rs"]; bp.inc = cfg["inc"]
bp.ecc = 0.0; bp.w = 90.0; bp.u = []; bp.limb_dark = "uniform"
bp.fp = 1e-3; bp.t_secondary = 0.0
UG = np.linspace(-1.5 * dur0, 1.5 * dur0, 6001)
GG = (batman.TransitModel(bp, UG, transittype="secondary").light_curve(bp) - 1.0) / bp.fp


def load(e):
    h5 = glob.glob(H5.format(e=e))[0]
    with h5py.File(h5, "r") as f:
        t = np.array(f["time"]); flux = np.array(f["aplev"])
        cx = np.array(f["centroid_x"]); cy = np.array(f["centroid_y"])
    if t[0] < 2400000:
        t = t + 2400000.5
    m = np.isfinite(t) & np.isfinite(flux) & np.isfinite(cx) & np.isfinite(cy)
    t, flux, cx, cy = t[m], flux[m], cx[m], cy[m]
    i = np.argsort(t); t, flux, cx, cy = t[i], flux[i], cx[i], cy[i]
    t, flux, cx, cy = t[TRIM:], flux[TRIM:], cx[TRIM:], cy[TRIM:]
    rel = flux / np.nanmedian(flux)
    d = np.diff(rel); sd = 1.4826 * np.median(np.abs(d - np.median(d)))
    keep = np.ones(len(rel), bool); keep[1:] &= np.abs(d) <= 6 * sd
    t, rel, cx, cy = t[keep], rel[keep], cx[keep], cy[keep]
    v = J.Visit(t, rel, cfg)
    return dict(name=f"{e}", t=t, f=rel, tsec0=v.t_secondary, tt0=t[0], td=t - t[0],
                cx=cx - np.median(cx), cy=cy - np.median(cy))


def orbit(secosw, sesinw):
    e = secosw ** 2 + sesinw ** 2
    if e >= 1.0:
        return None
    sq = np.sqrt(e) if e > 0 else 0.0
    dt = P * (2.0 / np.pi) * sq * secosw
    wdur = np.sqrt(max(1.0 - e ** 2, 1e-6)) / (1.0 + sq * sesinw)
    return dt, wdur


class Model:
    def __init__(self, visits):
        self.v = visits
        self.nv = len(visits)
        self.npv = 7
        self.ndim = 3 + self.npv * self.nv

    def log_prob(self, x):
        fp, secosw, sesinw = x[0], x[1], 0.0
        if not (1e-7 <= fp <= 1e-2) or abs(secosw) > 1:
            return -np.inf
        orb = orbit(secosw, sesinw)
        if orb is None:
            return -np.inf
        dt, wdur = orb
        if abs(dt) > 0.045:
            return -np.inf
        ll = 0.0
        for i, d in enumerate(self.v):
            C, A, tau, s, bx, by, jit = x[3 + i * self.npv:3 + (i + 1) * self.npv]
            if not (0.95 <= C <= 1.05 and -0.05 <= A <= 0.05 and 1e-4 <= tau <= 10.0
                    and -0.1 <= s <= 0.1 and abs(bx) < 1 and abs(by) < 1 and 1e-7 <= jit <= 1e-1):
                return -np.inf
            tc = d["tsec0"] + dt
            E = np.interp((d["t"] - tc) / wdur, UG, GG, left=1.0, right=1.0)
            base = C * (1.0 - A * np.exp(-d["td"] / tau)) * (1.0 - s * d["td"]) \
                * (1.0 + bx * d["cx"] + by * d["cy"])
            r = d["f"] - base * (1.0 + fp * E)
            ll += -0.5 * np.sum((r / jit) ** 2) - len(r) * np.log(jit)
        return ll

    def seed_sys(self, fp, secosw, sesinw):
        dt, wdur = orbit(secosw, sesinw)
        out = []
        for d in self.v:
            tc = d["tsec0"] + dt
            E = np.interp((d["t"] - tc) / wdur, UG, GG, left=1.0, right=1.0)
            def resid(y):
                C, A, ltau, s, bx, by = y
                base = C * (1.0 - A * np.exp(-d["td"] / 10 ** ltau)) * (1.0 - s * d["td"]) \
                    * (1.0 + bx * d["cx"] + by * d["cy"])
                return d["f"] - base * (1.0 + fp * E)
            res = least_squares(resid, [1.0, 0.001, np.log10(0.2), 0.0, 0.0, 0.0],
                                bounds=([0.95, -0.05, -4, -0.1, -1, -1],
                                        [1.05, 0.05, 1, 0.1, 1, 1]))
            C, A, ltau, s, bx, by = res.x
            out += [C, A, 10 ** ltau, s, bx, by, np.clip(np.std(res.fun), 1e-7, 1e-1)]
        return out


def main():
    visits = [load(e) for e in ECLIPSES]
    m = Model(visits)
    fp0 = 142e-6
    x0 = np.array([fp0, SECOSW_INIT, 0.0] + m.seed_sys(fp0, SECOSW_INIT, 0.0))
    print(f"init logprob: {m.log_prob(x0):.1f}", flush=True)

    nw = NW_FACTOR * m.ndim
    p0 = np.tile(x0, (nw, 1))
    p0[:, 0] = np.abs(fp0 + 30e-6 * np.random.randn(nw)) + 1e-6
    p0[:, 1] = SECOSW_INIT + 0.008 * np.random.randn(nw)
    p0[:, 2] = 1e-6 * np.random.randn(nw)
    for k in range(3, m.ndim):
        p0[:, k] = x0[k] * (1 + 1e-3 * np.random.randn(nw)) + 1e-7 * np.random.randn(nw)
    for i in range(nw):
        tries = 0
        while not np.isfinite(m.log_prob(p0[i])):
            p0[i] = p0[np.random.randint(nw)] * (1 + 1e-4 * np.random.randn(m.ndim))
            tries += 1
            if tries > 200:
                p0[i] = x0
                break

    s = emcee.EnsembleSampler(nw, m.ndim, m.log_prob)
    s.run_mcmc(p0, N_STEPS, progress=True)
    flat = s.get_chain(discard=N_BURN, thin=N_THIN, flat=True)
    lp = s.get_log_prob(discard=N_BURN, thin=N_THIN, flat=True)

    dep = 1e6 * flat[:, 0]
    sc = flat[:, 1]; ss = np.zeros_like(sc)
    e = sc ** 2 + ss ** 2
    dt_min = P * (2 / np.pi) * np.sqrt(e) * sc * 24 * 60

    print(f"\nGJ 3929 b joint fit eclipses {ECLIPSES} (shared depth, secosw, sesinw)")
    print(f"  depth_ecl: {np.median(dep):.1f} ppm  68% [{np.percentile(dep,16):.1f}, {np.percentile(dep,84):.1f}]")
    print(f"  secosw: {np.median(sc):+.4f} [{np.percentile(sc,16):+.4f}, {np.percentile(sc,84):+.4f}]")
    print(f"  sesinw: {np.median(ss):+.4f} [{np.percentile(ss,16):+.4f}, {np.percentile(ss,84):+.4f}]")
    print(f"  implied dt: {np.median(dt_min):+.1f} min [{np.percentile(dt_min,16):+.1f}, {np.percentile(dt_min,84):+.1f}]")
    print(f"  PAPER: 160 +/- 26 ppm")
    np.savez(OUT_NPZ, flat=flat, logprob=lp, dep=dep, secosw=sc, sesinw=ss, dt_min=dt_min,
             ndim=m.ndim, nv=m.nv, eclipses=ECLIPSES)


if __name__ == "__main__":
    main()

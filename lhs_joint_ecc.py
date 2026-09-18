import sys
import glob
import numpy as np
import batman
import emcee
from scipy.optimize import least_squares
import joint_fit as J

cfg = J.PLANETS["LHS 1140 b"]
TS_GLOB = "ts/lhs1140b_eclipse_*.txt"
SECOSW_INIT = 0.076
DT_SPREAD_H = 0.35
TRIM_H = float(sys.argv[1]) if len(sys.argv) > 1 else 0.0
FREE_SESINW = len(sys.argv) > 2 and sys.argv[2] == "1" 
N_STEPS = 9000
N_BURN = 3500
N_THIN = 10
NW_FACTOR = 3
OUT_NPZ = f"lhs_joint_ecc_chain_t{TRIM_H:.0f}_s{int(FREE_SESINW)}.npz"

P = cfg["period"]
dur0 = P / np.pi * np.arcsin(1.0 / cfg["a_rs"])
bp = batman.TransitParams()
bp.t0 = 0.0; bp.per = P; bp.rp = cfg["rp"]; bp.a = cfg["a_rs"]; bp.inc = cfg["inc"]
bp.ecc = 0.0; bp.w = 90.0; bp.u = []; bp.limb_dark = "uniform"
bp.fp = 1e-3; bp.t_secondary = 0.0
UG = np.linspace(-1.5 * dur0, 1.5 * dur0, 6001)
GG = (batman.TransitModel(bp, UG, transittype="secondary").light_curve(bp) - 1.0) / bp.fp


def load_visits():
    visits = []
    for p in sorted(glob.glob(TS_GLOB)):
        t, f = np.loadtxt(p, unpack=True, usecols=(0, 1))
        f = J.correct_outliers(f)
        keep = (t - t[0]) * 24 >= TRIM_H
        t, f = t[keep], f[keep]
        f = f / np.median(f)
        v = J.Visit(t, f, cfg)
        visits.append(dict(name=p.split("_")[-1].split(".")[0], t=t, f=f,
                           tsec0=v.t_secondary, tt0=t[0], td=t - t[0]))
    return visits


def orbit(secosw, sesinw):
    e = secosw ** 2 + sesinw ** 2
    if e >= 1.0:
        return None
    sq = np.sqrt(e) if e > 0 else 0.0
    ecosw = sq * secosw
    esinw = sq * sesinw
    dt = P * (2.0 / np.pi) * ecosw
    wdur = np.sqrt(max(1.0 - e ** 2, 1e-6)) / (1.0 + esinw)
    return dt, wdur


class Model:
    def __init__(self, visits):
        self.v = visits
        self.nv = len(visits)
        self.npv = 5
        self.ndim = 3 + self.npv * self.nv

    def log_prob(self, x):
        fp, secosw = x[0], x[1]
        sesinw = x[2] if FREE_SESINW else 0.0
        if not (1e-7 <= fp <= 1e-2) or not (-0.075 <= secosw <= 0.092) or abs(sesinw) > 1:
            return -np.inf
        orb = orbit(secosw, sesinw)
        if orb is None:
            return -np.inf
        dt, wdur = orb
        ll = 0.0
        for i, d in enumerate(self.v):
            C, A, tau, s, jit = x[3 + i * self.npv:3 + (i + 1) * self.npv]
            if not (0.95 <= C <= 1.05 and -0.05 <= A <= 0.05 and 1e-4 <= tau <= 10.0
                    and -0.1 <= s <= 0.1 and 1e-7 <= jit <= 1e-1):
                return -np.inf
            tc = d["tsec0"] + dt
            E = np.interp((d["t"] - tc) / wdur, UG, GG, left=1.0, right=1.0)
            base = C * (1.0 - A * np.exp(-d["td"] / tau)) * (1.0 - s * d["td"])
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
                C, A, ltau, s = y
                base = C * (1.0 - A * np.exp(-d["td"] / 10 ** ltau)) * (1.0 - s * d["td"])
                return d["f"] - base * (1.0 + fp * E)
            res = least_squares(resid, [1.0, 0.001, np.log10(0.2), 0.0],
                                bounds=([0.95, -0.05, -4, -0.1], [1.05, 0.05, 1, 0.1]))
            C, A, ltau, s = res.x
            jit = np.std(res.fun)
            out += [C, A, 10 ** ltau, s, np.clip(jit, 1e-7, 1e-1)]
        return out


def main():
    visits = load_visits()
    m = Model(visits)
    fp0 = 36e-6
    x0 = np.array([fp0, SECOSW_INIT, 0.0] + m.seed_sys(fp0, SECOSW_INIT, 0.0))
    print(f"init logprob: {m.log_prob(x0):.1f}", flush=True)

    nw = NW_FACTOR * m.ndim
    p0 = np.tile(x0, (nw, 1))
    p0[:, 0] = np.abs(fp0 + 20e-6 * np.random.randn(nw)) + 1e-6
    sc_spread = np.sqrt(np.abs(DT_SPREAD_H / 24 * np.pi / (2 * P)))
    p0[:, 1] = SECOSW_INIT + 0.006 * np.random.randn(nw)
    p0[:, 2] = (0.05 if FREE_SESINW else 1e-6) * np.random.randn(nw)
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
    s.run_mcmc(p0, N_STEPS, progress=True, thin_by=1)
    flat = s.get_chain(discard=N_BURN, thin=N_THIN, flat=True)
    lp = s.get_log_prob(discard=N_BURN, thin=N_THIN, flat=True)

    dep = 1e6 * flat[:, 0]
    sc = flat[:, 1]
    ss = flat[:, 2] if FREE_SESINW else np.zeros_like(sc)
    e = sc ** 2 + ss ** 2
    ecosw = np.sqrt(e) * sc
    dt_h = P * (2 / np.pi) * ecosw * 24

    print(f"\nLHS 1140 b joint 9-visit fit (shared depth, secosw, sesinw)")
    print(f"  depth_ecl: {np.median(dep):.1f} ppm  68% [{np.percentile(dep,16):.1f}, {np.percentile(dep,84):.1f}]"
          f"  95% [{np.percentile(dep,2.5):.1f}, {np.percentile(dep,97.5):.1f}]")
    print(f"  secosw: {np.median(sc):+.4f} [{np.percentile(sc,16):+.4f}, {np.percentile(sc,84):+.4f}]")
    print(f"  sesinw: {np.median(ss):+.4f} [{np.percentile(ss,16):+.4f}, {np.percentile(ss,84):+.4f}]")
    print(f"  implied dt: {np.median(dt_h):+.2f} h [{np.percentile(dt_h,16):+.2f}, {np.percentile(dt_h,84):+.2f}]")
    print(f"  P(depth>10ppm): {np.mean(dep>10):.3f}")
    np.savez(OUT_NPZ, flat=flat, logprob=lp, dep=dep, secosw=sc, sesinw=ss, dt_h=dt_h,
             ndim=m.ndim, nv=m.nv)


if __name__ == "__main__":
    main()

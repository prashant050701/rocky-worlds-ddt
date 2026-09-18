import glob
import sys
import numpy as np
import h5py
import batman
import emcee
from scipy.optimize import least_squares
import joint_fit as J

AP = int(sys.argv[1])
ECLIPSES = [int(c) for c in sys.argv[2]]
N_STEPS = int(sys.argv[3]) if len(sys.argv) > 3 else 2500
N_BURN = N_STEPS * 2 // 5
N_THIN = 5
TRIM = 800
T_TILT_BJD = 2461082.07162
SECOSW_INIT = -0.0959
OUT_NPZ = "gj_de_ap%d_e%s.npz" % (AP, sys.argv[2])

cfg = J.PLANETS["GJ 3929 b"]
P = cfg["period"]
dur0 = P / np.pi * np.arcsin(1.0 / cfg["a_rs"])
bp = batman.TransitParams()
bp.t0 = 0.0; bp.per = P; bp.rp = cfg["rp"]; bp.a = cfg["a_rs"]; bp.inc = cfg["inc"]
bp.ecc = 0.0; bp.w = 90.0; bp.u = []; bp.limb_dark = "uniform"
bp.fp = 1e-3; bp.t_secondary = 0.0
UG = np.linspace(-1.5 * dur0, 1.5 * dur0, 6001)
GG = (batman.TransitModel(bp, UG, transittype="secondary").light_curve(bp) - 1.0) / bp.fp


def find_h5(e, ap):
    pats = ["eureka/gj_e%02d/S3_ap%d/*/ap%d_bg12_32/*SpecData.h5" % (e, ap, ap),
            "eureka/gj_e%02d/S3/*/ap%d_bg12_32/*SpecData.h5" % (e, ap)]
    for p in pats:
        hits = glob.glob(p)
        if hits:
            return sorted(hits)[-1]
    raise FileNotFoundError("no S3 product for e%02d ap%d" % (e, ap))


def load(e, ap):
    with h5py.File(find_h5(e, ap), "r") as f:
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
    step = (t > T_TILT_BJD).astype(float) if e == 4 else None
    tpost = np.clip(t - T_TILT_BJD, 0, None) if e == 4 else None
    return dict(e=e, t=t, f=rel, tsec0=v.t_secondary, td=t - t[0],
                cx=cx - np.median(cx), cy=cy - np.median(cy), step=step, tpost=tpost)


def orbit(secosw):
    e = secosw ** 2
    return (P * (2.0 / np.pi) * np.sqrt(e) * secosw if e > 0 else 0.0), 1.0


class Model:
    def __init__(self, visits):
        self.v = visits
        self.npv = [10 if d["step"] is not None else 7 for d in visits]
        self.off = np.cumsum([2] + self.npv[:-1])
        self.ndim = 2 + sum(self.npv)

    def log_prob(self, x):
        fp, secosw = x[0], x[1]
        if not (1e-7 <= fp <= 1e-2) or abs(secosw) > 1:
            return -np.inf
        dt, wdur = orbit(secosw)
        if abs(dt) > 0.045:
            return -np.inf
        ll = 0.0
        for d, o, npv in zip(self.v, self.off, self.npv):
            C, A, tau, s, bx, by, jit = x[o:o + 7]
            if not (0.95 <= C <= 1.05 and -0.05 <= A <= 0.05 and 1e-4 <= tau <= 10.0
                    and -0.1 <= s <= 0.1 and abs(bx) < 1 and abs(by) < 1 and 1e-7 <= jit <= 1e-1):
                return -np.inf
            base = C * (1.0 - A * np.exp(-d["td"] / tau)) * (1.0 - s * d["td"]) \
                * (1.0 + bx * d["cx"] + by * d["cy"])
            if npv == 10:
                st, A2, tau2 = x[o + 7], x[o + 8], x[o + 9]
                if abs(st) > 0.05 or abs(A2) > 0.02 or not (0.002 <= tau2 <= 0.02):
                    return -np.inf
                base = base * (1.0 + st * d["step"] + A2 * d["step"] * np.exp(-d["tpost"] / tau2))
            tc = d["tsec0"] + dt
            E = np.interp(d["t"] - tc, UG, GG, left=1.0, right=1.0)
            r = d["f"] - base * (1.0 + fp * E)
            ll += -0.5 * np.sum((r / jit) ** 2) - len(r) * np.log(jit)
        return ll

    def seed_sys(self, fp, secosw):
        dt, wdur = orbit(secosw)
        out = []
        for d, npv in zip(self.v, self.npv):
            tc = d["tsec0"] + dt
            E = np.interp(d["t"] - tc, UG, GG, left=1.0, right=1.0)
            has_step = npv == 10

            def resid(y):
                C, A, ltau, s, bx, by = y[:6]
                base = C * (1.0 - A * np.exp(-d["td"] / 10 ** ltau)) * (1.0 - s * d["td"]) \
                    * (1.0 + bx * d["cx"] + by * d["cy"])
                if has_step:
                    base = base * (1.0 + y[6] * d["step"] + y[7] * d["step"] * np.exp(-d["tpost"] / y[8]))
                return d["f"] - base * (1.0 + fp * E)

            y0 = [1.0, 0.001, np.log10(0.2), 0.0, 0.0, 0.0] + ([0.0, 0.0, 0.008] if has_step else [])
            lo = [0.95, -0.05, -4, -0.1, -1, -1] + ([-0.05, -0.02, 0.002] if has_step else [])
            hi = [1.05, 0.05, 1, 0.1, 1, 1] + ([0.05, 0.02, 0.02] if has_step else [])
            res = least_squares(resid, y0, bounds=(lo, hi))
            y = res.x
            blk = [y[0], y[1], 10 ** y[2], y[3], y[4], y[5], np.clip(np.std(res.fun), 1e-7, 1e-1)]
            if has_step:
                blk += [y[6], y[7], y[8]]
            out += blk
        return out


def main():
    visits = [load(e, AP) for e in ECLIPSES]
    mad = [1e6 * 1.4826 * np.median(np.abs(np.diff(d["f"]) - np.median(np.diff(d["f"])))) / np.sqrt(2)
           for d in visits]
    m = Model(visits)
    fp0 = 140e-6
    x0 = np.array([fp0, SECOSW_INIT] + m.seed_sys(fp0, SECOSW_INIT))
    nw = 3 * m.ndim
    rng = np.random.default_rng(20260828 + AP)
    p0 = np.tile(x0, (nw, 1))
    p0[:, 0] = np.abs(fp0 + 30e-6 * rng.standard_normal(nw)) + 1e-6
    p0[:, 1] = SECOSW_INIT + 0.005 * rng.standard_normal(nw)
    for k in range(2, m.ndim):
        p0[:, k] = x0[k] * (1 + 1e-3 * rng.standard_normal(nw)) + 1e-7 * rng.standard_normal(nw)
    for i in range(nw):
        tries = 0
        while not np.isfinite(m.log_prob(p0[i])):
            p0[i] = p0[rng.integers(nw)] * (1 + 1e-4 * rng.standard_normal(m.ndim))
            tries += 1
            if tries > 200:
                p0[i] = x0
                break
    moves = [(emcee.moves.DEMove(), 0.8), (emcee.moves.DESnookerMove(), 0.2)]
    s = emcee.EnsembleSampler(nw, m.ndim, m.log_prob, moves=moves)
    s.run_mcmc(p0, N_STEPS, progress=False)
    try:
        tau = float(s.get_autocorr_time(discard=N_BURN, quiet=True)[0])
    except Exception:
        tau = float("nan")
    ntau = (N_STEPS - N_BURN) / tau if tau == tau else float("nan")
    acc = float(np.mean(s.acceptance_fraction))
    print("LADDER autocorr tau_depth=%.1f retained_steps=%d n_tau=%.1f acc=%.3f"
          % (tau, N_STEPS - N_BURN, ntau, acc), flush=True)
    flat = s.get_chain(discard=N_BURN, thin=N_THIN, flat=True)
    dep = 1e6 * flat[:, 0]
    q = np.percentile(dep, [16, 50, 84])
    print("LADDER ap=%d eclipses=%s ptp_MAD=%s" % (AP, ECLIPSES, [round(x) for x in mad]), flush=True)
    print("LADDER depth %.2f -%.2f +%.2f  mean %.2f sd %.2f  nsamp %d"
          % (q[1], q[1] - q[0], q[2] - q[1], dep.mean(), dep.std(), len(dep)), flush=True)
    np.savez(OUT_NPZ, dep=dep, secosw=flat[:, 1], ap=AP, eclipses=ECLIPSES, mad=mad,
             tau=tau, ntau=ntau, acceptance=acc)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()

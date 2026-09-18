import os
import glob
import copy
import numpy as np
from scipy.optimize import least_squares
from scipy.ndimage import median_filter
import batman
import emcee
import rocky_worlds_data_challenge as rw

WORK_DIR = os.path.expanduser("~/PycharmProjects/DDTChallenge/fulldata")
TS_DIR = os.path.join(WORK_DIR, "ts")
OUT_DIR = os.path.join(WORK_DIR, "out")
OUTPUT_ZIP = os.path.join(OUT_DIR, "submission_full.zip")
CHAIN_DIR = os.path.join(OUT_DIR, "chains")

N_WALKERS_FACTOR = 4
N_WALKERS_MIN = 100
N_STEPS = 8000
N_BURN = 3000
N_THIN = 10
RANDOM_SEED = 42

DEPTH_MAX_PPM = 1000.0
TAU_MAX_FRAC = 0.25

SUBMITTER_EMAIL = "divyansh@doktorant.umk.pl"
SECONDARY_NAME = "Prof Andrzej Niedzielski"
SECONDARY_EMAIL = "aniedzi@umk.pl"

TEAM = {
    "group_name": "Divyansh Srivastava",
    "submitter_name": "Divyansh Srivastava",
    "members": "Divyansh Srivastava",
}

PLANETS = {
    "GJ 3929 b": dict(tag="gj3929b", display="GJ 3929 b", t0=2458956.3962,
                      period=2.6162745, a_rs=17.56, inc=89.65, rp=0.03348),
    "LHS 1140 b": dict(tag="lhs1140b", display="LHS 1140 b", t0=2458399.9300,
                       period=24.73723, a_rs=95.3, inc=89.86, rp=0.07390),
}

SYS_PARAMS = ["C", "exp_amplitude", "exp_timescale", "slope", "jitter"]


def correct_outliers(data, window=51, nsigma=5):
    mf = median_filter(data, window)
    residuals = data - mf
    mad = np.median(np.abs(residuals - np.nanmedian(residuals)))
    sigma = 1.4 * mad
    idx = np.where(np.abs(residuals) > nsigma * sigma)[0]
    corrected = np.copy(data)
    corrected[idx] = mf[idx]
    return corrected


def exponential_function(t, exp_amplitude, exp_timescale, slope):
    return (1.0 - exp_amplitude * np.exp(-(t - t[0]) / exp_timescale)) * (
        1.0 - slope * (t - t[0])
    )


class Visit:
    def __init__(self, time, flux, cfg):
        idx = np.argsort(time)
        self.t = np.asarray(time)[idx]
        self.f = np.asarray(flux)[idx]
        self.cfg = cfg
        tmid = np.median(self.t)
        self.ncycle = int(np.round((tmid - cfg["t0"] - 0.5 * cfg["period"]) / cfg["period"]))
        self.t_secondary = cfg["t0"] + (self.ncycle + 0.5) * cfg["period"]
        self.visit = self.t.max() - self.t.min()
        self.tau_max = TAU_MAX_FRAC * self.visit
        self.sys_lower = np.array([0.95, -0.05, 1e-4, -0.1, 1e-7])
        self.sys_upper = np.array([1.05, 0.05, self.tau_max, 0.1, 1e-1])

    def eclipse_model(self, fp):
        params = batman.TransitParams()
        params.t0 = self.cfg["t0"]
        params.per = self.cfg["period"]
        params.rp = self.cfg["rp"]
        params.a = self.cfg["a_rs"]
        params.inc = self.cfg["inc"]
        params.ecc = 0.0
        params.w = 90.0
        params.u = []
        params.limb_dark = "uniform"
        params.fp = max(fp, 1e-12)
        params.t_secondary = self.t_secondary
        m = batman.TransitModel(params, self.t, transittype="secondary")
        return m.light_curve(params)

    def components(self, fp, sysp):
        astro = self.eclipse_model(fp)
        baseline = exponential_function(self.t, sysp[1], sysp[2], sysp[3])
        noise = sysp[0] * baseline
        return astro, noise, astro * noise

    def single_fit(self):
        def resid(xf):
            fp = xf[0]
            sysp = np.array([xf[1], xf[2], xf[3], xf[4], 1.0])
            _, _, full = self.components(fp, sysp)
            return self.f - full
        x0 = np.array([100e-6, 1.0, 0.001, min(0.05, 0.5 * self.tau_max), 0.0])
        lo = np.array([0.0, self.sys_lower[0], self.sys_lower[1], self.sys_lower[2], self.sys_lower[3]])
        hi = np.array([DEPTH_MAX_PPM * 1e-6, self.sys_upper[0], self.sys_upper[1], self.sys_upper[2], self.sys_upper[3]])
        res = least_squares(resid, x0, bounds=(lo, hi), method="trf", max_nfev=5000)
        fp = res.x[0]
        sysp = np.array([res.x[1], res.x[2], res.x[3], res.x[4], 1.0])
        _, _, full = self.components(fp, sysp)
        jitter = np.clip(np.std(self.f - full), self.sys_lower[4], self.sys_upper[4])
        return fp, np.array([res.x[1], res.x[2], res.x[3], res.x[4], jitter])


class JointModel:
    def __init__(self, visits):
        self.visits = visits
        self.nv = len(visits)
        self.ndim = 1 + 5 * self.nv
        self.lower = np.empty(self.ndim)
        self.upper = np.empty(self.ndim)
        self.lower[0] = 0.0
        self.upper[0] = DEPTH_MAX_PPM * 1e-6
        for i, v in enumerate(visits):
            o = 1 + 5 * i
            self.lower[o:o + 5] = v.sys_lower
            self.upper[o:o + 5] = v.sys_upper

    def log_prior(self, x):
        if np.any(x < self.lower) or np.any(x > self.upper):
            return -np.inf
        return 0.0

    def log_likelihood(self, x):
        fp = x[0]
        total = 0.0
        for i, v in enumerate(self.visits):
            o = 1 + 5 * i
            sysp = x[o:o + 5]
            _, _, full = v.components(fp, sysp)
            jit = sysp[4]
            r = (v.f - full) / jit
            total += -0.5 * np.sum(r**2 + np.log(2.0 * np.pi * jit**2))
        return total

    def log_probability(self, x):
        lp = self.log_prior(x)
        if not np.isfinite(lp):
            return -np.inf
        return lp + self.log_likelihood(x)

    def seed(self):
        fps = []
        x = np.empty(self.ndim)
        for i, v in enumerate(self.visits):
            fp_i, sysp_i = v.single_fit()
            fps.append(fp_i)
            x[1 + 5 * i:1 + 5 * i + 5] = sysp_i
        x[0] = np.clip(np.median(fps), 1e-6, self.upper[0])
        return x, np.array(fps)


def run_emcee(model, x_best):
    np.random.seed(RANDOM_SEED)
    ndim = model.ndim
    nwalkers = max(N_WALKERS_MIN, N_WALKERS_FACTOR * ndim)
    if nwalkers % 2:
        nwalkers += 1
    scale = np.abs(x_best) * 0.02 + 1e-6
    scale[0] = 0.1 * max(x_best[0], 1e-6)
    p0 = x_best + scale * np.random.randn(nwalkers, ndim)
    p0 = np.clip(p0, model.lower + 1e-9, model.upper - 1e-9)
    for i in range(nwalkers):
        tries = 0
        while not np.isfinite(model.log_prior(p0[i])):
            p0[i] = np.clip(x_best + scale * np.random.randn(ndim), model.lower + 1e-9, model.upper - 1e-9)
            tries += 1
            if tries > 2000:
                raise RuntimeError("walker init failed")
    sampler = emcee.EnsembleSampler(nwalkers, ndim, model.log_probability)
    sampler.run_mcmc(p0, N_STEPS, progress=True)
    try:
        tau_max = float(np.nanmax(sampler.get_autocorr_time(tol=0)))
    except Exception:
        tau_max = float("nan")
    flat = sampler.get_chain(discard=N_BURN, thin=N_THIN, flat=True)
    logprob = sampler.get_log_prob(discard=N_BURN, thin=N_THIN, flat=True)
    return flat, logprob, tau_max, nwalkers


def load_visits(cfg):
    paths = sorted(glob.glob(os.path.join(TS_DIR, f"{cfg['tag']}_eclipse_*.txt")))
    visits = []
    for p in paths:
        t, f = np.loadtxt(p, unpack=True, usecols=(0, 1))
        f = correct_outliers(f)
        visits.append(Visit(t, f, cfg))
    return visits, paths


def build_form(planet_name, cfg, nv, tau_max):
    d = copy.deepcopy(rw.Form.blank().dictionary)
    responses = {
        "01": planet_name, "02": TEAM["group_name"], "03": TEAM["submitter_name"],
        "04": SUBMITTER_EMAIL, "05": SECONDARY_NAME, "06": SECONDARY_EMAIL, "07": TEAM["members"],
        "12": "Reduction from the raw *_uncal ramps: a last-minus-first slope image per integration (second-to-last minus second group), following the Exoplanets 6 tutorial method, applied to all downloaded eclipses.",
        "13": "Circular aperture photometry (radius optimised to minimise point-to-point scatter: 4 px for GJ 3929 b, 5 px for LHS 1140 b) summed on the target after auto-centroiding the median slope image.",
        "17": "sigma_jitter (per-visit Gaussian jitter fit as the photometric uncertainty).",
        "18": "batman",
        "19": {"parameter": "depth_ecl", "prior_distribution_type": "uniform",
               "prior_parameters": f"lower=0 ppm, upper={DEPTH_MAX_PPM:.0f} ppm"},
        "20": "Depth restricted to non-negative values; negative (brightening) depths not explored.",
        "21": {"parameter": "inc", "prior_distribution_type": "fixed", "prior_parameters": f"value={cfg['inc']} deg"},
        "22": {"parameter": "Rp/Rs", "prior_distribution_type": "fixed", "prior_parameters": f"value={cfg['rp']}"},
        "23": {"parameter": "t_ecl", "prior_distribution_type": "fixed", "prior_parameters": "phase 0.5 per cycle (circular)"},
        "24": {"parameter": "ecosw", "prior_distribution_type": "fixed", "prior_parameters": "value=0"},
        "25": {"parameter": "esinw", "prior_distribution_type": "fixed", "prior_parameters": "value=0"},
        "26": {"parameter": "P", "prior_distribution_type": "fixed", "prior_parameters": f"value={cfg['period']} d; a/Rs={cfg['a_rs']}"},
        "27": f"Orbital period, a/Rs, inclination, planet radius and t0 fixed to literature values; circular orbit assumed. Mid-eclipse time per eclipse from its integer cycle number.",
        "28": "No separate light-travel-time correction; circular orbit, mid-eclipse fixed at phase 0.5 per cycle.",
        "31": "Per eclipse: a constant baseline C, an exponential ramp in time (timescale capped below the visit duration), and a linear slope in time.",
        "33": f"All {nv} eclipses of this target were fit jointly with a single shared eclipse depth and independent per-visit systematics.",
        "34": "Affine-invariant ensemble MCMC.",
        "35": "emcee (Foreman-Mackey et al. 2013), initialized at a per-visit least-squares fit.",
        "36": f"Discarded {N_BURN} burn-in steps and thinned by {N_THIN}; run length {N_STEPS} steps, max integrated autocorrelation time ~ {tau_max:.1f} steps.",
    }
    for k, v in responses.items():
        d[k]["response"] = v
    return d


def analyze(planet_name, cfg):
    visits, paths = load_visits(cfg)
    if len(visits) == 0:
        raise FileNotFoundError(f"no ts files for {cfg['tag']} in {TS_DIR}")
    model = JointModel(visits)
    x_best, fps = model.seed()
    flat, logprob, tau_max, nwalkers = run_emcee(model, x_best)

    depth_ppm = 1e6 * flat[:, 0]
    samples = depth_ppm.reshape(1, -1)

    imax = int(np.argmax(logprob))
    xmap = flat[imax]
    fp_map = xmap[0]

    all_t, all_f, all_ferr, all_astro, all_noise, all_full = [], [], [], [], [], []
    for i, v in enumerate(visits):
        o = 1 + 5 * i
        sysp = xmap[o:o + 5]
        astro, noise, full = v.components(fp_map, sysp)
        all_t.append(v.t)
        all_f.append(v.f)
        all_ferr.append(np.ones_like(v.f) * sysp[4])
        all_astro.append(astro)
        all_noise.append(noise)
        all_full.append(full)
    t = np.concatenate(all_t)
    order = np.argsort(t)
    photometry = dict(
        time=t[order],
        raw_flux=np.concatenate(all_f)[order],
        raw_flux_err=np.concatenate(all_ferr)[order],
        astro_model=np.concatenate(all_astro)[order],
        noise_model=np.concatenate(all_noise)[order],
        full_model=np.concatenate(all_full)[order],
    )

    os.makedirs(CHAIN_DIR, exist_ok=True)
    np.savez(os.path.join(CHAIN_DIR, f"chain_{cfg['tag']}.npz"),
             flat=flat, logprob=logprob, depth_ppm=depth_ppm,
             ndim=model.ndim, nv=model.nv, per_visit_fp=fps)

    print(f"--- {planet_name} ({model.nv} eclipses, ndim {model.ndim}, {nwalkers} walkers) ---")
    print(f"  per-visit lsq depths (ppm): {np.round(1e6*fps,1)}")
    print(f"  shared depth_ecl median: {np.median(depth_ppm):.2f} ppm  68% CI "
          f"[{np.percentile(depth_ppm,16):.2f}, {np.percentile(depth_ppm,84):.2f}]  ({len(depth_ppm)} samples)")
    print(f"  depth_ecl MAP: {1e6*fp_map:.2f} ppm   tau_max {tau_max:.0f}")

    posterior = rw.Posterior(samples=samples, parameter_keys=["depth_ecl"])
    posterior.validate()
    photometry_obj = rw.Photometry(**photometry)
    photometry_obj.validate()
    form = rw.Form(dictionary=build_form(planet_name, cfg, model.nv, tau_max))
    form.validate()
    return posterior, photometry_obj, form


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    p_gj, ph_gj, f_gj = analyze("GJ 3929 b", PLANETS["GJ 3929 b"])
    p_lhs, ph_lhs, f_lhs = analyze("LHS 1140 b", PLANETS["LHS 1140 b"])
    results = rw.Results(
        posterior_GJ_3929_b=p_gj, photometry_GJ_3929_b=ph_gj, form_GJ_3929_b=f_gj,
        posterior_LHS_1140_b=p_lhs, photometry_LHS_1140_b=ph_lhs, form_LHS_1140_b=f_lhs,
    )
    results.to_submission(OUTPUT_ZIP, overwrite=True)
    print(f"Wrote {OUTPUT_ZIP}")


if __name__ == "__main__":
    main()

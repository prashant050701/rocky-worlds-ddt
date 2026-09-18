import os
import copy
import shutil
import numpy as np
import rocky_worlds_data_challenge as rw
import joint_fit as J
import lhs_joint_ecc as L
import gj_joint_ecc as G

LHS_NPZ = "lhs_slice_chain.npz"
GJ_NPZ = "gj_joint_ecc_chain.npz"
OUT_ZIP = os.path.join(J.OUT_DIR, "submission_ecc_v3slice.zip")
FINAL_ZIP = os.path.join(J.OUT_DIR, "submission.zip")
GJ_ECLIPSES = [1, 2, 3]


def map_photometry_lhs(flat, lp):
    visits = L.load_visits()
    x = flat[int(np.argmax(lp))]
    fp, secosw = x[0], x[1]
    dt, wdur = L.orbit(secosw, 0.0)
    T, F, FE, AS, NO, FU = [], [], [], [], [], []
    for i, d in enumerate(visits):
        C, A, tau, s, jit = x[3 + 5 * i:3 + 5 * (i + 1)]
        tc = d["tsec0"] + dt
        E = np.interp((d["t"] - tc) / wdur, L.UG, L.GG, left=1.0, right=1.0)
        base = C * (1.0 - A * np.exp(-d["td"] / tau)) * (1.0 - s * d["td"])
        astro = 1.0 + fp * E
        T.append(d["t"]); F.append(d["f"]); FE.append(np.full(len(d["t"]), jit))
        AS.append(astro); NO.append(base); FU.append(astro * base)
    t = np.concatenate(T); o = np.argsort(t)
    return dict(time=t[o], raw_flux=np.concatenate(F)[o], raw_flux_err=np.concatenate(FE)[o],
                astro_model=np.concatenate(AS)[o], noise_model=np.concatenate(NO)[o],
                full_model=np.concatenate(FU)[o])


def map_photometry_gj(flat, lp):
    visits = [G.load(e) for e in GJ_ECLIPSES]
    x = flat[int(np.argmax(lp))]
    fp, secosw = x[0], x[1]
    dt, wdur = G.orbit(secosw, 0.0)
    T, F, FE, AS, NO, FU = [], [], [], [], [], []
    for i, d in enumerate(visits):
        C, A, tau, s, bx, by, jit = x[3 + 7 * i:3 + 7 * (i + 1)]
        tc = d["tsec0"] + dt
        E = np.interp((d["t"] - tc) / wdur, G.UG, G.GG, left=1.0, right=1.0)
        base = C * (1.0 - A * np.exp(-d["td"] / tau)) * (1.0 - s * d["td"]) \
            * (1.0 + bx * d["cx"] + by * d["cy"])
        astro = 1.0 + fp * E
        T.append(d["t"]); F.append(d["f"]); FE.append(np.full(len(d["t"]), jit))
        AS.append(astro); NO.append(base); FU.append(astro * base)
    t = np.concatenate(T); o = np.argsort(t)
    return dict(time=t[o], raw_flux=np.concatenate(F)[o], raw_flux_err=np.concatenate(FE)[o],
                astro_model=np.concatenate(AS)[o], noise_model=np.concatenate(NO)[o],
                full_model=np.concatenate(FU)[o])


def form_common(planet, cfg, nv, extra33):
    d = J.build_form(planet, cfg, nv, 100.0)
    islhs = planet.startswith("LHS")
    d["12"]["response"] = "Eureka! Stages 1-3 from the raw *_uncal ramps (CRDS context jwst_1348.pmap)."
    d["13"]["response"] = ("Eureka! Stage 3 circular aperture photometry, 5 px radius, sky annulus 12-32 px."
                           if islhs else
                           "Eureka! Stage 3 circular aperture photometry, 4 px radius, sky annulus 12-32 px.")
    d["19"]["response"] = {"parameter": "depth_ecl", "prior_distribution_type": "uniform",
                           "prior_parameters": "lower=0.1 ppm, upper=1e4 ppm"}
    d["23"]["response"] = {"parameter": "t_ecl", "prior_distribution_type": "custom / other",
                           "prior_parameters": "from fitted ecosw: t0+(n+0.5)P+P(2/pi)ecosw"}
    d["24"]["response"] = {"parameter": "ecosw", "prior_distribution_type": "uniform",
                           "prior_parameters": ("secosw U(-0.075, 0.092), eclipse within observed windows"
                                                if islhs else
                                                "secosw U(-1,1), |t_ecl offset| < 65 min")}
    d["25"]["response"] = {"parameter": "esinw", "prior_distribution_type": "fixed",
                           "prior_parameters": "value=0"}
    d["27"]["response"] = "P, a/Rs, inc, Rp/Rs fixed to literature; secosw fitted, esinw fixed 0."
    d["28"]["response"] = "No LTT correction; eclipse time from fitted ecosw."
    d["31"]["response"] = "Per visit: constant, exponential ramp, linear slope, white-noise jitter." \
        if planet.startswith("LHS") else \
        "Per visit: constant, exponential ramp, linear slope, x/y centroid decorrelation, jitter."
    d["33"]["response"] = extra33
    d["36"]["response"] = "9000 steps, 3500 burn-in, thin 10; emcee ensemble."
    return d


def main():
    lz = np.load(LHS_NPZ)
    gz = np.load(GJ_NPZ)

    def summary(tag, z):
        dep, sc = z["dep"], z["secosw"]
        print(f"{tag}: depth {np.median(dep):.1f} [{np.percentile(dep,16):.1f}, {np.percentile(dep,84):.1f}] ppm, "
              f"secosw {np.median(sc):+.4f}, N={len(dep)}")

    summary("LHS 1140 b", lz)
    summary("GJ 3929 b ", gz)

    p_lhs = rw.Posterior(samples=np.vstack([lz["dep"], lz["secosw"], lz["sesinw"]]),
                         parameter_keys=["depth_ecl", "secosw", "sesinw"])
    p_lhs.validate()
    p_gj = rw.Posterior(samples=np.vstack([gz["dep"], gz["secosw"], gz["sesinw"]]),
                        parameter_keys=["depth_ecl", "secosw", "sesinw"])
    p_gj.validate()

    ph_lhs = rw.Photometry(**map_photometry_lhs(lz["flat"], lz["logprob"]))
    ph_lhs.validate()
    ph_gj = rw.Photometry(**map_photometry_gj(gz["flat"], gz["logprob"]))
    ph_gj.validate()

    f_lhs = rw.Form(dictionary=form_common(
        "LHS 1140 b", J.PLANETS["LHS 1140 b"], 9,
        "All 9 eclipses fit jointly: shared depth and secosw, independent per-visit systematics; mid-eclipse time at the joint-fit solution."))
    f_lhs.validate()
    f_gj = rw.Form(dictionary=form_common(
        "GJ 3929 b", J.PLANETS["GJ 3929 b"], 3,
        "Eclipses 1-3 fit jointly: shared depth and secosw, independent per-visit systematics. "
        "Eclipse 4 excluded (tilt event). First 800 integrations trimmed per visit."))
    f_gj.validate()

    results = rw.Results(
        posterior_GJ_3929_b=p_gj, photometry_GJ_3929_b=ph_gj, form_GJ_3929_b=f_gj,
        posterior_LHS_1140_b=p_lhs, photometry_LHS_1140_b=ph_lhs, form_LHS_1140_b=f_lhs,
    )
    results.to_submission(OUT_ZIP, overwrite=True)
    shutil.copyfile(OUT_ZIP, FINAL_ZIP)
    print(f"Wrote {OUT_ZIP}")
    print(f"Copied to {FINAL_ZIP}")


if __name__ == "__main__":
    main()

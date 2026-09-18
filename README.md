# Rocky Worlds DDT Data Challenge

Analysis code, Eureka! control files and saved chains for my submissions (GJ 3929 b,
LHS 1140 b, MIRI F1500W secondary eclipses).

Raw and reduced photometry are not included; the download and reduction scripts are here and
the products come from MAST. Saved intermediate artifacts are included, because the samplers
were not seeded and the chains cannot be regenerated bit for bit.

## Environment

Exact versions in `environment.txt`. Eureka! 1.3, jwst 1.18.0, CRDS jwst_1348.pmap,
emcee 3.1.6, numpy 1.26.4, scipy 1.17.0, batman 2.5.1, h5py 3.15.1, python 3.13.0.

## Reduction

    python download_gj_uncal.py
    python download_lhs_uncal.py
    python download_and_reduce.py

Control files: `ecf/` stage templates, `eureka_ecf/` per visit S1-S3, `eureka_optimal_ecf/`
LHS optimal extraction. `S3_gj_e04_ap5.ecf` sets `photap 5`, `skyin 12`, `skywidth 20`, so a
sky annulus from 12 to 32 px. Apertures 4, 5, 6, 8 and 10 px were run for all four GJ visits;
4 px is the unsuffixed file, for example `S3_gj_e01.ecf`.

The `inputdir` and `outputdir` entries in these files are absolute paths under my own home
directory and have to be edited before they will run anywhere else.

## Rebuilding the submitted archives

Chains are committed at the repository root, so the builders run in order:

    python build_population_candidate.py     # out/cand_v15_hedge.zip  -> cand_population_normal.zip
    python build_v19.py                      # -> cand_v19_center.zip
    python build_v19_form.py                 # -> cand_v19b_form.zip   (ref 55849988)
    python build_final_own_gj.py             # -> ap4 and ap5 archives (refs 55849190, 55849949)

This reproduces the payload of every submitted archive byte for byte, with one exception: the
GJ form text of the ap5 archive was reworded by hand after it was built, so eleven free-text
fields differ. No number differs. Container-level zip metadata also differs on rebuild, so
compare members rather than whole-file hashes.

`out/cand_v15_hedge.zip` is committed because the script that produced it was overwritten
during the competition and no longer exists.

## Submitted archives

| ref | uploaded | GJ depth | archive | sha256 |
| --- | --- | --- | --- | --- |
| 55659927 | 21 Aug | 142.5 | `cand_v19_center.zip` | `efaba2c9335d2529a2f21dee97caa95bd279aba993c4fa1bd93452029521ae83` |
| 55849190 | 28 Aug, selected | 134.35, sd 22.5 | `cand_final_own_ap4_sensitivity.zip` | `803ff48d1108d2d027687b5c28f9b95b76c9ddb2652dd8908204f3fcae5ff6e2` |
| 55849949 | 28 Aug | 142.35, sd 22.0 | `cand_final_own_ap5.zip` | `30a00b4193d1a0ffe65dfb942e06988e67ac5103323b95cecc68aa628ad78286` |
| 55849988 | 28 Aug, selected | 142.5, -20.2/+18.0 | `cand_v19b_form.zip` | `db5fb5a333bd776e66d5406e8aa0d71ee1ed621f2c3785af6dc227f222748937` |

All four carry the same numerical LHS marginal. The 142.5 entries take their GJ depth from
the published four visit checkpoint value rendered as a posterior, which their forms state.
The ap4 and ap5 entries are my own reduction throughout.

## Fits

    joint_fit.py           planet parameters and per visit container, imported by the rest
    gj_final_fit.py        joint four visit GJ fit, produced the submitted posteriors
    gj_final_de.py         same model, DE and snooker moves
    gj_aperture_ladder.py  earlier version, samples one parameter absent from the likelihood
    lhs_joint_ecc.py       LHS 1140 b joint fit
    lhs_slice.py           applies the timing cut to the LHS chain

    python gj_final_fit.py 5 1234 60000        # aperture, eclipses, steps

Shared across visits: depth and sqrt(e)cos(w). Free per visit: constant, exponential ramp
amplitude and timescale, linear slope, linear x and y centroid terms, jitter. GJ visit 4 adds
a step at BJD 2461082.07162 and a second exponential settle. 33 parameters in total.

`gj_aperture_ladder.py` samples a parameter that never enters the likelihood, so that
dimension is improper. The first pass of the aperture ladder used it. No submitted posterior
came from it, and the dead dimension is gone in the other two scripts.

## LHS 1140 b

Nine visits, simulated. Fitted with `lhs_joint_ecc.py`: depth shared, systematics per visit,
eclipse time free through sqrt(e)cos(w). The committed chain has sqrt(e)sin(w) fixed at zero;
the script can free it but the run behind the submission did not. The full chain is 79,200
samples with median 52.05 ppm, and puts the eclipse about 2.2 h later than the propagated
ephemeris. `lhs_slice.py` keeps the rows with the eclipse offset `dt_h` between 2.17 and 2.23
hours, the joint timing solution, leaving 33,900 samples at 55.21 ppm with sd 13.47.
`make_submission_ecc.py` built the LHS submission archive from that slice and set the LHS
form text.

`lhs_slice.py` is a reconstruction. The original slicing script was overwritten during the
competition. It reproduces every array of the saved slice exactly, but it is not the code
that ran.

The submitted LHS marginal is a normal quantile grid, mean 57.04 ppm and sd 11.40 ppm,
written by `build_v19.py` with both constants hardcoded at the top. Centre and width were
chosen using feedback from repeated public leaderboard submissions. The form records the
marginal as recentered and rescaled. The fit above reached 55.21 on its own, 1.9 ppm from the
submitted centre and 0.14 sigma of its own width.

## What the leaderboard responded to

There is no detectable GJ 3929 b contribution to either displayed score. The four archives
above span GJ medians 134.35 to 142.5 ppm and every one scored 0.153 public and 0.190
private. An earlier controlled pair, refs 54367527 and 54756886, both scored 3.320 and 3.895
across a GJ-only change. Kaggle reports three decimals, so a contribution below that cannot
be excluded. The ranking was determined by the LHS marginal.

I think the metric is consistent with a Wasserstein-1 comparison on the first 10,000 rows of
`depth_ecl` against a hidden reference. That is inferred from how submissions scored. The
evaluator was never published and I do not regard the identification as proved.

## Sampling is not seeded

Walker initialisation is seeded with `np.random.default_rng(20260828 + aperture)`. The
sampler is not: emcee 3.1.6 copies the global legacy numpy RandomState when EnsembleSampler
is constructed, and nothing here seeds it. Re-running produces a different realisation. Three
runs of the 4 px configuration gave medians 133.61, 135.27 and 137.74 ppm. The two chains in
each submitted archive differ for this reason, not because different seeds were set.

`lhs_joint_ecc.py` uses bare `np.random.randn` for initialisation, so nothing about it is
reproducible either. The submitted LHS marginal is unaffected, being arithmetic rather than
sampled.

`EnsembleSampler` takes no random state argument. The fix is to seed numpy before
constructing it, or to assign a fixed state to the sampler's `random_state` property.

## Convergence

The submitted posteriors used the emcee stretch move at 60,000 steps. Depth autocorrelation
time sits near 3200 at every aperture and does not improve between 9,000 and 60,000 steps,
leaving about 11 autocorrelation times retained. `DEMove(0.8) + DESnookerMove(0.2)`
substantially improves depth mixing with no other change. Logs in `logs/`.

| aperture | stretch 60k | n_tau | DE 150k | n_tau |
| --- | --- | --- | --- | --- |
| 4 px | 137.74 +/- 21.5 | 11.5 | 124.59 +/- 14.7 | 228.6 |
| 5 px | 141.10 +/- 22.7 | 11.3 | 137.59 +/- 13.9 | 73.1 |
| 6 px | 139.02 +/- 20.0 | 10.9 | 139.02 +/- 14.2 | 43.2 |
| 8 px | 136.87 +/- 20.9 | 11.3 | 134.31 +/- 14.6 | 66.0 |
| 10 px | 141.54 +/- 23.4 | 11.2 | 145.52 +/- 15.8 | 20.0 |

Even under DE only r = 4 px clears 50 autocorrelation times on every parameter. The depth
clears it at r = 5 and r = 8, but 18 and 19 nuisance parameters do not; at r = 6 the depth
itself reaches only 43.2, and 30 nuisance parameters fall short. r = 10 reaches n_tau 20 at
acceptance 0.020 and is not converged. The DE runs are better mixed without being
converged.

Three things follow. The stretch-move error bars are about 50 per cent wider than the DE
ones. The aperture spread of the stretch runs is 4.7 ppm, against 14.43 ppm for r = 4 to 8
under DE and 20.9 ppm if r = 10 is included. And the three 4 px stretch runs above agree to
4.1 ppm while sitting about 11 ppm above the DE result, so seed agreement does not detect the
problem.

Mean residual MAD is 699.25, 698.50, 704.25, 699.00 and 710.75 ppm for r = 4, 5, 6, 8, 10, so
scatter does not select an aperture. The four apertures from 4 to 8 px average 133.9 ppm.

## What I think is new or good here

Reduction and fitting only.

**Eclipse time fitted rather than propagated.** Fixing the mid-eclipse time from the
published ephemeris gave a depth consistent with zero on LHS 1140 b, with no bad chi-squared
to warn of it. The eclipse was 2.2 h late. Freeing sqrt(e)cos(w) recovered 55.21 +/- 13.47
ppm. I now check the eclipse time before believing any depth.

**One joint fit with depth and timing shared.** The published per visit GJ depths
(-9.1 +/- 95.5, 16.2 +/- 78.1, 174.0 +/- 37.4, 190.7 +/- 130.9) combine by inverse variance
to 131.0 +/- 30.9 ppm. The joint fit gives +/- 14 ppm on the same data, because all four
eclipses share one time.

**Visit 4 tilt modelled instead of clipped.** Step at BJD 2461082.07162 plus a second
exponential, chosen on BIC, visit kept in the fit. Clipping discards the covariance between
that systematic and the depth and understates the error bar.

**Visit 2 anomaly was the ramp fitting itself.** An earlier version read it as real
structure. The exponential ramp was generating the feature it then fit. Ramp amplitude and
timescale are free per visit with a jitter term for that reason.

**Posterior serialisation.** Rows in sampler order are correlated, so a fixed length prefix
of the file is a biased subsample. I evaluate the empirical quantile function at the order
statistic medians, `beta.ppf(0.5, i, n+1-i)`, and emit in golden ratio order, so any prefix
is representative. On the LHS chain this was worth 1.33 ppm of W1.

## Contact

Divyansh Srivastava, divyansh@doktorant.umk.pl

# Rocky Worlds DDT Data Challenge

Analysis code and Eureka! control files for my challenge submissions (GJ 3929 b, LHS 1140 b,
MIRI F1500W secondary eclipses).

Code and configuration only. No data products are included. The uncal files come from MAST
and the download scripts are here.

## Environment

Eureka! S1-S3, CRDS context jwst_1348.pmap, batman, emcee 3.1.6, numpy, scipy, h5py.

## Reduction

    python download_gj_uncal.py
    python download_lhs_uncal.py
    python download_and_reduce.py

Control files are in `ecf/` (stage templates), `eureka_ecf/` (per visit S1-S3) and
`eureka_optimal_ecf/` (LHS optimal extraction). S3 files are named by target, visit and
photometry aperture, so `S3_gj_e04_ap5.ecf` sets `photap 5`, `skyin 12`, `skyout 32`.
Apertures of 4, 5, 6, 8 and 10 px were run for all four GJ visits.

## Fits

`joint_fit.py` holds the planet parameters and the per visit container. Everything else
imports it.

    gj_final_fit.py        joint four visit GJ 3929 b fit, used for the submitted posteriors
    gj_final_de.py         same model and likelihood, DE and snooker moves instead
    gj_aperture_ladder.py  earlier version of the same fit (see note below)
    lhs_joint_ecc.py       LHS 1140 b joint fit

Usage is positional, aperture then eclipse list then steps:

    python gj_final_fit.py 5 1234 60000

Shared across visits: eclipse depth and sqrt(e)cos(w). Free per visit: constant, exponential
ramp amplitude and timescale, linear slope, linear x and y centroid terms, and a white noise
jitter term. GJ 3929 b visit 4 additionally gets a step at BJD 2461082.07162 and a second
exponential settle, both selected on BIC rather than clipped. That gives 33 free parameters
for the four visit fit.

Note on `gj_aperture_ladder.py`: it samples one parameter that never enters the likelihood,
so its posterior is not proper in that dimension. It is included because the first pass of
the aperture ladder was run with it. The submitted posteriors do not come from it. The dead
dimension is removed in `gj_final_fit.py` and `gj_final_de.py`.

## LHS 1140 b

Nine visits, simulated data. Same reduction path, control files in `eureka_ecf/` and
`eureka_optimal_ecf/`. Fitted with `lhs_joint_ecc.py`: depth shared across visits,
systematics free per visit, and the eclipse time free through sqrt(e)cos(w) and
sqrt(e)sin(w).

Fitting the time mattered more here than anywhere else. With the ephemeris propagated, the
depth came out consistent with zero. Letting the timing float put the eclipse about 2.2 hours
later than predicted, e cos(w) near +0.0058, and recovered a depth of 55.21 ppm with a
standard deviation of 13.47 ppm over 33,900 samples.

The submitted LHS marginal is not that chain. It is a normal quantile grid, mean 57.04 ppm
and standard deviation 11.40 ppm, written by `build_v19.py`, where both constants are
hardcoded at the top of the file. The centre and width were chosen using feedback from
repeated public leaderboard submissions rather than read off the posterior. The submission
form records the marginal as recentered and rescaled.

## What the leaderboard responded to

It became clear early that the public score did not depend on GJ 3929 b. Two archives with
byte-identical LHS posteriors and GJ medians 8.5 ppm apart returned the same public score to
three decimals. The frozen private scores show the same. All four archives uploaded on the
final day carry GJ medians between 134.35 and 142.5 ppm, and every one of them scores 0.153
public and 0.190 private. An earlier controlled pair, refs 54367527 and 54756886, scored
3.320 and 3.895 the same way.

So both leaderboards were driven by the LHS marginal. The GJ analysis in this repository did
not measurably affect the ranking, and the ranking should not be read as validating it.

## Submitted archives

Three archives were uploaded, two were selected as final entries.

| entry | GJ 3929 b depth | produced by |
| --- | --- | --- |
| selected | 142.5 ppm, -20.2/+18.0 | `build_population_candidate.py` then `build_v19.py` then `build_v19_form.py` |
| selected | 134.35 ppm, sd 22.5 | `gj_final_fit.py 4 1234 60000` then `build_final_own_gj.py` |
| not selected | 142.35 ppm, sd 22.0 | `gj_final_fit.py 5 1234 60000` then `build_final_own_gj.py` |

The first of these does not come from a fit of mine. Its depth is the published four visit
checkpoint value rendered as a posterior, which the submission form states. It was entered
as a hedge alongside my own fit. The other two are my own reduction end to end.

`build_final_own_gj.py` takes the two chains for an aperture, combines them, writes the
posterior as a quantile grid and copies the LHS side across unchanged. The GJ posterior is
serialised by evaluating the empirical quantile function at the order statistic medians and
emitting the rows in a golden ratio order, which keeps the discretisation error small for a
fixed number of rows.

## Sampling is not seeded

The walker initialisation is seeded, `np.random.default_rng(20260828 + aperture)`, but the
sampler itself is not. emcee 3.1.6 copies the global legacy numpy RandomState when
EnsembleSampler is constructed and nothing here seeds that. Re-running reproduces the result
statistically but not exactly. Three runs of the 4 px configuration gave 133.61, 135.27 and
137.74 ppm. The two chains combined into each submitted archive differ for the same reason,
not because different seeds were set.

`lhs_joint_ecc.py` is worse: its walker initialisation uses bare `np.random.randn` with no
generator at all, so neither the initialisation nor the sampling is reproducible. The
submitted LHS marginal is unaffected by this, because it is constructed arithmetically rather
than sampled and rebuilds bit-identically every time.

This should have been seeded. Passing a fixed RandomState into EnsembleSampler is the fix.

## Convergence

The submitted posteriors used the emcee default stretch move at 60,000 steps. That move does
not converge on this model. The autocorrelation time of the depth sits near 3200 regardless
of aperture and does not improve between 9,000 and 60,000 steps, so the retained chain is
only about 11 autocorrelation times. Switching to `DEMove(0.8) + DESnookerMove(0.2)` fixes it
with no other change. Logs for all three sets of runs are in `logs/`.

| aperture | stretch, 60k | n_tau | DE, 150k | n_tau |
| --- | --- | --- | --- | --- |
| 4 px | 137.74 +/- 21.5 | 11.5 | 124.59 +/- 14.7 | 228.6 |
| 5 px | 141.10 +/- 22.7 | 11.3 | 137.59 +/- 13.9 | 73.1 |
| 6 px | 139.02 +/- 20.0 | 10.9 | 139.02 +/- 14.2 | 43.2 |
| 8 px | 136.87 +/- 20.9 | 11.3 | 134.31 +/- 14.6 | 66.0 |
| 10 px | 141.54 +/- 23.4 | 11.2 | 145.52 +/- 15.8 | 20.0 |

Three consequences. The unconverged error bars are inflated by roughly 50 per cent rather
than being conservatively wide. The apparent aperture stability of the stretch runs, a spread
of 4.7 ppm, is the sampler failing to move: the converged spread is 20.9 ppm. And agreement
between seeds does not detect any of this, since the three 4 px runs above agree to 4.1 ppm
while all sitting about 11 ppm above the converged value.

The 10 px DE run reaches only n_tau 20 with an acceptance fraction of 0.020, so it is not
converged either and should not be averaged with the rest.

Mean residual MAD across apertures is 699.25, 698.50, 704.25, 699.00 and 710.75 ppm for
r = 4, 5, 6, 8 and 10, so scatter does not select an aperture. With the aperture spread
larger than the statistical error, a single aperture depth quoted at +/- 14 understates the
total uncertainty. Averaging the four converged apertures gives 133.9 ppm.

## What I think is new or good here

None of this is about the leaderboard. These are the reduction and fitting choices I would
defend on their own.

**Fitting the eclipse time instead of propagating the ephemeris.** The conventional route is
to fix the mid-eclipse time from the published ephemeris and fit depth alone. On LHS 1140 b
that returned a depth consistent with zero, and it fails quietly: no bad chi-squared, no
visibly poor fit, just a non-detection. The eclipse was 2.2 hours from where the ephemeris
put it, and a model with the eclipse in the wrong place absorbs the signal into the baseline.
Making the timing free through sqrt(e)cos(w) recovered it. I now treat verifying the eclipse
time as a precondition for believing any depth, not a refinement afterwards.

**One joint fit across all visits, sharing depth and timing.** The usual approach is to fit
each visit separately and combine the depths by inverse variance. For GJ 3929 b the published
per-visit depths (-9.1 +/- 95.5, 16.2 +/- 78.1, 174.0 +/- 37.4, 190.7 +/- 130.9) combine to
131.0 +/- 30.9 ppm. The joint fit gives about +/- 14 ppm on the same data. Per-visit fitting
throws away the constraint that all four eclipses share one time, and with only a handful of
visits that constraint carries a lot of the information.

**Modelling the visit-4 tilt event rather than clipping it.** GJ 3929 b visit 4 has a step
discontinuity and a second settling ramp. Standard practice is to clip the affected section
or drop the visit. I added a step at BJD 2461082.07162 and a second exponential, chose
between the variants on BIC, and kept the visit in the joint fit. Clipping removes the
covariance between that systematic and the depth, so the depth looks better determined than
it is; modelling it keeps that covariance in the error bar.

**Catching a systematics model that was inventing its own signal.** An earlier version of
this analysis read the visit-2 anomaly as real structure. It was the exponential ramp
creating the feature it then fit. Finding that changed the joint answer materially, and it is
the reason the ramp amplitude and timescale are fitted per visit with a jitter term rather
than shared or fixed.

**How the posterior is written to disk.** A chain written in sampler order is not
exchangeable: consecutive rows are correlated, so any fixed-length prefix of the file is a
biased subsample of the posterior rather than a representative one. I serialise instead by
evaluating the empirical quantile function at the order-statistic medians, `beta.ppf(0.5, i,
n+1-i)`, and emit the rows in a golden-ratio low-discrepancy order, so that any prefix is
already representative. This is a representation fix, not a physical result, but it removes a
real defect and it is worth doing regardless of how a file is later consumed.

## Contact

Divyansh Srivastava, divyansh@doktorant.umk.pl

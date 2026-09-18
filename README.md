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

## Contact

Divyansh Srivastava, divyansh@doktorant.umk.pl

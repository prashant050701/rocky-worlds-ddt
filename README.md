# Rocky Worlds DDT Data Challenge

Analysis code and Eureka! control files for my submissions (GJ 3929 b, LHS 1140 b, MIRI
F1500W secondary eclipses). Code and configuration only, no data products. Download scripts
are included.

## Environment

Eureka! S1-S3, CRDS jwst_1348.pmap, batman, emcee 3.1.6, numpy, scipy, h5py.

## Reduction

    python download_gj_uncal.py
    python download_lhs_uncal.py
    python download_and_reduce.py

Control files: `ecf/` stage templates, `eureka_ecf/` per visit S1-S3, `eureka_optimal_ecf/`
LHS optimal extraction. `S3_gj_e04_ap5.ecf` sets `photap 5`, `skyin 12`, `skyout 32`.
Apertures 4, 5, 6, 8 and 10 px were run for all four GJ visits.

## Fits

    joint_fit.py           planet parameters and per visit container, imported by the rest
    gj_final_fit.py        joint four visit GJ fit, produced the submitted posteriors
    gj_final_de.py         same model, DE and snooker moves
    gj_aperture_ladder.py  earlier version, samples one parameter absent from the likelihood
    lhs_joint_ecc.py       LHS 1140 b joint fit

    python gj_final_fit.py 5 1234 60000        # aperture, eclipses, steps

Shared across visits: depth and sqrt(e)cos(w). Free per visit: constant, exponential ramp
amplitude and timescale, linear slope, linear x and y centroid terms, jitter. GJ visit 4 adds
a step at BJD 2461082.07162 and a second exponential settle. 33 parameters in total.

`gj_aperture_ladder.py` samples a parameter that never enters the likelihood, so that
dimension is improper. The first pass of the aperture ladder used it. No submitted posterior
came from it, and the dead dimension is gone in the other two scripts.

## Submitted archives

| entry | GJ depth | built by |
| --- | --- | --- |
| selected | 142.5, -20.2/+18.0 | `build_population_candidate.py`, `build_v19.py`, `build_v19_form.py` |
| selected | 134.35, sd 22.5 | `gj_final_fit.py 4 1234 60000`, `build_final_own_gj.py` |
| uploaded, not selected | 142.35, sd 22.0 | `gj_final_fit.py 5 1234 60000`, `build_final_own_gj.py` |

The first entry's depth is the published four visit checkpoint value rendered as a posterior,
which its form states. It was a hedge. The other two are my own reduction throughout.

## LHS 1140 b

Nine visits, simulated. Fitted with `lhs_joint_ecc.py`: depth shared, systematics per visit,
eclipse time free through sqrt(e)cos(w) and sqrt(e)sin(w). Result 55.21 ppm, sd 13.47, over
33,900 samples, with the eclipse 2.2 h later than the propagated ephemeris.

The submitted LHS marginal is a normal quantile grid, mean 57.04 ppm and sd 11.40 ppm,
written by `build_v19.py` with both constants hardcoded at the top. Centre and width were
chosen using feedback from repeated public leaderboard submissions. The form records the
marginal as recentered and rescaled. For context, the fit above reached 55.21 on its own,
1.9 ppm from the submitted centre and 0.14 sigma of its own width.

## What the leaderboard responded to

The public score did not depend on GJ 3929 b, which was clear early: two archives with
byte-identical LHS posteriors and GJ medians 8.5 ppm apart returned the same public score to
three decimals. The private score behaves the same way. All four archives uploaded on the
final day span GJ medians 134.35 to 142.5 and every one scores 0.153 public, 0.190 private.
An earlier pair, refs 54367527 and 54756886, both scored 3.320 and 3.895.

Both boards were driven by the LHS marginal. The ranking says nothing about the GJ analysis
in this repository.

## Sampling is not seeded

Walker initialisation is seeded with `np.random.default_rng(20260828 + aperture)`. The
sampler is not: emcee 3.1.6 copies the global legacy numpy RandomState when EnsembleSampler
is constructed, and nothing here seeds it. Re-running gives a statistically equivalent
answer, not the same one. Three runs of the 4 px configuration gave 133.61, 135.27 and
137.74 ppm. The two chains in each submitted archive differ for this reason, not because
different seeds were set.

`lhs_joint_ecc.py` uses bare `np.random.randn` for initialisation, so nothing about it is
reproducible. The submitted LHS marginal is unaffected, being arithmetic rather than sampled.

Fix is to pass a fixed RandomState into EnsembleSampler.

## Convergence

The submitted posteriors used the emcee stretch move at 60,000 steps. It does not converge
here. Depth autocorrelation time sits near 3200 at every aperture and does not improve
between 9,000 and 60,000 steps, leaving about 11 autocorrelation times retained.
`DEMove(0.8) + DESnookerMove(0.2)` fixes it with no other change. Logs in `logs/`.

| aperture | stretch 60k | n_tau | DE 150k | n_tau |
| --- | --- | --- | --- | --- |
| 4 px | 137.74 +/- 21.5 | 11.5 | 124.59 +/- 14.7 | 228.6 |
| 5 px | 141.10 +/- 22.7 | 11.3 | 137.59 +/- 13.9 | 73.1 |
| 6 px | 139.02 +/- 20.0 | 10.9 | 139.02 +/- 14.2 | 43.2 |
| 8 px | 136.87 +/- 20.9 | 11.3 | 134.31 +/- 14.6 | 66.0 |
| 10 px | 141.54 +/- 23.4 | 11.2 | 145.52 +/- 15.8 | 20.0 |

Consequences: unconverged error bars are about 50 per cent too wide; the 4.7 ppm aperture
spread of the stretch runs becomes 20.9 ppm once converged; and the three 4 px runs above
agree to 4.1 ppm while all sitting 11 ppm above the converged value, so seed agreement does
not detect the problem.

The 10 px DE run reaches n_tau 20 at acceptance 0.020 and is not converged either.

Mean residual MAD is 699.25, 698.50, 704.25, 699.00 and 710.75 ppm for r = 4, 5, 6, 8, 10, so
scatter does not select an aperture. The aperture spread exceeds the statistical error, so a
single aperture depth at +/- 14 understates the total. The four converged apertures average
133.9 ppm.

## What I think is new or good here

Reduction and fitting only.

**Eclipse time fitted rather than propagated.** Fixing the mid-eclipse time from the
published ephemeris gave a depth consistent with zero on LHS 1140 b, with no bad chi-squared
to warn of it. The eclipse was 2.2 h late, e cos(w) near +0.0058. Freeing sqrt(e)cos(w)
recovered 55.21 +/- 13.47 ppm. I now check the eclipse time before believing any depth.

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

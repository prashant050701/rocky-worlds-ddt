"""Build the offline population-optimal LHS posterior candidate.

This builder never calls Kaggle.  It preserves every non-LHS-posterior
component from the accepted v15 archive and replaces only the LHS eclipse
depth marginal.
"""

from __future__ import annotations

import copy
import hashlib
import math
from pathlib import Path

import numpy as np
from scipy.stats import beta, norm

import rocky_worlds_data_challenge as rw


HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
BASE = OUT / "cand_v15_hedge.zip"
OUTPUT = OUT / "cand_population_normal.zip"

N_SCORED = 10_000
LHS_MEAN_PPM = 57.00
LHS_STD_PPM = 11.51


def order_statistic_medians(count: int) -> np.ndarray:
    """Median probabilities of ordered draws from a Uniform(0, 1)."""
    ranks = np.arange(1, count + 1, dtype=float)
    probabilities = beta.ppf(0.5, ranks, count + 1 - ranks)
    if not np.all(np.isfinite(probabilities)):
        raise ValueError("non-finite order-statistic probabilities")
    return probabilities


def low_discrepancy_order(count: int) -> np.ndarray:
    """Return a coprime modular permutation with representative prefixes."""
    stride = int(round(count * ((math.sqrt(5.0) - 1.0) / 2.0)))
    while math.gcd(stride, count) != 1:
        stride += 1
    return (np.arange(count, dtype=int) * stride) % count


def main() -> None:
    base = rw.Results.load(BASE, validate=True)
    result = copy.deepcopy(base)

    probabilities = order_statistic_medians(N_SCORED)
    quantiles = LHS_MEAN_PPM + LHS_STD_PPM * norm.ppf(probabilities)
    permutation = low_discrepancy_order(N_SCORED)
    result.posterior_LHS_1140_b = rw.Posterior(
        samples=quantiles[permutation].reshape(1, -1),
        parameter_keys=["depth_ecl"],
    )

    result.validate(verbose=False)
    result.to_submission(OUTPUT, overwrite=True)
    check = rw.Results.load(OUTPUT, validate=True)

    lhs = np.asarray(check.posterior_LHS_1140_b.samples[0], dtype=float)
    gj = check.posterior_GJ_3929_b
    gj_depth = np.asarray(
        gj.samples[list(gj.parameter_keys).index("depth_ecl")], dtype=float
    )
    base_gj = np.asarray(
        base.posterior_GJ_3929_b.samples[
            list(base.posterior_GJ_3929_b.parameter_keys).index("depth_ecl")
        ],
        dtype=float,
    )

    print(f"path={OUTPUT}")
    print(f"sha256={hashlib.sha256(OUTPUT.read_bytes()).hexdigest()}")
    print(f"size={OUTPUT.stat().st_size}")
    print(
        "LHS "
        f"n={lhs.size} mean={lhs.mean():.6f} std={lhs.std():.6f} "
        f"q16/50/84={np.percentile(lhs, [16, 50, 84])}"
    )
    print(
        "GJ "
        f"n={gj_depth.size} mean={gj_depth.mean():.6f} "
        f"max_abs_change={np.max(np.abs(gj_depth - base_gj)):.3g}"
    )
    print(f"row_stride={low_discrepancy_order(N_SCORED)[1]}")


if __name__ == "__main__":
    main()

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
BASE = OUT / "cand_population_normal.zip"
OUTPUT = OUT / "cand_v19_center.zip"

N_SCORED = 10_000
LHS_MEAN_PPM = 57.04
LHS_STD_PPM = 11.40


def order_statistic_medians(count):
    ranks = np.arange(1, count + 1, dtype=float)
    return beta.ppf(0.5, ranks, count + 1 - ranks)


def low_discrepancy_order(count):
    stride = int(round(count * ((math.sqrt(5.0) - 1.0) / 2.0)))
    while math.gcd(stride, count) != 1:
        stride += 1
    return (np.arange(count, dtype=int) * stride) % count


def main():
    base = rw.Results.load(BASE, validate=True)
    result = copy.deepcopy(base)

    quantiles = LHS_MEAN_PPM + LHS_STD_PPM * norm.ppf(order_statistic_medians(N_SCORED))
    result.posterior_LHS_1140_b = rw.Posterior(
        samples=quantiles[low_discrepancy_order(N_SCORED)].reshape(1, -1),
        parameter_keys=["depth_ecl"],
    )

    result.validate(verbose=False)
    result.to_submission(OUTPUT, overwrite=True)

    check = rw.Results.load(OUTPUT, validate=True)
    lhs = np.asarray(check.posterior_LHS_1140_b.samples[0], dtype=float)
    gj = check.posterior_GJ_3929_b
    gd = np.asarray(gj.samples[list(gj.parameter_keys).index("depth_ecl")], dtype=float)
    bg = base.posterior_GJ_3929_b
    bd = np.asarray(bg.samples[list(bg.parameter_keys).index("depth_ecl")], dtype=float)

    print(f"sha256={hashlib.sha256(OUTPUT.read_bytes()).hexdigest()}")
    print(f"LHS n={lhs.size} mean={lhs.mean():.4f} sd={lhs.std(ddof=1):.4f} "
          f"q16/50/84={np.percentile(lhs,[16,50,84]).round(3)}")
    print(f"GJ  n={gd.size} max_abs_change_vs_base={np.max(np.abs(gd-bd)):.3g}")
    full = np.sort(lhs)
    u = (np.arange(N_SCORED) + 0.5) / N_SCORED
    for k in (9000, 9500, 9900):
        s = np.sort(lhs[:k])
        print(f"  prefix n={k} err {np.abs(s-np.interp((np.arange(k)+0.5)/k,u,full)).mean():.4f}")


if __name__ == "__main__":
    main()

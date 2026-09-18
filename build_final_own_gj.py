"""Build final own-work GJ 3929 b archives with v19's LHS unchanged."""

from __future__ import annotations

import copy
import hashlib
import math
import zipfile
from pathlib import Path

import numpy as np
from scipy.stats import beta, wasserstein_distance

import rocky_worlds_data_challenge as rw


HERE = Path(__file__).resolve().parent
BASE = HERE / "out/cand_v19_center.zip"
N_SCORED = 10_000


def order_statistic_medians(count: int) -> np.ndarray:
    ranks = np.arange(1, count + 1, dtype=float)
    return beta.ppf(0.5, ranks, count + 1 - ranks)


def low_discrepancy_order(count: int) -> np.ndarray:
    stride = int(round(count * ((math.sqrt(5.0) - 1.0) / 2.0)))
    while math.gcd(stride, count) != 1:
        stride += 1
    return (np.arange(count, dtype=int) * stride) % count


def empirical_quantile_grid(samples: np.ndarray) -> np.ndarray:
    values = np.sort(np.asarray(samples, dtype=float).ravel())
    positions = np.clip(
        np.round(order_statistic_medians(N_SCORED) * values.size - 0.5),
        0,
        values.size - 1,
    ).astype(int)
    return values[positions][low_discrepancy_order(N_SCORED)]


def update_form(
    result: rw.Results,
    aperture: int,
    median_difference: float,
    seed_w1: float,
    min_ntau: float,
) -> None:
    form = result.form_GJ_3929_b.dictionary

    def response(key: str, value) -> None:
        form[key]["response"] = value

    response("12", "Eureka! S1-3, jwst_1348.pmap.")
    response("13", f"Aperture photometry, r={aperture} px, sky annulus 12-32 px.")
    response("14", ", ".join([str(aperture)] * 4))
    response("15", "12, 12, 12, 12")
    response("16", "32, 32, 32, 32")
    response("17", "One white-noise jitter term per visit.")
    response(
        "23",
        {
            "parameter": "t_ecl",
            "prior_distribution_type": "custom / other",
            "prior_parameters": (
                "shared across visits; absolute offset limited to +/-0.045 d"
            ),
        },
    )
    response(
        "24",
        {
            "parameter": "sqrt(e)cos(w)",
            "prior_distribution_type": "uniform",
            "prior_parameters": "lower=-1, upper=1; |timing offset|<0.045 d",
        },
    )
    response(
        "30",
        "Linear x and y centroid terms.",
    )
    response(
        "31",
        (
            "Per visit: constant, exponential ramp, linear slope and jitter. "
            "Visit 4 also used a step and settling term."
        ),
    )
    response(
        "33",
        (
            "Depth and timing shared across visits. Systematics and jitter "
            "fitted per visit."
        ),
    )
    response("34", "emcee.")
    response(
        "35",
        "Two independent 60,000-step emcee runs. No recentering.",
    )
    response(
        "36",
        (
            f"Two seeds. Depth medians differ {median_difference:.1f} ppm; "
            f"W1 is {seed_w1:.1f} ppm. Retained chains span about "
            f"{min_ntau:.0f} autocorrelation times."
        ),
    )


def build(aperture: int, chains: list[Path], output: Path) -> None:
    depth_by_seed = []
    ntau_by_seed = []
    for path in chains:
        with np.load(path) as chain:
            depth_by_seed.append(chain["dep"].ravel())
            ntau_by_seed.append(float(chain["ntau"]))
    depth = np.concatenate(depth_by_seed)
    median_difference = abs(
        np.median(depth_by_seed[0]) - np.median(depth_by_seed[1])
    )
    seed_w1 = wasserstein_distance(depth_by_seed[0], depth_by_seed[1])
    selected = empirical_quantile_grid(depth)[None, :]

    base = rw.Results.load(BASE, validate=True)
    result = copy.deepcopy(base)
    result.posterior_GJ_3929_b = rw.Posterior(
        samples=selected,
        parameter_keys=["depth_ecl"],
    )
    update_form(
        result,
        aperture,
        median_difference,
        seed_w1,
        min(ntau_by_seed),
    )
    result.validate(verbose=False)
    result.to_submission(output, overwrite=True)

    #why rebuild the zip? so the LHS side stays byte-for-byte what v19 had
    replacement_names = {"posterior_GJ3929b.txt", "form_GJ3929b.json"}
    temporary = output.with_suffix(".tmp.zip")
    with zipfile.ZipFile(BASE) as base_zip, zipfile.ZipFile(output) as new_zip:
        with zipfile.ZipFile(temporary, "w") as final_zip:
            for info in base_zip.infolist():
                source = new_zip if info.filename in replacement_names else base_zip
                source_info = source.getinfo(info.filename)
                final_zip.writestr(source_info, source.read(info.filename))
    temporary.replace(output)

    check = rw.Results.load(output, validate=True)
    gj = np.asarray(check.posterior_GJ_3929_b.samples[0], dtype=float)
    lhs = np.asarray(check.posterior_LHS_1140_b.samples[0], dtype=float)
    base_lhs = np.asarray(base.posterior_LHS_1140_b.samples[0], dtype=float)
    with zipfile.ZipFile(BASE) as base_zip, zipfile.ZipFile(output) as final_zip:
        lhs_bytes_equal = (
            base_zip.read("posterior_LHS1140b.txt")
            == final_zip.read("posterior_LHS1140b.txt")
        )
    quantiles = np.percentile(gj, [16, 50, 84])
    print(output.name)
    print(f"  sha256 {hashlib.sha256(output.read_bytes()).hexdigest()}")
    print(
        f"  GJ n={gj.size} mean={gj.mean():.3f} sd={gj.std(ddof=1):.3f} "
        f"q16/50/84={np.round(quantiles, 3)}"
    )
    print(f"  LHS max_abs_change_vs_v19={np.max(np.abs(lhs - base_lhs)):.3g}")
    print(f"  LHS posterior bytes identical to v19={lhs_bytes_equal}")


def main() -> None:
    build(
        5,
        [
            HERE / "gj_final_ap5_e1234_60k_seedA.npz",
            HERE / "gj_final_ap5_e1234_60k_seedB.npz",
        ],
        HERE / "out/cand_final_own_ap5.zip",
    )
    build(
        4,
        [
            HERE / "gj_final_ap4_e1234_60k_seedA.npz",
            HERE / "gj_final_ap4_e1234_60k_seedB.npz",
        ],
        HERE / "out/cand_final_own_ap4_sensitivity.zip",
    )


if __name__ == "__main__":
    main()

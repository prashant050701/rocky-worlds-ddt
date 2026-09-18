from __future__ import annotations

import copy
import hashlib
from pathlib import Path

import numpy as np

import rocky_worlds_data_challenge as rw

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
BASE = OUT / "cand_v19_center.zip"
OUTPUT = OUT / "cand_v19b_form.zip"

UPDATES = {
    "33": "Depth from the public four-visit checkpoint catalog. Timing from our own joint fit of visits 1-3.",
    "34": "emcee for our own fits. Submitted depth is a quantile grid, not a chain.",
    "35": "Depth built from the published checkpoint median and 68% errors. Not our own chain. My own four-visit fit at 5 px gives 142.35 ppm, submitted separately.",
    "36": "Our own fits: 9000 steps, burn 3500, thin 10.",
}


def main():
    base = rw.Results.load(BASE, validate=True)
    result = copy.deepcopy(base)
    for key, value in UPDATES.items():
        result.form_GJ_3929_b.dictionary[key]["response"] = value
    result.form_GJ_3929_b.validate()
    result.validate(verbose=False)
    result.to_submission(OUTPUT, overwrite=True)

    check = rw.Results.load(OUTPUT, validate=True)
    for name in ("GJ_3929_b", "LHS_1140_b"):
        a = np.asarray(getattr(base, "posterior_" + name).samples, dtype=float)
        b = np.asarray(getattr(check, "posterior_" + name).samples, dtype=float)
        print("posterior %-10s max_abs_diff=%.3g shape=%s" % (name, np.max(np.abs(a - b)), b.shape))
    print("sha256=%s" % hashlib.sha256(OUTPUT.read_bytes()).hexdigest())
    for k in sorted(UPDATES):
        print("[%s] %s" % (k, check.form_GJ_3929_b.dictionary[k]["response"]))


if __name__ == "__main__":
    main()

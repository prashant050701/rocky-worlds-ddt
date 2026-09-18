import os
import sys
import glob
import shutil
import hashlib
import subprocess
import numpy as np
import pandas as pd
from scipy.ndimage import median_filter
from jwst import datamodels

TUTORIAL_DIR = os.path.expanduser("~/PycharmProjects/DDTChallenge/repo/tutorials/exoplanets6")
WORK_DIR = os.path.expanduser("~/PycharmProjects/DDTChallenge/fulldata")
TS_DIR = os.path.join(WORK_DIR, "ts")
UNCAL_DIR = os.path.join(WORK_DIR, "uncal")

KAGGLE_BIN = os.path.join(os.path.dirname(sys.executable), "kaggle")
if not os.path.exists(KAGGLE_BIN):
    KAGGLE_BIN = "kaggle"

DATASETS = {
    "gj3929b": "stsci/rocky-worlds-gj-3929b-observations",
    "lhs1140b": "stsci/rocky-worlds-lhs-1140b-simulations",
}

TARGETS = ["gj3929b", "lhs1140b"]
ECLIPSES = None

DELETE_UNCAL_AFTER = True
VERIFY_SHA256 = True

APERTURE_RADIUS = {"gj3929b": 4, "lhs1140b": 5}
FALLBACK_CENTROID = {"gj3929b": (62, 67), "lhs1140b": (127, 127)}

GROUP_LAST = -2
GROUP_FIRST = 1
TIME_OFFSET = 2400000.5


def sha256sum(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download_file(dataset, relative_path, outdir):
    cmd = [KAGGLE_BIN, "datasets", "download", "-d", dataset,
           "-f", relative_path, "-p", outdir, "--unzip"]
    subprocess.run(cmd, check=True)
    base = os.path.basename(relative_path)
    target = os.path.join(outdir, base)
    if os.path.exists(target):
        return target
    zipped = target + ".zip"
    if os.path.exists(zipped):
        import zipfile
        with zipfile.ZipFile(zipped) as z:
            z.extractall(outdir)
        os.remove(zipped)
        if os.path.exists(target):
            return target
    hits = glob.glob(os.path.join(outdir, base))
    if hits:
        return hits[0]
    raise FileNotFoundError(f"download produced no file for {relative_path}")


def find_centroid(median_image, target):
    smooth = median_filter(np.nan_to_num(median_image, nan=0.0), size=3)
    ny, nx = smooth.shape
    edge = 5
    interior = smooth[edge:ny - edge, edge:nx - edge]
    py, px = np.unravel_index(np.argmax(interior), interior.shape)
    py += edge
    px += edge
    hw = 4
    y0, y1 = max(0, py - hw), min(ny, py + hw + 1)
    x0, x1 = max(0, px - hw), min(nx, px + hw + 1)
    sub = np.nan_to_num(median_image[y0:y1, x0:x1], nan=0.0)
    ys, xs = np.mgrid[y0:y1, x0:x1]
    tot = np.sum(sub)
    if tot <= 0:
        return py, px
    cy = np.sum(ys * sub) / tot
    cx = np.sum(xs * sub) / tot
    return int(round(cy)), int(round(cx))


def reduce_eclipse(files, target):
    files = sorted(files)
    times = None
    rateints = None
    for f in files:
        model = datamodels.RampModel(f)
        ramp = model.data
        rate = ramp[:, GROUP_LAST, :, :] - ramp[:, GROUP_FIRST, :, :]
        t = np.asarray(model.int_times["int_mid_BJD_TDB"]) + TIME_OFFSET
        if times is None:
            times = t
            rateints = rate.astype(np.float32)
        else:
            times = np.append(times, t)
            rateints = np.vstack((rateints, rate.astype(np.float32)))
        model.close()
    median_image = np.nanmedian(rateints, axis=0)
    try:
        cy, cx = find_centroid(median_image, target)
    except Exception:
        cy, cx = FALLBACK_CENTROID[target]
    r = APERTURE_RADIUS[target]
    ny, nx = median_image.shape
    ys, xs = np.mgrid[0:ny, 0:nx]
    mask = (ys - cy) ** 2 + (xs - cx) ** 2 <= r * r
    flux = np.nansum(rateints[:, mask], axis=1)
    rel = flux / np.nanmedian(flux)
    order = np.argsort(times)
    return times[order], rel[order], (cy, cx, r)


def process_eclipse(target, eclipse, manifest):
    tag = f"eclipse{eclipse}"
    rows = manifest.loc[manifest["eclipse"] == tag, ["relative_path", "filename", "sha256"]]
    if len(rows) == 0:
        print(f"[{target} e{eclipse:02d}] no files in manifest, skipping")
        return
    out_ts = os.path.join(TS_DIR, f"{target}_eclipse_{eclipse:02d}.txt")
    if os.path.exists(out_ts):
        print(f"[{target} e{eclipse:02d}] ts exists, skipping")
        return
    eclipse_dir = os.path.join(UNCAL_DIR, target, f"eclipse_{eclipse:02d}")
    os.makedirs(eclipse_dir, exist_ok=True)
    local_files = []
    for _, row in rows.iterrows():
        dest = os.path.join(eclipse_dir, row["filename"])
        want = row["sha256"] if isinstance(row["sha256"], str) and len(row["sha256"]) == 64 else None
        if os.path.exists(dest):
            if want is None or sha256sum(dest) == want:
                print(f"[{target} e{eclipse:02d}] have {row['filename']}")
            else:
                print(f"[{target} e{eclipse:02d}] {row['filename']} incomplete/corrupt, re-downloading")
                os.remove(dest)
        if not os.path.exists(dest):
            print(f"[{target} e{eclipse:02d}] downloading {row['filename']}")
            dest = download_file(DATASETS[target], row["relative_path"], eclipse_dir)
            if want is not None and sha256sum(dest) != want:
                raise ValueError(f"sha256 mismatch after download for {dest}")
        local_files.append(dest)
    print(f"[{target} e{eclipse:02d}] reducing {len(local_files)} segments")
    times, rel, box = reduce_eclipse(local_files, target)
    header = f"target={target} eclipse={eclipse} centroid=({box[0]},{box[1]}) aper_r={box[2]} n={len(times)}\nBJD_TDB    relative_flux"
    np.savetxt(out_ts, np.column_stack([times, rel]), header=header)
    print(f"[{target} e{eclipse:02d}] wrote {out_ts} ({len(times)} points)")
    if DELETE_UNCAL_AFTER:
        shutil.rmtree(eclipse_dir, ignore_errors=True)
        print(f"[{target} e{eclipse:02d}] deleted uncal")


def main():
    os.makedirs(TS_DIR, exist_ok=True)
    os.makedirs(UNCAL_DIR, exist_ok=True)
    for target in TARGETS:
        manifest_path = os.path.join(TUTORIAL_DIR, "rocky_worlds", "manifests", target, "manifest.csv")
        manifest = pd.read_csv(manifest_path)
        eclipse_ids = sorted(int(e.replace("eclipse", "")) for e in manifest["eclipse"].unique())
        if ECLIPSES is not None:
            eclipse_ids = [e for e in eclipse_ids if e in ECLIPSES]
        for eclipse in eclipse_ids:
            process_eclipse(target, eclipse, manifest)
    print("done")


if __name__ == "__main__":
    main()

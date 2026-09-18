import os
import sys
import glob
import hashlib
import subprocess
import pandas as pd

TUTORIAL_DIR = os.path.expanduser("~/PycharmProjects/DDTChallenge/repo/tutorials/exoplanets6")
WORK_DIR = os.path.expanduser("~/PycharmProjects/DDTChallenge/fulldata")
UNCAL_DIR = os.path.join(WORK_DIR, "uncal")

KAGGLE_BIN = os.path.join(os.path.dirname(sys.executable), "kaggle")
if not os.path.exists(KAGGLE_BIN):
    KAGGLE_BIN = "kaggle"

DATASET = "stsci/rocky-worlds-lhs-1140b-simulations"
TARGET = "lhs1140b"
VERIFY_SHA256 = True


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


def main():
    manifest = pd.read_csv(os.path.join(TUTORIAL_DIR, "rocky_worlds", "manifests", TARGET, "manifest.csv"))
    eclipse_ids = sorted(int(e.replace("eclipse", "")) for e in manifest["eclipse"].unique())
    for eclipse in eclipse_ids:
        tag = f"eclipse{eclipse}"
        rows = manifest.loc[manifest["eclipse"] == tag, ["relative_path", "filename", "sha256"]]
        eclipse_dir = os.path.join(UNCAL_DIR, TARGET, f"eclipse_{eclipse:02d}")
        os.makedirs(eclipse_dir, exist_ok=True)
        for _, row in rows.iterrows():
            dest = os.path.join(eclipse_dir, row["filename"])
            want = row["sha256"] if isinstance(row["sha256"], str) and len(row["sha256"]) == 64 else None
            if os.path.exists(dest):
                if want is None or sha256sum(dest) == want:
                    print(f"[{TARGET} e{eclipse:02d}] have {row['filename']}", flush=True)
                    continue
                print(f"[{TARGET} e{eclipse:02d}] {row['filename']} corrupt, re-downloading", flush=True)
                os.remove(dest)
            print(f"[{TARGET} e{eclipse:02d}] downloading {row['filename']}", flush=True)
            dest = download_file(DATASET, row["relative_path"], eclipse_dir)
            if want is not None and sha256sum(dest) != want:
                raise ValueError(f"sha256 mismatch after download for {dest}")
        print(f"[{TARGET} e{eclipse:02d}] complete ({len(rows)} segments kept)", flush=True)
    print("all GJ uncal downloaded and kept", flush=True)


if __name__ == "__main__":
    main()

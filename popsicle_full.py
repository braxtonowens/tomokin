"""Fetch whole POPSICLE volumes (tomogram + every class mask) with the official split.

Split and file list: POPSICLE's HF metadata (biohub/popsicle, <set>/Croissant/{runs,tomograms,segmentations}.csv,
saved in data/popsicle_hf/). Bacterial: 68 train / 12 test; yeast: 16 train / 4 test.
Tomogram: the listed wbp-raw reconstruction, read from the portal's MRC next to the listed zarr (the same file our
earlier slices came from), stored as float32 (float16 changed features by ~1% through bf16 inference). Masks: every listed class zarr (level 0, full resolution), stored as
one uint8 bit field (bit i = class i of CLASSES[set]) so overlapping masks stay exact.
Output: data/popsicle_full/<set>/<run_name>/{tomo.npy, labels.npy, meta.json}
"""

import argparse
import csv
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import zarr

from popsicle_fetch import mrc_header, get

BASE = "https://files.cryoetdataportal.cziscience.com/"
HF = Path("data/popsicle_hf")
OUT = Path("data/popsicle_full")
CLASSES = {"bacterial": ["membrane", "flagellum", "inclusion", "intermembrane-space", "cytosole"],
           "yeast": None}  # yeast: taken from the metadata (sorted)


def rows(set_, name):
    return list(csv.DictReader(open(HF / set_ / "Croissant" / f"{name}.csv")))


def read_mrc(url):
    (nz, ny, nx), dt, off = mrc_header(url)
    plane = ny * nx * np.dtype(dt).itemsize
    chunk = 32
    out = np.empty((nz, ny, nx), np.float32)

    def part(z0):
        z1 = min(nz, z0 + chunk)
        b = np.frombuffer(get(url, off + z0 * plane, (z1 - z0) * plane), dt).reshape(z1 - z0, ny, nx)
        out[z0:z1] = b.astype(np.float32)
    with ThreadPoolExecutor(8) as ex:
        list(ex.map(part, range(0, nz, chunk)))
    return out


def fetch(set_, run, split, tomo, segs, classes):
    name = tomo["url"].split("/")[1]
    od = OUT / set_ / name
    if (od / "meta.json").exists():
        return f"{name}: exists"
    od.mkdir(parents=True, exist_ok=True)
    url = BASE + tomo["url"].replace(".zarr", ".mrc")
    vol = read_mrc(url)
    lab = np.zeros(vol.shape, np.uint8)
    present = []
    for s in segs:
        m = zarr.open_group(BASE + s["url"], mode="r")["0"][:]
        if m.shape != vol.shape:
            raise ValueError(f"{name} {s['name']}: mask {m.shape} vs tomogram {vol.shape}")
        lab |= ((m > 0).astype(np.uint8) << classes.index(s["name"]))
        present.append(s["name"])
    np.save(od / "tomo.npy", vol)
    np.save(od / "labels.npy", lab)
    json.dump(dict(run=name, portal_run_id=int(run["portal_run_id"]), split=split, classes=classes,
                   present=sorted(present), voxel=float(tomo["voxel_size"]), tomo_url=url,
                   shape=list(vol.shape)), open(od / "meta.json", "w"), indent=1)
    return f"{name} ({split}): {vol.shape}, classes {sorted(present)}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", nargs="+", default=["bacterial", "yeast"])
    ap.add_argument("--workers", type=int, default=3)
    args = ap.parse_args()
    for set_ in args.sets:
        runs = {r["name"]: r for r in rows(set_, "runs")}
        tomos = {r["run"]: r for r in rows(set_, "tomograms")}
        segs = {}
        for s in rows(set_, "segmentations"):
            segs.setdefault(s["run"], []).append(s)
        classes = CLASSES[set_] or sorted({s["name"] for v in segs.values() for s in v})
        print(f"{set_}: {len(runs)} runs, classes {classes}", flush=True)
        with ThreadPoolExecutor(args.workers) as ex:
            futs = [ex.submit(fetch, set_, runs[k], runs[k]["split"], tomos[k], segs.get(k, []), classes) for k in runs]
            for f in futs:
                print("  " + f.result(), flush=True)


if __name__ == "__main__":
    main()

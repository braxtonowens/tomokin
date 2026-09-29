"""Fetch POPSICLE segmentation slices from the CryoET Data Portal (no whole-volume downloads).

For every run in a POPSICLE deposition (10350 bacterial, 10351 yeast):
  - label volume at 4x binning from the masks' zarr level 2 (one small chunk per class) ->
    choose up to N_TARGET target slices with the largest labelled fraction, >= MIN_SEP apart,
    inside the central 70% of z (see pick_targets)
  - the matching tomogram (same voxel spacing and shape as the masks) as MRC; target slices plus
    the rc_exact template slices (RC position_bias.py rule: N_REF evenly spaced z in
    [nz/5, nz - nz/5), excluding |dz| <= 15 of every target) via HTTP range reads
  - full-resolution label slices at the targets (0 = unlabelled/background, else class index;
    overlapping masks resolved by PRIORITY, thin/specific structures first)
Writes data/popsicle/<deposition>/<run>.npz.
"""

import argparse
import struct
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numcodecs
import numpy as np
import requests
from cryoet_data_portal import Annotation, AnnotationFile, Client, Tomogram

N_TARGET, MIN_SEP, N_REF, EXCLUDE = 3, 40, 12, 15
PRIORITY = ["membrane", "nuclear envelope", "bacterial-type flagellum", "dense body", "vesicle",
            "mitochondrion", "membrane-enclosed lumen", "nucleus", "periplasmic space", "cytoplasm"]
MRC_DTYPES = {0: np.int8, 1: np.int16, 2: np.float32, 6: np.uint16, 12: np.float16}
S = requests.Session()


def get(url, start=None, length=None):
    h = {} if start is None else {"Range": f"bytes={start}-{start + length - 1}"}
    for _ in range(4):
        try:
            r = S.get(url, headers=h, timeout=120)
            if r.status_code == 404:
                break
            r.raise_for_status()
            if length is None or len(r.content) == length:
                return r.content
        except requests.RequestException:
            pass
    raise IOError(f"failed: {url} [{start}+{length}]")


def mrc_header(url):
    h = get(url, 0, 1024)
    nx, ny, nz, mode = struct.unpack("<4i", h[:16])
    return (nz, ny, nx), np.dtype(MRC_DTYPES[mode]).newbyteorder("<"), 1024 + struct.unpack("<i", h[92:96])[0]


def mrc_slices(url, zs):
    (nz, ny, nx), dt, off = mrc_header(url)
    plane = nx * ny * dt.itemsize
    with ThreadPoolExecutor(8) as ex:
        bufs = list(ex.map(lambda z: get(url, off + z * plane, plane), zs))
    return np.stack([np.frombuffer(b, dt).reshape(ny, nx) for b in bufs])


def zarr_level2(url):
    meta = requests.get(f"{url}/2/.zarray", timeout=60).json()
    assert all(s <= c for s, c in zip(meta["shape"], meta["chunks"])), meta
    key = meta.get("dimension_separator", ".").join("0" * len(meta["shape"]))
    raw = numcodecs.get_codec(meta["compressor"]).decode(get(f"{url}/2/{key}"))
    return np.frombuffer(raw, meta["dtype"]).reshape(meta["chunks"])[tuple(slice(0, s) for s in meta["shape"])]


def compose(masks):
    """masks: {class: bool array}. Label map with the PRIORITY order (index into classes list)."""
    classes = [c for c in PRIORITY if c in masks]
    lab = np.zeros(next(iter(masks.values())).shape, np.uint8)
    for c in reversed(classes):  # lowest priority first, so higher priority overwrites
        lab[masks[c]] = PRIORITY.index(c) + 1
    return lab


def pick_targets(lab2, nz):
    """lab2: label volume at 4x binning. Annotations can cover only part of z, and unlabelled
    voxels count as background, so rank slices by labelled fraction (#classes covering >= 0.5% of
    the slice as a tie-break) and keep only slices with >= half the best labelled fraction."""
    fg = (lab2 > 0).mean(axis=(1, 2))
    ncls = [(np.bincount(s.ravel(), minlength=len(PRIORITY) + 1)[1:] / s.size >= 0.005).sum() for s in lab2]
    score = fg + 0.01 * np.array(ncls)
    zs0 = np.clip(np.arange(lab2.shape[0]) * 4 + 2, 0, nz - 1)
    ok = (zs0 >= 0.15 * nz) & (zs0 <= 0.85 * nz)
    ok &= fg >= 0.5 * fg[ok].max()
    chosen = []
    for i in np.argsort(score)[::-1]:
        if ok[i] and all(abs(zs0[i] - c) >= MIN_SEP for c in chosen):
            chosen.append(int(zs0[i]))
        if len(chosen) == N_TARGET:
            break
    return sorted(chosen)


def ref_zs(nz, targets):
    cand = [z for z in range(nz // 5, nz - nz // 5) if all(abs(z - t) > EXCLUDE for t in targets)]
    return [cand[i] for i in np.linspace(0, len(cand) - 1, N_REF).astype(int)]


def fetch_run(client, run_id, anns, out):
    if out.exists():
        return f"{out.stem}: cached"
    files = {}
    for a in anns:
        fs = AnnotationFile.find(client, [AnnotationFile.annotation_shape.annotation_id == a.id])
        files[a.object_name] = {f.format: f for f in fs}
    vs_id = next(iter(files.values()))["mrc"].tomogram_voxel_spacing_id
    mask_shape = mrc_header(next(iter(files.values()))["mrc"].https_path)[0]
    tomos = [t for t in Tomogram.find(client, [Tomogram.tomogram_voxel_spacing_id == vs_id])
             if (t.size_z, t.size_y, t.size_x) == mask_shape]
    if not tomos:
        return f"run {run_id}: no tomogram matching mask shape {mask_shape}"
    tomo = sorted(tomos, key=lambda t: (t.processing != "raw", t.id))[0]
    nz = mask_shape[0]

    lab2 = compose({c: zarr_level2(f["zarr"].https_path) > 0 for c, f in files.items()})
    targets = pick_targets(lab2, nz)
    refs = ref_zs(nz, targets)
    tomo_sl = mrc_slices(tomo.https_mrc_file, targets + refs).astype(np.float32)
    labels = compose({c: mrc_slices(f["mrc"].https_path, targets) > 0 for c, f in files.items()})
    nt = len(targets)
    np.savez_compressed(out, targets=tomo_sl[:nt], refs=tomo_sl[nt:], labels=labels,
                        z_targets=targets, z_refs=refs, classes=np.array(PRIORITY), nz=nz,
                        voxel=tomo.voxel_spacing, tomo_id=tomo.id, run_name=out.stem,
                        dataset_id=tomo.run.dataset_id if hasattr(tomo, "run") else -1,
                        present=np.array(sorted(files)))
    return f"{out.stem}: tomo {tomo.id} {mask_shape} @ {tomo.voxel_spacing} A, targets {targets}, classes {sorted(files)}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("deposition", type=int, choices=[10350, 10351])
    ap.add_argument("--out", type=Path, default=Path("data/popsicle"))
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    out = args.out / str(args.deposition)
    out.mkdir(parents=True, exist_ok=True)
    client = Client()
    by_run = {}
    for a in Annotation.find(client, [Annotation.deposition_id == args.deposition]):
        by_run.setdefault(a.run_id, []).append(a)
    names = {r: by_run[r][0].run.name for r in by_run}
    print(f"deposition {args.deposition}: {len(by_run)} runs")

    def job(r):
        try:
            return fetch_run(Client(), r, by_run[r], out / f"{names[r]}.npz")
        except Exception as e:
            return f"{names[r]}: FAILED {type(e).__name__}: {e}"

    with ThreadPoolExecutor(args.workers) as ex:
        for msg in ex.map(job, sorted(by_run)):
            print(msg, flush=True)


if __name__ == "__main__":
    main()

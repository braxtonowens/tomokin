"""Slab inputs for C-RADIO: more signal per image than one z-slice.

For each target / reference z, read the 24-slice block [z-12, z+12) (one contiguous MRC range)
and build the model input four ways:
  g1      the centre slice alone (baseline, replicated to 3 channels)
  g8      mean of the middle 8 slices [z-4, z+4), replicated to 3 channels
  g24     mean of all 24, replicated
  rgb24   R = mean[z-12, z-4), G = mean[z-4, z+4), B = mean[z+4, z+12)
Intensity: one percentile clip (0.5/99.5) over all channels of the input, so channel
differences are kept. Resize as RC (native long side rounded to 16).

For every input type, two corrections of one target pass (as in seg_unsup round 2):
  rc_<in>    minus the RC position_bias.py template (12 ref slabs, random flips + token rolls)
  zfix_<in>  minus the mean of the same 12 ref slabs at fixed xy
Ref slabs: see ref_zs (RC rule, excluding |dz| <= 15 + slab width from every target).
Adds the keys to data/popsicle_feats/<dep>/<run>.npz.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel

from popsicle_fetch import get, mrc_header

PATCH, HALF, N_REF, EXCLUDE = 16, 12, 12, 15 + 24
INPUTS = ["g1", "g8", "g24", "rgb24"]


def tomo_url(d):
    """MRC address from saved metadata (no portal API): probe the tomogram index and accept the
    file whose shape matches the labels."""
    base = (f"https://files.cryoetdataportal.cziscience.com/{int(d['dataset_id'])}/{d['run_name']}/"
            f"Reconstructions/VoxelSpacing{float(d['voxel']):.3f}/Tomograms")
    want = (int(d["nz"]),) + d["labels"].shape[1:]
    for idx in range(100, 106):
        url = f"{base}/{idx}/{d['run_name']}.mrc"
        try:
            if mrc_header(url)[0] == want:
                return url
        except IOError:
            continue
    raise IOError(f"no tomogram matching {want} under {base}")


def read_block(url, z0, z1):
    (nz, ny, nx), dt, off = mrc_header(url)
    plane = nx * ny * dt.itemsize
    buf = get(url, off + z0 * plane, (z1 - z0) * plane)
    return np.frombuffer(buf, dt).reshape(z1 - z0, ny, nx).astype(np.float32)


def channels(block):
    """block: (24, H, W) -> {input: (3, H, W)}"""
    r, g, b = block[:8].mean(0), block[8:16].mean(0), block[16:].mean(0)
    g1 = block[HALF]
    a = block.mean(0)
    return {"g1": np.stack([g1] * 3), "g8": np.stack([g] * 3), "g24": np.stack([a] * 3), "rgb24": np.stack([r, g, b])}


def to_input(x3, scale=1.0):
    lo, hi = np.percentile(x3, [0.5, 99.5])
    x = torch.from_numpy(np.clip((x3 - lo) / (hi - lo + 1e-8), 0, 1).astype(np.float32))[None]
    H, W = x.shape[-2:]
    res = round(scale * max(H, W) / PATCH) * PATCH
    s = res / max(H, W)
    h, w = max(PATCH, round(H * s / PATCH) * PATCH), max(PATCH, round(W * s / PATCH) * PATCH)
    return F.interpolate(x, size=(h, w), mode="bilinear", align_corners=False, antialias=True).clamp(0, 1)[0]


@torch.no_grad()
def embed(model, xs, bs=4):
    out = []
    for i in range(0, len(xs), bs):
        x = torch.stack(xs[i:i + bs]).cuda()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _, f = model(x)
        out.append(f.float().transpose(1, 2).reshape(len(x), -1, x.shape[-2] // PATCH, x.shape[-1] // PATCH).cpu())
    return torch.cat(out)


def rolled(xs, seed=0):
    """RC template draws (unrotated-grid only; rotated-grid draws consumed, not run)."""
    rng = np.random.default_rng(seed)
    out = []
    for x in xs:
        for parity in (0, 1):
            xt = torch.rot90(x, parity, dims=(-2, -1))
            lr, ud = rng.random() < 0.5, rng.random() < 0.5
            gh, gw = xt.shape[-2] // PATCH, xt.shape[-1] // PATCH
            shift = (int(rng.integers(gh)) * PATCH, int(rng.integers(gw)) * PATCH)
            if parity == 0:
                xt = xt.flip(-1) if lr else xt
                xt = xt.flip(-2) if ud else xt
                out.append(torch.roll(xt, shifts=shift, dims=(-2, -1)))
    return out


def ref_zs(nz, targets, h=HALF):
    """RC rule (N_REF evenly spaced in [nz/5, nz - nz/5)), excluding |dz| <= 15 + slab width from every
    target so ref slabs never share slices with a target slab. If that leaves fewer than N_REF
    candidates (thin tomograms), widen to the whole usable z range; never repeat a slice."""
    excl = 15 + 2 * h
    for lo, hi in [(max(nz // 5, h), min(nz - nz // 5, nz - h)), (h, nz - h)]:
        cand = [z for z in range(lo, hi) if all(abs(z - t) > excl for t in targets)]
        if len(cand) >= N_REF:
            break
    idx = np.unique(np.linspace(0, len(cand) - 1, N_REF).round().astype(int))
    return [cand[i] for i in idx]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("deposition", type=int, choices=[10350, 10351])
    ap.add_argument("--data", type=Path, default=Path("data/popsicle"))
    ap.add_argument("--feats", type=Path, default=Path("data/popsicle_feats"))
    ap.add_argument("--inputs", nargs="+", default=INPUTS, choices=INPUTS)
    ap.add_argument("--force", action="store_true", help="recompute even if the keys exist")
    ap.add_argument("--scale", type=float, default=1.0, help="model input long side = scale x native")
    args = ap.parse_args()
    model = AutoModel.from_pretrained("nvidia/C-RADIOv4-H", trust_remote_code=True).eval().cuda()
    for p in sorted((args.data / str(args.deposition)).glob("*.npz")):
        fp = args.feats / str(args.deposition) / p.name
        f = dict(np.load(fp))
        sfx = "" if args.scale == 1.0 else f"_s{args.scale:g}".replace(".", "")
        if not args.force and all(f"rc_{i}{sfx}" in f for i in args.inputs):
            continue
        d = np.load(p)
        url = tomo_url(d)
        nz = int(d["nz"])
        targets = [int(z) for z in d["z_targets"]]
        h = 4 if set(args.inputs) <= {"g1", "g8"} else HALF  # read only the 8 middle slices if enough
        refs = ref_zs(nz, targets, h)
        zs = targets + refs
        spans = [int(np.clip(z - h, 0, nz - 2 * h)) for z in zs]

        def build(z0):
            blk = read_block(url, z0, z0 + 2 * h)
            if h == HALF:
                return channels(blk)
            return {"g1": np.stack([blk[4]] * 3), "g8": np.stack([blk.mean(0)] * 3)}

        with ThreadPoolExecutor(4) as ex:
            blocks = list(ex.map(build, spans))
        for inp in args.inputs:
            xs = [to_input(b[inp], args.scale) for b in blocks]
            tgt = embed(model, xs[:len(targets)])
            T0 = embed(model, rolled(xs[len(targets):])).mean(0)
            Pfix = embed(model, xs[len(targets):]).mean(0)
            f[f"rc_{inp}{sfx}"] = (tgt - T0).half().numpy()
            f[f"zfix_{inp}{sfx}"] = (tgt - Pfix).half().numpy()
        f["z_refs_slab"] = np.array(refs)
        np.savez(fp, **f)
        print(f"  {p.stem}: targets {targets}, ref slabs {refs}", flush=True)


if __name__ == "__main__":
    main()

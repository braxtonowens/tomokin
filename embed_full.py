"""Embed every z-slice of the whole POPSICLE volumes (popsicle_full.py) with C-RADIOv4-H, rc_g8 recipe.

Per slice z: gray-8 slab = mean of slices [z-4, z+4) (clipped at the volume ends, as lora_finetune.Run.slabs),
to_input (0.5/99.5% clip, native resolution rounded to 16), one bf16 C-RADIO pass, minus the run's RC template
(12 gray-8 ref slabs evenly spaced in [nz/5, nz - nz/5), random flip + whole-token roll, seed 0; label-free, so no
exclusion around targets is needed now that every slice is embedded).
Storage: 256-d PCA per set (bacterial / yeast), basis fitted on train runs only (16 slices per run, all tokens
subsampled 1/3), fp16 memmap (nz, 256, gh, gw) per run. Train runs also get a second orientation (the input
rotated 90 deg before the pass; stored in the rotated grid) as a real augmentation, since the ViT is not rotation
equivariant.
Outputs: data/popsicle_full/<set>/<run>/{feats.npy, feats_rot90.npy (train), T.pt}; data/popsicle_full/<set>/pca.pt
Check (--check): for runs also in data/popsicle, the new pipeline with the OLD template refs must reproduce the stored
rc_g8 at the old target slices (differences come only from float16 tomogram storage).
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from popsicle_slab import N_REF, PATCH, embed, rolled, to_input

ROOT = Path("data/popsicle_full")
PC = 256


def slab(vol, z):
    z0 = int(np.clip(z - 4, 0, vol.shape[0] - 8))
    return vol[z0:z0 + 8].astype(np.float32).mean(0)


def inp(s, rot=False):
    x = to_input(np.stack([s] * 3))
    return torch.rot90(x, 1, (-2, -1)).contiguous() if rot else x


def ref_z(nz):
    lo, hi = max(nz // 5, 4), min(nz - nz // 5, nz - 4)
    return [int(z) for z in np.unique(np.linspace(lo, hi - 1, N_REF).round().astype(int))]


def template(model, vol, zs, rot=False):
    return embed(model, rolled([inp(slab(vol, z), rot) for z in zs])).mean(0)


def runs(set_):
    return [json.load(open(p)) | {"dir": p.parent} for p in sorted((ROOT / set_).glob("*/meta.json"))]


def load_model():
    from transformers import AutoModel
    return AutoModel.from_pretrained("nvidia/C-RADIOv4-H", trust_remote_code=True).eval().cuda()


def fit_pca(model, set_):
    path = ROOT / set_ / "pca.pt"
    if path.exists():
        return torch.load(path)
    X = []
    for r in [r for r in runs(set_) if r["split"] == "train"]:
        vol = np.load(r["dir"] / "tomo.npy", mmap_mode="r")
        nz = vol.shape[0]
        T = get_T(model, r, vol)
        zs = np.linspace(nz // 10, nz - 1 - nz // 10, 16).round().astype(int)
        f = embed(model, [inp(slab(vol, z)) for z in zs]) - T
        X.append(f.permute(0, 2, 3, 1).reshape(-1, f.shape[1])[::3])
    X = torch.cat(X)
    mu = X.mean(0)
    torch.manual_seed(0)
    _, S, V = torch.pca_lowrank((X - mu).cuda(), q=PC, center=False, niter=6)
    ev = float((S ** 2).sum() / (X - mu).pow(2).sum())
    basis = {"mu": mu, "V": V[:, :PC].cpu().contiguous(), "explained": ev, "n_tokens": len(X)}
    torch.save(basis, path)
    print(f"{set_}: PCA from {len(X)} train tokens, top {PC} explain {ev:.3f}", flush=True)
    return basis


def get_T(model, r, vol, rot=False):
    f = r["dir"] / ("T_rot90.pt" if rot else "T.pt")
    if not f.exists():
        torch.save(template(model, vol, ref_z(vol.shape[0]), rot), f)
    return torch.load(f)


@torch.no_grad()
def embed_run(model, r, basis, rot=False, bs=8):
    out = r["dir"] / ("feats_rot90.npy" if rot else "feats.npy")
    if out.exists():
        return
    vol = np.load(r["dir"] / "tomo.npy", mmap_mode="r")
    nz = vol.shape[0]
    T = get_T(model, r, vol, rot).cuda()
    mu, V = basis["mu"].cuda(), basis["V"].cuda()
    mm = None
    for z0 in range(0, nz, bs):
        zs = list(range(z0, min(nz, z0 + bs)))
        f = embed(model, [inp(slab(vol, z), rot) for z in zs]).cuda() - T
        p = torch.einsum("bchw,cp->bphw", f - mu[:, None, None], V).half().cpu().numpy()
        if mm is None:
            mm = np.lib.format.open_memmap(out.with_suffix(".tmp.npy"), mode="w+", dtype=np.float16,
                                           shape=(nz,) + p.shape[1:])
        mm[z0:z0 + len(zs)] = p
    mm.flush()
    del mm
    out.with_suffix(".tmp.npy").rename(out)


def check(model, set_):
    """Old target slices with the old template refs -> compare with stored rc_g8 (1280-d, before PCA)."""
    from popsicle_slab import ref_zs
    dep = {"bacterial": "10350", "yeast": "10351"}[set_]
    for r in runs(set_)[:2]:
        old = Path("data/popsicle") / dep / f"{r['run']}.npz"
        if not old.exists():
            continue
        d = np.load(old)
        vol = np.load(r["dir"] / "tomo.npy", mmap_mode="r")
        zt = [int(z) for z in d["z_targets"]]
        T = template(model, vol, ref_zs(vol.shape[0], zt, 4))
        f = embed(model, [inp(slab(vol, z)) for z in zt]) - T
        ref = torch.from_numpy(np.load(Path("data/popsicle_feats") / dep / old.name)["rc_g8"]).float()
        print(f"  check {r['run']}: |new - stored rc_g8| / |stored| = {float((f - ref).norm() / ref.norm()):.4f}",
              flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", nargs="+", default=["bacterial", "yeast"])
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    model = load_model()
    for set_ in args.sets:
        if args.check:
            check(model, set_)
            continue
        basis = fit_pca(model, set_)
        rr = runs(set_)
        for i, r in enumerate(rr):
            embed_run(model, r, basis)
            if r["split"] == "train":
                embed_run(model, r, basis, rot=True)
            print(f"  {set_} {i + 1}/{len(rr)} {r['run']} ({r['split']}): {r['shape']}", flush=True)


if __name__ == "__main__":
    main()

"""Per-slice spectral entropy of C-RADIOv4-H features for the entropy GUI (webapp/frontend/entropy.html).

Tomograms: 5 whole POPSICLE volumes (data/popsicle_full), 3 bacterial (different source datasets) + 2 yeast.
Features: every slice embedded with the rc_g8 recipe (embed_full.py: gray-8 slab, native resolution, bf16, minus the
run's RC template), full 1280-d. Smoothed copy: Gaussian along z (sigma SIGMA slices, truncated at 3 sigma,
renormalized at the volume ends).
Entropy of slice z (raw and smoothed): 2000 patch embeddings (the same token positions on every slice, seed 0),
L2-normalized rows X, Gram matrix G = X X^T, G / trace(G), eigenvalues by eigvalsh, Shannon entropy
H = -sum lambda ln lambda over the positive eigenvalues (nats; max ln 2000 = 7.60).
Stored per tomogram in outputs/entropy/<run>/: entropy.json, pca35.npy (nz, 35, gh, gw) fp16 = top-35 PCA of the
smoothed features (basis fitted on 256 random tokens per slice), basis.pt.
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from embed_full import inp, load_model, ref_z, slab, template
from popsicle_slab import embed

OUT = Path("outputs/entropy")
TOMOS = [("bacterial", "dga2016-01-13-17"), ("bacterial", "ycw2012-09-23-39"), ("bacterial", "ycw2013-09-10-39"),
         ("yeast", "TS_0001"), ("yeast", "TS_030")]
N_TOK, SIGMA, K = 2000, 2.0, 35


def entropy(X):
    """X (n, D) -> Shannon entropy of the trace-normalized Gram spectrum of the L2-normalized rows."""
    X = torch.nn.functional.normalize(X.float(), dim=1)
    G = X @ X.T
    lam = torch.linalg.eigvalsh(G / torch.trace(G))
    lam = lam[lam > 1e-12]
    return float(-(lam * lam.log()).sum())


@torch.no_grad()
def run_one(model, set_, run, sigma, bs=8):
    out = OUT / run
    if (out / "entropy.json").exists():
        print(f"{run}: exists")
        return
    out.mkdir(parents=True, exist_ok=True)
    d = Path("data/popsicle_full") / set_ / run
    vol = np.load(d / "tomo.npy", mmap_mode="r")
    nz = vol.shape[0]
    T = template(model, vol, ref_z(nz))
    feats = []
    for z0 in range(0, nz, bs):
        f = embed(model, [inp(slab(vol, z)) for z in range(z0, min(nz, z0 + bs))]) - T
        feats.append(f.half())
    F = torch.cat(feats)  # (nz, 1280, gh, gw) fp16, cpu
    _, D, gh, gw = F.shape
    N = gh * gw
    idx = torch.from_numpy(np.random.default_rng(0).choice(N, min(N_TOK, N), replace=False))
    R = int(math.ceil(3 * sigma))
    w = torch.exp(-torch.arange(-R, R + 1).float() ** 2 / (2 * sigma ** 2))

    def smooth(z):
        ks = [k for k in range(-R, R + 1) if 0 <= z + k < nz]
        ww = w[[k + R for k in ks]].cuda()
        stack = F[z + ks[0]:z + ks[-1] + 1].cuda().float()
        return torch.einsum("k,kdhw->dhw", ww / ww.sum(), stack)

    H_raw, H_s, samp = [], [], []
    rng = np.random.default_rng(1)
    for z in range(nz):
        H_raw.append(entropy(F[z].cuda().flatten(1)[:, idx.cuda()].T))
        fs = smooth(z).flatten(1)
        H_s.append(entropy(fs[:, idx.cuda()].T))
        samp.append(fs[:, torch.from_numpy(rng.choice(N, 256, replace=False)).cuda()].T.cpu())
    S = torch.cat(samp)
    mu = S.mean(0)
    torch.manual_seed(0)
    _, sv, V = torch.pca_lowrank((S - mu).cuda(), q=K, center=False, niter=6)
    ev = float((sv ** 2).sum() / ((S - mu) ** 2).sum())
    V, mu = V[:, :K], mu.cuda()
    P = np.lib.format.open_memmap(out / "pca35.npy", mode="w+", dtype=np.float16, shape=(nz, K, gh, gw))
    for z in range(nz):
        P[z] = torch.einsum("dhw,dk->khw", smooth(z) - mu[:, None, None], V).half().cpu().numpy()
    P.flush()
    torch.save({"mu": mu.cpu(), "V": V.cpu(), "explained": ev}, out / "basis.pt")
    info = dict(run=run, set=set_, nz=nz, grid=[gh, gw], shape=list(vol.shape), n_tokens=int(len(idx)),
                sigma=sigma, H_raw=H_raw, H_smooth=H_s, argmin_raw=int(np.argmin(H_raw)),
                argmin_smooth=int(np.argmin(H_s)), pca_explained=ev,
                split=json.load(open(d / "meta.json"))["split"])
    json.dump(info, open(out / "entropy.json", "w"))
    print(f"{run}: nz {nz}, grid {gh}x{gw}, H_raw {min(H_raw):.3f}-{max(H_raw):.3f} (min z {info['argmin_raw']}), "
          f"H_smooth {min(H_s):.3f}-{max(H_s):.3f} (min z {info['argmin_smooth']}), PCA-35 explains {ev:.3f}",
          flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sigma", type=float, default=SIGMA)
    args = ap.parse_args()
    model = load_model()
    for set_, run in TOMOS:
        run_one(model, set_, run, args.sigma)


if __name__ == "__main__":
    main()

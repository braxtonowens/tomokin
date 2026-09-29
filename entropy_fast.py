"""Fast per-slice spectral entropy from the stored PCA-256 features (no C-RADIO pass), same definition as
entropy_cache.py otherwise.

The stored features are p = (f - mu) V (V: 1280 x 256 orthonormal, 94% of variance), so f ~ mu + V p lies in the
257-d span of [V, mu]. With B an orthonormal basis of that span, every reconstructed token is y = B^T f (257-d) and
||y|| = ||f_reconstructed||, so the L2-normalized Gram matrix of the 2000 sampled tokens (same positions as
entropy_cache.py, seed 0) has the same nonzero eigenvalues as the 257 x 257 matrix Y_n^T Y_n (Y_n = normalized
rows). Entropy is computed on raw and z-smoothed features (Gaussian sigma 2 slices along z; smoothing is linear, so it
is applied to p). PCA-35 overlay = top-35 PCs of the smoothed p (equal to PCA of the reconstructed 1280-d features,
since V is orthonormal). Approximation: the 6% of variance outside the 256 PCs is dropped.
--check compares with the exact caches (outputs/entropy/*) that already exist.
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

FULL, OUT = Path("data/popsicle_full"), Path("outputs/entropy")
N_TOK, SIGMA, K = 2000, 2.0, 35
DEV = "cuda"


def basis_for(set_):
    b = torch.load(FULL / ("bacterial" if set_ == "hylemonella_extra" else set_) / "pca.pt")
    mu, V = b["mu"].float(), b["V"].float()
    Q, _ = torch.linalg.qr(torch.cat([V, mu[:, None]], 1))  # (1280, 257)
    return mu.to(DEV), V.to(DEV), Q.to(DEV)


def entropies(P, idx, mu, V, Q, bs=64):
    """P (nz, 256, n) sampled PCA coords (fp16 cuda) -> per-slice entropy (nats)."""
    A = (V.T @ Q)  # (256, 257): y = (V p + mu)^T Q = p^T (V^T Q) + mu^T Q
    m = mu @ Q     # (257,)
    out = []
    for z0 in range(0, len(P), bs):
        Y = torch.einsum("bdn,dk->bnk", P[z0:z0 + bs].float(), A) + m  # (b, n, 257)
        Y = torch.nn.functional.normalize(Y, dim=2)
        S = Y.transpose(1, 2) @ Y
        lam = torch.linalg.eigvalsh(S / S.diagonal(dim1=1, dim2=2).sum(1)[:, None, None])
        lam = lam.clamp_min(0)
        H = -(torch.where(lam > 1e-12, lam * lam.clamp_min(1e-30).log(), torch.zeros_like(lam))).sum(1)
        out += H.tolist()
    return out


def smooth_z(F, sigma, chunk=65536):
    """Gaussian smoothing along z (dim 0) of a cuda tensor, renormalized at the ends; fp16 out, chunked columns."""
    R = int(math.ceil(3 * sigma))
    w = torch.exp(-torch.arange(-R, R + 1, device=F.device).float() ** 2 / (2 * sigma ** 2))[None, None]
    nz = F.shape[0]
    X = F.reshape(nz, -1)
    den = torch.nn.functional.conv1d(torch.nn.functional.pad(torch.ones(1, 1, nz, device=F.device), (R, R)), w)[0, 0]
    out = torch.empty_like(X)
    for c0 in range(0, X.shape[1], chunk):
        x = X[:, c0:c0 + chunk].float().T[:, None]
        out[:, c0:c0 + chunk] = (torch.nn.functional.conv1d(torch.nn.functional.pad(x, (R, R)), w)[:, 0] / den).T.half()
    return out.reshape(F.shape)


@torch.no_grad()
def run_one(set_, run, out_dir, sigma=SIGMA):
    d = FULL / set_ / run
    f = np.load(d / "feats.npy", mmap_mode="r")
    nz, D, gh, gw = f.shape
    N = gh * gw
    idx = torch.from_numpy(np.random.default_rng(0).choice(N, min(N_TOK, N), replace=False)).to(DEV)
    mu, V, Q = basis_for(set_)
    Fz = torch.from_numpy(np.ascontiguousarray(f)).to(DEV)  # (nz, 256, gh, gw) fp16
    Fs = smooth_z(Fz, sigma)
    H_raw = entropies(Fz.flatten(2)[:, :, idx], idx, mu, V, Q)
    H_s = entropies(Fs.flatten(2)[:, :, idx], idx, mu, V, Q)
    # PCA-35 of the smoothed features for the overlay
    rng = np.random.default_rng(1)
    samp = torch.cat([Fs[z].flatten(1)[:, torch.from_numpy(rng.choice(N, 256, replace=False)).to(DEV)].T.float()
                      for z in range(nz)])
    m35 = samp.mean(0)
    torch.manual_seed(0)
    _, sv, W = torch.pca_lowrank(samp - m35, q=K, center=False, niter=6)
    ev_rel = float((sv ** 2).sum() / ((samp - m35) ** 2).sum())
    out_dir.mkdir(parents=True, exist_ok=True)
    P35 = np.lib.format.open_memmap(out_dir / "pca35.npy", mode="w+", dtype=np.float16, shape=(nz, K, gh, gw))
    for z0 in range(0, nz, 64):
        x = Fs[z0:z0 + 64].float() - m35[None, :, None, None]
        P35[z0:z0 + len(x)] = torch.einsum("bdhw,dk->bkhw", x, W[:, :K]).half().cpu().numpy()
    P35.flush()
    meta = json.load(open(d / "meta.json"))
    info = dict(run=run, set=set_, nz=nz, grid=[gh, gw], shape=meta["shape"], n_tokens=int(len(idx)), sigma=sigma,
                H_raw=H_raw, H_smooth=H_s, argmin_raw=int(np.argmin(H_raw)), argmin_smooth=int(np.argmin(H_s)),
                pca_explained=ev_rel * 0.94, split=meta["split"], method="fast (PCA-256 features)")
    json.dump(info, open(out_dir / "entropy.json", "w"))
    return info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="compare with the existing exact caches")
    ap.add_argument("--runs", nargs="*", default=[], help="set/run pairs, e.g. bacterial/ycw2012-09-23-21")
    args = ap.parse_args()
    if args.check:
        import sys
        sys.path.insert(0, "webapp")
        from server import best_slice  # noqa: E402  (the page's slice rule)
        for f in sorted(OUT.glob("*/entropy.json")):
            ex = json.load(open(f))
            if ex.get("method", "").startswith("fast"):
                continue
            import time
            t = time.time()
            fa = run_one(ex["set"], ex["run"], Path("outputs/entropy_fast_check") / ex["run"])
            dt = time.time() - t
            a, b = np.array(ex["H_smooth"]), np.array(fa["H_smooth"])
            print(f"{ex['run']:18s} {dt:5.1f} s  r={np.corrcoef(a, b)[0, 1]:.4f}  mean offset {np.mean(a - b):+.3f} nats  "
                  f"best slice exact {best_slice(a)} fast {best_slice(b)}", flush=True)
        return
    for s in args.runs:
        set_, run = s.split("/")
        i = run_one(set_, run, OUT / run)
        print(f"{run}: nz {i['nz']}, H_smooth {min(i['H_smooth']):.3f}-{max(i['H_smooth']):.3f}", flush=True)


if __name__ == "__main__":
    main()

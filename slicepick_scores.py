"""torch-slicepick score for every slice of a tomogram (the package only returns the best index).

Uses the package's own measures and defaults (torch_slicepick.slicepick):
  H = 8-bit entropy / 8, L = Laplacian variance, E = edge density; L, E robust-normalized (5-95%)
  base = 0.3 H + 0.5 L + 0.2 E;  prior W = max(0, 1 - |z - centre| / (D/3));  score = base * W
Plotted per slice (thickness 1) and, as the package actually does it, per 4-slice slab (mean-pooled),
with the slice the package picks.
  python slicepick_scores.py data/10351/TS_030.mrc
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mrcfile
import numpy as np
import torch
from torch_slicepick import pick_slice_index_from_volume
from torch_slicepick.slicepick import _edge_density, _entropy_8bit, _lap_var, _quantile_approx, _robust_minmax_normalize


def scores(vol32, vol8, thickness, a=0.3, b=0.5, g=0.2, halfwidth=1 / 3):
    D = vol32.shape[0]
    bounds = [(z, min(z + thickness, D)) for z in range(0, D, thickness)]
    H, L, E, centres = [], [], [], []
    for z0, z1 in bounds:
        p8 = vol8[z0:z1].float().mean(0).round().clamp(0, 255).to(torch.uint8)
        p32 = vol32[z0:z1].mean(0)
        H.append(_entropy_8bit(p8))
        L.append(_lap_var(p32))
        E.append(_edge_density(p32))
        centres.append((z0 + z1 - 1) / 2)
    dev = vol32.device
    Ht = torch.tensor(H, device=dev) / 8
    Lt = _robust_minmax_normalize(torch.tensor(L, device=dev))
    Et = _robust_minmax_normalize(torch.tensor(E, device=dev))
    base = a * Ht + b * Lt + g * Et
    c = np.array(centres)
    W = np.clip(1 - np.abs(c - (D - 1) / 2) / max(1.0, D * halfwidth), 0, 1)
    return c, {"entropy H": Ht.cpu().numpy(), "Laplacian var L (norm.)": Lt.cpu().numpy(),
               "edge density E (norm.)": Et.cpu().numpy(), "base": base.cpu().numpy(), "score": base.cpu().numpy() * W}


def compute(vol: np.ndarray, device="cuda") -> dict:
    """All torch-slicepick scores for a [D, H, W] volume: per slice and per 4-slice slab, plus the
    package's own pick. Arrays are numpy."""
    dev = torch.device(device)
    vol32 = torch.from_numpy(np.asarray(vol, dtype=np.float32)).to(dev)
    vmin, vmax = _quantile_approx(vol32, 0.01), _quantile_approx(vol32, 0.99)
    vol8 = ((vol32 - vmin) / (vmax - vmin).clamp_min(1e-6) * 255).clamp(0, 255).to(torch.uint8)
    zs, per = scores(vol32, vol8, 1)
    cs, slab = scores(vol32, vol8, 4)
    del vol32, vol8
    best = pick_slice_index_from_volume(np.asarray(vol, dtype=np.float32), device=device)
    return {"z": zs, **{f"slice/{k}": v for k, v in per.items()}, "slab_z": cs, "slab_score": slab["score"],
            "pick": np.array(best)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tomogram", type=Path)
    ap.add_argument("--out", type=Path, default=Path("outputs/slicepick"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    vol = mrcfile.mmap(args.tomogram, mode="r", permissive=True).data
    r = compute(vol)
    zs, cs, best = r["z"], r["slab_z"], int(r["pick"])
    per = {k[6:]: v for k, v in r.items() if k.startswith("slice/")}
    slab = {"score": r["slab_score"]}

    fig, (a1, a2) = plt.subplots(2, 1, figsize=(12, 7.5), sharex=True)
    a1.plot(zs, per["score"], color="k", lw=1.4, label="score = base × centre prior (per slice)")
    a1.plot(zs, per["base"], color="0.6", lw=1, label="base (no prior)")
    a1.plot(cs, slab["score"], "o-", ms=2.5, color="#d93025", lw=1, label="score per 4-slice slab (what the package ranks)")
    a1.axvline(best, color="#1a73e8", ls="--", lw=1.2, label=f"package's pick: z={best}")
    a1.set_ylabel("score")
    a1.legend(fontsize=8)
    a1.set_title(f"torch-slicepick scores — {args.tomogram.name} ({vol.shape[0]} slices)")
    for k, col in [("entropy H", "#188038"), ("Laplacian var L (norm.)", "#e8710a"), ("edge density E (norm.)", "#9334e6")]:
        a2.plot(zs, per[k], color=col, lw=1, label=k)
    a2.set_xlabel("slice z")
    a2.set_ylabel("component (per slice)")
    a2.legend(fontsize=8)
    for a in (a1, a2):
        a.grid(alpha=0.3)
    fig.tight_layout()
    path = args.out / f"slicepick_{args.tomogram.stem}.png"
    fig.savefig(path, dpi=100)
    np.savetxt(args.out / f"slicepick_{args.tomogram.stem}.csv", np.column_stack([zs] + [per[k] for k in per]),
               delimiter=",", header="z," + ",".join(per), comments="")
    print(f"package pick z={best}; best per-slice score z={int(zs[np.argmax(per['score'])])}; wrote {path}")


if __name__ == "__main__":
    main()

"""C-RADIO embedding of chosen tomogram slices for the GUI: shared 256-d PCA per tomogram, on-disk
cache, linear interpolation between embedded slices.

Pipeline (the project's current choice):
  input      gray-8 slab = mean of the 8 slices [z-4, z+4) (clipped at the volume ends), clip
             0.5/99.5 percentile, native resolution (long side rounded to 16) -- popsicle_slab.to_input
  template   RC position_bias.py template, once per tomogram: 12 reference slabs evenly spaced in
             [nz/5, nz - nz/5), each randomly flipped + rolled by whole tokens (RC draws, seed 0),
             mean of their features (unrotated-grid template only)
  features   one C-RADIOv4-H pass on the slab minus the template, (1280, gh, gw)
  basis      one PCA basis per tomogram (mean + top 256 components), fitted on all tokens of up to
             24 embedded slices spread over the first selection; every slice is stored as its
             projection onto it, (256, gh, gw) float16. Sharing the basis keeps the channels
             comparable across slices, which interpolation and stable display colours need.
  display    RGB = shared components 1-3, stretched with the 1st-99th percentiles of the basis
             sample (same colour scale for every slice of the tomogram)
  interpolate  a slice between two embedded slices (gap <= MAX_GAP) = linear blend of their
             256-d features
Cache: outputs/gui_cache/<tomogram stem>/{template.npy, basis.npz, z0123.npy, z0123_pca.png}.
"""

from pathlib import Path

import numpy as np
import torch

from popsicle_slab import embed, rolled, to_input

CACHE = Path("outputs/gui_cache")
N_REF = 12
HALF = 4  # gray-8 slab = [z - 4, z + 4)
N_PC = 256
N_BASIS = 24
MAX_GAP = 4


class Embedder:
    def __init__(self, cache_root: Path = CACHE):
        self.cache_root = cache_root
        self.model = None
        self._basis = {}

    def load_model(self):
        if self.model is None:
            from transformers import AutoModel
            self.model = AutoModel.from_pretrained("nvidia/C-RADIOv4-H", trust_remote_code=True).eval().cuda()
        return self.model

    def cache_dir(self, name: str) -> Path:
        d = self.cache_root / name
        d.mkdir(parents=True, exist_ok=True)
        return d

    def cached(self, name: str) -> set[int]:
        d = self.cache_root / name
        return {int(p.stem[1:]) for p in d.glob("z[0-9][0-9][0-9][0-9].npy")} if d.exists() else set()

    # ------------------------------------------------------------ inputs and template
    @staticmethod
    def slab(vol, z: int) -> np.ndarray:
        nz = vol.shape[0]
        z0 = int(np.clip(z - HALF, 0, nz - 2 * HALF))
        g = np.asarray(vol[z0:z0 + 2 * HALF], dtype=np.float32).mean(0)
        return np.stack([g] * 3)

    @staticmethod
    def ref_zs(nz: int) -> list[int]:
        lo, hi = max(nz // 5, HALF), min(nz - nz // 5, nz - HALF)
        return [int(z) for z in np.linspace(lo, hi - 1, N_REF).round()]

    def template(self, name: str, vol) -> torch.Tensor:
        p = self.cache_dir(name) / "template.npy"
        if p.exists():
            return torch.from_numpy(np.load(p)).float()
        xs = [to_input(self.slab(vol, z)) for z in self.ref_zs(vol.shape[0])]
        T = embed(self.load_model(), rolled(xs)).mean(0)
        np.save(p, T.half().numpy())
        return T

    def full_features(self, name: str, vol, zs: list[int]) -> torch.Tensor:
        """(len(zs), 1280, gh, gw) processed features on the GPU (not cached)."""
        T = self.template(name, vol).cuda()
        return embed(self.load_model(), [to_input(self.slab(vol, z)) for z in zs]).cuda() - T

    # ------------------------------------------------------------ shared basis
    def has_basis(self, name: str) -> bool:
        return (self.cache_root / name / "basis.npz").exists()

    def basis(self, name: str) -> dict:
        if name not in self._basis:
            b = np.load(self.cache_root / name / "basis.npz")
            self._basis[name] = {k: torch.from_numpy(b[k]).float().cuda() for k in ("mean", "V", "lo", "hi")}
        return self._basis[name]

    def fit_basis(self, name: str, vol, selection: list[int]) -> list[int]:
        """Fit the shared basis on up to N_BASIS slices spread over the selection; embeds and stores
        those slices too. Returns the slices it embedded."""
        idx = np.unique(np.linspace(0, len(selection) - 1, min(N_BASIS, len(selection))).round().astype(int))
        zs = [selection[i] for i in idx]
        F = torch.cat([self.full_features(name, vol, zs[i:i + 4]) for i in range(0, len(zs), 4)])
        X = F.permute(0, 2, 3, 1).reshape(-1, F.shape[1]).double()
        mean = X.mean(0)
        C = (X - mean).T @ (X - mean) / (len(X) - 1)
        _, vecs = torch.linalg.eigh(C)
        V = vecs[:, -N_PC:].flip(1).T.float()  # (256, 1280), largest variance first
        P3 = ((X.float() - mean.float()) @ V[:3].T).cpu().numpy()
        lo, hi = np.percentile(P3, [1, 99], axis=0)
        np.savez(self.cache_dir(name) / "basis.npz", mean=mean.float().cpu().numpy(), V=V.cpu().numpy(),
                 lo=lo.astype(np.float32), hi=hi.astype(np.float32), zs=np.array(zs))
        self._basis.pop(name, None)
        for z, f in zip(zs, F):
            self.store(name, z, f, vol.shape[1:])
        return zs

    def project(self, name: str, f: torch.Tensor) -> torch.Tensor:
        """(1280, gh, gw) GPU features -> (256, gh, gw)."""
        b = self.basis(name)
        D, gh, gw = f.shape
        return ((f.flatten(1).T - b["mean"]) @ b["V"].T).T.reshape(-1, gh, gw)

    def rgb(self, name: str, f256: np.ndarray | torch.Tensor) -> np.ndarray:
        """(256, gh, gw) -> (gh, gw, 3) in [0, 1]: shared components 1-3, tomogram-wide stretch."""
        b = self.basis(name)
        p = f256[:3].float() if torch.is_tensor(f256) else torch.from_numpy(np.asarray(f256[:3], dtype=np.float32))
        p = p.cuda().permute(1, 2, 0)
        return ((p - b["lo"]) / (b["hi"] - b["lo"] + 1e-8)).clamp(0, 1).cpu().numpy()

    # ------------------------------------------------------------ per slice
    def store(self, name: str, z: int, f: torch.Tensor, shape: tuple[int, int]):
        p = self.cache_dir(name) / f"z{z:04d}.npy"
        f256 = self.project(name, f)
        np.save(p, f256.half().cpu().numpy())
        save_png(p.with_name(f"z{z:04d}_pca.png"), self.rgb(name, f256), shape)

    def embed_slice(self, name: str, vol, z: int):
        if not (self.cache_root / name / f"z{z:04d}.npy").exists():
            self.store(name, z, self.full_features(name, vol, [z])[0], vol.shape[1:])

    def load(self, name: str, z: int) -> np.ndarray | None:
        p = self.cache_root / name / f"z{z:04d}.npy"
        return np.load(p) if p.exists() else None

    def features(self, name: str, z: int, done: set[int]) -> tuple[np.ndarray | None, str]:
        """256-d features of slice z: stored, or interpolated between the nearest stored slices
        below and above (gap <= MAX_GAP). Returns (features or None, description)."""
        if z in done:
            return self.load(name, z), "embedded"
        below = [d for d in done if z - MAX_GAP <= d < z]
        above = [d for d in done if z < d <= z + MAX_GAP]
        if not below or not above:
            return None, ""
        a, b = max(below), min(above)
        w = (z - a) / (b - a)
        f = (1 - w) * self.load(name, a).astype(np.float32) + w * self.load(name, b).astype(np.float32)
        return f, f"interpolated from z={a}, {b}"


def save_png(path: Path, rgb: np.ndarray, shape: tuple[int, int]):
    """Write an RGB token map upsampled (nearest) to the slice's pixel size."""
    from PIL import Image
    im = Image.fromarray((rgb * 255).round().astype(np.uint8)).resize((shape[1], shape[0]), Image.NEAREST)
    im.save(path)

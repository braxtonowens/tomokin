"""Interactive segmentation from scribbles on frozen C-RADIO features (used by webapp/server.py, page entropy.html).

Features: data/popsicle_full/<set>/<run>/feats.npy, every slice's rc_g8 features as 256-d PCA (embed_full.py),
token grid (gh, gw) covering the slice (resized to multiples of 16, align_corners=False convention).
Annotations: one uint8 label image per annotated slice (0 = unlabelled, k = class k), full pixel resolution,
saved as outputs/seg_interactive/<run>/annot/z<z>.png, classes in classes.json.
Training: every annotated pixel samples its feature by bilinear interpolation of the token grid (grid_sample), at
most 20k pixels per class; features standardized with tomogram statistics; multinomial logistic regression
(class-balanced cross-entropy, AdamW, weight decay) on the GPU.
The open run's feature volume is kept on the GPU (fp16), so fitting takes well under a second and a slice's
prediction is computed on request (bilinearly upsampled probabilities, optionally averaged over z +-1). Each fit
also returns a per-slice uncertainty profile (mean 1 - margin between the two most likely classes). Magic wand: tokens connected to the clicked token whose cosine
similarity to it exceeds a threshold.
"""

import json
import threading
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy import ndimage as ndi

ROOT = Path(__file__).resolve().parents[1]
FULL = ROOT / "data" / "popsicle_full"
OUT = ROOT / "outputs" / "seg_interactive"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
LOCK = threading.Lock()
_feats, _stats, _models = {}, {}, {}


def run_dir(set_, run):
    d = FULL / set_ / Path(run).name
    if not (d / "feats.npy").exists():
        raise FileNotFoundError(f"no features for {run}")
    return d


def feats(set_, run):
    if run not in _feats:
        _feats[run] = np.load(run_dir(set_, run) / "feats.npy", mmap_mode="r")
    return _feats[run]


def stats(set_, run):
    """Per-channel mean / std of the tomogram's features (40 slices)."""
    if run not in _stats:
        f = feats(set_, run)
        s = torch.from_numpy(np.asarray(f[:: max(1, len(f) // 40)], dtype=np.float32))
        s = s.permute(1, 0, 2, 3).reshape(s.shape[1], -1)
        _stats[run] = (s.mean(1).to(DEV), (s.std(1) + 1e-6).to(DEV))
    return _stats[run]


def adir(run):
    d = OUT / Path(run).name / "annot"
    d.mkdir(parents=True, exist_ok=True)
    return d


# ---------------------------------------------------------------- annotations
def save_slice(run, z, png_bytes, classes):
    import io
    lab = np.array(Image.open(io.BytesIO(png_bytes)).convert("L"))
    f = adir(run) / f"z{int(z)}.png"
    if lab.any():
        Image.fromarray(lab).save(f)
    elif f.exists():
        f.unlink()
    json.dump(classes, open(adir(run).parent / "classes.json", "w"), indent=1)
    return int((lab > 0).sum())


def load_annotations(run):
    d = OUT / Path(run).name / "annot"  # read only: no folder is created for runs without scribbles
    zs = sorted(int(p.stem[1:]) for p in d.glob("z*.png"))
    cf = d.parent / "classes.json"
    return {"slices": zs, "classes": json.load(open(cf)) if cf.exists() else None}


def slice_png(run, z):
    f = OUT / Path(run).name / "annot" / f"z{int(z)}.png"
    return f.read_bytes() if f.exists() else None


# ---------------------------------------------------------------- training / prediction
_gpu = {"run": None, "F": None}


def gfeats(set_, run):
    """The run's whole feature volume (nz, 256, gh, gw) fp16 on the GPU; only the last-used run is kept."""
    if _gpu["run"] != run:
        _gpu["F"] = None
        torch.cuda.empty_cache()
        _gpu["F"] = torch.from_numpy(np.ascontiguousarray(feats(set_, run))).to(DEV)
        _gpu["run"] = run
    return _gpu["F"]


def sample(set_, run, z, lab, n_per_class, rng):
    """Features (N, D) at annotated pixels of slice z by bilinear interpolation of the token grid."""
    f = gfeats(set_, run)[z].float()  # (D, gh, gw)
    H, W = lab.shape
    X, y = [], []
    for c in np.unique(lab[lab > 0]):
        yy, xx = np.nonzero(lab == c)
        if len(yy) > n_per_class:
            k = rng.choice(len(yy), n_per_class, replace=False)
            yy, xx = yy[k], xx[k]
        g = torch.stack([torch.from_numpy((xx + 0.5) / W * 2 - 1), torch.from_numpy((yy + 0.5) / H * 2 - 1)], -1)
        v = F.grid_sample(f[None], g.float().to(DEV)[None, None], mode="bilinear", align_corners=False)[0, :, 0].T
        X.append(v)
        y.append(torch.full((len(v),), int(c), device=DEV))
    return X, y


class MLP(torch.nn.Sequential):
    def __init__(self, D, C, h=256):
        super().__init__(torch.nn.Linear(D, h), torch.nn.GELU(), torch.nn.Dropout(0.2), torch.nn.Linear(h, C))


def fit_classifier(X, y, C, steps=400, seed=0):
    """Small MLP on the (balanced, <= 20k per class) scribble pixels; class-weighted CE, AdamW, minibatches.
    Chosen over the linear model in the cell-hunt offline comparison (cellhunt_classifiers.py)."""
    present = torch.unique(y)
    cnt = torch.bincount(y, minlength=C).float()
    w = torch.where(cnt > 0, cnt.sum() / (cnt.clamp_min(1) * len(present)), torch.zeros_like(cnt))
    torch.manual_seed(seed)
    net = MLP(X.shape[1], C).to(DEV)
    opt = torch.optim.AdamW(net.parameters(), lr=3e-3, weight_decay=1e-2)
    for _ in range(steps):
        idx = torch.randint(len(X), (min(len(X), 8192),), device=DEV)
        opt.zero_grad()
        loss = F.cross_entropy(net(X[idx]), y[idx], weight=w)
        loss.backward()
        opt.step()
    net.eval()
    with torch.no_grad():
        acc = float((net(X).argmax(1) == y).float().mean())
    return net, acc, cnt, present


def prior_default(raw_counts):
    """Label-free background prior: painted class-1 (background) pixels : pixels of all other classes (uncapped)."""
    bg, rest = raw_counts[0], sum(raw_counts[1:])
    return float(np.clip(bg / max(rest, 1), 1.0, 500.0)) if bg and rest else 1.0


def logits(run, zs, key=None, set_=None, prior=1.0):
    """Logits (len(zs), C, gh, gw) of model `key` (default: the run's own model) for slices zs of `run`; features are
    standardized with the run's own statistics. The classifier is trained on balanced scribbles, so `prior`
    (background : other classes) is applied as a shift of the background logit by log(prior)."""
    m = _models[key or run]
    if isinstance(m["net"], UNet):
        f, im = unet_inputs(set_ or m["set"], run, zs)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            lg = m["net"](f, im).float()
        if prior and prior != 1.0:
            lg[:, 0] += float(np.log(prior))
        return lg
    x = _gpu["F"][zs].float() if _gpu["run"] == run else torch.from_numpy(
        np.asarray(_feats[run][zs], dtype=np.float32)).to(DEV)
    mu, sd = (stats(set_, run) if set_ else (m["mu"], m["sd"]))
    x = (x - mu[None, :, None, None]) / sd[None, :, None, None]
    B, D, gh, gw = x.shape
    lg = m["net"](x.permute(0, 2, 3, 1).reshape(-1, D)).reshape(B, gh, gw, -1).permute(0, 3, 1, 2)
    if prior and prior != 1.0:
        lg = lg.clone()
        lg[:, 0] += float(np.log(prior))
    return lg


def train(set_, run, n_classes, prior=None):
    ann = load_annotations(run)
    if not ann["slices"]:
        raise ValueError("no annotations yet")
    rng = np.random.default_rng(0)
    X, y = [], []
    for z in ann["slices"]:
        lab = np.array(Image.open(adir(run) / f"z{z}.png"))
        a, b = sample(set_, run, z, lab, 20000, rng)
        X += a
        y += b
    X, y = torch.cat(X), torch.cat(y) - 1  # classes 1..C -> 0..C-1
    present = torch.unique(y)
    if len(present) < 2:
        raise ValueError("paint at least two classes")
    mu, sd = stats(set_, run)
    Xs = (X - mu) / sd
    C = n_classes
    net, acc, cnt, present = fit_classifier(Xs, y, C)
    raw = [0] * C
    for z in ann["slices"]:
        lab = np.array(Image.open(adir(run) / f"z{z}.png"))
        for c in range(C):
            raw[c] += int((lab == c + 1).sum())
    pdef = prior_default(raw)
    prior = prior if prior else pdef
    version = _models.get(run, {}).get("version", 0) + 1
    _models[run] = {"net": net, "mu": mu, "sd": sd, "C": C, "version": version, "present": present.tolist()}
    # per-slice uncertainty (mean 1 - margin between the two most likely classes) and class fractions
    nz = len(gfeats(set_, run))
    unsure, frac = [], torch.zeros(C, device=DEV)
    with torch.no_grad():
        for z0 in range(0, nz, 64):
            p = torch.softmax(logits(run, list(range(z0, min(nz, z0 + 64))), prior=prior), 1)
            top = p.topk(2, dim=1).values
            unsure += (1 - (top[:, 0] - top[:, 1])).mean((1, 2)).tolist()
            frac += torch.bincount(p.argmax(1).flatten(), minlength=C).float()
    return {"n_pixels": int(len(y)), "per_class": cnt.int().tolist(), "train_acc": acc, "version": version,
            "slices": ann["slices"], "volume_fraction": (frac / frac.sum()).tolist(), "unsure": unsure,
            "prior_default": pdef, "painted": raw}


def predict_slice(run, z, shape, zavg=0, key=None, set_=None, prior=1.0):
    """-> class map (H, W) uint8 (0-based), confidence and uncertainty (1 - top-2 margin) at pixel resolution."""
    if (key or run) not in _models:
        return None
    if set_:
        feats(set_, run)
    nz = _gpu["F"].shape[0] if _gpu["run"] == run else len(_feats[run])
    zs = list(range(max(0, z - zavg), min(nz, z + zavg + 1)))
    m = _models[key or run]
    with torch.no_grad():
        p = torch.softmax(logits(run, zs, key, set_, 1.0 if m.get("calib") else prior), 1).mean(0, keepdim=True)
        p = F.interpolate(p, size=shape, mode="bilinear", align_corners=False)[0]
        if m.get("knn") is not None:  # U-Net + kNN, half and half (knn_unet16.py: 0.449 vs 0.339 / 0.400 alone)
            pk = F.interpolate(knn_probs(run, zs, set_ or m["set"], m)[None], size=shape, mode="bilinear", align_corners=False)[0]
            p = 0.5 * p + 0.5 * pk
        if m.get("calib"):
            lab, conf, uns = decide(p, m["calib"])
            return lab.cpu().numpy().astype(np.uint8), conf.cpu().numpy(), uns.cpu().numpy()
        top = p.topk(2, dim=0).values
    return p.argmax(0).cpu().numpy().astype(np.uint8), top[0].cpu().numpy(), (1 - (top[0] - top[1])).cpu().numpy()


def export(run, shape, zavg=0, key=None, set_=None, prior=1.0):
    if (key or run) not in _models:
        raise ValueError("segment first")
    out = OUT / Path(run).name
    out.mkdir(parents=True, exist_ok=True)
    if set_:
        feats(set_, run)
    nz = _gpu["F"].shape[0] if _gpu["run"] == run else len(_feats[run])
    vol = np.lib.format.open_memmap(out / "labels.npy", mode="w+", dtype=np.uint8, shape=(nz,) + tuple(shape))
    for z in range(nz):
        vol[z] = predict_slice(run, z, shape, zavg, key, set_, prior)[0] + 1  # 1..C (every voxel gets a class)
    vol.flush()
    return str((out / "labels.npy").relative_to(ROOT))


def wand(set_, run, z, x, y, shape, thr):
    """Token mask (gh, gw) of tokens connected to the clicked token with cosine similarity > thr."""
    f = gfeats(set_, run)[z].float()
    D, gh, gw = f.shape
    H, W = shape
    ty, tx = min(gh - 1, int(y / H * gh)), min(gw - 1, int(x / W * gw))
    fn = F.normalize(f - f.mean((1, 2), keepdim=True), dim=0)
    sim = torch.einsum("dhw,d->hw", fn, fn[:, ty, tx]).cpu().numpy()
    comp, _ = ndi.label(sim > thr)
    return (comp == comp[ty, tx]) & (sim > thr)


# ---------------------------------------------------------------- projects: one classifier pooled over many tomograms
PROJECTS = ROOT / "outputs"


def project(name):
    f = PROJECTS / Path(name).name / "project.json"
    if not f.exists():
        raise FileNotFoundError(f"no project {name}")
    return json.load(open(f))


def train_project(name, cur_set, cur_run, n_classes, prior=None, head="mlp"):
    """Pool the scribbles of every run in the project (each run's features standardized with its own statistics);
    log the fit (weights + what was annotated) for later hidden scoring; return the current run's uncertainty."""
    import time
    runs = project(name)["runs"]
    rng = np.random.default_rng(0)
    X, y, used = [], [], []
    raw = [0] * n_classes
    for r in runs:
        ann = load_annotations(r["run"])
        for z in ann["slices"]:
            lab = np.array(Image.open(adir(r["run"]) / f"z{z}.png"))
            feats(r["set"], r["run"])
            if _gpu["run"] != r["run"]:  # sample from the memory map without moving the whole volume to the GPU
                f = torch.from_numpy(np.asarray(_feats[r["run"]][z], dtype=np.float32)).to(DEV)
            else:
                f = _gpu["F"][z].float()
            mu, sd = stats(r["set"], r["run"])
            a, b = sample_from(f, lab, 20000, rng)
            X += [(v - mu) / sd for v in a]
            y += b
            used.append((r["run"], int(z), int((lab > 0).sum())))
            for c in range(n_classes):
                raw[c] += int((lab == c + 1).sum())
    y = [v - 1 for v in y]  # scribble classes 1..C -> 0..C-1
    tok_px = 256  # a token covers 16 x 16 px: dense token counts enter the prior in pixel units
    for r in runs:
        a_, b_, c_ = dense_samples(r["set"], r["run"], n_classes, 40000, rng)
        if a_:
            X += a_
            y += b_
            used.append((r["run"], "dense", int(sum(c_))))
            for c in range(n_classes):
                raw[c] += c_[c] * tok_px
    if not X:
        raise ValueError("no annotations yet")
    X, y = torch.cat(X), torch.cat(y)
    present = torch.unique(y)
    if head == "unet_pu":  # positive-only: one real class is enough (unassigned is implicit)
        if not (present > 0).any():
            raise ValueError("paint at least one class")
    elif len(present) < 2:
        raise ValueError("paint at least two classes")
    C = n_classes
    key = f"project:{name}"
    import time as _t
    t_fit = _t.time()
    calib = None
    pool = [(r["set"], r["run"]) for r in runs]
    prev = _models.get(key, {})
    warm = prev.get("net") if prev.get("kind") == head and prev.get("C") == C else None
    if head == "unet":  # class 1 = painted "unassigned" examples; automatic background prior (painted-pixel ratio)
        net, _ = fit_unet(unet_slices(runs), C, warm, steps=150 if warm is not None else 300)
        bank = knn_bank(runs)
        acc = None  # (JSON has no NaN)
        cnt = torch.bincount(y, minlength=C).float()
    elif head == "unet_pu":  # experimental: implicit unassigned + calibration, no painted background needed
        net, _, calib = fit_unet_pu(unet_slices(runs), C, pool, warm, steps=150 if warm is not None else 300)
        acc = None
        cnt = torch.bincount(y, minlength=C).float()
    elif head == "mlp":
        net, acc, cnt, present = fit_classifier(X, y, C)
    else:  # mlp_pu (experimental): implicit unassigned tokens from the whole project + calibration
        rng_u = np.random.default_rng(1)
        Xu = []
        for set_, run_, z_ in pool_slices(pool, 24, rng_u):
            f_ = torch.from_numpy(np.asarray(feats(set_, run_)[z_], dtype=np.float32)).to(DEV).flatten(1).T
            mu_, sd_ = stats(set_, run_)
            Xu.append(((f_ - mu_) / sd_)[torch.from_numpy(rng_u.choice(len(f_), min(len(f_), 800), replace=False)).to(DEV)])
        Xu = torch.cat(Xu)
        net, acc, cnt, present = fit_classifier(torch.cat([X, Xu]), torch.cat([y, torch.zeros(len(Xu), dtype=torch.long, device=DEV)]), C)
        with torch.no_grad():
            p_ = net(X).softmax(1)
        calib = [1.0] + [float(p_[y == k, k].mean()) if (y == k).any() else 1.0 for k in range(1, C)]
    fit_s = _t.time() - t_fit
    pdef = prior_default(raw)
    prior = prior if prior else pdef
    version = _models.get(key, {}).get("version", 0) + 1
    _models[key] = {"net": net, "mu": None, "sd": None, "C": C, "version": version, "present": present.tolist(),
                    "kind": head, "set": cur_set, "calib": calib}
    if head == "unet":
        _models[key]["knn"] = bank
    if calib is not None:
        prior = 1.0  # calibrated heads need no prior
    fits = PROJECTS / Path(name).name / "fits"
    fits.mkdir(parents=True, exist_ok=True)
    fit_file = fits / f"fit_{int(time.time() * 1000)}.pt"
    torch.save({"kind": head, "state": {k: v.cpu() for k, v in net.state_dict().items()}, "D": X.shape[1], "C": C,
                "version": version, "time": time.time(), "annotated": used, "per_class": cnt.int().tolist(),
                "painted": raw, "prior": prior, "prior_default": pdef, "train_acc": acc, "calib": calib}, fit_file)
    _models[key]["fit_file"] = str(fit_file)
    nz = len(gfeats(cur_set, cur_run))
    unsure, frac = [], torch.zeros(C, device=DEV)
    with torch.no_grad():
        for z0 in range(0, nz, 32):
            p = torch.softmax(logits(cur_run, list(range(z0, min(nz, z0 + 32))), key, cur_set, prior), 1)
            if calib is not None:
                lab_, _, un_ = decide(p, calib)
                unsure += un_.mean((1, 2)).tolist()
                frac += torch.bincount(lab_.flatten(), minlength=C).float()
            else:
                top = p.topk(2, dim=1).values
                unsure += (1 - (top[:, 0] - top[:, 1])).mean((1, 2)).tolist()
                frac += torch.bincount(p.argmax(1).flatten(), minlength=C).float()
    runs_used = sorted({u[0] for u in used})
    return {"n_pixels": int(len(y)), "per_class": cnt.int().tolist(), "train_acc": acc, "version": version,
            "slices": [u[1] for u in used if u[0] == cur_run], "n_slices": len(used), "n_tomograms": len(runs_used),
            "volume_fraction": (frac / frac.sum()).tolist(), "unsure": unsure, "prior_default": pdef, "painted": raw,
            "head": head, "fit_seconds": fit_s}


# ---------------------------------------------------------------- model history: undo restores the model from before a stroke
_history = {}  # classifier key -> OrderedDict {version: (copy of the _models entry, prior_default)}, last 30 fits


def remember_fit(key, result):
    import copy
    from collections import OrderedDict
    h = _history.setdefault(key, OrderedDict())
    m = _models[key]
    snap = copy.deepcopy({k: v for k, v in m.items() if k != "knn"})
    if m.get("knn") is not None:  # the kNN bank is never changed in place: keep a CPU reference, not a copy
        snap["knn"] = tuple(t.cpu() for t in m["knn"])
    h[m["version"]] = (snap, result.get("prior_default"))
    while len(h) > 30:
        h.popitem(last=False)


def restore(key, version, cur_set, cur_run, prior):
    """Make the model of fit `version` current again (under a new version number); the restore is logged as a copy of
    that fit's log file with 'restored_from'. -> dict for the page, or None if the version is no longer kept."""
    import copy
    h = _history.get(key)
    if not h or version not in h:
        return None
    m, pdef = h[version]
    new = max(max(h), _models.get(key, {}).get("version", 0)) + 1
    _models[key] = {**copy.deepcopy({k: v for k, v in m.items() if k != "knn"}), "version": new}
    if m.get("knn") is not None:
        _models[key]["knn"] = tuple(t.to(DEV) for t in m["knn"])
    h[new] = (m | {"version": new}, pdef)
    _cleared.discard(key)
    f = m.get("fit_file")
    if f and Path(f).exists():
        import time
        d = torch.load(f, weights_only=False)
        d.update(version=new, time=time.time(), restored_from=version)
        torch.save(d, Path(f).parent / f"fit_{int(time.time() * 1000)}.pt")
    r = profile(key, cur_set, cur_run, prior)
    return {"version": new, "restored_from": version, "prior_default": pdef, "unsure": r["unsure"] if r else []}


# ---------------------------------------------------------------- kNN half of the prediction
KNN_K, KNN_PER_SLICE, KNN_CAP = 16, 30000, 60000


def knn_bank(runs, seed=0):
    """Features (standardized per tomogram, L2-normalized, fp16) at painted pixels of every annotated slice of the
    project, sampled uniformly within each slice (so the class ratio is the painted ratio), at most KNN_PER_SLICE per
    slice and KNN_CAP in all; labels 0..C-1 (painted class - 1). -> (X (N, D), y (N,)) or None."""
    rng = np.random.default_rng(seed)
    Xs, ys = [], []
    for r in runs:
        for z in load_annotations(r["run"])["slices"]:
            lab = np.array(Image.open(adir(r["run"]) / f"z{z}.png"))
            yy, xx = np.nonzero(lab > 0)
            if not len(yy):
                continue
            if len(yy) > KNN_PER_SLICE:
                k = rng.choice(len(yy), KNN_PER_SLICE, replace=False)
                yy, xx = yy[k], xx[k]
            feats(r["set"], r["run"])
            f = _gpu["F"][z].float() if _gpu["run"] == r["run"] else torch.from_numpy(np.asarray(_feats[r["run"]][z], dtype=np.float32)).to(DEV)
            H, W = lab.shape
            g = torch.stack([torch.from_numpy((xx + 0.5) / W * 2 - 1), torch.from_numpy((yy + 0.5) / H * 2 - 1)], -1)
            v = F.grid_sample(f[None], g.float().to(DEV)[None, None], mode="bilinear", align_corners=False)[0, :, 0].T
            mu, sd = stats(r["set"], r["run"])
            Xs.append(F.normalize((v - mu) / sd, dim=1).half())
            ys.append(torch.from_numpy(lab[yy, xx].astype(np.int64) - 1).to(DEV))
    if not Xs:
        return None
    X, y = torch.cat(Xs), torch.cat(ys)
    if len(X) > KNN_CAP:
        k = torch.from_numpy(rng.choice(len(X), KNN_CAP, replace=False)).to(DEV)
        X, y = X[k], y[k]
    return X, y


def knn_probs(run, zs, set_, m):
    """Class fractions among the KNN_K most similar bank features, per token, averaged over zs -> (C, gh, gw)."""
    X, y = m["knn"]
    x, _ = unet_inputs(set_, run, zs)
    out = 0
    for f in F.normalize(x, dim=1):
        D, gh, gw = f.shape
        idx = (f.flatten(1).T.half() @ X.T).topk(min(KNN_K, len(X)), dim=1).indices
        out = out + F.one_hot(y[idx], m["C"]).float().mean(1).T.reshape(m["C"], gh, gw)
    return out / len(x)


def log_prior(name, prior):
    """Record the prior the annotator is currently using (for the hidden scoring of the latest fit)."""
    import time
    f = PROJECTS / Path(name).name / "priors.jsonl"
    with open(f, "a") as fh:
        fh.write(json.dumps({"time": time.time(), "prior": prior}) + "\n")


def sample_from(f, lab, n_per_class, rng):
    """Features at annotated pixels of one slice's feature map f (D, gh, gw), bilinear on the token grid."""
    H, W = lab.shape
    X, y = [], []
    for c in np.unique(lab[lab > 0]):
        yy, xx = np.nonzero(lab == c)
        if len(yy) > n_per_class:
            k = rng.choice(len(yy), n_per_class, replace=False)
            yy, xx = yy[k], xx[k]
        g = torch.stack([torch.from_numpy((xx + 0.5) / W * 2 - 1), torch.from_numpy((yy + 0.5) / H * 2 - 1)], -1)
        v = F.grid_sample(f[None], g.float().to(DEV)[None, None], mode="bilinear", align_corners=False)[0, :, 0].T
        X.append(v)
        y.append(torch.full((len(v),), int(c), device=DEV))
    return X, y


_cleared = set()  # classifier keys whose guess the user cleared: quiet (automatic) refits are refused


def clear(key):
    """Discard a classifier (the run's own, or 'project:<name>'); scribbles are kept. Until an explicit fit
    (a new stroke or Segment), automatic refits from any open tab are refused."""
    _cleared.add(key)
    return _models.pop(key, None) is not None


def check_cleared(key, explicit):
    if key in _cleared and not explicit:
        raise ValueError("guess cleared")
    _cleared.discard(key)


def remove_class(runs, k):
    """Delete class k (k >= 2) from these runs: its scribble pixels and accepted-object voxels become unlabelled,
    classes above k move down by one (k+1 -> k, ...), and it leaves classes.json (names / colours of the others kept).
    -> number of slices / volumes changed."""
    k = int(k)
    if k < 2:
        raise ValueError("class 1 (unassigned) cannot be removed")
    lut = np.arange(256, dtype=np.uint8)
    lut[k] = 0
    lut[k + 1:] = np.arange(k, 255, dtype=np.uint8)
    n = 0
    for run in runs:
        d = OUT / Path(run).name
        for f in (d / "annot").glob("z*.png"):
            a = np.array(Image.open(f))
            if (a >= k).any():
                b = lut[a]
                if b.any():
                    Image.fromarray(b).save(f)
                else:
                    f.unlink()
                n += 1
        if (d / "dense.npy").exists():
            a = np.load(d / "dense.npy")
            if (a >= k).any():
                np.save(d / "dense.npy", lut[a])
                n += 1
        cf = d / "classes.json"
        if cf.exists():
            cl = json.load(open(cf))
            cl = [{**c, "id": c["id"] - (c["id"] > k)} for c in cl if c["id"] != k]
            json.dump(cl, open(cf, "w"), indent=1)
    return n


def clear_scribbles(runs):
    """Delete every saved scribble slice of these runs (class names in classes.json are kept); returns the count."""
    n = 0
    for run in runs:
        for f in (OUT / Path(run).name / "annot").glob("z*.png"):
            f.unlink()
            n += 1
        dense = OUT / Path(run).name / "dense.npy"  # accepted nnInteractive objects
        if dense.exists():
            dense.unlink()
            n += 1
    return n


# ---------------------------------------------------------------- dense labels from accepted nnInteractive objects
_dense_cache = {}


def complete_flag(run):
    f = OUT / Path(run).name / "complete.json"
    return json.load(open(f)).get("complete", False) if f.exists() else False


def set_complete(run, value):
    d = OUT / Path(run).name
    d.mkdir(parents=True, exist_ok=True)
    json.dump({"complete": bool(value)}, open(d / "complete.json", "w"))


def dense_tokens(set_, run, n_classes, step=2):
    """Token labels from outputs/seg_interactive/<run>/dense.npy on every `step`-th slice: class k where >= 50% of the
    token's pixels are k; if the tomogram is marked complete, tokens with no labelled pixel are background (class 1),
    boundary tokens are skipped. Cached by file time. -> {class index (0-based): (z, ty, tx) int tensors} or None."""
    f = OUT / Path(run).name / "dense.npy"
    if not f.exists():
        return None
    key = (run, f.stat().st_mtime, complete_flag(run), n_classes)
    if _dense_cache.get(run, (None,))[0] == key:
        return _dense_cache[run][1]
    dense = np.load(f, mmap_mode="r")
    fz = feats(set_, run)
    gh, gw = fz.shape[2:]
    out = {c: [] for c in range(n_classes)}
    for z in range(0, len(dense), step):
        d = torch.from_numpy(np.asarray(dense[z])).to(DEV).long()
        oh = F.one_hot(d, n_classes + 1).permute(2, 0, 1).float()[None]
        frac = F.adaptive_avg_pool2d(oh, (gh, gw))[0]  # (C+1, gh, gw); channel 0 = unlabelled
        for c in range(1, n_classes + 1):
            ty, tx = torch.nonzero(frac[c] >= 0.5, as_tuple=True)
            out[c - 1].append(torch.stack([torch.full_like(ty, z), ty, tx], 1))
        if complete_flag(run):
            ty, tx = torch.nonzero(frac[0] >= 0.999, as_tuple=True)
            out[0].append(torch.stack([torch.full_like(ty, z), ty, tx], 1))
    res = {c: torch.cat(v) for c, v in out.items() if v and sum(len(x) for x in v)}
    _dense_cache[run] = (key, res)
    return res


def dense_samples(set_, run, n_classes, n_per_class, rng):
    """Standardized features + labels for up to n_per_class dense tokens per class of one run."""
    toks = dense_tokens(set_, run, n_classes)
    if not toks:
        return [], [], [0] * n_classes
    fz = feats(set_, run)
    mu, sd = stats(set_, run)
    X, y, cnt = [], [], [0] * n_classes
    for c, idx in toks.items():
        cnt[c] = len(idx)
        k = idx[torch.from_numpy(rng.choice(len(idx), min(n_per_class, len(idx)), replace=False)).to(idx.device)]
        zs = k[:, 0].cpu().numpy()
        if _gpu["run"] == run:
            v = _gpu["F"][k[:, 0], :, k[:, 1], k[:, 2]].float()
        else:
            uz, inv = np.unique(zs, return_inverse=True)
            block = torch.from_numpy(np.asarray(fz[uz], dtype=np.float32)).to(DEV)
            v = block[torch.from_numpy(inv).to(DEV), :, k[:, 1], k[:, 2]]
        X.append((v - mu) / sd)
        y.append(torch.full((len(v),), c, device=DEV))
    return X, y, cnt


# ---------------------------------------------------------------- fast U-Net head (quarter resolution)
Q = 4  # output cell = 4 x 4 px
_g8q = {}


def g8q(set_, run):
    """Gray-8 slab of every slice at quarter resolution (nz, ceil(H/4), ceil(W/4)), robustly standardized per
    tomogram; fp16 on the GPU, cached on disk (outputs/seg_interactive/<run>/g8q.npy)."""
    if run in _g8q:
        return _g8q[run]
    f = OUT / Path(run).name / "g8q.npy"
    if f.exists():
        g = np.load(f)
    else:
        vol = np.load(FULL / set_ / Path(run).name / "tomo.npy", mmap_mode="r")
        nz, H, W = vol.shape
        out = []
        for z0 in range(0, nz, 32):
            zs = range(z0, min(nz, z0 + 32))
            lo = max(0, z0 - 4)
            block = torch.from_numpy(np.asarray(vol[lo:min(nz, z0 + 32 + 4)], dtype=np.float32)).to(DEV)
            for z in zs:
                a = int(np.clip(z - 4, 0, nz - 8)) - lo
                s8 = block[a:a + 8].mean(0)
                out.append(F.avg_pool2d(s8[None, None], Q, Q, ceil_mode=True)[0, 0].cpu())
        g = torch.stack(out).numpy()
        med, sd = np.median(g[:: max(1, nz // 20)]), g[:: max(1, nz // 20)].std() + 1e-6
        g = ((g - med) / sd).astype(np.float16)
        f.parent.mkdir(parents=True, exist_ok=True)
        np.save(f, g)
    if len(_g8q) >= 24:  # a whole project's images fit (~56 MB each)
        _g8q.pop(next(iter(_g8q)))
    _g8q[run] = torch.from_numpy(g).to(DEV)
    return _g8q[run]


class UNet(torch.nn.Module):
    """RADIO features (256-d token grid, 1x1-reduced to 32 and bilinearly upsampled) + gray-8 image at quarter
    resolution -> class logits at quarter resolution. 3 levels, 32/64/64 channels (~0.15 M parameters)."""

    def __init__(self, D=256, C=2, c=32):
        super().__init__()
        blk = lambda i, o: torch.nn.Sequential(torch.nn.Conv2d(i, o, 3, padding=1), torch.nn.GELU(),
                                               torch.nn.Conv2d(o, o, 3, padding=1), torch.nn.GELU())
        self.red = torch.nn.Sequential(torch.nn.Conv2d(D, c, 1), torch.nn.GELU())
        self.e1, self.e2, self.b = blk(c + 1, c), blk(c, 2 * c), blk(2 * c, 2 * c)
        self.d2, self.d1 = blk(4 * c, c), blk(2 * c, c)
        self.out = torch.nn.Conv2d(c, C, 1)

    def forward(self, f, img):
        """f (B, D, gh, gw) standardized features; img (B, 1, h, w) quarter-res image."""
        h, w = img.shape[-2:]
        x = torch.cat([F.interpolate(self.red(f), size=(h, w), mode="bilinear", align_corners=False), img], 1)
        e1 = self.e1(x)
        e2 = self.e2(F.max_pool2d(e1, 2, ceil_mode=True))
        b = self.b(F.max_pool2d(e2, 2, ceil_mode=True))
        d2 = self.d2(torch.cat([F.interpolate(b, size=e2.shape[-2:], mode="bilinear", align_corners=False), e2], 1))
        d1 = self.d1(torch.cat([F.interpolate(d2, size=e1.shape[-2:], mode="bilinear", align_corners=False), e1], 1))
        return self.out(d1)


def unet_inputs(set_, run, zs):
    """Standardized features (B, D, gh, gw) and quarter-res image (B, 1, h, w) for slices zs of a run."""
    feats(set_, run)
    x = _gpu["F"][zs].float() if _gpu["run"] == run else torch.from_numpy(np.asarray(_feats[run][zs], dtype=np.float32)).to(DEV)
    mu, sd = stats(set_, run)
    x = (x - mu[None, :, None, None]) / sd[None, :, None, None]
    return x, g8q(set_, run)[zs].float()[:, None]


def unet_slices(runs):
    """Every annotated slice of the given runs: (set, run, z, quarter-res label map (h, w) long, 0 = unlabelled)."""
    out = []
    for r in runs:
        for z in load_annotations(r["run"])["slices"]:
            lab = np.array(Image.open(adir(r["run"]) / f"z{z}.png"))
            h, w = int(np.ceil(lab.shape[0] / Q)), int(np.ceil(lab.shape[1] / Q))
            lq = torch.from_numpy(lab[Q // 2::Q, Q // 2::Q][:h, :w].astype(np.int64))
            lq = F.pad(lq, (0, w - lq.shape[1], 0, h - lq.shape[0]))
            out.append((r["set"], r["run"], int(z), lq.to(DEV)))
    return out


def fit_unet(slices, C, net=None, steps=300, T=24, bs=8, seed=0):
    """Sparse-label training (loss only on painted pixels, class-balanced), token-aligned random crops of T x T tokens
    (= 4T x 4T quarter-res pixels) + flips, AdamW, bf16 autocast; warm-started from `net` when given."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    net = (net or UNet(C=C)).to(DEV).train()
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-4)
    data = [(unet_inputs(s, r, [z]), lab) for s, r, z, lab in slices]
    cnt = torch.zeros(C, device=DEV)
    for _, lab in data:
        cnt += torch.bincount(lab[lab > 0] - 1, minlength=C).float()
    w = torch.where(cnt > 0, cnt.sum() / (cnt.clamp_min(1) * (cnt > 0).sum()), torch.zeros_like(cnt))
    P = 4 * T
    for _ in range(steps):
        fs, ims, labs = [], [], []
        for i in rng.integers(len(data), size=bs):
            (f, im), lab = data[i]
            gh, gw = f.shape[-2:]
            h, w_ = lab.shape
            ys, xs = torch.nonzero(lab > 0, as_tuple=True)
            if len(ys) and rng.random() < 0.8:  # mostly crops around painted pixels
                k = int(rng.integers(len(ys)))
                cty, ctx = int(ys[k]) * gh // h, int(xs[k]) * gw // w_
            else:
                cty, ctx = int(rng.integers(gh)), int(rng.integers(gw))
            ty0 = int(np.clip(cty - T // 2, 0, max(0, gh - T)))
            tx0 = int(np.clip(ctx - T // 2, 0, max(0, gw - T)))
            y0, x0 = round(ty0 * h / gh), round(tx0 * w_ / gw)
            fc = F.pad(f[..., ty0:ty0 + T, tx0:tx0 + T], (0, max(0, T - (gw - tx0)), 0, max(0, T - (gh - ty0))))
            imc = im[..., y0:y0 + P, x0:x0 + P]
            lc = lab[y0:y0 + P, x0:x0 + P]
            imc = F.pad(imc, (0, P - imc.shape[-1], 0, P - imc.shape[-2]))
            lc = F.pad(lc, (0, P - lc.shape[-1], 0, P - lc.shape[-2]))
            if rng.random() < 0.5:
                fc, imc, lc = fc.flip(-1), imc.flip(-1), lc.flip(-1)
            if rng.random() < 0.5:
                fc, imc, lc = fc.flip(-2), imc.flip(-2), lc.flip(-2)
            fs.append(fc)
            ims.append(imc)
            labs.append(lc)
        fb, ib, lb = torch.cat(fs), torch.cat(ims), torch.stack(labs)
        m = lb > 0
        if not m.any():
            continue
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lg = net(fb, ib)
        loss = F.cross_entropy(lg.float().permute(0, 2, 3, 1)[m], lb[m] - 1, weight=w)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return net.eval(), cnt


def profile(key, cur_set, cur_run, prior):
    """Per-slice uncertainty of an existing classifier on a run (no training): for the "?" chips after a switch."""
    m = _models.get(key)
    if m is None:
        return None
    C = m["C"]
    nz = len(gfeats(cur_set, cur_run))
    unsure = []
    with torch.no_grad():
        for z0 in range(0, nz, 32):
            p = torch.softmax(logits(cur_run, list(range(z0, min(nz, z0 + 32))), key, cur_set,
                                     1.0 if m.get("calib") else prior), 1)
            if m.get("calib"):
                unsure += decide(p, m["calib"])[2].mean((1, 2)).tolist()
            else:
                top = p.topk(2, dim=1).values
                unsure += (1 - (top[:, 0] - top[:, 1])).mean((1, 2)).tolist()
    return {"unsure": unsure, "version": m["version"]}


# ---------------------------------------------------------------- implicit "unassigned" (positive-unlabelled learning)
# Output index 0 = unassigned. Painted class-1 pixels (the optional "unassigned" brush) are explicit examples of it;
# in addition every fit samples UNPAINTED pixels from all over the project's tomograms (all z) as implicit unassigned
# examples. Because those contain some objects too, scores are calibrated per class (Elkan & Noto 2008): c_k = mean
# predicted p_k over the pixels the user painted as class k; calibrated p_k / c_k > 0.5 -> class k, else unassigned.


def decide(p, calib):
    """p (..., C, h, w) softmax -> (label (..., h, w) with 0 = unassigned, confidence, uncertainty)."""
    c = torch.as_tensor(calib, device=p.device, dtype=p.dtype).clamp(0.04, 2.0)
    q = (p[..., 1:, :, :] / c[1:, None, None]).clamp(max=1.0)
    best, idx = q.max(dim=-3)
    lab = torch.where(best > 0.5, idx + 1, torch.zeros_like(idx))
    return lab, best, 1 - (2 * best - 1).abs()


def pool_slices(pool, n, rng):
    """n random (set, run, z) from the project's tomograms, all z, for implicit unassigned crops."""
    out = []
    for _ in range(n):
        set_, run = pool[int(rng.integers(len(pool)))]
        nz = len(feats(set_, run))
        out.append((set_, run, int(rng.integers(nz))))
    return out


def holdout_strokes(slices, frac, rng):
    """Split painted pixels by connected stroke (per class, per slice): ~frac of the strokes (at least one per class
    when a class has >= 2 strokes) are held out of training and used only for calibration."""
    train, held = [], []
    for s_, r_, z_, lab in slices:
        tl, hl = lab.clone(), torch.zeros_like(lab)
        L = lab.cpu().numpy()
        for c in np.unique(L[L > 0]):
            comp, n = ndi.label(L == c)
            if n < 2:
                continue
            k = max(1, int(round(frac * n)))
            for j in rng.choice(np.arange(1, n + 1), k, replace=False):
                m = torch.from_numpy(comp == j).to(lab.device)
                hl[m] = int(c)
                tl[m] = 0
        train.append((s_, r_, z_, tl))
        held.append(hl)
    return train, held


def fit_unet_pu(slices, C, pool, net=None, steps=300, T=24, bs=8, n_implicit=2, seed=0, holdout=0.25, w_implicit=0.5,
                calib_mode="quantile", recall=0.9, ignore_unpainted=True):
    """Like fit_unet, plus implicit unassigned supervision: unpainted pixels of the annotated slices and random crops
    of random slices of the whole project, label 0, total weight per batch = w_implicit x the painted pixels' total
    weight. Calibration (Elkan-Noto c_k) on held-out strokes (`holdout` of each class's strokes, see holdout_strokes);
    falls back to the training pixels when a class has a single stroke."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    net = (net or UNet(C=C)).to(DEV).train()
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-4)
    full = slices
    slices, held = holdout_strokes(slices, holdout, rng) if holdout else (slices, [None] * len(slices))
    data = [(unet_inputs(s, r, [z]), lab) for s, r, z, lab in slices]
    extra = [(unet_inputs(s, r, [z]), None) for s, r, z in pool_slices(pool, 32, rng)]
    cnt = torch.zeros(C, device=DEV)
    for _, lab in data:
        cnt += torch.bincount(lab[lab > 0] - 1, minlength=C).float()
    wc = torch.where(cnt > 0, cnt.sum() / (cnt.clamp_min(1) * (cnt > 0).sum()), torch.zeros_like(cnt))
    P = 4 * T

    def crop(item, around):
        (f, im), lab = item
        gh, gw = f.shape[-2:]
        h, w_ = im.shape[-2:]
        if lab is None:
            lab = torch.full((h, w_), -1, dtype=torch.long, device=DEV)  # -1 = implicit unassigned
        else:  # unpainted pixels of annotated slices are mostly unpainted object next to the strokes: ignore them
            lab = torch.where(lab > 0, lab, torch.full_like(lab, -2 if ignore_unpainted else -1))
        ys, xs = torch.nonzero(lab > 0, as_tuple=True)
        if around and len(ys):
            k = int(rng.integers(len(ys)))
            cty, ctx = int(ys[k]) * gh // h, int(xs[k]) * gw // w_
        else:
            cty, ctx = int(rng.integers(gh)), int(rng.integers(gw))
        ty0, tx0 = int(np.clip(cty - T // 2, 0, max(0, gh - T))), int(np.clip(ctx - T // 2, 0, max(0, gw - T)))
        y0, x0 = round(ty0 * h / gh), round(tx0 * w_ / gw)
        fc = F.pad(f[..., ty0:ty0 + T, tx0:tx0 + T], (0, max(0, T - (gw - tx0)), 0, max(0, T - (gh - ty0))))
        imc, lc = im[..., y0:y0 + P, x0:x0 + P], lab[y0:y0 + P, x0:x0 + P]
        imc = F.pad(imc, (0, P - imc.shape[-1], 0, P - imc.shape[-2]))
        lc = F.pad(lc, (0, P - lc.shape[-1], 0, P - lc.shape[-2]), value=-2)  # -2 = padding, ignored
        if rng.random() < 0.5:
            fc, imc, lc = fc.flip(-1), imc.flip(-1), lc.flip(-1)
        if rng.random() < 0.5:
            fc, imc, lc = fc.flip(-2), imc.flip(-2), lc.flip(-2)
        return fc, imc, lc

    for _ in range(steps):
        items = [crop(data[int(i)], rng.random() < 0.8) for i in rng.integers(len(data), size=bs - n_implicit)]
        items += [crop(extra[int(i)], False) for i in rng.integers(len(extra), size=n_implicit)]
        fb = torch.cat([a for a, _, _ in items])
        ib = torch.cat([b for _, b, _ in items])
        lb = torch.stack([c for _, _, c in items])
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lg = net(fb, ib).float().permute(0, 2, 3, 1)
        pos, unl = lb > 0, lb == -1
        if not pos.any():
            continue
        lp = F.cross_entropy(lg[pos], lb[pos] - 1, reduction="none") * wc[lb[pos] - 1]
        loss = lp.sum()
        if unl.any():
            lu = F.cross_entropy(lg[unl], torch.zeros(int(unl.sum()), dtype=torch.long, device=DEV), reduction="none")
            loss = loss + w_implicit * lu.mean() * wc[lb[pos] - 1].sum()  # implicit weight relative to painted
        loss = loss / ((1 + w_implicit) * wc[lb[pos] - 1].sum())
        opt.zero_grad()
        loss.backward()
        opt.step()
    net.eval()
    # Elkan-Noto calibration: mean p_k over held-out strokes of class k (training strokes if none held out)
    s_, n_ = torch.zeros(C, device=DEV), torch.zeros(C, device=DEV)
    s2, n2 = torch.zeros(C, device=DEV), torch.zeros(C, device=DEV)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for ((f, im), lab), hl in zip(data, held):
            p = net(f, im).float().softmax(1)[0]
            for k in range(1, C):
                if hl is not None and (hl == k + 1).any():
                    m = hl == k + 1
                    s_[k] += p[k][m].sum()
                    n_[k] += m.sum()
                m = lab == k + 1
                if m.any():
                    s2[k] += p[k][m].sum()
                    n2[k] += m.sum()
    s_ = torch.where(n_ > 0, s_, s2)
    n_ = torch.where(n_ > 0, n_, n2)
    if calib_mode == "quantile":  # target recall: threshold t_k = (1 - recall)-quantile of p_k on held-out strokes
        thr = torch.ones(C, device=DEV)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            vals = {k: [] for k in range(1, C)}
            for ((f, im), lab), hl in zip(data, held):
                p = net(f, im).float().softmax(1)[0]
                for k in range(1, C):
                    src = hl if hl is not None and (hl == k + 1).any() else lab
                    m = src == k + 1
                    if m.any():
                        vals[k].append(p[k][m])
        for k in range(1, C):
            if vals[k]:
                thr[k] = torch.quantile(torch.cat(vals[k])[:200000], 1 - recall).clamp(0.02, 0.98)
        return net, cnt, (2 * thr).tolist()  # decide() accepts p_k / c_k > 0.5, i.e. p_k > t_k
    calib = torch.ones(C, device=DEV)
    calib[1:] = torch.where(n_[1:] > 0, s_[1:] / n_[1:].clamp_min(1), torch.ones_like(n_[1:]))
    return net, cnt, calib.tolist()

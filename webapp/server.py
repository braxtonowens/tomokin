"""radio_tomo web app back end (FastAPI).

  cd ~/research/radio_tomo && .venv/bin/python webapp/server.py      # http://localhost:8770/
Serves the TypeScript front end (webapp/frontend) and a JSON/PNG API:
  GET  /api/tomograms                   MRC files under data/ (relative paths)
  GET  /api/open?path=                  shape, voxel size, embedded slices, slicepick pick
  GET  /api/slice?path=&z=              8-bit grayscale PNG of slice z (tomogram-wide 0.5-99.5% scaling;
                                        the browser applies the contrast window)
  GET  /api/overlay?path=&z=            processed PCA-RGB PNG at token resolution (embedded or
                                        interpolated slice; 404 otherwise); header X-Overlay = how
  GET  /api/scores?path=                torch-slicepick scores per slice (computed once, cached)
  POST /api/embed {path, zs}            start a background embedding job (gui_embed.Embedder pipeline)
  GET  /api/job                         progress of the current/last job
  GET  /api/ent/{list,info,slice,pca}   slice finder (entropy.html; outputs/entropy, entropy_cache.py)
  /api/seg/{annotations,annot,train,pred,export,wand}   scribble segmentation on the same page (seg_interactive.py)
  GET  /api/bands/items, POST /api/bands/label   band annotation (annotate.html); labels saved to
                                        outputs/nuisance/band_annot/labels.json on every click
Only binds to 127.0.0.1. The GPU is used by one thing at a time (lock).
"""

import io
import json
import sys
import threading
import time
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import os  # noqa: E402

os.environ.setdefault("HF_HOME", str(ROOT / ".hf_cache"))

import mrcfile  # noqa: E402
import numpy as np  # noqa: E402
import uvicorn  # noqa: E402
from fastapi import FastAPI, HTTPException  # noqa: E402
from fastapi.responses import FileResponse, Response  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402
from PIL import Image  # noqa: E402
from pydantic import BaseModel  # noqa: E402

from gui_embed import Embedder  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent))
import seg_interactive as si  # noqa: E402

DATA = ROOT / "data"
FRONT = Path(__file__).resolve().parent / "frontend"
GPU = threading.Lock()
embedder = Embedder(ROOT / "outputs" / "gui_cache")
app = FastAPI(title="radio_tomo")
job = {"state": "idle", "path": None, "done": 0, "total": 0, "message": "", "last": None}


# ---------------------------------------------------------------- tomograms
def resolve(path: str) -> Path:
    p = (ROOT / path).resolve()
    if not p.is_relative_to(ROOT) or not p.is_file():
        raise HTTPException(404, f"no such tomogram: {path}")
    return p


@lru_cache(maxsize=8)
def volume(path: str):
    """Memory-mapped volume and its tomogram-wide 8-bit scaling (0.5/99.5 percentile of ~20 slices)."""
    vol = mrcfile.mmap(resolve(path), mode="r", permissive=True).data
    nz = vol.shape[0]
    sample = np.asarray(vol[np.linspace(nz // 10, nz - 1 - nz // 10, 20).astype(int)], dtype=np.float32)
    lo, hi = np.percentile(sample, [0.5, 99.5])
    return vol, float(lo), float(hi)


def key(path: str) -> str:
    return Path(path).stem


def png(arr: np.ndarray) -> Response:
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG", compress_level=1)
    return Response(buf.getvalue(), media_type="image/png", headers={"Cache-Control": "no-store"})


@app.get("/api/tomograms")
def tomograms():
    files = sorted(p for ext in ("*.mrc", "*.rec") for p in DATA.rglob(ext)
                   if not any(s in p.stem for s in ("_cytoplasm", "_vesicle", "_membrane", "_mitochondrion", "_nucleus")))
    return [str(p.relative_to(ROOT)) for p in files]


@app.get("/api/open")
def open_tomogram(path: str):
    vol, lo, hi = volume(path)
    with mrcfile.open(resolve(path), permissive=True, header_only=True) as m:
        voxel = float(m.voxel_size.x)
    scores = embedder.cache_root / key(path) / "slicepick.npz"
    return {"path": path, "name": key(path), "shape": list(vol.shape), "voxel": voxel,
            "embedded": sorted(embedder.cached(key(path))),
            "pick": int(np.load(scores)["pick"]) if scores.exists() else None}


@app.get("/api/slice")
def slice_png(path: str, z: int):
    vol, lo, hi = volume(path)
    if not 0 <= z < vol.shape[0]:
        raise HTTPException(404, "slice out of range")
    s = np.asarray(vol[z], dtype=np.float32)
    return png(np.clip((s - lo) / (hi - lo) * 255, 0, 255).astype(np.uint8))


@app.get("/api/overlay")
def overlay_png(path: str, z: int):
    name = key(path)
    done = embedder.cached(name)
    f, how = embedder.features(name, z, done) if done else (None, "")
    if f is None:
        return Response(status_code=404)
    rgb = (embedder.rgb(name, f) * 255).round().astype(np.uint8)
    r = png(rgb)
    r.headers["X-Overlay"] = how
    return r


@app.get("/api/scores")
def scores(path: str):
    cache = embedder.cache_dir(key(path)) / "slicepick.npz"
    if not cache.exists():
        from slicepick_scores import compute
        vol, _, _ = volume(path)
        with GPU:
            np.savez(cache, **compute(vol))
    r = np.load(cache)
    return {k.replace("slice/", ""): (r[k].tolist() if r[k].ndim else r[k].item()) for k in r.files}


# ---------------------------------------------------------------- embedding job
class EmbedRequest(BaseModel):
    path: str
    zs: list[int]


def run_job(path: str, zs: list[int]):
    name = key(path)
    vol, _, _ = volume(path)
    try:
        with GPU:
            job.update(message="Loading C-RADIO…")
            embedder.load_model()
            job.update(message="Building the RC template…")
            embedder.template(name, vol)
            rest = [z for z in zs if z not in embedder.cached(name)]
            if not embedder.has_basis(name):
                job.update(message="Fitting the shared 256-PC basis…")
                for z in embedder.fit_basis(name, vol, rest):
                    rest.remove(z)
                    job.update(done=job["done"] + 1, last=z)
            job.update(message="Embedding…")
            for j in range(0, len(rest), 4):
                chunk = rest[j:j + 4]
                for z, f in zip(chunk, embedder.full_features(name, vol, chunk)):
                    embedder.store(name, z, f, vol.shape[1:])
                    job.update(done=job["done"] + 1, last=z)
        job.update(state="done", message=f"Embedded {job['done']} slices.")
    except Exception as e:  # report to the front end
        job.update(state="error", message=f"{type(e).__name__}: {e}")


@app.post("/api/embed")
def embed(req: EmbedRequest):
    if job["state"] == "running":
        raise HTTPException(409, "an embedding job is already running")
    vol, _, _ = volume(req.path)
    zs = sorted({z for z in req.zs if 0 <= z < vol.shape[0]} - embedder.cached(key(req.path)))
    job.update(state="running" if zs else "done", path=req.path, done=0, total=len(zs), last=None,
               message="Starting…" if zs else "Nothing new to embed.")
    if zs:
        threading.Thread(target=run_job, args=(req.path, zs), daemon=True).start()
    return job


@app.get("/api/job")
def job_status():
    return job


# ---------------------------------------------------------------- band annotation
BANDS = ROOT / "outputs" / "nuisance" / "band_annot"
LABELS = BANDS / "labels.json"
LABEL_LOCK = threading.Lock()


class BandLabel(BaseModel):
    id: str
    label: str
    note: str = ""


@app.get("/api/bands/items")
def band_items():
    items = json.loads((BANDS / "sample_items.json").read_text()) if (BANDS / "sample_items.json").exists() else []
    labels = json.loads(LABELS.read_text()) if LABELS.exists() else {}
    return {"items": items, "labels": labels}


@app.post("/api/bands/label")
def band_label(b: BandLabel):
    if b.label not in ("artifact", "structure", "unsure", ""):
        raise HTTPException(400, "label must be artifact / structure / unsure")
    with LABEL_LOCK:
        labels = json.loads(LABELS.read_text()) if LABELS.exists() else {}
        if b.label:
            labels[b.id] = {"label": b.label, "note": b.note, "time": time.strftime("%Y-%m-%d %H:%M:%S")}
        else:
            labels.pop(b.id, None)
        tmp = LABELS.with_suffix(".tmp")
        tmp.write_text(json.dumps(labels, indent=1))
        tmp.replace(LABELS)
    return {"n_labels": len(labels)}


if (BANDS / "img").exists():
    app.mount("/bandimg", StaticFiles(directory=BANDS / "img"), name="bandimg")


# ---------------------------------------------------------------- slice entropy explorer (entropy.html)
ENT = ROOT / "outputs" / "entropy"
FULL = ROOT / "data" / "popsicle_full"


def ent_info(run: str) -> dict:
    f = ENT / Path(run).name / "entropy.json"
    if not f.exists():
        raise HTTPException(404, f"no entropy cache for {run}")
    return json.loads(f.read_text())


@lru_cache(maxsize=8)
def ent_volume(run: str):
    """Memory-mapped tomogram + tomogram-wide 0.5/99.5% scaling of gray-8 slabs (20 sampled)."""
    info = ent_info(run)
    vol = np.load(FULL / info["set"] / info["run"] / "tomo.npy", mmap_mode="r")
    nz = vol.shape[0]
    zs = np.linspace(nz // 10, nz - 1 - nz // 10, 20).astype(int)
    sample = np.stack([gray8(vol, z) for z in zs])
    lo, hi = np.percentile(sample, [0.5, 99.5])
    return vol, float(lo), float(hi)


def gray8(vol, z):
    z0 = int(np.clip(z - 4, 0, vol.shape[0] - 8))
    return np.asarray(vol[z0:z0 + 8], dtype=np.float32).mean(0)


@lru_cache(maxsize=8)
def ent_pca(run: str):
    """PCA-35 memmap + per-channel 1/99% scaling of PCs 1-3 over the whole tomogram."""
    P = np.load(ENT / Path(run).name / "pca35.npy", mmap_mode="r")
    s = np.asarray(P[:: max(1, len(P) // 40), :3], dtype=np.float32)
    lo, hi = np.percentile(s, 1, axis=(0, 2, 3)), np.percentile(s, 99, axis=(0, 2, 3))
    return P, lo, hi


@app.get("/api/ent/list")
def ent_list(project: str = ""):
    """Tomograms with an entropy cache; with project=, only that project's runs, in project order."""
    keep = [r["run"] for r in si.project(project)["runs"]] if project else None
    out = []
    for f in sorted(ENT.glob("*/entropy.json")):
        i = json.loads(f.read_text())
        if keep is None or i["run"] in keep:
            out.append({k: i[k] for k in ("run", "set", "nz", "split", "argmin_raw", "argmin_smooth")})
    return sorted(out, key=lambda i: keep.index(i["run"])) if keep else out


def best_slice(h, cut=0.2, sigma=8.0, vmin=0.1):
    """Same rule as the page: very smooth curve, ends cut, deepest valley between humps, else centre-weighted."""
    h = np.asarray(h, float)
    n = len(h)
    R = int(np.ceil(3 * sigma))
    k = np.exp(-np.arange(-R, R + 1) ** 2 / (2 * sigma ** 2))
    num = np.convolve(h, k, mode="same")
    den = np.convolve(np.ones(n), k, mode="same")
    c = num / den
    lo, hi = int(np.floor(cut * n)), max(int(np.floor(cut * n)) + 1, int(np.ceil((1 - cut) * n)))
    amp = c[lo:hi].max() - c[lo:hi].min() or 1.0
    best = None
    for v_ in range(max(lo, 1), min(hi, n - 1)):
        if not (c[v_] < c[v_ - 1] and c[v_] <= c[v_ + 1]):
            continue
        l, lm = v_, v_
        while l > 0 and c[l - 1] >= c[v_]:
            l -= 1
            lm = l if c[l] > c[lm] else lm
        r, rm = v_, v_
        while r < n - 1 and c[r + 1] >= c[v_]:
            r += 1
            rm = r if c[r] > c[rm] else rm
        if lm in (0, v_) or rm in (n - 1, v_):
            continue
        d = min(c[lm], c[rm]) - c[v_]
        if d / amp >= vmin and (best is None or d > best[1]):
            best = (v_, d)
    if best:
        return int(best[0])
    zc, sd = (n - 1) / 2, 0.2 * n
    zz = np.arange(lo, hi)
    s = np.exp(-(zz - zc) ** 2 / (2 * sd * sd)) * (c[lo:hi].max() - c[lo:hi]) / amp
    return int(zz[np.argmax(s)])


class PriorLog(BaseModel):
    project: str
    prior: float


@app.post("/api/project/prior")
def project_prior(p: PriorLog):
    si.log_prior(p.project, p.prior)
    return {"ok": True}


@app.get("/api/project/cards")
def project_cards(name: str):
    """One card per project tomogram: its suggested slice and how many slices carry scribbles."""
    out = []
    for i, r in enumerate(si.project(name)["runs"]):
        try:
            info = ent_info(r["run"])
        except HTTPException:
            continue
        out.append({"i": i + 1, "run": r["run"], "nz": info["nz"], "z": best_slice(info["H_smooth"]),
                    "annotated": len(si.load_annotations(r["run"])["slices"])})
    return out


@app.get("/api/seg/thumb")
def seg_thumb(run: str, z: int, project: str = "", colors: str = "", size: int = 320, v: int = 0, prior: float = 1.0):
    """Gallery thumbnail: gray-8 slab with the project's current prediction blended in (alpha ~ confidence)."""
    info = ent_info(run)
    vol, lo, hi = ent_volume(run)
    g = np.clip((gray8(vol, z) - lo) / (hi - lo), 0, 1)
    rgb = np.repeat(g[..., None], 3, -1)
    if project and colors:
        with GPU:
            r = si.predict_slice(run, z, g.shape, 1, f"project:{project}", info["set"], prior)
        if r is not None:
            lab, conf, _ = r
            pal = np.array([[int(c[i:i + 2], 16) for i in (0, 2, 4)] for c in colors.split(",")], float) / 255
            k = len(pal)
            a = np.where(lab > 0, 0.45, 0.0)[..., None]  # unassigned stays plain
            rgb = rgb * (1 - a) + pal[np.minimum(lab, k - 1)] * a
    im = Image.fromarray((rgb * 255).astype(np.uint8))
    im.thumbnail((size, size), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=85)
    return Response(buf.getvalue(), media_type="image/jpeg", headers={"Cache-Control": "no-store"})


@app.get("/api/project")
def project_info(name: str):
    try:
        p = si.project(name)
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))
    return {"name": p["name"], "n_runs": len(p["runs"])}


@app.get("/api/ent/info")
def ent_info_api(run: str):
    return ent_info(run)


@app.get("/api/ent/slice")
def ent_slice(run: str, z: int):
    vol, lo, hi = ent_volume(run)
    if not 0 <= z < vol.shape[0]:
        raise HTTPException(404, "slice out of range")
    return png(np.clip((gray8(vol, z) - lo) / (hi - lo) * 255, 0, 255).astype(np.uint8))


class SegSlice(BaseModel):
    run: str
    z: int
    png: str  # base64 PNG, gray value = class id (0 = unlabelled)
    classes: list[dict]


class SegTrain(BaseModel):
    run: str
    n_classes: int
    project: str = ""
    prior: float = 0.0  # background prior; 0 = the label-free default (ratio of painted pixels)
    explicit: bool = False  # True only for fits after a stroke or Segment; automatic refits are refused after Clear guess
    head: str = "mlp"  # mlp | unet


@app.get("/api/seg/annotations")
def seg_annotations(run: str):
    return si.load_annotations(run)


@app.get("/api/seg/annot")
def seg_annot_slice(run: str, z: int):
    b = si.slice_png(run, z)
    return Response(b, media_type="image/png", headers={"Cache-Control": "no-store"}) if b else Response(status_code=404)


@app.post("/api/seg/annot")
def seg_save(s: SegSlice):
    import base64
    return {"n": si.save_slice(s.run, s.z, base64.b64decode(s.png.split(",")[-1]), s.classes)}


@app.post("/api/seg/clear")
def seg_clear(t: SegTrain):
    """Clear the current guess: drop the classifier (project-wide in project mode). Scribbles are untouched."""
    return {"cleared": si.clear(f"project:{t.project}" if t.project else t.run)}


class RemoveClass(BaseModel):
    run: str
    project: str = ""
    k: int


@app.post("/api/seg/remove_class")
def seg_remove_class(r: RemoveClass):
    """Delete a class from every tomogram of the project (or the one tomogram); the classifier is dropped (its class
    count changed) and the page refits."""
    runs = [x["run"] for x in si.project(r.project)["runs"]] if r.project else [r.run]
    key = f"project:{r.project}" if r.project else r.run
    try:
        n = si.remove_class(runs, r.k)
    except ValueError as e:
        raise HTTPException(400, str(e))
    si._models.pop(key, None)
    si._history.pop(key, None)  # earlier models have a different set of classes: undo cannot restore them
    return {"changed": n}


@app.post("/api/seg/clear_scribbles")
def seg_clear_scribbles(t: SegTrain):
    """Delete all scribbles of the project (or of the one tomogram) and the classifier built from them."""
    runs = [r["run"] for r in si.project(t.project)["runs"]] if t.project else [t.run]
    n = si.clear_scribbles(runs)
    si.clear(f"project:{t.project}" if t.project else t.run)
    return {"deleted_slices": n, "tomograms": len(runs)}


# ---------------------------------------------------------------- nnInteractive (separate service, .venv_nnint)
NNI = "http://127.0.0.1:8771"
_nni_proc = {"p": None}


def nni(method, path, **kw):
    import requests
    try:
        r = requests.request(method, NNI + path, timeout=120, **kw)
    except requests.ConnectionError:
        raise HTTPException(503, "nnInteractive service is not running")
    if r.status_code >= 400:
        raise HTTPException(r.status_code, r.text[:300])
    return r


@app.post("/api/nni/start")
def nni_start():
    """Start the nnInteractive service (own venv) if it is not running; the model loads in ~10-20 s."""
    import subprocess
    try:
        return {"running": True, **nni("GET", "/status").json()}
    except HTTPException:
        pass
    if _nni_proc["p"] is None or _nni_proc["p"].poll() is not None:
        log = open(ROOT / "outputs" / "nnint_service.log", "a")
        _nni_proc["p"] = subprocess.Popen([str(ROOT / ".venv_nnint" / "bin" / "python"), str(ROOT / "nnint_service.py")],
                                          cwd=ROOT, stdout=log, stderr=log)
    return {"running": False, "starting": True}


class NniOpen(BaseModel):
    run: str


@app.post("/api/nni/open")
def nni_open(o: NniOpen):
    return nni("POST", "/open", json={"set": ent_info(o.run)["set"], "run": o.run}).json()


@app.post("/api/nni/new")
def nni_new():
    return nni("POST", "/new").json()


@app.post("/api/nni/box")
def nni_box(b: dict):
    return nni("POST", "/box", json=b).json()


@app.post("/api/nni/point")
def nni_point(p: dict):
    return nni("POST", "/point", json=p).json()


@app.get("/api/nni/slice")
def nni_slice(z: int, v: int = 0):
    return Response(nni("GET", f"/slice?z={z}").content, media_type="image/png", headers={"Cache-Control": "no-store"})


@app.post("/api/nni/accept")
def nni_accept(a: dict):
    return nni("POST", "/accept", json=a).json()


@app.get("/api/seg/dense")
def seg_dense_slice(run: str, z: int):
    """Accepted objects on slice z as a gray PNG (value = class id), 404 if the tomogram has none."""
    f = si.OUT / Path(run).name / "dense.npy"
    if not f.exists():
        return Response(status_code=404)
    d = np.load(f, mmap_mode="r")
    return png(np.asarray(d[int(np.clip(z, 0, len(d) - 1))]))


class Complete(BaseModel):
    run: str
    complete: bool


@app.post("/api/seg/complete")
def seg_complete(c: Complete):
    si.set_complete(c.run, c.complete)
    return {"run": c.run, "complete": c.complete}


@app.get("/api/seg/complete")
def seg_complete_get(run: str):
    return {"run": run, "complete": si.complete_flag(run),
            "has_dense": (si.OUT / Path(run).name / "dense.npy").exists()}


@app.post("/api/seg/profile")
def seg_profile(t: SegTrain):
    """Uncertainty profile of the current classifier on another tomogram, without refitting."""
    info = ent_info(t.run)
    with GPU:
        r = si.profile(f"project:{t.project}" if t.project else t.run, info["set"], t.run, t.prior or 1.0)
    if r is None:
        raise HTTPException(404, "no classifier")
    return r


@app.post("/api/seg/train")
def seg_train(t: SegTrain):
    info = ent_info(t.run)
    with GPU:
        try:
            si.check_cleared(f"project:{t.project}" if t.project else t.run, t.explicit)
            key = f"project:{t.project}" if t.project else t.run
            r = (si.train_project(t.project, info["set"], t.run, t.n_classes, t.prior or None, t.head) if t.project
                 else si.train(info["set"], t.run, t.n_classes, t.prior or None))
            si.remember_fit(key, r)
            return r
        except ValueError as e:
            raise HTTPException(400, str(e))


class SegRestore(BaseModel):
    run: str
    project: str = ""
    version: int
    prior: float = 1.0


@app.post("/api/seg/restore")
def seg_restore(t: SegRestore):
    """Undo: make the model of an earlier fit current again (no retraining)."""
    info = ent_info(t.run)
    with GPU:
        r = si.restore(f"project:{t.project}" if t.project else t.run, t.version, info["set"], t.run, t.prior)
    if r is None:
        raise HTTPException(404, "that model is no longer kept")
    return r


@app.get("/api/seg/pred")
def seg_pred(run: str, z: int, colors: str, zavg: int = 1, mode: str = "class", v: int = 0, project: str = "",
             prior: float = 1.0):
    """RGBA overlay of the prediction. mode=class: class colours, alpha rising with confidence;
    mode=unsure: amber where the two most likely classes are close (1 - margin). v = model version (cache busting)."""
    info = ent_info(run)
    with GPU:
        r = (si.predict_slice(run, z, tuple(info["shape"][1:]), zavg, f"project:{project}", info["set"], prior)
             if project else si.predict_slice(run, z, tuple(info["shape"][1:]), zavg, prior=prior))
    if r is None:
        return Response(status_code=404)
    lab, conf, uns = r
    rgba = np.zeros(lab.shape + (4,), np.uint8)
    if mode == "unsure":
        rgba[..., :3] = (255, 176, 32)
        rgba[..., 3] = (np.clip(uns, 0, 1) ** 1.5 * 235).astype(np.uint8)
        return png(rgba)
    pal = np.array([[int(c[i:i + 2], 16) for i in (0, 2, 4)] for c in colors.split(",")], np.uint8)
    k = len(pal)
    rgba[..., :3] = pal[np.minimum(lab, k - 1)]
    rgba[..., 3] = np.where(lab > 0, 200, 0)  # constant; class 1 = "unassigned" is left uncoloured
    return png(rgba)


@app.post("/api/seg/warm")
def seg_warm(t: SegTrain):
    """Load the run's features onto the GPU ahead of the first fit."""
    info = ent_info(t.run)
    with GPU:
        si.gfeats(info["set"], t.run)
    return {"ok": True}


@app.post("/api/seg/export")
def seg_export(t: SegTrain):
    info = ent_info(t.run)
    with GPU:
        try:
            if t.project:
                return {"path": si.export(t.run, tuple(info["shape"][1:]), 1, f"project:{t.project}", info["set"],
                                          t.prior or 1.0)}
            return {"path": si.export(t.run, tuple(info["shape"][1:]), 1, prior=t.prior or 1.0)}
        except ValueError as e:
            raise HTTPException(400, str(e))


@app.get("/api/seg/wand")
def seg_wand(run: str, z: int, x: float, y: float, thr: float = 0.6):
    info = ent_info(run)
    m = si.wand(info["set"], run, z, x, y, tuple(info["shape"][1:]), thr)
    return png((m * 255).astype(np.uint8))


@app.get("/api/ent/pca")
def ent_pca_png(run: str, z: int):
    P, lo, hi = ent_pca(run)
    if not 0 <= z < len(P):
        raise HTTPException(404, "slice out of range")
    x = np.asarray(P[z, :3], dtype=np.float32)
    x = np.clip((x - lo[:, None, None]) / (hi - lo + 1e-8)[:, None, None], 0, 1)
    return png((x.transpose(1, 2, 0) * 255).round().astype(np.uint8))


# ---------------------------------------------------------------- front end
@app.get("/")
def index():
    return FileResponse(FRONT / "index.html", headers={"Cache-Control": "no-store"})


class FrontFiles(StaticFiles):
    """The front end, always revalidated (a changed page or script is picked up on a normal reload)."""
    async def get_response(self, path, scope):
        r = await super().get_response(path, scope)
        r.headers["Cache-Control"] = "no-cache"
        return r


app.mount("/", FrontFiles(directory=FRONT), name="frontend")

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8770, log_level="warning")

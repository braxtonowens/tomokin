"""Cell-hunt project: 20 Hylemonella gracilis tomograms (CZ portal dataset 10161, same microscope / camera / tilt scheme).

8 runs have POPSICLE labels (already in data/popsicle_full/bacterial at 20 A, used only for hidden scoring); 12 more
runs of the same dataset are drawn at random (seed 0) from those without labels and fetched at their native
19.73 A (1.4% from 20 A, used as is). Their tilt-series metadata is printed to confirm the acquisition matches.
Each new run is embedded exactly like embed_full.py (rc_g8, PCA-256 with the bacterial basis fitted on the POPSICLE
train runs) and gets an entropy cache (entropy_cache.py) for the slice finder.
Output: data/popsicle_full/hylemonella_extra/<run>/{tomo.npy, meta.json, feats.npy, T.pt};
outputs/cell_hunt/project.json (the 20 runs, which are labelled).
"""

import json
from pathlib import Path

import numpy as np
import torch
from cryoet_data_portal import Client, Run, TiltSeries, Tomogram

import embed_full as ef
import entropy_cache as ec
from popsicle_full import read_mrc

DS, N_EXTRA = 10161, 12
OUT = Path("data/popsicle_full/hylemonella_extra")
PROJ = Path("outputs/cell_hunt/project.json")


def main():
    c = Client()
    labelled = sorted(json.load(open(p))["run"] for p in Path("data/popsicle_full/bacterial").glob("*/meta.json")
                      if json.load(open(p))["tomo_url"].split("/")[3] == str(DS))
    runs = sorted(Run.find(c, [Run.dataset_id == DS]), key=lambda r: r.name)
    pool = [r for r in runs if r.name not in labelled]
    pick = [pool[i] for i in sorted(np.random.default_rng(0).choice(len(pool), N_EXTRA, replace=False))]
    print(f"dataset {DS}: {len(runs)} runs, {len(labelled)} labelled, picked {[r.name for r in pick]}", flush=True)
    for r in [x for x in runs if x.name in labelled][:1] + pick:
        ts = TiltSeries.find(c, [TiltSeries.run_id == r.id])
        for t in ts[:1]:
            print(f"  {r.name}: {t.microscope_model} {t.camera_model} px {t.pixel_spacing} tilt {t.tilt_min}..{t.tilt_max} "
                  f"step {t.tilt_step} flux {t.total_flux}", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    for r in pick:
        d = OUT / r.name
        if (d / "meta.json").exists():
            continue
        d.mkdir(exist_ok=True)
        tomos = [t for t in Tomogram.find(c, [Tomogram.run_id == r.id]) if abs(t.voxel_spacing - 19.73) < 0.05]
        t = sorted(tomos, key=lambda t: t.id)[0]
        vol = read_mrc(t.https_mrc_file)
        np.save(d / "tomo.npy", vol)
        json.dump(dict(run=r.name, portal_run_id=r.id, split="unlabelled", classes=[], present=[],
                       voxel=float(t.voxel_spacing), tomo_url=t.https_mrc_file, shape=list(vol.shape),
                       dataset=DS), open(d / "meta.json", "w"), indent=1)
        print(f"  fetched {r.name} {vol.shape} at {t.voxel_spacing} A", flush=True)
    model = ef.load_model()
    basis = torch.load(Path("data/popsicle_full/bacterial/pca.pt"))
    for m in ef.runs("hylemonella_extra"):
        ef.embed_run(model, m, basis)
        print(f"  embedded {m['run']}", flush=True)
    for set_, name in [("bacterial", n) for n in labelled] + [("hylemonella_extra", r.name) for r in pick]:
        ec.run_one(model, set_, name, ec.SIGMA)
    PROJ.parent.mkdir(parents=True, exist_ok=True)
    json.dump({"name": "Hylemonella gracilis (10161)", "dataset": DS,
               "runs": [{"set": "bacterial", "run": n, "labelled": True} for n in labelled] +
                       [{"set": "hylemonella_extra", "run": r.name, "labelled": False} for r in pick]},
              open(PROJ, "w"), indent=1)
    print(f"project with {len(labelled) + len(pick)} runs -> {PROJ}", flush=True)


if __name__ == "__main__":
    main()

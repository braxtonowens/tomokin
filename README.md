# tomokin

**Scribble on one cryo-ET slice, segment the whole tomogram, and find its kin in every tomogram like it.**

tomokin is a small interactive tool for segmenting cellular cryo-electron tomograms. It is built on frozen
[C-RADIOv4-H](https://huggingface.co/nvidia/C-RADIOv4-H) foundation-model features.

- You paint a few strokes per class.
- A small U-Net on the RADIO features plus the image retrains in about a second, and predicts every slice.
- One classifier is shared by all tomograms of a project. Strokes on one tomogram segment the others, and a gallery
  shows all of them live.

Version **0.0.2**: an early research prototype.

## What it does

- **Slice finder.** Every slice gets a spectral entropy computed from its RADIO patch features. The page suggests the
  slices in the valleys between entropy humps, which usually hold the biological content. The top and bottom 20% of
  the volume (reconstruction artefacts) are ignored.
- **Annotation tools.** Brush, eraser, lasso fill and a magic wand that selects regions of similar RADIO features.
  Classes can be renamed (✎) and removed (×; this deletes their scribbles on every tomogram of the project). Class 1
  is *unassigned*: a correction brush for places that belong to none of your classes, and it is always kept.
- **Live prediction.** Every edit refits the U-Net in about 1–2 s, warm-started from the previous fit. The prediction
  overlay updates on the current slice, and an *Unsure* layer shows where the model hesitates. Undo (Ctrl+Z) restores
  the labels *and* the exact model from before the edit, without retraining.
- **Entropy graph.** Under the image: the suggested slices (numbered dots), the slices the model is least sure about
  ("?" markers) and one coloured bar per annotated slice. Click any of them to go to that slice.
- **Projects.** A project is a list of tomograms, typically from one dataset. Scribbles from all of them train one
  classifier. The **Gallery** shows every tomogram's suggested slice with the current prediction.
- **Export.** Writes a full-resolution label volume per tomogram.

## How it works

1. **Features.** Each slice is an average of 8 z-slices, passed once through C-RADIOv4-H at native resolution (bf16).
   The model's position-locked pattern is removed by subtracting a template built from 12 randomly flipped and rolled
   reference slices of the same tomogram. The features are stored as a 256-d PCA.
2. **Slice entropy.** For each slice, take 2000 patch embeddings, normalise them, form the Gram matrix divided by its
   trace, and compute the Shannon entropy of its eigenvalues. `entropy_fast.py` computes this directly from the
   stored PCA features.
3. **U-Net head.** The RADIO features are reduced 256 → 32 and upsampled, then combined with the image at quarter
   resolution. The network has 3 levels and about 0.23 M parameters. It trains only on painted pixels (class-balanced
   cross-entropy, random crops aligned to the token grid, flips), and prediction happens at 4-px resolution. A
   background prior is set automatically from the ratio of painted pixels.
4. **kNN half.** After every fit, the features at the painted pixels (up to 60k, in the painted class ratio) form a
   nearest-neighbour bank. The shown prediction is the average of the U-Net's class probabilities and the class
   fractions among each token's 16 most similar painted pixels. In a simulated 16-slice annotation session on
   POPSICLE bacteria this scored a mean Dice of 0.45, against 0.34 for the U-Net and 0.40 for kNN alone.

## Requirements

- Linux with an NVIDIA GPU. It was developed on an RTX 5080 (16 GB), which needs a PyTorch build for CUDA 12.8.
- Python 3.12.
- Disk space: a 500-slice tomogram is about 1.7 GB as float32, and its features about 0.9 GB.

```bash
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

The C-RADIOv4-H weights download from Hugging Face on first use, into `.hf_cache/`.

## Getting data in

The tool works on whole tomograms with their features in `data/popsicle_full/<set>/<run>/`. The scripts below fetch
the [POPSICLE](https://huggingface.co/datasets/biohub/popsicle) benchmark volumes from the
[CryoET Data Portal](https://cryoetdataportal.czscience.com). The split metadata is included in `data/popsicle_hf/`.

```bash
.venv/bin/python popsicle_fetch.py --help     # low-level portal helpers
.venv/bin/python popsicle_full.py             # whole POPSICLE volumes + masks (bacterial, yeast), ~150 GB
.venv/bin/python embed_full.py                # features for every slice (PCA-256 per set)
.venv/bin/python entropy_fast.py --runs bacterial/<run> ...   # entropy caches for the slice finder
```

`cellhunt_fetch.py` is a complete example of a project. It fetches 12 extra *Hylemonella gracilis* tomograms from
the same portal dataset as 8 POPSICLE runs, embeds them, and writes `outputs/cell_hunt/project.json`:

```json
{"name": "Hylemonella gracilis (10161)", "runs": [{"set": "bacterial", "run": "ycw2012-09-23-21", "labelled": true}, ...]}
```

Any list of embedded runs in `outputs/<name>/project.json` is a project.

## Running

```bash
.venv/bin/python webapp/server.py            # serves on http://127.0.0.1:8770 (localhost only; no login)
```

Open `http://127.0.0.1:8770/entropy.html?project=<name>`, or `entropy.html` alone to use single tomograms. The
server listens only on localhost, because the app has no authentication and can delete annotations. To use it from
another machine, forward the port over SSH (`ssh -N -L 8770:127.0.0.1:8770 you@this-machine`).

Scribbles are saved per slice in `outputs/seg_interactive/<run>/annot/`. Every project fit is logged in
`outputs/<project>/fits/`, so a session can be scored afterwards against ground truth where it exists.

**Keys:**

| Key | Action |
|---|---|
| B | brush |
| E | eraser |
| L | lasso |
| W | magic wand |
| H or Space | pan |
| 1–9 | select class |
| `[` / `]` | brush size |
| ↑ / ↓ (Shift = ±10) | change slice |
| P | prediction layer |
| U | unsure layer |
| O | feature PCA overlay |
| G | gallery |
| F | fit view |
| Ctrl+Z | undo |

## Building the front end

The compiled JavaScript is in `webapp/frontend/dist/`. After editing `webapp/frontend/src/*.ts`, rebuild with:

```bash
cd webapp/frontend && npm install && npx tsc -p .
```

## Licences and data

- **Model:** C-RADIOv4-H weights are under the
  [NVIDIA Open Model License](https://developer.download.nvidia.com/licenses/nvidia-open-model-license-agreement-june-2024.pdf).
  They are downloaded, not redistributed.
- **Data:** the POPSICLE data and CryoET Data Portal depositions are CC0. The split metadata in `data/popsicle_hf/`
  comes from [biohub/popsicle](https://huggingface.co/datasets/biohub/popsicle).
- **Code:** [MIT](LICENSE). This covers the code in this repository only, not the model weights or the data.

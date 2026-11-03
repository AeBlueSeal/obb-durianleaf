# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.5
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Phase 4.3 — Hard-negative mining (Colab/GPU)
#
# Runs the current (pre-fix) `models/yolo11m-obb.pt` on the Phase 4.1
# background-only crops (`bgneg_*`) — images that by construction contain no
# leaf, so any detection is a false positive. Any false-positive Algal
# spot / Other detection is saved as an extra hard-negative training crop
# (empty label), added to train for the next training round.
#
# Notebook form of `analysis/dataset/phase4_hard_negative_mining.py` — see
# `results/source_confound_pipeline.md` (Phase 4). Run after
# `phase4_background_negatives.py` (local step) has populated `bgneg_*` images
# in Drive.

# %%
# !pip install -q ultralytics opencv-python-headless pyyaml

# %% [markdown]
# Mounts Google Drive and points `PROJECT_ROOT` at the project folder, per
# `results/colab_project_structure.md` (same convention as
# `notebooks/00_setup_project_structure.ipynb`).

# %%
from google.colab import drive
drive.mount('/content/drive')

PROJECT_NAME = 'ProjZ'
PROJECT_ROOT = f'/content/drive/MyDrive/{PROJECT_NAME}'

# %% [markdown]
# ## Paths and constants

# %%
import os, glob, uuid

import cv2
import yaml
from ultralytics import YOLO

DATA_YAML  = f"{PROJECT_ROOT}/data/dataset_resplit.yaml"
WEIGHTS    = f"{PROJECT_ROOT}/models/yolo11m-obb.pt"
CONF_THR   = 0.25
IMG_SZ     = 640
OUTPUT_DIR = f"{PROJECT_ROOT}/analysis/dataset/outputs/resplit"

with open(DATA_YAML) as f:
    cfg = yaml.safe_load(f)

CLASSES = cfg["names"]
# The yaml's `path:` field is a local machine path baked in at dataset-build
# time (e.g. /Volumes/...) — override with the Drive-mounted location instead
# of trusting it.
DATA_ROOT = f"{PROJECT_ROOT}/data/datasetv3_2_resplit"
TRAIN_IMG = os.path.join(DATA_ROOT, "train", "images")
TRAIN_LBL = os.path.join(DATA_ROOT, "train", "labels")
TARGET_CLASSES = {"Algal spot", "Other"}

# %% [markdown]
# ## Probe background-only crops with the current model

# %%
bg_paths = glob.glob(os.path.join(TRAIN_IMG, "bgneg_*"))
print(f"Probing {len(bg_paths)} background-only images with the current model …")

model = YOLO(WEIGHTS)

false_positives = 0
saved = 0
per_class_fp = {c: 0 for c in TARGET_CLASSES}

for img_path in bg_paths:
    result = model.predict(source=img_path, imgsz=IMG_SZ, conf=CONF_THR, verbose=False)[0]
    if len(result.obb) == 0:
        continue

    img = cv2.imread(img_path)
    if img is None:
        continue
    h, w = img.shape[:2]

    for cls_id, box in zip(result.obb.cls.int().tolist(), result.obb.xyxyxyxy.tolist()):
        cls_name = CLASSES[cls_id]
        if cls_name not in TARGET_CLASSES:
            continue
        false_positives += 1
        per_class_fp[cls_name] += 1

        xs = [p[0] for p in box]
        ys = [p[1] for p in box]
        x0, x1 = max(0, int(min(xs))), min(w, int(max(xs)))
        y0, y1 = max(0, int(min(ys))), min(h, int(max(ys)))
        if x1 - x0 < 8 or y1 - y0 < 8:
            continue

        crop = img[y0:y1, x0:x1]
        stem = os.path.splitext(os.path.basename(img_path))[0]
        out_stem = f"hardneg_{uuid.uuid4().hex[:8]}_{stem}"
        cv2.imwrite(os.path.join(TRAIN_IMG, out_stem + ".jpg"), crop)
        open(os.path.join(TRAIN_LBL, out_stem + ".txt"), "w").close()
        saved += 1

print(f"\nFalse positives on background-only images: {false_positives}")
for c, n in per_class_fp.items():
    print(f"  {c}: {n}")
print(f"Saved {saved} hard-negative crops into {TRAIN_IMG}")

# %% [markdown]
# ## Save report

# %%
os.makedirs(OUTPUT_DIR, exist_ok=True)
with open(os.path.join(OUTPUT_DIR, "phase4_hard_negative_report.txt"), "w") as f:
    f.write(f"Background-only images probed: {len(bg_paths)}\n")
    f.write(f"False positives (Algal spot / Other): {false_positives}\n")
    for c, n in per_class_fp.items():
        f.write(f"  {c}: {n}\n")
    f.write(f"Hard-negative crops saved: {saved}\n")
    if false_positives == 0:
        f.write("\nNo false positives found — either the current model doesn't "
                "exploit this shortcut on these crops, or CONF_THR is too high. "
                "Re-run Test C2 after retraining to confirm the shortcut is gone, "
                "not just untriggered here.\n")

print(f"Saved -> {OUTPUT_DIR}/phase4_hard_negative_report.txt")

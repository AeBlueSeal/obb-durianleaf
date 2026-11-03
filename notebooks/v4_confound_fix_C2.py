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
# # Test C2 — Background-only disease probe (v4_confound_fix)
#
# Masks out every annotated leaf polygon (filling with neutral grey) and runs
# the **v4_confound_fix retrained model** on background-only images. If the
# model still predicts the correct disease class well above chance
# (1/7 ≈ 14%), disease labels are leaking through background style — direct
# shortcut evidence.
#
# Runs against the weights produced by `notebooks/v4_confound_fix_train.py`
# (`runs/v4_confound_fix/train/weights/best.pt`).  Original baseline version:
# `notebooks/source_confound_C2.py`.
#
# See `results/source_confound_pipeline.md` (Phase 7).

# %%
import sys, subprocess
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "ultralytics", "pyyaml", "pandas", "opencv-python-headless"])

# %% [markdown]
# ## Setup — imports, paths & constants

# %%
import os, sys, glob
import numpy as np
import pandas as pd
import cv2
import yaml
from collections import defaultdict

from google.colab import drive
drive.mount('/content/drive')

PROJECT_NAME = 'ProjZ'
PROJECT_ROOT = f'/content/drive/MyDrive/{PROJECT_NAME}'

# ---------------------------------------------------------------------------
# Point at the v4_confound_fix retrained weights (not the old baseline model)
# ---------------------------------------------------------------------------
EXPERIMENT   = "v4_confound_fix"
DATA_YAML    = f"{PROJECT_ROOT}/data/dataset_resplit.yaml"
WEIGHTS      = f"{PROJECT_ROOT}/runs/{EXPERIMENT}/train/weights/best.pt"
CONF_THR     = 0.25
IMG_SZ       = 640
OUTPUT_DIR   = f"{PROJECT_ROOT}/runs/{EXPERIMENT}/test/C2_background_probe"

with open(DATA_YAML) as f:
    cfg = yaml.safe_load(f)

CLASSES = cfg["names"]
# The yaml's `path:` field is a local machine path baked in at dataset-build
# time — override with the Drive-mounted location instead of trusting it.
DATA_ROOT = f"{PROJECT_ROOT}/data/datasetv3_2_resplit"
TEST_IMG  = os.path.join(DATA_ROOT, "test", "images")
TEST_LBL  = os.path.join(DATA_ROOT, "test", "labels")

from ultralytics import YOLO
model = YOLO(WEIGHTS)

# %% [markdown]
# ## Helper functions

# %%
def load_obb_polys(label_path, img_w, img_h):
    """Return list of (cls_id, poly_px) where poly_px is (4,2) int array."""
    polys = []
    if not os.path.exists(label_path):
        return polys
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 9:
                continue
            cls_id = int(parts[0])
            coords = np.array(parts[1:9], dtype=float).reshape(4, 2)
            coords[:, 0] *= img_w
            coords[:, 1] *= img_h
            polys.append((cls_id, coords.astype(np.int32)))
    return polys


def mask_leaves(img_bgr, polys):
    """Grey-out every leaf polygon; keep background intact."""
    out = img_bgr.copy()
    for _, poly in polys:
        cv2.fillPoly(out, [poly], (127, 127, 127))
    return out


CHANCE = 1.0 / len(CLASSES)

# %% [markdown]
# ## Mask leaves and probe the model on background-only images

# %%
gt_counts = defaultdict(int)      # ground-truth class → images containing this class
hit_counts = defaultdict(int)     # ground-truth class → images where masked prediction matched
total_imgs = 0
detected_any = 0

img_paths = sorted(glob.glob(os.path.join(TEST_IMG, "*")))

for img_path in img_paths:
    stem      = os.path.splitext(os.path.basename(img_path))[0]
    lbl_path  = os.path.join(TEST_LBL, stem + ".txt")

    img = cv2.imread(img_path)
    if img is None:
        continue
    h, w = img.shape[:2]

    polys = load_obb_polys(lbl_path, w, h)
    if not polys:
        continue

    masked = mask_leaves(img, polys)
    total_imgs += 1

    # Ground-truth class set for this image
    gt_cls_set = {cls_id for cls_id, _ in polys}
    for c in gt_cls_set:
        gt_counts[c] += 1

    # Run model on masked image (in-memory array)
    result = model.predict(source=masked, imgsz=IMG_SZ, conf=CONF_THR,
                           verbose=False)[0]
    if len(result.obb) == 0:
        continue

    detected_any += 1
    pred_cls_set = set(result.obb.cls.int().tolist())

    for c in gt_cls_set:
        if c in pred_cls_set:
            hit_counts[c] += 1

print(f"\nProcessed {total_imgs} test images with annotations.")
print(f"Model detected something on {detected_any} masked images "
      f"({100*detected_any/max(total_imgs,1):.1f}%).\n")

# %% [markdown]
# ## Report per-class hit rates vs. chance

# %%
print(f"{'Class':<20}  {'GT imgs':>7}  {'Hits':>5}  {'Hit rate':>9}  {'Chance':>7}  {'Verdict'}")
print("-" * 70)

flagged = []
for i, cls_name in enumerate(CLASSES):
    gt  = gt_counts.get(i, 0)
    hit = hit_counts.get(i, 0)
    if gt == 0:
        continue
    rate = hit / gt
    flag = "❌ shortcut" if rate > CHANCE * 2 else ("✅ ok" if rate <= CHANCE * 1.5 else "⚠️  watch")
    print(f"{cls_name:<20}  {gt:>7}  {hit:>5}  {rate:>8.1%}  {CHANCE:>6.1%}  {flag}")
    if "❌" in flag:
        flagged.append(cls_name)

print(f"\nChance level: {CHANCE:.1%}  (1 / {len(CLASSES)} classes)")
if flagged:
    print(f"\n❌ Background shortcut confirmed for: {', '.join(flagged)}")
    print("   Disease labels are predictable from background alone.")
else:
    print("\n✅ No class predicted above 2× chance on masked images. Background is clean.")

# %% [markdown]
# ## Save outputs

# %%
os.makedirs(OUTPUT_DIR, exist_ok=True)

save_rows = []
for i, cls_name in enumerate(CLASSES):
    gt  = gt_counts.get(i, 0)
    hit = hit_counts.get(i, 0)
    if gt == 0:
        continue
    rate = hit / gt
    save_rows.append({
        "class":          cls_name,
        "gt_images":      gt,
        "hits":           hit,
        "hit_rate":       round(rate, 4),
        "chance":         round(CHANCE, 4),
        "ratio_to_chance": round(rate / CHANCE, 4),
        "verdict":        ("shortcut" if rate > CHANCE * 2
                           else "ok" if rate <= CHANCE * 1.5
                           else "watch"),
    })

c2_df = pd.DataFrame(save_rows).set_index("class")
c2_df.to_csv(os.path.join(OUTPUT_DIR, "C2_hit_rates.csv"))

verdict_line = (
    f"❌ Background shortcut confirmed for: {', '.join(flagged)}"
    if flagged else
    "✅ No class predicted above 2× chance on masked images. Background is clean."
)
with open(os.path.join(OUTPUT_DIR, "C2_report.txt"), "w") as _f:
    _f.write(f"Experiment: {EXPERIMENT}\n")
    _f.write(f"Weights:   {WEIGHTS}\n\n")
    _f.write(f"Processed {total_imgs} test images with annotations.\n")
    _f.write(f"Detections on masked images: {detected_any} "
             f"({100*detected_any/max(total_imgs,1):.1f}%)\n\n")
    _f.write(c2_df.to_string() + "\n\n")
    _f.write(f"Chance level: {CHANCE:.1%}  (1 / {len(CLASSES)} classes)\n")
    _f.write(verdict_line + "\n")

print(f"\nSaved → {OUTPUT_DIR}/C2_hit_rates.csv, C2_report.txt")

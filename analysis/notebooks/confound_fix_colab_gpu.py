# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.4
#   kernelspec:
#     display_name: Python 3
#     name: python3
# ---

# %% [markdown]
# # Confound-fix pipeline — Colab GPU steps
#
# Colab-runnable variant of the steps in the confound-fix pipeline that need real
# GPU compute: **Phase 6 retrain** and the model-dependent half of **Phase 7**
# (Tests C1, C2). Everything else (Phases 1-5, dataset-level Tests A/B1/D) already
# ran locally/CPU-only and is not repeated here.
#
# This notebook is a standalone copy — it does **not** modify the local scripts
# it mirrors:
# - `scripts/base/train.py` / `scripts/experiments/v4_confound_fix/config.yaml` → Phase 6
# - `analysis/evaluation/source_confound_C1.py` → Test C1
# - `analysis/evaluation/source_confound_C2.py` → Test C2
#
# See `results/source_confound_fix_pipeline.md` for the full local-vs-Colab file map
# and `results/source_confound_fix_phase7.md` for the before/after numbers this
# notebook is meant to complete.
#
# Drive layout follows `results/ml_project_structure.md`:
# ```
# DurianProj/                          (PROJECT_ROOT on Drive)
# ├── data/processed/                  datasetv3_2_resplit.zip
# ├── models/
# │   ├── final/                       best.pt, timestamped
# │   └── configs/                     the effective train config, yaml
# └── outputs/
#     ├── logs/                        full runs/v4_confound_fix/train dir
#     └── predictions/                 C1/C2 csvs + reports
# ```
# Raw/processed data and heavy training I/O still happen on local Colab disk
# (`/content/`) per that doc's note that Drive I/O is a bottleneck for training —
# Drive is only touched for upload-once / persist-results.
#
# **Before you start:** zip the confound-fixed dataset locally and have it ready to
# upload (or place directly at `data/processed/` in your Drive project folder):
# ```bash
# cd /Volumes/TheDisk/DurianProj/data
# zip -r datasetv3_2_resplit.zip datasetv3_2_resplit
# ```

# %% [markdown]
# ## 0. Setup

# %%
# !pip -q install ultralytics pyyaml pandas numpy opencv-python-headless

# %%
import torch
print('CUDA available:', torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else
      'No GPU attached — Runtime > Change runtime type > GPU')

# %%
from google.colab import drive
drive.mount('/content/drive')

# --- Edit this to match your Drive folder name ---
PROJECT_ROOT = '/content/drive/MyDrive/DurianProj'

import os
from pathlib import Path
for sub in ('data/raw', 'data/processed', 'data/external',
            'models/checkpoints', 'models/final', 'models/configs',
            'outputs/logs', 'outputs/figures', 'outputs/predictions'):
    Path(PROJECT_ROOT, sub).mkdir(parents=True, exist_ok=True)
print('Drive project structure ready at', PROJECT_ROOT)

# %% [markdown]
# ## 1. Get the confound-fixed dataset onto Colab
#
# Per `ml_project_structure.md`: raw/processed data is kept zipped in Drive
# (`data/processed/`) and unzipped to **local Colab disk** (`/content/`) at
# runtime — training directly off Drive-mounted files bottlenecks on Drive I/O.
#
# If `data/processed/datasetv3_2_resplit.zip` isn't already in your Drive project
# folder, the cell below prompts you to upload it and saves a copy there for
# next time.

# %%
DATA_ROOT        = '/content'                                    # local Colab disk — fast I/O for training
DRIVE_DATA_ZIP    = f'{PROJECT_ROOT}/data/processed/datasetv3_2_resplit.zip'
DATASET_ZIP       = f'{DATA_ROOT}/datasetv3_2_resplit.zip'        # local working copy
DATASET_DIR       = f'{DATA_ROOT}/datasetv3_2_resplit'
DATA_YAML         = f'{DATA_ROOT}/dataset_resplit.yaml'

import shutil

if Path(DRIVE_DATA_ZIP).exists():
    print(f'Found dataset in Drive project folder — copying to local disk …')
    shutil.copy(DRIVE_DATA_ZIP, DATASET_ZIP)
elif not Path(DATASET_ZIP).exists():
    print(f'{DRIVE_DATA_ZIP} not found — upload datasetv3_2_resplit.zip now.')
    from google.colab import files
    uploaded = files.upload()
    for fname in uploaded:
        if fname.endswith('.zip') and not Path(DATASET_ZIP).exists():
            os.rename(fname, DATASET_ZIP)
    # stash a copy in Drive's data/processed/ so future runs skip the upload
    shutil.copy(DATASET_ZIP, DRIVE_DATA_ZIP)
    print('Saved a copy to', DRIVE_DATA_ZIP, 'for next time.')

# %%
import zipfile

if not Path(DATASET_DIR).exists():
    print(f'Extracting {DATASET_ZIP} to local disk ...')
    with zipfile.ZipFile(DATASET_ZIP, 'r') as zf:
        zf.extractall(DATA_ROOT)
    print('Extraction complete.')
else:
    print(f'Dataset already extracted at {DATASET_DIR}')

for split in ('train', 'val', 'test'):
    n_img = len(list((Path(DATASET_DIR) / split / 'images').glob('*')))
    n_lbl = len(list((Path(DATASET_DIR) / split / 'labels').glob('*.txt')))
    print(f'{split:>5}: {n_img} images, {n_lbl} labels')

# %% [markdown]
# The repo's `data/dataset_resplit.yaml` has a Mac-local absolute `path:` — rebuild it
# here pointed at the local Colab path instead of uploading and editing it by hand.

# %%
import yaml

CLASSES = ['Algal spot', 'Anthracnose', 'Healthy', 'Other', 'Phomopsis', 'Phytophtora', 'Rhizoctonia']

with open(DATA_YAML, 'w') as f:
    yaml.dump({
        'path':  DATASET_DIR,
        'train': 'train/images',
        'val':   'val/images',
        'test':  'test/images',
        'nc':    len(CLASSES),
        'names': CLASSES,
    }, f)

print(open(DATA_YAML).read())

# %% [markdown]
# ## 2. Phase 6 — Retrain on the confound-fixed split
#
# Mirrors `scripts/experiments/v4_confound_fix/config.yaml` exactly (same model,
# epochs, imgsz, batch, patience). `'yolo11m-obb.pt'` here is the bare model name —
# Ultralytics auto-downloads the DOTA-pretrained checkpoint fresh; this is **not**
# the repo's `models/yolo11m-obb.pt`, which is the old confound-affected durian model
# and is intentionally not reused as a starting point.
#
# Training runs on local disk (`runs/`), same reasoning as the dataset — Drive I/O
# would slow down every epoch's checkpoint write.

# %%
from ultralytics import YOLO

TRAIN_CONFIG = dict(
    data     = DATA_YAML,
    epochs   = 500,
    imgsz    = 640,
    batch    = 16,
    patience = 50,
    project  = 'runs/v4_confound_fix',
    name     = 'train',
    exist_ok = True,
)

model = YOLO('yolo11m-obb.pt')
train_results = model.train(**TRAIN_CONFIG)

BEST_WEIGHTS = 'runs/v4_confound_fix/train/weights/best.pt'
print('Best weights:', BEST_WEIGHTS, '| exists:', Path(BEST_WEIGHTS).exists())

# %% [markdown]
# **Checklist (from `scripts/base/train.py`):**
# - Deploy `best.pt`, not `last.pt` — already the default above.
# - Re-check inference conf around ~0.53 after this run; the prior F1 peak was measured
#   on the old split/model and may shift.
# - Don't trust these weights until Tests C1/C2 below are re-run and reviewed.

# %%
# Persist to Drive immediately — Colab runtimes recycle and /content is not persistent.
# Layout matches results/ml_project_structure.md: models/final + models/configs + outputs/logs.
import datetime

timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M')
run_tag   = f'v4_confound_fix_{timestamp}'

final_weights_dst = f'{PROJECT_ROOT}/models/final/{run_tag}_best.pt'
shutil.copy(BEST_WEIGHTS, final_weights_dst)
print('Weights ->', final_weights_dst)

config_dst = f'{PROJECT_ROOT}/models/configs/{run_tag}.yaml'
with open(config_dst, 'w') as f:
    yaml.dump(TRAIN_CONFIG, f)
print('Config  ->', config_dst)

logs_dst = f'{PROJECT_ROOT}/outputs/logs/{run_tag}'
shutil.copytree('runs/v4_confound_fix', logs_dst, dirs_exist_ok=True)
print('Logs    ->', logs_dst)

# %% [markdown]
# ## 3. Phase 7 — Test C1: per-source AP breakdown
#
# Direct port of `analysis/evaluation/source_confound_C1.py`, pointed at the weights
# just trained above instead of the old `models/yolo11m-obb.pt`. Target: the
# curated-vs-field mAP@50 gap (0.4015 on Healthy at baseline) should shrink.

# %%
import glob, tempfile
import pandas as pd

WEIGHTS   = BEST_WEIGHTS
IMG_SZ    = 640
TEST_IMG  = os.path.join(DATASET_DIR, 'test', 'images')
TEST_LBL  = os.path.join(DATASET_DIR, 'test', 'labels')
OUTPUT_DIR = f'{DATA_ROOT}/outputs/predictions/source_confound_after_fix'   # local scratch, synced to Drive in step 5
os.makedirs(OUTPUT_DIR, exist_ok=True)

def source_of(filepath):
    name = os.path.basename(filepath)
    name = name.split('-', 1)[-1] if '-' in name else name
    return 'curated' if name.startswith('Leaf_') else 'field'

all_test = glob.glob(os.path.join(TEST_IMG, '*'))
by_source = {'curated': [], 'field': []}
for p in all_test:
    by_source[source_of(p)].append(p)
for src, imgs in by_source.items():
    print(f'  {src}: {len(imgs)} test images')

c1_model = YOLO(WEIGHTS)
c1_results = {}

for src, img_paths in by_source.items():
    if not img_paths:
        print(f'Skipping {src} — no images.')
        continue
    tmpdir = tempfile.mkdtemp(prefix=f'confound_{src}_')
    tmp_img = os.path.join(tmpdir, 'images', 'test')
    tmp_lbl = os.path.join(tmpdir, 'labels', 'test')
    os.makedirs(tmp_img, exist_ok=True)
    os.makedirs(tmp_lbl, exist_ok=True)
    for p in img_paths:
        shutil.copy(p, tmp_img)
        stem = os.path.splitext(os.path.basename(p))[0]
        label = os.path.join(TEST_LBL, stem + '.txt')
        if os.path.exists(label):
            shutil.copy(label, tmp_lbl)
    tmp_yaml = os.path.join(tmpdir, 'data.yaml')
    with open(tmp_yaml, 'w') as f:
        yaml.dump({'path': tmpdir, 'train': 'images/test', 'val': 'images/test',
                   'test': 'images/test', 'nc': len(CLASSES), 'names': CLASSES}, f)
    print(f'\n--- Evaluating on {src} test images ---')
    try:
        metrics = c1_model.val(data=tmp_yaml, split='test', imgsz=IMG_SZ,
                                task='obb', verbose=False, plots=False)
        c1_results[src] = metrics
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

# %%
overall_rows = []
for src, m in c1_results.items():
    overall_rows.append({'source': src, 'mAP@50': round(float(m.box.map50), 4),
                          'mAP@50-95': round(float(m.box.map), 4)})
overall_df = pd.DataFrame(overall_rows).set_index('source')
print(overall_df.to_string())

if 'curated' in c1_results and 'field' in c1_results:
    gap = abs(c1_results['curated'].box.map50 - c1_results['field'].box.map50)
    print(f'\nmAP@50 gap (curated vs field): {gap:.4f}  (baseline was 0.4015 on Healthy)')

per_class = {}
for src, m in c1_results.items():
    for idx, cls_idx in enumerate(m.box.ap_class_index):
        cls_name = CLASSES[int(cls_idx)]
        per_class.setdefault(cls_name, {})[src] = round(float(m.box.ap50[idx]), 4)
pc_df = pd.DataFrame(per_class).T.sort_index()
pc_df['gap'] = (pc_df.get('curated', float('nan')) - pc_df.get('field', float('nan'))).round(4)
print('\nPer-class AP@50 by source (watch Rhizoctonia & Healthy):')
print(pc_df.to_string())

overall_df.to_csv(os.path.join(OUTPUT_DIR, 'C1_overall_map.csv'))
pc_df.to_csv(os.path.join(OUTPUT_DIR, 'C1_per_class_ap.csv'))
print(f'\nSaved -> {OUTPUT_DIR}/C1_overall_map.csv, C1_per_class_ap.csv')

# %% [markdown]
# ## 4. Phase 7 — Test C2: background-only disease probe
#
# Direct port of `analysis/evaluation/source_confound_C2.py`. Target: Algal spot
# and Other hit-rate ratios should fall from 3.1x / 2.9x chance toward ~1.0x now
# that Phase 4's background negatives, background-swap augmentation, and
# hard-negative mining are baked into the training set.

# %%
import cv2, numpy as np
from collections import defaultdict

CONF_THR = 0.25
c2_model = YOLO(WEIGHTS)

def load_obb_polys(label_path, img_w, img_h):
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
    out = img_bgr.copy()
    for _, poly in polys:
        cv2.fillPoly(out, [poly], (127, 127, 127))
    return out

CHANCE = 1.0 / len(CLASSES)
gt_counts, hit_counts = defaultdict(int), defaultdict(int)
total_imgs, detected_any = 0, 0

for img_path in sorted(glob.glob(os.path.join(TEST_IMG, '*'))):
    stem = os.path.splitext(os.path.basename(img_path))[0]
    lbl_path = os.path.join(TEST_LBL, stem + '.txt')
    img = cv2.imread(img_path)
    if img is None:
        continue
    h, w = img.shape[:2]
    polys = load_obb_polys(lbl_path, w, h)
    if not polys:
        continue
    masked = mask_leaves(img, polys)
    total_imgs += 1
    gt_cls_set = {cls_id for cls_id, _ in polys}
    for c in gt_cls_set:
        gt_counts[c] += 1
    result = c2_model.predict(source=masked, imgsz=IMG_SZ, conf=CONF_THR, verbose=False)[0]
    if len(result.obb) == 0:
        continue
    detected_any += 1
    pred_cls_set = set(result.obb.cls.int().tolist())
    for c in gt_cls_set:
        if c in pred_cls_set:
            hit_counts[c] += 1

print(f'Processed {total_imgs} test images with annotations.')
print(f'Model detected something on {detected_any} masked images '
      f'({100*detected_any/max(total_imgs,1):.1f}%).\n')

# %%
save_rows = []
flagged = []
print(f"{'Class':<20}  {'GT imgs':>7}  {'Hits':>5}  {'Hit rate':>9}  {'Chance':>7}  {'Verdict'}")
print('-' * 70)
for i, cls_name in enumerate(CLASSES):
    gt = gt_counts.get(i, 0)
    hit = hit_counts.get(i, 0)
    if gt == 0:
        continue
    rate = hit / gt
    flag = ('shortcut' if rate > CHANCE * 2 else 'ok' if rate <= CHANCE * 1.5 else 'watch')
    print(f'{cls_name:<20}  {gt:>7}  {hit:>5}  {rate:>8.1%}  {CHANCE:>6.1%}  {flag}')
    if flag == 'shortcut':
        flagged.append(cls_name)
    save_rows.append({'class': cls_name, 'gt_images': gt, 'hits': hit,
                       'hit_rate': round(rate, 4), 'chance': round(CHANCE, 4),
                       'ratio_to_chance': round(rate / CHANCE, 4), 'verdict': flag})

print(f'\nChance level: {CHANCE:.1%}  (1 / {len(CLASSES)} classes)')
print(f'Background shortcut confirmed for: {", ".join(flagged)}' if flagged else
      'No class predicted above 2x chance on masked images — background is clean.')

c2_df = pd.DataFrame(save_rows).set_index('class')
c2_df.to_csv(os.path.join(OUTPUT_DIR, 'C2_hit_rates.csv'))
print(f'\nSaved -> {OUTPUT_DIR}/C2_hit_rates.csv')

# %% [markdown]
# ## 5. Pull everything back
#
# Weights/config/logs were already persisted to Drive in step 2 right after
# training. This last step syncs the C1/C2 outputs the same way, into
# `outputs/predictions/` per `ml_project_structure.md`.
#
# After downloading from Drive, drop the results into the repo at:
# - `models/final/<run_tag>_best.pt` → `models/yolo11m-obb.pt` (replacing the old
#   confound-affected one, once you're satisfied with the numbers)
# - `outputs/predictions/source_confound_after_fix/*.csv` →
#   `analysis/notebooks/outputs/source_confound_after_fix/`
#
# Then fill in `results/source_confound_fix_phase7.md`'s pending rows and
# `results/source_confound_fix_phase8_template.md`.

# %%
drive_predictions_dst = f'{PROJECT_ROOT}/outputs/predictions/source_confound_after_fix'
shutil.copytree(OUTPUT_DIR, drive_predictions_dst, dirs_exist_ok=True)
print('Copied C1/C2 outputs to', drive_predictions_dst)
print()
print('Already on Drive from step 2:')
print(' ', final_weights_dst)
print(' ', config_dst)
print(' ', logs_dst)

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
#     display_name: .venv (3.14.6)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Source-Confound Check
#
# Tests whether the model separates disease classes via **photographic source/style** (curated single-leaf vs. field photos) rather than the disease signal.
#
# **Run order (fastest path to an answer):**
# 1. Test A — contingency table (10 min) → stop if sources are mixed
# 2. Test B1 — metadata fingerprint (5 min)
# 3. Test C1 — per-source AP breakdown on existing model
# 4. Test C2 — background-only disease probe
# 5. Test D  — split-leakage audit

# %%
import os, glob, sys
# Run from repo root
if os.path.basename(os.getcwd()) == 'notebooks':
    os.chdir('../..')
print('Working dir:', os.getcwd())

# %%
import yaml, pandas as pd, numpy as np

DATA_YAML = 'data/dataset.yaml'
with open(DATA_YAML) as f:
    cfg = yaml.safe_load(f)

CLASSES   = cfg['names']
DATA_ROOT = cfg['path']
SPLITS    = ['train', 'val', 'test']

def source_of(filepath):
    name = os.path.basename(filepath)
    name = name.split('-', 1)[-1] if '-' in name else name
    return 'curated' if name.startswith('Leaf_') else 'field'

print('Classes:', CLASSES)

OUTPUT_DIR = 'analysis/notebooks/outputs/source_confound'
os.makedirs(OUTPUT_DIR, exist_ok=True)


# %% [markdown]
# ## Test A — Class × Source contingency table

# %%
rows = []
for split in SPLITS:
    ldir = os.path.join(DATA_ROOT, split, 'labels')
    for lbl in glob.glob(os.path.join(ldir, '*.txt')):
        src = source_of(lbl)
        with open(lbl) as f:
            for line in f:
                line = line.strip()
                if not line: continue
                cls_id = int(line.split()[0])
                rows.append({'class': CLASSES[cls_id], 'source': src, 'split': split})

df = pd.DataFrame(rows)
ct = pd.crosstab(df['class'], df['source'], margins=True)
display(ct)

print('\nRow-normalised (share of each class by source):')
display(pd.crosstab(df['class'], df['source'], normalize='index').round(3))

# %%
# Cramér's V
ct_raw   = pd.crosstab(df['class'], df['source']).values
n        = ct_raw.sum()
expected = np.outer(ct_raw.sum(axis=1), ct_raw.sum(axis=0)) / n
chi2     = ((ct_raw - expected) ** 2 / (expected + 1e-9)).sum()
k        = min(ct_raw.shape) - 1
cramers_v = np.sqrt(chi2 / (n * k)) if k > 0 else 0.0

print(f"Cramér's V (class ↔ source): {cramers_v:.3f}")
print('0 = independent | 1 = fully confounded')

if cramers_v < 0.15:
    print('\n✅ Sources well-mixed. No confound risk — can stop here.')
elif cramers_v < 0.40:
    print('\n⚠️  Moderate correlation. Proceed to Test B.')
else:
    print('\n❌ Strong correlation. Proceed to Tests B and C.')
# Save
_verdict = ('✅ Well-mixed.' if cramers_v < 0.15 else
            '⚠️  Moderate correlation.' if cramers_v < 0.40 else
            '❌ Strong correlation — confound likely.')
ct.to_csv(os.path.join(OUTPUT_DIR, 'A_contingency.csv'))
pd.crosstab(df['class'], df['source'], normalize='index').round(3).to_csv(
    os.path.join(OUTPUT_DIR, 'A_normalized.csv'))
with open(os.path.join(OUTPUT_DIR, 'A_report.txt'), 'w') as _f:
    _f.write(f"Cramér's V: {cramers_v:.3f}\nVerdict: {_verdict}\n\n")
    _f.write("Contingency:\n" + ct.to_string() + "\n")
print(f"Saved A results → {OUTPUT_DIR}/")


# %%
import matplotlib.pyplot as plt

ct_plot = pd.crosstab(df['class'], df['source'])
ct_plot.plot(kind='bar', stacked=True, figsize=(10, 5), colormap='Set2')
plt.title('Annotation counts by class and source')
plt.ylabel('Annotations')
plt.xticks(rotation=30, ha='right')
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Test B1 — Metadata / resolution fingerprint

# %%
from PIL import Image

recs = []
for split in SPLITS:
    for p in glob.glob(os.path.join(DATA_ROOT, split, 'images', '*')):
        try:
            im = Image.open(p)
            w, h   = im.size
            camera = im.getexif().get(272, 'unknown')
        except Exception:
            continue
        recs.append({'src': source_of(p), 'w': w, 'h': h,
                     'aspect': round(w/h, 3), 'camera': camera})

meta = pd.DataFrame(recs)
display(meta.groupby('src')[['w','h','aspect']].describe())
print('\nCamera model by source:')
display(meta.groupby('src')['camera'].value_counts())
# Save
meta.groupby('src')[['w','h','aspect']].describe().to_csv(
    os.path.join(OUTPUT_DIR, 'B1_resolution_stats.csv'))
meta.groupby('src')['camera'].value_counts().rename('count').to_csv(
    os.path.join(OUTPUT_DIR, 'B1_camera.csv'))
print(f"Saved B1 results → {OUTPUT_DIR}/")


# %% [markdown]
# ## Test C1 — Per-source AP breakdown

# %%
import shutil, tempfile
from ultralytics import YOLO

WEIGHTS  = 'models/yolo11m-obb.pt'
IMG_SZ   = 640
TEST_IMG = os.path.join(DATA_ROOT, 'test', 'images')
TEST_LBL = os.path.join(DATA_ROOT, 'test', 'labels')

all_test  = glob.glob(os.path.join(TEST_IMG, '*'))
by_source = {'curated': [p for p in all_test if source_of(p) == 'curated'],
             'field':   [p for p in all_test if source_of(p) == 'field']}
for src, imgs in by_source.items():
    print(f'{src}: {len(imgs)} test images')

model = YOLO(WEIGHTS)
results_c1 = {}

# %%
for src, img_paths in by_source.items():
    if not img_paths:
        continue
    tmpdir  = tempfile.mkdtemp(prefix=f'confound_{src}_')
    tmp_img = os.path.join(tmpdir, 'images', 'test')
    tmp_lbl = os.path.join(tmpdir, 'labels', 'test')
    os.makedirs(tmp_img); os.makedirs(tmp_lbl)

    for p in img_paths:
        shutil.copy(p, tmp_img)
        stem  = os.path.splitext(os.path.basename(p))[0]
        lbl   = os.path.join(TEST_LBL, stem + '.txt')
        if os.path.exists(lbl):
            shutil.copy(lbl, tmp_lbl)

    tmp_yaml = os.path.join(tmpdir, 'data.yaml')
    with open(tmp_yaml, 'w') as f:
        yaml.dump({'path': tmpdir, 'train': 'images/test',
                   'val': 'images/test', 'test': 'images/test',
                   'nc': cfg['nc'], 'names': cfg['names']}, f)

    print(f'\nEvaluating {src} …')
    try:
        m = model.val(data=tmp_yaml, split='test', imgsz=IMG_SZ,
                      task='obb', verbose=False, plots=False)
        results_c1[src] = m
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

# %%
rows = []
for src, m in results_c1.items():
    rows.append({'source': src, 'mAP@50': round(float(m.box.map50), 4),
                 'mAP@50-95': round(float(m.box.map), 4)})
summary = pd.DataFrame(rows).set_index('source')
display(summary)

if 'curated' in results_c1 and 'field' in results_c1:
    gap = abs(results_c1['curated'].box.map50 - results_c1['field'].box.map50)
    print(f'\nmAP@50 gap: {gap:.4f}')
    print('❌ Large gap — confound likely.' if gap > 0.10 else
          '⚠️  Moderate gap.' if gap > 0.05 else
          '✅ Small gap — generalises across sources.')

# Per-class AP@50 — the most diagnostic output (watch Rhizoctonia & Healthy)
per_class = {}
for src, m in results_c1.items():
    for idx, cls_idx in enumerate(m.box.ap_class_index):
        cls_name = CLASSES[int(cls_idx)]
        per_class.setdefault(cls_name, {})[src] = round(float(m.box.ap50[idx]), 4)

if per_class:
    pc_df = pd.DataFrame(per_class).T.sort_index()
    pc_df['gap'] = (pc_df.get('curated', float('nan')) - pc_df.get('field', float('nan'))).round(4)
    print('\nPer-class AP@50 by source:')
    display(pc_df)
    print('Large positive gap → curated AP ≫ field AP → shortcut on that class.')

# Save
summary.to_csv(os.path.join(OUTPUT_DIR, 'C1_overall_map.csv'))
if per_class:
    pc_df.to_csv(os.path.join(OUTPUT_DIR, 'C1_per_class_ap.csv'))
    with open(os.path.join(OUTPUT_DIR, 'C1_report.txt'), 'w') as _f:
        _f.write("Overall mAP by source:\n" + summary.to_string() + "\n\n")
        _f.write("Per-class AP@50 by source:\n" + pc_df.to_string() + "\n")
print(f"Saved C1 results → {OUTPUT_DIR}/")


# %% [markdown]
# ## Test C2 — Background-only disease probe

# %%
import cv2
from collections import defaultdict

CONF_THR = 0.25

def load_obb_polys(label_path, img_w, img_h):
    polys = []
    if not os.path.exists(label_path):
        return polys
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 9: continue
            cls_id = int(parts[0])
            coords = np.array(parts[1:9], dtype=float).reshape(4, 2)
            coords[:, 0] *= img_w
            coords[:, 1] *= img_h
            polys.append((cls_id, coords.astype(np.int32)))
    return polys

gt_counts  = defaultdict(int)
hit_counts = defaultdict(int)
total_imgs = 0

for img_path in sorted(glob.glob(os.path.join(TEST_IMG, '*'))):
    stem     = os.path.splitext(os.path.basename(img_path))[0]
    lbl_path = os.path.join(TEST_LBL, stem + '.txt')
    img      = cv2.imread(img_path)
    if img is None: continue
    h, w = img.shape[:2]
    polys = load_obb_polys(lbl_path, w, h)
    if not polys: continue

    masked = img.copy()
    for _, poly in polys:
        cv2.fillPoly(masked, [poly], (127, 127, 127))
    total_imgs += 1

    gt_cls_set = {cls_id for cls_id, _ in polys}
    for cls_id in gt_cls_set:
        gt_counts[cls_id] += 1

    res = model.predict(source=masked, imgsz=IMG_SZ, conf=CONF_THR, verbose=False)[0]
    if len(res.obb) == 0: continue
    pred_cls_set = set(res.obb.cls.int().tolist())
    for cls_id in gt_cls_set:
        if cls_id in pred_cls_set:
            hit_counts[cls_id] += 1

print(f'Processed {total_imgs} masked test images.')

# %%
CHANCE = 1 / len(CLASSES)
rows   = []
for i, name in enumerate(CLASSES):
    gt  = gt_counts.get(i, 0)
    hit = hit_counts.get(i, 0)
    if gt == 0: continue
    rate = hit / gt
    rows.append({'class': name, 'gt_images': gt, 'hits': hit,
                 'hit_rate': round(rate, 3), 'chance': round(CHANCE, 3),
                 'ratio_to_chance': round(rate / CHANCE, 2)})

c2_df = pd.DataFrame(rows).set_index('class')
display(c2_df)

flagged = c2_df[c2_df['ratio_to_chance'] > 2].index.tolist()
if flagged:
    print(f'\n❌ Background shortcut confirmed for: {flagged}')
else:
    print(f'\n✅ No class predicted at >2× chance on masked images. Background is clean.')
# Save
c2_df.to_csv(os.path.join(OUTPUT_DIR, 'C2_hit_rates.csv'))
with open(os.path.join(OUTPUT_DIR, 'C2_report.txt'), 'w') as _f:
    _f.write(c2_df.to_string() + "\n\n")
    _f.write(("❌ Background shortcut: " + str(flagged) if flagged
              else "✅ No background shortcut.") + "\n")
print(f"Saved C2 results → {OUTPUT_DIR}/")


# %% [markdown]
# ## Test D — Split-leakage audit

# %%
import imagehash
from PIL import Image as PILImage

HASH_THRESH = 5

def split_of(path):
    for s in SPLITS:
        if f'/{s}/' in path: return s
    return '?'

print('Computing perceptual hashes …')
paths  = []
for split in SPLITS:
    paths += glob.glob(os.path.join(DATA_ROOT, split, 'images', '*'))

hashes = {}
for p in paths:
    try: hashes[p] = imagehash.phash(PILImage.open(p))
    except Exception: pass

print(f'Hashed {len(hashes)} images. Checking pairs …')
items = list(hashes.items())
leaks = []
for i in range(len(items)):
    pa, ha = items[i]
    for j in range(i+1, len(items)):
        pb, hb = items[j]
        if split_of(pa) != split_of(pb) and (ha - hb) <= HASH_THRESH:
            leaks.append((pa, pb, int(ha - hb)))

print(f'\nCross-split near-duplicate pairs: {len(leaks)}')
if leaks:
    leak_df = pd.DataFrame(
        [{'dist': d, 'split_a': split_of(a), 'src_a': source_of(a), 'file_a': os.path.basename(a),
          'split_b': split_of(b), 'src_b': source_of(b), 'file_b': os.path.basename(b)}
         for a, b, d in sorted(leaks, key=lambda x: x[2])[:30]]
    )
    display(leak_df)
    print('\n❌ Leakage detected — re-split so duplicates stay in one split.')
else:
    print('✅ No cross-split near-duplicates found.')
# Save
_leak_rows = [{'dist': d, 'split_a': split_of(a), 'src_a': source_of(a),
               'file_a': os.path.basename(a), 'split_b': split_of(b),
               'src_b': source_of(b), 'file_b': os.path.basename(b)}
              for a, b, d in sorted(leaks, key=lambda x: x[2])]
_leak_df = pd.DataFrame(_leak_rows) if _leak_rows else pd.DataFrame(
    columns=['dist','split_a','src_a','file_a','split_b','src_b','file_b'])
_leak_df.to_csv(os.path.join(OUTPUT_DIR, 'D_leak_pairs.csv'), index=False)
print(f"Saved D results → {OUTPUT_DIR}/")


# %% [markdown]
# ## Summary
#
# | Test | Result | Verdict |
# |---|---|---|
# | A — Class×Source correlation | (fill in Cramér's V) | |
# | B1 — Resolution/metadata cue | (fill in) | |
# | C1 — Per-source AP gap | (fill in mAP gap) | |
# | C2 — Background-only probe | (fill in flagged classes) | |
# | D — Split leakage | (fill in pair count) | |

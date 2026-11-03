"""
Phase 4.2 — Background-randomization augmentation for Algal spot / Other.

Test C2 flagged these two classes as background-predictable (3.1x / 2.9x
chance on masked images) — the model can apparently key off "field-photo
background" rather than the lesion itself. This uses the existing OBB
polygons to cut the annotated leaf out of the image and recomposite it onto
a different background (drawn from Phase 4.1's background-negative pool),
so the same lesion appears against varied backgrounds and the background
itself stops being predictive.

Adds new augmented copies alongside the originals in train (doesn't
replace them) — labels are copied unchanged since only pixels outside the
leaf polygons change.

Run from project root (after phase4_background_negatives.py):
    python analysis/dataset/phase4_background_swap_augment.py
"""

import os, glob, random, uuid, shutil

import cv2
import numpy as np
import yaml

DATA_YAML  = "data/dataset_resplit.yaml"
OUTPUT_DIR = "analysis/dataset/outputs/resplit"

random.seed(42)
np.random.seed(42)

TARGET_CLASSES = {"Algal spot", "Other"}
SWAP_FRACTION = 0.4   # 30-50% per the plan; using the midpoint

with open(DATA_YAML) as f:
    cfg = yaml.safe_load(f)

CLASSES   = cfg["names"]
DATA_ROOT = cfg["path"]
TRAIN_IMG = os.path.join(DATA_ROOT, "train", "images")
TRAIN_LBL = os.path.join(DATA_ROOT, "train", "labels")
TARGET_IDS = {CLASSES.index(c) for c in TARGET_CLASSES}


def strip_uuid(fname):
    stem = os.path.splitext(fname)[0]
    parts = stem.split("-", 1)
    return parts[1] if len(parts) == 2 and len(parts[0]) == 8 else stem


def source_of(filepath):
    name = strip_uuid(os.path.basename(filepath))
    return "curated" if name.startswith("Leaf_") else "field"


def load_polys(lbl_path, w, h):
    polys = []
    if not os.path.exists(lbl_path):
        return polys
    with open(lbl_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 9:
                continue
            cls_id = int(parts[0])
            coords = np.array(parts[1:9], dtype=float).reshape(4, 2)
            coords[:, 0] *= w
            coords[:, 1] *= h
            polys.append((cls_id, coords.astype(np.int32)))
    return polys


def extract_leaf_mask(img_shape, obb_polys):
    mask = np.zeros(img_shape[:2], dtype=np.uint8)
    for _, poly in obb_polys:
        cv2.fillPoly(mask, [poly], 255)
    return mask


def swap_background(img, mask, bg):
    bg = cv2.resize(bg, (img.shape[1], img.shape[0]))
    mask_3ch = cv2.merge([mask, mask, mask]) / 255.0
    return (img * mask_3ch + bg * (1 - mask_3ch)).astype(np.uint8)


# ── Background pool: Phase 4.1's negatives (same field conditions) ─────
bg_pool_paths = glob.glob(os.path.join(TRAIN_IMG, "bgneg_*"))
if not bg_pool_paths:
    raise SystemExit("No bgneg_* images found — run phase4_background_negatives.py first.")
bg_pool = [cv2.imread(p) for p in bg_pool_paths]
bg_pool = [b for b in bg_pool if b is not None]
print(f"Background pool: {len(bg_pool)} images")

# ── Find candidate field images containing a target class ──────────────
candidates = []
for img_path in glob.glob(os.path.join(TRAIN_IMG, "*")):
    if os.path.basename(img_path).startswith("bgneg_"):
        continue
    if source_of(img_path) != "field":
        continue
    stem = os.path.splitext(os.path.basename(img_path))[0]
    lbl_path = os.path.join(TRAIN_LBL, stem + ".txt")
    if not os.path.exists(lbl_path):
        continue
    with open(lbl_path) as f:
        classes = {int(line.split()[0]) for line in f if line.strip()}
    if classes & TARGET_IDS:
        candidates.append(img_path)

print(f"Candidates containing {TARGET_CLASSES}: {len(candidates)}")

n_swap = int(len(candidates) * SWAP_FRACTION)
selected = random.sample(candidates, n_swap)
print(f"Augmenting {n_swap} ({SWAP_FRACTION:.0%})")

saved = 0
for img_path in selected:
    img = cv2.imread(img_path)
    if img is None:
        continue
    h, w = img.shape[:2]
    stem = os.path.splitext(os.path.basename(img_path))[0]
    lbl_path = os.path.join(TRAIN_LBL, stem + ".txt")
    polys = load_polys(lbl_path, w, h)
    if not polys:
        continue

    mask = extract_leaf_mask(img.shape, polys)
    bg = random.choice(bg_pool)
    swapped = swap_background(img, mask, bg)

    out_stem = f"bgswap_{uuid.uuid4().hex[:8]}_{stem}"
    cv2.imwrite(os.path.join(TRAIN_IMG, out_stem + ".jpg"), swapped)
    shutil.copy2(lbl_path, os.path.join(TRAIN_LBL, out_stem + ".txt"))
    saved += 1

print(f"\nSaved {saved} background-swapped augmentations into {TRAIN_IMG}")

os.makedirs(OUTPUT_DIR, exist_ok=True)
with open(os.path.join(OUTPUT_DIR, "phase4_background_swap_report.txt"), "w") as f:
    f.write(f"Background pool size: {len(bg_pool)}\n")
    f.write(f"Candidates containing {TARGET_CLASSES}: {len(candidates)}\n")
    f.write(f"Augmented (background-swapped): {saved} ({SWAP_FRACTION:.0%})\n")

print(f"Saved -> {OUTPUT_DIR}/phase4_background_swap_report.txt")

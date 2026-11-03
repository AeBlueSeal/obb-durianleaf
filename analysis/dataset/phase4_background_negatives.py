"""
Phase 4.1 — Background-only negatives for TRAIN.

The plan calls for ~150-300 new label-free field-condition images (grass,
soil, bark, non-durian foliage). Photographing new ones is outside this
session's reach, so instead this extracts genuine background-only crops
from the *existing* field train images: for every field training image,
find a crop window that doesn't overlap any annotated leaf polygon (with a
safety margin) and save it as a new unlabeled image. Same camera, same
orchard, same lighting as the rest of the field set — just no leaf in
frame — which is exactly the "same field conditions" requirement.

Adds an empty label file for each negative (Ultralytics treats this as
background, same as a missing label file, but explicit is safer).

Run from project root (after phase2_resplit.py):
    python analysis/dataset/phase4_background_negatives.py
"""

import os, glob, random, uuid

import cv2
import numpy as np
import yaml

DATA_YAML  = "data/dataset_resplit.yaml"
OUTPUT_DIR = "analysis/dataset/outputs/resplit"

random.seed(42)
np.random.seed(42)

TARGET_FRACTION = 0.08     # ~8% of train set, within the plan's 5-10% target
CROP_SIZE = 320            # square crop, matches the low end of the jitter pool
MARGIN_PX = 20             # keep crops this far from any leaf polygon
MAX_ATTEMPTS_PER_IMAGE = 15

with open(DATA_YAML) as f:
    cfg = yaml.safe_load(f)

DATA_ROOT = cfg["path"]
TRAIN_IMG = os.path.join(DATA_ROOT, "train", "images")
TRAIN_LBL = os.path.join(DATA_ROOT, "train", "labels")


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
            coords = np.array(parts[1:9], dtype=float).reshape(4, 2)
            coords[:, 0] *= w
            coords[:, 1] *= h
            polys.append(coords.astype(np.int32))
    return polys


n_train = len(glob.glob(os.path.join(TRAIN_IMG, "*")))
target_n = int(n_train * TARGET_FRACTION)
print(f"Train set has {n_train} images; targeting {target_n} background negatives "
      f"({TARGET_FRACTION:.0%})")

field_imgs = [p for p in glob.glob(os.path.join(TRAIN_IMG, "*")) if source_of(p) == "field"]
random.shuffle(field_imgs)

saved = 0
for img_path in field_imgs:
    if saved >= target_n:
        break
    img = cv2.imread(img_path)
    if img is None:
        continue
    h, w = img.shape[:2]
    if h < CROP_SIZE + 2 * MARGIN_PX or w < CROP_SIZE + 2 * MARGIN_PX:
        continue

    stem = os.path.splitext(os.path.basename(img_path))[0]
    lbl_path = os.path.join(TRAIN_LBL, stem + ".txt")
    polys = load_polys(lbl_path, w, h)

    # "forbidden" mask: leaf polygons dilated by MARGIN_PX
    forbidden = np.zeros((h, w), dtype=np.uint8)
    for poly in polys:
        cv2.fillPoly(forbidden, [poly], 255)
    if polys:
        kernel = np.ones((MARGIN_PX * 2 + 1, MARGIN_PX * 2 + 1), np.uint8)
        forbidden = cv2.dilate(forbidden, kernel)

    for _ in range(MAX_ATTEMPTS_PER_IMAGE):
        if saved >= target_n:
            break
        x0 = random.randint(0, w - CROP_SIZE)
        y0 = random.randint(0, h - CROP_SIZE)
        window = forbidden[y0:y0 + CROP_SIZE, x0:x0 + CROP_SIZE]
        if window.any():
            continue  # overlaps a (dilated) leaf polygon — try again

        crop = img[y0:y0 + CROP_SIZE, x0:x0 + CROP_SIZE]
        out_stem = f"bgneg_{uuid.uuid4().hex[:8]}_{stem}"
        cv2.imwrite(os.path.join(TRAIN_IMG, out_stem + ".jpg"), crop)
        open(os.path.join(TRAIN_LBL, out_stem + ".txt"), "w").close()  # empty = background
        saved += 1
        break  # one negative per source image keeps them visually diverse

print(f"\nSaved {saved} background-only negatives into {TRAIN_IMG} (empty label files).")
print(f"New train set size: {n_train + saved} images "
      f"({100 * saved / (n_train + saved):.1f}% background negatives)")

os.makedirs(OUTPUT_DIR, exist_ok=True)
with open(os.path.join(OUTPUT_DIR, "phase4_background_negatives_report.txt"), "w") as f:
    f.write(f"Background negatives added: {saved}\n")
    f.write(f"Source: crops from existing field train images, "
            f"{MARGIN_PX}px clear of any leaf polygon\n")
    f.write(f"Crop size: {CROP_SIZE}x{CROP_SIZE}\n")
    f.write(f"New train set size: {n_train + saved} "
            f"({100 * saved / (n_train + saved):.1f}% negatives)\n")

print(f"Saved -> {OUTPUT_DIR}/phase4_background_negatives_report.txt")

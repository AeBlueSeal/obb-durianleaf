"""
Phase 3 (cont.) — Remove the curated 400x400 fingerprint.

Test B1 showed every curated image is exactly 400x400, always — a
zero-effort proxy for "curated source" (and therefore, given the class/source
confound, a proxy for class). This mutates curated images *in place* inside
the resplit dataset (data/datasetv3_2_resplit — a copy, so this never touches
the original data/datasetv3_2) with:
  - resolution jitter (resize to a randomly chosen size), applied to most
    curated images — cheap, always safe, no label changes needed since OBB
    labels are stored normalized.
  - random crop-then-resize on a subset, which changes the fixed square
    framing too. Falls back to resize-only if the crop would clip an
    annotated leaf.

This directly targets Phase 5's letterbox-uniformity concern too: a
constant 1:1 aspect ratio produces a constant letterbox padding pattern the
model can key off just as easily as raw resolution.

Run from project root (after phase2_resplit.py has materialized the split):
    python analysis/dataset/phase3_augment_curated.py
"""

import os, glob, random
from collections import Counter

import pandas as pd
import yaml
from PIL import Image

DATA_YAML  = "data/dataset_resplit.yaml"
SPLITS     = ["train", "val", "test"]
OUTPUT_DIR = "analysis/dataset/outputs/resplit"

random.seed(42)

CROP_PROB   = 0.4                              # rest get resize-only jitter
TARGET_SIZES = [320, 384, 448, 512, 576, 640, 704, 768]  # jitter pool (independent w/h)

with open(DATA_YAML) as f:
    cfg = yaml.safe_load(f)

DATA_ROOT = cfg["path"]


def strip_uuid(fname):
    stem = os.path.splitext(fname)[0]
    parts = stem.split("-", 1)
    return parts[1] if len(parts) == 2 and len(parts[0]) == 8 else stem


def source_of(filepath):
    name = strip_uuid(os.path.basename(filepath))
    return "curated" if name.startswith("Leaf_") else "field"


def load_labels(lbl_path):
    """Return list of (cls_id, [(x,y), ...4 pts]) with x,y normalized 0-1."""
    rows = []
    if not os.path.exists(lbl_path):
        return rows
    with open(lbl_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 9:
                continue
            cls_id = int(parts[0])
            coords = [float(v) for v in parts[1:9]]
            pts = list(zip(coords[0::2], coords[1::2]))
            rows.append((cls_id, pts))
    return rows


def write_labels(lbl_path, rows):
    with open(lbl_path, "w") as f:
        for cls_id, pts in rows:
            flat = " ".join(f"{v:.6f}" for xy in pts for v in xy)
            f.write(f"{cls_id} {flat}\n")


records = []
for split in SPLITS:
    img_dir = os.path.join(DATA_ROOT, split, "images")
    lbl_dir = os.path.join(DATA_ROOT, split, "labels")
    for img_path in glob.glob(os.path.join(img_dir, "*")):
        if source_of(img_path) != "curated":
            continue
        stem = os.path.splitext(os.path.basename(img_path))[0]
        lbl_path = os.path.join(lbl_dir, stem + ".txt")

        img = Image.open(img_path).convert("RGB")
        orig_w, orig_h = img.size
        rows = load_labels(lbl_path)

        target_w = random.choice(TARGET_SIZES)
        target_h = random.choice(TARGET_SIZES)
        mode = "resize_only"

        if rows and random.random() < CROP_PROB:
            xs = [x for _, pts in rows for x, y in pts]
            ys = [y for _, pts in rows for x, y in pts]
            bbox_xmin, bbox_xmax = min(xs) * orig_w, max(xs) * orig_w
            bbox_ymin, bbox_ymax = min(ys) * orig_h, max(ys) * orig_h
            bbox_w, bbox_h = bbox_xmax - bbox_xmin, bbox_ymax - bbox_ymin

            # Curated leaf photos are framed edge-to-edge (bbox height is
            # often ~100% of the frame), so a fixed crop *fraction* usually
            # has no feasible placement. Instead draw crop_w/crop_h uniformly
            # between "just the bbox" and "the whole frame" — by construction
            # there is always a valid x0/y0 range, and axes with no real
            # margin (bbox spans the full dimension) just degrade to ~no
            # crop on that axis while the other axis can still vary.
            crop_w = random.uniform(bbox_w, orig_w)
            crop_h = random.uniform(bbox_h, orig_h)

            x0_lo, x0_hi = max(0.0, bbox_xmax - crop_w), min(bbox_xmin, orig_w - crop_w)
            y0_lo, y0_hi = max(0.0, bbox_ymax - crop_h), min(bbox_ymin, orig_h - crop_h)

            x0 = random.uniform(x0_lo, x0_hi)
            y0 = random.uniform(y0_lo, y0_hi)
            img = img.crop((int(x0), int(y0), int(x0 + crop_w), int(y0 + crop_h)))
            new_rows = []
            for cls_id, pts in rows:
                new_pts = []
                for x, y in pts:
                    px, py = x * orig_w - x0, y * orig_h - y0
                    nx = min(1.0, max(0.0, px / crop_w))
                    ny = min(1.0, max(0.0, py / crop_h))
                    new_pts.append((nx, ny))
                new_rows.append((cls_id, new_pts))
            rows = new_rows
            mode = "crop_then_resize"

        img = img.resize((target_w, target_h), Image.BILINEAR)
        img.save(img_path)
        if mode == "crop_then_resize":
            write_labels(lbl_path, rows)

        records.append({
            "path": img_path, "split": split, "mode": mode,
            "orig_size": f"{orig_w}x{orig_h}", "new_size": f"{target_w}x{target_h}",
        })

df = pd.DataFrame(records)
print(f"Augmented {len(df)} curated images")
print(df["mode"].value_counts().to_string())

print(f"\nUnique resolutions before: {df['orig_size'].nunique()} "
      f"(expect 1 — the fixed 400x400 fingerprint)")
print(f"Unique resolutions after:  {df['new_size'].nunique()} out of "
      f"{len(TARGET_SIZES)}^2 possible combos")

os.makedirs(OUTPUT_DIR, exist_ok=True)
df.to_csv(os.path.join(OUTPUT_DIR, "phase3_augment_log.csv"), index=False)
with open(os.path.join(OUTPUT_DIR, "phase3_augment_report.txt"), "w") as f:
    f.write(f"Augmented {len(df)} curated images\n")
    f.write(df["mode"].value_counts().to_string() + "\n\n")
    f.write(f"Unique resolutions before: {df['orig_size'].nunique()}\n")
    f.write(f"Unique resolutions after:  {df['new_size'].nunique()}\n")

print(f"\nSaved -> {OUTPUT_DIR}/phase3_augment_log.csv, phase3_augment_report.txt")
print("\n⚠️  This mutated images in data/datasetv3_2_resplit in place. "
      "Re-run phase2_resplit.py first if you need a clean copy again.")

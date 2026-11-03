"""
Phase 5 — Strip EXIF camera-model fingerprint.

Test B1 showed camera model is a clean source/field-device fingerprint
(iPhone 15 Pro, DNY-NX9, iPhone 16 Pro Max — all field-only). Re-saves
every image in the resplit dataset through PIL without forwarding EXIF,
which drops the tag entirely (pixels are untouched).

Note: the resize/letterbox pipeline itself is already uniform across
sources — scripts/base/train.py, evaluate.py, detect.py call Ultralytics'
model.train()/val()/predict() directly with no source-specific
preprocessing branch, so every image (curated or field, any split) already
goes through the same letterbox at imgsz. Nothing to change there.

Run from project root (after phase2_resplit.py, and phase3/4 augmentation
if you're running the whole pipeline in order):
    python analysis/dataset/phase5_strip_exif.py
"""

import os, glob

import pandas as pd
from PIL import Image

DATA_ROOT  = "data/datasetv3_2_resplit"
SPLITS     = ["train", "val", "test"]
OUTPUT_DIR = "analysis/dataset/outputs/resplit"

stripped = 0
had_camera = 0
errors = []

for split in SPLITS:
    img_dir = os.path.join(DATA_ROOT, split, "images")
    for p in glob.glob(os.path.join(img_dir, "*")):
        try:
            im = Image.open(p)
            exif = im.getexif()
            if exif.get(272):  # tag 272 = camera Model
                had_camera += 1
            im = im.convert("RGB")
            im.save(p)  # re-save without passing exif -> metadata dropped
            stripped += 1
        except Exception as e:
            errors.append((p, str(e)))

print(f"Re-saved {stripped} images (stripping EXIF); "
      f"{had_camera} previously carried a camera-model tag")
if errors:
    print(f"{len(errors)} errors:")
    for p, e in errors[:10]:
        print(f"  {p}: {e}")

# ── Verify ────────────────────────────────────────────────────────────
remaining = 0
for split in SPLITS:
    img_dir = os.path.join(DATA_ROOT, split, "images")
    for p in glob.glob(os.path.join(img_dir, "*")):
        im = Image.open(p)
        if im.getexif().get(272):
            remaining += 1

print(f"\nImages still carrying a camera-model tag after stripping: {remaining}")
verdict = "✅ EXIF camera fingerprint removed." if remaining == 0 else "❌ Some EXIF remains."
print(verdict)

os.makedirs(OUTPUT_DIR, exist_ok=True)
with open(os.path.join(OUTPUT_DIR, "phase5_exif_report.txt"), "w") as f:
    f.write(f"Images re-saved: {stripped}\n")
    f.write(f"Previously had camera-model tag: {had_camera}\n")
    f.write(f"Remaining after stripping: {remaining}\n")
    f.write(verdict + "\n")

print(f"Saved -> {OUTPUT_DIR}/phase5_exif_report.txt")

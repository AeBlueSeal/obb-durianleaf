"""
Test B1 — Metadata / resolution fingerprint.

Checks whether curated vs field images differ in resolution, aspect ratio,
or EXIF camera model — confirming a learnable non-disease cue exists.

Run from project root:
    python analysis/dataset/source_confound_B1.py
"""

import os, glob
import pandas as pd
from PIL import Image
import yaml

DATA_YAML  = os.environ.get("CONFOUND_DATA_YAML", "data/dataset.yaml")
SPLITS     = ["train", "val", "test"]
OUTPUT_DIR = os.environ.get("CONFOUND_OUTPUT_DIR", "analysis/notebooks/outputs/source_confound")

with open(DATA_YAML) as f:
    cfg = yaml.safe_load(f)

DATA_ROOT = cfg["path"]


def source_of(filepath):
    name = os.path.basename(filepath)
    name = name.split("-", 1)[-1] if "-" in name else name
    return "curated" if name.startswith("Leaf_") else "field"


recs = []
for split in SPLITS:
    img_dir = os.path.join(DATA_ROOT, split, "images")
    for p in glob.glob(os.path.join(img_dir, "*")):
        try:
            im = Image.open(p)
            w, h   = im.size
            exif   = im.getexif()
            camera = exif.get(272, "unknown")   # tag 272 = Model
        except Exception:
            continue
        recs.append({
            "src":    source_of(p),
            "split":  split,
            "w":      w,
            "h":      h,
            "aspect": round(w / h, 3),
            "camera": camera,
        })

df = pd.DataFrame(recs)

print("=" * 60)
print("Resolution and aspect ratio by source")
print("=" * 60)
print(df.groupby("src")[["w", "h", "aspect"]].describe().T)

print("\nCamera model by source:")
print(df.groupby("src")["camera"].value_counts().to_string())

print("\nResolution uniqueness:")
for src, grp in df.groupby("src"):
    sizes = grp[["w", "h"]].drop_duplicates()
    print(f"  {src}: {len(sizes)} unique resolutions out of {len(grp)} images")

# ── Save outputs ──────────────────────────────────────────────────────
os.makedirs(OUTPUT_DIR, exist_ok=True)

df.groupby("src")[["w", "h", "aspect"]].describe().to_csv(
    os.path.join(OUTPUT_DIR, "B1_resolution_stats.csv"))
df.groupby("src")["camera"].value_counts().rename("count").to_csv(
    os.path.join(OUTPUT_DIR, "B1_camera.csv"))

with open(os.path.join(OUTPUT_DIR, "B1_report.txt"), "w") as _f:
    _f.write("Resolution and aspect ratio by source:\n")
    _f.write(df.groupby("src")[["w", "h", "aspect"]].describe().T.to_string() + "\n\n")
    _f.write("Camera model by source:\n")
    _f.write(df.groupby("src")["camera"].value_counts().to_string() + "\n\n")
    _f.write("Resolution uniqueness:\n")
    for src, grp in df.groupby("src"):
        sizes = grp[["w", "h"]].drop_duplicates()
        _f.write(f"  {src}: {len(sizes)} unique resolutions out of {len(grp)} images\n")

print(f"Saved → {OUTPUT_DIR}/B1_resolution_stats.csv, B1_camera.csv, B1_report.txt")

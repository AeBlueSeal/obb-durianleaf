"""
Test D — Split-leakage audit via perceptual hashing.

Finds near-duplicate images that cross train/val/test boundaries.
Near-duplicates from curated datasets (same plant, slightly different crop)
can silently inflate validation and test scores.

Install:  pip install imagehash
Run from project root:
    python analysis/dataset/source_confound_D.py
"""

import os, glob
import pandas as pd
import yaml
from PIL import Image
import imagehash

DATA_YAML  = os.environ.get("CONFOUND_DATA_YAML", "data/dataset.yaml")
SPLITS      = ["train", "val", "test"]
HASH_THRESH = 5   # Hamming distance ≤ 5 ≈ near-duplicate
OUTPUT_DIR  = os.environ.get("CONFOUND_OUTPUT_DIR", "analysis/notebooks/outputs/source_confound")

with open(DATA_YAML) as f:
    cfg = yaml.safe_load(f)

DATA_ROOT = cfg["path"]


def split_of(path):
    for s in SPLITS:
        if f"/{s}/" in path:
            return s
    return "?"


def source_of(filepath):
    name = os.path.basename(filepath)
    name = name.split("-", 1)[-1] if "-" in name else name
    return "curated" if name.startswith("Leaf_") else "field"


print("Computing perceptual hashes …")
paths  = []
for split in SPLITS:
    paths += glob.glob(os.path.join(DATA_ROOT, split, "images", "*"))

hashes = {}
for p in paths:
    try:
        hashes[p] = imagehash.phash(Image.open(p))
    except Exception:
        pass  # skip unreadable files

print(f"Hashed {len(hashes)} images. Checking pairs …")

items = list(hashes.items())
leaks = []
for i in range(len(items)):
    pa, ha = items[i]
    for j in range(i + 1, len(items)):
        pb, hb = items[j]
        if split_of(pa) != split_of(pb) and (ha - hb) <= HASH_THRESH:
            leaks.append((pa, pb, int(ha - hb)))

print(f"\nCross-split near-duplicate pairs found: {len(leaks)}")
if leaks:
    print(f"\n{'Dist':>4}  {'Split A':>5}  {'Src A':>7}  {'File A':<40}  {'Split B':>5}  {'Src B':>7}  {'File B'}")
    print("-" * 110)
    for a, b, d in sorted(leaks, key=lambda x: x[2])[:40]:
        print(f"{d:>4}  {split_of(a):>5}  {source_of(a):>7}  "
              f"{os.path.basename(a):<40}  "
              f"{split_of(b):>5}  {source_of(b):>7}  {os.path.basename(b)}")
    if len(leaks) > 40:
        print(f"  … and {len(leaks) - 40} more")
    print("\n❌ Leakage detected — re-split so duplicates stay in one split.")
else:
    print("✅ No cross-split near-duplicates found.")

# ── Save outputs ──────────────────────────────────────────────────────
os.makedirs(OUTPUT_DIR, exist_ok=True)

leak_rows = [
    {
        "dist":    d,
        "split_a": split_of(a),
        "src_a":   source_of(a),
        "file_a":  os.path.basename(a),
        "split_b": split_of(b),
        "src_b":   source_of(b),
        "file_b":  os.path.basename(b),
    }
    for a, b, d in sorted(leaks, key=lambda x: x[2])
]
leak_df = pd.DataFrame(leak_rows) if leak_rows else pd.DataFrame(
    columns=["dist", "split_a", "src_a", "file_a", "split_b", "src_b", "file_b"])
leak_df.to_csv(os.path.join(OUTPUT_DIR, "D_leak_pairs.csv"), index=False)

verdict_d = (
    f"❌ {len(leaks)} cross-split near-duplicate pairs detected. Re-split required."
    if leaks else
    "✅ No cross-split near-duplicates found."
)
with open(os.path.join(OUTPUT_DIR, "D_report.txt"), "w") as _f:
    _f.write(f"Images hashed: {len(hashes)}\n")
    _f.write(f"Cross-split near-duplicate pairs: {len(leaks)}\n")
    _f.write(f"Hash threshold (Hamming): ≤ {HASH_THRESH}\n\n")
    _f.write(verdict_d + "\n")
    if leak_rows:
        _f.write("\nAll pairs:\n")
        _f.write(leak_df.to_string(index=False) + "\n")

print(f"\nSaved → {OUTPUT_DIR}/D_leak_pairs.csv, D_report.txt")

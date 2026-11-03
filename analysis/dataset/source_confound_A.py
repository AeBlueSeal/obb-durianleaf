"""
Test A — Class × Source contingency table.

Run from project root:
    python analysis/dataset/source_confound_A.py
"""

import os, glob
import pandas as pd
import numpy as np
import yaml

DATA_YAML  = os.environ.get("CONFOUND_DATA_YAML", "data/dataset.yaml")
SPLITS     = ["train", "val", "test"]
OUTPUT_DIR = os.environ.get("CONFOUND_OUTPUT_DIR", "analysis/notebooks/outputs/source_confound")

with open(DATA_YAML) as f:
    cfg = yaml.safe_load(f)

CLASSES   = cfg["names"]
DATA_ROOT = cfg["path"]


def source_of(filepath):
    name = os.path.basename(filepath)
    # strip UUID prefix (e.g. "a1b2c3d4-Leaf_Healthy_24.txt" → "Leaf_Healthy_24.txt")
    name = name.split("-", 1)[-1] if "-" in name else name
    if name.startswith("Leaf_"):
        return "curated"
    return "field"


rows = []
for split in SPLITS:
    label_dir = os.path.join(DATA_ROOT, split, "labels")
    for lbl in glob.glob(os.path.join(label_dir, "*.txt")):
        src = source_of(lbl)
        with open(lbl) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                cls_id = int(line.split()[0])
                rows.append({"class": CLASSES[cls_id], "source": src, "split": split})

df = pd.DataFrame(rows)

print("=" * 60)
print("Class × Source contingency table (annotation counts)")
print("=" * 60)
ct = pd.crosstab(df["class"], df["source"], margins=True)
print(ct)

print("\nRow-normalised — share of each class by source:")
rn = pd.crosstab(df["class"], df["source"], normalize="index").round(3)
print(rn)

# Cramér's V
ct_raw   = pd.crosstab(df["class"], df["source"]).values
n        = ct_raw.sum()
expected = np.outer(ct_raw.sum(axis=1), ct_raw.sum(axis=0)) / n
chi2     = ((ct_raw - expected) ** 2 / (expected + 1e-9)).sum()
k    = min(ct_raw.shape) - 1
cramers_v = np.sqrt(chi2 / (n * k)) if k > 0 else 0.0

print(f"\nCramér's V (class ↔ source): {cramers_v:.3f}")
print("  0.0 → fully independent  |  1.0 → fully confounded")

if cramers_v < 0.15:
    verdict = "✅ Sources are well-mixed within classes. No confound risk."
elif cramers_v < 0.40:
    verdict = "⚠️  Moderate correlation — proceed to Test B."
else:
    verdict = "❌ Strong correlation — sources are confounded with classes. Proceed to Test B."
print(f"\n{verdict}")

# ── Save outputs ──────────────────────────────────────────────────────
os.makedirs(OUTPUT_DIR, exist_ok=True)

ct.to_csv(os.path.join(OUTPUT_DIR, "A_contingency.csv"))
rn.to_csv(os.path.join(OUTPUT_DIR, "A_normalized.csv"))

with open(os.path.join(OUTPUT_DIR, "A_report.txt"), "w") as _f:
    _f.write(f"Cramér's V (class ↔ source): {cramers_v:.3f}\n")
    _f.write(f"Verdict: {verdict}\n\n")
    _f.write("Contingency table (annotation counts):\n")
    _f.write(ct.to_string() + "\n\n")
    _f.write("Row-normalised (share of each class by source):\n")
    _f.write(rn.to_string() + "\n")

print(f"Saved → {OUTPUT_DIR}/A_contingency.csv, A_normalized.csv, A_report.txt")

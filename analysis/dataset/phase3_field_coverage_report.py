"""
Phase 3 — Field-coverage report (Option B: scope honestly, no new photography).

For every class, reports curated vs field instance/image counts on the
resplit dataset and flags any class that cannot be reliably field-validated
(too few field instances to trust a per-source AP split). Feeds directly
into Phase 8's honest reporting requirement: classes flagged here must be
reported as curated-validated only, with field AP (if any) carrying an
explicit small-N caveat.

Run from project root (after phase2_resplit.py):
    python analysis/dataset/phase3_field_coverage_report.py
"""

import os, glob
from collections import Counter, defaultdict

import pandas as pd
import yaml

DATA_YAML  = "data/dataset_resplit.yaml"
SPLITS     = ["train", "val", "test"]
OUTPUT_DIR = "analysis/dataset/outputs/resplit"

# Below this many field *instances* (annotation count, not images), a class's
# field AP should not be reported with equal confidence to a high-N class.
MIN_RELIABLE_FIELD_INSTANCES = 50

with open(DATA_YAML) as f:
    cfg = yaml.safe_load(f)

CLASSES   = cfg["names"]
DATA_ROOT = cfg["path"]


def strip_uuid(fname):
    stem = os.path.splitext(fname)[0]
    parts = stem.split("-", 1)
    return parts[1] if len(parts) == 2 and len(parts[0]) == 8 else stem


def source_of(filepath):
    name = strip_uuid(os.path.basename(filepath))
    return "curated" if name.startswith("Leaf_") else "field"


rows = []
image_counts = defaultdict(lambda: defaultdict(set))  # (class, source) -> split -> set(images)

for split in SPLITS:
    lbl_dir = os.path.join(DATA_ROOT, split, "labels")
    for lbl in glob.glob(os.path.join(lbl_dir, "*.txt")):
        src = source_of(lbl)
        stem = os.path.splitext(os.path.basename(lbl))[0]
        with open(lbl) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                cls_id = int(line.split()[0])
                cls = CLASSES[cls_id]
                rows.append({"class": cls, "source": src, "split": split})
                image_counts[(cls, src)][split].add(stem)

df = pd.DataFrame(rows)

print("=" * 70)
print("Per-class instance counts by source (resplit dataset)")
print("=" * 70)
inst = pd.crosstab(df["class"], df["source"])
inst = inst.reindex(CLASSES).fillna(0).astype(int)
print(inst.to_string())

print("\nPer-class image counts by source (unique images containing that class):")
img_rows = []
for cls in CLASSES:
    for src in ["curated", "field"]:
        per_split = {s: len(image_counts[(cls, src)].get(s, set())) for s in SPLITS}
        total = sum(per_split.values())
        img_rows.append({"class": cls, "source": src, **per_split, "total": total})
img_df = pd.DataFrame(img_rows).set_index(["class", "source"])
print(img_df.to_string())

# ── Flag classes needing honest scoping ─────────────────────────────────
print("\n" + "=" * 70)
print("Field-validation reliability (Option B scoping)")
print("=" * 70)

flag_rows = []
for cls in CLASSES:
    field_n = int(inst.loc[cls, "field"]) if "field" in inst.columns else 0
    curated_n = int(inst.loc[cls, "curated"]) if "curated" in inst.columns else 0
    reliable = field_n >= MIN_RELIABLE_FIELD_INSTANCES
    verdict = ("field-validated" if reliable
               else "CURATED-VALIDATED ONLY — report field AP (if any) with explicit small-N caveat"
               if field_n > 0 else
               "CURATED-ONLY — no field examples exist for this class")
    flag_rows.append({
        "class": cls, "field_instances": field_n, "curated_instances": curated_n,
        "verdict": verdict,
    })
    marker = "✅" if reliable else "❌"
    print(f"{marker} {cls:<15} field={field_n:>5}  curated={curated_n:>5}  -> {verdict}")

scoping_df = pd.DataFrame(flag_rows).set_index("class")
needs_scoping = scoping_df[scoping_df["field_instances"] < MIN_RELIABLE_FIELD_INSTANCES]

print(f"\n{len(needs_scoping)} / {len(CLASSES)} classes fall below the "
      f"{MIN_RELIABLE_FIELD_INSTANCES}-instance reliability bar for field AP:")
for cls in needs_scoping.index:
    print(f"  - {cls}")

# ── Save outputs ──────────────────────────────────────────────────────
os.makedirs(OUTPUT_DIR, exist_ok=True)
inst.to_csv(os.path.join(OUTPUT_DIR, "phase3_instance_counts.csv"))
img_df.to_csv(os.path.join(OUTPUT_DIR, "phase3_image_counts.csv"))
scoping_df.to_csv(os.path.join(OUTPUT_DIR, "phase3_scoping.csv"))

with open(os.path.join(OUTPUT_DIR, "phase3_report.txt"), "w") as f:
    f.write("Per-class instance counts by source:\n")
    f.write(inst.to_string() + "\n\n")
    f.write("Field-validation reliability (Option B — scope honestly):\n")
    f.write(scoping_df.to_string() + "\n\n")
    f.write(f"Classes below {MIN_RELIABLE_FIELD_INSTANCES}-instance reliability bar: "
            f"{', '.join(needs_scoping.index)}\n")

print(f"\nSaved -> {OUTPUT_DIR}/phase3_instance_counts.csv, phase3_image_counts.csv, "
      f"phase3_scoping.csv, phase3_report.txt")

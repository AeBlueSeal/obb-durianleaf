# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.5
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Test C1 — Per-source AP breakdown (Colab/GPU)
#
# Evaluates the trained model separately on curated-only and field-only test
# images. A large gap in per-class AP means the model rides the source
# shortcut instead of learning the disease signal.
#
# Notebook form of `analysis/evaluation/source_confound_C1.py` — see
# `results/source_confound_pipeline.md` (Phase 7). Needs the Phase 6 weights;
# update `WEIGHTS` below to the new `best.pt` before running.

# %%
# !pip install -q ultralytics pyyaml pandas opencv-python-headless

# %% [markdown]
# Mounts Google Drive and points `PROJECT_ROOT` at the project folder, per
# `results/colab_project_structure.md` (same convention as
# `notebooks/00_setup_project_structure.ipynb`).

# %%
from google.colab import drive
drive.mount('/content/drive')

PROJECT_NAME = 'ProjZ'
PROJECT_ROOT = f'/content/drive/MyDrive/{PROJECT_NAME}'

# %% [markdown]
# ## Paths and constants
#
# Edit `DATA_YAML` / `OUTPUT_DIR` to re-run against a different split without overwriting the original baseline outputs (equivalent of the script's `CONFOUND_DATA_YAML` / `CONFOUND_OUTPUT_DIR` env vars).

# %%
import os, glob, shutil, tempfile
import yaml
import pandas as pd
from ultralytics import YOLO

DATA_YAML  = f"{PROJECT_ROOT}/data/dataset_resplit.yaml"
WEIGHTS    = f"{PROJECT_ROOT}/models/yolo11m-obb.pt"
IMG_SZ     = 640
OUTPUT_DIR = f"{PROJECT_ROOT}/analysis/notebooks/outputs/source_confound_after_fix"

with open(DATA_YAML) as f:
    cfg = yaml.safe_load(f)

CLASSES = cfg["names"]
# The yaml's `path:` field is a local machine path baked in at dataset-build
# time — override with the Drive-mounted location instead of trusting it.
DATA_ROOT = f"{PROJECT_ROOT}/data/datasetv3_2_resplit"
TEST_IMG  = os.path.join(DATA_ROOT, "test", "images")
TEST_LBL  = os.path.join(DATA_ROOT, "test", "labels")


# %% [markdown]
# ## Partition test images by source

# %%
def source_of(filepath):
    name = os.path.basename(filepath)
    name = name.split("-", 1)[-1] if "-" in name else name
    return "curated" if name.startswith("Leaf_") else "field"


all_test = glob.glob(os.path.join(TEST_IMG, "*"))
by_source = {"curated": [], "field": []}
for p in all_test:
    by_source[source_of(p)].append(p)

for src, imgs in by_source.items():
    print(f"  {src}: {len(imgs)} test images")

# %% [markdown]
# ## Evaluate per source

# %%
model   = YOLO(WEIGHTS)
results = {}

for src, img_paths in by_source.items():
    if not img_paths:
        print(f"Skipping {src} — no images.")
        continue

    # Build a temporary dataset structure
    tmpdir = tempfile.mkdtemp(prefix=f"confound_{src}_")
    tmp_img = os.path.join(tmpdir, "images", "test")
    tmp_lbl = os.path.join(tmpdir, "labels", "test")
    os.makedirs(tmp_img, exist_ok=True)
    os.makedirs(tmp_lbl, exist_ok=True)

    for p in img_paths:
        shutil.copy(p, tmp_img)
        stem  = os.path.splitext(os.path.basename(p))[0]
        label = os.path.join(TEST_LBL, stem + ".txt")
        if os.path.exists(label):
            shutil.copy(label, tmp_lbl)

    tmp_yaml = os.path.join(tmpdir, "data.yaml")
    with open(tmp_yaml, "w") as f:
        yaml.dump({
            "path":  tmpdir,
            "train": "images/test",
            "val":   "images/test",
            "test":  "images/test",
            "nc":    cfg["nc"],
            "names": cfg["names"],
        }, f)

    print(f"\n--- Evaluating on {src} test images ---")
    try:
        metrics = model.val(
            data=tmp_yaml,
            split="test",
            imgsz=IMG_SZ,
            task="obb",
            verbose=False,
            plots=False,
        )
        results[src] = metrics
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

# %% [markdown]
# ## Overall mAP comparison

# %%
print("\n" + "=" * 70)
print("Overall mAP by source")
print("=" * 70)
overall_rows = []
for src, m in results.items():
    overall_rows.append({
        "source":    src,
        "mAP@50":    round(float(m.box.map50), 4),
        "mAP@50-95": round(float(m.box.map),   4),
    })
print(pd.DataFrame(overall_rows).set_index("source").to_string())

if "curated" in results and "field" in results:
    gap = abs(results["curated"].box.map50 - results["field"].box.map50)
    print(f"\nmAP@50 gap (curated vs field): {gap:.4f}")
    if gap > 0.10:
        print("❌ Large gap — model performance is source-dependent. Confound likely.")
    elif gap > 0.05:
        print("⚠️  Moderate gap — worth investigating per-class breakdown.")
    else:
        print("✅ Small gap — model generalises across sources.")

# %% [markdown]
# ## Per-class AP@50 comparison

# %%
print("\n" + "=" * 70)
print("Per-class AP@50 by source  (most diagnostic — watch Rhizoctonia & Healthy)")
print("=" * 70)
per_class = {}
for src, m in results.items():
    for idx, cls_idx in enumerate(m.box.ap_class_index):
        cls_name = CLASSES[int(cls_idx)]
        per_class.setdefault(cls_name, {})[src] = round(float(m.box.ap50[idx]), 4)

pc_df = pd.DataFrame(per_class).T.sort_index()
pc_df["gap"] = (pc_df.get("curated", float("nan")) - pc_df.get("field", float("nan"))).round(4)
print(pc_df.to_string())
print("\nLarge positive gap → curated AP ≫ field AP → shortcut on that class.")

# %% [markdown]
# ## Save outputs

# %%
os.makedirs(OUTPUT_DIR, exist_ok=True)

pd.DataFrame(overall_rows).set_index("source").to_csv(
    os.path.join(OUTPUT_DIR, "C1_overall_map.csv"))
pc_df.to_csv(os.path.join(OUTPUT_DIR, "C1_per_class_ap.csv"))

with open(os.path.join(OUTPUT_DIR, "C1_report.txt"), "w") as _f:
    _f.write("Overall mAP by source:\n")
    _f.write(pd.DataFrame(overall_rows).set_index("source").to_string() + "\n\n")
    if "curated" in results and "field" in results:
        gap = abs(results["curated"].box.map50 - results["field"].box.map50)
        _f.write(f"mAP@50 gap (curated vs field): {gap:.4f}\n\n")
    _f.write("Per-class AP@50 by source (most diagnostic):\n")
    _f.write(pc_df.to_string() + "\n")

print(f"\nSaved → {OUTPUT_DIR}/C1_overall_map.csv, C1_per_class_ap.csv, C1_report.txt")

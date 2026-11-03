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
# # Experiment v4_confound_fix — Retrain (Colab/GPU)
#
# Notebook form of `scripts/run.py --experiment v4_confound_fix`, wired
# through this repo's experiment-runner instead of calling `train.py`
# directly. Same retrain as `train.ipynb`, driven by
# `scripts/experiments/v4_confound_fix/config.yaml`.
#
# Retrain on the confound-fixed dataset (see
# `results/source_confound_scoping.md` and `implementation_plan.md` Phases
# 1-5). Uses `data/dataset_resplit.yaml`: grouped/dedup'd/source+class-stratified
# 80/10/10 split, curated images resolution-jittered (no more fixed 400x400),
# background negatives + background-swap augmentation for Algal spot/Other
# added to train, EXIF camera tags stripped. Run
# `analysis/dataset/phase1`-`5` scripts first if `data/dataset_resplit.yaml`
# doesn't exist yet.

# %%
import sys, subprocess
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "ultralytics", "pyyaml"])

# %% [markdown]
# ## Setup — imports, config, dataset, hyperparameters & E-IRFS
#
# One cell that does every import, defines every variable, loads config,
# copies the dataset to local Colab disk, merges tuned hyperparameters, and
# monkey-patches Ultralytics with the capped E-IRFS sampler — so the Train
# and Test cells below only contain the actual `model.train()` / `model.val()`
# calls.

# %%
# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
import os, sys, shutil, json, datetime, yaml
from ultralytics import YOLO

from google.colab import drive
drive.mount('/content/drive')

PROJECT_NAME = 'ProjZ'
PROJECT_ROOT = f'/content/drive/MyDrive/{PROJECT_NAME}'

# ---------------------------------------------------------------------------
# Experiment config (scripts/experiments/v4_confound_fix/config.yaml)
# ---------------------------------------------------------------------------
EXPERIMENT  = "v4_confound_fix"
CONFIG_PATH = f"{PROJECT_ROOT}/scripts/experiments/{EXPERIMENT}/config.yaml"

with open(CONFIG_PATH) as f:
    cfg = yaml.safe_load(f)
cfg.pop("status", None)
cfg.pop("notes", None)

# ---------------------------------------------------------------------------
# Fix up the dataset yaml for Colab
#
# cfg["data"]'s `path:` field is a local machine path baked in at
# dataset-build time -- rewrite it to the Drive-mounted location, copy the
# dataset to local Colab disk (Drive I/O bottlenecks every epoch), and point
# Ultralytics at the local copy.
# ---------------------------------------------------------------------------
data_yaml_path = f"{PROJECT_ROOT}/{cfg['data']}"
with open(data_yaml_path) as f:
    data_cfg = yaml.safe_load(f)

DRIVE_DATA_ROOT = f"{PROJECT_ROOT}/data/{os.path.basename(data_cfg['path'].rstrip('/'))}"
LOCAL_DATA_ROOT = f"/content/{os.path.basename(DRIVE_DATA_ROOT)}"
if not os.path.exists(LOCAL_DATA_ROOT):
    print(f"Copying dataset to local disk ({LOCAL_DATA_ROOT}) ...")
    shutil.copytree(DRIVE_DATA_ROOT, LOCAL_DATA_ROOT)
else:
    print(f"Local dataset copy already exists at {LOCAL_DATA_ROOT}")
data_cfg["path"] = LOCAL_DATA_ROOT

DATA_COLAB = "/content/dataset_resplit_colab_v4_confound_fix.yaml"
with open(DATA_COLAB, "w") as f:
    yaml.dump(data_cfg, f)

cfg["data"] = DATA_COLAB
# cfg["model"] is left as the bare name from config.yaml ('yolo11m-obb.pt') --
# Ultralytics auto-downloads the DOTA-pretrained checkpoint fresh. This is NOT
# PROJECT_ROOT/models/yolo11m-obb.pt, which is the current durian model
# already fine-tuned on the pre-fix (confound-affected) split -- reusing it as
# the training seed would carry the shortcuts into the "fixed" retrain and
# invalidate the before/after comparison (see
# analysis/notebooks/confound_fix_colab_gpu.ipynb, which documents the same
# rule).

# ---------------------------------------------------------------------------
# Load tuned hyperparameters (prior Optuna run)
#
# Phase 6.1 (implementation_plan.md): reuse the optimizer/lr/momentum/
# loss-weight/augmentation values the prior 3-fold-CV Optuna search already
# found -- a re-split alone isn't expected to move those much, so don't
# re-search them from scratch.  best_params.json is the output of that search
# (notebooks/Template_Clean_E-IRFS_optuna_datasetv3.ipynb), synced to Drive
# at ProjZ/models/configs/best_params.json.
#
# cfg's own keys (data, epochs, imgsz, batch, patience, model) come from
# config.yaml and are retrain-specific overrides -- they win where both
# define the same key.
# ---------------------------------------------------------------------------
BEST_PARAMS_PATH = f"{PROJECT_ROOT}/models/configs/best_params.json"
with open(BEST_PARAMS_PATH) as f:
    best_params = json.load(f)

# Guard against silently training off the slow Drive-mounted path if this
# cell is re-run out of order after a runtime restart.
assert cfg.get("data", "").startswith("/content/"), (
    "cfg['data'] doesn't look like a local Colab path -- re-run this "
    "cell from the top (the dataset-fixup section rewrites it)."
)

cfg = {**best_params, **cfg}
# cos_lr=True was hardcoded (never a tuned param) in the Optuna search that
# produced best_params.json, so it isn't a key in that file -- re-set it
# explicitly here, otherwise training silently falls back to Ultralytics'
# default cos_lr=False, a different LR schedule than the one these "best"
# hyperparameters were actually tuned under.
cfg["cos_lr"] = True

# ---------------------------------------------------------------------------
# Re-sequence E-IRFS (Phase 5.5)
#
# results/source_confound_pipeline.md Phase 5.5: E-IRFS oversamples *images*,
# not *content*. Recomputed here from scratch on **this** dataset (the new
# dataset_resplit + Phase 4.1 background negatives), not carried over from
# the pre-fix run.
#
# Per results/source_confound_scoping.md, only **Rhizoctonia** is capped --
# it's curated-only (8 field vs. 325 curated instances), so letting E-IRFS
# fully oversample it would just repeat the same narrow curated cluster
# rather than teach the model more about the disease.  **Healthy is left
# uncapped**: it has plenty of field instances (179) and its AP gap was a
# background/preprocessing shortcut already addressed by Phases 3-5, not a
# data-starvation problem.
#
# Background-only negatives (bgneg_*/bgswap_*, empty label files) get a
# fixed neutral sampling weight of 1.0 -- confirmed below via the printed
# class stats -- so Phase 4.1's negative ratio isn't silently amplified or
# crowded out by sampling.
#
# The sampler class itself lives in src/data_loader.py (one shared
# implementation instead of a copy in every notebook).  If that file is
# missing from Drive or empty (the setup notebook creates empty placeholders),
# write it inline first.
# ---------------------------------------------------------------------------
sys.path.insert(0, PROJECT_ROOT)  # src/ lives at the Drive-mounted project root

# Colab kernels persist across notebook runs -- wipe any previously-cached
# src.* modules so the import below always reads from disk.
for mod in list(sys.modules.keys()):
    if mod == 'src' or mod.startswith('src.'):
        del sys.modules[mod]

DL_PATH = os.path.join(PROJECT_ROOT, 'src', 'data_loader.py')
_has_class = False
if os.path.exists(DL_PATH):
    with open(DL_PATH) as _f:
        _has_class = 'YOLOEIRFSCappedDataset' in _f.read()

if not _has_class:
    print('Writing src/data_loader.py to Drive (missing or empty placeholder) ...')
    os.makedirs(os.path.dirname(DL_PATH), exist_ok=True)
    with open(DL_PATH, 'w') as _f:
        _f.write(r'''
"""Shared E-IRFS class-imbalance sampler for durian OBB training."""
import csv
import os as _os
import numpy as np
from ultralytics.data.dataset import YOLODataset

EIRFS_ALPHA = 2.0
EIRFS_THRESHOLD = 0.0001
CAP_MULTIPLIER = 1.5
DEFAULT_CAPPED_CLASS_NAMES = {"Rhizoctonia"}
PHASE3_SCOPING_CSV = "analysis/dataset/outputs/resplit/phase3_scoping.csv"

def load_capped_class_names(scoping_csv=PHASE3_SCOPING_CSV, default=DEFAULT_CAPPED_CLASS_NAMES):
    if not _os.path.exists(scoping_csv):
        return set(default)
    capped = set()
    with open(scoping_csv) as f:
        for row in csv.DictReader(f):
            verdict = row.get("verdict", "")
            if verdict.startswith("CURATED-VALIDATED") or verdict.startswith("CURATED-ONLY"):
                capped.add(row["class"])
    return capped or set(default)

_stats_cache = {}

def _cache_key(dataset):
    im_files = getattr(dataset, "im_files", None)
    if not im_files:
        return None
    return (len(dataset.data["names"]), tuple(im_files))

class YOLOEIRFSCappedDataset(YOLODataset):
    def __init__(self, *args, alpha=EIRFS_ALPHA, t=EIRFS_THRESHOLD,
                 capped_class_names=None, cap_multiplier=CAP_MULTIPLIER, **kwargs):
        super().__init__(*args, **kwargs)
        self.train_mode = "train" in self.prefix
        self.alpha = alpha
        self.t = t
        self.capped_class_names = (
            load_capped_class_names() if capped_class_names is None else set(capped_class_names)
        )
        self.cap_multiplier = cap_multiplier
        self.num_classes = len(self.data["names"])
        cache_key = _cache_key(self)
        cached = _stats_cache.get(cache_key) if cache_key is not None else None
        if cached is not None:
            self.image_counts, self.instance_counts, self.class_weights = (
                cached[0].copy(), cached[1].copy(), cached[2].copy()
            )
        else:
            self.instance_counts = np.ones(self.num_classes, dtype=np.float32)
            self._count_class_frequencies()
            self._compute_class_weights()
            self._cap_confounded_classes()
            if cache_key is not None:
                _stats_cache[cache_key] = (
                    self.image_counts.copy(), self.instance_counts.copy(), self.class_weights.copy()
                )
        self._compute_image_weights()
        self.print_class_stats()

    def print_class_stats(self):
        print("\n=== E-IRFS Class Statistics (capped) ===")
        for class_id, name in self.data["names"].items():
            img_count = int(self.image_counts[class_id])
            inst_count = int(self.instance_counts[class_id])
            tag = " [CAPPED]" if name in self.capped_class_names else ""
            print(f"Class {class_id:2d} ({name:15}): {img_count:5d} images, "
                  f"{inst_count:6d} instances, weight={self.class_weights[class_id]:.3f}{tag}")
        n_bg = sum(1 for label in self.labels if not label['cls'].size)
        print(f"Background-only negatives: {n_bg} images, sampling weight fixed at 1.0 (neutral)")
        print("=========================================\n")

    def _count_class_frequencies(self):
        seen_in_image = [set() for _ in range(self.num_classes)]
        for label in self.labels:
            cls_ids = label['cls'].astype(int).flatten()
            unique = np.unique(cls_ids)
            for c in unique:
                seen_in_image[c].add(id(label))
            for c in cls_ids:
                self.instance_counts[c] += 1
        self.image_counts = np.array([len(s) if len(s) > 0 else 1 for s in seen_in_image], dtype=np.float32)

    def _compute_class_weights(self):
        f_ic = self.image_counts / np.sum(self.image_counts)
        f_bc = self.instance_counts / np.sum(self.instance_counts)
        self.class_weights = np.exp(self.alpha * np.sqrt(self.t / np.sqrt(f_ic * f_bc)))

    def _cap_confounded_classes(self):
        names = self.data["names"]
        capped_ids = [cid for cid, name in names.items() if name in self.capped_class_names]
        uncapped_ids = [cid for cid in names if cid not in capped_ids]
        if self.capped_class_names and not capped_ids:
            print(f"WARNING: E-IRFS cap requested for {sorted(self.capped_class_names)} "
                  f"but none of these class names matched this dataset's classes "
                  f"({list(names.values())}) -- cap NOT applied.")
        if not capped_ids or not uncapped_ids:
            return
        cap_value = self.cap_multiplier * float(np.median(self.class_weights[uncapped_ids]))
        for cid in capped_ids:
            if self.class_weights[cid] > cap_value:
                print(f"E-IRFS cap: {names[cid]} weight {self.class_weights[cid]:.3f} -> {cap_value:.3f}")
                self.class_weights[cid] = cap_value

    def _compute_image_weights(self):
        self.weights = np.array([
            np.max(self.class_weights[label['cls'].astype(int).flatten()])
            if label['cls'].size else 1.0
            for label in self.labels
        ])
        total = np.sum(self.weights)
        self.probabilities = self.weights / total if total > 0 else np.ones_like(self.weights) / max(len(self.weights), 1)
        self._cum_probabilities = np.cumsum(self.probabilities)

    def __getitem__(self, index):
        if self.train_mode and len(self._cum_probabilities):
            index = int(np.searchsorted(self._cum_probabilities, np.random.random(), side="right"))
            index = min(index, len(self.labels) - 1)
        return self.transforms(self.get_image_and_label(index))
'''.lstrip('\n'))
    # Clear the cache again so Python picks up the freshly-written file
    for mod in list(sys.modules.keys()):
        if mod == 'src' or mod.startswith('src.'):
            del sys.modules[mod]

import ultralytics.data.build as build
from src.data_loader import YOLOEIRFSCappedDataset

# Override Ultralytics dataset with the shared capped E-IRFS variant
build.YOLODataset = YOLOEIRFSCappedDataset

print("\nSetup complete.  Ready to train.\n")

# %% [markdown]
# ## Train

# %%
model = YOLO(cfg.pop("model"))
# Pop these before the splat -- best_params.json doesn't currently define
# them, but nothing prevents a future regenerated best_params.json from
# doing so, which would otherwise collide with the explicit kwargs below
# and raise "got multiple values for keyword argument".
for _key in ("project", "name", "exist_ok"):
    cfg.pop(_key, None)

model.train(
    **cfg,
    project  = f"{PROJECT_ROOT}/runs/{EXPERIMENT}",
    name     = "train",
    exist_ok = True,
)

# %% [markdown]
# ## Test
#
# Evaluate the just-trained `best.pt` against the held-out **test** split
# (never touched during training/validation) so every run gets an honest
# number before it's trusted — see `results/source_confound_pipeline.md`
# Phase 7 for the fuller confound-check suite to run on top of this.

# %%
best_weights = f"{PROJECT_ROOT}/runs/{EXPERIMENT}/train/weights/best.pt"
test_model = YOLO(best_weights)

test_results_dir = f"{PROJECT_ROOT}/runs/{EXPERIMENT}/test"

test_metrics = test_model.val(
    data=cfg["data"],
    split="test",
    task="obb",
    plots=True,
    project=f"{PROJECT_ROOT}/runs/{EXPERIMENT}",
    name="test",
    exist_ok=True,
)
print(test_metrics)

# The val call above writes confusion_matrix.png, results.png, and other
# plots into runs/<experiment>/test/ (Colab runtime local disk if the
# training project= pointed there, or Drive if the project root is on
# Drive).  Copy any remaining key training-curve artifacts from the
# train/ directory that aren't already in test/, so a single folder holds
# metrics + curves + confusion matrix.
train_dir = f"{PROJECT_ROOT}/runs/{EXPERIMENT}/train"
for _stem in ["results.png", "confusion_matrix.png", "results.csv"]:
    _src = os.path.join(train_dir, _stem)
    _dst = os.path.join(test_results_dir, _stem)
    if os.path.exists(_src) and not os.path.exists(_dst):
        shutil.copy2(_src, _dst)
        print(f"Copied {_stem} to test/")

# Persist test metrics alongside the training run so they survive
# Colab runtime recycling and are available for offline comparison
# (before/after confound fix, across experiments, etc.).
os.makedirs(test_results_dir, exist_ok=True)

# Machine-readable JSON (full metrics object, best-effort serialization)
try:
    metrics_dict = {
        "timestamp": datetime.datetime.now().isoformat(),
        "weights": best_weights,
        "data": cfg["data"],
        "split": "test",
        # Ultralytics Metrics objects have a .results_dict attribute
        # (dict of metric_name -> float) in recent versions
        "results": getattr(test_metrics, "results_dict", str(test_metrics)),
    }
    json_path = os.path.join(test_results_dir, "test_metrics.json")
    with open(json_path, "w") as f:
        json.dump(metrics_dict, f, indent=2, default=str)
    print(f"Test metrics saved to {json_path}")
except Exception as e:
    print(f"WARNING: Could not save test metrics JSON: {e}")

# Human-readable summary (same as the printed output, to a text file)
try:
    txt_path = os.path.join(test_results_dir, "test_metrics.txt")
    with open(txt_path, "w") as f:
        f.write(f"Timestamp: {datetime.datetime.now().isoformat()}\n")
        f.write(f"Weights:   {best_weights}\n")
        f.write(f"Data:      {cfg['data']}\n")
        f.write(f"Split:     test\n")
        f.write(f"\n{test_metrics}\n")
    print(f"Test metrics summary saved to {txt_path}")
except Exception as e:
    print(f"WARNING: Could not save test metrics summary: {e}")

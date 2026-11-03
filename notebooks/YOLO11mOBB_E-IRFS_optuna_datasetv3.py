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
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # YOLO11m-OBB -- E-IRFS + Optuna, datasetv3
#
# Baseline arm of the OBB comparison (`results/obb_model_comparison.md`).
# Byte-identical to `notebooks/Template_Clean_E-IRFS_optuna_datasetv3.py`
# except for the run/output paths below (`PROJECT`, `RUN_LABEL`,
# `best_final_yolo11m_obb`) so this arm and the YOLO26 arm
# (`notebooks/YOLO26mOBB_E-IRFS_optuna_datasetv3.py`) can be run into the same
# Drive project without overwriting each other's runs. If the template changes
# (bug fix, search-space edit), re-sync this file the same way --
# see the "propagate template changes" pattern already used for the
# Strip R-CNN / Oriented-DETR / RF-DETR-OBB arms.

# %% [markdown] id="KD8Ql8ZC23mh"
# ## file setup

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 6361, "status": "ok", "timestamp": 1770117844964, "user": {"displayName": "Pakorn Wongaroon \u0e21\u0e32\u0e1f\u0e34\u0e19 \u0e1c\u0e39\u0e49\u0e40\u0e25\u0e35\u0e49\u0e22\u0e07\u0e1a\u0e39\u0e48", "userId": "10427902391031551812"}, "user_tz": -420} id="wBzRgP4g0Cjn" outputId="f667b8b0-76ee-43b8-8f35-fd78f8160042"
# !pip3 install pyyaml ultralytics optuna pandas numpy

# %% [markdown]
# Mounts Google Drive and points `PROJECT_ROOT` at the project folder, per
# `results/colab_project_structure.md` (same convention as
# `notebooks/00_setup_project_structure.ipynb`). Training itself still runs
# against the local Colab disk (`PROJECT` below) since Drive I/O would
# bottleneck the many small per-trial writes during the Optuna search --
# only the final archived results get copied to Drive, at the end of this
# notebook.

# %%
from google.colab import drive
drive.mount('/content/drive')

PROJECT_NAME = 'ProjZ'
PROJECT_ROOT = f'/content/drive/MyDrive/{PROJECT_NAME}'

# %%
# !unzip -q /content/drive/MyDrive/Krong\ Si\ Su\ Pap/datasetv3_2.zip -d /content
# !cp /content/drive/MyDrive/Krong\ Si\ Su\ Pap/classes.txt /content/classes.txt

# %%
# !mv /content/datasetv3_2 /content/data

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 10429, "status": "ok", "timestamp": 1768336037894, "user": {"displayName": "Pakorn Wongaroon \u0e21\u0e32\u0e1f\u0e34\u0e19 \u0e1c\u0e39\u0e49\u0e40\u0e25\u0e35\u0e49\u0e22\u0e07\u0e1a\u0e39\u0e48", "userId": "10427902391031551812"}, "user_tz": -420} id="spXiuRos30O_" outputId="4c909cae-6134-4716-e9cd-e3c4f4688b9c"
import yaml
import os
import numpy as np
from ultralytics.data.dataset import YOLODataset
import ultralytics.data.build as build
import random
from glob import glob
from pathlib import Path
import optuna
from ultralytics import YOLO
import sys
import traceback
import pandas as pd
from statistics import mean
import json

EIRFS_THRESHOLD = 0.0001
TRAIN_EPOCHS = 250
IMG_EXTS = [".jpg", ".jpeg", ".png"]

# Phase 5.5 (results/source_confound_pipeline.md): only Rhizoctonia is capped
# -- it's curated-only (source_confound_scoping.md: 8 field vs. 325 curated
# instances), so letting E-IRFS fully oversample it just repeats the same
# narrow curated cluster. Healthy is NOT capped -- it has plenty of field
# instances and its AP gap was a background/preprocessing shortcut already
# addressed elsewhere, not a data-starvation problem.
CAPPED_CLASS_NAMES = {"Rhizoctonia"}
CAP_MULTIPLIER = 1.5  # capped classes' weight clipped to this x the median weight of uncapped classes

# --- Edit these paths ---
# NOTE: these still point at a flat pre-fix dataset dump. If reusing this
# notebook for the Phase 6.1 targeted HPO re-run, repoint these at the
# grouped/dedup'd/stratified resplit dataset (data/dataset_resplit.yaml +
# Phase 4.1 background negatives) before running -- otherwise this search
# recomputes E-IRFS frequencies on the pre-fix split, not the corrected one.
DATA_YAML  = "/content/data.yaml"      # base yaml (for nc/names). Can be anywhere.
IMAGES_DIR = "/content/data/train/images"         # flat images dir
LABELS_DIR = "/content/data/train/labels"         # flat OBB labels dir (.txt)
PRETRAINED = "yolo11m-obb.pt"          # auto-downloads if online; or absolute path to local .pt

# --- Training/Optuna knobs (you can keep defaults first run) ---
DEVICE      = 0            # GPU index or 'cpu'
# PROJECT MUST BE ABSOLUTE. Ultralytics re-roots a *relative* project= under
# {SETTINGS['runs_dir']}/{task}/, so project="runs/optuna_obb" actually trained
# into /content/runs/obb/runs/optuna_obb/ while this notebook's own plain-Python
# writes (cv_trial_*.csv, best_params.json) went to ./runs/optuna_obb/. That
# split is what caused the original bug: read_final_map_from_results() was given
# the un-rooted path, found no results.csv, and scored every fold 0.0 even
# though training itself was fine -- which tied every trial and made Optuna
# permanently report trial 0's seeded params. An absolute path can't be
# re-rooted, so training output and our own writes stay in one directory.
PROJECT     = "/content/runs/optuna_obb_yolo11m"
EPOCHS_PER_TRIAL = 50
# Two different patience regimes, same intent as the RF-DETR-OBB/StripRCNN
# templates: short patience for cheap HPO trials, much longer patience once
# we commit to the real final-training run so it isn't cut off before it's
# had a chance to converge over its much larger epoch budget.
PATIENCE         = 7    # HPO trials only
FINAL_PATIENCE   = 50   # final training on the full split only
SEED             = 42
K_FOLDS          = 3
SHUFFLE_BEFORE_SPLIT = True

# Repro + folders
random.seed(SEED)
assert Path(PROJECT).is_absolute(), (
    f"PROJECT must be an absolute path, got {PROJECT!r} -- a relative one gets "
    f"re-rooted by Ultralytics and splits training output away from this "
    f"notebook's own writes (see the note above)."
)
Path(PROJECT).mkdir(parents=True, exist_ok=True)


# %% [markdown] id="CFjedlSS3c4o"
# ### class definition

# %% id="irfv0ATc2yNV"
# used
# Find the image file that corresponds to a given label file.
# It tries all known image extensions (e.g. .jpg, .png, .jpeg)
# and returns the first match it finds. Checks both the usual case and the
# all-caps variant (e.g. ".jpg" and ".JPG") -- on a case-INsensitive
# filesystem (default macOS) an exact-case check alone already matches
# either, which is why a lowercase-only version of this went unnoticed
# locally; on a case-sensitive one (Colab/Linux, where training actually
# runs) it silently drops every upper-case-extension image -- confirmed on
# this dataset to be ~27% of every split (datasetv3_2 mixes ".jpg" and
# ".JPG" filenames). Only affects this notebook's own Optuna CV loop; final
# training below hands Ultralytics the plain data.yaml directory, which it
# scans natively and case-insensitively.
def find_image_for_label(label_path: Path) -> None:
    stem = label_path.stem
    for ext in IMG_EXTS:
        for candidate_ext in (ext, ext.upper()):
            p = Path(IMAGES_DIR) / f"{stem}{candidate_ext}"
            if p.exists():
                return p.resolve()

# used
# Collect all (image, label) pairs in the dataset.
# Only labels that have a matching image are included.
# This produces the list of items that will later be split into folds.
def collect_items() -> list:
    label_files = sorted(Path(LABELS_DIR).glob("*.txt"))
    items = []
    for lf in label_files:
        img = find_image_for_label(lf)
        if img is not None:
            items.append((str(img), str(lf.resolve())))
    return items

# used
# Assign each dataset item to one of K folds.
# This returns a list "fold_of" such that:
#   fold_of[i] = fold index (0..K-1) for item i
# If shuffle=True, items are randomly permuted before assignment.
def kfold_assignments(n_items: int, k: int, shuffle: bool=True) -> list:
    idxs = list(range(n_items))
    if shuffle:
        random.shuffle(idxs)
    fold_of = [None]*n_items
    for r, idx in enumerate(idxs):
        fold_of[idx] = r % k
    return fold_of

#used
# Load the base dataset YAML (YOLO format).
# This typically contains fields like:
#   nc: number of classes
#   names: list of class names
def load_base_yaml(yaml_path: str) -> dict:
    with open(yaml_path, "r") as f:
        return yaml.safe_load(f)

# used
# Create K-fold split files for YOLO training.
#
# For each fold k:
#   - train_fold{k}.txt contains image paths for training
#   - val_fold{k}.txt contains image paths for validation
#   - data_fold{k}.yaml is a YOLO data file pointing to those lists
#
# This allows you to run YOLO K times and average performance.
def make_folds_files(items, K, base_yaml_path, out_dir: Path) -> list:
    out_dir.mkdir(parents=True, exist_ok=True)
    base_yaml = load_base_yaml(base_yaml_path)

    # Extract class metadata from the base YAML
    names = base_yaml.get("names", None)
    nc    = base_yaml.get("nc", (len(names) if isinstance(names, list) else None))

    # Assign each item to a fold
    fold_of = kfold_assignments(len(items), K, SHUFFLE_BEFORE_SPLIT)
    fold_specs = []

    for k in range(K):
        # Training = all items NOT in fold k
        # Validation = all items IN fold k
        train_imgs = [items[i][0] for i in range(len(items)) if fold_of[i] != k]
        val_imgs   = [items[i][0] for i in range(len(items)) if fold_of[i] == k]

        # guarantee non-empty val
        if len(val_imgs) == 0 and len(train_imgs) > 0:
            val_imgs.append(train_imgs.pop())

        # Write image path lists
        train_txt = (out_dir / f"train_fold{k}.txt").resolve()
        val_txt   = (out_dir / f"val_fold{k}.txt").resolve()
        train_txt.write_text("\n".join(train_imgs))
        val_txt.write_text("\n".join(val_imgs))

        # Create YOLO data YAML for this fold
        data_yaml = (out_dir / f"data_fold{k}.yaml").resolve()
        fold_yaml_dict = {
            "task": "obb",              # rotated bounding box task
            "train": str(train_txt),
            "val": str(val_txt),
        }
        if nc is not None:    fold_yaml_dict["nc"] = nc
        if names is not None: fold_yaml_dict["names"] = names

        with open(data_yaml, "w") as f:
            yaml.safe_dump(fold_yaml_dict, f, sort_keys=False)

        fold_specs.append({"data_yaml": str(data_yaml)})
    return fold_specs

# used
# Read the final mAP50–95 score from a YOLO training run.
# It looks into results.csv and tries multiple possible column names
# because Ultralytics changes naming across versions.
def read_final_map_from_results(run_dir: Path) -> float:
    csv_path = run_dir / "results.csv"
    if not csv_path.exists():
        return float("nan")
    df = pd.read_csv(csv_path)
    if len(df) == 0:
        return float("nan")
    last = df.iloc[-1]
    # Known column names used by different YOLO versions
    candidates = [
        "metrics/mAP50-95(B)", "metrics/mAP50-95", "map50-95", "mAP50-95(B)", "metrics/mAP50-95(all)"
    ]
    for c in candidates:
        if c in df.columns:
            return float(last[c])

    # Fallback: find any column that looks like mAP50-95
    for c in df.columns:
        if "mAP50-95" in c or "map50-95" in c.lower():
            return float(last[c])
    return float("nan")

def clip01(x) -> float:
    return max(0.0, min(1.0, float(x)))

# used
# Python function to automatically create data.yaml config file
# 1. Reads "classes.txt" file to get list of class names
# 2. Creates data dictionary with correct paths to folders, number of classes, and names of classes
# 3. Writes data in YAML format to data.yaml
def create_data_yaml(path_to_classes_txt, path_to_data_yaml) -> None:
  # Read class.txt to get class names
  if not os.path.exists(path_to_classes_txt):
    print(f'classes.txt file not found! Please create a classes.txt labelmap and move it to {path_to_classes_txt}')
    return
  with open(path_to_classes_txt, 'r') as f:
    classes = []
    for line in f.readlines():
      if len(line.strip()) == 0: continue
      classes.append(line.strip())
  number_of_classes = len(classes)

  # Create data dictionary
  data = {
      'path': '/content/data',
      'train': 'train/images',
      'val': 'val/images',
      'test': 'test/images',
      'nc': number_of_classes,
      'names': classes
  }

  # Write data to YAML file
  with open(path_to_data_yaml, 'w') as f:
    yaml.dump(data, f, sort_keys=False)
  print(f'Created config file at {path_to_data_yaml}')

# used
# Phase 5.5 (results/source_confound_pipeline.md): capped E-IRFS variant --
# recomputes class frequencies fresh from whatever dataset this is pointed
# at, then clips CAPPED_CLASS_NAMES' weight so E-IRFS can't amplify the
# source confound for classes still data-starved on one source (see
# results/source_confound_scoping.md). Background-only negatives (empty
# label files) keep a neutral weight of 1.0, same as before.
#
# Canonical implementation lives in src/data_loader.py (imported by the
# Drive-mounted retrain notebooks); this notebook doesn't mount Drive and
# can't rely on that path being importable, so it keeps its own copy -- kept
# in sync with the canonical version, including the fixes below:
#   - warns (rather than silently no-op'ing) if CAPPED_CLASS_NAMES doesn't
#     match any class in this dataset
#   - caches per-fold class-weight computation so repeated Optuna trials on
#     the same fold don't redo the full O(n_labels) count/weight/cap pass
#   - precomputes the sampling CDF once instead of rebuilding it on every
#     __getitem__ call
#   - drops the unused `mode="train"` constructor kwarg (dead in the
#     original; train/val is derived from self.prefix, not this parameter)
_eirfs_stats_cache = {}


def _eirfs_cache_key(dataset):
    im_files = getattr(dataset, "im_files", None)
    if not im_files:
        return None
    return (len(dataset.data["names"]), tuple(im_files))


class YOLOEIRFSCappedDataset(YOLODataset):
    def __init__(self, *args, alpha=2.0, t=EIRFS_THRESHOLD,
                 capped_class_names=CAPPED_CLASS_NAMES, cap_multiplier=CAP_MULTIPLIER, **kwargs):
        super().__init__(*args, **kwargs)

        self.train_mode = "train" in self.prefix
        self.alpha = alpha
        self.t = t
        self.capped_class_names = capped_class_names
        self.cap_multiplier = cap_multiplier
        self.num_classes = len(self.data["names"])

        cache_key = _eirfs_cache_key(self)
        cached = _eirfs_stats_cache.get(cache_key) if cache_key is not None else None
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
                _eirfs_stats_cache[cache_key] = (
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
        """Count how often each class appears across images and instances."""
        seen_in_image = [set() for _ in range(self.num_classes)]
        for label in self.labels:
            cls_ids = label['cls'].astype(int).flatten()
            unique = np.unique(cls_ids)
            for c in unique:
                seen_in_image[c].add(id(label))  # ensure image-level uniqueness
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
            if label['cls'].size else 1.0   # background-only negatives: neutral, never capped or boosted
            for label in self.labels
        ])
        total = np.sum(self.weights)
        self.probabilities = self.weights / total if total > 0 else np.ones_like(self.weights) / max(len(self.weights), 1)
        # Precomputed once here rather than rebuilt by np.random.choice(p=...)
        # on every single __getitem__ call.
        self._cum_probabilities = np.cumsum(self.probabilities)

    def __getitem__(self, index):
        """Sample images based on precomputed probabilities (E-IRFS) during training."""
        if self.train_mode and len(self._cum_probabilities):
            index = int(np.searchsorted(self._cum_probabilities, np.random.random(), side="right"))
            index = min(index, len(self.labels) - 1)  # guard float-rounding at the tail
        return self.transforms(self.get_image_and_label(index))

# used
def objective(trial: optuna.trial.Trial) -> float:
    # --- Search space ---
    optimizer = trial.suggest_categorical("optimizer", ["SGD", "Adam", "AdamW"])

    lr0      = trial.suggest_float("lr0", 1e-4, 1e-1, log=True)
    lrf      = trial.suggest_float("lrf", 0.01, 1.0)
    momentum = trial.suggest_float("momentum", 0.7, 0.98)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)  # log can't include 0

    warmup_epochs   = trial.suggest_float("warmup_epochs", 0.0, 5.0)
    warmup_momentum = trial.suggest_float("warmup_momentum", 0.2, 0.95)
    warmup_bias_lr  = trial.suggest_float("warmup_bias_lr", 1e-3, 1e-1, log=True)

    box = trial.suggest_float("box", 2.0, 8.0)
    cls = trial.suggest_float("cls", 0.2, 2.0)
    dfl = trial.suggest_float("dfl", 0.5, 2.0)

    imgsz = trial.suggest_categorical("imgsz", [320, 640, 832, 1024])
    batch = trial.suggest_categorical("batch", [8, 16, 32])

    hsv_h = trial.suggest_float("hsv_h", 0.0, 0.015)
    hsv_s = trial.suggest_float("hsv_s", 0.0, 0.7)
    hsv_v = trial.suggest_float("hsv_v", 0.0, 0.7)
    translate   = trial.suggest_float("translate", 0.0, 0.2)
    scale       = trial.suggest_float("scale", 0.0, 1.0)   # must be <= 1.0 for recent Ultralytics
    shear       = trial.suggest_float("shear", 0.0, 5.0)
    perspective = trial.suggest_float("perspective", 0.0, 0.001)
    mosaic      = trial.suggest_float("mosaic", 0.0, 1.0)
    mixup       = trial.suggest_float("mixup", 0.0, 0.3)
    flipud      = trial.suggest_float("flipud", 0.0, 0.5)
    fliplr      = trial.suggest_float("fliplr", 0.0, 0.9)

    # --- Train K folds, average mAP ---
    fold_scores = []
    rows = []
    fold_errors = []

    for k, spec in enumerate(fold_specs):
        run_name = f"trial_{trial.number:03d}_fold{k}"
        model = YOLO(PRETRAINED)

        overrides = dict(
            task="obb",
            data=spec["data_yaml"],           # fold-specific yaml (absolute paths)
            epochs=EPOCHS_PER_TRIAL,
            patience=PATIENCE,
            imgsz=imgsz,
            batch=batch,
            device=DEVICE,
            project=PROJECT,
            name=run_name,
            exist_ok=True,
            seed=SEED,

            optimizer=optimizer,
            lr0=lr0, lrf=lrf, momentum=momentum, weight_decay=weight_decay,
            warmup_epochs=warmup_epochs, warmup_momentum=warmup_momentum, warmup_bias_lr=warmup_bias_lr,
            cos_lr=True,

            box=box, cls=cls, dfl=dfl,

            hsv_h=clip01(hsv_h), hsv_s=clip01(hsv_s), hsv_v=clip01(hsv_v),
            translate=clip01(translate), scale=clip01(scale), shear=shear,
            perspective=clip01(perspective), mosaic=clip01(mosaic), mixup=clip01(mixup),
            flipud=clip01(flipud), fliplr=clip01(fliplr),
        )

        try:
            model.train(**overrides)
            # NOTE: don't reconstruct this path from PROJECT/run_name -- Ultralytics
            # resolves a *relative* project= to {SETTINGS['runs_dir']}/{task}/{project}
            # internally (confirmed from this notebook's own prior run logs: with
            # project="runs/optuna_obb", the real save_dir was
            # "/content/runs/obb/runs/optuna_obb/..."), so a naive Path(PROJECT)/run_name
            # silently pointed at a directory with no results.csv in it -- every trial
            # scored 0.0 and the Optuna search never actually got real feedback. Read
            # the trainer's own resolved save_dir instead of guessing.
            run_dir = Path(model.trainer.save_dir)
            if not (run_dir / "results.csv").exists():
                # Belt-and-braces: with an absolute PROJECT this shouldn't happen,
                # but if Ultralytics ever relocates the run again, say so with both
                # paths rather than silently scoring 0.0 like the original code did.
                raise RuntimeError(
                    f"no results.csv in the trainer's save_dir {run_dir} "
                    f"(PROJECT={PROJECT}). Training output and score-reading have "
                    f"diverged again -- check how Ultralytics resolved project=."
                )
            score = read_final_map_from_results(run_dir)
            if pd.isna(score):
                # A missing/unparseable results.csv is a *failure*, not a real
                # score of zero -- see the degenerate-study note below.
                raise RuntimeError(
                    f"training produced no readable mAP50-95 in {run_dir}/results.csv"
                )
            score = float(score)
        except Exception as e:
            # Print the FULL traceback to stdout (not a one-line stderr note that
            # gets buried under Ultralytics' per-epoch logs). The original code
            # swallowed this and substituted score=0.0, which made every trial tie
            # at 0.0; Optuna only replaces best_trial on a *strict* improvement, so
            # a fully-tied study permanently reports trial 0 -- whose params come
            # from the seeded TPESampler before any training runs, hence the
            # "best_params.json is always identical regardless of model/epochs" bug.
            print(f"\n{'!'*70}\n[Trial {trial.number}] Fold {k} FAILED: {e}\n{'!'*70}")
            traceback.print_exc(file=sys.stdout)
            sys.stdout.flush()
            fold_errors.append(f"fold {k}: {type(e).__name__}: {e}")
            rows.append({"trial": trial.number, "fold": k, "mAP50-95": None, "error": str(e)})
            continue

        fold_scores.append(score)
        rows.append({"trial": trial.number, "fold": k, "mAP50-95": score, "error": None})

        trial.report(mean(fold_scores), step=k+1)
        # Pruning is deliberately NOT enabled. `trial.report()` above feeds the
        # PercentilePruner configured below, but the objective never calls
        # `should_prune()`, so no trial is ever cut short: every trial runs
        # every fold to completion. This is a standing project decision (the
        # pruning path was judged not worth the implementation cost), not an
        # oversight -- do not "fix" it. Note the consequence for the
        # cross-arm comparison: Strip R-CNN and S2A-Net DO prune, so those two
        # arms search under a different HPO budget than this one. See
        # results/kfold_comparability_audit.md, Finding 2.

    # Save per-trial fold scores (including failed folds, so a degenerate run is
    # visible in the CSVs rather than looking like a legitimate all-zero search).
    pd.DataFrame(rows).to_csv(Path(PROJECT) / f"cv_trial_{trial.number:03d}.csv", index=False)

    if not fold_scores:
        # Every fold failed. Fail fast on the very first trial so a setup/config
        # error surfaces immediately instead of after a full n_trials x K_FOLDS
        # run that silently produces nothing. Later trials raise TrialPruned so
        # they're recorded as PRUNED and excluded from best_trial entirely --
        # never as a 0.0 that can win the study by tie-break.
        detail = "; ".join(fold_errors)
        if trial.number == 0:
            raise RuntimeError(
                f"Trial 0 failed on all {len(fold_specs)} folds -- aborting the study "
                f"instead of reporting a meaningless best_params.json. Causes: {detail}"
            )
        raise optuna.TrialPruned(f"all folds failed: {detail}")

    if fold_errors:
        print(f"[Trial {trial.number}] WARNING: {len(fold_errors)}/{len(fold_specs)} folds "
              f"failed; CV mean is over the {len(fold_scores)} that succeeded. {'; '.join(fold_errors)}")

    return float(mean(fold_scores))


# %% [markdown] id="y4lQ9y8M32u4"
# ### Pipeline

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 30, "status": "ok", "timestamp": 1768336038228, "user": {"displayName": "Pakorn Wongaroon \u0e21\u0e32\u0e1f\u0e34\u0e19 \u0e1c\u0e39\u0e49\u0e40\u0e25\u0e35\u0e49\u0e22\u0e07\u0e1a\u0e39\u0e48", "userId": "10427902391031551812"}, "user_tz": -420} id="lNiMD1Br3s1u" outputId="ec9d92ea-cf45-4470-e2b7-7c14261e327f"
# Define path to classes.txt and create ymal
path_to_classes_txt = '/content/classes.txt'
path_to_data_yaml = '/content/data.yaml'

create_data_yaml(path_to_classes_txt, path_to_data_yaml)

print('\nFile contents:\n')
# !cat /content/data.yaml

# %% [markdown] id="-seM39vS3Az0"
# ### Overide sampling behavior

# %% id="sbJ2KIgo0r8G"
# Override Ultralytics dataset with this capped E-IRFS variant
build.YOLODataset = YOLOEIRFSCappedDataset

# %% [markdown] id="mYvx4N1z-koV"
# ## Optuna

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 12, "status": "ok", "timestamp": 1768336038413, "user": {"displayName": "Pakorn Wongaroon \u0e21\u0e32\u0e1f\u0e34\u0e19 \u0e1c\u0e39\u0e49\u0e40\u0e25\u0e35\u0e49\u0e22\u0e07\u0e1a\u0e39\u0e48", "userId": "10427902391031551812"}, "user_tz": -420} id="zO_xrMzN_o2O" outputId="7706eae2-679a-4a77-9894-d2b6b447f759"
#check
print("Images:", len(glob(f"{IMAGES_DIR}/*")))
print("Labels:", len(glob(f"{LABELS_DIR}/*.txt")))

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 25, "status": "ok", "timestamp": 1768336038439, "user": {"displayName": "Pakorn Wongaroon \u0e21\u0e32\u0e1f\u0e34\u0e19 \u0e1c\u0e39\u0e49\u0e40\u0e25\u0e35\u0e49\u0e22\u0e07\u0e1a\u0e39\u0e48", "userId": "10427902391031551812"}, "user_tz": -420} id="NxSfJuyAnC_H" outputId="4fa844b0-35a7-4121-ecd5-0c76ec770963"
items = collect_items()
print(f"Found {len(items)} labeled images.")
fold_files_dir = Path(PROJECT) / "fold_files"
fold_specs = make_folds_files(items, K_FOLDS, DATA_YAML, fold_files_dir)
# Peek
fold_specs[:2], os.path.exists(fold_specs[0]["data_yaml"])

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 12047442, "status": "ok", "timestamp": 1768348085900, "user": {"displayName": "Pakorn Wongaroon \u0e21\u0e32\u0e1f\u0e34\u0e19 \u0e1c\u0e39\u0e49\u0e40\u0e25\u0e35\u0e49\u0e22\u0e07\u0e1a\u0e39\u0e48", "userId": "10427902391031551812"}, "user_tz": -420} id="Lm9fE_SVAJWK" outputId="a262c520-960b-4210-d4d9-97441d90b7dc"
N_TRIALS = 10          # start small; scale up after it runs cleanly
N_STARTUP_TRIALS = 3   # must stay well below N_TRIALS -- see note below

print(f"Using TPESampler.")
# n_startup_trials MUST be set explicitly. Its default is 10, which equalled the
# old hardcoded n_trials=10 -- so the sampler never left its seeded random
# startup phase and TPE's Bayesian modelling never actually engaged; the whole
# "search" was seeded random sampling. Kept in sync with the pruner's own
# n_startup_trials below.
sampler = optuna.samplers.TPESampler(seed=SEED, n_startup_trials=N_STARTUP_TRIALS)

pruner = optuna.pruners.PercentilePruner(
    percentile=50, n_startup_trials=N_STARTUP_TRIALS, n_warmup_steps=2
)

study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)
study.optimize(objective, n_trials=N_TRIALS)

# --- Degenerate-study guard -------------------------------------------------
# Optuna only replaces best_trial on a STRICT improvement, so if every completed
# trial returns the same value the study permanently reports trial 0 -- whose
# params are fixed by the seeded sampler before any training happens, and are
# therefore identical across models/epochs. That's the exact failure this
# notebook hit. Refuse to write a best_params.json we know is meaningless.
completed = study.get_trials(deepcopy=False, states=(optuna.trial.TrialState.COMPLETE,))
print(f"Completed trials: {len(completed)} / {N_TRIALS} "
      f"(pruned/failed: {N_TRIALS - len(completed)})")

if not completed:
    raise RuntimeError(
        "No trial completed successfully -- every trial failed or was pruned. "
        "Scroll up for the per-fold tracebacks; best_params.json NOT written."
    )

distinct_values = {round(t.value, 10) for t in completed if t.value is not None}
if len(distinct_values) == 1 and len(completed) > 1:
    raise RuntimeError(
        f"All {len(completed)} completed trials returned the identical objective value "
        f"{distinct_values.pop()}. Optuna's tie-break would make trial 0 'best' regardless "
        f"of the search, producing a best_params.json that is the same every run no matter "
        f"which model/epochs you use. Fix the underlying training/scoring issue first; "
        f"best_params.json NOT written."
    )

print("Best CV mAP50-95:", study.best_value)
print("Best trial number:", study.best_trial.number)
print("Best params:", study.best_params)

if not os.path.exists(os.path.join(PROJECT, "best_final_yolo11m_obb")):
    os.makedirs(os.path.join(PROJECT, "best_final_yolo11m_obb"))

Path(os.path.join(PROJECT, "best_final_yolo11m_obb", "best_params.json")).write_text(json.dumps(study.best_params, indent=2))

# %% [markdown]
# ## Train

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 10560451, "status": "ok", "timestamp": 1768359168341, "user": {"displayName": "Pakorn Wongaroon \u0e21\u0e32\u0e1f\u0e34\u0e19 \u0e1c\u0e39\u0e49\u0e40\u0e25\u0e35\u0e49\u0e22\u0e07\u0e1a\u0e39\u0e48", "userId": "10427902391031551812"}, "user_tz": -420} id="f9H-lNRU8jRb" outputId="8bd923ee-df90-4cb9-eed9-9c7ca8d9dabb"
# Train with best params
best_params = json.loads((Path(PROJECT) / "best_final_yolo11m_obb" / "best_params.json").read_text())
model = YOLO(PRETRAINED)
final_name = "best_final_yolo11m_obb"

final_overrides = dict(
    task="obb",
    data="/content/data.yaml",
    epochs=TRAIN_EPOCHS,
    patience=FINAL_PATIENCE,
    imgsz=best_params.get("imgsz", 1024),
    batch=best_params.get("batch", 16),
    device=DEVICE,
    project=PROJECT,
    name=final_name,
    exist_ok=True,
    seed=SEED,

    optimizer=best_params.get("optimizer", "AdamW"),
    lr0=best_params.get("lr0", 0.003),
    lrf=best_params.get("lrf", 0.1),
    momentum=best_params.get("momentum", 0.9),
    weight_decay=best_params.get("weight_decay", 5e-4),
    warmup_epochs=best_params.get("warmup_epochs", 3.0),
    warmup_momentum=best_params.get("warmup_momentum", 0.8),
    warmup_bias_lr=best_params.get("warmup_bias_lr", 0.05),
    cos_lr=True,

    box=best_params.get("box", 6.0),
    cls=best_params.get("cls", 0.5),
    dfl=best_params.get("dfl", 1.5),

    hsv_h=best_params.get("hsv_h", 0.015),
    hsv_s=best_params.get("hsv_s", 0.7),
    hsv_v=best_params.get("hsv_v", 0.7),
    translate=best_params.get("translate", 0.2),
    scale=best_params.get("scale", 1.0),
    shear=best_params.get("shear", 0.0),
    perspective=best_params.get("perspective", 0.0),
    mosaic=best_params.get("mosaic", 1.0),
    mixup=best_params.get("mixup", 0.0),
    flipud=best_params.get("flipud", 0.0),
    fliplr=best_params.get("fliplr", 0.5),
)
model.train(**final_overrides)

# Read the trainer's own resolved save_dir rather than reconstructing
# Path(PROJECT)/final_name -- see the NOTE in objective() above; a relative
# project= doesn't end up where you'd naively expect.
FINAL_RUN_DIR = model.trainer.save_dir
print(f"Final run saved at: {FINAL_RUN_DIR}")

# %% [markdown] id="erKqaKk7plmI"
# ## test

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 19131, "status": "ok", "timestamp": 1768359187475, "user": {"displayName": "Pakorn Wongaroon \u0e21\u0e32\u0e1f\u0e34\u0e19 \u0e1c\u0e39\u0e49\u0e40\u0e25\u0e35\u0e49\u0e22\u0e07\u0e1a\u0e39\u0e48", "userId": "10427902391031551812"}, "user_tz": -420} id="SbGQihHFoKq6" outputId="dc3710d9-49d6-4d73-d376-b3703356de58"
from ultralytics import YOLO

# Restore the stock dataset class before evaluating -- build.YOLODataset was
# monkeypatched to YOLOEIRFSCappedDataset for training and was never reset,
# so without this, test-set evaluation would also construct the E-IRFS
# dataset (redoing its weight/cap computation and printing "E-IRFS Class
# Statistics" into what should be a clean held-out eval log for no benefit,
# since eval never consumes the sampling weights anyway).
import ultralytics.data.build as build
from ultralytics.data.dataset import YOLODataset as _StockYOLODataset
build.YOLODataset = _StockYOLODataset

# Load trained OBB model -- FINAL_RUN_DIR is the trainer's own resolved
# save_dir from the cell above (not a reconstructed path; see its NOTE).
model = YOLO(FINAL_RUN_DIR / "weights" / "best.pt")

# Run TEST set evaluation
metrics = model.val(
    data=DATA_YAML,
    split="test",   # use test set (not validation)
    task="obb",      # OBB (rotated boxes)
    plots=True
)

print(metrics)

# %%
import shutil
from pathlib import Path

PROJECT_NAME = 'ProjZ'
PROJECT_ROOT = f'/content/drive/MyDrive/{PROJECT_NAME}'


def _unique_drive_dir(base_dir: Path) -> Path:
    """Avoid merging into a stale previous archive: if base_dir already
    exists on Drive, return the first base_dir2, base_dir3, ... that doesn't."""
    if not base_dir.exists():
        return base_dir
    n = 2
    while True:
        candidate = base_dir.with_name(f"{base_dir.name}{n}")
        if not candidate.exists():
            return candidate
        n += 1


# Edit this to whatever label you want the Drive folder to have.
RUN_LABEL = "YOLO11m_E-IRFS_Optuna"

LOCAL_RUNS = Path("/content/runs")
DRIVE_DEST = _unique_drive_dir(Path(f"{PROJECT_ROOT}/runs/{RUN_LABEL}"))
DRIVE_DEST.mkdir(parents=True, exist_ok=True)
if DRIVE_DEST.name != RUN_LABEL:
    print(f"NOTE: '{RUN_LABEL}' already exists on Drive -- archiving to '{DRIVE_DEST.name}' instead.")

shutil.copytree(LOCAL_RUNS, DRIVE_DEST, dirs_exist_ok=True)

print("Local  :", LOCAL_RUNS, "->", sorted(p.name for p in LOCAL_RUNS.iterdir()))
print("Drive  :", DRIVE_DEST, "->", sorted(p.name for p in DRIVE_DEST.iterdir()))

# %% [markdown]
# ## Re-test from Drive (resume without retraining)
#
# If you're in a fresh runtime and just want to re-verify a previously
# archived run without retraining, run this section after the mount/constants
# cells above and skip Optuna/training entirely.

# %% id="91ehgnBAPKkC"
# Restores the whole /content/runs tree from its RUN_LABEL folder on Drive --
# PROJECT is a subfolder of /content/runs, so this repopulates it along with
# everything else that was archived alongside it.
import shutil
from pathlib import Path

RUN_LABEL = "YOLO11m_E-IRFS_Optuna"
LOCAL_RUNS = Path("/content/runs")
DRIVE_DEST = Path(f"{PROJECT_ROOT}/runs/{RUN_LABEL}")
FINAL_RUN_DIR = Path(PROJECT) / "best_final_yolo11m_obb"

LOCAL_RUNS.mkdir(parents=True, exist_ok=True)
shutil.copytree(DRIVE_DEST, LOCAL_RUNS, dirs_exist_ok=True)
print(f"Restored {LOCAL_RUNS} <- {DRIVE_DEST}")

# %% id="dh4QWFPtPJ_Z"
from ultralytics import YOLO

# Restore the stock dataset class before evaluating (see comment in the
# earlier eval cell -- harmless no-op if this is a fresh kernel where the
# monkeypatch was never applied).
import ultralytics.data.build as build
from ultralytics.data.dataset import YOLODataset as _StockYOLODataset
build.YOLODataset = _StockYOLODataset

model = YOLO(FINAL_RUN_DIR / "weights" / "best.pt")

# Run TEST set evaluation
metrics = model.val(
    data=DATA_YAML,
    split="test",   # use test set (not validation)
    task="obb",      # OBB (rotated boxes)
    plots=True
)

print(metrics)

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
#     name: python3
# ---

# %% [markdown] id="KD8Ql8ZC23mh"
# ## file setup

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 6361, "status": "ok", "timestamp": 1770117844964, "user": {"displayName": "Pakorn Wongaroon \u0e21\u0e32\u0e1f\u0e34\u0e19 \u0e1c\u0e39\u0e49\u0e40\u0e25\u0e35\u0e49\u0e22\u0e07\u0e1a\u0e39\u0e48", "userId": "10427902391031551812"}, "user_tz": -420} id="wBzRgP4g0Cjn" outputId="f667b8b0-76ee-43b8-8f35-fd78f8160042"
# !pip3 install pyyaml ultralytics optuna pandas numpy sahi

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

# %% [markdown]
# Unzips the 640x640 SAHI-tiled dataset (from
# `notebooks/SAHI_Tile_Dataset_datasetv3_2.py`) that Optuna's K-fold HPO and
# the final training run below use. The ORIGINAL, full-size `datasetv3_2.zip`
# is unzipped separately, right before the "SAHI: Slicing Aided Hyper
# Inference" section further down -- that's the only place in this notebook
# that reads it, since it slices these full-size images into tiles ITSELF at
# inference time to simulate real deployment. Running that section against
# the already-tiled dataset instead would be degenerate -- SAHI slicing a
# 640x640 image into 640x640 tiles with 20% overlap produces one
# near-identical tile, which defeats the entire point of testing sliced
# inference.

# %%
# !unzip -q "/content/drive/MyDrive/Krong Si Su Pap/datasetv3_2_sahi_tiles.zip" -d /content/data_sahi_tiles

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 10429, "status": "ok", "timestamp": 1768336037894, "user": {"displayName": "Pakorn Wongaroon \u0e21\u0e32\u0e1f\u0e34\u0e19 \u0e1c\u0e39\u0e49\u0e40\u0e25\u0e35\u0e49\u0e22\u0e07\u0e1a\u0e39\u0e48", "userId": "10427902391031551812"}, "user_tz": -420} id="spXiuRos30O_" outputId="4c909cae-6134-4716-e9cd-e3c4f4688b9c"
import yaml
import os
import re
import numpy as np
from ultralytics.data.dataset import YOLODataset
import ultralytics.data.build as build
import random
from glob import glob
from pathlib import Path
import optuna
from ultralytics import YOLO
import sys
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
# Optuna's K-fold HPO and the final training run below both read from the
# TILED dataset (notebooks/SAHI_Tile_Dataset_datasetv3_2.py's output), not
# the flat pre-fix dump the other OBB notebooks in this repo default to --
# this notebook exists specifically to train tile-native. Its own data.yaml
# (nc/names/paths) was already written correctly by that notebook; nothing
# here regenerates it.
DATA_YAML  = "/content/data_sahi_tiles/data.yaml"
IMAGES_DIR = "/content/data_sahi_tiles/train/images"
LABELS_DIR = "/content/data_sahi_tiles/train/labels"
# Deliberately the ORIGINAL, full-size test images -- not the tiled test
# split -- because the SAHI sliced-inference section below slices these
# itself to simulate real deployment. See the bootstrap cell's note above.
TEST_IMAGES_DIR = "/content/data/test/images"
PRETRAINED = "yolo11m-obb.pt"          # auto-downloads if online; or absolute path to local .pt

# --- Training/Optuna knobs (you can keep defaults first run) ---
DEVICE      = 0            # GPU index or 'cpu'
PROJECT     = "runs/optuna_obb"
EPOCHS_PER_TRIAL = 25
PATIENCE         = 5
SEED             = 42
K_FOLDS          = 3
SHUFFLE_BEFORE_SPLIT = True
IMG_EXTS   = [".jpg", ".jpeg", ".png"]

# --- SAHI inference settings ---
SAHI_SLICE_HEIGHT         = 640
SAHI_SLICE_WIDTH          = 640
SAHI_OVERLAP_HEIGHT_RATIO = 0.2
SAHI_OVERLAP_WIDTH_RATIO  = 0.2
SAHI_CONFIDENCE_THRESHOLD = 0.25

# Training image size is FIXED, not Optuna-searched, because every training
# image is already a 640x640 tile (SAHI_SLICE_HEIGHT/WIDTH above) -- resizing
# tiles to some other size at train time would reintroduce a train/inference
# scale mismatch, which is exactly what tile-native training exists to avoid.
IMGSZ = 640

# Tiles are named "<source_stem>_tile<NNN>" by
# notebooks/SAHI_Tile_Dataset_datasetv3_2.py's slice_image(). Adjacent tiles
# from the same source image overlap 20% and are near-duplicates of each
# other -- used below to keep every tile from one source image in the same
# K-fold split, so the Optuna CV can't leak a near-duplicate crop between a
# fold's train and val partitions (the same class of leakage
# results/source_confound_pipeline.md's Phase D fix already had to clean up
# once for whole-image duplicates, recreated at tile granularity by training
# on tiles at all).
_TILE_STEM_RE = re.compile(r"^(.*)_tile\d{3}$")


def source_stem_of(path) -> str:
    """Map a tile's stem back to its source image's stem. Falls back to the
    stem itself for any file that isn't a recognisable tile name, so this
    degrades gracefully to today's per-image fold assignment if this notebook
    is ever pointed back at a non-tiled dataset."""
    stem = Path(path).stem
    m = _TILE_STEM_RE.match(stem)
    return m.group(1) if m else stem


# Repro + folders
random.seed(SEED)
Path(PROJECT).mkdir(parents=True, exist_ok=True)


# %% [markdown] id="CFjedlSS3c4o"
# ### class definition

# %% id="irfv0ATc2yNV"
# used
# Find the image file that corresponds to a given label file.
# It tries all known image extensions (e.g. .jpg, .png, .jpeg)
# and returns the first match it finds.
def find_image_for_label(label_path: Path) -> None:
    stem = label_path.stem
    for ext in IMG_EXTS:
        p = Path(IMAGES_DIR) / f"{stem}{ext}"
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
# Assign each dataset item to one of K folds, GROUPED by source image (see
# source_stem_of above) rather than per-tile -- every tile from one source
# image lands in the same fold, together, so overlapping near-duplicate tiles
# can never straddle a fold's train/val boundary. Returns "fold_of" such that
# fold_of[i] = fold index (0..K-1) for items[i].
def kfold_assignments(items: list, k: int, shuffle: bool=True) -> list:
    groups = sorted({source_stem_of(img) for img, _ in items})
    if shuffle:
        random.shuffle(groups)
    fold_of_group = {g: i % k for i, g in enumerate(groups)}
    return [fold_of_group[source_stem_of(img)] for img, _ in items]

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

    # Assign each item to a fold, grouped by source image (see
    # kfold_assignments/source_stem_of above) so overlapping tiles of the
    # same source image never straddle a fold's train/val boundary.
    fold_of = kfold_assignments(items, K, SHUFFLE_BEFORE_SPLIT)
    fold_specs = []

    for k in range(K):
        # Training = all items NOT in fold k
        # Validation = all items IN fold k
        train_imgs = [items[i][0] for i in range(len(items)) if fold_of[i] != k]
        val_imgs   = [items[i][0] for i in range(len(items)) if fold_of[i] == k]

        # guarantee non-empty val -- move a WHOLE source-image group from
        # train to val, not a single tile. Moving just one tile while its
        # overlapping siblings stayed in train would recreate exactly the
        # cross-fold leakage the grouping above exists to prevent.
        if len(val_imgs) == 0 and len(train_imgs) > 0:
            stray_group = source_stem_of(train_imgs[-1])
            moved = [p for p in train_imgs if source_stem_of(p) == stray_group]
            train_imgs = [p for p in train_imgs if source_stem_of(p) != stray_group]
            val_imgs.extend(moved)
            print(f"WARNING: fold {k} had an empty val split -- moved source "
                  f"image group '{stray_group}' ({len(moved)} tiles) from "
                  f"train to val instead of a single tile.")

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

    # imgsz is NOT searched here -- fixed at IMGSZ (640, matching the tile
    # size) per the note where IMGSZ is defined above.
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

    for k, spec in enumerate(fold_specs):
        run_name = f"trial_{trial.number:03d}_fold{k}"
        model = YOLO(PRETRAINED)

        overrides = dict(
            task="obb",
            data=spec["data_yaml"],           # fold-specific yaml (absolute paths)
            epochs=EPOCHS_PER_TRIAL,
            patience=PATIENCE,
            imgsz=IMGSZ,
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
            run_dir = model.trainer.save_dir
            score = read_final_map_from_results(run_dir)
            if pd.isna(score):
                score = 0.0
        except Exception as e:
            print(f"[Trial {trial.number}] Fold {k} failed: {e}", file=sys.stderr)
            score = 0.0

        fold_scores.append(float(score))
        rows.append({"trial": trial.number, "fold": k, "mAP50-95": float(score)})

        trial.report(mean(fold_scores), step=k+1)
        # Optional pruning inside-CV:
        # if trial.should_prune(): raise optuna.TrialPruned()

    cv_mean = float(mean(fold_scores)) if fold_scores else 0.0

    # Save per-trial fold scores
    pd.DataFrame(rows).to_csv(Path(PROJECT) / f"cv_trial_{trial.number:03d}.csv", index=False)
    return cv_mean


# %% [markdown] id="y4lQ9y8M32u4"
# ### Pipeline

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 30, "status": "ok", "timestamp": 1768336038228, "user": {"displayName": "Pakorn Wongaroon \u0e21\u0e32\u0e1f\u0e34\u0e19 \u0e1c\u0e39\u0e49\u0e40\u0e25\u0e35\u0e49\u0e22\u0e07\u0e1a\u0e39\u0e48", "userId": "10427902391031551812"}, "user_tz": -420} id="lNiMD1Br3s1u" outputId="ec9d92ea-cf45-4470-e2b7-7c14261e327f"
# create_data_yaml() (defined above, unused here now) used to regenerate
# /content/data.yaml for the FLAT dataset. DATA_YAML now points at the tiled
# dataset's own data.yaml, already written correctly by
# notebooks/SAHI_Tile_Dataset_datasetv3_2.py -- don't overwrite it; just
# confirm it's actually there before Optuna starts.
assert Path(DATA_YAML).exists(), (
    f"{DATA_YAML} not found -- did the datasetv3_2_sahi_tiles.zip bootstrap "
    f"cell run? See the note on the unzip cells above."
)
print(f"Using {DATA_YAML}:\n")
print(Path(DATA_YAML).read_text())

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
print(f"Using TPESampler.")
sampler = optuna.samplers.TPESampler(seed=SEED)

pruner = optuna.pruners.PercentilePruner(percentile=50, n_startup_trials=3, n_warmup_steps=2)

study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)
study.optimize(objective, n_trials=10)   # start small; scale up after it runs cleanly

print("Best CV mAP50-95:", study.best_value)
print("Best params:", study.best_params)

if not os.path.exists(os.path.join(PROJECT, "best_final_all_m")):
    os.makedirs(os.path.join(PROJECT, "best_final_all_m"))

Path(os.path.join(PROJECT, "best_final_all_m", "best_params.json")).write_text(json.dumps(study.best_params, indent=2))

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 10560451, "status": "ok", "timestamp": 1768359168341, "user": {"displayName": "Pakorn Wongaroon \u0e21\u0e32\u0e1f\u0e34\u0e19 \u0e1c\u0e39\u0e49\u0e40\u0e25\u0e35\u0e49\u0e22\u0e07\u0e1a\u0e39\u0e48", "userId": "10427902391031551812"}, "user_tz": -420} id="f9H-lNRU8jRb" outputId="8bd923ee-df90-4cb9-eed9-9c7ca8d9dabb"
# Train with best params
best_params = json.loads((Path(PROJECT) / "best_final_all_m" / "best_params.json").read_text())
model = YOLO(PRETRAINED)
final_name = "best_final_all_m"

final_overrides = dict(
    task="obb",
    # Was hardcoded to "/content/data.yaml" -- silently the FLAT dataset's
    # yaml regardless of what DATA_YAML above was set to. Now that DATA_YAML
    # points at the tiled dataset, that hardcoding would have trained the
    # final model on the wrong data after everything upstream (Optuna,
    # E-IRFS, the K-fold CV) was correctly switched to tiles.
    data=DATA_YAML,
    epochs=TRAIN_EPOCHS,
    patience=50,
    imgsz=IMGSZ,   # fixed, not from best_params -- see IMGSZ's definition above
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
# ### test

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

# TILE-domain test evaluation: DATA_YAML points at the tiled dataset, so this
# scores the model on its own training-domain test split (same 640x640 tile
# scale it was trained on). This is a different, complementary number to the
# "SAHI: Slicing Aided Hyper Inference" section below, which evaluates the
# SAME model against the ORIGINAL, full-size test images sliced at inference
# time -- that section is the realistic deployment-scenario number; this one
# is the training-domain sanity check.
metrics = model.val(
    data=DATA_YAML,
    split="test",   # use test set (not validation)
    task="obb",      # OBB (rotated boxes)
    plots=True
)

print(metrics)

# %% [markdown]
# ## SAHI: Slicing Aided Hyper Inference
#
# SAHI slices each high-resolution test image into overlapping tiles, runs the OBB detector on every tile, then merges predictions with NMS. This dramatically improves recall for small lesions on large leaf images.
#
# Needs the ORIGINAL, full-size `datasetv3_2.zip` -- unzipped here rather than
# up top, since this is the only place in the notebook that reads it (see the
# note on the tile-dataset bootstrap cell near the top).

# %%
# !unzip -q /content/drive/MyDrive/Krong\ Si\ Su\ Pap/datasetv3_2.zip -d /content
# !cp /content/drive/MyDrive/Krong\ Si\ Su\ Pap/classes.txt /content/classes.txt

# %%
# !mv /content/datasetv3_2 /content/data

# %%
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from sahi.utils.cv import visualize_object_predictions
import cv2
from PIL import Image

best_weights = FINAL_RUN_DIR / "weights" / "best.pt"

sahi_model = AutoDetectionModel.from_pretrained(
    model_type='ultralytics',
    model_path=str(best_weights),
    confidence_threshold=SAHI_CONFIDENCE_THRESHOLD,
    device=f"cuda:{DEVICE}" if isinstance(DEVICE, int) else str(DEVICE),
)
print("SAHI detection model loaded.")

# %%
SAHI_OUTPUT_DIR = Path(PROJECT) / "sahi_predictions"
SAHI_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

test_images = sorted(
    p for ext in IMG_EXTS
    for p in Path(TEST_IMAGES_DIR).glob(f"*{ext}")
)
print(f"Found {len(test_images)} test images.")

sahi_results = []

for img_path in test_images:
    result = get_sliced_prediction(
        str(img_path),
        sahi_model,
        slice_height=SAHI_SLICE_HEIGHT,
        slice_width=SAHI_SLICE_WIDTH,
        overlap_height_ratio=SAHI_OVERLAP_HEIGHT_RATIO,
        overlap_width_ratio=SAHI_OVERLAP_WIDTH_RATIO,
    )

    # Collect per-image stats
    preds = result.object_prediction_list
    sahi_results.append({
        "image":      img_path.name,
        "n_dets":     len(preds),
        "categories": [p.category.name for p in preds],
        "scores":     [round(p.score.value, 4) for p in preds],
    })

    # Save visualisation
    result.export_visuals(export_dir=str(SAHI_OUTPUT_DIR), file_name=img_path.stem)

print(f"SAHI inference complete. Results saved to {SAHI_OUTPUT_DIR}")

# %%
sahi_df = pd.DataFrame(sahi_results)
sahi_csv = Path(PROJECT) / "sahi_test_predictions.csv"
sahi_df.to_csv(sahi_csv, index=False)

print(f"Per-image detection counts saved to {sahi_csv}")
print(sahi_df.describe())

# %% id="0mCZaslOc2Y7"
# Archive the run to Drive, per results/colab_project_structure.md's `runs/`
# convention. Two separate local sources get merged into one clean Drive
# folder, because Ultralytics resolved the relative project= to a different
# real directory (FINAL_RUN_DIR, e.g. /content/runs/obb/runs/optuna_obb/...)
# than the plain-Python writes below PROJECT itself (best_params.json,
# per-trial CSVs, fold_files/, sahi_predictions/) -- see the NOTE in
# objective() above.
import shutil
from pathlib import Path

DRIVE_RUNS_DIR = f"{PROJECT_ROOT}/runs/optuna_obb"
Path(DRIVE_RUNS_DIR).mkdir(parents=True, exist_ok=True)

shutil.copytree(PROJECT, DRIVE_RUNS_DIR, dirs_exist_ok=True)
shutil.copytree(FINAL_RUN_DIR, f"{DRIVE_RUNS_DIR}/best_final_all_m", dirs_exist_ok=True)
print(f"Archived {PROJECT} + {FINAL_RUN_DIR} -> {DRIVE_RUNS_DIR}")

# %% id="91ehgnBAPKkC"
# Resume path: if you're in a fresh runtime and just want to re-verify a
# previously archived run without retraining, restore it from Drive first
# (run the mount/constants cells above, then this one, skipping Optuna/training).
# Restores into a clean local path we control directly (PROJECT/best_final_all_m)
# rather than trying to recreate whatever quirky path Ultralytics originally
# resolved to -- there's no live model/trainer object to ask this time.
import shutil
from pathlib import Path

DRIVE_RUNS_DIR = f"{PROJECT_ROOT}/runs/optuna_obb"
FINAL_RUN_DIR = Path(PROJECT) / "best_final_all_m"
shutil.copytree(DRIVE_RUNS_DIR, PROJECT, dirs_exist_ok=True)
print(f"Restored {PROJECT} <- {DRIVE_RUNS_DIR}")

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

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
# # YOLO11m-OBB -- E-IRFS + K-fold final training, datasetv3
#
# K-fold variant of `notebooks/YOLO11mOBB_E-IRFS_optuna_datasetv3.py`. The
# Optuna HPO search itself is **not** re-run here -- this notebook loads the
# `best_params.json` produced by that notebook's search (from this session's
# local run, or from its Drive archive under `YOLO11m_E-IRFS_Optuna`) and
# replaces the single final train/val split with K-fold final training: one
# model is trained per fold with the same best hyperparameters, and every
# fold's model is evaluated on the same held-out test set, giving a mean +/-
# std test score instead of a single-split point estimate.
#
# The fold-construction *mechanics* (E-IRFS capped sampler, `make_folds_files`)
# are unchanged from the Optuna notebook -- kept byte-identical on purpose. If
# the template's fold-construction logic changes, re-sync this file the same
# way. One deliberate content difference: the *item pool* folded over is
# `train/` + `val/` pooled together, not `train/` alone. A K-fold split
# carves its own held-out validation portion out of every fold, so the
# dataset's fixed `val/` split -- which the Optuna notebook needed as final
# training's own validation -- serves no purpose here and would otherwise sit
# unused throughout every fold. This means fold membership (and fold count)
# here is *not* identical to the Optuna notebook's CV folds -- that's fine
# since only `best_params.json` (hyperparameters, not fold-conditioned
# results) is carried over from that search. `test/` is untouched throughout,
# as always.

# %% [markdown] id="KD8Ql8ZC23mh"
# ## file setup

# %%
# shapely/pillow/tqdm are for src/obb_eval.py + src/obb_common.py: this
# notebook now scores its test set with the SHARED cross-arm evaluator
# (exact polygon IoU via shapely), not only with Ultralytics' own
# metric. Same install line the other four arms carry.
# !pip3 install pyyaml ultralytics pandas numpy shapely pillow tqdm

# %% [markdown]
# Mounts Google Drive and points `PROJECT_ROOT` at the project folder, per
# `results/colab_project_structure.md` (same convention as
# `notebooks/00_setup_project_structure.ipynb`). Training itself still runs
# against the local Colab disk (`PROJECT` below) since Drive I/O would
# bottleneck the many small per-fold writes -- only the final archived
# results get copied to Drive, at the end of this notebook.

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

# %%
import yaml
import os
import numpy as np
from ultralytics.data.dataset import YOLODataset
import ultralytics.data.build as build
import random
from glob import glob
from pathlib import Path
from ultralytics import YOLO
import pandas as pd
import json
import sys
import time
import traceback

# The other four arms (Strip R-CNN, Oriented-DETR, RF-DETR-OBB, S2A-Net) all
# report `src/obb_eval.py`'s rotated-polygon-IoU AP. Ultralytics' own
# `model.val()` uses ProbIoU matching and its own AP integration, so the two
# differ by roughly the same 1-3 mAP as the architectural effect being
# measured -- putting them in one table compares metrics, not models. These
# imports let this notebook report BOTH: the shared metric (the comparable
# number) and Ultralytics' native one (kept under `ultra_*` column names for
# continuity with previously archived runs).
sys.path.insert(0, PROJECT_ROOT)
from src.obb_common import collect_items as _shared_collect_items  # noqa: E402
from src.obb_eval import (  # noqa: E402
    dump_predictions, evaluate_obb, load_ground_truth, precision_recall_f1,
    predict_ultralytics,
)
from src.obb_logging import (  # noqa: E402
    Timer, collect_epoch_log, fmt_dt, write_fold_composition,
    write_per_class_tables, write_run_env,
)

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
# notebook for the Phase 6.1 targeted re-run, repoint these at the
# grouped/dedup'd/stratified resplit dataset (data/dataset_resplit.yaml +
# Phase 4.1 background negatives) before running -- otherwise this recomputes
# E-IRFS frequencies on the pre-fix split, not the corrected one. Must match
# whatever split the source Optuna notebook used to produce best_params.json.
DATA_YAML  = "/content/data.yaml"      # base yaml (for nc/names). Can be anywhere.
IMAGES_DIR = "/content/data/train/images"         # flat images dir
LABELS_DIR = "/content/data/train/labels"         # flat OBB labels dir (.txt)
# K-fold carves its own held-out validation portion out of every fold, so a
# separate fixed val split serves no purpose here (unlike the single-split
# Optuna notebook, which used it as final training's own validation) --
# pooling it into the fold-able item set below puts those labeled images to
# use instead of leaving them untouched throughout every fold. TEST_IMAGES/
# TEST_LABELS remain the one split that must never enter this pool.
VAL_IMAGES_DIR = "/content/data/val/images"
VAL_LABELS_DIR = "/content/data/val/labels"
# The held-out test split, addressed as (images, labels) directories rather
# than only through DATA_YAML's `split="test"`. The shared evaluator scores
# the SAME (image, label) pairs the other four arms score -- built by
# src/obb_common.collect_items, exactly as they build theirs -- instead of
# whatever Ultralytics' own dataloader happens to enumerate.
TEST_IMAGES_DIR = "/content/data/test/images"
TEST_LABELS_DIR = "/content/data/test/labels"
PRETRAINED = "yolo11m-obb.pt"          # auto-downloads if online; or absolute path to local .pt

# --- Training knobs (you can keep defaults first run) ---
DEVICE      = 0            # GPU index or 'cpu'
# PROJECT MUST BE ABSOLUTE -- see the note in the Optuna notebook (Ultralytics
# re-roots a relative project= under {SETTINGS['runs_dir']}/{task}/, which
# split training output away from this notebook's own writes there). Kept
# in a separate directory from the Optuna notebook's PROJECT so K-fold final
# training never collides with that search's trial runs.
PROJECT     = "/content/runs/kfold_final_yolo11m"
SEED             = 42
K_FOLDS          = 9
SHUFFLE_BEFORE_SPLIT = True
FINAL_PATIENCE   = 50   # early-stopping patience for each fold's final training

# Where to find the archived Optuna run's best_params.json if this session
# doesn't already have one locally (see load_best_params() below).
OPTUNA_PROJECT_LOCAL = "/content/runs/optuna_obb_yolo11m"
OPTUNA_RUN_LABEL = "YOLO11m_E-IRFS_Optuna2"
OPTUNA_FINAL_SUBDIR = "best_final_yolo11m_obb"

# Repro + folders
random.seed(SEED)
assert Path(PROJECT).is_absolute(), (
    f"PROJECT must be an absolute path, got {PROJECT!r} -- a relative one gets "
    f"re-rooted by Ultralytics and splits training output away from this "
    f"notebook's own writes (see the note above)."
)
Path(PROJECT).mkdir(parents=True, exist_ok=True)

# Best-effort run metadata -- platform, GPU, framework versions, git commit,
# and this arm's own control-variable constants. Never blocks training on
# failure; see src/obb_logging.py's module docstring.
write_run_env(Path(PROJECT) / "run_env.json", repo_root=PROJECT_ROOT, extra={
    "arm": "YOLO11m_E-IRFS_KFold_Final", "SEED": SEED, "K_FOLDS": K_FOLDS,
    "TRAIN_EPOCHS": TRAIN_EPOCHS, "EIRFS_THRESHOLD": EIRFS_THRESHOLD,
    "PROJECT": PROJECT, "PRETRAINED": PRETRAINED,
})


# %% [markdown] id="CFjedlSS3c4o"
# ### class definition

# %% id="irfv0ATc2yNV"
# used
# Find the image file that corresponds to a given label file, inside
# images_dir. It tries all known image extensions (e.g. .jpg, .png, .jpeg)
# and returns the first match it finds. Checks both the usual case and the
# all-caps variant (e.g. ".jpg" and ".JPG") -- on a case-INsensitive
# filesystem (default macOS) an exact-case check alone already matches
# either, which is why a lowercase-only version of this went unnoticed
# locally; on a case-sensitive one (Colab/Linux, where training actually
# runs) it silently drops every upper-case-extension image -- confirmed on
# this dataset to be ~27% of every split (datasetv3_2 mixes ".jpg" and
# ".JPG" filenames).
def find_image_for_label(label_path: Path, images_dir) -> None:
    stem = label_path.stem
    for ext in IMG_EXTS:
        for candidate_ext in (ext, ext.upper()):
            p = Path(images_dir) / f"{stem}{candidate_ext}"
            if p.exists():
                return p.resolve()

# used
# Collect all (image, label) pairs under one images_dir/labels_dir.
# Only labels that have a matching image are included.
# This produces the list of items that will later be split into folds.
def collect_items(images_dir, labels_dir) -> list:
    label_files = sorted(Path(labels_dir).glob("*.txt"))
    items = []
    for lf in label_files:
        img = find_image_for_label(lf, images_dir)
        if img is not None:
            items.append((str(img), str(lf.resolve())))
    return items

# used
# Assign each dataset item to one of K folds.
# This returns a list "fold_of" such that:
#   fold_of[i] = fold index (0..K-1) for item i
# If shuffle=True, items are randomly permuted before assignment.
def kfold_assignments(n_items: int, k: int, shuffle: bool=True, seed: int=None) -> list:
    idxs = list(range(n_items))
    if shuffle:
        # Reseed immediately before shuffling rather than relying on the
        # module-level random.seed(SEED) run far earlier in the config cell.
        # Same folds on a clean top-to-bottom run, but it makes the fold
        # assignment independent of how much other `random` consumption
        # happened in between -- notably, re-running THIS cell alone in Colab
        # (the normal interactive workflow) without re-running the config cell
        # would otherwise hand out a different split than the other arms got.
        # src/obb_common.kfold_assignments() -- which every non-YOLO arm uses
        # -- reseeds for exactly this reason; matching it here is what keeps
        # all six arms on byte-identical folds.
        random.seed(SEED if seed is None else seed)
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
    # The LAST row is the last epoch trained, which after early stopping is
    # FINAL_PATIENCE epochs past the checkpoint actually saved as best.pt --
    # and best.pt is what the test evaluation below loads. Reporting the last
    # epoch's val score beside best.pt's test score describes two different
    # checkpoints. Take the best row instead.
    #
    # Caveat, stated rather than glossed: Ultralytics selects best.pt on a
    # fitness blend (0.1*mAP50 + 0.9*mAP50-95), not on mAP50-95 alone, so the
    # argmax below is a close proxy for best.pt's epoch, not provably the same
    # row. That formula was not re-verified against the installed package
    # here. This is a diagnostic column only -- every reported result comes
    # from the test-set block below.
    def _pick(col):
        return float(df[col].max())

    # Known column names used by different YOLO versions
    candidates = [
        "metrics/mAP50-95(B)", "metrics/mAP50-95", "map50-95", "mAP50-95(B)", "metrics/mAP50-95(all)"
    ]
    for c in candidates:
        if c in df.columns:
            return _pick(c)

    # Fallback: find any column that looks like mAP50-95
    for c in df.columns:
        if "mAP50-95" in c or "map50-95" in c.lower():
            return _pick(c)
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
#   - caches per-fold class-weight computation so repeated trials on
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
# Load best_params.json from the sibling Optuna notebook's search, without
# re-running Optuna here. Checks this session's own local run first (if you
# already ran the Optuna notebook earlier in this same runtime), then falls
# back to the archived copy on Drive under OPTUNA_RUN_LABEL (see that
# notebook's "archive to Drive" cell -- RUN_LABEL="YOLO11m_E-IRFS_Optuna").
def load_best_params() -> dict:
    optuna_basename = Path(OPTUNA_PROJECT_LOCAL).name  # "optuna_obb_yolo11m"
    candidates = [
        # 1. This session's own local Optuna run (same runtime, not yet archived).
        Path(OPTUNA_PROJECT_LOCAL) / OPTUNA_FINAL_SUBDIR / "best_params.json",
        # 2. What the Optuna notebook's own "Archive to Drive" cell actually
        #    produces: it copytree()s the whole local /content/runs into
        #    Drive under RUN_LABEL, so PROJECT's basename survives as a
        #    nested subdirectory.
        Path(f"{PROJECT_ROOT}/runs/{OPTUNA_RUN_LABEL}/{optuna_basename}/{OPTUNA_FINAL_SUBDIR}/best_params.json"),
        # 3. A flat archive directly under RUN_LABEL (in case only that
        #    final-run subfolder was copied, not the whole /content/runs tree).
        Path(f"{PROJECT_ROOT}/runs/{OPTUNA_RUN_LABEL}/{OPTUNA_FINAL_SUBDIR}/best_params.json"),
        # 4. The local PROJECT folder copied straight to Drive as its own
        #    top-level dir (e.g. via the Colab file browser, or a Drive sync
        #    that bypassed the archive cell) -- no RUN_LABEL folder at all.
        Path(f"{PROJECT_ROOT}/runs/{optuna_basename}/{OPTUNA_FINAL_SUBDIR}/best_params.json"),
    ]

    for path in candidates:
        if path.exists():
            print(f"Loading best_params from: {path}")
            return json.loads(path.read_text())

    tried = "\n  ".join(str(p) for p in candidates)
    raise FileNotFoundError(
        f"No best_params.json found in any of:\n  {tried}\n"
        f"Run notebooks/YOLO11mOBB_E-IRFS_optuna_datasetv3.py's Optuna search "
        f"first, then either let its 'archive to Drive' cell run (RUN_LABEL="
        f"{OPTUNA_RUN_LABEL!r}), or copy its best_params.json to Drive "
        f"yourself under one of the paths above. If you archived more than "
        f"once, _unique_drive_dir() may have named it "
        f"'{OPTUNA_RUN_LABEL}2', '{OPTUNA_RUN_LABEL}3', etc. -- point "
        f"OPTUNA_RUN_LABEL at the archive you want."
    )


# %% [markdown] id="y4lQ9y8M32u4"
# ### Pipeline

# %%
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
from ultralytics.data.dataset import YOLODataset as _StockYOLODataset
build.YOLODataset = YOLOEIRFSCappedDataset

# %% [markdown]
# ## Load best hyperparameters
#
# No Optuna search runs in this notebook -- see the header markdown. This
# just loads the previous search's winning params so K-fold final training
# below uses the same hyperparameters the single-split notebook would have.

# %%
best_params = load_best_params()
print("Using best_params:", json.dumps(best_params, indent=2))

# %% [markdown] id="mYvx4N1z-koV"
# ## Folds

# %%
#check
print("Train images:", len(glob(f"{IMAGES_DIR}/*")), " Train labels:", len(glob(f"{LABELS_DIR}/*.txt")))
print("Val images  :", len(glob(f"{VAL_IMAGES_DIR}/*")), " Val labels  :", len(glob(f"{VAL_LABELS_DIR}/*.txt")))

# %%
# Pool train + val into one fold-able set -- see the note by VAL_IMAGES_DIR
# above for why the fixed val split doesn't need to stay separate here.
items = collect_items(IMAGES_DIR, LABELS_DIR) + collect_items(VAL_IMAGES_DIR, VAL_LABELS_DIR)
print(f"Found {len(items)} labeled images (train + val pooled).")
fold_files_dir = Path(PROJECT) / "fold_files"
fold_specs = make_folds_files(items, K_FOLDS, DATA_YAML, fold_files_dir)
# Peek
fold_specs[:2], os.path.exists(fold_specs[0]["data_yaml"])

# %%
# Per-(fold, split, class) image/instance counts, for auditing class balance
# across folds (e.g. which folds have zero Rhizoctonia GT -- evaluate_obb()
# silently excludes such classes from that fold's mean). Recomputes the same
# fold_of assignment make_folds_files() used internally, over the same items.
_fold_of_for_composition = kfold_assignments(len(items), K_FOLDS, SHUFFLE_BEFORE_SPLIT)
_folds_for_composition = [{
    "fold": k,
    "train": [items[i] for i in range(len(items)) if _fold_of_for_composition[i] != k],
    "val": [items[i] for i in range(len(items)) if _fold_of_for_composition[i] == k],
} for k in range(K_FOLDS)]

# %% [markdown]
# ## K-fold final training
#
# Trains K models -- one per fold above -- each for the full `TRAIN_EPOCHS`
# using `best_params`, then evaluates every fold's model on the same
# held-out test set (`DATA_YAML`, `split="test"`). This replaces the
# Optuna notebook's single final train/val run with a mean +/- std test
# score across folds, which is more robust to reviewer questions about a
# single lucky/unlucky split. The fold with the best test mAP50-95 is kept
# as the representative "best" model for downstream use.

# %%
KFOLD_FINAL_DIR = "kfold_final_yolo11m_obb"
kfold_root = Path(PROJECT) / KFOLD_FINAL_DIR
kfold_root.mkdir(parents=True, exist_ok=True)

# --- shared-evaluator setup -------------------------------------------------
# Every arm in this comparison must score the SAME test images with the SAME
# code. `_shared_collect_items` is src/obb_common.collect_items -- the exact
# function the other four arms use -- so the (image, label) pairs scored here
# are the ones they score, rather than whatever Ultralytics' own dataloader
# happens to enumerate from DATA_YAML.
_base_yaml = load_base_yaml(DATA_YAML)
_names = _base_yaml.get("names")
CLASS_NAMES = ([_names[i] for i in sorted(_names)] if isinstance(_names, dict)
               else list(_names))
NUM_CLASSES = len(CLASS_NAMES)

write_fold_composition(Path(PROJECT) / "fold_composition.csv",
                       _folds_for_composition, CLASS_NAMES, NUM_CLASSES)

test_items = _shared_collect_items(TEST_IMAGES_DIR, TEST_LABELS_DIR)
if not test_items:
    raise RuntimeError(
        f"no labelled test images under {TEST_IMAGES_DIR} / {TEST_LABELS_DIR} -- "
        f"the shared evaluator has nothing to score."
    )
print(f"Test split: {len(test_items)} labelled images, {NUM_CLASSES} classes "
      f"(scored by src/obb_eval.py, the shared cross-arm metric).")
shared_gt = load_ground_truth(test_items)

fold_rows = []
fold_errors = []
per_fold_per_class = []
for k, spec in enumerate(fold_specs):
    started_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    try:
        run_name = f"{KFOLD_FINAL_DIR}_fold{k}"
        model = YOLO(PRETRAINED)

        final_overrides = dict(
            task="obb",
            data=spec["data_yaml"],           # fold-specific yaml (absolute paths)
            epochs=TRAIN_EPOCHS,
            patience=FINAL_PATIENCE,
            imgsz=best_params.get("imgsz", 1024),
            batch=best_params.get("batch", 16),
            device=DEVICE,
            project=PROJECT,
            name=run_name,
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

        with Timer() as train_timer:
            model.train(**final_overrides)
        # Read the trainer's own resolved save_dir rather than reconstructing
        # Path(PROJECT)/run_name -- see the NOTE in the Optuna notebook's
        # objective(): a relative project= doesn't end up where you'd naively
        # expect, though PROJECT here is absolute so this is belt-and-braces.
        fold_run_dir = Path(model.trainer.save_dir)
        val_map = read_final_map_from_results(fold_run_dir)

        (fold_run_dir / "resolved_params.json").write_text(
            json.dumps(final_overrides, indent=2, default=str))
        collect_epoch_log(fold_run_dir, "ultralytics")

        # Restore the stock dataset class before test-set eval -- E-IRFS
        # sampling weights are a training-time concern only (see the note in
        # the Optuna notebook's own test cell for why this matters).
        build.YOLODataset = _StockYOLODataset
        eval_model = YOLO(fold_run_dir / "weights" / "best.pt")
        # Ultralytics' OWN metric (ProbIoU matching, its own AP integration).
        # Kept for its plots -- confusion matrix, PR curves -- and for continuity
        # with runs archived before this notebook moved onto the shared
        # evaluator, but recorded under `ultra_*` column names because it is NOT
        # the number to put beside the other arms'. plots=True + an explicit
        # project/name so every fold's plots land in a findable, per-fold
        # directory under PROJECT instead of an anonymously-numbered
        # runs/obb/valN next to it.
        ultra_metrics = eval_model.val(
            data=DATA_YAML, split="test", task="obb", plots=True,
            project=PROJECT, name=f"{KFOLD_FINAL_DIR}_fold{k}_test_eval", exist_ok=True,
        )
        ultra_map = float(ultra_metrics.box.map)      # mAP50-95, Ultralytics' metric
        ultra_map50 = float(ultra_metrics.box.map50)

        # THE cross-arm comparable number: same weights, same test images,
        # scored by the same src/obb_eval.py code as Strip R-CNN,
        # Oriented-DETR, RF-DETR-OBB and S2A-Net.
        #   * conf=0.001 is the operating point all five other arms evaluate at
        #     -- AP integrates the whole precision/recall curve, so a higher
        #     threshold truncates it and understates AP.
        #   * imgsz is passed EXPLICITLY rather than left to Ultralytics'
        #     restore-imgsz-from-checkpoint behaviour, so train and test
        #     resolution are pinned together here exactly as they are in every
        #     other arm (the B1 resolution confound).
        #   * precision/recall/F1 come from src/obb_eval.precision_recall_f1,
        #     which reproduces Ultralytics' own max-mean-F1 operating point --
        #     so those columns stay comparable to the other arms' too, not just
        #     the mAPs.
        with Timer() as eval_timer:
            records = predict_ultralytics(
                fold_run_dir / "weights" / "best.pt", test_items,
                imgsz=best_params.get("imgsz", 1024), conf=0.001, iou=0.7, device=DEVICE,
            )
            dump_predictions(fold_run_dir / "test_predictions.json", records)
            metrics = evaluate_obb(shared_gt, records, NUM_CLASSES, CLASS_NAMES)
            prf = precision_recall_f1(shared_gt, records, NUM_CLASSES, class_names=CLASS_NAMES)
        metrics["precision"], metrics["recall"], metrics["F1"] = (
            prf["precision"], prf["recall"], prf["F1"]
        )
        # Merge P/R/F1 + confusion counts into the same per-class dict AP
        # already lives in, so kfold_per_class_results.csv has one row per
        # (fold, class) with every metric this arm computes.
        for _name, _pc in prf.get("per_class", {}).items():
            metrics["per_class"].setdefault(_name, {}).update(_pc)
        for _name, _c in prf.get("counts", {}).items():
            metrics["per_class"].setdefault(_name, {}).update(_c)
        per_fold_per_class.append((k, metrics["per_class"]))
        (fold_run_dir / "test_metrics.json").write_text(json.dumps(metrics, indent=2))
        build.YOLODataset = YOLOEIRFSCappedDataset  # re-enable for the next fold's training

        epoch_summary_path = fold_run_dir / "epoch_log_summary.json"
        epoch_summary = (json.loads(epoch_summary_path.read_text())
                         if epoch_summary_path.exists() else {})

        fold_rows.append({
            "fold": k,
            "val_mAP50-95": val_map,
            # Shared evaluator (src/obb_eval.py) -- the comparable columns.
            "test_mAP50-95": metrics["mAP50-95"],
            "test_mAP50": metrics["mAP50"],
            "test_mAP75": metrics["mAP75"],
            "test_precision": prf["precision"],
            "test_recall": prf["recall"],
            "test_F1": prf["F1"],
            # Ultralytics' native metric -- diagnostic/continuity only, NOT
            # comparable across arms (see the note above).
            "ultra_test_mAP50-95": ultra_map,
            "ultra_test_mAP50": ultra_map50,
            "run_dir": str(fold_run_dir),
            "error": None,
            "started_at": started_at,
            "finished_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "train_seconds": train_timer.seconds,
            "eval_seconds": eval_timer.seconds,
            "epochs_completed": epoch_summary.get("epochs_completed"),
            "final_train_loss": epoch_summary.get("final_loss"),
        })
        print(f"[Fold {k}] test mAP50-95={metrics['mAP50-95']:.4f}  "
              f"mAP50={metrics['mAP50']:.4f}  mAP75={metrics['mAP75']:.4f}  "
              f"P={prf['precision']:.4f}  R={prf['recall']:.4f}  F1={prf['F1']:.4f}"
              f"  | ultralytics mAP50-95={ultra_map:.4f}, val {val_map:.4f}"
              f"  | train {fmt_dt(train_timer.seconds or 0)}")
    except Exception as e:
        # One fold's failure must not take down the whole K-fold run and
        # leave no CSV behind -- the same guard Strip R-CNN, RF-DETR-OBB and
        # S2A-Net already carry, and which the all-NaN check below already
        # assumed was here. Re-enable E-IRFS first: if the exception came out
        # of training, the stock class may still be swapped in (or not), and
        # the NEXT fold must train with E-IRFS active either way.
        build.YOLODataset = YOLOEIRFSCappedDataset
        print(f"\n{'!'*70}\n[Fold {k}] FAILED: {e}\n{'!'*70}")
        traceback.print_exc(file=sys.stdout)
        sys.stdout.flush()
        fold_errors.append(f"fold {k}: {type(e).__name__}: {e}")
        fold_rows.append({
            "fold": k, "val_mAP50-95": None,
            "test_mAP50-95": None, "test_mAP50": None, "test_mAP75": None,
            "test_precision": None, "test_recall": None, "test_F1": None,
            "ultra_test_mAP50-95": None, "ultra_test_mAP50": None,
            "run_dir": None, "error": str(e),
            "started_at": started_at, "finished_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "train_seconds": None, "eval_seconds": None,
            "epochs_completed": None, "final_train_loss": None,
        })
        continue

if fold_errors:
    print(f"WARNING: {len(fold_errors)}/{len(fold_specs)} folds failed: "
          f"{'; '.join(fold_errors)}")
fold_df = pd.DataFrame(fold_rows)
fold_df.to_csv(kfold_root / "kfold_final_results.csv", index=False)
print(fold_df)
write_per_class_tables(kfold_root, per_fold_per_class, CLASS_NAMES)

# %%
if fold_df["test_mAP50-95"].isna().all():
    raise RuntimeError(
        "Every fold's test mAP50-95 came back NaN -- scroll up for per-fold "
        "training/eval errors. kfold_final_summary.json NOT written."
    )

summary = {
    "n_folds": int(len(fold_df)),
    "test_mAP50-95_mean": float(fold_df["test_mAP50-95"].mean()),
    "test_mAP50-95_std": float(fold_df["test_mAP50-95"].std(ddof=0)),
    "test_mAP50_mean": float(fold_df["test_mAP50"].mean()),
    "test_mAP50_std": float(fold_df["test_mAP50"].std(ddof=0)),
    "test_mAP75_mean": float(fold_df["test_mAP75"].mean()),
    "test_mAP75_std": float(fold_df["test_mAP75"].std(ddof=0)),
    "test_precision_mean": float(fold_df["test_precision"].mean()),
    "test_precision_std": float(fold_df["test_precision"].std(ddof=0)),
    "test_recall_mean": float(fold_df["test_recall"].mean()),
    "test_recall_std": float(fold_df["test_recall"].std(ddof=0)),
    "test_F1_mean": float(fold_df["test_F1"].mean()),
    "test_F1_std": float(fold_df["test_F1"].std(ddof=0)),
    "val_mAP50-95_mean": float(fold_df["val_mAP50-95"].mean()),
    "val_mAP50-95_std": float(fold_df["val_mAP50-95"].std(ddof=0)),
    # Ultralytics' own metric, kept alongside for continuity. Do NOT report
    # this beside another architecture's mAP -- the `test_*` keys above are
    # the cross-arm comparable ones.
    "ultra_test_mAP50-95_mean": float(fold_df["ultra_test_mAP50-95"].mean()),
    "ultra_test_mAP50-95_std": float(fold_df["ultra_test_mAP50-95"].std(ddof=0)),
    "metric": "src/obb_eval.py evaluate_obb (polygon-IoU COCO AP50:5:95)",
    "n_folds_failed": int(len(fold_errors)),
    # NOTE: `best_fold` peeks at every fold's TEST score to pick which
    # checkpoint gets archived, so it is an optimistically biased number.
    # The unbiased result to report is test_mAP50-95_mean +/- _std.
    "best_fold": int(fold_df.loc[fold_df["test_mAP50-95"].idxmax(), "fold"]),
    "best_params": best_params,
}
print("K-fold final test summary:", json.dumps(summary, indent=2))
(kfold_root / "kfold_final_summary.json").write_text(json.dumps(summary, indent=2))

FINAL_RUN_DIR = Path(fold_df.loc[fold_df["test_mAP50-95"].idxmax(), "run_dir"])
print(f"Best fold: {summary['best_fold']} -> {FINAL_RUN_DIR}")

# %% [markdown] id="erKqaKk7plmI"
# ## test
#
# Re-prints the held-out test metrics for the best fold (already computed
# during the K-fold loop above) plus full per-class breakdown via `plots=True`.

# %%
from ultralytics import YOLO

# Restore the stock dataset class before evaluating (harmless no-op if
# already restored above).
build.YOLODataset = _StockYOLODataset

# Load the best fold's trained OBB model.
model = YOLO(FINAL_RUN_DIR / "weights" / "best.pt")

# Run TEST set evaluation
metrics = model.val(
    data=DATA_YAML,
    split="test",   # use test set (not validation)
    task="obb",      # OBB (rotated boxes)
    plots=True,
    project=PROJECT, name="best_fold_test_eval", exist_ok=True,
)

# Ultralytics' own numbers -- printed for its per-class table and plots,
# and labelled as such. They are NOT the figures to quote beside another
# architecture: `model.val()` matches with ProbIoU and integrates AP its own
# way, differing from src/obb_eval.py by roughly the same margin as the
# architectural effect under study.
print(metrics)
print(f"\n[Ultralytics' own metric -- NOT cross-arm comparable]"
      f"\n  mAP50-95={float(metrics.box.map):.4f}  mAP50={float(metrics.box.map50):.4f}  "
      f"precision={float(metrics.box.mp):.4f}  recall={float(metrics.box.mr):.4f}")

# The figures to report: written during this fold's K-fold iteration by the
# shared evaluator (same code, same test images, same conf=0.001 operating
# point as Strip R-CNN / Oriented-DETR / RF-DETR-OBB / S2A-Net).
_shared_path = FINAL_RUN_DIR / "test_metrics.json"
if _shared_path.exists():
    _shared = json.loads(_shared_path.read_text())
    print(f"\n[src/obb_eval.py -- THE cross-arm comparable metric]"
          f"\n  mAP50-95={_shared['mAP50-95']:.4f}  mAP50={_shared['mAP50']:.4f}  "
          f"mAP75={_shared['mAP75']:.4f}  precision={_shared['precision']:.4f}  "
          f"recall={_shared['recall']:.4f}  F1={_shared['F1']:.4f}")
    print("  per-class (shared metric):")
    for _name, _m in _shared["per_class"].items():
        print(f"    {_name:15} AP50-95={_m['AP50-95']:.4f}  AP50={_m['AP50']:.4f}  "
              f"n={_m['n_gt']}")
else:
    print(f"\nWARNING: {_shared_path} not found -- this fold predates the shared "
          f"evaluator, so only Ultralytics' own (non-comparable) numbers are "
          f"available. Re-run the K-fold loop, or rescore the checkpoint with "
          f"analysis/comparison/rescore_baseline_yolo.py.")
print("\nFull K-fold summary (all folds):", json.dumps(summary, indent=2))

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
RUN_LABEL = "YOLO11m_E-IRFS_KFold_Final"

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
# cells above and skip the K-fold training loop entirely.

# %% id="91ehgnBAPKkC"
# Restores the whole /content/runs tree from its RUN_LABEL folder on Drive,
# then reads kfold_final_summary.json to find which fold was best.
import shutil
from pathlib import Path

RUN_LABEL = "YOLO11m_E-IRFS_KFold_Final"
LOCAL_RUNS = Path("/content/runs")
DRIVE_DEST = Path(f"{PROJECT_ROOT}/runs/{RUN_LABEL}")

LOCAL_RUNS.mkdir(parents=True, exist_ok=True)
shutil.copytree(DRIVE_DEST, LOCAL_RUNS, dirs_exist_ok=True)
print(f"Restored {LOCAL_RUNS} <- {DRIVE_DEST}")

summary = json.loads((Path(PROJECT) / KFOLD_FINAL_DIR / "kfold_final_summary.json").read_text())
fold_df = pd.read_csv(Path(PROJECT) / KFOLD_FINAL_DIR / "kfold_final_results.csv")
FINAL_RUN_DIR = Path(fold_df.loc[fold_df["test_mAP50-95"].idxmax(), "run_dir"])
print(f"Best fold: {summary['best_fold']} -> {FINAL_RUN_DIR}")

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
    plots=True,
    project=PROJECT, name="best_fold_test_eval", exist_ok=True,
)

# Ultralytics' own numbers -- printed for its per-class table and plots,
# and labelled as such. They are NOT the figures to quote beside another
# architecture: `model.val()` matches with ProbIoU and integrates AP its own
# way, differing from src/obb_eval.py by roughly the same margin as the
# architectural effect under study.
print(metrics)
print(f"\n[Ultralytics' own metric -- NOT cross-arm comparable]"
      f"\n  mAP50-95={float(metrics.box.map):.4f}  mAP50={float(metrics.box.map50):.4f}  "
      f"precision={float(metrics.box.mp):.4f}  recall={float(metrics.box.mr):.4f}")

# The figures to report: written during this fold's K-fold iteration by the
# shared evaluator (same code, same test images, same conf=0.001 operating
# point as Strip R-CNN / Oriented-DETR / RF-DETR-OBB / S2A-Net).
_shared_path = FINAL_RUN_DIR / "test_metrics.json"
if _shared_path.exists():
    _shared = json.loads(_shared_path.read_text())
    print(f"\n[src/obb_eval.py -- THE cross-arm comparable metric]"
          f"\n  mAP50-95={_shared['mAP50-95']:.4f}  mAP50={_shared['mAP50']:.4f}  "
          f"mAP75={_shared['mAP75']:.4f}  precision={_shared['precision']:.4f}  "
          f"recall={_shared['recall']:.4f}  F1={_shared['F1']:.4f}")
    print("  per-class (shared metric):")
    for _name, _m in _shared["per_class"].items():
        print(f"    {_name:15} AP50-95={_m['AP50-95']:.4f}  AP50={_m['AP50']:.4f}  "
              f"n={_m['n_gt']}")
else:
    print(f"\nWARNING: {_shared_path} not found -- this fold predates the shared "
          f"evaluator, so only Ultralytics' own (non-comparable) numbers are "
          f"available. Re-run the K-fold loop, or rescore the checkpoint with "
          f"analysis/comparison/rescore_baseline_yolo.py.")
print("\nFull K-fold summary (all folds):", json.dumps(summary, indent=2))

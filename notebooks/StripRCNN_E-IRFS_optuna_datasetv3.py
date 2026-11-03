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

# %% [markdown]
# # Strip R-CNN (AAAI 2026) -- E-IRFS + Optuna, datasetv3
#
# Architecture arm 2 of 4 in the OBB comparison. Structurally parallel to
# `notebooks/Template_Clean_E-IRFS_optuna_datasetv3.py` (YOLO11m-OBB): same
# folds, same E-IRFS sampler, same Optuna budget, same final-train protocol,
# same held-out metric. Everything that differs between the two notebooks
# differs because the *framework* forces it, and every such difference is
# enumerated in `results/obb_model_comparison.md`.
#
# **Read that document before reporting any number from this notebook.** Two
# differences in particular are not cosmetic:
#
# 1. Strip R-CNN is a **two-stage** detector on **MMRotate 0.x**. It has no
#    counterpart to YOLO's `box`/`cls`/`dfl` loss weights, no mosaic/mixup, and
#    no HSV augmentation. Those axes of the YOLO search space are dropped here
#    rather than faked, so the two searches explore spaces of different
#    dimensionality (17 vs 12 tunable params).
# 2. MMRotate's own DOTA evaluation reports **VOC-style mAP at IoU 0.5 only**.
#    It is not comparable to Ultralytics' mAP50-95. This notebook therefore
#    ignores MMRotate's metric entirely and scores predictions with
#    `src/obb_eval.py`, the single shared rotated-IoU implementation used by
#    all four arms.
#
# ## Environment warning -- read before running
#
# Strip R-CNN targets **MMRotate 0.x**, which needs `mmcv-full` (not `mmcv>=2`),
# `mmdet` 2.x, and PyTorch <= 1.13 on Python <= 3.10. Recent Colab runtimes ship
# Python 3.11+/torch 2.x, where `mmcv-full` has no wheel and its source build
# fails. If the install cell below fails, you need a runtime with an older
# Python -- this is a property of the upstream project, not of this notebook.
# There is no version of this comparison in which Strip R-CNN runs in the same
# environment as RF-DETR; they are mutually exclusive dependency sets.

# %% [markdown]
# ## file setup

# %%
# This notebook's own Colab kernel installs only what it needs to run
# framework-free code (Optuna, the shared src/obb_eval.py evaluator, etc.).
# The MMRotate 0.x stack itself (torch<=1.13, Python<=3.10) is NOT installed
# here -- see the next cell for why and how.
# tqdm and pillow are both genuinely used here (progress bars; src/obb_common.py's
# image_size()/PNG conversion) -- listed explicitly even though Colab's default
# runtime already ships both, matching this project's convention of not relying
# on implicit/preinstalled availability (see numpy/pandas below, also preinstalled).
# !pip3 install -q optuna pandas numpy pyyaml shapely pillow tqdm

# %%
# MMRotate 0.x pins (mmcv-full, mmdet 2.x, torch<=1.13) require Python<=3.10.
# Modern Colab's default runtime now ships Python 3.11+ (3.13 as of writing),
# for which no torch==1.13.1 wheel exists at all -- `pip install torch==1.13.1`
# fails outright, and even `mim` itself crashes on 3.12+ (pkgutil.ImpImporter
# was removed from the stdlib, which old setuptools/pkg_resources still call).
# There is no way to `pip install` around this on the notebook's own kernel.
#
# Fix: install Miniconda and build a SEPARATE Python 3.10 environment inside
# it, and run all of Strip R-CNN's actual training/inference through THAT
# environment via subprocess (see src/strip_rcnn_worker.py) -- this
# notebook's own kernel never imports mmcv/mmdet/mmrotate directly. Re-running
# this cell in an already-set-up runtime just re-installs into the same env
# (mostly a fast no-op via pip/conda's own caching), it does not need to be
# guarded like the Oriented-DETR ops-build cell.
# !wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
# !bash /tmp/miniconda.sh -b -p /content/miniconda -u
# Newer Anaconda requires explicit Terms-of-Service acceptance for the
# default channels before `conda create` will run non-interactively --
# without this, `conda create` fails with CondaToSNonInteractiveError, the
# strip_rcnn env is never actually created, and every cell below it then
# fails with "EnvironmentLocationNotFound: Not a conda environment", which
# only surfaces much later as a bare FileNotFoundError when training tries
# to launch the conda-env subprocess. Idempotent -- a no-op if already accepted.
# !/content/miniconda/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
# !/content/miniconda/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
# !/content/miniconda/bin/conda create -y -n strip_rcnn python=3.10
# !/content/miniconda/bin/conda run -n strip_rcnn pip install -q torch==1.13.1 torchvision==0.14.1 --index-url https://download.pytorch.org/whl/cu117
# !/content/miniconda/bin/conda run -n strip_rcnn pip install -q -U openmim
# !/content/miniconda/bin/conda run -n strip_rcnn mim install -q "mmcv-full==1.7.2"
# !/content/miniconda/bin/conda run -n strip_rcnn mim install -q "mmdet==2.28.2"
# mmcv-full 1.7.2 (torch 1.13.1-era) ships compiled C extensions built
# against NumPy 1.x's ABI -- an unpinned `pip install numpy` here pulls
# NumPy 2.x, which torchvision already warns about at import time ("A
# module that was compiled using NumPy 1.x cannot be run in NumPy 2.2.6...
# it may crash"). Still a warning today, not yet a crash, but this is
# exactly the kind of silent ABI mismatch that can corrupt array data or
# hard-crash inside a C extension call later -- pinned before it becomes one.
# !/content/miniconda/bin/conda run -n strip_rcnn pip install -q "numpy<2" pyyaml

# %%
# Strip R-CNN is a fork of MMRotate, installed in place of it (not alongside)
# -- into the strip_rcnn conda env, not this notebook's own kernel.
# !git clone https://github.com/HVision-NKU/Strip-R-CNN.git /content/Strip-R-CNN
# !/content/miniconda/bin/conda run -n strip_rcnn pip install -q -v -e /content/Strip-R-CNN
# Another real gap in the repo's own packaging (confirmed against its
# requirements/runtime.txt, which `pip install -e .` installs from): its
# StripNet backbone (mmrotate/models/backbones/stripnet.py) imports
# `timm.models.layers` directly, but timm is not listed as a dependency
# anywhere in the repo, so the editable install above never installs it --
# ModuleNotFoundError: No module named 'timm' the first time StripNet is
# actually built (i.e. as soon as training starts, not at import time above).
# !/content/miniconda/bin/conda run -n strip_rcnn pip install -q timm
# The earlier `numpy<2` pin above does NOT survive these last two installs --
# confirmed live: `RuntimeError: Numpy is not available` from
# torch.from_numpy() the first time a DataLoader worker actually ran (i.e.
# once training got far enough to load real data), the exact failure mode
# of that earlier "compiled using NumPy 1.x... may crash" warning becoming
# fatal. Root cause: `pip install -e /content/Strip-R-CNN` (its own
# requirements/runtime.txt lists a bare, unconstrained `numpy`) and/or
# `pip install timm` each resolve their own dependency graph independently
# of the earlier `numpy<2` pip invocation -- with no memory of that
# constraint, either one can silently pull numpy back up to 2.x if its own
# resolution prefers a newer version. Re-asserted as the actual LAST
# install step instead of trying to pin down exactly which of the two did
# it -- whatever happens in between, this guarantees the final state.
# !/content/miniconda/bin/conda run -n strip_rcnn pip install -q "numpy<2"

# %%
# Patch a real bug in the Strip-R-CNN repo's own HEAD (confirmed by cloning
# it directly): mmrotate/datasets/__init__.py imports DOTAv15Dataset and
# DOTAv2Dataset from .dota, but mmrotate/datasets/dota.py only defines
# DOTADataset -- neither of those two names exists anywhere in the repo (a
# separate, differently-named DOTADataset15 lives in dota_1_5.py instead,
# already imported correctly on the line below). This ImportError blocks
# every `from mmrotate.datasets import build_dataset` the worker script
# does, before any of this project's own code runs -- and since
# /content/Strip-R-CNN is a fresh clone every runtime, it must be
# re-patched every runtime too. Only DOTADataset is used here
# (build_config() sets dataset_type = "DOTADataset"), so the missing names
# are dropped rather than stubbed. Idempotent -- a no-op if already patched.
from pathlib import Path as _Path

def _patch_mmrotate_datasets_init():
    init_path = _Path("/content/Strip-R-CNN/mmrotate/datasets/__init__.py")
    src = init_path.read_text()
    patched = (
        src.replace(
            "from .dota import DOTADataset,DOTAv15Dataset, DOTAv2Dataset  # noqa: F401, F403",
            "from .dota import DOTADataset  # noqa: F401, F403",
        )
        .replace(
            "__all__ = ['SARDataset','DIORDataset', 'DOTADataset', 'DOTAv15Dataset', 'DOTAv2Dataset', "
            "'build_dataset', 'HRSCDataset', 'FairDataset', 'DOTADataset15']",
            "__all__ = ['SARDataset','DIORDataset', 'DOTADataset', "
            "'build_dataset', 'HRSCDataset', 'FairDataset', 'DOTADataset15']",
        )
    )
    if patched != src:
        init_path.write_text(patched)
        print(f"Patched {init_path}: dropped missing DOTAv15Dataset/DOTAv2Dataset imports")
    else:
        print(f"{init_path}: no change (already patched, or upstream text has shifted -- "
              f"check the ImportError hasn't changed shape before assuming this is stale).")

_patch_mmrotate_datasets_init()

# %%
# Patch a second real bug in the Strip-R-CNN repo's own HEAD, hit live during
# inference (predict_fold() -> inference_detector() -> ... ->
# multiclass_nms_rotated()): `labels = torch.arange(num_classes,
# dtype=torch.long)` has no `device=` argument, so it's always built on CPU
# regardless of what device bboxes/scores are actually on (GPU, during real
# inference). Training never exercises this path (inference_detector is
# eval-only), so it silently sat broken until the first real prediction
# call: `RuntimeError: indices should be either on cpu or on the same
# device as the indexed tensor (cpu)` at `labels[inds]`, where `inds` is a
# CUDA tensor (derived from `scores`, which IS on the right device) but
# `labels` is stuck on CPU. Confirmed by cloning the repo directly -- this
# is the only `torch.arange(...)` of its kind anywhere in mmrotate/, not a
# config or environment issue on this project's end. Fixed the same way as
# the dataset patch above: add `device=multi_bboxes.device` so `labels`
# lands on the same device as everything else in the function.
def _patch_multiclass_nms_rotated():
    nms_path = _Path("/content/Strip-R-CNN/mmrotate/core/post_processing/bbox_nms_rotated.py")
    src = nms_path.read_text()
    patched = src.replace(
        "labels = torch.arange(num_classes, dtype=torch.long)",
        "labels = torch.arange(num_classes, dtype=torch.long, device=multi_bboxes.device)",
    )
    if patched != src:
        nms_path.write_text(patched)
        print(f"Patched {nms_path}: labels tensor now built on multi_bboxes.device")
    else:
        print(f"{nms_path}: no change (already patched, or upstream text has shifted -- "
              f"check the RuntimeError hasn't changed shape before assuming this is stale).")

_patch_multiclass_nms_rotated()

# %%
# ImageNet-300ep pretrained Strip R-CNN-S backbone (id from the repo's own
# README, "Imagenet 300-epoch pre-trained Strip R-CNN-S backbone" row) --
# training below (`build_config()`'s `backbone=dict(init_cfg=dict(type=
# "Pretrained", checkpoint=PRETRAINED))`) hard-requires this file to already
# exist locally, and fails with a bare FileNotFoundError deep inside
# model.init_weights() -- after HPO has already mounted Drive and
# materialised every fold -- if it's missing. `/content` is local-runtime
# disk, so this has to be re-downloaded on every fresh Colab runtime, easy
# to forget starting from a clean one. Plain wget/curl chokes on Google
# Drive's large-file "can't scan for viruses, download anyway?" confirmation
# page instead of the actual bytes -- gdown handles that token exchange.
# !pip install -q gdown
# !gdown --id 1_c2aXANKHl0cIBb370LNIkCyDmQpA3_o -O /content/strip_rcnn_s_imagenet_300e.pth

# %%
# Fail loudly here, not deep inside model.init_weights() three folds later --
# a failed/quota-limited gdown pull commonly saves an HTML error page under
# the requested filename instead of raising, which looks like a "successful"
# download until mmcv tries to torch.load() it. A few KB of HTML vs. tens of
# MB of real weights is an easy, cheap distinguishing check.
_pretrained_check = _Path("/content/strip_rcnn_s_imagenet_300e.pth")
assert _pretrained_check.exists(), (
    f"{_pretrained_check} not found -- the gdown cell above didn't run or "
    f"failed. Re-run it, or download manually from the Strip-R-CNN repo "
    f"README's 'Imagenet 300-epoch pre-trained Strip R-CNN-S backbone' link "
    f"and place it at this exact path."
)
_pretrained_size = _pretrained_check.stat().st_size
assert _pretrained_size > 10_000_000, (
    f"{_pretrained_check} is only {_pretrained_size} bytes -- almost "
    f"certainly Google Drive's HTML confirmation/quota page saved under "
    f"the checkpoint's filename, not real weights. Delete it and re-run "
    f"the gdown cell above (Drive's per-file download quota can take a few "
    f"hours to reset if this keeps happening)."
)

# %% [markdown]
# Mounts Drive and points `PROJECT_ROOT` at the project folder, per
# `results/colab_project_structure.md`. Same rationale as the YOLO template:
# training runs against local Colab disk (`PROJECT`) because Drive I/O would
# bottleneck the many small per-trial writes, and only the final archived
# results get copied to Drive at the end.

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
import json
import os
import shutil
import subprocess
import sys
import traceback
from glob import glob
from pathlib import Path
from statistics import mean

import numpy as np
import optuna
import pandas as pd
from tqdm.auto import tqdm

# src/ ships the pieces that MUST be identical across all four architecture
# arms -- fold assignment, E-IRFS, and the shared rotated-IoU metric. Importing
# them (rather than re-pasting them per notebook, as the older notebooks did)
# is what makes "same data, same sampler, same metric" an enforced property
# instead of a claim.
sys.path.insert(0, PROJECT_ROOT)
from src.obb_common import (  # noqa: E402
    build_png_cache, collect_items, load_class_names, make_folds,
    yolo_obb_to_dota,
)
from src.obb_eval import (  # noqa: E402
    evaluate_obb, load_ground_truth, load_predictions, poly_iou_matrix,
)

EIRFS_THRESHOLD = 0.0001
TRAIN_EPOCHS = 250

# --- Edit these paths ---
# NOTE: matches the YOLO template's current default -- this still points at a
# flat pre-fix dataset dump, same as Template_Clean_E-IRFS_optuna_datasetv3.py's
# own DATA_YAML. If reusing this notebook for the Phase 6.1 targeted HPO re-run,
# repoint DATA_ROOT at the grouped/dedup'd/stratified resplit dataset
# (data/dataset_resplit.yaml + Phase 4.1 background negatives) IN LOCKSTEP with
# the template -- otherwise this arm's E-IRFS frequencies and folds would be
# computed on a different split than the other three, breaking the "same data"
# guarantee in results/obb_model_comparison.md.
DATA_ROOT   = "/content/data"
IMAGES_DIR  = f"{DATA_ROOT}/train/images"
LABELS_DIR  = f"{DATA_ROOT}/train/labels"
TEST_IMAGES = f"{DATA_ROOT}/test/images"
TEST_LABELS = f"{DATA_ROOT}/test/labels"
CLASSES_TXT = f"{DATA_ROOT}/classes.txt"

STRIP_RCNN_REPO = "/content/Strip-R-CNN"
# Strip R-CNN-S: the 30M-param variant that carries the paper's headline
# 82.75 mAP50 on DOTA-v1.0. Download the ImageNet-300ep backbone from the
# repo README's Google Drive link and place it here.
BASE_CONFIG = f"{STRIP_RCNN_REPO}/configs/strip_rcnn/strip_rcnn_s_fpn_1x_dota_le90.py"
PRETRAINED  = "/content/strip_rcnn_s_imagenet_300e.pth"

# The strip_rcnn conda env's own Python -- everything mmcv/mmdet/mmrotate
# dependent runs as a subprocess through this binary, never imported directly
# into this notebook's own Python 3.13 kernel. See src/strip_rcnn_worker.py.
CONDA_PY = "/content/miniconda/envs/strip_rcnn/bin/python"
WORKER_SCRIPT = f"{PROJECT_ROOT}/src/strip_rcnn_worker.py"

# Fail loudly here rather than deep inside subprocess.run() -- Colab does
# NOT stop notebook execution just because a `!shell` cell in "file setup"
# above fails (wget/conda/pip cells run to completion regardless of exit
# code), so a partial/failed conda-env build can silently be run past, only
# to surface as a bare FileNotFoundError after this notebook has already
# spent time materialising folds and starting an Optuna study.
assert Path(CONDA_PY).exists(), (
    f"{CONDA_PY} not found -- the strip_rcnn conda env hasn't been built in "
    f"this runtime. Re-run the 'file setup' cells above (Miniconda install, "
    f"conda create -n strip_rcnn, then the torch/openmim/mmcv-full/mmdet and "
    f"Strip-R-CNN install cells) and check each cell's own output for an "
    f"error before re-running this cell."
)
assert Path(WORKER_SCRIPT).exists(), f"Worker script not found at {WORKER_SCRIPT}"

# --- Training/Optuna knobs: held IDENTICAL to the YOLO template ---
DEVICE      = 0
# Same absolute-path requirement as the YOLO template, for the same reason:
# a relative work_dir gets re-rooted by the framework and splits training
# output away from this notebook's own writes, which is what made every fold
# score 0.0 in the original Optuna bug.
PROJECT     = "/content/runs/optuna_strip_rcnn"
EPOCHS_PER_TRIAL = 50
SEED             = 42
K_FOLDS          = 3
SHUFFLE_BEFORE_SPLIT = True

assert Path(PROJECT).is_absolute(), (
    f"PROJECT must be an absolute path, got {PROJECT!r}."
)
Path(PROJECT).mkdir(parents=True, exist_ok=True)

# A source image's PNG bytes don't depend on which fold (or the final
# full-train/val split, below) uses it, so every
# yolo_obb_to_dota(..., convert_to_png=True) call in this notebook shares this
# cache and re-encodes each image exactly once instead of once per call -- see
# png_cache_dir's docstring in src/obb_common.py.
PNG_CACHE_DIR = Path(PROJECT) / "png_cache"

# Pure CPU/PIL work with no GPU dependency, but this notebook's own kernel
# runs on a GPU-attached Colab runtime -- see the identical note in
# StripRCNN_E-IRFS_kfold_datasetv3.py. Restore the zip
# notebooks/StripRCNN_png_cache_prep_cpu.py built under a CPU-only runtime
# (one Drive read + local unzip, like datasetv3_2.zip above, instead of
# ~1,445 individual small-file reads through Drive's slow FUSE mount)
# instead of re-encoding here; falls back to building it locally if that
# Drive archive isn't there yet.
DRIVE_PNG_CACHE_RUN_LABEL = "StripRCNN_png_cache"
DRIVE_PNG_CACHE_ZIP = Path(f"{PROJECT_ROOT}/data/{DRIVE_PNG_CACHE_RUN_LABEL}.zip")
if DRIVE_PNG_CACHE_ZIP.exists():
    print(f"Restoring PNG cache from Drive: {DRIVE_PNG_CACHE_ZIP} -> {PNG_CACHE_DIR}")
    PNG_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    shutil.unpack_archive(str(DRIVE_PNG_CACHE_ZIP), str(PNG_CACHE_DIR), format="zip")
else:
    print(f"No Drive PNG cache found at {DRIVE_PNG_CACHE_ZIP} -- run "
          f"notebooks/StripRCNN_png_cache_prep_cpu.py once under a CPU-only "
          f"runtime first to avoid paying this notebook's GPU-runtime time "
          f"for pure image-encoding work. Falling back to building it here.")

# Shared across every yolo_obb_to_dota call below (HPO folds and the final
# full-train/val split alike) so a given image's (w, h) is read from disk at
# most once per runtime -- see size_cache's docstring in src/obb_common.py.
SIZE_CACHE = {}

CLASS_NAMES = load_class_names(CLASSES_TXT)
NUM_CLASSES = len(CLASS_NAMES)
# DOTA's label format is whitespace-delimited, so class names containing a
# space ("Algal spot") are written underscored on disk and mapped back here.
DOTA_CLASSES = tuple(n.replace(" ", "_") for n in CLASS_NAMES)
print(f"{NUM_CLASSES} classes: {CLASS_NAMES}")

# %% [markdown]
# ### Dataset materialisation
#
# The folds themselves come from `make_folds(..., seed=SEED)` -- the same call,
# with the same seed and the same sorted item list, that the other three
# notebooks make. The four arms therefore train and validate on byte-identical
# image sets. Only the on-disk *encoding* differs: YOLO reads txt path lists of
# normalised polygons; MMRotate reads DOTA `labelTxt` directories of absolute
# pixel polygons. `yolo_obb_to_dota` is that transcription and nothing more --
# it denormalises coordinates and renames the file, it does not filter, clip,
# or re-order any instance.

# %%
items = collect_items(IMAGES_DIR, LABELS_DIR)
print(f"Found {len(items)} labeled images.")

folds = make_folds(items, K_FOLDS, seed=SEED, shuffle=SHUFFLE_BEFORE_SPLIT)
print("Fold sizes (train/val):", [(len(f["train"]), len(f["val"])) for f in folds])

FOLD_ROOT = Path(PROJECT) / "folds"
fold_specs = []
for f in tqdm(folds, desc="Materialising HPO folds", unit="fold"):
    k = f["fold"]
    train_dir = yolo_obb_to_dota(f["train"], FOLD_ROOT / f"fold{k}" / "train", CLASS_NAMES,
                                 convert_to_png=True, png_cache_dir=PNG_CACHE_DIR,
                                 size_cache=SIZE_CACHE)
    val_dir   = yolo_obb_to_dota(f["val"],   FOLD_ROOT / f"fold{k}" / "val",   CLASS_NAMES,
                                 convert_to_png=True, png_cache_dir=PNG_CACHE_DIR,
                                 size_cache=SIZE_CACHE)
    fold_specs.append({
        "fold": k,
        "train_dir": str(train_dir),
        "val_dir": str(val_dir),
        "val_items": f["val"],          # kept for the shared evaluator
        "train_labels": [lbl for _, lbl in f["train"]],
    })
print("Materialised DOTA-format folds under", FOLD_ROOT)

# %% [markdown]
# ### Config construction
#
# MMRotate is config-file-driven rather than kwarg-driven, so an Optuna trial
# has to be written out as a config file before it can be run. This builds
# one config per (trial, fold) by inheriting the stock Strip R-CNN-S config
# and overriding only the dataset wiring and the searched hyperparameters --
# the backbone, neck, RoI head, and assigner are left exactly as the authors
# published them.

# %%
def build_config(trial_params, fold_spec, work_dir, epochs, seed=SEED):
    """Write an MMRotate config for one (trial, fold) and return its path."""
    p = trial_params
    cfg = f'''_base_ = ["{BASE_CONFIG}"]

# --- dataset -------------------------------------------------------------
dataset_type = "DOTADataset"
classes = {DOTA_CLASSES!r}

# imgsz enters MMRotate as the Resize pipeline scale rather than as a single
# `imgsz=` kwarg. Both train and test pipelines must be overridden together;
# overriding only one silently evaluates at a different resolution to training,
# which is exactly the B1 resolution confound this project already fixed once.
img_scale = ({p["imgsz"]}, {p["imgsz"]})

train_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="LoadAnnotations", with_bbox=True),
    dict(type="RResize", img_scale=img_scale),
    dict(type="RRandomFlip",
         flip_ratio=[{p["fliplr"]:.6f}, {p["flipud"]:.6f}],
         direction=["horizontal", "vertical"],
         version="le90"),
    dict(type="PolyRandomRotate",
         rotate_ratio={p["rotate_ratio"]:.6f},
         angles_range=180,
         auto_bound=False,
         version="le90"),
    dict(type="Normalize", mean=[123.675, 116.28, 103.53],
         std=[58.395, 57.12, 57.375], to_rgb=True),
    dict(type="Pad", size_divisor=32),
    dict(type="DefaultFormatBundle"),
    dict(type="Collect", keys=["img", "gt_bboxes", "gt_labels"]),
]

test_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(
        type="MultiScaleFlipAug",
        img_scale=img_scale,
        flip=False,
        transforms=[
            dict(type="RResize"),
            dict(type="Normalize", mean=[123.675, 116.28, 103.53],
                 std=[58.395, 57.12, 57.375], to_rgb=True),
            dict(type="Pad", size_divisor=32),
            dict(type="DefaultFormatBundle"),
            dict(type="Collect", keys=["img"]),
        ],
    ),
]

data = dict(
    samples_per_gpu={p["batch"]},
    # A live run showed data_time consistently 52-57% of total iteration
    # time (e.g. time=2.652s, data_time=1.523s) -- the GPU was idle waiting
    # on the CPU-bound image decode + RResize/RRandomFlip/PolyRandomRotate
    # augmentation pipeline more than half of every step. workers_per_gpu=2
    # was a hardcoded floor regardless of the actual Colab instance's CPU
    # count. Neither this nor persistent_workers below changes WHAT data
    # reaches the model or in what order (same dataset, same sampler,
    # same augmentation) -- only how many CPU workers prefetch/decode it in
    # parallel -- so this doesn't affect trained-model results or the
    # cross-architecture comparison methodology, only wall-clock time.
    workers_per_gpu={max(2, (os.cpu_count() or 2) - 1)},
    # mmdet's own DataLoader default (persistent_workers=False) tears down
    # and respawns the entire worker pool at every epoch boundary -- real,
    # avoidable overhead given TRAIN_EPOCHS=250 for final K-fold training
    # (250 respawns per fold x 9 folds). Safe with this repo's own
    # mp_start_method='fork' default (default_runtime.py): workers are
    # forked AFTER attach_eirfs() has already monkey-patched the dataset's
    # __getitem__ in the main process, so every fork -- persistent or not --
    # inherits the patched method identically.
    persistent_workers=True,
    train=dict(type=dataset_type, classes=classes,
               ann_file="{fold_spec["train_dir"]}/labelTxt/",
               img_prefix="{fold_spec["train_dir"]}/images/",
               pipeline=train_pipeline),
    val=dict(type=dataset_type, classes=classes,
             ann_file="{fold_spec["val_dir"]}/labelTxt/",
             img_prefix="{fold_spec["val_dir"]}/images/",
             pipeline=test_pipeline),
    test=dict(type=dataset_type, classes=classes,
              ann_file="{fold_spec["val_dir"]}/labelTxt/",
              img_prefix="{fold_spec["val_dir"]}/images/",
              pipeline=test_pipeline),
)

# --- head width ----------------------------------------------------------
# Both stages carry their own num_classes; setting only one leaves a 15-class
# head from the DOTA config in the model and the run fails deep inside the
# loss with a shape mismatch rather than at config time.
model = dict(
    backbone=dict(init_cfg=dict(type="Pretrained", checkpoint="{PRETRAINED}")),
    roi_head=dict(
        bbox_head=dict(num_classes={NUM_CLASSES}),
    ),
    # score_thr 0.05 -> 0.001, for cross-arm comparability. The base config
    # (strip_rcnn_s_fpn_1x_dota_le90.py) sets test_cfg.rcnn.score_thr=0.05,
    # which is applied inside the rotated NMS during inference -- i.e.
    # BEFORE src/strip_rcnn_worker.py's own predict_fold(score_thr=0.001)
    # ever sees a detection, making that function's "deliberately near-zero"
    # docstring untrue in practice. It caps recall and truncates the
    # precision/recall curve evaluate_obb() integrates, understating this
    # arm's AP against the YOLO and Oriented-DETR arms, which both evaluate
    # at 0.001. No _delete_ here on purpose: MMCV's recursive merge is what
    # we want, so nms/nms_pre/max_per_img keep their inherited values and
    # only the threshold changes. Evaluation operating point only -- the
    # architecture and training objective are untouched.
    test_cfg=dict(rcnn=dict(score_thr=0.001)),
)

# --- optimisation --------------------------------------------------------
# _delete_=True on both dicts below: MMCV's config inheritance does a
# RECURSIVE MERGE of dict-typed fields by default, not a replace -- without
# it, keys from the inherited base config that this dict doesn't happen to
# also set survive alongside ours. Confirmed against the actual base config
# chain: strip_rcnn_s_fpn_1x_dota_le90.py -> configs/_base_/schedules/
# schedule_1x.py sets lr_config = dict(policy='step', ..., step=[8, 11]) --
# our own lr_config below (policy="CosineAnnealing") merged onto it rather
# than replacing it, leaving step=[8, 11] attached and raising
# `CosineAnnealingLrUpdaterHook: LrUpdaterHook.__init__() got an unexpected
# keyword argument 'step'` at train_detector() -> register_lr_hook(). The
# strip_rcnn_s config's own optimizer override already uses this same
# `_delete_=True` pattern for exactly this reason (it also fully replaces
# the schedule's SGD optimizer with AdamW) -- our optimizer dict below
# didn't, which is a latent version of the identical bug: when Optuna picks
# optimizer="SGD" (in the search space), our dict sets `momentum=` but not
# `betas=`, so the base's leftover `betas=(0.9, 0.999)` (from
# strip_rcnn_s's own AdamW override) would merge in and crash
# torch.optim.SGD(..., betas=...) the same way.
optimizer = dict(
    _delete_=True,
    type="{p["optimizer"]}",
    lr={p["lr0"]:.8f},
    weight_decay={p["weight_decay"]:.10f},
    {p["_optimizer_extra"]}
)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))

# Cosine annealing to lr0 * lrf, matching the YOLO template's cos_lr=True with
# its lrf final-LR fraction. MMRotate expresses that as min_lr_ratio.
lr_config = dict(
    _delete_=True,
    policy="CosineAnnealing",
    min_lr_ratio={p["lrf"]:.6f},
    warmup="linear",
    warmup_iters={p["warmup_iters"]},
    warmup_ratio={p["warmup_ratio"]:.6f},
)

runner = dict(type="EpochBasedRunner", max_epochs={epochs})
checkpoint_config = dict(interval={epochs})   # only the final epoch is kept
evaluation = dict(interval={epochs} + 1)      # MMRotate's own mAP50 is not used
log_config = dict(interval=50, hooks=[dict(type="TextLoggerHook")])

work_dir = "{work_dir}"
seed = {seed}
gpu_ids = [{DEVICE}]
# mmdet's build_dp() looks this up in a dict keyed by the bare device TYPE
# ('cuda'/'cpu'), not a torch device string -- confirmed against mmdet
# 2.28.2's own util_distribution.py (`dp_factory = {{'cuda': ...,
# 'cpu': ...}}`, `dp_factory[device]`). "cuda:{{DEVICE}}" raised
# KeyError('cuda:0') at train_detector() -> build_dp(). The actual GPU
# index is carried by gpu_ids above (-> device_ids kwarg -> model.cuda(
# device_ids[0])), not by this field.
device = "cuda"
'''
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = work_dir / "config.py"
    cfg_path.write_text(cfg)
    return str(cfg_path)


def optimizer_extra(name, momentum):
    """MMRotate optimizers take different extra kwargs by type.

    SGD takes `momentum`; Adam/AdamW take `betas`, where beta1 plays the
    equivalent role. Mapping the single searched `momentum` onto beta1 keeps
    the search space the same shape as the YOLO template's rather than adding
    an optimizer-conditional parameter that Optuna would have to model.
    """
    if name == "SGD":
        return f"momentum={momentum:.6f},"
    return f"betas=({momentum:.6f}, 0.999),"

# %% [markdown]
# ### Train / predict one fold
#
# The mmcv/mmdet/mmrotate stack this needs (torch<=1.13, Python<=3.10) cannot
# be imported into this notebook's own Python 3.13 kernel -- see the "file
# setup" section's Miniconda cell. `run_fold_via_conda` hands the whole
# train+predict job to `src/strip_rcnn_worker.py`, run as a subprocess inside
# the `strip_rcnn` conda env, and reads back the predictions it writes.

# %%
def run_fold_via_conda(cfg_path, work_dir, train_labels, test_items, epochs, verbose=False):
    """Train one fold and predict on `test_items`, entirely inside the
    `strip_rcnn` conda env (a separate Python 3.10 process), and return the
    resulting checkpoint path.

    A subprocess boundary can't pass live Python objects, so the arguments
    are written to a JSON file the worker script reads, and the predictions
    it computes are written to `work_dir/test_predictions.json` for this
    kernel to read back via `load_predictions` -- same shared prediction
    format every other arm's own inference code produces.
    """
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    ckpt = work_dir / f"epoch_{epochs}.pth"
    predictions_out = work_dir / "test_predictions.json"
    args_path = work_dir / "worker_args.json"
    args_path.write_text(json.dumps({
        "project_root": PROJECT_ROOT,
        "cfg_path": str(cfg_path),
        "work_dir": str(work_dir),
        "epochs": epochs,
        "train_labels": [str(p) for p in train_labels],
        "class_names": list(CLASS_NAMES),
        "num_classes": NUM_CLASSES,
        "eirfs_threshold": EIRFS_THRESHOLD,
        "verbose": verbose,
        "test_items": [[str(img), str(lbl)] for img, lbl in test_items],
        "device": DEVICE,
        "predictions_out": str(predictions_out),
    }))

    # subprocess.run() inherits the whole parent (Colab kernel) environment
    # by default -- two vars from it are actively wrong for a subprocess
    # running under the separate strip_rcnn conda env:
    #   MPLBACKEND: Colab's IPython kernel sets this to
    #     "module://matplotlib_inline.backend_inline" so plots render
    #     inline. mmdet's own visualization module imports matplotlib.pyplot
    #     at import time, which sets rcParams['backend'] from this env var --
    #     the strip_rcnn env's matplotlib doesn't recognise that inline
    #     backend string and raises ValueError before training even starts.
    #     Forced to "Agg" (headless, no display needed) instead.
    #   PYTHONPATH: conda itself warns about this at env-creation time
    #     ("You currently have a PYTHONPATH environment variable set...").
    #     Colab's own kernel sets it to point at ITS OWN (Python 3.13)
    #     site-packages, which could shadow or ABI-mismatch against the
    #     strip_rcnn env's own Python 3.10 mmcv/mmdet if left inherited.
    worker_env = os.environ.copy()
    worker_env["MPLBACKEND"] = "Agg"
    worker_env.pop("PYTHONPATH", None)

    # subprocess.run(..., capture_output=True) buffers the ENTIRE subprocess's
    # stdout/stderr internally and only prints it once the process exits --
    # for a fold that trains for real minutes, the Colab cell shows nothing
    # at all in the meantime, which is indistinguishable from a genuine hang.
    # Popen + streaming line-by-line as it arrives fixes that -- but only if
    # the CHILD process itself is unbuffered too: Python defaults to
    # block-buffered stdout when it isn't connected to a real terminal (i.e.
    # exactly this pipe), so without `-u` here, output would still arrive in
    # infrequent large chunks rather than as it's actually produced.
    process = subprocess.Popen(
        [CONDA_PY, "-u", WORKER_SCRIPT, str(args_path)],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, env=worker_env, bufsize=1,
    )
    for line in process.stdout:
        print(line, end="")
    process.wait()
    if process.returncode != 0:
        raise RuntimeError(f"strip_rcnn_worker.py failed for {work_dir} -- see output above.")
    if not ckpt.exists():
        raise RuntimeError(f"no checkpoint at {ckpt} after worker run -- see output above.")
    if not predictions_out.exists():
        raise RuntimeError(f"no predictions at {predictions_out} after worker run.")
    return ckpt

# %% [markdown]
# ## Precision / recall / F1 at the best-F1 confidence threshold
#
# `evaluate_obb()` (the shared AP metric) deliberately has no single
# confidence operating point -- AP integrates over the whole precision/recall
# curve. Ultralytics' own `box.mp`/`box.mr`/`box.f1` (used by the YOLO K-fold
# notebooks) DO report at one point: the confidence threshold that maximises
# mean F1 across classes. This reproduces that same operating point here so
# this arm's P/R/F1 are comparable to the YOLO arms', without touching
# `src/obb_eval.py`'s shared evaluator (which every arm's AP number, and the
# YOLO-baseline rescoring script, depend on).
#
# Ported in from StripRCNN_E-IRFS_kfold_datasetv3.py (byte-identical) so
# this notebook's own objective() can report the same P/R/F1 per fold that
# notebook's HPO and final K-fold training already do, not just mAP50-95.

# %%
def precision_recall_f1(gt, predictions, num_classes, iou_thr=0.5, n_points=1000):
    """Overall precision/recall/F1 at the confidence threshold that maximises
    mean F1 across classes with GT -- the same operating point Ultralytics'
    ap_per_class() uses for box.mp/box.mr/box.f1. Matching is greedy
    per-image IoU>=iou_thr (COCO-style, one GT claimed once), independent of
    evaluate_obb()'s multi-IoU-threshold AP computation.
    """
    by_image = {}
    for p in predictions:
        by_image.setdefault(p["image"], []).append(p)

    px = np.linspace(0.0, 1.0, n_points)
    p_curve = np.zeros((num_classes, n_points))
    r_curve = np.zeros((num_classes, n_points))
    valid = np.zeros(num_classes, dtype=bool)

    for cid in range(num_classes):
        gt_by_image, n_gt = {}, 0
        for stem, g in gt.items():
            mask = g["cls"] == cid
            polys = g["polys"][mask]
            gt_by_image[stem] = polys
            n_gt += len(polys)
        if n_gt == 0:
            continue
        valid[cid] = True

        dets = []
        for stem, preds in by_image.items():
            if stem not in gt:
                continue
            for p in preds:
                if p["cls"] == cid:
                    dets.append((p["score"], stem, p["poly"]))
        if not dets:
            r_curve[cid, :] = 0.0
            p_curve[cid, :] = 1.0
            continue
        dets.sort(key=lambda d: -d[0])

        claimed = {stem: np.zeros(len(g), dtype=bool) for stem, g in gt_by_image.items()}
        tp = np.zeros(len(dets), dtype=np.float64)
        fp = np.zeros(len(dets), dtype=np.float64)
        for di, (_, stem, poly) in enumerate(dets):
            stem_gt = gt_by_image.get(stem, np.zeros((0, 8)))
            ious = poly_iou_matrix([poly], stem_gt)[0] if len(stem_gt) else np.zeros(0)
            best_iou, best_j = iou_thr, -1
            for j, v in enumerate(ious):
                if claimed[stem][j] or v < best_iou:
                    continue
                best_iou, best_j = v, j
            if best_j >= 0:
                claimed[stem][best_j] = True
                tp[di] = 1
            else:
                fp[di] = 1

        cum_tp = np.cumsum(tp)
        cum_fp = np.cumsum(fp)
        recall = cum_tp / n_gt
        precision = cum_tp / np.maximum(cum_tp + cum_fp, np.finfo(np.float64).eps)
        scores = np.array([d[0] for d in dets])

        # scores is sorted descending; np.interp needs ascending x, so both
        # the confidence grid and scores are negated. This reproduces the
        # step function "recall/precision using every detection with
        # score >= threshold" at each grid point, same trick ap_per_class()
        # uses to build its P/R-vs-confidence curves.
        r_curve[cid] = np.interp(-px, -scores, recall, left=0.0)
        p_curve[cid] = np.interp(-px, -scores, precision, left=1.0)

    if not valid.any():
        return {"precision": 0.0, "recall": 0.0, "F1": 0.0}

    p_valid, r_valid = p_curve[valid], r_curve[valid]
    f1_curve = 2 * p_valid * r_valid / np.maximum(p_valid + r_valid, np.finfo(np.float64).eps)
    best_i = int(f1_curve.mean(axis=0).argmax())

    precision = float(p_valid[:, best_i].mean())
    recall = float(r_valid[:, best_i].mean())
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {"precision": precision, "recall": recall, "F1": f1}


# %% [markdown]
# ## Optuna
#
# Same budget and same guards as the YOLO template: TPE with an explicit
# `n_startup_trials` well below `n_trials` (its default of 10 equalled the old
# `n_trials=10`, so the sampler never left random startup and TPE never
# engaged), a failed fold raises rather than scoring 0.0, and a fully-tied
# study refuses to write `best_params.json`.
#
# The search space is the YOLO space minus the parameters Strip R-CNN has no
# equivalent for. What is dropped, and why, is tabulated in
# `results/obb_model_comparison.md`; the short version is that `box`/`cls`/`dfl`
# are YOLO loss weights with no two-stage counterpart, and mosaic/mixup/HSV
# are not in MMRotate 0.x's rotated pipeline. `rotate_ratio` is added because
# `PolyRandomRotate` is MMRotate's nearest analogue to the geometric jitter
# YOLO gets from `degrees`/`shear`/`perspective`.

# %%
def suggest_params(trial):
    optimizer = trial.suggest_categorical("optimizer", ["SGD", "Adam", "AdamW"])
    momentum = trial.suggest_float("momentum", 0.7, 0.98)
    # fliplr sampled first so flipud's upper bound can depend on it -- see
    # the note by flip_ratio below.
    fliplr = trial.suggest_float("fliplr", 0.0, 0.9)
    p = {
        "optimizer": optimizer,
        "lr0": trial.suggest_float("lr0", 1e-4, 1e-1, log=True),
        "lrf": trial.suggest_float("lrf", 0.01, 1.0),
        "momentum": momentum,
        "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True),
        "warmup_iters": trial.suggest_int("warmup_iters", 0, 1000),
        "warmup_ratio": trial.suggest_float("warmup_ratio", 1e-3, 1e-1, log=True),
        "imgsz": trial.suggest_categorical("imgsz", [320, 640, 832, 1024]),
        "batch": trial.suggest_categorical("batch", [2, 4, 8]),
        "fliplr": fliplr,
        # Capped to keep fliplr + flipud <= 1 -- build_config() passes both
        # as RRandomFlip's flip_ratio=[fliplr, flipud], and RRandomFlip's
        # base class (mmdet's RandomFlip.__init__) hard-asserts
        # `0 <= sum(flip_ratio) <= 1`, confirmed against mmdet 2.28.2's own
        # source. That assertion fires at dataset-build time -- before any
        # real training starts -- so independently sampling fliplr in
        # [0, 0.9] and flipud in [0, 0.5] (their originally-intended ranges)
        # could draw a combination violating it on roughly 1 in 6 trials by
        # search-space area, aborting that trial outright (and the whole
        # study, if it happened to be trial 0 -- see objective()'s own
        # "Trial 0 failed on all folds" guard).
        "flipud": trial.suggest_float("flipud", 0.0, min(0.5, 1.0 - fliplr)),
        "rotate_ratio": trial.suggest_float("rotate_ratio", 0.0, 1.0),
    }
    p["_optimizer_extra"] = optimizer_extra(optimizer, momentum)
    return p


def objective(trial):
    p = suggest_params(trial)
    fold_scores, rows, fold_errors = [], [], []

    for spec in fold_specs:
        k = spec["fold"]
        work_dir = Path(PROJECT) / f"trial_{trial.number:03d}_fold{k}"
        try:
            cfg_path = build_config(p, spec, work_dir, EPOCHS_PER_TRIAL)
            run_fold_via_conda(cfg_path, work_dir, spec["train_labels"], spec["val_items"],
                               EPOCHS_PER_TRIAL, verbose=(trial.number == 0))

            records = load_predictions(work_dir / "test_predictions.json")
            gt = load_ground_truth(spec["val_items"])
            result = evaluate_obb(gt, records, NUM_CLASSES, CLASS_NAMES)
            score = float(result["mAP50-95"])
            if not np.isfinite(score):
                raise RuntimeError(f"shared evaluator returned {score} for fold {k}")
            # Only mAP50-95 (the pruning/selection metric) used to be saved
            # here -- mAP50/mAP75 come for free from the same evaluate_obb()
            # call, and precision/recall/F1 give the same operating-point
            # numbers the final training run's own test_metrics.json reports.
            prf = precision_recall_f1(gt, records, NUM_CLASSES)
        except Exception as e:
            # Full traceback to stdout, not a one-line stderr note that gets
            # buried under MMRotate's per-iteration logs. Substituting 0.0 here
            # (as the original YOLO notebook did) ties every trial, and Optuna
            # only replaces best_trial on a STRICT improvement -- so a tied
            # study permanently reports trial 0's pre-search seeded params.
            print(f"\n{'!'*70}\n[Trial {trial.number}] Fold {k} FAILED: {e}\n{'!'*70}")
            traceback.print_exc(file=sys.stdout)
            sys.stdout.flush()
            fold_errors.append(f"fold {k}: {type(e).__name__}: {e}")
            rows.append({
                "trial": trial.number, "fold": k,
                "mAP50-95": None, "mAP50": None, "mAP75": None,
                "precision": None, "recall": None, "F1": None,
                "error": str(e),
            })
            continue

        fold_scores.append(score)
        rows.append({
            "trial": trial.number, "fold": k,
            "mAP50-95": score, "mAP50": result["mAP50"], "mAP75": result["mAP75"],
            "precision": prf["precision"], "recall": prf["recall"], "F1": prf["F1"],
            "error": None,
        })
        trial.report(mean(fold_scores), step=k + 1)
        # trial.report() alone does NOT prune anything -- Optuna requires the
        # objective to explicitly check should_prune() and raise TrialPruned
        # itself. Without this, the PercentilePruner configured below (and
        # every trial.report() call above) was dead weight: every trial
        # always ran every remaining fold to completion regardless of how
        # bad an early fold's score was, defeating the entire point of
        # configuring a pruner (saving compute on trials unlikely to win).
        if trial.should_prune():
            pd.DataFrame(rows).to_csv(Path(PROJECT) / f"cv_trial_{trial.number:03d}.csv", index=False)
            raise optuna.TrialPruned(
                f"pruned after fold {k} (intermediate mean mAP50-95={mean(fold_scores):.4f})"
            )

    pd.DataFrame(rows).to_csv(Path(PROJECT) / f"cv_trial_{trial.number:03d}.csv", index=False)

    if not fold_scores:
        detail = "; ".join(fold_errors)
        if trial.number == 0:
            raise RuntimeError(
                f"Trial 0 failed on all {len(fold_specs)} folds -- aborting the study "
                f"instead of reporting a meaningless best_params.json. Causes: {detail}"
            )
        raise optuna.TrialPruned(f"all folds failed: {detail}")

    if fold_errors:
        print(f"[Trial {trial.number}] WARNING: {len(fold_errors)}/{len(fold_specs)} folds "
              f"failed; CV mean is over the {len(fold_scores)} that succeeded.")
    return float(mean(fold_scores))

# %%
N_TRIALS = 10
N_STARTUP_TRIALS = 3

sampler = optuna.samplers.TPESampler(seed=SEED, n_startup_trials=N_STARTUP_TRIALS)
pruner = optuna.pruners.PercentilePruner(
    percentile=50, n_startup_trials=N_STARTUP_TRIALS, n_warmup_steps=2
)
study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)
study.optimize(objective, n_trials=N_TRIALS)

completed = study.get_trials(deepcopy=False, states=(optuna.trial.TrialState.COMPLETE,))
print(f"Completed trials: {len(completed)} / {N_TRIALS} "
      f"(pruned/failed: {N_TRIALS - len(completed)})")
if not completed:
    raise RuntimeError(
        "No trial completed successfully. Scroll up for the per-fold "
        "tracebacks; best_params.json NOT written."
    )
distinct_values = {round(t.value, 10) for t in completed if t.value is not None}
if len(distinct_values) == 1 and len(completed) > 1:
    raise RuntimeError(
        f"All {len(completed)} completed trials returned the identical objective value "
        f"{distinct_values.pop()}. Optuna's tie-break would make trial 0 'best' regardless "
        f"of the search. Fix the underlying training/scoring issue first; "
        f"best_params.json NOT written."
    )

print("Best CV mAP50-95:", study.best_value)
print("Best trial number:", study.best_trial.number)
print("Best params:", study.best_params)

FINAL_DIR = Path(PROJECT) / "best_final_strip_rcnn"
FINAL_DIR.mkdir(parents=True, exist_ok=True)
(FINAL_DIR / "best_params.json").write_text(json.dumps(study.best_params, indent=2))

# %% [markdown]
# ### Final training on the full train split

# %%
best_params = json.loads((FINAL_DIR / "best_params.json").read_text())
best_params["_optimizer_extra"] = optimizer_extra(
    best_params["optimizer"], best_params["momentum"]
)

FULL_TRAIN_DIR = yolo_obb_to_dota(items, FOLD_ROOT / "full" / "train", CLASS_NAMES,
                                  convert_to_png=True, png_cache_dir=PNG_CACHE_DIR,
                                  size_cache=SIZE_CACHE)
val_items = collect_items(f"{DATA_ROOT}/val/images", f"{DATA_ROOT}/val/labels")
FULL_VAL_DIR = yolo_obb_to_dota(val_items, FOLD_ROOT / "full" / "val", CLASS_NAMES,
                                convert_to_png=True, png_cache_dir=PNG_CACHE_DIR,
                                size_cache=SIZE_CACHE)

final_spec = {"train_dir": str(FULL_TRAIN_DIR), "val_dir": str(FULL_VAL_DIR)}
final_cfg_path = build_config(best_params, final_spec, FINAL_DIR, TRAIN_EPOCHS)

# Collected before training (rather than after, as the original order had it)
# since run_fold_via_conda needs test_items up front to predict on them in
# the same subprocess call that trains the final model.
test_items = collect_items(TEST_IMAGES, TEST_LABELS)
print(f"Test images: {len(test_items)}")

FINAL_CKPT = run_fold_via_conda(final_cfg_path, FINAL_DIR, [lbl for _, lbl in items],
                                test_items, TRAIN_EPOCHS, verbose=True)
print(f"Final run saved at: {FINAL_DIR} (checkpoint {FINAL_CKPT})")

# %% [markdown]
# ### test
#
# Held-out test evaluation through `src/obb_eval.py` -- the same function, on
# the same test items, that scores the YOLO, Oriented-DETR, and RF-DETR arms.
# MMRotate's own `--eval mAP` is intentionally not called: it reports VOC-style
# AP at IoU 0.5 only and would not be comparable.

# %%
records = load_predictions(FINAL_DIR / "test_predictions.json")

gt = load_ground_truth(test_items)
metrics = evaluate_obb(gt, records, NUM_CLASSES, CLASS_NAMES)
# precision/recall/F1 at the best-F1 confidence threshold, same operating
# point the K-fold notebook's own test_metrics.json already reports --
# previously this final-training test_metrics.json only had mAP figures.
prf = precision_recall_f1(gt, records, NUM_CLASSES)
metrics["precision"] = prf["precision"]
metrics["recall"] = prf["recall"]
metrics["F1"] = prf["F1"]
(FINAL_DIR / "test_metrics.json").write_text(json.dumps(metrics, indent=2))

print(f"\n=== Strip R-CNN-S | held-out test (shared evaluator) ===")
print(f"mAP50-95: {metrics['mAP50-95']:.4f}")
print(f"mAP50   : {metrics['mAP50']:.4f}")
print(f"mAP75   : {metrics['mAP75']:.4f}")
print(f"precision={metrics['precision']:.4f}  recall={metrics['recall']:.4f}  F1={metrics['F1']:.4f}")
for name, m in metrics["per_class"].items():
    print(f"  {name:15} AP50-95={m['AP50-95']:.4f}  AP50={m['AP50']:.4f}  n={m['n_gt']}")

# %% [markdown]
# ### Archive to Drive

# %%
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


# Archives the ENTIRE local /content/runs tree, not just PROJECT -- each
# notebook gets its own Colab runtime, so its /content/runs only ever holds
# this notebook's own output -- into a notebook-labeled subfolder on Drive.
# Naming the Drive destination after RUN_LABEL rather than PROJECT's basename
# keeps runs/ readable across the whole comparison (results/obb_model_comparison.md)
# instead of five siblings named after internal PROJECT strings.
RUN_LABEL = "StripRCNN_E-IRFS_Optuna"

LOCAL_RUNS = Path("/content/runs")
DRIVE_DEST = _unique_drive_dir(Path(f"{PROJECT_ROOT}/runs/{RUN_LABEL}"))
DRIVE_DEST.mkdir(parents=True, exist_ok=True)
if DRIVE_DEST.name != RUN_LABEL:
    print(f"NOTE: '{RUN_LABEL}' already exists on Drive -- archiving to '{DRIVE_DEST.name}' instead.")
# The materialised fold directories are symlinked copies of the dataset and
# would either bloat the archive or copy as broken links; the fold *assignment*
# is reproducible from make_folds(items, K_FOLDS, seed=SEED) anyway.
shutil.copytree(LOCAL_RUNS, DRIVE_DEST, dirs_exist_ok=True,
                ignore=shutil.ignore_patterns("folds", "png_cache"))

print("Local  :", LOCAL_RUNS, "->", sorted(p.name for p in LOCAL_RUNS.iterdir()))
print("Drive  :", DRIVE_DEST, "->", sorted(p.name for p in DRIVE_DEST.iterdir()))

# %% [markdown]
# ## Re-test from Drive (resume without retraining)
#
# If you're in a fresh runtime and just want to re-verify a previously
# archived run without retraining, run this section after the mount/constants
# cells above and skip Optuna/training entirely.

# %%
# Restores the whole /content/runs tree from its RUN_LABEL folder on Drive --
# PROJECT is a subfolder of /content/runs, so this repopulates it along with
# everything else that was archived alongside it.
RUN_LABEL = "StripRCNN_E-IRFS_Optuna"
LOCAL_RUNS = Path("/content/runs")
DRIVE_DEST = Path(f"{PROJECT_ROOT}/runs/{RUN_LABEL}")

LOCAL_RUNS.mkdir(parents=True, exist_ok=True)
shutil.copytree(DRIVE_DEST, LOCAL_RUNS, dirs_exist_ok=True)
print(f"Restored {LOCAL_RUNS} <- {DRIVE_DEST}")

FINAL_DIR = Path(PROJECT) / "best_final_strip_rcnn"

# %%
# Re-prints the held-out test metrics already computed -- no re-inference
# needed since the worker/evaluate_obb() already wrote test_metrics.json
# during the original run.
metrics = json.loads((FINAL_DIR / "test_metrics.json").read_text())

print(f"\n=== Strip R-CNN-S | held-out test (shared evaluator) ===")
print(f"mAP50-95: {metrics['mAP50-95']:.4f}")
print(f"mAP50   : {metrics['mAP50']:.4f}")
print(f"mAP75   : {metrics['mAP75']:.4f}")
# .get(..., nan) rather than a plain key lookup -- a test_metrics.json
# archived from before precision/recall/F1 were added here won't have
# these keys, and this cell reads whatever was actually archived on Drive.
print(f"precision={metrics.get('precision', float('nan')):.4f}  "
      f"recall={metrics.get('recall', float('nan')):.4f}  "
      f"F1={metrics.get('F1', float('nan')):.4f}")
for name, m in metrics["per_class"].items():
    print(f"  {name:15} AP50-95={m['AP50-95']:.4f}  AP50={m['AP50']:.4f}  n={m['n_gt']}")

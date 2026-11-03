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
# # Strip R-CNN (AAAI 2026) -- E-IRFS + K-fold final training, datasetv3
#
# K-fold variant of `notebooks/StripRCNN_E-IRFS_optuna_datasetv3.py`. The
# Optuna HPO search itself is **not** re-run here -- this notebook loads the
# `best_params.json` produced by that notebook's search (from this session's
# local run, or from its Drive archive under `StripRCNN_E-IRFS_Optuna`) and
# replaces the single final full-dataset training run with K-fold final
# training: one model is trained per fold with the same best hyperparameters,
# and every fold's model is evaluated on the same held-out test set through
# the shared `src/obb_eval.py` evaluator, giving a mean +/- std test score
# instead of a single-split point estimate.
#
# The fold-construction *mechanics* and the config/train/predict helpers are
# unchanged from the Optuna notebook -- kept byte-identical on purpose. If
# the template's fold-construction or config-building logic changes,
# re-sync this file the same way. One deliberate content difference: the
# *item pool* folded over is `train/` + `val/` pooled together, not `train/`
# alone -- a K-fold split carves its own held-out validation portion out of
# every fold, so the dataset's fixed `val/` split (which the Optuna notebook
# needed as final training's own validation) serves no purpose here. Fold
# membership is therefore *not* identical to the Optuna notebook's CV folds
# -- that's fine since only `best_params.json` (hyperparameters, not
# fold-conditioned results) is carried over from that search. `test/` is
# untouched throughout, as always. See that notebook's header for the two
# Strip R-CNN-specific caveats (12 vs 17 tunable params; MMRotate's own
# mAP50 not used, scored via the shared evaluator instead) and the
# environment warning (MMRotate 0.x needs mmcv-full, mmdet 2.x, torch<=1.13,
# Python<=3.10).

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
# bottleneck the many small per-fold writes, and only the final archived
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
import time
import traceback
from pathlib import Path
from statistics import mean

import numpy as np
import optuna
import pandas as pd
from tqdm.auto import tqdm

# src/ ships the pieces that MUST be identical across all four architecture
# arms -- fold assignment, E-IRFS, and the shared rotated-IoU metric. E-IRFS
# itself (EIRFSWeights/attach_eirfs) is NOT imported here -- it's applied
# inside src/strip_rcnn_worker.py, which runs under the strip_rcnn conda env
# (see "file setup" above), not this notebook's own kernel.
sys.path.insert(0, PROJECT_ROOT)
from src.obb_common import (  # noqa: E402
    build_png_cache, collect_items, load_class_names, make_folds,
    yolo_obb_to_dota,
)
from src.obb_eval import (  # noqa: E402
    evaluate_obb, load_ground_truth, load_predictions, poly_iou_matrix,
    precision_recall_f1,
)
from src.obb_logging import (  # noqa: E402
    Timer, collect_epoch_log, dump_study, fmt_dt, open_study,
    write_fold_composition, write_per_class_tables, write_run_env,
)

EIRFS_THRESHOLD = 0.0001
TRAIN_EPOCHS = 250

# Where the strip_rcnn conda env (Python 3.10, torch 1.13.1, mmcv-full/mmdet)
# actually lives, and the worker script that runs inside it -- see "file
# setup" above and src/strip_rcnn_worker.py's own docstring for why this
# split exists.
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

# --- Edit these paths ---
# NOTE: these still point at a flat pre-fix dataset dump, matching the Optuna
# notebook's own DATA_ROOT. Must match whatever split that notebook used to
# produce best_params.json -- see the note there before repointing.
DATA_ROOT   = "/content/data"
IMAGES_DIR  = f"{DATA_ROOT}/train/images"
LABELS_DIR  = f"{DATA_ROOT}/train/labels"
# K-fold carves its own held-out validation portion out of every fold, so a
# separate fixed val split serves no purpose here (unlike the single-split
# Optuna notebook, which used it as final training's own validation) --
# pooled into `items` below instead of left untouched throughout every fold.
VAL_IMAGES  = f"{DATA_ROOT}/val/images"
VAL_LABELS  = f"{DATA_ROOT}/val/labels"
TEST_IMAGES = f"{DATA_ROOT}/test/images"
TEST_LABELS = f"{DATA_ROOT}/test/labels"
CLASSES_TXT = f"{DATA_ROOT}/classes.txt"

STRIP_RCNN_REPO = "/content/Strip-R-CNN"
# Strip R-CNN-S: the 30M-param variant that carries the paper's headline
# 82.75 mAP50 on DOTA-v1.0. Download the ImageNet-300ep backbone from the
# repo README's Google Drive link and place it here.
BASE_CONFIG = f"{STRIP_RCNN_REPO}/configs/strip_rcnn/strip_rcnn_s_fpn_1x_dota_le90.py"
PRETRAINED  = "/content/strip_rcnn_s_imagenet_300e.pth"

# --- Training knobs: held IDENTICAL to the Optuna notebook ---
DEVICE      = 0
# Same absolute-path requirement as the Optuna notebook, for the same reason:
# a relative work_dir gets re-rooted by the framework and splits training
# output away from this notebook's own writes.
# Kept in a separate directory from the Optuna notebook's PROJECT so K-fold
# final training never collides with that search's trial runs.
PROJECT     = "/content/runs/kfold_final_strip_rcnn"
SEED             = 42
K_FOLDS          = 9
SHUFFLE_BEFORE_SPLIT = True

# Where to find the archived Optuna run's best_params.json if this session
# doesn't already have one locally (see load_best_params() below).
OPTUNA_PROJECT_LOCAL = "/content/runs/optuna_strip_rcnn"
OPTUNA_RUN_LABEL = "StripRCNN_E-IRFS_Optuna"
OPTUNA_FINAL_SUBDIR = "best_final_strip_rcnn"

# If True, this notebook runs its own Optuna search (identical search space
# and objective to notebooks/StripRCNN_E-IRFS_optuna_datasetv3.py) against
# its own fold_specs below instead of loading a previously-archived
# best_params.json. Off by default -- the documented/comparable path is to
# reuse the sibling Optuna notebook's winning params (see the header
# markdown), which is cheaper and keeps HPO decoupled from K-fold reporting.
RUN_HPO = True
# HPO_K_FOLDS is deliberately separate from K_FOLDS above: the embedded
# search below (only used when RUN_HPO=True) mirrors the sibling Optuna
# notebook's own fold count, not this notebook's final-training fold count --
# they are different stages with different budgets.
HPO_K_FOLDS      = 3    # only used when RUN_HPO=True
EPOCHS_PER_TRIAL = 50   # only used when RUN_HPO=True
N_TRIALS         = 10   # only used when RUN_HPO=True
N_STARTUP_TRIALS = 3    # only used when RUN_HPO=True

assert Path(PROJECT).is_absolute(), (
    f"PROJECT must be an absolute path, got {PROJECT!r}."
)
Path(PROJECT).mkdir(parents=True, exist_ok=True)

# A source image's PNG bytes don't depend on which fold uses it, so every
# yolo_obb_to_dota(..., convert_to_png=True) call below (both the HPO folds
# and the final-training folds share this notebook's PROJECT) re-encodes each
# image exactly once here instead of once per fold -- see png_cache_dir's
# docstring in src/obb_common.py.
PNG_CACHE_DIR = Path(PROJECT) / "png_cache"

# The PNG re-encode itself (and the redundant per-fold image_size() reads --
# see size_cache below) is pure CPU/PIL work with no GPU dependency, but this
# notebook's own kernel runs on a GPU-attached Colab runtime, so building the
# cache here burns GPU-runtime compute units on work that never touches the
# GPU. Run notebooks/StripRCNN_png_cache_prep_cpu.py ONCE under a CPU-only
# runtime (Runtime > Change runtime type > CPU) to build and archive this
# same cache to Drive as a single zip -- restoring it here is then one
# Drive read + local unzip (like datasetv3_2.zip above) instead of ~1,445
# individual small-file reads through Drive's slow FUSE mount, which turns
# every convert_to_png=True call below into a cheap symlink + label-write.
# Safe to skip -- falls back to building the cache locally (on the GPU
# runtime) if the Drive archive isn't there yet.
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

# Shared across every yolo_obb_to_dota call below (HPO folds and final-
# training folds alike) so a given image's (w, h) is read from disk at most
# once per runtime, not once per fold -- see size_cache's docstring in
# src/obb_common.py.
SIZE_CACHE = {}

CLASS_NAMES = load_class_names(CLASSES_TXT)
NUM_CLASSES = len(CLASS_NAMES)
# DOTA's label format is whitespace-delimited, so class names containing a
# space ("Algal spot") are written underscored on disk and mapped back here.
DOTA_CLASSES = tuple(n.replace(" ", "_") for n in CLASS_NAMES)
print(f"{NUM_CLASSES} classes: {CLASS_NAMES}")

# Best-effort run metadata -- platform, GPU, framework versions, git commit,
# and this arm's own control-variable constants. Never blocks training on
# failure; see src/obb_logging.py's module docstring. Framework versions
# (mmcv/mmdet/torch) live in the strip_rcnn conda env, invisible to this
# kernel's own imports -- not probed here; see the worker's own log instead.
write_run_env(Path(PROJECT) / "run_env.json", repo_root=PROJECT_ROOT, extra={
    "arm": "StripRCNN_E-IRFS_KFold_Final", "SEED": SEED, "K_FOLDS": K_FOLDS,
    "TRAIN_EPOCHS": TRAIN_EPOCHS, "EIRFS_THRESHOLD": EIRFS_THRESHOLD,
    "PROJECT": PROJECT, "RUN_HPO": RUN_HPO,
})

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
# Pool train + val into one fold-able set -- see the note by VAL_IMAGES
# above for why the fixed val split doesn't need to stay separate here.
items = collect_items(IMAGES_DIR, LABELS_DIR) + collect_items(VAL_IMAGES, VAL_LABELS)
print(f"Found {len(items)} labeled images (train + val pooled).")

folds = make_folds(items, K_FOLDS, seed=SEED, shuffle=SHUFFLE_BEFORE_SPLIT)
print("Fold sizes (train/val):", [(len(f["train"]), len(f["val"])) for f in folds])
write_fold_composition(Path(PROJECT) / "fold_composition.csv", folds, CLASS_NAMES, NUM_CLASSES)

FOLD_ROOT = Path(PROJECT) / "folds"
fold_specs = []
for f in tqdm(folds, desc="Materialising final-training folds", unit="fold"):
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
# MMRotate is config-file-driven rather than kwarg-driven, so best_params has
# to be written out as a config file before it can be used to train. This
# builds one config per fold by inheriting the stock Strip R-CNN-S config and
# overriding only the dataset wiring and `best_params` -- the backbone, neck,
# RoI head, and assigner are left exactly as the authors published them.

# %%
def build_config(trial_params, fold_spec, work_dir, epochs, seed=SEED):
    """Write an MMRotate config for one fold and return its path."""
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
               # This dataset has label-free images on purpose -- ~8% of train/
               # is background-only field crops added by
               # analysis/dataset/phase4_background_negatives.py so the model
               # learns to predict nothing on them -- and the YOLO arms train
               # on them (Ultralytics treats an empty label file as
               # background). MMRotate discards them by default, twice over and
               # both times silently: DOTADataset.load_annotations has
               # `if os.path.getsize(ann_file) == 0 and self.filter_empty_gt:
               # continue`, and _filter_imgs drops any that get past that under
               # the same flag, which mmdet 2.x's CustomDataset defaults to
               # True. One flag covers both gates. Applies to train only: val
               # and test are built with test_mode=True, where no filtering
               # happens. See Finding 5 in
               # results/kfold_comparability_audit.md.
               filter_empty_gt=False,
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

# Cosine annealing to lr0 * lrf, matching the Optuna notebook's cos_lr=True
# with its lrf final-LR fraction. MMRotate expresses that as min_lr_ratio.
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
    the search space the same shape as the Optuna notebook's rather than
    adding an optimizer-conditional parameter that Optuna would have to
    model.
    """
    if name == "SGD":
        return f"momentum={momentum:.6f},"
    return f"betas=({momentum:.6f}, 0.999),"

# %% [markdown]
# ### Train / predict one fold
#
# Runs entirely inside the `strip_rcnn` conda env via subprocess (see "file
# setup" above and `src/strip_rcnn_worker.py`'s own docstring) -- this
# notebook's own kernel can no longer import mmcv/mmdet/mmrotate directly on
# modern Colab (Python 3.11+, no torch==1.13.1 wheel).

# %%
def run_fold_via_conda(cfg_path, work_dir, train_labels, test_items, epochs, verbose=False):
    """Train one fold and predict on `test_items`, entirely inside the
    `strip_rcnn` conda env. Writes `work_dir/test_predictions.json` in the
    shared prediction format (read it back with `src.obb_eval.load_predictions`)
    and returns the checkpoint path.
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
    # Teed to a file so the worker's stdout (which carries mmdet's own
    # per-iteration TextLoggerHook lines) survives outside the committed
    # .ipynb cell output -- previously it only ever reached the printed cell.
    with open(work_dir / "train_log.txt", "w") as log_f:
        for line in process.stdout:
            print(line, end="")
            log_f.write(line)
    process.wait()
    if process.returncode != 0:
        raise RuntimeError(
            f"strip_rcnn_worker.py failed for {work_dir} -- see output above."
        )
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
# mean F1 across classes. `src/obb_eval.precision_recall_f1` reproduces that
# same operating point so this arm's P/R/F1 are comparable to the YOLO arms'.
# This notebook used to carry its own inline copy of that function (verified
# byte-identical to the module version before deleting it) -- now imported
# above instead, so per-class P/R/F1 additions land in one place for every arm.

# %% [markdown]
# ## Hyperparameters: run Optuna here, or load a previous search
#
# `RUN_HPO` (set above) picks one of two paths:
# - `False` (default): load the `best_params.json` produced by
#   `notebooks/StripRCNN_E-IRFS_optuna_datasetv3.py`'s own search (from this
#   session's local run, or its Drive archive under `StripRCNN_E-IRFS_Optuna`)
#   -- the documented, cheaper path used by the header markdown.
# - `True`: run that same search space and objective directly against this
#   notebook's own `fold_specs` (already built above) and skip loading
#   entirely. This costs `N_TRIALS * HPO_K_FOLDS` extra fold-trainings at
#   `EPOCHS_PER_TRIAL` epochs before K-fold final training even starts.
#
# `_optimizer_extra` is re-derived from `optimizer`/`momentum` rather than
# stored either way -- `build_config` needs it, but Optuna never sees it as a
# search parameter, so it isn't part of `best_params.json`.

# %%
def load_best_params() -> dict:
    local_path = Path(OPTUNA_PROJECT_LOCAL) / OPTUNA_FINAL_SUBDIR / "best_params.json"
    if local_path.exists():
        print(f"Loading best_params from local Optuna run: {local_path}")
        params = json.loads(local_path.read_text())
    else:
        drive_path = Path(f"{PROJECT_ROOT}/runs/{OPTUNA_RUN_LABEL}/{OPTUNA_FINAL_SUBDIR}/best_params.json")
        if drive_path.exists():
            print(f"Loading best_params from Drive archive: {drive_path}")
            params = json.loads(drive_path.read_text())
        else:
            raise FileNotFoundError(
                f"No best_params.json found locally ({local_path}) or on Drive "
                f"({drive_path}). Run notebooks/StripRCNN_E-IRFS_optuna_datasetv3.py's "
                f"Optuna search first and archive it to Drive (its 'Archive to Drive' "
                f"cell, RUN_LABEL={OPTUNA_RUN_LABEL!r}) before running this K-fold "
                f"final-training notebook, or set RUN_HPO=True above to search here "
                f"instead. If you archived more than once, _unique_drive_dir() may "
                f"have named it '{OPTUNA_RUN_LABEL}2', '{OPTUNA_RUN_LABEL}3', etc. -- "
                f"point OPTUNA_RUN_LABEL at the archive you want."
            )
    params["_optimizer_extra"] = optimizer_extra(params["optimizer"], params["momentum"])
    return params


# Same search space and objective as notebooks/StripRCNN_E-IRFS_optuna_datasetv3.py
# -- kept byte-identical on purpose, just pointed at this notebook's own
# fold_specs instead of the sibling notebook's. Only defined/run when
# RUN_HPO=True.
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


# When RUN_HPO=True, build a SEPARATE train-only fold split for the embedded
# search -- mirrors notebooks/StripRCNN_E-IRFS_optuna_datasetv3.py's own fold
# construction (train/ only, not train/+val/ pooled) so HPO searches over the
# same kind of data the sibling Optuna notebook would, instead of silently
# sharing the final-training folds above (which pool in val/, so HPO and
# final training would otherwise see the same held-out val images).
if RUN_HPO:
    hpo_items = collect_items(IMAGES_DIR, LABELS_DIR)
    print(f"[RUN_HPO] Found {len(hpo_items)} labeled images (train only).")
    hpo_folds = make_folds(hpo_items, HPO_K_FOLDS, seed=SEED, shuffle=SHUFFLE_BEFORE_SPLIT)

    HPO_FOLD_ROOT = Path(PROJECT) / "hpo_folds"
    hpo_fold_specs = []
    for f in tqdm(hpo_folds, desc="Materialising HPO folds", unit="fold"):
        k = f["fold"]
        train_dir = yolo_obb_to_dota(f["train"], HPO_FOLD_ROOT / f"fold{k}" / "train", CLASS_NAMES,
                                     convert_to_png=True, png_cache_dir=PNG_CACHE_DIR,
                                     size_cache=SIZE_CACHE)
        val_dir   = yolo_obb_to_dota(f["val"],   HPO_FOLD_ROOT / f"fold{k}" / "val",   CLASS_NAMES,
                                     convert_to_png=True, png_cache_dir=PNG_CACHE_DIR,
                                     size_cache=SIZE_CACHE)
        hpo_fold_specs.append({
            "fold": k,
            "train_dir": str(train_dir),
            "val_dir": str(val_dir),
            "val_items": f["val"],
            "train_labels": [lbl for _, lbl in f["train"]],
        })
    print("Materialised train-only HPO folds under", HPO_FOLD_ROOT)


def objective(trial):
    p = suggest_params(trial)
    fold_scores, rows, fold_errors = [], [], []

    for spec in hpo_fold_specs:
        k = spec["fold"]
        work_dir = Path(PROJECT) / f"trial_{trial.number:03d}_fold{k}"
        fold_t0 = time.perf_counter()
        try:
            cfg_path = build_config(p, spec, work_dir, EPOCHS_PER_TRIAL)
            run_fold_via_conda(cfg_path, work_dir, spec["train_labels"], spec["val_items"],
                               EPOCHS_PER_TRIAL, verbose=(trial.number == 0))
            collect_epoch_log(work_dir, "mmdet")
            records = load_predictions(work_dir / "test_predictions.json")
            gt = load_ground_truth(spec["val_items"])
            result = evaluate_obb(gt, records, NUM_CLASSES, CLASS_NAMES)
            score = float(result["mAP50-95"])
            if not np.isfinite(score):
                raise RuntimeError(f"shared evaluator returned {score} for fold {k}")
            # Only mAP50-95 (the pruning/selection metric) used to be saved
            # here -- mAP50/mAP75 come for free from the same evaluate_obb()
            # call, and precision/recall/F1 give the same operating-point
            # numbers the final K-fold training loop's own
            # kfold_final_results.csv already reports, so a trial's per-fold
            # CSV is no longer a strictly narrower view than the final one.
            prf = precision_recall_f1(gt, records, NUM_CLASSES)
        except Exception as e:
            print(f"\n{'!'*70}\n[Trial {trial.number}] Fold {k} FAILED: {e}\n{'!'*70}")
            traceback.print_exc(file=sys.stdout)
            sys.stdout.flush()
            fold_errors.append(f"fold {k}: {type(e).__name__}: {e}")
            rows.append({
                "trial": trial.number, "fold": k,
                "mAP50-95": None, "mAP50": None, "mAP75": None,
                "precision": None, "recall": None, "F1": None,
                "seconds": time.perf_counter() - fold_t0, "error": str(e),
            })
            continue

        fold_scores.append(score)
        rows.append({
            "trial": trial.number, "fold": k,
            "mAP50-95": score, "mAP50": result["mAP50"], "mAP75": result["mAP75"],
            "precision": prf["precision"], "recall": prf["recall"], "F1": prf["F1"],
            "seconds": time.perf_counter() - fold_t0, "error": None,
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
                f"Trial 0 failed on all {len(hpo_fold_specs)} folds -- aborting the study "
                f"instead of reporting a meaningless best_params.json. Causes: {detail}"
            )
        raise optuna.TrialPruned(f"all folds failed: {detail}")

    if fold_errors:
        print(f"[Trial {trial.number}] WARNING: {len(fold_errors)}/{len(hpo_fold_specs)} folds "
              f"failed; CV mean is over the {len(fold_scores)} that succeeded.")
    return float(mean(fold_scores))


def run_hpo() -> dict:
    sampler = optuna.samplers.TPESampler(seed=SEED, n_startup_trials=N_STARTUP_TRIALS)
    pruner = optuna.pruners.PercentilePruner(
        percentile=50, n_startup_trials=N_STARTUP_TRIALS, n_warmup_steps=2
    )
    # sqlite-backed (not in-memory) so a Colab disconnect mid-search does not
    # lose every trial's params/values -- same pattern S2A-Net already uses.
    study = open_study(
        "strip_rcnn_kfold", Path(PROJECT) / "hpo.db",
        Path(PROJECT_ROOT) / "runs" / "StripRCNN_E-IRFS_KFold_Final" / "hpo.db",
        sampler, pruner,
    )
    n_remaining = max(0, N_TRIALS - len(study.trials))
    if n_remaining < N_TRIALS:
        print(f"Resuming study: {len(study.trials)} trial(s) already recorded, "
              f"{n_remaining} remaining.")
    if n_remaining:
        study.optimize(objective, n_trials=n_remaining)

    completed = study.get_trials(deepcopy=False, states=(optuna.trial.TrialState.COMPLETE,))
    print(f"Completed trials: {len(completed)} / {N_TRIALS} "
          f"(pruned/failed: {N_TRIALS - len(completed)})")
    if not completed:
        raise RuntimeError("No trial completed successfully; best_params.json NOT written.")
    distinct_values = {round(t.value, 10) for t in completed if t.value is not None}
    if len(distinct_values) == 1 and len(completed) > 1:
        raise RuntimeError(
            f"All {len(completed)} completed trials returned the identical value "
            f"{distinct_values.pop()} -- trial 0 would win by tie-break regardless of "
            f"the search. best_params.json NOT written."
        )

    print("Best CV mAP50-95:", study.best_value)
    print("Best params:", study.best_params)

    final_dir = Path(PROJECT) / "best_final_strip_rcnn"
    final_dir.mkdir(parents=True, exist_ok=True)
    (final_dir / "best_params.json").write_text(json.dumps(study.best_params, indent=2))
    dump_study(study, final_dir)
    params = dict(study.best_params)
    params["_optimizer_extra"] = optimizer_extra(params["optimizer"], params["momentum"])
    return params


best_params = run_hpo() if RUN_HPO else load_best_params()
print("Using best_params:", json.dumps(
    {k: v for k, v in best_params.items() if k != "_optimizer_extra"}, indent=2))

# %% [markdown]
# ## K-fold final training
#
# Trains K models -- one per fold above -- each for the full `TRAIN_EPOCHS`
# using `best_params`, then evaluates every fold's model on the same
# held-out test set through the shared `src/obb_eval.py` evaluator (not
# MMRotate's own `--eval mAP`, which is VOC-style AP at IoU 0.5 only and
# would not be comparable to the other arms). This replaces the Optuna
# notebook's single full-dataset final training run with a mean +/- std
# test score across folds, which is more robust to reviewer questions about
# a single lucky/unlucky split. The fold with the best test mAP50-95 is kept
# as the representative "best" model for downstream use.

# %%
KFOLD_FINAL_DIR = Path(PROJECT) / "kfold_final_strip_rcnn"
KFOLD_FINAL_DIR.mkdir(parents=True, exist_ok=True)

test_items = collect_items(TEST_IMAGES, TEST_LABELS)
print(f"Test images: {len(test_items)}")
gt = load_ground_truth(test_items)

fold_rows = []
fold_errors = []
per_fold_per_class = []
for spec in fold_specs:
    k = spec["fold"]
    work_dir = KFOLD_FINAL_DIR / f"fold{k}"
    cfg_path = build_config(best_params, spec, work_dir, TRAIN_EPOCHS)
    started_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    try:
        with Timer() as train_timer:
            ckpt = run_fold_via_conda(cfg_path, work_dir, spec["train_labels"], test_items,
                                      TRAIN_EPOCHS, verbose=(k == 0))
        collect_epoch_log(work_dir, "mmdet")
        (work_dir / "resolved_params.json").write_text(json.dumps(
            {k_: v for k_, v in best_params.items() if k_ != "_optimizer_extra"}, indent=2))

        with Timer() as eval_timer:
            records = load_predictions(work_dir / "test_predictions.json")
            metrics = evaluate_obb(gt, records, NUM_CLASSES, CLASS_NAMES)
            prf = precision_recall_f1(gt, records, NUM_CLASSES, class_names=CLASS_NAMES)
        metrics["precision"] = prf["precision"]
        metrics["recall"] = prf["recall"]
        metrics["F1"] = prf["F1"]
        for _name, _pc in prf.get("per_class", {}).items():
            metrics["per_class"].setdefault(_name, {}).update(_pc)
        for _name, _c in prf.get("counts", {}).items():
            metrics["per_class"].setdefault(_name, {}).update(_c)
        per_fold_per_class.append((k, metrics["per_class"]))
        (work_dir / "test_metrics.json").write_text(json.dumps(metrics, indent=2))
    except Exception as e:
        # One fold's failure (a transient CUDA OOM, a conda-subprocess
        # hiccup) must not take down the whole 9-fold x TRAIN_EPOCHS run --
        # earlier folds' checkpoints/test_metrics.json are already on disk,
        # and kfold_final_results.csv/summary.json should still get written
        # from whatever folds did succeed. Mirrors every other arm's own
        # K-fold final loop (e.g. RFDETR_OBB_E-IRFS_kfold_datasetv3.py's
        # objective()/final-loop error handling) -- the NaN-tolerant check
        # right below this loop already assumed this existed.
        print(f"\n{'!'*70}\n[Fold {k}] FAILED: {e}\n{'!'*70}")
        traceback.print_exc(file=sys.stdout)
        sys.stdout.flush()
        fold_errors.append(f"fold {k}: {type(e).__name__}: {e}")
        fold_rows.append({
            "fold": k, "test_mAP50-95": None, "test_mAP50": None, "test_mAP75": None,
            "test_precision": None, "test_recall": None, "test_F1": None,
            "ckpt": None, "cfg_path": cfg_path, "run_dir": str(work_dir), "error": str(e),
            "started_at": started_at, "finished_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "train_seconds": None, "eval_seconds": None,
            "epochs_completed": None, "final_train_loss": None,
        })
        continue

    epoch_summary_path = work_dir / "epoch_log_summary.json"
    epoch_summary = (json.loads(epoch_summary_path.read_text())
                     if epoch_summary_path.exists() else {})

    fold_rows.append({
        "fold": k,
        "test_mAP50-95": metrics["mAP50-95"],
        "test_mAP50": metrics["mAP50"],
        "test_mAP75": metrics["mAP75"],
        "test_precision": prf["precision"],
        "test_recall": prf["recall"],
        "test_F1": prf["F1"],
        "ckpt": str(ckpt),
        "cfg_path": cfg_path,
        "run_dir": str(work_dir),
        "error": None,
        "started_at": started_at,
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "train_seconds": train_timer.seconds,
        "eval_seconds": eval_timer.seconds,
        "epochs_completed": epoch_summary.get("epochs_completed"),
        "final_train_loss": epoch_summary.get("final_loss"),
    })
    print(f"[Fold {k}] test mAP50-95={metrics['mAP50-95']:.4f}  mAP50={metrics['mAP50']:.4f}  "
          f"P={prf['precision']:.4f}  R={prf['recall']:.4f}  F1={prf['F1']:.4f}"
          f"  | train {fmt_dt(train_timer.seconds or 0)}")

if fold_errors:
    print(f"WARNING: {len(fold_errors)}/{len(fold_specs)} folds failed: {'; '.join(fold_errors)}")

fold_df = pd.DataFrame(fold_rows)
fold_df.to_csv(KFOLD_FINAL_DIR / "kfold_final_results.csv", index=False)
print(fold_df)
write_per_class_tables(KFOLD_FINAL_DIR, per_fold_per_class, CLASS_NAMES)

# %%
if fold_df["test_mAP50-95"].isna().all():
    raise RuntimeError(
        "Every fold's test mAP50-95 came back NaN -- scroll up for per-fold "
        "training/eval errors. kfold_final_summary.json NOT written."
    )

summary = {
    "n_folds": int(len(fold_df)),
    # Folds that raised and were skipped. The mean/std below are over the
    # SURVIVING folds only, so n_folds alone overstates the sample.
    "n_folds_failed": int(len(fold_errors)),
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
    "best_fold": int(fold_df.loc[fold_df["test_mAP50-95"].idxmax(), "fold"]),
    "best_params": {k: v for k, v in best_params.items() if k != "_optimizer_extra"},
}
print("K-fold final test summary:", json.dumps(summary, indent=2))
(KFOLD_FINAL_DIR / "kfold_final_summary.json").write_text(json.dumps(summary, indent=2))

best_row = fold_df.loc[fold_df["test_mAP50-95"].idxmax()]
FINAL_CKPT = Path(best_row["ckpt"])
FINAL_DIR = Path(best_row["run_dir"])
print(f"Best fold: {summary['best_fold']} -> {FINAL_CKPT}")

# %% [markdown]
# ### test
#
# Re-prints the held-out test metrics for the best fold (already computed
# during the K-fold loop above) plus the full K-fold summary.

# %%
metrics = json.loads((FINAL_DIR / "test_metrics.json").read_text())

print(f"\n=== Strip R-CNN-S | held-out test, best fold {summary['best_fold']} (shared evaluator) ===")
print(f"mAP50-95: {metrics['mAP50-95']:.4f}")
print(f"mAP50   : {metrics['mAP50']:.4f}")
print(f"mAP75   : {metrics['mAP75']:.4f}")
print(f"precision={metrics.get('precision', float('nan')):.4f}  "
      f"recall={metrics.get('recall', float('nan')):.4f}  "
      f"F1={metrics.get('F1', float('nan')):.4f}")
for name, m in metrics["per_class"].items():
    print(f"  {name:15} AP50-95={m['AP50-95']:.4f}  AP50={m['AP50']:.4f}  n={m['n_gt']}")

print("\nFull K-fold summary (all folds):", json.dumps(summary, indent=2))

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
RUN_LABEL = "StripRCNN_E-IRFS_KFold_Final"

LOCAL_RUNS = Path("/content/runs")
DRIVE_DEST = _unique_drive_dir(Path(f"{PROJECT_ROOT}/runs/{RUN_LABEL}"))
DRIVE_DEST.mkdir(parents=True, exist_ok=True)
if DRIVE_DEST.name != RUN_LABEL:
    print(f"NOTE: '{RUN_LABEL}' already exists on Drive -- archiving to '{DRIVE_DEST.name}' instead.")
# The materialised fold directories are symlinked copies of the dataset and
# would either bloat the archive or copy as broken links; the fold *assignment*
# is reproducible from make_folds(items, K_FOLDS, seed=SEED) anyway.
# "hpo_folds" (only materialised when RUN_HPO=True, above) is a SEPARATE
# name from "folds" -- ignore_patterns does exact-name glob matching, not
# substring, so listing only "folds" (and "png_cache") here left the entire
# HPO fold-materialised dataset getting archived every time, unnoticed,
# alongside the K-fold final training dirs this exclusion was meant to
# keep out.
shutil.copytree(LOCAL_RUNS, DRIVE_DEST, dirs_exist_ok=True,
                ignore=shutil.ignore_patterns("folds", "hpo_folds", "png_cache"))

print("Local  :", LOCAL_RUNS, "->", sorted(p.name for p in LOCAL_RUNS.iterdir()))
print("Drive  :", DRIVE_DEST, "->", sorted(p.name for p in DRIVE_DEST.iterdir()))

# %% [markdown]
# ## Re-test from Drive (resume without retraining)
#
# If you're in a fresh runtime and just want to re-verify a previously
# archived run without retraining, run this section after the mount/constants
# cells above and skip the K-fold training loop entirely.

# %%
# Restores the whole /content/runs tree from its RUN_LABEL folder on Drive,
# then reads kfold_final_summary.json to find which fold was best.
RUN_LABEL = "StripRCNN_E-IRFS_KFold_Final"
LOCAL_RUNS = Path("/content/runs")
DRIVE_DEST = Path(f"{PROJECT_ROOT}/runs/{RUN_LABEL}")

LOCAL_RUNS.mkdir(parents=True, exist_ok=True)
shutil.copytree(DRIVE_DEST, LOCAL_RUNS, dirs_exist_ok=True)
print(f"Restored {LOCAL_RUNS} <- {DRIVE_DEST}")

summary = json.loads((KFOLD_FINAL_DIR / "kfold_final_summary.json").read_text())
fold_df = pd.read_csv(KFOLD_FINAL_DIR / "kfold_final_results.csv")
best_row = fold_df.loc[fold_df["test_mAP50-95"].idxmax()]
FINAL_CKPT = Path(best_row["ckpt"])
FINAL_DIR = Path(best_row["run_dir"])
print(f"Best fold: {summary['best_fold']} -> {FINAL_CKPT}")

# %%
# Re-prints the held-out test metrics already computed for the best fold --
# no re-inference needed since predict_split()/evaluate_obb() already wrote
# test_metrics.json during the original K-fold loop.
metrics = json.loads((FINAL_DIR / "test_metrics.json").read_text())

print(f"\n=== Strip R-CNN-S | held-out test, best fold {summary['best_fold']} (shared evaluator) ===")
print(f"mAP50-95: {metrics['mAP50-95']:.4f}")
print(f"mAP50   : {metrics['mAP50']:.4f}")
print(f"mAP75   : {metrics['mAP75']:.4f}")
print(f"precision={metrics.get('precision', float('nan')):.4f}  "
      f"recall={metrics.get('recall', float('nan')):.4f}  "
      f"F1={metrics.get('F1', float('nan')):.4f}")
for name, m in metrics["per_class"].items():
    print(f"  {name:15} AP50-95={m['AP50-95']:.4f}  AP50={m['AP50']:.4f}  n={m['n_gt']}")

print("\nFull K-fold summary (all folds):", json.dumps(summary, indent=2))

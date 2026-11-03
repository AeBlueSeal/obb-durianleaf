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
# # Oriented-DETR (ECCV 2024 Oral) -- No E-IRFS (ablation) + K-fold final training, datasetv3
#
# Ablation twin of `notebooks/OrientedDETR_E-IRFS_kfold_datasetv3.py` with the
# E-IRFS resampling removed entirely -- `datasets.dota.build` is still
# wrapped the same way, but `USE_EIRFS = False` below means
# `EIRFSWeights`/`attach_eirfs` are never constructed/attached, so training
# uses `datasets.dota.build`'s own stock (uniform) sampling. Nothing else
# changes.
#
# **This twin runs its own embedded Optuna search under no-E-IRFS folds**
# (`RUN_HPO=True`, matching the E-IRFS notebook's own live default of
# searching rather than loading) -- the same convention as **S2A-Net's**
# NoEIRFS twin, not the YOLO11m/YOLO26m one (those load the sibling Optuna
# notebook's already-searched `best_params.json` instead). The rationale:
# since the E-IRFS K-fold notebook itself now searches its own
# hyperparameters under E-IRFS sampling rather than inheriting a fixed set,
# holding "the same numbers" constant here would mean holding constant a
# search result that was itself shaped by E-IRFS being active -- letting
# this twin search independently under its own (no-E-IRFS) folds keeps each
# arm's hyperparameters matched to its own sampling regime, at the cost of
# `N_TRIALS x HPO_K_FOLDS` extra fold-trainings before final K-fold training
# starts. Setting `RUN_HPO=False` instead loads `best_params.json` from the
# E-IRFS K-fold notebook's own archived run
# (`RUN_LABEL=OrientedDETR_E-IRFS_KFold_Final`) via `load_best_params()`
# below, kept as a fallback path for a strict-isolation comparison if ever
# wanted.
#
# The fold-construction *mechanics*, the E-IRFS `build()` patch mechanism,
# and the train/predict helpers are otherwise unchanged from the E-IRFS
# notebook -- kept byte-identical on purpose. If the template's
# fold-construction or patch logic changes, re-sync this file the same way.
# `test/` is untouched throughout, as always.

# %% [markdown]
# ## file setup

# %%
# Standalone DINO-style codebase -- no mmcv/mmdet, so this arm is compatible
# with a modern torch, unlike the Strip R-CNN notebook.
# !git clone https://github.com/zhaozeyang108/Oriented-DETR.git /content/Oriented-DETR
# !cd /content/Oriented-DETR && pip install -q -r requirements.txt
# !pip3 install -q optuna pandas numpy pyyaml shapely

# %%
# Self-healing build of Oriented-DETR's two native extensions -- tries the
# import/compiled-file check first, and only builds if actually missing, so
# re-running this cell in an already-set-up runtime is a fast no-op.
#
# 1. MultiScaleDeformableAttention (models/odetr/ops): custom CUDA ops for
#    deformable attention, `setup.py build install` puts it in site-packages.
#    Without it, the encoder silently has no attention kernel (or, since
#    Colab now defaults to Python 3.11+, it just fails to import at all).
# 2. DOTA_devkit/polyiou: a SWIG-wrapped C++ extension `main.py` transitively
#    imports (via engine.py -> util/dota_eval.py). The repo only ships
#    prebuilt _polyiou*.so for Python 3.7/3.8 (checked via GitHub API), which
#    don't match modern Colab runtimes. polyiou_wrap.cxx is already
#    pre-generated in the repo, so no `swig` install is needed -- just a C++
#    compile (`build_ext --inplace`) against the *running* Python's ABI.
#    This one builds in-place rather than installing to site-packages, so we
#    check for the compiled file directly instead of importing (DOTA_devkit
#    isn't on sys.path yet at this point in the notebook).
#
# A build failure here (GPU-enabled runtime + matching CUDA toolkit required)
# prints the actual compiler output before raising, rather than surfacing
# later as a bare "ModuleNotFoundError" mid-training.
import glob
import importlib
import site
import subprocess
import sys
import sysconfig
from pathlib import Path

# MUST come before the MultiScaleDeformableAttention import below, even on a
# fast-path re-run where the extension is already built: its .so links
# against libc10.so/libtorch.so (PyTorch's own shared libraries, confirmed
# via the actual link command's `-L .../torch/lib -lc10 -ltorch ...`), but
# that -L path is only a build-time search path, not an embedded rpath --
# nothing makes those libraries resolvable at import time unless `torch`
# has already been imported in this same process (torch's own package init
# loads them with global symbol visibility, which is how every PyTorch CUDA
# extension normally resolves its dependencies). Importing torch here, before
# it would otherwise first appear later in this notebook, is what fixes a
# real ImportError: "libc10.so: cannot open shared object file" hit live.
import torch  # noqa: F401,E402

REPO = "/content/Oriented-DETR"
OPS_DIR = f"{REPO}/models/odetr/ops"
DOTA_DIR = f"{REPO}/DOTA_devkit"


def _patch_ms_deform_attn_cuda():
    """Patch a real PyTorch-version incompatibility in the repo's own CUDA
    source, applied fresh every time since /content/Oriented-DETR is a fresh
    git clone each runtime (see the "file setup" cell above).

    `ms_deform_attn_cuda.cu` calls `AT_DISPATCH_FLOATING_TYPES(value.type(), ...)`
    at two call sites (forward + backward). `Tensor::type()` returns the
    deprecated `at::DeprecatedTypeProperties`, which no longer implicitly
    converts to `c10::ScalarType` in the PyTorch build Colab now ships --
    confirmed via the actual nvcc error: "no suitable conversion function
    from 'const at::DeprecatedTypeProperties' to 'c10::ScalarType' exists".
    `Tensor::scalar_type()` returns `c10::ScalarType` directly and is the
    standard fix used across the whole Deformable-DETR family of repos for
    this exact issue. Idempotent: a no-op if the pattern is already patched
    or absent.
    """
    cu_path = Path(OPS_DIR) / "src/cuda/ms_deform_attn_cuda.cu"
    src = cu_path.read_text()
    patched = src.replace("AT_DISPATCH_FLOATING_TYPES(value.type(),",
                          "AT_DISPATCH_FLOATING_TYPES(value.scalar_type(),")
    if patched != src:
        cu_path.write_text(patched)
        print(f"Patched value.type() -> value.scalar_type() in {cu_path}")


def _run_build(cmd, cwd):
    print(f"$ {' '.join(cmd)}  (cwd={cwd})")
    result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    print(result.stdout[-4000:])
    if result.returncode != 0:
        print(result.stderr[-4000:])
        raise RuntimeError(
            "Build failed (see compiler output above) -- common cause: the "
            "Colab runtime isn't GPU-enabled, or its CUDA toolkit doesn't "
            "match the installed torch build."
        )


def _ensure_egg_on_path(module_name):
    """`setup.py ... install`'s legacy egg mechanism installs the built
    extension into a `<module_name>*.egg` directory and appends it to
    `easy-install.pth` -- but `.pth` files are only read by the `site`
    module at interpreter startup, so a package installed this way mid-session
    is invisible to the CURRENTLY RUNNING kernel's sys.path (confirmed live:
    the build cell's own log shows "Installed ... Finished processing
    dependencies", yet the very next `import` in the same cell still raised
    ModuleNotFoundError). Search every site-packages dir for the freshly
    installed egg and prepend it to sys.path so the retry import below
    succeeds without needing a runtime restart.
    """
    for site_dir in site.getsitepackages():
        for egg in glob.glob(f"{site_dir}/{module_name}*.egg"):
            if egg not in sys.path:
                sys.path.insert(0, egg)
                print(f"Added {egg} to sys.path")
    importlib.invalidate_caches()


try:
    import MultiScaleDeformableAttention  # noqa: F401
    print("MultiScaleDeformableAttention already importable.")
except ModuleNotFoundError:
    print("MultiScaleDeformableAttention not found -- building...")
    _patch_ms_deform_attn_cuda()
    _run_build([sys.executable, "setup.py", "build", "install"], OPS_DIR)
    _ensure_egg_on_path("MultiScaleDeformableAttention")
    import MultiScaleDeformableAttention  # noqa: F401  (raises again if still missing)
    print("MultiScaleDeformableAttention built and imported OK.")

ext_suffix = sysconfig.get_config_var("EXT_SUFFIX") or ".so"
# Must NOT fall back to a bare "*.so" glob: the repo ships prebuilt
# _polyiou.cpython-37m-*.so / _polyiou.cpython-38-*.so binaries (Python
# 3.7/3.8 only) directly in DOTA_DIR, and a loose ".so" pattern matches
# those too -- confirmed live, this exact fallback made the check report
# "already built" on Python 3.13 while only those incompatible files
# existed, silently skipping the real build. Only trust an exact
# EXT_SUFFIX match for the CURRENTLY RUNNING Python.
built = glob.glob(f"{DOTA_DIR}/_polyiou*{ext_suffix}")
if built:
    print(f"DOTA_devkit/polyiou already built: {built}")
else:
    print("DOTA_devkit/polyiou not built for this Python -- building...")
    _run_build([sys.executable, "setup.py", "build_ext", "--inplace"], DOTA_DIR)
    built = glob.glob(f"{DOTA_DIR}/_polyiou*{ext_suffix}")
    if not built:
        raise RuntimeError(
            f"Build reported success but no _polyiou*{ext_suffix} found in "
            f"{DOTA_DIR} -- check the build output above."
        )
    print(f"DOTA_devkit/polyiou built: {built}")

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
import sys
import time
import traceback
from pathlib import Path
from statistics import mean

import numpy as np
import optuna
import pandas as pd
import torch

ODETR_REPO = "/content/Oriented-DETR"
sys.path.insert(0, ODETR_REPO)
sys.path.insert(0, PROJECT_ROOT)

from src.obb_common import (  # noqa: E402
    EIRFSWeights, attach_eirfs, collect_items, load_class_names, make_folds,
    yolo_obb_to_coco,
)
from src.obb_eval import (  # noqa: E402
    dump_predictions, evaluate_obb, load_ground_truth, poly_iou_matrix,
    precision_recall_f1,
)
from src.obb_logging import (  # noqa: E402
    Timer, collect_epoch_log, dump_study, fmt_dt, open_study,
    write_fold_composition, write_per_class_tables, write_run_env,
)

EIRFS_THRESHOLD = 0.0001
# Ablation switch -- when False, EIRFSWeights is never constructed and
# `_ACTIVE_EIRFS["weights"]` stays None, so `eirfs_build` (below) never
# attaches E-IRFS resampling and training falls back to
# `datasets.dota.build`'s own stock (uniform) sampling. This is the ONLY
# functional difference from OrientedDETR_E-IRFS_kfold_datasetv3.py.
USE_EIRFS = False
TRAIN_EPOCHS = 250

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

# 4-scale ResNet-50 single-scale DOTA config -- the R50 arm, chosen because it
# is the closest capacity match to YOLO11m and Strip R-CNN-S. The Swin-T/L
# configs in the same directory would confound architecture with backbone
# capacity.
BASE_CONFIG = f"{ODETR_REPO}/config/dota/4scale_res50_ss_dota.py"
PRETRAINED  = "/content/odetr_r50_coco_pretrain.pth"   # COCO-pretrained, from the repo README

# --- Training knobs: held IDENTICAL to the E-IRFS K-fold notebook ---
DEVICE      = 0
# Suffixed _noeirfs so this twin's own output never collides with the
# E-IRFS notebook's PROJECT.
PROJECT     = "/content/runs/kfold_final_oriented_detr_noeirfs"
SEED             = 42
K_FOLDS          = 9
SHUFFLE_BEFORE_SPLIT = True

# Only used when RUN_HPO=False below -- points at the E-IRFS K-fold
# notebook's OWN archived run (RUN_LABEL=OrientedDETR_E-IRFS_KFold_Final),
# NOT the sibling Optuna notebook, since that E-IRFS notebook's own live
# default is RUN_HPO=True (its own embedded search, E-IRFS folds).
SOURCE_PROJECT_LOCAL = "/content/runs/kfold_final_oriented_detr"
SOURCE_RUN_LABEL = "OrientedDETR_E-IRFS_KFold_Final"
SOURCE_FINAL_SUBDIR = "best_final_oriented_detr"

# On by default -- see the header's rationale: this twin runs its OWN
# embedded Optuna search under no-E-IRFS folds (S2A-Net-style), matching the
# E-IRFS notebook's own live default of searching rather than loading.
# Setting this False instead loads best_params.json from
# SOURCE_RUN_LABEL's archive above (strict-isolation fallback).
RUN_HPO = True
# HPO_K_FOLDS is deliberately separate from K_FOLDS above: the embedded
# search below (only used when RUN_HPO=True) mirrors the sibling Optuna
# notebook's own fold count, not this notebook's final-training fold count --
# they are different stages with different budgets.
HPO_K_FOLDS      = 3    # only used when RUN_HPO=True
EPOCHS_PER_TRIAL = 50   # only used when RUN_HPO=True
N_TRIALS         = 10   # only used when RUN_HPO=True
N_STARTUP_TRIALS = 3    # only used when RUN_HPO=True

assert Path(PROJECT).is_absolute(), f"PROJECT must be absolute, got {PROJECT!r}"
Path(PROJECT).mkdir(parents=True, exist_ok=True)

CLASS_NAMES = load_class_names(CLASSES_TXT)
NUM_CLASSES = len(CLASS_NAMES)
print(f"{NUM_CLASSES} classes: {CLASS_NAMES}")

# Best-effort run metadata -- platform, GPU, framework versions, git commit,
# and this arm's own control-variable constants. Never blocks training on
# failure; see src/obb_logging.py's module docstring.
write_run_env(Path(PROJECT) / "run_env.json", repo_root=PROJECT_ROOT, extra={
    "arm": "OrientedDETR_NoEIRFS_KFold_Final", "SEED": SEED, "K_FOLDS": K_FOLDS,
    "TRAIN_EPOCHS": TRAIN_EPOCHS, "USE_EIRFS": USE_EIRFS,
    "PROJECT": PROJECT, "PRETRAINED": PRETRAINED, "RUN_HPO": RUN_HPO,
})

# %% [markdown]
# ### Dataset materialisation
#
# Same folds as every other arm (`make_folds(..., seed=SEED)` over the same
# sorted item list). The encoding differs: Oriented-DETR reads a COCO JSON
# whose `bbox` field holds an **8-value polygon** rather than standard COCO
# `[x, y, w, h]` -- `datasets/dota.py` builds a 9-wide row from
# `[category_id] + bbox` and `poly2csl` asserts the polygon part is 8 wide.
# `yolo_obb_to_coco` writes exactly that.
#
# The directory names below (`trainval2017`, `val2017`, `annotations/`) are
# hardcoded in the repo's `build()` and cannot be configured; they are a COCO
# naming convention inherited from DINO, not a statement about the data.

# %%
# Pool train + val into one fold-able set -- see the note by VAL_IMAGES
# above for why the fixed val split doesn't need to stay separate here.
items = collect_items(IMAGES_DIR, LABELS_DIR) + collect_items(VAL_IMAGES, VAL_LABELS)
print(f"Found {len(items)} labeled images (train + val pooled).")

folds = make_folds(items, K_FOLDS, seed=SEED, shuffle=SHUFFLE_BEFORE_SPLIT)
print("Fold sizes (train/val):", [(len(f["train"]), len(f["val"])) for f in folds])
write_fold_composition(Path(PROJECT) / "fold_composition.csv", folds, CLASS_NAMES, NUM_CLASSES)


def materialise_coco_split(fold_items, root, split_dirname, ann_name):
    """Symlink images into a COCO-style split dir and write its annotation JSON."""
    root = Path(root)
    img_dir = root / split_dirname
    img_dir.mkdir(parents=True, exist_ok=True)
    (root / "annotations").mkdir(parents=True, exist_ok=True)
    for img_path, _ in fold_items:
        dst = img_dir / Path(img_path).name
        if not dst.exists():
            os.symlink(img_path, dst)
    yolo_obb_to_coco(fold_items, root / "annotations" / ann_name, CLASS_NAMES)
    return root


FOLD_ROOT = Path(PROJECT) / "folds"
fold_specs = []
for f in folds:
    k = f["fold"]
    root = FOLD_ROOT / f"fold{k}"
    materialise_coco_split(f["train"], root, "trainval2017", "instances_trainval2017.json")
    materialise_coco_split(f["val"], root, "val2017", "instances_val2017.json")
    fold_specs.append({
        "fold": k,
        "coco_path": str(root),
        "val_items": f["val"],
        "train_labels": [lbl for _, lbl in f["train"]],
    })
print("Materialised COCO-format folds under", FOLD_ROOT)

# %% [markdown]
# ### Override sampling behavior
#
# (Or, in this ablation twin, skip it.) E-IRFS is attached by wrapping
# `datasets.dota.build` so that the dataset it returns for
# `image_set == "train"` has its `__getitem__` index-substituted -- same
# mechanism as `attach_eirfs` in `src/obb_common.py`: sampling with
# replacement, epoch length unchanged, val and test untouched. The wrapper
# below is kept exactly as-is (unlike removing it entirely) so this file
# stays a minimal diff from the E-IRFS notebook -- `USE_EIRFS = False` above
# means `_ACTIVE_EIRFS["weights"]` is never set to anything but `None`, so
# `eirfs_build`'s `is not None` check simply never attaches anything.
#
# Wrapping `build` rather than patching `CocoDetection.__getitem__` on the
# class matters here -- `build()` returns the *same class* for train and val,
# so a class-level patch would silently apply resampling to validation too,
# which would make the fold score meaningless.

# %%
import datasets.dota as odetr_dota  # noqa: E402

# Guard against re-running this cell in the same kernel (common after fixing
# an earlier cell without restarting the runtime): a second run would
# capture the ALREADY-WRAPPED `eirfs_build` as `_ORIGINAL_BUILD`, and since
# `eirfs_build` looks up the `_ORIGINAL_BUILD` global by name at call time
# (not by closure value), the new wrapper would end up calling the old
# wrapper, which calls itself -- infinite recursion that only surfaces once
# training reaches `build_dataset()`, not at cell-run time.
if not getattr(odetr_dota.build, "_is_eirfs_wrapper", False):
    _ORIGINAL_BUILD = odetr_dota.build
_ACTIVE_EIRFS = {"weights": None}


def eirfs_build(image_set, args):
    dataset = _ORIGINAL_BUILD(image_set, args)
    if image_set in ("train", "train_reg") and _ACTIVE_EIRFS["weights"] is not None:
        attach_eirfs(dataset, _ACTIVE_EIRFS["weights"], train_mode=True)
        print(f"E-IRFS attached to '{image_set}' dataset ({len(dataset)} images)")
    return dataset


eirfs_build._is_eirfs_wrapper = True
odetr_dota.build = eirfs_build
# datasets/__init__.py did `from .dota import build as build_dota` at import
# time, binding the original function object -- rebinding the module attribute
# above is not enough on its own.
import datasets as odetr_datasets  # noqa: E402
odetr_datasets.build_dota = eirfs_build

# %% [markdown]
# ### Only persist the final-epoch weight; log every epoch's metrics instead
#
# `main.py`'s training loop unconditionally `torch.save`s the full model +
# optimizer state (`checkpoint.pth`, ~500MB) after EVERY epoch, purely to
# support mid-training resume -- a capability this notebook's K-fold/HPO
# loop never uses (a crashed fold is just retried from scratch). Across
# `EPOCHS_PER_TRIAL * N_TRIALS * HPO_K_FOLDS` HPO epochs plus `K_FOLDS` final
# epochs, that is a lot of redundant full-weight disk I/O for files that are
# overwritten before ever being read -- `train_one_fold`/`predict_split` only
# load the checkpoint once training has fully finished.
#
# Fixed the same way as the E-IRFS `build()` hook above: wrap two functions
# `main.py` calls by bare name (so patching the *source* module, e.g.
# `util.misc.save_on_master`, is enough -- `main.py`'s own `utils` name is a
# live reference to that same module object, re-looked-up by attribute on
# every call, unlike the direct-name-imported `train_one_epoch`, which needs
# patching on the `main` module object itself):
#
# 1. `utils.save_on_master` -- now only actually writes when the checkpoint
#    dict's own `epoch` equals `args.epochs - 1` (both already embedded in
#    the dict `main.py` passes), i.e. the true final epoch. Every
#    intermediate call becomes a no-op; resume-from-crash is not a path this
#    pipeline exercises, so nothing is lost.
# 2. `train_one_epoch` -- wrapped to temporarily track the `MetricLogger`
#    instance it creates (via a scoped `MetricLogger.__init__` patch,
#    restored immediately after the call), then append that epoch's
#    `{epoch, **global_avg per metric}` -- the same averaged numbers already
#    shown in each epoch's final progress-bar line -- as one JSON line to
#    `<output_dir>/epoch_metrics.jsonl`. Cheap (a few KB/epoch) where the
#    checkpoint write was ~500MB/epoch.

# %%
import util.misc as odetr_misc  # noqa: E402
import main as odetr_main_module  # noqa: E402

if not getattr(odetr_misc.save_on_master, "_is_ckpt_wrapper", False):
    _ORIGINAL_SAVE_ON_MASTER = odetr_misc.save_on_master


def _save_on_master_final_epoch_only(obj, checkpoint_path, *a, **kw):
    is_final_epoch = (
        isinstance(obj, dict) and "epoch" in obj and "args" in obj
        and obj["epoch"] == obj["args"].epochs - 1
    )
    if not is_final_epoch:
        return
    return _ORIGINAL_SAVE_ON_MASTER(obj, checkpoint_path, *a, **kw)


_save_on_master_final_epoch_only._is_ckpt_wrapper = True
odetr_misc.save_on_master = _save_on_master_final_epoch_only

if not getattr(odetr_main_module.train_one_epoch, "_is_metrics_wrapper", False):
    _ORIGINAL_TRAIN_ONE_EPOCH = odetr_main_module.train_one_epoch


def _train_one_epoch_with_metrics(*a, **kw):
    epoch = a[5] if len(a) > 5 else kw.get("epoch")
    train_args = kw.get("args")
    output_dir = getattr(train_args, "output_dir", None) if train_args is not None else None

    created_loggers = []
    _orig_ml_init = odetr_misc.MetricLogger.__init__

    def _tracking_init(self, *ia, **ikw):
        _orig_ml_init(self, *ia, **ikw)
        created_loggers.append(self)

    odetr_misc.MetricLogger.__init__ = _tracking_init
    try:
        result = _ORIGINAL_TRAIN_ONE_EPOCH(*a, **kw)
    finally:
        odetr_misc.MetricLogger.__init__ = _orig_ml_init

    if created_loggers and output_dir is not None:
        stats = {"epoch": epoch}
        for name, meter in created_loggers[-1].meters.items():
            if meter.count > 0:
                stats[name] = meter.global_avg
        with open(Path(output_dir) / "epoch_metrics.jsonl", "a") as f:
            f.write(json.dumps(stats) + "\n")
    return result


_train_one_epoch_with_metrics._is_metrics_wrapper = True
odetr_main_module.train_one_epoch = _train_one_epoch_with_metrics

# %% [markdown]
# ### Train / predict one fold

# %%
def make_args(coco_path, output_dir, params, epochs, eval_only=False, resume=""):
    """Build the argparse Namespace main.py expects.

    Goes through the repo's own `get_args_parser()` with a CLI-shaped argv
    rather than hand-constructing a Namespace, so any default this notebook
    does not set stays at whatever the repo intends -- hand-built Namespaces
    are how DETR-style codebases end up silently running without EMA or
    without the auxiliary decoder losses.
    """
    from main import get_args_parser

    options = [
        f"epochs={epochs}",
        # Repo default is save_checkpoint_interval=1, which writes an extra
        # full-weight `checkpointNNNN.pth` every single epoch on top of the
        # always-overwritten `checkpoint.pth` -- and nothing downstream
        # (train_one_fold/predict_split) ever reads those numbered files,
        # only the final `checkpoint.pth`. Setting the interval past the
        # last epoch means that per-epoch snapshot is never written; only
        # the per-epoch log.txt (loss/metrics) and the one necessary final
        # checkpoint remain.
        f"save_checkpoint_interval={epochs + 1}",
        f"lr={params['lr0']}",
        f"lr_backbone={params['lr0'] * params['lr_backbone_mult']}",
        f"weight_decay={params['weight_decay']}",
        f"batch_size={params['batch']}",
        f"img_size={params['imgsz']}",
        f"num_classes={NUM_CLASSES + 1}",   # +1: labels are 1-based, id 0 is background
        f"dn_labelbook_size={NUM_CLASSES + 1}",
        f"lr_drop={max(1, int(epochs * params['lr_drop_frac']))}",
        f"clip_max_norm={params['clip_max_norm']}",
        # DETR set-matching cost weights -- this arm's analogue of the YOLO
        # template's box/cls/dfl loss weights.
        f"set_cost_class={params['set_cost_class']}",
        f"cls_loss_coef={params['cls_loss_coef']}",
        # YOLOv5-derived augmentation, same names/meaning as the YOLO template
        "augment=True",
        f"degrees={params['degrees']}",
        f"translate={params['translate']}",
        f"scale={params['scale']}",
        f"shear={params['shear']}",
        f"perspective={params['perspective']}",
        f"hsv_h={params['hsv_h']}",
        f"hsv_s={params['hsv_s']}",
        f"hsv_v={params['hsv_v']}",
        f"flipud={params['flipud']}",
        f"fliplr={params['fliplr']}",
        f"mosaic={params['mosaic'] > 0}",
        f"mosaic_p={params['mosaic']}",
    ]
    argv = [
        "-c", BASE_CONFIG,
        "--coco_path", str(coco_path),
        "--output_dir", str(output_dir),
        "--device", f"cuda:{DEVICE}",
        "--seed", str(SEED),
        "--num_workers", "2",
        "--options", *options,
    ]
    if resume:
        argv += ["--resume", resume]
    elif Path(PRETRAINED).exists():
        argv += ["--pretrain_model_path", PRETRAINED]
    if eval_only:
        argv += ["--eval"]
    return get_args_parser().parse_args(argv)


def merge_config_into_args(args):
    """Reproduce the config-file merge `main.main()` performs on `args`
    before it ever calls `build_model_main()`.

    `get_args_parser()` alone does NOT set `args.modelname` (or `backbone`,
    `hidden_dim`, and every other architecture field) -- those only exist in
    the config file (`-c`/`--config_file`, e.g. `4scale_res50_ss_dota.py`)
    and are merged onto `args` by `main()` via `SLConfig.fromfile(...)` +
    `setattr`. `train_one_fold()` goes through `main()` (`odetr_main(args)`)
    so it gets this for free; `predict_split()` below calls
    `build_model_main(args)` directly (to run pure inference without kicking
    off a training loop), which skips that merge entirely and crashes with
    `AttributeError: 'Namespace' object has no attribute 'modelname'` --
    confirmed against main.py's own source (github.com/zhaozeyang108/Oriented-DETR).
    """
    from util.slconfig import SLConfig

    cfg = SLConfig.fromfile(args.config_file)
    if args.options is not None:
        cfg.merge_from_dict(args.options)
    cfg_dict = cfg._cfg_dict.to_dict()
    args_vars = vars(args)
    for k, v in cfg_dict.items():
        if k not in args_vars:
            setattr(args, k, v)
    if not getattr(args, "use_ema", None):
        args.use_ema = False
    if not getattr(args, "debug", None):
        args.debug = False
    return args


def train_one_fold(coco_path, output_dir, params, epochs, eirfs_weights):
    """Run main.main() for one fold (with E-IRFS active only if
    `eirfs_weights` is not None -- always None in this ablation twin)."""
    from main import main as odetr_main

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    _ACTIVE_EIRFS["weights"] = eirfs_weights
    try:
        args = make_args(coco_path, output_dir, params, epochs)
        odetr_main(args)
    finally:
        # Always clear, so a crashed trial cannot leave a stale weight vector
        # attached to the next fold's dataset -- that would resample fold k+1
        # under fold k's class frequencies and be almost invisible in the logs.
        _ACTIVE_EIRFS["weights"] = None
    ckpt = Path(output_dir) / "checkpoint.pth"
    if not ckpt.exists():
        raise RuntimeError(f"no checkpoint at {ckpt} -- training produced no weights")
    return ckpt


@torch.no_grad()
def predict_split(ckpt, coco_path, items, params, conf=0.001):
    """Run inference and return records in the shared prediction format.

    Replicates `engine.detect()`'s postprocessing rather than calling it,
    because `detect()` writes DOTA submission txts keyed by class name and
    discards the per-image association this notebook needs. The parts that
    matter are copied verbatim:
      * `postprocessors['bbox'](outputs, orig_target_sizes, conf)` returns
        `scores`, `labels`, `boxes` (points), `angles`;
      * the rotated quadrilateral is `engine.poly2bbox(box, angle)`;
      * `orig_target_sizes` makes those coordinates ORIGINAL-image pixels,
        which is the frame `src/obb_eval.py` scores in.

    `labels` are 1-based (`class_id = i + 1` in `detect()`), so 1 is
    subtracted to reach this project's 0-based class indices. Getting that
    wrong shifts every class by one and produces a plausible-looking but
    entirely wrong per-class table.
    """
    from datasets import build_dataset
    from engine import poly2bbox
    from main import build_model_main
    import util.misc as utils

    args = make_args(coco_path, Path(ckpt).parent, params, TRAIN_EPOCHS, eval_only=True)
    args = merge_config_into_args(args)
    model, _, postprocessors = build_model_main(args)
    # weights_only=False: PyTorch 2.6 defaults torch.load to weights_only=True,
    # which rejects this repo's checkpoints (they pickle an argparse.Namespace
    # under state["args"]). These checkpoints are our own K-fold/HPO training
    # output, not third-party files, so full unpickling is safe here.
    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"])
    model.to(args.device).eval()

    dataset_val = build_dataset(image_set="val", args=args)
    sampler = torch.utils.data.SequentialSampler(dataset_val)
    loader = torch.utils.data.DataLoader(
        dataset_val, batch_size=1, sampler=sampler, drop_last=False,
        collate_fn=utils.collate_fn, num_workers=2,
    )
    # image_id -> stem, so predictions can be keyed the same way the shared
    # ground truth is.
    id_to_stem = {img["id"]: Path(img["file_name"]).stem
                  for img in dataset_val.coco.dataset["images"]}

    records = []
    for samples, targets in loader:
        samples = samples.to(args.device)
        outputs = model(samples)
        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0).to(args.device)
        results = postprocessors["bbox"](outputs, orig_target_sizes, conf)
        for i, result in enumerate(results):
            scores = result["scores"].cpu().numpy()
            labels = result["labels"].cpu().numpy()
            boxes, angles = result["boxes"], result["angles"]
            if boxes.shape[0] == 0:
                continue
            stem = id_to_stem[int(targets[i]["image_id"].item())]
            for box, angle, label, score in zip(boxes, angles, labels, scores):
                poly = poly2bbox(box, angle).cpu().numpy().reshape(-1)
                assert poly.size == 8, (
                    f"poly2bbox returned {poly.size} values, expected 8 -- the "
                    f"point-axis decoding has changed upstream and these "
                    f"polygons would be scored against the wrong geometry."
                )
                records.append({"image": stem, "cls": int(label) - 1,
                                "score": float(score),
                                "poly": [float(v) for v in poly]})
    return records

# %% [markdown]
# ## Hyperparameters: run Optuna here, or load from the E-IRFS K-fold notebook's own archive
#
# `RUN_HPO` (set above, default `True`) picks one of two paths:
# - `True` (default -- see header): run the same search space and objective
#   directly against this notebook's own `fold_specs`, under no-E-IRFS
#   sampling, instead of loading. Hyperparameters are searched independently
#   per arm, matched to each arm's own sampling regime (S2A-Net convention).
# - `False` (strict-isolation fallback -- see header): load the
#   `best_params.json` produced by
#   `notebooks/OrientedDETR_E-IRFS_kfold_datasetv3.py`'s own embedded search
#   (from this session's local run, or its Drive archive under
#   `OrientedDETR_E-IRFS_KFold_Final`), holding hyperparameters constant so
#   E-IRFS is the only varying factor.

# %%
def load_best_params() -> dict:
    local_path = Path(SOURCE_PROJECT_LOCAL) / SOURCE_FINAL_SUBDIR / "best_params.json"
    if local_path.exists():
        print(f"Loading best_params from local E-IRFS K-fold run: {local_path}")
        return json.loads(local_path.read_text())

    drive_path = Path(
        f"{PROJECT_ROOT}/runs/{SOURCE_RUN_LABEL}/"
        f"{Path(SOURCE_PROJECT_LOCAL).name}/{SOURCE_FINAL_SUBDIR}/best_params.json"
    )
    if drive_path.exists():
        print(f"Loading best_params from Drive archive: {drive_path}")
        return json.loads(drive_path.read_text())

    raise FileNotFoundError(
        f"No best_params.json found locally ({local_path}) or on Drive "
        f"({drive_path}). Run notebooks/OrientedDETR_E-IRFS_kfold_datasetv3.py "
        f"first and archive it to Drive (its 'Archive to Drive' cell, "
        f"RUN_LABEL={SOURCE_RUN_LABEL!r}) before running this NoEIRFS "
        f"K-fold final-training notebook, or leave RUN_HPO=True above to search "
        f"independently here instead (this twin's default -- see header). If "
        f"you archived more than once, _unique_drive_dir() may have named it "
        f"'{SOURCE_RUN_LABEL}2', '{SOURCE_RUN_LABEL}3', etc. -- point "
        f"SOURCE_RUN_LABEL at the archive you want."
    )


# Same search space and objective as notebooks/OrientedDETR_E-IRFS_optuna_datasetv3.py
# -- kept byte-identical on purpose, just pointed at this notebook's own
# fold_specs instead of the sibling notebook's. Only defined/run when
# RUN_HPO=True.
def suggest_params(trial):
    return {
        "lr0": trial.suggest_float("lr0", 1e-5, 1e-3, log=True),
        "lr_backbone_mult": trial.suggest_float("lr_backbone_mult", 0.01, 1.0, log=True),
        "lr_drop_frac": trial.suggest_float("lr_drop_frac", 0.5, 0.95),
        "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True),
        "clip_max_norm": trial.suggest_float("clip_max_norm", 0.1, 1.0),
        "set_cost_class": trial.suggest_float("set_cost_class", 1.0, 4.0),
        "cls_loss_coef": trial.suggest_float("cls_loss_coef", 0.5, 4.0),
        "imgsz": trial.suggest_categorical("imgsz", [320, 640, 832, 1024]),
        "batch": trial.suggest_categorical("batch", [1, 2, 4]),
        "degrees": trial.suggest_float("degrees", 0.0, 180.0),
        "translate": trial.suggest_float("translate", 0.0, 0.2),
        "scale": trial.suggest_float("scale", 0.0, 1.0),
        "shear": trial.suggest_float("shear", 0.0, 5.0),
        "perspective": trial.suggest_float("perspective", 0.0, 0.001),
        "hsv_h": trial.suggest_float("hsv_h", 0.0, 0.015),
        "hsv_s": trial.suggest_float("hsv_s", 0.0, 0.7),
        "hsv_v": trial.suggest_float("hsv_v", 0.0, 0.7),
        "flipud": trial.suggest_float("flipud", 0.0, 0.5),
        "fliplr": trial.suggest_float("fliplr", 0.0, 0.9),
        "mosaic": trial.suggest_float("mosaic", 0.0, 1.0),
    }


# When RUN_HPO=True, build a SEPARATE train-only fold split for the embedded
# search -- mirrors notebooks/OrientedDETR_E-IRFS_optuna_datasetv3.py's own
# fold construction (train/ only, not train/+val/ pooled) so HPO searches
# over the same kind of data the sibling Optuna notebook would, instead of
# silently sharing the final-training folds above (which pool in val/, so
# HPO and final training would otherwise see the same held-out val images).
if RUN_HPO:
    hpo_items = collect_items(IMAGES_DIR, LABELS_DIR)
    print(f"[RUN_HPO] Found {len(hpo_items)} labeled images (train only).")
    hpo_folds = make_folds(hpo_items, HPO_K_FOLDS, seed=SEED, shuffle=SHUFFLE_BEFORE_SPLIT)

    HPO_FOLD_ROOT = Path(PROJECT) / "hpo_folds"
    hpo_fold_specs = []
    for f in hpo_folds:
        k = f["fold"]
        root = HPO_FOLD_ROOT / f"fold{k}"
        materialise_coco_split(f["train"], root, "trainval2017", "instances_trainval2017.json")
        materialise_coco_split(f["val"], root, "val2017", "instances_val2017.json")
        hpo_fold_specs.append({
            "fold": k,
            "coco_path": str(root),
            "val_items": f["val"],
            "train_labels": [lbl for _, lbl in f["train"]],
        })
    print("Materialised train-only HPO folds under", HPO_FOLD_ROOT)


def objective(trial):
    p = suggest_params(trial)
    fold_scores, rows, fold_errors = [], [], []

    for spec in hpo_fold_specs:
        k = spec["fold"]
        out_dir = Path(PROJECT) / f"trial_{trial.number:03d}_fold{k}"
        fold_t0 = time.perf_counter()
        try:
            weights = (EIRFSWeights(spec["train_labels"], NUM_CLASSES, CLASS_NAMES,
                                   t=EIRFS_THRESHOLD, verbose=(trial.number == 0))
                       if USE_EIRFS else None)
            ckpt = train_one_fold(spec["coco_path"], out_dir, p, EPOCHS_PER_TRIAL, weights)
            collect_epoch_log(out_dir, "detr")
            records = predict_split(ckpt, spec["coco_path"], spec["val_items"], p)
            gt = load_ground_truth(spec["val_items"])
            result = evaluate_obb(gt, records, NUM_CLASSES, CLASS_NAMES)
            prf = precision_recall_f1(gt, records, NUM_CLASSES)
            score = float(result["mAP50-95"])
            if not np.isfinite(score):
                raise RuntimeError(f"shared evaluator returned {score} for fold {k}")
        except Exception as e:
            print(f"\n{'!'*70}\n[Trial {trial.number}] Fold {k} FAILED: {e}\n{'!'*70}")
            traceback.print_exc(file=sys.stdout)
            sys.stdout.flush()
            fold_errors.append(f"fold {k}: {type(e).__name__}: {e}")
            rows.append({"trial": trial.number, "fold": k, "mAP50-95": None, "mAP50": None,
                        "mAP75": None, "precision": None, "recall": None, "F1": None,
                        "seconds": time.perf_counter() - fold_t0, "error": str(e)})
            continue

        fold_scores.append(score)
        rows.append({"trial": trial.number, "fold": k, "mAP50-95": score,
                    "mAP50": result["mAP50"], "mAP75": result["mAP75"],
                    "precision": prf["precision"], "recall": prf["recall"], "F1": prf["F1"],
                    "seconds": time.perf_counter() - fold_t0, "error": None})
        trial.report(mean(fold_scores), step=k + 1)
        # Pruning is deliberately NOT enabled -- see the same note in
        # OrientedDETR_E-IRFS_kfold_datasetv3.py's objective(). Not a
        # bug/oversight; do not "fix" it here either.

    pd.DataFrame(rows).to_csv(Path(PROJECT) / f"cv_trial_{trial.number:03d}.csv", index=False)

    if not fold_scores:
        detail = "; ".join(fold_errors)
        if trial.number == 0:
            raise RuntimeError(
                f"Trial 0 failed on all {len(hpo_fold_specs)} folds -- aborting the study. "
                f"Causes: {detail}"
            )
        raise optuna.TrialPruned(f"all folds failed: {detail}")

    if fold_errors:
        print(f"[Trial {trial.number}] WARNING: {len(fold_errors)}/{len(hpo_fold_specs)} folds failed.")
    return float(mean(fold_scores))


def run_hpo() -> dict:
    sampler = optuna.samplers.TPESampler(seed=SEED, n_startup_trials=N_STARTUP_TRIALS)
    pruner = optuna.pruners.PercentilePruner(
        percentile=50, n_startup_trials=N_STARTUP_TRIALS, n_warmup_steps=2
    )
    # sqlite-backed (not in-memory) so a Colab disconnect mid-search does not
    # lose every trial's params/values -- same pattern S2A-Net already uses.
    study = open_study(
        "oriented_detr_kfold_noeirfs", Path(PROJECT) / "hpo.db",
        Path(PROJECT_ROOT) / "runs" / "OrientedDETR_NoEIRFS_KFold_Final" / "hpo.db",
        sampler, pruner,
    )
    n_remaining = max(0, N_TRIALS - len(study.trials))
    if n_remaining < N_TRIALS:
        print(f"Resuming study: {len(study.trials)} trial(s) already recorded, "
              f"{n_remaining} remaining.")
    if n_remaining:
        study.optimize(objective, n_trials=n_remaining)

    completed = study.get_trials(deepcopy=False, states=(optuna.trial.TrialState.COMPLETE,))
    print(f"Completed trials: {len(completed)} / {N_TRIALS}")
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

    final_dir = Path(PROJECT) / "best_final_oriented_detr_noeirfs"
    final_dir.mkdir(parents=True, exist_ok=True)
    (final_dir / "best_params.json").write_text(json.dumps(study.best_params, indent=2))
    dump_study(study, final_dir)
    return study.best_params


best_params = run_hpo() if RUN_HPO else load_best_params()
print("Using best_params:", json.dumps(best_params, indent=2))

# %% [markdown]
# ## Precision / recall / F1 at the best-F1 confidence threshold
#
# `evaluate_obb()` (the shared AP metric) deliberately has no single
# confidence operating point -- AP integrates over the whole precision/recall
# curve. Ultralytics' own `box.mp`/`box.mr`/`box.f1` (used by the YOLO K-fold
# notebooks) DO report at one point: the confidence threshold that maximises
# mean F1 across classes. `src/obb_eval.precision_recall_f1` reproduces that
# same operating point so this arm's P/R/F1 are comparable to the YOLO arms'.

# %% [markdown]
# ## K-fold final training
#
# Trains K models -- one per fold above -- each for the full `TRAIN_EPOCHS`
# using `best_params` and, since `USE_EIRFS=False`, no E-IRFS resampling
# (stock `datasets.dota.build` sampling), then evaluates every fold's model
# on the same held-out test set through the shared `src/obb_eval.py`
# evaluator. The fold with the best test mAP50-95 is kept as the
# representative "best" model for downstream use.
#
# `build_dataset` reads the "val" split, so test images are staged under
# `val2017` here, and the full labeled train set under `trainval2017` (only
# its existence matters for `eval_only=True` inference, not its content).

# %%
KFOLD_FINAL_DIR = Path(PROJECT) / "kfold_final_oriented_detr_noeirfs"
KFOLD_FINAL_DIR.mkdir(parents=True, exist_ok=True)

test_items = collect_items(TEST_IMAGES, TEST_LABELS)
print(f"Test images: {len(test_items)}")

TEST_ROOT = FOLD_ROOT / "test"
materialise_coco_split(test_items, TEST_ROOT, "val2017", "instances_val2017.json")
materialise_coco_split(items, TEST_ROOT, "trainval2017", "instances_trainval2017.json")

gt = load_ground_truth(test_items)

fold_rows = []
fold_errors = []
per_fold_per_class = []
for spec in fold_specs:
    k = spec["fold"]
    out_dir = KFOLD_FINAL_DIR / f"fold{k}"
    started_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    # One fold's failure must not take down the whole 9-fold run and leave no
    # CSV/summary behind -- the same guard Strip R-CNN, RF-DETR-OBB and
    # S2A-Net already carry (it was a real regression in RF-DETR-OBB), and
    # which the all-NaN check below already assumed was here.
    try:
        weights = (EIRFSWeights(spec["train_labels"], NUM_CLASSES, CLASS_NAMES,
                               t=EIRFS_THRESHOLD, verbose=(k == 0))
                   if USE_EIRFS else None)
        with Timer() as train_timer:
            ckpt = train_one_fold(spec["coco_path"], out_dir, best_params, TRAIN_EPOCHS, weights)
        collect_epoch_log(out_dir, "detr")
        (out_dir / "resolved_params.json").write_text(json.dumps(best_params, indent=2))

        with Timer() as eval_timer:
            records = predict_split(ckpt, TEST_ROOT, test_items, best_params)
            dump_predictions(out_dir / "test_predictions.json", records)
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
        (out_dir / "test_metrics.json").write_text(json.dumps(metrics, indent=2))
    except Exception as e:
        print(f"\n{'!'*70}\n[Fold {k}] FAILED: {e}\n{'!'*70}")
        traceback.print_exc(file=sys.stdout)
        sys.stdout.flush()
        fold_errors.append(f"fold {k}: {type(e).__name__}: {e}")
        fold_rows.append({
            "fold": k, "test_mAP50-95": None, "test_mAP50": None, "test_mAP75": None,
            "test_precision": None, "test_recall": None, "test_F1": None,
            "ckpt": None, "run_dir": str(out_dir), "error": str(e),
            "started_at": started_at, "finished_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "train_seconds": None, "eval_seconds": None,
            "epochs_completed": None, "final_train_loss": None,
        })
        continue

    epoch_summary_path = out_dir / "epoch_log_summary.json"
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
        "run_dir": str(out_dir),
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
    "best_params": best_params,
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

print(f"\n=== Oriented-DETR R50 (NoEIRFS) | held-out test, best fold {summary['best_fold']} (shared evaluator) ===")
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
RUN_LABEL = "OrientedDETR_NoEIRFS_KFold_Final"

LOCAL_RUNS = Path("/content/runs")
DRIVE_DEST = _unique_drive_dir(Path(f"{PROJECT_ROOT}/runs/{RUN_LABEL}"))
DRIVE_DEST.mkdir(parents=True, exist_ok=True)
if DRIVE_DEST.name != RUN_LABEL:
    print(f"NOTE: '{RUN_LABEL}' already exists on Drive -- archiving to '{DRIVE_DEST.name}' instead.")
shutil.copytree(LOCAL_RUNS, DRIVE_DEST, dirs_exist_ok=True,
                ignore=shutil.ignore_patterns("folds", "hpo_folds"))

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
import shutil
from pathlib import Path

RUN_LABEL = "OrientedDETR_NoEIRFS_KFold_Final"
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

print(f"\n=== Oriented-DETR R50 (NoEIRFS) | held-out test, best fold {summary['best_fold']} (shared evaluator) ===")
print(f"mAP50-95: {metrics['mAP50-95']:.4f}")
print(f"mAP50   : {metrics['mAP50']:.4f}")
print(f"mAP75   : {metrics['mAP75']:.4f}")
print(f"precision={metrics.get('precision', float('nan')):.4f}  "
      f"recall={metrics.get('recall', float('nan')):.4f}  "
      f"F1={metrics.get('F1', float('nan')):.4f}")
for name, m in metrics["per_class"].items():
    print(f"  {name:15} AP50-95={m['AP50-95']:.4f}  AP50={m['AP50']:.4f}  n={m['n_gt']}")

print("\nFull K-fold summary (all folds):", json.dumps(summary, indent=2))

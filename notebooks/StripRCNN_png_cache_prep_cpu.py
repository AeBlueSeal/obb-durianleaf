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
# # Strip R-CNN dataset PNG-cache prep (CPU-only)
#
# `notebooks/StripRCNN_E-IRFS_kfold_datasetv3.py` and
# `StripRCNN_E-IRFS_optuna_datasetv3.py` both re-encode every source image
# to PNG (MMRotate hardcodes `.png` regardless of the real extension --
# see `yolo_obb_to_dota`'s docstring in `src/obb_common.py`) before
# materialising each fold. That re-encode is pure CPU/PIL work with zero GPU
# dependency, but both notebooks run it inside their own kernel, which is
# attached to a GPU Colab runtime -- so it burns GPU-runtime compute units
# on work that never touches the GPU. For the K-fold notebook specifically
# (9 final folds + 3 HPO folds, each visiting every pooled train+val image),
# this alone has been observed to run 30+ minutes.
#
# This notebook factors that step out: run it under **Runtime > Change
# runtime type > CPU** (no GPU/TPU) to avoid paying GPU-runtime rates for
# it, then archive the resulting cache to Drive. Both notebooks above check
# for this exact Drive path at the start of their own dataset-materialisation
# cell and restore it (a bulk local copy, no PIL work) instead of rebuilding
# it -- see `DRIVE_PNG_CACHE` in each. They still fall back to building the
# cache themselves if this notebook hasn't been run yet, so this step is an
# optional compute-unit optimisation, not a hard dependency.
#
# Re-run this whenever `datasetv3_2.zip` changes; already-cached images are
# skipped, so a re-run after a small dataset update is cheap.

# %% [markdown]
# ## file setup
#
# No torch/mmcv/mmdet/optuna needed here -- only numpy/Pillow/tqdm. Installed
# explicitly below even though Colab's default runtime already ships all
# three, matching the two training notebooks' own convention of not relying
# on implicit/preinstalled availability.

# %%
# !pip3 install -q numpy pillow tqdm

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

# %% [markdown]
# ## Build the cache locally, then archive to Drive
#
# Same `DATA_ROOT` and pooled train+val item list as
# `StripRCNN_E-IRFS_kfold_datasetv3.py`'s own dataset-materialisation cell --
# kept in sync by hand, like the other small per-arm pieces this project
# already documents. The Optuna notebook's own pools (train-only for its HPO
# folds, val alone for its final full-train run) are both subsets of this
# same train+val union, so one cache here covers both notebooks.

# %%
import sys
from pathlib import Path

sys.path.insert(0, PROJECT_ROOT)
from src.obb_common import build_png_cache, collect_items  # noqa: E402

DATA_ROOT   = "/content/data"
IMAGES_DIR  = f"{DATA_ROOT}/train/images"
LABELS_DIR  = f"{DATA_ROOT}/train/labels"
VAL_IMAGES  = f"{DATA_ROOT}/val/images"
VAL_LABELS  = f"{DATA_ROOT}/val/labels"

LOCAL_PNG_CACHE = Path("/content/png_cache")

items = collect_items(IMAGES_DIR, LABELS_DIR) + collect_items(VAL_IMAGES, VAL_LABELS)
print(f"Found {len(items)} labeled images (train + val pooled).")

n_encoded = build_png_cache(items, LOCAL_PNG_CACHE)
n_total = len(list(LOCAL_PNG_CACHE.glob("*.png")))
print(f"PNG cache ready at {LOCAL_PNG_CACHE}: {n_encoded} newly encoded, {n_total} total.")

# %% [markdown]
# ### Archive to Drive as a single zip
#
# Packed as one archive, the same way `datasetv3_2.zip` ships the dataset
# itself -- Drive's FUSE mount is slow for many small files, so writing
# ~1,445 individual PNGs there directly (`shutil.copytree` against a
# Drive-mounted path) would pay that per-file cost on write, and restoring
# them later would pay it again on read. One `shutil.copy2` of a single zip
# avoids both. `StripRCNN_E-IRFS_kfold_datasetv3.py` and
# `StripRCNN_E-IRFS_optuna_datasetv3.py` both look for this exact path
# (`{PROJECT_ROOT}/data/StripRCNN_png_cache.zip`) at the start of their own
# dataset-materialisation cell and unzip it locally.

# %%
import shutil  # noqa: E402

DRIVE_PNG_CACHE_RUN_LABEL = "StripRCNN_png_cache"
LOCAL_ZIP = Path(f"/content/{DRIVE_PNG_CACHE_RUN_LABEL}.zip")
DRIVE_DEST = Path(f"{PROJECT_ROOT}/data/{DRIVE_PNG_CACHE_RUN_LABEL}.zip")

# Zip locally first (fast, local disk), then a single copy to Drive --
# not shutil.make_archive(..., base_dir=Drive path) directly, which would
# write the zip's data through the slow Drive mount incrementally instead
# of once.
shutil.make_archive(str(LOCAL_ZIP.with_suffix("")), "zip", root_dir=LOCAL_PNG_CACHE)
DRIVE_DEST.parent.mkdir(parents=True, exist_ok=True)
shutil.copy2(LOCAL_ZIP, DRIVE_DEST)

print(f"Archived {LOCAL_PNG_CACHE} -> {LOCAL_ZIP} -> {DRIVE_DEST} "
      f"({DRIVE_DEST.stat().st_size / 1e6:.1f} MB)")

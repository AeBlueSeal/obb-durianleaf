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
# # S2A-Net -- re-test an archived K-fold run (no retraining)
#
# Companion to `notebooks/S2ANet_E-IRFS_kfold_datasetv3.py`. That notebook
# no longer has its own "re-test from Drive" section -- it's been split out
# into this dedicated notebook instead, so the main notebook can run
# start-to-finish non-interactively (e.g. via a Colab CLI runner) without a
# manual "skip training" decision partway through a single file.
#
# Run this after `S2ANet_E-IRFS_kfold_datasetv3.py`'s own "Archive to Drive"
# cell has produced a `RUN_LABEL` folder on Drive. This notebook restores
# that whole `/content/runs` tree locally and re-prints the held-out test
# metrics already computed for the best fold during that original run -- no
# retraining, and no s2anet/mmcv/mmdet environment needed at all (everything
# here is plain JSON/CSV reads).

# %% [markdown]
# ## file setup
#
# Deliberately NOT the Miniconda/s2anet conda-env bootstrap the main
# notebook needs -- re-printing already-computed metrics from
# `test_metrics.json`/`kfold_final_summary.json` is pure JSON/CSV work, so
# this notebook never imports mmcv/mmdet and never needs that environment.

# %%
# !pip3 install -q pandas

# %%
from google.colab import drive
drive.mount('/content/drive')

PROJECT_NAME = 'ProjZ'
PROJECT_ROOT = f'/content/drive/MyDrive/{PROJECT_NAME}'

# %%
import json
import shutil
from pathlib import Path

import pandas as pd

# Must match the main notebook's own PROJECT/RUN_LABEL/KFOLD_FINAL_DIR
# exactly -- these are how the archived run is found and re-parsed.
PROJECT = "/content/runs/s2anet"
RUN_LABEL = "S2ANet_E-IRFS_KFold_Final"
KFOLD_FINAL_DIR = Path(PROJECT) / "kfold_final_s2anet"

# %% [markdown]
# ## Restore from Drive

# %%
# Restores the whole /content/runs tree from its RUN_LABEL folder on Drive,
# then reads kfold_final_summary.json to find which fold was best.
LOCAL_RUNS = Path("/content/runs")
DRIVE_DEST = Path(f"{PROJECT_ROOT}/runs/{RUN_LABEL}")

if not DRIVE_DEST.exists():
    raise FileNotFoundError(
        f"No archived run found at {DRIVE_DEST}. Run "
        f"notebooks/S2ANet_E-IRFS_kfold_datasetv3.py first and let its "
        f"'Archive to Drive' cell finish (RUN_LABEL={RUN_LABEL!r}) -- or, if "
        f"you archived more than once, that cell's own _unique_drive_dir() "
        f"may have named it '{RUN_LABEL}2', '{RUN_LABEL}3', etc.; point "
        f"RUN_LABEL above at the archive you want."
    )

LOCAL_RUNS.mkdir(parents=True, exist_ok=True)
shutil.copytree(DRIVE_DEST, LOCAL_RUNS, dirs_exist_ok=True)
print(f"Restored {LOCAL_RUNS} <- {DRIVE_DEST}")

summary = json.loads((KFOLD_FINAL_DIR / "kfold_final_summary.json").read_text())
fold_df = pd.read_csv(KFOLD_FINAL_DIR / "kfold_final_results.csv")
best_row = fold_df.loc[fold_df["test_mAP50-95"].idxmax()]
FINAL_CKPT = Path(best_row["ckpt"])
FINAL_DIR = Path(best_row["run_dir"])
print(f"Best fold: {summary['best_fold']} -> {FINAL_CKPT}")

# %% [markdown]
# ## test
#
# Re-prints the held-out test metrics already computed for the best fold --
# no re-inference needed since the worker/evaluate_obb() already wrote
# test_metrics.json during the original K-fold run.

# %%
metrics = json.loads((FINAL_DIR / "test_metrics.json").read_text())

print(f"\n=== S2A-Net R50-FPN | held-out test, best fold {summary['best_fold']} (shared evaluator) ===")
print(f"mAP50-95: {metrics['mAP50-95']:.4f}")
print(f"mAP50   : {metrics['mAP50']:.4f}")
print(f"mAP75   : {metrics['mAP75']:.4f}")
print(f"precision={metrics.get('precision', float('nan')):.4f}  "
      f"recall={metrics.get('recall', float('nan')):.4f}  "
      f"F1={metrics.get('F1', float('nan')):.4f}")
for name, m in metrics["per_class"].items():
    print(f"  {name:15} AP50-95={m['AP50-95']:.4f}  AP50={m['AP50']:.4f}  n={m['n_gt']}")

print("\nFull K-fold summary (all folds):", json.dumps(summary, indent=2))

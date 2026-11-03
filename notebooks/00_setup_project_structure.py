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
# # ProjZ — Project Structure Setup
#
# Run this notebook once in Google Colab to scaffold the `ProjZ` folder structure on Google Drive, following `results/colab_project_structure.md`.

# %%
from google.colab import drive
drive.mount('/content/drive')

# %%
PROJECT_NAME = 'ProjZ'
PROJECT_ROOT = f'/content/drive/MyDrive/{PROJECT_NAME}'

# %%
import os

FOLDERS = [
    'data/datasetv3_2/train/images',
    'data/datasetv3_2/train/labels',
    'data/datasetv3_2/val/images',
    'data/datasetv3_2/val/labels',
    'data/datasetv3_2/test/images',
    'data/datasetv3_2/test/labels',
    'data/datasetv3_2_resplit/train/images',
    'data/datasetv3_2_resplit/train/labels',
    'data/datasetv3_2_resplit/val/images',
    'data/datasetv3_2_resplit/val/labels',
    'data/datasetv3_2_resplit/test/images',
    'data/datasetv3_2_resplit/test/labels',
    'notebooks',
    'src',
    'models/checkpoints',
    'models/final',
    'models/configs',
    'outputs/logs',
    'outputs/figures',
    'outputs/predictions',
]

FILES = [
    'data/dataset.yaml',
    'data/dataset_resplit.yaml',
    'data/datasetv3_2/data.yaml',
    'data/datasetv3_2/classes.txt',
    'data/datasetv3_2/notes.json',
    'src/__init__.py',       # empty __init__.py is correct — makes src/ a package
    'requirements.txt',
    'README.md',
]

for folder in FOLDERS:
    os.makedirs(os.path.join(PROJECT_ROOT, folder), exist_ok=True)

for file in FILES:
    path = os.path.join(PROJECT_ROOT, file)
    if not os.path.exists(path):
        open(path, 'a').close()

# NOTE: Python source files (src/*.py) are NOT created as empty placeholders —
# empty .py files import successfully but define nothing, which silently masks
# missing real files and causes confusing ImportError later (e.g.
# "cannot import name YOLOEIRFSCappedDataset from src.data_loader").
# Instead, the training notebooks (train.py, v4_confound_fix_train.py, etc.)
# write the required src/*.py modules inline if they're missing from Drive.
# If you are syncing the entire project folder via Google Drive for Desktop,
# those files will already be present and the notebooks will skip the write.

# dataset archives (datasetv3_2.zip, datasetv3_ISNet.zip, datasetv3_BEN2.zip) are not
# scaffolded here — upload/copy the real archives into data/ instead of creating placeholders.

print(f'{PROJECT_NAME} structure created at {PROJECT_ROOT}')

# %%
for root, dirs, files in os.walk(PROJECT_ROOT):
    dirs.sort()
    level = root.replace(PROJECT_ROOT, '').count(os.sep)
    indent = '  ' * level
    print(f'{indent}{os.path.basename(root)}/')
    for f in sorted(files):
        print(f'{indent}  {f}')

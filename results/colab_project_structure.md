# ML Project File Structure — Google Drive / Colab

This structure is designed for a machine learning project stored on **Google Drive** and accessed/run through **Google Colab**.

```
MyProject/
├── data/
│   ├── datasetv3_2/             # main dataset (YOLO format)
│   │   ├── data.yaml
│   │   ├── classes.txt
│   │   ├── notes.json
│   │   ├── train/
│   │   │   ├── images/
│   │   │   └── labels/
│   │   ├── val/
│   │   │   ├── images/
│   │   │   └── labels/
│   │   └── test/
│   │       ├── images/
│   │       └── labels/
│   ├── datasetv3_2_resplit/     # resplit variant of the dataset
│   │   ├── train/
│   │   │   ├── images/
│   │   │   └── labels/
│   │   ├── val/
│   │   │   ├── images/
│   │   │   └── labels/
│   │   └── test/
│   │       ├── images/
│   │       └── labels/
│   ├── dataset.yaml             # Ultralytics data config → datasetv3_2
│   ├── dataset_resplit.yaml     # Ultralytics data config → datasetv3_2_resplit
│   ├── datasetv3_2.zip
│   ├── datasetv3_ISNet.zip
│   └── datasetv3_BEN2.zip
│
├── notebooks/                # each notebook is paired via jupytext (ipynb,py:percent) --
│   ├── 01_eda.ipynb          # the .py is the versioned source (readable diffs, no
│   ├── 01_eda.py             # output noise); .ipynb is gitignored, a local build
│   ├── 02_preprocessing.ipynb  # artifact you still need to actually open/run/upload
│   ├── 02_preprocessing.py     # (Jupyter, VS Code, Colab). Regenerate a missing
│   ├── 03_training.ipynb       # .ipynb with `jupytext --sync notebooks/<name>.py`.
│   ├── 03_training.py
│   ├── 04_evaluation.ipynb
│   └── 04_evaluation.py
│
├── src/                     # reusable code (import into notebooks)
│   ├── __init__.py
│   ├── data_loader.py
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   └── utils.py
│
├── models/
│   ├── checkpoints/         # intermediate training checkpoints
│   ├── final/               # best/final saved models
│   └── configs/             # hyperparameter configs (yaml/json)
│
├── runs/                    # raw Ultralytics training-run output, one subfolder
│   │                        # per Colab training notebook (weights/, results.csv,
│   │                        # args.yaml, confusion_matrix.png, etc.) -- every run
│   │                        # kept for traceability; hand-pick the winner into
│   │                        # models/final/ rather than pruning this folder
│   ├── train/               # notebooks/train.ipynb
│   ├── v4_confound_fix/
│   │   └── train/           # notebooks/v4_confound_fix_train.ipynb
│   └── optuna_obb/
│       └── best_final_all_m/  # notebooks/Template_Clean_E-IRFS_optuna_datasetv3.ipynb
│
├── outputs/
│   ├── logs/                # training logs, tensorboard files
│   ├── figures/             # plots, confusion matrices, etc.
│   └── predictions/         # inference results
│
├── requirements.txt
└── README.md
```

## Mounting in Colab

```python
from google.colab import drive
drive.mount('/content/drive')

PROJECT_ROOT = '/content/drive/MyDrive/MyProject'
```

## Notes

- Keep `data/raw/` read-only in practice — never overwrite originals, since Drive has no built-in version control.
- Checkpoints can eat storage fast; periodically clear `models/checkpoints/` once you've picked a final model.
- If your dataset is huge, consider keeping raw data zipped in Drive and unzipping to local Colab disk (`/content/`) at runtime — Drive I/O is much slower than local disk, so training directly off Drive-mounted data can bottleneck you.
- Put shared constants (paths, seeds, hyperparams) in one `config.py` in `src/` so notebooks stay in sync.
- Every Colab notebook that calls `model.train()` writes its Ultralytics `project=`/`name=` output straight into `runs/` on Drive (not local `/content/`), so results survive the runtime being recycled. Training itself still reads from the locally-copied dataset (see the note above) — only the run *output* (weights, logs, plots) goes directly to Drive as it's produced.
- Each of those training notebooks ends with a test-split evaluation (`model.val(split="test")`) against the `best.pt` it just produced, so every run gets an honest held-out number before you trust it — see `results/source_confound_pipeline.md` Phase 7 for the fuller confound-check suite to run on top of that.
- Notebooks are version-controlled as jupytext-paired `.py` files (`git diff` on JSON `.ipynb` is unreadable and every re-run dirties the diff with new outputs/execution counts even when no code changed). `.ipynb` is gitignored. Workflow: edit either the `.ipynb` (in Jupyter/Colab/VS Code) or the `.py`, then run `jupytext --sync notebooks/<name>.ipynb` before committing so both stay in sync — only the `.py` actually gets committed. On a fresh clone, generate the missing `.ipynb` files with `jupytext --to notebook notebooks/*.py` (requires `pip install jupytext`).

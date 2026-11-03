# Drive results folder structure — OBB comparison notebooks

How the 14 active Optuna/K-fold notebooks (YOLO11m-OBB, YOLO26m-OBB,
Oriented-DETR, Strip R-CNN — each with an E-IRFS and a NoEIRFS-ablation
K-fold variant —, plus RF-DETR-OBB) archive their results to Google Drive,
and how to find the number you actually want inside them.

Everything lives under:

```
<Drive>/ProjZ/runs/<RUN_LABEL>/
```

`ProjZ` is `PROJECT_NAME` (constant at the top of every notebook).
`<RUN_LABEL>` is set in each notebook's own "Archive to Drive" cell — see the
table below. **The archive is not scoped to that one notebook's own output**:
every notebook's archive cell does `shutil.copytree("/content/runs", DRIVE_DEST)`
— i.e. it copies *the entire local Colab `/content/runs` tree*, not just its
own `PROJECT` subfolder. In practice this only matters if you ran more than
one notebook in the same Colab runtime before archiving; normally each
notebook gets a fresh runtime, so `<RUN_LABEL>/` ends up holding exactly one
`PROJECT`-basename subfolder, named after that notebook's own `PROJECT`
constant.

## Collision handling

Every archive cell resolves its destination through `_unique_drive_dir()`:
if `ProjZ/runs/<RUN_LABEL>/` already exists, it archives to `<RUN_LABEL>2`,
then `<RUN_LABEL>3`, etc. instead of merging into the existing one. It
prints `NOTE: '<RUN_LABEL>' already exists on Drive -- archiving to
'<RUN_LABEL>N' instead.` when this happens — check your session log if
you're not sure which numbered copy is the one you want, and always check
`ProjZ/runs/` itself for the highest suffix before assuming `<RUN_LABEL>` is
the latest.

## RUN_LABEL / PROJECT lookup table

| Notebook | `RUN_LABEL` | `PROJECT` basename (subfolder under it) |
|---|---|---|
| `YOLO11mOBB_E-IRFS_optuna_datasetv3.py` | `YOLO11m_E-IRFS_Optuna` | `optuna_obb_yolo11m` |
| `YOLO26mOBB_E-IRFS_optuna_datasetv3.py` | `YOLO26m_E-IRFS_Optuna` | `optuna_obb_yolo26m` |
| `OrientedDETR_E-IRFS_optuna_datasetv3.py` | `OrientedDETR_E-IRFS_Optuna` | `optuna_oriented_detr` |
| `RFDETR_OBB_E-IRFS_optuna_datasetv3.py` | `RFDETR_OBB_E-IRFS_Optuna` | `optuna_rfdetr_obb` |
| `StripRCNN_E-IRFS_optuna_datasetv3.py` | `StripRCNN_E-IRFS_Optuna` | `optuna_strip_rcnn` |
| `YOLO11mOBB_E-IRFS_kfold_datasetv3.py` | `YOLO11m_E-IRFS_KFold_Final` | `kfold_final_yolo11m` |
| `YOLO26mOBB_E-IRFS_kfold_datasetv3.py` | `YOLO26m_E-IRFS_KFold_Final` | `kfold_final_yolo26m` |
| `YOLO11mOBB_NoEIRFS_kfold_datasetv3.py` | `YOLO11m_NoEIRFS_KFold_Final` | `kfold_final_yolo11m_noeirfs` |
| `YOLO26mOBB_NoEIRFS_kfold_datasetv3.py` | `YOLO26m_NoEIRFS_KFold_Final` | `kfold_final_yolo26m_noeirfs` |
| `OrientedDETR_E-IRFS_kfold_datasetv3.py` | `OrientedDETR_E-IRFS_KFold_Final` | `kfold_final_oriented_detr` |
| `OrientedDETR_NoEIRFS_kfold_datasetv3.py` | `OrientedDETR_NoEIRFS_KFold_Final` | `kfold_final_oriented_detr_noeirfs` |
| `RFDETR_OBB_E-IRFS_kfold_datasetv3.py` | `RFDETR_OBB_E-IRFS_KFold_Final` | `kfold_final_rfdetr_obb` |
| `StripRCNN_E-IRFS_kfold_datasetv3.py` | `StripRCNN_E-IRFS_KFold_Final` | `kfold_final_strip_rcnn` |
| `StripRCNN_NoEIRFS_kfold_datasetv3.py` | `StripRCNN_NoEIRFS_KFold_Final` | `kfold_final_strip_rcnn_noeirfs` |

Each notebook's own "Re-test from Drive" section hardcodes the matching
`RUN_LABEL`/`PROJECT` pair above to restore from — if you archived under a
`...2`/`...3` suffix, edit `RUN_LABEL` there before running it.

## Shape 1: YOLO11m-OBB / YOLO26m-OBB (Ultralytics)

### Optuna notebooks

```
<RUN_LABEL>/optuna_obb_yolo11m/
├── trial_000_fold0/ ... trial_NNN_fold2/    # one Ultralytics run dir per (trial, fold)
│   └── weights/, results.csv, args.yaml, ...
├── cv_trial_000.csv ... cv_trial_NNN.csv    # per-trial per-fold scores
├── best_final_yolo11m_obb/                 # best_params.json AND the final
│   ├── best_params.json                    #   full-split training run's own
│   ├── weights/best.pt                     #   Ultralytics output land in the
│   └── results.csv, args.yaml, ...          #   SAME folder (same project=/name=)
└── (nothing else under this PROJECT folder for the final test-eval plots --
     see the next paragraph)
```

**Quirk worth knowing:** the Optuna notebook's own final test-set evaluation
(`model.val(split="test", plots=True)`) does **not** pass `project=`/`name=`,
so Ultralytics writes it to its own default location, `/content/runs/obb/valN`
(auto-incremented per call in that Colab session) — a **sibling** of
`optuna_obb_yolo11m/`, not nested inside it. It's still captured by the
archive (which copies the whole `/content/runs` tree), just under:

```
<RUN_LABEL>/obb/valN/     # confusion_matrix.png, PR curves, etc. for the ONE test eval this notebook ran
```

If you archived more than once in the same runtime, `valN` keeps incrementing
across every `model.val()` call made so far that session (not just this
notebook's) — match it up by timestamp if there's more than one `obb/valN`.

### K-fold notebooks (E-IRFS and NoEIRFS)

These *do* pass `project=`/`name=` on every eval call (fixed this way
deliberately, precisely so you don't have to hunt through auto-numbered
`valN` folders), so everything lands inside the `PROJECT` folder:

```
<RUN_LABEL>/kfold_final_yolo11m/
├── fold_files/                              # train_fold{k}.txt, val_fold{k}.txt, data_fold{k}.yaml
├── kfold_final_yolo11m_obb_fold0/ ... _fold8/   # one Ultralytics run dir per fold (K=9)
│   └── weights/best.pt, results.csv, args.yaml, ...
├── kfold_final_yolo11m_obb_fold0_test_eval/ ... _fold8_test_eval/  # per-fold TEST plots
├── best_fold_test_eval/                     # best fold's re-verification, full plots
└── kfold_final_yolo11m_obb/                 # <-- the numbers you actually want
    ├── kfold_final_results.csv              #     per-fold val/test mAP50-95/mAP50/P/R/F1
    └── kfold_final_summary.json             #     mean±std across folds + best_fold index + best_params
```

Note the naming trap: `kfold_final_yolo11m_obb_fold0/` (a training-run
directory) and `kfold_final_yolo11m_obb/` (the results-CSV directory) are
**siblings with almost-identical names** — the one *without* a `_foldN`
suffix is the one holding `kfold_final_results.csv`/`kfold_final_summary.json`.

## Shape 2: Oriented-DETR / RF-DETR-OBB / Strip R-CNN

Same two-notebook split, but everything nests cleanly under one directory
per stage (no Ultralytics-style sibling-folder surprises), and the archive
cell explicitly excludes materialised dataset copies:

```
<RUN_LABEL>/<project_basename>/
├── folds/fold0/ ... fold2/         # EXCLUDED from the Drive archive (ignore=shutil.ignore_patterns("folds"))
│                                    #   -- reproducible from make_folds(items, K_FOLDS, seed=SEED) anyway
├── trial_000_fold0/ ... (Optuna notebooks; or K-fold notebooks with RUN_HPO=True)
├── cv_trial_000.csv ...
├── best_final_<model>/             # Optuna notebooks only
│   ├── best_params.json
│   ├── checkpoint(s)               # checkpoint.pth / checkpoint_best_total.pth / epoch_N.pth
│   ├── test_predictions.json
│   └── test_metrics.json           # <-- the numbers you want (mAP50-95/50/75, per-class)
└── kfold_final_<model>/            # K-fold notebooks only
    ├── fold0/ ... fold8/           # K=9, one per fold
    │   ├── checkpoint(s)
    │   ├── test_predictions.json
    │   └── test_metrics.json       # this fold's mAP50-95/50/75 + precision/recall/F1
    ├── kfold_final_results.csv     # <-- the numbers you want
    └── kfold_final_summary.json    # mean±std across folds + best_fold index + best_params
```

**Gap closed (2026-09):** the K-fold notebooks' `RUN_HPO=True` path
materialises its own train-only folds under `hpo_folds/fold0/ ... fold2/`
(K=3) — a *different* directory name from `folds/`, which the archive's
`ignore_patterns("folds")` exclusion did not cover. Oriented-DETR and
RF-DETR-OBB's archive cells now also exclude `"hpo_folds"`; Strip R-CNN and
S2A-Net already did.

## New logging files (2026-09 logging pass)

Every arm now writes these alongside what's described above (see
`src/obb_logging.py` and `results/comparison_control_variables.md`'s "what
every arm logs" section for the full rationale):

| File | Where | What |
|---|---|---|
| `run_env.json` | `PROJECT`/ root, once per notebook run | timestamp, platform, GPU (`nvidia-smi`), importable framework versions, git commit, this arm's own config constants |
| `fold_composition.csv` | `PROJECT`/ root, once per notebook run | per (fold, split, class) image/instance counts — audits class balance across folds |
| `epoch_log.csv` + `epoch_log_summary.json` | beside each fold's/trial's other output | that framework's own per-epoch loss/lr curve, normalised; summary has `epochs_completed`, `first_loss`/`final_loss`/`min_loss`, `total_seconds` |
| `resolved_params.json` | beside each fold's other output | the hyperparameter dict actually handed to the trainer (not just what Optuna searched) |
| `kfold_per_class_results.csv` / `kfold_per_class_summary.csv` | beside `kfold_final_results.csv` | one row per (fold, class) / per class mean±std of AP50-95, AP50, AP75, precision, recall, F1, tp, fp, fn |
| `hpo_trials.csv` + `hpo_best.json` | beside `best_params.json` | `study.trials_dataframe()` and the winning trial, so the Optuna study is readable without an Optuna install or the sqlite file |
| `train_log.txt` | beside each fold's other output (Strip R-CNN, S2A-Net) | the conda-subprocess worker's full raw stdout, previously visible only in the committed `.ipynb` cell output |

All of it is write-only side output: no training config, sampler, evaluation
operating point, or metric math changed. A missing file for a given fold
means that fold's logging failed soft (see `src/obb_logging.py`'s module
docstring) — it does not mean the fold itself failed; check
`kfold_final_results.csv`'s own `error` column for that.

Oriented-DETR, RF-DETR-OBB and Strip R-CNN's Optuna studies are now sqlite-
backed (`hpo.db` under `PROJECT`, mirrored to Drive) rather than in-memory,
matching the pattern S2A-Net already used — a Colab disconnect mid-search no
longer loses every completed trial.

## Quick "where's my number" cheat sheet

- **Best single result to quote for an architecture's K-fold run:**
  `kfold_final_summary.json`'s `test_mAP50-95_mean`/`_std` (and the
  `test_mAP50`/`test_precision`/`test_recall`/`test_F1` equivalents) — not
  `best_fold`'s number alone, which is a cherry-picked max across folds (see
  `CLAUDE.md`'s cross-architecture audit notes on why).
- **Which fold is "best" and where its weights live:** `kfold_final_summary.json["best_fold"]`,
  then look up that fold number's `run_dir`/`ckpt` column in
  `kfold_final_results.csv`.
- **Winning hyperparameters for a from-scratch retrain:** `best_params.json`
  under `best_final_<model>*/` (Optuna notebook) — the K-fold notebooks load
  this same file rather than re-deriving it (unless `RUN_HPO=True`, in which
  case they write their own copy in the same relative location).

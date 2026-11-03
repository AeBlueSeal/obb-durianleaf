# Control variables of the OBB architecture comparison (K-fold generation)

The **independent variable is the architecture** — six arms:

| Arm | K-fold notebook | Framework |
|---|---|---|
| YOLO11m-OBB (baseline) | `YOLO11mOBB_E-IRFS_kfold_datasetv3.py` | Ultralytics |
| YOLO26m-OBB | `YOLO26mOBB_E-IRFS_kfold_datasetv3.py` | Ultralytics |
| Oriented-DETR R50 | `OrientedDETR_E-IRFS_kfold_datasetv3.py` | standalone (DINO-derived) |
| RF-DETR-Medium OBB | `RFDETR_OBB_E-IRFS_kfold_datasetv3.py` | rfdetr (open PR) |
| Strip R-CNN-S | `StripRCNN_E-IRFS_kfold_datasetv3.py` | MMRotate 0.x / mmdet 2.x |
| S2A-Net R50-FPN | `S2ANet_E-IRFS_kfold_datasetv3.py` | csuhan/s2anet (mmdet 1.x) |

A second, smaller experiment rides along: **E-IRFS on/off**, as ablation twins
for YOLO11m, YOLO26m and S2A-Net (`*_NoEIRFS_kfold_datasetv3.py`).

Everything below is what is held constant so that a difference in the reported
number is attributable to those two factors rather than to plumbing.

**Enforcement:** `analysis/comparison/check_kfold_comparability.py` — runs in
seconds, no GPU, no framework imports (four arms pin mutually incompatible
torch/mmcv/Python versions, which is exactly why they drift). Run it after
touching any arm.

**Related docs.** [kfold_comparability_audit.md](kfold_comparability_audit.md)
is the findings history — what was found broken, when, and what it invalidated.
[obb_model_comparison.md](obb_model_comparison.md) has the per-arm search-space,
pretraining and angle-representation detail, which is still the best reference
for *why* certain things cannot be equalised — but note it describes the earlier
**Optuna** generation (five arms, 25 epochs/trial, single train split) and some
of its statements have since been superseded.

---

## 1. Enforced by the guard

A violation here fails the script, deliberately.

| # | Control | Value / rule |
|---|---|---|
| 1 | Seed, folds, budget | `SEED = 42`, `K_FOLDS = 9`, `TRAIN_EPOCHS = 250` |
| 2 | E-IRFS threshold | `EIRFS_THRESHOLD = 0.0001` (skipped for the `*_NoEIRFS` twins) |
| 3 | Reported metric | every arm calls `evaluate_obb()` **and** `precision_recall_f1()` from `src/obb_eval.py`, and records `test_mAP50-95` / `test_mAP50` / `test_mAP75` |
| 4 | Framework metrics quarantined | Ultralytics' `box.map` may appear only under an `ultra_*` name; it is not cross-arm comparable |
| 5 | Evaluation operating point | conf / `score_thr` = **0.001** in all six. The mmdet-family arms must set it *explicitly*: their base DOTA configs inherit 0.05, applied inside NMS before the worker's own `predict_fold` ever sees a detection, so absence of "0.05" from the notebook proves nothing |
| 6 | Background-only images | trained on, not discarded — `skip_img_without_anno=False` (S2A-Net, mmdet 1.x) and `filter_empty_gt=False` (Strip R-CNN, MMRotate). Both frameworks *default* to discarding them, so absence of the flag is itself the violation |
| 7 | One fold cannot kill a run | the final K-fold loop needs a per-fold `try/except`, or one transient OOM discards every other fold's completed work |
| 8 | Fold identity | the YOLO arms' own local fold split must equal `src/obb_common.kfold_assignments`, and must reseed on every call |

## 2. Held constant by convention — documented, not machine-checked

Changing one of these will not fail the guard. It still breaks the comparison,
so it needs an entry in the audit.

- **HPO budget shape:** `N_TRIALS = 10`, `HPO_K_FOLDS = 3`,
  `EPOCHS_PER_TRIAL = 50`, `N_STARTUP_TRIALS = 3`.
- **`imgsz` search space** `[320, 640, 832, 1024]` — except RF-DETR-OBB, which
  requires multiples of 56 and uses `336/616/840/1008`.
- **Item pools:** `train/` only for the HPO folds; `train/` + `val/` pooled for
  the final folds; test is always `/content/data/test/`. No arm resamples its
  val split.
- **Fold construction:** `collect_items` → `make_folds(..., seed=SEED,
  shuffle=True)` over the same sorted item list.
- **E-IRFS parameters:** `capped_class_names={"Rhizoctonia"}`, `alpha=2.0`,
  `cap_multiplier=1.5`; weights built from train-only labels.
- **Resolution confound:** train and test resolution are always overridden
  together (the B1 confound).
- **Detection cap:** 300 per image — Ultralytics' `max_det` default, which
  S2A-Net was brought to on 2026-09-12 (it had the stock 2000). **Not audited
  for Strip R-CNN, Oriented-DETR or RF-DETR-OBB.**
- **`warmup_iters` / `warmup_ratio`** search ranges.
- **Data:** `datasetv3_2` staged to `/content/data`, one `classes.txt`.

## 3. Cannot be held constant, and what is done instead

- **`batch` search space** differs by arm — a genuine memory constraint, not drift.
- **Early stopping.** The YOLO arms stop on patience (`FINAL_PATIENCE = 50`);
  Strip R-CNN, Oriented-DETR and S2A-Net have no early-stopping hook in their
  framework and always burn the full budget. So `TRAIN_EPOCHS = 250` is *not*
  the same amount of training across arms — it is an upper bound for three of
  them and the actual figure for the others.
- **Optuna pruning.** S2A-Net and Strip R-CNN call `should_prune()`; the others
  run every trial to completion, so the effective HPO budget differs. The guard
  reports this rather than failing, as an accepted difference. Note
  `n_warmup_steps=2` against `HPO_K_FOLDS = 3` means pruning can only fire
  after the final fold, i.e. it currently saves nothing — uniformly, in both
  arms that configure it.
- **HPO selection criterion.** The non-YOLO arms score HPO folds with
  `evaluate_obb`; the YOLO Optuna notebooks use Ultralytics' own val mAP from
  `results.csv`. This changes which hyperparameters win, not the reported
  metric.
- **Search spaces for optimizer / LR schedule / augmentation** are
  per-framework — see `obb_model_comparison.md` §2.1 for the full mapping.
  S2A-Net is the narrowest: its pipeline offers only `RotatedResize` +
  `RotatedRandomFlip`, with no HSV/mosaic/mixup equivalent to search at all.
- **Pretraining** differs by arm and is the largest single confound — see
  `obb_model_comparison.md` §2.2.
- **Angle representation** differs (`le90` / `le135` / `oc` / S2A-Net's own
  `norm_angle`); all arms are converted to the same polygon form before the
  shared evaluator sees them.
- **GPU.** Most arms run on modern hardware. **S2A-Net is capped at compute
  capability 7.5** (T4/V100 or older): the csuhan/s2anet fork needs CUDA 10.0 /
  cuDNN 7.6, which ship no kernels past Turing and cannot JIT from PTX, so a
  newer card dies in the first convolution with
  `CUDNN_STATUS_EXECUTION_FAILED`. This changes runtime, not results — but by
  5–8×, which is why this arm's `9 x 250`-epoch budget is tens to hundreds of
  hours where other arms' is a few. MMRotate ships an S2A-Net
  (`configs/s2anet/`, 73.91 DOTA mAP against the paper's ~74.1) that would lift
  the cap if this ever becomes blocking.
- **Mixed precision.** S2A-Net trains in fp16 as of 2026-09-12, and the YOLO
  arms default to Ultralytics' `amp=True`. Strip R-CNN, Oriented-DETR and
  RF-DETR-OBB are **unaudited**, so "all six train in mixed precision" is not a
  claim this list supports yet.

## 4. Biases that apply equally to every arm

- `best_fold` is selected by `idxmax()` over the **test** mAP50-95, which is
  optimistically biased. It affects only which checkpoint gets archived.
  **The number to report is `test_mAP50-95_mean ± _std`.**

## 5. What every arm logs (2026-09 logging pass)

Beyond the columns machine-checked in §1, every arm now also writes (see
`src/obb_logging.py` and `results/drive_results_structure.md`'s "New logging
files" section for exact paths):

- **`run_env.json`** and **`fold_composition.csv`**, once per notebook run —
  platform/GPU/framework versions/git commit/config constants, and per-fold
  per-class image/instance counts.
- **A per-epoch curve** (`epoch_log.csv` + `epoch_log_summary.json`) per fold
  and per HPO trial-fold, normalised from whatever that framework already
  writes (Ultralytics `results.csv`, mmdet `*.log.json`, DETR-family
  `log.txt`). This is what makes the early-stopping asymmetry in §3 above
  *measurable* rather than merely documented: `epochs_completed` now exists
  for every arm, including RF-DETR-OBB, whose actual stopping epoch
  (`early_stopping=True`) was previously recorded nowhere at all.
- **`resolved_params.json`** per fold — the hyperparameter dict actually
  handed to the trainer, not just the Optuna-searched subset.
- **A per-class table across all folds** (`kfold_per_class_results.csv` /
  `kfold_per_class_summary.csv`) — AP50-95/AP50/AP75/precision/recall/F1/tp/
  fp/fn per class, mean±std across the folds that had GT for it. Previously
  per-class AP existed only inside each fold's own `test_metrics.json` and
  was printed for the best fold only.
- **Sqlite-backed Optuna studies** (`hpo.db`, mirrored to Drive, plus
  `hpo_trials.csv`/`hpo_best.json`) for Oriented-DETR, RF-DETR-OBB and Strip
  R-CNN — previously in-memory, so a Colab disconnect mid-search lost every
  completed trial's params/values. S2A-Net already had this; the other three
  now share the same `src/obb_logging.open_study`/`dump_study` helpers.

**Explicitly rejected as too expensive to add here:** turning on per-epoch
*validation* for the mmdet-family arms (Strip R-CNN, S2A-Net) — their
`validate=False` training call is what makes `TRAIN_EPOCHS=250` cheap for
those two arms in the first place, and adding a real eval pass every epoch
would burn GPU time this logging pass was supposed to avoid costing (see the
"no meaningful Colab time added" constraint each of these arms' notebooks
now documents alongside the new logging calls). Also rejected: TensorBoard
event files and W&B, for the same reason -- more moving parts and Drive
volume for the same numbers `epoch_log.csv` already recovers from each
framework's own text log.

Every new field is additive, write-only side output: no training config,
sampler, evaluation operating point, or metric math changed, and a
bit-identical check against the pre-change `evaluate_obb()`/
`precision_recall_f1()` return values confirmed this before any arm was
touched. A missing new file for a given fold means that fold's *logging*
failed soft (by design — see `src/obb_logging.py`'s module docstring), not
that the fold itself failed; `kfold_final_results.csv`'s own `error` column
is still the source of truth for that.

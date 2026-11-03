# E-IRFS K-fold cross-arm comparability audit (2026-09-11)

> This document is the findings *history* — what was found broken, when, and
> what it invalidated. For the current list of control variables (what is
> enforced, what is convention, and what cannot be held constant), see
> [comparison_control_variables.md](comparison_control_variables.md).

Audit question: **what, in the six E-IRFS K-fold notebooks, makes their
numbers uncomparable to each other?** Same question the S2A-Net-specific pass
answered for one arm, asked across all of them at once.

Arms audited (the notebooks whose `test_mAP50-95` lands in the comparison
table):

| arm | notebook |
|---|---|
| YOLO11m-OBB | `notebooks/YOLO11mOBB_E-IRFS_kfold_datasetv3.py` |
| YOLO26m-OBB | `notebooks/YOLO26mOBB_E-IRFS_kfold_datasetv3.py` |
| Oriented-DETR | `notebooks/OrientedDETR_E-IRFS_kfold_datasetv3.py` |
| RF-DETR-OBB | `notebooks/RFDETR_OBB_E-IRFS_kfold_datasetv3.py` |
| Strip R-CNN | `notebooks/StripRCNN_E-IRFS_kfold_datasetv3.py` |
| S2A-Net | `notebooks/S2ANet_E-IRFS_kfold_datasetv3.py` |

Plus the two E-IRFS-ablation twins (`YOLO{11,26}mOBB_NoEIRFS_kfold_*`) — if
they drift from their E-IRFS partners, the ablation stops being an ablation —
and the two YOLO Optuna siblings, which produce the `best_params.json` the
YOLO K-fold arms train on.

---

## Finding 1 — the YOLO arms reported a different metric entirely (critical)

**This is the one that invalidates the table.** The four non-YOLO arms score
their held-out test set with `src/obb_eval.py`'s `evaluate_obb` — COCO-style
AP at IoU 0.50:0.05:0.95 over exact polygon IoU. The two YOLO arms (and both
NoEIRFS twins) never called it. They reported `eval_model.val(...).box.map`:
Ultralytics' own OBB mAP, which uses **ProbIoU matching and its own AP
integration**.

`src/obb_eval.py`'s module docstring states the stakes plainly, and they are
the project's own words, not an inference: differences of 1–3 mAP arise from
the metric code alone, *the same order as the architectural differences being
measured*. The file exists specifically so this cannot happen — it even ships
`predict_ultralytics()`, described in its own docstring as being there "so the
YOLO11m-OBB baseline is scored by the same code as the other three."

### Why the two numbers differ at all

Two detectors can produce byte-identical predictions and still be scored
differently, because "mAP" is not one algorithm. Three things diverge:

1. **What counts as an overlap.** `evaluate_obb` computes *exact polygon
   intersection over union* — `poly_iou_matrix` builds real shapely polygons
   and intersects them. Ultralytics' OBB validator matches with **ProbIoU**,
   which models each rotated box as a 2-D Gaussian and scores overlap by a
   Bhattacharyya-style distance. ProbIoU is a smooth *approximation* of
   rotated-box IoU, designed as a differentiable training loss, not an exact
   area ratio. The two agree closely for well-aligned boxes and disagree most
   for elongated or strongly rotated ones — which is what leaf-lesion OBBs
   frequently are. A detection whose true polygon IoU is 0.52 can score
   below 0.5 under ProbIoU and flip from true positive to false positive at
   the IoU=0.50 threshold, and again independently at each of the ten
   thresholds 0.50…0.95.

2. **How the P/R curve is integrated into AP.** `evaluate_obb` uses COCO's
   rule: monotonise precision from the right, sample at 101 fixed recall
   points, take the mean. Ultralytics integrates its own interpolated curve.
   Same curve, different quadrature, different number.

3. **Which classes enter the mean.** `evaluate_obb` averages per-class AP
   over classes that actually have ground truth in the split, and documents
   why: scoring an absent class as 0 would drag the mean down by 1/nc for
   reasons unrelated to the model. Ultralytics applies its own rule.

The errors do not cancel. They are systematic per architecture — they depend
on box shape statistics and on the score distribution each model produces —
so the bias is *not* a constant offset you could subtract from the YOLO
column.

**Verification note, so this is not over-claimed:** items 1–3 as they apply
to `src/obb_eval.py` were read directly from this repo's source. The
characterisation of Ultralytics' side is taken from `src/obb_eval.py`'s own
module docstring, which names ProbIoU matching explicitly; Ultralytics is not
installed in this environment and its internals were **not** independently
re-verified here. That does not weaken the conclusion — two different
implementations were being reported in one table, which is the problem
regardless of exactly how they differ.

That function was never called from any notebook. Its only caller in the repo
is `analysis/comparison/rescore_baseline_yolo.py`, which rescores a *single*
baseline checkpoint (`runs/v4_confound_fix`) — not the nine K-fold
checkpoints. So the shared-metric path existed, was correct, and was wired to
nothing that produced a K-fold number.

Knock-on effects, all from the same root cause:

* **No `test_mAP75`.** Every non-YOLO arm records it; the YOLO arms' per-fold
  CSV and summary had no such column, so the arms' result tables could not be
  lined up even column-for-column.
* **P/R/F1 at a different operating point.** The non-YOLO arms use a
  `precision_recall_f1` helper written specifically to reproduce Ultralytics'
  max-mean-F1 operating point. The YOLO arms read `box.mp`/`box.mr` directly.
  Those happen to agree in intent — but only because the helper was written
  to chase them; nothing enforced it.
* **No `test_predictions.json`.** The other arms dump predictions in the
  shared format, so a fold can be rescored later without re-running
  inference. The YOLO arms dumped nothing, so there was no way to retrofit
  the shared metric onto an archived run short of full re-inference.
* **Test-time resolution was implicit.** `eval_model.val()` was called with
  no `imgsz`, relying on Ultralytics restoring it from the checkpoint's
  training args. Probably correct, but it is the B1 resolution confound left
  to a framework default, while every other arm passes `params["imgsz"]`
  explicitly.

**Fixed.** All four YOLO K-fold notebooks now, per fold: run
`predict_ultralytics(best.pt, test_items, imgsz=best_params["imgsz"],
conf=0.001)`, dump `test_predictions.json`, and score with `evaluate_obb` +
`precision_recall_f1` against ground truth built by
`src/obb_common.collect_items` — the same function, over the same `test/`
directories, that the other four arms use. `test_mAP50-95/50/75`,
`test_precision/recall/F1` now all come from the shared code.

Ultralytics' own numbers are **kept, not deleted** — its `val()` still runs
for its plots (confusion matrix, PR curves) and for continuity with archived
runs — but recorded as `ultra_test_mAP50-95` / `ultra_test_mAP50`, under names
that cannot be mistaken for the comparable ones. The summary JSON gains a
`"metric"` field naming the evaluator explicitly.

> **Implication for archived results:** every YOLO11m / YOLO26m K-fold number
> produced before this fix — E-IRFS and NoEIRFS alike — is on Ultralytics'
> metric. Those numbers are internally valid and fine to compare *to each
> other* (E-IRFS vs NoEIRFS, fold vs fold), but must not be placed beside a
> Strip R-CNN / Oriented-DETR / RF-DETR-OBB / S2A-Net number. Re-running the
> K-fold loop is what produces the comparable figure.

## Finding 2 — three different HPO budgets across six arms

All six configure an Optuna `PercentilePruner` and call `trial.report(...)`
after each HPO fold. But `trial.report()` prunes nothing on its own — Optuna
requires the objective to check `should_prune()` and raise `TrialPruned`
itself. State found:

| arm | `should_prune()` |
|---|---|
| Strip R-CNN, S2A-Net | called (fixed in earlier passes) |
| Oriented-DETR, RF-DETR-OBB | **missing** — pruner dead |
| YOLO11m, YOLO26m (Optuna siblings) | **commented out** — `# if trial.should_prune(): ...`, labelled "Optional pruning inside-CV" |

So four of six arms run every trial's every fold to completion regardless of
how bad an early fold was, while two prune. That is a different search
protocol, and for the YOLO arms it propagates into the K-fold numbers, since
`best_params.json` from that search is what K-fold training consumes.

### NOT fixed — deliberate, by project decision

This audit initially "fixed" the four non-pruning arms by adding the
`should_prune()` check. **That change was reverted at the project owner's
direction: leaving pruning off is a standing decision, taken because the
pruning path was judged not worth the implementation cost.** It is not an
oversight and should not be re-fixed by a future audit. Each of the four
affected notebooks now carries an inline comment saying so at the
`trial.report()` call site, so the next reader does not have to rediscover
this.

What that leaves standing, stated plainly so it is not mistaken for
resolved:

* Strip R-CNN and S2A-Net **do** prune; the other four run every trial in
  full. The HPO budgets therefore genuinely differ between those two groups.
* This affects which *hyperparameters* each arm ends up with, not which
  *metric* its reported score is on. After Finding 1, every arm's reported
  number comes from the same evaluator — so this does not make the results
  uncomparable in the way Finding 1 did. It is a difference in how hard each
  arm's hyperparameters were searched for.
* Worth one sentence in the paper's methods if HPO budget is described, so
  the asymmetry is disclosed rather than implied away.

If the goal is uniformity, the cheaper direction is to **remove** the
`should_prune()` blocks from Strip R-CNN and S2A-Net rather than add them to
the other four — that makes all six match the standing decision and costs
nothing to implement. Not done here, since it was not asked for and those two
arms' pruning was added deliberately in earlier passes.

`analysis/comparison/check_kfold_comparability.py` reports pruning status as
INFO rather than failing on it, so the split stays visible without being
treated as a bug.

## Finding 3 — one fold's failure destroyed the whole run (5 of 8 notebooks)

Strip R-CNN, RF-DETR-OBB and S2A-Net wrap each final-training fold in
`try/except`, log the failure, record a null row and continue — a guard added
after this was a real regression in RF-DETR-OBB. **Oriented-DETR and all four
YOLO K-fold notebooks had no such guard**, so one fold raising anywhere in
9 × 250 epochs would take down the run and write no `kfold_final_results.csv`
and no summary, discarding the already-completed folds' results.

Both YOLO notebooks' own all-NaN check below the loop —

```python
if fold_df["test_mAP50-95"].isna().all():
    raise RuntimeError("Every fold's test mAP50-95 came back NaN ...")
```

— already *assumed* the guard existed: it can only ever fire if failed folds
produce NaN rows, which requires the `except` branch that was missing.

**Fixed** in all five, mirroring Strip R-CNN's version. Each now appends a
null row with the error text and prints a `WARNING: n/K folds failed` summary.

Adding the `except` is only half the fix, since the null row then flows into
`fold_df` and every aggregation below it. The summary path was replayed
against a `fold_df` with failed folds to confirm three things actually hold:
the mean/std are computed over the **surviving** folds only (not over K with
zeros substituted, which would silently understate the result), `best_fold`
and `FINAL_RUN_DIR` skip the null rows, and an all-folds-failed run still
raises instead of writing a bogus summary.

Because a run can now end with fewer than K usable folds, every arm's summary
JSON also records `n_folds_failed` — `"n_folds": 9` on its own would overstate
the sample once failures are survivable. The four non-YOLO arms were brought
to parity on this field.

## Finding 4 — the YOLO arms' folds were reproducible only by luck

The four non-YOLO arms import `src/obb_common.make_folds`, whose
`kfold_assignments` reseeds `random` *immediately before shuffling* — its
docstring says this is "the only way the four notebooks can be guaranteed to
train and validate on byte-identical folds."

The YOLO K-fold notebooks keep a local `kfold_assignments(n, k, shuffle)` with
no `seed` parameter, relying on a `random.seed(SEED)` executed in a config
cell several hundred lines earlier.

### The mechanism

`random.shuffle()` draws from Python's single global Mersenne Twister state.
Fixing the permutation therefore requires fixing that state *at the moment of
the shuffle*, not at some earlier point in the program.

* **Shared version:** `random.seed(seed)` then `random.shuffle(idxs)`, back to
  back inside the same function. The permutation depends only on `seed` and
  `n_items`. Nothing else can reach in between.
* **YOLO version (pre-fix):** `random.seed(SEED)` runs in the config cell
  (line 136); `kfold_assignments` is called from `make_folds_files` around
  line 535, in a *different Colab cell*. The permutation depends on
  `seed`, `n_items`, **and everything that touched `random` in between**.

On a clean top-to-bottom run nothing does, so the folds matched exactly —
**verified, not assumed**. Previously archived YOLO K-fold runs executed
straight through are fine on this point.

The hazard is that Colab cells are independently re-runnable, which is the
normal way people use them: fix a path, hit re-run on the fold cell, don't
re-run the config cell 400 lines up. The seed is then *not* re-applied, but
the RNG has already advanced (the first shuffle consumed from it).

### How badly it diverges

Not subtly. Burning even a single extra RNG draw before the shuffle scatters
the split to essentially chance level:

| n_items | extra draws burned | items landing in a different fold |
|---|---|---|
| 1500 | 0 | 0 / 1500 (0.0%) |
| 1500 | 1 | 1400 / 1500 (93.3%) |
| 1500 | 5 | 1368 / 1500 (91.2%) |
| 1500 | 50 | 1377 / 1500 (91.8%) |

Two *independent* random 9-fold assignments would differ on ~88.9% of items.
So this is not a near-miss that perturbs a few boundary cases — one stray
draw and the YOLO arm is training on a completely unrelated partition from
the one Strip R-CNN, Oriented-DETR, RF-DETR-OBB and S2A-Net use. There is
also no warning: the split is still a valid, balanced, correctly-sized
9-fold split. It is simply a *different* one.

### What it would and would not have broken

* **Not a validity/leakage bug.** `test/` is a fixed, separate split that
  never enters the fold pool, so a different partition of `train/`+`val/`
  cannot leak test data. Each arm's own number stays internally honest.
* **A comparability bug.** The whole point of `make_folds` being shared is
  that per-fold variance and the reported mean ± std reflect the *same* nine
  partitions across arms. If YOLO's nine differ, part of the gap between
  YOLO's mean and Strip R-CNN's is just "different data in each fold."
* **A reproducibility bug.** Re-running would not reproduce the archived
  result, and nothing recorded which split was used — the fold lists are
  written to disk but the notebooks' own archive step excludes `folds/` on
  the grounds that the assignment "is reproducible from
  `make_folds(items, K_FOLDS, seed=SEED)` anyway." That justification is
  exactly what silently stopped being true.

**Fixed**: the local `kfold_assignments` now reseeds internally, matching
`src/obb_common`. `analysis/comparison/check_kfold_comparability.py` asserts
the two implementations agree — both on a clean run and after deliberate
drift — and that assertion fails on 8/8 cases against the pre-fix code, so it
is genuinely sensitive rather than vacuously passing.

---

## Finding 5 — two arms silently discarded the dataset's background negatives (2026-09-12)

`analysis/dataset/phase4_background_negatives.py` adds label-free images to
`train/` **on purpose** — ~8% of the split, background-only field crops, and its
own comment says outright "Ultralytics treats this as background, same as a
missing label file". Four arms do train on them. Two did not, each through a
different framework default, and neither reported it anywhere a reader would
look.

* **S2A-Net.** mmdet 1.x's `LoadAnnotations` defaults to
  `skip_img_without_anno=True`, so `_load_bboxes` returns `None` for an image
  with no instances and `CustomDataset.__getitem__` loops
  `idx = self._rand_another(idx)` — training on a *different* image instead. An
  empty label file therefore trained nothing, raised nothing, and silently
  duplicated some other image in that epoch. A live run showed 17 of 963 images
  in HPO fold 0 taking that path. In the E-IRFS twin it is worse than a no-op:
  `_rand_another` is uniform, so it also overrode E-IRFS's own sampling choice
  for those draws.
* **Strip R-CNN.** MMRotate's `DOTADataset` drops them earlier and more
  completely: `load_annotations` has
  `if os.path.getsize(ann_file) == 0 and self.filter_empty_gt: continue`, and
  `_filter_imgs` drops any zero-label image that gets past it under the same
  flag — which mmdet 2.x's `CustomDataset` defaults to `True`.

Net effect: the two arms with the most capacity to exploit hard negatives were
the two never shown any, while the four YOLO/DETR arms were.

### Fixed

* **S2A-Net**, both twins: `LoadAnnotations(..., skip_img_without_anno=False)`
  in `build_config`, plus `patch_empty_gt_support()` in
  `src/s2anet_worker.py`. The flag alone is not sufficient and fails loudly if
  you try it: `MaxIoUAssigner.assign` opens with
  `raise ValueError('No gt or bboxes')`, and `rotated_box_to_poly_np` /
  `poly_to_rotated_box_np` return `(0,)`-shaped arrays for empty input, which
  breaks `RotatedResize._resize_bboxes`. Both are monkey-patched in this repo's
  own worker rather than in the clone, so the fix survives the clone being
  rebuilt in a fresh runtime. Neither S2A-Net arm had been run yet, so no
  existing result was invalidated.
* **Strip R-CNN**: `filter_empty_gt=False` on its train dataset — one flag,
  since mmdet 2.x's assigner already tolerates empty GT. **Takes effect only on
  a re-run of that arm**, and has not been smoke-tested: it is two-stage, and
  RPN-with-zero-positive-proposals was not traced.

### Regression guard

`check_kfold_comparability.py` now asserts both flag strings. As with the
inherited `score_thr` above, the bad value is a *default*, so there is nothing
to grep for — absence of the fix string is itself the violation. The same pass
registered `S2ANet_NoEIRFS` in the guard's `ABLATION` table, which it had been
missing from entirely.

## Finding 6 — S2A-Net kept 6.7x more detections per image than the YOLO arms (2026-09-12)

`test_cfg.max_per_img` was the stock DOTA value of 2000 while the YOLO arms keep
Ultralytics' `max_det=300` default. Extra low-confidence detections can only
extend the tail of the precision/recall curve `evaluate_obb` integrates, so this
was a small one-directional advantage to S2A-Net — on top of making each fold
write ~185k predictions, which is slow to score.

**Fixed:** 300 in both S2A-Net twins, before either was run. **Not audited for
Strip R-CNN, Oriented-DETR or RF-DETR-OBB**, whose own caps were not examined;
worth a pass if their numbers are ever re-derived.

## Checked and found already consistent

Verified across all six arms, no change needed:

* `SEED = 42`, `K_FOLDS = 9`, `TRAIN_EPOCHS = 250`, `EIRFS_THRESHOLD = 1e-4`.
* HPO budget shape: `N_TRIALS = 10`, `HPO_K_FOLDS = 3`,
  `EPOCHS_PER_TRIAL = 50`, `N_STARTUP_TRIALS = 3`.
* `imgsz` search space `[320, 640, 832, 1024]` everywhere. (`batch` ranges
  differ by arm — a genuine memory constraint, not drift.)
* Item pool: `train/` + `val/` pooled for final folds, `train/` only for the
  HPO folds, in every arm.
* Evaluation operating point: conf / `score_thr` = 0.001 in all six.
* E-IRFS weights built from `spec["train_labels"]` (train-only) in every
  non-YOLO arm; the YOLO arms' `YOLOEIRFSCappedDataset` guards on
  `"train" in self.prefix`, so no arm resamples its val split.
* E-IRFS cap parameters: `capped_class_names={"Rhizoctonia"}`, `alpha=2.0`,
  `cap_multiplier=1.5` — the non-YOLO arms take `EIRFSWeights`' defaults,
  which equal the YOLO notebooks' explicit constants.
* Test split: all six read `/content/data/test/`.

## Known, deliberate, still open

* **Mixed precision is now on for S2A-Net, and unaudited elsewhere.** Its
  configs set `fp16 = dict(loss_scale=512.)` (2026-09-12), which mmdet 1.x
  turns into an `Fp16OptimizerHook`. This *reduces* drift rather than adding it
  — Ultralytics defaults `amp=True` and the YOLO notebooks never override it, so
  fp32 here was the outlier — and `S2ANetHead` already carries `@force_fp32` on
  `loss()`/`get_bboxes()`, so the losses, rotated IoU assignment and NMS stay
  fp32; only backbone/FPN/head convolutions use the tensor cores. Evaluation is
  fp32 regardless, since `init_detector` has no fp16 path. What is *not*
  established is whether Strip R-CNN, Oriented-DETR and RF-DETR-OBB train in
  fp32 or mixed precision, so "all six use mixed precision" is not a claim this
  audit supports yet.

* **`best_fold` peeks at test scores.** Every arm selects its archived
  checkpoint with `fold_df.loc[fold_df["test_mAP50-95"].idxmax()]`, which is
  optimistically biased. Previously flagged in CLAUDE.md as "flagged, not
  fixed"; left as-is, since it is a settled methodology decision and affects
  only which checkpoint is archived, not the reported statistic. A comment
  saying so is now in the YOLO summaries. **The unbiased number to report is
  `test_mAP50-95_mean ± _std`.**
* **HPO *selection criterion* still differs.** The non-YOLO arms score HPO
  folds with `evaluate_obb`; the YOLO Optuna notebooks score them with
  `read_final_map_from_results` (Ultralytics' val mAP from `results.csv`).
  Not fixed: it changes only which hyperparameters get picked, not the
  reported metric, and changing it invalidates every existing
  `best_params.json` and forces a full re-search. Worth doing if the YOLO
  arms are ever re-tuned from scratch; not worth forcing now.
* **`precision_recall_f1` is duplicated.** The canonical copy now lives in
  `src/obb_eval.py` (verified byte-identical in behaviour to the notebooks'
  copies across 30 randomised cases plus degenerate inputs). The four
  non-YOLO notebooks still carry their own inline copies and were left
  untouched — consolidating them is a safe follow-up, not a correctness fix.
* **Diagnostic only:** the YOLO arms' `val_mAP50-95` column read the *last*
  row of `results.csv` (the last epoch trained, i.e. `FINAL_PATIENCE` epochs
  past the saved checkpoint) while test eval loads `best.pt` — two different
  checkpoints in adjacent columns. Now takes the best row instead. Caveat
  kept in the code: Ultralytics selects `best.pt` on a fitness blend
  (0.1·mAP50 + 0.9·mAP50-95), not mAP50-95 alone, so this is a close proxy,
  not provably the same epoch. That formula was **not** re-verified against
  the installed package. No reported result depends on this column.
  *Naming wart, flagged deliberately:* the function is still called
  `read_final_map_from_results`, and the identically-named copies in the
  Optuna siblings (and `SAHI_*`, `Template_Clean_*`) still return the last
  row. Those were left alone on purpose — there the value **is** the trial's
  Optuna objective, so changing it would silently change which
  hyperparameters win. Renaming across all six notebook families was judged
  wider churn than the diagnostic column justifies; the K-fold copies carry
  an explicit comment saying what they now return.

## Regression guard

`analysis/comparison/check_kfold_comparability.py` — runs in seconds, no GPU,
no framework imports (four of the six arms pin mutually incompatible
torch/mmcv/Python versions and cannot share one interpreter, which is
precisely why they drift). It asserts every invariant above.

```
python analysis/comparison/check_kfold_comparability.py
```

Confirmed to flag all 28 violations on the pre-fix tree. It is a source-text
audit, so it cannot catch runtime-only drift — see
`results/eirfs_noop_nonyolo_arms.md` for the canonical example of a bug
(`attach_eirfs` silently inert) that only a live run would reveal.

### One expected violation on this branch

Run as-is, the guard reports exactly one failure:

```
[StripRCNN] does not explicitly set score_thr=0.001; its base config
inherits the stock 0.05, ...
```

This is a **true positive, and not a regression from this audit.** The
Strip R-CNN `score_thr` fix (`test_cfg=dict(rcnn=dict(score_thr=0.001))`)
exists only as an *uncommitted working-tree change* in the main checkout, so
it is absent from the commit this branch was cut from. It clears as soon as
that change is committed. It was left alone here rather than duplicated, to
avoid conflicting with work in progress.

Note the shape of this check: unlike the others it asserts the *presence* of
an explicit override, because Strip R-CNN and S2A-Net inherit `score_thr=0.05`
from their upstream DOTA base configs. The value is never written in the
notebook, so grepping for `0.05` proves nothing — absence of the bad value and
absence of any value look identical. That is the same trap the original
Strip R-CNN / S2A-Net `score_thr` bug hid in.
